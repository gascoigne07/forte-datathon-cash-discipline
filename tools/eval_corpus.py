"""Оценка правил по корпусу `data/corpus` (tools/gpu_corpus.py) при заданном KF_CAP.

Кадры выбираются `pipeline.select_keyframes`, факты считаются `pipeline.frame_facts`,
события — `rules.detect_events`: всё теми же функциями, что и в сервисе. Отличие от
живого прогона одно — нет дедлайна, то есть это оценка при полном бюджете кадров.

    PYTHONPATH=service:tools .venv/bin/python tools/eval_corpus.py --cap 150 [--cap 45 ...]
"""
import argparse, glob, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "service"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from app import pipeline, rules  # noqa: E402
from app.objects import ObjectTrack  # noqa: E402
from app.perception import Frame  # noqa: E402
from eval_caps import DUPLICATES  # noqa: E402
from evaluate import LABELS  # noqa: E402

CORPUS = "data/corpus"


def load(name: str, corpus: str = CORPUS):
    """Кадры (без пикселей, но с формой кадра), зона, треки и подписи из корпуса."""
    d = json.loads(Path(f"{corpus}/{name}.json").read_text())
    shape = np.broadcast_to(np.uint8(0), (d["h"], d["w"], 3))   # форма без памяти
    frames = [Frame(t=f["t"], img=shape, persons=f["persons"], objects=f["objects"],
                    motion=f["motion"]) for f in d["frames"]]
    tracks = [ObjectTrack(t_start=t["t_start"], t_end=t["t_end"], box=tuple(t["box"]),
                          area=t["area"], frames=t["frames"], near_person=t["near_person"])
              for t in d["tracks"]]
    caps = [f["caption"] for f in d["frames"]]
    return d, frames, tracks, caps


def series_at(name: str, cap: int, extra=None, corpus: str = CORPUS):
    """Ряд фактов, который получил бы сервис при KF_CAP=cap.

    extra(name, i, facts) — дописывает факты второго канала (VLM) по индексу кадра.
    """
    d, frames, tracks, caps = load(name, corpus)
    zone = tuple(d["zone"])
    idx = pipeline.select_keyframes(frames, tracks, cap)
    series = []
    for i in idx:
        facts = pipeline.frame_facts(frames[i], caps[i], zone, tracks)
        if extra is not None:
            extra(name, i, facts)
        series.append((frames[i].t, facts))
    return series, tracks, idx, d


def vlm_extra(name, i, facts, _cache={}):
    """Факты второго описателя (tools/vlm_corpus.py) по индексу кадра сетки 2 fps.

    Ответы хранятся сырыми и разбираются `vlm.parse` — той же функцией, что в сервисе.
    """
    from app import vlm
    if name not in _cache:
        fp = Path(f"data/vlm/{name}.json")
        _cache[name] = json.loads(fp.read_text())["frames"] if fp.exists() else []
    rows = _cache[name]
    if i < len(rows):
        facts.update(vlm.parse(rows[i]["raw"]))
    skey = ("safe", name)
    if skey not in _cache:
        fp = Path(f"data/vlm_safe/{name}.json")
        _cache[skey] = json.loads(fp.read_text()) if fp.exists() else {}
    if str(i) in _cache[skey]:
        facts["safe_open"] = _cache[skey][str(i)].startswith("open")
    key = ("drink", name)
    if key not in _cache:
        fp = Path(f"data/vlm_drink/{name}.json")
        _cache[key] = json.loads(fp.read_text()) if fp.exists() else {}
    if str(i) in _cache[key] and facts.get("drink_or_food"):
        facts["drink_in_hand_vlm"] = vlm.yes(_cache[key][str(i)])


def names(corpus: str = CORPUS):
    out = []
    for p in sorted(glob.glob(f"{corpus}/*.json")):
        n = Path(p).stem
        if n in LABELS and n not in DUPLICATES:
            out.append(n)
    return out


def evaluate(cap: int, extra=None, detect=None, corpus: str = CORPUS, verbose=False):
    detect = detect or rules.detect_events
    tp = fp = fn = tn = 0
    errs, rows = [], []
    for name in names(corpus):
        series, tracks, idx, d = series_at(name, cap, extra, corpus)
        ev = detect(series, tracks=tracks)
        gold = LABELS[name]
        truth, pred = bool(gold), bool(ev)
        tp += pred and truth; fp += pred and not truth
        fn += (not pred) and truth; tn += (not pred) and not truth
        dd = None
        if gold and ev:
            dd = min(abs(e.t - g) for e in ev for g in gold); errs.append(dd)
        rows.append({"name": name, "truth": truth, "pred": pred, "dt": dd, "frames": len(idx),
                     "events": [f"{e.key}@{e.t:.0f}" for e in ev]})
    n = tp + fp + fn + tn
    errs.sort()
    res = {"cap": cap, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "n": n,
           "acc": (tp + tn) / max(1, n),
           "dt_med": errs[len(errs) // 2] if errs else None,
           "dt_le3": sum(e <= 3 for e in errs) / len(errs) if errs else None,
           "dt_mean": float(np.mean(errs)) if errs else None,
           "found": [r["name"] for r in rows if r["truth"] and r["pred"]],
           "false": [r["name"] for r in rows if r["pred"] and not r["truth"]],
           "rows": rows}
    if verbose:
        for r in rows:
            mark = "OK" if r["pred"] == r["truth"] else ("ЛОЖНЯК" if r["pred"] else "ПРОПУСК")
            d = f"Δ={r['dt']:.1f}с" if r["dt"] is not None else ""
            print(f"{r['name']:7s} {'V' if r['truth'] else '.':2s} {mark:8s} {d:9s} "
                  f"[{r['frames']:3d} к] {', '.join(r['events'])}")
    return res


def summary(r) -> str:
    s = (f"KF_CAP={r['cap']}: вердикт {r['tp'] + r['tn']}/{r['n']} ({100 * r['acc']:.0f} %), "
         f"найдено {r['tp']}/{r['tp'] + r['fn']}, ложных {r['fp']}/{r['fp'] + r['tn']}")
    if r["dt_med"] is not None:
        s += (f", Δt медиана {r['dt_med']:.1f} с, ≤3 с у {100 * r['dt_le3']:.0f} %, "
              f"средняя {r['dt_mean']:.1f} с")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, action="append")
    ap.add_argument("--corpus", default=CORPUS)
    ap.add_argument("--vlm", action="store_true", help="факты второго описателя из data/vlm")
    ap.add_argument("-v", action="store_true")
    a = ap.parse_args()
    for cap in a.cap or [150]:
        r = evaluate(cap, extra=vlm_extra if a.vlm else None, corpus=a.corpus, verbose=a.v)
        print(summary(r))
        print(f"  найдены: {', '.join(r['found'])}")
        print(f"  ложные: {', '.join(r['false']) or '—'}")


if __name__ == "__main__":
    main()
