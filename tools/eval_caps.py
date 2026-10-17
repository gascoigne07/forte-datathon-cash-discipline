"""Оценка правил по кэшу подписей + геометрии. Быстрый цикл тюнинга без моделей.

Каталог подписей задаётся первым аргументом — так два описателя (Florence-2 и
Qwen3-VL) сравниваются на ОДНИХ И ТЕХ ЖЕ правилах, а не на разных.
"""
import json, sys, glob
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "service"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from app import caption, rules  # noqa: E402
from app.objects import ObjectTrack  # noqa: E402
from app.pipeline import TRACK_MIN_AREA  # noqa: E402
from evaluate import LABELS  # noqa: E402

# Х-47 побайтово равен Х-4, Х-48 равен Х-5 (md5 совпадает). Оценивать их как
# отдельные ролики — значит дважды засчитывать один и тот же результат и искажать
# метрику: один раз это уже привело к ложному выводу об улучшении.
DUPLICATES = {"Х-47", "Х-48"}


CAPS_DIR = "data/captions"


def series_for(name):
    cp = Path(f"{CAPS_DIR}/{name}.caps.json")
    fp = Path(f"data/feats/{name}.feats.json")
    if not cp.exists():
        return None, None
    caps = json.loads(cp.read_text())["captions"]
    feats = json.loads(fp.read_text()) if fp.exists() else {"rows": [], "tracks": []}
    byt = {r["t"]: r for r in feats["rows"]}
    series = []
    for c in caps:
        f = caption.facts_from_caption(c["caption"])
        row = byt.get(c["t"]) or min(feats["rows"], key=lambda r: abs(r["t"] - c["t"])) if feats["rows"] else None
        f["people_in_cabin"] = row["ppl"] if row else 0
        f["teller_present"] = bool(row and row["ppl"] > 0 and row.get("teller_dist") is not None)
        f["track_on_desk"] = any(t["t0"] - 3 <= c["t"] <= t["t1"] + 3
                                 and t["area"] >= TRACK_MIN_AREA
                                 for t in feats.get("tracks", []))
        series.append((c["t"], f))
    # Треки приводим к тому же типу, что и в сервисе: detect_events уточняет по ним
    # таймкод, и без этого замер Δt измерял бы не то, что реально отдаёт сервис.
    tracks = [ObjectTrack(t_start=t["t0"], t_end=t["t1"], box=tuple(t["box"]),
                          area=t["area"], frames=0, near_person=t["near_person"])
              for t in feats.get("tracks", []) if t["area"] >= TRACK_MIN_AREA]
    return series, tracks


def main():
    global CAPS_DIR
    if len(sys.argv) > 1:
        CAPS_DIR = sys.argv[1].rstrip("/")
    print(f"подписи из: {CAPS_DIR}")
    tp = fp = fn = tn = 0; errs = []; rows = []
    for p in sorted(glob.glob(f"{CAPS_DIR}/*.caps.json")):
        name = Path(p).stem.replace(".caps", "")
        if name not in LABELS or name in DUPLICATES:
            continue
        series, tracks = series_for(name)
        if not series:
            continue
        ev = rules.detect_events(series, tracks=tracks)
        gold, pred, truth = LABELS[name], bool(ev), bool(LABELS[name])
        tp += pred and truth; fp += pred and not truth
        fn += (not pred) and truth; tn += (not pred) and not truth
        d = ""
        if gold and ev:
            dd = min(abs(e.t - g) for e in ev for g in gold); errs.append(dd); d = f"Δ={dd:.1f}с"
        mark = "OK" if pred == truth else ("ЛОЖНЯК" if pred else "ПРОПУСК")
        rows.append((name, "V" if truth else ".", mark, d,
                     ", ".join(f"{e.key}@{e.t:.0f}" for e in ev)))
    for r in rows:
        print(f"{r[0]:7s} {r[1]:2s} {r[2]:8s} {r[3]:9s} {r[4]}")
    n = tp + fp + fn + tn
    print(f"\nTP={tp} FP={fp} FN={fn} TN={tn}  точность={100*(tp+tn)/max(1,n):.0f}%  (уникальных роликов {n})")
    print(f"recall={100*tp/max(1,tp+fn):.0f}%  specificity={100*tn/max(1,tn+fp):.0f}%")
    if errs:
        errs.sort(); print(f"таймкод: медиана {errs[len(errs)//2]:.1f}с, ≤3с у {100*sum(e<=3 for e in errs)/len(errs):.0f}%")


if __name__ == "__main__":
    main()
