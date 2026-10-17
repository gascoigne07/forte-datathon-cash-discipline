"""Оценка объединения двух каналов описания: обычный кадр + накопленный.

Накопление медианой по соседним кадрам убирает шум сенсора и артефакты сжатия
на статичной сцене — мелкий предмет на столешнице становится различимее. Но оно
же размазывает движущееся: стакан в руке исчезает вместе с человеком.

Поэтому каналы объединяются, а не заменяют друг друга: факт считается истинным,
если его увидел хотя бы один из двух. Здесь измеряется, что это даёт по метрикам.
"""
import glob, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "service"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from app import caption, rules  # noqa: E402
from evaluate import LABELS  # noqa: E402

DUPLICATES = {"Х-47", "Х-48"}
OBJ_KEYS = ("cash_seen", "drink_or_food", "personal_bag", "sealed_bag", "paper_seen")


def load(dirname, name):
    p = Path(f"{dirname}/{name}.caps.json")
    return json.loads(p.read_text())["captions"] if p.exists() else None


def series_for(name, use_fused=True):
    raw = load("data/captions", name)
    if raw is None:
        return None
    fused = load("data/captions_fused", name) if use_fused else None
    fmap = {c["t"]: c["caption"] for c in (fused or [])}

    fp = Path(f"data/feats/{name}.feats.json")
    feats = json.loads(fp.read_text()) if fp.exists() else {"rows": [], "tracks": []}
    rows = feats["rows"]; byt = {r["t"]: r for r in rows}
    tracks = [t for t in feats.get("tracks", []) if t["area"] >= 60]

    out = []
    for c in raw:
        f = caption.facts_from_caption(c["caption"])
        if c["t"] in fmap:
            g = caption.facts_from_caption(fmap[c["t"]])
            # предметные факты объединяем, признак «нет человека» берём только из
            # обычного кадра: на накопленном человек размазывается и исчезает,
            # то есть этот признак там системно завышен.
            for k in OBJ_KEYS:
                f[k] = f[k] or g[k]
            for k in ("bag_ctx", "sealed_ctx", "cash_ctx"):
                if f[k] == "unknown":
                    f[k] = g[k]
        row = byt.get(c["t"]) or (min(rows, key=lambda r: abs(r["t"] - c["t"])) if rows else None)
        f["people_in_cabin"] = row["ppl"] if row else 0
        f["teller_present"] = bool(row and row["ppl"] > 0 and row.get("teller_dist") is not None)
        f["track_on_desk"] = any(t["t0"] - 3 <= c["t"] <= t["t1"] + 3 for t in tracks)
        out.append((c["t"], f))
    return out


def run(use_fused):
    tp = fp = fn = tn = 0; errs = []; fps = []; hits = []
    for p in sorted(glob.glob("data/captions/*.caps.json")):
        name = Path(p).stem.replace(".caps", "")
        if name not in LABELS or name in DUPLICATES:
            continue
        s = series_for(name, use_fused)
        if not s:
            continue
        ev = rules.detect_events(s)
        gold, pred, truth = LABELS[name], bool(ev), bool(LABELS[name])
        tp += pred and truth; fp += pred and not truth
        fn += (not pred) and truth; tn += (not pred) and not truth
        if pred and truth:
            hits.append(name); errs.append(min(abs(e.t - g) for e in ev for g in gold))
        if pred and not truth:
            fps.append(name)
    errs.sort()
    med = errs[len(errs) // 2] if errs else -1
    le3 = 100 * sum(e <= 3 for e in errs) / len(errs) if errs else 0
    n = tp + fp + fn + tn
    return tp, fp, 100 * (tp + tn) / max(1, n), med, le3, fps, hits


if __name__ == "__main__":
    have = len(glob.glob("data/captions_fused/*.caps.json"))
    print(f"накопленных подписей собрано: {have} роликов\n")
    for use in (False, True):
        tp, fp, acc, med, le3, fps, hits = run(use)
        tag = "обычный + накопленный" if use else "только обычный"
        print(f"{tag:24s} TP={tp:2d} FP={fp:2d} точность={acc:3.0f}%  "
              f"Δt медиана={med:4.1f}с  ≤3с у {le3:3.0f}%")
        if fps:
            print(f"{'':24s} ложные: {','.join(fps)}")
        print(f"{'':24s} найдены: {','.join(hits)}")
