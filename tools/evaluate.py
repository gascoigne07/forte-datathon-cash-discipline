"""Оценка правил по кэшу фактов против разметки датасета.

Метрики ровно те, по которым считает жюри:
  критерий 1 — вердикт на всё видео (пропуск хуже ложняка);
  критерий 2 — ошибка таймкода относительно размеченной секунды.
"""
import argparse, glob, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "service"))
from app import rules  # noqa: E402

LABELS = {
    "X-1": [22], "X-2": [10], "Х-4": [57, 68], "Х-11": [18], "Х-13": [47],
    "Х-15": [15], "Х-17": [7], "Х-18": [20], "Х-20": [5], "Х-22": [14],
    "Х-24": [8], "Х-26": [9], "Х-28": [7], "Х-30": [13], "Х-32": [5],
    "Х-33": [6], "Х-34": [13], "Х-38": [33], "Х-40": [17], "Х-44": [16],
    "Х-46": [11], "Х-47": [68],
    "Х-3": [], "Х-5": [], "Х-8": [], "Х-10": [], "Х-12": [], "Х-14": [],
    "Х-16": [], "Х-19": [], "Х-21": [], "Х-23": [], "Х-25": [], "Х-27": [],
    "Х-29": [], "Х-31": [], "Х-35": [], "Х-37": [], "Х-39": [], "Х-43": [],
    "Х-45": [], "Х-48": [],
}


def load(path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    series = [(f["_t"], f) for f in d["series"]]
    return d["video"], sorted(series, key=lambda x: x[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--facts", default="data/facts")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    tp = fp = fn = tn = 0
    errs, rows = [], []
    for p in sorted(glob.glob(f"{a.facts}/*.facts.json")):
        name, series = load(p)
        if name not in LABELS:
            continue
        gold = LABELS[name]
        ev = rules.detect_events(series)
        pred = bool(ev)
        truth = bool(gold)
        if pred and truth: tp += 1
        elif pred and not truth: fp += 1
        elif not pred and truth: fn += 1
        else: tn += 1

        best = ""
        if gold and ev:
            d = min(abs(e.t - g) for e in ev for g in gold)
            errs.append(d)
            best = f"Δt={d:.1f}с"
        mark = "OK " if pred == truth else ("ЛОЖНЯК" if pred else "ПРОПУСК")
        rows.append((name, "V" if truth else ".", mark, best,
                     ", ".join(f"{e.key}@{e.t:.0f}" for e in ev)))

    w = max(len(r[0]) for r in rows) if rows else 8
    print(f"{'видео':{w}s} GT  статус   {'Δtaймкод':10s} события")
    for r in rows:
        print(f"{r[0]:{w}s} {r[1]:3s} {r[2]:8s} {r[3]:10s} {r[4]}")

    n = tp + fp + fn + tn
    print(f"\nвердикт: TP={tp} FP={fp} FN={fn} TN={tn}  точность={100*(tp+tn)/max(1,n):.0f}%")
    print(f"пропуски (критично!): {fn} из {tp+fn} нарушений")
    print(f"ложные срабатывания: {fp} из {fp+tn} чистых")
    if errs:
        errs.sort()
        print(f"таймкод: медиана {errs[len(errs)//2]:.1f}с, "
              f"≤2с у {100*sum(e<=2 for e in errs)/len(errs):.0f}%, "
              f"≤5с у {100*sum(e<=5 for e in errs)/len(errs):.0f}%")


if __name__ == "__main__":
    main()
