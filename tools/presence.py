"""Временные ряды присутствия людей из кэша RT-DETR + сверка с разметкой.

«В кабине» = крупная детекция person (люди за стеклом в опенспейсе мельче).
Порог по высоте бокса задаётся относительно высоты кадра, чтобы не зависеть от разрешения.
"""
import json, glob, sys
from pathlib import Path
import numpy as np

H = 848  # высота кадра датасета
MIN_H_REL = 0.115   # ~98 px: человек внутри кабины
PERSON_THR = 0.45

LABELS = {  # видео -> секунды нарушений
    "X-1": [22], "X-2": [10], "Х-4": [57, 68], "Х-11": [18], "Х-13": [47],
    "Х-15": [15], "Х-17": [7], "Х-18": [20], "Х-20": [5], "Х-22": [14],
    "Х-24": [8], "Х-26": [9], "Х-28": [7], "Х-30": [13], "Х-32": [5],
    "Х-33": [6], "Х-34": [13], "Х-38": [33], "Х-40": [17], "Х-44": [16],
    "Х-46": [11], "Х-47": [68], "Х-48": [],
    "Х-3": [], "Х-5": [], "Х-8": [], "Х-10": [], "Х-12": [], "Х-14": [],
    "Х-16": [], "Х-19": [], "Х-21": [], "Х-23": [], "Х-25": [], "Х-27": [],
    "Х-29": [], "Х-31": [], "Х-35": [], "Х-37": [], "Х-39": [], "Х-43": [],
    "Х-45": [],
}


def series(path):
    d = json.load(open(path))
    out = []
    for fr in d["frames"]:
        big = [x for x in fr["det"]
               if x["cls"] == "person" and x["score"] >= PERSON_THR
               and (x["box"][3] - x["box"][1]) >= MIN_H_REL * H]
        # «за столом»: нижний край бокса в нижней половине кадра
        at_desk = [x for x in big if x["box"][3] > 0.60 * H]
        out.append((fr["t"], len(big), len(at_desk)))
    return d["video"], out


def main():
    rows = []
    for f in sorted(glob.glob("data/cache/*.detr.json")):
        name, s = series(f)
        lab = LABELS.get(name)
        if lab is None:
            continue
        viol = "V" if lab else "."
        counts = "".join(str(min(c, 9)) for _, c, _ in s)
        desk = "".join(str(min(c, 9)) for _, _, c in s)
        # отметка секунд нарушения в строке (шаг 0.5 c -> индекс = сек*2)
        mark = [" "] * len(counts)
        for sec in lab:
            i = int(sec * 2)
            if 0 <= i < len(mark):
                mark[i] = "^"
        rows.append((name, viol, counts, desk, "".join(mark)))

    print("video    V  люди в кабине (шаг 0.5 с) / ^ = размеченная секунда")
    for name, viol, counts, desk, mark in rows:
        print(f"{name:7s} {viol}  {counts}")
        print(f"{'':7s}    {desk}  <- за столом")
        if viol == "V":
            print(f"{'':7s}    {mark}")

    # агрегат: доля кадров без людей за столом
    print("\n=== агрегаты ===")
    print(f"{'video':8s} {'V':2s} {'кадров':>6s} {'max_люди':>8s} {'%без_стола':>10s} {'смен_состава':>12s}")
    for f in sorted(glob.glob("data/cache/*.detr.json")):
        name, s = series(f)
        if LABELS.get(name) is None:
            continue
        c = np.array([x[1] for x in s]); d = np.array([x[2] for x in s])
        changes = int((np.diff(c) != 0).sum())
        print(f"{name:8s} {'V' if LABELS[name] else '.':2s} {len(s):6d} {c.max():8d} "
              f"{100*(d == 0).mean():10.0f} {changes:12d}")


if __name__ == "__main__":
    main()
