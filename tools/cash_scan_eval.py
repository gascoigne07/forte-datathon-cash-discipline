"""Проверка классификатора наличных на РЕАЛЬНЫХ кадрах, с двусторонним контролем.

Протокол без утечки: для каждой вырезки-источника классификатор обучается БЕЗ неё,
а затем сканирует реальный кадр, из которого она взята. Негативы — столешницы
из чистых роликов, которые в обучении участвовали только как фон.

Если максимум скора на реальных кадрах с деньгами не отделяется от максимума
на чистых — канал бесполезен, сколько бы ни было точности на синтетике.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "service"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import synth_cash as sc  # noqa: E402
from app import objects, perception  # noqa: E402
from evaluate import LABELS  # noqa: E402

VID = "/Users/miras/Downloads/open/videos"


def scan(img, zone, coef, intercept, stride=12):
    """Максимум скора по скользящему окну внутри зоны столешницы."""
    x1, y1, x2, y2 = zone
    half = sc.WIN // 2
    wins, centers = [], []
    for cy in range(y1 + half, y2 - half, stride):
        for cx in range(x1 + half, x2 - half, stride):
            w = sc.window(img, cx, cy)
            if w is not None:
                wins.append(w); centers.append((cx, cy))
    if not wins:
        return -9.0, None
    E = sc.embed(wins)
    scores = E @ coef.ravel() + intercept[0]
    i = int(np.argmax(scores))
    return float(scores[i]), centers[i]


def desk_of(name, sec=None):
    fr = perception.sample(f"{VID}/{name}.mov", fps=1.0)
    cache = Path(f"data/cache/{name}.detr.json")
    if cache.exists():
        import json
        det = json.loads(cache.read_text())["frames"]
        byt = {round(d["t"], 2): d["det"] for d in det}
        for f in fr:
            ds = byt.get(round(f.t, 2), [])
            f.persons = [d for d in ds if d["cls"] == "person" and d["score"] >= 0.35]
            f.objects = [d for d in ds if d["cls"] != "person" and d["score"] >= 0.35]
    else:
        perception.detect(fr)
    zone = objects.desk_zone(fr)
    if sec is None:
        return [(f.img, zone) for f in fr], zone
    f = min(fr, key=lambda x: abs(x.t - sec))
    return f.img, zone


def main():
    from sklearn.linear_model import LogisticRegression
    X, y, grp, srcs = sc.build()
    E = sc.embed(list(X))
    print(f"обучающая синтетика: {len(y)} окон\n", file=sys.stderr)

    print(f"{'кадр':16s} {'класс':8s} {'макс.скор':>10s}  где")
    pos_scores, neg_scores = [], []

    # положительные: реальный кадр источника, классификатор обучен БЕЗ него
    for si, (label, _) in enumerate(srcs):
        name, sec = label.split("@")
        clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")
        m = grp != si
        clf.fit(E[m], y[m])
        img, zone = desk_of(name, float(sec))
        s, c = scan(img, zone, clf.coef_, clf.intercept_)
        pos_scores.append(s)
        print(f"{label:16s} {'ДЕНЬГИ':8s} {s:10.2f}  {c}")

    # негативы: кадры из чистых роликов (в обучении были только фоном)
    clf_all = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced").fit(E, y)
    for name in ("Х-3", "Х-43", "Х-10", "Х-21", "Х-29", "Х-16", "Х-23", "Х-37"):
        if not Path(f"{VID}/{name}.mov").exists():
            continue
        frames, zone = desk_of(name)
        best = -9.0; where = None
        for img, z in frames[:6]:
            s, c = scan(img, z or zone, clf_all.coef_, clf_all.intercept_)
            if s > best:
                best, where = s, c
        neg_scores.append(best)
        print(f"{name:16s} {'чистый':8s} {best:10.2f}  {where}")

    print(f"\nмедиана на деньгах: {np.median(pos_scores):.2f}, "
          f"на чистых: {np.median(neg_scores):.2f}")
    print(f"минимум на деньгах: {min(pos_scores):.2f}, "
          f"максимум на чистых: {max(neg_scores):.2f}")
    if min(pos_scores) > max(neg_scores):
        print(f"РАЗДЕЛЯЕТСЯ: порог можно поставить на "
              f"{(min(pos_scores) + max(neg_scores)) / 2:.2f}")
    else:
        print("НЕ РАЗДЕЛЯЕТСЯ: распределения пересекаются, канал как есть не годится")


if __name__ == "__main__":
    main()
