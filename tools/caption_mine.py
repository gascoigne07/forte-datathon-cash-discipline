"""Поиск фраз, которые реально отличают нарушение от нормы.

Объективная замена разглядыванию кадров: берём корпус подписей Florence-2 и
считаем для каждой n-граммы, в скольких роликах с нарушением она встречается
против чистых. Ранжируем по точному тесту Фишера.
"""
import json, re, sys
from collections import Counter
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import LABELS  # noqa: E402

CAPS = Path("data/captions")
STOP = {"the", "a", "an", "is", "are", "and", "of", "on", "in", "with", "to", "there",
        "it", "its", "this", "that", "image", "shows", "appears", "be", "at", "has",
        "have", "from", "by", "as", "or", "also", "which", "while", "her", "his",
        "she", "he", "they", "them", "their", "background", "foreground", "left",
        "right", "side", "front", "behind", "next", "near", "above", "below"}


def ngrams(text, n):
    words = [w for w in re.findall(r"[a-z]+", text.lower()) if w not in STOP]
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def fisher(a, b, c, d):
    """Односторонний точный тест Фишера: a из (a+c) против b из (b+d)."""
    n = a + b + c + d
    p = 0.0
    for i in range(a, min(a + b, a + c) + 1):
        j = a + c - i
        if j < 0 or j > b + d:
            continue
        p += comb(a + b, i) * comb(c + d, j) / comb(n, a + c)
    return p


def main():
    docs = {}
    for f in sorted(CAPS.glob("*.caps.json")):
        d = json.loads(f.read_text())
        if d["video"] not in LABELS:
            continue
        docs[d["video"]] = " ".join(c["caption"] for c in d["captions"])
    if not docs:
        print("корпус подписей пуст"); return

    viol = [n for n in docs if LABELS[n]]
    clean = [n for n in docs if not LABELS[n]]
    print(f"роликов с подписями: {len(docs)} (нарушений {len(viol)}, чистых {len(clean)})\n")

    rows = []
    for n in (1, 2):
        sets = {k: ngrams(v, n) for k, v in docs.items()}
        vocab = Counter()
        for s in sets.values():
            vocab.update(s)
        for term, _ in vocab.most_common():
            a = sum(term in sets[k] for k in viol)
            b = sum(term in sets[k] for k in clean)
            if a + b < 3 or (a <= 1 and b <= 1):
                continue
            pv = fisher(a, b, len(viol) - a, len(clean) - b)
            pc = fisher(b, a, len(clean) - b, len(viol) - a)
            rows.append((min(pv, pc), "V" if pv < pc else ".", term, a, b))

    rows.sort(key=lambda r: r[0])
    print(f"{'p':>8s} {'класс':5s} {'нарушения':>10s} {'чистые':>8s}  фраза")
    for p, cls, term, a, b in rows[:40]:
        print(f"{p:8.4f} {cls:5s} {a:6d}/{len(viol):<3d} {b:5d}/{len(clean):<3d}  {term}")


if __name__ == "__main__":
    main()
