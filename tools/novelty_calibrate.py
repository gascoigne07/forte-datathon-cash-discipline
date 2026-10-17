"""Калибровка канала новизны строго по leave-one-video-out.

Скор кадра = среднее косинусное расстояние до k ближайших эмбеддингов в банке нормы.
Порог подбирается так, чтобы в LOO на чистых роликах не было НИ ОДНОГО ложного
срабатывания: для каждого чистого ролика банк собирается БЕЗ его собственных
кадров. Без этого ноль ложных получился бы по построению и ничего не значил.

Затем тем же порогом измеряется, сколько нарушений канал ловит и с какой Δt.
"""
import glob, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import LABELS  # noqa: E402

EMB = Path("data/embeds")
DUPLICATES = {"Х-47", "Х-48"}
K = 5
MIN_RUN = 2          # событие: порог превышен на >=2 соседних ключевых кадрах


def load_all():
    data = {}
    for p in sorted(EMB.glob("*.npz")):
        name = p.stem
        if name not in LABELS or name in DUPLICATES:
            continue
        d = np.load(p)
        if len(d["emb"]):
            data[name] = (d["emb"], d["t"])
    return data


def scores(emb, bank):
    """Среднее расстояние до K ближайших. Эмбеддинги нормированы -> 1 - косинус."""
    sim = emb @ bank.T
    k = min(K, bank.shape[0])
    top = np.partition(sim, -k, axis=1)[:, -k:]
    return 1.0 - top.mean(axis=1)


def runs_over(sc, thr, min_run=MIN_RUN):
    """Индексы начала серий, где скор выше порога min_run кадров подряд."""
    out, cur, start = [], 0, 0
    for i, v in enumerate(sc):
        if v > thr:
            if cur == 0:
                start = i
            cur += 1
            if cur == min_run:
                out.append((start, i))
        else:
            if cur >= min_run:
                out[-1] = (out[-1][0], i - 1)
            cur = 0
    return out


def main():
    data = load_all()
    clean = [n for n in data if not LABELS[n]]
    viol = [n for n in data if LABELS[n]]
    print(f"банк нормы: {len(clean)} чистых роликов, проверяем на {len(viol)} нарушениях\n")

    # LOO: для каждого чистого ролика — скоры по банку без него самого
    loo_scores = {}
    for n in clean:
        bank = np.concatenate([data[m][0] for m in clean if m != n])
        loo_scores[n] = scores(data[n][0], bank)

    full_bank = np.concatenate([data[n][0] for n in clean])
    viol_scores = {n: scores(data[n][0], full_bank) for n in viol}

    # максимум по чистым в LOO задаёт нижнюю границу безопасного порога
    worst_clean = max(float(s.max()) for s in loo_scores.values())
    print(f"максимальный LOO-скор на чистых: {worst_clean:.4f}")
    print(f"{'порог':>8s} {'ложных LOO':>11s} {'найдено наруш.':>15s} {'Δt медиана':>11s}")

    results = []
    for thr in np.quantile(np.concatenate(list(loo_scores.values())),
                           [0.90, 0.95, 0.98, 0.99, 0.995, 1.0]):
        fp = sum(1 for n in clean if runs_over(loo_scores[n], thr))
        hits, errs = [], []
        for n in viol:
            rr = runs_over(viol_scores[n], thr)
            if rr:
                hits.append(n)
                ts = [float(data[n][1][lo]) for lo, _ in rr]
                errs.append(min(abs(t - g) for t in ts for g in LABELS[n]))
        med = float(np.median(errs)) if errs else -1
        results.append((thr, fp, len(hits), med, hits))
        print(f"{thr:8.4f} {fp:11d} {len(hits):15d} {med:11.1f}")

    ok = [r for r in results if r[1] == 0]
    if ok:
        best = max(ok, key=lambda r: r[2])
        print(f"\nвыбран порог {best[0]:.4f}: ложных в LOO 0, найдено {best[2]} из {len(viol)}")
        print(f"ловит: {','.join(sorted(best[4]))}")
        np.savez("data/novelty_thr.npz", thr=np.float32(best[0]), k=np.int32(K),
                 min_run=np.int32(MIN_RUN))
        print("порог сохранён в data/novelty_thr.npz")
    else:
        print("\nпорога с нулём ложных в LOO не существует — канал не годится")


if __name__ == "__main__":
    main()
