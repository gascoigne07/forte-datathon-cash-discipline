"""Канал новизны: кадр, непохожий на банк нормы, как «иное отклонение».

Задуман как ответ на п.9 FAQ: в тесте будут типы нарушений, которых нет в
обучающих данных, и правила под них написать нельзя по определению.

ИЗМЕРЕННЫЙ РЕЗУЛЬТАТ — канал не разделяет классы и ПО УМОЛЧАНИЮ ВЫКЛЮЧЕН.
Калибровка строго leave-one-video-out (`tools/novelty_calibrate.py`): для каждого
чистого ролика банк собирается без его собственных кадров. При пороге, дающем
ноль ложных срабатываний в LOO, канал находит 0 нарушений из 21; чтобы найти
5 нарушений, порог приходится опустить до 5 ложных из 19. Кадры с нарушениями
в пространстве SigLIP не «новее» чистых — то же самое показал более ранний замер
обучения на этих эмбеддингах (leave-one-out 48 % при 50 % у монетки).

Код оставлен рабочим: включается переменной окружения NOVELTY=1 и пригодится,
если появится банк нормы с другого объекта или более подходящий энкодер.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

BANK_PATH = Path(os.environ.get("NORMAL_BANK", "data/normal_bank.npz"))
ENABLED = os.environ.get("NOVELTY", "0") == "1"
K = 5
MIN_RUN = 2

_BANK: np.ndarray | None = None
_THR: float | None = None


def available() -> bool:
    return ENABLED and BANK_PATH.exists()


def _load():
    global _BANK, _THR
    if _BANK is None:
        d = np.load(BANK_PATH)
        _BANK = d["bank"].astype(np.float32)
        _THR = float(d["thr"])
    return _BANK, _THR


def score_frames(embeddings: np.ndarray) -> np.ndarray:
    """Среднее косинусное расстояние до K ближайших в банке нормы."""
    bank, _ = _load()
    sim = embeddings @ bank.T
    k = min(K, bank.shape[0])
    top = np.partition(sim, -k, axis=1)[:, -k:]
    return 1.0 - top.mean(axis=1)


def find_runs(scores: np.ndarray, thr: float | None = None) -> list[tuple[int, int]]:
    """Серии кадров, где скор выше порога не менее MIN_RUN подряд."""
    if thr is None:
        _, thr = _load()
    out: list[tuple[int, int]] = []
    cur = start = 0
    for i, v in enumerate(scores):
        if v > thr:
            if cur == 0:
                start = i
            cur += 1
            if cur == MIN_RUN:
                out.append((start, i))
        else:
            if cur >= MIN_RUN and out:
                out[-1] = (out[-1][0], i - 1)
            cur = 0
    return out
