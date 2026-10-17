"""Оркестрация анализа одного видео.

Каскад «дешёвое -> дорогое», чтобы уложиться в лимит 5 минут на видео:
декодирование и детектор людей стоят десятки миллисекунд на кадр, описание
кадра моделью — секунды, поэтому модель запускается только по ключевым кадрам.
"""
from __future__ import annotations

import os
import time
from typing import Callable

from . import caption, objects, perception, rules

BUDGET_SEC = float(os.environ.get("ANALYZE_BUDGET_SEC", 260))
SAMPLE_FPS = 2.0
KF_STEP_SEC = float(os.environ.get("KF_STEP_SEC", 1.5))
KF_HARD_CAP = int(os.environ.get("KF_CAP", 45))
DESK_CROP_PX = 430_000

Progress = Callable[[float, str], None]


def _noop(p: float, s: str) -> None:  # pragma: no cover
    pass


def analyze(path: str, progress: Progress = _noop) -> dict:
    t0 = time.time()
    left = lambda: BUDGET_SEC - (time.time() - t0)      # noqa: E731

    progress(0.03, "чтение видео")
    duration, fps = perception.probe(path)
    frames = perception.sample(path, fps=SAMPLE_FPS)
    if not frames:
        raise ValueError("не удалось декодировать видео")

    progress(0.10, "детекция людей и предметов")
    perception.detect(frames)

    progress(0.18, "разметка сцены")
    bg = objects.background(frames)
    zone = objects.desk_zone(frames, bg=bg)
    if zone is None:                                   # запасная зона: нижние две трети
        h, w = frames[0].img.shape[:2]
        zone = (0, int(0.35 * h), w, h)
    tracks = objects.detect_objects(frames, zone=zone)

    progress(0.22, "выбор ключевых кадров")
    idx = perception.keyframes(frames, budget=KF_HARD_CAP, step_sec=KF_STEP_SEC)
    for tr in tracks:                                   # моменты появления предметов — обязательно
        j = min(range(len(frames)), key=lambda i: abs(frames[i].t - tr.t_start))
        if j not in idx:
            idx.append(j)
    idx = sorted(set(idx))

    progress(0.25, "описание кадров моделью")
    series: list[tuple[float, dict]] = []
    durations: list[float] = []
    reserve = 15.0                                      # на сборку ответа
    MIN_FRAMES = 8                                      # ниже этого вердикт недостоверен
    for n, i in enumerate(idx):
        # Стоимость кадра оцениваем по медиане уже посчитанных, а не по первому:
        # в первый кадр попадает загрузка весов, и страж бюджета обрубал анализ
        # после одного кадра (наблюдалось на Х-40__dark: кадров=1, 210 с).
        if durations and len(series) >= MIN_FRAMES:
            med = sorted(durations)[len(durations) // 2]
            if left() < reserve + med:
                break
        f = frames[i]
        crop = objects.crop_object(f.img, zone, target_px=DESK_CROP_PX, pad_rel=0.05)
        t_frame = time.time()
        try:
            text = caption.caption(crop)
        except Exception:                               # noqa: BLE001
            continue
        durations.append(time.time() - t_frame)
        facts = caption.facts_from_caption(text)
        ppl = perception.persons_in_room(f, desk_top=zone[1])
        facts["people_in_cabin"] = max(len(ppl), caption.count_people(text))
        facts["teller_present"] = _teller_at_desk(ppl, zone)
        series.append((f.t, facts))
        progress(0.25 + 0.6 * (n + 1) / max(1, len(idx)), "описание кадров моделью")

    progress(0.87, "применение правил")
    events = rules.detect_events(series, tracks=tracks)

    progress(0.92, "подготовка доказательств")
    out = []
    for e in events:
        out.append({
            "type": e.type,
            "timecode_sec": e.t,
            "description": _describe(e),
            "evidence_frames": _evidence(frames, e),
            "confidence": e.confidence,
            "rule": rules.rule_ref(e.key),
            "window_sec": list(e.window),
        })

    progress(1.0, "готово")
    low_coverage = len(series) < MIN_FRAMES
    return {
        "verdict": "violation" if out else "no_violation",
        "low_coverage": low_coverage,
        "violations": out,
        "video": {"duration_sec": round(duration, 2), "fps": round(fps, 2)},
        "processing_sec": round(time.time() - t0, 1),
        "frames_analyzed": len(series),
        "frames_total_sampled": len(frames),
    }


def _teller_at_desk(persons, zone) -> bool:
    """Работник на месте = есть человек, чей бокс пересекается с зоной столешницы."""
    zx1, zy1, zx2, zy2 = zone
    for p in persons:
        x1, y1, x2, y2 = p["box"]
        if x1 < zx2 and x2 > zx1 and y1 < zy2 and y2 > zy1:
            return True
    return False


def _evidence(frames, e: rules.Event, k: int = 2) -> list[dict]:
    """1-2 кадра из окна события: сам таймкод и середина окна."""
    want = [e.t]
    mid = (e.window[0] + e.window[1]) / 2
    if abs(mid - e.t) > 1.0:
        want.append(mid)
    out = []
    for t in want[:k]:
        f = min(frames, key=lambda x: abs(x.t - t))
        if f.img is None:
            continue
        out.append({"timestamp_sec": round(f.t, 2),
                    "image_base64": perception.jpeg_b64(f.img)})
    return out


# Описание собирается из фактов детерминированно. Подпись модели на английском
# и может ошибаться в деталях, поэтому в текст для проверяющего она не попадает:
# юридически значимый вывод не должен опираться на вольный пересказ модели.
_PHRASES = {
    "food_drink": "на рабочем месте кассира виден напиток или еда",
    "personal_bag": "на рабочем месте кассира видна личная сумка или пакет",
    "sealed_bag": "в кадре видна инкассаторская сумка вне сейфа",
    "cash_unattended": "на столешнице видны наличные деньги, работника рядом с рабочим местом нет",
    "left_workplace": "работник отсутствует на рабочем месте, при этом в кадре фигурируют ценности",
    "outsider_with_cash": "в помещении находятся трое и более человек одновременно с наличными на столе",
    "manual_count": "наличные пересчитываются в руках, счётно-денежная машина не задействована",
    "unattended_desk": "работника на рабочем месте не видно, при этом на столешнице остаются предметы",
    "other_anomaly": "зафиксировано отклонение от типичной картины рабочего места",
}


def _describe(e: rules.Event) -> str:
    base = _PHRASES.get(e.key, "зафиксирован наблюдаемый признак")
    w0, w1 = e.window
    span = f"Признак наблюдается с {w0:.1f} по {w1:.1f} с" if w1 > w0 else f"Признак наблюдается на {w0:.1f} с"
    conf = ("подтверждён на нескольких кадрах" if e.support >= 3
            else "подтверждён на двух кадрах")
    return (f"На кадре {base}. {span} ({conf}). "
            f"Вывод носит характер подсказки: окончательную оценку даёт проверяющий.")
