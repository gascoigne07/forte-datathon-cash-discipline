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
# Порог площади трека подобран перебором на обучающей выборке (40 уникальных
# роликов): при 45 правило давало 2 ложных срабатывания из 19 чистых, при 60 —
# одно. Ложные приходятся на кадры, где подпись перечисляет только мебель и
# технику, а трек мелкий (площадь 30-50 против 68-254 у настоящих срабатываний).
TRACK_MIN_AREA = 60

Progress = Callable[[float, str], None]


def _noop(p: float, s: str) -> None:  # pragma: no cover
    pass


def analyze(path: str, progress: Progress = _noop, queued: int = 0,
            deadline_ts: float | None = None) -> dict:
    """queued — сколько заданий ждёт за этим: чем длиннее очередь, тем меньше кадров.

    deadline_ts — абсолютный момент, к которому обязан быть готов результат.
    Жюри меряет пятиминутный лимит от отправки видео, а не от старта обработки,
    поэтому время, проведённое в очереди, вычитается из бюджета этого видео.
    В прогоне пяти роликов подряд последний финишировал на 285-й секунде —
    без абсолютного дедлайна шестой уже не успел бы.
    """
    t0 = time.time()
    if deadline_ts is None:
        left = lambda: BUDGET_SEC - (time.time() - t0)   # noqa: E731
    else:
        left = lambda: deadline_ts - time.time()         # noqa: E731

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
    cap = KF_HARD_CAP if queued <= 0 else max(12, int(KF_HARD_CAP / (1 + queued)))
    idx = perception.keyframes(frames, budget=cap, step_sec=KF_STEP_SEC)
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
        # Жёсткий стоп: бюджет кончился — выходим независимо от числа кадров.
        # Иначе при дорогих первых кадрах цикл не прерывался до MIN_FRAMES и
        # обработка вылетала за 300 с, то есть видео засчитывалось необработанным.
        if left() < reserve:
            break
        # Предсказательный стоп: стоимость кадра берём по медиане уже посчитанных,
        # а не по первому — в первый кадр попадает загрузка весов, из-за чего
        # страж обрубал анализ после одного кадра (наблюдалось на Х-40__dark).
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
        # Независимый от модели канал: в этот момент на столешнице есть предмет,
        # появившийся по ходу записи (вычитание фона). Нужен как второе
        # подтверждение к «в описании кадра нет ни одного человека».
        facts["track_on_desk"] = any(
            t.t_start - 3 <= f.t <= t.t_end + 3 and t.area >= TRACK_MIN_AREA
            for t in tracks)
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
            "evidence_frames": _evidence(frames, e, zone, tracks),
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
        "queued_behind": queued,
    }


def _teller_at_desk(persons, zone) -> bool:
    """Работник на месте = есть человек, чей бокс пересекается с зоной столешницы."""
    zx1, zy1, zx2, zy2 = zone
    for p in persons:
        x1, y1, x2, y2 = p["box"]
        if x1 < zx2 and x2 > zx1 and y1 < zy2 and y2 > zy1:
            return True
    return False


def _evidence(frames, e: rules.Event, zone, tracks, k: int = 3) -> list[dict]:
    """До трёх кадров (ТЗ раздел 10 разрешает 1-3): сам таймкод плюс до и после.

    На кадре отмечается зона, к которой относится признак. ТЗ требует кадр,
    «позволяющий человеку проверить вывод» — без отметки проверяющему пришлось бы
    искать мелкий предмет на столешнице глазами.
    """
    w0, w1 = e.window
    want = [e.t]
    for extra in (w0 - 1.5, w1 + 1.5, (w0 + w1) / 2):
        if all(abs(extra - t) > 1.0 for t in want) and 0 <= extra:
            want.append(extra)
    want = sorted(want[:k])

    # Область для отметки: бокс ближайшего трека предмета, иначе зона столешницы.
    box = zone
    near = [t for t in (tracks or []) if t.t_start - 3 <= e.t <= t.t_end + 3]
    if near:
        box = min(near, key=lambda t: abs(t.t_start - e.t)).box

    out = []
    for t in want:
        f = min(frames, key=lambda x: abs(x.t - t))
        if f.img is None:
            continue
        img = perception.annotate(f.img, box, f.t, highlight=abs(f.t - e.t) < 0.6)
        out.append({"timestamp_sec": round(f.t, 2),
                    "image_base64": perception.jpeg_b64(img)})
    return out


# Описание собирается из фактов детерминированно. Подпись модели на английском
# и может ошибаться в деталях, поэтому в текст для проверяющего она не попадает:
# юридически значимый вывод не должен опираться на вольный пересказ модели.
#
# Каждая запись — две части, как требует раздел 4 п.6 ТЗ: ЧТО видно на кадре
# и ПОЧЕМУ это может указывать на нарушение. Вторая часть — связь с требованием
# норматива (раздел 6 ТЗ требует показать связь источник -> нарушение -> фрагмент).
_PHRASES = {
    "food_drink": (
        "на рабочем месте кассира виден напиток или еда",
        "перечень предметов, которые разрешено вносить в кассовый узел, закрыт и "
        "посуды с напитками не содержит"),
    "personal_bag": (
        "на рабочем месте кассира видна личная сумка или пакет",
        "личные вещи работника должны храниться вне кассового узла"),
    "sealed_bag": (
        "в кадре видна инкассаторская сумка вне сейфа",
        "сумка с ценностями вне операции приёма-передачи должна находиться в "
        "денежном хранилище, а не на столешнице"),
    "cash_unattended": (
        "на столешнице видны наличные деньги, работника рядом с рабочим местом нет",
        "наличные в течение операционного дня должны находиться под контролем "
        "кассира, иначе утрата или подмена не отслеживается"),
    "left_workplace": (
        "работник отсутствует на рабочем месте, при этом в кадре фигурируют ценности",
        "при оставлении рабочего места ценности убираются, а доступ к ним закрывается"),
    "outsider_with_cash": (
        "в помещении находятся трое и более человек одновременно с наличными на столе",
        "доступ посторонних в кассовое помещение во время операций с наличностью "
        "ограничен требованиями к устройству кассового узла"),
    "manual_count": (
        "наличные пересчитываются в руках, счётно-денежная машина не задействована",
        "пересчёт принимаемых и выдаваемых наличных предусмотрен с применением "
        "счётно-денежного оборудования"),
    "unattended_desk": (
        "работника на рабочем месте не видно, при этом на столешнице остаются предметы",
        "рабочее место кассира не должно оставаться без контроля с находящимися "
        "на нём предметами и ценностями"),
    "other_anomaly": (
        "зафиксировано отклонение от типичной картины рабочего места",
        "отклонение требует проверки человеком, автоматический вывод здесь не делается"),
}


def _describe(e: rules.Event) -> str:
    what, why = _PHRASES.get(
        e.key, ("зафиксирован наблюдаемый признак", "требуется проверка человеком"))
    w0, w1 = e.window
    span = (f"с {w0:.1f} по {w1:.1f} с" if w1 > w0 else f"на {w0:.1f} с")
    conf = ("признак подтверждён на нескольких кадрах" if e.support >= 3
            else "признак подтверждён на двух кадрах")
    return (f"Что видно: {what} ({span}, {conf}). "
            f"Почему это может указывать на нарушение: {why}. "
            f"Вывод носит характер подсказки — окончательную оценку даёт проверяющий.")
