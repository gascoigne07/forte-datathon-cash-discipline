"""Оркестрация анализа одного видео.

Каскад «дешёвое -> дорогое», чтобы уложиться в лимит 5 минут на видео:
декодирование и детектор людей стоят десятки миллисекунд на кадр, описание
кадра моделью — секунды, поэтому модель запускается только по ключевым кадрам.
"""
from __future__ import annotations

import os
import time
from typing import Callable

from . import caption, objects, perception, rules, vlm

BUDGET_SEC = float(os.environ.get("ANALYZE_BUDGET_SEC", 260))
SAMPLE_FPS = 2.0
KF_STEP_SEC = float(os.environ.get("KF_STEP_SEC", 1.5))
KF_HARD_CAP = int(os.environ.get("KF_CAP", 45))
# Кадров за один вызов generate. На L40S батч 16 даёт 0.09 с/кадр против 0.49 с
# покадрово (замер, шаг G1 в NIGHT-LOG); подписи совпадают в 15 из 16. По
# умолчанию 1: на MPS батч не проверялся, включается в юните GPU-сервера.
CAPTION_BATCH = int(os.environ.get("CAPTION_BATCH", 1))
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
    zone, zone_source = objects.resolve_desk_zone(frames, bg=bg)
    tracks = objects.detect_objects(frames, zone=zone)

    progress(0.22, "выбор ключевых кадров")
    # Нижняя граница 18, а не 12: перебор по прореженному корпусу показал, что до
    # 18 кадров метрика держится (вердикт 72 %), на 15 появляется второе ложное
    # срабатывание, а на 12-13 теряются Х-4, Х-26 и Х-40 (вердикт 65-68 %). Именно
    # это и случилось на приёмочном прогоне пяти видео: Х-40 получил 13 кадров и
    # нарушение не нашлось. Граница безопасна, потому что реальным ограничителем
    # остаётся абсолютный дедлайн ниже: не хватит времени — цикл оборвётся сам и
    # выставит low_coverage, а не вылезет за 300 с.
    cap = KF_HARD_CAP if queued <= 0 else max(18, int(KF_HARD_CAP / (1 + queued)))
    idx = select_keyframes(frames, tracks, cap)

    progress(0.25, "описание кадров моделью")
    series: list[tuple[float, dict]] = []
    per_frame: list[float] = []                         # стоимость кадра по каждому батчу
    reserve = 15.0                                      # на сборку ответа
    MIN_FRAMES = 8                                      # ниже этого вердикт недостоверен
    n = 0
    while n < len(idx):
        # Жёсткий стоп: бюджет кончился — выходим независимо от числа кадров.
        # Иначе при дорогих первых кадрах цикл не прерывался до MIN_FRAMES и
        # обработка вылетала за 300 с, то есть видео засчитывалось необработанным.
        if left() < reserve:
            break
        chunk = idx[n:n + CAPTION_BATCH]
        # Предсказательный стоп: стоимость кадра берём по медиане уже посчитанных
        # батчей, а не по первому — в первый попадает загрузка весов, из-за чего
        # страж обрубал анализ после одного кадра (наблюдалось на Х-40__dark).
        # Не влезает весь батч — укорачиваем его, а не бросаем целиком.
        if per_frame and len(series) >= MIN_FRAMES:
            med = sorted(per_frame)[len(per_frame) // 2]
            fit = int((left() - reserve) / med)
            if fit <= 0:
                break
            chunk = chunk[:fit]
        crops = [objects.crop_object(frames[i].img, zone, target_px=DESK_CROP_PX, pad_rel=0.05)
                 for i in chunk]
        t_batch = time.time()
        try:
            texts = caption.caption_batch(crops)
        except Exception:                               # noqa: BLE001
            n += len(chunk)
            continue
        per_frame.append((time.time() - t_batch) / len(chunk))
        for i, text in zip(chunk, texts):
            series.append((frames[i].t, frame_facts(frames[i], text, zone, tracks)))
        n += len(chunk)
        progress(0.25 + 0.6 * n / max(1, len(idx)), "описание кадров моделью")

    # Второй описатель — по тем же ключевым кадрам, что успели получить подпись.
    # Он дополняет факты, а не заменяет: не хватило времени — правила работают на
    # фактах Florence-2 и детектора, как без него.
    if vlm.ENABLED and series:
        progress(0.80, "вопросы второй модели")
        done = {round(t, 3) for t, _ in series}
        vidx = [i for i in idx if round(frames[i].t, 3) in done]
        pos = {round(t, 3): k for k, (t, _) in enumerate(series)}
        vcost: list[float] = []
        for s0 in range(0, len(vidx), vlm.BATCH):
            chunk = vidx[s0:s0 + vlm.BATCH]
            need = reserve + (max(vcost) * len(chunk) if vcost else 0.0)
            if left() < need:
                break
            t_batch = time.time()
            try:
                answers = vlm.ask([frames[i].img for i in chunk])
            except Exception:                           # noqa: BLE001
                break
            vcost.append((time.time() - t_batch) / len(chunk))
            for i, raw in zip(chunk, answers):
                series[pos[round(frames[i].t, 3)]][1].update(vlm.parse(raw))

    progress(0.87, "применение правил")
    events = rules.detect_events(series, tracks=tracks)

    progress(0.92, "подготовка доказательств")
    out = []
    for e in events:
        out.append({
            "type": e.type,
            "timecode_sec": e.t,
            "description": _describe(e, tracks),
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
        "zone_source": zone_source,
    }


def select_keyframes(frames, tracks, cap: int) -> list[int]:
    """Кадры для описателя: сетка + смена числа людей + пики движения, плюс
    обязательные моменты появления предметов.

    Вынесено в функцию, чтобы оценочный харнесс выбирал кадры тем же кодом, что и
    сервис: за ночь расхождение харнесса с сервисом трижды давало замеры
    конфигурации, которой не существует.
    """
    idx = perception.keyframes(frames, budget=cap, step_sec=KF_STEP_SEC)
    for tr in tracks:                                   # моменты появления предметов — обязательно
        j = min(range(len(frames)), key=lambda i: abs(frames[i].t - tr.t_start))
        if j not in idx:
            idx.append(j)
    return sorted(set(idx))


def frame_facts(f, text: str, zone, tracks) -> dict:
    """Факты одного кадра: разбор подписи + детектор + треки. Одна функция на
    сервис и харнесс (tools/gpu_corpus.py, tools/eval_corpus.py)."""
    facts = caption.facts_from_caption(text)
    ppl = perception.persons_in_room(f, desk_top=zone[1])
    facts["people_in_cabin"] = max(len(ppl), caption.count_people(text))
    facts["teller_present"] = _teller_at_desk(ppl, zone)
    # Кто за столешницей и кто в комнате, но не за ней. По этому признаку
    # работает правило о пункте 22: «покидать рабочее место до окончания
    # обслуживания клиента не допускается».
    facts["at_desk"], facts["off_desk"] = perception.split_by_desk(ppl, zone)
    # Независимый от модели канал: в этот момент на столешнице есть предмет,
    # появившийся по ходу записи (вычитание фона). Нужен как второе
    # подтверждение к «в описании кадра нет ни одного человека».
    facts["track_on_desk"] = any(
        t.t_start - 3 <= f.t <= t.t_end + 3 and t.area >= TRACK_MIN_AREA
        for t in tracks)
    return facts


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
        "в помещении находятся трое и более человек, на столешнице есть предметы",
        "доступ посторонних в кассовое помещение во время операций с наличностью "
        "ограничен требованиями к устройству кассового узла"),
    "outsider_vlm": (
        "по оценке второй модели в помещении находятся трое и более человек",
        "доступ посторонних в кассовое помещение во время операций с наличностью "
        "ограничен требованиями к устройству кассового узла"),
    "left_via_door": (
        "по оценке второй модели работник выходит через дверь помещения, его стул пуст",
        "кассовому работнику не допускается покидать рабочее место до окончания "
        "обслуживания клиента"),
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


# Факт -> как он называется по-русски в описании. Описание собирается ИЗ ФАКТОВ,
# а английский текст описателя в ответ не попадает: он служебный, проверяющему нужен
# разбор на его языке, а не пересказ модели.
_FACT_RU = {
    "drink_or_food": "посуда с напитком или еда",
    "personal_bag": "личная сумка или пакет",
    "sealed_bag": "инкассаторская сумка",
    "paper_seen": "документы",
    "machine_seen": "счётно-денежная машина",
}
_ON_DESK = {"personal_bag": "bag_ctx", "sealed_bag": "sealed_ctx"}
# 400 символов не хватало: на `Х-11` и `Х-13` лимит срезал именно отсутствие
# работника — главный факт события «рабочее место без контроля» — с перевесом
# в два символа. 460 вмещает полный перечень на всех 13 событиях выборки.
DESC_MAX = 460


def _observed(f: dict, appeared: float | None) -> list[str]:
    """Наблюдаемое на кадре события — только из признаков, без текста модели.

    Порядок важен: при превышении лимита длины отрезается хвост списка, поэтому
    предметы идут первыми, а служебные уточнения — последними.
    """
    out: list[str] = []
    seen = []
    for k, ru in _FACT_RU.items():
        if not f.get(k):
            continue
        ctx = _ON_DESK.get(k)
        seen.append(f"{ru} на столешнице" if ctx and f.get(ctx) == "on_desk" else ru)
    if seen:
        out.append("в кадре: " + ", ".join(seen))

    # Два независимых канала о присутствии людей могут расходиться: детектор видит
    # человека через стеклянную стену, а в зоне столешницы его нет. Пишем это прямо,
    # иначе проверяющий читает описание как противоречивое.
    n = f.get("people_in_cabin")
    if f.get("no_person_in_caption"):
        out.append(f"у столешницы работника нет (детектор видит {n} чел. в кадре)"
                   if n else "людей в кадре не обнаружено")
    elif n is not None:
        out.append("людей в кадре не обнаружено" if n == 0 else f"людей в кадре: {n}")
        if f.get("teller_present"):
            out.append("работник у столешницы")
    if f.get("people_in_booth_vlm") is not None:
        out.append(f"людей в помещении по второй модели: {f['people_in_booth_vlm']}")

    if appeared is not None:
        out.append(f"предмет появился на столешнице на {appeared:.1f} с")
    return out


def _describe(e: rules.Event, tracks=None) -> str:
    what, why = _PHRASES.get(
        e.key, ("зафиксирован наблюдаемый признак", "требуется проверка человеком"))
    w0, w1 = e.window
    span = (f"{w0:.1f}–{w1:.1f} с" if w1 > w0 else f"{w0:.1f} с")

    appeared = None
    near = [t for t in (tracks or [])
            if t.t_start > 1.0 and abs(t.t_start - e.t) <= 3.0]
    if near:
        appeared = min(near, key=lambda t: abs(t.t_start - e.t)).t_start

    obs = _observed(e.facts_at_t or {}, appeared)
    tail = (f" Почему это может указывать на нарушение: {why}."
            " Окончательную оценку даёт проверяющий.")
    head = f"Что видно ({span}, кадров с признаком: {e.support}): {what}"
    # Перечень фактов отрезается первым: он уточняет, а связь с нормой — обязательна.
    while obs and len(head) + len(" — " + "; ".join(obs) + ".") + len(tail) > DESC_MAX:
        obs.pop()
    if obs:
        head += " — " + "; ".join(obs)
    return head + "." + tail
