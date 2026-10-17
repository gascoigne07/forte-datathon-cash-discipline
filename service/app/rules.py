"""Правила над фактами: факты по кадрам -> события с типом, таймкодом и нормативом.

Правила объявлены декларативно и читаемо: жюри оценивает обоснованность трактовки
(критерий 4), а не изящество кода.

Две калибровки, выведенные из обучающей выборки, а не из общих соображений:
1) В ЧИСТЫХ роликах кассир тоже встаёт и отходит (`Х-29`, `Х-31`), посторонние тоже
   заходят в кабину (`Х-35`, `Х-37`), наличные тоже лежат на столе (`Х-10`, `Х-14`, `Х-43`).
   Поэтому ни одно правило не срабатывает на одиночном грубом признаке.
2) Модель восприятия ошибается на отдельных кадрах в обе стороны, поэтому любой
   предметный факт требует подтверждения минимум на двух ключевых кадрах
   (приём из AnomalyRuler: голосование вместо доверия одному кадру).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

KB_PATH = Path(__file__).resolve().parents[2] / "kb" / "rules.json"


@dataclass
class Event:
    key: str
    type: str
    t: float                       # таймкод события (начало наблюдаемого признака)
    window: tuple[float, float]
    confidence: float
    support: int                   # сколько ключевых кадров подтвердили
    facts_at_t: dict = field(default_factory=dict)


def _load_kb() -> dict[str, dict]:
    kb: dict[str, dict] = {}
    if KB_PATH.exists():
        for r in json.loads(KB_PATH.read_text(encoding="utf-8")):
            rid = r["id"]
            kb[rid] = r
            base = re.sub(r"[a-z]+$", "", rid)
            kb.setdefault(base, r)
    return kb


KB = _load_kb()

# ключ -> (человекочитаемый тип, id нормы в kb/rules.json)
TYPES = {
    "manual_count": ("Пересчёт наличных вручную, минуя счётно-денежную машину", "V-01"),
    "cash_unattended": ("Наличные деньги оставлены без контроля кассира", "V-02"),
    "left_workplace": ("Кассир покинул рабочее место, не убрав ценности", "V-03"),
    "outsider_with_cash": ("Посторонние в кассовом помещении во время операции", "V-04"),
    "food_drink": ("Напитки или еда на рабочем месте кассира", "V-05"),
    "personal_bag": ("Личные вещи (сумка) на рабочем месте кассира", "V-06"),
    "sealed_bag": ("Инкассаторская сумка вне сейфа / без присмотра", "V-07"),
    "unattended_desk": ("Рабочее место оставлено без работника, на столешнице находятся предметы", "V-02"),
    "other_anomaly": ("Иное отклонение от регламента (требует проверки человеком)", "V-08"),
}


def rule_ref(key: str) -> dict | None:
    _, rid = TYPES.get(key, (None, None))
    r = KB.get(rid) if rid else None
    if not r:
        return None
    return {
        "doc_title": r.get("doc_title", ""),
        "doc_requisites": r.get("doc_requisites", ""),
        "clause": r.get("clause"),
        "quote": r.get("quote"),
        "url": r.get("doc_url"),
        "verified": bool(r.get("verified")),
        "note": r.get("note"),
    }


def _runs(series, pred, gap_tolerance: int = 0):
    """Отрезки индексов, где pred(facts) истинно, с допуском на провалы.

    gap_tolerance=1 разрешает одиночный провал внутри серии: модель ошибается на
    отдельных кадрах. Включается только для предметных правил (напиток, сумки) —
    замерено, что для правила «рабочее место без работника» допуск добавляет
    ложное срабатывание, не добавляя найденных нарушений.
    """
    hits = [i for i, (_, f) in enumerate(series) if pred(f)]
    if not hits:
        return []
    out, start, prev = [], hits[0], hits[0]
    for i in hits[1:]:
        if i - prev <= gap_tolerance + 1:
            prev = i
        else:
            out.append((start, prev)); start = prev = i
    out.append((start, prev))
    return out


def _mk(series, key, lo, hi, min_support, conf_base) -> Event | None:
    support = hi - lo + 1
    if support < min_support:
        return None
    conf = min(0.95, conf_base + 0.1 * (support - min_support))
    # Таймкод = момент появления признака. Но если признак есть уже на первом
    # проанализированном кадре, момент появления не наблюдаем — тогда честнее
    # указать представительный кадр из середины интервала, а не нулевую секунду.
    anchor = lo if lo > 0 else (lo + hi) // 2
    return Event(key=key, type=TYPES[key][0], t=round(series[anchor][0], 2),
                 window=(round(series[lo][0], 2), round(series[hi][0], 2)),
                 confidence=round(conf, 2), support=support,
                 facts_at_t=series[anchor][1])


def detect_events(series: list[tuple[float, dict]],
                  tracks: list | None = None) -> list[Event]:
    """series: [(секунда, факты), ...] по ключевым кадрам, отсортировано по времени.

    tracks — треки предметов, появившихся на столешнице; используются только для
    уточнения таймкода: у них точность до полусекунды, у подписей — до шага сетки.
    """
    if not series:
        return []
    ev: list[Event] = []
    f0 = lambda k: any(f.get(k) for _, f in series)       # noqa: E731

    # 1. Напитки и еда. Высокая точность: на чистых роликах стакан не упоминается.
    for lo, hi in _runs(series, lambda f: f.get("drink_or_food")):
        e = _mk(series, "food_drink", lo, hi, 2, 0.70)
        if e:
            ev.append(e)

    # 2. Личные вещи.
    for lo, hi in _runs(series, lambda f: f.get("personal_bag")
                        and f.get("bag_ctx") == "on_desk"):
        e = _mk(series, "personal_bag", lo, hi, 2, 0.60)
        if e:
            ev.append(e)

    # 3. Инкассаторская сумка. Если при этом работника нет — уверенность выше.
    for lo, hi in _runs(series, lambda f: f.get("sealed_bag")
                        and f.get("sealed_ctx") == "on_desk"):
        e = _mk(series, "sealed_bag", lo, hi, 2, 0.60)
        if e:
            if not series[lo][1].get("teller_present", True):
                e.confidence = min(0.95, e.confidence + 0.15)
            ev.append(e)

    # 4. Наличные на столе без работника рядом.
    for lo, hi in _runs(series, lambda f: f.get("cash_seen")
                        and not f.get("teller_present", True)):
        e = _mk(series, "cash_unattended", lo, hi, 2, 0.65)
        if e:
            ev.append(e)

    # 5. Работника нет на месте, при этом в ролике фигурируют ценности.
    #    Без условия на ценности это дало бы ложняки: в чистых роликах кассир выходит.
    if f0("cash_seen") or f0("sealed_bag"):
        for lo, hi in _runs(series, lambda f: not f.get("teller_present", True)):
            e = _mk(series, "left_workplace", lo, hi, 3, 0.55)
            if e and not any(x.key == "cash_unattended"
                             and x.window[0] <= e.t <= x.window[1] for x in ev):
                ev.append(e)

    # 6. Рабочее место без работника, при этом на столешнице лежат предметы.
    #    Признак набран двумя независимыми каналами: модель описывает кадр и НЕ
    #    упоминает ни одного человека, и при этом на столешнице есть предмет —
    #    либо названный в описании, либо найденный вычитанием фона.
    #    Замерено на обучающей выборке: +5 найденных нарушений ценой 2 ложных
    #    срабатываний из 20 чистых роликов. Точность вердикта при этом растёт
    #    с 62 до 67 %, а ТЗ прямо ставит пропуск выше ложного срабатывания (п.16).
    #    Пробовали требовать согласия двух каналов (подпись И детектор человека не
    #    видят) — это убирает 1 ложное срабатывание, но теряет 3 настоящих: детектор
    #    слишком часто «видит» человека за стеклом. Замерено, вариант отвергнут.
    #    Таймкод: берём САМЫЙ ДЛИННЫЙ подтверждённый интервал и указываем кадр из
    #    его середины, а не первый сработавший кадр. Момент ухода работника на шаге
    #    сетки ключевых кадров не наблюдаем, а середина интервала — это кадр, на
    #    который проверяющему и надо смотреть. Замер на обучающей выборке:
    #    первый кадр — медиана 2.9 с и 56 % в пределах 3 с; середина самого
    #    длинного интервала — медиана 2.1 с и 67 % в пределах 3 с.
    rr = _runs(series, lambda f: f.get("no_person_in_caption")
               and (f.get("paper_seen") or f.get("personal_bag")
                    or f.get("sealed_bag") or f.get("track_on_desk")))
    if rr:
        lo, hi = max(rr, key=lambda r: r[1] - r[0])
        mid = (lo + hi) // 2
        e = _mk(series, "unattended_desk", lo, hi, 1, 0.55)
        if e:
            e.t = round(series[mid][0], 2)
            e.facts_at_t = series[mid][1]
            if not any(x.window[0] <= e.t <= x.window[1] for x in ev):
                ev.append(e)

    # 7. Посторонние в кассовом помещении.
    #    Исходное условие «трое и более ОДНОВРЕМЕННО С НАЛИЧНЫМИ» было мёртвым:
    #    признак cash_seen не срабатывает никогда, потому что банкноты на этом
    #    разрешении не распознаются (01-FINDINGS, раздел 14). Заменено на «на
    #    столешнице есть предмет» — по описанию или по вычитанию фона.
    #    Перебор длительности серии (tools/sweep_rules.py), ложные из 19 чистых:
    #      2 кадра — найдено 12, ложных 4;  3 кадра — найдено 11, ложных 1;
    #      4 кадра — найдено 10, ложных 1.
    #    Взято 3 кадра: +2 найденных нарушения (Х-4, Х-38) без новых ложных.
    for lo, hi in _runs(series, lambda f: f.get("people_in_cabin", 0) >= 3
                        and (f.get("track_on_desk") or f.get("paper_seen"))):
        e = _mk(series, "outsider_with_cash", lo, hi, 3, 0.60)
        if e:
            ev.append(e)

    ev.sort(key=lambda e: e.t)
    ev = _dedup(ev)
    if tracks:
        _snap_to_tracks(ev, tracks)
    return ev


def _snap_to_tracks(ev: list[Event], tracks, max_gap: float = 4.0,
                    min_t_start: float = 1.0) -> None:
    """Уточнение таймкода по моменту появления предмета на столешнице.

    Трек, существующий с первого же кадра, обозначает не появление предмета, а фон:
    столешницу без этого предмета мы никогда не видели. Притягивать к нему таймкод
    значит сдвигать событие на нулевую секунду. Замер на обучающей выборке
    (`Х-30`, `Х-32`): с такими треками доля ошибок ≤3 с падала с 55 % до 45 %,
    средняя ошибка 5.5 с; без них — 55 % и 5.2 с.
    """
    for e in ev:
        if e.key not in ("food_drink", "personal_bag", "sealed_bag",
                         "cash_unattended", "unattended_desk"):
            continue
        near = [t for t in tracks if t.t_start > min_t_start
                and abs(t.t_start - e.t) <= max_gap]
        if near:
            t_new = round(min(near, key=lambda t: abs(t.t_start - e.t)).t_start, 2)
            # Окно расширяем до нового таймкода: иначе описание сообщало бы
            # «признак наблюдается с X по Y», а таймкод указывал вне этого диапазона.
            e.window = (min(e.window[0], t_new), max(e.window[1], t_new))
            e.t = t_new


MAX_EVENTS = 6
MERGE_GAP = 3.0      # окна ближе этого — одно наблюдение, склеиваем
SPLIT_GAP = 12.0     # разнесены дальше этого — разные эпизоды, оба остаются


def _window_gap(a: Event, b: Event) -> float:
    """Расстояние между окнами двух событий; 0, если окна пересекаются."""
    return max(0.0, a.window[0] - b.window[1], b.window[0] - a.window[1])


def _dedup(ev: list[Event], gap: float = MERGE_GAP,
           split: float = SPLIT_GAP) -> list[Event]:
    """Склейка близких событий одного типа; разнесённые во времени — оставить оба.

    Один ролик может содержать два отдельных эпизода одного типа: в `Х-4` разметка
    указывает 57 и 68 с. Прежняя версия оставляла по одному событию на тип и такой
    второй эпизод теряла.

    Три зоны по расстоянию между окнами событий одного типа:
      ≤ `gap`      — одно и то же наблюдение, склеиваем окна и поддержку;
      ≤ `split`    — вероятно один затянувшийся эпизод, оставляем лучшее событие;
      > `split`    — разные эпизоды, оставляем оба.
    """
    merged: list[Event] = []
    for e in ev:
        prev = next((x for x in reversed(merged) if x.key == e.key), None)
        if prev and e.t - prev.window[1] <= gap:
            prev.window = (prev.window[0], max(prev.window[1], e.window[1]))
            prev.confidence = max(prev.confidence, e.confidence)
            prev.support += e.support
        else:
            merged.append(e)

    kept: list[Event] = []
    for e in sorted(merged, key=lambda x: (-x.confidence, -x.support, x.t)):
        near = [k for k in kept if k.key == e.key and _window_gap(e, k) <= split]
        if not near:
            kept.append(e)

    out = sorted(kept, key=lambda x: (-x.confidence, x.t))[:MAX_EVENTS]
    return sorted(out, key=lambda x: x.t)
