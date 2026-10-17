"""Локальная VLM (Qwen2.5-VL, Apache-2.0) как извлекатель фактов по кадру.

Почему VLM, а не правила по пиксельным зонам: камера переставляется между
записями, а финальное тестовое видео снято под другим углом (FAQ п.4).
Фактам вида «кассир на месте», «на столе банкноты» смена ракурса не мешает.

Модель крутится локально через MLX — никаких облачных вызовов (ТЗ п.7, п.21, п.35).
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from typing import Any

import cv2
import numpy as np

MODEL_ID = os.environ.get("VLM_MODEL", "mlx-community/Qwen2.5-VL-7B-Instruct-4bit")

_LOCK = threading.Lock()
_MODEL: Any = None

FACT_KEYS = {
    "people_in_cabin": int,
    "teller_present": bool,
    "cash_on_desk": bool,
    "cash_in_hands": bool,
    "machine_in_use": bool,
    "drink_or_food": bool,
    "personal_bag": bool,
    "sealed_bag": bool,
    "door_open": bool,
}

PROMPT = """На кадре — рабочее место кассира банка, вид сверху. За столом работник, напротив через стол может сидеть посетитель.
На столе обычно: ноутбук, стационарный телефон, счётно-денежная машина (крупный тёмный аппарат слева), рядом серый металлический шкаф.
Через стеклянную стену видно соседнее помещение — людей ЗА СТЕКЛОМ не считай и не описывай.

Ответь строго четырьмя короткими строками, без markdown и без пояснений:
ЛЮДЕЙ: <число людей внутри этой комнаты>
РАБОТНИК_ЗА_СТОЛОМ: <да / нет>
НА_СТОЛЕ: <перечисли через запятую, что лежит на столешнице>
В_РУКАХ: <что держат в руках; если ничего — «ничего»>

Называй предметы своими словами: банкноты, пачка денег, банковская карта, лист бумаги, паспорт,
стакан, кружка, банка, бутылка, еда, сумка, рюкзак, пакет, телефон, ноутбук, счётная машина.
Пиши только то, что действительно различимо на кадре."""

# ключевые слова -> факт. Семантику решает код, модель только перечисляет предметы.
_CASH = ("банкнот", "купюр", "деньг", "наличн", "пачка", "тенге", "валют")
_DRINK = ("стакан", "кружк", "банка", "банку", "бутыл", "кофе", "чай", "напит", "еда", "еду",
          "сэндвич", "печень", "шоколад", "термос")
_BAG = ("сумк", "рюкзак", "пакет", "портфел", "барсетк")
_SEALED = ("инкассатор", "мешок", "мешк", "опломбир", "пломб", "сумка-мешок", "брезент")
_MACHINE = ("счётн", "счетн", "машин", "аппарат", "купюросч")


def _load():
    global _MODEL
    if _MODEL is None:
        from mlx_vlm import load
        from mlx_vlm.utils import load_config
        model, processor = load(MODEL_ID)
        _MODEL = (model, processor, load_config(MODEL_ID))
    return _MODEL


def warmup() -> None:
    with _LOCK:
        _load()


def _field(text: str, name: str) -> str:
    m = re.search(rf"^\s*{name}\s*[:：]\s*(.+)$", text, re.M | re.I)
    return (m.group(1).strip() if m else "").strip("«»\"' .")


def _has(s: str, words) -> bool:
    low = s.lower().replace("ё", "е")
    return any(w.replace("ё", "е") in low for w in words)


def _parse_lines(text: str) -> dict:
    """Разбор построчного ответа. Семантику выводим из перечня предметов."""
    text = re.sub(r"^```.*?$", "", text.strip(), flags=re.M)
    people_raw = _field(text, "ЛЮДЕЙ")
    m = re.search(r"\d+", people_raw)
    people = int(m.group(0)) if m else 0

    at_desk = _field(text, "РАБОТНИК_ЗА_СТОЛОМ").lower()
    on_desk = _field(text, "НА_СТОЛЕ")
    in_hands = _field(text, "В_РУКАХ")
    door = _field(text, "ДВЕРЬ").lower()
    note = ""

    if not any((people_raw, at_desk, on_desk, in_hands, note)):
        raise ValueError(f"ответ не разобран: {text[:160]!r}")

    hands_empty = _has(in_hands, ("ничего", "пусто", "не держ"))
    return {
        "people_in_cabin": people,
        "teller_present": at_desk.startswith("да") or (people > 0 and not at_desk.startswith("нет")),
        "cash_on_desk": _has(on_desk, _CASH),
        "cash_in_hands": (not hands_empty) and _has(in_hands, _CASH),
        "machine_in_use": _has(in_hands, _MACHINE) or _has(note, _MACHINE),
        "drink_or_food": _has(on_desk, _DRINK) or ((not hands_empty) and _has(in_hands, _DRINK)),
        "personal_bag": _has(on_desk, _BAG) and not _has(on_desk, _SEALED),
        "sealed_bag": _has(on_desk, _SEALED) or _has(in_hands, _SEALED) or _has(note, _SEALED),
        "door_open": door.startswith("откр"),
        "what_happens": note[:300],
        "_raw": {"on_desk": on_desk, "in_hands": in_hands, "door": door},
    }


def facts(img: np.ndarray, upscale: float = 1.0, max_tokens: int = 110) -> dict:
    """Факты по одному кадру. Кадр BGR (как отдаёт OpenCV)."""
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    model, processor, config = _load()
    if upscale != 1.0:
        img = cv2.resize(img, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_LANCZOS4)

    path = tempfile.mktemp(suffix=".jpg")
    cv2.imwrite(path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    try:
        formatted = apply_chat_template(processor, config, PROMPT, num_images=1)
        with _LOCK:
            res = generate(model, processor, formatted, [path],
                           max_tokens=max_tokens, temperature=0.0, verbose=False)
        text = getattr(res, "text", res if isinstance(res, str) else str(res))
        return _parse_lines(text)
    finally:
        if os.path.exists(path):
            os.unlink(path)


def describe(img: np.ndarray, violation_type: str, upscale: float = 1.0,
             max_tokens: int = 160) -> str:
    """Короткое человекочитаемое обоснование по кадру-доказательству."""
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    model, processor, config = _load()
    if upscale != 1.0:
        img = cv2.resize(img, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_LANCZOS4)
    prompt = (
        "Кадр с записи рабочего места кассира банка (вид сверху). "
        f"Проверяющий подозревает: «{violation_type}».\n"
        "Опиши в 1-2 предложениях по-русски ТОЛЬКО то, что видно на кадре: где находится работник, "
        "что у него в руках, какие предметы на столе. Не делай юридических выводов, "
        "не утверждай того, чего на кадре не видно. Без вступлений."
    )
    path = tempfile.mktemp(suffix=".jpg")
    cv2.imwrite(path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    try:
        formatted = apply_chat_template(processor, config, prompt, num_images=1)
        with _LOCK:
            res = generate(model, processor, formatted, [path],
                           max_tokens=max_tokens, temperature=0.0, verbose=False)
        text = getattr(res, "text", res if isinstance(res, str) else str(res))
        return text.strip().strip('"')
    finally:
        if os.path.exists(path):
            os.unlink(path)
