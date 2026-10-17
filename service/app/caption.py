"""Florence-2 (MIT) — описание кадра + разбор в факты.

Почему не Qwen2.5-VL-7B: на Apple M4 он даёт ~4 токена/с (13 с на кадр), то есть
12-18 кадров на видео в пятиминутный лимит. Florence-2-large — 2.7 с/кадр,
это 40-60 кадров, и именно он устойчиво называет стакан и сумки.
Лицензия MIT вместо Apache-2.0 — чище для передачи прав заказчику (п.34 ТЗ).
"""
from __future__ import annotations

import os
import re
import threading
from typing import Any

import cv2
import numpy as np

MODEL_ID = os.environ.get("CAPTION_MODEL", "florence-community/Florence-2-large")
TASK = "<MORE_DETAILED_CAPTION>"

_LOCK = threading.Lock()
_MODEL: Any = None

# Ключевые слова в подписи -> факт. Модель описывает, семантику решает код.
CASH = ("banknote", "bank note", "money", "cash", "currency", "bill", "dollar")
DRINK = ("coffee", "cup", "mug", "can of", "soda", "bottle", "drink", "tea",
         "snack", "food", "sandwich", "eating")
BAG = ("bag", "backpack", "purse", "handbag", "tote", "briefcase", "sack")
SEALED = ("green bag", "green office bag", "canvas bag", "sealed bag", "pouch")
MACHINE = ("printer", "counting machine", "cash register", "scanner", "fax")
PAPER = ("paper", "document", "card", "passport", "notebook", "note")
# Если модель описывает кадр и НИ РАЗУ не упоминает человека — на рабочем месте
# никого нет. Это отдельный канал, независимый от детектора людей.
# По границам слова, а не по подстроке: с проверкой на " she " подпись,
# начинающаяся со слова She, не находила человека и правило unattended_desk
# срабатывало на кадре, где работник описан.
PERSON_RE = re.compile(
    r"\b(wom[ae]n|m[ae]n|person|people|she|he|her|his|lady|girl|guy|"
    r"worker|employee|cashier|client|customer|visitor)\b", re.I)


# Описатель кадра переключается переменной окружения, а факты из его текста
# извлекаются одним и тем же кодом — так две модели сравниваются честно:
# одинаковые правила и ключевые слова, разный источник описания.
BACKEND = "qwen" if "Qwen" in MODEL_ID else "florence"

# Для Qwen просим описание по-английски: ключевые слова ниже собраны под английский
# текст Florence-2, и менять их ради второй модели значило бы менять и правила.
QWEN_PROMPT = (
    "This is an overhead frame of a bank teller's desk. "
    "Describe in English, in 2-3 sentences, exactly what is visible: how many people are in "
    "the room, whether a worker is at the desk, and what lies on the desktop. "
    "List only objects that are actually there. Do not mention objects that are absent: "
    "a sentence like \"there are no cups\" is wrong, simply omit what you do not see."
)


def _load():
    global _MODEL
    if _MODEL is None:
        if BACKEND == "qwen":
            from mlx_vlm import load
            from mlx_vlm.utils import load_config
            model, proc = load(MODEL_ID)
            _MODEL = (model, proc, load_config(MODEL_ID), None)
        else:
            import torch
            from transformers import Florence2ForConditionalGeneration, Florence2Processor
            dev = "mps" if torch.backends.mps.is_available() else "cpu"
            proc = Florence2Processor.from_pretrained(MODEL_ID)
            model = Florence2ForConditionalGeneration.from_pretrained(
                MODEL_ID, dtype=torch.float32, attn_implementation="eager").to(dev).eval()
            _MODEL = (model, proc, dev, torch)
    return _MODEL


def warmup() -> None:
    with _LOCK:
        _load()


def _caption_qwen(img: np.ndarray, max_new_tokens: int) -> str:
    import os as _os
    import tempfile
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template
    model, proc, config, _ = _load()
    path = tempfile.mktemp(suffix=".jpg")
    cv2.imwrite(path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    try:
        fmt = apply_chat_template(proc, config, QWEN_PROMPT, num_images=1)
        with _LOCK:
            res = generate(model, proc, fmt, [path], max_tokens=max_new_tokens,
                           temperature=0.0, verbose=False)
        return getattr(res, "text", str(res)).strip()
    finally:
        if _os.path.exists(path):
            _os.unlink(path)


def caption(img: np.ndarray, max_new_tokens: int = 140) -> str:
    """Подробное описание кадра (английский — под него собраны ключевые слова)."""
    if BACKEND == "qwen":
        return _caption_qwen(img, max_new_tokens)
    from PIL import Image
    model, proc, dev, torch = _load()
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    inputs = proc(text=TASK, images=pil, return_tensors="pt").to(dev)
    with _LOCK, torch.inference_mode():
        ids = model.generate(input_ids=inputs["input_ids"],
                             pixel_values=inputs["pixel_values"],
                             max_new_tokens=max_new_tokens, num_beams=1, do_sample=False)
    raw = proc.batch_decode(ids, skip_special_tokens=False)[0]
    return proc.post_process_generation(raw, task=TASK, image_size=pil.size)[TASK].strip()


# Отрицание перед ключевым словом. На текущем описателе не встречается ни разу
# (0 случаев на 906 подписей), но при смене модели или ракурса фраза вида
# «no cup on the desk» дала бы ложное срабатывание: матчер видит «cup».
_NEG_RE = re.compile(
    r"\b(no|not|without|nothing|none|aren't|isn't|does not|do not)\b[^.;]{0,40}$",
    re.I)


def _has(text: str, words) -> bool:
    """Упоминание ключевого слова без отрицания перед ним."""
    low = text.lower()
    for w in words:
        i = low.find(w)
        while i != -1:
            # смотрим на фрагмент предложения ДО слова: нет ли там отрицания
            start = max(low.rfind(".", 0, i), low.rfind(";", 0, i)) + 1
            if not _NEG_RE.search(low[start:i]):
                return True
            i = low.find(w, i + 1)
    return False


# Где находится предмет, решает предлог в том же предложении. Это не придирка к
# формулировке: на чистом ролике Х-16 модель пишет «a man holding a green bag»
# (сумку несут), а на нарушении Х-15 — «a green office bag on the desk»
# (сумка брошена на столешнице). Без разбора контекста это один и тот же факт.
ON_DESK = ("on the desk", "on the table", "on a desk", "on a table",
           "on the wooden desk", "on top of the desk", "lying on")
IN_HAND = ("holding", "in her hand", "in his hand", "in their hands",
           "in hand", "carrying", "hands")


def object_context(text: str, words) -> str:
    """Где упомянут предмет: 'on_desk' / 'in_hand' / 'unknown'."""
    low = text.lower()
    for sent in re.split(r"(?<=[.!?])\s+", low):
        if not any(w in sent for w in words):
            continue
        on = any(w in sent for w in ON_DESK)
        hand = any(w in sent for w in IN_HAND)
        if on and not hand:
            return "on_desk"
        if hand and not on:
            return "in_hand"
        if on and hand:
            return "on_desk"
    return "unknown"


def facts_from_caption(text: str) -> dict:
    """Разбор подписи в факты. Никаких выводов, только наличие упоминаний."""
    sealed = _has(text, SEALED)
    bag = _has(text, BAG) and not sealed
    return {
        "no_person_in_caption": not PERSON_RE.search(text),
        "cash_seen": _has(text, CASH),
        "cash_ctx": object_context(text, CASH) if _has(text, CASH) else "unknown",
        "drink_or_food": _has(text, DRINK),
        "personal_bag": bag,
        "bag_ctx": object_context(text, BAG) if bag else "unknown",
        "sealed_bag": sealed,
        "sealed_ctx": object_context(text, SEALED) if sealed else "unknown",
        "machine_seen": _has(text, MACHINE),
        "paper_seen": _has(text, PAPER),
        "caption": text,
    }


def count_people(text: str) -> int:
    """Грубая оценка числа людей по подписи — только как доп. сигнал к детектору."""
    low = text.lower()
    n = 0
    for w, k in (("two people", 2), ("two men", 2), ("two women", 2),
                 ("three people", 3), ("a man and a woman", 2)):
        if w in low:
            n = max(n, k)
    if n == 0 and re.search(r"\b(a|one)\s+(woman|man|person)\b", low):
        n = 1
    return n
