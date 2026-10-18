"""Второй описатель: Qwen3-VL-8B-Instruct (Apache-2.0) отвечает на короткие вопросы
по ПОЛНОМУ кадру — сидит ли работник, есть ли кто-то в двери, сколько людей в
помещении, пуст ли стул работника.

Зачем: Florence-2 описывает кроп столешницы и про людей в помещении знает только
то, что попало в кроп; детектор считает людей и за стеклянной стеной. Прямой вопрос
о числе людей в комнате модель покрупнее отвечает по всему кадру.

На CUDA включён по умолчанию (17 ГБ весов в bf16), на MPS/CPU — выключен; VLM=0/1. Ответ — четыре строки
фиксированного формата; разбор в факты — функция `parse`, общая для сервиса и
харнесса (`tools/vlm_corpus.py`).
"""
from __future__ import annotations

import os
import re
import threading
from typing import Any

import cv2
import numpy as np

from .config import env

ENABLED = env("VLM", 1, 0) == "1"
MODEL_ID = os.environ.get("VLM_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
BATCH = int(os.environ.get("VLM_BATCH", 8))

PROMPT = (
    "This is a CCTV frame of a bank teller's room. The teller works behind the wooden desk "
    "with the laptop and the cash counting machine. The room has a glass wall and a glass door; "
    "people seen through the glass in the office outside are NOT in the room.\n"
    "Q1. Is the teller sitting or standing at the desk? Answer at_desk or away.\n"
    "Q2. Is anyone standing in the doorway or walking through the door of the room? Answer yes or no.\n"
    "Q3. How many people are inside the room? Answer a number.\n"
    "Q4. Is the teller's chair behind the desk empty? Answer yes or no.\n"
    "Reply with exactly four lines:\nQ1: ...\nQ2: ...\nQ3: ...\nQ4: ..."
)

# Отдельный вопрос, задаётся только на кадрах, где Florence-2 назвал напиток или еду:
# нарушением считается напиток в руке / у лица, а не банка, стоящая на столе.
# Отдельным вызовом, а не пятым вопросом в PROMPT: смена общего промпта поменяла бы
# ответы на остальные четыре вопроса, по которым уже откалиброваны правила.
DRINK_PROMPT = ("Look at the person at the desk. Is the person holding a cup, mug, can or "
                "bottle in their hand, or drinking? Answer yes or no.")

# Сейф/шкаф по кропу области за столешницей (П5). Три варианта ответа: шкафа в
# кропе может не быть вовсе — тогда «закрыт» был бы ложным утверждением.
SAFE_PROMPT = ("This is a close-up of the area behind a bank teller's desk. Is there a grey "
               "metal cabinet or safe, and is its door open? Answer with one word: open, "
               "closed, or none.")

_LOCK = threading.Lock()
_MODEL: Any = None


def parse(text: str) -> dict:
    """Четыре ответа -> факты. Неразобранный ответ = None, а не «нет»."""
    ans = {}
    for k, v in re.findall(r"Q([1-4])\s*[:.)]\s*([^\n]+)", text):
        ans[k] = v.strip().lower()

    def yn(s):
        if s is None:
            return None
        if s.startswith("yes"):
            return True
        if s.startswith("no"):
            return False
        return None

    q1 = ans.get("1")
    at = None if q1 is None else (True if "at_desk" in q1 or "at desk" in q1 or q1.startswith("sit")
                                   else False if "away" in q1 else None)
    m = re.search(r"\d+", ans.get("3", ""))
    # Q1 модель иногда отвечает «standing» вместо одного из двух вариантов: это не
    # «за столом» и не «ушёл», поэтому хранится отдельно, а не приводится насильно.
    pose = None if q1 is None else ("at_desk" if at else "away" if at is False
                                    else "standing" if "stand" in q1 else "other")
    return {"teller_at_desk_vlm": at, "teller_pose_vlm": pose, "person_at_door": yn(ans.get("2")),
            "people_in_booth_vlm": int(m.group()) if m else None,
            "chair_empty": yn(ans.get("4"))}


def _load():
    global _MODEL
    if _MODEL is None:
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        proc = AutoProcessor.from_pretrained(MODEL_ID)
        proc.tokenizer.padding_side = "left"           # батч генерации: паддинг слева
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_ID, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
        _MODEL = (model, proc, torch)
    return _MODEL


def warmup() -> None:
    with _LOCK:
        _load()


def yes(text: str) -> bool | None:
    """Ответ «yes/no» -> bool; неразобранный -> None."""
    t = text.strip().lower()
    return True if t.startswith("yes") else False if t.startswith("no") else None


def ask(imgs_bgr: list[np.ndarray], max_new_tokens: int = 32,
        prompt: str = PROMPT) -> list[str]:
    """Сырые ответы модели на prompt по каждому кадру, одним вызовом generate."""
    from PIL import Image
    model, proc, torch = _load()
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    pils = [Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)) for im in imgs_bgr]
    inputs = proc(text=[text] * len(pils), images=pils, return_tensors="pt",
                  padding=True).to("cuda")
    with _LOCK, torch.inference_mode():
        ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    ids = ids[:, inputs["input_ids"].shape[1]:]
    return [t.strip() for t in proc.batch_decode(ids, skip_special_tokens=True)]
