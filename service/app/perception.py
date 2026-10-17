"""Дешёвый слой восприятия: кадры, люди, ключевые кадры.

Задача слоя — не «понять нарушение», а сократить видео до десятков кадров,
которые стоит показать тяжёлой модели. Всё остальное решает vlm.py + rules.py.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass, field

import cv2
import numpy as np

_DETR = None  # ленивый синглтон


@dataclass
class Frame:
    t: float                      # секунда от начала
    img: np.ndarray               # BGR
    persons: list[dict] = field(default_factory=list)
    objects: list[dict] = field(default_factory=list)
    motion: float = 0.0           # энергия изменения относительно предыдущего кадра


def probe(path: str) -> tuple[float, float]:
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    return (n / fps if fps else 0.0), fps


def sample(path: str, fps: float = 2.0, max_side: int = 960) -> list[Frame]:
    """Разреженная выборка кадров. max_side — ограничение на всякий случай,
    датасет 464x848, но тестовое видео может быть крупнее."""
    cap = cv2.VideoCapture(path)
    src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(src / fps)))
    out: list[Frame] = []
    i = 0
    prev_small = None
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if i % step == 0:
            h, w = img.shape[:2]
            if max(h, w) > max_side:
                k = max_side / max(h, w)
                img = cv2.resize(img, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)
            small = cv2.cvtColor(cv2.resize(img, (96, 160)), cv2.COLOR_BGR2GRAY)
            motion = 0.0 if prev_small is None else float(
                np.abs(small.astype(np.int16) - prev_small.astype(np.int16)).mean())
            prev_small = small
            out.append(Frame(t=i / src, img=img, motion=motion))
        i += 1
    cap.release()
    return out


def _load_detr():
    global _DETR
    if _DETR is None:
        import torch
        from transformers import AutoImageProcessor, AutoModelForObjectDetection
        mid = "PekingU/rtdetr_v2_r50vd"
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        proc = AutoImageProcessor.from_pretrained(mid)
        model = AutoModelForObjectDetection.from_pretrained(mid).to(dev).eval()
        _DETR = (proc, model, dev, torch)
    return _DETR


def detect(frames: list[Frame], batch: int = 8, thr: float = 0.35) -> None:
    """Проставляет frame.persons / frame.objects (in-place)."""
    proc, model, dev, torch = _load_detr()
    id2label = model.config.id2label
    with torch.inference_mode():
        for s in range(0, len(frames), batch):
            chunk = frames[s:s + batch]
            imgs = [cv2.cvtColor(f.img, cv2.COLOR_BGR2RGB) for f in chunk]
            inputs = proc(images=imgs, return_tensors="pt").to(dev)
            res = model(**inputs)
            sizes = torch.tensor([im.shape[:2] for im in imgs]).to(dev)
            post = proc.post_process_object_detection(res, target_sizes=sizes, threshold=thr)
            for f, p in zip(chunk, post):
                for sc, lb, bx in zip(p["scores"], p["labels"], p["boxes"]):
                    d = {"cls": id2label[int(lb)], "score": round(float(sc), 3),
                         "box": [round(float(v), 1) for v in bx.tolist()]}
                    (f.persons if d["cls"] == "person" else f.objects).append(d)


def persons_in_room(f: Frame, min_h_rel: float = 0.115,
                    desk_top: float | None = None) -> list[dict]:
    """Люди внутри кабины.

    Два отсечения, оба относительные (не зависят от разрешения и крупности плана):
    по высоте бокса — люди в соседнем помещении за стеклом мельче; по нижнему краю —
    тот, кто действительно в комнате, доходит как минимум до уровня столешницы.
    Без второго отсечения подсчёт людей ломается: за стеклянной стеной виден опенспейс.
    """
    h = f.img.shape[0]
    top = desk_top if desk_top is not None else 0.35 * h
    return [p for p in f.persons
            if (p["box"][3] - p["box"][1]) >= min_h_rel * h and p["box"][3] >= top]


# Классы-«якоря» рабочей зоны. Стулья и столы исключены намеренно: в кадре есть
# крупная белая столешница на переднем плане, она растянула бы зону на весь кадр.
ANCHOR_CLS = {"laptop", "suitcase", "backpack", "tvmonitor", "keyboard",
              "cell phone", "book", "cup", "bottle", "mouse", "handbag"}


def roi(frames: list[Frame], pad: float = 0.06) -> tuple[int, int, int, int]:
    """Зона интереса на всё видео: объединение людей в кабине и якорных предметов.

    Выводится из детекций, а не из пиксельных констант, — поэтому переживает
    смену ракурса и крупности плана (критерий устойчивости).
    """
    h, w = frames[0].img.shape[:2]
    xs1, ys1, xs2, ys2 = [], [], [], []
    for f in frames:
        for p in persons_in_room(f):
            b = p["box"]; xs1.append(b[0]); ys1.append(b[1]); xs2.append(b[2]); ys2.append(b[3])
        for o in f.objects:
            if o["cls"] in ANCHOR_CLS and o["score"] >= 0.4:
                b = o["box"]; xs1.append(b[0]); ys1.append(b[1]); xs2.append(b[2]); ys2.append(b[3])
    if not xs1:
        return 0, 0, w, h
    px, py = pad * w, pad * h
    x1 = max(0, int(min(xs1) - px)); y1 = max(0, int(min(ys1) - py))
    x2 = min(w, int(max(xs2) + px)); y2 = min(h, int(max(ys2) + py))
    if x2 - x1 < w * 0.25 or y2 - y1 < h * 0.2:   # вырожденная зона — берём кадр целиком
        return 0, 0, w, h
    return x1, y1, x2, y2


def crop_scaled(img: np.ndarray, box: tuple[int, int, int, int],
                target_px: int = 700_000) -> np.ndarray:
    """Кроп зоны интереса с апскейлом до бюджета по пикселям.

    Объекты становятся крупнее для модели, а стоимость инференса остаётся прежней:
    число пикселей фиксировано, меняется только масштаб.
    """
    x1, y1, x2, y2 = box
    sub = img[y1:y2, x1:x2]
    if sub.size == 0:
        return img
    k = (target_px / (sub.shape[0] * sub.shape[1])) ** 0.5
    if k > 1.05:
        sub = cv2.resize(sub, None, fx=k, fy=k, interpolation=cv2.INTER_LANCZOS4)
    return sub


def keyframes(frames: list[Frame], budget: int, step_sec: float = 2.0) -> list[int]:
    """Индексы кадров для тяжёлого анализа.

    Берём: равномерную сетку + точки смены числа людей + пики движения,
    затем прореживаем до budget штук, сохраняя хронологию.
    """
    if not frames:
        return []
    picked: set[int] = set()

    # 1. равномерная сетка
    dt = frames[1].t - frames[0].t if len(frames) > 1 else 0.5
    grid = max(1, int(round(step_sec / dt)))
    picked.update(range(0, len(frames), grid))

    # 2. смена числа людей в кабине
    counts = [len(persons_in_room(f)) for f in frames]
    for i in range(1, len(counts)):
        if counts[i] != counts[i - 1]:
            picked.update({max(0, i - 1), i, min(len(frames) - 1, i + 1)})

    # 3. пики движения
    mot = np.array([f.motion for f in frames])
    if len(mot) > 4:
        thr = float(np.percentile(mot, 85))
        picked.update(int(i) for i in np.where(mot >= thr)[0])

    idx = sorted(picked)
    if len(idx) > budget:  # равномерно проредить
        sel = np.linspace(0, len(idx) - 1, budget).round().astype(int)
        idx = [idx[i] for i in sorted(set(sel.tolist()))]
    return idx


def jpeg_b64(img: np.ndarray, quality: int = 85, max_side: int = 1280) -> str:
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        k = max_side / max(h, w)
        img = cv2.resize(img, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("не удалось закодировать кадр в JPEG")
    return base64.b64encode(buf.tobytes()).decode("ascii")
