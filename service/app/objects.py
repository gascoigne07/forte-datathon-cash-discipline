"""Канал «новые предметы на сцене»: фон + вычитание + трекинг блобов.

Зачем отдельный канал: VLM на этом железе стоит ~13 с/кадр, её хватает на
десяток кадров за видео. А момент «на столе появилась пачка денег / стакан /
сумка» надёжно и бесплатно ловится классическим вычитанием фона — сцена
статична, двигаются только люди и предметы, которые кто-то принёс.

Люди вырезаются по боксам детектора: иначе каждый взмах руки давал бы «предмет».
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from . import perception
from .perception import Frame, persons_in_room


@dataclass
class ObjectTrack:
    t_start: float
    t_end: float
    box: tuple[int, int, int, int]      # в координатах исходного кадра
    area: float
    frames: int
    near_person: bool                    # появился в зоне рук (значит, его принесли/положили)


def _person_mask(f: Frame, shape, pad_rel: float = 0.05) -> np.ndarray:
    h, w = shape
    m = np.zeros((h, w), np.uint8)
    kh, kw = f.img.shape[0] / h, f.img.shape[1] / w
    for p in f.persons:                      # все люди, включая мелких за стеклом
        x1, y1, x2, y2 = p["box"]
        px, py = pad_rel * w, pad_rel * h
        cv2.rectangle(m, (int(x1 / kw - px), int(y1 / kh - py)),
                      (int(x2 / kw + px), int(y2 / kh + py)), 255, -1)
    return m


def detect_objects(frames: list[Frame], zone: tuple[int, int, int, int] | None = None,
                   min_frames: int = 3, min_area_rel: float = 1.2e-3,
                   max_area_rel: float = 8e-2, work_h: int = 240) -> list[ObjectTrack]:
    """Треки предметов, появившихся в зоне стола и задержавшихся там.

    zone — прямоугольник столешницы в координатах кадра; без него канал
    ловит людей и блики за стеклянной стеной, а не предметы.
    """
    if len(frames) < 6:
        return []
    h0, w0 = frames[0].img.shape[:2]
    k = work_h / h0
    ww, hh = int(w0 * k), work_h

    small = [cv2.cvtColor(cv2.resize(f.img, (ww, hh), interpolation=cv2.INTER_AREA),
                          cv2.COLOR_BGR2GRAY) for f in frames]
    bg = np.median(np.stack(small), axis=0).astype(np.uint8)

    zmask = np.zeros((hh, ww), np.uint8)
    if zone:
        zx1, zy1, zx2, zy2 = (int(v * k) for v in zone)
        zmask[max(0, zy1):zy2, max(0, zx1):zx2] = 255
        zarea = max(1, (zy2 - zy1) * (zx2 - zx1))
    else:
        zmask[:] = 255
        zarea = ww * hh
    min_area = min_area_rel * zarea
    max_area = max_area_rel * zarea
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    per_frame: list[list[tuple[int, int, int, int, float]]] = []
    for f, g in zip(frames, small):
        diff = cv2.absdiff(g, bg)
        _, th = cv2.threshold(diff, 28, 255, cv2.THRESH_BINARY)
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
        th[_person_mask(f, (hh, ww)) > 0] = 0
        th[zmask == 0] = 0
        n, _, stats, _ = cv2.connectedComponentsWithStats(th, 8)
        boxes = []
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            if min_area <= area <= max_area and bw < ww * 0.5 and bh < hh * 0.5:
                boxes.append((x, y, x + bw, y + bh, float(area)))
        per_frame.append(boxes)

    # трекинг: сопоставляем блоки соседних кадров по пересечению
    tracks: list[dict] = []
    for fi, boxes in enumerate(per_frame):
        for b in boxes:
            # Берём трек с наибольшим пересечением, а не первый подходящий:
            # рядом лежащие предметы иначе склеиваются в один трек и момент
            # появления привязывается не к тому предмету.
            hit, best_iou = None, 0.25
            for tr in tracks:
                if tr["last_fi"] < fi - 2:
                    continue
                iou = _iou(tr["box"], b[:4])
                if iou > best_iou:
                    hit, best_iou = tr, iou
            if hit:
                hit["last_fi"] = fi; hit["n"] += 1
                hit["areas"].append(b[4])
                hit["box"] = _merge(hit["box"], b[:4])
            else:
                tracks.append({"first_fi": fi, "last_fi": fi, "n": 1,
                               "box": b[:4], "areas": [b[4]]})

    out: list[ObjectTrack] = []
    for tr in tracks:
        if tr["n"] < min_frames:
            continue
        x1, y1, x2, y2 = tr["box"]
        box = (int(x1 / k), int(y1 / k), int(x2 / k), int(y2 / k))
        f0 = frames[tr["first_fi"]]
        out.append(ObjectTrack(
            t_start=round(f0.t, 2), t_end=round(frames[tr["last_fi"]].t, 2),
            box=box, area=float(np.median(tr["areas"])), frames=tr["n"],
            near_person=_near_any_person(box, f0),
        ))
    out.sort(key=lambda t: t.t_start)
    return out


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua else 0.0


def _merge(a, b):
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _near_any_person(box, f: Frame, margin_rel: float = 0.12) -> bool:
    h, w = f.img.shape[:2]
    mx, my = margin_rel * w, margin_rel * h
    for p in persons_in_room(f):
        x1, y1, x2, y2 = p["box"]
        if _iou((x1 - mx, y1 - my, x2 + mx, y2 + my), box) > 0:
            return True
    return False


def crop_object(img: np.ndarray, box, target_px: int = 160_000,
                pad_rel: float = 0.6) -> np.ndarray:
    """Вырезает предмет с запасом и увеличивает — чтобы VLM отвечала на
    вопрос «что это?» по картинке, где предмет занимает полкадра, а не 25 пикселей."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    px, py = int(bw * pad_rel) + 8, int(bh * pad_rel) + 8
    x1, y1 = max(0, x1 - px), max(0, y1 - py)
    x2, y2 = min(w, x2 + px), min(h, y2 + py)
    sub = img[y1:y2, x1:x2]
    if sub.size == 0:
        return img
    k = (target_px / (sub.shape[0] * sub.shape[1])) ** 0.5
    if k > 1.05:
        sub = cv2.resize(sub, None, fx=k, fy=k, interpolation=cv2.INTER_LANCZOS4)
    return sub


# --- зона стола -------------------------------------------------------------
# Предметы на столе ищем только на столешнице: иначе блобы ловят людей и блики
# за стеклянной стеной. Зона выводится из статичной техники, а не из констант.
DESK_ANCHORS = {"laptop", "keyboard", "mouse", "tvmonitor", "suitcase",
                "backpack", "cell phone", "book", "remote"}


def background(frames: list[Frame]) -> np.ndarray:
    """Медианный фон: люди усредняются, остаётся пустая сцена."""
    step = max(1, len(frames) // 60)          # 60 кадров достаточно для медианы
    return np.median(np.stack([f.img for f in frames[::step]]), axis=0).astype(np.uint8)


def resolve_desk_zone(frames: list[Frame], bg: np.ndarray | None = None
                      ) -> tuple[tuple[int, int, int, int], str]:
    """Зона столешницы и способ, которым она получена.

    Четыре уровня, от самого точного к самому грубому. Способ возвращается наружу
    и попадает в ответ API: на новом ракурсе проверяющему важно знать, выведена
    зона из сцены или сервис скатился в константу.

    - `wood`     — тон дерева на медианном фоне (основной путь);
    - `anchors`  — по статичной технике на столе (другой интерьер);
    - `roi`      — объединение людей и якорных предметов за всё видео: грубее, но
                   всё ещё выведено из детекций, а не из пиксельных констант;
    - `fallback` — нижние две трети кадра. Последний рубеж.
    """
    zone = _wood_zone(background(frames) if bg is None else bg)
    if zone is not None:
        return zone, "wood"
    zone = _anchor_zone(frames)
    if zone is not None:
        return zone, "anchors"

    h, w = frames[0].img.shape[:2]
    r = perception.roi(frames)
    if r != (0, 0, w, h):
        return r, "roi"
    return (0, int(0.35 * h), w, h), "fallback"


def desk_zone(frames: list[Frame], bg: np.ndarray | None = None
              ) -> tuple[int, int, int, int] | None:
    """Столешница по тону дерева, иначе по статичной технике, иначе None.

    Оставлено для вызовов, которым не нужен способ получения зоны
    (`tools/build_normal_bank.py` и прочая оснастка).
    """
    return (_wood_zone(background(frames) if bg is None else bg)
            or _anchor_zone(frames))


def _wood_zone(img: np.ndarray) -> tuple[int, int, int, int] | None:
    """Столешница = крупнейшая область тона дерева на медианном фоне.

    Считается по фону, а не по кадру: люди и их руки не мешают.
    """
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Нижний порог яркости — относительный, от медианы кадра. С фиксированным
    # порогом затемнение записи выбивало столешницу из диапазона (ширина зоны
    # уезжала на 39% при -12% яркости).
    v_med = float(np.median(hsv[:, :, 2]))
    v_lo = max(40, min(110, 0.55 * v_med))
    mask = cv2.inRange(hsv, (8, 30, int(v_lo)), (30, 140, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n >= 2:
        i = 1 + int(np.argmax(stats[1:, 4]))
        x, y, bw, bh, area = stats[i]
        if area >= 0.02 * h * w:
            pad = int(0.03 * h)
            return (int(max(0, x - pad)), int(max(0, y - pad)),
                    int(min(w, x + bw + pad)), int(min(h, y + bh + pad)))
    return None


def _anchor_zone(frames: list[Frame], min_share: float = 0.4,
                 expand: float = 0.22) -> tuple[int, int, int, int] | None:
    """Запасной вариант: зона по статичной технике на столе."""
    h, w = frames[0].img.shape[:2]
    buckets: dict[str, list] = {}
    for f in frames:
        for o in f.objects:
            if o["cls"] in DESK_ANCHORS and o["score"] >= 0.4:
                buckets.setdefault(o["cls"], []).append(o["box"])
    boxes = []
    for cls, bs in buckets.items():
        if len(bs) < min_share * len(frames):      # предмет виден почти всегда => статичен
            continue
        arr = np.array(bs)
        med = np.median(arr, axis=0)
        spread = np.abs(arr - med).mean()
        if spread < 0.06 * max(h, w):              # и почти не двигается
            boxes.append(med)
    if not boxes:
        return None
    arr = np.array(boxes)
    x1, y1 = arr[:, 0].min(), arr[:, 1].min()
    x2, y2 = arr[:, 2].max(), arr[:, 3].max()
    bw, bh = x2 - x1, y2 - y1
    return (int(max(0, x1 - expand * bw)), int(max(0, y1 - expand * bh)),
            int(min(w, x2 + expand * bw)), int(min(h, y2 + expand * bh)))


def safe_crop(img: np.ndarray, zone, target_px: int = 300_000) -> np.ndarray:
    """Область за столешницей, где стоит сейф/шкаф: полоса вокруг верхнего края
    зоны стола (шкаф стоит вплотную за столом и попадает в верх зоны), по ширине —
    сама зона с запасом. Выводится из зоны
    стола, а не из координат: ракурс между записями менялся."""
    h, w = img.shape[:2]
    zx1, zy1, zx2, zy2 = zone
    zh, zw = zy2 - zy1, zx2 - zx1
    x1, x2 = max(0, int(zx1 - 0.1 * zw)), min(w, int(zx2 + 0.1 * zw))
    y1, y2 = max(0, int(zy1 - 0.4 * zh)), min(h, int(zy1 + 0.45 * zh))
    sub = img[y1:y2, x1:x2]
    if sub.size == 0:
        return img
    k = (target_px / (sub.shape[0] * sub.shape[1])) ** 0.5
    if k > 1.05:
        sub = cv2.resize(sub, None, fx=k, fy=k, interpolation=cv2.INTER_LANCZOS4)
    return sub
