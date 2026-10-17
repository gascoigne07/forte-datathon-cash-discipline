"""Единый пер-секундный экстрактор признаков по кэшу детекций + геометрии сцены.

Все величины нормированы на размер стола, а не на пиксели кадра: иначе смена
ракурса и крупности плана (а финальное видео жюри снято под другим углом)
обнуляет любые пороги.
"""
import json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "service"))
from app import objects, perception  # noqa: E402

VID = "/Users/miras/Downloads/open/videos"


def load(name, fps=2.0):
    fr = perception.sample(f"{VID}/{name}.mov", fps=fps)
    cache = Path(f"data/cache/{name}.detr.json")
    if cache.exists():
        det = json.loads(cache.read_text())["frames"]
        byt = {round(d["t"], 2): d["det"] for d in det}
        for f in fr:
            ds = byt.get(round(f.t, 2), [])
            f.persons = [d for d in ds if d["cls"] == "person" and d["score"] >= 0.35]
            f.objects = [d for d in ds if d["cls"] != "person" and d["score"] >= 0.35]
    else:
        perception.detect(fr)
    return fr


def _dist_to_zone(box, zone):
    """Расстояние от бокса до прямоугольника зоны (0 если пересекаются)."""
    x1, y1, x2, y2 = box
    zx1, zy1, zx2, zy2 = zone
    dx = max(zx1 - x2, x1 - zx2, 0)
    dy = max(zy1 - y2, y1 - zy2, 0)
    return (dx * dx + dy * dy) ** 0.5


def machine_box(frames):
    """Счётно-денежная машина: самый устойчивый тёмный объект на столе.
    RT-DETR видит её как suitcase/backpack/tvmonitor."""
    cand = {}
    for f in frames:
        for o in f.objects:
            if o["cls"] in ("suitcase", "backpack", "tvmonitor") and o["score"] >= 0.4:
                cand.setdefault(o["cls"], []).append(o["box"])
    best = None
    for cls, boxes in cand.items():
        if len(boxes) < 0.5 * len(frames):
            continue
        arr = np.array(boxes)
        med = np.median(arr, axis=0)
        spread = float(np.abs(arr - med).mean())
        if best is None or spread < best[0]:
            best = (spread, med)
    return None if best is None else tuple(float(v) for v in best[1])


def extract(name):
    fr = load(name)
    if not fr:
        return None
    h, w = fr[0].img.shape[:2]
    bg = objects.background(fr)
    zone = objects.desk_zone(fr, bg=bg)
    if zone is None:
        zone = (0, int(0.35 * h), w, h)
    zw = max(1.0, zone[2] - zone[0])
    mach = machine_box(fr)
    tracks = objects.detect_objects(fr, zone=zone)

    rows = []
    for f in fr:
        ppl = perception.persons_in_room(f, desk_top=zone[1])
        # «кассир» = самый крупный человек в кабине
        teller = max(ppl, key=lambda p: (p["box"][3] - p["box"][1]) * (p["box"][2] - p["box"][0]),
                     default=None)
        d = _dist_to_zone(teller["box"], zone) / zw if teller else None
        hands_at_machine = None
        if teller and mach:
            hands_at_machine = _dist_to_zone(teller["box"], mach) / zw
        # Тем же кодом, что и сервис (perception.split_by_desk): признак «за
        # столешницей никого, но в комнате кто-то есть» — наблюдаемая часть п.22.
        at_desk, off_desk = perception.split_by_desk(ppl, zone)
        rows.append({
            "t": round(f.t, 2),
            "ppl": len(ppl),
            "at_desk": at_desk,
            "off_desk": off_desk,
            "teller_dist": None if d is None else round(d, 3),
            "machine_dist": None if hands_at_machine is None else round(hands_at_machine, 3),
            "motion": round(f.motion, 2),
        })
        f.img = None

    return {
        "video": name, "zone": [int(v) for v in zone],
        "zone_w": round(zw, 1), "machine": None if mach is None else [round(v, 1) for v in mach],
        "tracks": [{"t0": t.t_start, "t1": t.t_end, "area": round(t.area, 1),
                    "box": [int(v) for v in t.box], "near_person": t.near_person}
                   for t in tracks],
        "rows": rows,
    }


def main():
    from evaluate import LABELS
    out = Path("data/feats"); out.mkdir(parents=True, exist_ok=True)
    names = sys.argv[1:] or sorted(LABELS)
    for name in names:
        dst = out / f"{name}.feats.json"
        if dst.exists():
            continue
        if not Path(f"{VID}/{name}.mov").exists():
            continue
        d = extract(name)
        if d:
            dst.write_text(json.dumps(d, ensure_ascii=False))
            print(f"{name}: {len(d['rows'])} строк, зона={d['zone']}, машина={'да' if d['machine'] else 'нет'}",
                  file=sys.stderr)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
