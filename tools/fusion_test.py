"""Повышение эффективного разрешения: многокадровое накопление и супер-разрешение.

Камера статична, поэтому соседние кадры отличаются только шумом сенсора и
артефактами сжатия. Медиана по 5-7 кадрам убирает их и даёт более чистый мелкий
объект — классический приём для видеонаблюдения со статичной камеры.

Контролируемый эксперимент: берём кадры, где известно, что должно быть видно,
и сравниваем базовый кроп против накопленного. Среди случаев — вклеенный стакан
18x24 px, который базовый пайплайн НЕ нашёл: если накопление его вытягивает,
значит предел сдвигается.
"""
import json, sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "service"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from app import caption, objects, perception  # noqa: E402

VID = "/Users/miras/Downloads/open/videos"

CASES = [
    ("data/planted/Х-43__planted_стакан.mov", 12.0, "стакан 18x24 — базовый НЕ нашёл"),
    ("data/planted/Х-43__planted_стакан_x1.5.mov", 12.0, "стакан 27x36 — базовый нашёл"),
    (f"{VID}/Х-46.mov", 11.0, "реальный стакан — базовый нашёл"),
    (f"{VID}/X-2.mov", 10.0, "пачка купюр на столе"),
    (f"{VID}/Х-43.mov", 12.0, "КОНТРОЛЬ: чистый, предметов нет"),
]


def frames_around(path, sec, n=7, fps=2.0):
    """n кадров вокруг секунды. Камера статична — выравнивание не нужно."""
    fr = perception.sample(path, fps=fps)
    if not fr:
        return [], None
    i = min(range(len(fr)), key=lambda k: abs(fr[k].t - sec))
    lo = max(0, i - n // 2); hi = min(len(fr), lo + n)
    return fr[lo:hi], fr


def fuse(imgs, mode="median"):
    st = np.stack([i.astype(np.float32) for i in imgs])
    out = np.median(st, axis=0) if mode == "median" else st.mean(axis=0)
    return np.clip(out, 0, 255).astype(np.uint8)


def sharpen(img, amount=0.6):
    """Лёгкое повышение резкости после накопления: медиана слегка размывает."""
    blur = cv2.GaussianBlur(img, (0, 0), 1.0)
    return cv2.addWeighted(img, 1 + amount, blur, -amount, 0)


def main():
    caption.warmup()
    print(f"описатель: {caption.MODEL_ID}\n")
    for path, sec, label in CASES:
        if not Path(path).exists():
            print(f"{path}: нет файла"); continue
        window, allfr = frames_around(path, sec)
        if not window:
            continue
        name = Path(path).stem
        cache = Path(f"data/cache/{name.split('__')[0]}.detr.json")
        if cache.exists():
            det = json.loads(cache.read_text())["frames"]
            byt = {round(d["t"], 2): d["det"] for d in det}
            for f in allfr:
                ds = byt.get(round(f.t, 2), [])
                f.persons = [d for d in ds if d["cls"] == "person" and d["score"] >= 0.35]
                f.objects = [d for d in ds if d["cls"] != "person" and d["score"] >= 0.35]
        else:
            perception.detect(allfr)
        zone = objects.desk_zone(allfr)
        mid = window[len(window) // 2]

        variants = {
            "базовый": mid.img,
            "медиана 7 кадров": fuse([f.img for f in window]),
            "медиана + резкость": sharpen(fuse([f.img for f in window])),
        }
        print(f"=== {name} @{sec} с  [{label}]")
        for vname, img in variants.items():
            crop = objects.crop_object(img, zone, target_px=430_000, pad_rel=0.05)
            txt = caption.caption(crop)
            f = caption.facts_from_caption(txt)
            flags = [k for k in ("cash_seen", "drink_or_food", "personal_bag",
                                 "sealed_bag", "paper_seen") if f[k]]
            print(f"   {vname:20s} [{' '.join(flags) if flags else '—'}]")
            print(f"      {txt[:150]}")
        print()


if __name__ == "__main__":
    main()
