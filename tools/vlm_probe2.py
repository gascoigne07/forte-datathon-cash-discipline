"""Та же проверка VLM, но кадр обрезается по автоматически найденной зоне интереса.

Гипотеза: модель не видит мелкие предметы не из-за «слабости», а из-за того,
что пачка купюр занимает ~25x15 пикселей в кадре 464x848.
"""
import sys, time
from pathlib import Path
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "service"))
import vlm_qwen25_measured as vlm
from app import perception  # noqa: E402

VID = "/Users/miras/Downloads/open/videos"

CASES = [
    ("X-2", 10, "cash_on_desk (пачка 1000₸ на столешнице)"),
    ("X-1", 22, "cash_in_hands (веер купюр в руке)"),
    ("Х-46", 11, "drink_or_food (стакан кофе)"),
    ("Х-24", 8, "drink_or_food (банка)"),
    ("Х-40", 17, "personal_bag (бежевая сумка)"),
    ("Х-15", 15, "sealed_bag (зелёная сумка) + teller_present=False"),
    ("Х-13", 47, "teller_present=False (пустое место)"),
    ("Х-18", 20, "people_in_cabin>=3"),
    ("Х-43", 12, "КОНТРОЛЬ: всё False"),
    ("Х-3", 10, "КОНТРОЛЬ: кассир+клиент"),
    ("Х-29", 17, "КОНТРОЛЬ: кассир выходит, НЕ нарушение"),
    ("Х-35", 28, "КОНТРОЛЬ: посторонний в кабине, НЕ нарушение"),
]

_cache: dict[str, tuple] = {}


def prep(video, sec, dump=None):
    """ROI считаем по всему видео (как в бою), кадр берём нужный."""
    if video not in _cache:
        fr = perception.sample(f"{VID}/{video}.mov", fps=2.0)
        perception.detect(fr)
        _cache[video] = (fr, perception.roi(fr))
    frames, box = _cache[video]
    f = min(frames, key=lambda x: abs(x.t - sec))
    img = perception.crop_scaled(f.img, box)
    if dump:
        cv2.imwrite(dump, img)
    return img, box


def main():
    t0 = time.time(); vlm.warmup()
    print(f"модель {vlm.MODEL_ID}, загрузка {time.time()-t0:.0f}с\n")
    times = []
    outdir = Path("data/roi_dump"); outdir.mkdir(parents=True, exist_ok=True)
    for video, sec, expect in CASES:
        img, box = prep(video, sec, str(outdir / f"{video}_{sec}.jpg"))
        t = time.time()
        try:
            f = vlm.facts(img)
        except Exception as exc:                            # noqa: BLE001
            print(f"{video}@{sec}: ОШИБКА {type(exc).__name__}: {exc}"); continue
        times.append(time.time() - t)
        flags = " ".join(k.replace("_", "")[:9] + "=" + ("1" if f[k] else "0")
                         for k in ("teller_present", "cash_on_desk", "cash_in_hands",
                                   "machine_in_use", "drink_or_food", "personal_bag",
                                   "sealed_bag", "door_open"))
        print(f"{video}@{sec}s  ждём: {expect}   roi={box} -> {img.shape[1]}x{img.shape[0]}")
        print(f"   люди={f['people_in_cabin']} {flags}")
        print(f"   «{f['what_happens']}»  [{times[-1]:.1f}с]\n")
    if times:
        print(f"скорость: среднее {sum(times)/len(times):.1f} с/кадр")


if __name__ == "__main__":
    main()
