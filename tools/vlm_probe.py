"""Решающая проверка: видит ли локальная VLM нужные факты на кадрах датасета.

Контроль двусторонний — кадры с известным нарушением И кадры из чистых роликов.
Если модель «видит» деньги и на чистых кадрах, факт бесполезен как признак.
"""
import subprocess, sys, tempfile, os, time
from pathlib import Path
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "service"))
import vlm_qwen25_measured as vlm  # noqa: E402

VID = "/Users/miras/Downloads/open/videos"

CASES = [
    # (видео, сек, что ожидаем увидеть)
    ("X-2", 10, "cash_on_desk=True (пачка 1000₸ на столешнице)"),
    ("X-1", 22, "cash_in_hands=True (веер купюр в руке)"),
    ("Х-46", 11, "drink_or_food=True (стакан кофе)"),
    ("Х-24", 8, "drink_or_food=True (банка) + teller_present=False"),
    ("Х-40", 17, "personal_bag=True (бежевая сумка на столе)"),
    ("Х-15", 15, "sealed_bag=True (зелёная сумка) + teller_present=False"),
    ("Х-13", 47, "teller_present=False (пустое рабочее место)"),
    ("Х-18", 20, "people_in_cabin>=3 (двое посторонних)"),
    ("Х-43", 12, "КОНТРОЛЬ: всё False"),
    ("Х-3", 10, "КОНТРОЛЬ: кассир+клиент, предметов нет"),
    ("Х-29", 17, "КОНТРОЛЬ: кассир выходит, но это НЕ нарушение"),
    ("Х-35", 28, "КОНТРОЛЬ: посторонний в кабине, но это НЕ нарушение"),
]


def grab(video, sec):
    out = tempfile.mktemp(suffix=".png")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(sec),
                    "-i", f"{VID}/{video}.mov", "-frames:v", "1", out], check=True)
    img = cv2.imread(out); os.unlink(out)
    return img


def main():
    up = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    t0 = time.time(); vlm.warmup()
    print(f"модель {vlm.MODEL_ID} загружена за {time.time()-t0:.0f}с, upscale=x{up}\n")
    times = []
    for video, sec, expect in CASES:
        img = grab(video, sec)
        t = time.time()
        try:
            f = vlm.facts(img, upscale=up)
        except Exception as exc:                            # noqa: BLE001
            print(f"{video}@{sec}: ОШИБКА {type(exc).__name__}: {exc}"); continue
        times.append(time.time() - t)
        flags = " ".join(k.replace("_", "")[:9] + "=" + ("1" if f[k] else "0")
                         for k in ("teller_present", "cash_on_desk", "cash_in_hands",
                                   "machine_in_use", "drink_or_food", "personal_bag",
                                   "sealed_bag", "door_open"))
        print(f"{video}@{sec}s  ожидали: {expect}")
        print(f"   люди={f['people_in_cabin']} {flags}")
        print(f"   «{f['what_happens']}»  [{times[-1]:.1f}с]\n")
    if times:
        print(f"скорость: среднее {sum(times)/len(times):.1f} с/кадр, макс {max(times):.1f} с")


if __name__ == "__main__":
    main()
