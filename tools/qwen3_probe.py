"""Проверка Qwen3-VL на кадрах, где провалились все предыдущие модели.

Задача одна: отличить пачку банкнот от листа бумаги или карты на 25x15 пикселях.
Контроль двусторонний — кадры с деньгами и кадры из чистых роликов без денег.
Вопрос задаётся как выбор из вариантов: свободное описание модель уводит в фантазии.
"""
import os, sys, time, tempfile
from pathlib import Path

import cv2

MODEL = os.environ.get("Q3_MODEL", "mlx-community/Qwen3-VL-4B-Instruct-4bit")
VID = "/Users/miras/Downloads/open/videos"

# (видео, секунда, бокс предмета, что это на самом деле)
CASES = [
    ("X-2", 10, (143, 420, 180, 445), "ДЕНЬГИ: пачка купюр 1000₸ на столе"),
    ("X-1", 22, (170, 405, 210, 450), "ДЕНЬГИ: веер купюр в руке"),
    ("Х-4", 57, (60, 390, 150, 450), "ДЕНЬГИ: пачка на столе"),
    ("Х-43", 12, (160, 375, 200, 405), "КОНТРОЛЬ: бумага/карта, денег нет"),
    ("Х-10", 7, (150, 390, 195, 420), "КОНТРОЛЬ: визитка, денег нет"),
    ("Х-46", 11, (195, 330, 225, 360), "КОНТРОЛЬ+: стакан кофе"),
]

PROMPT = ("На фото — фрагмент стола банковского кассира, снято сверху, кадр увеличен.\n"
          "Что находится в центре кадра? Выбери ОДИН вариант и ответь только им:\n"
          "A) пачка банкнот или деньги\n"
          "B) лист бумаги или документ\n"
          "C) пластиковая карта или визитка\n"
          "D) стакан, кружка или банка\n"
          "E) ничего из перечисленного\n"
          "Ответ: ")


def crop(video, sec, box, target_px=220_000):
    cap = cv2.VideoCapture(f"{VID}/{video}.mov")
    cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
    ok, img = cap.read(); cap.release()
    if not ok:
        return None
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    px, py = int(bw * 1.2), int(bh * 1.2)
    sub = img[max(0, y1 - py):y2 + py, max(0, x1 - px):x2 + px]
    k = (target_px / max(1, sub.shape[0] * sub.shape[1])) ** 0.5
    if k > 1:
        sub = cv2.resize(sub, None, fx=k, fy=k, interpolation=cv2.INTER_LANCZOS4)
    return sub


def main():
    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config
    t0 = time.time()
    model, processor = load(MODEL)
    config = load_config(MODEL)
    print(f"{MODEL}\nзагрузка {time.time()-t0:.0f}с\n")

    for video, sec, box, truth in CASES:
        img = crop(video, sec, box)
        if img is None:
            print(f"{video}@{sec}: кадр не прочитан"); continue
        p = tempfile.mktemp(suffix=".jpg")
        cv2.imwrite(p, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        fmt = apply_chat_template(processor, config, PROMPT, num_images=1)
        t = time.time()
        res = generate(model, processor, fmt, [p], max_tokens=24,
                       temperature=0.0, verbose=False)
        txt = getattr(res, "text", str(res)).strip().replace("\n", " ")
        os.unlink(p)
        print(f"{video}@{sec:3d} [{truth}]")
        print(f"    -> {txt[:90]}   [{time.time()-t:.1f}с, {img.shape[1]}x{img.shape[0]}]")


if __name__ == "__main__":
    main()
