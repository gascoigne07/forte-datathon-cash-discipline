"""Проверка OWLv2 (open-vocab детекция) на кадрах датасета.

Контроль двусторонний: позитивный (кадры, где предмет ТОЧНО есть) и
негативный (чистый кадр, где его быть не должно). Без негативного контроля
нельзя отличить «нашёл» от «находит везде».
"""
import sys, subprocess, tempfile, os
from pathlib import Path
import numpy as np, torch
from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection

MODEL = "google/owlv2-base-patch16-ensemble"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
VID = "/Users/miras/Downloads/open/videos"

PROMPTS = [
    "a stack of paper money banknotes",
    "money in a person's hand",
    "a paper coffee cup",
    "a drink can",
    "a handbag",
    "a mobile phone",
    "a white paper document",
]

# (видео, секунда, что ожидаем, crop|full)
CASES = [
    ("X-2", 10, "деньги на столе"),
    ("X-1", 22, "деньги в руке"),
    ("Х-46", 11, "стакан кофе"),
    ("Х-24", 8, "банка напитка"),
    ("Х-40", 17, "сумка на столе"),
    ("Х-43", 12, "НЕГАТИВНЫЙ КОНТРОЛЬ (чистый ролик)"),
    ("Х-3", 10, "НЕГАТИВНЫЙ КОНТРОЛЬ (чистый ролик)"),
]


def grab(video, sec, scale=1):
    out = tempfile.mktemp(suffix=".png")
    vf = f"scale=iw*{scale}:ih*{scale}:flags=lanczos" if scale != 1 else "null"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(sec), "-i",
                    f"{VID}/{video}.mov", "-vf", vf, "-frames:v", "1", out], check=True)
    img = Image.open(out).convert("RGB")
    os.unlink(out)
    return img


@torch.inference_mode()
def detect(proc, model, img, thr=0.08):
    inputs = proc(text=[PROMPTS], images=img, return_tensors="pt").to(DEV)
    out = model(**inputs)
    res = proc.post_process_grounded_object_detection(
        out, target_sizes=torch.tensor([img.size[::-1]]).to(DEV), threshold=thr)[0]
    hits = []
    for s, l, b in zip(res["scores"], res["labels"], res["boxes"]):
        hits.append((PROMPTS[int(l)], round(float(s), 3),
                     [int(v) for v in b.tolist()]))
    return sorted(hits, key=lambda x: -x[1])


def main():
    scale = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    proc = Owlv2Processor.from_pretrained(MODEL)
    model = Owlv2ForObjectDetection.from_pretrained(MODEL).to(DEV).eval()
    print(f"device={DEV} scale=x{scale}\n")
    for video, sec, expect in CASES:
        img = grab(video, sec, scale)
        hits = detect(proc, model, img)
        print(f"=== {video} @{sec}s [{expect}] {img.size}")
        for name, score, box in hits[:6]:
            print(f"    {score:.3f}  {name:34s} {box}")
        if not hits:
            print("    (ничего выше порога)")
        print()


if __name__ == "__main__":
    main()
