"""Florence-2 (MIT) как быстрый объектный канал: grounding по фразе.

Проверяем на кадрах с известным содержимым + контрольных кадрах из чистых роликов.
Интересует: находит ли банкноты/стакан/сумку на столешнице и не находит ли их там,
где их нет. Плюс скорость: 0.23B/0.77B против 7B — разница на порядок.
"""
import json, sys, time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "service"))
from app import objects, perception  # noqa: E402

VID = "/Users/miras/Downloads/open/videos"
MODEL = sys.argv[1] if len(sys.argv) > 1 else "microsoft/Florence-2-base"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"

PHRASES = ["banknotes", "paper money", "a cup", "a can", "a bag", "a mobile phone"]

CASES = [
    ("X-2", 10, "деньги на столе"),
    ("X-1", 22, "деньги в руке"),
    ("Х-46", 11, "стакан кофе"),
    ("Х-24", 8, "банка напитка"),
    ("Х-40", 17, "личная сумка"),
    ("Х-15", 15, "инкассаторская сумка"),
    ("Х-43", 12, "КОНТРОЛЬ чистый"),
    ("Х-3", 10, "КОНТРОЛЬ чистый"),
    ("Х-10", 7, "КОНТРОЛЬ чистый"),
    ("Х-35", 28, "КОНТРОЛЬ чистый"),
]


def load_frames(name):
    fr = perception.sample(f"{VID}/{name}.mov", fps=2.0)
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


def main():
    # transformers 5.x поддерживает Florence-2 нативно: remote code не нужен
    from transformers import Florence2ForConditionalGeneration, Florence2Processor
    proc = Florence2Processor.from_pretrained(MODEL)
    model = Florence2ForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.float32, attn_implementation="eager").to(DEV).eval()
    print(f"{MODEL} на {DEV}\n")

    times = []
    for name, sec, exp in CASES:
        fr = load_frames(name)
        zone = objects.desk_zone(fr)
        f = min(fr, key=lambda x: abs(x.t - sec))
        crop = objects.crop_object(f.img, zone, target_px=430_000, pad_rel=0.05)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        from PIL import Image
        img = Image.fromarray(rgb)

        print(f"=== {name}@{sec}s [{exp}]  {img.size}")
        for task, text in (("<MORE_DETAILED_CAPTION>", None),):
            prompt = task if text is None else task + text
            inputs = proc(text=prompt, images=img, return_tensors="pt").to(DEV)
            t = time.time()
            with torch.inference_mode():
                ids = model.generate(input_ids=inputs["input_ids"],
                                     pixel_values=inputs["pixel_values"],
                                     max_new_tokens=256, num_beams=1, do_sample=False)
            dt = time.time() - t; times.append(dt)
            out = proc.batch_decode(ids, skip_special_tokens=False)[0]
            parsed = proc.post_process_generation(out, task=task, image_size=img.size)
            labels = parsed.get(task, {})
            if isinstance(labels, dict):
                labels = list(zip(labels.get("labels", []), labels.get("bboxes", [])))
                short = ", ".join(f"{l}" for l, _ in labels[:8])
            else:
                short = str(labels)
            print(f"   [{dt:4.1f}с] {short}")
        print()
    print(f"скорость: среднее {np.mean(times):.2f} с/запрос")


if __name__ == "__main__":
    main()
