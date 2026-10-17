"""Прогон детектора по видео -> покадровые детекции в JSON-кэш.

Модель: RT-DETR v2 (Apache-2.0, COCO-80). Даёт person / cup / bottle / cell phone /
handbag / backpack / laptop / chair — почти все «посторонние предметы» из таксономии.
Денег в COCO нет, они добираются отдельно (OWLv2, см. owl.py).
"""
import argparse, json, os, sys, time
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection

MODEL_ID = "PekingU/rtdetr_v2_r50vd"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
# классы COCO, которые нам интересны (остальные пишем тоже, фильтруем позже)
KEEP = None  # None = писать всё выше порога


def load_model():
    proc = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForObjectDetection.from_pretrained(MODEL_ID).to(DEVICE).eval()
    return proc, model


def sample_frames(path: str, fps: float):
    """Возвращает (timestamp_sec, BGR-кадр) с заданной частотой."""
    cap = cv2.VideoCapture(path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(src_fps / fps)))
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            yield i / src_fps, frame
        i += 1
    cap.release()


@torch.inference_mode()
def run_video(path: str, proc, model, fps: float, batch: int, thr: float):
    frames, stamps, out = [], [], []
    id2label = model.config.id2label

    def flush():
        if not frames:
            return
        imgs = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
        inputs = proc(images=imgs, return_tensors="pt").to(DEVICE)
        res = model(**inputs)
        sizes = torch.tensor([im.shape[:2] for im in imgs]).to(DEVICE)
        post = proc.post_process_object_detection(res, target_sizes=sizes, threshold=thr)
        for ts, p in zip(stamps, post):
            dets = []
            for score, label, box in zip(p["scores"], p["labels"], p["boxes"]):
                name = id2label[int(label)]
                if KEEP and name not in KEEP:
                    continue
                dets.append({
                    "cls": name,
                    "score": round(float(score), 3),
                    "box": [round(float(v), 1) for v in box.tolist()],  # xyxy
                })
            out.append({"t": round(ts, 3), "det": dets})
        frames.clear(); stamps.clear()

    for ts, frame in sample_frames(path, fps):
        frames.append(frame); stamps.append(ts)
        if len(frames) >= batch:
            flush()
    flush()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--out", default="data/cache")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--thr", type=float, default=0.30)
    a = ap.parse_args()

    Path(a.out).mkdir(parents=True, exist_ok=True)
    proc, model = load_model()
    print(f"device={DEVICE} model={MODEL_ID}", file=sys.stderr)

    for v in a.videos:
        name = Path(v).stem
        dst = Path(a.out) / f"{name}.detr.json"
        if dst.exists():
            print(f"skip {name}", file=sys.stderr); continue
        t0 = time.time()
        res = run_video(v, proc, model, a.fps, a.batch, a.thr)
        dst.write_text(json.dumps({
            "video": name, "fps_sampled": a.fps, "thr": a.thr,
            "model": MODEL_ID, "frames": res,
        }, ensure_ascii=False))
        print(f"{name}: {len(res)} frames in {time.time()-t0:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
