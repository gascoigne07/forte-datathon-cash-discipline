"""Корпус подписей Florence-2 по датасету (подход LAVAD, но локально).

Florence-2-large на зоне столешницы: ~4 с/кадр против 13 с у Qwen-7B, и именно
он надёжно называет стакан и сумки. Подписи кэшируются, дальше их майнят правила.
"""
import argparse, json, sys, time
from pathlib import Path

import cv2
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "service"))
from app import objects, perception  # noqa: E402

MODEL = "florence-community/Florence-2-large"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
TASK = "<MORE_DETAILED_CAPTION>"


def load_frames(path, name):
    fr = perception.sample(path, fps=2.0)
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
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--out", default="data/captions")
    ap.add_argument("--step", type=float, default=1.0, help="шаг по времени, с")
    ap.add_argument("--cap", type=int, default=25)
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)

    from transformers import Florence2ForConditionalGeneration, Florence2Processor
    proc = Florence2Processor.from_pretrained(MODEL)
    model = Florence2ForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.float32, attn_implementation="eager").to(DEV).eval()
    print(f"{MODEL} на {DEV}", file=sys.stderr)

    for v in a.videos:
        name = Path(v).stem
        dst = Path(a.out) / f"{name}.caps.json"
        if dst.exists():
            continue
        t0 = time.time()
        fr = load_frames(v, name)
        if not fr:
            continue
        zone = objects.desk_zone(fr)
        dt = fr[1].t - fr[0].t if len(fr) > 1 else 0.5
        stride = max(1, int(round(a.step / dt)))
        idx = list(range(0, len(fr), stride))[:a.cap]

        caps = []
        for i in idx:
            crop = objects.crop_object(fr[i].img, zone, target_px=430_000, pad_rel=0.05)
            img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            inputs = proc(text=TASK, images=img, return_tensors="pt").to(DEV)
            with torch.inference_mode():
                ids = model.generate(input_ids=inputs["input_ids"],
                                     pixel_values=inputs["pixel_values"],
                                     max_new_tokens=140, num_beams=1, do_sample=False)
            out = proc.batch_decode(ids, skip_special_tokens=False)[0]
            txt = proc.post_process_generation(out, task=TASK, image_size=img.size)[TASK]
            caps.append({"t": round(fr[i].t, 2), "caption": txt.strip()})
            fr[i].img = None
        dst.write_text(json.dumps({"video": name, "model": MODEL, "zone": [int(v) for v in zone] if zone else None,
                                   "captions": caps}, ensure_ascii=False, indent=1))
        d = time.time() - t0
        print(f"{name}: {len(caps)} подписей за {d:.0f}с ({d/max(1,len(caps)):.1f} с/кадр)",
              file=sys.stderr)


if __name__ == "__main__":
    main()
