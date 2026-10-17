"""OWLv2 по кропу зоны стола + апскейл. Проверяем, улучшается ли ярлык объекта.

Также меряем время инференса — важно для лимита 5 минут на видео.
"""
import subprocess, tempfile, os, time
import torch
from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection

MODEL = "google/owlv2-base-patch16-ensemble"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
VID = "/Users/miras/Downloads/open/videos"

PROMPTS = [
    "banknotes", "a stack of cash money", "paper currency bills",
    "a coffee cup", "a soda can", "a bottle",
    "a handbag", "a purse", "a smartphone",
    "a sheet of paper", "a plastic card",
]

# видео, секунда, crop(w,h,x,y) по зоне стола, ожидание
CASES = [
    ("X-2", 10, (300, 160, 60, 380), "деньги на столе"),
    ("X-1", 22, (300, 160, 60, 380), "деньги в руке"),
    ("Х-24", 8, (340, 200, 100, 380), "банка напитка"),
    ("Х-40", 17, (340, 200, 100, 380), "личная сумка"),
    ("Х-46", 11, (340, 200, 100, 330), "стакан кофе"),
    ("Х-43", 12, (340, 200, 100, 330), "НЕГАТИВНЫЙ КОНТРОЛЬ"),
    ("Х-3", 10, (300, 160, 60, 380), "НЕГАТИВНЫЙ КОНТРОЛЬ"),
    ("Х-10", 7, (300, 160, 60, 380), "НЕГАТИВНЫЙ КОНТРОЛЬ"),
]


def grab(video, sec, crop, up=3):
    out = tempfile.mktemp(suffix=".png")
    w, h, x, y = crop
    vf = f"crop={w}:{h}:{x}:{y},scale=iw*{up}:ih*{up}:flags=lanczos"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(sec), "-i",
                    f"{VID}/{video}.mov", "-vf", vf, "-frames:v", "1", out], check=True)
    img = Image.open(out).convert("RGB"); os.unlink(out)
    return img


@torch.inference_mode()
def main():
    proc = Owlv2Processor.from_pretrained(MODEL)
    model = Owlv2ForObjectDetection.from_pretrained(MODEL).to(DEV).eval()
    times = []
    for video, sec, crop, expect in CASES:
        img = grab(video, sec, crop)
        t0 = time.time()
        inputs = proc(text=[PROMPTS], images=img, return_tensors="pt").to(DEV)
        out = model(**inputs)
        res = proc.post_process_grounded_object_detection(
            out, target_sizes=torch.tensor([img.size[::-1]]).to(DEV), threshold=0.07)[0]
        times.append(time.time() - t0)
        hits = sorted(((PROMPTS[int(l)], float(s), [int(v) for v in b.tolist()])
                       for s, l, b in zip(res["scores"], res["labels"], res["boxes"])),
                      key=lambda x: -x[1])
        print(f"=== {video} @{sec}s [{expect}] crop{crop} -> {img.size}")
        for name, score, box in hits[:5]:
            print(f"    {score:.3f}  {name:24s} {box}")
        if not hits:
            print("    (ничего выше порога)")
        print()
    print(f"инференс: среднее {sum(times)/len(times):.2f} с/кадр, макс {max(times):.2f} с")


if __name__ == "__main__":
    main()
