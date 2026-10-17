"""Банк «нормы»: эмбеддинги столешницы по ключевым кадрам чистых роликов.

Нужен для канала новизны (слой 4 из 00-ANALYSIS.md): кадр, непохожий ни на что
в банке, помечается как «иное отклонение от регламента». Это ответ на п.9 FAQ —
в тесте будут типы нарушений, которых нет в обучающих данных, и правила под них
написать нельзя по определению.

Эмбеддинги считаются по ТЕМ ЖЕ кропам, что уходят в Florence-2, и по тем же
секундам, что в корпусе подписей, — чтобы канал новизны и канал описаний
говорили об одних и тех же кадрах.

Результат (data/embeds/) в репозиторий не попадает: это производная данных
организаторов, FAQ A.6 требует использовать их только в рамках хакатона.
"""
import json, sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "service"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from app import objects, perception  # noqa: E402
from evaluate import LABELS  # noqa: E402

VID = "/Users/miras/Downloads/open/videos"
OUT = Path("data/embeds")
MODEL = "google/siglip-base-patch16-224"
_EMB = None


def embedder():
    global _EMB
    if _EMB is None:
        import torch
        from transformers import AutoImageProcessor, AutoModel
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        proc = AutoImageProcessor.from_pretrained(MODEL)
        model = AutoModel.from_pretrained(MODEL).to(dev).eval()
        _EMB = (proc, model, dev, torch)
    return _EMB


def embed(imgs):
    proc, model, dev, torch = embedder()
    out = []
    with torch.inference_mode():
        for s in range(0, len(imgs), 32):
            batch = [cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in imgs[s:s + 32]]
            inp = proc(images=batch, return_tensors="pt").to(dev)
            e = model.get_image_features(**inp)
            e = getattr(e, "pooler_output", e)
            if not hasattr(e, "norm"):
                e = e.last_hidden_state.mean(1)
            e = e / e.norm(dim=-1, keepdim=True)
            out.append(e.cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 768), np.float32)


def process(name):
    dst = OUT / f"{name}.npz"
    if dst.exists():
        return
    cp = Path(f"data/captions/{name}.caps.json")
    if not cp.exists():
        print(f"{name}: нет подписей, пропуск", file=sys.stderr); return
    want = [c["t"] for c in json.loads(cp.read_text())["captions"]]

    fr = perception.sample(f"{VID}/{name}.mov", fps=2.0)
    if not fr:
        return
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
    zone = objects.desk_zone(fr)
    if zone is None:
        h, w = fr[0].img.shape[:2]
        zone = (0, int(0.35 * h), w, h)

    crops, ts = [], []
    for t in want:
        f = min(fr, key=lambda x: abs(x.t - t))
        crops.append(objects.crop_object(f.img, zone, target_px=430_000, pad_rel=0.05))
        ts.append(round(f.t, 2))
    E = embed(crops)
    np.savez(dst, emb=E.astype(np.float32), t=np.array(ts, np.float32))
    print(f"{name}: {len(ts)} кадров", file=sys.stderr)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    names = sys.argv[1:] or sorted(LABELS)
    for n in names:
        if not Path(f"{VID}/{n}.mov").exists():
            continue
        process(n)
    print(f"\nготово: {len(list(OUT.glob('*.npz')))} роликов в {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
