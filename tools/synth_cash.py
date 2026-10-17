"""Синтетические данные для детекции наличных в своём домене (cut-paste).

Приём из Dwibedi et al., ICCV 2017 «Cut, Paste and Learn»: для обучения детектора
достаточно реализма на уровне патча. Берём реальные вырезки банкнот из кадров
датасета, вклеиваем их в чистые столешницы с вариациями масштаба, поворота и
яркости — получаем тысячи размеченных положительных примеров в ТОМ ЖЕ домене
(та же камера, тот же стол, то же разрешение), где zero-shot модели не работают.

Обучается классификатор окна: «в этом окне столешницы лежат банкноты или нет».
Проверка — leave-one-source-out: обучаемся на двух вырезках, проверяемся на третьей
и на реальных кадрах из чистых роликов.
"""
import argparse, json, random, sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "service"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from app import objects, perception  # noqa: E402
from evaluate import LABELS  # noqa: E402

VID = "/Users/miras/Downloads/open/videos"
SYNTH = Path("data/synth")

# Вырезки, где наличные подтверждены просмотром кадра (см. data/synth/sources.jpg)
SOURCES = [
    ("X-2", 10, (143, 420, 180, 445)),    # пачка 1000 тг на столешнице
    ("X-1", 22, (172, 408, 206, 446)),    # веер купюр в руке
    ("Х-4", 57, (137, 442, 163, 456)),    # пачка на столешнице у счётной машины
]
WIN = 56          # сторона окна в пикселях исходного кадра
RNG = random.Random(20261017)


def grab(name, sec):
    cap = cv2.VideoCapture(f"{VID}/{name}.mov")
    cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
    ok, img = cap.read(); cap.release()
    return img if ok else None


def load_sources():
    out = []
    for name, sec, box in SOURCES:
        img = grab(name, sec)
        if img is None:
            continue
        x1, y1, x2, y2 = box
        patch = img[y1:y2, x1:x2]
        if patch.size:
            out.append((f"{name}@{sec}", patch))
    return out


def clean_desk_frames(limit_videos=10, per_video=6):
    """Кадры столешницы из ЧИСТЫХ роликов — фон для вклейки и источник негативов."""
    frames = []
    names = [n for n in sorted(LABELS) if not LABELS[n]][:limit_videos]
    for n in names:
        p = Path(f"{VID}/{n}.mov")
        if not p.exists():
            continue
        fr = perception.sample(str(p), fps=1.0)
        cache = Path(f"data/cache/{n}.detr.json")
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
            continue
        step = max(1, len(fr) // per_video)
        for f in fr[::step][:per_video]:
            frames.append((n, f.img, zone))
    return frames


def paste(bg, patch, zone):
    """Вклейка патча в случайное место столешницы с вариациями."""
    x1, y1, x2, y2 = zone
    k = RNG.uniform(0.8, 1.35)
    ang = RNG.uniform(-18, 18)
    p = cv2.resize(patch, None, fx=k, fy=k, interpolation=cv2.INTER_LINEAR)
    h, w = p.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    p = cv2.warpAffine(p, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if max(h, w) >= min(x2 - x1, y2 - y1) - 4:
        return None, None
    px = RNG.randint(x1 + 2, max(x1 + 3, x2 - w - 2))
    py = RNG.randint(y1 + 2, max(y1 + 3, y2 - h - 2))

    out = bg.copy()
    roi = out[py:py + h, px:px + w]
    if roi.shape[:2] != p.shape[:2]:
        return None, None
    # подгоняем яркость патча под локальный фон — иначе классификатор выучит «яркое пятно»
    gain = (float(roi.mean()) + 1e-3) / (float(p.mean()) + 1e-3)
    p = np.clip(p.astype(np.float32) * np.clip(gain, 0.7, 1.4), 0, 255).astype(np.uint8)
    if RNG.random() < 0.5:
        p = cv2.GaussianBlur(p, (3, 3), 0.6)
    # мягкие края: резкая граница — главный артефакт вклейки
    mask = np.zeros((h, w), np.float32)
    cv2.rectangle(mask, (1, 1), (w - 2, h - 2), 1.0, -1)
    mask = cv2.GaussianBlur(mask, (5, 5), 1.2)[..., None]
    out[py:py + h, px:px + w] = (p * mask + roi * (1 - mask)).astype(np.uint8)
    return out, (px + w // 2, py + h // 2)


def window(img, cx, cy, size=WIN):
    h, w = img.shape[:2]
    half = size // 2
    x1, y1 = max(0, cx - half), max(0, cy - half)
    x2, y2 = min(w, cx + half), min(h, cy + half)
    sub = img[y1:y2, x1:x2]
    if sub.size == 0:
        return None
    return cv2.resize(sub, (96, 96), interpolation=cv2.INTER_LINEAR)


def build(n_pos_per_bg=3, n_neg_per_bg=6):
    srcs = load_sources()
    bgs = clean_desk_frames()
    print(f"источников банкнот: {len(srcs)}, фоновых кадров столешницы: {len(bgs)}",
          file=sys.stderr)
    X, y, grp = [], [], []
    for vid, bg, zone in bgs:
        for _ in range(n_pos_per_bg):
            si = RNG.randrange(len(srcs))
            comp, c = paste(bg, srcs[si][1], zone)
            if comp is None:
                continue
            win = window(comp, *c)
            if win is not None:
                X.append(win); y.append(1); grp.append(si)
        zx1, zy1, zx2, zy2 = zone
        for _ in range(n_neg_per_bg):
            cx = RNG.randint(zx1 + WIN // 2, max(zx1 + WIN // 2 + 1, zx2 - WIN // 2))
            cy = RNG.randint(zy1 + WIN // 2, max(zy1 + WIN // 2 + 1, zy2 - WIN // 2))
            win = window(bg, cx, cy)
            if win is not None:
                X.append(win); y.append(0); grp.append(-1)
    return np.array(X), np.array(y), np.array(grp), srcs


def embed(batch_imgs):
    import torch
    from transformers import AutoImageProcessor, AutoModel
    global _EMB
    try:
        proc, model, dev = _EMB
    except NameError:
        mid = "google/siglip-base-patch16-224"
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        proc = AutoImageProcessor.from_pretrained(mid)
        model = AutoModel.from_pretrained(mid).to(dev).eval()
        _EMB = (proc, model, dev)
    out = []
    with torch.inference_mode():
        for s in range(0, len(batch_imgs), 64):
            imgs = [cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in batch_imgs[s:s + 64]]
            inp = proc(images=imgs, return_tensors="pt").to(dev)
            e = model.get_image_features(**inp)
            e = getattr(e, "pooler_output", e)
            if not hasattr(e, "norm"):
                e = e.last_hidden_state.mean(1)
            e = e / e.norm(dim=-1, keepdim=True)
            out.append(e.cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", default="data/synth/cash_clf.npz")
    a = ap.parse_args()
    SYNTH.mkdir(parents=True, exist_ok=True)

    X, y, grp, srcs = build()
    print(f"выборка: {len(y)} окон, положительных {int(y.sum())}", file=sys.stderr)
    E = embed(list(X))

    from sklearn.linear_model import LogisticRegression
    # leave-one-source-out: обучаемся на двух вырезках, проверяемся на третьей
    print(f"\n{'held-out источник':22s} {'точность':>9s} {'на положительных':>17s}")
    for si, (label, _) in enumerate(srcs):
        tr = (grp != si)
        te = (grp == si)
        if te.sum() == 0:
            continue
        clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")
        clf.fit(E[tr], y[tr])
        # негативы в тест добавляем из общего пула (у них grp=-1)
        neg = np.where(grp == -1)[0][:int(te.sum())]
        idx = np.concatenate([np.where(te)[0], neg])
        acc = (clf.predict(E[idx]) == y[idx]).mean()
        rec = (clf.predict(E[np.where(te)[0]]) == 1).mean()
        print(f"{label:22s} {100*acc:8.0f}% {100*rec:16.0f}%")

    clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced").fit(E, y)
    np.savez(a.save, coef=clf.coef_, intercept=clf.intercept_, win=WIN)
    print(f"\nклассификатор сохранён: {a.save}", file=sys.stderr)


if __name__ == "__main__":
    main()
