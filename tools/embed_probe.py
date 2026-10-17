"""Есть ли в кадрах обучаемый сигнал? Эмбеддинги SigLIP + leave-one-out.

Честная проверка потолка: если по 40 роликам классификатор на замороженных
признаках не отличает нарушение от нормы, значит признак не в облике кадра,
а в тонком действии — и надо менять подход, а не крутить пороги.
"""
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "service"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import LABELS  # noqa: E402

VID = "/Users/miras/Downloads/open/videos"
CACHE = Path("data/emb")
MODEL = "google/siglip-base-patch16-224"


def extract():
    import cv2, torch
    from transformers import AutoImageProcessor, AutoModel
    from app import perception
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    proc = AutoImageProcessor.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).to(dev).eval()
    CACHE.mkdir(parents=True, exist_ok=True)

    for name in sorted(LABELS):
        dst = CACHE / f"{name}.npz"
        if dst.exists():
            continue
        p = f"{VID}/{name}.mov"
        if not Path(p).exists():
            continue
        frames = perception.sample(p, fps=2.0)
        embs, ts = [], []
        with torch.inference_mode():
            for s in range(0, len(frames), 16):
                chunk = frames[s:s + 16]
                imgs = [cv2.cvtColor(f.img, cv2.COLOR_BGR2RGB) for f in chunk]
                inp = proc(images=imgs, return_tensors="pt").to(dev)
                e = model.get_image_features(**inp)
                e = getattr(e, 'pooler_output', e)
                if not hasattr(e, 'norm'):
                    e = e.last_hidden_state.mean(1)
                e = e / e.norm(dim=-1, keepdim=True)
                embs.append(e.cpu().numpy())
                ts += [f.t for f in chunk]
        np.savez(dst, emb=np.concatenate(embs), t=np.array(ts))
        print(f"{name}: {len(ts)} кадров", file=sys.stderr)


def evaluate():
    from sklearn.linear_model import LogisticRegression
    X, y, names, raw = [], [], [], {}
    for f in sorted(CACHE.glob("*.npz")):
        name = f.stem
        if name not in LABELS:
            continue
        d = np.load(f)
        e = d["emb"]
        raw[name] = (e, d["t"])
        X.append(np.concatenate([e.mean(0), e.max(0), e.std(0)]))
        y.append(int(bool(LABELS[name])))
        names.append(name)
    X = np.array(X); y = np.array(y)
    print(f"выборка: {len(y)} роликов, нарушений {y.sum()}, признак {X.shape[1]}")

    # leave-one-out: единственный честный протокол на 40 примерах
    ok = 0; preds = []
    for i in range(len(y)):
        m = np.ones(len(y), bool); m[i] = False
        clf = LogisticRegression(max_iter=2000, C=0.5).fit(X[m], y[m])
        p = clf.predict_proba(X[i:i + 1])[0, 1]
        preds.append(p); ok += int((p > 0.5) == bool(y[i]))
    preds = np.array(preds)
    print(f"LOO точность: {100*ok/len(y):.0f}%  (случайное угадывание ~50%)")
    auc = np.mean([(preds[i] > preds[j]) + 0.5 * (preds[i] == preds[j])
                   for i in np.where(y == 1)[0] for j in np.where(y == 0)[0]])
    print(f"LOO AUC: {auc:.2f}")
    for n, p, t in sorted(zip(names, preds, y), key=lambda r: -r[1]):
        print(f"  {n:7s} GT={'V' if t else '.'} p={p:.2f} {'OK' if (p>0.5)==bool(t) else 'ОШИБКА'}")

    # где внутри видео максимум «нарушенности» -> проверка таймкода
    clf = LogisticRegression(max_iter=2000, C=0.5).fit(X, y)
    w = clf.coef_[0][:X.shape[1] // 3]          # веса для mean-части
    errs = []
    for n, (e, t) in raw.items():
        if not LABELS.get(n):
            continue
        score = e @ w
        tp = t[int(np.argmax(score))]
        d = min(abs(tp - g) for g in LABELS[n])
        errs.append(d)
    if errs:
        errs.sort()
        print(f"\nтаймкод по максимуму score: медиана {errs[len(errs)//2]:.1f}с, "
              f"≤3с у {100*sum(e<=3 for e in errs)/len(errs):.0f}%")


if __name__ == "__main__":
    if "--eval" not in sys.argv:
        extract()
    evaluate()
