"""Скрининг дешёвых признаков против разметки: что вообще разделяет классы.

Смысл: прежде чем вкладываться в правило, проверить, есть ли в признаке сигнал.
Считается по кэшу детекций, без VLM — секунды на всю выборку.
"""
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "service"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from app import perception, objects  # noqa: E402
from evaluate import LABELS  # noqa: E402

VID = "/Users/miras/Downloads/open/videos"


def load(name):
    fr = perception.sample(f"{VID}/{name}.mov", fps=2.0)
    det = json.loads(Path(f"data/cache/{name}.detr.json").read_text())["frames"]
    byt = {round(d["t"], 2): d["det"] for d in det}
    for f in fr:
        ds = byt.get(round(f.t, 2), [])
        f.persons = [d for d in ds if d["cls"] == "person" and d["score"] >= 0.35]
        f.objects = [d for d in ds if d["cls"] != "person" and d["score"] >= 0.35]
    return fr


def in_cabin(f, zone):
    """Человек в кабине = крупный бокс, нижний край которого доходит до стола.
    Люди в соседнем помещении видны через стекло выше уровня столешницы."""
    h = f.img.shape[0]
    out = []
    top = zone[1] if zone else 0.35 * h
    for p in f.persons:
        x1, y1, x2, y2 = p["box"]
        if (y2 - y1) >= 0.115 * h and y2 >= top:
            out.append(p)
    return out


def feats(name):
    fr = load(name)
    z = objects.desk_zone(fr)
    dt = (fr[1].t - fr[0].t) if len(fr) > 1 else 0.5
    cnt = np.array([len(in_cabin(f, z)) for f in fr])
    mot = np.array([f.motion for f in fr])
    tracks = objects.detect_objects(fr, zone=z)
    big = [t for t in tracks if t.area >= 40]

    absent = cnt == 0
    best = cur = 0
    for v in absent:
        cur = cur + 1 if v else 0
        best = max(best, cur)

    return {
        "dur": round(fr[-1].t, 1),
        "ppl_max": int(cnt.max()) if len(cnt) else 0,
        "ppl_mean": round(float(cnt.mean()), 2),
        "ppl_chg": int((np.diff(cnt) != 0).sum()),
        "absent_s": round(best * dt, 1),
        "absent_pct": round(100 * float(absent.mean())),
        "mot_p90": round(float(np.percentile(mot, 90)), 1),
        "trk": len(tracks),
        "trk_big": len(big),
        "trk_area": round(max([t.area for t in big], default=0)),
    }


def main():
    rows = []
    for name in sorted(LABELS):
        if not Path(f"data/cache/{name}.detr.json").exists():
            continue
        try:
            rows.append((name, bool(LABELS[name]), feats(name)))
        except Exception as exc:                      # noqa: BLE001
            print(f"{name}: {type(exc).__name__}: {exc}", file=sys.stderr)

    keys = list(rows[0][2])
    print(f"{'видео':7s} {'GT':3s} " + " ".join(f"{k:>10s}" for k in keys))
    for n, v, f in sorted(rows, key=lambda r: (not r[1], r[0])):
        print(f"{n:7s} {'V' if v else '.':3s} " + " ".join(f"{f[k]:>10}" for k in keys))

    print("\n=== разделяющая способность (медиана V против медианы чистых) ===")
    for k in keys:
        a = [f[k] for _, v, f in rows if v]
        b = [f[k] for _, v, f in rows if not v]
        ma, mb = np.median(a), np.median(b)
        # AUC через долю пар, где V > чистого
        pairs = [(x > y) + 0.5 * (x == y) for x in a for y in b]
        print(f"{k:12s} V={ma:8.1f}  clean={mb:8.1f}  AUC={np.mean(pairs):.2f}")


if __name__ == "__main__":
    main()
