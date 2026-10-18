"""П2: признак screen_on — яркость экрана ноутбука в рамке класса laptop (RT-DETR).

Рамка ноутбука — медиана детекций laptop за ролик (техника на столе статична).
На каждом кадре сетки 2 fps: доля пикселей рамки с V > 200 и доля пикселей ярче
90-го перцентиля яркости всего кадра. Сравнивается на кадрах, где работника у
столешницы нет (подпись не упоминает людей), и где он есть.

    PYTHONPATH=service:tools .venv/bin/python tools/screen_probe.py
"""
import json, sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "service"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from app import caption, perception  # noqa: E402
from app.pipeline import SAMPLE_FPS  # noqa: E402
from eval_corpus import CORPUS, names  # noqa: E402
from evaluate import LABELS  # noqa: E402

VID = "/Users/miras/Downloads/open/videos"
OUT = Path("data/screen")


def laptop_box(frames_json):
    boxes = [o["box"] for f in frames_json for o in f["objects"]
             if o["cls"] == "laptop" and o["score"] >= 0.4]
    if len(boxes) < 0.2 * len(frames_json):
        return None
    return [int(v) for v in np.median(np.array(boxes), axis=0)]


def screen_stats(img, box):
    x1, y1, x2, y2 = box
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    roi = v[y1:y2, x1:x2]
    if roi.size == 0:
        return None, None
    p90 = np.percentile(v, 90)
    return float((roi > 200).mean()), float((roi > p90).mean())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for n in names():
        d = json.loads(Path(f"{CORPUS}/{n}.json").read_text())
        box = laptop_box(d["frames"])
        rows = []
        if box:
            frames = perception.sample(f"{VID}/{n}.mov", fps=SAMPLE_FPS)
            for f, fj in zip(frames, d["frames"]):
                a, b = screen_stats(f.img, box)
                rows.append({"t": fj["t"], "v200": a, "p90": b,
                             "absent": caption.facts_from_caption(fj["caption"])["no_person_in_caption"]})
        (OUT / f"{n}.json").write_text(json.dumps({"box": box, "rows": rows}))
        ab = [r for r in rows if r["absent"]]; pr = [r for r in rows if not r["absent"]]
        med = lambda xs, k: np.median([x[k] for x in xs]) if xs else float("nan")  # noqa: E731
        print(f"{n:6s} {'V' if LABELS[n] else '.'} рамка={box} нет-работника {len(ab):3d} к: "
              f"v200={med(ab,'v200'):.3f} p90={med(ab,'p90'):.3f} | есть {len(pr):3d} к: "
              f"v200={med(pr,'v200'):.3f} p90={med(pr,'p90'):.3f}", flush=True)


if __name__ == "__main__":
    main()
