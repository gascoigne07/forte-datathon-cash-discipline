"""Корпус «как в сервисе» по всем кадрам сетки 2 fps — для перебора бюджета кадров.

Каждый ролик проходит ровно те стадии, что и в `pipeline.analyze`: выборка,
детектор, фон, зона столешницы (`resolve_desk_zone`), треки предметов, кроп
столешницы и описатель. Описываются ВСЕ кадры сетки, поэтому любой бюджет
`KF_CAP` дальше оценивается без моделей: `tools/eval_corpus.py --cap N` выбирает
кадры функцией сервиса `pipeline.select_keyframes` и считает факты функцией
сервиса `pipeline.frame_facts`.

Прежний корпус (`tools/captions.py`) выбирал кадры своей сеткой «1 с, не больше
N», а не ключевыми кадрами сервиса, — то есть измерял конфигурацию, которой в
сервисе нет. Здесь харнесс и сервис совпадают по построению.

    PYTHONPATH=service .venv/bin/python tools/gpu_corpus.py [--batch 16] [имена]
"""
import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "service"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from app import caption, objects, perception  # noqa: E402
from app.pipeline import DESK_CROP_PX, SAMPLE_FPS  # noqa: E402

VID = "/Users/miras/Downloads/open/videos"
OUT = Path("data/corpus")


def build(name: str, batch: int) -> dict:
    t0 = time.time()
    frames = perception.sample(f"{VID}/{name}.mov", fps=SAMPLE_FPS)
    perception.detect(frames)
    bg = objects.background(frames)
    zone, zone_source = objects.resolve_desk_zone(frames, bg=bg)
    tracks = objects.detect_objects(frames, zone=zone)
    t_prep = time.time() - t0

    t1 = time.time()
    caps: list[str] = []
    for s in range(0, len(frames), batch):
        crops = [objects.crop_object(f.img, zone, target_px=DESK_CROP_PX, pad_rel=0.05)
                 for f in frames[s:s + batch]]
        caps += caption.caption_batch(crops)
    t_cap = time.time() - t1
    h, w = frames[0].img.shape[:2]
    return {
        "video": name, "h": h, "w": w, "model": caption.MODEL_ID,
        "zone": [int(v) for v in zone], "zone_source": zone_source,
        "tracks": [{"t_start": t.t_start, "t_end": t.t_end, "box": [int(v) for v in t.box],
                    "area": float(t.area), "frames": t.frames, "near_person": bool(t.near_person)}
                   for t in tracks],
        "frames": [{"t": round(f.t, 3), "motion": round(f.motion, 3),
                    "persons": f.persons, "objects": f.objects, "caption": c}
                   for f, c in zip(frames, caps)],
        "timing": {"prep_sec": round(t_prep, 2), "caption_sec": round(t_cap, 2),
                   "sec_per_frame": round(t_cap / max(1, len(frames)), 3), "batch": batch},
    }


def main():
    from evaluate import LABELS
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    caption.warmup()
    for name in a.names or sorted(LABELS):
        dst = OUT / f"{name}.json"
        if dst.exists() or not Path(f"{VID}/{name}.mov").exists():
            continue
        d = build(name, a.batch)
        dst.write_text(json.dumps(d, ensure_ascii=False))
        tm = d["timing"]
        print(f"{name}: {len(d['frames'])} кадров, зона {d['zone_source']}, "
              f"подготовка {tm['prep_sec']:.1f} с, описание {tm['caption_sec']:.1f} с "
              f"({tm['sec_per_frame']:.3f} с/кадр)", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
