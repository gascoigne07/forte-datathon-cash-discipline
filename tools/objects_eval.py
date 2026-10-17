"""Проверка канала «новые предметы» против разметки: ловит ли он нужную секунду.

Без VLM и без правил — чистое измерение локализации.
"""
import sys, glob
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "service"))
from app import perception, objects  # noqa: E402
from evaluate import LABELS  # noqa: E402

VID = "/Users/miras/Downloads/open/videos"


def main():
    which = sys.argv[1:] or sorted(LABELS)
    hits = miss = 0
    fp_clean = 0
    for name in which:
        gold = LABELS.get(name)
        if gold is None:
            continue
        p = f"{VID}/{name}.mov"
        if not Path(p).exists():
            continue
        frames = perception.sample(p, fps=2.0)
        cache = Path(f"data/cache/{name}.detr.json")
        if cache.exists():            # переиспользуем прогон детектора
            import json
            det = json.loads(cache.read_text())["frames"]
            byt = {round(d["t"], 2): d["det"] for d in det}
            for f in frames:
                ds = byt.get(round(f.t, 2), [])
                f.persons = [d for d in ds if d["cls"] == "person" and d["score"] >= 0.35]
                f.objects = [d for d in ds if d["cls"] != "person" and d["score"] >= 0.35]
        else:
            perception.detect(frames)
        z = objects.desk_zone(frames)
        tr = objects.detect_objects(frames, zone=z)
        best = ""
        if gold and tr:
            d = min(abs(t.t_start - g) for t in tr for g in gold)
            best = f"Δ={d:.1f}с"
            hits += d <= 3
            miss += d > 3
        elif gold:
            miss += 1; best = "предметов не найдено"
        elif tr:
            fp_clean += 1
        desc = "; ".join(f"{t.t_start:.0f}-{t.t_end:.0f}с S={t.area:.0f}"
                         f"{'~рука' if t.near_person else ''}" for t in tr[:5])
        print(f"{name:6s} {'V' if gold else '.':2s} gold={str(gold):10s} {best:20s} зона={z} треков={len(tr):2d}  {desc}")
    print(f"\nпопаданий ≤3с: {hits}, промахов: {miss}, чистых роликов с треками: {fp_clean}")


if __name__ == "__main__":
    main()
