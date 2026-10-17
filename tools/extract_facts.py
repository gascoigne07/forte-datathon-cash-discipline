"""Прогон VLM по ключевым кадрам датасета -> кэш фактов на диск.

Дорогой этап отделён от дешёвого: факты извлекаются ОДИН раз, а правила
потом крутятся по кэшу мгновенно. Иначе каждая правка порога стоила бы час.
"""
import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "service"))
import vlm_qwen25_measured as vlm
from app import objects, perception  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--out", default="data/facts")
    ap.add_argument("--step", type=float, default=2.0, help="шаг сетки ключевых кадров, с")
    ap.add_argument("--cap", type=int, default=30, help="максимум кадров на видео")
    ap.add_argument("--upscale", type=float, default=1.0)
    a = ap.parse_args()

    Path(a.out).mkdir(parents=True, exist_ok=True)
    vlm.warmup()
    print(f"модель загружена: {vlm.MODEL_ID}", file=sys.stderr)

    for v in a.videos:
        name = Path(v).stem
        dst = Path(a.out) / f"{name}.facts.json"
        if dst.exists():
            print(f"skip {name}", file=sys.stderr); continue
        t0 = time.time()
        frames = perception.sample(v, fps=2.0)
        if not frames:
            print(f"!! {name}: не декодируется", file=sys.stderr); continue
        cache = Path("data/cache") / f"{name}.detr.json"
        if cache.exists():                       # переиспользуем прогон детектора
            det = json.loads(cache.read_text())["frames"]
            byt = {round(d["t"], 2): d["det"] for d in det}
            for fr in frames:
                ds = byt.get(round(fr.t, 2), [])
                fr.persons = [d for d in ds if d["cls"] == "person" and d["score"] >= 0.35]
                fr.objects = [d for d in ds if d["cls"] != "person" and d["score"] >= 0.35]
        else:
            perception.detect(frames)
        box = perception.roi(frames)
        idx = perception.keyframes(frames, budget=a.cap, step_sec=a.step)
        keep = set(idx)                          # остальные кадры в памяти не держим
        for j, fr in enumerate(frames):
            if j not in keep:
                fr.img = None
        series = []
        for i in idx:
            try:
                f = vlm.facts(perception.crop_scaled(frames[i].img, box, target_px=430_000),
                              upscale=a.upscale)
            except Exception as exc:                        # noqa: BLE001
                print(f"   {name}@{frames[i].t:.1f}: {type(exc).__name__}: {exc}", file=sys.stderr)
                continue
            f["_t"] = round(frames[i].t, 2)
            f["_persons_detr"] = len(perception.persons_in_room(frames[i]))
            frames[i].img = None                 # кадр больше не нужен
            series.append(f)
        dst.write_text(json.dumps({"video": name, "upscale": a.upscale,
                                   "model": vlm.MODEL_ID, "series": series},
                                  ensure_ascii=False, indent=1))
        dur = time.time() - t0
        print(f"{name}: {len(series)} кадров за {dur:.0f}с ({dur/max(1,len(series)):.1f} с/кадр)",
              file=sys.stderr)


if __name__ == "__main__":
    main()
