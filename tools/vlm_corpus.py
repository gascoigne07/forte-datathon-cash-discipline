"""Второй описатель: Qwen3-VL-8B-Instruct (Apache-2.0) отвечает на короткие вопросы
по ПОЛНОМУ кадру — про рабочее место, дверь и число людей.

Florence-2 описывает кроп столешницы и за 906 подписей ни разу не написал «дверь
открыта», а половина пропусков визуально про уход кассира и посторонних. Здесь
проверяется, видит ли это модель покрупнее, если спросить прямо.

Промпт, вызов модели и разбор ответа — из `service/app/vlm.py`, тем же кодом, что
и в сервисе. Факты: teller_at_desk_vlm, person_at_door, people_in_booth_vlm, chair_empty.

    PYTHONPATH=service .venv/bin/python tools/vlm_corpus.py [--batch 8] [--limit N] [имена]
"""
import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "service"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from app import perception, vlm  # noqa: E402
from app.pipeline import SAMPLE_FPS  # noqa: E402

VID = "/Users/miras/Downloads/open/videos"
OUT = Path("data/vlm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="только первые N кадров (замер)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    vlm.warmup()
    from evaluate import LABELS
    for name in a.names or sorted(LABELS):
        dst = OUT / f"{name}.json"
        if dst.exists() and not a.limit:
            continue
        if not Path(f"{VID}/{name}.mov").exists():
            continue
        frames = perception.sample(f"{VID}/{name}.mov", fps=SAMPLE_FPS)
        if a.limit:
            frames = frames[:a.limit]
        t0 = time.time(); rows = []
        for s in range(0, len(frames), a.batch):
            chunk = frames[s:s + a.batch]
            for f, raw in zip(chunk, vlm.ask([f.img for f in chunk])):
                rows.append({"t": round(f.t, 3), "raw": raw, **vlm.parse(raw)})
        d = time.time() - t0
        if not a.limit:
            dst.write_text(json.dumps({"video": name, "model": vlm.MODEL_ID, "frames": rows,
                                       "sec_per_frame": round(d / len(rows), 3)},
                                      ensure_ascii=False))
        print(f"{name}: {len(rows)} кадров, {d / len(rows):.3f} с/кадр", file=sys.stderr, flush=True)
        if a.limit:
            for r in rows:
                print(r, file=sys.stderr)


if __name__ == "__main__":
    main()
