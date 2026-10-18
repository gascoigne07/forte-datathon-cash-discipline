"""П5: вопрос второго описателя «открыт ли сейф/шкаф» по кропу области за столешницей
на всех кадрах сетки 2 fps корпуса. Результат — data/vlm_safe/<имя>.json.

    PYTHONPATH=service:tools .venv/bin/python tools/safe_probe.py [имена] [--limit N] [--save-crops]
"""
import json, sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "service"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from app import objects, perception, vlm  # noqa: E402
from app.pipeline import SAMPLE_FPS  # noqa: E402
from eval_corpus import CORPUS, names  # noqa: E402

VID = "/Users/miras/Downloads/open/videos"
OUT = Path("data/vlm_safe")


def main():
    args = sys.argv[1:]
    limit = 0
    if "--limit" in args:
        i = args.index("--limit"); limit = int(args[i + 1]); del args[i:i + 2]
    save = "--save-crops" in args
    args = [a for a in args if a != "--save-crops"]
    OUT.mkdir(parents=True, exist_ok=True)
    vlm.warmup()
    for n in args or names():
        dst = OUT / f"{n}.json"
        if dst.exists() and not limit:
            continue
        d = json.loads(Path(f"{CORPUS}/{n}.json").read_text())
        zone = tuple(d["zone"])
        frames = perception.sample(f"{VID}/{n}.mov", fps=SAMPLE_FPS)
        idx = list(range(len(frames)))[: limit or None]
        rows = {}
        for s in range(0, len(idx), vlm.BATCH):
            chunk = idx[s:s + vlm.BATCH]
            crops = [objects.safe_crop(frames[i].img, zone) for i in chunk]
            if save and s == 0:
                cv2.imwrite(f"data/vlm_safe/{n}_crop.jpg", crops[0])
            for i, raw in zip(chunk, vlm.ask(crops, max_new_tokens=4, prompt=vlm.SAFE_PROMPT)):
                rows[str(i)] = raw.strip().lower()
        if not limit:
            dst.write_text(json.dumps(rows, ensure_ascii=False))
        seq = "".join({"open": "O", "closed": "c", "none": "-"}.get(rows[k].split()[0].strip(".,") if rows[k] else "", "?")
                      for k in sorted(rows, key=int))
        print(f"{n}: {seq}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
