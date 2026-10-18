"""Вопрос второго описателя «держит ли человек напиток / пьёт» по кадрам корпуса,
где Florence-2 назвал напиток или еду. Результат — data/vlm_drink/<имя>.json.

    PYTHONPATH=service:tools .venv/bin/python tools/vlm_drink.py
"""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "service"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from app import caption, perception, vlm  # noqa: E402
from app.pipeline import SAMPLE_FPS  # noqa: E402
from eval_corpus import CORPUS, names  # noqa: E402

VID = "/Users/miras/Downloads/open/videos"
OUT = Path("data/vlm_drink")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    vlm.warmup()
    for n in names():
        d = json.loads(Path(f"{CORPUS}/{n}.json").read_text())
        want = [i for i, f in enumerate(d["frames"])
                if caption.facts_from_caption(f["caption"])["drink_or_food"]]
        rows = {}
        if want:
            frames = perception.sample(f"{VID}/{n}.mov", fps=SAMPLE_FPS)
            for s in range(0, len(want), vlm.BATCH):
                chunk = want[s:s + vlm.BATCH]
                for i, raw in zip(chunk, vlm.ask([frames[i].img for i in chunk],
                                                 max_new_tokens=4, prompt=vlm.DRINK_PROMPT)):
                    rows[str(i)] = raw
        (OUT / f"{n}.json").write_text(json.dumps(rows, ensure_ascii=False))
        if rows:
            print(n, rows, file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
