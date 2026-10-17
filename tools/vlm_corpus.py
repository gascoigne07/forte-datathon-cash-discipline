"""Второй описатель: Qwen3-VL-8B-Instruct (Apache-2.0) отвечает на короткие вопросы
по ПОЛНОМУ кадру — про рабочее место, дверь и число людей.

Florence-2 описывает кроп столешницы и за 906 подписей ни разу не написал «дверь
открыта», а половина пропусков визуально про уход кассира и посторонних. Здесь
проверяется, видит ли это модель покрупнее, если спросить прямо.

Ответ — четыре строки фиксированного формата, разбираются в факты:
teller_at_desk_vlm, person_at_door, people_in_booth_vlm, chair_empty.

    PYTHONPATH=service .venv/bin/python tools/vlm_corpus.py [--batch 8] [--limit N] [имена]
"""
import argparse, json, re, sys, time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "service"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from app import perception  # noqa: E402
from app.pipeline import SAMPLE_FPS  # noqa: E402

MODEL = "Qwen/Qwen3-VL-8B-Instruct"
VID = "/Users/miras/Downloads/open/videos"
OUT = Path("data/vlm")

PROMPT = (
    "This is a CCTV frame of a bank teller's room. The teller works behind the wooden desk "
    "with the laptop and the cash counting machine. The room has a glass wall and a glass door; "
    "people seen through the glass in the office outside are NOT in the room.\n"
    "Q1. Is the teller sitting or standing at the desk? Answer at_desk or away.\n"
    "Q2. Is anyone standing in the doorway or walking through the door of the room? Answer yes or no.\n"
    "Q3. How many people are inside the room? Answer a number.\n"
    "Q4. Is the teller's chair behind the desk empty? Answer yes or no.\n"
    "Reply with exactly four lines:\nQ1: ...\nQ2: ...\nQ3: ...\nQ4: ..."
)


def parse(text: str) -> dict:
    """Четыре ответа -> факты. Неразобранный ответ = None, а не «нет»."""
    ans = {}
    for k, v in re.findall(r"Q([1-4])\s*[:.)]\s*([^\n]+)", text):
        ans[k] = v.strip().lower()
    def yn(s):
        if s is None:
            return None
        if s.startswith("yes"):
            return True
        if s.startswith("no"):
            return False
        return None
    q1 = ans.get("1")
    at = None if q1 is None else (True if "at_desk" in q1 or "at desk" in q1 or q1.startswith("sit")
                                   else False if "away" in q1 else None)
    m = re.search(r"\d+", ans.get("3", ""))
    # Q1 модель иногда отвечает «standing» вместо одного из двух вариантов: это не
    # «за столом» и не «ушёл», поэтому хранится отдельно, а не приводится насильно.
    pose = None if q1 is None else ("at_desk" if at else "away" if at is False
                                    else "standing" if "stand" in q1 else "other")
    return {"teller_at_desk_vlm": at, "teller_pose_vlm": pose, "person_at_door": yn(ans.get("2")),
            "people_in_booth_vlm": int(m.group()) if m else None,
            "chair_empty": yn(ans.get("4"))}


class Asker:
    def __init__(self):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        self.torch = torch
        self.proc = AutoProcessor.from_pretrained(MODEL)
        self.proc.tokenizer.padding_side = "left"
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()

    def ask(self, imgs_bgr, max_new_tokens: int = 32) -> list[str]:
        from PIL import Image
        msgs = [[{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
                for _ in imgs_bgr]
        texts = [self.proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                 for m in msgs]
        pils = [Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)) for im in imgs_bgr]
        inputs = self.proc(text=texts, images=pils, return_tensors="pt", padding=True).to("cuda")
        with self.torch.inference_mode():
            ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        ids = ids[:, inputs["input_ids"].shape[1]:]
        return [t.strip() for t in self.proc.batch_decode(ids, skip_special_tokens=True)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="только первые N кадров (замер)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    asker = Asker()
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
            for f, raw in zip(chunk, asker.ask([f.img for f in chunk])):
                rows.append({"t": round(f.t, 3), "raw": raw, **parse(raw)})
        d = time.time() - t0
        if not a.limit:
            dst.write_text(json.dumps({"video": name, "model": MODEL, "frames": rows,
                                       "sec_per_frame": round(d / len(rows), 3)},
                                      ensure_ascii=False))
        print(f"{name}: {len(rows)} кадров, {d / len(rows):.3f} с/кадр", file=sys.stderr, flush=True)
        if a.limit:
            for r in rows:
                print(r, file=sys.stderr)


if __name__ == "__main__":
    main()
