"""Двухстадийная схема: кандидатные треки предметов -> тесный кроп -> называние моделью.

Проверяем гипотезу: мелкий объект не виден не из-за слабости модели, а из-за
масштаба. На кропе столешницы пачка купюр ~40x25 px, на тесном кропе ~200x150.

Контроль двусторонний: те же кандидаты берутся и из роликов с нарушением,
и из чистых. Если модель называет деньги одинаково часто в обоих — признак
бесполезен, сколько бы ни было уверенности.
"""
import json, os, sys, tempfile, time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "service"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from app import objects, perception  # noqa: E402
from evaluate import LABELS  # noqa: E402

VID = "/Users/miras/Downloads/open/videos"
MODEL = os.environ.get("Q3_MODEL", "mlx-community/Qwen3-VL-4B-Instruct-4bit")

PROMPT = ("На фото — увеличенный фрагмент стола банковского кассира (вид сверху).\n"
          "Что лежит или находится в центре кадра? Ответь ОДНОЙ буквой:\n"
          "A) банкноты, деньги\nB) лист бумаги или документ\nC) карта или визитка\n"
          "D) стакан, кружка, банка, бутылка\nE) сумка, пакет\nF) техника "
          "(ноутбук, телефон, счётная машина)\nG) ничего из перечисленного\nОтвет: ")


def load(name):
    fr = perception.sample(f"{VID}/{name}.mov", fps=2.0)
    cache = Path(f"data/cache/{name}.detr.json")
    if cache.exists():
        det = json.loads(cache.read_text())["frames"]
        byt = {round(d["t"], 2): d["det"] for d in det}
        for f in fr:
            ds = byt.get(round(f.t, 2), [])
            f.persons = [d for d in ds if d["cls"] == "person" and d["score"] >= 0.35]
            f.objects = [d for d in ds if d["cls"] != "person" and d["score"] >= 0.35]
    else:
        perception.detect(fr)
    return fr


def main():
    names = sys.argv[1:] or ["X-2", "Х-4", "Х-26", "Х-13", "Х-15",
                             "Х-3", "Х-43", "Х-10", "Х-35", "Х-12"]
    from mlx_vlm import generate, load as mload
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config
    model, proc = mload(MODEL)
    config = load_config(MODEL)
    print(f"{MODEL}\n")

    tally = {}
    for name in names:
        if name not in LABELS:
            continue
        fr = load(name)
        zone = objects.desk_zone(fr)
        tracks = [t for t in objects.detect_objects(fr, zone=zone) if t.area >= 30][:3]
        gold = LABELS[name]
        if not tracks:
            print(f"{name:6s} {'V' if gold else '.'}  кандидатов нет"); continue
        for tr in tracks:
            f = min(fr, key=lambda x: abs(x.t - tr.t_start))
            crop = objects.crop_object(f.img, tr.box, target_px=220_000, pad_rel=1.2)
            p = tempfile.mktemp(suffix=".jpg")
            cv2.imwrite(p, crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            fmt = apply_chat_template(proc, config, PROMPT, num_images=1)
            t = time.time()
            res = generate(model, proc, fmt, [p], max_tokens=8, temperature=0.0, verbose=False)
            ans = getattr(res, "text", str(res)).strip()[:40].replace("\n", " ")
            os.unlink(p)
            letter = next((c for c in ans if c in "ABCDEFG"), "?")
            key = ("V" if gold else ".", letter)
            tally[key] = tally.get(key, 0) + 1
            print(f"{name:6s} {'V' if gold else '.'}  трек {tr.t_start:5.1f}-{tr.t_end:5.1f}с "
                  f"S={tr.area:4.0f} -> {letter}  ({ans[:28]}) [{time.time()-t:.1f}с]")

    print("\n=== сводка: какую букву модель даёт на кандидатах ===")
    letters = sorted({k[1] for k in tally})
    print(f"{'буква':6s} {'нарушения':>10s} {'чистые':>8s}")
    for L in letters:
        print(f"{L:6s} {tally.get(('V', L), 0):10d} {tally.get(('.', L), 0):8d}")


if __name__ == "__main__":
    main()
