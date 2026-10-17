"""Синтетические нарушения: вклейка предмета в ЧИСТОЕ видео с известной секундой.

Зачем. Устойчивость к искажениям я уже мерил, но это проверка «не сломается ли
на том же событии». Здесь другое: берём чистый ролик, в котором правила заведомо
ничего не находят, и кладём на столешницу стакан или сумку начиная с заданной
секунды. Получается нарушение с ТОЧНО известным таймкодом на фоне, под который
ничего не настраивалось, — это измеряет чувствительность, а не живучесть.

Источники предметов — реальные вырезки из датасета (data/synth/obj), поэтому
внешний вид настоящий: та же камера, то же освещение, тот же масштаб.
"""
import argparse, json, random, sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "service"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from app import objects, perception  # noqa: E402

VID = "/Users/miras/Downloads/open/videos"
OBJ = Path("data/synth/obj")
RNG = random.Random(4242)


def load_frames(name):
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


def composite(frame, patch, pos, jitter=1):
    """Вклейка с подгонкой яркости под фон и мягкими краями."""
    x, y = pos
    x += RNG.randint(-jitter, jitter); y += RNG.randint(-jitter, jitter)
    h, w = patch.shape[:2]
    H, W = frame.shape[:2]
    if x < 0 or y < 0 or x + w > W or y + h > H:
        return frame
    out = frame.copy()
    roi = out[y:y + h, x:x + w].astype(np.float32)
    p = patch.astype(np.float32)
    gain = (roi.mean() + 1e-3) / (p.mean() + 1e-3)
    p = np.clip(p * np.clip(gain, 0.75, 1.3), 0, 255)
    # Эллиптическая маска с растушёвкой вместо прямоугольной: прямоугольный шов
    # сам по себе заметнее предмета, и сработавший на него детектор ничего не доказывает.
    mask = np.zeros((h, w), np.float32)
    cv2.ellipse(mask, (w // 2, h // 2), (max(1, w // 2 - 1), max(1, h // 2 - 1)),
                0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (7, 7), 2.0)[..., None]
    out[y:y + h, x:x + w] = (p * mask + roi * (1 - mask)).astype(np.uint8)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", nargs="+", default=["Х-43", "Х-29", "Х-21", "Х-3"])
    ap.add_argument("--objects", nargs="+", default=["стакан_Х-46", "сумка_Х-40", "банка_Х-24"])
    ap.add_argument("--out", default="data/planted")
    a = ap.parse_args()
    out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    patches = {}
    for key in a.objects:
        p = OBJ / f"{key}.png"
        if p.exists():
            patches[key] = cv2.imread(str(p))
    if not patches:
        print("нет вырезок предметов в data/synth/obj", file=sys.stderr); return

    for ci, name in enumerate(a.clean):
        src = Path(f"{VID}/{name}.mov")
        if not src.exists():
            continue
        fr = load_frames(name)
        zone = objects.desk_zone(fr)
        if zone is None:
            print(f"{name}: зона стола не найдена", file=sys.stderr); continue
        zx1, zy1, zx2, zy2 = zone

        key = list(patches)[ci % len(patches)]
        patch = patches[key]
        ph, pw = patch.shape[:2]
        # кладём в правую половину столешницы, подальше от счётной машины
        px = int(zx1 + (zx2 - zx1) * 0.55)
        py = int(zy1 + (zy2 - zy1) * 0.45)
        px = min(px, zx2 - pw - 2); py = min(py, zy2 - ph - 2)

        cap = cv2.VideoCapture(str(src))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        plant_sec = round((n / fps) * 0.35, 1)        # предмет появляется на 35 % длительности
        dst = out_dir / f"{name}__planted_{key.split('_')[0]}.mov"
        vw = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        i = 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            if i / fps >= plant_sec:
                img = composite(img, patch, (px, py))
            vw.write(img); i += 1
        cap.release(); vw.release()
        manifest.append({"video": dst.name, "base": name, "object": key.split("_")[0],
                         "plant_sec": plant_sec, "pos": [px, py]})
        print(f"{dst.name}: предмет «{key.split('_')[0]}» с {plant_sec} с, позиция ({px},{py})",
              file=sys.stderr)

    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    print(f"\nманифест: {out_dir}/manifest.json", file=sys.stderr)


if __name__ == "__main__":
    main()
