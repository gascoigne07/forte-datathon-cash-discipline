"""Искажения ракурса для проверки устойчивости (критерий 5 ТЗ).

Финальное видео жюри снято под другим углом. Проверяем это честно: прогоняем
пайплайн по геометрически искажённым копиям обучающих роликов и смотрим,
сохраняется ли вердикт и таймкод.
"""
import argparse, subprocess, sys
from pathlib import Path

# имя -> фильтр ffmpeg. Значения подобраны так, чтобы имитировать перестановку
# камеры, а не превратить кадр в неузнаваемый.
VARIANTS = {
    "rot+7":   "rotate=7*PI/180:fillcolor=black@0:bilinear=1",
    "rot-7":   "rotate=-7*PI/180:fillcolor=black@0:bilinear=1",
    "persp_l": "perspective=x0=40:y0=0:x1=W:y1=0:x2=0:y2=H:x3=W-40:y3=H:interpolation=linear",
    "persp_r": "perspective=x0=0:y0=0:x1=W-40:y1=0:x2=40:y2=H:x3=W:y3=H:interpolation=linear",
    "zoom085": "scale=iw*0.85:ih*0.85,pad=iw/0.85:ih/0.85:(ow-iw)/2:(oh-ih)/2:black",
    "dark":    "eq=brightness=-0.12:contrast=1.05",
    "bright":  "eq=brightness=0.12:saturation=0.9",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--out", default="data/aug")
    ap.add_argument("--only", nargs="*", help="подмножество вариантов")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    names = a.only or list(VARIANTS)
    for v in a.videos:
        stem = Path(v).stem
        for name in names:
            dst = out / f"{stem}__{name}.mov"
            if dst.exists():
                continue
            r = subprocess.run(
                ["ffmpeg", "-v", "error", "-y", "-i", v, "-vf", VARIANTS[name],
                 "-c:v", "libx264", "-crf", "20", "-an", str(dst)],
                capture_output=True, text=True)
            status = "ok" if r.returncode == 0 else f"FAIL {r.stderr[:120]}"
            print(f"{dst.name}: {status}", file=sys.stderr)


if __name__ == "__main__":
    main()
