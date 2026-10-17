"""Искажения ракурса для проверки устойчивости (критерий 5 ТЗ).

Финальное видео жюри снято под другим углом. Проверяем это честно: прогоняем
пайплайн по геометрически искажённым копиям обучающих роликов и смотрим,
сохраняется ли вердикт и таймкод.
"""
import argparse, subprocess, sys
from pathlib import Path

# имя -> фильтр ffmpeg. Значения подобраны так, чтобы имитировать перестановку
# камеры, а не превратить кадр в неузнаваемый.
# Набор искажений по таксономии бенчмарка устойчивости пространственно-временных
# моделей (NeurIPS 2021 Datasets & Benchmarks): шум, размытие, цифровые артефакты,
# временные и движение камеры. Плюс зеркало — самая сильная смена ракурса:
# камера как будто стоит с другой стороны, и левое меняется с правым.
VARIANTS = {
    # движение камеры
    "mirror":  "hflip",
    "rot+7":   "rotate=7*PI/180:fillcolor=black@0:bilinear=1",
    "rot-7":   "rotate=-7*PI/180:fillcolor=black@0:bilinear=1",
    "persp_l": "perspective=x0=40:y0=0:x1=W:y1=0:x2=0:y2=H:x3=W-40:y3=H:interpolation=linear",
    "persp_r": "perspective=x0=0:y0=0:x1=W-40:y1=0:x2=40:y2=H:x3=W:y3=H:interpolation=linear",
    "shift":   "crop=iw-40:ih-40:40:0,pad=iw+40:ih+40:0:40:black",
    "zoom085": "scale=iw*0.85:ih*0.85,pad=iw/0.85:ih/0.85:(ow-iw)/2:(oh-ih)/2:black",
    # освещение
    "dark":    "eq=brightness=-0.12:contrast=1.05",
    "bright":  "eq=brightness=0.12:saturation=0.9",
    # размытие
    "blur_motion":  "tblend=all_mode=average,boxblur=2:1",
    "blur_defocus": "gblur=sigma=1.6",
    # шум
    "noise":   "noise=alls=18:allf=t+u",
    # временные
    "fps15":   "fps=15",
    "fps10":   "fps=10",
    # развёртка по силе: ищем точку излома, а не бинарный ответ
    "blur_s08":  "gblur=sigma=0.8",
    "blur_s12":  "gblur=sigma=1.2",
    "noise_s30": "noise=alls=30:allf=t+u",
    "gray":      "format=gray,format=yuv420p",
    "res060":    "scale=trunc(iw*0.6/2)*2:trunc(ih*0.6/2)*2,scale=trunc(iw/0.6/2)*2:trunc(ih/0.6/2)*2:flags=bilinear",
    "rot+15":    "rotate=15*PI/180:fillcolor=black@0:bilinear=1",
    # составное: настоящая «другая камера» это сочетание, а не один эффект
    "mirror_rot": "hflip,rotate=8*PI/180:fillcolor=black@0:bilinear=1",
    "persp_dark": "perspective=x0=40:y0=0:x1=W:y1=0:x2=0:y2=H:x3=W-40:y3=H:interpolation=linear,eq=brightness=-0.1",
}

# Цифровые артефакты задаются не фильтром, а параметрами кодирования.
ENCODE = {
    "compress":    ["-c:v", "libx264", "-crf", "40", "-preset", "veryfast"],
    "compress_28": ["-c:v", "libx264", "-crf", "28", "-preset", "veryfast"],
    "compress_34": ["-c:v", "libx264", "-crf", "34", "-preset", "veryfast"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--out", default="data/aug")
    ap.add_argument("--only", nargs="*", help="подмножество вариантов")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    names = a.only or (list(VARIANTS) + list(ENCODE))
    for v in a.videos:
        stem = Path(v).stem
        for name in names:
            dst = out / f"{stem}__{name}.mov"
            if dst.exists():
                continue
            if name in ENCODE:
                cmd = ["ffmpeg", "-v", "error", "-y", "-i", v, *ENCODE[name], "-an", str(dst)]
            else:
                cmd = ["ffmpeg", "-v", "error", "-y", "-i", v, "-vf", VARIANTS[name],
                       "-c:v", "libx264", "-crf", "20", "-an", str(dst)]
            r = subprocess.run(cmd, capture_output=True, text=True)
            status = "ok" if r.returncode == 0 else f"FAIL {r.stderr[:120]}"
            print(f"{dst.name}: {status}", file=sys.stderr)


if __name__ == "__main__":
    main()
