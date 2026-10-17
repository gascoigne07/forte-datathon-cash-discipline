"""Проверка разметки сцены на незнакомом видео — без моделей описания.

Отвечает на один вопрос: на этом ролике зона столешницы ВЫВЕДЕНА из сцены или
сервис скатился в константу «нижние две трети кадра»? Это главный риск закрытого
теста: ракурс там другой, а все правила работают внутри зоны.

Запуск (видео может быть любым, не только из выборки):
    .venv/bin/python tools/smoke_new_angle.py data/aug/*.mov
    .venv/bin/python tools/smoke_new_angle.py ~/Downloads/new_angle.mov

Модель описания не грузится: детектор и вычитание фона стоят десятки миллисекунд
на кадр, поэтому проверка нового ролика занимает секунды, а не минуты.
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "service"))
from app import objects, perception  # noqa: E402

SAMPLE_FPS = 2.0
# Доля кадра, ниже которой зона подозрительно мелкая: внутри неё правила
# предметов работать не смогут.
MIN_AREA_SHARE = 0.05


def check(path: str) -> tuple[str, str]:
    frames = perception.sample(path, fps=SAMPLE_FPS)
    if not frames:
        return "ОШИБКА", "видео не декодируется"
    perception.detect(frames)
    h, w = frames[0].img.shape[:2]
    zone, src = objects.resolve_desk_zone(frames)
    zw, zh = zone[2] - zone[0], zone[3] - zone[1]
    share = zw * zh / (w * h)
    tracks = objects.detect_objects(frames, zone=zone)
    big = [t for t in tracks if t.area >= 60]
    ppl = max(len(perception.persons_in_room(f, desk_top=zone[1])) for f in frames)

    note = (f"зона {src:8s} {zw}x{zh} ({share:.0%} кадра), кадр {w}x{h}, "
            f"кадров {len(frames)}, людей макс. {ppl}, треков {len(tracks)} "
            f"(крупных {len(big)})")
    if src == "fallback":
        return "СКАТИЛСЯ В КОНСТАНТУ", note
    if share < MIN_AREA_SHARE:
        return "ЗОНА СЛИШКОМ МЕЛКАЯ", note
    return "ok", note


def main():
    paths = sys.argv[1:]
    if not paths:
        print(__doc__)
        return 1
    verdicts = Counter()
    for p in paths:
        v, note = check(p)
        verdicts[v] += 1
        print(f"{Path(p).stem:26s} {v:22s} {note}")
    print("\nитог: " + ", ".join(f"{k} — {n}" for k, n in verdicts.most_common()))
    return 0 if set(verdicts) <= {"ok"} else 1


if __name__ == "__main__":
    sys.exit(main())
