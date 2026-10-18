"""П8: длительность отсутствия работника для событий ухода.

Уход — таймкод события (якорь «начало ухода»); возвращение — первый ключевой кадр
после окна события, где работник снова сидит за столешницей (`rules._seated`), иначе
конец записи.

    PYTHONPATH=service:tools .venv/bin/python tools/absence_probe.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "service"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from app import rules  # noqa: E402
from eval_corpus import names, series_at, vlm_extra  # noqa: E402
from evaluate import LABELS  # noqa: E402


def absence(series, e):
    after = [(t, f) for t, f in series if t > e.window[1]]
    back = next((t for t, f in after if rules._seated(f)), None)
    end = series[-1][0]
    return (back if back is not None else end), back is None


def main(keys=rules.LEAVE_KEYS):
    for n in names():
        series, tracks, _, _ = series_at(n, 90, vlm_extra)
        for e in rules.detect_events(series, tracks=tracks):
            if e.key not in keys:
                continue
            t_back, to_end = absence(series, e)
            print(f"{n:6s} {'V' if LABELS[n] else '.'} {e.key:16s} уход {e.t:5.1f} окно {e.window[0]:5.1f}–{e.window[1]:5.1f} "
                  f"возврат {'конец записи' if to_end else f'{t_back:5.1f}'} ({t_back:5.1f}) "
                  f"отсутствие {t_back - e.t:5.1f} с")


if __name__ == "__main__":
    main()
