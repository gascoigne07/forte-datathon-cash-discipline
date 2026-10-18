"""П3: выбор якоря таймкода по кэшу фактов (вердикты не меняются).

Для каждого найденного события сравниваются якоря:
  уход (unattended_desk, left_via_door, left_workplace):
    current — текущий; door — начало первой серии «дверь + пустой стул»;
    empty-3 — первый кадр, где помещение пусто, минус 3 с;
    leave   — первый кадр после последнего, где работник сидел за столешницей,
              перед окном события (начало ухода);
  предметные (food_drink, personal_bag, sealed_bag, outsider_*):
    current; start — начало окна; mid — середина окна.

    PYTHONPATH=service:tools .venv/bin/python tools/anchor_probe.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "service"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from app import rules  # noqa: E402
from eval_corpus import names, series_at, vlm_extra  # noqa: E402
from evaluate import LABELS  # noqa: E402

LEAVE = {"unattended_desk", "left_via_door", "left_workplace"}


def seated(f):
    return (f.get("at_desk", 0) > 0 and not f.get("no_person_in_caption")
            and not f.get("chair_empty"))


def leave_anchors(series, e):
    out = {"current": e.t}
    door = rules._runs(series, lambda f: bool(f.get("person_at_door")) and bool(f.get("chair_empty")))
    door = [r for r in door if r[1] - r[0] + 1 >= 2]
    out["door"] = series[door[0][0]][0] if door else e.t
    emp = [t for t, f in series if rules._is_empty(f)]
    out["empty-3"] = max(0.0, emp[0] - 3) if emp else e.t
    before = [k for k, (t, f) in enumerate(series) if t <= e.window[0] and seated(f)]
    if before and before[-1] + 1 < len(series):
        out["leave"] = series[before[-1] + 1][0]
    else:
        out["leave"] = e.t
    return out


def obj_anchors(e):
    return {"current": e.t, "start": e.window[0], "mid": (e.window[0] + e.window[1]) / 2}


def main():
    per = {"уход": {}, "предметные": {}}
    for n in names():
        gold = LABELS[n]
        if not gold:
            continue
        series, tracks, _, _ = series_at(n, 90, vlm_extra)
        ev = rules.detect_events(series, tracks=tracks)
        if not ev:
            continue
        for grp, sel, fn in (("уход", lambda e: e.key in LEAVE, lambda e: leave_anchors(series, e)),
                             ("предметные", lambda e: e.key not in LEAVE, obj_anchors)):
            es = [e for e in ev if sel(e)]
            if not es:
                continue
            best = {}
            for e in es:
                for k, t in fn(e).items():
                    d = min(abs(t - g) for g in gold)
                    best[k] = min(best.get(k, 1e9), d)
            per[grp][n] = best
    for grp, rows in per.items():
        keys = sorted({k for r in rows.values() for k in r})
        print(f"\n== {grp}: {len(rows)} роликов")
        print("ролик   " + " ".join(f"{k:>8s}" for k in keys))
        for n, r in rows.items():
            print(f"{n:7s} " + " ".join(f"{r[k]:8.1f}" for k in keys))
        for k in keys:
            xs = np.array([r[k] for r in rows.values()])
            print(f"  {k:8s}: медиана {np.median(xs):.1f} с, ≤3 с у {100 * (xs <= 3).mean():.0f} %")


if __name__ == "__main__":
    main()
