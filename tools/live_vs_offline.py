"""Сверка живого сервиса с офлайн-замером по корпусу (урок ночи: харнесс и сервис
обязаны совпадать, а проверить это может только живой прогон).

    PYTHONPATH=service:tools .venv/bin/python tools/live_vs_offline.py [data/runs/live]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "service"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from app import rules  # noqa: E402
from eval_caps import DUPLICATES  # noqa: E402
from eval_corpus import names, series_at, vlm_extra  # noqa: E402
from evaluate import LABELS  # noqa: E402

CAP = 90


def main():
    live_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "data/runs/live")
    tp = fp = fn = tn = same = 0
    diff = []
    for n in names():
        p = live_dir / f"{n}.run.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        s, tr, _, _ = series_at(n, CAP, vlm_extra)
        off = rules.detect_events(s, tracks=tr)
        live_v = d["verdict"] == "violation"
        truth = bool(LABELS[n])
        tp += live_v and truth; fp += live_v and not truth
        fn += (not live_v) and truth; tn += (not live_v) and not truth
        lk = sorted(f"{x['type'][:20]}@{x['timecode_sec']:.1f}" for x in d["violations"])
        ok = sorted(f"{rules.TYPES[e.key][0][:20]}@{e.t:.1f}" for e in off)
        if live_v == bool(off):
            same += 1
        else:
            diff.append(n)
        if lk != ok:
            print(f"{n:6s} живой: {lk or '—'}\n       офлайн: {ok or '—'}")
    n = tp + fp + fn + tn
    print(f"\nживой сервис: вердикт {tp + tn}/{n}, найдено {tp}/{tp + fn}, ложных {fp}/{fp + tn}")
    print(f"вердикт живой = офлайн: {same}/{n}; расходятся: {', '.join(diff) or '—'}")


if __name__ == "__main__":
    main()
