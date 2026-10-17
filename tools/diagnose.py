"""Диагностика промахов и ложных срабатываний по кэшу подписей.

Для каждого пропущенного нарушения печатает факты вокруг размеченной секунды:
видно, какого признака не хватило. Для ложных — что именно сработало.
Это прямой путь к следующему улучшению: не гадать, а смотреть на конкретный кадр.
"""
import json, sys, glob
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "service"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from app import caption, rules  # noqa: E402
from eval_caps import series_for  # noqa: E402
from evaluate import LABELS  # noqa: E402

FLAGS = ("no_person_in_caption", "cash_seen", "drink_or_food", "personal_bag",
         "sealed_bag", "paper_seen", "track_on_desk")
SHORT = {"no_person_in_caption": "нетЧел", "cash_seen": "деньги", "drink_or_food": "напиток",
         "personal_bag": "сумка", "sealed_bag": "инкас", "paper_seen": "бумага",
         "track_on_desk": "трек"}


def main():
    want = sys.argv[1:] if len(sys.argv) > 1 else None
    missed, fps = [], []
    for p in sorted(glob.glob("data/captions/*.caps.json")):
        name = Path(p).stem.replace(".caps", "")
        if name not in LABELS or (want and name not in want):
            continue
        series, _ = series_for(name)
        if not series:
            continue
        ev = rules.detect_events(series)
        gold = LABELS[name]
        if gold and not ev:
            missed.append((name, gold, series))
        elif not gold and ev:
            fps.append((name, ev, series))

    print(f"=== ПРОПУЩЕНО: {len(missed)} ===\n")
    for name, gold, series in missed:
        print(f"{name}  разметка={gold}")
        for t, f in series:
            if any(abs(t - g) <= 4 for g in gold):
                on = [SHORT[k] for k in FLAGS if f.get(k)]
                mark = "<<<" if any(abs(t - g) <= 1.6 for g in gold) else "   "
                print(f"  {mark} t={t:5.1f} люди={f.get('people_in_cabin', 0)} "
                      f"работник={'да' if f.get('teller_present') else 'НЕТ'} "
                      f"[{' '.join(on) if on else '—'}]")
                print(f"        {f.get('caption', '')[:180]}")
        print()

    print(f"=== ЛОЖНЫЕ СРАБАТЫВАНИЯ: {len(fps)} ===\n")
    for name, ev, series in fps:
        print(f"{name}  события: {', '.join(f'{e.key}@{e.t:.1f}' for e in ev)}")
        for e in ev:
            for t, f in series:
                if abs(t - e.t) < 0.1:
                    on = [SHORT[k] for k in FLAGS if f.get(k)]
                    print(f"  t={t:5.1f} [{' '.join(on)}] работник="
                          f"{'да' if f.get('teller_present') else 'НЕТ'}")
                    print(f"        {f.get('caption', '')[:220]}")
        print()


if __name__ == "__main__":
    main()
