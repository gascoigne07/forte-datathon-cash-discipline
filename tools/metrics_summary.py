"""Единственный источник цифр для README, SUBMISSION и PITCH.

Проблема, которую решает: одни и те же метрики переписывались в трёх документах
руками, и после каждой правки правил часть цифр оставалась старой. Теперь таблица
генерируется из выборки и вставляется во все документы одним и тем же блоком между
маркерами `<!-- МЕТРИКИ:начало -->` и `<!-- МЕТРИКИ:конец -->`.

    .venv/bin/python tools/metrics_summary.py            # напечатать таблицу
    .venv/bin/python tools/metrics_summary.py --write     # вставить в документы
    .venv/bin/python tools/metrics_summary.py --check     # сверить, не устарело ли

Считает ровно то, что отдаёт сервис: те же `rules.detect_events(series, tracks)`
и тот же порог площади трека.
"""
import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "service"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from app import rules  # noqa: E402
from eval_caps import DUPLICATES, series_for  # noqa: E402
from evaluate import LABELS  # noqa: E402

BEGIN = "<!-- МЕТРИКИ:начало -->"
END = "<!-- МЕТРИКИ:конец -->"
TARGETS = ["README.md", "SUBMISSION.md", "PITCH.md"]


def measure() -> dict:
    tp = fp = fn = tn = 0
    errs: list[float] = []
    hits: list[str] = []
    fps: list[str] = []
    for p in sorted(glob.glob("data/captions/*.caps.json")):
        name = Path(p).stem.replace(".caps", "")
        if name not in LABELS or name in DUPLICATES:
            continue
        series, tracks = series_for(name)
        if not series:
            continue
        ev = rules.detect_events(series, tracks=tracks)
        gold, pred = LABELS[name], bool(ev)
        if pred and gold:
            tp += 1
            hits.append(name)
            errs.append(min(abs(e.t - g) for e in ev for g in gold))
        elif pred:
            fp += 1
            fps.append(name)
        elif gold:
            fn += 1
        else:
            tn += 1
    errs.sort()
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n": tp + fp + fn + tn,
        "viol": tp + fn, "clean": tn + fp,
        "acc": 100 * (tp + tn) / max(1, tp + fp + fn + tn),
        "recall": 100 * tp / max(1, tp + fn),
        "spec": 100 * tn / max(1, tn + fp),
        "med": errs[len(errs) // 2] if errs else -1,
        "le3": 100 * sum(e <= 3 for e in errs) / max(1, len(errs)),
        "mean": sum(errs) / max(1, len(errs)),
        "hits": hits, "fps": fps,
    }


def robustness() -> dict:
    """Варианты роликов, прогнанные через боевой пайплайн (`data/runs/*__*.run.json`).

    Три группы с РАЗНЫМ ожидаемым вердиктом — их нельзя считать одной цифрой:
      `искажения нарушений` — съёмочные искажения ролика с нарушением, вердикт
          обязан остаться `violation`;
      `искажения чистых`    — то же на чистом ролике, вердикт обязан остаться
          `no_violation`;
      `вклейки`             — предмет врисован в ЧИСТЫЙ ролик, поэтому правильный
          вердикт здесь `violation`, а не `no_violation`. Первая версия этого
          счётчика брала метку исходного ролика и записывала успешные вклейки в
          ложные срабатывания.
    """
    g = {k: [0, 0] for k in ("искажения нарушений", "искажения чистых", "вклейки")}
    for p in sorted(glob.glob("data/runs/*__*.run.json")):
        stem = Path(p).stem.replace(".run", "")
        base, _, variant = stem.partition("__")
        if base not in LABELS:
            continue
        d = json.loads(Path(p).read_text())
        if d.get("error"):
            continue
        planted = variant.startswith("planted")
        want = True if planted else bool(LABELS[base])
        key = ("вклейки" if planted
               else "искажения нарушений" if LABELS[base] else "искажения чистых")
        g[key][0 if (d.get("verdict") == "violation") == want else 1] += 1
    return {k: {"ok": v[0], "total": v[0] + v[1]} for k, v in g.items() if v[0] + v[1]}


def table(m: dict, r: dict) -> str:
    return "\n".join([
        BEGIN,
        "",
        f"Замер: {m['n']} уникальных роликов обучающей выборки "
        f"({m['viol']} с нарушением, {m['clean']} чистых). `Х-47` и `Х-48` — побайтовые "
        "копии `Х-4` и `Х-5`, из счёта исключены. Воспроизводится командой "
        "`PYTHONPATH=tools .venv/bin/python tools/eval_caps.py`.",
        "",
        "| Метрика | Значение |",
        "|---|---|",
        f"| Верный вердикт | **{m['acc']:.0f} %** ({m['tp'] + m['tn']} из {m['n']}) |",
        f"| Найдено нарушений (полнота) | **{m['tp']} из {m['viol']}** ({m['recall']:.0f} %) |",
        f"| Ложные срабатывания | **{m['fp']} из {m['clean']}** "
        f"(специфичность {m['spec']:.0f} %) |",
        f"| Ошибка таймкода | медиана **{m['med']:.1f} с**, ≤3 с у {m['le3']:.0f} %, "
        f"средняя {m['mean']:.1f} с |",
        *(f"| Устойчивость: {k} | {v['ok']} из {v['total']} прогонов с верным вердиктом |"
          for k, v in r.items()),
        "",
        f"Найдены: {', '.join('`' + h + '`' for h in m['hits'])}. "
        + (f"Ложное срабатывание: {', '.join('`' + f + '`' for f in m['fps'])}."
           if m['fps'] else "Ложных срабатываний нет."),
        "",
        END,
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="вставить блок в документы")
    ap.add_argument("--check", action="store_true", help="сверить блок в документах")
    a = ap.parse_args()

    m, r = measure(), robustness()
    block = table(m, r)
    if not (a.write or a.check):
        print(block)
        return 0

    rc = 0
    for name in TARGETS:
        p = Path(name)
        if not p.exists():
            print(f"{name}: файла нет"); rc = 1; continue
        s = p.read_text(encoding="utf-8")
        if BEGIN not in s or END not in s:
            print(f"{name}: нет маркеров {BEGIN} ... {END} — вставьте их вручную")
            rc = 1
            continue
        head, rest = s.split(BEGIN, 1)
        _, tail = rest.split(END, 1)
        new = head + block + tail
        if a.check:
            print(f"{name}: {'актуально' if new == s else 'УСТАРЕЛО'}")
            rc |= int(new != s)
        else:
            if new != s:
                p.write_text(new, encoding="utf-8")
            print(f"{name}: {'обновлён' if new != s else 'без изменений'}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
