"""Точность ТИПА события (критерий 3 ТЗ, 15 баллов).

Вердикт и таймкод мерит `tools/eval_caps.py`. Тип события им не проверить: в
`open_labels.xlsx` указаны только секунды, без типа нарушения. Тип берётся из
ручной разметки `annotation/labels.csv` (формат описан в `annotation/LABELING.md`).

    PYTHONPATH=tools .venv/bin/python tools/type_accuracy.py

Печатает по каждому ролику ожидаемый и выданный тип, матрицу путаницы и список
размеченных типов, которых сервис не находит вовсе, — последнее важнее процентов:
оно показывает, под какие типы правил просто нет.

Пока `labels.csv` не заполнен, скрипт честно сообщает, что мерить нечего. Это
«ТРЕБУЕТ ЧЕЛОВЕКА» из `NIGHT-LOG.md`: разметка типов — ручная работа напарника.
"""
import csv
import glob
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "service"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from app import rules  # noqa: E402
from eval_caps import DUPLICATES, series_for  # noqa: E402
from evaluate import LABELS  # noqa: E402

# Напарник возвращает .xlsx (его собирает tools/make_labeling_xlsx.py): CSV в
# Excel на macOS открывается в MacRoman и кириллица рассыпается. Читаем оба формата.
SOURCES = [Path("annotation/РАЗМЕТКА.xlsx"), Path("annotation/labels.xlsx"),
           Path("annotation/labels.csv")]
# код в разметке -> ключ правила в сервисе. Несколько кодов могут вести к одному
# ключу: сервис описывает наблюдаемое, а не юридическую квалификацию.
MAP = {
    "manual_count": {"manual_count"},
    "cash_unattended": {"cash_unattended", "unattended_desk"},
    "cash_handover": {"cash_unattended", "manual_count"},
    "left_workplace": {"left_workplace", "unattended_desk"},
    "cabin_open": {"outsider_with_cash"},
    "outsider": {"outsider_with_cash"},
    "food_drink": {"food_drink"},
    "personal_bag": {"personal_bag"},
    "sealed_bag": {"sealed_bag"},
    "unclear": set(),
}


def source() -> Path | None:
    return next((p for p in SOURCES if p.exists()), None)


def _from_xlsx(path: Path) -> list[dict]:
    from openpyxl import load_workbook
    wb = load_workbook(path, data_only=True)
    ws = wb["разметка"] if "разметка" in wb.sheetnames else wb[wb.sheetnames[0]]
    # Колонки в файле для человека названы по-русски; приводим к ключам разметки
    # по порядку, а не по заголовку: заголовок напарник может переписать.
    # По ЗАГОЛОВКУ, а не по позиции: колонки уже один раз менялись (добавлены
    # статус и времена), и позиционное чтение молча сдвинуло бы все поля.
    ALIAS = {
        "video": ("ролик", "video"),
        "sec": ("секунда организаторов", "секунда", "sec"),
        "type": ("что наблюдается (код)", "тип (выбрать из списка)", "тип", "type", "код"),
        "what": ("что происходит", "what"),
        "cue": ("чем это видно", "cue"),
        "conf": ("уверенность", "conf"),
        "status": ("статус", "status"),
        "t_start": ("t_начало", "t_start"),
        "t_evidence": ("t_доказательство", "t_evidence"),
        "t_end": ("t_конец", "t_end"),
        "unclear_reason": ("что мешает решить", "unclear_reason"),
    }
    head = [str(c.value or "").strip().lower() for c in ws[1]]
    idx = {}
    for key, names in ALIAS.items():
        for i, h in enumerate(head):
            if any(h.startswith(nm) for nm in names):
                idx[key] = i
                break
    missing = {"video", "type"} - set(idx)
    if missing:
        raise SystemExit(f"в разметке нет колонок {missing}; заголовок: {head}")
    out = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        if not row or not row[idx["video"]]:
            continue
        out.append({k: (str(row[i]).strip() if i < len(row) and row[i] is not None else "")
                    for k, i in idx.items()})
    return out


def _from_csv(path: Path) -> list[dict]:
    # utf-8-sig снимает BOM, если файл пересохраняли из Excel
    with path.open(encoding="utf-8-sig") as fh:
        return [dict(r) for r in csv.DictReader(fh, delimiter=";")]


def load_labels() -> dict[str, list[dict]]:
    path = source()
    raw = _from_xlsx(path) if path.suffix == ".xlsx" else _from_csv(path)
    rows: dict[str, list[dict]] = {}
    for r in raw:
        name = (r.get("video") or "").strip()
        if name:
            rows.setdefault(name, []).append(r)
    return rows


def main():
    if source() is None:
        print("Разметки нет — тип события мерить нечем. Ожидается один из файлов:")
        for p in SOURCES:
            print(f"  {p}")
        print("Разметка типов делается руками по annotation/LABELING.md.")
        print("Без неё критерий 3 (корректность классификации, 15 баллов) оценивается")
        print("только качественно: сервис выдаёт тип из закрытого списка в TYPES,")
        print("каждый тип привязан к пункту НПА с дословной цитатой.")
        return 1

    gold = load_labels()
    hit = miss = wrong = empty = 0
    confusion: Counter = Counter()
    uncaught: Counter = Counter()
    print(f"{'ролик':8s} {'размечено':18s} {'выдано сервисом':34s} итог")
    for p in sorted(glob.glob("data/captions/*.caps.json")):
        name = Path(p).stem.replace(".caps", "")
        if name not in LABELS or name in DUPLICATES or name not in gold:
            continue
        series, tracks = series_for(name)
        if not series:
            continue
        got = {e.key for e in rules.detect_events(series, tracks=tracks)}
        for row in gold[name]:
            code = (row.get("type") or "").strip()
            if not code:                       # строка ещё не заполнена
                empty += 1
                continue
            want = MAP.get(code)
            if want is None:
                print(f"{name:8s} {code:18s} {'—':34s} код не из списка LABELING.md")
                continue
            if not want:                       # unclear — не предъявляем сервису
                continue
            # Статус «это норма» — мнение разметчика, а НЕ отмена метки
            # организаторов: их положительная метка остаётся основой вердикта
            # (её мерит eval_caps). Здесь такая строка просто не оценивается.
            if (row.get("status") or "").startswith("это_норма"):
                continue
            if got & want:
                hit += 1
                verdict = "тип совпал"
            elif got:
                wrong += 1
                verdict = "ТИП НЕ ТОТ"
                for g in got:
                    confusion[(code, g)] += 1
            else:
                miss += 1
                verdict = "пропуск"
                uncaught[code] += 1
            print(f"{name:8s} {code:18s} {', '.join(sorted(got)) or '—':34s} {verdict}")

    total = hit + wrong + miss
    if not total:
        print(f"\nв разметке нет заполненных типов (пустых строк: {empty})")
        return 1
    print(f"\nтип совпал: {hit} из {total} ({100 * hit / total:.0f} %), "
          f"не тот тип: {wrong}, пропусков: {miss}"
          + (f", ещё не заполнено строк: {empty}" if empty else ""))
    if confusion:
        print("\nматрица путаницы (размечено -> выдано):")
        for (a, b), n in confusion.most_common():
            print(f"  {a:18s} -> {b:24s} {n}")
    if uncaught:
        print("\nтипы, которые сервис не находит вовсе (нужны новые правила):")
        for code, n in uncaught.most_common():
            print(f"  {code:18s} {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
