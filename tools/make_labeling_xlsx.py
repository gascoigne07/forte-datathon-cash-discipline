"""Сборка файла разметки для напарника: .xlsx с подсказками и .csv с BOM.

Зачем .xlsx, а не только .csv: Excel на macOS открывает CSV в MacRoman, если в
файле нет BOM, и кириллица превращается в мусор — это и произошло у напарника.
BOM добавлен, но .xlsx надёжнее в принципе: кодировка там не при чём, а заодно
можно дать выпадающие списки, чтобы коды типов нельзя было написать с опечаткой.

    PYTHONPATH=tools .venv/bin/python tools/make_labeling_xlsx.py [каталог]
"""
import sys
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import LABELS  # noqa: E402

DUP = {"Х-47", "Х-48"}

TYPES = [
    ("manual_count", "пересчёт наличных руками, счётная машина не используется"),
    ("cash_unattended", "наличные лежат на столе, кассир их не контролирует"),
    ("cash_handover", "деньги переданы из рук в руки / положены мимо лотка"),
    ("left_workplace", "кассир ушёл с рабочего места, не убрав ценности"),
    ("cabin_open", "дверь кассовой кабины открыта / кабина не заперта"),
    ("safe_open", "сейф или шкаф оставлен открытым"),
    ("outsider", "посторонний в помещении во время операции с наличными"),
    ("food_drink", "стакан, кружка, банка, бутылка, еда на рабочем месте"),
    ("personal_bag", "личная сумка, рюкзак, пакет на рабочем месте"),
    ("sealed_bag", "инкассаторская сумка вне сейфа / передана без оформления"),
    ("phone_use", "кассир пользуется телефоном на рабочем месте"),
    ("other", "другое — обязательно описать словами в колонке «что происходит»"),
    ("unclear", "не видно, что именно нарушено"),
]

# Ролик -> что сейчас выдаёт сервис. Первыми идут те, где он не находит ничего:
# про них у нас нет никакой информации, и там каждая строка ценнее всего.
FOUND = {
    "Х-4": "находит «посторонние в помещении» на 42 с",
    "Х-11": "находит «рабочее место без работника»",
    "Х-13": "находит «рабочее место без работника» на 57 с",
    "Х-15": "находит «инкассаторская сумка вне сейфа»",
    "Х-24": "находит «напиток или еда»",
    "Х-26": "находит «рабочее место без работника» на 17 с",
    "Х-30": "находит «рабочее место без работника» на 0 с",
    "Х-32": "находит «рабочее место без работника»",
    "Х-38": "находит «посторонние в помещении»",
    "Х-40": "находит «личная сумка на рабочем месте»",
    "Х-46": "находит «напиток или еда»",
}
HEAD = PatternFill("solid", fgColor="1F3864")
ALT = PatternFill("solid", fgColor="EDF2FA")


def rows() -> list[tuple[str, int, str]]:
    out = []
    for name, secs in LABELS.items():
        if not secs or name in DUP:
            continue
        for s in secs:
            out.append((name, s, FOUND.get(name, "программа не находит ничего")))
    # сначала ненайденные, внутри групп — по номеру ролика
    out.sort(key=lambda r: (r[2] != "программа не находит ничего",
                            len(r[0]), r[0], r[1]))
    return out


def build(dst_dir: Path) -> None:
    wb = Workbook()

    ws = wb.active
    ws.title = "разметка"
    cols = [("ролик", 12), ("секунда", 10), ("тип (выбрать из списка)", 26),
            ("что происходит, 5-15 слов", 46),
            ("чем это видно на экране — предмет, место, действие", 60),
            ("уверенность", 14), ("что выдаёт программа сейчас", 42)]
    ws.append([c[0] for c in cols])
    for i, (_, w) in enumerate(cols, 1):
        ws.column_dimensions[get_column_letter(i)].width = w
        c = ws.cell(row=1, column=i)
        c.font = Font(bold=True, color="FFFFFF")
        c.fill = HEAD
        c.alignment = Alignment(wrap_text=True, vertical="center")
    ws.row_dimensions[1].height = 34

    data = rows()
    for n, (name, sec, note) in enumerate(data, start=2):
        ws.append([name, sec, "", "", "", "", note])
        for i in range(1, len(cols) + 1):
            cell = ws.cell(row=n, column=i)
            cell.alignment = Alignment(wrap_text=True, vertical="top")
            if n % 2 == 0:
                cell.fill = ALT
        ws.cell(row=n, column=7).font = Font(italic=True, color="808080")
        ws.row_dimensions[n].height = 30

    last = len(data) + 1
    dv = DataValidation(type="list", formula1='"' + ",".join(t for t, _ in TYPES) + '"',
                        allow_blank=True, showDropDown=False)
    dv.error = "Выберите код из списка на листе «коды типов»"
    dv.errorTitle = "Неизвестный код"
    ws.add_data_validation(dv)
    dv.add(f"C2:C{last}")

    dvc = DataValidation(type="list", formula1='"high,med,low"', allow_blank=True,
                         showDropDown=False)
    ws.add_data_validation(dvc)
    dvc.add(f"F2:F{last}")

    ws.freeze_panes = "C2"

    # лист с кодами — чтобы не держать инструкцию открытой рядом
    ws2 = wb.create_sheet("коды типов")
    ws2.append(["код", "что значит"])
    for i, w in ((1, 22), (2, 70)):
        ws2.column_dimensions[get_column_letter(i)].width = w
        c = ws2.cell(row=1, column=i)
        c.font = Font(bold=True, color="FFFFFF")
        c.fill = HEAD
    for code, desc in TYPES:
        ws2.append([code, desc])
    for r in range(2, len(TYPES) + 2):
        ws2.cell(row=r, column=2).alignment = Alignment(wrap_text=True)

    ws3 = wb.create_sheet("как заполнять")
    text = [
        "Как заполнять",
        "",
        "1. Открыть ролик в QuickTime или VLC, перемотать на (секунда - 3).",
        "   QuickTime: стрелки влево/вправо листают по кадру. VLC: клавиша E.",
        "2. Проиграть покадрово до (секунда + 3) и понять, что происходит.",
        "3. Заполнить колонки «тип», «что происходит», «чем это видно», «уверенность».",
        "   Колонки «ролик» и «секунда» уже заполнены — их менять не нужно.",
        "4. Если на указанной секунде ничего особенного не видно — посмотреть весь",
        "   ролик целиком. Если всё равно непонятно, поставить тип unclear, а в",
        "   «что происходит» описать, что вообще в ролике делается.",
        "",
        "Три правила, которые важнее скорости",
        "",
        "1. Не угадывать. unclear — нормальный и полезный ответ. Выдуманный тип хуже",
        "   отсутствующего: по нему будет построено правило, которое даст ложные",
        "   срабатывания на проверке у жюри. Лучше 8 честных строк и 13 unclear,",
        "   чем 21 придуманная.",
        "",
        "2. Колонка «чем это видно» — главная. Писать физически: не «нарушение",
        "   кассовой дисциплины», а «зелёная сумка лежит на столешнице, кассир стоит",
        "   у стеклянной стены в двух метрах от стола». Из этой колонки выводится",
        "   признак, по которому будет работать программа.",
        "",
        "3. Если в ролике несколько событий — добавить строку с тем же роликом и",
        "   своей секундой. Про Х-4 это уже известно: он размечен на 57 и 68 секунде,",
        "   обе строки в таблице есть.",
        "",
        "Колонка «что выдаёт программа сейчас» — подсказка, а не ответ. Если видно",
        "другое, писать то, что видно: расхождение программы с человеком — это и есть",
        "то, что мы ищем.",
        "",
        "Первые 10 строк — ролики, где программа не находит ничего. Про них у нас нет",
        "никакой информации, и каждая заполненная строка здесь даёт больше, чем все",
        "остальные вместе.",
        "",
        "Один ролик — 2-3 минуты. Всего около часа.",
        "",
        "Пример заполненной строки",
        "",
        "  ролик:            Х-46",
        "  секунда:          11",
        "  тип:              food_drink",
        "  что происходит:   кассир работает, рядом на столе стоит стакан с напитком",
        "  чем это видно:    прозрачный стакан слева от клавиатуры, в 20 см от рук,",
        "                    стоит весь ролик, никто его не убирает",
        "  уверенность:      high",
        "",
        "Ещё пример — когда непонятно",
        "",
        "  ролик:            Х-20",
        "  секунда:          5",
        "  тип:              unclear",
        "  что происходит:   кассир перебирает бумаги, ничего необычного не вижу",
        "  чем это видно:    на столе папка и листы, денег в кадре нет,человек один",
        "  уверенность:      low",
    ]
    ws3.column_dimensions["A"].width = 100
    for i, line in enumerate(text, start=1):
        ws3.cell(row=i, column=1, value=line)
        if i == 1 or line in ("Три правила, которые важнее скорости",
                              "Пример заполненной строки",
                              "Ещё пример — когда непонятно"):
            ws3.cell(row=i, column=1).font = Font(bold=True, size=13)

    dst_dir.mkdir(parents=True, exist_ok=True)
    xlsx = dst_dir / "РАЗМЕТКА.xlsx"
    wb.save(xlsx)

    # CSV с BOM — на случай, если удобнее текстом; Excel на macOS без BOM
    # открывает файл в MacRoman и кириллица превращается в мусор
    csv = dst_dir / "РАЗМЕТКА-запасной-вариант.csv"
    lines = ["video;sec;type;what;cue;conf"]
    lines += [f"{n};{s};;;;" for n, s, _ in data]
    csv.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")

    print(f"{xlsx}  — {len(data)} строк, выпадающие списки, 3 листа")
    print(f"{csv}  — то же в CSV, UTF-8 с BOM")


if __name__ == "__main__":
    build(Path(sys.argv[1] if len(sys.argv) > 1 else "annotation/out"))
