"""Сверка цитируемых пунктов с действующей редакцией акта.

Нормативные акты РК правятся постановлениями, и у каждого пункта в тексте на
adilet стоит сноска «Пункт N – в редакции постановления ... от ДД.ММ.ГГГГ № NN».
Если сервис цитирует пункт, не указав редакцию, проверяющий не может быть уверен,
что цитата соответствует действующему тексту.

Скрипт идёт по `kb/rules.json`, находит номер пункта, ищет в выгруженном тексте
соответствующую сноску и сообщает, указана ли она в поле `edition`.

    .venv/bin/python tools/check_editions.py           # отчёт
    .venv/bin/python tools/check_editions.py --write    # проставить поле edition
"""
import argparse
import json
import re
from pathlib import Path

KB = Path("kb/rules.json")
DOCS = {
    "231": Path("kb/doc1_plain.txt"),
    "14": Path("kb/doc2_plain.txt"),
}
# Сноска занимает несколько строк: номер постановления переносится на следующую,
# а дата введения в действие — на третью. Пример из `kb/doc1_plain.txt`:
#   '\xa0... Сноска. Пункт 12 – в редакции постановления ... РК от 10.11.2025 '
#   '№ 71'
#   ' (вводится в действие ...).'
# Поэтому берём фиксированное окно после номера пункта, а не ищем конец сноски:
# первая версия искала следующую «Сноску» как границу и возвращала «сносок нет»
# на всех 14 нормах — то есть молча мерила не то.
FOOTNOTE = re.compile(r"Сноска\.\s*Пункт\s+(\S+?)\s*[-–—]\s*(.{0,160})", re.S)


def doc_key(requisites: str) -> str | None:
    m = re.search(r"№\s*(\d+)", requisites or "")
    return m.group(1) if m and m.group(1) in DOCS else None


def footnotes(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    out: dict[str, str] = {}
    for num, body in FOOTNOTE.findall(text):
        body = " ".join(body.split())
        out[num.strip()] = body
    return out


def clause_num(clause: str) -> str | None:
    m = re.search(r"пункт\s+(\d+(?:-\d+)?)", clause or "", re.I)
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    cache = {k: footnotes(p) for k, p in DOCS.items() if p.exists()}
    rules = json.loads(KB.read_text(encoding="utf-8"))
    changed = 0
    print(f"{'id':6s} {'акт':5s} {'пункт':7s} {'редакция по тексту акта':48s} статус")
    for r in rules:
        key = doc_key(r.get("doc_requisites", ""))
        num = clause_num(r.get("clause", ""))
        if not key or not num:
            print(f"{r['id']:6s} {'?':5s} {str(num):7s} {'—':48s} не разобрано")
            continue
        fn = cache.get(key, {}).get(num)
        if not fn:
            status = "сносок нет — пункт в первоначальной редакции"
            short = "—"
        else:
            m = re.search(r"(в редакции|с изменени\S+).{0,80}?(\d{2}\.\d{2}\.\d{4})"
                          r"\s*№?\s*(\d+)?", fn)
            short = (f"{m.group(1)} постановления от {m.group(2)}"
                     + (f" № {m.group(3)}" if m and m.group(3) else "")) if m else fn[:46]
            if r.get("edition") == short:
                status = "указана"
            else:
                status = "НЕ УКАЗАНА"
                if a.write:
                    r["edition"] = short
                    changed += 1
        print(f"{r['id']:6s} {key:5s} {num:7s} {short[:48]:48s} {status}")

    if a.write and changed:
        KB.write_text(json.dumps(rules, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
        print(f"\nполе edition проставлено у {changed} норм")
    elif a.write:
        print("\nвсе редакции уже указаны")


if __name__ == "__main__":
    main()
