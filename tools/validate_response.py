"""Проверка ответа сервиса на соответствие контракту раздела 10 ТЗ.

Читает JSON из stdin, печатает сводку и выходит с кодом 1, если контракт нарушен.
Вызывается из tools/jury_check.sh, но годится и отдельно:
    curl ... | python3 tools/validate_response.py [прошедшие_секунды]
"""
import json
import sys


def main() -> int:
    elapsed = int(sys.argv[1]) if len(sys.argv) > 1 else -1
    try:
        d = json.load(sys.stdin)
    except json.JSONDecodeError as exc:
        print(f"  ОТВЕТ НЕ JSON: {exc}")
        return 1

    bad = []
    status = d.get("status")
    if status != "done":
        bad.append(f"status={status!r}, error={d.get('error')!r}")

    verdict = d.get("verdict")
    if verdict not in ("violation", "no_violation"):
        bad.append(f"verdict={verdict!r}, ожидается violation|no_violation")

    violations = d.get("violations") or []
    for v in violations:
        for key in ("type", "timecode_sec", "description", "evidence_frames"):
            if key not in v:
                bad.append(f"в элементе violations нет поля {key}")
        frames = v.get("evidence_frames") or []
        if not 1 <= len(frames) <= 3:
            bad.append(f"кадров-доказательств {len(frames)}, ТЗ разрешает 1-3")
        for f in frames:
            if "timestamp_sec" not in f or "image_base64" not in f:
                bad.append("кадр без timestamp_sec или image_base64")
            elif len(f.get("image_base64") or "") < 1000:
                bad.append("кадр подозрительно мал — возможно, пустое изображение")

    if verdict == "violation" and not violations:
        bad.append("вердикт violation, но список violations пуст")
    if verdict == "no_violation" and violations:
        bad.append("вердикт no_violation, но список violations не пуст")
    if elapsed > 300:
        bad.append(f"превышен лимит 5 минут: {elapsed} с")

    print(f"  вердикт: {verdict}  событий: {len(violations)}  "
          f"время: {elapsed} с (обработка {d.get('processing_sec')} с, "
          f"кадров {d.get('frames_analyzed')}/{d.get('frames_total_sampled')})")
    if d.get("low_coverage"):
        print("  ВНИМАНИЕ: low_coverage — разобрано мало кадров, вердикт недостоверен")
    for v in violations:
        rule = v.get("rule") or {}
        mark = " (трактовка)" if rule.get("verified") is False else ""
        print(f"    - {v.get('type')} @ {v.get('timecode_sec')} с, "
              f"кадров {len(v.get('evidence_frames') or [])}, "
              f"норматив: {rule.get('clause')}{mark}")
        print(f"      {(v.get('description') or '')[:150]}")

    if bad:
        print("  НАРУШЕНИЯ КОНТРАКТА:")
        for b in bad:
            print(f"    * {b}")
        return 1
    print("  контракт соблюдён")
    return 0


if __name__ == "__main__":
    sys.exit(main())
