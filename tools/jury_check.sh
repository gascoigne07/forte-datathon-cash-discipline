#!/usr/bin/env bash
# Прогон ровно по сценарию жюри: загрузить видео, опрашивать статус, проверить схему ответа.
# Падает с ненулевым кодом, если нарушен контракт раздела 10 ТЗ или превышен лимит 5 минут.
#
#   tools/jury_check.sh <base_url> <api_key> <видео> [<видео> ...]
set -uo pipefail

BASE="${1:?укажите базовый URL, например https://xxx.trycloudflare.com}"
KEY="${2:?укажите API-ключ}"
shift 2
[ $# -gt 0 ] || { echo "укажите хотя бы один видеофайл"; exit 2; }

fail=0

echo "== проверка доступности =="
code=$(curl -s -o /dev/null -w '%{http_code}' --connect-timeout 10 "$BASE/api/v1/health")
echo "health: $code"; [ "$code" = "200" ] || { echo "ОШИБКА: сервис недоступен"; exit 1; }

echo "== проверка отказа без ключа =="
code=$(curl -s -o /dev/null -w '%{http_code}' --connect-timeout 10 \
  -X POST "$BASE/api/v1/analyze" -F "file=@$1")
echo "без ключа: $code"; [ "$code" = "401" ] || { echo "ОШИБКА: ожидался 401"; fail=1; }

for video in "$@"; do
  name=$(basename "$video")
  echo "== $name =="
  t0=$(date +%s)
  resp=$(curl -s --connect-timeout 15 -X POST "$BASE/api/v1/analyze" \
    -H "X-API-Key: $KEY" -F "file=@$video")
  jid=$(printf '%s' "$resp" | python3 -c 'import sys,json;print(json.load(sys.stdin).get("job_id",""))' 2>/dev/null)
  if [ -z "$jid" ]; then echo "ОШИБКА: не получен job_id, ответ: $resp"; fail=1; continue; fi
  echo "job_id: $jid"

  status=""
  while :; do
    sleep 5
    body=$(curl -s --connect-timeout 15 "$BASE/api/v1/jobs/$jid" -H "X-API-Key: $KEY")
    status=$(printf '%s' "$body" | python3 -c 'import sys,json;print(json.load(sys.stdin).get("status",""))' 2>/dev/null)
    el=$(( $(date +%s) - t0 ))
    [ "$status" = "done" ] && break
    [ "$status" = "failed" ] && break
    if [ "$el" -gt 300 ]; then echo "ОШИБКА: превышен лимит 5 минут ($el с)"; fail=1; break; fi
  done

  el=$(( $(date +%s) - t0 ))
  printf '%s' "$body" | API_EL="$el" python3 -c '
import json,os,sys
d=json.load(sys.stdin); el=int(os.environ["API_EL"]); bad=[]
if d.get("status")!="done": bad.append(f"status={d.get(\"status\")} error={d.get(\"error\")}")
if d.get("verdict") not in ("violation","no_violation"): bad.append(f"verdict={d.get(\"verdict\")!r}")
for v in d.get("violations") or []:
    for k in ("type","timecode_sec","description","evidence_frames"):
        if k not in v: bad.append(f"в violations нет поля {k}")
    fr=v.get("evidence_frames") or []
    if not 1 <= len(fr) <= 3: bad.append(f"кадров-доказательств {len(fr)}, ТЗ разрешает 1-3")
    for f in fr:
        if "timestamp_sec" not in f or "image_base64" not in f: bad.append("кадр без timestamp_sec/image_base64")
        elif len(f["image_base64"]) < 1000: bad.append("кадр подозрительно мал")
print(f"  вердикт: {d.get(\"verdict\")}  событий: {len(d.get(\"violations\") or [])}  "
      f"время: {el} с (обработка {d.get(\"processing_sec\")} с, кадров {d.get(\"frames_analyzed\")})")
if d.get("low_coverage"): print("  ВНИМАНИЕ: low_coverage — разобрано мало кадров, вердикт недостоверен")
for v in d.get("violations") or []:
    r=v.get("rule") or {}
    print(f"    - {v[\"type\"]} @ {v[\"timecode_sec\"]} с, кадров {len(v[\"evidence_frames\"])}, "
          f"норматив: {r.get(\"clause\")}{\" (трактовка)\" if r.get(\"verified\") is False else \"\"}")
if bad:
    print("  НАРУШЕНИЯ КОНТРАКТА:"); [print("    *",b) for b in bad]; sys.exit(1)
print("  контракт соблюдён")
' || fail=1
done

echo
[ "$fail" = "0 " ] 2>/dev/null
if [ "$fail" -eq 0 ]; then echo "ИТОГ: все проверки пройдены"; else echo "ИТОГ: есть нарушения, см. выше"; fi
exit "$fail"
