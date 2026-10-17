#!/usr/bin/env bash
# Запуск сервиса. Модели крутятся локально, наружу торчит только HTTP-вход.
#
#   ./run.sh                 — локально на http://127.0.0.1:8000
#   ./run.sh --tunnel        — плюс публичный URL через Cloudflare Tunnel
#
# Ключ API: берётся из переменной API_KEY, иначе генерируется и печатается.
set -euo pipefail
cd "$(dirname "$0")"

: "${API_KEY:=}"
if [[ -z "$API_KEY" ]]; then
  if [[ -f .api_key ]]; then API_KEY="$(cat .api_key)"
  else
    # Генерируем через python, а не `tr </dev/urandom | head -c 40`: в том конвейере
    # head закрывает поток после 40 байт, tr получает SIGPIPE, и при `set -o pipefail`
    # падает весь скрипт. Срабатывало только при ОТСУТСТВУЮЩЕМ .api_key, то есть
    # ровно при первом запуске у жюри: ./run.sh умирал молча с кодом 141.
    API_KEY="$(.venv/bin/python -c 'import secrets; print(secrets.token_urlsafe(30))')"
    echo "$API_KEY" > .api_key
    chmod 600 .api_key
  fi
fi
export API_KEY

export PYTHONPATH="$PWD/service"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"   # 1 = полный офлайн (веса уже скачаны)
export ANALYZE_BUDGET_SEC="${ANALYZE_BUDGET_SEC:-260}"
PORT="${PORT:-8000}"

echo "================================================================"
echo " API-ключ для жюри:  $API_KEY"
echo " Локально:           http://127.0.0.1:$PORT"
echo " Документация API:   http://127.0.0.1:$PORT/api/docs"
echo "================================================================"

if [[ "${1:-}" == "--tunnel" ]]; then
  command -v cloudflared >/dev/null || { echo "нет cloudflared: brew install cloudflared"; exit 1; }
  cloudflared tunnel --url "http://127.0.0.1:$PORT" --no-autoupdate > .tunnel.log 2>&1 &
  echo "туннель поднимается, публичный URL появится в .tunnel.log ..."
  for _ in $(seq 1 30); do
    url=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' .tunnel.log | head -1 || true)
    [[ -n "$url" ]] && { echo " ПУБЛИЧНЫЙ URL:      $url"; break; }
    sleep 1
  done
fi

exec .venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port "$PORT" --timeout-keep-alive 75
