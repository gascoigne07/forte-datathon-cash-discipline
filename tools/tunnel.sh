#!/usr/bin/env bash
# Запасной публичный вход: quick tunnel Cloudflare на локальный сервис.
# Адрес quick tunnel меняется при каждом перезапуске, поэтому актуальный URL
# пишется в data/public_url.txt (юнит forte-tunnel, Restart=always).
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
: > data/public_url.txt
cloudflared tunnel --no-autoupdate --url http://127.0.0.1:8000 2>&1 | while IFS= read -r line; do
  echo "$line"
  url=$(printf '%s' "$line" | grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' | head -1)
  if [ -n "$url" ]; then
    printf '%s\n' "$url" > data/public_url.txt
    echo "PUBLIC_URL $url"
  fi
done
