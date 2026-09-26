#!/usr/bin/env bash
set -euo pipefail

ollama serve &

until curl -sf http://127.0.0.1:11434/api/tags >/dev/null 2>&1; do
  sleep 1
done

exec python3 -u handler.py
