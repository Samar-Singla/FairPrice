#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -x .venv/bin/python ]]; then
  echo 'Create .venv and install requirements.lock.txt first; see README.md.' >&2
  exit 1
fi
if [[ ! -f frontend/dist/index.html ]]; then
  npm ci --prefix frontend
  npm run build --prefix frontend
fi
exec .venv/bin/python -m uvicorn backend.app:app --host 127.0.0.1 --port "${PORT:-8000}"
