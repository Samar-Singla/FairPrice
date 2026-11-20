#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"
echo '1/4  Python lint'
.venv/bin/ruff check backend tests scripts
echo '2/4  Statistical and complete API workflow tests (isolated temporary storage)'
.venv/bin/python -m pytest -q
echo '3/4  TypeScript checks and production build'
npm run build --prefix frontend
echo '4/4  Python source compilation'
.venv/bin/python -m compileall -q backend scripts
echo 'PASS: automated verification completed. See docs/ACCEPTANCE.md for browser checks.'
