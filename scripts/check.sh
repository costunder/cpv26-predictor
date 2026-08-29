#!/usr/bin/env bash
set -euo pipefail

if [[ ! -x .venv/bin/python ]]; then
  echo "Missing .venv. Run: bash scripts/setup.sh dev" >&2
  exit 1
fi

source .venv/bin/activate
python -m compileall -q src tests scripts/build_code_summary.py
ruff check src tests scripts/build_code_summary.py
mypy --no-incremental src/cpv26
pytest
python -m pip check
