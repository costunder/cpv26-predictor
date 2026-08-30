#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
  echo "Run this script with: bash scripts/check.sh" >&2
  return 1
fi
set -euo pipefail

cpv26_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${cpv26_script_dir}/conda_guard.sh"
cpv26_require_conda || exit 1
"${cpv26_python}" -m compileall -q src tests scripts/build_code_summary.py scripts/conda_guard.py
"${cpv26_python}" -m ruff check src tests scripts/build_code_summary.py scripts/conda_guard.py
"${cpv26_python}" -m mypy --no-incremental src/cpv26
"${cpv26_python}" -m pytest
"${cpv26_python}" -m pip check
