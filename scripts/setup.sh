#!/usr/bin/env bash
set -euo pipefail

profile="${1:-base}"
python_bin="${PYTHON_BIN:-python3}"
constraints="requirements/constraints.txt"

if [[ ! -f pyproject.toml || ! -f "${constraints}" ]]; then
  echo "Run this command from the cpv26-predictor repository root." >&2
  exit 1
fi

if [[ ! -x .venv/bin/python ]]; then
  if ! command -v "${python_bin}" >/dev/null 2>&1; then
    echo "Python executable not found: ${python_bin}" >&2
    echo "Set PYTHON_BIN to an installed Python 3.10-3.12 executable." >&2
    exit 1
  fi
  if ! "${python_bin}" -c 'import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] < (3, 13) else 1)'; then
    cpv26_python_version="$("${python_bin}" --version 2>&1)"
    echo "Unsupported Python: ${cpv26_python_version}" >&2
    echo "CPV26 requires Python 3.10-3.12." >&2
    exit 1
  fi
  "${python_bin}" -m venv .venv
fi
source .venv/bin/activate

if ! python -c 'import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] < (3, 13) else 1)'; then
  echo "The existing .venv does not use Python 3.10-3.12." >&2
  echo "Move the stale .venv aside and rerun setup." >&2
  exit 1
fi
if ! python -m pip --version >/dev/null 2>&1; then
  echo "The existing .venv is incomplete because pip is unavailable." >&2
  echo "Move the incomplete .venv aside and rerun setup." >&2
  exit 1
fi

python -m pip install --upgrade pip wheel

case "${profile}" in
  base)
    python -m pip install -c "${constraints}" -e .
    ;;
  dev)
    python -m pip install -c "${constraints}" -e '.[dev]'
    ;;
  ml-cpu)
    python -m pip install -c "${constraints}" -e '.[dev,tabular]'
    python -m pip install 'torch>=2.4,<3' \
      --index-url https://download.pytorch.org/whl/cpu
    ;;
  ml-cuda)
    python -m pip install -c "${constraints}" -e '.[dev,tabular]'
    if ! python -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)'; then
      echo "CUDA-enabled PyTorch is missing or cannot see the GPU." >&2
      echo "Activate .venv, install the server-compatible PyTorch wheel, then rerun:" >&2
      echo "  bash scripts/setup.sh ml-cuda" >&2
      exit 3
    fi
    ;;
  *)
    echo "Usage: bash scripts/setup.sh [base|dev|ml-cpu|ml-cuda]" >&2
    exit 2
    ;;
esac

echo "Environment ready."
echo "Create .env if needed, then run: source scripts/activate.sh"
