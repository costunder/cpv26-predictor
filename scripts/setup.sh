#!/usr/bin/env bash
set -euo pipefail

profile="${1:-base}"
python_bin="${PYTHON_BIN:-python3.10}"
constraints="requirements/constraints.txt"

if ! command -v "${python_bin}" >/dev/null 2>&1; then
  echo "Python executable not found: ${python_bin}" >&2
  echo "Set PYTHON_BIN to an installed Python 3.10-3.12 executable." >&2
  exit 1
fi

if [[ ! -x .venv/bin/python ]]; then
  "${python_bin}" -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip wheel

case "${profile}" in
  base)
    python -m pip install -c "${constraints}" -e .
    ;;
  dev)
    python -m pip install -c "${constraints}" -e '.[dev]'
    ;;
  ml-cpu)
    python -m pip install -c "${constraints}" -e '.[dev,tabular,neural]'
    ;;
  ml-cuda)
    python -m pip install -c "${constraints}" -e '.[tabular]'
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

echo "Environment ready. Activate it with: source .venv/bin/activate"
