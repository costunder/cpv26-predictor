#!/usr/bin/env bash
set -euo pipefail

profile="${1:-base}"
python_bin="${PYTHON_BIN:-python3}"
constraints="requirements/constraints.txt"

case "${profile}" in
  base|dev|tabular|ml-cpu|ml-cuda) ;;
  *)
    echo "Usage: bash scripts/setup.sh [base|dev|tabular|ml-cpu|ml-cuda]" >&2
    exit 2
    ;;
esac

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
  if ! "${python_bin}" -m venv .venv; then
    echo "Virtual environment creation failed; any partial .venv was preserved." >&2
    echo "Install the matching Python venv package, move .venv aside, and rerun setup." >&2
    exit 1
  fi
fi
if [[ ! -f .venv/bin/activate ]]; then
  echo "The existing .venv is incomplete because bin/activate is unavailable." >&2
  echo "Move the incomplete .venv aside and rerun setup." >&2
  exit 1
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

cuda_torch_ready() {
  python - <<'PY'
import torch
from packaging.version import Version

if not Version("2.4") <= Version(torch.__version__) < Version("3"):
    raise SystemExit("This project requires torch>=2.4,<3.")
if not hasattr(torch.amp, "GradScaler") or not hasattr(torch, "autocast"):
    raise SystemExit("The installed PyTorch lacks the required AMP APIs.")
if torch.version.cuda is None or not torch.cuda.is_available():
    raise SystemExit("A CUDA-enabled PyTorch build and an accessible NVIDIA GPU are required.")
device = torch.device("cuda:0")
value = torch.randn((8, 8), device=device, requires_grad=True)
loss = (value @ value.T).square().mean()
loss.backward()
torch.cuda.synchronize(device)
if value.grad is None or not bool(torch.isfinite(value.grad).all().item()):
    raise SystemExit("CUDA backward produced invalid gradients.")
print(f"CUDA PyTorch ready: torch={torch.__version__}, GPU={torch.cuda.get_device_name(device)}")
PY
}

case "${profile}" in
  base)
    python -m pip install -c "${constraints}" -e .
    ;;
  dev)
    python -m pip install -c "${constraints}" -e '.[dev]'
    ;;
  tabular)
    python -m pip install -c "${constraints}" -e '.[dev,tabular]'
    ;;
  ml-cpu)
    python -m pip install -c "${constraints}" -e '.[dev,tabular]'
    python -m pip install 'torch>=2.4,<3' \
      --index-url https://download.pytorch.org/whl/cpu
    ;;
  ml-cuda)
    python -m pip install -c "${constraints}" -e '.[dev]'
    if cuda_torch_ready >/dev/null 2>&1; then
      echo "Existing CUDA PyTorch passed a forward/backward check; leaving it unchanged."
    else
      torch_index_url="${TORCH_INDEX_URL:-}"
      torch_index_url="${torch_index_url%/}"
      if [[ -z "${torch_index_url}" ]]; then
        echo "CUDA-enabled PyTorch is missing or failed the CUDA kernel check." >&2
        echo "No CUDA wheel was selected or installed automatically." >&2
        echo "Choose the CUDA index for this server at https://pytorch.org/get-started/locally/" >&2
        echo "Then rerun with TORCH_INDEX_URL set to that command's --index-url value." >&2
        exit 3
      fi
      if [[ ! "${torch_index_url}" =~ ^https://download[.]pytorch[.]org/whl/cu[0-9]+$ ]]; then
        echo "TORCH_INDEX_URL must be an official stable CUDA index:" >&2
        echo "  https://download.pytorch.org/whl/cu<version>" >&2
        echo "Copy the exact URL from the official PyTorch installation selector." >&2
        exit 2
      fi
      # An explicit index authorizes replacing an unusable/CPU build, even when
      # that installed build has a newer public version than the CUDA index.
      python -m pip install --upgrade --force-reinstall 'torch>=2.4,<3' \
        --index-url "${torch_index_url}"
      # Restore this project's pinned core/dev dependencies after torch's
      # dependency resolution, then fail if the selected wheel conflicts.
      python -m pip install -c "${constraints}" -e '.[dev]'
      if ! cuda_torch_ready; then
        echo "CUDA PyTorch still cannot execute a forward/backward kernel on this GPU." >&2
        echo "Check the selected CUDA build, NVIDIA driver, and GPU visibility." >&2
        exit 3
      fi
    fi
    ;;
esac

python -m pip check
echo "Environment ready."
echo "Create .env if needed, then run: source scripts/activate.sh"
