#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
  echo "Run this script with: bash scripts/setup.sh [profile]" >&2
  return 1
fi
set -euo pipefail

cpv26_setup_main() {
profile="${1:-base}"
constraints="requirements/constraints.txt"

case "${profile}" in
  base|dev|tabular|ml-cpu|ml-cuda) ;;
  *)
    echo "Usage: bash scripts/setup.sh [base|dev|tabular|ml-cpu|ml-cuda]" >&2
    return 2
    ;;
esac

if [[ ! -f pyproject.toml || ! -f "${constraints}" ]]; then
  echo "Run this command from the cpv26-predictor repository root." >&2
  return 1
fi

cpv26_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${cpv26_script_dir}/conda_guard.sh"
cpv26_require_conda || return 1

"${cpv26_python}" -m pip install --upgrade pip wheel

cuda_torch_ready() {
  "${cpv26_python}" - <<'PY'
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
    "${cpv26_python}" -m pip install -c "${constraints}" -e .
    ;;
  dev)
    "${cpv26_python}" -m pip install -c "${constraints}" -e '.[dev]'
    ;;
  tabular)
    "${cpv26_python}" -m pip install -c "${constraints}" -e '.[dev,tabular]'
    ;;
  ml-cpu)
    "${cpv26_python}" -m pip install -c "${constraints}" -e '.[dev,tabular]'
    "${cpv26_python}" -m pip install 'torch>=2.4,<3' \
      --index-url https://download.pytorch.org/whl/cpu
    ;;
  ml-cuda)
    "${cpv26_python}" -m pip install -c "${constraints}" -e '.[dev]'
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
        return 3
      fi
      if [[ ! "${torch_index_url}" =~ ^https://download[.]pytorch[.]org/whl/cu[0-9]+$ ]]; then
        echo "TORCH_INDEX_URL must be an official stable CUDA index:" >&2
        echo "  https://download.pytorch.org/whl/cu<version>" >&2
        echo "Copy the exact URL from the official PyTorch installation selector." >&2
        return 2
      fi
      # An explicit index authorizes replacing an unusable/CPU build, even when
      # that installed build has a newer public version than the CUDA index.
      "${cpv26_python}" -m pip install --upgrade --force-reinstall 'torch>=2.4,<3' \
        --index-url "${torch_index_url}"
      # Restore this project's pinned core/dev dependencies after torch's
      # dependency resolution, then fail if the selected wheel conflicts.
      "${cpv26_python}" -m pip install -c "${constraints}" -e '.[dev]'
      if ! cuda_torch_ready; then
        echo "CUDA PyTorch still cannot execute a forward/backward kernel on this GPU." >&2
        echo "Check the selected CUDA build, NVIDIA driver, and GPU visibility." >&2
        return 3
      fi
    fi
    ;;
esac

"${cpv26_python}" -m pip check
echo "Conda environment ready: ${CONDA_PREFIX}"
echo "Create .env if needed, then run: source scripts/activate.sh"
}

cpv26_setup_main "$@"
