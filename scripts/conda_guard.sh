#!/usr/bin/env bash

# Source this helper; it never creates, activates, or modifies an environment.
cpv26_require_conda() {
  local cpv26_guard_dir
  cpv26_guard_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" || return 1
  if ! command -v python >/dev/null 2>&1; then
    echo "Python is unavailable. Run: conda activate cpv26" >&2
    return 1
  fi
  if ! cpv26_python="$(python "${cpv26_guard_dir}/conda_guard.py")"; then
    unset cpv26_python
    return 1
  fi
}
