#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "This script must be sourced so it can update the current shell." >&2
  echo "Run: source scripts/activate.sh" >&2
  exit 1
fi

cpv26_project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || return 1
if ! source "${cpv26_project_root}/scripts/conda_guard.sh"; then
  return 1
fi
if ! cpv26_require_conda; then
  return 1
fi

if [[ ! -f "${cpv26_project_root}/.env" ]]; then
  echo "Missing .env. Run: cp .env.example .env" >&2
  return 1
fi
if [[ ! -r "${cpv26_project_root}/.env" ]]; then
  echo "Cannot read .env; check its ownership and permissions." >&2
  return 1
fi

cpv26_env_entries=()
if ! while IFS= read -r cpv26_env_line || [[ -n "${cpv26_env_line}" ]]; do
  cpv26_env_line="${cpv26_env_line%$'\r'}"
  if [[ -z "${cpv26_env_line}" || "${cpv26_env_line}" =~ ^[[:space:]]*# ]]; then
    continue
  fi
  if [[ ! "${cpv26_env_line}" =~ ^CPV26_[A-Z0-9_]+= ]]; then
    echo "Invalid .env key: only CPV26_[A-Z0-9_]+ configuration keys are allowed." >&2
    unset cpv26_env_entries cpv26_env_line
    return 1
  fi
  cpv26_env_entries+=("${cpv26_env_line}")
done < "${cpv26_project_root}/.env"; then
  echo "Could not load .env; no project configuration was exported." >&2
  unset cpv26_env_entries cpv26_env_line
  return 1
fi

# Validate the entire file before changing the directory or exporting any values.
cd "${cpv26_project_root}" || return 1
for cpv26_env_line in "${cpv26_env_entries[@]}"; do
  if ! export "${cpv26_env_line}"; then
    echo "Could not export a project configuration value from .env." >&2
    unset cpv26_env_entries cpv26_env_line
    return 1
  fi
done
unset cpv26_env_entries cpv26_env_line

echo "CPV26 configuration loaded in Conda environment ${CONDA_DEFAULT_ENV}: ${cpv26_project_root}"
