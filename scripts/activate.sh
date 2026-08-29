#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "This script must be sourced so it can update the current shell." >&2
  echo "Run: source scripts/activate.sh" >&2
  exit 1
fi

cpv26_project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ ! -f "${cpv26_project_root}/.venv/bin/activate" ]]; then
  echo "Missing .venv. Run: bash scripts/setup.sh base" >&2
  return 1
fi

if [[ ! -f "${cpv26_project_root}/.env" ]]; then
  echo "Missing .env. Run: cp .env.example .env" >&2
  return 1
fi

cd "${cpv26_project_root}" || return 1
if ! source .venv/bin/activate; then
  echo "Failed to activate .venv." >&2
  return 1
fi

while IFS= read -r cpv26_env_line || [[ -n "${cpv26_env_line}" ]]; do
  cpv26_env_line="${cpv26_env_line%$'\r'}"
  if [[ -z "${cpv26_env_line}" || "${cpv26_env_line}" =~ ^[[:space:]]*# ]]; then
    continue
  fi
  if [[ ! "${cpv26_env_line}" =~ ^[A-Za-z_][A-Za-z0-9_]*=.*$ ]]; then
    echo "Invalid .env entry: ${cpv26_env_line}" >&2
    return 1
  fi
  export "${cpv26_env_line}"
done < .env

echo "CPV26 environment activated: ${cpv26_project_root}"
