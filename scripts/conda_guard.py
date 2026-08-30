"""Validate an active, non-base Conda environment before project commands run."""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path


class CondaEnvironmentError(ValueError):
    """The current Python cannot safely be used for this project's environment."""


def validate_conda_environment(
    environ: Mapping[str, str],
    *,
    python_prefix: str,
    python_base_prefix: str,
    python_executable: str,
    python_version: tuple[int, int],
    pip_available: bool,
    conda_base_prefix: str | None = None,
) -> None:
    """Check supplied runtime facts without invoking Conda or mutating packages.

    The environment prefix must contain ``conda-meta`` and the Python executable.
    ``conda_base_prefix`` must be independently obtained from ``conda info --base``;
    trusting the environment name alone could allow a renamed base environment.
    """
    raw_prefix = environ.get("CONDA_PREFIX", "").strip()
    if not raw_prefix:
        raise CondaEnvironmentError("CONDA_PREFIX is missing; run: conda activate cpv26")
    environment_name = environ.get("CONDA_DEFAULT_ENV", "").strip()
    if not environment_name:
        raise CondaEnvironmentError("CONDA_DEFAULT_ENV is missing; run: conda activate cpv26")
    if environment_name.lower() in {"base", "root"}:
        raise CondaEnvironmentError("Conda base/root is not allowed; run: conda activate cpv26")
    if environ.get("VIRTUAL_ENV", "").strip():
        raise CondaEnvironmentError(
            "VIRTUAL_ENV is set; deactivate the nested venv, then conda activate cpv26"
        )
    if not (3, 10) <= python_version < (3, 13):
        raise CondaEnvironmentError("CPV26 requires Python 3.10-3.12 in the Conda environment")

    prefix = Path(raw_prefix).expanduser().resolve()
    runtime_prefix = Path(python_prefix).expanduser().resolve()
    if runtime_prefix != Path(python_base_prefix).expanduser().resolve():
        raise CondaEnvironmentError("A nested venv is active; use the Conda Python directly")
    if prefix != runtime_prefix:
        raise CondaEnvironmentError(
            "CONDA_PREFIX does not match the active Python; run: conda activate cpv26"
        )
    if not conda_base_prefix or not conda_base_prefix.strip():
        raise CondaEnvironmentError(
            "Cannot verify the Conda base environment with conda info --base; "
            "restore the conda command before continuing"
        )
    if prefix == Path(conda_base_prefix).expanduser().resolve():
        raise CondaEnvironmentError("Conda base/root is not allowed; run: conda activate cpv26")
    if not (prefix / "conda-meta").is_dir():
        raise CondaEnvironmentError("CONDA_PREFIX has no conda-meta directory")
    executable = Path(python_executable).expanduser().resolve()
    if not executable.is_file() or not executable.is_relative_to(prefix):
        raise CondaEnvironmentError("Python executable is not inside the active CONDA_PREFIX")
    if not pip_available:
        raise CondaEnvironmentError("pip is missing from the active Conda environment")


def _conda_base_prefix(environ: Mapping[str, str]) -> str | None:
    """Ask Conda itself; fail closed when its base prefix cannot be established."""
    candidates = [environ.get("CONDA_EXE"), shutil.which("conda")]
    checked: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in checked:
            continue
        checked.add(candidate)
        try:
            result = subprocess.run(
                [candidate, "info", "--base"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if result.returncode == 0:
            value = result.stdout.strip()
            if value and "\n" not in value and Path(value).is_dir():
                return value
    return None


def main() -> int:
    try:
        validate_conda_environment(
            os.environ,
            python_prefix=sys.prefix,
            python_base_prefix=sys.base_prefix,
            python_executable=sys.executable,
            python_version=sys.version_info[:2],
            pip_available=importlib.util.find_spec("pip") is not None,
            conda_base_prefix=_conda_base_prefix(os.environ),
        )
    except (CondaEnvironmentError, OSError, ValueError) as exc:
        print(f"Conda environment check failed: {exc}", file=sys.stderr)
        return 1
    print(Path(sys.executable).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
