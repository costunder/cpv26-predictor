"""Exercise Bash entry points without installing packages or requiring Conda/GPU."""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[1]


def _shell_env(shell: str, extra: dict[str, str] | None = None) -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("BASH_FUNC_", "CONDA_", "CPV26_", "PIP_"))
        and key
        not in {
            "BASH_ENV", "BASHOPTS", "SHELLOPTS", "ENV", "VIRTUAL_ENV", "PYTHONHOME", "PYTHONPATH"
        }
    }
    environment["PATH"] = str(Path(shell).parent) + os.pathsep + environment.get("PATH", "")
    if extra:
        environment.update(extra)
    return environment


@pytest.fixture(scope="module")
def bash_executable() -> str:
    candidates = (
        os.environ.get("CPV26_TEST_BASH"),
        shutil.which("bash"),
        shutil.which("sh"),
    )
    for candidate in candidates:
        if not candidate:
            continue
        try:
            probe = subprocess.run(
                [candidate, "--noprofile", "--norc", "-c", 'printf "%s" "$BASH_VERSION"'],
                env=_shell_env(candidate),
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if probe.returncode == 0 and probe.stdout.strip():
            return candidate
    pytest.skip("GNU Bash unavailable; set CPV26_TEST_BASH to its executable")


@pytest.fixture
def shell_project(tmp_path: Path) -> Path:
    project = tmp_path / "project with spaces"
    (project / "scripts").mkdir(parents=True)
    for name in ("activate.sh", "setup.sh", "check.sh", "conda_guard.sh", "conda_guard.py"):
        shutil.copyfile(REPOSITORY / "scripts" / name, project / "scripts" / name)
    (project / "requirements").mkdir()
    (project / "requirements" / "constraints.txt").write_text("", encoding="utf-8")
    (project / "pyproject.toml").write_text("", encoding="utf-8")
    return project


@pytest.fixture
def loader_project(shell_project: Path) -> Path:
    # Boundary stub: these tests isolate configuration loading from Conda discovery.
    (shell_project / "scripts" / "conda_guard.sh").write_text(
        'cpv26_require_conda() { cpv26_python="$CPV26_TEST_PYTHON"; }\n',
        encoding="utf-8",
    )
    return shell_project


def _run(
    shell: str, project: Path, command: str, extra: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    environment = {
        "CPV26_TEST_PROJECT": project.as_posix(),
        "CPV26_TEST_PYTHON": Path(sys.executable).as_posix(),
        "CONDA_DEFAULT_ENV": "cpv26",
        **(extra or {}),
    }
    # GNU Bash named sh starts in POSIX mode; the project scripts require Bash mode.
    command = "set +o posix\n" + command
    # Files avoid Windows pipe-reader threads when Bash launches native Python.
    with (
        tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace") as stdout,
        tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace") as stderr,
    ):
        result = subprocess.run(
            [shell, "--noprofile", "--norc", "-c", command],
            cwd=project.parent,
            env=_shell_env(shell, environment),
            stdout=stdout,
            stderr=stderr,
            timeout=30,
            check=False,
        )
        stdout.seek(0)
        stderr.seek(0)
        return subprocess.CompletedProcess(
            result.args, result.returncode, stdout.read(), stderr.read()
        )


def test_activation_loads_literal_values_without_evaluation(
    bash_executable: str, loader_project: Path
) -> None:
    (loader_project / ".env").write_bytes(
        b"# configuration\r\n\r\nCPV26_HOME=./var\r\n"
        b"CPV26_TEST_LITERAL=$(touch evaluated)\r\n"
        b"CPV26_TEST_BACKTICK=`touch evaluated-backtick`\r\n"
        b"CPV26_TEST_SPACES=a b # literal\r\nCPV26_TEST_EQUALS=a=b"
    )
    result = _run(
        bash_executable,
        loader_project,
        """
source "$CPV26_TEST_PROJECT/scripts/activate.sh" || exit 31
[[ "$PWD" == "$(cd "$CPV26_TEST_PROJECT" && pwd)" ]] || exit 32
printf 'literal:%s\n' "$CPV26_TEST_LITERAL" "$CPV26_TEST_BACKTICK"
printf 'spaces:%s\nequals:%s\nhome:%s\n' \
  "$CPV26_TEST_SPACES" "$CPV26_TEST_EQUALS" "$CPV26_HOME"
""",
    )
    assert result.returncode == 0, result.stderr
    assert "literal:$(touch evaluated)" in result.stdout
    assert "literal:`touch evaluated-backtick`" in result.stdout
    assert "spaces:a b # literal\nequals:a=b\nhome:./var" in result.stdout
    assert not (loader_project / "evaluated").exists()
    assert not (loader_project / "evaluated-backtick").exists()
    assert not (loader_project / ".venv").exists()


@pytest.mark.parametrize(
    "invalid_entry", ["PATH=/wrong", "CONDA_PREFIX=/wrong", "VIRTUAL_ENV=/wrong", "invalid"]
)
def test_activation_rejects_late_invalid_key_without_partial_changes(
    bash_executable: str, loader_project: Path, invalid_entry: str
) -> None:
    (loader_project / ".env").write_text(
        f"CPV26_HOME=changed\n{invalid_entry}\n", encoding="utf-8"
    )
    result = _run(
        bash_executable,
        loader_project,
        """
original_directory=$PWD
original_path=$PATH
source "$CPV26_TEST_PROJECT/scripts/activate.sh"
status=$?
[[ $status -ne 0 ]] || exit 31
[[ "$CPV26_HOME" == unchanged && "$PWD" == "$original_directory" ]] || exit 32
[[ "$PATH" == "$original_path" && -z "${VIRTUAL_ENV:-}" ]] || exit 33
[[ -z "${CONDA_PREFIX:-}" && $- != *e* ]] || exit 34
printf 'shell survived\n'
""",
        {"CPV26_HOME": "unchanged"},
    )
    assert result.returncode == 0, result.stderr
    assert "only CPV26_" in result.stderr
    assert "shell survived" in result.stdout


@pytest.mark.parametrize("failure", ["missing_env", "guard_failure"])
def test_activation_failure_returns_to_the_calling_shell(
    bash_executable: str, loader_project: Path, failure: str
) -> None:
    if failure == "guard_failure":
        (loader_project / ".env").write_text("CPV26_HOME=changed\n", encoding="utf-8")
        (loader_project / "scripts" / "conda_guard.sh").write_text(
            'cpv26_require_conda() { echo "invalid Conda" >&2; return 1; }\n',
            encoding="utf-8",
        )
    result = _run(
        bash_executable,
        loader_project,
        """
original_directory=$PWD
source "$CPV26_TEST_PROJECT/scripts/activate.sh"
status=$?
[[ $status -ne 0 && $- != *e* ]] || exit 31
[[ "$CPV26_HOME" == unchanged && "$PWD" == "$original_directory" ]] || exit 32
printf 'shell survived\n'
""",
        {"CPV26_HOME": "unchanged"},
    )
    assert result.returncode == 0, result.stderr
    assert ("Missing .env" if failure == "missing_env" else "invalid Conda") in result.stderr
    assert "shell survived" in result.stdout


def test_activation_reports_readonly_export_failure(
    bash_executable: str, loader_project: Path
) -> None:
    (loader_project / ".env").write_text("CPV26_HOME=changed\n", encoding="utf-8")
    result = _run(
        bash_executable,
        loader_project,
        """
readonly CPV26_HOME=unchanged
source "$CPV26_TEST_PROJECT/scripts/activate.sh"
status=$?
[[ $status -ne 0 && "$CPV26_HOME" == unchanged ]] || exit 31
printf 'shell survived\n'
""",
    )
    assert result.returncode == 0, result.stderr
    assert "Could not export" in result.stderr
    assert "configuration loaded" not in result.stdout
    assert "shell survived" in result.stdout


def test_activation_reports_unreadable_configuration(
    bash_executable: str, loader_project: Path
) -> None:
    configuration = loader_project / ".env"
    configuration.write_text("CPV26_HOME=changed\n", encoding="utf-8")
    original_mode = stat.S_IMODE(configuration.stat().st_mode)
    try:
        configuration.chmod(0)
        if os.access(configuration, os.R_OK):
            pytest.skip("Current privileges/filesystem cannot make the fixture unreadable")
        result = _run(
            bash_executable,
            loader_project,
            """
source "$CPV26_TEST_PROJECT/scripts/activate.sh"
status=$?
[[ $status -ne 0 && "$CPV26_HOME" == unchanged ]] || exit 31
printf 'shell survived\n'
""",
            {"CPV26_HOME": "unchanged"},
        )
        assert result.returncode == 0, result.stderr
        assert "Cannot read .env" in result.stderr
        assert "configuration loaded" not in result.stdout
        assert "shell survived" in result.stdout
    finally:
        configuration.chmod(original_mode)


@pytest.mark.parametrize("script", ["setup.sh", "check.sh"])
def test_setup_and_check_refuse_sourcing_without_changing_shell_options(
    bash_executable: str, shell_project: Path, script: str
) -> None:
    result = _run(
        bash_executable,
        shell_project,
        """
original_options=$-
source "$CPV26_TEST_PROJECT/scripts/$CPV26_TEST_SCRIPT"
status=$?
[[ $status -ne 0 && "$-" == "$original_options" ]] || exit 31
printf 'shell survived\n'
""",
        {"CPV26_TEST_SCRIPT": script},
    )
    assert result.returncode == 0, result.stderr
    assert f"Run this script with: bash scripts/{script}" in result.stderr
    assert "shell survived" in result.stdout
    assert not (shell_project / ".venv").exists()


@pytest.mark.parametrize("script", ["setup.sh", "check.sh"])
@pytest.mark.parametrize("invalid_environment", ["missing", "base", "nested"])
def test_invalid_conda_stops_before_package_commands(
    bash_executable: str, shell_project: Path, script: str, invalid_environment: str
) -> None:
    calls = shell_project / "python-calls.log"
    guard_output = shell_project / "guard-output.log"
    environment = {
        "CPV26_TEST_SCRIPT": script,
        "CPV26_TEST_CALLS": calls.as_posix(),
        "CPV26_TEST_GUARD_OUTPUT": guard_output.as_posix(),
    }
    expected = "CONDA_PREFIX is missing"
    if invalid_environment != "missing":
        environment["CONDA_PREFIX"] = sys.prefix
        if invalid_environment == "base":
            environment["CONDA_DEFAULT_ENV"] = "base"
            expected = "base/root is not allowed"
        else:
            environment["VIRTUAL_ENV"] = str(shell_project / "nested")
            expected = "VIRTUAL_ENV is set"
    result = _run(
        bash_executable,
        shell_project,
        """
python() {
  printf '%s\n' "$*" >> "$CPV26_TEST_CALLS"
  [[ "$1" == */conda_guard.py ]] || return 97
  local status=0
  "$CPV26_TEST_PYTHON" "$@" > "$CPV26_TEST_GUARD_OUTPUT" || status=$?
  # Never expose an install-capable interpreter, even if a guard regression occurs.
  if [[ $status -eq 0 ]]; then
    printf 'unexpected guard success\n' >> "$CPV26_TEST_CALLS"
    return 98
  fi
  return "$status"
}
export -f python
cd "$CPV26_TEST_PROJECT" || exit 31
"$BASH" "scripts/$CPV26_TEST_SCRIPT"
""",
        environment,
    )
    assert result.returncode != 0
    assert expected in result.stderr
    recorded_calls = calls.read_text(encoding="utf-8").splitlines()
    assert len(recorded_calls) == 1 and recorded_calls[0].endswith("/conda_guard.py")
    assert guard_output.read_text(encoding="utf-8") == ""
    assert not (shell_project / ".venv").exists()
