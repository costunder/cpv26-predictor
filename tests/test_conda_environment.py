from dataclasses import dataclass, replace
from pathlib import Path

import pytest
from scripts.build_code_summary import LANGUAGE_BY_SUFFIX, _included_files
from scripts.conda_guard import CondaEnvironmentError, validate_conda_environment


@dataclass(frozen=True)
class CondaRuntime:
    environ: dict[str, str]
    python_prefix: str
    python_base_prefix: str
    python_executable: str
    python_version: tuple[int, int] = (3, 12)
    pip_available: bool = True
    conda_base_prefix: str | None = None

    def validate(self) -> None:
        validate_conda_environment(
            self.environ,
            python_prefix=self.python_prefix,
            python_base_prefix=self.python_base_prefix,
            python_executable=self.python_executable,
            python_version=self.python_version,
            pip_available=self.pip_available,
            conda_base_prefix=self.conda_base_prefix,
        )


@pytest.fixture
def conda_runtime(tmp_path: Path) -> CondaRuntime:
    base = tmp_path / "miniforge"
    prefix = base / "envs" / "cpv26"
    (prefix / "conda-meta").mkdir(parents=True)
    executable = prefix / "bin" / "python"
    executable.parent.mkdir()
    executable.touch()
    return CondaRuntime(
        environ={"CONDA_PREFIX": str(prefix), "CONDA_DEFAULT_ENV": "cpv26"},
        python_prefix=str(prefix),
        python_base_prefix=str(prefix),
        python_executable=str(executable),
        conda_base_prefix=str(base),
    )


@pytest.mark.parametrize("version", [(3, 10), (3, 11), (3, 12)])
@pytest.mark.parametrize("name", ["cpv26", "cpv26-cpu", "/custom/project-env"])
def test_accepts_dedicated_conda(
    conda_runtime: CondaRuntime, version: tuple[int, int], name: str
) -> None:
    replace(
        conda_runtime,
        environ={**conda_runtime.environ, "CONDA_DEFAULT_ENV": name},
        python_version=version,
    ).validate()


@pytest.mark.parametrize("key", ["CONDA_PREFIX", "CONDA_DEFAULT_ENV"])
@pytest.mark.parametrize("value", [None, "", "   "])
def test_requires_conda_activation(
    conda_runtime: CondaRuntime, key: str, value: str | None
) -> None:
    environ = dict(conda_runtime.environ)
    if value is None:
        del environ[key]
    else:
        environ[key] = value
    with pytest.raises(CondaEnvironmentError, match=key):
        replace(conda_runtime, environ=environ).validate()


@pytest.mark.parametrize("name", ["base", "root", "BASE", " root "])
def test_rejects_base_environment_name(conda_runtime: CondaRuntime, name: str) -> None:
    with pytest.raises(CondaEnvironmentError, match="base/root"):
        replace(
            conda_runtime, environ={**conda_runtime.environ, "CONDA_DEFAULT_ENV": name}
        ).validate()


def test_rejects_actual_base_even_with_project_name(conda_runtime: CondaRuntime) -> None:
    with pytest.raises(CondaEnvironmentError, match="base/root"):
        replace(conda_runtime, conda_base_prefix=conda_runtime.python_prefix).validate()


@pytest.mark.parametrize("base", [None, "", "   "])
def test_requires_independent_base_verification(
    conda_runtime: CondaRuntime, base: str | None
) -> None:
    with pytest.raises(CondaEnvironmentError, match="Cannot verify the Conda base"):
        replace(conda_runtime, conda_base_prefix=base).validate()


def test_rejects_virtual_env_variable(conda_runtime: CondaRuntime) -> None:
    with pytest.raises(CondaEnvironmentError, match="VIRTUAL_ENV"):
        replace(
            conda_runtime, environ={**conda_runtime.environ, "VIRTUAL_ENV": "/legacy/.venv"}
        ).validate()


def test_rejects_nested_venv_without_variable(conda_runtime: CondaRuntime) -> None:
    with pytest.raises(CondaEnvironmentError, match="nested venv"):
        # The inherited Conda variables alone must not make a venv valid.
        replace(conda_runtime, python_base_prefix=conda_runtime.conda_base_prefix or "").validate()


def test_rejects_python_from_another_environment(conda_runtime: CondaRuntime) -> None:
    with pytest.raises(CondaEnvironmentError, match="CONDA_PREFIX does not match"):
        replace(
            conda_runtime,
            python_prefix="/other/environment",
            python_base_prefix="/other/environment",
        ).validate()


@pytest.mark.parametrize("version", [(3, 9), (3, 13), (4, 0)])
def test_rejects_unsupported_python(
    conda_runtime: CondaRuntime, version: tuple[int, int]
) -> None:
    with pytest.raises(CondaEnvironmentError, match="Python 3.10-3.12"):
        replace(conda_runtime, python_version=version).validate()


def test_requires_conda_metadata(conda_runtime: CondaRuntime) -> None:
    (Path(conda_runtime.python_prefix) / "conda-meta").rmdir()
    with pytest.raises(CondaEnvironmentError, match="conda-meta"):
        conda_runtime.validate()


def test_requires_pip(conda_runtime: CondaRuntime) -> None:
    with pytest.raises(CondaEnvironmentError, match="pip"):
        replace(conda_runtime, pip_available=False).validate()


def test_rejects_missing_python_executable(conda_runtime: CondaRuntime) -> None:
    with pytest.raises(CondaEnvironmentError, match="Python executable"):
        replace(
            conda_runtime, python_executable=str(Path(conda_runtime.python_prefix) / "missing")
        ).validate()


def test_rejects_sibling_with_same_path_prefix(conda_runtime: CondaRuntime) -> None:
    sibling = Path(conda_runtime.python_prefix + "-other")
    sibling.mkdir()
    executable = sibling / "python"
    executable.touch()
    with pytest.raises(CondaEnvironmentError, match="Python executable"):
        replace(conda_runtime, python_executable=str(executable)).validate()


def test_compares_normalized_paths(conda_runtime: CondaRuntime) -> None:
    alternate = str(Path(conda_runtime.python_prefix) / ".." / "cpv26")
    replace(
        conda_runtime, environ={**conda_runtime.environ, "CONDA_PREFIX": alternate}
    ).validate()


def test_conda_manifest_is_in_source_handoff() -> None:
    paths = _included_files()
    manifest = Path(__file__).resolve().parents[1] / "environment.yml"
    assert manifest in paths
    contents = manifest.read_text(encoding="utf-8")
    assert "name: cpv26" in contents
    assert "  - python=3.12" in contents
    assert "  - pip" in contents
    assert LANGUAGE_BY_SUFFIX[manifest.suffix] == "yaml"
