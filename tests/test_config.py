from __future__ import annotations

from pathlib import Path

import pytest

from cpv26.config import Settings


def test_settings_resolve_linux_style_relative_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CPV26_HOME", "runtime")
    monkeypatch.setenv("CPV26_DB_PATH", "runtime/predictor.duckdb")
    monkeypatch.setenv("CPV26_TIMEZONE", "Asia/Seoul")
    monkeypatch.setenv("CPV26_DEVICE", "cpu")
    monkeypatch.setenv("CPV26_RANDOM_SEED", "17")

    settings = Settings.from_environment(tmp_path)

    assert settings.home == (tmp_path / "runtime").resolve()
    assert settings.database_path == (tmp_path / "runtime/predictor.duckdb").resolve()
    assert settings.random_seed == 17
    assert settings.device == "cpu"


def test_settings_reject_unknown_device(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CPV26_DEVICE", "tpu")

    with pytest.raises(ValueError, match="CPV26_DEVICE"):
        Settings.from_environment(tmp_path)
