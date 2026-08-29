from pathlib import Path

from pytest import MonkeyPatch
from typer.testing import CliRunner

from cpv26.cli import app
from cpv26.data import SCHEMA_VERSION


def test_database_cli_lifecycle(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    runtime = tmp_path / "runtime"
    database = runtime / "cpv26.duckdb"
    monkeypatch.setenv("CPV26_HOME", str(runtime))
    monkeypatch.setenv("CPV26_DB_PATH", str(database))

    runner = CliRunner()
    initialised = runner.invoke(app, ["db-init"])
    assert initialised.exit_code == 0, initialised.output
    assert database.is_file()
    assert "Database ready" in initialised.output
    assert f"schema={SCHEMA_VERSION}" in initialised.output

    checked = runner.invoke(app, ["db-check"])
    assert checked.exit_code == 0, checked.output
    assert "Database schema and references are current" in checked.output
    assert f"version {SCHEMA_VERSION}" in checked.output

    shown = runner.invoke(app, ["show-config"])
    assert shown.exit_code == 0, shown.output
    assert "database_path" in shown.output
    assert "Asia/Seoul" in shown.output
