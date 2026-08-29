"""Linux-oriented command line entry points for CPV26 batch jobs."""

from __future__ import annotations

from pathlib import Path

import typer
from duckdb import Error as DuckDBError
from rich.console import Console
from rich.table import Table

from cpv26.config import Settings
from cpv26.data import SCHEMA_VERSION, DuckDBStore, SnapshotBuilder, table_names

app = typer.Typer(
    name="cpv26",
    help="Point-in-time baseball prediction infrastructure.",
    no_args_is_help=True,
)
console = Console()
error_console = Console(stderr=True)


def _settings() -> Settings:
    try:
        return Settings.from_environment()
    except ValueError as exc:
        error_console.print(f"[red]Configuration error:[/red] {exc}")
        raise typer.Exit(code=2) from exc


def _require_database(path: Path) -> None:
    if not path.is_file():
        error_console.print(f"[red]Database not found:[/red] {path}")
        error_console.print("Run `cpv26 db-init` first.")
        raise typer.Exit(code=1)


@app.command("show-config")
def show_config() -> None:
    """Print the resolved, non-secret runtime configuration."""

    settings = _settings()
    table = Table(title="CPV26 runtime configuration", show_header=False)
    table.add_column("Setting", style="cyan")
    table.add_column("Value")
    table.add_row("repository_root", str(settings.repository_root))
    table.add_row("home", str(settings.home))
    table.add_row("database_path", str(settings.database_path))
    table.add_row("timezone", settings.timezone)
    table.add_row("device", settings.device)
    table.add_row("random_seed", str(settings.random_seed))
    table.add_row("log_level", settings.log_level)
    console.print(table)


@app.command("db-init")
def db_init() -> None:
    """Create or migrate the append-only DuckDB database."""

    settings = _settings()
    settings.ensure_runtime_directories()
    with DuckDBStore(settings.database_path):
        pass
    console.print(
        f"[green]Database ready[/green]: {settings.database_path} "
        f"(schema={SCHEMA_VERSION}, tables={len(table_names(include_metadata=True))})"
    )


@app.command("db-check")
def db_check() -> None:
    """Open the database read-only and verify the complete schema."""

    settings = _settings()
    _require_database(settings.database_path)
    try:
        with DuckDBStore(settings.database_path, read_only=True) as store:
            store.assert_referential_integrity()
    except (DuckDBError, OSError, RuntimeError) as exc:
        error_console.print(f"[red]Database check failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(
        f"[green]Database schema and references are current[/green]: "
        f"version {SCHEMA_VERSION}, {len(table_names(include_metadata=True))} tables"
    )


@app.command("snapshot-build")
def snapshot_build(
    prediction_run_id: str = typer.Argument(help="Existing prediction_run identifier."),
) -> None:
    """Materialise a deterministic point-in-time Parquet snapshot."""

    settings = _settings()
    _require_database(settings.database_path)
    snapshot_root = settings.home / "snapshots"
    try:
        with DuckDBStore(settings.database_path, read_only=True) as store:
            manifest = SnapshotBuilder(store, snapshot_root).build(prediction_run_id)
    except (DuckDBError, KeyError, OSError, RuntimeError, ValueError) as exc:
        error_console.print(f"[red]Snapshot build failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"[green]Snapshot ready[/green]: {snapshot_root / prediction_run_id}")
    console.print(f"fingerprint: {manifest.fingerprint}")


if __name__ == "__main__":
    app()
