"""Linux-oriented command line entry points for CPV26 batch jobs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated
from uuid import uuid4

import typer
from duckdb import Error as DuckDBError
from rich.console import Console
from rich.table import Table

from cpv26.config import Settings
from cpv26.data import (
    KBO_PLAYBYPLAY_REVISION,
    SCHEMA_VERSION,
    DuckDBStore,
    SnapshotBuilder,
    download_kbo_playbyplay,
    import_kbo_playbyplay,
    table_names,
    write_import_report,
)
from cpv26.data.kbo_playbyplay import sha256_file
from cpv26.pipelines import (
    LIVE_HIT_CANONICAL_SQL,
    evaluate_fixed_season_catboost_json,
    evaluate_live_hit_fixed_season_catboost_json,
)
from cpv26.pipelines.kbo_match_baseline import MATCH_CANONICAL_SQL

app = typer.Typer(
    name="cpv26",
    help="Point-in-time baseball prediction infrastructure.",
    no_args_is_help=True,
)
console = Console()
error_console = Console(stderr=True)
COMPLETE_KBO_SEASONS = (2023, 2024, 2025)


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


def _kbo_dataset_directory(settings: Settings) -> Path:
    return settings.home / "datasets" / "kbo_playbyplay" / "v0"


def _model_run_directory(settings: Settings, task: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return (settings.home / "models" / task / f"{stamp}-{uuid4().hex[:8]}").resolve()


def _write_evaluation_outputs(output: Path, model_directory: Path, payload: str) -> str:
    """Keep an immutable report beside each run's models and update the report pointer."""

    result = json.loads(payload)
    result["artifact_run_id"] = model_directory.name
    result["model_directory"] = str(model_directory)
    for fold in result["folds"]:
        if fold.get("model_path"):
            fold["model_sha256"] = sha256_file(Path(fold["model_path"]))
    saved = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    model_directory.mkdir(parents=True, exist_ok=True)
    with (model_directory / "evaluation.json").open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(saved)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(f".{output.name}.{model_directory.name}.part")
    try:
        partial.write_text(saved, encoding="utf-8", newline="\n")
        partial.replace(output)
    finally:
        partial.unlink(missing_ok=True)
    return saved


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
            store.assert_composite_referential_integrity()
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


@app.command("kbo-fetch")
def kbo_fetch(
    destination: Annotated[
        Path | None,
        typer.Option(
            "--destination",
            "-d",
            help="Dataset directory (default: CPV26_HOME/datasets/kbo_playbyplay/v0).",
        ),
    ] = None,
    year: Annotated[
        list[int] | None,
        typer.Option(
            "--year",
            help="Season to download; repeat for multiple seasons (default: 2023-2025).",
        ),
    ] = None,
) -> None:
    """Download the pinned public KBO Parquet files and verify SHA-256."""

    settings = _settings()
    selected_years = tuple(year) if year else COMPLETE_KBO_SEASONS
    output = (destination or _kbo_dataset_directory(settings)).expanduser().resolve()
    try:
        paths = download_kbo_playbyplay(output, years=selected_years)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        error_console.print(f"[red]KBO dataset download failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(
        f"[green]KBO dataset ready[/green]: {output} (revision={KBO_PLAYBYPLAY_REVISION})"
    )
    for path in paths:
        console.print(f"  {path.name}")
    console.print("  SOURCE.json")


@app.command("kbo-import")
def kbo_import(
    source_dir: Annotated[
        Path | None,
        typer.Option("--source-dir", help="Directory produced by kbo-fetch."),
    ] = None,
    year: Annotated[
        list[int] | None,
        typer.Option(
            "--year",
            help="Season to import; repeat for multiple seasons (default: 2023-2025).",
        ),
    ] = None,
    report_path: Annotated[
        Path | None,
        typer.Option(
            "--report",
            help="Import report path (default: CPV26_HOME/reports/kbo_import.json).",
        ),
    ] = None,
) -> None:
    """Convert pitch rows into canonical games, teams, players, and plate appearances."""

    settings = _settings()
    settings.ensure_runtime_directories()
    selected_years = tuple(year) if year else COMPLETE_KBO_SEASONS
    directory = (source_dir or _kbo_dataset_directory(settings)).expanduser().resolve()
    files = tuple(directory / f"kbo_pbp_{season}.parquet" for season in selected_years)
    missing = tuple(path for path in files if not path.is_file())
    if missing:
        error_console.print("[red]KBO source file not found:[/red]")
        for path in missing:
            error_console.print(f"  {path}")
        error_console.print("Run `cpv26 kbo-fetch` first.")
        raise typer.Exit(code=1)
    output = (report_path or settings.home / "reports" / "kbo_import.json").expanduser().resolve()
    try:
        with DuckDBStore(settings.database_path) as store:
            report = import_kbo_playbyplay(store, files)
            store.assert_referential_integrity()
            store.assert_composite_referential_integrity()
        write_import_report(report, output)
    except (DuckDBError, OSError, RuntimeError, TypeError, ValueError) as exc:
        error_console.print(f"[red]KBO import failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    table = Table(title="KBO canonical import")
    table.add_column("Table")
    table.add_column("Inserted", justify="right")
    table.add_column("Total", justify="right")
    for name, total in report.total_rows.items():
        table.add_row(name, str(report.inserted_rows[name]), str(total))
    console.print(table)
    console.print(
        "completed PA: "
        f"{report.completed_plate_appearances:,}; unlabelled: "
        f"{report.unlabelled_plate_appearances:,}"
    )
    console.print(
        "score audit: "
        f"{report.invalid_score_transitions} inconsistent PA transitions; "
        f"{report.unreconciled_score_games} unreconciled games; "
        f"{report.source_unallocated_runs} source-unallocated runs"
    )
    console.print(f"[green]Import report ready[/green]: {output}")


@app.command("kbo-match-evaluate")
def kbo_match_evaluate(
    report_path: Annotated[
        Path | None,
        typer.Option(
            "--report",
            help="Evaluation JSON path (default: CPV26_HOME/reports/kbo_match_baseline.json).",
        ),
    ] = None,
    iterations: Annotated[
        int,
        typer.Option(min=1, help="CatBoost tree count per fold."),
    ] = 400,
) -> None:
    """Train and evaluate the distinct three-class match-result baseline."""

    settings = _settings()
    _require_database(settings.database_path)
    output = (
        (report_path or settings.home / "reports" / "kbo_match_baseline.json")
        .expanduser()
        .resolve()
    )
    model_directory = _model_run_directory(settings, "kbo_match_baseline")
    try:
        with DuckDBStore(settings.database_path, read_only=True) as store:
            rows = store.connection.execute(MATCH_CANONICAL_SQL).fetchall()
        payload = evaluate_fixed_season_catboost_json(
            rows,
            model_output_directory=model_directory,
            catboost_parameters={
                "iterations": iterations,
                "random_seed": settings.random_seed,
            },
        )
        payload = _write_evaluation_outputs(output, model_directory, payload)
    except (DuckDBError, OSError, RuntimeError, TypeError, ValueError) as exc:
        error_console.print(f"[red]KBO match evaluation failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    result = json.loads(payload)
    table = Table(title="KBO match baseline (home L/D/W)")
    table.add_column("Fold", no_wrap=True)
    table.add_column("Train", justify="right")
    table.add_column("Eval", justify="right")
    table.add_column("LL", justify="right")
    table.add_column("Prior LL", justify="right")
    table.add_column("Accuracy", justify="right")
    for fold in result["folds"]:
        table.add_row(
            fold["name"],
            str(fold["train_games"]),
            str(fold["evaluation_games"]),
            f"{fold['metrics']['log_loss']:.4f}",
            f"{fold['prior_baseline']['metrics']['log_loss']:.4f}",
            f"{fold['metrics']['accuracy']:.4f}",
        )
    console.print(table)
    console.print(f"[green]Evaluation report ready[/green]: {output}")
    console.print(f"Models and archived report: {model_directory}")


@app.command("kbo-live-hit-evaluate")
def kbo_live_hit_evaluate(
    report_path: Annotated[
        Path | None,
        typer.Option(
            "--report",
            help="JSON path (default: CPV26_HOME/reports/kbo_live_hit_baseline.json).",
        ),
    ] = None,
    iterations: Annotated[
        int,
        typer.Option(min=1, help="CatBoost tree count per fold."),
    ] = 400,
) -> None:
    """Evaluate player-game any-hit probability conditional on observed appearance."""

    settings = _settings()
    _require_database(settings.database_path)
    output = (
        (report_path or settings.home / "reports" / "kbo_live_hit_baseline.json")
        .expanduser()
        .resolve()
    )
    model_directory = _model_run_directory(settings, "kbo_live_hit_baseline")
    try:
        with DuckDBStore(settings.database_path, read_only=True) as store:
            rows = store.connection.execute(LIVE_HIT_CANONICAL_SQL).fetchall()
        complete_season_rows = [row for row in rows if 2023 <= row[2].year <= 2025]
        payload = evaluate_live_hit_fixed_season_catboost_json(
            complete_season_rows,
            model_output_directory=model_directory,
            catboost_parameters={
                "iterations": iterations,
                "random_seed": settings.random_seed,
            },
        )
        payload = _write_evaluation_outputs(output, model_directory, payload)
    except (DuckDBError, OSError, RuntimeError, TypeError, ValueError) as exc:
        error_console.print(f"[red]KBO Live Hit evaluation failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    result = json.loads(payload)
    table = Table(title="KBO Live Hit baseline (any hit | appeared)")
    table.add_column("Fold", no_wrap=True)
    table.add_column("Train", justify="right")
    table.add_column("Eval", justify="right")
    table.add_column("LL", justify="right")
    table.add_column("Prior LL", justify="right")
    table.add_column("Accuracy", justify="right")
    for fold in result["folds"]:
        table.add_row(
            fold["name"],
            str(fold["train_player_games"]),
            str(fold["evaluation_player_games"]),
            f"{fold['metrics']['log_loss']:.4f}",
            f"{fold['prior_baseline']['metrics']['log_loss']:.4f}",
            f"{fold['metrics']['accuracy']:.4f}",
        )
    console.print(table)
    console.print(
        "This experiment is conditional on PA >= 1. "
        "Candidate selection and appearance probability are not estimated here."
    )
    console.print(f"[green]Evaluation report ready[/green]: {output}")
    console.print(f"Models and archived report: {model_directory}")


if __name__ == "__main__":
    app()
