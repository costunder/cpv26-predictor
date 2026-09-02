"""Linux-oriented command line entry points for CPV26 batch jobs."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
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
from cpv26.data.kbo_history_ingest import import_kbo_history
from cpv26.data.kbo_history_source import (
    KBO_HISTORY_FILES,
    download_kbo_history,
    select_history_artifacts,
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


def _history_years(start_year: int, end_year: int) -> tuple[int, ...]:
    if start_year > end_year:
        raise typer.BadParameter("--start-year must not exceed --end-year")
    return tuple(range(start_year, end_year + 1))


def _write_json_report(output: Path, value: object) -> None:
    saved = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(f".{output.name}.{uuid4().hex}.part")
    try:
        with partial.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(saved)
        partial.replace(output)
    finally:
        partial.unlink(missing_ok=True)


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
    output = (report_path or settings.home / "reports" / "kbo_import.json").expanduser().resolve()
    if (
        output == settings.database_path.resolve()
        or output == directory
        or directory in output.parents
    ):
        raise typer.BadParameter("--report must not overwrite the database or source archive")
    files = tuple(directory / f"kbo_pbp_{season}.parquet" for season in selected_years)
    missing = tuple(path for path in files if not path.is_file())
    if missing:
        error_console.print("[red]KBO source file not found:[/red]")
        for path in missing:
            error_console.print(f"  {path}")
        error_console.print("Run `cpv26 kbo-fetch` first.")
        raise typer.Exit(code=1)
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


@app.command("kbo-history-fetch")
def kbo_history_fetch(
    start_year: Annotated[
        int, typer.Option("--start-year", min=2001, max=2022, help="First season, inclusive.")
    ] = 2001,
    end_year: Annotated[
        int, typer.Option("--end-year", min=2001, max=2022, help="Last season, inclusive.")
    ] = 2022,
    destination: Annotated[
        Path | None,
        typer.Option(
            "--destination", "-d",
            help="Archive directory (default: CPV26_HOME/datasets/kbo_history).",
        ),
    ] = None,
) -> None:
    """Download checksum-pinned 2001-2022 game and player box-score archives."""

    years = _history_years(start_year, end_year)
    settings = _settings()
    output = (destination or settings.home / "datasets" / "kbo_history").expanduser().resolve()
    try:
        paths = download_kbo_history(output, years=years)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        error_console.print(f"[red]KBO history download failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"[green]KBO history ready[/green]: {output} ({start_year}-{end_year})")
    for path in paths:
        console.print(f"  {path.name}")
    console.print("  SOURCE.json")


@app.command("kbo-history-import")
def kbo_history_import(
    start_year: Annotated[
        int, typer.Option("--start-year", min=2001, max=2022, help="First season, inclusive.")
    ] = 2001,
    end_year: Annotated[
        int, typer.Option("--end-year", min=2001, max=2022, help="Last season, inclusive.")
    ] = 2022,
    source_dir: Annotated[
        Path | None, typer.Option("--source-dir", help="Directory produced by kbo-history-fetch.")
    ] = None,
    report_path: Annotated[
        Path | None,
        typer.Option(
            "--report", help="Report path (default: CPV26_HOME/reports/kbo_history_import.json)."
        ),
    ] = None,
) -> None:
    """Import games, batting and pitching records with raw values and missing-field masks."""

    years = _history_years(start_year, end_year)
    settings = _settings()
    directory = (source_dir or settings.home / "datasets" / "kbo_history").expanduser().resolve()
    output = (
        report_path or settings.home / "reports" / "kbo_history_import.json"
    ).expanduser().resolve()
    if (
        output == settings.database_path.resolve()
        or output == directory
        or directory in output.parents
    ):
        raise typer.BadParameter("--report must not overwrite the database or source archive")
    missing = [
        directory / artifact.filename
        for artifact in select_history_artifacts(KBO_HISTORY_FILES, years=years)
        if not (directory / artifact.filename).is_file()
    ]
    if missing:
        error_console.print(f"[red]KBO history source file not found:[/red] {missing[0]}")
        error_console.print("Run `cpv26 kbo-history-fetch` first.")
        raise typer.Exit(code=1)
    try:
        settings.ensure_runtime_directories()
        with DuckDBStore(settings.database_path) as store:
            report = import_kbo_history(store, directory, years=years, progress=console.print)
            store.assert_referential_integrity()
            store.assert_composite_referential_integrity()
        _write_json_report(output, report)
    except (DuckDBError, OSError, RuntimeError, TypeError, ValueError) as exc:
        error_console.print(f"[red]KBO history import failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    table = Table(title="KBO historical game and player coverage")
    table.add_column("Season")
    table.add_column("Games", justify="right")
    table.add_column("Batters", justify="right")
    table.add_column("Pitchers", justify="right")
    table.add_column("Hit labels", justify="right")
    table.add_column("Verified outcomes", justify="right")
    for season in report["season_coverage"]:
        table.add_row(
            str(season["year"]), str(season["games"]),
            str(season["batter_rows"]), str(season["pitcher_rows"]),
            str(season["hit_labels"]), str(season["verified_batting_outcomes"]),
        )
    console.print(table)
    console.print(f"Historical games: {report['games']:,}; partial player records retained.")
    console.print("Unresolved names remain separate source observations, not career player IDs.")
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


@app.command("gpu-check")
def gpu_check(
    device: Annotated[
        str, typer.Option(help="Explicit CUDA device; no automatic CPU fallback.")
    ] = "cuda:0",
    amp: Annotated[str, typer.Option(help="auto, off, fp16, or bf16.")] = "auto",
) -> None:
    """Verify the actual GPU runtime with forward/backward kernels."""
    try:
        from cpv26.training.kbo_runner import check_gpu

        result = check_gpu(device, amp=amp)
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        error_console.print(f"[red]Device verification failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print_json(data=result)


@app.command("kbo-graph-build")
def kbo_graph_build(
    output: Annotated[
        Path | None,
        typer.Option(
            help="Default: datasets/kbo_graph for v5, datasets/kbo_graph_vnext for vnext."
        ),
    ] = None,
    rolling_days: Annotated[int, typer.Option(min=1, help="Past-only graph history window.")] = 90,
    start_date: Annotated[
        str, typer.Option(help="First prediction date (YYYY-MM-DD).")
    ] = "2023-01-01",
    end_date: Annotated[
        str, typer.Option(help="Last prediction date (YYYY-MM-DD).")
    ] = "2025-12-31",
    graph_schema: Annotated[
        str,
        typer.Option(
            help="Graph cache schema: v5 (legacy aggregate graph) or vnext (game-resolved v6)."
        ),
    ] = "v5",
) -> None:
    """Materialize actual KBO history into leakage-resistant day graphs."""
    settings = _settings()
    _require_database(settings.database_path)
    default_name = "kbo_graph_vnext" if graph_schema == "vnext" else "kbo_graph"
    directory = (output or settings.home / "datasets" / default_name).expanduser().resolve()
    try:
        from cpv26.data.kbo_graph_dataset import build_kbo_graph_dataset

        dataset = build_kbo_graph_dataset(
            settings.database_path,
            directory,
            rolling_days=rolling_days,
            start_day=date.fromisoformat(start_date),
            end_day=date.fromisoformat(end_date),
            graph_schema=graph_schema,
        )
    except (DuckDBError, ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        error_console.print(f"[red]KBO graph preparation failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"[green]Graph dataset ready[/green]: {directory}")
    console.print(f"Days: {len(dataset.days())}; fingerprint: {dataset.manifest['fingerprint']}")


@app.command("kbo-graph-audit")
def kbo_graph_audit(
    end_date: Annotated[
        str,
        typer.Option(
            help="Last audited date (YYYY-MM-DD); set this to validation end, before test."
        ),
    ],
    dataset: Annotated[
        Path | None, typer.Option(help="Default: CPV26_HOME/datasets/kbo_graph.")
    ] = None,
    output: Annotated[
        Path | None, typer.Option(help="Default: CPV26_HOME/reports/kbo_graph_audit.json.")
    ] = None,
    start_date: Annotated[
        str | None, typer.Option(help="Optional first audited date (YYYY-MM-DD).")
    ] = None,
) -> None:
    """Audit graph size, relation compression, isolation, and one/two-hop coverage."""

    settings = _settings()
    directory = (dataset or settings.home / "datasets" / "kbo_graph").expanduser().resolve()
    target = (
        output or settings.home / "reports" / "kbo_graph_audit.json"
    ).expanduser().resolve()
    try:
        from cpv26.data.kbo_graph_audit import audit_kbo_graph_dataset

        report = audit_kbo_graph_dataset(
            directory,
            start_day=date.fromisoformat(start_date) if start_date is not None else None,
            end_day=date.fromisoformat(end_date),
        )
        _write_json_report(target, report)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
        error_console.print(f"[red]KBO graph audit failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    summary = report["totals"]
    node_occurrences = sum(summary["node_occurrences"].values())
    edge_occurrences = summary["history_compression"]["unique_edge_occurrences"]
    console.print(f"[green]Graph audit ready[/green]: {target}")
    console.print(
        f"Days: {summary['days']}; node occurrences: {node_occurrences}; "
        f"unique edge occurrences: {edge_occurrences}"
    )


@app.command("relgnn-train")
def relgnn_train(
    dataset: Annotated[
        Path | None, typer.Option(help="Default: CPV26_HOME/datasets/kbo_graph.")
    ] = None,
    run_dir: Annotated[
        Path | None, typer.Option(help="New run directory; keep models out of Git.")
    ] = None,
    resume: Annotated[
        Path | None,
        typer.Option(
            help="Resume from this run's last.pt; repeat original season and date-order options."
        ),
    ] = None,
    device: Annotated[
        str, typer.Option(help="CUDA device; CPU is only an explicit validation mode.")
    ] = "cuda:0",
    epochs: Annotated[
        int, typer.Option(min=1, help="Total target epochs, including resumed epochs.")
    ] = 30,
    train_start_year: Annotated[
        int, typer.Option(min=1, max=9999, help="First training season, inclusive.")
    ] = 2023,
    train_end_year: Annotated[
        int, typer.Option(min=1, max=9999, help="Last training season, inclusive.")
    ] = 2023,
    validation_year: Annotated[
        int, typer.Option(min=1, max=9999, help="Validation season, after all training seasons.")
    ] = 2024,
    test_year: Annotated[
        int, typer.Option(min=1, max=9999, help="Held-out test season, after validation.")
    ] = 2025,
    chronological: Annotated[
        bool,
        typer.Option(
            help="Use date order within each epoch; not streaming predict-then-learn."
        ),
    ] = False,
    batch_days: Annotated[int, typer.Option(min=1, help="Disjoint day graphs per minibatch.")] = 2,
    hidden_dim: Annotated[int, typer.Option(min=4)] = 64,
    layers: Annotated[int, typer.Option(min=1)] = 2,
    heads: Annotated[int, typer.Option(min=1)] = 4,
    dropout: Annotated[float, typer.Option(min=0.0, max=0.99)] = 0.1,
    learning_rate: Annotated[float, typer.Option(min=1e-10)] = 0.0003,
    weight_decay: Annotated[float, typer.Option(min=0.0)] = 0.0001,
    amp: Annotated[str, typer.Option(help="auto selects GPU bf16/fp16; off uses fp32.")] = "auto",
    workers: Annotated[int, typer.Option(min=0, help="CPU graph-loading worker processes.")] = 2,
    seed: Annotated[
        int | None, typer.Option(min=0, help="Training/order seed; defaults to CPV26_RANDOM_SEED.")
    ] = None,
    route_message_normalization: Annotated[
        str, typer.Option(help="Route message normalization: none or layer_norm.")
    ] = "none",
    route_schedule: Annotated[
        str, typer.Option(help="Route schedule: full, staged, core, or node_only.")
    ] = "full",
    graph_control: Annotated[
        str, typer.Option(help="Graph control: intact or permuted_endpoints.")
    ] = "intact",
    graph_control_seed: Annotated[
        int, typer.Option(min=0, help="Epoch-independent graph-control seed.")
    ] = 2026,
    accumulate_steps: Annotated[int, typer.Option(min=1)] = 1,
    max_pa_per_day: Annotated[
        int, typer.Option(min=0, help="Training PA limit per day; 0 uses every query.")
    ] = 0,
    max_edges_per_route: Annotated[
        int, typer.Option(min=0, help="Relation edge limit per route; 0 uses every edge.")
    ] = 0,
    box_pa_weight: Annotated[
        float, typer.Option(min=0.0, help="Historical aggregate batting outcome loss weight.")
    ] = 0.2,
    box_pitch_weight: Annotated[
        float, typer.Option(min=0.0, help="Historical masked pitching-count loss weight.")
    ] = 0.1,
    selection_target: Annotated[
        str, typer.Option(help="Checkpoint criterion: auto/weighted or match log loss.")
    ] = "auto",
    box_gradient_mode: Annotated[
        str, typer.Option(help="auto/shared trains all layers; head_only isolates box decoders.")
    ] = "auto",
    patience: Annotated[
        int, typer.Option(min=0, help="Validation early stopping; 0 disables it.")
    ] = 6,
    max_days_per_split: Annotated[
        int | None, typer.Option(min=1, help="Explicit smoke test limit; not a full-season result.")
    ] = None,
) -> None:
    """Train role-aware RelGNN on selected seasons with later validation and held-out test."""
    if train_start_year > train_end_year:
        raise typer.BadParameter(
            "--train-start-year must not exceed --train-end-year.",
            param_hint="--train-start-year",
        )
    settings = _settings()
    directory = (dataset or settings.home / "datasets" / "kbo_graph").expanduser().resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = (
        (
            run_dir
            or (
                resume.expanduser().resolve().parent
                if resume
                else settings.home / "runs" / "relgnn" / f"{stamp}-{uuid4().hex[:8]}"
            )
        )
        .expanduser()
        .resolve()
    )
    try:
        from cpv26.training.kbo_runner import KBOTrainingConfig, train_kbo_relgnn

        config = KBOTrainingConfig(
            device=device,
            epochs=epochs,
            batch_days=batch_days,
            hidden_dim=hidden_dim,
            layers=layers,
            heads=heads,
            dropout=dropout,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            amp=amp,
            workers=workers,
            accumulate_steps=accumulate_steps,
            max_pa_per_day=max_pa_per_day,
            max_edges_per_route_per_day=max_edges_per_route,
            box_pa_weight=box_pa_weight,
            box_pitch_weight=box_pitch_weight,
            selection_target=selection_target,
            box_gradient_mode=box_gradient_mode,
            patience=patience,
            seed=settings.random_seed if seed is None else seed,
            max_days_per_split=max_days_per_split,
            train_seasons=tuple(range(train_start_year, train_end_year + 1)),
            validation_season=validation_year,
            test_season=test_year,
            chronological=chronological,
            route_message_normalization=route_message_normalization,
            route_schedule=route_schedule,
            graph_control=graph_control,
            graph_control_seed=graph_control_seed,
        )
        report = train_kbo_relgnn(
            directory, output, config=config, resume=resume, progress=console.print
        )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
        error_console.print(f"[red]RelGNN training failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"[green]RelGNN training finished[/green]: {output}")
    console.print(f"Epochs: {report['completed_epochs']}; best: {report['best_epoch']}")
    if report["smoke_test_only"]:
        console.print("[yellow]Limited-date verification only, not a full-season result.[/yellow]")
    console.print(
        f"{config.test_season} test was not used; "
        "evaluate best.pt explicitly with relgnn-evaluate."
    )


@app.command("relgnn-ablation-train")
def relgnn_ablation_train(
    dataset: Annotated[
        Path | None, typer.Option(help="Default: CPV26_HOME/datasets/kbo_graph.")
    ] = None,
    suite_dir: Annotated[
        Path | None,
        typer.Option(help="Matched-suite directory; reuse it to resume interrupted runs."),
    ] = None,
    device: Annotated[str, typer.Option(help="CUDA device for every matched run.")] = "cuda:0",
    epochs: Annotated[int, typer.Option(min=1, help="Equal total epoch budget per run.")] = 30,
    seed: Annotated[
        list[int] | None,
        typer.Option(
            "--seed",
            min=0,
            help="Repeat for paired training seeds; default: CPV26_RANDOM_SEED.",
        ),
    ] = None,
    graph_control_seed: Annotated[
        int,
        typer.Option(
            min=0,
            help="Fixed rewiring seed, independent of training seed and epoch.",
        ),
    ] = 2026,
    train_start_year: Annotated[
        int, typer.Option(min=1, max=9999, help="First training season, inclusive.")
    ] = 2023,
    train_end_year: Annotated[
        int, typer.Option(min=1, max=9999, help="Last training season, inclusive.")
    ] = 2023,
    validation_year: Annotated[
        int,
        typer.Option(min=1, max=9999, help="Only split used to select and compare models."),
    ] = 2024,
    test_year: Annotated[
        int,
        typer.Option(
            min=1,
            max=9999,
            help="Held-out metadata only; this command never loads or evaluates it.",
        ),
    ] = 2025,
    chronological: Annotated[
        bool, typer.Option(help="Visit training dates oldest first within every epoch.")
    ] = False,
    batch_days: Annotated[int, typer.Option(min=1)] = 2,
    hidden_dim: Annotated[int, typer.Option(min=4)] = 64,
    layers: Annotated[int, typer.Option(min=1)] = 2,
    heads: Annotated[int, typer.Option(min=1)] = 4,
    dropout: Annotated[float, typer.Option(min=0.0, max=0.99)] = 0.1,
    learning_rate: Annotated[float, typer.Option(min=1e-10)] = 0.0003,
    weight_decay: Annotated[float, typer.Option(min=0.0)] = 0.0001,
    amp: Annotated[str, typer.Option(help="auto selects GPU bf16/fp16; off uses fp32.")] = "auto",
    workers: Annotated[int, typer.Option(min=0)] = 2,
    accumulate_steps: Annotated[int, typer.Option(min=1)] = 1,
    max_pa_per_day: Annotated[
        int, typer.Option(min=0, help="Training PA limit per day; 0 uses every query.")
    ] = 0,
    max_edges_per_route: Annotated[
        int, typer.Option(min=0, help="Relation edge limit per route; 0 uses every edge.")
    ] = 0,
    box_pa_weight: Annotated[float, typer.Option(min=0.0)] = 0.2,
    box_pitch_weight: Annotated[float, typer.Option(min=0.0)] = 0.1,
    selection_target: Annotated[
        str, typer.Option(help="Checkpoint criterion: auto/weighted or match log loss.")
    ] = "auto",
    box_gradient_mode: Annotated[
        str, typer.Option(help="auto/shared trains all layers; head_only isolates box decoders.")
    ] = "auto",
    max_days_per_split: Annotated[
        int | None,
        typer.Option(min=1, help="Explicit smoke limit per train/validation split."),
    ] = None,
) -> None:
    """Retrain six graph variants from matched seeds and compare validation only."""
    if train_start_year > train_end_year:
        raise typer.BadParameter(
            "--train-start-year must not exceed --train-end-year.",
            param_hint="--train-start-year",
        )
    settings = _settings()
    selected_seeds = seed or [settings.random_seed]
    directory = (dataset or settings.home / "datasets" / "kbo_graph").expanduser().resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = (
        suite_dir
        or settings.home / "runs" / "relgnn_ablations" / f"{stamp}-{uuid4().hex[:8]}"
    ).expanduser().resolve()
    console.print(
        f"[yellow]Matched retraining starts {6 * len(selected_seeds)} runs "
        f"(6 variants x {len(selected_seeds)} seeds); this can take much longer "
        "than one run.[/yellow]"
    )
    console.print(
        f"[yellow]{test_year} test is metadata only and will not be loaded or evaluated.[/yellow]"
    )
    try:
        from cpv26.training.kbo_matched_ablation import train_matched_graph_ablations
        from cpv26.training.kbo_runner import KBOTrainingConfig

        config = KBOTrainingConfig(
            device=device,
            epochs=epochs,
            batch_days=batch_days,
            hidden_dim=hidden_dim,
            layers=layers,
            heads=heads,
            dropout=dropout,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            amp=amp,
            workers=workers,
            accumulate_steps=accumulate_steps,
            max_pa_per_day=max_pa_per_day,
            max_edges_per_route_per_day=max_edges_per_route,
            box_pa_weight=box_pa_weight,
            box_pitch_weight=box_pitch_weight,
            selection_target=selection_target,
            box_gradient_mode=box_gradient_mode,
            patience=0,
            seed=selected_seeds[0],
            max_days_per_split=max_days_per_split,
            train_seasons=tuple(range(train_start_year, train_end_year + 1)),
            validation_season=validation_year,
            test_season=test_year,
            chronological=chronological,
            graph_control_seed=graph_control_seed,
        )
        report = train_matched_graph_ablations(
            directory,
            output,
            base_config=config,
            seeds=selected_seeds,
            progress=console.print,
        )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
        error_console.print(f"[red]Matched RelGNN ablation failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    table = Table(title="Matched validation-only graph ablation")
    table.add_column("variant")
    table.add_column("selection loss mean", justify="right")
    table.add_column("population std", justify="right")
    table.add_column("paired delta vs full", justify="right")
    table.add_column("parameters", justify="right")
    for variant, values in report["aggregate"].items():
        selection = values["validation_selection_loss"]
        table.add_row(
            variant,
            f"{selection['mean']:.6f}",
            f"{selection['population_std']:.6f}",
            f"{selection['paired_delta_vs_full_mean']:+.6f}",
            f"{values['parameter_count']:,}",
        )
    console.print(table)
    console.print(f"[green]Matched ablation finished[/green]: {output}")
    console.print(f"Report: {output / 'matched_retraining_report.json'}")
    console.print(
        f"{test_year} test was not loaded or evaluated; all comparisons above use validation only."
    )


@app.command("relgnn-capacity-compare")
def relgnn_capacity_compare(
    baseline_suite: Annotated[
        Path,
        typer.Option(
            exists=True,
            file_okay=False,
            help="Existing 64x2 matched suite to reuse; it is never retrained.",
        ),
    ],
    baseline_seed: Annotated[
        int | None,
        typer.Option(
            min=0,
            help=(
                "Existing seed to reuse. Required when the matched suite declares "
                "multiple seeds."
            ),
        ),
    ] = None,
    dataset: Annotated[
        Path | None, typer.Option(help="Default: CPV26_HOME/datasets/kbo_graph.")
    ] = None,
    output: Annotated[
        Path | None, typer.Option(help="Comparison directory; reuse it to resume safely.")
    ] = None,
) -> None:
    """Select one existing 64x2 seed and train only 128x3 full/node_only."""

    from dataclasses import replace

    settings = _settings()
    directory = (dataset or settings.home / "datasets" / "kbo_graph").expanduser().resolve()
    baseline = baseline_suite.expanduser().resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    target = (
        output
        or settings.home / "runs" / "relgnn_capacity" / f"{stamp}-{uuid4().hex[:8]}"
    ).expanduser().resolve()
    try:
        from cpv26.training.kbo_capacity_comparison import (
            train_kbo_capacity_comparison,
        )
        from cpv26.training.kbo_runner import KBOTrainingConfig

        with (baseline / "matched_retraining_report.json").open(encoding="utf-8") as handle:
            baseline_report = json.load(handle)
        seeds = baseline_report.get("seeds")
        if not isinstance(seeds, list) or not seeds:
            raise ValueError("baseline suite does not declare any seeds")
        if baseline_seed is None:
            if len(seeds) != 1:
                raise ValueError(
                    "baseline suite declares multiple seeds; pass --baseline-seed"
                )
            selected_seed = seeds[0]
        else:
            selected_seed = baseline_seed
        if (
            isinstance(selected_seed, bool)
            or not isinstance(selected_seed, int)
            or selected_seed < 0
            or selected_seed not in seeds
        ):
            raise ValueError("selected baseline seed is not declared by the suite")
        raw_config = baseline_report.get("base_training_config")
        if not isinstance(raw_config, dict):
            raise ValueError("baseline suite has no base training configuration")
        base = KBOTrainingConfig.from_dict(raw_config)
        config = replace(base, seed=selected_seed, hidden_dim=128, layers=3)
        report = train_kbo_capacity_comparison(
            directory,
            baseline,
            target,
            config=config,
            baseline_seed=selected_seed,
            progress=console.print,
        )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
        error_console.print(f"[red]RelGNN capacity comparison failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    comparison = report["validation_selection_comparison"]
    console.print(f"[green]Capacity comparison ready[/green]: {target}")
    console.print(
        "node_only-full validation gap: "
        f"64x2={comparison['baseline_64x2']['node_only_minus_full']:+.6f}; "
        f"128x3={comparison['expanded_128x3']['node_only_minus_full']:+.6f}"
    )
    console.print("Held-out test was not loaded; this is a one-seed comparison.")
    if report.get("smoke_test_only"):
        console.print("[yellow]Limited-date smoke comparison, not a full-split result.[/yellow]")


@app.command("relgnn-pair-train")
def relgnn_pair_train(
    dataset: Annotated[
        Path | None, typer.Option(help="Graph dataset, including graph-vNext v6.")
    ] = None,
    output: Annotated[
        Path | None, typer.Option(help="Two-condition directory; reuse it to resume safely.")
    ] = None,
    device: Annotated[str, typer.Option()] = "cuda:0",
    epochs: Annotated[int, typer.Option(min=1)] = 30,
    train_start_year: Annotated[int, typer.Option(min=1, max=9999)] = 2023,
    train_end_year: Annotated[int, typer.Option(min=1, max=9999)] = 2023,
    validation_year: Annotated[int, typer.Option(min=1, max=9999)] = 2024,
    test_year: Annotated[int, typer.Option(min=1, max=9999)] = 2025,
    chronological: Annotated[bool, typer.Option()] = False,
    batch_days: Annotated[int, typer.Option(min=1)] = 2,
    hidden_dim: Annotated[int, typer.Option(min=4)] = 128,
    layers: Annotated[int, typer.Option(min=1)] = 3,
    heads: Annotated[int, typer.Option(min=1)] = 4,
    dropout: Annotated[float, typer.Option(min=0.0, max=0.99)] = 0.1,
    learning_rate: Annotated[float, typer.Option(min=1e-10)] = 0.0003,
    weight_decay: Annotated[float, typer.Option(min=0.0)] = 0.0001,
    amp: Annotated[str, typer.Option()] = "auto",
    workers: Annotated[int, typer.Option(min=0)] = 2,
    seed: Annotated[int | None, typer.Option(min=0)] = None,
    accumulate_steps: Annotated[int, typer.Option(min=1)] = 1,
    max_pa_per_day: Annotated[int, typer.Option(min=0)] = 0,
    max_edges_per_route: Annotated[int, typer.Option(min=0)] = 0,
    box_pa_weight: Annotated[float, typer.Option(min=0.0)] = 0.2,
    box_pitch_weight: Annotated[float, typer.Option(min=0.0)] = 0.1,
    selection_target: Annotated[str, typer.Option()] = "auto",
    box_gradient_mode: Annotated[str, typer.Option()] = "auto",
    max_days_per_split: Annotated[int | None, typer.Option(min=1)] = None,
) -> None:
    """Train exactly full and node_only once with matched initialization."""

    if train_start_year > train_end_year:
        raise typer.BadParameter("--train-start-year must not exceed --train-end-year")
    settings = _settings()
    directory = (dataset or settings.home / "datasets" / "kbo_graph_vnext").expanduser().resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    target = (
        output or settings.home / "runs" / "relgnn_pairs" / f"{stamp}-{uuid4().hex[:8]}"
    ).expanduser().resolve()
    try:
        from cpv26.training.kbo_capacity_comparison import (
            train_kbo_full_node_comparison,
        )
        from cpv26.training.kbo_runner import KBOTrainingConfig

        config = KBOTrainingConfig(
            device=device,
            epochs=epochs,
            batch_days=batch_days,
            hidden_dim=hidden_dim,
            layers=layers,
            heads=heads,
            dropout=dropout,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            amp=amp,
            workers=workers,
            accumulate_steps=accumulate_steps,
            max_pa_per_day=max_pa_per_day,
            max_edges_per_route_per_day=max_edges_per_route,
            box_pa_weight=box_pa_weight,
            box_pitch_weight=box_pitch_weight,
            selection_target=selection_target,
            box_gradient_mode=box_gradient_mode,
            patience=0,
            seed=settings.random_seed if seed is None else seed,
            max_days_per_split=max_days_per_split,
            train_seasons=tuple(range(train_start_year, train_end_year + 1)),
            validation_season=validation_year,
            test_season=test_year,
            chronological=chronological,
            route_message_normalization="none",
            route_schedule="full",
            graph_control="intact",
        )
        report = train_kbo_full_node_comparison(
            directory, target, config=config, progress=console.print
        )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
        error_console.print(f"[red]RelGNN pair comparison failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    comparison = report["validation_selection_comparison"]
    console.print(f"[green]Full/node-only comparison ready[/green]: {target}")
    console.print(
        f"node_only-full validation gap: {comparison['node_only_minus_full']:+.6f}"
    )
    console.print("Exactly two runs were trained; held-out test was not loaded.")
    if report.get("smoke_test_only"):
        console.print("[yellow]Limited-date smoke comparison, not a full-split result.[/yellow]")


@app.command("relgnn-ablation-report")
def relgnn_ablation_report(
    suite_dir: Annotated[
        Path,
        typer.Option(
            exists=True,
            file_okay=False,
            help="Completed matched-suite directory; reads saved suite/training JSON only.",
        ),
    ],
) -> None:
    """Explain validation-loss changes by task without training or evaluating test."""

    try:
        from cpv26.training.kbo_matched_ablation import analyze_matched_graph_ablations

        analysis = analyze_matched_graph_ablations(suite_dir)
    except (OSError, TypeError, ValueError, KeyError) as exc:
        error_console.print(f"[red]Matched RelGNN report failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    aggregate = analysis["aggregate"]
    contrasts = analysis["named_contrasts"]
    selection_target = str(analysis["selection_target"])

    def mean(block: object) -> str:
        if not isinstance(block, dict) or not isinstance(block.get("mean"), (int, float)):
            return "-"
        return f"{float(block['mean']):.6f}"

    def signed(block: object, key: str) -> str:
        if not isinstance(block, dict) or not isinstance(block.get(key), (int, float)):
            return "-"
        return f"{float(block[key]):+.6f}"

    console.print(
        f"Selection target: [bold]{selection_target}[/bold]; "
        f"training seeds in this saved suite: {len(analysis['seeds'])}"
    )

    selection_table = Table(title="Matched validation checkpoint selection")
    selection_table.add_column("variant")
    selection_table.add_column("selection loss", justify="right")
    selection_table.add_column("delta vs full", justify="right")
    selection_table.add_column("delta vs core", justify="right")
    selection_table.add_column("best epoch", justify="right")
    selection_table.add_column("final - best", justify="right")
    for variant, values in aggregate.items():
        selection = values["validation_selection_loss"]
        checkpoint = values.get("checkpoint_selection", {})
        selection_table.add_row(
            variant,
            mean(selection),
            signed(selection, "paired_delta_vs_full_mean"),
            signed(selection, "paired_delta_vs_core_mean"),
            mean(checkpoint.get("best_epoch")),
            mean(checkpoint.get("final_minus_best_selection_loss")),
        )
    console.print(selection_table)

    contrast_table = Table(
        title="Predefined matched contrasts (candidate - reference; loss lower is better)"
    )
    contrast_table.add_column("candidate-reference", no_wrap=True)
    contrast_table.add_column("best delta", justify="right")
    contrast_table.add_column("final delta", justify="right")
    contrast_table.add_column("last-5 delta", justify="right")
    for values in contrasts.values():
        selection = values["validation_selection_loss"]
        checkpoint = values.get("checkpoint_selection", {})
        contrast_table.add_row(
            f"{values['candidate']}-{values['reference']}",
            signed(selection, "delta_mean"),
            signed(
                checkpoint.get("final_validation_selection_loss"),
                "delta_mean",
            ),
            signed(
                checkpoint.get("last_five_validation_selection_loss_mean"),
                "delta_mean",
            ),
        )
    console.print(contrast_table)

    contribution_label = (
        "Weighted task contribution deltas"
        if selection_target == "weighted"
        else "Weighted task deltas (diagnostic only; checkpoint selection uses match)"
    )
    console.print(f"[bold]{contribution_label}[/bold] (candidate-reference; lower is better)")
    task_labels = {
        "match": "match",
        "live_hit": "hit",
        "pa": "pa",
        "run": "run",
        "box_pa": "box_pa",
        "box_pitch": "box_pitch",
    }
    for values in contrasts.values():
        deltas = values.get("weighted_contribution_deltas", {})
        pair = f"{values['candidate']}-{values['reference']}"
        console.print(
            f"{pair}: "
            + " ".join(
                f"{label}={signed(deltas.get(task), 'delta_mean')}"
                for task, label in task_labels.items()
            )
        )

    console.print(
        "[bold]Raw task-loss deltas by matched contrast[/bold] "
        "(candidate-reference; lower is better)"
    )
    for values in contrasts.values():
        deltas = values.get("validation_loss_deltas", {})
        pair = f"{values['candidate']}-{values['reference']}"
        console.print(
            f"{pair}: "
            + " ".join(
                f"{label}={signed(deltas.get(task), 'delta_mean')}"
                for task, label in task_labels.items()
            )
        )

    core_contrast = contrasts.get("core_pruning", {})
    checkpoint_tasks = core_contrast.get("checkpoint_task_deltas", {})
    if isinstance(checkpoint_tasks, dict) and checkpoint_tasks:
        console.print(
            "[bold]Core-normalized weighted task attribution by checkpoint view[/bold]"
        )
        views = {
            "best": core_contrast.get("weighted_contribution_deltas", {}),
            "final": checkpoint_tasks.get("final", {}).get(
                "weighted_contribution_deltas", {}
            ),
            "last_five": checkpoint_tasks.get("last_five", {}).get(
                "weighted_contribution_deltas", {}
            ),
        }
        for view, deltas in views.items():
            console.print(
                f"{view}: "
                + " ".join(
                    f"{label}={signed(deltas.get(task), 'delta_mean')}"
                    for task, label in task_labels.items()
                )
            )

    metric_table = Table(title="Validation prediction metrics")
    metric_table.add_column("variant")
    for label in (
        "match LL",
        "match acc",
        "PA LL",
        "PA acc",
    ):
        metric_table.add_column(label, justify="right", no_wrap=True)
    for variant, values in aggregate.items():
        metrics = values.get("validation_metrics", {})
        match = metrics.get("match", {})
        pa = metrics.get("pa", {})
        metric_table.add_row(
            variant,
            mean(match.get("log_loss")),
            mean(match.get("accuracy")),
            mean(pa.get("log_loss")),
            mean(pa.get("accuracy")),
        )
    console.print(metric_table)

    console.print("[bold]Validation LiveHit metrics[/bold]")
    for variant, values in aggregate.items():
        live_hit = values.get("validation_metrics", {}).get("live_hit", {})
        console.print(
            f"{variant}: marginal_LL={mean(live_hit.get('log_loss'))} "
            f"joint_NLL={mean(live_hit.get('joint_nll'))} "
            f"observed_NLL={mean(live_hit.get('observed_nll'))} "
            f"hits_MAE={mean(live_hit.get('expected_hits_lower_bound_mae'))} "
            f"pa_MAE={mean(live_hit.get('expected_pa_lower_bound_mae'))}"
        )
    for warning in analysis["warnings"]:
        console.print(f"[yellow]Warning:[/yellow] {warning}")
    console.print(f"Report: {analysis['report_path']}")
    console.print(
        f"{analysis['held_out_test_season']} test was not loaded or evaluated; "
        "this command only audited saved validation and training JSON."
    )


@app.command("relgnn-evaluate")
def relgnn_evaluate(
    checkpoint: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    dataset: Annotated[
        Path | None, typer.Option(help="Defaults to the checkpoint's dataset.")
    ] = None,
    split: Annotated[
        str, typer.Option(help="test, validation, or train; years come from the checkpoint.")
    ] = "test",
    device: Annotated[str, typer.Option()] = "cuda:0",
    amp: Annotated[str, typer.Option()] = "auto",
    batch_days: Annotated[int, typer.Option(min=1)] = 2,
    workers: Annotated[int, typer.Option(min=0)] = 2,
    output: Annotated[Path | None, typer.Option(help="New evaluation output directory.")] = None,
) -> None:
    """Reload a checkpoint and evaluate held-out data with per-query Parquet predictions."""
    try:
        from cpv26.training.kbo_runner import evaluate_kbo_relgnn

        report = evaluate_kbo_relgnn(
            checkpoint,
            dataset_directory=dataset,
            split=split,
            device=device,
            amp=amp,
            batch_days=batch_days,
            workers=workers,
            output_directory=output,
        )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
        error_console.print(f"[red]RelGNN evaluation failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print_json(data=report["metrics"])
    console.print(f"[green]Evaluation ready[/green]: {report['output_directory']}")
    if report["smoke_test_only"]:
        console.print("[yellow]Checkpoint came from a limited-date smoke test.[/yellow]")


@app.command("relgnn-graph-diagnose")
def relgnn_graph_diagnose(
    checkpoint: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    dataset: Annotated[
        Path | None, typer.Option(help="Defaults to the checkpoint's graph dataset.")
    ] = None,
    split: Annotated[
        str,
        typer.Option(
            help="validation (default), train, or test; use validation for redesign decisions."
        ),
    ] = "validation",
    device: Annotated[str, typer.Option()] = "cuda:0",
    amp: Annotated[str, typer.Option()] = "auto",
    batch_days: Annotated[int, typer.Option(min=1)] = 2,
    workers: Annotated[int, typer.Option(min=0)] = 2,
    seed: Annotated[
        int, typer.Option(min=0, help="Deterministic graph intervention seed.")
    ] = 2026,
    max_days: Annotated[
        int | None,
        typer.Option(min=1, help="Evenly sampled smoke limit; omit for the complete split."),
    ] = None,
    output: Annotated[Path | None, typer.Option(help="New diagnostic output directory.")] = None,
) -> None:
    """Measure whether one fixed checkpoint depends on graph routes and topology."""

    try:
        from cpv26.training.kbo_graph_diagnostic import diagnose_kbo_graph_dependence

        report = diagnose_kbo_graph_dependence(
            checkpoint,
            dataset_directory=dataset,
            split=split,
            device=device,
            amp=amp,
            batch_days=batch_days,
            workers=workers,
            seed=seed,
            max_days=max_days,
            output_directory=output,
        )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
        error_console.print(f"[red]RelGNN graph diagnostic failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    table = Table(title="Fixed-checkpoint graph dependence")
    table.add_column("Condition")
    table.add_column("Selection loss", justify="right")
    table.add_column("Delta", justify="right")
    table.add_column("Mean TV: match / hit marginal / pa", justify="right")
    for name, condition in report["conditions"].items():
        metrics = condition["metrics"]
        delta = (condition.get("metric_delta_vs_intact") or {}).get("selection_loss")
        sensitivity = condition.get("prediction_sensitivity_vs_intact") or {}
        shifts = []
        for task in ("match", "live_hit", "pa"):
            value = (sensitivity.get(task) or {}).get("mean_total_variation")
            shifts.append("-" if value is None else f"{float(value):.6f}")
        table.add_row(
            name,
            f"{float(metrics['selection_loss']):.6f}",
            "-" if delta is None else f"{float(delta):+.6f}",
            " / ".join(shifts),
        )
    console.print(table)
    report_path = Path(report["output_directory"]) / "report.json"
    console.print(f"[green]Graph dependence report ready[/green]: {report_path}")
    console.print(
        "This is a fixed-checkpoint dependence test. Proving graph benefit requires "
        "matched retraining against node-only and rewired controls."
    )
    if split == "test":
        console.print(
            "[yellow]Test was inspected; do not use this result to choose the redesign.[/yellow]"
        )
    if max_days is not None or report.get("smoke_test_only", False):
        console.print(
            "[yellow]Limited-date smoke diagnostic, not a complete-split result.[/yellow]"
        )


if __name__ == "__main__":
    app()
