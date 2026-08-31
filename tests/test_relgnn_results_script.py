"""Saved-result inspection requires neither Torch nor installed project packages."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "show_relgnn_results.py"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _run(run: Path) -> subprocess.CompletedProcess[str]:
    # Isolated mode also proves this script does not depend on src/ or editable installation.
    return subprocess.run(
        [sys.executable, "-I", "-X", "utf8", str(SCRIPT), "--run-dir", str(run)],
        cwd=run.parent, capture_output=True, text=True, encoding="utf-8", check=False,
    )


def _test_report(**changes: Any) -> dict[str, Any]:
    return {
        "split": "test", "checkpoint_epoch": 11, "checkpoint_sha256": "best-sha",
        "dataset_fingerprint": "graph-sha", "date_start": "2026-03-28",
        "date_end": "2026-07-26", "held_out_test_season": 2026,
        "metrics": {"match": {"samples": 470, "accuracy": 0.53}, "box_pa": None},
        **changes,
    }


def _hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*") if path.is_file()
    }


def test_prints_training_history_and_latest_test_without_changing_sources(tmp_path: Path) -> None:
    run = tmp_path / "한글 run"
    training = {
        "status": "completed", "completed_epochs": 17, "best_epoch": 11,
        "best_validation_loss": 4.069, "configuration": {"batch_days": 1},
        "runtime": {"gpu_name": "A100 MIG 1g.10gb", "precision": "bf16"},
        "sampling_limits": {"training_pa_per_day": None},
        "split_summary": {"train": {"days": 3500}}, "dataset_fingerprint": "graph-sha",
        "best_checkpoint_sha256": "best-sha", "history": [{"epoch": 1}],
    }
    _write_json(run / "training_report.json", training)
    (run / "history.jsonl").write_text('{"epoch": 16}\n{"epoch": 17}\n', encoding="utf-8")
    # These intentionally are NOT valid Torch files and must never be loaded.
    (run / "best.pt").write_bytes(b"not a checkpoint")
    (run / "last.pt").write_bytes(b"do not deserialize this")
    old = run / "evaluations" / "test-20260830T010000Z-aa" / "metrics.json"
    latest = run / "evaluations" / "test-20260901T010000Z-bb" / "metrics.json"
    _write_json(old, _test_report(checkpoint_epoch=2))
    _write_json(latest, _test_report())
    os.utime(old, (2_000_000_000, 2_000_000_000))
    before = _hashes(run)
    output = _run(run)
    assert output.returncode == 0, output.stderr
    result = json.loads(output.stdout)
    assert result["training_report"]["report"] == {
        key: value for key, value in training.items() if key != "history"
    }
    assert result["history"]["last_complete_record"] == {"epoch": 17}
    assert result["history"]["complete_records"] == 2
    assert result["latest_test_evaluation"]["path"] == str(latest.resolve())
    assert result["latest_test_evaluation"]["report"] == _test_report()
    assert result["warnings"] == []
    assert _hashes(run) == before


def test_missing_run_is_clear_error_and_does_not_create_it(tmp_path: Path) -> None:
    run = tmp_path / "absent"
    output = _run(run)
    assert output.returncode == 2
    assert str(run) in output.stderr
    assert not run.exists()


def test_missing_artifacts_do_not_invent_training_or_evaluation_status(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    output = _run(run)
    assert output.returncode == 0
    result = json.loads(output.stdout)
    assert result["training_report"]["availability"] == "missing"
    assert result["history"]["availability"] == "missing"
    assert result["latest_test_evaluation"]["availability"] == "missing"
    assert "단정할 수 없습니다" in result["training_report"]["message"]
    assert list(run.iterdir()) == []


def test_incomplete_final_history_line_is_explicit_not_silently_used(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "history.jsonl").write_bytes(b'{"epoch": 1}\n{"epoch": 2')
    result = json.loads(_run(run).stdout)
    assert result["history"]["last_complete_record"] == {"epoch": 1}
    assert result["history"]["incomplete_tail_ignored"] is True
    assert result["warnings"]


@pytest.mark.parametrize("content", [b'{"epoch": 1}', b'\n{"epoch": 1}\n\n'])
def test_complete_history_record_with_or_without_final_newline(
    tmp_path: Path, content: bytes,
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "history.jsonl").write_bytes(content)
    result = json.loads(_run(run).stdout)
    assert result["history"]["last_complete_record"] == {"epoch": 1}
    assert result["history"]["incomplete_tail_ignored"] is False


@pytest.mark.parametrize("content", [b'bad\n{"epoch": 2}\n', b'{"epoch": 1}\nbad\n'])
def test_complete_corrupt_history_line_is_error(tmp_path: Path, content: bytes) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "history.jsonl").write_bytes(content)
    output = _run(run)
    assert output.returncode == 2
    assert "history.jsonl" in output.stderr
    assert not output.stdout


@pytest.mark.parametrize("content", ['{"status":', '[]', '{"loss": NaN}'])
def test_corrupt_training_report_is_error(tmp_path: Path, content: str) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "training_report.json").write_text(content, encoding="utf-8")
    output = _run(run)
    assert output.returncode == 2
    assert "training_report.json" in output.stderr


@pytest.mark.parametrize("content", ['{"metrics":', '{}', '{"split":"validation","metrics":{}}'])
def test_corrupt_latest_test_does_not_fall_back_to_older_result(
    tmp_path: Path, content: str,
) -> None:
    run = tmp_path / "run"
    _write_json(run / "evaluations" / "test-20260101" / "metrics.json", _test_report())
    latest = run / "evaluations" / "test-20260901" / "metrics.json"
    latest.parent.mkdir()
    latest.write_text(content, encoding="utf-8")
    output = _run(run)
    assert output.returncode == 2
    assert str(latest) in output.stderr
    assert not output.stdout


def test_missing_latest_metrics_directory_is_disclosed(tmp_path: Path) -> None:
    run = tmp_path / "run"
    completed = run / "evaluations" / "test-20260101" / "metrics.json"
    _write_json(completed, _test_report())
    pending = run / "evaluations" / "test-20260901"
    pending.mkdir()
    result = json.loads(_run(run).stdout)
    assert result["latest_test_evaluation"]["path"] == str(completed)
    assert result["latest_test_evaluation"]["directories_without_metrics"] == [str(pending)]
    assert result["warnings"]


def test_checkpoint_and_dataset_mismatches_are_warnings(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _write_json(run / "training_report.json", {
        "best_checkpoint_sha256": "other-sha", "dataset_fingerprint": "other-graph",
        "completed_epochs": 1,
    })
    (run / "history.jsonl").write_text('{"epoch": 2}\n', encoding="utf-8")
    _write_json(run / "evaluations" / "test-20260901" / "metrics.json", _test_report())
    result = json.loads(_run(run).stdout)
    assert any("SHA256" in message for message in result["warnings"])
    assert any("fingerprint" in message for message in result["warnings"])
    assert any("오래됐습니다" in message for message in result["warnings"])
