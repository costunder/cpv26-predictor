"""Print saved RelGNN results using only the standard library; never run a model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


class ResultsError(ValueError):
    """A saved result cannot be read without hiding an error."""


def _reject_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant: {value}")


def _json_object(content: str) -> dict[str, Any]:
    value = json.loads(content, parse_constant=_reject_constant)
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object")
    return value


def _read_object(path: Path) -> dict[str, Any]:
    try:
        return _json_object(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ResultsError(f"결과 JSON을 읽을 수 없습니다: {path}: {exc}") from exc


def _history(path: Path, warnings: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path), "availability": "missing"}
    if not path.exists():
        result["message"] = "history.jsonl이 없습니다. 실행 상태는 판단할 수 없습니다."
        return result
    try:
        lines = path.read_bytes().splitlines(keepends=True)
    except OSError as exc:
        raise ResultsError(f"학습 기록을 읽을 수 없습니다: {path}: {exc}") from exc
    count = 0
    last: dict[str, Any] | None = None
    last_line: int | None = None
    incomplete_tail = False
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = _json_object(line.decode("utf-8"))
        except (UnicodeError, ValueError) as exc:
            if number == len(lines) and not line.endswith((b"\n", b"\r")):
                incomplete_tail = True
                warnings.append(
                    f"{path}:{number}: 마지막 줄이 불완전하거나 손상됐습니다. "
                    "그 줄은 제외하고 마지막 완전한 기록을 표시합니다."
                )
                continue
            raise ResultsError(f"학습 기록 JSON 오류: {path}:{number}: {exc}") from exc
        count += 1
        last, last_line = record, number
    result.update(
        availability="available" if last is not None else "no_complete_records",
        complete_records=count,
        last_complete_record_line=last_line,
        last_complete_record=last,
        incomplete_tail_ignored=incomplete_tail,
    )
    return result


def read_results(run_directory: str | Path) -> dict[str, Any]:
    """Read JSON artifacts only, preserving their values and exposing missing evidence."""
    run = Path(run_directory).expanduser().resolve()
    if not run.is_dir():
        raise ResultsError(f"학습 run 디렉터리가 없습니다: {run}")
    warnings: list[str] = []
    result: dict[str, Any] = {
        "run_directory": str(run),
        "read_only": True,
        "status_note": "저장된 기록만 표시합니다. 현재 학습 프로세스 상태는 조회하지 않습니다.",
        "warnings": warnings,
    }
    report_path = run / "training_report.json"
    report: dict[str, Any] | None = None
    training: dict[str, Any] = {"path": str(report_path), "availability": "missing"}
    if report_path.exists():
        report = _read_object(report_path)
        training.update(
            availability="available",
            report={key: value for key, value in report.items() if key != "history"},
        )
    else:
        training["message"] = (
            "완료된 학습 보고서가 없습니다. 미실행·실행 중·중단 여부는 단정할 수 없습니다."
        )
    result["training_report"] = training
    result["history"] = _history(run / "history.jsonl", warnings)
    last = result["history"].get("last_complete_record")
    if report and isinstance(last, dict):
        completed, recorded = report.get("completed_epochs"), last.get("epoch")
        if isinstance(completed, int) and isinstance(recorded, int) and recorded > completed:
            warnings.append(
                "history.jsonl에 training_report.json보다 뒤의 epoch가 있습니다. "
                "학습 보고서가 마지막 기록보다 오래됐습니다."
            )

    evaluation_root = run / "evaluations"
    directories = sorted(
        (path for path in evaluation_root.glob("test-*") if path.is_dir()),
        key=lambda path: path.name,
    )
    candidates = [path / "metrics.json" for path in directories if (path / "metrics.json").exists()]
    missing = [str(path) for path in directories if not (path / "metrics.json").exists()]
    evaluation: dict[str, Any] = {
        "availability": "missing",
        "selection": "latest test-* directory name with metrics.json; not file modification time",
        "directories_without_metrics": missing,
    }
    if missing:
        warnings.append(
            "metrics.json이 없는 test-* 폴더가 있습니다. "
            "해당 평가의 완료 여부는 확인할 수 없습니다."
        )
    if candidates:
        path = candidates[-1]
        # Never fall back to an older result if the latest metrics file is invalid.
        metrics = _read_object(path)
        if metrics.get("split") != "test" or not isinstance(metrics.get("metrics"), dict):
            raise ResultsError(f"test 평가 보고서 형식이 올바르지 않습니다: {path}")
        evaluation.update(availability="available", path=str(path), report=metrics)
        if report:
            for training_key, evaluation_key, label in (
                ("best_checkpoint_sha256", "checkpoint_sha256", "best 체크포인트 SHA256"),
                ("dataset_fingerprint", "dataset_fingerprint", "데이터셋 fingerprint"),
            ):
                expected, actual = report.get(training_key), metrics.get(evaluation_key)
                if expected is not None and actual is not None and expected != actual:
                    warnings.append(f"학습 보고서와 최신 test 평가의 {label}가 다릅니다.")
                elif expected is None or actual is None:
                    warnings.append(
                        f"{label} 기록이 없어 학습/test 결과 일치 여부를 확인할 수 없습니다."
                    )
    else:
        evaluation["message"] = (
            "저장된 test metrics.json이 없습니다. "
            "미실행·실행 중·실패 여부는 단정할 수 없습니다."
        )
    result["latest_test_evaluation"] = evaluation
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = read_results(args.run_dir)
    except (ResultsError, OSError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
