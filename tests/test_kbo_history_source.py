from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

import cpv26.data.kbo_history_source as source_module
from cpv26.data.kbo_history_source import (
    DIALEKTIKE_REVISION,
    EXPECTED_HISTORY_GAMES,
    KBO_HISTORY_FILES,
    LOPES_REVISION,
    KBOHistoryArtifact,
    download_kbo_history,
    select_history_artifacts,
)
from cpv26.data.kbo_playbyplay import ChecksumMismatchError


def _artifact(remote: Path, *, year: int = 2001, month: int | None = None) -> KBOHistoryArtifact:
    filename = f"kbo_history_{year}{f'_{month:02d}' if month is not None else ''}.json"
    payload = json.dumps({"fixture_only": year, "month": month}).encode("utf-8")
    path = remote / filename
    path.write_bytes(payload)
    return KBOHistoryArtifact(
        year=year,
        month=month,
        filename=filename,
        sha256=hashlib.sha256(payload).hexdigest(),
        url=path.as_uri(),
        game_count=1,
        repository_license="test fixture only",
    )


def test_pins_all_years_2001_through_2022_without_claiming_complete_raw_coverage() -> None:
    assert {artifact.year for artifact in KBO_HISTORY_FILES} == set(range(2001, 2023))
    assert len(KBO_HISTORY_FILES) == 37
    assert len({artifact.filename for artifact in KBO_HISTORY_FILES}) == 37
    for artifact in KBO_HISTORY_FILES:
        if artifact.bundled_resource is not None:
            assert artifact.revision == "verified-20260831"
            assert artifact.provenance["scope"] == "final_game_score_only"
            continue
        if artifact.year <= 2020:
            assert artifact.revision == DIALEKTIKE_REVISION
            assert artifact.format == "game_map"
            assert artifact.repository_license == "GPL-3.0"
        else:
            assert artifact.revision == LOPES_REVISION
            assert artifact.format == "game_list"
            assert artifact.repository_license is None
        assert artifact.revision in artifact.url
    totals = {
        year: sum(artifact.game_count for artifact in KBO_HISTORY_FILES if artifact.year == year)
        for year in range(2001, 2023)
    }
    assert totals[2001] == 532
    assert totals[2015] == totals[2018] == 720
    assert totals[2021] == 729  # 720 regular + eight duplicates + the first-place tiebreaker.
    assert totals[2022] == 720
    assert sum(EXPECTED_HISTORY_GAMES.values()) == 13184
    assert KBO_HISTORY_FILES[0].sha256 == (
        "b963977b7a9ba6cae80cc962abcfd1164f6d6c9f94f76d79222ef111322fb0f4"
    )


def test_selects_all_months_in_year_and_deduplicates_requested_years() -> None:
    selected = select_history_artifacts(KBO_HISTORY_FILES, [2022, 2001, 2022])
    assert len(selected) == 8
    assert selected[0].year == 2001
    assert [artifact.month for artifact in selected[1:]] == list(range(4, 11))


@pytest.mark.parametrize("years", [[], [2000], [2023], [True], [2001.0], ["2001"]])
def test_rejects_invalid_years_before_creating_destination(
    tmp_path: Path, years: list[object]
) -> None:
    destination = tmp_path / "not-created"
    with pytest.raises((TypeError, ValueError)):
        download_kbo_history(destination, years=years)  # type: ignore[arg-type]
    assert not destination.exists()


def test_download_verifies_raw_bytes_and_records_provenance(tmp_path: Path) -> None:
    remote = tmp_path / "remote"
    remote.mkdir()
    artifact = _artifact(remote)
    destination = tmp_path / "download"

    paths = download_kbo_history(destination, years=[2001], artifacts=(artifact,))

    assert paths == (destination / artifact.filename,)
    assert paths[0].read_bytes() == (remote / artifact.filename).read_bytes()
    manifest = json.loads((destination / "SOURCE.json").read_text(encoding="utf-8"))
    assert manifest["files"][0]["sha256"] == artifact.sha256
    assert manifest["files"][0]["url"] == artifact.url
    assert manifest["files"][0]["raw_record_count"] == 1
    assert manifest["files"][0]["repository_license"] == "test fixture only"
    assert "not grant blanket rights" in manifest["usage_notice"]
    assert manifest["retrieved_at"].endswith("Z")
    assert not list(destination.glob("*.part"))


def test_valid_cache_never_contacts_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    remote = tmp_path / "remote"
    remote.mkdir()
    artifact = _artifact(remote)
    destination = tmp_path / "download"
    first = download_kbo_history(destination, artifacts=(artifact,))

    def unexpected_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("valid cache should not contact the source")

    monkeypatch.setattr(source_module, "urlopen", unexpected_network)
    assert download_kbo_history(destination, artifacts=(artifact,)) == first


def test_failed_verification_preserves_existing_file_and_leaves_no_partial(tmp_path: Path) -> None:
    remote = tmp_path / "remote"
    remote.mkdir()
    artifact = replace(_artifact(remote), sha256="0" * 64)
    destination = tmp_path / "download"
    destination.mkdir()
    target = destination / artifact.filename
    target.write_bytes(b"existing file is not overwritten before verification")

    with pytest.raises(ChecksumMismatchError, match="SHA-256 mismatch"):
        download_kbo_history(destination, artifacts=(artifact,))

    assert target.read_bytes() == b"existing file is not overwritten before verification"
    assert not list(destination.glob("*.part"))
    assert not (destination / "SOURCE.json").exists()


def test_successful_verified_download_atomically_replaces_invalid_cache(tmp_path: Path) -> None:
    remote = tmp_path / "remote"
    remote.mkdir()
    artifact = _artifact(remote)
    destination = tmp_path / "download"
    destination.mkdir()
    target = destination / artifact.filename
    target.write_bytes(b"invalid cache")

    download_kbo_history(destination, artifacts=(artifact,))

    assert target.read_bytes() == (remote / artifact.filename).read_bytes()
    assert not list(destination.glob("*.part"))


def test_manifest_retains_previous_verified_years_and_all_months(tmp_path: Path) -> None:
    remote = tmp_path / "remote"
    remote.mkdir()
    artifacts = (
        _artifact(remote),
        _artifact(remote, year=2022, month=4),
        _artifact(remote, year=2022, month=5),
    )
    destination = tmp_path / "download"
    download_kbo_history(destination, years=[2001], artifacts=artifacts)
    result = download_kbo_history(destination, years=[2022], artifacts=artifacts)
    manifest = json.loads((destination / "SOURCE.json").read_text(encoding="utf-8"))

    assert len(result) == 2
    assert [item["filename"] for item in manifest["files"]] == [
        artifact.filename for artifact in artifacts
    ]


def test_duplicate_artifact_filename_is_rejected(tmp_path: Path) -> None:
    artifact = KBO_HISTORY_FILES[0]
    with pytest.raises(ValueError, match="filenames must be unique"):
        download_kbo_history(tmp_path / "download", artifacts=(artifact, artifact))


def test_packaged_real_final_score_supplements_verify_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    supplements = tuple(
        artifact for artifact in KBO_HISTORY_FILES if artifact.bundled_resource is not None
    )

    def unexpected_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("the tiny verified supplements must not query KBO again")

    monkeypatch.setattr(source_module, "urlopen", unexpected_network)
    paths = download_kbo_history(tmp_path / "download", artifacts=supplements)
    assert len(paths) == 3
    for artifact, path in zip(supplements, paths, strict=True):
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact.sha256
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert len(payload) == artifact.game_count
        assert len(artifact.provenance["responses"]) == 5  # type: ignore[arg-type]
    assert (
        json.loads(paths[0].read_text(encoding="utf-8"))["20150708_HTWO0"]["scoreboard"][0]["R"]
        == 3
    )


def test_bad_packaged_supplement_checksum_is_never_published(tmp_path: Path) -> None:
    supplement = next(
        artifact for artifact in KBO_HISTORY_FILES if artifact.bundled_resource is not None
    )
    wrong_digest = replace(supplement, sha256="0" * 64)
    destination = tmp_path / "download"

    with pytest.raises(ChecksumMismatchError):
        download_kbo_history(destination, artifacts=(wrong_digest,))

    assert not (destination / wrong_digest.filename).exists()
    assert not list(destination.glob("*.part"))


@pytest.mark.parametrize(
    "changes",
    [
        {"filename": "../escape.json"},
        {"filename": "nested\\escape.json"},
        {"filename": "SOURCE.json.part"},
        {"sha256": "A" * 64},
        {"year": True},
        {"month": 13},
        {"month": True},
        {"game_count": 0},
        {"format": "imagined_pa_events"},
        {"revision": "main"},
    ],
)
def test_artifact_metadata_rejects_unsafe_or_invalid_fields(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        replace(KBO_HISTORY_FILES[0], **changes)  # type: ignore[arg-type]
