from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

import cpv26.data.kbo_playbyplay as source_module
from cpv26.data.kbo_playbyplay import (
    KBO_PLAYBYPLAY_FILES,
    KBO_PLAYBYPLAY_REVISION,
    KBO_PLAYBYPLAY_SOURCE,
    ChecksumMismatchError,
    KBOPlayByPlayArtifact,
    KBOPlayByPlaySource,
    download_kbo_playbyplay,
)


def _fixture_source(
    source_directory: Path,
    *,
    payload: bytes,
    expected_sha256: str | None = None,
) -> KBOPlayByPlaySource:
    filename = "kbo_pbp_2099.parquet"
    (source_directory / filename).write_bytes(payload)
    artifact = KBOPlayByPlayArtifact(
        year=2099,
        filename=filename,
        sha256=expected_sha256 or hashlib.sha256(payload).hexdigest(),
    )
    return KBOPlayByPlaySource(
        dataset_id="fixture/kbo_playbyplay",
        revision="f" * 40,
        repository_url=source_directory.as_uri(),
        resolve_base_url=source_directory.as_uri(),
        license="fixture-only",
        artifacts=(artifact,),
    )


def test_pinned_source_has_published_revision_filenames_and_sha256() -> None:
    assert KBO_PLAYBYPLAY_REVISION == "6afc8af044e3bba5f326b688e8cb41d7ff7065ec"
    assert all(
        f"/resolve/{KBO_PLAYBYPLAY_REVISION}/v0/" in KBO_PLAYBYPLAY_SOURCE.artifact_url(artifact)
        for artifact in KBO_PLAYBYPLAY_FILES
    )
    assert {
        artifact.year: (artifact.filename, artifact.sha256) for artifact in KBO_PLAYBYPLAY_FILES
    } == {
        2023: (
            "kbo_pbp_2023.parquet",
            "818f6016655b02fe48b8118281d1b04bfe3548d376fdc70131a41ea539341edb",
        ),
        2024: (
            "kbo_pbp_2024.parquet",
            "8332cd716cf0126a4ab0bf390383f43deff22ab320a57fb70d02b31025bdf553",
        ),
        2025: (
            "kbo_pbp_2025.parquet",
            "2c824919495809722a5ff0290a823ff9a44d88f61640ad9b288ff3dca2652f2c",
        ),
        2026: (
            "kbo_pbp_2026.parquet",
            "9d330311d28371806028b878191fcc85b9170839c8951b00ff9c64ec8aa28630",
        ),
    }


def test_downloads_from_file_url_and_writes_source_manifest(tmp_path: Path) -> None:
    source_directory = tmp_path / "remote"
    source_directory.mkdir()
    payload = b"PAR1fixture parquet bytesPAR1"
    source = _fixture_source(source_directory, payload=payload)
    destination = tmp_path / "download"

    paths = download_kbo_playbyplay(destination, years=[2099], source=source)

    expected_path = destination / "kbo_pbp_2099.parquet"
    assert paths == (expected_path,)
    assert expected_path.read_bytes() == payload
    assert not (destination / "kbo_pbp_2099.parquet.part").exists()
    manifest = json.loads((destination / "SOURCE.json").read_text(encoding="utf-8"))
    assert manifest["dataset_id"] == "fixture/kbo_playbyplay"
    assert manifest["revision"] == "f" * 40
    assert manifest["files"] == [
        {
            "filename": "kbo_pbp_2099.parquet",
            "regular_season_complete": True,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "url": f"{source_directory.as_uri()}/kbo_pbp_2099.parquet",
            "year": 2099,
        }
    ]
    assert manifest["retrieved_at"].endswith("Z")
    assert not (destination / "SOURCE.json.part").exists()


def test_valid_existing_file_is_reused_without_opening_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_directory = tmp_path / "remote"
    source_directory.mkdir()
    payload = b"already valid"
    source = _fixture_source(source_directory, payload=payload)
    destination = tmp_path / "download"
    destination.mkdir()
    target = destination / "kbo_pbp_2099.parquet"
    target.write_bytes(payload)
    stale_partial = destination / "kbo_pbp_2099.parquet.part"
    stale_partial.write_bytes(b"stale")

    def fail_if_opened(*args: object, **kwargs: object) -> None:
        raise AssertionError("valid cached file must skip network access")

    monkeypatch.setattr(source_module, "urlopen", fail_if_opened)

    assert download_kbo_playbyplay(destination, source=source) == (target,)
    assert target.read_bytes() == payload
    assert not stale_partial.exists()
    assert (destination / "SOURCE.json").is_file()


def test_invalid_existing_file_is_replaced_only_after_verified_download(tmp_path: Path) -> None:
    source_directory = tmp_path / "remote"
    source_directory.mkdir()
    payload = b"verified replacement"
    source = _fixture_source(source_directory, payload=payload)
    destination = tmp_path / "download"
    destination.mkdir()
    target = destination / "kbo_pbp_2099.parquet"
    target.write_bytes(b"corrupt cached bytes")

    download_kbo_playbyplay(destination, source=source)

    assert target.read_bytes() == payload
    assert not (destination / "kbo_pbp_2099.parquet.part").exists()


def test_checksum_mismatch_never_publishes_partial_file(tmp_path: Path) -> None:
    source_directory = tmp_path / "remote"
    source_directory.mkdir()
    source = _fixture_source(
        source_directory,
        payload=b"unexpected remote bytes",
        expected_sha256="0" * 64,
    )
    destination = tmp_path / "download"
    destination.mkdir()
    final_path = destination / "kbo_pbp_2099.parquet"
    final_path.write_bytes(b"known-invalid cached bytes")

    with pytest.raises(ChecksumMismatchError, match="SHA-256 mismatch") as caught:
        download_kbo_playbyplay(destination, source=source)

    assert caught.value.expected_sha256 == "0" * 64
    assert not final_path.exists()
    assert not (destination / "kbo_pbp_2099.parquet.part").exists()
    assert not (destination / "SOURCE.json").exists()


def test_manifest_keeps_previously_verified_seasons_on_incremental_download(
    tmp_path: Path,
) -> None:
    remote = tmp_path / "remote"
    remote.mkdir()
    source = _fixture_source(remote, payload=b"first season")
    second_payload = b"second season"
    second = KBOPlayByPlayArtifact(
        year=2100,
        filename="kbo_pbp_2100.parquet",
        sha256=hashlib.sha256(second_payload).hexdigest(),
    )
    (remote / second.filename).write_bytes(second_payload)
    source = replace(source, artifacts=(*source.artifacts, second))
    destination = tmp_path / "download"

    download_kbo_playbyplay(destination, years=[2099], source=source)
    download_kbo_playbyplay(destination, years=[2100], source=source)

    manifest = json.loads((destination / "SOURCE.json").read_text(encoding="utf-8"))
    assert [item["year"] for item in manifest["files"]] == [2099, 2100]
