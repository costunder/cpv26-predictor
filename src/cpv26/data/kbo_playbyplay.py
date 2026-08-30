"""Reproducible downloader for the published KBO play-by-play Parquet snapshot.

This module downloads an existing, revision-pinned Hugging Face dataset.  It
does not crawl KBO, NAVER Sports, or Statiz.  The dataset author publishes the
derived records under CC BY 4.0 while noting that upstream source terms and
rights remain the user's responsibility.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

KBO_PLAYBYPLAY_DATASET_ID = "slothman3878/kbo_playbyplay"
KBO_PLAYBYPLAY_REVISION = "6afc8af044e3bba5f326b688e8cb41d7ff7065ec"
KBO_PLAYBYPLAY_REPOSITORY_URL = "https://huggingface.co/datasets/slothman3878/kbo_playbyplay"
KBO_PLAYBYPLAY_RESOLVE_BASE_URL = (
    f"{KBO_PLAYBYPLAY_REPOSITORY_URL}/resolve/{KBO_PLAYBYPLAY_REVISION}/v0"
)
KBO_PLAYBYPLAY_LICENSE = "CC BY 4.0"
SOURCE_MANIFEST_FILENAME = "SOURCE.json"
_DOWNLOAD_CHUNK_SIZE = 1024 * 1024
_USER_AGENT = "cpv26-predictor/0.4 public-dataset-downloader"


@dataclass(frozen=True)
class KBOPlayByPlayArtifact:
    """One immutable season artifact within a dataset revision."""

    year: int
    filename: str
    sha256: str
    regular_season_complete: bool = True
    coverage_through: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.year, bool) or not isinstance(self.year, int) or self.year < 1:
            raise ValueError("artifact year must be a positive integer")
        if Path(self.filename).name != self.filename or not self.filename.endswith(".parquet"):
            raise ValueError("artifact filename must be a plain .parquet filename")
        if len(self.sha256) != 64:
            raise ValueError("artifact sha256 must be a 64-character hexadecimal digest")
        try:
            int(self.sha256, 16)
        except ValueError as exc:
            raise ValueError("artifact sha256 must be hexadecimal") from exc
        if self.sha256 != self.sha256.lower():
            raise ValueError("artifact sha256 must use lowercase hexadecimal")
        if self.regular_season_complete and self.coverage_through is not None:
            raise ValueError("a complete season cannot have a partial coverage date")
        if not self.regular_season_complete and self.coverage_through is None:
            raise ValueError("a partial season must declare coverage_through")


@dataclass(frozen=True)
class KBOPlayByPlaySource:
    """Metadata needed to resolve and attribute an immutable source snapshot."""

    dataset_id: str
    revision: str
    repository_url: str
    resolve_base_url: str
    license: str
    artifacts: tuple[KBOPlayByPlayArtifact, ...]
    upstream_description: str = "Derived from NAVER Sports KBO play-by-play"
    usage_notice: str = (
        "Unofficial dataset. Upstream source material may be subject to separate terms and rights."
    )

    def __post_init__(self) -> None:
        if not self.dataset_id:
            raise ValueError("dataset_id cannot be empty")
        if len(self.revision) != 40:
            raise ValueError("revision must be a full 40-character Git commit")
        try:
            int(self.revision, 16)
        except ValueError as exc:
            raise ValueError("revision must be hexadecimal") from exc
        if self.revision != self.revision.lower():
            raise ValueError("revision must use lowercase hexadecimal")
        if not self.repository_url or not self.resolve_base_url:
            raise ValueError("source URLs cannot be empty")
        if not self.artifacts:
            raise ValueError("source must declare at least one artifact")
        years = [artifact.year for artifact in self.artifacts]
        filenames = [artifact.filename for artifact in self.artifacts]
        if len(set(years)) != len(years):
            raise ValueError("source artifact years must be unique")
        if len(set(filenames)) != len(filenames):
            raise ValueError("source artifact filenames must be unique")

    def artifact_url(self, artifact: KBOPlayByPlayArtifact) -> str:
        """Return the source URL for an artifact in this pinned snapshot."""

        return f"{self.resolve_base_url.rstrip('/')}/{quote(artifact.filename, safe='')}"


KBO_PLAYBYPLAY_FILES = (
    KBOPlayByPlayArtifact(
        year=2023,
        filename="kbo_pbp_2023.parquet",
        sha256="818f6016655b02fe48b8118281d1b04bfe3548d376fdc70131a41ea539341edb",
    ),
    KBOPlayByPlayArtifact(
        year=2024,
        filename="kbo_pbp_2024.parquet",
        sha256="8332cd716cf0126a4ab0bf390383f43deff22ab320a57fb70d02b31025bdf553",
    ),
    KBOPlayByPlayArtifact(
        year=2025,
        filename="kbo_pbp_2025.parquet",
        sha256="2c824919495809722a5ff0290a823ff9a44d88f61640ad9b288ff3dca2652f2c",
    ),
    KBOPlayByPlayArtifact(
        year=2026,
        filename="kbo_pbp_2026.parquet",
        sha256="9d330311d28371806028b878191fcc85b9170839c8951b00ff9c64ec8aa28630",
        regular_season_complete=False,
        coverage_through="2026-07-26",
    ),
)

KBO_PLAYBYPLAY_SOURCE = KBOPlayByPlaySource(
    dataset_id=KBO_PLAYBYPLAY_DATASET_ID,
    revision=KBO_PLAYBYPLAY_REVISION,
    repository_url=KBO_PLAYBYPLAY_REPOSITORY_URL,
    resolve_base_url=KBO_PLAYBYPLAY_RESOLVE_BASE_URL,
    license=KBO_PLAYBYPLAY_LICENSE,
    artifacts=KBO_PLAYBYPLAY_FILES,
)


class ChecksumMismatchError(RuntimeError):
    """Raised when a downloaded artifact does not match its pinned SHA-256."""

    def __init__(
        self,
        *,
        filename: str,
        expected_sha256: str,
        actual_sha256: str,
    ) -> None:
        self.filename = filename
        self.expected_sha256 = expected_sha256
        self.actual_sha256 = actual_sha256
        super().__init__(
            f"SHA-256 mismatch for {filename}: expected {expected_sha256}, got {actual_sha256}"
        )


def sha256_file(path: str | Path) -> str:
    """Return the lowercase SHA-256 digest of a local file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(_DOWNLOAD_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def download_kbo_playbyplay(
    destination: str | Path,
    *,
    years: Iterable[int] | None = None,
    source: KBOPlayByPlaySource = KBO_PLAYBYPLAY_SOURCE,
    timeout_seconds: float = 60.0,
) -> tuple[Path, ...]:
    """Download and verify selected season files into ``destination``.

    Files are streamed to ``<filename>.part``, verified against the pinned
    SHA-256, and atomically promoted to their final name.  An existing final
    file is reused only when its checksum is already valid.  ``SOURCE.json``
    is atomically written after every selected file is known to be valid.

    The ``source`` argument exists so callers can mirror an immutable snapshot
    and so tests can use ``file://`` fixtures.  The default always targets the
    published, revision-pinned Hugging Face snapshot above.
    """

    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    selected = _select_artifacts(source, years)
    destination_path = Path(destination).expanduser()
    destination_path.mkdir(parents=True, exist_ok=True)
    if not destination_path.is_dir():
        raise NotADirectoryError(destination_path)

    paths = tuple(
        _ensure_artifact(
            destination_path,
            artifact=artifact,
            url=source.artifact_url(artifact),
            timeout_seconds=timeout_seconds,
        )
        for artifact in selected
    )
    verified_cached_artifacts = tuple(
        artifact
        for artifact in source.artifacts
        if (destination_path / artifact.filename).is_file()
        and sha256_file(destination_path / artifact.filename) == artifact.sha256
    )
    _write_source_manifest(
        destination_path, source=source, artifacts=verified_cached_artifacts
    )
    return paths


def _select_artifacts(
    source: KBOPlayByPlaySource,
    years: Iterable[int] | None,
) -> tuple[KBOPlayByPlayArtifact, ...]:
    if years is None:
        return source.artifacts
    requested = tuple(years)
    if not requested:
        raise ValueError("years cannot be empty")
    if any(isinstance(year, bool) or not isinstance(year, int) for year in requested):
        raise TypeError("years must contain integers")
    requested_set = set(requested)
    known_years = {artifact.year for artifact in source.artifacts}
    unknown = sorted(requested_set - known_years)
    if unknown:
        raise ValueError(f"unknown KBO play-by-play years: {', '.join(map(str, unknown))}")
    return tuple(artifact for artifact in source.artifacts if artifact.year in requested_set)


def _ensure_artifact(
    destination: Path,
    *,
    artifact: KBOPlayByPlayArtifact,
    url: str,
    timeout_seconds: float,
) -> Path:
    target = destination / artifact.filename
    partial = destination / f"{artifact.filename}.part"
    if target.exists() and not target.is_file():
        raise IsADirectoryError(target)
    if partial.exists() and not partial.is_file():
        raise IsADirectoryError(partial)
    if target.is_file():
        if sha256_file(target) == artifact.sha256:
            partial.unlink(missing_ok=True)
            return target
        # A known-invalid cache entry must never remain under the publishable
        # final filename if the replacement download later fails.
        target.unlink()

    request = Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urlopen(request, timeout=timeout_seconds) as response, partial.open("wb") as output:
            while chunk := response.read(_DOWNLOAD_CHUNK_SIZE):
                output.write(chunk)
        actual_sha256 = sha256_file(partial)
        if actual_sha256 != artifact.sha256:
            raise ChecksumMismatchError(
                filename=artifact.filename,
                expected_sha256=artifact.sha256,
                actual_sha256=actual_sha256,
            )
        os.replace(partial, target)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return target


def _write_source_manifest(
    destination: Path,
    *,
    source: KBOPlayByPlaySource,
    artifacts: tuple[KBOPlayByPlayArtifact, ...],
) -> None:
    manifest_path = destination / SOURCE_MANIFEST_FILENAME
    partial_path = destination / f"{SOURCE_MANIFEST_FILENAME}.part"
    payload = {
        "schema_version": 1,
        "dataset_id": source.dataset_id,
        "repository_url": source.repository_url,
        "revision": source.revision,
        "license": source.license,
        "upstream_description": source.upstream_description,
        "usage_notice": source.usage_notice,
        "retrieved_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "files": [
            {
                "year": artifact.year,
                "filename": artifact.filename,
                "sha256": artifact.sha256,
                "url": source.artifact_url(artifact),
                "regular_season_complete": artifact.regular_season_complete,
                **(
                    {"coverage_through": artifact.coverage_through}
                    if artifact.coverage_through is not None
                    else {}
                ),
            }
            for artifact in artifacts
        ],
    }
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    try:
        partial_path.write_bytes(encoded)
        os.replace(partial_path, manifest_path)
    except BaseException:
        partial_path.unlink(missing_ok=True)
        raise


__all__ = [
    "KBO_PLAYBYPLAY_DATASET_ID",
    "KBO_PLAYBYPLAY_FILES",
    "KBO_PLAYBYPLAY_LICENSE",
    "KBO_PLAYBYPLAY_REPOSITORY_URL",
    "KBO_PLAYBYPLAY_RESOLVE_BASE_URL",
    "KBO_PLAYBYPLAY_REVISION",
    "KBO_PLAYBYPLAY_SOURCE",
    "SOURCE_MANIFEST_FILENAME",
    "ChecksumMismatchError",
    "KBOPlayByPlayArtifact",
    "KBOPlayByPlaySource",
    "download_kbo_playbyplay",
    "sha256_file",
]
