"""Download immutable, publicly archived KBO game box scores from 2001 onward.

No upstream scraper code is imported, and no play-by-play events are invented
from the box scores. Repository licenses describe the archives, not a blanket
grant of rights over the original KBO material.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from importlib.resources import files
from pathlib import Path
from urllib.request import Request, urlopen

from cpv26.data.kbo_playbyplay import ChecksumMismatchError, sha256_file

DIALEKTIKE_REVISION = "00c63c74c3c0590f3ca2fae5c03d4d2eeaa18296"
LOPES_REVISION = "94e72c797e07b3b72167c92258728bef599ed5fc"
SOURCE_MANIFEST_FILENAME = "SOURCE.json"
_CHUNK_SIZE = 1024 * 1024
_USER_AGENT = "cpv26-predictor/0.4 public-history-archive-downloader"
EXPECTED_HISTORY_GAMES = {
    year: (
        532
        if year <= 2004
        else 504
        if year <= 2008
        else 532
        if year <= 2012
        else 576
        if year <= 2014
        else 720
    )
    for year in range(2001, 2023)
}


@dataclass(frozen=True)
class KBOHistoryArtifact:
    """A raw annual or monthly JSON file with its immutable source identity.

    ``game_count`` is the number of raw records, not a claim that they are
    unique or that the entire regular season is present.
    """

    year: int
    filename: str
    sha256: str
    url: str
    game_count: int
    format: str = "game_map"
    repository_url: str = ""
    revision: str = ""
    repository_license: str | None = None
    month: int | None = None
    bundled_resource: str | None = None
    provenance: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.year, bool) or not isinstance(self.year, int) or self.year < 1:
            raise ValueError("artifact year must be a positive integer")
        if not re.fullmatch(r"[A-Za-z0-9_-]+\.json", self.filename):
            raise ValueError("artifact filename must be a plain JSON filename")
        if not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise ValueError("artifact sha256 must be a lowercase 64-character hex digest")
        if not self.url:
            raise ValueError("artifact url cannot be empty")
        if (
            isinstance(self.game_count, bool)
            or not isinstance(self.game_count, int)
            or self.game_count < 1
        ):
            raise ValueError("artifact game_count must be a positive raw record count")
        if self.format not in {"game_map", "game_list"}:
            raise ValueError("artifact format must be game_map or game_list")
        if self.bundled_resource is not None and not re.fullmatch(
            r"[A-Za-z0-9_-]+\.json", self.bundled_resource
        ):
            raise ValueError("bundled_resource must be a plain JSON resource filename")
        if self.revision and not (
            re.fullmatch(r"[0-9a-f]{40}", self.revision)
            or (
                self.bundled_resource is not None and re.fullmatch(r"verified-\d{8}", self.revision)
            )
        ):
            raise ValueError("artifact revision must be a Git commit or bundled verification date")
        if self.month is not None and (
            isinstance(self.month, bool)
            or not isinstance(self.month, int)
            or not 1 <= self.month <= 12
        ):
            raise ValueError("artifact month must be between 1 and 12")


# Hashes and raw record counts verified against the archived bytes. The 2015
# and 2018 archives each contain 719 games, one short of a full 720-game season.
_ANNUAL_ARCHIVES = (
    (2001, 532, "b963977b7a9ba6cae80cc962abcfd1164f6d6c9f94f76d79222ef111322fb0f4"),
    (2002, 532, "70e6428301b335965155c710aa98886ab0c7f9d7abcec0435e0d65f2c234e727"),
    (2003, 532, "7374d4a410807f1538be8ad376660b2458bf3de26dbcbc9771a99a8d6c8d853f"),
    (2004, 532, "21cf782bac2c553114779485b4ff150703ba05d90491a801c5b45043faa808c4"),
    (2005, 504, "057834987d5bfed6c49d3c83336c2dca995c4380393955012dd7764ae67e13b0"),
    (2006, 504, "8d4636129a2d5f0d9540e6196a1b7df9287fc59565f1347d930796cdd7175cf7"),
    (2007, 504, "18459be3c3b7883311142a2574bdbe8497e9cdfc13058acfbbf9cf1f44e7b9dd"),
    (2008, 504, "5930758bc5c92d604976f04c9292f972184facf669b6c67d2a83df42d21deaf7"),
    (2009, 532, "ee0067803cf7599de15aa4018c2ae159a321282e98a4f49a237d34ff33e507e3"),
    (2010, 532, "5ee5096094e4330e21fb689477c313d3b08f0e17f009780e8b347c5b97f1d11e"),
    (2011, 532, "2d816974e134041d84858d37709c50fd140658edb8386cf18315b61c89ee3ce7"),
    (2012, 532, "229bdcaf4a87e0096ac5f2e6624213cce6811df5b2f5934df5ed9adeda9950d8"),
    (2013, 576, "a9d377ca9a487eb19f097a08119c07dacb0e08c49ca9cc2f2f1d2b59de0a8f0f"),
    (2014, 576, "25d0065778c289872e1975b60f7a68a6d1694ce7956a6a8e20d29381f5b35271"),
    (2015, 719, "98d81ea97ec7f26af29dca758f7ef287b47548fcf4a6ad69a8986ca4498232d4"),
    (2016, 720, "44757daeb2ba9cb9f6a2e635f80bc7eb73ce7c93ac9b925228f4835f93c87a48"),
    (2017, 720, "04d0e6fde33478c2139fe69b683a8778e44bcd36098f3bde39b23fd8d9636b25"),
    (2018, 719, "83c2ab8cc4aa37e9f4a2403ff53c110da672d6e038d43ff9ac2189f3fbd8e3fe"),
    (2019, 720, "4836424adfcd45d730ca25d9d4fb3803d2558b1ba81bf71d3089aa33b3b0741b"),
    (2020, 720, "e1177911f96d3e36b8a1210a34555d04524dc8d48f3f35de1baf3bbef5422ac8"),
)
_MONTHLY_ARCHIVES = (
    (2021, 4, 116, "1a12b42c15de7c7d5229aa7a9a589621ec20c969aa9bfb67731ed7c5872e517c"),
    (2021, 5, 113, "a11c1233a7ac7e1e4b1697eaec144f7f5522db492ac87542c4b796e691706f4e"),
    (2021, 6, 123, "ef12b2a4822c0f6d9c8396b1ecde94f8b68abfeb3099b7fa46991222d0217e3d"),
    (2021, 7, 33, "53e1b459ab3b5829904bb90c172957fc9577956d4b2932139f6550adcaf8f3d2"),
    (2021, 8, 81, "34801b01ab1442ba6b0e5e3f62d767a847f4d49f4030f777e9747a0b8ff928dc"),
    (2021, 9, 132, "c5b852a38c25ad713e6bec295ca7fed7a15837673d785e7dc8414de351ed0a30"),
    (2021, 10, 123, "3d16bb37d79558cb14a1f9024cbe554cf479b0039aac2a7115cafb06cc6311cc"),
    (2022, 4, 123, "496af302e7108a0e5b8681753da739a75b1e03a6528a96c911d2637cfad03eb5"),
    (2022, 5, 129, "4c3f6ef1b4598ac4fd2884249837bf36caeef1c51a120d14db7ca38a14f63d31"),
    (2022, 6, 117, "2ddfa1737253797689f4187505d93e9c2a784093ca3548de77c2bc173e087482"),
    (2022, 7, 95, "3070b301cb31eb50f8aaa25d51174b653e89480a3c2db6cf65bf7d667fe4090a"),
    (2022, 8, 109, "be289dfb2d29cde65292c3b33923bb39ed7068cf65455fe8263e5413c3ee990d"),
    (2022, 9, 119, "dd4734c80f8cfad38eeaae5452d57bcc81b402ed7d879cfa3c35defe031b9474"),
    (2022, 10, 28, "df32286d69cf962bd1da13f2665a036698c9fa0a2353fa17b40a3734b57db4cf"),
)

_ARCHIVE_FILES = tuple(
    KBOHistoryArtifact(
        year=year,
        filename=f"kbo_history_{year}.json",
        sha256=digest,
        url=(
            "https://raw.githubusercontent.com/dialektike/KBO-league/"
            f"{DIALEKTIKE_REVISION}/data/temp/temp_data_{year}.json"
        ),
        game_count=count,
        repository_url="https://github.com/dialektike/KBO-league",
        revision=DIALEKTIKE_REVISION,
        repository_license="GPL-3.0",
    )
    for year, count, digest in _ANNUAL_ARCHIVES
) + tuple(
    KBOHistoryArtifact(
        year=year,
        month=month,
        filename=f"kbo_history_{year}_{month:02d}.json",
        sha256=digest,
        url=(
            "https://raw.githubusercontent.com/LOPES-HUFS/KBO_data/"
            f"{LOPES_REVISION}/sample_data/{year}/{year}_{month:02d}.json"
        ),
        game_count=count,
        format="game_list",
        repository_url="https://github.com/LOPES-HUFS/KBO_data",
        revision=LOPES_REVISION,
        repository_license=None,
    )
    for year, month, count, digest in _MONTHLY_ARCHIVES
)
_SUPPLEMENT_ARCHIVES = (
    (2015, 1, "efba4b0bca8e58d2e9711c94e54445f65cacab26de3d1dee75aa7febcb734682"),
    (2018, 1, "4652c4dad9c150f12956ed510e8392a088f0b7c88c7d4f2079ce279ba6166ca4"),
    (2021, 8, "b048c8630b1145fe102e9bed6ed83e1a37d244cfb338f2cd0a260e69a5e0c8c1"),
)
_PROVENANCE_RESOURCE = "history_supplement_sources.json"
_PROVENANCE_SHA256 = "5189943f1fe1fc6b5478be3d43d54950b596e2ed72a5e41601029cb5e1e701f9"
_provenance_bytes = files("cpv26.data").joinpath(_PROVENANCE_RESOURCE).read_bytes()
if hashlib.sha256(_provenance_bytes).hexdigest() != _PROVENANCE_SHA256:
    raise ValueError("bundled historical score provenance checksum mismatch")
_SUPPLEMENT_PROVENANCE = json.loads(_provenance_bytes)
KBO_HISTORY_FILES = tuple(
    sorted(
        _ARCHIVE_FILES
        + tuple(
            KBOHistoryArtifact(
                year=year,
                filename=f"kbo_history_{year}_supplement.json",
                sha256=digest,
                url="https://www.koreabaseball.com/ws/Main.asmx/GetKboGameList",
                game_count=count,
                repository_url="https://www.koreabaseball.com/",
                revision="verified-20260831",
                bundled_resource=f"history_supplement_{year}.json",
                provenance=_SUPPLEMENT_PROVENANCE,
            )
            for year, count, digest in _SUPPLEMENT_ARCHIVES
        ),
        key=lambda artifact: (artifact.year, artifact.filename),
    )
)


def select_history_artifacts(
    artifacts: Iterable[KBOHistoryArtifact], years: Iterable[int] | None
) -> tuple[KBOHistoryArtifact, ...]:
    """Select every monthly/annual artifact for each explicitly requested year."""

    available = tuple(artifacts)
    if not available:
        raise ValueError("history artifacts cannot be empty")
    if len({artifact.filename for artifact in available}) != len(available):
        raise ValueError("history artifact filenames must be unique")
    if years is None:
        return available
    requested = tuple(years)
    if not requested:
        raise ValueError("years cannot be empty")
    if any(isinstance(year, bool) or not isinstance(year, int) for year in requested):
        raise TypeError("years must contain integers")
    unknown = set(requested) - {artifact.year for artifact in available}
    if unknown:
        raise ValueError(f"unknown KBO history years: {', '.join(map(str, sorted(unknown)))}")
    return tuple(artifact for artifact in available if artifact.year in set(requested))


def download_kbo_history(
    destination: str | Path,
    *,
    years: Iterable[int] | None = None,
    artifacts: tuple[KBOHistoryArtifact, ...] = KBO_HISTORY_FILES,
    timeout_seconds: float = 60.0,
) -> tuple[Path, ...]:
    """Download pinned historical archives without altering their original bytes.

    Valid cached files need no network access. New bytes are atomically promoted
    only after SHA-256 verification; an unsuccessful download never overwrites
    an existing file. The manifest includes all verified cached artifacts, not
    just the years selected by this invocation.
    """

    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    selected = select_history_artifacts(artifacts, years)
    destination_path = Path(destination).expanduser()
    destination_path.mkdir(parents=True, exist_ok=True)
    paths = tuple(
        _ensure_artifact(destination_path, artifact, timeout_seconds) for artifact in selected
    )
    verified = tuple(
        artifact
        for artifact in artifacts
        if (destination_path / artifact.filename).is_file()
        and sha256_file(destination_path / artifact.filename) == artifact.sha256
    )
    payload = {
        "schema_version": 1,
        "dataset_id": "cpv26/kbo-history-public-archives",
        "upstream_description": "Historical KBO game box scores archived from the KBO website",
        "usage_notice": (
            "Unofficial archives. Repository license metadata does not grant blanket rights "
            "to original KBO material. LOPES-HUFS/KBO_data declares no repository license. "
            "Review upstream terms before redistribution or other use. Large raw archives are "
            "not redistributed with this project; ten factual final-score supplements are "
            "packaged with their verification provenance. No play-by-play is reconstructed."
        ),
        "coverage_notice": (
            "Raw record counts are not unique game counts. The 2015 and 2018 archives have "
            "719 games each. The 2021 monthly files contain 721 records but 713 unique IDs, "
            "including the October 31 first-place tiebreaker and eight duplicate records. "
            "Ten missing finals are supplied as small, independently verified factual records, "
            "with official response hashes and request details. Deduplication yields full "
            "regular-season final-score coverage from 2001 to 2022; the tiebreaker is retained "
            "with its own game type. Canonical import verifies regular-season counts separately, "
            "retains every player box-score row and records partial-field quality masks. "
            "Missing PA sequence, identities and game state are not fabricated."
        ),
        "expected_regular_season_games": {
            str(year): count
            for year, count in EXPECTED_HISTORY_GAMES.items()
            if any(artifact.year == year for artifact in verified)
        },
        "retrieved_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "files": [
            {**asdict(artifact), "raw_record_count": artifact.game_count} for artifact in verified
        ],
    }
    _write_manifest(destination_path, payload)
    return paths


def _ensure_artifact(
    destination: Path, artifact: KBOHistoryArtifact, timeout_seconds: float
) -> Path:
    target = destination / artifact.filename
    if target.exists() and not target.is_file():
        raise IsADirectoryError(target)
    if target.is_file() and sha256_file(target) == artifact.sha256:
        return target
    partial: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{artifact.filename}.",
            suffix=".part",
            dir=destination,
            delete=False,
        ) as output:
            partial = Path(output.name)
            if artifact.bundled_resource is not None:
                output.write(files("cpv26.data").joinpath(artifact.bundled_resource).read_bytes())
            else:
                request = Request(artifact.url, headers={"User-Agent": _USER_AGENT})
                with urlopen(request, timeout=timeout_seconds) as response:
                    while chunk := response.read(_CHUNK_SIZE):
                        output.write(chunk)
        actual = sha256_file(partial)
        if actual != artifact.sha256:
            raise ChecksumMismatchError(
                filename=artifact.filename,
                expected_sha256=artifact.sha256,
                actual_sha256=actual,
            )
        os.replace(partial, target)
    finally:
        if partial is not None:
            partial.unlink(missing_ok=True)
    return target


def _write_manifest(destination: Path, payload: dict[str, object]) -> None:
    partial: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=".SOURCE.",
            suffix=".part",
            dir=destination,
            delete=False,
        ) as output:
            partial = Path(output.name)
            json.dump(payload, output, ensure_ascii=False, indent=2, sort_keys=True)
            output.write("\n")
        os.replace(partial, destination / SOURCE_MANIFEST_FILENAME)
    finally:
        if partial is not None:
            partial.unlink(missing_ok=True)
