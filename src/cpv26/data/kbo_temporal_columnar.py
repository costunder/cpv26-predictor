"""Memory-mapped primitive storage for the production temporal KBO graph.

The archive owns one immutable set of column files.  Worker processes map
those files read-only and retain only integer state plus graph aggregates;
source dictionaries are decoded into short-lived views when a changed record
is applied.  Cutoff advancement consumes a precomputed, sorted key-change
stream, so it never scans or rebuilds the complete raw history per sample.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator, Mapping, Sequence
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, overload
from zoneinfo import ZoneInfo

import numpy as np

from .kbo_graph_dataset import Array, _json_default, _Record

COLUMNAR_RECORD_STORE_SCHEMA_VERSION = 1
_KST = ZoneInfo("Asia/Seoul")
_UTC_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
_MISSING_TIME = np.iinfo(np.int64).min


def _datetime_to_us(value: datetime) -> int:
    if value.utcoffset() is None:
        raise ValueError("columnar temporal datetime must be timezone-aware")
    delta = value.astimezone(timezone.utc) - _UTC_EPOCH
    return ((delta.days * 86_400 + delta.seconds) * 1_000_000) + delta.microseconds


def _us_to_datetime(value: int) -> datetime:
    if value == _MISSING_TIME:
        raise ValueError("missing temporal timestamp cannot be decoded")
    return _UTC_EPOCH + timedelta(microseconds=value)


def _first_cutoff(value: datetime) -> date:
    local = value.astimezone(_KST)
    return local.date() + timedelta(days=local.time().replace(tzinfo=None) != time.min)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_npy(path: Path, values: Array) -> dict[str, Any]:
    array = np.asarray(values)
    if array.dtype.hasobject:
        raise ValueError(f"object arrays are forbidden in columnar storage: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    os.replace(temporary, path)
    return {
        "file": path.as_posix(),
        "sha256": _sha256_file(path),
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "bytes": path.stat().st_size,
    }


def _encode_utf8(values: Sequence[str]) -> tuple[Array, Array]:
    encoded = [value.encode("utf-8") for value in values]
    offsets = np.zeros(len(encoded) + 1, dtype=np.int64)
    for index, value in enumerate(encoded, start=1):
        offsets[index] = offsets[index - 1] + len(value)
    blob = np.frombuffer(b"".join(encoded), dtype=np.uint8).copy()
    return blob, offsets


def _write_utf8_column(root: Path, name: str, values: Sequence[str]) -> dict[str, Any]:
    blob, offsets = _encode_utf8(values)
    return {
        "encoding": "utf8_blob_offsets_v1",
        "blob": _write_npy(root / f"{name}.utf8.npy", blob),
        "offsets": _write_npy(root / f"{name}.offsets.npy", offsets),
    }


def write_columnar_record_store(
    directory: Path,
    records: Sequence[_Record],
    record_keys: Sequence[str],
) -> dict[str, Any]:
    """Write every source version once and build its cutoff change stream."""

    if len(records) != len(record_keys):
        raise ValueError("columnar record keys disagree with record count")
    root = directory / "columnar" / "records"
    root.mkdir(parents=True, exist_ok=True)
    relative_root = root.relative_to(directory)
    columns: dict[str, Any] = {}
    for name, string_values in (
        ("kind", [record.kind for record in records]),
        ("entity", [record.entity for record in records]),
        ("row_id", [record.row_id for record in records]),
        ("source_id", [record.source_id for record in records]),
        ("record_key", list(record_keys)),
        ("digest_hex", [f"{record.digest:064x}" for record in records]),
        (
            "data_json",
            [
                json.dumps(
                    record.data,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                    default=_json_default,
                )
                for record in records
            ],
        ),
    ):
        columns[name] = _write_utf8_column(root, name, string_values)

    box_width = max((len(record.box_values) for record in records), default=0)
    box_values = np.zeros((len(records), box_width), dtype=np.float64)
    box_lengths = np.zeros(len(records), dtype=np.int16)
    for index, record in enumerate(records):
        length = len(record.box_values)
        box_lengths[index] = length
        if length:
            box_values[index, :length] = record.box_values
    numeric: dict[str, Array] = {
        "day_ordinal": np.asarray([record.day.toordinal() for record in records], dtype=np.int32),
        "event_us": np.asarray(
            [_datetime_to_us(record.event_at) for record in records], dtype=np.int64
        ),
        "available_us": np.asarray(
            [_datetime_to_us(record.available_at) for record in records], dtype=np.int64
        ),
        "ingested_us": np.asarray(
            [_datetime_to_us(record.ingested_at) for record in records], dtype=np.int64
        ),
        "valid_from_us": np.asarray(
            [_datetime_to_us(record.valid_from) for record in records], dtype=np.int64
        ),
        "valid_to_us": np.asarray(
            [
                _MISSING_TIME if record.valid_to is None else _datetime_to_us(record.valid_to)
                for record in records
            ],
            dtype=np.int64,
        ),
        "values": np.asarray(
            [record.values for record in records], dtype=np.float64
        ).reshape(-1, 7),
        "box_values": box_values,
        "box_value_lengths": box_lengths,
    }
    for name, numeric_values in numeric.items():
        columns[name] = _write_npy(root / f"{name}.npy", numeric_values)

    grouped: dict[tuple[str, str], list[int]] = {}
    for index, record in enumerate(records):
        grouped.setdefault((record.kind, record.entity), []).append(index)
    keys = sorted(grouped)
    key_offsets = np.zeros(len(keys) + 1, dtype=np.int64)
    key_record_indices: list[int] = []
    schedule: set[tuple[int, int]] = set()
    for key_index, key in enumerate(keys):
        ordered = sorted(grouped[key], key=lambda index: records[index].rank)
        key_record_indices.extend(ordered)
        key_offsets[key_index + 1] = len(key_record_indices)
        for record_index in ordered:
            record = records[record_index]
            activation = max(
                record.day + timedelta(days=1),
                _first_cutoff(record.available_at),
                _first_cutoff(record.ingested_at),
                _first_cutoff(record.valid_from),
            )
            schedule.add((activation.toordinal(), key_index))
            if record.valid_to is not None:
                schedule.add((_first_cutoff(record.valid_to).toordinal(), key_index))
    schedule_rows = sorted(schedule)
    index_columns: dict[str, Array] = {
        "key_offsets": key_offsets,
        "key_record_indices": np.asarray(key_record_indices, dtype=np.int64),
        "schedule_day_ordinal": np.asarray([row[0] for row in schedule_rows], dtype=np.int32),
        "schedule_key_index": np.asarray([row[1] for row in schedule_rows], dtype=np.int64),
    }
    for name, index_values in index_columns.items():
        columns[name] = _write_npy(root / f"{name}.npy", index_values)
    columns["key_kind"] = _write_utf8_column(root, "key_kind", [key[0] for key in keys])
    columns["key_entity"] = _write_utf8_column(root, "key_entity", [key[1] for key in keys])

    # Paths in entries are archive-relative, not staging-directory absolute paths.
    for descriptor in columns.values():
        items = (
            (descriptor["blob"], descriptor["offsets"])
            if "encoding" in descriptor
            else (descriptor,)
        )
        for item in items:
            item["file"] = (relative_root / Path(item["file"]).name).as_posix()
    return {
        "schema_version": COLUMNAR_RECORD_STORE_SCHEMA_VERSION,
        "backend": "numpy_npy_mmap_utf8_blob_v1",
        "record_count": len(records),
        "logical_key_count": len(keys),
        "pickle_allowed": False,
        "read_mode": "read_only_mmap",
        "point_in_time_fields": [
            "day_ordinal",
            "available_us",
            "ingested_us",
            "valid_from_us",
            "valid_to_us",
        ],
        "cutoff_index": "sorted_key_change_stream_with_searchsorted_prefix",
        "columns": columns,
    }


class _UTF8Column:
    __slots__ = ("blob", "offsets")

    def __init__(self, blob: Array, offsets: Array) -> None:
        self.blob = blob
        self.offsets = offsets

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def __getitem__(self, index: int) -> str:
        start = int(self.offsets[index])
        stop = int(self.offsets[index + 1])
        return self.blob[start:stop].tobytes().decode("utf-8")


class _RecordDataView(Mapping[str, Any]):
    __slots__ = ("_store", "_index", "_decoded")

    def __init__(self, store: MMapTemporalRecordStore, index: int) -> None:
        self._store = store
        self._index = index
        self._decoded: dict[str, Any] | None = None

    def _mapping(self) -> dict[str, Any]:
        if self._decoded is None:
            raw = json.loads(self._store.strings["data_json"][self._index])
            if not isinstance(raw, dict):
                raise ValueError("columnar temporal data_json must decode to an object")
            for name in (
                "scheduled_start",
                "event_at",
                "available_at",
                "ingested_at",
                "valid_from",
                "valid_to",
            ):
                value = raw.get(name)
                if isinstance(value, str):
                    parsed = datetime.fromisoformat(value)
                    if parsed.utcoffset() is None:
                        raise ValueError("columnar data_json datetime must be timezone-aware")
                    raw[name] = parsed
            self._decoded = raw
        return self._decoded

    def __getitem__(self, key: str) -> Any:
        return self._mapping()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._mapping())

    def __len__(self) -> int:
        return len(self._mapping())


class ColumnarRecordRef:
    """A two-word handle; source data is never retained on the Python object."""

    __slots__ = ("store", "index")

    def __init__(self, store: MMapTemporalRecordStore, index: int) -> None:
        self.store = store
        self.index = index

    @property
    def kind(self) -> str:
        return self.store.strings["kind"][self.index]

    @property
    def entity(self) -> str:
        return self.store.strings["entity"][self.index]

    @property
    def row_id(self) -> str:
        return self.store.strings["row_id"][self.index]

    @property
    def source_id(self) -> str:
        return self.store.strings["source_id"][self.index]

    @property
    def day(self) -> date:
        return date.fromordinal(int(self.store.arrays["day_ordinal"][self.index]))

    @property
    def event_at(self) -> datetime:
        return _us_to_datetime(int(self.store.arrays["event_us"][self.index]))

    @property
    def available_at(self) -> datetime:
        return _us_to_datetime(int(self.store.arrays["available_us"][self.index]))

    @property
    def ingested_at(self) -> datetime:
        return _us_to_datetime(int(self.store.arrays["ingested_us"][self.index]))

    @property
    def valid_from(self) -> datetime:
        return _us_to_datetime(int(self.store.arrays["valid_from_us"][self.index]))

    @property
    def valid_to(self) -> datetime | None:
        value = int(self.store.arrays["valid_to_us"][self.index])
        return None if value == _MISSING_TIME else _us_to_datetime(value)

    @property
    def data(self) -> Mapping[str, Any]:
        return _RecordDataView(self.store, self.index)

    @property
    def digest(self) -> int:
        return int(self.store.strings["digest_hex"][self.index], 16)

    @property
    def values(self) -> Array:
        return np.asarray(self.store.arrays["values"][self.index])

    @property
    def box_values(self) -> Array:
        length = int(self.store.arrays["box_value_lengths"][self.index])
        return np.asarray(self.store.arrays["box_values"][self.index, :length])


class ColumnarRecordSelection(Sequence[ColumnarRecordRef]):
    """One sample-scoped integer view over active memory-mapped records."""

    __slots__ = ("store", "indices")

    def __init__(self, store: MMapTemporalRecordStore, indices: Array) -> None:
        self.store = store
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    @overload
    def __getitem__(self, index: int) -> ColumnarRecordRef: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[ColumnarRecordRef]: ...

    def __getitem__(self, index: int | slice) -> ColumnarRecordRef | Sequence[ColumnarRecordRef]:
        if isinstance(index, slice):
            return ColumnarRecordSelection(self.store, self.indices[index])
        return self.store.ref(int(self.indices[index]))


class MMapTemporalRecordStore:
    """Verified read-only memory maps plus cutoff/key prefix operations."""

    def __init__(
        self,
        directory: Path,
        descriptor: Mapping[str, Any],
        *,
        label_year_ceiling: int | None,
        trusted_file_attestation: Mapping[str, tuple[int, int, int, int]] | None = None,
    ) -> None:
        if descriptor.get("schema_version") != COLUMNAR_RECORD_STORE_SCHEMA_VERSION:
            raise ValueError("unsupported temporal columnar record store")
        if (
            descriptor.get("backend") != "numpy_npy_mmap_utf8_blob_v1"
            or descriptor.get("pickle_allowed") is not False
            or descriptor.get("read_mode") != "read_only_mmap"
            or descriptor.get("cutoff_index")
            != "sorted_key_change_stream_with_searchsorted_prefix"
        ):
            raise ValueError("temporal columnar backend contract is incomplete")
        columns = descriptor.get("columns")
        if not isinstance(columns, Mapping):
            raise ValueError("temporal columnar columns must be an object")
        self.directory = directory
        self._trusted_file_attestation = (
            dict(trusted_file_attestation) if trusted_file_attestation is not None else None
        )
        self.file_attestation: dict[str, tuple[int, int, int, int]] = {}
        self.arrays: dict[str, Array] = {}
        self.strings: dict[str, _UTF8Column] = {}
        for name, raw in columns.items():
            if not isinstance(name, str) or not isinstance(raw, Mapping):
                raise ValueError("invalid temporal column descriptor")
            if raw.get("encoding") == "utf8_blob_offsets_v1":
                blob = self._open_array(raw.get("blob"), f"{name}.blob")
                offsets = self._open_array(raw.get("offsets"), f"{name}.offsets")
                if blob.dtype != np.uint8 or offsets.dtype != np.int64 or offsets.ndim != 1:
                    raise ValueError(f"invalid UTF-8 column storage: {name}")
                if len(offsets) < 1 or int(offsets[0]) != 0 or int(offsets[-1]) != len(blob):
                    raise ValueError(f"invalid UTF-8 offsets: {name}")
                self.strings[name] = _UTF8Column(blob, offsets)
            else:
                self.arrays[name] = self._open_array(raw, name)
        self.record_count = int(descriptor.get("record_count", -1))
        self.key_count = int(descriptor.get("logical_key_count", -1))
        if self.record_count < 0 or self.key_count < 0:
            raise ValueError("invalid temporal columnar row counts")
        for name in (
            "kind",
            "entity",
            "row_id",
            "source_id",
            "record_key",
            "digest_hex",
            "data_json",
        ):
            if name not in self.strings or len(self.strings[name]) != self.record_count:
                raise ValueError(f"temporal columnar string count disagrees: {name}")
        for name in (
            "day_ordinal",
            "event_us",
            "available_us",
            "ingested_us",
            "valid_from_us",
            "valid_to_us",
            "values",
            "box_values",
            "box_value_lengths",
        ):
            if name not in self.arrays or len(self.arrays[name]) != self.record_count:
                raise ValueError(f"temporal columnar numeric count disagrees: {name}")
        if len(self.strings.get("key_kind", ())) != self.key_count or len(
            self.strings.get("key_entity", ())
        ) != self.key_count:
            raise ValueError("temporal columnar key count disagrees")
        key_offsets_array = self.arrays.get("key_offsets")
        key_indices_array = self.arrays.get("key_record_indices")
        if (
            key_offsets_array is None
            or key_indices_array is None
            or key_offsets_array.shape != (self.key_count + 1,)
            or key_indices_array.shape != (self.record_count,)
            or int(key_offsets_array[0]) != 0
            or int(key_offsets_array[-1]) != self.record_count
        ):
            raise ValueError("temporal columnar key index is invalid")
        schedule_days_raw = self.arrays.get("schedule_day_ordinal")
        schedule_keys_raw = self.arrays.get("schedule_key_index")
        if (
            schedule_days_raw is None
            or schedule_keys_raw is None
            or schedule_days_raw.ndim != 1
            or schedule_days_raw.shape != schedule_keys_raw.shape
            or np.any(schedule_days_raw[1:] < schedule_days_raw[:-1])
            or np.any(schedule_keys_raw < 0)
            or np.any(schedule_keys_raw >= self.key_count)
        ):
            raise ValueError("temporal columnar cutoff schedule is invalid")
        self.label_year_ceiling = label_year_ceiling
        self.allowed_day_ordinal = (
            date(label_year_ceiling, 12, 31).toordinal()
            if label_year_ceiling is not None
            else date.max.toordinal()
        )
        self.allowed_record_count = int(
            np.count_nonzero(self.arrays["day_ordinal"] <= self.allowed_day_ordinal)
        )

    def _open_array(self, raw: Any, context: str) -> Array:
        if not isinstance(raw, Mapping):
            raise ValueError(f"missing temporal column descriptor: {context}")
        relative = raw.get("file")
        if not isinstance(relative, str):
            raise ValueError(f"invalid temporal column path: {context}")
        path = (self.directory / relative).resolve()
        if self.directory not in path.parents or not path.is_file():
            raise ValueError(f"temporal column path escapes or is missing: {context}")
        stat = path.stat()
        attestation = (
            int(stat.st_size),
            int(stat.st_mtime_ns),
            int(stat.st_dev),
            int(stat.st_ino),
        )
        expected_bytes = raw.get("bytes")
        if isinstance(expected_bytes, bool) or not isinstance(expected_bytes, int):
            raise ValueError(f"temporal column byte count is invalid: {context}")
        if stat.st_size != expected_bytes:
            raise ValueError(f"temporal column byte count mismatch: {context}")
        if self._trusted_file_attestation is None:
            # The parent process verifies content once. Spawned workers receive
            # this exact file identity attestation and reopen mmap views without
            # rescanning every multi-gigabyte column.
            if _sha256_file(path) != raw.get("sha256"):
                raise ValueError(f"temporal column checksum mismatch: {context}")
        elif self._trusted_file_attestation.get(relative) != attestation:
            raise ValueError(f"temporal column changed after parent verification: {context}")
        self.file_attestation[relative] = attestation
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if not isinstance(array, np.memmap) or array.dtype.hasobject:
            raise ValueError(f"temporal column is not a pickle-free mmap: {context}")
        if array.dtype.str != raw.get("dtype") or list(array.shape) != raw.get("shape"):
            raise ValueError(f"temporal column metadata mismatch: {context}")
        return array

    @property
    def mmap_backed(self) -> bool:
        arrays = list(self.arrays.values()) + [
            value for column in self.strings.values() for value in (column.blob, column.offsets)
        ]
        return bool(arrays) and all(isinstance(array, np.memmap) for array in arrays)

    def ref(self, index: int) -> ColumnarRecordRef:
        if not 0 <= index < self.record_count:
            raise IndexError(index)
        if int(self.arrays["day_ordinal"][index]) > self.allowed_day_ordinal:
            raise PermissionError("record is sealed by label_year_ceiling")
        return ColumnarRecordRef(self, index)

    def schedule_stop(self, day: date) -> int:
        return int(
            np.searchsorted(
                self.arrays["schedule_day_ordinal"], day.toordinal(), side="right"
            )
        )

    def changed_keys(self, start: int, stop: int) -> Array:
        return np.unique(self.arrays["schedule_key_index"][start:stop])

    def select_record(self, key_index: int, day: date) -> int:
        offsets = self.arrays["key_offsets"]
        ordered = self.arrays["key_record_indices"][
            int(offsets[key_index]) : int(offsets[key_index + 1])
        ]
        cutoff_us = _datetime_to_us(datetime.combine(day, time.min, tzinfo=_KST))
        for raw_index in ordered[::-1]:
            index = int(raw_index)
            if (
                int(self.arrays["day_ordinal"][index]) < day.toordinal()
                and int(self.arrays["day_ordinal"][index]) <= self.allowed_day_ordinal
                and int(self.arrays["available_us"][index]) <= cutoff_us
                and int(self.arrays["ingested_us"][index]) <= cutoff_us
                and int(self.arrays["valid_from_us"][index]) <= cutoff_us
            ):
                # Match the canonical History rule exactly: select the latest
                # eligible version by rank first.  If that selected version is
                # no longer valid, the logical entity is absent rather than
                # silently falling back to an older superseded version.
                valid_to = int(self.arrays["valid_to_us"][index])
                return index if valid_to == _MISSING_TIME or cutoff_us < valid_to else -1
        return -1

    def record_key(self, index: int) -> str:
        return self.strings["record_key"][index]

    def key(self, key_index: int) -> tuple[str, str]:
        return self.strings["key_kind"][key_index], self.strings["key_entity"][key_index]


__all__ = [
    "COLUMNAR_RECORD_STORE_SCHEMA_VERSION",
    "ColumnarRecordRef",
    "ColumnarRecordSelection",
    "MMapTemporalRecordStore",
    "write_columnar_record_store",
]
