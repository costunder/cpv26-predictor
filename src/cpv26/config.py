"""Environment-backed runtime settings with Linux-safe paths."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_path(raw_value: str, root: Path) -> Path:
    candidate = Path(raw_value).expanduser()
    return candidate if candidate.is_absolute() else (root / candidate).resolve()


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings loaded from exported shell environment variables."""

    repository_root: Path
    home: Path
    database_path: Path
    timezone: str
    device: str
    random_seed: int
    log_level: str

    @classmethod
    def from_environment(cls, repository_root: Path | None = None) -> Settings:
        root = (repository_root or _repository_root()).resolve()
        home = _resolve_path(os.getenv("CPV26_HOME", "./var"), root)
        database_path = _resolve_path(os.getenv("CPV26_DB_PATH", str(home / "cpv26.duckdb")), root)
        timezone = os.getenv("CPV26_TIMEZONE", "Asia/Seoul")
        try:
            ZoneInfo(timezone)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"Unknown CPV26_TIMEZONE: {timezone}") from exc

        device = os.getenv("CPV26_DEVICE", "auto").lower()
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("CPV26_DEVICE must be one of: auto, cpu, cuda")

        try:
            random_seed = int(os.getenv("CPV26_RANDOM_SEED", "2026"))
        except ValueError as exc:
            raise ValueError("CPV26_RANDOM_SEED must be an integer") from exc

        log_level = os.getenv("CPV26_LOG_LEVEL", "INFO").upper()
        return cls(
            repository_root=root,
            home=home,
            database_path=database_path,
            timezone=timezone,
            device=device,
            random_seed=random_seed,
            log_level=log_level,
        )

    def ensure_runtime_directories(self) -> None:
        self.home.mkdir(parents=True, exist_ok=True)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
