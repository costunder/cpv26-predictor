"""Select annual PBP snapshots without deleting superseded source evidence.

An annual file is a replacement snapshot of that provider's season, not an
incremental stream. Entity-level revision ranking alone cannot represent a PA
or game removed from a later file. Filter superseded sources first instead.
The selection boundary is system knowledge time, not a forecast cutoff: source
availability is retrospectively imputed from its first *remaining* game.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from cpv26.domain import utc_datetime

ANNUAL_SNAPSHOT_POLICY = "annual_snapshot"
_SOURCE_NAME = "slothman3878/kbo_playbyplay"
_COLUMN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?\Z")


def superseded_source_ids_sql(*, knowledge_bound: bool = False) -> str:
    """Return a SELECT of replaced source IDs, with one optional time parameter.

    Existing adapter-v1 imports predate ``snapshot_policy``. Recognize only
    their complete generated source identity/metadata contract, not arbitrary
    similarly named sources. Explicit non-snapshot policies are left alone.
    Ties are stable: later adapter version, then source ID wins.
    """
    knowledge = "AND ingested_at <= ?" if knowledge_bound else ""
    return f"""
        WITH annual_sources AS (
            SELECT source_revision_id, source_name, ingested_at,
                   try_cast(metadata_json ->> 'season' AS INTEGER) AS snapshot_season,
                   try_cast(metadata_json ->> 'adapter_version' AS INTEGER) AS adapter_version
            FROM source_revision
            WHERE source_name = '{_SOURCE_NAME}'
              AND try_cast(metadata_json ->> 'season' AS INTEGER) BETWEEN 2000 AND 2099
              AND try_cast(metadata_json ->> 'adapter_version' AS INTEGER) >= 1
              AND coalesce(metadata_json ->> 'snapshot_policy', '{ANNUAL_SNAPSHOT_POLICY}')
                    = '{ANNUAL_SNAPSHOT_POLICY}'
              AND coalesce(metadata_json ->> 'dataset_revision', '') <> ''
              AND source_revision_id = 'hf-kbo-playbyplay:'
                    || (metadata_json ->> 'dataset_revision') || ':'
                    || (metadata_json ->> 'season') || ':adapter-v'
                    || (metadata_json ->> 'adapter_version')
              {knowledge}
        ), ranked_sources AS (
            SELECT source_revision_id, row_number() OVER (
                PARTITION BY source_name, snapshot_season
                ORDER BY ingested_at DESC, adapter_version DESC, source_revision_id DESC
            ) AS snapshot_rank
            FROM annual_sources
        )
        SELECT source_revision_id FROM ranked_sources WHERE snapshot_rank > 1
    """


def source_snapshot_filter_sql(
    source_column: str = "source_revision_id", *, knowledge_bound: bool = False,
) -> str:
    """SQL predicate applied before logical-row/state filters and ranking."""
    if _COLUMN.fullmatch(source_column) is None:
        raise ValueError("source_column must be a simple column or alias.column")
    outdated = superseded_source_ids_sql(knowledge_bound=knowledge_bound)
    return f"({source_column} IS NULL OR {source_column} NOT IN ({outdated}))"


def superseded_source_ids(connection: Any, knowledge_at: datetime | None = None) -> frozenset[str]:
    """Use the same policy for readers that build records outside SQL."""
    parameters = [] if knowledge_at is None else [
        utc_datetime(knowledge_at, field_name="knowledge_at")
    ]
    rows = connection.execute(
        superseded_source_ids_sql(knowledge_bound=knowledge_at is not None), parameters,
    ).fetchall()
    return frozenset(str(row[0]) for row in rows)
