"""Validate every cached graph and report source-era feature coverage.

Run from an installed project environment, for example:
``python scripts/audit_cross_era_graph.py var/datasets/kbo_graph --output audit.json``.
This reads graph artifacts only; it never opens or changes the canonical database.
Historical name cohorts are uncertain priors, not verified individual identities.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from cpv26.data.kbo_graph_dataset import GraphDay, KBOGraphDataset

_BOX_BLOCKS = (
    "player_box_batting_features",
    "player_box_pitching_features",
    "team_box_batting_features",
    "team_box_pitching_features",
)
_BLOCKS = (*_BOX_BLOCKS, "player_batting_features", "player_pitching_features")


def _new_season() -> dict[str, Any]:
    return {
        "days": 0,
        "games": 0,
        "pa_queries": 0,
        "eligible_modern_pa_days": 0,
        "modern_missing_box_blocks": [],
        "feature_blocks": {
            name: {"nonzero_days": 0, "nonzero_rows": 0, "rows": 0} for name in _BLOCKS
        },
        "queries": {},
        "training_targets": {},
        "player_node_days_by_identifier_kind": {},
    }


def _query_coverage(graph: GraphDay, prefix: str, role: str) -> dict[str, Any]:
    players = graph.arrays[f"{prefix}_player_index"]
    teams = graph.arrays[f"{prefix}_team_index"]
    values = graph.arrays[f"player_box_{role}_features"][players]
    field_count = (values.shape[1] - 1) // 2
    # A recency value alone is not evidence that any field was observed.
    observed = np.any(values[:, field_count : 2 * field_count] > 0, axis=1)
    result: dict[str, Any] = {
        "queries": len(players),
        "queries_with_observed_history": int(observed.sum()),
        "multiplayer_team_groups": 0,
        "team_groups_with_two_observed_priors": 0,
        "team_groups_with_distinct_observed_priors": 0,
        "max_distinct_observed_priors": 0,
        "distinct_prior_examples": [],
    }
    for team in np.unique(teams):
        selected = teams == team
        if len(np.unique(players[selected])) < 2:
            continue
        result["multiplayer_team_groups"] += 1
        known_players = np.unique(players[selected & observed])
        if len(known_players) < 2:
            continue
        result["team_groups_with_two_observed_priors"] += 1
        known_values = graph.arrays[f"player_box_{role}_features"][known_players, :-1]
        # Exclude recency so different dates alone cannot satisfy this check.
        distinct = len(np.unique(known_values, axis=0))
        result["max_distinct_observed_priors"] = max(
            result["max_distinct_observed_priors"], distinct
        )
        if distinct > 1:
            result["team_groups_with_distinct_observed_priors"] += 1
            if len(result["distinct_prior_examples"]) < 3:
                result["distinct_prior_examples"].append(
                    {
                        "day": graph.day_id,
                        "team_id": graph.team_ids[int(team)],
                        "query_nodes_with_history": len(known_players),
                        "distinct_feature_vectors": distinct,
                    }
                )
    return result


def _add_query_coverage(total: dict[str, Any], current: dict[str, Any]) -> None:
    for name, value in current.items():
        if name == "distinct_prior_examples":
            total.setdefault(name, []).extend(value[: 3 - len(total.get(name, []))])
        elif name == "max_distinct_observed_priors":
            total[name] = max(total.get(name, 0), value)
        else:
            total[name] = total.get(name, 0) + value


def _identifier_kind(identifier: str) -> str:
    if identifier.startswith("kbo-name-team-role-cohort:"):
        return "uncertain_name_team_role_cohort"
    if identifier.startswith("kbo-team-role-prior:"):
        return "team_role_fallback"
    if identifier.startswith(("kbo-box-observation:", "observed:")):
        return "distinct_source_observation"
    return "other_source_player_id_not_independently_verified_by_this_audit"


def audit_dataset(directory: str | Path) -> dict[str, Any]:
    """Check file integrity/PIT and return reproducible coverage, not model scores."""
    dataset = KBOGraphDataset(directory)
    manifest = dataset.manifest
    version = int(manifest["dataset_version"])
    entries = {entry["day"]: entry for entry in manifest["days"]}
    seasons: dict[str, dict[str, Any]] = {}
    gaps: list[dict[str, Any]] = []
    count_mismatches: list[dict[str, Any]] = []
    arrays_checked = routes_checked = 0
    for day in dataset.days():
        # load_day validates the exact NPZ SHA, dimensions, numeric finiteness,
        # route publication cutoff, indices and count-label observation masks.
        graph = dataset.load_day(day)
        arrays_checked += len(graph.arrays)
        for route in graph.routes.values():
            age = route["event_age_seconds"]
            delay = route["publication_delay_seconds"]
            if np.any(age <= 0) or np.any(delay < 0):
                raise ValueError(f"invalid past-only route timing: {graph.day_id}")
            routes_checked += len(age)
        season = seasons.setdefault(str(day.year), _new_season())
        season["days"] += 1
        season["games"] += len(graph.match_targets)
        season["pa_queries"] += len(graph.pa_targets)
        actual_counts = {
            "live_hit_queries": len(graph.live_hit_query_ids),
            "live_hit_unknown_pa_queries": int(np.sum(graph.live_hit_pa < 0)),
            "pa_queries": len(graph.pa_targets),
            "box_pa_queries": len(graph.box_pa_counts),
            "box_pa_outcomes": int(graph.box_pa_counts.sum()),
            "box_pitch_queries": len(graph.box_pitch_targets),
            "box_pitch_observed_counts": int(graph.box_pitch_mask.sum()),
            "pa_derived_batting_queries": sum(
                str(key).startswith("observed-pa-box:") for key in graph.box_pa_query_ids
            ),
            "pa_derived_pitching_queries": sum(
                str(key).startswith("observed-pa-box:") for key in graph.box_pitch_query_ids
            ),
        }
        for name, actual in actual_counts.items():
            season["training_targets"][name] = season["training_targets"].get(name, 0) + actual
            if version >= 5 and entries[graph.day_id].get(name) != actual:
                count_mismatches.append(
                    {
                        "scope": graph.day_id,
                        "field": name,
                        "actual": actual,
                        "manifest": entries[graph.day_id].get(name),
                    }
                )
        for name in _BLOCKS:
            values = graph.arrays[name]
            nonzero = int(np.any(values != 0, axis=1).sum())
            block = season["feature_blocks"][name]
            block["nonzero_days"] += int(nonzero > 0)
            block["nonzero_rows"] += nonzero
            block["rows"] += len(values)
        for identifier in graph.player_ids:
            kinds = season["player_node_days_by_identifier_kind"]
            kind = _identifier_kind(identifier)
            kinds[kind] = kinds.get(kind, 0) + 1
        for prefix, role in (
            ("live_hit", "batting"),
            ("box_pa", "batting"),
            ("box_pitch", "pitching"),
        ):
            total = season["queries"].setdefault(prefix, {})
            _add_query_coverage(total, _query_coverage(graph, prefix, role))
        eligible = (
            day.year >= 2023
            and len(graph.pa_targets) > 0
            and np.any(graph.player_batting_features[:, 0] > 0)
            and np.any(graph.player_pitching_features[:, 0] > 0)
        )
        if eligible:
            season["eligible_modern_pa_days"] += 1
            missing = [name for name in _BOX_BLOCKS if not graph.arrays[name].any()]
            if missing:
                gap = {"day": graph.day_id, "missing_blocks": missing}
                season["modern_missing_box_blocks"].append(gap)
                gaps.append(gap)
    totals: dict[str, int] = {}
    season_entries = {str(row["season"]): row for row in manifest.get("season_coverage", [])}
    for year, season in seasons.items():
        for name, actual in season["training_targets"].items():
            totals[name] = totals.get(name, 0) + actual
            if version >= 5 and season_entries.get(year, {}).get(name) != actual:
                count_mismatches.append(
                    {
                        "scope": year,
                        "field": name,
                        "actual": actual,
                        "manifest": season_entries.get(year, {}).get(name),
                    }
                )
    reported_totals = manifest.get("label_quality", {}).get("training_targets", {})
    if version >= 5:
        for name, actual in totals.items():
            if reported_totals.get(name) != actual:
                count_mismatches.append(
                    {
                        "scope": "all",
                        "field": name,
                        "actual": actual,
                        "manifest": reported_totals.get(name),
                    }
                )
    identity_audit = manifest.get("label_quality", {}).get("historical_boxscore_identity")
    failures = gaps if version >= 4 else []
    return {
        "dataset_directory": str(dataset.directory),
        "dataset_version": version,
        "fingerprint": manifest["fingerprint"],
        "validation": {
            "passed": not failures and not count_mismatches,
            "sha_verified_npz_files": len(dataset.days()),
            "arrays_checked": arrays_checked,
            "past_only_route_edges_checked": routes_checked,
            "modern_zero_block_days": len(gaps),
            "v4_bridge_requirement_enforced": version >= 4,
            "v4_bridge_failure_days": len(failures),
            "v5_all_source_target_counts_enforced": version >= 5,
            "target_count_mismatches": len(count_mismatches),
        },
        "training_targets_from_arrays": totals,
        "target_count_mismatch_details": count_mismatches,
        "raw_archive_boxscore_from_manifest": manifest.get("label_quality", {}).get(
            "raw_archive_boxscore"
        ),
        "source_identity_counts_from_manifest": identity_audit,
        "identity_policy": manifest.get("policies", {}).get("boxscore_identity"),
        "identity_count_note": (
            "Manifest source-row counts are reported only when explicitly provided. "
            "Node-day identifier categories below are not unique players or verified identities. "
            "Name/team/role cohorts are uncertain priors, not linked career identities."
        ),
        "source_provenance_count": len(manifest.get("source_provenance", [])),
        "season_coverage_from_manifest": manifest.get("season_coverage"),
        "seasons": seasons,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graph", type=Path, help="Completed graph cache directory")
    parser.add_argument("--output", type=Path, help="Also write the JSON audit to this path")
    args = parser.parse_args()
    if args.output is not None:
        graph_directory = args.graph.expanduser().resolve()
        destination = args.output.expanduser().resolve()
        if destination == graph_directory or graph_directory in destination.parents:
            parser.error(
                "--output must be outside the graph directory; input artifacts are read-only"
            )
        args.output = destination
    report = audit_dataset(args.graph)
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    "output": str(args.output.resolve()),
                    "dataset_version": report["dataset_version"],
                    "validation": report["validation"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(encoded)
    return 0 if report["validation"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
