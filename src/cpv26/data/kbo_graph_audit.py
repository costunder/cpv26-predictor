"""Read-only capacity and reachability audit for cached KBO graph datasets.

The graph cache stores one edge per endpoint pair and rolling cutoff day.  This
module makes that aggregation visible without rebuilding or mutating the cache.
All counts ending in ``_occurrences`` count the same node/edge again when it is
present on another cutoff day; ``unique_*`` counts use stable entity IDs within
the reported scope.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from cpv26.data.kbo_graph_dataset import ROUTE_METADATA, GraphDay, KBOGraphDataset

Array = NDArray[Any]
Node = tuple[str, str]
Edge = tuple[Node, Node]

_QUERY_SPECS: dict[str, tuple[tuple[str, str, str], ...]] = {
    "match": (
        ("home_team", "match_home_team_index", "team"),
        ("away_team", "match_away_team_index", "team"),
    ),
    "live_hit": (
        ("player", "live_hit_player_index", "player"),
        ("team", "live_hit_team_index", "team"),
        ("opponent", "live_hit_opponent_index", "team"),
    ),
    "pa": (
        ("batter", "pa_batter_index", "player"),
        ("pitcher", "pa_pitcher_index", "player"),
    ),
    "box_pa": (
        ("player", "box_pa_player_index", "player"),
        ("team", "box_pa_team_index", "team"),
        ("opponent", "box_pa_opponent_index", "team"),
    ),
    "box_pitch": (
        ("player", "box_pitch_player_index", "player"),
        ("team", "box_pitch_team_index", "team"),
        ("opponent", "box_pitch_opponent_index", "team"),
    ),
}


@dataclass(frozen=True, slots=True)
class _Reachability:
    relation_degree: int
    neighbor_degree: int
    one_hop_nodes: int
    within_two_hop_nodes: int
    exact_two_hop_nodes: int
    possible_nodes: int


@dataclass(frozen=True, slots=True)
class _BitGraph:
    nodes: tuple[Node, ...]
    positions: dict[Node, int]
    neighbors: dict[Node, int]


@dataclass(slots=True)
class _QueryStats:
    occurrences: int = 0
    unique_nodes: set[Node] = field(default_factory=set)
    isolated: int = 0
    relation_degree_sum: int = 0
    neighbor_degree_sum: int = 0
    relation_degree_histogram: Counter[int] = field(default_factory=Counter)
    neighbor_degree_histogram: Counter[int] = field(default_factory=Counter)
    one_hop_nodes: int = 0
    within_two_hop_nodes: int = 0
    exact_two_hop_nodes: int = 0
    one_hop_coverage_sum: float = 0.0
    two_hop_coverage_sum: float = 0.0
    nodes_with_one_hop: int = 0
    nodes_with_exact_two_hops: int = 0

    def add(self, node: Node, reachability: _Reachability, weight: int) -> None:
        if weight < 1:
            return
        self.occurrences += weight
        self.unique_nodes.add(node)
        if reachability.relation_degree == 0:
            self.isolated += weight
        self.relation_degree_sum += reachability.relation_degree * weight
        self.neighbor_degree_sum += reachability.neighbor_degree * weight
        self.relation_degree_histogram[reachability.relation_degree] += weight
        self.neighbor_degree_histogram[reachability.neighbor_degree] += weight
        self.one_hop_nodes += reachability.one_hop_nodes * weight
        self.within_two_hop_nodes += reachability.within_two_hop_nodes * weight
        self.exact_two_hop_nodes += reachability.exact_two_hop_nodes * weight
        if reachability.possible_nodes:
            self.one_hop_coverage_sum += (
                reachability.one_hop_nodes / reachability.possible_nodes
            ) * weight
            self.two_hop_coverage_sum += (
                reachability.within_two_hop_nodes / reachability.possible_nodes
            ) * weight
        if reachability.one_hop_nodes:
            self.nodes_with_one_hop += weight
        if reachability.exact_two_hop_nodes:
            self.nodes_with_exact_two_hops += weight

    def report(self) -> dict[str, Any]:
        count = self.occurrences
        return {
            "query_node_occurrences": count,
            "unique_query_nodes": len(self.unique_nodes),
            "isolated_query_node_occurrences": self.isolated,
            "isolation_fraction": _ratio(self.isolated, count),
            "degree": _degree_report(
                self.relation_degree_histogram, self.relation_degree_sum, count
            ),
            "unique_neighbor_degree": _degree_report(
                self.neighbor_degree_histogram, self.neighbor_degree_sum, count
            ),
            "one_hop": {
                "reachable_node_occurrences": self.one_hop_nodes,
                "mean_reachable_nodes": _ratio(self.one_hop_nodes, count),
                "mean_graph_coverage": _ratio(self.one_hop_coverage_sum, count),
                "query_node_occurrences_with_reach": self.nodes_with_one_hop,
                "query_node_reach_fraction": _ratio(self.nodes_with_one_hop, count),
            },
            "within_two_hops": {
                "reachable_node_occurrences": self.within_two_hop_nodes,
                "mean_reachable_nodes": _ratio(self.within_two_hop_nodes, count),
                "mean_graph_coverage": _ratio(self.two_hop_coverage_sum, count),
                "exact_second_hop_node_occurrences": self.exact_two_hop_nodes,
                "mean_exact_second_hop_nodes": _ratio(self.exact_two_hop_nodes, count),
                "query_node_occurrences_with_exact_second_hop": (
                    self.nodes_with_exact_two_hops
                ),
                "exact_second_hop_reach_fraction": _ratio(
                    self.nodes_with_exact_two_hops, count
                ),
            },
            # Short aliases make the two headline coverage numbers easy to consume.
            "one_hop_coverage": _ratio(self.one_hop_coverage_sum, count),
            "two_hop_coverage": _ratio(self.two_hop_coverage_sum, count),
        }


@dataclass(slots=True)
class _RouteStats:
    source_type: str
    destination_type: str
    stored_edges: int = 0
    unique_edge_occurrences: int = 0
    duplicate_edges: int = 0
    unique_pairs: set[Edge] = field(default_factory=set)
    source_node_occurrences: int = 0
    destination_node_occurrences: int = 0
    incident_node_occurrences: int = 0
    unique_source_nodes: set[Node] = field(default_factory=set)
    unique_destination_nodes: set[Node] = field(default_factory=set)
    unique_incident_nodes: set[Node] = field(default_factory=set)
    encoded_raw_events: int = 0
    zero_count_edges: int = 0
    query_nodes: _QueryStats = field(default_factory=_QueryStats)

    def report(self) -> dict[str, Any]:
        raw_complete = self.zero_count_edges == 0
        raw_events: int | None = self.encoded_raw_events if raw_complete else None
        edge_count = self.unique_edge_occurrences
        compression = _compression(raw_events, edge_count)
        return {
            "source_type": self.source_type,
            "destination_type": self.destination_type,
            "stored_edge_occurrences": self.stored_edges,
            "unique_edge_occurrences": edge_count,
            "duplicate_edge_occurrences": self.duplicate_edges,
            "unique_endpoint_pairs": len(self.unique_pairs),
            "source_node_occurrences": self.source_node_occurrences,
            "destination_node_occurrences": self.destination_node_occurrences,
            "incident_node_occurrences": self.incident_node_occurrences,
            "unique_source_nodes": len(self.unique_source_nodes),
            "unique_destination_nodes": len(self.unique_destination_nodes),
            "unique_incident_nodes": len(self.unique_incident_nodes),
            "encoded_raw_event_occurrences": raw_events,
            "encoded_raw_event_occurrences_lower_bound": self.encoded_raw_events,
            "zero_encoded_count_edges": self.zero_count_edges,
            "history_compression": compression,
            "query_nodes": self.query_nodes.report(),
            # Compact count aliases; their occurrence semantics are defined at top level.
            "nodes": self.incident_node_occurrences,
            "edges": edge_count,
        }


@dataclass(slots=True)
class _ScopeStats:
    days: int = 0
    date_start: str | None = None
    date_end: str | None = None
    node_occurrences: Counter[str] = field(default_factory=Counter)
    unique_nodes: dict[str, set[str]] = field(
        default_factory=lambda: {"player": set(), "team": set()}
    )
    routes: dict[str, _RouteStats] = field(default_factory=dict)
    queries: dict[str, _QueryStats] = field(
        default_factory=lambda: {name: _QueryStats() for name in _QUERY_SPECS}
    )
    query_roles: dict[str, dict[str, _QueryStats]] = field(
        default_factory=lambda: {
            task: {role: _QueryStats() for role, _, _ in specs}
            for task, specs in _QUERY_SPECS.items()
        }
    )
    all_queries: _QueryStats = field(default_factory=_QueryStats)
    raw_history_rows: int = 0
    history_rows_available_days: int = 0
    history_rows_missing_days: int = 0

    def report(self, *, year: int | None = None) -> dict[str, Any]:
        route_reports = {name: self.routes[name].report() for name in sorted(self.routes)}
        unique_edges = sum(
            route.unique_edge_occurrences for route in self.routes.values()
        )
        observed_relation_events = sum(
            route.encoded_raw_events for route in self.routes.values()
        )
        relation_events_complete = all(
            route.zero_count_edges == 0 for route in self.routes.values()
        )
        exact_relation_events = (
            observed_relation_events if relation_events_complete else None
        )
        exact_history_rows: int | None = (
            self.raw_history_rows if self.history_rows_missing_days == 0 else None
        )
        nodes = {
            kind: {
                "occurrences": int(self.node_occurrences[kind]),
                "unique": len(self.unique_nodes[kind]),
            }
            for kind in ("player", "team")
        }
        report: dict[str, Any] = {
            "days": self.days,
            "date_start": self.date_start,
            "date_end": self.date_end,
            "nodes": nodes,
            "node_occurrences": {
                kind: int(self.node_occurrences[kind]) for kind in ("player", "team")
            },
            "unique_nodes": {
                kind: len(self.unique_nodes[kind]) for kind in ("player", "team")
            },
            "routes": route_reports,
            "history_compression": {
                **_compression(exact_relation_events, unique_edges),
                "raw_relation_event_occurrences": exact_relation_events,
                "observed_raw_relation_event_occurrences": observed_relation_events,
                "relation_event_counts_complete": relation_events_complete,
                "raw_history_row_occurrences": exact_history_rows,
                "observed_raw_history_row_occurrences": self.raw_history_rows,
                "history_rows_available_days": self.history_rows_available_days,
                "history_rows_missing_days": self.history_rows_missing_days,
                "unique_edge_occurrences": unique_edges,
            },
            "queries": {
                task: {
                    **self.queries[task].report(),
                    "by_role": {
                        role: stats.report()
                        for role, stats in sorted(self.query_roles[task].items())
                    },
                }
                for task in _QUERY_SPECS
            },
            "all_queries": self.all_queries.report(),
        }
        if year is not None:
            report = {"year": year, **report}
        return report


def audit_kbo_graph_dataset(
    dataset_or_directory: KBOGraphDataset | str | Path,
    *,
    start_day: date | str | None = None,
    end_day: date | str | None = None,
) -> dict[str, Any]:
    """Return a JSON-serializable, read-only audit of a v2--v5 graph cache.

    ``history_rows`` is read from each manifest day.  Route-local raw event
    counts are decoded from the first documented route feature (log-scaled PA
    or game count); an edge whose encoded count is zero makes that route's raw
    compression ratio unknown instead of silently inventing an event count.

    Hop coverage is computed on the union of the reviewed, bidirectional route
    channels for each cutoff day.  Query node occurrences include every model
    endpoint (for example both home and away team for one match query), so the
    report also breaks them out by task role.
    """

    if isinstance(dataset_or_directory, KBOGraphDataset):
        dataset = dataset_or_directory
    elif isinstance(dataset_or_directory, (str, Path)):
        dataset = KBOGraphDataset(dataset_or_directory)
    else:
        raise TypeError("expected a KBOGraphDataset or dataset directory")
    first = date.fromisoformat(start_day) if isinstance(start_day, str) else start_day
    last = date.fromisoformat(end_day) if isinstance(end_day, str) else end_day
    if last is None:
        raise ValueError("end_day is required so held-out test dates are not audited accidentally")
    if first is not None and first > last:
        raise ValueError("start_day must not exceed end_day")

    version = dataset.manifest.get("dataset_version")
    if version not in (2, 3, 4, 5):
        raise ValueError("graph audit supports KBO graph dataset versions 2 through 5")

    manifest_entries = dataset.manifest.get("days")
    if not isinstance(manifest_entries, list):
        raise ValueError("KBO graph manifest days must be a list")
    entries: dict[str, dict[str, Any]] = {}
    for raw_entry in manifest_entries:
        if not isinstance(raw_entry, dict) or not isinstance(raw_entry.get("day"), str):
            raise ValueError("KBO graph manifest contains an invalid day entry")
        entries[raw_entry["day"]] = raw_entry

    total = _new_scope()
    years: dict[int, _ScopeStats] = {}
    selected_days = tuple(
        day
        for day in dataset.days()
        if (first is None or day >= first) and day <= last
    )
    if not selected_days:
        raise ValueError("no graph days fall inside the requested audit range")
    for day in selected_days:
        graph = dataset.load_day(day)
        day_id = day.isoformat()
        entry = entries.get(day_id)
        if entry is None:
            raise ValueError(f"KBO graph manifest is missing day metadata: {day_id}")
        year_scope = years.setdefault(day.year, _new_scope())
        _add_day(graph, entry, (total, year_scope))

    year_reports = [years[year].report(year=year) for year in sorted(years)]
    return {
        "schema_version": 1,
        "dataset_version": int(version),
        "days": total.days,
        "date_start": total.date_start,
        "date_end": total.date_end,
        "definitions": {
            "occurrences": (
                "Nodes and endpoint-pair edges are counted once per cutoff day; the same "
                "entity or pair can occur again on another day."
            ),
            "unique": "Stable entity IDs or typed endpoint-ID pairs within the report scope.",
            "degree": (
                "Number of incident reviewed route edges; parallel relation types count "
                "separately. unique_neighbor_degree collapses parallel relations."
            ),
            "one_hop_coverage": (
                "Mean fraction of other nodes on the same cutoff day reachable in one hop."
            ),
            "two_hop_coverage": (
                "Mean fraction of other nodes on the same cutoff day reachable within two hops."
            ),
            "history_compression": (
                "Relation events decoded route-by-route are compared with stored endpoint-pair "
                "edges. Manifest history_rows are reported separately and never used as the "
                "compression denominator."
            ),
        },
        "totals": total.report(),
        "years": year_reports,
        "by_year": {str(item["year"]): item for item in year_reports},
    }


def analyze_kbo_graph_dataset(
    dataset_or_directory: KBOGraphDataset | str | Path,
    *,
    start_day: date | str | None = None,
    end_day: date | str | None = None,
) -> dict[str, Any]:
    """Alias for :func:`audit_kbo_graph_dataset`."""

    return audit_kbo_graph_dataset(
        dataset_or_directory, start_day=start_day, end_day=end_day
    )


def _new_scope() -> _ScopeStats:
    routes = {
        name: _RouteStats(
            source_type=str(metadata["source_type"]),
            destination_type=str(metadata["destination_type"]),
        )
        for name, metadata in ROUTE_METADATA.items()
    }
    return _ScopeStats(routes=routes)


def _add_day(
    graph: GraphDay,
    manifest_entry: dict[str, Any],
    scopes: tuple[_ScopeStats, _ScopeStats],
) -> None:
    day_id = graph.day_id
    nodes_by_type = {
        "player": tuple(str(value) for value in graph.player_ids),
        "team": tuple(str(value) for value in graph.team_ids),
    }
    all_nodes = {
        (kind, node_id) for kind, node_ids in nodes_by_type.items() for node_id in node_ids
    }
    for scope in scopes:
        scope.days += 1
        scope.date_start = day_id if scope.date_start is None else min(scope.date_start, day_id)
        scope.date_end = day_id if scope.date_end is None else max(scope.date_end, day_id)
        for kind, node_ids in nodes_by_type.items():
            scope.node_occurrences[kind] += len(node_ids)
            scope.unique_nodes[kind].update(node_ids)
        _add_history_rows(scope, manifest_entry, day_id)

    adjacency: dict[Node, set[Node]] = {node: set() for node in all_nodes}
    relation_degree: Counter[Node] = Counter()
    route_adjacencies: dict[str, dict[Node, set[Node]]] = {}
    route_degrees: dict[str, Counter[Node]] = {}

    for route_name, columns in graph.routes.items():
        metadata = ROUTE_METADATA[route_name]
        source_type = str(metadata["source_type"])
        destination_type = str(metadata["destination_type"])
        source = _indices(columns["source_index"], f"{route_name} source_index")
        destination = _indices(
            columns["destination_index"], f"{route_name} destination_index"
        )
        if source.size != destination.size:
            raise ValueError(f"route endpoint columns disagree: {route_name}")
        features = np.asarray(columns["event_features"])
        if features.ndim != 2 or features.shape[0] != source.size or features.shape[1] < 1:
            raise ValueError(f"route event features are invalid: {route_name}")

        pairs: set[Edge] = set()
        raw_events = 0
        zero_count_edges = 0
        for position, (source_index, destination_index) in enumerate(
            zip(source.tolist(), destination.tolist(), strict=True)
        ):
            source_node = _node(nodes_by_type, source_type, source_index, route_name)
            destination_node = _node(
                nodes_by_type, destination_type, destination_index, route_name
            )
            pairs.add((source_node, destination_node))
            encoded_count = _decode_route_count(route_name, float(features[position, 0]))
            raw_events += encoded_count
            zero_count_edges += int(encoded_count == 0)

        sources = {pair[0] for pair in pairs}
        destinations = {pair[1] for pair in pairs}
        incidents = sources | destinations
        for scope in scopes:
            stats = scope.routes[route_name]
            stats.stored_edges += int(source.size)
            stats.unique_edge_occurrences += len(pairs)
            stats.duplicate_edges += int(source.size) - len(pairs)
            stats.unique_pairs.update(pairs)
            stats.source_node_occurrences += len(sources)
            stats.destination_node_occurrences += len(destinations)
            stats.incident_node_occurrences += len(incidents)
            stats.unique_source_nodes.update(sources)
            stats.unique_destination_nodes.update(destinations)
            stats.unique_incident_nodes.update(incidents)
            stats.encoded_raw_events += raw_events
            stats.zero_count_edges += zero_count_edges

        route_adjacency: dict[Node, set[Node]] = {node: set() for node in all_nodes}
        route_degree: Counter[Node] = Counter()
        bidirectional = bool(metadata.get("bidirectional", False))
        for source_node, destination_node in pairs:
            route_adjacency[source_node].add(destination_node)
            route_degree[source_node] += 1
            if bidirectional:
                route_adjacency[destination_node].add(source_node)
                route_degree[destination_node] += 1
            adjacency[source_node].add(destination_node)
            relation_degree[source_node] += 1
            if bidirectional:
                adjacency[destination_node].add(source_node)
                relation_degree[destination_node] += 1
        route_adjacencies[route_name] = route_adjacency
        route_degrees[route_name] = route_degree

    query_nodes = _query_nodes(graph, nodes_by_type)
    ordered_nodes = tuple(sorted(all_nodes))
    positions = {node: index for index, node in enumerate(ordered_nodes)}
    bit_adjacency = _bit_graph(adjacency, ordered_nodes, positions)
    route_bit_adjacencies = {
        name: _bit_graph(values, ordered_nodes, positions)
        for name, values in route_adjacencies.items()
    }
    reachability_cache: dict[Node, _Reachability] = {}
    route_cache: dict[tuple[str, Node], _Reachability] = {}
    possible_nodes = max(0, len(all_nodes) - 1)
    for task, roles in query_nodes.items():
        for role, occurrences in roles.items():
            for node, weight in occurrences.items():
                reachability = reachability_cache.get(node)
                if reachability is None:
                    reachability = _reachability(
                        node, bit_adjacency, relation_degree, possible_nodes
                    )
                    reachability_cache[node] = reachability
                for scope in scopes:
                    scope.queries[task].add(node, reachability, weight)
                    scope.query_roles[task][role].add(node, reachability, weight)
                    scope.all_queries.add(node, reachability, weight)
                for route_name, route_graph in route_bit_adjacencies.items():
                    cache_key = (route_name, node)
                    route_reachability = route_cache.get(cache_key)
                    if route_reachability is None:
                        route_reachability = _reachability(
                            node,
                            route_graph,
                            route_degrees[route_name],
                            possible_nodes,
                        )
                        route_cache[cache_key] = route_reachability
                    for scope in scopes:
                        scope.routes[route_name].query_nodes.add(
                            node, route_reachability, weight
                        )


def _add_history_rows(scope: _ScopeStats, entry: dict[str, Any], day_id: str) -> None:
    value = entry.get("history_rows")
    if value is None:
        scope.history_rows_missing_days += 1
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"invalid history_rows for graph day {day_id}")
    scope.raw_history_rows += value
    scope.history_rows_available_days += 1


def _query_nodes(
    graph: GraphDay, nodes_by_type: dict[str, tuple[str, ...]]
) -> dict[str, dict[str, Counter[Node]]]:
    result: dict[str, dict[str, Counter[Node]]] = {}
    for task, specs in _QUERY_SPECS.items():
        columns = [
            _indices(graph.arrays.get(array_name, np.empty(0, dtype=np.int64)), array_name)
            for _, array_name, _ in specs
        ]
        sizes = {int(column.size) for column in columns}
        if len(sizes) != 1:
            raise ValueError(f"{task} query index columns disagree")
        roles: dict[str, Counter[Node]] = {}
        for (role, array_name, kind), column in zip(specs, columns, strict=True):
            counter: Counter[Node] = Counter()
            for index in column.tolist():
                counter[_node(nodes_by_type, kind, index, array_name)] += 1
            roles[role] = counter
        result[task] = roles
    return result


def _node(
    nodes_by_type: dict[str, tuple[str, ...]], kind: str, index: int, context: str
) -> Node:
    node_ids = nodes_by_type[kind]
    if index < 0 or index >= len(node_ids):
        raise ValueError(f"node index is out of range in {context}")
    return kind, node_ids[index]


def _indices(raw: Array, name: str) -> NDArray[np.int64]:
    values = np.asarray(raw)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError(f"{name} must be a one-dimensional integer array")
    return values.astype(np.int64, copy=False)


def _decode_route_count(route_name: str, encoded: float) -> int:
    if not math.isfinite(encoded) or encoded < 0:
        raise ValueError(f"route count feature is invalid: {route_name}")
    scale = 20 if route_name == "home_team_game_away_team" else 100
    decoded = math.expm1(encoded * math.log1p(scale))
    nearest = round(decoded)
    if not math.isclose(decoded, nearest, rel_tol=2e-5, abs_tol=2e-5):
        raise ValueError(f"route count feature does not encode an integer: {route_name}")
    return max(0, int(nearest))


def _reachability(
    node: Node,
    adjacency: _BitGraph,
    relation_degree: Counter[Node],
    possible_nodes: int,
) -> _Reachability:
    position = adjacency.positions[node]
    self_bit = 1 << position
    one_hop = adjacency.neighbors[node] & ~self_bit
    within_two = one_hop
    pending = one_hop
    while pending:
        least_significant = pending & -pending
        neighbor_index = least_significant.bit_length() - 1
        within_two |= adjacency.neighbors[adjacency.nodes[neighbor_index]]
        pending ^= least_significant
    within_two &= ~self_bit
    exact_two = within_two & ~one_hop
    return _Reachability(
        relation_degree=int(relation_degree[node]),
        neighbor_degree=one_hop.bit_count(),
        one_hop_nodes=one_hop.bit_count(),
        within_two_hop_nodes=within_two.bit_count(),
        exact_two_hop_nodes=exact_two.bit_count(),
        possible_nodes=possible_nodes,
    )


def _bit_graph(
    adjacency: dict[Node, set[Node]],
    nodes: tuple[Node, ...],
    positions: dict[Node, int],
) -> _BitGraph:
    encoded: dict[Node, int] = {}
    for node in nodes:
        mask = 0
        for neighbor in adjacency[node]:
            mask |= 1 << positions[neighbor]
        encoded[node] = mask
    return _BitGraph(nodes=nodes, positions=positions, neighbors=encoded)


def _compression(raw_rows: int | None, edges: int) -> dict[str, int | float | None]:
    return {
        "raw_row_occurrences": raw_rows,
        "unique_edge_occurrences": edges,
        "raw_rows_per_unique_edge": _ratio(raw_rows, edges),
        "unique_edges_per_raw_row": _ratio(edges, raw_rows),
        "compression_fraction": (
            None if raw_rows is None or raw_rows == 0 else 1.0 - edges / raw_rows
        ),
    }


def _degree_report(
    histogram: Counter[int], total: int, count: int
) -> dict[str, int | float | None]:
    return {
        "min": min(histogram) if histogram else None,
        "max": max(histogram) if histogram else None,
        "mean": _ratio(total, count),
        "p50": _weighted_percentile(histogram, 0.50),
        "p90": _weighted_percentile(histogram, 0.90),
    }


def _weighted_percentile(histogram: Counter[int], quantile: float) -> int | None:
    count = sum(histogram.values())
    if not count:
        return None
    target = max(1, math.ceil(count * quantile))
    cumulative = 0
    for value in sorted(histogram):
        cumulative += histogram[value]
        if cumulative >= target:
            return value
    raise AssertionError("unreachable weighted percentile")


def _ratio(numerator: int | float | None, denominator: int | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return float(numerator / denominator)


__all__ = ["analyze_kbo_graph_dataset", "audit_kbo_graph_dataset"]
