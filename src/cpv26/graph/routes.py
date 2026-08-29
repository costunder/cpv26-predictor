"""Whitelisted atomic routes used by the point-in-time baseball graph.

An atomic route is a two-hop relation through an event or bridge table.  The
registry is deliberately closed: data preparation must select one of the
reviewed routes instead of automatically exposing every foreign-key path in
the relational database.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType

PLAYER_ROLE_NAMES: tuple[str, ...] = (
    "batting",
    "pitching",
    "defense",
    "baserunning",
    "catcher",
)
SHARED_PLAYER_ROLE = "shared"

_PLAYER_ROLE_ALIASES: Mapping[str, str] = MappingProxyType(
    {
        "batter": "batting",
        "pitcher": "pitching",
        "fielder": "defense",
        "runner": "baserunning",
        "lineup_player": SHARED_PLAYER_ROLE,
        "roster_player": SHARED_PLAYER_ROLE,
        "player": SHARED_PLAYER_ROLE,
    }
)


def normalize_endpoint_role(node_type: str, role: str) -> str:
    """Normalize player endpoint aliases to role-state channel names.

    Non-player roles remain semantic route labels. Player endpoints use the
    same names as :class:`cpv26.models.PlayerRole`, plus ``shared`` for routes
    that should consume the common player state.
    """

    normalized = role.strip()
    if node_type != "player":
        return normalized
    normalized = _PLAYER_ROLE_ALIASES.get(normalized, normalized)
    allowed = (*PLAYER_ROLE_NAMES, SHARED_PLAYER_ROLE)
    if normalized not in allowed:
        raise ValueError(
            f"player endpoint role {role!r} is invalid; expected one of: "
            f"{', '.join(allowed)}"
        )
    return normalized


@dataclass(frozen=True, slots=True)
class AtomicRoute:
    """Definition of a reviewed two-hop route.

    ``source_role`` and ``destination_role`` distinguish two endpoints that
    share a physical node type, such as a batter and a pitcher (both players).
    ``bidirectional`` controls whether the model also propagates a reverse
    message along the same observed event.
    """

    name: str
    source_type: str
    event_type: str
    destination_type: str
    source_role: str
    destination_role: str
    bidirectional: bool = True

    def __post_init__(self) -> None:
        values = {
            "name": self.name,
            "source_type": self.source_type,
            "event_type": self.event_type,
            "destination_type": self.destination_type,
            "source_role": self.source_role,
            "destination_role": self.destination_role,
        }
        for field_name, value in values.items():
            if not value or value.strip() != value:
                raise ValueError(f"{field_name} must be a non-empty, trimmed string")
        object.__setattr__(
            self,
            "source_role",
            normalize_endpoint_role(self.source_type, self.source_role),
        )
        object.__setattr__(
            self,
            "destination_role",
            normalize_endpoint_role(self.destination_type, self.destination_role),
        )


class RouteRegistry:
    """Immutable-by-default registry that enforces an explicit route whitelist."""

    def __init__(self, routes: Iterable[AtomicRoute] = ()) -> None:
        route_map: dict[str, AtomicRoute] = {}
        for route in routes:
            if route.name in route_map:
                raise ValueError(f"duplicate atomic route: {route.name}")
            route_map[route.name] = route
        self._routes = route_map

    def register(self, route: AtomicRoute) -> None:
        """Add a reviewed route, rejecting silent replacement."""

        if route.name in self._routes:
            raise ValueError(f"atomic route is already registered: {route.name}")
        self._routes[route.name] = route

    def require(self, name: str) -> AtomicRoute:
        """Return a route or fail with a whitelist-specific error."""

        try:
            return self._routes[name]
        except KeyError as exc:
            allowed = ", ".join(sorted(self._routes)) or "<none>"
            raise KeyError(
                f"atomic route {name!r} is not whitelisted; allowed routes: {allowed}"
            ) from exc

    def validate_endpoints(
        self,
        name: str,
        source_type: str,
        destination_type: str,
    ) -> AtomicRoute:
        route = self.require(name)
        if route.source_type != source_type or route.destination_type != destination_type:
            raise ValueError(
                f"route {name!r} expects {route.source_type!r} -> "
                f"{route.destination_type!r}, got {source_type!r} -> "
                f"{destination_type!r}"
            )
        return route

    @property
    def routes(self) -> Mapping[str, AtomicRoute]:
        return MappingProxyType(self._routes)

    def names(self) -> tuple[str, ...]:
        return tuple(self._routes)

    def __contains__(self, name: object) -> bool:
        return name in self._routes

    def __len__(self) -> int:
        return len(self._routes)

    def __iter__(self) -> Iterator[AtomicRoute]:
        return iter(self._routes.values())

    def copy(self) -> RouteRegistry:
        """Return a mutable copy so callers can add project-approved routes."""

        return RouteRegistry(self._routes.values())


DEFAULT_ROUTES: tuple[AtomicRoute, ...] = (
    AtomicRoute(
        name="batter_pa_pitcher",
        source_type="player",
        event_type="observed_plate_appearance",
        destination_type="player",
        source_role="batting",
        destination_role="pitching",
    ),
    AtomicRoute(
        name="pitcher_pa_catcher",
        source_type="player",
        event_type="observed_plate_appearance",
        destination_type="player",
        source_role="pitching",
        destination_role="catcher",
    ),
    AtomicRoute(
        name="batter_pa_game",
        source_type="player",
        event_type="observed_plate_appearance",
        destination_type="game",
        source_role="batting",
        destination_role="game",
    ),
    AtomicRoute(
        name="player_lineup_game",
        source_type="player",
        event_type="lineup_entry",
        destination_type="game",
        source_role="shared",
        destination_role="game",
    ),
    AtomicRoute(
        name="pitcher_appearance_team_game",
        source_type="player",
        event_type="pitching_appearance",
        destination_type="team_game",
        source_role="pitching",
        destination_role="team_game",
    ),
    AtomicRoute(
        name="player_roster_team_season",
        source_type="player",
        event_type="roster_spell",
        destination_type="team_season",
        source_role="shared",
        destination_role="team_season",
    ),
    AtomicRoute(
        name="home_team_game_away_team",
        source_type="team",
        event_type="game",
        destination_type="team",
        source_role="home_team",
        destination_role="away_team",
    ),
    AtomicRoute(
        name="player_candidate_game",
        source_type="player",
        event_type="player_game_candidate",
        destination_type="game",
        source_role="batting",
        destination_role="game",
    ),
)


def default_route_registry() -> RouteRegistry:
    """Create an independent registry containing the reviewed baseball routes."""

    return RouteRegistry(DEFAULT_ROUTES)
