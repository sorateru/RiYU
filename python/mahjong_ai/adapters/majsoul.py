"""Map an authorized Mahjong Soul observation feed to the common GameEvent.

This module deliberately does not connect to, inspect, or inject into the
Mahjong Soul client.  An operator-authorized collector must emit the small,
versioned observation envelope accepted here.  Everything after that boundary
uses the same Rust ``GameEvent`` and ``GameState`` as logs and the simulator.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

from mahjong_ai.schema import TILE_TYPES, canonical_json

MAJSOUL_EVENT_PROFILE = "majsoul-normalized-event/v1"

_EVENT_NAMES = frozenset(
    {
        "round_start",
        "draw",
        "discard",
        "chi",
        "pon",
        "kan",
        "riichi",
        "dora_reveal",
        "pass",
        "ron",
        "tsumo",
        "draw_game",
        "round_end",
    }
)
_HONORS = {
    "1z": "east",
    "2z": "south",
    "3z": "west",
    "4z": "north",
    "5z": "white",
    "6z": "green",
    "7z": "red",
}
_TILE_PATTERN = re.compile(r"^([0-9])([mps])$")


class MahjongSoulAdapterError(ValueError):
    """A stable adapter error suitable for a WebSocket error response."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class MahjongSoulEvent:
    event_id: str
    sequence: int
    round_id: str
    name: str
    data: dict[str, Any]

    @property
    def fingerprint(self) -> str:
        value = {
            "event_id": self.event_id,
            "sequence": self.sequence,
            "round_id": self.round_id,
            "name": self.name,
            "data": self.data,
        }
        return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def parse_majsoul_event(value: Any) -> MahjongSoulEvent:
    """Parse the normalized adapter envelope and reject silent schema drift."""

    if not isinstance(value, dict):
        raise MahjongSoulAdapterError("invalid_source_event", "event must be a JSON object")
    unknown = set(value) - {"event_id", "sequence", "round_id", "name", "data"}
    if unknown:
        raise MahjongSoulAdapterError(
            "unknown_source_fields", f"unsupported event fields: {sorted(unknown)}"
        )
    event_id = _bounded_text(value.get("event_id"), "event_id")
    round_id = _bounded_text(value.get("round_id"), "round_id")
    sequence = value.get("sequence")
    if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 0:
        raise MahjongSoulAdapterError(
            "invalid_source_sequence", "sequence must be a non-negative integer"
        )
    name = value.get("name")
    if name not in _EVENT_NAMES:
        raise MahjongSoulAdapterError(
            "unsupported_source_event", f"unsupported Mahjong Soul event name: {name!r}"
        )
    data = value.get("data")
    if not isinstance(data, dict):
        raise MahjongSoulAdapterError("invalid_source_event", "data must be a JSON object")
    return MahjongSoulEvent(event_id, sequence, round_id, name, data)


def map_majsoul_event(
    source: MahjongSoulEvent, *, game_sequence: int, observer: int
) -> dict[str, Any]:
    """Convert one service-boundary event to the authoritative GameEvent JSON."""

    if not 0 <= observer <= 3:
        raise MahjongSoulAdapterError("invalid_observer", "observer must be between zero and three")
    if game_sequence < 0:
        raise MahjongSoulAdapterError(
            "invalid_game_sequence", "game_sequence must be non-negative"
        )
    data = source.data
    base: dict[str, Any] = {"sequence": game_sequence, "type": source.name}
    if source.name == "round_start":
        return {
            **base,
            "round": {
                "wind": _enum(data, "wind", {"east", "south", "west", "north"}),
                "hand_number": _integer(data, "hand_number", 1, 4),
                "honba": _integer(data, "honba", 0, 65_535),
                "riichi_sticks": _integer(data, "riichi_sticks", 0, 65_535),
                "dealer": _player(data, "dealer"),
            },
            "scores": _score_array(data, "scores"),
            "initial_hand": _tile_list(data, "initial_hand", expected=13),
            "dora_indicator": _tile(data.get("dora_indicator"), "dora_indicator"),
            "remaining_tiles": _integer(data, "remaining_tiles", 0, 70),
        }
    if source.name == "draw":
        player = _player(data, "player")
        raw_tile = data.get("tile")
        if player != observer and raw_tile is not None:
            raise MahjongSoulAdapterError(
                "secret_information_rejected", "opponent draw tile must be null"
            )
        if player == observer and raw_tile is None:
            raise MahjongSoulAdapterError(
                "observer_tile_required", "observer draw must include tile"
            )
        return {
            **base,
            "player": player,
            "tile": None if raw_tile is None else _tile(raw_tile, "tile"),
            "rinshan": _boolean(data, "rinshan", default=False),
        }
    if source.name == "discard":
        return {
            **base,
            "player": _player(data, "player"),
            "tile": _tile(data.get("tile"), "tile"),
            "tsumogiri": _boolean(data, "tsumogiri"),
            "riichi_declaration": _boolean(data, "riichi_declaration", default=False),
        }
    if source.name in {"chi", "pon", "kan"}:
        kind = source.name if source.name != "kan" else _enum(
            data, "kind", {"daiminkan", "ankan", "shouminkan"}
        )
        called = data.get("called_tile")
        from_player = data.get("from")
        return {
            **base,
            "player": _player(data, "player"),
            "meld": {
                "kind": kind,
                "tiles": _tile_list(data, "tiles", expected=4 if source.name == "kan" else 3),
                "called_tile": None if called is None else _tile(called, "called_tile"),
                "from": None if from_player is None else _player_value(from_player, "from"),
            },
        }
    if source.name == "riichi":
        return {
            **base,
            "player": _player(data, "player"),
            "step": _enum(data, "step", {"declared", "accepted"}),
        }
    if source.name == "dora_reveal":
        return {**base, "tile": _tile(data.get("tile"), "tile")}
    if source.name == "pass":
        return {
            **base,
            "player": _player(data, "player"),
            "ron_available": _boolean(data, "ron_available"),
        }
    if source.name == "ron":
        winners = data.get("winners")
        if not isinstance(winners, list) or not winners:
            raise MahjongSoulAdapterError("invalid_winners", "winners must be a non-empty array")
        return {
            **base,
            "winners": [_player_value(value, "winners") for value in winners],
            "loser": _player(data, "loser"),
            "score_deltas": _score_array(data, "score_deltas"),
        }
    if source.name == "tsumo":
        return {
            **base,
            "winner": _player(data, "winner"),
            "score_deltas": _score_array(data, "score_deltas"),
        }
    if source.name == "draw_game":
        return {
            **base,
            "reason": _enum(
                data,
                "reason",
                {"exhaustive", "nine_terminals", "four_winds", "four_riichi", "four_kan", "triple_ron"},
            ),
            "score_deltas": _score_array(data, "score_deltas"),
        }
    if source.name == "round_end":
        return {
            **base,
            "reason": _enum(data, "reason", {"ron", "tsumo", "draw", "abortive_draw"}),
            "scores": _score_array(data, "scores"),
        }
    raise AssertionError(f"unhandled event {source.name}")


def _bounded_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 128:
        raise MahjongSoulAdapterError(
            "invalid_source_event", f"{field} must be a non-empty string of at most 128 characters"
        )
    return value.strip()


def _integer(data: dict[str, Any], field: str, minimum: int, maximum: int) -> int:
    value = data.get(field)
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not minimum <= value <= maximum
    ):
        raise MahjongSoulAdapterError(
            "invalid_source_event", f"{field} must be an integer in {minimum}..={maximum}"
        )
    return value


def _player(data: dict[str, Any], field: str) -> int:
    return _player_value(data.get(field), field)


def _player_value(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 3:
        raise MahjongSoulAdapterError(
            "invalid_source_event", f"{field} must be a player index between zero and three"
        )
    return value


def _boolean(data: dict[str, Any], field: str, default: bool | None = None) -> bool:
    value = data.get(field, default)
    if not isinstance(value, bool):
        raise MahjongSoulAdapterError("invalid_source_event", f"{field} must be boolean")
    return value


def _enum(data: dict[str, Any], field: str, allowed: set[str]) -> str:
    value = data.get(field)
    if value not in allowed:
        raise MahjongSoulAdapterError(
            "invalid_source_event", f"{field} must be one of {sorted(allowed)}"
        )
    return value


def _score_array(data: dict[str, Any], field: str) -> list[int]:
    value = data.get(field)
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(not isinstance(item, int) or isinstance(item, bool) for item in value)
    ):
        raise MahjongSoulAdapterError(
            "invalid_source_event", f"{field} must contain four integer scores"
        )
    return value


def _tile_list(data: dict[str, Any], field: str, expected: int) -> list[dict[str, Any]]:
    value = data.get(field)
    if not isinstance(value, list) or len(value) != expected:
        raise MahjongSoulAdapterError(
            "invalid_source_event", f"{field} must contain exactly {expected} tiles"
        )
    return [_tile(item, field) for item in value]


def _tile(value: Any, field: str) -> dict[str, Any]:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _HONORS:
            return {"kind": _HONORS[lowered], "red": False}
        match = _TILE_PATTERN.fullmatch(lowered)
        if match:
            rank, suit = match.groups()
            if rank == "0":
                return {"kind": f"{suit}5", "red": True}
            return {"kind": f"{suit}{rank}", "red": False}
        if lowered in TILE_TYPES:
            return {"kind": lowered, "red": False}
    elif isinstance(value, dict):
        if set(value) - {"kind", "red"}:
            raise MahjongSoulAdapterError("invalid_tile", f"{field} tile has unknown fields")
        kind = value.get("kind")
        red = value.get("red", False)
        if kind in TILE_TYPES and isinstance(red, bool):
            if red and kind not in {"m5", "p5", "s5"}:
                raise MahjongSoulAdapterError("invalid_tile", f"{kind} cannot be red")
            return {"kind": kind, "red": red}
    raise MahjongSoulAdapterError("invalid_tile", f"{field} contains an invalid tile")
