"""Shared Python-side schema helpers; Rust remains authoritative for rules."""

from __future__ import annotations

import json
from typing import Any

TILE_TYPES = (
    *(f"m{n}" for n in range(1, 10)),
    *(f"p{n}" for n in range(1, 10)),
    *(f"s{n}" for n in range(1, 10)),
    "east",
    "south",
    "west",
    "north",
    "white",
    "green",
    "red",
)
TILE_TO_INDEX = {name: index for index, name in enumerate(TILE_TYPES)}
ACTION_TYPES = ("discard", "riichi", "chi", "pon", "kan", "ron", "tsumo", "abortive_draw", "pass")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def tile_index(tile: dict[str, Any]) -> int:
    return TILE_TO_INDEX[tile["kind"]]


def action_key(action: dict[str, Any]) -> str:
    return canonical_json(action)
