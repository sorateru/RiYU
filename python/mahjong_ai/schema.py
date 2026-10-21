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
    # Tile consumption is a multiset. Service logs and the rules engine may
    # enumerate the same physical tiles in different orders (especially when a
    # red five is involved), but they represent the same legal action.
    normalized = json.loads(canonical_json(action))

    def sort_tiles(container: dict[str, Any], key: str) -> None:
        value = container.get(key)
        if isinstance(value, list) and all(isinstance(tile, dict) for tile in value):
            value.sort(key=lambda tile: (tile_index(tile), bool(tile.get("red", False))))

    sort_tiles(normalized, "consume")
    kan = normalized.get("action")
    if isinstance(kan, dict):
        sort_tiles(kan, "consume")
        sort_tiles(kan, "tiles")
    return canonical_json(normalized)
