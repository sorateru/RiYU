"""Offline conversion of Tenhou XML logs into validated observer event logs.

This module intentionally contains no downloader or service client. It accepts
only already acquired SQLite archives and removes player names and raw log IDs.
"""

from __future__ import annotations

import bz2
import hashlib
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any

from .log import EventLog


_DRAW_PLAYERS = {"T": 0, "U": 1, "V": 2, "W": 3}
_DISCARD_PLAYERS = {"D": 0, "E": 1, "F": 2, "G": 3}
_WINDS = ("east", "south", "west", "north")


@dataclass(frozen=True)
class TenhouRound:
    log: EventLog
    source_events: int


def decode_archive_xml(blob: bytes) -> ET.Element:
    """Decode one bzip2-compressed XML document from a phoenix-logs DB."""

    return ET.fromstring(bz2.decompress(blob))


def convert_document(
    root: ET.Element,
    raw_log_id: str,
    *,
    source: str = "Tenhou Phoenix offline archive",
    converter_version: str = "tenhou-phoenix-v3-outcome-kan-labels",
) -> list[TenhouRound]:
    """Convert one hanchan into independently validated round event logs."""

    final_scores = _final_scores(root)
    red_fives = _uses_red_fives(root)
    anonymous_id = hashlib.sha256(raw_log_id.encode("utf-8")).hexdigest()
    output: list[TenhouRound] = []
    current: list[ET.Element] = []
    round_index = 0
    for element in root:
        if element.tag == "INIT":
            if current:
                converted = _convert_round(
                    current,
                    anonymous_id,
                    round_index,
                    final_scores,
                    red_fives,
                    source,
                    converter_version,
                )
                if converted is not None:
                    output.append(converted)
                round_index += 1
            current = [element]
        elif current:
            current.append(element)
    if current:
        converted = _convert_round(
            current,
            anonymous_id,
            round_index,
            final_scores,
            red_fives,
            source,
            converter_version,
        )
        if converted is not None:
            output.append(converted)
    return output


def _convert_round(
    elements: list[ET.Element],
    anonymous_id: str,
    round_index: int,
    final_scores: tuple[int, int, int, int],
    red_fives: bool,
    source: str,
    converter_version: str,
) -> TenhouRound | None:
    init = elements[0]
    if init.tag != "INIT":
        raise ValueError("Tenhou round must begin with INIT")
    seed = _csv_ints(init.attrib["seed"])
    if len(seed) < 6:
        raise ValueError("INIT seed lacks round metadata")
    wind_index = seed[0] // 4
    if wind_index >= len(_WINDS):
        return None
    dealer = int(init.attrib["oya"])
    scores = tuple(value * 100 for value in _csv_ints(init.attrib["ten"]))
    if len(scores) != 4:
        raise ValueError("INIT ten must contain four scores")

    streams: dict[int, list[dict[str, Any]]] = {seat: [] for seat in range(4)}
    for observer in range(4):
        initial_ids = _csv_ints(init.attrib[f"hai{observer}"])
        if len(initial_ids) != 13:
            raise ValueError("INIT observer hand must contain 13 tiles")
        streams[observer].append(
            {
                "sequence": 0,
                "type": "round_start",
                "round": {
                    "wind": _WINDS[wind_index],
                    "hand_number": seed[0] % 4 + 1,
                    "honba": seed[1],
                    "riichi_sticks": seed[2],
                    "dealer": dealer,
                },
                "scores": list(scores),
                "initial_hand": [_tile(tile_id, red_fives) for tile_id in initial_ids],
                "dora_indicator": _tile(seed[5], red_fives),
                "remaining_tiles": 70,
            }
        )

    pending_riichi: set[int] = set()
    rinshan_player: int | None = None
    last_draws: list[int | None] = [None] * 4
    public_melds: list[list[dict[str, Any]]] = [[] for _ in range(4)]
    source_events = 1
    for element in elements[1:]:
        tag = element.tag
        first = tag[:1]
        if first in _DRAW_PLAYERS and tag[1:].isdigit():
            player = _DRAW_PLAYERS[first]
            tile_id = int(tag[1:])
            tile = _tile(tile_id, red_fives)
            last_draws[player] = tile_id
            rinshan = rinshan_player == player
            rinshan_player = None
            _append_projected(
                streams,
                lambda observer: {
                    "type": "draw",
                    "player": player,
                    "tile": tile if observer == player else None,
                    "rinshan": rinshan,
                },
            )
            source_events += 1
            continue
        if first in _DISCARD_PLAYERS and tag[1:].isdigit():
            player = _DISCARD_PLAYERS[first]
            tile_id = int(tag[1:])
            event = {
                "type": "discard",
                "player": player,
                "tile": _tile(tile_id, red_fives),
                "tsumogiri": last_draws[player] == tile_id,
                "riichi_declaration": player in pending_riichi,
            }
            pending_riichi.discard(player)
            last_draws[player] = None
            _append_public(streams, event)
            source_events += 1
            continue
        if tag == "REACH":
            player = int(element.attrib["who"])
            step = int(element.attrib["step"])
            if step == 1:
                pending_riichi.add(player)
                _append_public(streams, {"type": "riichi", "player": player, "step": "declared"})
            elif step == 2:
                _append_public(streams, {"type": "riichi", "player": player, "step": "accepted"})
            source_events += 1
            continue
        if tag == "DORA":
            _append_public(
                streams,
                {"type": "dora_reveal", "tile": _tile(int(element.attrib["hai"]), red_fives)},
            )
            source_events += 1
            continue
        if tag == "N":
            player = int(element.attrib["who"])
            event = _meld_event(int(element.attrib["m"]), player, red_fives, public_melds)
            _append_public(streams, event)
            last_draws[player] = None
            if event["type"] == "kan":
                rinshan_player = player
            source_events += 1
            continue
        if tag == "AGARI":
            agari = [candidate for candidate in elements if candidate.tag == "AGARI"]
            if all("who" in candidate.attrib and "fromWho" in candidate.attrib for candidate in agari):
                winners = [int(candidate.attrib["who"]) for candidate in agari]
                losers = {int(candidate.attrib["fromWho"]) for candidate in agari}
                if len(winners) == 1 and losers == {winners[0]}:
                    _append_public(streams, {"type": "tsumo", "winner": winners[0]})
                elif len(losers) == 1:
                    _append_public(
                        streams,
                        {"type": "ron", "winners": winners, "loser": losers.pop()},
                    )
                source_events += len(agari)
            break
        if tag in {"RYUUKYOKU", "INIT"}:
            break

    round_result = _round_result(elements, red_fives)
    log_id = f"{anonymous_id}-r{round_index:02d}"
    return TenhouRound(
        EventLog(
            log_id=log_id,
            rules=None,
            observations=streams,
            final_scores=final_scores,
            source=source,
            converter_version=converter_version,
            initial_scores=scores,  # type: ignore[arg-type]
            round_score_deltas=tuple(round_result["score_deltas"]),
            round_result=round_result,
        ),
        source_events,
    )


def _append_public(streams: dict[int, list[dict[str, Any]]], event: dict[str, Any]) -> None:
    _append_projected(streams, lambda _observer: dict(event))


def _append_projected(streams: dict[int, list[dict[str, Any]]], factory: Any) -> None:
    for observer, events in streams.items():
        event = factory(observer)
        event["sequence"] = len(events)
        events.append(event)


def _tile(tile_id: int, red_fives: bool) -> dict[str, Any]:
    if not 0 <= tile_id < 136:
        raise ValueError(f"invalid Tenhou tile ID {tile_id}")
    kind_index = tile_id // 4
    if kind_index < 9:
        kind = f"m{kind_index + 1}"
    elif kind_index < 18:
        kind = f"p{kind_index - 8}"
    elif kind_index < 27:
        kind = f"s{kind_index - 17}"
    else:
        kind = ("east", "south", "west", "north", "white", "green", "red")[kind_index - 27]
    return {"kind": kind, "red": red_fives and tile_id in {16, 52, 88}}


def _meld_event(
    value: int,
    player: int,
    red_fives: bool,
    public_melds: list[list[dict[str, Any]]],
) -> dict[str, Any]:
    relative_from = value & 0x3
    source = (player + relative_from) % 4
    if value & 0x4:
        copies = ((value >> 3) & 3, (value >> 5) & 3, (value >> 7) & 3)
        encoded = value >> 10
        called_index = encoded % 3
        base = encoded // 3
        base = (base // 7) * 9 + base % 7
        ids = [4 * (base + offset) + copies[offset] for offset in range(3)]
        meld = {
            "kind": "chi",
            "tiles": [_tile(tile_id, red_fives) for tile_id in ids],
            "called_tile": _tile(ids[called_index], red_fives),
            "from": source,
        }
        public_melds[player].append(meld)
        return {"type": "chi", "player": player, "meld": meld}

    if value & 0x18:
        unused_copy = (value >> 5) & 3
        encoded = value >> 9
        called_index = encoded % 3
        kind_index = encoded // 3
        pon_ids = [4 * kind_index + copy for copy in range(4) if copy != unused_copy]
        if value & 0x8:
            meld = {
                "kind": "pon",
                "tiles": [_tile(tile_id, red_fives) for tile_id in pon_ids],
                "called_tile": _tile(pon_ids[called_index], red_fives),
                "from": source,
            }
            public_melds[player].append(meld)
            return {"type": "pon", "player": player, "meld": meld}

        existing_index = next(
            (
                index
                for index, meld in enumerate(public_melds[player])
                if meld["kind"] == "pon" and _kind_index(meld["tiles"][0]) == kind_index
            ),
            None,
        )
        if existing_index is None:
            raise ValueError("shouminkan lacks its original pon")
        previous = public_melds[player][existing_index]
        all_ids = [4 * kind_index + copy for copy in range(4)]
        meld = {
            "kind": "shouminkan",
            "tiles": [_tile(tile_id, red_fives) for tile_id in all_ids],
            "called_tile": previous["called_tile"],
            "from": previous["from"],
        }
        public_melds[player][existing_index] = meld
        return {"type": "kan", "player": player, "meld": meld}

    encoded = value >> 8
    called_copy = encoded % 4
    kind_index = encoded // 4
    ids = [4 * kind_index + copy for copy in range(4)]
    if relative_from == 0:
        meld = {
            "kind": "ankan",
            "tiles": [_tile(tile_id, red_fives) for tile_id in ids],
            "called_tile": None,
            "from": None,
        }
    else:
        meld = {
            "kind": "daiminkan",
            "tiles": [_tile(tile_id, red_fives) for tile_id in ids],
            "called_tile": _tile(4 * kind_index + called_copy, red_fives),
            "from": source,
        }
    public_melds[player].append(meld)
    return {"type": "kan", "player": player, "meld": meld}


def _kind_index(tile: dict[str, Any]) -> int:
    kind = tile["kind"]
    if kind[0] in "mps":
        return {"m": 0, "p": 9, "s": 18}[kind[0]] + int(kind[1:]) - 1
    return {"east": 27, "south": 28, "west": 29, "north": 30, "white": 31, "green": 32, "red": 33}[kind]


def _round_result(elements: list[ET.Element], red_fives: bool) -> dict[str, Any]:
    agari = [element for element in elements if element.tag == "AGARI"]
    draw = next((element for element in elements if element.tag == "RYUUKYOKU"), None)
    terminal = agari or ([draw] if draw is not None else [])
    deltas = [0, 0, 0, 0]
    for element in terminal:
        raw = [float(value) for value in element.attrib.get("sc", "").split(",") if value]
        if len(raw) >= 8:
            for player in range(4):
                deltas[player] += int(round(raw[player * 2 + 1] * 100))
    if agari:
        wins = []
        for element in agari:
            yaku = _csv_ints(element.attrib.get("yaku", ""))
            wins.append(
                {
                    "winner": int(element.attrib["who"]),
                    "from": int(element.attrib["fromWho"]),
                    "winning_tile": _tile(int(element.attrib["machi"]), red_fives)
                    if "machi" in element.attrib
                    else None,
                    "ten": _csv_ints(element.attrib.get("ten", "")),
                    "yaku": [yaku[index : index + 2] for index in range(0, len(yaku), 2)],
                    "yakuman": _csv_ints(element.attrib.get("yakuman", "")),
                    "dora_indicators": _csv_ints(element.attrib.get("doraHai", "")),
                    "ura_indicators": _csv_ints(element.attrib.get("doraHaiUra", "")),
                }
            )
        return {"type": "agari", "score_deltas": deltas, "wins": wins}
    if draw is not None:
        return {
            "type": "ryuukyoku",
            "reason": draw.attrib.get("type", "exhaustive"),
            "score_deltas": deltas,
            "tenpai_hands": {
                str(player): [_tile(tile_id, red_fives) for tile_id in _csv_ints(draw.attrib[key])]
                for player in range(4)
                if (key := f"hai{player}") in draw.attrib
            },
        }
    return {"type": "unknown", "score_deltas": deltas}


def _final_scores(root: ET.Element) -> tuple[int, int, int, int]:
    elements = list(root)
    for element in reversed(elements):
        if "owari" in element.attrib:
            values = [float(value) for value in element.attrib["owari"].split(",")]
            if len(values) >= 8:
                return tuple(int(round(values[index] * 100)) for index in range(0, 8, 2))  # type: ignore[return-value]
    for element in reversed(elements):
        if "sc" in element.attrib:
            values = [float(value) for value in element.attrib["sc"].split(",")]
            if len(values) >= 8:
                return tuple(
                    int(round((values[index] + values[index + 1]) * 100))
                    for index in range(0, 8, 2)
                )  # type: ignore[return-value]
    for element in reversed(elements):
        if element.tag == "INIT" and "ten" in element.attrib:
            scores = tuple(value * 100 for value in _csv_ints(element.attrib["ten"]))
            if len(scores) == 4:
                return scores  # type: ignore[return-value]
    raise ValueError("Tenhou document has no final score information")


def _uses_red_fives(root: ET.Element) -> bool:
    go = next((element for element in root if element.tag == "GO"), None)
    return go is None or not (int(go.attrib.get("type", "0")) & 0x02)


def _csv_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item != ""]
