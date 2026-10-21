"""Convert user-owned, locally decoded Mahjong Soul records.

This module is deliberately an offline adapter.  It does not download a
replay, authenticate to Mahjong Soul, or decode the service's wire protocol.
It accepts a small reviewed JSON interchange format and the equivalent
``Record*`` names commonly emitted by local decoders.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable

from .log import LOG_FORMAT, EventLog, parse_event_log
from .replay_bundle import create_replay_bundle

DECODED_FORMAT = "mahjong-soul-decoded-replay/v1"
CONVERTER_NAME = "mahjong-soul-local-decoded"
CONVERTER_VERSION = "1.0.0"
_WINDS = ("east", "south", "west", "north")
_HONORS = {
    "1z": "east",
    "2z": "south",
    "3z": "west",
    "4z": "north",
    "5z": "white",
    "6z": "green",
    "7z": "red",
    "e": "east",
    "s": "south",
    "w": "west",
    "n": "north",
    "p": "white",
    "f": "green",
    "c": "red",
}
_RECORD_TYPES = {
    "recordnewround": "new_round",
    "recorddealtile": "draw",
    "recorddiscardtile": "discard",
    "recordchipenggang": "call",
    "recordangangaddgang": "kan",
    "recordrevealtiles": "dora_reveal",
    "recordrevealtile": "dora_reveal",
    "recordhule": "win",
    "recordnotile": "draw_game",
    "recordliuju": "abortive_draw",
}
_DRAW_REASONS = {
    "exhaustive": "exhaustive",
    "nine_terminals": "nine_terminals",
    "four_winds": "four_winds",
    "four_riichi": "four_riichi",
    "four_kan": "four_kan",
    "triple_ron": "triple_ron",
}


class MahjongSoulConversionError(ValueError):
    """The decoded input cannot be converted without guessing."""


@dataclass
class _RoundBuilder:
    index: int
    initial_scores: tuple[int, int, int, int]
    streams: dict[int, list[dict[str, Any]]]
    public_melds: list[list[dict[str, Any]]]
    dora_indicators: list[dict[str, Any]]
    source_events: int = 1
    terminal: bool = False
    score_deltas: tuple[int, int, int, int] = (0, 0, 0, 0)
    result: dict[str, Any] | None = None
    last_discarder: int | None = None
    riichi_count: int = 0
    kan_count: int = 0

    def public(self, event: dict[str, Any]) -> None:
        self.project(lambda _observer: dict(event))

    def project(self, factory: Callable[[int], dict[str, Any]]) -> None:
        for observer, events in self.streams.items():
            event = factory(observer)
            event["sequence"] = len(events)
            events.append(event)


def convert_decoded_match(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert a decoded four-player match to normalized EventLog documents."""

    records = _records(raw)
    anonymous_id = hashlib.sha256(
        json.dumps(raw, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    rounds: list[_RoundBuilder] = []
    current: _RoundBuilder | None = None
    pending_rinshan: int | None = None

    for position, original in enumerate(records):
        record_type, record = _normalize_record(original, position)
        if record_type == "new_round":
            if current is not None and not current.terminal:
                raise MahjongSoulConversionError(
                    f"record {position}: previous round has no terminal record"
                )
            current = _new_round(record, len(rounds))
            rounds.append(current)
            pending_rinshan = None
            continue
        if current is None:
            raise MahjongSoulConversionError(
                f"record {position}: {record_type} appears before new_round"
            )
        if current.terminal:
            raise MahjongSoulConversionError(
                f"record {position}: event appears after the round terminal"
            )

        if record_type != "dora_reveal":
            _append_cumulative_doras(current, record, position)

        if record_type == "draw":
            player = _seat(record, position)
            tile = _tile(_required(record, ("tile",), position))
            rinshan = bool(record.get("rinshan", record.get("lingshang", False)))
            if pending_rinshan is not None:
                if pending_rinshan != player:
                    raise MahjongSoulConversionError(
                        f"record {position}: replacement draw belongs to another player"
                    )
                rinshan = True
            pending_rinshan = None
            current.project(
                lambda observer: {
                    "type": "draw",
                    "player": player,
                    "tile": tile if observer == player else None,
                    "rinshan": rinshan,
                }
            )
        elif record_type == "discard":
            player = _seat(record, position)
            tile = _tile(_required(record, ("tile",), position))
            riichi = bool(record.get("riichi", record.get("is_liqi", False)))
            if riichi:
                current.riichi_count += 1
                current.public({"type": "riichi", "player": player, "step": "declared"})
            current.public(
                {
                    "type": "discard",
                    "player": player,
                    "tile": tile,
                    "tsumogiri": bool(record.get("tsumogiri", record.get("moqie", False))),
                    "riichi_declaration": riichi,
                }
            )
            if riichi and bool(record.get("riichi_accepted", True)):
                current.public({"type": "riichi", "player": player, "step": "accepted"})
            current.last_discarder = player
        elif record_type in {"call", "kan"}:
            event = _call_event(record_type, record, current.public_melds, position)
            current.public(event)
            if event["type"] == "kan":
                pending_rinshan = int(event["player"])
                current.last_discarder = int(event["player"])
                current.kan_count += 1
        elif record_type == "dora_reveal":
            values = record.get("tiles", record.get("doras"))
            if values is None:
                values = [_required(record, ("tile", "dora"), position)]
            if not isinstance(values, list) or not values:
                raise MahjongSoulConversionError(f"record {position}: dora tiles must be a list")
            for value in values:
                tile = _tile(value)
                current.public({"type": "dora_reveal", "tile": tile})
                current.dora_indicators.append(tile)
        elif record_type == "win":
            _append_win(current, record, position)
        elif record_type in {"draw_game", "abortive_draw"}:
            _append_draw_game(current, record, record_type, position)
        else:
            raise MahjongSoulConversionError(
                f"record {position}: unsupported record type {record_type!r}"
            )
        current.source_events += 1

    if not rounds:
        raise MahjongSoulConversionError("decoded replay contains no rounds")
    if current is not None and not current.terminal:
        raise MahjongSoulConversionError("last round has no terminal record")

    final_scores = _scores(
        raw.get("final_scores", raw.get("result", {}).get("final_scores"))
        if isinstance(raw.get("result", {}), dict)
        else raw.get("final_scores"),
        "final_scores",
        default=tuple(
            rounds[-1].initial_scores[index] + rounds[-1].score_deltas[index]
            for index in range(4)
        ),
    )
    documents = [
        _document(builder, anonymous_id, final_scores) for builder in rounds
    ]
    for document in documents:
        parse_event_log(document)
    return documents


def create_mahjong_soul_bundle(
    raw: dict[str, Any],
    *,
    authorization: str = "user_owned",
    acquisition: str = "user_export",
    reference: str | None = None,
    share_url: str | None = None,
) -> dict[str, Any]:
    """Convert decoded data and wrap it in a validated Replay Bundle."""

    return create_replay_bundle(
        convert_decoded_match(raw),
        platform="mahjong_soul",
        acquisition=acquisition,
        authorization=authorization,
        converter_name=CONVERTER_NAME,
        converter_version=CONVERTER_VERSION,
        reference=reference,
        share_url=share_url,
    )


def _records(raw: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(raw, dict):
        raise MahjongSoulConversionError("decoded replay must be a JSON object")
    value = raw.get("records", raw.get("actions"))
    if not isinstance(value, list) or not value:
        raise MahjongSoulConversionError("decoded replay records must be a non-empty array")
    if raw.get("format") not in {None, DECODED_FORMAT}:
        raise MahjongSoulConversionError(f"expected format {DECODED_FORMAT!r}")
    if not all(isinstance(item, dict) for item in value):
        raise MahjongSoulConversionError("every decoded replay record must be an object")
    return value


def _normalize_record(raw: dict[str, Any], position: int) -> tuple[str, dict[str, Any]]:
    name = raw.get("type", raw.get("name"))
    if not isinstance(name, str) or not name:
        raise MahjongSoulConversionError(f"record {position}: missing type/name")
    short = name.rsplit(".", 1)[-1]
    normalized = _RECORD_TYPES.get(short.lower(), short.lower())
    payload = raw.get("data")
    if payload is None:
        payload = {key: value for key, value in raw.items() if key not in {"type", "name"}}
    if not isinstance(payload, dict):
        raise MahjongSoulConversionError(f"record {position}: data must be an object")
    return normalized, payload


def _new_round(record: dict[str, Any], index: int) -> _RoundBuilder:
    # protobuf JSON omits scalar zero defaults, so East 1's ``ju`` is often absent.
    dealer = _integer(
        record.get("dealer", record.get("ju", 0)), f"round {index} dealer", 0, 3
    )
    hands = record.get("hands", record.get("tiles"))
    if hands is None:
        numbered = [record.get(f"tiles{seat}") for seat in range(4)]
        hands = numbered if all(hand is not None for hand in numbered) else None
    if not isinstance(hands, list) or len(hands) != 4:
        raise MahjongSoulConversionError(
            f"round {index}: all four 13-tile starting hands are required"
        )
    parsed_hands: list[list[dict[str, Any]]] = []
    dealer_initial_draw: dict[str, Any] | None = None
    for seat, hand in enumerate(hands):
        expected_lengths = {13, 14} if seat == dealer else {13}
        if not isinstance(hand, list) or len(hand) not in expected_lengths:
            raise MahjongSoulConversionError(
                f"round {index}: starting hand {seat} must contain "
                f"{'13 or 14' if seat == dealer else '13'} tiles"
            )
        parsed = [_tile(value) for value in hand]
        if seat == dealer and len(parsed) == 14:
            dealer_initial_draw = parsed.pop()
        parsed_hands.append(parsed)

    scores = _scores(record.get("scores"), f"round {index} scores")
    chang = record.get("chang", 0)
    wind_value = record.get("wind", chang)
    if isinstance(wind_value, int):
        if not 0 <= wind_value < 4:
            raise MahjongSoulConversionError(f"round {index}: invalid wind")
        wind = _WINDS[wind_value]
    elif isinstance(wind_value, str) and wind_value.lower() in _WINDS:
        wind = wind_value.lower()
    else:
        raise MahjongSoulConversionError(f"round {index}: invalid wind")
    hand_number = _integer(
        record.get("hand_number", dealer + 1), f"round {index} hand_number", 1, 4
    )
    dora_value = record.get("dora_indicator", record.get("dora"))
    if dora_value is None:
        doras = record.get("doras")
        if isinstance(doras, list) and doras:
            dora_value = doras[0]
    if dora_value is None:
        raise MahjongSoulConversionError(f"round {index}: dora indicator is required")
    dora = _tile(dora_value)
    remaining = _integer(
        record.get("remaining_tiles", 70), f"round {index} remaining_tiles", 0, 70
    )
    round_state = {
        "wind": wind,
        "hand_number": hand_number,
        "honba": _integer(record.get("honba", record.get("ben", 0)), "honba", 0),
        "riichi_sticks": _integer(
            record.get("riichi_sticks", record.get("liqibang", 0)), "riichi_sticks", 0
        ),
        "dealer": dealer,
    }
    streams: dict[int, list[dict[str, Any]]] = {seat: [] for seat in range(4)}
    for observer in range(4):
        streams[observer].append(
            {
                "sequence": 0,
                "type": "round_start",
                "round": round_state,
                "scores": list(scores),
                "initial_hand": parsed_hands[observer],
                "dora_indicator": dora,
                "remaining_tiles": remaining,
            }
        )
    builder = _RoundBuilder(index, scores, streams, [[] for _ in range(4)], [dora])
    if dealer_initial_draw is not None:
        builder.project(
            lambda observer: {
                "type": "draw",
                "player": dealer,
                "tile": dealer_initial_draw if observer == dealer else None,
                "rinshan": False,
            }
        )
        builder.source_events += 1
    return builder


def _call_event(
    record_type: str,
    record: dict[str, Any],
    public_melds: list[list[dict[str, Any]]],
    position: int,
) -> dict[str, Any]:
    player = _seat(record, position)
    meld_raw = record.get("meld") if isinstance(record.get("meld"), dict) else record
    kind_value = meld_raw.get("kind", meld_raw.get("call_type", meld_raw.get("type")))
    numeric = {
        0: "chi",
        1: "pon",
        2: "shouminkan" if record_type == "kan" else "daiminkan",
        3: "ankan",
    }
    aliases = {"peng": "pon", "minggang": "daiminkan", "angang": "ankan", "jiagang": "shouminkan", "addgang": "shouminkan"}
    kind = numeric.get(kind_value, str(kind_value).lower())
    kind = aliases.get(kind, kind)
    if kind not in {"chi", "pon", "ankan", "daiminkan", "shouminkan"}:
        raise MahjongSoulConversionError(f"record {position}: unknown call kind {kind_value!r}")
    tiles_raw = meld_raw.get("tiles")
    existing = None
    if kind == "shouminkan":
        if isinstance(tiles_raw, list) and tiles_raw:
            added = _tile(tiles_raw[-1])
            lookup_tile = _tile(tiles_raw[0])
        else:
            added = _tile(tiles_raw)
            lookup_tile = added
        existing = next(
            (
                meld
                for meld in public_melds[player]
                if meld["kind"] == "pon"
                and meld["tiles"][0]["kind"] == lookup_tile["kind"]
            ),
            None,
        )
        if existing is None:
            raise MahjongSoulConversionError(f"record {position}: shouminkan lacks its pon")
        if not isinstance(tiles_raw, list):
            tiles_raw = [*existing["tiles"], added]
    elif kind == "ankan" and isinstance(tiles_raw, str):
        tiles_raw = [tiles_raw] * 4
    if not isinstance(tiles_raw, list) or len(tiles_raw) != (3 if kind in {"chi", "pon"} else 4):
        raise MahjongSoulConversionError(f"record {position}: invalid {kind} tiles")
    tiles = [_tile(value) for value in tiles_raw]
    source = meld_raw.get("from")
    called_value = meld_raw.get("called_tile")
    froms = meld_raw.get("froms")
    if (source is None or called_value is None) and isinstance(froms, list) and len(froms) == len(tiles):
        external = [i for i, seat in enumerate(froms) if seat != player]
        if len(external) == 1:
            source = froms[external[0]]
            called_value = tiles_raw[external[0]]
    if kind == "ankan":
        source = None
        called_tile = None
    elif kind == "shouminkan":
        assert existing is not None
        source = existing["from"]
        called_tile = existing["called_tile"]
    else:
        if source is None or called_value is None:
            raise MahjongSoulConversionError(
                f"record {position}: {kind} requires an unambiguous source and called tile"
            )
        source = _integer(source, f"record {position} from", 0, 3)
        called_tile = _tile(called_value)
    meld = {"kind": kind, "tiles": tiles, "called_tile": called_tile, "from": source}
    if kind == "shouminkan":
        public_melds[player] = [
            meld if old is existing else old for old in public_melds[player]
        ]
    else:
        public_melds[player].append(meld)
    return {"type": "chi" if kind == "chi" else "pon" if kind == "pon" else "kan", "player": player, "meld": meld}


def _append_win(builder: _RoundBuilder, record: dict[str, Any], position: int) -> None:
    raw_wins = record.get("wins", record.get("hules"))
    if raw_wins is None:
        raw_wins = [record]
    if not isinstance(raw_wins, list) or not raw_wins:
        raise MahjongSoulConversionError(f"record {position}: win list is empty")
    wins: list[dict[str, Any]] = []
    for raw in raw_wins:
        if not isinstance(raw, dict):
            raise MahjongSoulConversionError(f"record {position}: win must be an object")
        winner = _integer(raw.get("winner", raw.get("seat")), "winner", 0, 3)
        source = raw.get(
            "from",
            raw.get(
                "from_seat",
                winner if raw.get("zimo") else builder.last_discarder,
            ),
        )
        if source is None:
            raise MahjongSoulConversionError(f"record {position}: win source is required")
        wins.append({"winner": winner, "from": _integer(source, "win source", 0, 3)})
    deltas = _score_deltas(record, position)
    losers = {win["from"] for win in wins}
    if len(wins) == 1 and wins[0]["winner"] == wins[0]["from"]:
        builder.public({"type": "tsumo", "winner": wins[0]["winner"], "score_deltas": list(deltas)})
    elif len(losers) == 1:
        builder.public({"type": "ron", "winners": [win["winner"] for win in wins], "loser": next(iter(losers)), "score_deltas": list(deltas)})
    else:
        raise MahjongSoulConversionError(f"record {position}: mixed win sources are unsupported")
    builder.score_deltas = deltas
    builder.result = {"type": "agari", "score_deltas": list(deltas), "wins": wins}
    builder.terminal = True


def _append_draw_game(
    builder: _RoundBuilder, record: dict[str, Any], record_type: str, position: int
) -> None:
    raw_reason = record.get("reason", "exhaustive" if record_type == "draw_game" else None)
    if record_type == "abortive_draw" and isinstance(record.get("type"), int):
        raw_reason = {
            1: "nine_terminals",
            2: "four_winds",
        }.get(record["type"])
        if raw_reason is None:
            if builder.riichi_count == 4:
                raw_reason = "four_riichi"
            elif builder.kan_count >= 4:
                raw_reason = "four_kan"
            else:
                raw_reason = "triple_ron"
    reason = _DRAW_REASONS.get(str(raw_reason).lower())
    if reason is None:
        raise MahjongSoulConversionError(
            f"record {position}: explicit supported abortive-draw reason is required"
        )
    # RecordLiuJu normally carries no score array. Riichi deductions have already
    # been represented by preceding accepted declarations, so the terminal delta
    # for an abortive draw is zero unless the service explicitly supplies one.
    if record_type == "abortive_draw" and not _has_score_deltas(record):
        deltas = (0, 0, 0, 0)
    else:
        deltas = _score_deltas(record, position)
    builder.public({"type": "draw_game", "reason": reason, "score_deltas": list(deltas)})
    builder.score_deltas = deltas
    builder.result = {"type": "ryuukyoku", "reason": reason, "score_deltas": list(deltas)}
    builder.terminal = True


def _document(
    builder: _RoundBuilder, anonymous_id: str, final_scores: tuple[int, int, int, int]
) -> dict[str, Any]:
    return {
        "format": LOG_FORMAT,
        "log_id": f"{anonymous_id}-r{builder.index:02d}",
        "rules": None,
        "observations": {str(seat): events for seat, events in builder.streams.items()},
        "result": {
            "final_scores": list(final_scores),
            "initial_scores": list(builder.initial_scores),
            "round_score_deltas": list(builder.score_deltas),
            "round_result": builder.result,
        },
        "provenance": {
            "source": "Mahjong Soul user-owned local decoded replay",
            "converter_version": CONVERTER_VERSION,
        },
    }


def _tile(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        kind = value.get("kind")
        red = bool(value.get("red", False))
        if not isinstance(kind, str):
            raise MahjongSoulConversionError("tile.kind must be a string")
        if kind in {"east", "south", "west", "north", "white", "green", "red"}:
            return {"kind": kind, "red": red}
        value = f"{kind}{'r' if red else ''}"
    if not isinstance(value, str):
        raise MahjongSoulConversionError(f"invalid tile {value!r}")
    token = value.strip().lower()
    if token in _HONORS:
        return {"kind": _HONORS[token], "red": False}
    red = token.endswith("r")
    if red:
        token = token[:-1]
    if len(token) == 2 and token[0].isdigit() and token[1] in "mps":
        rank = int(token[0])
        if rank == 0:
            rank, red = 5, True
        if 1 <= rank <= 9:
            if red and rank != 5:
                raise MahjongSoulConversionError("only suited fives can be red")
            return {"kind": f"{token[1]}{rank}", "red": red}
    if len(token) == 2 and token[0] in "mps" and token[1].isdigit():
        return _tile(token[1] + token[0] + ("r" if red else ""))
    raise MahjongSoulConversionError(f"unsupported Mahjong Soul tile {value!r}")


def _seat(record: dict[str, Any], position: int) -> int:
    return _integer(record.get("player", record.get("seat")), f"record {position} seat", 0, 3)


def _score_deltas(record: dict[str, Any], position: int) -> tuple[int, int, int, int]:
    value = record.get("score_deltas", record.get("delta_scores"))
    if value is None and isinstance(record.get("scores"), list):
        nested = [item.get("delta_scores") for item in record["scores"] if isinstance(item, dict)]
        if nested and all(isinstance(item, list) and len(item) == 4 for item in nested):
            value = [sum(int(item[seat]) for item in nested) for seat in range(4)]
    if value is None:
        raise MahjongSoulConversionError(
            f"record {position}: terminal score_deltas/delta_scores are required"
        )
    return _scores(value, f"record {position} score_deltas")


def _has_score_deltas(record: dict[str, Any]) -> bool:
    if record.get("score_deltas") is not None or record.get("delta_scores") is not None:
        return True
    return isinstance(record.get("scores"), list) and any(
        isinstance(item, dict) and item.get("delta_scores") is not None
        for item in record["scores"]
    )


def _append_cumulative_doras(
    builder: _RoundBuilder, record: dict[str, Any], position: int
) -> None:
    raw = record.get("doras")
    if raw is None:
        return
    if not isinstance(raw, list):
        raise MahjongSoulConversionError(f"record {position}: doras must be a list")
    # Empty repeated protobuf fields are serialized as [] by some decoders.
    if not raw:
        return
    parsed = [_tile(value) for value in raw]
    known = builder.dora_indicators
    if len(parsed) < len(known) or parsed[: len(known)] != known:
        raise MahjongSoulConversionError(
            f"record {position}: cumulative doras do not extend previous indicators"
        )
    for tile in parsed[len(known) :]:
        builder.public({"type": "dora_reveal", "tile": tile})
        builder.dora_indicators.append(tile)


def _scores(
    value: Any, name: str, default: tuple[int, int, int, int] | None = None
) -> tuple[int, int, int, int]:
    if value is None and default is not None:
        return default
    if not isinstance(value, (list, tuple)) or len(value) != 4 or not all(
        isinstance(item, int) and not isinstance(item, bool) for item in value
    ):
        raise MahjongSoulConversionError(f"{name} must contain four integers")
    return tuple(value)  # type: ignore[return-value]


def _integer(value: Any, name: str, minimum: int, maximum: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise MahjongSoulConversionError(f"{name} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        raise MahjongSoulConversionError(f"{name} is out of range")
    return value


def _required(record: dict[str, Any], names: tuple[str, ...], position: int) -> Any:
    for name in names:
        if name in record:
            return record[name]
    raise MahjongSoulConversionError(
        f"record {position}: missing required field {'/'.join(names)}"
    )
