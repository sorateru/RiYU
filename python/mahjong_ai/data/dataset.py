"""Validated decision records and Parquet export for supervised learning."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from mahjong_ai import (
    analyze_state_json,
    apply_event_json,
    calculate_shanten_json,
    replay_events_json,
)
from mahjong_ai.schema import action_key, canonical_json, tile_index

from .log import EventLog


@dataclass(frozen=True)
class DecisionRecord:
    log_id: str
    observer: int
    event_sequence: int
    observation: str
    legal_actions: str
    expert_action: str
    final_rank: int
    final_score: int
    source: str
    converter_version: str
    state_schema_version: int


def build_decision_records(
    log: EventLog, *, supplemental_responses_only: bool = False
) -> list[DecisionRecord]:
    ranks = _ranks(log.final_scores)
    records: list[DecisionRecord] = []
    rules_json = canonical_json(log.rules) if log.rules is not None else None
    for observer, events in log.observations.items():
        state_json = replay_events_json(observer, "[]", rules_json)
        for index, event in enumerate(events):
            state = json.loads(state_json)
            phase = state["phase"]["phase"]

            # Tenhou records successful calls but has no explicit tag for a
            # declined call.  A following normal draw proves that every player
            # declined the preceding response window, so it is an uncensored
            # and safe `pass` training label.  Responses lost to a higher
            # priority call are deliberately not labelled as pass.
            if phase in {"awaiting_response", "awaiting_kan_response"} and event.get(
                "type"
            ) == "draw" and _may_have_non_pass_response(state):
                legal = _legal_actions(state_json)
                if any(candidate["type"] != "pass" for candidate in legal):
                    _append_record(
                        records,
                        log,
                        observer,
                        int(event["sequence"]),
                        state,
                        legal,
                        {"type": "pass"},
                        ranks,
                    )
                    # A declined ron changes future furiten state. Calls that
                    # were merely declined require no persistent state event.
                    if any(candidate["type"] == "ron" for candidate in legal):
                        synthetic = {
                            "sequence": state["next_event_sequence"],
                            "type": "pass",
                            "player": observer,
                            "ron_available": True,
                        }
                        state_json = apply_event_json(state_json, canonical_json(synthetic))
                        state = json.loads(state_json)

            action = _decision_for_event(events, index, observer, state)
            if supplemental_responses_only and event.get("type") not in {"ron", "tsumo"}:
                action = None
            if action is not None:
                legal = _legal_actions(state_json)
                _append_record(
                    records,
                    log,
                    observer,
                    int(event["sequence"]),
                    state,
                    legal,
                    action,
                    ranks,
                )

            # Terminal tags are needed as policy labels, but applying them is
            # unnecessary because there are no later decisions in this round.
            # The converter intentionally omits score deltas from these labels.
            if event.get("type") in {"ron", "tsumo"}:
                break

            replay_event = dict(event)
            replay_event["sequence"] = json.loads(state_json)["next_event_sequence"]
            state_json = apply_event_json(state_json, canonical_json(replay_event))
    return records


def _legal_actions(state_json: str) -> list[dict[str, Any]]:
    analysis = json.loads(analyze_state_json(state_json))
    return analysis["legal_actions"]["value"] or []


def _may_have_non_pass_response(state: dict[str, Any]) -> bool:
    phase = state["phase"]
    if phase["phase"] == "awaiting_kan_response":
        return True
    if phase["phase"] != "awaiting_response":
        return False
    observer = int(state["information"]["observer"])
    discarder = int(phase["discarder"])
    called = phase["tile"]
    called_index = tile_index(called)
    counts = [0] * 34
    for tile in state["hand"]:
        counts[tile_index(tile)] += 1
    own_riichi = state["players"][observer]["riichi"] in {"accepted", "double_accepted"}
    if not own_riichi:
        if counts[called_index] >= 2:
            return True
        if discarder == (observer + 3) % 4 and called_index < 27:
            rank = called_index % 9
            suit_base = called_index - rank
            sequences = ((rank - 2, rank - 1), (rank - 1, rank + 1), (rank + 1, rank + 2))
            for left, right in sequences:
                if 0 <= left < 9 and 0 <= right < 9:
                    if counts[suit_base + left] and counts[suit_base + right]:
                        return True
    counts[called_index] += 1
    fixed_melds = len(state["players"][observer]["melds"])
    shanten = json.loads(calculate_shanten_json(canonical_json(counts), fixed_melds))
    return int(shanten["minimum"]) == -1


def _append_record(
    records: list[DecisionRecord],
    log: EventLog,
    observer: int,
    event_sequence: int,
    state: dict[str, Any],
    legal: list[dict[str, Any]],
    action: dict[str, Any],
    ranks: tuple[int, int, int, int],
) -> None:
    legal_keys = {action_key(candidate) for candidate in legal}
    if action_key(action) not in legal_keys:
        raise ValueError(
            f"log {log.log_id} observer {observer} event {event_sequence}: "
            "expert action is not legal"
        )
    records.append(
        DecisionRecord(
            log_id=log.log_id,
            observer=observer,
            event_sequence=event_sequence,
            observation=canonical_json(state),
            legal_actions=canonical_json(legal),
            expert_action=canonical_json(action),
            final_rank=ranks[observer],
            final_score=log.final_scores[observer],
            source=log.source,
            converter_version=log.converter_version,
            state_schema_version=state["schema_version"],
        )
    )


def write_parquet(records: Iterable[DecisionRecord], path: str | Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = [asdict(record) for record in records]
    if not rows:
        raise ValueError("cannot write an empty decision dataset")
    table = pa.Table.from_pylist(rows)
    metadata = dict(table.schema.metadata or {})
    metadata[b"mahjong_ai_dataset"] = b"decision/v1"
    table = table.replace_schema_metadata(metadata)
    pq.write_table(table, path, compression="zstd")


def _event_to_action(event: dict[str, Any], observer: int) -> dict[str, Any] | None:
    event_type = event.get("type")
    if event_type == "discard" and event.get("player") == observer:
        if event.get("riichi_declaration", False):
            return {
                "type": "riichi",
                "discard": event["tile"],
                "tsumogiri": event["tsumogiri"],
            }
        return {"type": "discard", "tile": event["tile"], "tsumogiri": event["tsumogiri"]}
    if event_type in {"chi", "pon"} and event.get("player") == observer:
        meld = event["meld"]
        consumed = list(meld["tiles"])
        consumed.remove(meld["called_tile"])
        return {
            "type": event_type,
            "called_tile": meld["called_tile"],
            "from": meld["from"],
            "consume": consumed,
        }
    if event_type == "ron" and observer in event.get("winners", []):
        return {"type": "ron", "target": event["loser"]}
    if event_type == "tsumo" and event.get("winner") == observer:
        return {"type": "tsumo"}
    if event_type == "pass" and event.get("player") == observer:
        return {"type": "pass"}
    # Kan conversion needs the prior state (pon index/physical tiles), and is
    # intentionally rejected until an adapter supplies an explicit action.
    return event.get("expert_action") if event.get("player") == observer else None


def _decision_for_event(
    events: list[dict[str, Any]],
    index: int,
    observer: int,
    state: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Place a riichi decision before its declaration state transition.

    Service logs encode riichi as DECLARE, discard, ACCEPT.  The legal action
    must be learned from the state before DECLARE, while the discard event is
    still needed for exact state replay.
    """

    event = events[index]
    if (
        event.get("type") == "riichi"
        and event.get("step") == "declared"
        and event.get("player") == observer
    ):
        if index + 1 >= len(events):
            raise ValueError("riichi declaration lacks its discard")
        discard = events[index + 1]
        if discard.get("type") != "discard" or discard.get("player") != observer:
            raise ValueError("riichi declaration must be immediately followed by its discard")
        return {
            "type": "riichi",
            "discard": discard["tile"],
            "tsumogiri": discard["tsumogiri"],
        }
    if event.get("type") == "discard" and event.get("riichi_declaration", False):
        if index > 0 and events[index - 1].get("type") == "riichi":
            return None
    if event.get("type") == "kan" and event.get("player") == observer:
        if state is None:
            return None
        meld = event["meld"]
        kind = meld["kind"]
        if kind == "ankan":
            return {"type": "kan", "action": {"kind": "ankan", "tiles": meld["tiles"]}}
        if kind == "daiminkan":
            consumed = list(meld["tiles"])
            consumed.remove(meld["called_tile"])
            return {
                "type": "kan",
                "action": {
                    "kind": "daiminkan",
                    "called_tile": meld["called_tile"],
                    "from": meld["from"],
                    "consume": consumed,
                },
            }
        if kind == "shouminkan":
            previous = state["players"][observer]["melds"]
            meld_index = next(
                (
                    position
                    for position, candidate in enumerate(previous)
                    if candidate["kind"] == "pon"
                    and tile_index(candidate["tiles"][0]) == tile_index(meld["tiles"][0])
                ),
                None,
            )
            if meld_index is None:
                raise ValueError("shouminkan teacher action lacks its original pon")
            old_tiles = list(previous[meld_index]["tiles"])
            added = next((tile for tile in meld["tiles"] if tile not in old_tiles), None)
            if added is None:
                # Red/non-red identity can make all JSON objects compare equal
                # after an archive normalization. Base kind is still enough
                # for the legal action encoder.
                added = meld["tiles"][-1]
            return {
                "type": "kan",
                "action": {"kind": "shouminkan", "tile": added, "meld_index": meld_index},
            }
    return _event_to_action(event, observer)


def _ranks(scores: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    order = sorted(range(4), key=lambda seat: (-scores[seat], seat))
    ranks = [0] * 4
    for rank, seat in enumerate(order, start=1):
        ranks[seat] = rank
    return tuple(ranks)  # type: ignore[return-value]
