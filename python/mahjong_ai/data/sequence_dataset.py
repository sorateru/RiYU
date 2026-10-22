"""Sequence-aware, observable-input training records with oracle-only labels.

Inputs remain normal player observations.  Hidden hands and future draws are
read only while producing targets and are never copied into ``observation`` or
``public_events``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from mahjong_ai import ObservationReplayer, calculate_shanten_json
from mahjong_ai.schema import action_key, canonical_json, tile_index
from mahjong_ai.ai.sequence import public_event_tokens

from .dataset import DecisionRecord, build_decision_records
from .log import EventLog


SEQUENCE_DATASET_FORMAT = "mahjong-ai-sequence-decision/v4"
LEGACY_SEQUENCE_DATASET_FORMAT = "mahjong-ai-sequence-decision/v3"
SEQUENCE_LABEL_VERSION = "sequence-labels-v2"


@dataclass(frozen=True)
class SequenceDecisionRecord:
    log_id: str
    observer: int
    event_sequence: int
    observation: str
    public_events: str
    legal_actions: str
    chosen_action: str
    chosen_action_index: int
    final_rank: int
    final_score: int
    round_score_delta: int
    round_delta: str
    opponent_shanten: str
    opponent_tenpai: str
    opponent_wait: str
    future_draw: str
    round_result: str
    factual_deal_in_by_action: str
    factual_deal_in_points_by_action: str
    factual_deal_in_mask: str
    counterfactual_deal_in_by_action: str
    counterfactual_deal_in_points_by_action: str
    counterfactual_deal_in_mask: str
    factual_round_result_by_action: str
    factual_round_result_mask: str
    counterfactual_round_result_by_action: str
    counterfactual_round_result_mask: str
    factual_action_round_delta: str
    factual_action_round_delta_mask: str
    counterfactual_action_round_delta: str
    counterfactual_action_round_delta_mask: str
    factual_action_final_rank: str
    factual_action_final_score: str
    factual_action_hanchan_mask: str
    counterfactual_action_final_rank: str
    counterfactual_action_final_score: str
    counterfactual_action_hanchan_mask: str
    opponent_next_discard: str
    opponent_next_discard_mask: str
    opponent_next_response: str
    opponent_next_response_mask: str
    opponent_next_call_type: str
    opponent_next_call_type_mask: str
    opponent_tile_count: str
    opponent_tile_count_mask: str
    label_masks: str
    source: str
    converter_version: str
    state_schema_version: int
    dataset_version: str = SEQUENCE_DATASET_FORMAT
    label_version: str = SEQUENCE_LABEL_VERSION


def build_sequence_decision_records(
    log: EventLog,
    *,
    future_draw_horizon: int = 12,
) -> list[SequenceDecisionRecord]:
    if future_draw_horizon <= 0:
        raise ValueError("future_draw_horizon must be positive")
    base = build_decision_records(log)
    by_key: dict[tuple[int, int], DecisionRecord] = {
        (record.observer, record.event_sequence): record for record in base
    }
    if len(by_key) != len(base):
        raise ValueError("sequence dataset requires unique observer/sequence decisions")

    streams = log.observations
    lengths = {len(events) for events in streams.values()}
    if len(lengths) != 1:
        raise ValueError("observer event streams must have identical lengths")
    rules_json = canonical_json(log.rules) if log.rules is not None else None
    replayers = [ObservationReplayer(observer, rules_json) for observer in range(4)]
    output: list[SequenceDecisionRecord] = []
    deal_in_index, deal_in_winners, deal_in_points = _terminal_deal_in_labels(log)
    for index in range(next(iter(lengths))):
        relevant = any((observer, index) in by_key for observer in range(4))
        decoded = (
            [json.loads(replayer.state_json()) for replayer in replayers]
            if relevant
            else []
        )
        labels: tuple[list[int], list[bool], list[list[int]]] | None = None
        future_draws: list[int] | None = None
        for observer in range(4):
            record = by_key.get((observer, index))
            if record is None:
                continue
            if labels is None:
                labels = _hidden_labels(decoded)
            if future_draws is None:
                future_draws = _future_draws(streams, index, future_draw_horizon)
            shanten, tenpai, waits = labels
            public = json.loads(record.observation)
            legal_actions = json.loads(record.legal_actions)
            chosen_action = json.loads(record.expert_action)
            legal_keys = [action_key(action) for action in legal_actions]
            chosen_action_index = legal_keys.index(action_key(chosen_action))
            opponents = [(observer + relative) % 4 for relative in range(1, 4)]
            relative_deltas = [
                int(log.round_score_deltas[(observer + relative) % 4])
                for relative in range(4)
            ]
            event = streams[observer][index]
            riichi_discard_decision = (
                event.get("type") == "riichi"
                and event.get("step") == "declared"
                and index + 1 == deal_in_index
            )
            is_deal_in_decision = index == deal_in_index or riichi_discard_decision
            factual_deal_in = [[False] * 3 for _ in legal_actions]
            factual_points = [[0] * 3 for _ in legal_actions]
            factual_mask = [[False] * 3 for _ in legal_actions]
            if chosen_action["type"] in {"discard", "riichi"}:
                factual_mask[chosen_action_index] = [True] * 3
                if is_deal_in_decision:
                    factual_deal_in[chosen_action_index] = [
                        player in deal_in_winners for player in opponents
                    ]
                    factual_points[chosen_action_index] = [
                        int(deal_in_points[player]) for player in opponents
                    ]
            counterfactual_deal_in = [[False] * 3 for _ in legal_actions]
            counterfactual_points = [[0] * 3 for _ in legal_actions]
            counterfactual_mask = [[False] * 3 for _ in legal_actions]
            round_result = _outcome_name(_outcome_target(log, observer))
            round_result_index = _outcome_target(log, observer)
            factual_round_result = [-1] * len(legal_actions)
            factual_round_result[chosen_action_index] = round_result_index
            factual_action_round_delta = [0] * len(legal_actions)
            factual_action_round_delta[chosen_action_index] = int(
                log.round_score_deltas[observer]
            )
            factual_action_final_rank = [0] * len(legal_actions)
            factual_action_final_score = [0] * len(legal_actions)
            factual_action_final_rank[chosen_action_index] = record.final_rank
            factual_action_final_score[chosen_action_index] = record.final_score
            factual_action_mask = [False] * len(legal_actions)
            factual_action_mask[chosen_action_index] = True
            counterfactual_action_mask = [False] * len(legal_actions)
            next_actions = _future_opponent_actions(streams[observer], index, opponents)
            tile_counts = _hidden_tile_counts(decoded, opponents)
            output.append(
                SequenceDecisionRecord(
                    log_id=record.log_id,
                    observer=observer,
                    event_sequence=index,
                    observation=record.observation,
                    public_events=canonical_json(
                        public_event_tokens(public, maximum=None)
                    ),
                    legal_actions=record.legal_actions,
                    chosen_action=record.expert_action,
                    chosen_action_index=chosen_action_index,
                    final_rank=record.final_rank,
                    final_score=record.final_score,
                    round_score_delta=int(log.round_score_deltas[observer]),
                    round_delta=canonical_json(relative_deltas),
                    opponent_shanten=canonical_json(
                        [shanten[player] for player in opponents]
                    ),
                    opponent_tenpai=canonical_json(
                        [tenpai[player] for player in opponents]
                    ),
                    opponent_wait=canonical_json(
                        [
                            [tile in set(waits[player]) for tile in range(34)]
                            for player in opponents
                        ]
                    ),
                    future_draw=canonical_json(
                        [tile in set(future_draws) for tile in range(34)]
                    ),
                    round_result=round_result,
                    factual_deal_in_by_action=canonical_json(factual_deal_in),
                    factual_deal_in_points_by_action=canonical_json(factual_points),
                    factual_deal_in_mask=canonical_json(factual_mask),
                    counterfactual_deal_in_by_action=canonical_json(
                        counterfactual_deal_in
                    ),
                    counterfactual_deal_in_points_by_action=canonical_json(
                        counterfactual_points
                    ),
                    counterfactual_deal_in_mask=canonical_json(counterfactual_mask),
                    factual_round_result_by_action=canonical_json(
                        factual_round_result
                    ),
                    factual_round_result_mask=canonical_json(factual_action_mask),
                    counterfactual_round_result_by_action=canonical_json(
                        [-1] * len(legal_actions)
                    ),
                    counterfactual_round_result_mask=canonical_json(
                        counterfactual_action_mask
                    ),
                    factual_action_round_delta=canonical_json(
                        factual_action_round_delta
                    ),
                    factual_action_round_delta_mask=canonical_json(
                        factual_action_mask
                    ),
                    counterfactual_action_round_delta=canonical_json(
                        [0] * len(legal_actions)
                    ),
                    counterfactual_action_round_delta_mask=canonical_json(
                        counterfactual_action_mask
                    ),
                    factual_action_final_rank=canonical_json(
                        factual_action_final_rank
                    ),
                    factual_action_final_score=canonical_json(
                        factual_action_final_score
                    ),
                    factual_action_hanchan_mask=canonical_json(factual_action_mask),
                    counterfactual_action_final_rank=canonical_json(
                        [0] * len(legal_actions)
                    ),
                    counterfactual_action_final_score=canonical_json(
                        [0] * len(legal_actions)
                    ),
                    counterfactual_action_hanchan_mask=canonical_json(
                        counterfactual_action_mask
                    ),
                    opponent_next_discard=canonical_json(next_actions["discard"]),
                    opponent_next_discard_mask=canonical_json(
                        next_actions["discard_mask"]
                    ),
                    opponent_next_response=canonical_json(next_actions["response"]),
                    opponent_next_response_mask=canonical_json(
                        next_actions["response_mask"]
                    ),
                    opponent_next_call_type=canonical_json(next_actions["call_type"]),
                    opponent_next_call_type_mask=canonical_json(
                        next_actions["call_type_mask"]
                    ),
                    opponent_tile_count=canonical_json(tile_counts),
                    opponent_tile_count_mask=canonical_json([True] * 3),
                    label_masks=canonical_json(
                        {
                            "policy": True,
                            "rank": True,
                            "final_score": True,
                            "round_delta": [True] * 4,
                            "opponent_state": [True] * 3,
                            "future_draw": True,
                            "round_result": True,
                        }
                    ),
                    source=record.source,
                    converter_version=record.converter_version,
                    state_schema_version=record.state_schema_version,
                )
            )

        if streams[0][index].get("type") in {"ron", "tsumo"}:
            break
        for observer in range(4):
            event = dict(streams[observer][index])
            event["sequence"] = replayers[observer].next_event_sequence()
            replayers[observer].apply_event_json(canonical_json(event))
    return output


def _terminal_deal_in_labels(
    log: EventLog,
) -> tuple[int | None, set[int], list[int]]:
    events = log.observations[0]
    terminal_index = next(
        (index for index, event in enumerate(events) if event.get("type") == "ron"),
        None,
    )
    if terminal_index is None:
        return None, set(), [0, 0, 0, 0]
    terminal = events[terminal_index]
    loser = int(terminal["loser"])
    discard_index = next(
        (
            index
            for index in range(terminal_index - 1, -1, -1)
            if events[index].get("type") == "discard"
            and int(events[index]["player"]) == loser
        ),
        None,
    )
    if discard_index is None:
        return None, set(), [0, 0, 0, 0]
    loss = max(0, -int(log.round_score_deltas[loser]))
    winners = {int(value) for value in terminal["winners"]}
    points = [0, 0, 0, 0]
    if len(winners) == 1:
        points[next(iter(winners))] = loss
    elif winners:
        result_wins = {
            int(win["winner"]): win
            for win in (log.round_result or {}).get("wins", [])
            if win.get("winner") is not None and win.get("from") == loser
        }
        honba = int(events[0].get("round", {}).get("honba", 0))
        estimates: dict[int, int] = {}
        for winner in winners:
            ten = result_wins.get(winner, {}).get("ten") or []
            if len(ten) >= 2 and int(ten[1]) > 0:
                estimates[winner] = int(ten[1]) + 300 * honba
        if len(estimates) == len(winners) and sum(estimates.values()) == loss:
            for winner, value in estimates.items():
                points[winner] = value
        else:
            # Legacy/custom logs may omit per-win payments. Preserve the exact
            # aggregate loss and split only the ambiguous multi-ron component.
            ordered = sorted(winners)
            share, remainder = divmod(loss, len(ordered))
            for offset, winner in enumerate(ordered):
                points[winner] = share + (remainder if offset == 0 else 0)
    return discard_index, winners, points


def _outcome_target(log: EventLog, observer: int) -> int:
    """0 self win, 1 deal-in, 2 opponent tsumo, 3 horizontal, 4 draw."""

    result = log.round_result or {}
    wins = result.get("wins") or _wins_from_terminal_events(log)
    if any(win.get("winner") == observer for win in wins):
        return 0
    if any(win.get("from") == observer and win.get("winner") != observer for win in wins):
        return 1
    if wins:
        return 2 if all(win.get("winner") == win.get("from") for win in wins) else 3
    return 4


def _outcome_name(target: int) -> str:
    return ("self_win", "deal_in", "opponent_tsumo", "horizontal", "draw")[target]


def _wins_from_terminal_events(log: EventLog) -> list[dict[str, int]]:
    """Recover outcome labels for simulator/custom logs without round_result.

    Event streams are observable and contain only terminal public facts, so
    using them as labels cannot expose a hidden hand or wall to model inputs.
    """

    for event in reversed(log.observations[0]):
        if event.get("type") == "tsumo":
            winner = int(event["winner"])
            return [{"winner": winner, "from": winner}]
        if event.get("type") == "ron":
            loser = int(event["loser"])
            return [
                {"winner": int(winner), "from": loser}
                for winner in event.get("winners", [])
            ]
        if event.get("type") == "draw_game":
            return []
    return []


def write_sequence_parquet(
    records: Iterable[SequenceDecisionRecord], path: str | Path
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    materialized = list(records)
    for record in materialized:
        validate_sequence_decision_record(record)
    rows = [asdict(record) for record in materialized]
    if not rows:
        raise ValueError("cannot write an empty sequence dataset")
    table = pa.Table.from_pylist(rows)
    metadata = dict(table.schema.metadata or {})
    metadata[b"mahjong_ai_dataset"] = SEQUENCE_DATASET_FORMAT.encode()
    table = table.replace_schema_metadata(metadata)
    pq.write_table(table, path, compression="zstd")


def validate_sequence_decision_record(record: SequenceDecisionRecord) -> None:
    """Fail closed when a V4 record violates a frozen axis or mask contract."""

    if record.dataset_version != SEQUENCE_DATASET_FORMAT:
        raise ValueError("record does not use the Decision V4 schema")
    if record.label_version != SEQUENCE_LABEL_VERSION:
        raise ValueError("record does not use the Decision V4 label contract")
    actions = json.loads(record.legal_actions)
    chosen = json.loads(record.chosen_action)
    if not 0 <= record.chosen_action_index < len(actions):
        raise ValueError("chosen_action_index is outside legal_actions")
    if action_key(actions[record.chosen_action_index]) != action_key(chosen):
        raise ValueError("chosen_action_index does not identify chosen_action")
    if len(json.loads(record.opponent_tenpai)) != 3:
        raise ValueError("opponent_tenpai must have three relative opponents")
    waits = json.loads(record.opponent_wait)
    if len(waits) != 3 or any(len(row) != 34 for row in waits):
        raise ValueError("opponent_wait must have shape [3, 34]")
    if len(json.loads(record.opponent_shanten)) != 3:
        raise ValueError("opponent_shanten must have three relative opponents")
    if len(json.loads(record.round_delta)) != 4:
        raise ValueError("round_delta must have four relative seats")
    if len(json.loads(record.future_draw)) != 34:
        raise ValueError("future_draw must have 34 tile indicators")
    for field in (
        "factual_deal_in_by_action",
        "factual_deal_in_points_by_action",
        "factual_deal_in_mask",
        "counterfactual_deal_in_by_action",
        "counterfactual_deal_in_points_by_action",
        "counterfactual_deal_in_mask",
    ):
        values = json.loads(getattr(record, field))
        if len(values) != len(actions) or any(len(row) != 3 for row in values):
            raise ValueError(f"{field} must have shape [legal_actions, 3]")
    for field in (
        "factual_round_result_by_action",
        "factual_round_result_mask",
        "counterfactual_round_result_by_action",
        "counterfactual_round_result_mask",
        "factual_action_round_delta",
        "factual_action_round_delta_mask",
        "counterfactual_action_round_delta",
        "counterfactual_action_round_delta_mask",
        "factual_action_final_rank",
        "factual_action_final_score",
        "factual_action_hanchan_mask",
        "counterfactual_action_final_rank",
        "counterfactual_action_final_score",
        "counterfactual_action_hanchan_mask",
    ):
        if len(json.loads(getattr(record, field))) != len(actions):
            raise ValueError(f"{field} must follow the legal action axis")
    for field in (
        "opponent_next_discard",
        "opponent_next_discard_mask",
        "opponent_next_response",
        "opponent_next_response_mask",
        "opponent_next_call_type",
        "opponent_next_call_type_mask",
        "opponent_tile_count_mask",
    ):
        if len(json.loads(getattr(record, field))) != 3:
            raise ValueError(f"{field} must follow the three-opponent axis")
    tile_counts = json.loads(record.opponent_tile_count)
    if len(tile_counts) != 3 or any(len(row) != 34 for row in tile_counts):
        raise ValueError("opponent_tile_count must have shape [3, 34]")
    if any(not 0 <= int(value) <= 4 for row in tile_counts for value in row):
        raise ValueError("opponent_tile_count values must be between zero and four")

    def checked_mask(field: str) -> list[bool]:
        values = json.loads(getattr(record, field))
        flattened = (
            [value for row in values for value in row]
            if values and isinstance(values[0], list)
            else values
        )
        if any(not isinstance(value, bool) for value in flattened):
            raise ValueError(f"{field} must contain booleans")
        return flattened

    factual_action_masks = (
        "factual_round_result_mask",
        "factual_action_round_delta_mask",
        "factual_action_hanchan_mask",
    )
    for field in factual_action_masks:
        mask = json.loads(getattr(record, field))
        if any(value and index != record.chosen_action_index for index, value in enumerate(mask)):
            raise ValueError(f"{field} may label only chosen_action_index")
    factual_deal_mask = json.loads(record.factual_deal_in_mask)
    if any(
        value and action_index != record.chosen_action_index
        for action_index, row in enumerate(factual_deal_mask)
        for value in row
    ):
        raise ValueError("factual_deal_in_mask may label only chosen_action_index")

    result_fields = (
        ("factual_round_result_by_action", "factual_round_result_mask"),
        ("counterfactual_round_result_by_action", "counterfactual_round_result_mask"),
    )
    for value_field, mask_field in result_fields:
        values = json.loads(getattr(record, value_field))
        mask = json.loads(getattr(record, mask_field))
        if any(valid and not 0 <= int(value) < 5 for value, valid in zip(values, mask, strict=True)):
            raise ValueError(f"{value_field} contains an invalid result class")
    for value_field, mask_field in (
        ("factual_action_final_rank", "factual_action_hanchan_mask"),
        ("counterfactual_action_final_rank", "counterfactual_action_hanchan_mask"),
    ):
        values = json.loads(getattr(record, value_field))
        mask = json.loads(getattr(record, mask_field))
        if any(
            valid and not 1 <= int(value) <= 4
            for value, valid in zip(values, mask, strict=True)
        ):
            raise ValueError(f"{value_field} must contain final ranks one through four")
    for value_field, mask_field, class_count in (
        ("opponent_next_discard", "opponent_next_discard_mask", 34),
        ("opponent_next_response", "opponent_next_response_mask", 4),
        ("opponent_next_call_type", "opponent_next_call_type_mask", 3),
    ):
        values = json.loads(getattr(record, value_field))
        mask = json.loads(getattr(record, mask_field))
        if any(
            valid and not 0 <= int(value) < class_count
            for value, valid in zip(values, mask, strict=True)
        ):
            raise ValueError(f"{value_field} contains an invalid class")
    mask_pairs = (
        ("factual_deal_in_mask", "counterfactual_deal_in_mask"),
        ("factual_round_result_mask", "counterfactual_round_result_mask"),
        (
            "factual_action_round_delta_mask",
            "counterfactual_action_round_delta_mask",
        ),
        ("factual_action_hanchan_mask", "counterfactual_action_hanchan_mask"),
    )
    mask_fields = {field for pair in mask_pairs for field in pair}
    mask_fields.update(
        {
            "opponent_next_discard_mask",
            "opponent_next_response_mask",
            "opponent_next_call_type_mask",
            "opponent_tile_count_mask",
        }
    )
    for field in mask_fields:
        checked_mask(field)
    for factual_field, counterfactual_field in mask_pairs:
        factual = json.loads(getattr(record, factual_field))
        counterfactual = json.loads(getattr(record, counterfactual_field))
        factual_flat = [value for row in factual for value in row] if factual and isinstance(factual[0], list) else factual
        counterfactual_flat = (
            [value for row in counterfactual for value in row]
            if counterfactual and isinstance(counterfactual[0], list)
            else counterfactual
        )
        if any(bool(left) and bool(right) for left, right in zip(factual_flat, counterfactual_flat, strict=True)):
            raise ValueError(
                f"{factual_field} and {counterfactual_field} must not overlap"
            )


def _hidden_tile_counts(
    states: list[dict[str, Any]], opponents: list[int]
) -> list[list[int]]:
    result: list[list[int]] = []
    for player in opponents:
        counts = [0] * 34
        state = states[player]
        for tile in state.get("hand") or []:
            counts[tile_index(tile)] += 1
        if state.get("drawn_tile") is not None:
            counts[tile_index(state["drawn_tile"])] += 1
        result.append(counts)
    return result


def _future_opponent_actions(
    events: list[dict[str, Any]], start: int, opponents: list[int]
) -> dict[str, list[int] | list[bool]]:
    discard = [0] * 3
    discard_mask = [False] * 3
    response = [0] * 3
    response_mask = [False] * 3
    call_type = [0] * 3
    call_type_mask = [False] * 3
    seat_to_relative = {player: index for index, player in enumerate(opponents)}
    for event in events[start + 1 :]:
        event_type = str(event.get("type", ""))
        player = event.get("player", event.get("winner"))
        if event_type == "ron":
            for winner in event.get("winners", []):
                relative = seat_to_relative.get(int(winner))
                if relative is not None and not response_mask[relative]:
                    response[relative] = 3
                    response_mask[relative] = True
            continue
        if player is None or int(player) not in seat_to_relative:
            continue
        relative = seat_to_relative[int(player)]
        if event_type == "discard" and not discard_mask[relative]:
            discard[relative] = tile_index(event["tile"])
            discard_mask[relative] = True
        if response_mask[relative]:
            continue
        if event_type == "pass":
            response[relative] = 0
            response_mask[relative] = True
        elif event_type in {"chi", "pon", "kan"}:
            response[relative] = 1
            response_mask[relative] = True
            call_type[relative] = {"chi": 0, "pon": 1, "kan": 2}[event_type]
            call_type_mask[relative] = True
        elif event_type == "riichi" and event.get("step") == "declared":
            response[relative] = 2
            response_mask[relative] = True
        elif event_type == "tsumo":
            response[relative] = 3
            response_mask[relative] = True
        if all(discard_mask) and all(response_mask):
            break
    return {
        "discard": discard,
        "discard_mask": discard_mask,
        "response": response,
        "response_mask": response_mask,
        "call_type": call_type,
        "call_type_mask": call_type_mask,
    }


def _hidden_labels(
    states: list[dict[str, Any]],
) -> tuple[list[int], list[bool], list[list[int]]]:
    shanten: list[int] = []
    tenpai: list[bool] = []
    waits: list[list[int]] = []
    for player, state in enumerate(states):
        counts = [0] * 34
        for tile in state.get("hand") or []:
            counts[tile_index(tile)] += 1
        if state.get("drawn_tile") is not None:
            counts[tile_index(state["drawn_tile"])] += 1
        fixed_melds = len(state["players"][player]["melds"])
        current = _shanten(counts, fixed_melds)
        shanten.append(current)
        player_waits = _winning_tiles(counts, fixed_melds)
        tenpai.append(bool(player_waits) or current == -1)
        waits.append(player_waits)
    return shanten, tenpai, waits


def _winning_tiles(counts: list[int], fixed_melds: int) -> list[int]:
    """Return waits for either a 13-tile or pre-discard 14-tile state.

    Decision records also occur immediately before a discard/riichi, where the
    player has a drawn tile.  In that case the useful oracle target is the union
    of winning tiles after every discard that leaves the hand in tenpai.
    """

    concealed = sum(counts)
    maximum = 14 - fixed_melds * 3
    if concealed == maximum - 1:
        bases = [counts]
    elif concealed == maximum:
        bases = []
        seen: set[tuple[int, ...]] = set()
        for discarded, amount in enumerate(counts):
            if amount <= 0:
                continue
            base = list(counts)
            base[discarded] -= 1
            key = tuple(base)
            if key not in seen and _shanten(base, fixed_melds) == 0:
                seen.add(key)
                bases.append(base)
    else:
        return []

    waits: set[int] = set()
    for base in bases:
        if _shanten(base, fixed_melds) != 0:
            continue
        for tile in range(34):
            if base[tile] >= 4:
                continue
            candidate = list(base)
            candidate[tile] += 1
            if _shanten(candidate, fixed_melds) == -1:
                waits.add(tile)
    return sorted(waits)


def _shanten(counts: list[int], fixed_melds: int) -> int:
    return _cached_shanten(tuple(counts), fixed_melds)


@lru_cache(maxsize=500_000)
def _cached_shanten(counts: tuple[int, ...], fixed_melds: int) -> int:
    return int(
        json.loads(calculate_shanten_json(canonical_json(counts), fixed_melds))["minimum"]
    )


def _future_draws(
    streams: dict[int, list[dict[str, Any]]], start: int, horizon: int
) -> list[int]:
    output: list[int] = []
    for index in range(start, len(streams[0])):
        public = streams[0][index]
        if public.get("type") != "draw":
            continue
        player = int(public["player"])
        private = streams[player][index]
        tile = private.get("tile")
        if tile is not None:
            output.append(tile_index(tile))
        if len(output) >= horizon:
            break
    return output
