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
from mahjong_ai.schema import canonical_json, tile_index
from mahjong_ai.ai.sequence import public_event_tokens

from .dataset import DecisionRecord, build_decision_records
from .log import EventLog


SEQUENCE_DATASET_FORMAT = "mahjong-ai-sequence-decision/v3"


@dataclass(frozen=True)
class SequenceDecisionRecord:
    log_id: str
    observer: int
    event_sequence: int
    observation: str
    public_events: str
    legal_actions: str
    expert_action: str
    final_rank: int
    final_score: int
    round_score_delta: int
    all_round_score_deltas: str
    all_player_shanten: str
    opponent_tenpai: str
    opponent_waits: str
    future_draws: str
    result_type: str
    source: str
    converter_version: str
    state_schema_version: int
    expert_deal_in_by_player: str = "[false,false,false,false]"
    expert_deal_in_points: int = 0
    expert_deal_in_points_by_player: str = "[0,0,0,0]"
    expert_outcome_target: int = 4
    dataset_version: str = SEQUENCE_DATASET_FORMAT


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
            event = streams[observer][index]
            riichi_discard_decision = (
                event.get("type") == "riichi"
                and event.get("step") == "declared"
                and index + 1 == deal_in_index
            )
            is_deal_in_decision = index == deal_in_index or riichi_discard_decision
            output.append(
                SequenceDecisionRecord(
                    log_id=record.log_id,
                    observer=observer,
                    event_sequence=index,
                    observation=record.observation,
                    public_events=canonical_json(public_event_tokens(public)),
                    legal_actions=record.legal_actions,
                    expert_action=record.expert_action,
                    final_rank=record.final_rank,
                    final_score=record.final_score,
                    round_score_delta=int(log.round_score_deltas[observer]),
                    all_round_score_deltas=canonical_json(log.round_score_deltas),
                    all_player_shanten=canonical_json(shanten),
                    opponent_tenpai=canonical_json(
                        [tenpai[player] if player != observer else False for player in range(4)]
                    ),
                    opponent_waits=canonical_json(
                        [waits[player] if player != observer else [] for player in range(4)]
                    ),
                    future_draws=canonical_json(future_draws),
                    result_type=str((log.round_result or {}).get("type", "unknown")),
                    source=record.source,
                    converter_version=record.converter_version,
                    state_schema_version=record.state_schema_version,
                    expert_deal_in_by_player=canonical_json(
                        [
                            is_deal_in_decision and player in deal_in_winners
                            for player in range(4)
                        ]
                    ),
                    expert_deal_in_points=(
                        sum(deal_in_points) if is_deal_in_decision else 0
                    ),
                    expert_deal_in_points_by_player=canonical_json(
                        deal_in_points if is_deal_in_decision else [0, 0, 0, 0]
                    ),
                    expert_outcome_target=_outcome_target(log, observer),
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

    rows = [asdict(record) for record in records]
    if not rows:
        raise ValueError("cannot write an empty sequence dataset")
    table = pa.Table.from_pylist(rows)
    metadata = dict(table.schema.metadata or {})
    metadata[b"mahjong_ai_dataset"] = SEQUENCE_DATASET_FORMAT.encode()
    table = table.replace_schema_metadata(metadata)
    pq.write_table(table, path, compression="zstd")


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
