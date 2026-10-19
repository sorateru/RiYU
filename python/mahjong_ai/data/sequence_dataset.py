"""Sequence-aware, observable-input training records with oracle-only labels.

Inputs remain normal player observations.  Hidden hands and future draws are
read only while producing targets and are never copied into ``observation`` or
``public_events``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from mahjong_ai import apply_event_json, calculate_shanten_json, replay_events_json
from mahjong_ai.schema import canonical_json, tile_index
from mahjong_ai.ai.sequence import public_event_tokens

from .dataset import DecisionRecord, build_decision_records
from .log import EventLog


SEQUENCE_DATASET_FORMAT = "mahjong-ai-sequence-decision/v2"


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
    states = [replay_events_json(observer, "[]", rules_json) for observer in range(4)]
    output: list[SequenceDecisionRecord] = []
    for index in range(next(iter(lengths))):
        decoded = [json.loads(value) for value in states]
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
                )
            )

        if streams[0][index].get("type") in {"ron", "tsumo"}:
            break
        for observer in range(4):
            event = dict(streams[observer][index])
            event["sequence"] = json.loads(states[observer])["next_event_sequence"]
            states[observer] = apply_event_json(states[observer], canonical_json(event))
    return output


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
