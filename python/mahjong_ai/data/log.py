"""Offline event-log contract and parser.

The contract stores a separately projected event stream for each observer, so
an opponent's concealed draw can never leak into a normal model input.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOG_FORMAT = "mahjong-ai-event-log/v1"


class EventLogError(ValueError):
    pass


@dataclass(frozen=True)
class EventLog:
    log_id: str
    rules: dict[str, Any] | None
    observations: dict[int, list[dict[str, Any]]]
    final_scores: tuple[int, int, int, int]
    source: str
    converter_version: str
    initial_scores: tuple[int, int, int, int] | None = None
    round_score_deltas: tuple[int, int, int, int] = (0, 0, 0, 0)
    round_result: dict[str, Any] | None = None


def load_event_log(path: str | Path) -> EventLog:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("format") != LOG_FORMAT:
        raise EventLogError(f"expected format {LOG_FORMAT!r}")
    streams = raw.get("observations")
    if not isinstance(streams, dict) or set(streams) != {"0", "1", "2", "3"}:
        raise EventLogError("observations must contain exactly observer keys 0..3")
    observations: dict[int, list[dict[str, Any]]] = {}
    for observer in range(4):
        events = streams[str(observer)]
        if not isinstance(events, list):
            raise EventLogError(f"observer {observer} stream must be an array")
        for expected, event in enumerate(events):
            if event.get("sequence") != expected:
                raise EventLogError(f"observer {observer}: non-contiguous sequence at {expected}")
            if event.get("type") == "draw":
                player = event.get("player")
                if player != observer and event.get("tile") is not None:
                    raise EventLogError(
                        f"observer {observer}: opponent draw at {expected} leaks a tile"
                    )
                if player == observer and event.get("tile") is None:
                    raise EventLogError(
                        f"observer {observer}: own draw at {expected} is missing a tile"
                    )
        observations[observer] = events
    scores = raw.get("result", {}).get("final_scores")
    if not isinstance(scores, list) or len(scores) != 4 or not all(isinstance(x, int) for x in scores):
        raise EventLogError("result.final_scores must contain four integers")
    provenance = raw.get("provenance", {})
    initial_scores = raw.get("result", {}).get("initial_scores")
    if initial_scores is not None and (
        not isinstance(initial_scores, list)
        or len(initial_scores) != 4
        or not all(isinstance(x, int) for x in initial_scores)
    ):
        raise EventLogError("result.initial_scores must contain four integers")
    deltas = raw.get("result", {}).get("round_score_deltas", [0, 0, 0, 0])
    if not isinstance(deltas, list) or len(deltas) != 4 or not all(
        isinstance(x, int) for x in deltas
    ):
        raise EventLogError("result.round_score_deltas must contain four integers")
    return EventLog(
        log_id=str(raw.get("log_id", Path(path).stem)),
        rules=raw.get("rules"),
        observations=observations,
        final_scores=tuple(scores),
        source=str(provenance.get("source", "unknown")),
        converter_version=str(provenance.get("converter_version", "unknown")),
        initial_scores=tuple(initial_scores) if initial_scores is not None else None,
        round_score_deltas=tuple(deltas),
        round_result=raw.get("result", {}).get("round_result"),
    )
