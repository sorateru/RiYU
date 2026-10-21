"""Adapter from normalized observer logs to the web replay-analysis contract."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from typing import Any

from mahjong_ai.data import EventLog, build_decision_records


MAX_REPLAY_DECISIONS = 256
MAX_REPLAY_LOGS = 64
_WINDS = {"east": "東", "south": "南", "west": "西", "north": "北"}
_HONORS = {
    "east": "東",
    "south": "南",
    "west": "西",
    "north": "北",
    "white": "白",
    "green": "發",
    "red": "中",
}


def build_web_replay_analysis(
    log: EventLog,
    observer: int,
    analyze: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    if observer not in range(4):
        raise ValueError("observer must be between 0 and 3")
    records = [
        record
        for record in build_decision_records(log)
        if record.observer == observer
        and json.loads(record.expert_action).get("type") in {"discard", "riichi"}
    ]
    if not records:
        raise ValueError("the uploaded log has no discard decisions for this observer")
    if len(records) > MAX_REPLAY_DECISIONS:
        raise ValueError(f"replay exceeds the {MAX_REPLAY_DECISIONS}-decision safety limit")

    turns: list[dict[str, Any]] = []
    model_used = False
    for record in records:
        state = json.loads(record.observation)
        analysis = analyze(state)
        candidates, uses_model = _candidates(analysis)
        model_used = model_used or uses_model
        if not candidates:
            continue
        recommendations = {candidate["tile"]: candidate["score"] for candidate in candidates}
        hand = list(state.get("hand") or [])
        drawn = state.get("drawn_tile")
        hand_tiles = [
            {
                "tile": _tile_label(tile),
                "recommendation": recommendations.get(_tile_label(tile), 0.0),
                "drawn": False,
            }
            for tile in hand
        ]
        if drawn is not None:
            hand_tiles.append(
                {
                    "tile": _tile_label(drawn),
                    "recommendation": recommendations.get(_tile_label(drawn), 0.0),
                    "drawn": True,
                }
            )
        players = state["players"]
        seats = {
            "self": observer,
            "right": (observer + 1) % 4,
            "opposite": (observer + 2) % 4,
            "left": (observer + 3) % 4,
        }
        turns.append(
            {
                "turn": len(players[observer]["discards"]) + 1,
                "handTiles": hand_tiles,
                "candidates": candidates,
                "scores": {name: int(state["scores"][seat]) for name, seat in seats.items()},
                "discards": {
                    name: [
                        _tile_label(discard["tile"])
                        for discard in players[seat]["discards"]
                        if discard.get("called_by") is None
                    ]
                    for name, seat in seats.items()
                },
                "remainingTiles": int(state["remaining_tiles"]),
                "dora": " ".join(_tile_label(tile) for tile in state["dora_indicators"]),
                "eventSequence": record.event_sequence,
            }
        )
    if not turns:
        raise ValueError("the uploaded log has no analyzable discard candidates")
    first_state = json.loads(records[0].observation)
    round_state = first_state.get("round") or {}
    wind = str(round_state.get("wind", "east"))
    number = int(round_state.get("hand_number", 1))
    honba = int(round_state.get("honba", 0))
    label = f"{_WINDS.get(wind, wind)}{number}局"
    if honba:
        label += f" {honba}本場"
    return {
        "format": "riyu-replay-analysis/v1",
        "sourceLogId": log.log_id,
        "observer": observer,
        "analysisMode": "model_estimate" if model_used else "exact_tile_efficiency",
        "decisionCount": len(turns),
        "rounds": [
            {
                "id": log.log_id,
                "label": label,
                "honba": honba,
                "riichiSticks": int(round_state.get("riichi_sticks", 0)),
                "decisionCount": len(turns),
                "turns": turns,
            }
        ],
    }


def build_web_replay_collection(
    logs: list[EventLog],
    observer: int,
    analyze: Callable[[dict[str, Any]], dict[str, Any]],
    *,
    source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Analyze a bounded ordered collection of round logs as one replay."""

    if not logs:
        raise ValueError("at least one replay log is required")
    if len(logs) > MAX_REPLAY_LOGS:
        raise ValueError(f"replay exceeds the {MAX_REPLAY_LOGS}-round safety limit")

    rounds: list[dict[str, Any]] = []
    source_ids: list[str] = []
    model_used = False
    decision_count = 0
    for index, log in enumerate(logs):
        analysis = build_web_replay_analysis(log, observer, analyze)
        source_ids.append(log.log_id)
        model_used = model_used or analysis["analysisMode"] == "model_estimate"
        for round_value in analysis["rounds"]:
            decision_count += len(round_value["turns"])
            if decision_count > MAX_REPLAY_DECISIONS:
                raise ValueError(
                    f"replay exceeds the {MAX_REPLAY_DECISIONS}-decision safety limit"
                )
            rounds.append({**round_value, "id": f"{index}-{round_value['id']}"})

    result = {
        "format": "riyu-replay-analysis/v1",
        "sourceLogIds": source_ids,
        "observer": observer,
        "analysisMode": "model_estimate" if model_used else "exact_tile_efficiency",
        "decisionCount": decision_count,
        "rounds": rounds,
    }
    if len(source_ids) == 1:
        result["sourceLogId"] = source_ids[0]
    if source is not None:
        result["source"] = source
    return result


def _candidates(analysis: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    policy = analysis.get("model_policy") or {}
    ranked = policy.get("value") if policy.get("status") == "model_estimate" else None
    if ranked:
        usable = [item for item in ranked if _discard_tile(item.get("action") or {}) is not None]
        scores = _normalized_scores([max(float(item.get("policy_score", 0.0)), 0.0) for item in usable])
        return [
            {
                "tile": _tile_label(_discard_tile(item["action"])),
                "score": score,
                "shanten": int(item.get("shanten", 0)),
                "ukeire": int(item.get("ukeire_count", 0)),
                "dealInRisk": _percent_or_none(item.get("deal_in_probability")),
                "expectedValue": _number_or_none(item.get("estimated_action_value")),
            }
            for item, score in zip(usable, scores, strict=True)
        ], True

    efficiency = (analysis.get("discard_efficiency") or {}).get("value") or []
    if not efficiency:
        return [], False
    minimum = min(int(item["shanten"]["minimum"]) for item in efficiency)
    weights = [
        math.exp(-2.0 * (int(item["shanten"]["minimum"]) - minimum))
        * (int(item.get("ukeire_count", 0)) + 1)
        for item in efficiency
    ]
    scores = _normalized_scores(weights)
    candidates = [
        {
            "tile": _tile_label(item["discard"]),
            "score": score,
            "shanten": int(item["shanten"]["minimum"]),
            "ukeire": int(item.get("ukeire_count", 0)),
            "dealInRisk": None,
            "expectedValue": None,
        }
        for item, score in zip(efficiency, scores, strict=True)
    ]
    candidates.sort(key=lambda item: (-item["score"], item["tile"]))
    return candidates, False


def _normalized_scores(values: list[float]) -> list[float]:
    total = sum(values)
    if total <= 0:
        return [round(100.0 / len(values), 1) for _ in values] if values else []
    return [round(value * 100.0 / total, 1) for value in values]


def _discard_tile(action: dict[str, Any]) -> dict[str, Any] | None:
    if action.get("type") == "discard":
        return action.get("tile")
    if action.get("type") == "riichi":
        return action.get("discard")
    return None


def _tile_label(tile: dict[str, Any]) -> str:
    kind = str(tile["kind"])
    if kind in _HONORS:
        return _HONORS[kind]
    label = f"{kind[1:]}{kind[0]}"
    return f"赤{label}" if tile.get("red") else label


def _percent_or_none(value: Any) -> float | None:
    return None if value is None else round(float(value) * 100.0, 1)


def _number_or_none(value: Any) -> float | None:
    return None if value is None else round(float(value), 1)
