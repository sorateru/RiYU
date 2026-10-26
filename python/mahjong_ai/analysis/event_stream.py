"""Ordered external events applied to the authoritative observable GameState."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from mahjong_ai import apply_event_json, legal_actions_json, replay_events_json, validate_state_json
from mahjong_ai.adapters import MahjongSoulEvent, map_majsoul_event, parse_majsoul_event
from mahjong_ai.schema import canonical_json


class RealtimeStateError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class DecisionPoint:
    kind: str
    legal_actions: list[dict[str, Any]]


@dataclass(frozen=True)
class EventApplyResult:
    status: str
    source_event: MahjongSoulEvent | None
    game_event: dict[str, Any] | None
    state: dict[str, Any]
    decision: DecisionPoint | None


class RealtimeStateManager:
    """Deduplicate, sequence-check and atomically apply normalized source events."""

    def __init__(self, observer: int, rules: dict[str, Any] | None = None) -> None:
        if not isinstance(observer, int) or isinstance(observer, bool) or not 0 <= observer <= 3:
            raise RealtimeStateError("invalid_observer", "observer must be between zero and three")
        rules_json = None if rules is None else canonical_json(rules)
        self.observer = observer
        self._state = json.loads(replay_events_json(observer, "[]", rules_json))
        self._next_source_sequence = 0
        self._round_id: str | None = None
        self._seen: dict[str, str] = {}

    @property
    def state(self) -> dict[str, Any]:
        return self._state

    @property
    def next_source_sequence(self) -> int:
        return self._next_source_sequence

    @property
    def round_id(self) -> str | None:
        return self._round_id

    def apply(self, value: Any) -> EventApplyResult:
        source = parse_majsoul_event(value)
        fingerprint = source.fingerprint
        previous = self._seen.get(source.event_id)
        if previous is not None:
            if previous != fingerprint:
                raise RealtimeStateError(
                    "event_id_conflict", "event_id was reused with different content"
                )
            return EventApplyResult("duplicate", source, None, self._state, self._decision())
        if source.sequence < self._next_source_sequence:
            raise RealtimeStateError(
                "stale_source_event",
                f"source sequence {source.sequence} is older than {self._next_source_sequence}",
            )
        if source.sequence > self._next_source_sequence:
            raise RealtimeStateError(
                "missing_source_event",
                f"expected source sequence {self._next_source_sequence}, got {source.sequence}",
            )
        if source.name == "round_start":
            if source.round_id == self._round_id:
                raise RealtimeStateError(
                    "round_id_reused", "a new round_start must use a new round_id"
                )
        elif self._round_id is None:
            raise RealtimeStateError("round_start_required", "first event must start a round")
        elif source.round_id != self._round_id:
            raise RealtimeStateError(
                "round_id_mismatch",
                f"event round_id {source.round_id!r} does not match {self._round_id!r}",
            )

        game_event = map_majsoul_event(
            source,
            game_sequence=int(self._state["next_event_sequence"]),
            observer=self.observer,
        )
        try:
            updated = json.loads(
                apply_event_json(canonical_json(self._state), canonical_json(game_event))
            )
        except ValueError as error:
            raise RealtimeStateError("invalid_game_event", str(error)) from error

        self._state = updated
        self._next_source_sequence += 1
        if source.name == "round_start":
            self._round_id = source.round_id
        self._remember(source.event_id, fingerprint)
        return EventApplyResult("applied", source, game_event, updated, self._decision())

    def restore(
        self,
        state: Any,
        *,
        next_source_sequence: int,
        round_id: str | None,
    ) -> EventApplyResult:
        if (
            not isinstance(next_source_sequence, int)
            or isinstance(next_source_sequence, bool)
            or next_source_sequence < self._next_source_sequence
        ):
            raise RealtimeStateError(
                "snapshot_rollback",
                f"next_source_sequence must be at least {self._next_source_sequence}",
            )
        if not isinstance(state, dict):
            raise RealtimeStateError("invalid_snapshot", "snapshot state must be an object")
        try:
            normalized = json.loads(validate_state_json(canonical_json(state)))
        except ValueError as error:
            raise RealtimeStateError("invalid_snapshot", str(error)) from error
        if normalized.get("information", {}).get("observer") != self.observer:
            raise RealtimeStateError(
                "observer_mismatch", "snapshot observer does not match the session observer"
            )
        has_round = normalized.get("round") is not None
        if has_round and (not isinstance(round_id, str) or not round_id.strip()):
            raise RealtimeStateError(
                "round_id_required", "an active snapshot requires round_id"
            )
        if not has_round and round_id is not None:
            raise RealtimeStateError(
                "round_id_mismatch", "round_id must be null before a match starts"
            )
        self._state = normalized
        self._next_source_sequence = next_source_sequence
        self._round_id = None if round_id is None else round_id.strip()
        self._seen.clear()
        return EventApplyResult("restored", None, None, normalized, self._decision())

    def validate_assertion(self, assertion: Any) -> dict[str, Any]:
        """Compare a future vision/collector assertion without mutating state."""

        if not isinstance(assertion, dict):
            raise RealtimeStateError("invalid_state_assertion", "assertion must be an object")
        allowed = {
            "round_id",
            "turn",
            "hand_count",
            "dora_count",
            "current_actor",
            "phase",
        }
        unknown = set(assertion) - allowed
        if unknown:
            raise RealtimeStateError(
                "unknown_assertion_fields", f"unsupported assertion fields: {sorted(unknown)}"
            )
        actual = self.summary()
        expected = {
            "round_id": actual["round_id"],
            "turn": actual["turn"],
            "hand_count": len(actual["hand"]),
            "dora_count": len(self._state.get("dora_indicators") or []),
            "current_actor": actual["current_actor"],
            "phase": (actual.get("phase") or {}).get("phase"),
        }
        mismatches = {
            key: {"expected": expected[key], "observed": value}
            for key, value in assertion.items()
            if value != expected[key]
        }
        return {"valid": not mismatches, "mismatches": mismatches, "actual": expected}

    def summary(self, decision: DecisionPoint | None = None) -> dict[str, Any]:
        decision = self._decision() if decision is None else decision
        round_state = self._state.get("round")
        phase = self._state.get("phase") or {}
        hand = list(self._state.get("hand") or [])
        if self._state.get("drawn_tile") is not None:
            hand.append(self._state["drawn_tile"])
        return {
            "round_id": self._round_id,
            "round": round_state,
            "turn": self._state.get("turn"),
            "hand": hand,
            "scores": self._state.get("scores"),
            "remaining_tiles": self._state.get("remaining_tiles"),
            "current_actor": self._state.get("current_actor"),
            "phase": phase,
            "decision_required": decision is not None,
            "decision_kind": None if decision is None else decision.kind,
            "legal_action_count": 0 if decision is None else len(decision.legal_actions),
            "next_source_sequence": self._next_source_sequence,
            "next_game_sequence": self._state.get("next_event_sequence"),
        }

    def _decision(self) -> DecisionPoint | None:
        try:
            actions = json.loads(legal_actions_json(canonical_json(self._state)))
        except ValueError as error:
            raise RealtimeStateError("legal_action_failure", str(error)) from error
        meaningful = [action for action in actions if action.get("type") != "pass"]
        if not meaningful:
            return None
        phase = (self._state.get("phase") or {}).get("phase")
        if phase == "awaiting_discard":
            kind = "self_turn"
        elif phase == "awaiting_kan_response":
            kind = "kan_response"
        else:
            types = {action.get("type") for action in meaningful}
            kind = "ron_response" if types == {"ron"} else "call_or_win_response"
        return DecisionPoint(kind, actions)

    def _remember(self, event_id: str, fingerprint: str) -> None:
        self._seen[event_id] = fingerprint
        if len(self._seen) > 4_096:
            self._seen.pop(next(iter(self._seen)))
