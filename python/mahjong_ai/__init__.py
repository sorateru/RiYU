"""Python access to the authoritative Rust mahjong state engine."""

import json

from ._core import (
    FourPlayerSimulator,
    analyze_state_json,
    apply_event_json,
    calculate_shanten_json,
    replay_events_json,
    score_hand_json,
    validate_state_json,
)

try:
    from ._core import ObservationReplayer, legal_actions_json
except ImportError:  # Compatibility with an extension built before this API.
    def legal_actions_json(state_json: str) -> str:
        analysis = json.loads(analyze_state_json(state_json))
        return json.dumps(analysis["legal_actions"]["value"] or [], separators=(",", ":"))

    class ObservationReplayer:
        def __init__(self, observer: int, rules_json: str | None = None) -> None:
            self._state_json = replay_events_json(observer, "[]", rules_json)

        def state_json(self) -> str:
            return self._state_json

        def apply_event_json(self, event_json: str) -> None:
            self._state_json = apply_event_json(self._state_json, event_json)

        def legal_actions_json(self) -> str:
            return legal_actions_json(self._state_json)

        def next_event_sequence(self) -> int:
            return int(json.loads(self._state_json)["next_event_sequence"])

__all__ = [
    "FourPlayerSimulator",
    "ObservationReplayer",
    "analyze_state_json",
    "apply_event_json",
    "calculate_shanten_json",
    "legal_actions_json",
    "replay_events_json",
    "score_hand_json",
    "validate_state_json",
]
__version__ = "0.1.0"
