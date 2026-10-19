"""Python access to the authoritative Rust mahjong state engine."""

from ._core import (
    FourPlayerSimulator,
    analyze_state_json,
    apply_event_json,
    calculate_shanten_json,
    replay_events_json,
    validate_state_json,
)

__all__ = [
    "FourPlayerSimulator",
    "analyze_state_json",
    "apply_event_json",
    "calculate_shanten_json",
    "replay_events_json",
    "validate_state_json",
]
__version__ = "0.1.0"
