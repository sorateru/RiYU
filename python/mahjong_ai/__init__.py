"""Python access to the authoritative Rust mahjong state engine."""

from ._core import apply_event_json, replay_events_json, validate_state_json

__all__ = ["apply_event_json", "replay_events_json", "validate_state_json"]
__version__ = "0.1.0"

