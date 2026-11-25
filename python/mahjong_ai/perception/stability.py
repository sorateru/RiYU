"""Require repeated, high-confidence frames before changing game state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mahjong_ai.schema import canonical_json

from .parser import PerceptionResult


@dataclass(frozen=True)
class StableObservation:
    recognized: dict[str, tuple[dict[str, object], ...]]
    confidences: dict[str, tuple[float, ...]]
    fingerprint: str
    confirmation_frames: int
    source: str
    metadata: dict[str, int | str] = field(default_factory=dict)
    metadata_confidences: dict[str, float] = field(default_factory=dict)
    markers: dict[str, bool | None] = field(default_factory=dict)
    marker_confidences: dict[str, float] = field(default_factory=dict)
    slot_states: dict[str, tuple[dict[str, Any], ...]] = field(default_factory=dict)
    melds: dict[str, dict[str, Any]] = field(default_factory=dict)


class StableObservationFilter:
    def __init__(self, required_frames: int = 3, minimum_confidence: float = 0.78) -> None:
        if not 2 <= required_frames <= 30:
            raise ValueError("required_frames must be in 2..=30")
        if not 0.0 <= minimum_confidence <= 1.0:
            raise ValueError("minimum_confidence must be in 0..=1")
        self.required_frames = required_frames
        self.minimum_confidence = minimum_confidence
        self._candidate: str | None = None
        self._candidate_count = 0
        self._last_emitted: str | None = None
        self.last_status: dict[str, Any] = {
            "reason": "waiting_for_frame",
            "candidate_frames": 0,
            "required_frames": required_frames,
        }

    def push(self, result: PerceptionResult) -> StableObservation | None:
        if not result.scene_found:
            self._candidate = None
            self._candidate_count = 0
            self.last_status = {
                "reason": "game_screen_not_found",
                "candidate_frames": 0,
                "required_frames": self.required_frames,
                "minimum_confidence": result.alignment_confidence,
                "unknown_slots": list(result.unknown_slots[:32]),
                "recognized_counts": {},
                "alignment_confidence": result.alignment_confidence,
            }
            return None
        terminal = any(
            value is True and name.startswith("result_")
            for name, value in result.markers.items()
        ) or all(
            isinstance(result.metadata.get(f"delta_{player}"), int)
            for player in range(4)
        )
        annotation_unknown = tuple(
            value
            for value in result.unknown_slots
            if value.startswith("text_fields.") or value.startswith("markers.")
        )
        minimum_observed = self._minimum_confidence(result.confidences)
        if annotation_unknown or (not terminal and result.unknown_slots):
            self._candidate = None
            self._candidate_count = 0
            self.last_status = {
                "reason": "unknown_slots",
                "candidate_frames": 0,
                "required_frames": self.required_frames,
                "minimum_confidence": minimum_observed,
                "unknown_slots": list(result.unknown_slots[:32]),
                "recognized_counts": {
                    name: len(values) for name, values in result.recognized.items()
                },
            }
            return None
        if not terminal and not self._confident(result.confidences):
            self._candidate = None
            self._candidate_count = 0
            self.last_status = {
                "reason": "low_confidence",
                "candidate_frames": 0,
                "required_frames": self.required_frames,
                "minimum_confidence": minimum_observed,
                "unknown_slots": [],
                "recognized_counts": {
                    name: len(values) for name, values in result.recognized.items()
                },
            }
            return None
        payload = {
            "recognized": {} if terminal else result.recognized,
            # OCR and UI marker values participate in the same multi-frame
            # confirmation as tiles. This is essential for safe mid-round
            # restoration: a single plausible score or dealer OCR result must
            # never be paired with an otherwise stable tile snapshot.
            "metadata": result.metadata,
            "markers": result.markers,
            "slot_states": result.slot_states,
            "melds": result.melds,
        }
        fingerprint = canonical_json(payload)
        if fingerprint == self._candidate:
            self._candidate_count += 1
        else:
            self._candidate = fingerprint
            self._candidate_count = 1
        if self._candidate_count < self.required_frames:
            self.last_status = {
                "reason": "stabilizing",
                "candidate_frames": self._candidate_count,
                "required_frames": self.required_frames,
                "minimum_confidence": minimum_observed,
                "unknown_slots": [],
                "recognized_counts": {
                    name: len(values) for name, values in result.recognized.items()
                },
            }
            return None
        if fingerprint == self._last_emitted:
            self.last_status = {
                "reason": "stable_unchanged",
                "candidate_frames": self._candidate_count,
                "required_frames": self.required_frames,
                "minimum_confidence": minimum_observed,
                "unknown_slots": [],
                "recognized_counts": {
                    name: len(values) for name, values in result.recognized.items()
                },
            }
            return None
        self._last_emitted = fingerprint
        self.last_status = {
            "reason": "stable",
            "candidate_frames": self._candidate_count,
            "required_frames": self.required_frames,
            "minimum_confidence": minimum_observed,
            "unknown_slots": [],
            "recognized_counts": {
                name: len(values) for name, values in result.recognized.items()
            },
        }
        return StableObservation(
            result.recognized,
            result.confidences,
            fingerprint,
            self._candidate_count,
            result.source,
            result.metadata,
            result.metadata_confidences,
            result.markers,
            result.marker_confidences,
            result.slot_states,
            result.melds,
        )

    def _confident(self, values: dict[str, tuple[float, ...]]) -> bool:
        all_scores = [score for scores in values.values() for score in scores]
        return bool(all_scores) and min(all_scores) >= self.minimum_confidence

    @staticmethod
    def _minimum_confidence(values: dict[str, tuple[float, ...]]) -> float | None:
        all_scores = [score for scores in values.values() for score in scores]
        return min(all_scores) if all_scores else None


def observed_tiles(
    observation: StableObservation | dict[str, Any], region: str
) -> tuple[dict[str, object], ...]:
    recognized = (
        observation.recognized
        if isinstance(observation, StableObservation)
        else observation
    )
    return tuple(recognized.get(region, ()))


def observed_metadata(
    observation: StableObservation | dict[str, Any], name: str
) -> int | str | None:
    metadata = (
        observation.metadata
        if isinstance(observation, StableObservation)
        else observation
    )
    return metadata.get(name)


def observed_marker(
    observation: StableObservation | dict[str, Any], name: str
) -> bool | None:
    markers = (
        observation.markers
        if isinstance(observation, StableObservation)
        else observation
    )
    value = markers.get(name)
    return value if isinstance(value, bool) else None


def observed_slot_state(
    observation: StableObservation | dict[str, Any], region: str, index: int
) -> dict[str, Any] | None:
    values = (
        observation.slot_states
        if isinstance(observation, StableObservation)
        else observation
    ).get(region, ())
    if not isinstance(values, (list, tuple)) or not 0 <= index < len(values):
        return None
    value = values[index]
    return value if isinstance(value, dict) else None


def observed_meld(
    observation: StableObservation | dict[str, Any], region: str
) -> dict[str, Any] | None:
    values = (
        observation.melds
        if isinstance(observation, StableObservation)
        else observation
    )
    value = values.get(region)
    return value if isinstance(value, dict) else None


def _looks_like_fresh_round(
    recognized: dict[str, tuple[dict[str, object], ...]],
) -> bool:
    if len(recognized.get("hand", ())) != 13:
        return False
    if recognized.get("drawn_tile", ()):
        return False
    if len(recognized.get("dora_indicators", ())) != 1:
        return False
    for name, values in recognized.items():
        if (name.startswith("river_") or name.startswith("meld_")) and values:
            return False
    return True
