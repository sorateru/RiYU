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

    def push(self, result: PerceptionResult) -> StableObservation | None:
        terminal = any(
            value is True and name.startswith("result_")
            for name, value in result.markers.items()
        )
        annotation_unknown = tuple(
            value
            for value in result.unknown_slots
            if value.startswith("text_fields.") or value.startswith("markers.")
        )
        if (
            annotation_unknown
            or (not terminal and result.unknown_slots)
            or (not terminal and not self._confident(result.confidences))
        ):
            self._candidate = None
            self._candidate_count = 0
            return None
        fresh_round = _looks_like_fresh_round(result.recognized)
        event_markers = {
            name: value
            for name, value in result.markers.items()
            if name.startswith("riichi_") or name.startswith("tsumogiri_")
        }
        payload = {
            "recognized": {} if terminal else result.recognized,
            "metadata": result.metadata if terminal or fresh_round else {},
            "markers": result.markers if terminal or fresh_round else event_markers,
        }
        fingerprint = canonical_json(payload)
        if fingerprint == self._candidate:
            self._candidate_count += 1
        else:
            self._candidate = fingerprint
            self._candidate_count = 1
        if self._candidate_count < self.required_frames or fingerprint == self._last_emitted:
            return None
        self._last_emitted = fingerprint
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
        )

    def _confident(self, values: dict[str, tuple[float, ...]]) -> bool:
        all_scores = [score for scores in values.values() for score in scores]
        return bool(all_scores) and min(all_scores) >= self.minimum_confidence


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
