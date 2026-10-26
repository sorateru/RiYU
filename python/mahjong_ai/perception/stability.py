"""Require repeated, high-confidence frames before changing game state."""

from __future__ import annotations

from dataclasses import dataclass
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
        if result.unknown_slots or not self._confident(result.confidences):
            self._candidate = None
            self._candidate_count = 0
            return None
        fingerprint = canonical_json(result.recognized)
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
