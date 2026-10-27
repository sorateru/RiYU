"""Deterministic OpenCV template classifier for offline screenshots."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from mahjong_ai.schema import TILE_TO_INDEX

_HONOR_ALIASES = {
    "z1": "east",
    "z2": "south",
    "z3": "west",
    "z4": "north",
    "z5": "white",
    "z6": "green",
    "z7": "red",
}


@dataclass(frozen=True)
class Classification:
    tile: dict[str, object] | None
    confidence: float
    runner_up_confidence: float
    reason: str | None = None
    empty: bool = False


class TemplateTileClassifier:
    def __init__(
        self,
        template_directory: str | Path,
        image_size: tuple[int, int] = (40, 60),
        minimum_confidence: float = 0.72,
        minimum_margin: float = 0.03,
    ) -> None:
        self.image_size = image_size
        self.minimum_confidence = minimum_confidence
        self.minimum_margin = minimum_margin
        self.templates: list[tuple[dict[str, object] | None, np.ndarray]] = []
        for path in sorted(Path(template_directory).glob("*.png")):
            # ``__suffix`` allows multiple visual variants (seat/orientation/theme)
            # without changing the semantic label used by the engine.
            stem = path.stem.split("__", 1)[0]
            if stem == "empty":
                image = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if image is not None:
                    self.templates.append((None, self._prepare(image)))
                continue
            red = stem.endswith("_red")
            kind = stem.removesuffix("_red")
            kind = _HONOR_ALIASES.get(kind, kind)
            if kind not in TILE_TO_INDEX:
                continue
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            self.templates.append(({"kind": kind, "red": red}, self._prepare(image)))
        if not self.templates:
            raise ValueError("template directory contains no recognized tile PNG files")

    def classify(self, crop: np.ndarray) -> Classification:
        query = self._prepare(crop)
        # Visual variants of the same semantic tile (seat, theme, animation
        # frame, or empty-slot background) are alternatives, not competing
        # labels.  Compare the best variant for each label so two near-identical
        # variants cannot make an otherwise exact match look ambiguous.
        best_by_label: dict[
            tuple[str, bool] | None,
            tuple[float, dict[str, object] | None],
        ] = {}
        for tile, template in self.templates:
            label = (
                None
                if tile is None
                else (str(tile["kind"]), bool(tile.get("red", False)))
            )
            score = self._similarity(query, template)
            previous = best_by_label.get(label)
            if previous is None or score > previous[0]:
                best_by_label[label] = (score, tile)
        scores = sorted(best_by_label.values(), key=lambda item: item[0], reverse=True)
        best_score, best_tile = scores[0]
        runner_up = scores[1][0] if len(scores) > 1 else 0.0
        if best_score < self.minimum_confidence:
            return Classification(None, best_score, runner_up, "confidence_below_threshold")
        required_margin = (
            min(self.minimum_margin, 0.01)
            if best_tile is None
            else self.minimum_margin
        )
        if best_score - runner_up < required_margin:
            return Classification(None, best_score, runner_up, "ambiguous_templates")
        if best_tile is None:
            return Classification(None, best_score, runner_up, empty=True)
        return Classification(best_tile, best_score, runner_up)

    def _prepare(self, image: np.ndarray) -> np.ndarray:
        color = (
            cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            if image.ndim == 2
            else image
        )
        resized = cv2.resize(color, self.image_size, interpolation=cv2.INTER_AREA)
        # Comparing the whole gray tile made the large, nearly identical white
        # background dominate the score.  Work on the interior ink mask instead:
        # this keeps 3m/4m, 6p/7p and 4s/6s well separated while remaining
        # tolerant of small lighting changes in a live screen capture.
        height, width = resized.shape[:2]
        left = max(1, round(width * 0.10))
        right = min(width, width - left)
        top = max(1, round(height * 0.05))
        bottom = min(height, height - max(1, round(height * 0.08)))
        interior = resized[top:bottom, left:right]
        gray = cv2.cvtColor(interior, cv2.COLOR_BGR2GRAY)
        # A fixed screen-space threshold is intentional. Otsu becomes unstable
        # on an empty, low-contrast board crop: a one-level grayscale rounding
        # difference between color capture and PNG reload can flip most pixels.
        # Mahjong Soul tile faces remain much brighter than both ink and board.
        _threshold, ink = cv2.threshold(
            gray,
            180,
            1,
            cv2.THRESH_BINARY_INV,
        )
        hsv = cv2.cvtColor(interior, cv2.COLOR_BGR2HSV)
        hue, saturation, value = cv2.split(hsv)
        red = (
            (((hue <= 10) | (hue >= 170)) & (saturation >= 80) & (value >= 60))
            .astype(np.float32)
        )
        return np.stack((ink.astype(np.float32), red), axis=0)

    @staticmethod
    def _similarity(left: np.ndarray, right: np.ndarray) -> float:
        ink_difference = float(np.mean(np.abs(left[0] - right[0])))
        red_difference = float(np.mean(np.abs(left[1] - right[1])))
        difference = 0.75 * ink_difference + 0.25 * red_difference
        return max(0.0, 1.0 - difference)
