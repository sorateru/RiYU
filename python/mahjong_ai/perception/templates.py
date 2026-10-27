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
        minimum_margin: float = 0.05,
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
                image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    self.templates.append((None, self._prepare(image)))
                continue
            red = stem.endswith("_red")
            kind = stem.removesuffix("_red")
            kind = _HONOR_ALIASES.get(kind, kind)
            if kind not in TILE_TO_INDEX:
                continue
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
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
        if best_score - runner_up < self.minimum_margin:
            return Classification(None, best_score, runner_up, "ambiguous_templates")
        if best_tile is None:
            return Classification(None, best_score, runner_up, empty=True)
        return Classification(best_tile, best_score, runner_up)

    def _prepare(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(image, self.image_size, interpolation=cv2.INTER_AREA)
        normalized = cv2.equalizeHist(resized)
        return normalized.astype(np.float32) / 255.0

    @staticmethod
    def _similarity(left: np.ndarray, right: np.ndarray) -> float:
        mse = float(np.mean((left - right) ** 2))
        return max(0.0, 1.0 - mse)
