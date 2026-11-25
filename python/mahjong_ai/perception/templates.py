"""Deterministic OpenCV template classifier for offline screenshots."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

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
    candidates: tuple[dict[str, Any], ...] = ()
    template_difference: float | None = None
    red_ratio: float | None = None


@dataclass(frozen=True)
class VisualClassification:
    label: str | None
    confidence: float
    runner_up_confidence: float
    reason: str | None = None
    candidates: tuple[dict[str, Any], ...] = ()


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
        root = Path(template_directory)
        self.metadata = _load_metadata(root)
        self.templates: list[
            tuple[dict[str, object] | None, np.ndarray, Path, dict[str, Any] | None]
        ] = []
        for path in sorted(root.glob("*.png")):
            # ``__suffix`` allows multiple visual variants (seat/orientation/theme)
            # without changing the semantic label used by the engine.
            stem = path.stem.split("__", 1)[0]
            if stem == "empty":
                image = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if image is not None:
                    self.templates.append(
                        (None, self._prepare(image), path, self.metadata.get(path.name))
                    )
                continue
            red = stem.endswith("_red")
            kind = stem.removesuffix("_red")
            kind = _HONOR_ALIASES.get(kind, kind)
            if kind not in TILE_TO_INDEX:
                continue
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            self.templates.append(
                (
                    {"kind": kind, "red": red},
                    self._prepare(image),
                    path,
                    self.metadata.get(path.name),
                )
            )
        if not self.templates:
            raise ValueError("template directory contains no recognized tile PNG files")

    def classify(self, crop: np.ndarray) -> Classification:
        query = self._prepare(crop)
        red_ratio = float(np.mean(query[1]))
        # Visual variants of the same semantic tile (seat, theme, animation
        # frame, or empty-slot background) are alternatives, not competing
        # labels.  Compare the best variant for each label so two near-identical
        # variants cannot make an otherwise exact match look ambiguous.
        best_by_label: dict[
            tuple[str, bool] | None,
            tuple[
                float,
                dict[str, object] | None,
                Path,
                dict[str, Any] | None,
            ],
        ] = {}
        for tile, template, path, metadata in self.templates:
            label = (
                None
                if tile is None
                else (str(tile["kind"]), bool(tile.get("red", False)))
            )
            score = self._similarity(query, template)
            previous = best_by_label.get(label)
            if previous is None or score > previous[0]:
                best_by_label[label] = (score, tile, path, metadata)
        scores = sorted(best_by_label.values(), key=lambda item: item[0], reverse=True)
        best_score, best_tile, _best_path, _best_metadata = scores[0]
        runner_up = scores[1][0] if len(scores) > 1 else 0.0
        candidates = tuple(
            {
                "label": _label(candidate_tile),
                "confidence": float(score),
                "template": str(path),
            }
            for score, candidate_tile, path, _metadata in scores[:2]
        )
        difference = max(0.0, 1.0 - best_score)
        if best_score < self.minimum_confidence:
            return Classification(
                None,
                best_score,
                runner_up,
                "confidence_below_threshold",
                candidates=candidates,
                template_difference=difference,
                red_ratio=red_ratio,
            )
        required_margin = (
            min(self.minimum_margin, 0.01)
            if best_tile is None
            else self.minimum_margin
        )
        if best_score - runner_up < required_margin:
            return Classification(
                None,
                best_score,
                runner_up,
                "ambiguous_templates",
                candidates=candidates,
                template_difference=difference,
                red_ratio=red_ratio,
            )
        if best_tile is None:
            return Classification(
                None,
                best_score,
                runner_up,
                empty=True,
                candidates=candidates,
                template_difference=difference,
                red_ratio=red_ratio,
            )
        return Classification(
            best_tile,
            best_score,
            runner_up,
            candidates=candidates,
            template_difference=difference,
            red_ratio=red_ratio,
        )

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


class VisualTemplateClassifier:
    """Match a small shared set of non-semantic tile visuals.

    These templates live below ``templates/visuals`` and are intentionally
    separate from semantic tile labels.  A tile back, for example, must never
    become a valid result in a hand or river slot merely because an ankan uses
    the same artwork.  Callers therefore request the allowed label explicitly.
    """

    def __init__(
        self,
        template_directory: str | Path,
        *,
        minimum_confidence: float = 0.72,
        minimum_margin: float = 0.03,
    ) -> None:
        self.minimum_confidence = minimum_confidence
        self.minimum_margin = minimum_margin
        root = Path(template_directory) / "visuals"
        templates: dict[str, list[np.ndarray]] = {}
        for path in sorted(root.glob("*.png")):
            label = path.stem.split("__", 1)[0]
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is not None:
                templates.setdefault(label, []).append(self._prepare(image))
        self.templates = {name: tuple(values) for name, values in templates.items()}

    def classify(self, crop: np.ndarray, allowed: set[str]) -> VisualClassification:
        query = self._prepare(crop)
        scores: list[tuple[float, str]] = []
        for label in sorted(allowed):
            values = self.templates.get(label, ())
            if values:
                scores.append((max(self._similarity(query, value) for value in values), label))
        if not scores:
            return VisualClassification(None, 0.0, 0.0, "visual_template_missing")
        scores.sort(reverse=True)
        best_score, best_label = scores[0]
        runner_up = scores[1][0] if len(scores) > 1 else 0.0
        candidates = tuple(
            {"label": label, "confidence": float(score)}
            for score, label in scores[:2]
        )
        if best_score < self.minimum_confidence:
            return VisualClassification(
                None, best_score, runner_up, "confidence_below_threshold", candidates
            )
        if len(scores) > 1 and best_score - runner_up < self.minimum_margin:
            return VisualClassification(
                None, best_score, runner_up, "ambiguous_templates", candidates
            )
        return VisualClassification(best_label, best_score, runner_up, candidates=candidates)

    @staticmethod
    def _prepare(image: np.ndarray) -> np.ndarray:
        color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image
        resized = cv2.resize(color, (48, 64), interpolation=cv2.INTER_AREA)
        lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB).astype(np.float32) / 255.0
        return lab

    @staticmethod
    def _similarity(left: np.ndarray, right: np.ndarray) -> float:
        difference = float(np.mean(np.abs(left - right)))
        return max(0.0, 1.0 - 2.0 * difference)


def _label(tile: dict[str, object] | None) -> str:
    if tile is None:
        return "empty"
    return f"{tile['kind']}{'_red' if tile.get('red') else ''}"


def _load_metadata(root: Path) -> dict[str, dict[str, Any]]:
    """Load optional local template provenance without requiring it at runtime."""

    path = root / "template_manifest.json"
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("format") != "riyu-vision-template-manifest/v1":
        raise ValueError("unsupported vision template manifest format")
    entries = raw.get("templates")
    if not isinstance(entries, dict):
        raise ValueError("template manifest templates must be an object")
    output: dict[str, dict[str, Any]] = {}
    for name, value in entries.items():
        if not isinstance(name, str) or not isinstance(value, dict):
            raise ValueError("template manifest entries must be objects keyed by filename")
        output[name] = dict(value)
    return output
