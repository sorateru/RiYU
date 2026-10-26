"""Layout-driven screenshot parser with explicit confidence/unknown handling."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from mahjong_ai import validate_state_json
from mahjong_ai.schema import canonical_json

from .templates import TemplateTileClassifier


@dataclass(frozen=True)
class LayoutConfig:
    image_width: int
    image_height: int
    slots: dict[str, tuple[tuple[int, int, int, int], ...]]

    @classmethod
    def load(cls, path: str | Path) -> "LayoutConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        slots = {
            name: tuple(tuple(int(value) for value in rectangle) for rectangle in rectangles)
            for name, rectangles in raw["slots"].items()
        }
        return cls(int(raw["image_width"]), int(raw["image_height"]), slots)


@dataclass(frozen=True)
class PerceptionResult:
    state: dict[str, Any] | None
    recognized: dict[str, tuple[dict[str, object], ...]]
    confidences: dict[str, tuple[float, ...]]
    unknown_slots: tuple[str, ...]
    source: str


class ScreenshotParser:
    def __init__(
        self,
        layout: LayoutConfig,
        classifier: TemplateTileClassifier,
    ) -> None:
        self.layout = layout
        self.classifier = classifier

    def parse(
        self,
        image: str | Path | np.ndarray,
        base_state: dict[str, Any] | None = None,
    ) -> PerceptionResult:
        source = str(image) if isinstance(image, (str, Path)) else "numpy_array"
        frame = cv2.imread(str(image), cv2.IMREAD_COLOR) if isinstance(image, (str, Path)) else image
        if frame is None:
            raise ValueError(f"cannot read image: {source}")
        if frame.shape[1] != self.layout.image_width or frame.shape[0] != self.layout.image_height:
            raise ValueError(
                f"image size {frame.shape[1]}x{frame.shape[0]} does not match layout "
                f"{self.layout.image_width}x{self.layout.image_height}"
            )
        recognized: dict[str, tuple[dict[str, object], ...]] = {}
        confidences: dict[str, tuple[float, ...]] = {}
        unknown: list[str] = []
        for region, rectangles in self.layout.slots.items():
            tiles: list[dict[str, object]] = []
            scores: list[float] = []
            for index, (x, y, width, height) in enumerate(rectangles):
                if min(x, y, width, height) < 0 or x + width > frame.shape[1] or y + height > frame.shape[0]:
                    raise ValueError(f"slot {region}[{index}] lies outside the image")
                result = self.classifier.classify(frame[y : y + height, x : x + width])
                scores.append(result.confidence)
                if result.tile is None and not result.empty:
                    unknown.append(f"{region}[{index}]:{result.reason}")
                elif result.tile is not None:
                    tiles.append(result.tile)
            recognized[region] = tuple(tiles)
            confidences[region] = tuple(scores)

        state = None
        if base_state is not None and not unknown:
            candidate = copy.deepcopy(base_state)
            if "hand" in recognized:
                candidate["hand"] = list(recognized["hand"])
            if "drawn_tile" in recognized:
                values = recognized["drawn_tile"]
                if len(values) > 1:
                    raise ValueError("drawn_tile region must contain at most one slot")
                candidate["drawn_tile"] = values[0] if values else None
            if "dora_indicators" in recognized:
                candidate["dora_indicators"] = list(recognized["dora_indicators"])
            state = json.loads(validate_state_json(canonical_json(candidate)))
        return PerceptionResult(
            state=state,
            recognized=recognized,
            confidences=confidences,
            unknown_slots=tuple(unknown),
            source=source,
        )
