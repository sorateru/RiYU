"""Layout-driven screenshot parser with explicit confidence/unknown handling."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from mahjong_ai import validate_state_json
from mahjong_ai.schema import canonical_json

from .templates import TemplateTileClassifier


Rectangle = tuple[int, int, int, int]


@dataclass(frozen=True)
class TextFieldConfig:
    rectangle: Rectangle
    value_type: str = "integer"
    minimum: int | None = None
    maximum: int | None = None
    rotation: int = 0
    required: bool = False


@dataclass(frozen=True)
class MarkerFieldConfig:
    rectangle: Rectangle
    required: bool = False


@dataclass(frozen=True)
class LayoutConfig:
    image_width: int
    image_height: int
    slots: dict[str, tuple[Rectangle, ...]]
    text_fields: dict[str, TextFieldConfig] = field(default_factory=dict)
    markers: dict[str, MarkerFieldConfig] = field(default_factory=dict)
    rotations: dict[str, int] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "LayoutConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        slots = {
            name: tuple(tuple(int(value) for value in rectangle) for rectangle in rectangles)
            for name, rectangles in raw["slots"].items()
        }
        text_fields = {
            name: TextFieldConfig(
                _rectangle(value["rect"], f"text_fields.{name}"),
                str(value.get("type", "integer")),
                None if value.get("min") is None else int(value["min"]),
                None if value.get("max") is None else int(value["max"]),
                int(value.get("rotation", 0)),
                bool(value.get("required", False)),
            )
            for name, value in (raw.get("text_fields") or {}).items()
        }
        markers = {
            name: MarkerFieldConfig(
                _rectangle(value["rect"], f"markers.{name}"),
                bool(value.get("required", False)),
            )
            for name, value in (raw.get("markers") or {}).items()
        }
        result = cls(
            int(raw["image_width"]),
            int(raw["image_height"]),
            slots,
            text_fields,
            markers,
            {name: int(value) for name, value in (raw.get("rotations") or {}).items()},
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("layout image dimensions must be positive")
        rectangles = [
            (f"slots.{name}[{index}]", rectangle)
            for name, values in self.slots.items()
            for index, rectangle in enumerate(values)
        ]
        rectangles.extend(
            (f"text_fields.{name}", value.rectangle)
            for name, value in self.text_fields.items()
        )
        rectangles.extend(
            (f"markers.{name}", value.rectangle)
            for name, value in self.markers.items()
        )
        for name, (x, y, width, height) in rectangles:
            if (
                min(x, y) < 0
                or width <= 0
                or height <= 0
                or x + width > self.image_width
                or y + height > self.image_height
            ):
                raise ValueError(f"{name} lies outside the image")
        for name, value in self.text_fields.items():
            if value.value_type not in {"integer", "text"}:
                raise ValueError(f"text_fields.{name}.type must be integer or text")
            if value.rotation not in {0, 90, 180, 270}:
                raise ValueError(f"text_fields.{name}.rotation must be 0, 90, 180, or 270")
            if (
                value.minimum is not None
                and value.maximum is not None
                and value.minimum > value.maximum
            ):
                raise ValueError(f"text_fields.{name} has min greater than max")
        for name, rotation in self.rotations.items():
            if name not in self.slots:
                raise ValueError(f"rotations.{name} does not name a slot region")
            if rotation not in {0, 90, 180, 270}:
                raise ValueError(f"rotations.{name} must be 0, 90, 180, or 270")


@dataclass(frozen=True)
class PerceptionResult:
    state: dict[str, Any] | None
    recognized: dict[str, tuple[dict[str, object], ...]]
    confidences: dict[str, tuple[float, ...]]
    unknown_slots: tuple[str, ...]
    source: str
    metadata: dict[str, int | str] = field(default_factory=dict)
    metadata_confidences: dict[str, float] = field(default_factory=dict)
    markers: dict[str, bool | None] = field(default_factory=dict)
    marker_confidences: dict[str, float] = field(default_factory=dict)


class ScreenshotParser:
    def __init__(
        self,
        layout: LayoutConfig,
        classifier: TemplateTileClassifier,
        annotation_parser: Any | None = None,
    ) -> None:
        self.layout = layout
        self.classifier = classifier
        self.annotation_parser = annotation_parser

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
                crop = rotate_slot_crop(
                    frame[y : y + height, x : x + width],
                    self.layout.rotations.get(region, 0),
                )
                result = self.classifier.classify(crop)
                scores.append(result.confidence)
                if result.tile is None and not result.empty:
                    unknown.append(f"{region}[{index}]:{result.reason}")
                elif result.tile is not None:
                    tiles.append(result.tile)
            recognized[region] = tuple(tiles)
            confidences[region] = tuple(scores)

        metadata: dict[str, int | str] = {}
        metadata_confidences: dict[str, float] = {}
        markers: dict[str, bool | None] = {}
        marker_confidences: dict[str, float] = {}
        if self.annotation_parser is not None:
            annotations = self.annotation_parser.parse(frame)
            metadata = annotations.metadata
            metadata_confidences = annotations.metadata_confidences
            markers = annotations.markers
            marker_confidences = annotations.marker_confidences
            unknown.extend(annotations.required_unknown)

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
            metadata=metadata,
            metadata_confidences=metadata_confidences,
            markers=markers,
            marker_confidences=marker_confidences,
        )


def _rectangle(value: Any, name: str) -> Rectangle:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"{name}.rect must contain x, y, width, height")
    return tuple(int(item) for item in value)  # type: ignore[return-value]


def rotate_slot_crop(crop: np.ndarray, rotation: int) -> np.ndarray:
    if rotation == 0:
        return crop
    if rotation == 90:
        return cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
    if rotation == 180:
        return cv2.rotate(crop, cv2.ROTATE_180)
    if rotation == 270:
        return cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError("slot rotation must be 0, 90, 180, or 270")
