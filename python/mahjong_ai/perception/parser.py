"""Layout-driven screenshot parser with explicit confidence/unknown handling."""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from mahjong_ai import validate_state_json
from mahjong_ai.schema import TILE_TO_INDEX, canonical_json

from .templates import TemplateTileClassifier, VisualTemplateClassifier

if TYPE_CHECKING:
    from .alignment import SceneAligner


Rectangle = tuple[int, int, int, int]
_MELD_REGION = re.compile(r"^meld_([0-3])_([0-3])$")


@dataclass(frozen=True)
class TextFieldConfig:
    rectangle: Rectangle
    value_type: str = "integer"
    minimum: int | None = None
    maximum: int | None = None
    rotation: int = 0
    required: bool = False
    ocr_psm: int | None = None
    ocr_minimum_confidence: float | None = None


@dataclass(frozen=True)
class MarkerFieldConfig:
    rectangle: Rectangle
    required: bool = False


@dataclass(frozen=True)
class SceneAnchorConfig:
    rectangle: Rectangle
    search_rectangle: Rectangle | None = None
    minimum_confidence: float = 0.82
    required: bool = True
    minimum_scale: float = 0.9
    maximum_scale: float = 1.1
    scale_step: float = 0.05
    align: bool = False


@dataclass(frozen=True)
class SlotVariantConfig:
    rectangle: Rectangle
    rotation: int = 0
    orientation: str = "vertical"


@dataclass(frozen=True)
class MeldSlotConfig:
    rectangle: Rectangle
    role: str = "tile"
    rotation: int = 0
    called: bool = False


@dataclass(frozen=True)
class MeldPatternConfig:
    name: str
    kind: str
    slots: tuple[MeldSlotConfig, ...]
    from_offset: int | None = None


@dataclass(frozen=True)
class MeldScanConfig:
    """Article-style right-to-left scan of one normalized meld area.

    The three scan rectangles use an x offset relative to the moving right
    cut-line.  This mirrors the Mahjong Soul technique described in Qiita Vol.4:
    compare upright and sideways candidates, then advance by the width of the
    selected orientation.  ``perspective_destination`` is optional so synthetic
    and already-normalized captures do not pay for a warp.
    """

    rectangle: Rectangle
    horizontal_rect: Rectangle
    vertical_rect: Rectangle
    upper_horizontal_rect: Rectangle
    horizontal_step: int
    vertical_step: int
    max_tiles: int = 4
    rotation: int = 0
    perspective_destination: tuple[tuple[float, float], ...] | None = None
    from_offsets: tuple[int | None, ...] = ()


@dataclass(frozen=True)
class LayoutConfig:
    image_width: int
    image_height: int
    slots: dict[str, tuple[Rectangle, ...]]
    text_fields: dict[str, TextFieldConfig] = field(default_factory=dict)
    markers: dict[str, MarkerFieldConfig] = field(default_factory=dict)
    rotations: dict[str, int] = field(default_factory=dict)
    scene_anchors: dict[str, SceneAnchorConfig] = field(default_factory=dict)
    slot_variants: dict[str, tuple[tuple[SlotVariantConfig, ...], ...]] = field(
        default_factory=dict
    )
    meld_layouts: dict[str, tuple[MeldPatternConfig, ...]] = field(default_factory=dict)
    meld_scans: dict[str, MeldScanConfig] = field(default_factory=dict)

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
                None if value.get("ocr_psm") is None else int(value["ocr_psm"]),
                (
                    None
                    if value.get("ocr_minimum_confidence") is None
                    else float(value["ocr_minimum_confidence"])
                ),
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
        scene_anchors = {
            name: SceneAnchorConfig(
                _rectangle(value["rect"], f"scene_anchors.{name}"),
                (
                    None
                    if value.get("search_rect") is None
                    else _rectangle(value["search_rect"], f"scene_anchors.{name}.search_rect")
                ),
                float(value.get("minimum_confidence", 0.82)),
                bool(value.get("required", True)),
                float(value.get("minimum_scale", 0.9)),
                float(value.get("maximum_scale", 1.1)),
                float(value.get("scale_step", 0.05)),
                bool(value.get("align", False)),
            )
            for name, value in (raw.get("scene_anchors") or {}).items()
        }
        slot_variants = {
            name: tuple(
                tuple(
                    SlotVariantConfig(
                        _rectangle(candidate["rect"], f"slot_variants.{name}[{index}]"),
                        int(candidate.get("rotation", 0)),
                        str(candidate.get("orientation", "vertical")),
                    )
                    for candidate in candidates
                )
                for index, candidates in enumerate(values)
            )
            for name, values in (raw.get("slot_variants") or {}).items()
        }
        meld_layouts = {
            name: tuple(
                MeldPatternConfig(
                    str(pattern.get("name", pattern["kind"])),
                    str(pattern["kind"]),
                    tuple(
                        MeldSlotConfig(
                            _rectangle(
                                slot["rect"],
                                f"meld_layouts.{name}.{pattern.get('name', pattern['kind'])}",
                            ),
                            str(slot.get("role", "tile")),
                            int(slot.get("rotation", 0)),
                            bool(slot.get("called", False)),
                        )
                        for slot in pattern["slots"]
                    ),
                    (
                        None
                        if pattern.get("from_offset") is None
                        else int(pattern["from_offset"])
                    ),
                )
                for pattern in patterns
            )
            for name, patterns in (raw.get("meld_layouts") or {}).items()
        }
        meld_scans = {
            name: MeldScanConfig(
                _rectangle(value["rect"], f"meld_scans.{name}"),
                _scan_rectangle(
                    value["horizontal_rect"], f"meld_scans.{name}.horizontal_rect"
                ),
                _scan_rectangle(
                    value["vertical_rect"], f"meld_scans.{name}.vertical_rect"
                ),
                _scan_rectangle(
                    value["upper_horizontal_rect"],
                    f"meld_scans.{name}.upper_horizontal_rect",
                ),
                int(value["horizontal_step"]),
                int(value["vertical_step"]),
                int(value.get("max_tiles", 4)),
                int(value.get("rotation", 0)),
                _perspective_destination(
                    value.get("perspective_destination"),
                    f"meld_scans.{name}.perspective_destination",
                ),
                tuple(
                    None if item is None else int(item)
                    for item in value.get("from_offsets", [])
                ),
            )
            for name, value in (raw.get("meld_scans") or {}).items()
        }
        result = cls(
            int(raw["image_width"]),
            int(raw["image_height"]),
            slots,
            text_fields,
            markers,
            {name: int(value) for name, value in (raw.get("rotations") or {}).items()},
            scene_anchors,
            slot_variants,
            meld_layouts,
            meld_scans,
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
            (f"scene_anchors.{name}", value.rectangle)
            for name, value in self.scene_anchors.items()
        )
        rectangles.extend(
            (f"scene_anchors.{name}.search_rect", value.search_rectangle)
            for name, value in self.scene_anchors.items()
            if value.search_rectangle is not None
        )
        rectangles.extend(
            (f"markers.{name}", value.rectangle)
            for name, value in self.markers.items()
        )
        rectangles.extend(
            (f"slot_variants.{name}[{index}]", candidate.rectangle)
            for name, values in self.slot_variants.items()
            for index, candidates in enumerate(values)
            for candidate in candidates
        )
        rectangles.extend(
            (f"meld_layouts.{name}.{pattern.name}", slot.rectangle)
            for name, patterns in self.meld_layouts.items()
            for pattern in patterns
            for slot in pattern.slots
        )
        rectangles.extend(
            (f"meld_scans.{name}", value.rectangle)
            for name, value in self.meld_scans.items()
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
            if value.ocr_psm not in {None, 6, 7, 10, 13}:
                raise ValueError(
                    f"text_fields.{name}.ocr_psm must be 6, 7, 10, or 13"
                )
            if (
                value.ocr_minimum_confidence is not None
                and not 0 <= value.ocr_minimum_confidence <= 1
            ):
                raise ValueError(
                    f"text_fields.{name}.ocr_minimum_confidence must be in 0..=1"
                )
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
        for name, value in self.scene_anchors.items():
            if not 0 <= value.minimum_confidence <= 1:
                raise ValueError(f"scene_anchors.{name}.minimum_confidence must be in 0..=1")
            if not 0.5 <= value.minimum_scale <= value.maximum_scale <= 2.0:
                raise ValueError(f"scene_anchors.{name} scale range must be within 0.5..=2")
            if not 0.005 <= value.scale_step <= 0.25:
                raise ValueError(f"scene_anchors.{name}.scale_step must be in 0.005..=0.25")
        for name, values in self.slot_variants.items():
            if name not in self.slots:
                raise ValueError(f"slot_variants.{name} does not name a slot region")
            if len(values) != len(self.slots[name]):
                raise ValueError(
                    f"slot_variants.{name} must contain one candidate list per logical slot"
                )
            for index, candidates in enumerate(values):
                if not candidates:
                    raise ValueError(f"slot_variants.{name}[{index}] has no candidates")
                for candidate in candidates:
                    if candidate.rotation not in {0, 90, 180, 270}:
                        raise ValueError(
                            f"slot_variants.{name}[{index}] rotation is invalid"
                        )
                    if candidate.orientation not in {"vertical", "horizontal"}:
                        raise ValueError(
                            f"slot_variants.{name}[{index}] orientation must be vertical or horizontal"
                        )
        for name, patterns in self.meld_layouts.items():
            if not _MELD_REGION.fullmatch(name):
                raise ValueError(f"meld_layouts.{name} must be named meld_<player>_<group>")
            if not patterns:
                raise ValueError(f"meld_layouts.{name} has no patterns")
            for pattern in patterns:
                if pattern.kind not in {
                    "empty", "chi", "pon", "daiminkan", "ankan", "shouminkan"
                }:
                    raise ValueError(f"meld_layouts.{name}.{pattern.name} kind is invalid")
                if not pattern.slots:
                    raise ValueError(f"meld_layouts.{name}.{pattern.name} has no slots")
                if pattern.from_offset not in {None, 1, 2, 3}:
                    raise ValueError(
                        f"meld_layouts.{name}.{pattern.name}.from_offset must be 1, 2, or 3"
                    )
                called_slots = [slot for slot in pattern.slots if slot.called]
                if pattern.kind in {"empty", "ankan"}:
                    if called_slots or pattern.from_offset is not None:
                        raise ValueError(
                            f"meld_layouts.{name}.{pattern.name} cannot have called/from metadata"
                        )
                else:
                    if len(called_slots) != 1 or called_slots[0].role != "tile":
                        raise ValueError(
                            f"meld_layouts.{name}.{pattern.name} needs exactly one called tile"
                        )
                    if pattern.from_offset is None:
                        raise ValueError(
                            f"meld_layouts.{name}.{pattern.name} needs from_offset"
                        )
                    if pattern.kind == "chi" and pattern.from_offset != 3:
                        raise ValueError(
                            f"meld_layouts.{name}.{pattern.name} chi must come from offset 3"
                        )
                for slot in pattern.slots:
                    if slot.role not in {"tile", "back", "empty"}:
                        raise ValueError(
                            f"meld_layouts.{name}.{pattern.name} slot role is invalid"
                        )
                    if slot.rotation not in {0, 90, 180, 270}:
                        raise ValueError(
                            f"meld_layouts.{name}.{pattern.name} slot rotation is invalid"
                        )
        for name, scan in self.meld_scans.items():
            if not _MELD_REGION.fullmatch(name):
                raise ValueError(f"meld_scans.{name} must be named meld_<player>_<group>")
            if scan.rotation not in {0, 90, 180, 270}:
                raise ValueError(f"meld_scans.{name}.rotation is invalid")
            if scan.horizontal_step <= 0 or scan.vertical_step <= 0:
                raise ValueError(f"meld_scans.{name} steps must be positive")
            if not 1 <= scan.max_tiles <= 4:
                raise ValueError(f"meld_scans.{name}.max_tiles must be in 1..=4")
            for field_name, rectangle in (
                ("horizontal_rect", scan.horizontal_rect),
                ("vertical_rect", scan.vertical_rect),
                ("upper_horizontal_rect", scan.upper_horizontal_rect),
            ):
                _x_offset, y, width, height = rectangle
                if y < 0 or width <= 0 or height <= 0:
                    raise ValueError(f"meld_scans.{name}.{field_name} is invalid")
            if any(value not in {None, 1, 2, 3} for value in scan.from_offsets):
                raise ValueError(f"meld_scans.{name}.from_offsets values are invalid")


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
    scene_found: bool = True
    alignment_confidence: float | None = None
    alignment_offset: tuple[float, float] | None = None
    alignment_scale: float | None = None
    alignment_rotation_degrees: float | None = None
    slot_diagnostics: tuple[dict[str, Any], ...] = ()
    slot_states: dict[str, tuple[dict[str, Any], ...]] = field(default_factory=dict)
    melds: dict[str, dict[str, Any]] = field(default_factory=dict)
    meld_diagnostics: tuple[dict[str, Any], ...] = ()


class ScreenshotParser:
    def __init__(
        self,
        layout: LayoutConfig,
        classifier: TemplateTileClassifier,
        annotation_parser: Any | None = None,
        scene_aligner: "SceneAligner | None" = None,
        visual_classifier: VisualTemplateClassifier | None = None,
    ) -> None:
        self.layout = layout
        self.classifier = classifier
        self.annotation_parser = annotation_parser
        self.scene_aligner = scene_aligner
        self.visual_classifier = visual_classifier

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
        alignment = self.scene_aligner.align(frame) if self.scene_aligner is not None else None
        if alignment is not None and not alignment.found:
            return PerceptionResult(
                state=None,
                recognized={},
                confidences={},
                unknown_slots=(f"scene:{alignment.reason or 'game_screen_not_found'}",),
                source=source,
                scene_found=False,
                alignment_confidence=alignment.confidence,
            )
        if alignment is not None:
            frame = alignment.frame
        recognized: dict[str, tuple[dict[str, object], ...]] = {}
        confidences: dict[str, tuple[float, ...]] = {}
        slot_states: dict[str, tuple[dict[str, Any], ...]] = {}
        melds: dict[str, dict[str, Any]] = {}
        unknown: list[str] = []
        slot_diagnostics: list[dict[str, Any]] = []
        meld_diagnostics: list[dict[str, Any]] = []
        for region, rectangles in self.layout.slots.items():
            if region in self.layout.meld_layouts or region in self.layout.meld_scans:
                continue
            tiles: list[dict[str, object]] = []
            scores: list[float] = []
            states: list[dict[str, Any]] = []
            for index, (x, y, width, height) in enumerate(rectangles):
                variants = self.layout.slot_variants.get(region)
                if variants is None:
                    candidates = (
                        SlotVariantConfig(
                            (x, y, width, height),
                            self.layout.rotations.get(region, 0),
                            "vertical",
                        ),
                    )
                else:
                    candidates = variants[index]
                selected, candidate_diagnostics = self._classify_slot_variants(
                    frame, region, index, candidates
                )
                if selected is None:
                    scores.append(
                        max(
                            (float(item["confidence"]) for item in candidate_diagnostics),
                            default=0.0,
                        )
                    )
                    unknown.append(f"{region}[{index}]:variant_ambiguous")
                    slot_diagnostics.extend(candidate_diagnostics)
                    states.append(
                        {"index": index, "label": None, "orientation": None}
                    )
                    continue
                candidate, result = selected
                x, y, width, height = candidate.rectangle
                scores.append(result.confidence)
                chosen_diagnostic = next(
                    item for item in candidate_diagnostics if item.get("selected")
                )
                slot_diagnostics.extend(candidate_diagnostics)
                states.append(
                    {
                        "index": index,
                        "label": chosen_diagnostic["label"],
                        "orientation": candidate.orientation,
                        "rectangle": [x, y, width, height],
                        "confidence": result.confidence,
                    }
                )
                if result.tile is None and not result.empty:
                    unknown.append(f"{region}[{index}]:{result.reason}")
                elif result.tile is not None:
                    tiles.append(result.tile)
            recognized[region] = tuple(tiles)
            confidences[region] = tuple(scores)
            slot_states[region] = tuple(states)

        meld_regions = set(self.layout.meld_layouts) | set(self.layout.meld_scans)
        for region in sorted(meld_regions):
            scan = self.layout.meld_scans.get(region)
            if scan is not None:
                meld, tiles, scores, diagnostics, reason = self._scan_meld(
                    frame, region, scan
                )
            else:
                meld, tiles, scores, diagnostics, reason = self._classify_meld(
                    frame, region, self.layout.meld_layouts[region]
                )
            recognized[region] = tiles
            confidences[region] = scores
            meld_diagnostics.extend(diagnostics)
            if meld is not None:
                melds[region] = meld
            if reason is not None:
                unknown.append(f"{region}:{reason}")

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
            scene_found=True,
            alignment_confidence=(alignment.confidence if alignment is not None else None),
            alignment_offset=(alignment.offset if alignment is not None else None),
            alignment_scale=(alignment.scale if alignment is not None else None),
            alignment_rotation_degrees=(
                alignment.rotation_degrees if alignment is not None else None
            ),
            slot_diagnostics=tuple(slot_diagnostics),
            slot_states=slot_states,
            melds=melds,
            meld_diagnostics=tuple(meld_diagnostics),
        )

    def _classify_slot_variants(
        self,
        frame: np.ndarray,
        region: str,
        index: int,
        candidates: tuple[SlotVariantConfig, ...],
    ) -> tuple[
        tuple[SlotVariantConfig, Any] | None,
        list[dict[str, Any]],
    ]:
        classified: list[tuple[float, SlotVariantConfig, Any]] = []
        diagnostics: list[dict[str, Any]] = []
        for candidate in candidates:
            x, y, width, height = candidate.rectangle
            crop = rotate_slot_crop(
                frame[y : y + height, x : x + width], candidate.rotation
            )
            result = self.classifier.classify(crop)
            classified.append((result.confidence, candidate, result))
        classified.sort(key=lambda item: item[0], reverse=True)
        selected: tuple[SlotVariantConfig, Any] | None = None
        if classified:
            best_score, best_candidate, best_result = classified[0]
            runner_up = classified[1][0] if len(classified) > 1 else 0.0
            if (
                best_result.reason is None
                and (
                    len(classified) == 1
                    or best_score - runner_up >= self.classifier.minimum_margin
                )
            ):
                selected = (best_candidate, best_result)
        for _score, candidate, result in classified:
            x, y, width, height = candidate.rectangle
            diagnostics.append(
                {
                    "region": region,
                    "index": index,
                    "rectangle": [x, y, width, height],
                    "rotation": candidate.rotation,
                    "orientation": candidate.orientation,
                    "selected": selected is not None and candidate == selected[0],
                    "label": (
                        "empty"
                        if result.empty
                        else None
                        if result.tile is None
                        else _tile_label(result.tile)
                    ),
                    "confidence": result.confidence,
                    "runner_up_confidence": result.runner_up_confidence,
                    "reason": result.reason,
                    "candidates": list(result.candidates),
                    "template_difference": result.template_difference,
                    "red_ratio": result.red_ratio,
                }
            )
        return selected, diagnostics

    def _scan_meld(
        self,
        frame: np.ndarray,
        region: str,
        scan: MeldScanConfig,
    ) -> tuple[
        dict[str, Any] | None,
        tuple[dict[str, object], ...],
        tuple[float, ...],
        list[dict[str, Any]],
        str | None,
    ]:
        """Scan a meld with the same geometry-first flow as Qiita Vol.4.

        Unlike the reference script, an ambiguous choice aborts the entire
        meld instead of moving the cut-line.  A wrong first stride would shift
        every later crop and is therefore more dangerous than an unknown meld.
        """

        match = _MELD_REGION.fullmatch(region)
        assert match is not None
        player = int(match.group(1))
        x, y, width, height = scan.rectangle
        normalized = rotate_slot_crop(frame[y : y + height, x : x + width], scan.rotation)
        if scan.perspective_destination is not None:
            normalized = _warp_meld_area(normalized, scan.perspective_destination)

        cut_line = normalized.shape[1]
        visible_tiles: list[dict[str, object]] = []
        scores: list[float] = []
        diagnostics: list[dict[str, Any]] = []
        orientations: list[str] = []
        back_indices: list[int] = []
        called_base_index: int | None = None
        stacked_tile: dict[str, object] | None = None

        for base_index in range(scan.max_tiles):
            horizontal_crop = _cut_line_crop(normalized, cut_line, scan.horizontal_rect)
            vertical_crop = _cut_line_crop(normalized, cut_line, scan.vertical_rect)
            if horizontal_crop is None or vertical_crop is None:
                return None, (), tuple(scores), diagnostics, "scan_out_of_bounds"
            horizontal = self.classifier.classify(
                rotate_slot_crop(horizontal_crop, 90)
            )
            vertical = self.classifier.classify(vertical_crop)
            selected = _select_scan_orientation(
                horizontal,
                vertical,
                self.classifier.minimum_margin,
            )
            detail: dict[str, Any] = {
                "mode": "right-to-left-scan",
                "region": region,
                "index": base_index,
                "scan_rectangle": [x, y, width, height],
                "cut_line": cut_line,
                "horizontal": _classification_diagnostic(horizontal),
                "vertical": _classification_diagnostic(vertical),
                "selected": None if selected is None else selected[0],
            }

            # Ankan backs are visual-only.  They are checked only after the
            # semantic classifier refuses the upright crop, so a back can never
            # become a legal hand/river tile or compete with white dragon.
            if selected is None and self.visual_classifier is not None:
                back = self.visual_classifier.classify(vertical_crop, {"back"})
                detail["back"] = {
                    "label": back.label,
                    "confidence": back.confidence,
                    "reason": back.reason,
                    "candidates": list(back.candidates),
                }
                if back.label == "back":
                    selected = ("back", back.confidence, None)

            diagnostics.append(detail)
            if selected is None:
                return None, (), tuple(scores), diagnostics, "scan_orientation_ambiguous"
            orientation, confidence, result = selected
            if orientation == "empty":
                if base_index == 0:
                    return None, (), tuple(scores + [confidence]), diagnostics, None
                break
            scores.append(confidence)
            orientations.append(orientation)
            if orientation == "back":
                back_indices.append(base_index)
                cut_line -= scan.vertical_step
                continue
            assert result is not None and result.tile is not None
            visible_tiles.append(result.tile)
            if orientation == "horizontal":
                if called_base_index is not None:
                    return None, (), tuple(scores), diagnostics, "multiple_sideways_tiles"
                called_base_index = base_index
                upper_crop = _cut_line_crop(
                    normalized, cut_line, scan.upper_horizontal_rect
                )
                if upper_crop is None:
                    return None, (), tuple(scores), diagnostics, "upper_scan_out_of_bounds"
                upper = self.classifier.classify(rotate_slot_crop(upper_crop, 90))
                detail["upper_horizontal"] = _classification_diagnostic(upper)
                if upper.tile is not None:
                    if _tile_label(upper.tile) != _tile_label(result.tile):
                        return None, (), tuple(scores), diagnostics, "stacked_tile_mismatch"
                    stacked_tile = upper.tile
                    scores.append(upper.confidence)
                elif not upper.empty and upper.reason != "confidence_below_threshold":
                    return None, (), tuple(scores), diagnostics, "upper_tile_ambiguous"
                cut_line -= scan.horizontal_step
            else:
                cut_line -= scan.vertical_step

        kind, tiles, called_index, reason = _normalize_scanned_meld(
            visible_tiles,
            orientations,
            back_indices,
            called_base_index,
            stacked_tile,
        )
        if reason is not None:
            return None, (), tuple(scores), diagnostics, reason
        assert kind is not None and tiles is not None
        if kind == "ankan":
            from_player = None
        else:
            if called_base_index is None:
                return None, (), tuple(scores), diagnostics, "called_tile_ambiguous"
            if kind == "chi":
                from_offset = 3
            else:
                from_offset = (
                    scan.from_offsets[called_base_index]
                    if called_base_index < len(scan.from_offsets)
                    else None
                )
            if from_offset not in {1, 2, 3}:
                return None, (), tuple(scores), diagnostics, "source_ambiguous"
            from_player = (player + from_offset) % 4
        meld = {
            "kind": kind,
            "tiles": tuple(tiles),
            "called_index": called_index,
            "from": from_player,
            "confidence": min(scores) if scores else 0.0,
            "pattern": "right-to-left-scan",
        }
        return meld, tuple(tiles), tuple(scores), diagnostics, None

    def _classify_meld(
        self,
        frame: np.ndarray,
        region: str,
        patterns: tuple[MeldPatternConfig, ...],
    ) -> tuple[
        dict[str, Any] | None,
        tuple[dict[str, object], ...],
        tuple[float, ...],
        list[dict[str, Any]],
        str | None,
    ]:
        match = _MELD_REGION.fullmatch(region)
        assert match is not None
        player = int(match.group(1))
        accepted: list[
            tuple[float, MeldPatternConfig, list[dict[str, object]], list[float], int | None]
        ] = []
        diagnostics: list[dict[str, Any]] = []
        for pattern in patterns:
            tiles: list[dict[str, object]] = []
            scores: list[float] = []
            slot_details: list[dict[str, Any]] = []
            called_index: int | None = None
            rejection: str | None = None
            for slot in pattern.slots:
                x, y, width, height = slot.rectangle
                crop = rotate_slot_crop(
                    frame[y : y + height, x : x + width], slot.rotation
                )
                if slot.role == "back":
                    if self.visual_classifier is None:
                        rejection = "visual_template_missing"
                        break
                    visual = self.visual_classifier.classify(crop, {"back"})
                    scores.append(visual.confidence)
                    slot_details.append(
                        {
                            "rectangle": [x, y, width, height],
                            "role": slot.role,
                            "rotation": slot.rotation,
                            "label": visual.label,
                            "confidence": visual.confidence,
                            "runner_up_confidence": visual.runner_up_confidence,
                            "reason": visual.reason,
                            "candidates": list(visual.candidates),
                        }
                    )
                    if visual.label != "back":
                        rejection = visual.reason or "back_not_recognized"
                        break
                    continue
                result = self.classifier.classify(crop)
                scores.append(result.confidence)
                slot_details.append(
                    {
                        "rectangle": [x, y, width, height],
                        "role": slot.role,
                        "rotation": slot.rotation,
                        "label": (
                            "empty"
                            if result.empty
                            else None
                            if result.tile is None
                            else _tile_label(result.tile)
                        ),
                        "confidence": result.confidence,
                        "runner_up_confidence": result.runner_up_confidence,
                        "reason": result.reason,
                        "candidates": list(result.candidates),
                        "template_difference": result.template_difference,
                        "red_ratio": result.red_ratio,
                    }
                )
                if slot.role == "empty":
                    if not result.empty:
                        rejection = result.reason or "empty_guard_failed"
                        break
                    continue
                if result.tile is None:
                    rejection = result.reason or "tile_not_recognized"
                    break
                if slot.called:
                    if called_index is not None:
                        rejection = "multiple_called_slots"
                        break
                    called_index = len(tiles)
                tiles.append(result.tile)
            if rejection is None:
                normalized = _normalize_meld_pattern(pattern, tiles)
                if normalized is None:
                    rejection = "invalid_shape"
                else:
                    tiles = normalized
                    accepted.append(
                        (
                            min(scores) if scores else 0.0,
                            pattern,
                            tiles,
                            scores,
                            called_index,
                        )
                    )
            diagnostics.append(
                {
                    "region": region,
                    "pattern": pattern.name,
                    "kind": pattern.kind,
                    "confidence": min(scores) if scores else 0.0,
                    "rejection": rejection,
                    "tiles": [_tile_label(tile) for tile in tiles],
                    "slots": slot_details,
                }
            )
        accepted.sort(key=lambda item: item[0], reverse=True)
        if not accepted:
            return None, (), (), diagnostics, "no_meld_layout_matched"
        if (
            len(accepted) > 1
            and accepted[0][0] - accepted[1][0] < self.classifier.minimum_margin
        ):
            return None, (), (), diagnostics, "meld_layout_ambiguous"
        score, pattern, tiles, scores, called_index = accepted[0]
        for diagnostic in diagnostics:
            diagnostic["selected"] = diagnostic["pattern"] == pattern.name
        if pattern.kind == "empty":
            return None, (), tuple(scores), diagnostics, None
        if pattern.kind != "ankan" and called_index is None:
            return None, (), tuple(scores), diagnostics, "called_tile_ambiguous"
        from_player = (
            None
            if pattern.kind == "ankan" or pattern.from_offset is None
            else (player + pattern.from_offset) % 4
        )
        if pattern.kind != "ankan" and from_player is None:
            return None, (), tuple(scores), diagnostics, "source_ambiguous"
        meld = {
            "kind": pattern.kind,
            "tiles": tuple(tiles),
            "called_index": called_index,
            "from": from_player,
            "confidence": score,
            "pattern": pattern.name,
        }
        return meld, tuple(tiles), tuple(scores), diagnostics, None


def _classification_diagnostic(result: Any) -> dict[str, Any]:
    return {
        "label": (
            "empty"
            if result.empty
            else None
            if result.tile is None
            else _tile_label(result.tile)
        ),
        "confidence": result.confidence,
        "runner_up_confidence": result.runner_up_confidence,
        "reason": result.reason,
        "candidates": list(result.candidates),
        "template_difference": result.template_difference,
        "red_ratio": result.red_ratio,
    }


def _select_scan_orientation(
    horizontal: Any,
    vertical: Any,
    minimum_margin: float,
) -> tuple[str, float, Any | None] | None:
    candidates: list[tuple[float, str, Any]] = []
    for orientation, result in (("horizontal", horizontal), ("vertical", vertical)):
        if result.reason is None and (result.tile is not None or result.empty):
            candidates.append((float(result.confidence), orientation, result))
    candidates.sort(reverse=True)
    if not candidates:
        return None
    if len(candidates) > 1 and all(result.empty for _score, _orientation, result in candidates):
        score, _orientation, result = candidates[0]
        return "empty", score, result
    if len(candidates) > 1 and candidates[0][0] - candidates[1][0] < minimum_margin:
        return None
    score, orientation, result = candidates[0]
    return ("empty" if result.empty else orientation, score, result)


def _cut_line_crop(
    image: np.ndarray,
    cut_line: int,
    rectangle: Rectangle,
) -> np.ndarray | None:
    x_offset, y, width, height = rectangle
    x = cut_line + x_offset
    if (
        x < 0
        or y < 0
        or width <= 0
        or height <= 0
        or x + width > image.shape[1]
        or y + height > image.shape[0]
    ):
        return None
    return image[y : y + height, x : x + width]


def _warp_meld_area(
    image: np.ndarray,
    destination: tuple[tuple[float, float], ...],
) -> np.ndarray:
    height, width = image.shape[:2]
    source = np.float32(
        ((0, 0), (0, height - 1), (width - 1, height - 1), (width - 1, 0))
    )
    matrix = cv2.getPerspectiveTransform(source, np.float32(destination))
    return cv2.warpPerspective(image, matrix, (width, height))


def _normalize_scanned_meld(
    visible_tiles: list[dict[str, object]],
    orientations: list[str],
    back_indices: list[int],
    called_base_index: int | None,
    stacked_tile: dict[str, object] | None,
) -> tuple[
    str | None,
    list[dict[str, object]] | None,
    int | None,
    str | None,
]:
    if back_indices:
        if (
            len(orientations) != 4
            or len(back_indices) != 2
            or len(visible_tiles) != 2
            or called_base_index is not None
            or stacked_tile is not None
            or len({_tile_label(tile) for tile in visible_tiles}) != 1
        ):
            return None, None, None, "invalid_ankan_shape"
        return "ankan", [dict(visible_tiles[0]) for _ in range(4)], None, None

    if called_base_index is None:
        return None, None, None, "called_tile_ambiguous"
    if stacked_tile is not None:
        tiles = [*visible_tiles, stacked_tile]
        if len(visible_tiles) != 3 or len({_tile_label(tile) for tile in tiles}) != 1:
            return None, None, None, "invalid_shouminkan_shape"
        return "shouminkan", tiles, called_base_index, None
    labels = [_tile_label(tile) for tile in visible_tiles]
    if len(visible_tiles) == 4:
        if len(set(labels)) != 1:
            return None, None, None, "invalid_daiminkan_shape"
        return "daiminkan", visible_tiles, called_base_index, None
    if len(visible_tiles) != 3:
        return None, None, None, "invalid_meld_tile_count"
    if len(set(labels)) == 1:
        return "pon", visible_tiles, called_base_index, None

    indexes = [TILE_TO_INDEX.get(str(tile["kind"])) for tile in visible_tiles]
    if any(value is None or int(value) >= 27 for value in indexes):
        return None, None, None, "invalid_chi_shape"
    indexed = sorted(
        zip((int(value) for value in indexes if value is not None), visible_tiles),
        key=lambda item: item[0],
    )
    values = [value for value, _tile in indexed]
    if values[0] // 9 != values[-1] // 9 or values != list(range(values[0], values[0] + 3)):
        return None, None, None, "invalid_chi_shape"
    called_label = labels[called_base_index]
    sorted_tiles = [tile for _value, tile in indexed]
    called_index = next(
        index for index, tile in enumerate(sorted_tiles) if _tile_label(tile) == called_label
    )
    return "chi", sorted_tiles, called_index, None


def _rectangle(value: Any, name: str) -> Rectangle:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"{name}.rect must contain x, y, width, height")
    return tuple(int(item) for item in value)  # type: ignore[return-value]


def _scan_rectangle(value: Any, name: str) -> Rectangle:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"{name} must contain x_offset, y, width, height")
    return tuple(int(item) for item in value)  # type: ignore[return-value]


def _perspective_destination(
    value: Any,
    name: str,
) -> tuple[tuple[float, float], ...] | None:
    if value is None:
        return None
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(not isinstance(point, list) or len(point) != 2 for point in value)
    ):
        raise ValueError(f"{name} must contain four [x, y] points")
    return tuple((float(point[0]), float(point[1])) for point in value)


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


def _normalize_meld_pattern(
    pattern: MeldPatternConfig,
    tiles: list[dict[str, object]],
) -> list[dict[str, object]] | None:
    kinds = [str(tile["kind"]) for tile in tiles]
    if pattern.kind == "empty":
        return [] if not tiles else None
    if pattern.kind == "chi":
        if len(tiles) != 3:
            return None
        indexes = [TILE_TO_INDEX.get(kind) for kind in kinds]
        if any(value is None or int(value) >= 27 for value in indexes):
            return None
        values = sorted(int(value) for value in indexes if value is not None)
        if values[0] // 9 != values[-1] // 9 or values != list(range(values[0], values[0] + 3)):
            return None
        return tiles
    if pattern.kind == "pon":
        return tiles if len(tiles) == 3 and len(set(kinds)) == 1 else None
    if pattern.kind in {"daiminkan", "shouminkan"}:
        return tiles if len(tiles) == 4 and len(set(kinds)) == 1 else None
    if pattern.kind == "ankan":
        if len(tiles) != 2 or len(set(kinds)) != 1:
            return None
        return [dict(tiles[0]) for _ in range(4)]
    return None


def _tile_label(tile: dict[str, object]) -> str:
    return f"{tile['kind']}{'_red' if tile.get('red') else ''}"
