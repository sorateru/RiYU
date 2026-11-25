from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np
import pytest

from mahjong_ai.ai.sequence import public_event_tokens
from mahjong_ai.analysis import RealtimeStateManager
from mahjong_ai.perception import (
    CaptureRegion,
    Classification,
    LayoutConfig,
    MeldPatternConfig,
    MeldScanConfig,
    MeldSlotConfig,
    MarkerFieldConfig,
    SceneAligner,
    SceneAnchorConfig,
    MarkerTemplateBank,
    PerceptionResult,
    ScreenshotParser,
    SlotVariantConfig,
    StableObservation,
    StableObservationFilter,
    TemplateTileClassifier,
    TesseractFieldRecognizer,
    TextFieldConfig,
    VisionAdapterConfig,
    VisionAdapterRuntime,
    VisionEventDetector,
    VisualTemplateClassifier,
)
from mahjong_ai.perception.profile import (
    export_vision_profile,
    import_vision_profile,
    inspect_vision_profile,
    prune_template_metadata,
    record_template_metadata,
)


HAND = ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "1p", "2p", "3p", "7s", "1z", "5z"]


def tile(kind: str, red: bool = False) -> dict[str, object]:
    return {"kind": kind, "red": red}


def source_event(sequence: int, name: str, data: dict) -> dict:
    return {
        "event_id": f"event-{sequence}",
        "sequence": sequence,
        "round_id": "east-1-0",
        "name": name,
        "data": data,
    }


def start(manager: RealtimeStateManager, dealer: int = 0) -> None:
    manager.apply(
        source_event(
            0,
            "round_start",
            {
                "wind": "east",
                "hand_number": 1,
                "honba": 0,
                "riichi_sticks": 0,
                "dealer": dealer,
                "scores": [25_000] * 4,
                "initial_hand": HAND,
                "dora_indicator": "4z",
                "remaining_tiles": 70,
            },
        )
    )


def observation(
    hand: tuple[dict[str, object], ...],
    *,
    drawn: tuple[dict[str, object], ...] = (),
    rivers: dict[int, tuple[dict[str, object], ...]] | None = None,
    metadata: dict[str, int | str] | None = None,
    markers: dict[str, bool | None] | None = None,
) -> StableObservation:
    recognized = {
        "hand": hand,
        "drawn_tile": drawn,
        "dora_indicators": (tile("north"),),
        **{f"river_{player}": () for player in range(4)},
        **{
            f"meld_{player}_{group}": ()
            for player in range(4)
            for group in range(4)
        },
    }
    for player, values in (rivers or {}).items():
        recognized[f"river_{player}"] = values
    return StableObservation(
        recognized,
        {},
        repr((recognized, metadata, markers)),
        3,
        "test",
        metadata or {},
        {},
        markers or {},
        {},
    )


def restorable_observation(
    hand: tuple[dict[str, object], ...],
    *,
    drawn: dict[str, object] | None = None,
    melds: dict[tuple[int, int], tuple[dict[str, object], ...]] | None = None,
    extra_markers: dict[str, bool | None] | None = None,
    metadata_overrides: dict[str, int | str] | None = None,
    rivers: dict[int, tuple[dict[str, object], ...]] | None = None,
    dora: tuple[dict[str, object], ...] = (tile("north"),),
) -> StableObservation:
    metadata: dict[str, int | str] = {
        "hand_number": 1,
        "honba": 0,
        "riichi_sticks": 0,
        "remaining_tiles": 66,
        "score_0": 25_000,
        "score_1": 25_000,
        "score_2": 25_000,
        "score_3": 25_000,
    }
    metadata.update(metadata_overrides or {})
    markers: dict[str, bool | None] = {
        "wind_east": True,
        "dealer_0": True,
        **{f"riichi_{player}": False for player in range(4)},
        **{f"double_riichi_{player}": False for player in range(4)},
    }
    markers.update(extra_markers or {})
    value = observation(
        hand,
        drawn=() if drawn is None else (drawn,),
        rivers=rivers,
        metadata=metadata,
        markers=markers,
    )
    value.recognized["dora_indicators"] = dora
    for (player, group), tiles in (melds or {}).items():
        value.recognized[f"meld_{player}_{group}"] = tiles
    return value


def make_templates(directory: Path) -> None:
    directory.mkdir()
    empty = np.zeros((12, 8), dtype=np.uint8)
    empty[:, 4:] = 255
    one = np.zeros((12, 8), dtype=np.uint8)
    one[6:, :] = 255
    assert cv2.imwrite(str(directory / "empty__bottom.png"), empty)
    assert cv2.imwrite(str(directory / "m1__bottom.png"), one)


def make_runtime(tmp_path: Path) -> VisionAdapterRuntime:
    tmp_path.mkdir(parents=True, exist_ok=True)
    templates = tmp_path / "templates"
    make_templates(templates)
    replay = tmp_path / "replay"
    replay.mkdir()
    assert cv2.imwrite(str(replay / "000.png"), np.zeros((12, 8, 3), dtype=np.uint8))
    layout = tmp_path / "layout.json"
    layout.write_text(
        json.dumps({"image_width": 8, "image_height": 12, "slots": {"hand": []}}),
        encoding="utf-8",
    )
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "session_id": "restore-test",
                "observer": 0,
                "layout": "layout.json",
                "templates": "templates",
                "capture": {"mode": "directory_replay", "directory": "replay"},
                "bootstrap": {"dealer": 0, "scores": [25_000] * 4},
            }
        ),
        encoding="utf-8",
    )
    return VisionAdapterRuntime(VisionAdapterConfig.load(config))


def test_capture_region_and_template_empty_slot(tmp_path: Path) -> None:
    assert CaptureRegion(10, 20, 30, 40).macos_argument == "10,20,30,40"
    with pytest.raises(ValueError):
        CaptureRegion(-1, 0, 10, 10)

    templates = tmp_path / "templates"
    make_templates(templates)
    layout = LayoutConfig(8, 12, {"hand": ((0, 0, 8, 12),)})
    parser = ScreenshotParser(layout, TemplateTileClassifier(templates))
    frame = cv2.imread(str(templates / "empty__bottom.png"), cv2.IMREAD_COLOR)
    result = parser.parse(frame)
    assert result.recognized["hand"] == ()
    assert result.unknown_slots == ()


def test_same_semantic_template_variants_do_not_compete(tmp_path: Path) -> None:
    templates = tmp_path / "templates"
    make_templates(templates)
    duplicate = cv2.imread(str(templates / "empty__bottom.png"), cv2.IMREAD_GRAYSCALE)
    assert duplicate is not None
    assert cv2.imwrite(str(templates / "empty__top.png"), duplicate)

    classifier = TemplateTileClassifier(templates)
    result = classifier.classify(duplicate)

    assert result.empty is True
    assert result.reason is None
    assert result.confidence > result.runner_up_confidence


def test_tile_ink_outweighs_shared_white_background(tmp_path: Path) -> None:
    templates = tmp_path / "templates"
    templates.mkdir()
    three = np.full((60, 40), 240, dtype=np.uint8)
    four = three.copy()
    cv2.line(three, (10, 15), (30, 15), 0, 3)
    cv2.line(three, (10, 25), (30, 25), 0, 3)
    cv2.line(three, (10, 35), (30, 35), 0, 3)
    cv2.line(four, (10, 12), (30, 12), 0, 3)
    cv2.line(four, (10, 22), (30, 22), 0, 3)
    cv2.line(four, (10, 32), (30, 32), 0, 3)
    cv2.line(four, (10, 42), (30, 42), 0, 3)
    assert cv2.imwrite(str(templates / "m3.png"), three)
    assert cv2.imwrite(str(templates / "m4.png"), four)

    result = TemplateTileClassifier(templates).classify(three)

    assert result.tile == tile("m3")
    assert result.confidence - result.runner_up_confidence >= 0.05


def test_red_five_uses_color_evidence_and_keeps_ambiguous_crop_unknown(
    tmp_path: Path,
) -> None:
    templates = tmp_path / "templates"
    templates.mkdir()
    normal = np.full((60, 40, 3), 240, dtype=np.uint8)
    red = normal.copy()
    cv2.rectangle(normal, (8, 12), (31, 47), (40, 150, 40), -1)
    cv2.rectangle(red, (8, 12), (31, 47), (20, 20, 220), -1)
    assert cv2.imwrite(str(templates / "p5.png"), normal)
    assert cv2.imwrite(str(templates / "p5_red.png"), red)
    classifier = TemplateTileClassifier(templates)

    red_result = classifier.classify(red)
    ambiguous = normal.copy()
    ambiguous[:, :20] = red[:, :20]
    ambiguous_result = classifier.classify(ambiguous)

    assert red_result.tile == tile("p5", True)
    assert red_result.red_ratio is not None and red_result.red_ratio > 0.2
    assert ambiguous_result.tile is None
    assert ambiguous_result.reason == "ambiguous_templates"


def test_layout_rotates_side_seat_crop_before_classification(tmp_path: Path) -> None:
    templates = tmp_path / "templates"
    make_templates(templates)
    upright = cv2.imread(str(templates / "m1__bottom.png"), cv2.IMREAD_COLOR)
    assert upright is not None
    sideways = cv2.rotate(upright, cv2.ROTATE_90_COUNTERCLOCKWISE)
    layout = LayoutConfig(
        12,
        8,
        {"river_1": ((0, 0, 12, 8),)},
        rotations={"river_1": 90},
    )

    result = ScreenshotParser(layout, TemplateTileClassifier(templates)).parse(sideways)

    assert result.recognized["river_1"] == (tile("m1"),)

    with pytest.raises(ValueError, match="does not name a slot region"):
        LayoutConfig(12, 8, {}, rotations={"river_1": 90}).validate()


def test_slot_variants_detect_horizontal_riichi_tile(tmp_path: Path) -> None:
    templates = tmp_path / "templates"
    make_templates(templates)
    empty = np.zeros((12, 8), dtype=np.uint8)
    empty[::2, ::2] = 255
    empty[1::2, 1::2] = 255
    assert cv2.imwrite(str(templates / "empty__bottom.png"), empty)
    upright = cv2.imread(str(templates / "m1__bottom.png"), cv2.IMREAD_COLOR)
    assert upright is not None
    sideways = cv2.rotate(upright, cv2.ROTATE_90_COUNTERCLOCKWISE)
    layout = LayoutConfig(
        12,
        8,
        {"river_0": ((0, 0, 12, 8),)},
        slot_variants={
            "river_0": (
                (
                    SlotVariantConfig((0, 0, 12, 8), 0, "vertical"),
                    SlotVariantConfig((0, 0, 12, 8), 90, "horizontal"),
                ),
            )
        },
    )
    layout.validate()

    result = ScreenshotParser(layout, TemplateTileClassifier(templates)).parse(sideways)

    assert result.recognized["river_0"] == (tile("m1"),)
    assert result.slot_states["river_0"][0]["orientation"] == "horizontal"
    assert result.unknown_slots == ()


def test_meld_geometry_recognizes_pon_without_per_region_markers(tmp_path: Path) -> None:
    templates = tmp_path / "templates"
    make_templates(templates)
    upright = cv2.imread(str(templates / "m1__bottom.png"), cv2.IMREAD_COLOR)
    assert upright is not None
    frame = np.zeros((12, 24, 3), dtype=np.uint8)
    for left in (0, 8, 16):
        frame[:, left : left + 8] = upright
    pattern = MeldPatternConfig(
        "pon-right",
        "pon",
        (
            MeldSlotConfig((0, 0, 8, 12)),
            MeldSlotConfig((8, 0, 8, 12), called=True),
            MeldSlotConfig((16, 0, 8, 12)),
        ),
        from_offset=1,
    )
    layout = LayoutConfig(
        24,
        12,
        {"meld_0_0": ()},
        meld_layouts={"meld_0_0": (pattern,)},
    )
    layout.validate()
    parser = ScreenshotParser(
        layout,
        TemplateTileClassifier(templates),
        visual_classifier=VisualTemplateClassifier(templates),
    )

    result = parser.parse(frame)

    assert result.unknown_slots == ()
    assert result.recognized["meld_0_0"] == (tile("m1"),) * 3
    assert result.melds["meld_0_0"]["kind"] == "pon"
    assert result.melds["meld_0_0"]["called_index"] == 1
    assert result.melds["meld_0_0"]["from"] == 1


def test_meld_geometry_ankan_uses_one_shared_back_template(tmp_path: Path) -> None:
    templates = tmp_path / "templates"
    make_templates(templates)
    visual_root = templates / "visuals"
    visual_root.mkdir()
    back = np.full((12, 8, 3), (180, 80, 20), dtype=np.uint8)
    cv2.rectangle(back, (1, 1), (6, 10), (230, 160, 70), 1)
    assert cv2.imwrite(str(visual_root / "back__default.png"), back)
    upright = cv2.imread(str(templates / "m1__bottom.png"), cv2.IMREAD_COLOR)
    assert upright is not None
    frame = np.zeros((12, 32, 3), dtype=np.uint8)
    frame[:, 0:8] = back
    frame[:, 8:16] = upright
    frame[:, 16:24] = upright
    frame[:, 24:32] = back
    pattern = MeldPatternConfig(
        "ankan",
        "ankan",
        (
            MeldSlotConfig((0, 0, 8, 12), "back"),
            MeldSlotConfig((8, 0, 8, 12)),
            MeldSlotConfig((16, 0, 8, 12)),
            MeldSlotConfig((24, 0, 8, 12), "back"),
        ),
    )
    layout = LayoutConfig(
        32,
        12,
        {"meld_0_0": ()},
        meld_layouts={"meld_0_0": (pattern,)},
    )
    parser = ScreenshotParser(
        layout,
        TemplateTileClassifier(templates),
        visual_classifier=VisualTemplateClassifier(templates),
    )

    result = parser.parse(frame)

    assert result.unknown_slots == ()
    assert result.recognized["meld_0_0"] == (tile("m1"),) * 4
    assert result.melds["meld_0_0"]["kind"] == "ankan"


class _SequenceTileClassifier:
    minimum_margin = 0.03

    def __init__(self, values: list[Classification]) -> None:
        self.values = values

    def classify(self, _crop: np.ndarray) -> Classification:
        assert self.values
        return self.values.pop(0)


def _recognized(kind: str = "m1", confidence: float = 0.95) -> Classification:
    return Classification(tile(kind), confidence, 0.2)


def _unknown(confidence: float = 0.4) -> Classification:
    return Classification(None, confidence, 0.2, "confidence_below_threshold")


def _empty(confidence: float = 0.95) -> Classification:
    return Classification(None, confidence, 0.2, empty=True)


def _article_scan_layout(*, max_tiles: int = 3) -> LayoutConfig:
    scan = MeldScanConfig(
        rectangle=(0, 0, 40, 24),
        horizontal_rect=(-12, 16, 12, 8),
        vertical_rect=(-8, 12, 8, 12),
        upper_horizontal_rect=(-12, 0, 12, 8),
        horizontal_step=12,
        vertical_step=8,
        max_tiles=max_tiles,
        perspective_destination=((0, 0), (0, 23), (39, 23), (39, 0)),
        from_offsets=(3, 1, 2, 3),
    )
    return LayoutConfig(
        40,
        24,
        {"meld_0_0": ()},
        meld_scans={"meld_0_0": scan},
    )


def test_article_style_meld_scan_chooses_vertical_and_horizontal_safely() -> None:
    # Right-to-left: upright, called sideways, upright.  Each orientation is
    # compared against the alternative crop before the cut-line advances.
    classifier = _SequenceTileClassifier(
        [
            _unknown(), _recognized(),
            _recognized(), _unknown(), _empty(),
            _unknown(), _recognized(),
        ]
    )
    result = ScreenshotParser(_article_scan_layout(), classifier).parse(
        np.zeros((24, 40, 3), dtype=np.uint8)
    )

    assert result.unknown_slots == ()
    assert result.recognized["meld_0_0"] == (tile("m1"),) * 3
    assert result.melds["meld_0_0"]["kind"] == "pon"
    assert result.melds["meld_0_0"]["called_index"] == 1
    assert result.melds["meld_0_0"]["from"] == 1
    assert result.melds["meld_0_0"]["pattern"] == "right-to-left-scan"


def test_article_style_meld_scan_detects_upper_shouminkan_tile() -> None:
    classifier = _SequenceTileClassifier(
        [
            _recognized(), _unknown(), _recognized(),
            _unknown(), _recognized(),
            _unknown(), _recognized(),
        ]
    )
    result = ScreenshotParser(_article_scan_layout(), classifier).parse(
        np.zeros((24, 40, 3), dtype=np.uint8)
    )

    assert result.unknown_slots == ()
    assert result.melds["meld_0_0"]["kind"] == "shouminkan"
    assert result.recognized["meld_0_0"] == (tile("m1"),) * 4


def test_article_style_meld_scan_stops_before_ambiguous_stride() -> None:
    classifier = _SequenceTileClassifier(
        [_recognized(confidence=0.95), _recognized(confidence=0.94)]
    )
    result = ScreenshotParser(_article_scan_layout(), classifier).parse(
        np.zeros((24, 40, 3), dtype=np.uint8)
    )

    assert result.recognized["meld_0_0"] == ()
    assert result.melds == {}
    assert result.unknown_slots == ("meld_0_0:scan_orientation_ambiguous",)


def test_article_style_meld_scan_accepts_empty_area_without_direction() -> None:
    classifier = _SequenceTileClassifier([_empty(0.96), _empty(0.95)])
    result = ScreenshotParser(_article_scan_layout(), classifier).parse(
        np.zeros((24, 40, 3), dtype=np.uint8)
    )

    assert result.unknown_slots == ()
    assert result.recognized["meld_0_0"] == ()
    assert result.melds == {}


def test_layout_json_loads_slot_variants_and_meld_geometry(tmp_path: Path) -> None:
    path = tmp_path / "layout.json"
    path.write_text(
        json.dumps(
            {
                "image_width": 40,
                "image_height": 20,
                "slots": {"river_0": [[0, 0, 8, 12]], "meld_0_0": []},
                "slot_variants": {
                    "river_0": [
                        [
                            {"rect": [0, 0, 8, 12], "orientation": "vertical"},
                            {
                                "rect": [0, 0, 12, 8],
                                "rotation": 90,
                                "orientation": "horizontal",
                            },
                        ]
                    ]
                },
                "meld_layouts": {
                    "meld_0_0": [
                        {
                            "name": "pon-left",
                            "kind": "pon",
                            "from_offset": 3,
                            "slots": [
                                {"rect": [0, 0, 8, 12], "role": "tile", "called": True},
                                {"rect": [8, 0, 8, 12], "role": "tile"},
                                {"rect": [16, 0, 8, 12], "role": "tile"},
                            ],
                        }
                    ]
                },
                "meld_scans": {
                    "meld_0_1": {
                        "rect": [0, 0, 40, 20],
                        "horizontal_rect": [-12, 12, 12, 8],
                        "vertical_rect": [-8, 8, 8, 12],
                        "upper_horizontal_rect": [-12, 0, 12, 8],
                        "horizontal_step": 12,
                        "vertical_step": 8,
                        "max_tiles": 3,
                        "from_offsets": [3, 1, None]
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    layout = LayoutConfig.load(path)

    assert layout.slot_variants["river_0"][0][1].orientation == "horizontal"
    assert layout.meld_layouts["meld_0_0"][0].from_offset == 3
    assert layout.meld_scans["meld_0_1"].horizontal_step == 12
    assert layout.meld_scans["meld_0_1"].from_offsets == (3, 1, None)


def test_scene_anchor_rejects_non_game_and_corrects_translation(tmp_path: Path) -> None:
    templates = tmp_path / "templates"
    make_templates(templates)
    anchor_root = templates / "anchors"
    anchor_root.mkdir()
    anchor = np.zeros((10, 12, 3), dtype=np.uint8)
    cv2.rectangle(anchor, (1, 1), (10, 8), (255, 255, 255), 1)
    cv2.line(anchor, (2, 7), (9, 2), (255, 255, 255), 1)
    assert cv2.imwrite(str(anchor_root / "board__test.png"), anchor)
    layout = LayoutConfig(
        80,
        60,
        {"hand": ((10, 20, 8, 12),)},
        scene_anchors={
            "board": SceneAnchorConfig(
                (40, 8, 12, 10),
                (20, 0, 55, 35),
                0.8,
                True,
                1.0,
                1.0,
                0.05,
                True,
            )
        },
    )
    reference = np.zeros((60, 80, 3), dtype=np.uint8)
    reference[8:18, 40:52] = anchor
    reference[20:32, 10:18] = cv2.imread(
        str(templates / "m1__bottom.png"), cv2.IMREAD_COLOR
    )
    translated = np.zeros_like(reference)
    translated[13:23, 47:59] = anchor
    translated[25:37, 17:25] = reference[20:32, 10:18]
    parser = ScreenshotParser(
        layout,
        TemplateTileClassifier(templates),
        scene_aligner=SceneAligner(layout, templates),
    )

    result = parser.parse(translated)

    assert result.scene_found is True
    assert result.alignment_offset == pytest.approx((7.0, 5.0), abs=0.1)
    assert result.recognized["hand"] == (tile("m1"),)
    missing = parser.parse(np.zeros_like(reference))
    assert missing.scene_found is False
    assert missing.unknown_slots[0].startswith("scene:")


def test_multiple_scene_anchors_correct_zoom_and_small_rotation(tmp_path: Path) -> None:
    templates = tmp_path / "templates"
    make_templates(templates)
    anchors = templates / "anchors"
    anchors.mkdir()
    first = np.zeros((10, 12, 3), dtype=np.uint8)
    second = np.zeros((9, 11, 3), dtype=np.uint8)
    cv2.line(first, (1, 1), (10, 8), (255, 255, 255), 2)
    cv2.circle(second, (5, 4), 3, (255, 255, 255), 1)
    assert cv2.imwrite(str(anchors / "first.png"), first)
    assert cv2.imwrite(str(anchors / "second.png"), second)
    layout = LayoutConfig(
        120,
        90,
        {"hand": ((18, 62, 8, 12),)},
        scene_anchors={
            "first": SceneAnchorConfig((12, 10, 12, 10), None, 0.7, True, 0.9, 1.15, 0.01, True),
            "second": SceneAnchorConfig((82, 36, 11, 9), None, 0.7, True, 0.9, 1.15, 0.01, True),
        },
    )
    reference = np.zeros((90, 120, 3), dtype=np.uint8)
    reference[10:20, 12:24] = first
    reference[36:45, 82:93] = second
    reference[62:74, 18:26] = cv2.imread(
        str(templates / "m1__bottom.png"), cv2.IMREAD_COLOR
    )
    transform = cv2.getRotationMatrix2D((0, 0), 1.2, 1.04)
    transform[:, 2] += (4.0, 3.0)
    moved = cv2.warpAffine(reference, transform, (120, 90))
    parser = ScreenshotParser(
        layout,
        TemplateTileClassifier(templates),
        scene_aligner=SceneAligner(layout, templates),
    )

    result = parser.parse(moved)

    assert result.scene_found is True
    assert result.alignment_scale == pytest.approx(1.04, abs=0.04)
    assert result.alignment_rotation_degrees == pytest.approx(-1.2, abs=1.0)
    assert result.recognized["hand"] == (tile("m1"),)


def test_profile_report_and_round_trip_bundle(tmp_path: Path) -> None:
    templates = tmp_path / "templates"
    make_templates(templates)
    layout = tmp_path / "layout.json"
    layout.write_text(
        json.dumps(
            {
                "image_width": 8,
                "image_height": 12,
                "slots": {"hand": [[0, 0, 8, 12]]},
            }
        ),
        encoding="utf-8",
    )
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "layout": "layout.json",
                "templates": "templates",
                "capture": {"mode": "directory_replay", "directory": "frames"},
                "bootstrap": {"dealer": 0},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "frames").mkdir()
    report = inspect_vision_profile(layout, templates)
    assert report.complete is False
    assert "m2" in report.missing_tile_labels
    issue_codes = {issue.code for issue in report.issues}
    assert "meld_geometry_incomplete" in issue_codes
    assert "meld_resync_markers_incomplete" not in issue_codes
    assert "winners_markers_incomplete" not in issue_codes
    assert "losers_markers_incomplete" not in issue_codes
    assert "results_markers_incomplete" not in issue_codes
    assert "riichi_evidence_incomplete" in issue_codes
    assert "result_score_deltas_incomplete" in issue_codes
    record_template_metadata(
        templates,
        templates / "m1__bottom.png",
        label="m1",
        orientation=0,
        target_region="hand[0]",
        source_image="authorized-local-frame.png",
        crop=(0, 0, 8, 12),
        image_size=(8, 12),
    )
    metadata_report = inspect_vision_profile(layout, templates)
    assert metadata_report.template_metadata["registered"] == 1
    stale = templates / "s1__archived.png"
    stale.write_bytes((templates / "m1__bottom.png").read_bytes())
    record_template_metadata(
        templates,
        stale,
        label="s1",
        orientation=0,
        target_region="test",
        source_image="authorized-local-frame.png",
        crop=(0, 0, 8, 12),
        image_size=(8, 12),
    )
    stale.unlink()
    _manifest, removed = prune_template_metadata(templates)
    assert removed == 1
    assert inspect_vision_profile(layout, templates).template_metadata["stale"] == 0
    bundle = export_vision_profile(config, tmp_path / "profile.riyu-vision")
    imported = import_vision_profile(bundle, tmp_path / "imported")
    assert imported.is_file()
    assert (imported.parent / "layout.json").read_bytes() == layout.read_bytes()
    assert (imported.parent / "templates" / "template_manifest.json").is_file()


def test_macos_capture_normalizes_retina_pixels_to_logical_layout(tmp_path: Path) -> None:
    from mahjong_ai.perception.capture import MacOSScreenCapture

    executable = tmp_path / "screencapture"
    executable.touch()
    source = MacOSScreenCapture(CaptureRegion(0, 0, 100, 80), executable)
    retina = np.zeros((160, 200, 3), dtype=np.uint8)
    completed = Mock(returncode=0, stderr="")
    with patch("mahjong_ai.perception.capture.subprocess.run", return_value=completed), patch(
        "mahjong_ai.perception.capture.cv2.imread", return_value=retina
    ):
        frame = source.capture()
    assert frame.shape == (80, 100, 3)


def test_macos_window_capture_crops_authorized_window(tmp_path: Path) -> None:
    from mahjong_ai.perception.capture import MacOSWindowCapture

    executable = tmp_path / "screencapture"
    executable.touch()
    window = np.zeros((120, 180, 3), dtype=np.uint8)
    window[20:100, 30:150] = 200
    completed = Mock(returncode=0, stderr="")
    source = MacOSWindowCapture(
        1234,
        CaptureRegion(30, 20, 120, 80),
        (60, 40),
        executable,
    )
    with patch("mahjong_ai.perception.capture.subprocess.run", return_value=completed), patch(
        "mahjong_ai.perception.capture.cv2.imread", return_value=window
    ):
        frame = source.capture()
    assert frame.shape == (40, 60, 3)
    assert np.all(frame == 200)


def test_front_safari_window_id_is_read_only_and_validated(tmp_path: Path) -> None:
    from mahjong_ai.perception.capture import front_safari_window_id

    executable = tmp_path / "osascript"
    executable.touch()
    completed = Mock(returncode=0, stdout="4567\n", stderr="")
    with patch("mahjong_ai.perception.capture.subprocess.run", return_value=completed) as run:
        assert front_safari_window_id(executable) == 4567
    command = run.call_args.args[0]
    assert command[1:] == [
        "-e",
        'tell application "Safari" to get id of front window',
    ]


def test_stability_filter_requires_repeated_confident_frames() -> None:
    recognized = {"hand": (tile("m1"),)}
    result = PerceptionResult(None, recognized, {"hand": (0.99,)}, (), "frame")
    filter_ = StableObservationFilter(required_frames=3, minimum_confidence=0.8)
    assert filter_.push(result) is None
    assert filter_.last_status["reason"] == "stabilizing"
    assert filter_.last_status["candidate_frames"] == 1
    assert filter_.push(result) is None
    stable = filter_.push(result)
    assert stable is not None
    assert stable.confirmation_frames == 3
    assert filter_.last_status["reason"] == "stable"
    assert filter_.push(result) is None
    assert filter_.last_status["reason"] == "stable_unchanged"

    unknown = PerceptionResult(None, recognized, {"hand": (0.99,)}, ("hand[0]",), "frame")
    assert filter_.push(unknown) is None
    assert filter_.last_status["reason"] == "unknown_slots"
    assert filter_.last_status["unknown_slots"] == ["hand[0]"]


def test_result_marker_can_stabilize_while_overlay_hides_tiles() -> None:
    result = PerceptionResult(
        None,
        {"hand": ()},
        {"hand": (0.1,)},
        ("hand[0]:confidence_below_threshold",),
        "result-frame",
        {"delta_0": 8_000, "delta_1": -8_000, "delta_2": 0, "delta_3": 0},
        {"delta_0": 0.9, "delta_1": 0.9, "delta_2": 0.9, "delta_3": 0.9},
        {"result_ron": True},
        {"result_ron": 0.99},
    )
    filter_ = StableObservationFilter(required_frames=2, minimum_confidence=0.8)
    assert filter_.push(result) is None
    stable = filter_.push(result)
    assert stable is not None
    assert stable.markers["result_ron"] is True


def test_score_delta_overlay_can_stabilize_without_result_marker() -> None:
    result = PerceptionResult(
        None,
        {"hand": ()},
        {"hand": (0.1,)},
        ("hand[0]:confidence_below_threshold",),
        "result-frame",
        {"delta_0": 8_000, "delta_1": -8_000, "delta_2": 0, "delta_3": 0},
    )
    filter_ = StableObservationFilter(required_frames=2, minimum_confidence=0.8)

    assert filter_.push(result) is None
    stable = filter_.push(result)

    assert stable is not None
    assert stable.metadata["delta_0"] == 8_000


def test_vision_delta_detects_observer_draw_then_discard() -> None:
    manager = RealtimeStateManager(observer=0)
    start(manager)
    # Use the canonical engine state to avoid duplicating honor conversion rules.
    hand = tuple(manager.state["hand"])
    detector = VisionEventDetector(observer=0)
    detector.prime(observation(hand))

    with_draw = observation(hand, drawn=(tile("p9"),))
    events = detector.detect(with_draw, manager.state)
    assert [(event.name, event.data["tile"]) for event in events] == [("draw", tile("p9"))]
    manager.apply(source_event(1, events[0].name, events[0].data))

    events = detector.detect(observation(hand), manager.state)
    assert len(events) == 1
    assert events[0].name == "discard"
    assert events[0].data == {
        "player": 0,
        "tile": tile("p9"),
        "tsumogiri": True,
        "tsumogiri_known": True,
        "riichi_declaration": False,
    }


def test_vision_delta_infers_hidden_opponent_draw_from_river_append() -> None:
    manager = RealtimeStateManager(observer=0)
    start(manager, dealer=1)
    hand = tuple(manager.state["hand"])
    detector = VisionEventDetector(observer=0)
    detector.prime(observation(hand))
    events = detector.detect(
        observation(hand, rivers={1: (tile("m9"),)}),
        manager.state,
    )
    assert [event.name for event in events] == ["draw", "discard"]
    assert events[0].data["tile"] is None
    assert events[1].data["tile"] == tile("m9")
    assert events[1].confidence == "tsumogiri_unavailable"
    assert events[1].data["tsumogiri_known"] is False
    for sequence, event in enumerate(events, start=1):
        manager.apply(source_event(sequence, event.name, event.data))
    discard = manager.state["players"][1]["discards"][-1]
    assert discard["tsumogiri"] is False
    assert discard["tsumogiri_known"] is False
    discard_token = [
        value for value in public_event_tokens(manager.state) if value["type"] == "discard"
    ][-1]
    assert discard_token["tsumogiri"] == 0.5


def test_indexed_tsumogiri_marker_makes_opponent_value_known() -> None:
    manager = RealtimeStateManager(observer=0)
    start(manager, dealer=1)
    hand = tuple(manager.state["hand"])
    detector = VisionEventDetector(observer=0)
    detector.prime(observation(hand))
    events = detector.detect(
        observation(
            hand,
            rivers={1: (tile("m9"),)},
            markers={"tsumogiri_1_0": True},
        ),
        manager.state,
    )
    assert events[1].data["tsumogiri"] is True
    assert events[1].data["tsumogiri_known"] is True
    assert events[1].confidence == "visual_tsumogiri_marker"


def test_vision_delta_infers_hidden_draw_before_opponent_ankan() -> None:
    manager = RealtimeStateManager(observer=0)
    start(manager, dealer=1)
    hand = tuple(manager.state["hand"])
    before = observation(hand)
    before.recognized["meld_1_0"] = ()
    after = observation(hand)
    after.recognized["meld_1_0"] = (tile("m1"),) * 4
    detector = VisionEventDetector(observer=0)
    detector.prime(before)
    events = detector.detect(after, manager.state)
    assert [event.name for event in events] == ["draw", "kan"]
    assert events[0].data == {"player": 1, "tile": None, "rinshan": False}
    assert events[1].data["kind"] == "ankan"


def test_called_discard_may_disappear_from_river_with_the_new_meld() -> None:
    manager = RealtimeStateManager(observer=0)
    start(manager, dealer=1)
    hand = tuple(manager.state["hand"])
    detector = VisionEventDetector(observer=0)
    initial = observation(hand)
    initial.recognized["meld_2_0"] = ()
    detector.prime(initial)

    discard_frame = observation(hand, rivers={1: (tile("m9"),)})
    discard_frame.recognized["meld_2_0"] = ()
    events = detector.detect(discard_frame, manager.state)
    assert [event.name for event in events] == ["draw", "discard"]
    for sequence, event in enumerate(events, start=1):
        manager.apply(source_event(sequence, event.name, event.data))

    call_frame = observation(hand)
    call_frame.recognized["meld_2_0"] = (tile("m9"),) * 3
    events = detector.detect(call_frame, manager.state)
    assert [event.name for event in events] == ["pon"]
    assert events[0].data["player"] == 2
    assert events[0].data["from"] == 1


def test_marker_templates_and_ocr_value_validation(tmp_path: Path) -> None:
    from mahjong_ai.perception.annotations import _remove_border_components

    binary = np.full((20, 30), 255, dtype=np.uint8)
    binary[0:8, 0:5] = 0
    binary[6:15, 12:16] = 0
    cleaned = _remove_border_components(binary)
    assert np.all(cleaned[0:8, 0:5] == 255)
    assert np.all(cleaned[6:15, 12:16] == 0)

    marker_root = tmp_path / "templates" / "markers"
    marker_root.mkdir(parents=True)
    on = np.zeros((12, 8), dtype=np.uint8)
    on[:, 4:] = 255
    off = np.zeros((12, 8), dtype=np.uint8)
    off[6:, :] = 255
    assert cv2.imwrite(str(marker_root / "riichi_0__on.png"), on)
    assert cv2.imwrite(str(marker_root / "riichi_0__off.png"), off)
    bank = MarkerTemplateBank(tmp_path / "templates", {"riichi_0"})
    assert bank.classify("riichi_0", on).value is True
    assert bank.classify("riichi_0", off).value is False

    only_on = tmp_path / "only-on" / "markers"
    only_on.mkdir(parents=True)
    assert cv2.imwrite(str(only_on / "riichi_0__on.png"), on)
    safe_bank = MarkerTemplateBank(tmp_path / "only-on", {"riichi_0"})
    assert safe_bank.classify("riichi_0", np.full_like(on, 127)).value is False

    integer = TextFieldConfig((0, 0, 10, 10), "integer", 0, 50_000)
    assert TesseractFieldRecognizer._parse_value("+12,300", integer) == 12_300
    assert TesseractFieldRecognizer._parse_value("99999", integer) is None

    with pytest.raises(ValueError, match="ocr_psm"):
        LayoutConfig(
            10,
            10,
            {},
            text_fields={
                "round": TextFieldConfig((0, 0, 10, 10), ocr_psm=8),
            },
        ).validate()

    with pytest.raises(ValueError, match="ocr_minimum_confidence"):
        LayoutConfig(
            10,
            10,
            {},
            text_fields={
                "round": TextFieldConfig(
                    (0, 0, 10, 10), ocr_minimum_confidence=1.1
                ),
            },
        ).validate()


def test_isolated_ocr_field_overrides_composite_result(monkeypatch: pytest.MonkeyPatch) -> None:
    config = TextFieldConfig(
        (0, 0, 8, 8),
        "integer",
        1,
        4,
        ocr_psm=7,
        ocr_minimum_confidence=0.0,
    )
    recognizer = TesseractFieldRecognizer(
        {"hand_number": config},
        executable="/usr/bin/true",
    )
    monkeypatch.setattr(
        recognizer,
        "_composite",
        lambda _frame: (np.zeros((8, 8), dtype=np.uint8), {"hand_number": (0, 8)}),
    )
    calls: list[tuple[int, float | None]] = []

    def recognize(
        _image: np.ndarray,
        _rows: dict[str, tuple[int, int]],
        _fields: dict[str, TextFieldConfig],
        *,
        page_segmentation_mode: int,
        minimum_confidence: float | None = None,
    ) -> tuple[dict[str, int], dict[str, float], tuple[str, ...]]:
        calls.append((page_segmentation_mode, minimum_confidence))
        if page_segmentation_mode == 7:
            return {"hand_number": 2}, {"hand_number": 0.0}, ()
        return {}, {}, ()

    monkeypatch.setattr(recognizer, "_recognize_composite", recognize)

    values, confidences, unknown = recognizer.recognize(
        np.zeros((8, 8, 3), dtype=np.uint8)
    )

    assert values == {"hand_number": 2}
    assert confidences == {"hand_number": 0.0}
    assert unknown == ()
    assert calls == [(6, None), (7, 0.0)]


def test_riichi_is_declared_on_discard_and_accepted_on_later_frame() -> None:
    manager = RealtimeStateManager(observer=0)
    start(manager)
    hand = tuple(manager.state["hand"])
    manager.apply(source_event(1, "draw", {"player": 0, "tile": "9p", "rinshan": False}))
    detector = VisionEventDetector(observer=0)
    detector.prime(observation(hand, drawn=(tile("p9"),)))

    discarded = observation(
        hand,
        rivers={0: (tile("p9"),)},
        markers={"riichi_0": True},
    )
    events = detector.detect(discarded, manager.state)
    assert [event.name for event in events] == ["riichi", "discard"]
    assert events[1].data["riichi_declaration"] is True
    for sequence, event in enumerate(events, start=2):
        manager.apply(source_event(sequence, event.name, event.data))
    assert manager.state["players"][0]["riichi"].endswith("declared")

    progressed = observation(
        hand,
        rivers={0: (tile("p9"),), 1: (tile("m9"),)},
        markers={"riichi_0": True},
    )
    events = detector.detect(progressed, manager.state)
    assert [event.name for event in events] == ["riichi", "draw", "discard"]
    assert events[0].data == {"player": 0, "step": "accepted"}


def test_horizontal_river_tile_declares_riichi_without_unique_marker() -> None:
    manager = RealtimeStateManager(observer=0)
    start(manager)
    hand = tuple(manager.state["hand"])
    manager.apply(source_event(1, "draw", {"player": 0, "tile": "9p", "rinshan": False}))
    detector = VisionEventDetector(observer=0)
    detector.prime(observation(hand, drawn=(tile("p9"),)))
    discarded = replace(
        observation(hand, rivers={0: (tile("p9"),)}),
        slot_states={
            "river_0": (
                {"index": 0, "orientation": "horizontal", "confidence": 0.99},
            )
        },
    )

    events = detector.detect(discarded, manager.state)

    assert [event.name for event in events] == ["riichi", "discard"]
    assert events[0].confidence == "horizontal_riichi_discard"
    assert events[1].data["riichi_declaration"] is True


def test_result_overlay_emits_result_and_matching_round_end() -> None:
    manager = RealtimeStateManager(observer=0)
    start(manager, dealer=1)
    manager.apply(source_event(1, "draw", {"player": 1, "tile": None, "rinshan": False}))
    manager.apply(
        source_event(
            2,
            "discard",
            {
                "player": 1,
                "tile": "9m",
                "tsumogiri": False,
                "riichi_declaration": False,
            },
        )
    )
    hand = tuple(manager.state["hand"])
    detector = VisionEventDetector(observer=0)
    detector.prime(observation(hand, rivers={1: (tile("m9"),)}))
    result = observation(
        hand,
        metadata={"delta_0": 8_000, "delta_1": -8_000, "delta_2": 0, "delta_3": 0},
        markers={
            "result_ron": True,
            "winner_0": True,
            "winner_1": False,
            "winner_2": False,
            "winner_3": False,
            "loser_1": True,
        },
    )
    events = detector.detect(result, manager.state)
    assert [event.name for event in events] == ["ron", "round_end"]
    assert events[0].data["winners"] == [0]
    for sequence, event in enumerate(events, start=3):
        manager.apply(source_event(sequence, event.name, event.data))
    assert manager.state["phase"]["phase"] == "ended"
    animated_result = observation(
        (),
        metadata={"delta_0": 8_000, "delta_1": -8_000, "delta_2": 0, "delta_3": 0},
        markers={"result_ron": True},
    )
    assert detector.detect(animated_result, manager.state) == []


def test_score_deltas_and_phase_infer_ron_winner_and_loser_without_markers() -> None:
    manager = RealtimeStateManager(observer=0)
    start(manager, dealer=1)
    manager.apply(source_event(1, "draw", {"player": 1, "tile": None, "rinshan": False}))
    manager.apply(
        source_event(
            2,
            "discard",
            {
                "player": 1,
                "tile": "9m",
                "tsumogiri": False,
                "riichi_declaration": False,
            },
        )
    )
    hand = tuple(manager.state["hand"])
    detector = VisionEventDetector(observer=0)
    detector.prime(observation(hand, rivers={1: (tile("m9"),)}))

    events = detector.detect(
        observation(
            hand,
            metadata={"delta_0": 8_000, "delta_1": -8_000, "delta_2": 0, "delta_3": 0},
        ),
        manager.state,
    )

    assert [event.name for event in events] == ["ron", "round_end"]
    assert events[0].data["winners"] == [0]
    assert events[0].data["loser"] == 1
    assert events[0].confidence == "score_delta_phase_inferred"


def test_score_deltas_and_phase_infer_tsumo_winner_without_markers() -> None:
    manager = RealtimeStateManager(observer=0)
    start(manager, dealer=1)
    manager.apply(source_event(1, "draw", {"player": 1, "tile": None, "rinshan": False}))
    hand = tuple(manager.state["hand"])
    detector = VisionEventDetector(observer=0)
    detector.prime(observation(hand))

    events = detector.detect(
        observation(
            hand,
            metadata={
                "delta_0": -2_000,
                "delta_1": 6_000,
                "delta_2": -2_000,
                "delta_3": -2_000,
            },
        ),
        manager.state,
    )

    assert [event.name for event in events] == ["tsumo", "round_end"]
    assert events[0].data["winner"] == 1


def test_zero_score_delta_overlay_infers_exhaustive_draw_from_remaining_tiles() -> None:
    manager = RealtimeStateManager(observer=0)
    start(manager)
    state = json.loads(json.dumps(manager.state))
    state["remaining_tiles"] = 0
    hand = tuple(state["hand"])
    detector = VisionEventDetector(observer=0)
    detector.prime(observation(hand))

    events = detector.detect(
        observation(
            hand,
            metadata={
                "remaining_tiles": 0,
                "delta_0": 0,
                "delta_1": 0,
                "delta_2": 0,
                "delta_3": 0,
            },
        ),
        state,
    )

    assert [event.name for event in events] == ["draw_game", "round_end"]
    assert events[0].data["reason"] == "exhaustive"


def test_abortive_draw_uses_only_exact_reason_marker() -> None:
    manager = RealtimeStateManager(observer=0)
    start(manager)
    hand = tuple(manager.state["hand"])
    detector = VisionEventDetector(observer=0)
    detector.prime(observation(hand))

    events = detector.detect(
        observation(
            hand,
            metadata={"delta_0": 0, "delta_1": 0, "delta_2": 0, "delta_3": 0},
            markers={"draw_reason_four_winds": True},
        ),
        manager.state,
    )

    assert [event.name for event in events] == ["draw_game", "round_end"]
    assert events[0].data["reason"] == "four_winds"


def test_config_rejects_remote_url_and_unauthorized_live_capture(tmp_path: Path) -> None:
    templates = tmp_path / "templates"
    make_templates(templates)
    layout = tmp_path / "layout.json"
    layout.write_text(
        json.dumps({"image_width": 8, "image_height": 12, "slots": {"hand": []}}),
        encoding="utf-8",
    )
    base = {
        "session_id": "vision-test",
        "observer": 0,
        "layout": "layout.json",
        "templates": "templates",
        "capture": {"mode": "macos_region", "left": 0, "top": 0, "width": 8, "height": 12},
        "bootstrap": {"dealer": 0, "scores": [25_000] * 4},
    }
    config = tmp_path / "config.json"
    config.write_text(json.dumps(base), encoding="utf-8")
    with pytest.raises(ValueError, match="authorization_reference"):
        VisionAdapterConfig.load(config)

    base["authorization_reference"] = "operator-permission-record"
    config.write_text(json.dumps(base), encoding="utf-8")
    live_runtime = VisionAdapterRuntime(VisionAdapterConfig.load(config))
    assert live_runtime._hello()["source_acquisition"] == "screen_capture"

    base["websocket_url"] = "ws://example.com/ws/realtime/events/source"
    config.write_text(json.dumps(base), encoding="utf-8")
    with pytest.raises(ValueError, match="loopback"):
        VisionAdapterConfig.load(config)


def test_directory_replay_bootstrap_sends_checked_round_start(tmp_path: Path) -> None:
    templates = tmp_path / "templates"
    make_templates(templates)
    replay = tmp_path / "replay"
    replay.mkdir()
    assert cv2.imwrite(str(replay / "000.png"), np.zeros((12, 8, 3), dtype=np.uint8))
    layout = tmp_path / "layout.json"
    layout.write_text(
        json.dumps({"image_width": 8, "image_height": 12, "slots": {"hand": []}}),
        encoding="utf-8",
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "session_id": "replay-test",
                "observer": 0,
                "layout": "layout.json",
                "templates": "templates",
                "capture": {"mode": "directory_replay", "directory": "replay"},
                "bootstrap": {"dealer": 0, "scores": [25_000] * 4},
            }
        ),
        encoding="utf-8",
    )
    runtime = VisionAdapterRuntime(VisionAdapterConfig.load(config_path))
    assert runtime._hello()["source_kind"] == "offline_replay"
    assert runtime._hello()["source_acquisition"] == "directory_replay"

    class Socket:
        def __init__(self) -> None:
            self.sent: list[dict] = []
            self.responses = [
                {"type": "state_update"},
                {"type": "state_check", "valid": True},
            ]

        async def send(self, value: str) -> None:
            self.sent.append(json.loads(value))

        async def recv(self) -> str:
            return json.dumps(self.responses.pop(0))

    socket = Socket()
    initial = observation(tuple(_canonical_hand()))
    asyncio.run(runtime._process_stable(socket, initial))
    assert runtime.manager.round_id == "east-1-0"
    assert socket.sent[0]["event"]["name"] == "round_start"
    assert socket.sent[1]["type"] == "state_check"

    skipped = observation(
        tuple(_canonical_hand()),
        rivers={0: (tile("m1"), tile("m2"))},
    )
    asyncio.run(runtime._process_stable(socket, skipped))
    assert runtime.desynced is True
    socket.responses.append({"type": "state_update"})
    next_round = observation(
        tuple(_canonical_hand()),
        metadata={
            "hand_number": 2,
            "honba": 0,
            "riichi_sticks": 0,
            "score_0": 24_000,
            "score_1": 26_000,
            "score_2": 25_000,
            "score_3": 25_000,
        },
        markers={"wind_east": True, "dealer_1": True},
    )
    asyncio.run(runtime._process_stable(socket, next_round))
    assert runtime.manager.round_id == "east-2-0"
    assert runtime.manager.state["round"]["hand_number"] == 2
    assert runtime.desynced is False
    assert socket.sent[2]["type"] == "snapshot"


def test_dealer_bootstrap_restores_initial_drawn_tile(tmp_path: Path) -> None:
    templates = tmp_path / "templates"
    make_templates(templates)
    replay = tmp_path / "replay"
    replay.mkdir()
    assert cv2.imwrite(str(replay / "000.png"), np.zeros((12, 8, 3), dtype=np.uint8))
    layout = tmp_path / "layout.json"
    layout.write_text(
        json.dumps({"image_width": 8, "image_height": 12, "slots": {"hand": []}}),
        encoding="utf-8",
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "session_id": "dealer-bootstrap-test",
                "observer": 0,
                "layout": "layout.json",
                "templates": "templates",
                "capture": {"mode": "directory_replay", "directory": "replay"},
                "bootstrap": {
                    "dealer": 0,
                    "scores": [25_000] * 4,
                    "remaining_tiles": 69,
                },
            }
        ),
        encoding="utf-8",
    )
    runtime = VisionAdapterRuntime(VisionAdapterConfig.load(config_path))

    class Socket:
        def __init__(self) -> None:
            self.sent: list[dict] = []
            self.responses = [
                {"type": "state_update"},
                {"type": "state_update"},
                {"type": "state_check", "valid": True},
            ]

        async def send(self, value: str) -> None:
            self.sent.append(json.loads(value))

        async def recv(self) -> str:
            return json.dumps(self.responses.pop(0))

    socket = Socket()
    initial = observation(tuple(_canonical_hand()), drawn=(tile("s1"),))
    asyncio.run(runtime._process_stable(socket, initial))

    assert [message["event"]["name"] for message in socket.sent[:2]] == [
        "round_start",
        "draw",
    ]
    assert socket.sent[0]["event"]["data"]["remaining_tiles"] == 70
    assert socket.sent[1]["event"]["data"] == {
        "player": 0,
        "tile": tile("s1"),
        "rinshan": False,
    }
    assert socket.sent[2]["type"] == "state_check"
    assert runtime.manager.state["drawn_tile"] == tile("s1")
    assert runtime.manager.state["phase"]["phase"] == "awaiting_discard"


def test_midround_self_turn_restores_visible_public_state(tmp_path: Path) -> None:
    templates = tmp_path / "templates"
    make_templates(templates)
    replay = tmp_path / "replay"
    replay.mkdir()
    assert cv2.imwrite(str(replay / "000.png"), np.zeros((12, 8, 3), dtype=np.uint8))
    layout = tmp_path / "layout.json"
    layout.write_text(
        json.dumps({"image_width": 8, "image_height": 12, "slots": {"hand": []}}),
        encoding="utf-8",
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "session_id": "midround-test",
                "observer": 0,
                "layout": "layout.json",
                "templates": "templates",
                "capture": {"mode": "directory_replay", "directory": "replay"},
                "bootstrap": {"dealer": 0, "scores": [25_000] * 4},
            }
        ),
        encoding="utf-8",
    )
    runtime = VisionAdapterRuntime(VisionAdapterConfig.load(config_path))

    class Socket:
        def __init__(self) -> None:
            self.sent: list[dict] = []

        async def send(self, value: str) -> None:
            self.sent.append(json.loads(value))

        async def recv(self) -> str:
            return json.dumps({"type": "state_update"})

    midround = restorable_observation(
        tuple(_canonical_hand()),
        drawn=tile("p8"),
        rivers={
            0: (tile("m9"),),
            1: (tile("p9"),),
            2: (tile("s9"),),
        },
    )
    socket = Socket()
    asyncio.run(runtime._process_stable(socket, midround))

    assert socket.sent[0]["type"] == "snapshot"
    assert runtime.manager.round_id == "east-1-0"
    assert runtime.manager.state["remaining_tiles"] == 66
    assert runtime.manager.state["turn"] == 4
    assert runtime.manager.state["phase"] == {"phase": "awaiting_discard", "player": 0}
    assert runtime.manager.state["players"][1]["discards"][0]["tile"] == tile("p9")
    assert runtime.manager.state["players"][1]["discards"][0]["tsumogiri_known"] is False


@pytest.mark.parametrize(
    ("name", "meld", "markers", "expected_kind", "dora"),
    [
        (
            "chi",
            (tile("m1"), tile("m2"), tile("m3")),
            {"meld_0_0_called_2": True},
            "chi",
            (tile("north"),),
        ),
        (
            "pon",
            (tile("p5"),) * 3,
            {"meld_0_0_from_2": True},
            "pon",
            (tile("north"),),
        ),
        (
            "ankan",
            (tile("s9"),) * 4,
            {"meld_0_0_ankan": True},
            "ankan",
            (tile("north"), tile("east")),
        ),
        (
            "daiminkan",
            (tile("s8"),) * 4,
            {"meld_0_0_daiminkan": True, "meld_0_0_from_1": True},
            "daiminkan",
            (tile("north"), tile("east")),
        ),
        (
            "shouminkan",
            (tile("p7"),) * 4,
            {"meld_0_0_shouminkan": True, "meld_0_0_from_3": True},
            "shouminkan",
            (tile("north"), tile("east")),
        ),
    ],
)
def test_midround_restore_supports_visible_melds(
    tmp_path: Path,
    name: str,
    meld: tuple[dict[str, object], ...],
    markers: dict[str, bool],
    expected_kind: str,
    dora: tuple[dict[str, object], ...],
) -> None:
    runtime = make_runtime(tmp_path / name)
    hand = tuple(_canonical_hand()[:10])
    observed = restorable_observation(
        hand,
        drawn=tile("p8"),
        melds={(0, 0): meld},
        extra_markers=markers,
        dora=dora,
    )

    plan = runtime._build_midround_restore_plan(observed)

    assert plan.unknown_fields == ()
    assert plan.state is not None
    restored = plan.state["players"][0]["melds"][0]
    assert restored["kind"] == expected_kind
    if expected_kind == "chi":
        assert restored["from"] == 3
        assert restored["called_tile"] == tile("m3")


def test_midround_restore_uses_structured_meld_geometry_without_markers(
    tmp_path: Path,
) -> None:
    runtime = make_runtime(tmp_path)
    meld = (tile("p5"),) * 3
    observed = restorable_observation(
        tuple(_canonical_hand()[:10]),
        drawn=tile("p8"),
        melds={(0, 0): meld},
    )
    observed = replace(
        observed,
        melds={
            "meld_0_0": {
                "kind": "pon",
                "tiles": meld,
                "called_index": 1,
                "from": 3,
                "confidence": 0.98,
                "pattern": "pon-left",
            }
        },
    )

    plan = runtime._build_midround_restore_plan(observed)

    assert plan.unknown_fields == ()
    assert plan.state is not None
    restored = plan.state["players"][0]["melds"][0]
    assert restored == {
        "kind": "pon",
        "tiles": list(meld),
        "called_tile": tile("p5"),
        "from": 3,
    }


@pytest.mark.parametrize(
    ("name", "kind", "meld", "called_index", "from_player", "dora"),
    [
        (
            "chi",
            "chi",
            (tile("m2"), tile("m3"), tile("m4")),
            1,
            3,
            (tile("north"),),
        ),
        (
            "daiminkan",
            "daiminkan",
            (tile("s8"),) * 4,
            0,
            1,
            (tile("north"), tile("east")),
        ),
        (
            "shouminkan",
            "shouminkan",
            (tile("p7"),) * 4,
            1,
            3,
            (tile("north"), tile("east")),
        ),
        (
            "ankan",
            "ankan",
            (tile("m9"),) * 4,
            None,
            None,
            (tile("north"), tile("east")),
        ),
    ],
)
def test_midround_restore_structured_geometry_covers_all_meld_kinds(
    tmp_path: Path,
    name: str,
    kind: str,
    meld: tuple[dict[str, object], ...],
    called_index: int | None,
    from_player: int | None,
    dora: tuple[dict[str, object], ...],
) -> None:
    runtime = make_runtime(tmp_path / name)
    observed = restorable_observation(
        tuple(_canonical_hand()[:10]),
        drawn=tile("p8"),
        melds={(0, 0): meld},
        dora=dora,
    )
    observed = replace(
        observed,
        melds={
            "meld_0_0": {
                "kind": kind,
                "tiles": meld,
                "called_index": called_index,
                "from": from_player,
                "confidence": 0.98,
                "pattern": name,
            }
        },
    )

    plan = runtime._build_midround_restore_plan(observed)

    assert plan.unknown_fields == ()
    assert plan.state is not None
    assert plan.state["players"][0]["melds"][0]["kind"] == kind


def test_midround_restore_preserves_confirmed_riichi_state(tmp_path: Path) -> None:
    runtime = make_runtime(tmp_path)
    observed = restorable_observation(
        tuple(_canonical_hand()),
        drawn=tile("p8"),
        extra_markers={"riichi_1": True, "double_riichi_1": False},
        metadata_overrides={"riichi_sticks": 1, "score_1": 24_000},
    )

    plan = runtime._build_midround_restore_plan(observed)

    assert plan.state is not None
    assert plan.state["players"][1]["riichi"] == "accepted"


def test_midround_restore_derives_riichi_from_horizontal_river_without_markers(
    tmp_path: Path,
) -> None:
    runtime = make_runtime(tmp_path)
    observed = restorable_observation(
        tuple(_canonical_hand()),
        drawn=tile("p8"),
        rivers={1: (tile("m1"), tile("m2"))},
        metadata_overrides={"riichi_sticks": 1, "score_1": 24_000},
    )
    for player in range(4):
        observed.markers.pop(f"riichi_{player}", None)
        observed.markers.pop(f"double_riichi_{player}", None)
    observed = replace(
        observed,
        slot_states={
            "river_1": (
                {"index": 0, "orientation": "vertical", "confidence": 0.99},
                {"index": 1, "orientation": "horizontal", "confidence": 0.99},
            )
        },
    )

    plan = runtime._build_midround_restore_plan(observed)

    assert plan.unknown_fields == ()
    assert plan.state is not None
    assert plan.state["players"][1]["riichi"] == "accepted"


def test_midround_ocr_failure_stays_partial_without_losing_tiles(tmp_path: Path) -> None:
    runtime = make_runtime(tmp_path)
    observed = restorable_observation(tuple(_canonical_hand()), drawn=tile("p8"))
    del observed.metadata["score_2"]

    plan = runtime._build_midround_restore_plan(observed)

    assert plan.state is None
    assert "metadata.score_2:unknown" in plan.unknown_fields
    assert observed.recognized["hand"] == tuple(_canonical_hand())


def test_desynchronized_runtime_recovers_at_safe_midround_snapshot(tmp_path: Path) -> None:
    runtime = make_runtime(tmp_path)
    start(runtime.manager)
    runtime.current_round_id = "east-1-0"
    runtime.desynced = True
    runtime.sync_status = "DESYNCHRONIZED"
    observed = restorable_observation(tuple(_canonical_hand()), drawn=tile("p8"))

    class Socket:
        def __init__(self) -> None:
            self.sent: list[dict] = []

        async def send(self, value: str) -> None:
            self.sent.append(json.loads(value))

        async def recv(self) -> str:
            return json.dumps({"type": "state_update"})

    socket = Socket()
    asyncio.run(runtime._process_stable(socket, observed))

    assert socket.sent[0]["type"] == "snapshot"
    assert runtime.desynced is False
    assert runtime.sync_status == "SYNCED"


def test_ambiguous_meld_provenance_is_rejected(tmp_path: Path) -> None:
    runtime = make_runtime(tmp_path)
    observed = restorable_observation(
        tuple(_canonical_hand()[:10]),
        drawn=tile("p8"),
        melds={(0, 0): (tile("p5"),) * 3},
    )

    plan = runtime._build_midround_restore_plan(observed)

    assert plan.state is None
    assert "recognized.meld_0_0:source_ambiguous" in plan.unknown_fields


def test_saved_frame_benchmark_writes_json_metrics(tmp_path: Path) -> None:
    templates = tmp_path / "templates"
    make_templates(templates)
    frames = tmp_path / "frames"
    frames.mkdir()
    m1 = cv2.imread(str(templates / "m1__bottom.png"), cv2.IMREAD_COLOR)
    assert m1 is not None and cv2.imwrite(str(frames / "000.png"), m1)
    layout = tmp_path / "layout.json"
    layout.write_text(
        json.dumps({"image_width": 8, "image_height": 12, "slots": {"hand": [[0, 0, 8, 12]]}}),
        encoding="utf-8",
    )
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "layout": "layout.json",
                "templates": "templates",
                "capture": {"mode": "directory_replay", "directory": "frames"},
                "bootstrap": {"dealer": 0},
            }
        ),
        encoding="utf-8",
    )
    annotations = tmp_path / "annotations.json"
    annotations.write_text(
        json.dumps(
            {
                "format": "riyu-vision-benchmark/v1",
                "frames": [
                    {
                        "file": "000.png",
                        "round_id": "east-1-0",
                        "game_id": "game-1",
                        "tiles": {"hand": ["m1"]},
                        "events": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    report = tmp_path / "report.json"
    repository = Path(__file__).resolve().parents[2]

    completed = subprocess.run(
        [
            sys.executable,
            str(repository / "tools" / "benchmark_vision_profile.py"),
            str(config),
            str(frames),
            "--annotations",
            str(annotations),
            "--output",
            str(report),
        ],
        cwd=repository,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    metrics = json.loads(report.read_text(encoding="utf-8"))
    assert metrics["tile_slot_count"] == 1
    assert metrics["tile_accuracy"] == 1.0
    assert metrics["round_full_sync_rate"] == 1.0
    assert metrics["game_full_sync_rate"] == 1.0


def _canonical_hand() -> list[dict[str, object]]:
    aliases = {
        "1z": "east",
        "5z": "white",
    }
    return [
        tile(aliases.get(value, value[1] + value[0]))
        for value in HAND
    ]
