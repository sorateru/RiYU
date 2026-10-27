from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np
import pytest

from mahjong_ai.ai.sequence import public_event_tokens
from mahjong_ai.analysis import RealtimeStateManager
from mahjong_ai.perception import (
    CaptureRegion,
    LayoutConfig,
    MarkerFieldConfig,
    MarkerTemplateBank,
    PerceptionResult,
    ScreenshotParser,
    StableObservation,
    StableObservationFilter,
    TemplateTileClassifier,
    TesseractFieldRecognizer,
    TextFieldConfig,
    VisionAdapterConfig,
    VisionAdapterRuntime,
    VisionEventDetector,
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


def make_templates(directory: Path) -> None:
    directory.mkdir()
    empty = np.zeros((12, 8), dtype=np.uint8)
    empty[:, 4:] = 255
    one = np.zeros((12, 8), dtype=np.uint8)
    one[6:, :] = 255
    assert cv2.imwrite(str(directory / "empty__bottom.png"), empty)
    assert cv2.imwrite(str(directory / "m1__bottom.png"), one)


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

    midround = observation(
        tuple(_canonical_hand()),
        drawn=(tile("p8"),),
        rivers={
            0: (tile("m9"),),
            1: (tile("p9"),),
            2: (tile("s9"),),
        },
        metadata={"remaining_tiles": 66},
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


def _canonical_hand() -> list[dict[str, object]]:
    aliases = {
        "1z": "east",
        "5z": "white",
    }
    return [
        tile(aliases.get(value, value[1] + value[0]))
        for value in HAND
    ]
