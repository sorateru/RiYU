from __future__ import annotations

import asyncio
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from mahjong_ai.analysis import RealtimeStateManager
from mahjong_ai.perception import (
    CaptureRegion,
    LayoutConfig,
    PerceptionResult,
    ScreenshotParser,
    StableObservation,
    StableObservationFilter,
    TemplateTileClassifier,
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
) -> StableObservation:
    recognized = {
        "hand": hand,
        "drawn_tile": drawn,
        "dora_indicators": (tile("north"),),
        **{f"river_{player}": () for player in range(4)},
    }
    for player, values in (rivers or {}).items():
        recognized[f"river_{player}"] = values
    return StableObservation(recognized, {}, repr(recognized), 3, "test")


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


def test_stability_filter_requires_repeated_confident_frames() -> None:
    recognized = {"hand": (tile("m1"),)}
    result = PerceptionResult(None, recognized, {"hand": (0.99,)}, (), "frame")
    filter_ = StableObservationFilter(required_frames=3, minimum_confidence=0.8)
    assert filter_.push(result) is None
    assert filter_.push(result) is None
    stable = filter_.push(result)
    assert stable is not None
    assert stable.confirmation_frames == 3
    assert filter_.push(result) is None

    unknown = PerceptionResult(None, recognized, {"hand": (0.99,)}, ("hand[0]",), "frame")
    assert filter_.push(unknown) is None


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


def _canonical_hand() -> list[dict[str, object]]:
    aliases = {
        "1z": "east",
        "5z": "white",
    }
    return [
        tile(aliases.get(value, value[1] + value[0]))
        for value in HAND
    ]
