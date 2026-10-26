"""Vision frames to normalized events to the local realtime service."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from mahjong_ai.adapters import MAJSOUL_EVENT_PROFILE
from mahjong_ai.analysis import (
    EXTERNAL_AUTHORIZATION,
    MAJSOUL_SOURCE_SERVICE,
    REALTIME_PROTOCOL,
    RealtimeStateManager,
)

from .capture import CaptureRegion, DirectoryFrameSource, FrameSource, MacOSScreenCapture
from .parser import LayoutConfig, ScreenshotParser
from .stability import StableObservation, StableObservationFilter, observed_tiles
from .templates import TemplateTileClassifier
from .vision_events import DetectedEvent, VisionDeltaError, VisionEventDetector

logger = logging.getLogger(__name__)


class VisionRuntimeError(RuntimeError):
    pass


@dataclass(frozen=True)
class VisionBootstrap:
    round_id: str
    wind: str
    hand_number: int
    honba: int
    riichi_sticks: int
    dealer: int
    scores: tuple[int, int, int, int]
    remaining_tiles: int = 70


@dataclass(frozen=True)
class VisionAdapterConfig:
    session_id: str
    observer: int
    websocket_url: str
    layout_path: Path
    template_directory: Path
    frames_per_second: float
    confirmation_frames: int
    minimum_confidence: float
    capture_region: CaptureRegion | None
    replay_directory: Path | None
    authorization_reference: str | None
    bootstrap: VisionBootstrap

    @classmethod
    def load(cls, path: str | Path) -> "VisionAdapterConfig":
        config_path = Path(path).resolve()
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        base = config_path.parent
        capture = raw.get("capture") or {}
        capture_region = None
        replay_directory = None
        mode = capture.get("mode", "macos_region")
        if mode == "macos_region":
            capture_region = CaptureRegion(
                int(capture["left"]),
                int(capture["top"]),
                int(capture["width"]),
                int(capture["height"]),
            )
        elif mode == "directory_replay":
            replay_directory = _resolve(base, capture["directory"])
        else:
            raise ValueError("capture.mode must be macos_region or directory_replay")
        bootstrap = raw["bootstrap"]
        scores = tuple(int(value) for value in bootstrap.get("scores", [25_000] * 4))
        if len(scores) != 4:
            raise ValueError("bootstrap.scores must contain four values")
        result = cls(
            session_id=str(raw.get("session_id", "majsoul-vision")),
            observer=int(raw.get("observer", 0)),
            websocket_url=str(
                raw.get(
                    "websocket_url",
                    "ws://127.0.0.1:8787/ws/realtime/events/source",
                )
            ),
            layout_path=_resolve(base, raw["layout"]),
            template_directory=_resolve(base, raw["templates"]),
            frames_per_second=float(raw.get("frames_per_second", 3.0)),
            confirmation_frames=int(raw.get("confirmation_frames", 3)),
            minimum_confidence=float(raw.get("minimum_confidence", 0.78)),
            capture_region=capture_region,
            replay_directory=replay_directory,
            authorization_reference=raw.get("authorization_reference"),
            bootstrap=VisionBootstrap(
                round_id=str(bootstrap.get("round_id", "east-1-0")),
                wind=str(bootstrap.get("wind", "east")),
                hand_number=int(bootstrap.get("hand_number", 1)),
                honba=int(bootstrap.get("honba", 0)),
                riichi_sticks=int(bootstrap.get("riichi_sticks", 0)),
                dealer=int(bootstrap["dealer"]),
                scores=scores,  # type: ignore[arg-type]
                remaining_tiles=int(bootstrap.get("remaining_tiles", 70)),
            ),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if not 0 <= self.observer <= 3 or not 0 <= self.bootstrap.dealer <= 3:
            raise ValueError("observer and bootstrap.dealer must be in 0..=3")
        if self.bootstrap.wind not in {"east", "south", "west", "north"}:
            raise ValueError("bootstrap.wind must be east, south, west, or north")
        if not 1 <= self.bootstrap.hand_number <= 4:
            raise ValueError("bootstrap.hand_number must be in 1..=4")
        if self.bootstrap.honba < 0 or self.bootstrap.riichi_sticks < 0:
            raise ValueError("bootstrap honba and riichi_sticks must be non-negative")
        if not 0 <= self.bootstrap.remaining_tiles <= 70:
            raise ValueError("bootstrap.remaining_tiles must be in 0..=70")
        if not self.session_id or len(self.session_id) > 128:
            raise ValueError("session_id must contain 1 to 128 characters")
        parsed_url = urlparse(self.websocket_url)
        if parsed_url.scheme not in {"ws", "wss"} or parsed_url.hostname not in {
            "127.0.0.1",
            "localhost",
            "::1",
        }:
            raise ValueError("websocket_url must be a loopback ws:// or wss:// URL")
        if not 0.2 <= self.frames_per_second <= 10.0:
            raise ValueError("frames_per_second must be between 0.2 and 10")
        if not self.layout_path.is_file():
            raise ValueError(f"layout does not exist: {self.layout_path}")
        if not self.template_directory.is_dir():
            raise ValueError(
                f"template directory does not exist: {self.template_directory}"
            )
        if self.replay_directory is not None and not self.replay_directory.is_dir():
            raise ValueError(
                f"replay directory does not exist: {self.replay_directory}"
            )
        StableObservationFilter(
            self.confirmation_frames,
            self.minimum_confidence,
        )
        if self.capture_region is not None and (
            not isinstance(self.authorization_reference, str)
            or not self.authorization_reference.strip()
            or len(self.authorization_reference) > 256
        ):
            raise ValueError(
                "live screen capture requires authorization_reference; use directory_replay for offline tests"
            )
        if self.capture_region is not None:
            layout = LayoutConfig.load(self.layout_path)
            if (
                layout.image_width != self.capture_region.width
                or layout.image_height != self.capture_region.height
            ):
                raise ValueError("layout dimensions must match capture width and height")

    def frame_source(self) -> FrameSource:
        if self.capture_region is not None:
            return MacOSScreenCapture(self.capture_region)
        assert self.replay_directory is not None
        return DirectoryFrameSource(self.replay_directory)


class VisionAdapterRuntime:
    def __init__(self, config: VisionAdapterConfig) -> None:
        self.config = config
        self.parser = ScreenshotParser(
            LayoutConfig.load(config.layout_path),
            TemplateTileClassifier(
                config.template_directory,
                minimum_confidence=config.minimum_confidence,
            ),
        )
        self.filter = StableObservationFilter(
            config.confirmation_frames, config.minimum_confidence
        )
        self.detector = VisionEventDetector(config.observer)
        self.manager = RealtimeStateManager(config.observer)
        self.source = config.frame_source()

    async def run(self, max_frames: int | None = None) -> None:
        if max_frames is not None and max_frames <= 0:
            raise ValueError("max_frames must be positive")
        try:
            import websockets
        except ImportError as error:
            raise VisionRuntimeError(
                "Vision Adapter requires the service dependency 'websockets'"
            ) from error
        async with websockets.connect(self.config.websocket_url, max_size=1_000_000) as socket:
            await socket.send(json.dumps(self._hello()))
            ready = json.loads(await socket.recv())
            if ready.get("type") != "ready":
                raise VisionRuntimeError(f"Vision Adapter handshake failed: {ready}")
            logger.info("vision source connected session=%s", self.config.session_id)
            interval = 1.0 / self.config.frames_per_second
            frames = 0
            while max_frames is None or frames < max_frames:
                started = asyncio.get_running_loop().time()
                frame = await asyncio.to_thread(self.source.capture)
                perception = await asyncio.to_thread(self.parser.parse, frame)
                frames += 1
                stable = self.filter.push(perception)
                if stable is not None:
                    await self._process_stable(socket, stable)
                remaining = interval - (asyncio.get_running_loop().time() - started)
                if remaining > 0:
                    await asyncio.sleep(remaining)

    async def _process_stable(self, socket: Any, observation: StableObservation) -> None:
        if self.manager.round_id is None:
            events = [self._round_start(observation)]
            self.detector.prime(observation)
        else:
            try:
                events = self.detector.detect(observation, self.manager.state)
            except VisionDeltaError as error:
                logger.error("vision delta rejected: %s", error)
                raise VisionRuntimeError(str(error)) from error
        for detected in events:
            envelope = self._envelope(detected)
            await socket.send(json.dumps({"type": "event", "event": envelope}))
            response = json.loads(await socket.recv())
            if response.get("type") == "error":
                raise VisionRuntimeError(f"realtime service rejected vision event: {response}")
            self.manager.apply(envelope)
            logger.info(
                "vision event applied sequence=%s name=%s confidence=%s",
                envelope["sequence"],
                envelope["name"],
                detected.confidence,
            )
        if self.manager.round_id is not None:
            assertion = {
                "round_id": self.manager.round_id,
                "hand_count": len(observed_tiles(observation, "hand"))
                + len(observed_tiles(observation, "drawn_tile")),
                "dora_count": len(observed_tiles(observation, "dora_indicators")),
            }
            await socket.send(json.dumps({"type": "state_check", "assertion": assertion}))
            check = json.loads(await socket.recv())
            if check.get("type") == "error" or check.get("valid") is not True:
                raise VisionRuntimeError(f"vision/state inconsistency: {check}")

    def _round_start(self, observation: StableObservation) -> DetectedEvent:
        hand = list(observed_tiles(observation, "hand"))
        drawn = observed_tiles(observation, "drawn_tile")
        dora = observed_tiles(observation, "dora_indicators")
        if len(hand) != 13 or drawn or len(dora) != 1:
            raise VisionRuntimeError(
                "fresh round bootstrap requires exactly 13 hand tiles, no drawn tile, and one dora indicator"
            )
        bootstrap = self.config.bootstrap
        return DetectedEvent(
            "round_start",
            {
                "wind": bootstrap.wind,
                "hand_number": bootstrap.hand_number,
                "honba": bootstrap.honba,
                "riichi_sticks": bootstrap.riichi_sticks,
                "dealer": bootstrap.dealer,
                "scores": list(bootstrap.scores),
                "initial_hand": hand,
                "dora_indicator": dora[0],
                "remaining_tiles": bootstrap.remaining_tiles,
            },
            "stable_visual_bootstrap",
            "new round initialized from confirmed hand and dora slots",
        )

    def _envelope(self, detected: DetectedEvent) -> dict[str, Any]:
        sequence = self.manager.next_source_sequence
        return {
            "event_id": f"vision-{sequence}-{detected.name}",
            "sequence": sequence,
            "round_id": self.config.bootstrap.round_id,
            "name": detected.name,
            "data": detected.data,
        }

    def _hello(self) -> dict[str, Any]:
        live = self.config.capture_region is not None
        hello = {
            "type": "hello",
            "protocol": REALTIME_PROTOCOL,
            "session_id": self.config.session_id,
            "observer": self.config.observer,
            "source_kind": "authorized_external_adapter" if live else "offline_replay",
            "source_service": MAJSOUL_SOURCE_SERVICE,
            "source_profile": MAJSOUL_EVENT_PROFILE,
        }
        if live:
            hello["authorization"] = EXTERNAL_AUTHORIZATION
            hello["authorization_reference"] = self.config.authorization_reference
        return hello


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()
