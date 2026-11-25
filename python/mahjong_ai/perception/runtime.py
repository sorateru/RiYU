"""Vision frames to normalized events to the local realtime service."""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import cv2

from mahjong_ai import validate_state_json
from mahjong_ai.adapters import MAJSOUL_EVENT_PROFILE
from mahjong_ai.analysis import (
    EXTERNAL_AUTHORIZATION,
    MAJSOUL_SOURCE_SERVICE,
    REALTIME_PROTOCOL,
    RealtimeStateManager,
)
from mahjong_ai.schema import TILE_TO_INDEX, canonical_json

from .annotations import ScreenAnnotationParser
from .alignment import SceneAligner
from .capture import (
    CaptureRegion,
    DirectoryFrameSource,
    FrameSource,
    MacOSScreenCapture,
    MacOSWindowCapture,
    front_safari_window_id,
)
from .parser import LayoutConfig, ScreenshotParser
from .profile import inspect_vision_profile
from .stability import (
    StableObservation,
    StableObservationFilter,
    observed_marker,
    observed_meld,
    observed_metadata,
    observed_slot_state,
    observed_tiles,
)
from .templates import TemplateTileClassifier, VisualTemplateClassifier
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
class MidroundRestorePlan:
    state: dict[str, Any] | None
    bootstrap: VisionBootstrap | None
    unknown_fields: tuple[str, ...]


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
    capture_window_id: int | None
    capture_window_owner: str | None
    capture_window_crop: CaptureRegion | None
    replay_directory: Path | None
    authorization_reference: str | None
    bootstrap: VisionBootstrap
    tesseract_executable: str | None = None
    marker_minimum_confidence: float = 0.72
    marker_minimum_margin: float = 0.05
    ocr_minimum_confidence: float = 0.55
    tile_minimum_margin: float = 0.03
    latest_frame_path: Path | None = None
    latest_frame_interval_seconds: float = 2.0

    @classmethod
    def load(cls, path: str | Path) -> "VisionAdapterConfig":
        config_path = Path(path).resolve()
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        base = config_path.parent
        capture = raw.get("capture") or {}
        capture_region = None
        capture_window_id = None
        capture_window_owner = None
        capture_window_crop = None
        replay_directory = None
        mode = capture.get("mode", "macos_region")
        if mode == "macos_region":
            capture_region = CaptureRegion(
                int(capture["left"]),
                int(capture["top"]),
                int(capture["width"]),
                int(capture["height"]),
            )
        elif mode == "macos_window":
            capture_region = CaptureRegion(
                0,
                0,
                int(capture["width"]),
                int(capture["height"]),
            )
            capture_window_id = (
                None if capture.get("window_id") is None else int(capture["window_id"])
            )
            capture_window_owner = capture.get("window_owner")
            content = capture["content_region"]
            capture_window_crop = CaptureRegion(
                int(content["left"]),
                int(content["top"]),
                int(content["width"]),
                int(content["height"]),
            )
        elif mode == "directory_replay":
            replay_directory = _resolve(base, capture["directory"])
        else:
            raise ValueError(
                "capture.mode must be macos_region, macos_window, or directory_replay"
            )
        bootstrap = raw["bootstrap"]
        annotation = raw.get("annotation") or {}
        diagnostics = raw.get("diagnostics") or {}
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
            capture_window_id=capture_window_id,
            capture_window_owner=capture_window_owner,
            capture_window_crop=capture_window_crop,
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
            tesseract_executable=annotation.get("tesseract_executable"),
            marker_minimum_confidence=float(
                annotation.get("marker_minimum_confidence", 0.72)
            ),
            marker_minimum_margin=float(annotation.get("marker_minimum_margin", 0.05)),
            ocr_minimum_confidence=float(
                annotation.get("ocr_minimum_confidence", 0.55)
            ),
            tile_minimum_margin=float(raw.get("tile_minimum_margin", 0.03)),
            latest_frame_path=(
                None
                if not diagnostics.get("latest_frame")
                else _resolve(base, diagnostics["latest_frame"])
            ),
            latest_frame_interval_seconds=float(
                diagnostics.get("latest_frame_interval_seconds", 2.0)
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
        if not 0 <= self.marker_minimum_confidence <= 1:
            raise ValueError("marker_minimum_confidence must be in 0..=1")
        if not 0 <= self.marker_minimum_margin <= 1:
            raise ValueError("marker_minimum_margin must be in 0..=1")
        if not 0 <= self.ocr_minimum_confidence <= 1:
            raise ValueError("ocr_minimum_confidence must be in 0..=1")
        if not 0 <= self.tile_minimum_margin <= 1:
            raise ValueError("tile_minimum_margin must be in 0..=1")
        if not 0.5 <= self.latest_frame_interval_seconds <= 60:
            raise ValueError("latest_frame_interval_seconds must be in 0.5..=60")
        if self.capture_region is not None and (
            not isinstance(self.authorization_reference, str)
            or not self.authorization_reference.strip()
            or len(self.authorization_reference) > 256
        ):
            raise ValueError(
                "live screen capture requires authorization_reference; use directory_replay for offline tests"
            )
        window_selected = (
            self.capture_window_id is not None
            or self.capture_window_owner is not None
        )
        if window_selected != (self.capture_window_crop is not None):
            raise ValueError(
                "macos_window requires window_id or window_owner plus content_region"
            )
        if (
            self.capture_window_id is not None
            and self.capture_window_owner is not None
        ):
            raise ValueError("macos_window accepts either window_id or window_owner, not both")
        if self.capture_window_owner not in {None, "Safari"}:
            raise ValueError("capture.window_owner currently supports only Safari")
        if self.capture_window_id is not None and self.capture_window_id <= 0:
            raise ValueError("capture.window_id must be positive")
        if self.capture_region is not None:
            layout = LayoutConfig.load(self.layout_path)
            if (
                layout.image_width != self.capture_region.width
                or layout.image_height != self.capture_region.height
            ):
                raise ValueError("layout dimensions must match capture width and height")

    def frame_source(self) -> FrameSource:
        if self.capture_window_id is not None or self.capture_window_owner is not None:
            assert self.capture_region is not None
            assert self.capture_window_crop is not None
            window_id = (
                self.capture_window_id
                if self.capture_window_id is not None
                else front_safari_window_id()
            )
            return MacOSWindowCapture(
                window_id,
                self.capture_window_crop,
                (self.capture_region.width, self.capture_region.height),
            )
        if self.capture_region is not None:
            return MacOSScreenCapture(self.capture_region)
        assert self.replay_directory is not None
        return DirectoryFrameSource(self.replay_directory)


class VisionAdapterRuntime:
    def __init__(self, config: VisionAdapterConfig) -> None:
        self.config = config
        layout = LayoutConfig.load(config.layout_path)
        annotation_parser = (
            ScreenAnnotationParser(
                layout,
                config.template_directory,
                tesseract_executable=config.tesseract_executable,
                minimum_confidence=config.marker_minimum_confidence,
                minimum_margin=config.marker_minimum_margin,
                ocr_minimum_confidence=config.ocr_minimum_confidence,
            )
            if layout.text_fields or layout.markers
            else None
        )
        self.parser = ScreenshotParser(
            layout,
            TemplateTileClassifier(
                config.template_directory,
                minimum_confidence=config.minimum_confidence,
                minimum_margin=config.tile_minimum_margin,
            ),
            annotation_parser,
            (
                SceneAligner(layout, config.template_directory)
                if layout.scene_anchors
                else None
            ),
            VisualTemplateClassifier(
                config.template_directory,
                minimum_confidence=config.minimum_confidence,
                minimum_margin=config.tile_minimum_margin,
            ),
        )
        self.filter = StableObservationFilter(
            config.confirmation_frames, config.minimum_confidence
        )
        self.detector = VisionEventDetector(config.observer)
        self.manager = RealtimeStateManager(config.observer)
        self.source = config.frame_source()
        self.current_round_id = config.bootstrap.round_id
        self.desynced = False
        self.sync_status = "PARTIAL"
        self.partial_unknown_fields: tuple[str, ...] = ("stable_observation",)
        self._partial_observation: StableObservation | None = None
        self._last_status_payload: str | None = None
        self._last_status_sent_at = 0.0
        self._runtime_block_reason: str | None = None
        self._last_frame_saved_at = 0.0
        self.profile_report = inspect_vision_profile(
            config.layout_path, config.template_directory
        )

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
                await asyncio.to_thread(self._maybe_save_latest_frame, frame)
                perception = await asyncio.to_thread(self.parser.parse, frame)
                frames += 1
                stable = self.filter.push(perception)
                await self._send_perception_status(socket)
                if stable is not None:
                    await self._process_stable(socket, stable)
                remaining = interval - (asyncio.get_running_loop().time() - started)
                if remaining > 0:
                    await asyncio.sleep(remaining)

    def _maybe_save_latest_frame(self, frame: Any) -> None:
        destination = self.config.latest_frame_path
        if destination is None:
            return
        now = time.monotonic()
        if now - self._last_frame_saved_at < self.config.latest_frame_interval_seconds:
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.stem}.tmp{destination.suffix}")
        if not cv2.imwrite(str(temporary), frame):
            raise VisionRuntimeError(f"could not save diagnostic frame: {destination}")
        temporary.replace(destination)
        self._last_frame_saved_at = now

    async def _send_perception_status(self, socket: Any) -> None:
        status = {
            **self.filter.last_status,
            "desynchronized": self.desynced,
            "sync_status": self.sync_status,
            "partial_unknown_fields": list(self.partial_unknown_fields[:64]),
            "round_initialized": self.manager.round_id is not None,
            "profile_complete": self.profile_report.complete,
            "profile_capabilities": self.profile_report.capabilities,
            "profile_issues": [
                {"code": issue.code, "detail": issue.detail}
                for issue in self.profile_report.issues
            ],
        }
        if (
            self._runtime_block_reason is not None
            and status["reason"] in {"stable", "stable_unchanged"}
        ):
            status["reason"] = self._runtime_block_reason
        encoded = json.dumps(status, sort_keys=True, separators=(",", ":"))
        now = time.monotonic()
        if encoded == self._last_status_payload and now - self._last_status_sent_at < 1.0:
            return
        await socket.send(json.dumps({"type": "perception_status", "status": status}))
        response = json.loads(await socket.recv())
        if response.get("type") == "error":
            raise VisionRuntimeError(
                f"realtime service rejected perception status: {response}"
            )
        if response.get("type") != "perception_status_ack":
            raise VisionRuntimeError(
                f"unexpected perception status response: {response}"
            )
        self._last_status_payload = encoded
        self._last_status_sent_at = now

    async def _process_stable(self, socket: Any, observation: StableObservation) -> None:
        if self.manager.round_id is None:
            if self._fresh_round(observation):
                bootstrap = self._round_metadata(observation, initial=True)
                self.current_round_id = bootstrap.round_id
                self._runtime_block_reason = None
                events = self._bootstrap_events(observation, bootstrap)
                self.detector.prime(observation)
            else:
                plan = self._build_midround_restore_plan(observation)
                if plan.state is not None and plan.bootstrap is not None:
                    await self._restore_midround(socket, observation, plan)
                    return
                self.sync_status = "PARTIAL"
                self.partial_unknown_fields = plan.unknown_fields
                self._partial_observation = observation
                self._runtime_block_reason = "midround_partial"
                await self._send_perception_status(socket)
                logger.info(
                    "mid-round vision is partial; missing=%s",
                    ",".join(plan.unknown_fields),
                )
                return
        elif self._fresh_round(observation) and self._new_round_visible(observation):
            bootstrap = self._round_metadata(observation, initial=False)
            if self.desynced or (self.manager.state.get("phase") or {}).get("phase") != "ended":
                await self._restore_fresh_round(socket, observation, bootstrap)
                return
            self.current_round_id = bootstrap.round_id
            events = self._bootstrap_events(observation, bootstrap)
            self.detector.prime(observation)
        else:
            if self.desynced:
                plan = self._build_midround_restore_plan(observation)
                if plan.state is not None and plan.bootstrap is not None:
                    await self._restore_midround(socket, observation, plan)
                    return
                self.sync_status = "DESYNCHRONIZED"
                self.partial_unknown_fields = plan.unknown_fields
                self._runtime_block_reason = "desynchronized_waiting_for_safe_snapshot"
                logger.warning("vision remains desynchronized; waiting for a safe snapshot")
                return
            try:
                events = self.detector.detect(observation, self.manager.state)
            except VisionDeltaError as error:
                self.desynced = True
                self.sync_status = "DESYNCHRONIZED"
                self.partial_unknown_fields = (f"event_delta:{error}",)
                self._runtime_block_reason = "desynchronized_waiting_for_safe_snapshot"
                logger.error(
                    "vision delta rejected; waiting for a safe resync snapshot: %s", error
                )
                return
        for detected in events:
            await self._send_event(socket, detected)
        if events or self.manager.round_id is not None:
            self.sync_status = "SYNCED"
            self.partial_unknown_fields = ()
            self._partial_observation = None
        if self.manager.round_id is not None and not self._terminal_visible(observation):
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

    def _can_restore_midround(self, observation: StableObservation) -> bool:
        return self._build_midround_restore_plan(observation).state is not None

    def _build_midround_restore_plan(
        self, observation: StableObservation
    ) -> MidroundRestorePlan:
        """Build only a fully public, uniquely determined self-turn snapshot.

        Opponent hands and wall order never enter ``GameState``. Ambiguous meld
        provenance, riichi kind, OCR, or turn ownership remains explicit in the
        returned unknown list instead of being guessed.
        """

        unknown: list[str] = []
        hand = list(observed_tiles(observation, "hand"))
        drawn = list(observed_tiles(observation, "drawn_tile"))
        dora = list(observed_tiles(observation, "dora_indicators"))
        if len(drawn) != 1:
            unknown.append("current_turn:not_confirmed_self_discard")

        melds, meld_unknown = self._restored_melds(observation)
        unknown.extend(meld_unknown)
        observer_meld_count = len(melds[self.config.observer])
        expected_hand = 13 - 3 * observer_meld_count
        if len(hand) != expected_hand:
            unknown.append(
                f"hand:expected_{expected_hand}_with_{observer_meld_count}_melds_got_{len(hand)}"
            )

        kan_count = sum(
            meld["kind"] in {"daiminkan", "ankan", "shouminkan"}
            for player_melds in melds
            for meld in player_melds
        )
        expected_dora = 1 + kan_count
        if len(dora) != expected_dora:
            unknown.append(f"dora_indicators:expected_{expected_dora}_got_{len(dora)}")

        required_integers = (
            "hand_number",
            "honba",
            "riichi_sticks",
            "remaining_tiles",
            "score_0",
            "score_1",
            "score_2",
            "score_3",
        )
        metadata: dict[str, int] = {}
        for name in required_integers:
            value = observed_metadata(observation, name)
            if isinstance(value, int) and not isinstance(value, bool):
                metadata[name] = value
            else:
                unknown.append(f"metadata.{name}:unknown")
        wind = self._wind(observation)
        dealer = self._marked_player(observation, "dealer")
        if wind is None:
            unknown.append("metadata.wind:unknown")
        if dealer is None:
            unknown.append("metadata.dealer:unknown")

        riichi_states: list[str] = []
        any_meld = any(player_melds for player_melds in melds)
        for player in range(4):
            riichi = observed_marker(observation, f"riichi_{player}")
            double = observed_marker(observation, f"double_riichi_{player}")
            horizontal_indexes = [
                index
                for index in range(len(observed_tiles(observation, f"river_{player}")))
                if (observed_slot_state(observation, f"river_{player}", index) or {}).get(
                    "orientation"
                )
                == "horizontal"
            ]
            if len(horizontal_indexes) > 1:
                unknown.append(f"river_{player}:multiple_horizontal_riichi_tiles")
                riichi_states.append("none")
                continue
            horizontal_declaration = bool(horizontal_indexes)
            if riichi is False and horizontal_declaration:
                unknown.append(f"markers.riichi_{player}:inconsistent_with_river")
                riichi_states.append("none")
                continue
            if riichi is None:
                # A completed riichi always leaves its declaration discard
                # sideways. If there is no sideways discard, the public state
                # safely says "none" without a seat-specific UI marker.
                riichi = horizontal_declaration
            if not riichi:
                if double is True:
                    unknown.append(f"markers.double_riichi_{player}:inconsistent")
                riichi_states.append("none")
                continue
            if double is None:
                if not horizontal_indexes:
                    unknown.append(f"riichi_{player}:declaration_tile_unknown")
                    riichi_states.append("none")
                    continue
                declaration_index = horizontal_indexes[0]
                if declaration_index > 0:
                    double = False
                elif not any_meld:
                    # First discard and no call has occurred anywhere: double
                    # riichi is uniquely determined from public information.
                    double = True
                else:
                    unknown.append(f"riichi_{player}:double_status_ambiguous")
                    riichi_states.append("none")
                    continue
            if double and not riichi:
                unknown.append(f"markers.double_riichi_{player}:inconsistent")
            riichi_states.append(
                "double_accepted" if double else "accepted" if riichi else "none"
            )

        if unknown:
            return MidroundRestorePlan(None, None, tuple(sorted(set(unknown))))

        assert wind is not None and dealer is not None and len(drawn) == 1
        remaining = metadata["remaining_tiles"]
        bootstrap = VisionBootstrap(
            f"{wind}-{metadata['hand_number']}-{metadata['honba']}",
            wind,
            metadata["hand_number"],
            metadata["honba"],
            metadata["riichi_sticks"],
            dealer,
            tuple(metadata[f"score_{player}"] for player in range(4)),  # type: ignore[arg-type]
            remaining,
        )
        state = copy.deepcopy(self.manager.state)
        state.update(
            {
                "round": {
                    "wind": bootstrap.wind,
                    "hand_number": bootstrap.hand_number,
                    "honba": bootstrap.honba,
                    "riichi_sticks": bootstrap.riichi_sticks,
                    "dealer": bootstrap.dealer,
                },
                "scores": list(bootstrap.scores),
                "hand": hand,
                "drawn_tile": drawn[0],
                "dora_indicators": dora,
                "remaining_tiles": remaining,
                "turn": 70 - remaining,
                "current_actor": self.config.observer,
                "phase": {
                    "phase": "awaiting_discard",
                    "player": self.config.observer,
                },
                "pending_kan": None,
                "last_event": None,
                "event_history": [],
                "next_event_sequence": 0,
            }
        )
        rivers = [
            list(observed_tiles(observation, f"river_{player}"))
            for player in range(4)
        ]
        for player, values in enumerate(rivers):
            state["players"][player] = {
                "discards": [
                    {
                        "tile": tile,
                        "tsumogiri": False,
                        "tsumogiri_known": False,
                        "riichi_declaration": False,
                        "called_by": None,
                    }
                    for tile in values
                ],
                "melds": melds[player],
                "riichi": riichi_states[player],
            }
        try:
            state = json.loads(validate_state_json(canonical_json(state)))
        except ValueError as error:
            return MidroundRestorePlan(
                None,
                None,
                (f"rust_state_validation:{error}",),
            )
        return MidroundRestorePlan(state, bootstrap, ())

    def _restored_melds(
        self, observation: StableObservation
    ) -> tuple[list[list[dict[str, Any]]], list[str]]:
        output: list[list[dict[str, Any]]] = [[] for _ in range(4)]
        unknown: list[str] = []
        for player in range(4):
            seen_empty = False
            for group in range(4):
                region = f"meld_{player}_{group}"
                if region not in observation.recognized:
                    unknown.append(f"recognized.{region}:not_configured")
                    continue
                tiles = list(observed_tiles(observation, region))
                if not tiles:
                    seen_empty = True
                    continue
                if seen_empty:
                    unknown.append(f"recognized.{region}:non_contiguous_group")
                meld, reason = self._restored_meld(
                    observation, player, group, tiles
                )
                if meld is None:
                    unknown.append(f"recognized.{region}:{reason}")
                else:
                    output[player].append(meld)
        return output, unknown

    def _restored_meld(
        self,
        observation: StableObservation,
        player: int,
        group: int,
        tiles: list[dict[str, object]],
    ) -> tuple[dict[str, Any] | None, str]:
        region = f"meld_{player}_{group}"
        geometry = observed_meld(observation, region)
        if geometry is not None:
            return self._restored_geometry_meld(player, tiles, geometry)
        kinds = [str(tile["kind"]) for tile in tiles]
        if len(tiles) == 3 and len(set(kinds)) == 1:
            kind = "pon"
        elif len(tiles) == 3 and _is_chi(kinds):
            kind = "chi"
        elif len(tiles) == 4 and len(set(kinds)) == 1:
            active = [
                value
                for value in ("ankan", "daiminkan", "shouminkan")
                if observed_marker(observation, f"{region}_{value}") is True
            ]
            if len(active) != 1:
                return None, "kan_kind_ambiguous"
            kind = active[0]
        else:
            return None, "invalid_shape"

        if kind == "ankan":
            return {
                "kind": kind,
                "tiles": tiles,
                "called_tile": None,
                "from": None,
            }, ""

        if kind == "chi":
            from_player = (player + 3) % 4
        else:
            sources = [
                source
                for source in range(4)
                if observed_marker(observation, f"{region}_from_{source}") is True
            ]
            if len(sources) != 1 or sources[0] == player:
                return None, "source_ambiguous"
            from_player = sources[0]

        if all(tile == tiles[0] for tile in tiles):
            called_tile = tiles[0]
        else:
            indexes = [
                index
                for index in range(len(tiles))
                if observed_marker(observation, f"{region}_called_{index}") is True
            ]
            if len(indexes) != 1:
                return None, "called_tile_ambiguous"
            called_tile = tiles[indexes[0]]
        return {
            "kind": kind,
            "tiles": tiles,
            "called_tile": called_tile,
            "from": from_player,
        }, ""

    @staticmethod
    def _restored_geometry_meld(
        player: int,
        tiles: list[dict[str, object]],
        geometry: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, str]:
        kind = geometry.get("kind")
        if kind not in {"chi", "pon", "daiminkan", "ankan", "shouminkan"}:
            return None, "geometry_kind_invalid"
        geometry_tiles = geometry.get("tiles")
        if not isinstance(geometry_tiles, (list, tuple)) or list(geometry_tiles) != tiles:
            return None, "geometry_tiles_inconsistent"
        if kind == "ankan":
            if len(tiles) != 4 or len({str(tile["kind"]) for tile in tiles}) != 1:
                return None, "invalid_shape"
            return {
                "kind": kind,
                "tiles": tiles,
                "called_tile": None,
                "from": None,
            }, ""
        from_player = geometry.get("from")
        called_index = geometry.get("called_index")
        if not isinstance(from_player, int) or from_player == player or not 0 <= from_player <= 3:
            return None, "source_ambiguous"
        if kind == "chi" and from_player != (player + 3) % 4:
            return None, "chi_source_invalid"
        if not isinstance(called_index, int) or not 0 <= called_index < len(tiles):
            return None, "called_tile_ambiguous"
        return {
            "kind": kind,
            "tiles": tiles,
            "called_tile": tiles[called_index],
            "from": from_player,
        }, ""

    async def _restore_midround(
        self,
        socket: Any,
        observation: StableObservation,
        plan: MidroundRestorePlan,
    ) -> None:
        assert plan.state is not None and plan.bootstrap is not None
        state = plan.state
        bootstrap = plan.bootstrap
        request = {
            "type": "snapshot",
            "round_id": bootstrap.round_id,
            "next_source_sequence": self.manager.next_source_sequence,
            "state": state,
        }
        await socket.send(json.dumps(request))
        response = json.loads(await socket.recv())
        if response.get("type") == "error":
            raise VisionRuntimeError(f"mid-round snapshot was rejected: {response}")
        self.manager.restore(
            state,
            next_source_sequence=self.manager.next_source_sequence,
            round_id=bootstrap.round_id,
        )
        self.current_round_id = bootstrap.round_id
        self.desynced = False
        self.sync_status = "SYNCED"
        self.partial_unknown_fields = ()
        self._partial_observation = None
        self._runtime_block_reason = None
        self.detector.prime(observation)
        logger.info(
            "vision restored at mid-round self turn round=%s turn=%s",
            bootstrap.round_id,
            state["turn"],
        )

    async def _send_event(self, socket: Any, detected: DetectedEvent) -> None:
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

    async def _restore_fresh_round(
        self,
        socket: Any,
        observation: StableObservation,
        bootstrap: VisionBootstrap,
    ) -> None:
        temporary = RealtimeStateManager(self.config.observer)
        for sequence, detected in enumerate(
            self._bootstrap_events(observation, bootstrap)
        ):
            temporary.apply(
                {
                    "event_id": f"vision-resync-{bootstrap.round_id}-{sequence}",
                    "sequence": sequence,
                    "round_id": bootstrap.round_id,
                    "name": detected.name,
                    "data": detected.data,
                }
            )
        next_sequence = self.manager.next_source_sequence
        request = {
            "type": "snapshot",
            "round_id": bootstrap.round_id,
            "next_source_sequence": next_sequence,
            "state": temporary.state,
        }
        await socket.send(json.dumps(request))
        response = json.loads(await socket.recv())
        if response.get("type") == "error":
            raise VisionRuntimeError(f"fresh-round resync was rejected: {response}")
        self.manager.restore(
            temporary.state,
            next_source_sequence=next_sequence,
            round_id=bootstrap.round_id,
        )
        self.current_round_id = bootstrap.round_id
        self.desynced = False
        self.sync_status = "SYNCED"
        self.partial_unknown_fields = ()
        self._partial_observation = None
        self._runtime_block_reason = None
        self.detector.prime(observation)
        logger.info("vision resynchronized at fresh round=%s", bootstrap.round_id)

    def _bootstrap_events(
        self, observation: StableObservation, bootstrap: VisionBootstrap
    ) -> list[DetectedEvent]:
        events = [self._round_start(observation, bootstrap)]
        drawn = observed_tiles(observation, "drawn_tile")
        if drawn:
            events.append(
                DetectedEvent(
                    "draw",
                    {
                        "player": self.config.observer,
                        "tile": drawn[0],
                        "rinshan": False,
                    },
                    "stable_visual_bootstrap",
                    "dealer's initial drawn tile restored from the confirmed draw slot",
                )
            )
        return events

    def _round_start(
        self, observation: StableObservation, bootstrap: VisionBootstrap
    ) -> DetectedEvent:
        hand = list(observed_tiles(observation, "hand"))
        drawn = observed_tiles(observation, "drawn_tile")
        dora = observed_tiles(observation, "dora_indicators")
        drawn_is_valid = not drawn or (
            len(drawn) == 1 and bootstrap.dealer == self.config.observer
        )
        if len(hand) != 13 or not drawn_is_valid or len(dora) != 1:
            raise VisionRuntimeError(
                "fresh round bootstrap requires exactly 13 hand tiles, one dora indicator, "
                "and at most the observing dealer's initial drawn tile"
            )
        remaining_tiles = min(
            70,
            bootstrap.remaining_tiles + (1 if drawn else 0),
        )
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
                "remaining_tiles": remaining_tiles,
            },
            "stable_visual_bootstrap",
            "new round initialized from confirmed hand and dora slots",
        )

    def _envelope(self, detected: DetectedEvent) -> dict[str, Any]:
        sequence = self.manager.next_source_sequence
        return {
            "event_id": f"vision-{sequence}-{detected.name}",
            "sequence": sequence,
            "round_id": self.current_round_id,
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
            "source_acquisition": "screen_capture" if live else "directory_replay",
        }
        if live:
            hello["authorization"] = EXTERNAL_AUTHORIZATION
            hello["authorization_reference"] = self.config.authorization_reference
        return hello

    def _round_metadata(
        self, observation: StableObservation, *, initial: bool
    ) -> VisionBootstrap:
        fallback = self.config.bootstrap if initial else None
        wind = self._wind(observation)
        hand_number = observed_metadata(observation, "hand_number")
        honba = observed_metadata(observation, "honba")
        riichi_sticks = observed_metadata(observation, "riichi_sticks")
        dealer = self._marked_player(observation, "dealer")
        scores = tuple(observed_metadata(observation, f"score_{player}") for player in range(4))
        if (
            fallback is not None
            and all(isinstance(value, int) for value in scores)
            and sum(int(value) for value in scores) > sum(fallback.scores)
        ):
            logger.warning(
                "discarding impossible OCR score total=%s; using configured bootstrap scores",
                sum(int(value) for value in scores),
            )
            scores = fallback.scores
        observed_identity = (
            wind in {"east", "south", "west", "north"}
            and isinstance(hand_number, int)
            and isinstance(honba, int)
        )
        if fallback is not None:
            wind = wind or fallback.wind
            hand_number = hand_number if isinstance(hand_number, int) else fallback.hand_number
            honba = honba if isinstance(honba, int) else fallback.honba
            riichi_sticks = (
                riichi_sticks
                if isinstance(riichi_sticks, int)
                else fallback.riichi_sticks
            )
            dealer = dealer if dealer is not None else fallback.dealer
            scores = tuple(
                value if isinstance(value, int) else fallback.scores[index]
                for index, value in enumerate(scores)
            )
        if (
            wind not in {"east", "south", "west", "north"}
            or not isinstance(hand_number, int)
            or not isinstance(honba, int)
            or not isinstance(riichi_sticks, int)
            or dealer is None
            or any(not isinstance(value, int) for value in scores)
        ):
            raise VisionRuntimeError(
                "automatic round transition requires wind/dealer markers and OCR fields "
                "hand_number, honba, riichi_sticks, score_0..score_3"
            )
        round_id = (
            f"{wind}-{hand_number}-{honba}"
            if observed_identity or fallback is None
            else fallback.round_id
        )
        remaining = observed_metadata(observation, "remaining_tiles")
        return VisionBootstrap(
            round_id,
            wind,
            hand_number,
            honba,
            riichi_sticks,
            dealer,
            tuple(int(value) for value in scores),  # type: ignore[arg-type]
            int(remaining) if isinstance(remaining, int) else 70,
        )

    def _new_round_visible(self, observation: StableObservation) -> bool:
        try:
            metadata = self._round_metadata(observation, initial=False)
        except VisionRuntimeError:
            return False
        return metadata.round_id != self.current_round_id

    @staticmethod
    def _fresh_round(observation: StableObservation) -> bool:
        if len(observed_tiles(observation, "hand")) != 13:
            return False
        if len(observed_tiles(observation, "drawn_tile")) > 1:
            return False
        if len(observed_tiles(observation, "dora_indicators")) != 1:
            return False
        for player in range(4):
            if observed_tiles(observation, f"river_{player}"):
                return False
        for name, values in observation.recognized.items():
            if name.startswith("meld_") and values:
                return False
        return not VisionAdapterRuntime._terminal_visible(observation)

    @staticmethod
    def _terminal_visible(observation: StableObservation) -> bool:
        return any(
            observed_marker(observation, f"result_{kind}") is True
            for kind in ("ron", "tsumo", "draw", "abortive_draw")
        ) or all(
            isinstance(observed_metadata(observation, f"delta_{player}"), int)
            for player in range(4)
        )

    @staticmethod
    def _wind(observation: StableObservation) -> str | None:
        value = observed_metadata(observation, "wind")
        aliases = {
            "E": "east",
            "S": "south",
            "W": "west",
            "N": "north",
            "east": "east",
            "south": "south",
            "west": "west",
            "north": "north",
        }
        if isinstance(value, str) and value in aliases:
            return aliases[value]
        active = [
            wind
            for wind in ("east", "south", "west", "north")
            if observed_marker(observation, f"wind_{wind}") is True
        ]
        return active[0] if len(active) == 1 else None

    @staticmethod
    def _marked_player(
        observation: StableObservation, prefix: str
    ) -> int | None:
        value = observed_metadata(observation, prefix)
        if isinstance(value, int) and 0 <= value <= 3:
            return value
        active = [
            player
            for player in range(4)
            if observed_marker(observation, f"{prefix}_{player}") is True
        ]
        return active[0] if len(active) == 1 else None


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _is_chi(kinds: list[str]) -> bool:
    if len(kinds) != 3 or any(kind not in TILE_TO_INDEX for kind in kinds):
        return False
    indexes = sorted(TILE_TO_INDEX[kind] for kind in kinds)
    return (
        indexes[2] < 27
        and indexes[0] // 9 == indexes[2] // 9
        and indexes == list(range(indexes[0], indexes[0] + 3))
    )
