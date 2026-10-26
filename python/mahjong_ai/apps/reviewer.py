"""FastAPI service for offline review and authorized normalized streams."""

from __future__ import annotations

import json
import re
import secrets
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from mahjong_ai import analyze_state_json, validate_state_json
from mahjong_ai.ai.inference import InferenceEngine
from mahjong_ai.analysis import RealtimeAnalysisSession, RealtimeProtocolError, explain_analysis
from mahjong_ai.apps.replay_analysis import build_web_replay_collection
from mahjong_ai.data import (
    EventLogError,
    LocalReplayURLCache,
    MahjongSoulConversionError,
    ReplayBundleError,
    ReplayURLCacheError,
    ReplayURLNotFound,
    create_mahjong_soul_bundle,
    mahjong_soul_replay_id,
    parse_event_log,
    parse_replay_bundle,
)
from mahjong_ai.schema import canonical_json

from .local_play import LocalPlayError, LocalPlayManager, LocalPlayUnavailable


class StateRequest(BaseModel):
    state: dict[str, Any]


class StartLocalPlayRequest(BaseModel):
    human_seat: int = 0
    seed: int | None = None


class LocalPlayActionRequest(BaseModel):
    action: dict[str, Any]


class ReplayAnalysisRequest(BaseModel):
    log: dict[str, Any] | None = None
    logs: list[dict[str, Any]] | None = None
    bundle: dict[str, Any] | None = None
    decoded_mahjong_soul: dict[str, Any] | None = None
    url: str | None = None
    observer: int = 0


class ExtensionImportRequest(BaseModel):
    decoded_mahjong_soul: dict[str, Any]
    observer: int = 0
    share_url: str


ALLOWED_BROWSER_ORIGINS = frozenset(
    {
        "http://127.0.0.1:1420",
        "http://localhost:1420",
        "http://127.0.0.1:4173",
        "http://localhost:4173",
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    }
)
MAX_STREAM_MESSAGE_BYTES = 1_000_000
MAX_REPLAY_UPLOAD_BYTES = 16 * 1024 * 1024
EXTENSION_IMPORT_TTL_SECONDS = 5 * 60
MAX_EXTENSION_IMPORTS = 8
EXTENSION_ORIGIN_RE = re.compile(r"chrome-extension://[a-p]{32}")


class _ExtensionImportStore:
    def __init__(self) -> None:
        self._values: OrderedDict[str, tuple[float, dict[str, Any]]] = OrderedDict()
        self._lock = threading.Lock()

    def put(self, analysis: dict[str, Any]) -> str:
        now = time.monotonic()
        with self._lock:
            self._expire(now)
            while len(self._values) >= MAX_EXTENSION_IMPORTS:
                self._values.popitem(last=False)
            token = secrets.token_urlsafe(24)
            self._values[token] = (now + EXTENSION_IMPORT_TTL_SECONDS, analysis)
            return token

    def get(self, token: str) -> dict[str, Any]:
        if not isinstance(token, str) or len(token) > 64:
            raise KeyError(token)
        now = time.monotonic()
        with self._lock:
            self._expire(now)
            expires, analysis = self._values[token]
            if expires <= now:
                del self._values[token]
                raise KeyError(token)
            return analysis

    def _expire(self, now: float) -> None:
        expired = [token for token, (expires, _) in self._values.items() if expires <= now]
        for token in expired:
            del self._values[token]


def create_app(
    engine: InferenceEngine | None = None,
    local_play: LocalPlayManager | None = None,
    web_dist: str | Path | None = None,
    replay_cache_dir: str | Path = "data/replay-url-cache",
) -> FastAPI:
    app = FastAPI(title="mahjong-ai local analysis service", version="0.3.0")
    play_manager = local_play or LocalPlayManager()
    replay_url_cache = LocalReplayURLCache(replay_cache_dir)
    extension_imports = _ExtensionImportStore()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=sorted(ALLOWED_BROWSER_ORIGINS),
        allow_origin_regex=r"^chrome-extension://[a-p]{32}$",
        allow_methods=["POST", "GET", "DELETE"],
        allow_headers=["content-type"],
    )

    @app.middleware("http")
    async def limit_replay_upload(request: Request, call_next: Any) -> Any:
        if request.method == "POST" and request.url.path in {
            "/api/replay-analysis",
            "/api/extension-import",
        }:
            raw_length = request.headers.get("content-length")
            try:
                content_length = int(raw_length) if raw_length is not None else None
            except ValueError:
                return JSONResponse(status_code=400, content={"detail": "invalid content-length"})
            if content_length is not None and content_length > MAX_REPLAY_UPLOAD_BYTES:
                return JSONResponse(
                    status_code=413,
                    content={
                        "detail": f"replay upload exceeds {MAX_REPLAY_UPLOAD_BYTES} bytes"
                    },
                )
        return await call_next(request)

    def run_analysis(state: dict[str, Any]) -> dict[str, Any]:
        return engine.analyze(state) if engine else json.loads(analyze_state_json(canonical_json(state)))

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "mode": "offline_review_authorized_stream_and_separate_local_play",
            "local_play_available": play_manager.available,
            "replay_import": {
                "formats": [
                    "mahjong-ai-event-log/v1",
                    "mahjong-ai-replay-bundle/v1",
                    "mahjong-soul-decoded-replay/v1",
                ],
                "maximum_rounds": 64,
                "maximum_decisions": 256,
                "maximum_upload_bytes": MAX_REPLAY_UPLOAD_BYTES,
                "direct_mahjong_soul_fetch": False,
                "mahjong_soul_url_import": "local_authorized_cache",
                "browser_extension_import": True,
            },
        }

    @app.post("/local-play/sessions")
    def start_local_play(request: StartLocalPlayRequest) -> dict[str, Any]:
        try:
            seed = request.seed if request.seed is not None else secrets.randbits(63)
            return play_manager.start(seed=seed, human_seat=request.human_seat)
        except LocalPlayUnavailable as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        except LocalPlayError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.get("/local-play/sessions/{session_id}")
    def get_local_play(session_id: str) -> dict[str, Any]:
        try:
            return play_manager.get(session_id).snapshot()
        except LocalPlayError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.post("/local-play/sessions/{session_id}/actions")
    def act_local_play(session_id: str, request: LocalPlayActionRequest) -> dict[str, Any]:
        try:
            return play_manager.get(session_id).act(request.action)
        except LocalPlayError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.delete("/local-play/sessions/{session_id}", status_code=204)
    def close_local_play(session_id: str) -> None:
        try:
            play_manager.close(session_id)
        except LocalPlayError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.post("/analyze")
    def analyze(request: StateRequest) -> dict[str, Any]:
        try:
            normalized = json.loads(validate_state_json(canonical_json(request.state)))
            result = run_analysis(normalized)
            result["explanations"] = explain_analysis(result)
            result["safety_scope"] = "offline_saved_log_or_local_simulator_only"
            return result
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.get("/api/health")
    def web_health() -> dict[str, Any]:
        return health()

    @app.post("/api/replay-analysis")
    def analyze_replay(request: ReplayAnalysisRequest) -> dict[str, Any]:
        try:
            provided = sum(
                value is not None
                for value in (
                    request.log,
                    request.logs,
                    request.bundle,
                    request.decoded_mahjong_soul,
                    request.url,
                )
            )
            if provided != 1:
                raise ValueError(
                    "provide exactly one of log, logs, bundle, decoded_mahjong_soul, or url"
                )
            source = None
            if request.bundle is not None:
                bundle = parse_replay_bundle(request.bundle)
                logs = list(bundle.logs)
                source = bundle.source.public_summary()
            elif request.decoded_mahjong_soul is not None:
                bundle = parse_replay_bundle(
                    create_mahjong_soul_bundle(request.decoded_mahjong_soul)
                )
                logs = list(bundle.logs)
                source = bundle.source.public_summary()
            elif request.url is not None:
                bundle = parse_replay_bundle(replay_url_cache.resolve(request.url))
                logs = list(bundle.logs)
                source = bundle.source.public_summary()
            else:
                documents = request.logs if request.logs is not None else [request.log]
                logs = [
                    parse_event_log(document, default_log_id=f"uploaded-round-{index + 1}")
                    for index, document in enumerate(documents)
                    if document is not None
                ]
            return build_web_replay_collection(
                logs, request.observer, run_analysis, source=source
            )
        except ReplayURLNotFound as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except (
            EventLogError,
            MahjongSoulConversionError,
            ReplayBundleError,
            ReplayURLCacheError,
            ValueError,
        ) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.post("/api/extension-import")
    def import_from_extension(
        body: ExtensionImportRequest, request: Request
    ) -> dict[str, str]:
        origin = request.headers.get("origin")
        if origin is not None and EXTENSION_ORIGIN_RE.fullmatch(origin) is None:
            raise HTTPException(status_code=403, detail="extension import requires a Chrome extension origin")
        try:
            # This also ensures a stale captured record cannot be submitted from
            # an ordinary lobby/game URL: the active tab must still carry one
            # valid official paipu identifier.
            mahjong_soul_replay_id(body.share_url)
            bundle = parse_replay_bundle(
                create_mahjong_soul_bundle(
                    body.decoded_mahjong_soul,
                    acquisition="authorized_adapter",
                    share_url=body.share_url,
                )
            )
            analysis = build_web_replay_collection(
                list(bundle.logs),
                body.observer,
                run_analysis,
                source=bundle.source.public_summary(),
            )
            token = extension_imports.put(analysis)
            base = str(request.base_url).rstrip("/")
            return {"review_url": f"{base}/?import={token}"}
        except (
            EventLogError,
            MahjongSoulConversionError,
            ReplayBundleError,
            ValueError,
        ) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.get("/api/extension-import/{token}")
    def get_extension_import(token: str) -> dict[str, Any]:
        try:
            return extension_imports.get(token)
        except KeyError as error:
            raise HTTPException(
                status_code=404, detail="extension import expired or does not exist"
            ) from error

    @app.websocket("/ws/lab")
    async def realtime_lab(websocket: WebSocket) -> None:
        origin = websocket.headers.get("origin")
        if origin is not None and origin not in ALLOWED_BROWSER_ORIGINS:
            await websocket.close(code=1008, reason="browser origin is not allowed")
            return
        await websocket.accept()
        session = RealtimeAnalysisSession(run_analysis)
        try:
            while True:
                event = await websocket.receive()
                if event["type"] == "websocket.disconnect":
                    return
                raw = event.get("text")
                if raw is None:
                    await websocket.send_json(_stream_error("text_messages_only", "binary messages are not accepted"))
                    continue
                if len(raw.encode("utf-8")) > MAX_STREAM_MESSAGE_BYTES:
                    await websocket.close(code=1009, reason="message exceeds the one-megabyte limit")
                    return
                try:
                    message = json.loads(raw)
                    if not isinstance(message, dict):
                        raise RealtimeProtocolError("invalid_message", "message must be a JSON object")
                    if not session.is_open:
                        response = session.open(message)
                    elif message.get("type") == "ping":
                        response = {"type": "pong"}
                    else:
                        response = await run_in_threadpool(session.analyze, message)
                    await websocket.send_json(response)
                except json.JSONDecodeError:
                    await websocket.send_json(_stream_error("invalid_json", "message is not valid JSON"))
                except RealtimeProtocolError as error:
                    await websocket.send_json(_stream_error(error.code, str(error)))
        except WebSocketDisconnect:
            return

    if web_dist is not None:
        distribution = Path(web_dist).resolve()
        if not distribution.is_dir() or not (distribution / "index.html").is_file():
            raise ValueError(f"web distribution is missing index.html: {distribution}")
        app.mount("/", StaticFiles(directory=distribution, html=True), name="riyu-web")

    return app


def _stream_error(code: str, message: str) -> dict[str, str]:
    return {"type": "error", "code": code, "message": message}


app = create_app()
