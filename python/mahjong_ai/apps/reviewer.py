"""FastAPI service for offline review and authorized normalized streams."""

from __future__ import annotations

import asyncio
import json
import secrets
from contextlib import suppress
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from mahjong_ai import analyze_state_json, validate_state_json
from mahjong_ai.ai.inference import InferenceEngine
from mahjong_ai.analysis import (
    RealtimeAnalysisSession,
    RealtimeProtocolError,
    RealtimeSessionHub,
    explain_analysis,
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


ALLOWED_BROWSER_ORIGINS = frozenset({"http://127.0.0.1:1420", "http://127.0.0.1:4173"})
MAX_STREAM_MESSAGE_BYTES = 1_000_000


def create_app(
    engine: InferenceEngine | None = None,
    local_play: LocalPlayManager | None = None,
    realtime_hub: RealtimeSessionHub | None = None,
) -> FastAPI:
    app = FastAPI(title="mahjong-ai local analysis service", version="0.3.0")
    play_manager = local_play or LocalPlayManager()
    stream_hub = realtime_hub or RealtimeSessionHub()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=sorted(ALLOWED_BROWSER_ORIGINS),
        allow_methods=["POST", "GET", "DELETE"],
        allow_headers=["content-type"],
    )

    def run_analysis(state: dict[str, Any]) -> dict[str, Any]:
        return engine.analyze(state) if engine else json.loads(analyze_state_json(canonical_json(state)))

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "mode": "offline_review_authorized_stream_and_separate_local_play",
            "local_play_available": play_manager.available,
            "realtime_source_endpoint": "/ws/realtime/source",
            "realtime_view_endpoint": "/ws/realtime/view/{session_id}",
        }

    @app.get("/realtime/sessions")
    def realtime_sessions() -> dict[str, Any]:
        return {"sessions": stream_hub.sessions()}

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

    @app.websocket("/ws/realtime/source")
    async def realtime_source(websocket: WebSocket) -> None:
        if not _origin_allowed(websocket):
            await websocket.close(code=1008, reason="browser origin is not allowed")
            return
        await websocket.accept()
        session = RealtimeAnalysisSession(run_analysis)
        registered = False
        send_lock = asyncio.Lock()

        async def send(payload: dict[str, Any]) -> None:
            async with send_lock:
                await websocket.send_json(payload)

        try:
            while not registered:
                try:
                    message = await _receive_stream_message(websocket)
                except RealtimeProtocolError as error:
                    await send(_stream_error(error.code, str(error)))
                    continue
                if message is None:
                    return
                try:
                    response = session.open(message)
                    stream_hub.register_source(session.hello)
                    registered = True
                    await send(response)
                except RealtimeProtocolError as error:
                    await send(_stream_error(error.code, str(error)))
                    if session.is_open and not registered:
                        await websocket.close(code=1008, reason=str(error))
                        return

            pending: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=1)

            async def analyze_latest() -> None:
                while True:
                    message = await pending.get()
                    try:
                        response = await run_in_threadpool(session.analyze, message)
                        viewers = stream_hub.publish(session.hello.session_id, response)
                        response["delivery"] = {"viewer_count": viewers}
                        await send(response)
                    except RealtimeProtocolError as error:
                        await send(_stream_error(error.code, str(error)))

            worker = asyncio.create_task(analyze_latest())
            last_received_sequence: int | None = None
            try:
                while True:
                    try:
                        message = await _receive_stream_message(websocket)
                    except RealtimeProtocolError as error:
                        await send(_stream_error(error.code, str(error)))
                        continue
                    if message is None:
                        return
                    if message.get("type") == "ping":
                        await send(
                            {"type": "pong", "session_id": session.hello.session_id}
                        )
                        continue
                    if message.get("type") != "analyze":
                        await send(
                            _stream_error(
                                "invalid_message_type", "expected analyze or ping"
                            )
                        )
                        continue
                    sequence = message.get("sequence")
                    if (
                        not isinstance(sequence, int)
                        or isinstance(sequence, bool)
                        or sequence < 0
                    ):
                        await send(
                            _stream_error(
                                "invalid_sequence",
                                "sequence must be a non-negative integer",
                            )
                        )
                        continue
                    if (
                        last_received_sequence is not None
                        and sequence <= last_received_sequence
                    ):
                        await send(
                            _stream_error(
                                "sequence_not_monotonic",
                                f"sequence must be greater than {last_received_sequence}",
                            )
                        )
                        continue
                    last_received_sequence = sequence
                    if pending.full():
                        superseded = pending.get_nowait()
                        await send(
                            {
                                "type": "superseded",
                                "session_id": session.hello.session_id,
                                "sequence": superseded["sequence"],
                                "replacement_sequence": sequence,
                            }
                        )
                    pending.put_nowait(message)
            finally:
                worker.cancel()
                with suppress(asyncio.CancelledError):
                    await worker
        except WebSocketDisconnect:
            return
        finally:
            if registered:
                stream_hub.unregister_source(session.hello.session_id)

    @app.websocket("/ws/realtime/view/{session_id}")
    async def realtime_view(websocket: WebSocket, session_id: str) -> None:
        if not _origin_allowed(websocket):
            await websocket.close(code=1008, reason="browser origin is not allowed")
            return
        await websocket.accept()
        try:
            token, queue, latest = stream_hub.subscribe(session_id)
        except RealtimeProtocolError as error:
            await websocket.send_json(_stream_error(error.code, str(error)))
            await websocket.close(code=1008, reason=str(error))
            return
        try:
            await websocket.send_json(
                {"type": "subscribed", "session_id": session_id, "latest": latest}
            )
            while True:
                try:
                    message = await asyncio.wait_for(queue.get(), timeout=15.0)
                except TimeoutError:
                    message = {"type": "heartbeat", "session_id": session_id}
                await websocket.send_json(message)
                if message.get("type") == "source_offline":
                    await websocket.close(code=1000, reason="source disconnected")
                    return
        except WebSocketDisconnect:
            return
        finally:
            stream_hub.unsubscribe(session_id, token)

    return app


def _stream_error(code: str, message: str) -> dict[str, str]:
    return {"type": "error", "code": code, "message": message}


def _origin_allowed(websocket: WebSocket) -> bool:
    origin = websocket.headers.get("origin")
    return origin is None or origin in ALLOWED_BROWSER_ORIGINS


async def _receive_stream_message(websocket: WebSocket) -> dict[str, Any] | None:
    event = await websocket.receive()
    if event["type"] == "websocket.disconnect":
        return None
    raw = event.get("text")
    if raw is None:
        raise RealtimeProtocolError("text_messages_only", "binary messages are not accepted")
    if len(raw.encode("utf-8")) > MAX_STREAM_MESSAGE_BYTES:
        await websocket.close(code=1009, reason="message exceeds the one-megabyte limit")
        return None
    try:
        message = json.loads(raw)
    except json.JSONDecodeError as error:
        raise RealtimeProtocolError("invalid_json", "message is not valid JSON") from error
    if not isinstance(message, dict):
        raise RealtimeProtocolError("invalid_message", "message must be a JSON object")
    return message


app = create_app()
