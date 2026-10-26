"""In-process fan-out for loopback realtime analysis sessions."""

from __future__ import annotations

import asyncio
import time
from dataclasses import asdict, dataclass, field
from threading import RLock
from typing import Any
from uuid import uuid4

from .realtime import RealtimeProtocolError, SessionHello


@dataclass
class _LiveSession:
    hello: SessionHello
    connected_at: float = field(default_factory=time.time)
    latest: dict[str, Any] | None = None
    last_sequence: int | None = None
    viewers: dict[str, asyncio.Queue[dict[str, Any]]] = field(default_factory=dict)


class RealtimeSessionHub:
    """Keep source ownership separate from read-only browser subscribers.

    Every viewer has a queue of size one. If analysis arrives faster than the
    browser renders it, the stale pending result is discarded and the newest
    state wins. Source messages themselves remain strictly ordered by the
    ``RealtimeAnalysisSession`` protocol.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, _LiveSession] = {}
        self._lock = RLock()

    def register_source(self, hello: SessionHello) -> None:
        with self._lock:
            if hello.session_id in self._sessions:
                raise RealtimeProtocolError(
                    "session_already_connected",
                    "a realtime source already owns this session_id",
                )
            self._sessions[hello.session_id] = _LiveSession(hello)

    def publish(self, session_id: str, message: dict[str, Any]) -> int:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise RealtimeProtocolError("unknown_session", "realtime session is not connected")
            session.latest = message
            sequence = message.get("sequence")
            if isinstance(sequence, int):
                session.last_sequence = sequence
            queues = tuple(session.viewers.values())
        for queue in queues:
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            queue.put_nowait(message)
        return len(queues)

    def subscribe(
        self, session_id: str
    ) -> tuple[str, asyncio.Queue[dict[str, Any]], dict[str, Any] | None]:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise RealtimeProtocolError("unknown_session", "realtime session is not connected")
            token = uuid4().hex
            queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=1)
            session.viewers[token] = queue
            return token, queue, session.latest

    def unsubscribe(self, session_id: str, token: str) -> None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is not None:
                session.viewers.pop(token, None)

    def unregister_source(self, session_id: str) -> None:
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is None:
            return
        offline = {
            "type": "source_offline",
            "session_id": session_id,
            "last_sequence": session.last_sequence,
        }
        for queue in session.viewers.values():
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            queue.put_nowait(offline)

    def sessions(self) -> list[dict[str, Any]]:
        with self._lock:
            sessions = tuple(self._sessions.values())
        result: list[dict[str, Any]] = []
        for session in sessions:
            hello = asdict(session.hello)
            authorization_recorded = bool(hello.pop("authorization_reference", None))
            result.append(
                {
                    **hello,
                    "authorization_recorded": authorization_recorded,
                    "connected_at_unix": session.connected_at,
                    "last_sequence": session.last_sequence,
                    "viewer_count": len(session.viewers),
                    "has_analysis": session.latest is not None,
                }
            )
        return result
