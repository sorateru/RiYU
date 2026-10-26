#!/usr/bin/env python3
"""Replay normalized Mahjong Soul observation events into the local runtime."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from urllib.parse import urlparse

from mahjong_ai.adapters import MAJSOUL_EVENT_PROFILE
from mahjong_ai.analysis import MAJSOUL_SOURCE_SERVICE, REALTIME_PROTOCOL


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("events", help="JSONL containing normalized Mahjong Soul events")
    parser.add_argument("--session-id", default="majsoul-event-replay")
    parser.add_argument("--observer", type=int, default=0)
    parser.add_argument(
        "--url", default="ws://127.0.0.1:8787/ws/realtime/events/source"
    )
    parser.add_argument("--interval", type=float, default=0.05)
    parser.add_argument("--linger", type=float, default=0.0)
    args = parser.parse_args()
    if not 0 <= args.observer <= 3:
        parser.error("--observer must be between zero and three")
    if args.interval < 0 or args.linger < 0:
        parser.error("--interval and --linger cannot be negative")
    _require_loopback(args.url, parser)
    events = _load_events(Path(args.events))
    asyncio.run(
        _replay(
            args.url,
            args.session_id,
            args.observer,
            events,
            args.interval,
            args.linger,
        )
    )


def _require_loopback(url: str, parser: argparse.ArgumentParser) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"ws", "wss"} or parsed.hostname not in {
        "127.0.0.1",
        "localhost",
        "::1",
    }:
        parser.error("--url must target a loopback WebSocket")


def _load_events(path: Path) -> list[dict]:
    events: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            event = value.get("event") if isinstance(value, dict) and "event" in value else value
            if not isinstance(event, dict):
                raise ValueError(f"line {line_number} is not an event object")
            events.append(event)
    if not events:
        raise ValueError("the replay file contains no events")
    return events


async def _replay(
    url: str,
    session_id: str,
    observer: int,
    events: list[dict],
    interval: float,
    linger: float,
) -> None:
    try:
        import websockets
    except ImportError as error:
        raise RuntimeError("event replay requires the service dependency 'websockets'") from error

    async with websockets.connect(url, max_size=1_000_000) as socket:
        await socket.send(
            json.dumps(
                {
                    "type": "hello",
                    "protocol": REALTIME_PROTOCOL,
                    "session_id": session_id,
                    "observer": observer,
                    "source_kind": "offline_replay",
                    "source_service": MAJSOUL_SOURCE_SERVICE,
                    "source_profile": MAJSOUL_EVENT_PROFILE,
                }
            )
        )
        ready = json.loads(await socket.recv())
        if ready.get("type") != "ready":
            raise RuntimeError(f"event handshake failed: {ready}")
        for index, event in enumerate(events):
            await socket.send(json.dumps({"type": "event", "event": event}))
            response = json.loads(await socket.recv())
            if response.get("type") == "error":
                raise RuntimeError(f"event replay failed: {response}")
            print(
                json.dumps(
                    {
                        "source_sequence": event.get("sequence"),
                        "event": event.get("name"),
                        "response": response.get("type"),
                        "decision_required": bool(response.get("decision")),
                    },
                    ensure_ascii=False,
                )
            )
            if interval and index + 1 < len(events):
                await asyncio.sleep(interval)
        if linger:
            await asyncio.sleep(linger)


if __name__ == "__main__":
    main()
