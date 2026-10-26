#!/usr/bin/env python3
"""Replay saved observable GameState JSONL into the loopback realtime source."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from urllib.parse import urlparse

from mahjong_ai.analysis import (
    MAJSOUL_REALTIME_PROFILE,
    MAJSOUL_SOURCE_SERVICE,
    REALTIME_PROTOCOL,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("states", help="JSONL containing observable GameState objects")
    parser.add_argument("--session-id", default="majsoul-offline-replay")
    parser.add_argument("--observer", type=int, default=0)
    parser.add_argument(
        "--url", default="ws://127.0.0.1:8787/ws/realtime/source"
    )
    parser.add_argument("--interval", type=float, default=0.25)
    parser.add_argument(
        "--linger",
        type=float,
        default=0.0,
        help="seconds to keep the source connected after the final state",
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if not 0 <= args.observer <= 3:
        parser.error("--observer must be between zero and three")
    if args.interval < 0:
        parser.error("--interval cannot be negative")
    if args.linger < 0:
        parser.error("--linger cannot be negative")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    _require_loopback(args.url, parser)
    states = _load_states(Path(args.states), args.limit)
    asyncio.run(
        _replay(
            args.url,
            args.session_id,
            args.observer,
            states,
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


def _load_states(path: Path, limit: int | None) -> list[dict]:
    states: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            state = value.get("state") if isinstance(value, dict) and "state" in value else value
            if not isinstance(state, dict):
                raise ValueError(f"line {line_number} is not a GameState object")
            states.append(state)
            if limit is not None and len(states) >= limit:
                break
    if not states:
        raise ValueError("the replay file contains no GameState objects")
    return states


async def _replay(
    url: str,
    session_id: str,
    observer: int,
    states: list[dict],
    interval: float,
    linger: float,
) -> None:
    try:
        import websockets
    except ImportError as error:
        raise RuntimeError(
            "the realtime replay tool requires the service dependency 'websockets'"
        ) from error

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
                    "source_profile": MAJSOUL_REALTIME_PROFILE,
                }
            )
        )
        ready = json.loads(await socket.recv())
        if ready.get("type") != "ready":
            raise RuntimeError(f"realtime handshake failed: {ready}")
        for sequence, state in enumerate(states):
            await socket.send(
                json.dumps({"type": "analyze", "sequence": sequence, "state": state})
            )
            response = json.loads(await socket.recv())
            if response.get("type") != "analysis":
                raise RuntimeError(f"realtime analysis failed: {response}")
            realtime = response["result"].get("realtime") or {}
            print(
                json.dumps(
                    {
                        "sequence": sequence,
                        "analysis_ms": realtime.get("analysis_ms"),
                        "state_changed": realtime.get("state_changed"),
                    },
                    ensure_ascii=False,
                )
            )
            if interval and sequence + 1 < len(states):
                await asyncio.sleep(interval)
        if linger:
            await asyncio.sleep(linger)


if __name__ == "__main__":
    main()
