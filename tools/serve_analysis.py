#!/usr/bin/env python3
"""Run the loopback-only offline and streaming analysis service."""

from __future__ import annotations

import argparse

import uvicorn

from mahjong_ai.ai.inference import InferenceEngine
from mahjong_ai.apps.local_play import LocalPlayManager
from mahjong_ai.apps.reviewer import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--checkpoint", help="validated model checkpoint; exact analysis works without it")
    parser.add_argument(
        "--belief-checkpoint",
        help="validated sequence-transformer-v2 checkpoint for opponent/wall priors",
    )
    parser.add_argument(
        "--play-checkpoint",
        help="model checkpoint for the separate local human-versus-AI mode",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if not 1 <= args.port <= 65_535:
        parser.error("--port must be between 1 and 65535")
    if args.belief_checkpoint and not args.checkpoint:
        parser.error("--belief-checkpoint requires --checkpoint")
    engine = (
        InferenceEngine.load(args.checkpoint, args.device, args.belief_checkpoint)
        if args.checkpoint
        else None
    )
    play_manager = (
        LocalPlayManager.from_checkpoint(args.play_checkpoint, args.device)
        if args.play_checkpoint
        else None
    )
    uvicorn.run(
        create_app(engine, play_manager),
        host="127.0.0.1",
        port=args.port,
        access_log=False,
    )


if __name__ == "__main__":
    main()
