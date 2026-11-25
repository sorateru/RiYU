#!/usr/bin/env python3
"""Run the read-only Mahjong Soul screen recognition adapter."""

from __future__ import annotations

import argparse
import asyncio
import logging
from dataclasses import replace
from pathlib import Path

from mahjong_ai.perception import VisionAdapterConfig, VisionAdapterRuntime


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Recognize an authorized Mahjong Soul screen region and forward "
            "stable, validated deltas to the local RiYU realtime service."
        )
    )
    parser.add_argument("config", help="path to the Vision Adapter JSON config")
    parser.add_argument(
        "--max-frames",
        type=int,
        help="stop after this many frames (primarily for saved-frame replay)",
    )
    parser.add_argument(
        "--save-latest-frame",
        help="local-only PNG overwritten periodically for capture diagnostics",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = VisionAdapterConfig.load(args.config)
    if args.save_latest_frame:
        config = replace(config, latest_frame_path=Path(args.save_latest_frame).resolve())
    try:
        asyncio.run(VisionAdapterRuntime(config).run(args.max_frames))
    except KeyboardInterrupt:
        logging.getLogger(__name__).info("Vision Adapter stopped")


if __name__ == "__main__":
    main()
