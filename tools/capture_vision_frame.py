#!/usr/bin/env python3
"""Capture one authorized screen rectangle for local Vision Adapter calibration."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from mahjong_ai.perception import CaptureRegion, MacOSScreenCapture


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", help="output PNG path")
    parser.add_argument("--left", type=int, required=True)
    parser.add_argument("--top", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    args = parser.parse_args()

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = MacOSScreenCapture(
        CaptureRegion(args.left, args.top, args.width, args.height)
    ).capture()
    if not cv2.imwrite(str(output), frame):
        raise RuntimeError(f"could not write capture: {output}")
    print(output)


if __name__ == "__main__":
    main()
