#!/usr/bin/env python3
"""Render a Vision Adapter layout over a calibration screenshot."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from mahjong_ai.perception import LayoutConfig


SLOT_COLOR = (74, 193, 255)
TEXT_COLOR = (112, 224, 112)
MARKER_COLOR = (214, 125, 255)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("--layout", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    layout = LayoutConfig.load(args.layout)
    frame = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"cannot read image: {args.image}")
    if frame.shape[1] != layout.image_width or frame.shape[0] != layout.image_height:
        raise ValueError(
            f"image is {frame.shape[1]}x{frame.shape[0]}, "
            f"layout is {layout.image_width}x{layout.image_height}"
        )

    overlay = frame.copy()
    for region, rectangles in layout.slots.items():
        for index, rectangle in enumerate(rectangles):
            _draw(overlay, rectangle, f"{region}[{index}]", SLOT_COLOR)
    for name, config in layout.text_fields.items():
        _draw(overlay, config.rectangle, f"text:{name}", TEXT_COLOR)
    for name, config in layout.markers.items():
        _draw(overlay, config.rectangle, f"marker:{name}", MARKER_COLOR)

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), overlay):
        raise RuntimeError(f"could not write overlay: {output}")
    print(output)


def _draw(
    frame: object,
    rectangle: tuple[int, int, int, int],
    label: str,
    color: tuple[int, int, int],
) -> None:
    x, y, width, height = rectangle
    cv2.rectangle(frame, (x, y), (x + width, y + height), color, 1)
    cv2.putText(
        frame,
        label,
        (x + 2, max(12, y + 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.30,
        color,
        1,
        cv2.LINE_AA,
    )


if __name__ == "__main__":
    main()
