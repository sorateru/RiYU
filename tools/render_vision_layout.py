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
ANCHOR_COLOR = (255, 180, 70)
VARIANT_COLOR = (255, 210, 90)
MELD_COLOR = (80, 150, 255)


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
    for region, values in layout.slot_variants.items():
        for index, candidates in enumerate(values):
            for candidate_index, candidate in enumerate(candidates):
                _draw(
                    overlay,
                    candidate.rectangle,
                    f"{region}[{index}].{candidate.orientation}:{candidate_index}",
                    VARIANT_COLOR,
                )
    for region, patterns in layout.meld_layouts.items():
        for pattern in patterns:
            for index, slot in enumerate(pattern.slots):
                _draw(
                    overlay,
                    slot.rectangle,
                    f"{region}.{pattern.name}[{index}]:{slot.role}",
                    MELD_COLOR,
                )
    for region, scan in layout.meld_scans.items():
        _draw(overlay, scan.rectangle, f"{region}:right-to-left-scan", MELD_COLOR)
        if scan.rotation == 0:
            scan_x, scan_y, scan_width, _scan_height = scan.rectangle
            for name, relative in (
                ("H", scan.horizontal_rect),
                ("V", scan.vertical_rect),
                ("H+", scan.upper_horizontal_rect),
            ):
                x_offset, y, width, height = relative
                _draw(
                    overlay,
                    (scan_x + scan_width + x_offset, scan_y + y, width, height),
                    f"{region}:{name}@start",
                    MELD_COLOR,
                )
    for name, config in layout.text_fields.items():
        _draw(overlay, config.rectangle, f"text:{name}", TEXT_COLOR)
    for name, config in layout.markers.items():
        _draw(overlay, config.rectangle, f"marker:{name}", MARKER_COLOR)
    for name, config in layout.scene_anchors.items():
        _draw(overlay, config.rectangle, f"anchor:{name}", ANCHOR_COLOR)

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
