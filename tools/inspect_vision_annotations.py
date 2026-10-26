#!/usr/bin/env python3
"""Inspect OCR fields and marker confidence on one saved screenshot."""

from __future__ import annotations

import argparse
import json

import cv2

from mahjong_ai.perception import LayoutConfig, ScreenAnnotationParser


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--layout", required=True)
    parser.add_argument("--templates", required=True)
    parser.add_argument("--tesseract")
    args = parser.parse_args()
    frame = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"cannot read image: {args.image}")
    layout = LayoutConfig.load(args.layout)
    if frame.shape[1] != layout.image_width or frame.shape[0] != layout.image_height:
        raise ValueError("image dimensions do not match the layout")
    result = ScreenAnnotationParser(
        layout,
        args.templates,
        tesseract_executable=args.tesseract,
    ).parse(frame)
    print(
        json.dumps(
            {
                "metadata": result.metadata,
                "metadata_confidences": result.metadata_confidences,
                "markers": result.markers,
                "marker_confidences": result.marker_confidences,
                "required_unknown": result.required_unknown,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
