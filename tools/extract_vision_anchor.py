#!/usr/bin/env python3
"""Extract a local scene-anchor template from an authorized calibration frame."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from mahjong_ai.perception import LayoutConfig
from mahjong_ai.perception.profile import record_template_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("--layout", required=True)
    parser.add_argument("--anchor", required=True)
    parser.add_argument("--variant")
    parser.add_argument("--templates", required=True)
    parser.add_argument("--source-reference")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.variant is not None and (
        not args.variant
        or any(not (character.isalnum() or character in "-_") for character in args.variant)
    ):
        parser.error("variant may contain only letters, digits, '-' and '_'")
    layout = LayoutConfig.load(args.layout)
    config = layout.scene_anchors.get(args.anchor)
    if config is None:
        parser.error(f"unknown scene anchor: {args.anchor}")
    frame = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"cannot read image: {args.image}")
    if frame.shape[1] != layout.image_width or frame.shape[0] != layout.image_height:
        raise ValueError("image dimensions do not match the layout")
    x, y, width, height = config.rectangle
    crop = frame[y : y + height, x : x + width]
    suffix = "" if args.variant is None else f"__{args.variant}"
    destination = (
        Path(args.templates).resolve()
        / "anchors"
        / f"{args.anchor}{suffix}.png"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not args.overwrite:
        raise FileExistsError(f"anchor template already exists: {destination}")
    if not cv2.imwrite(str(destination), crop):
        raise RuntimeError(f"could not write anchor template: {destination}")
    record_template_metadata(
        args.templates,
        destination,
        label=args.anchor,
        orientation=0,
        target_region=f"scene_anchors.{args.anchor}",
        source_image=args.source_reference or str(Path(args.image).resolve()),
        crop=(x, y, width, height),
        image_size=(crop.shape[1], crop.shape[0]),
    )
    print(destination)


if __name__ == "__main__":
    main()
