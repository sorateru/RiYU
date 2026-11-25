#!/usr/bin/env python3
"""Extract one on/off visual marker template from a calibrated screenshot."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from mahjong_ai.perception import LayoutConfig
from mahjong_ai.perception.profile import record_template_metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--layout", required=True)
    parser.add_argument("--marker", required=True)
    parser.add_argument("--state", choices=("on", "off"), required=True)
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
    config = layout.markers.get(args.marker)
    if config is None:
        parser.error(f"unknown layout marker: {args.marker}")
    frame = cv2.imread(str(Path(args.image)), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"cannot read image: {args.image}")
    if frame.shape[1] != layout.image_width or frame.shape[0] != layout.image_height:
        raise ValueError("image dimensions do not match the layout")
    x, y, width, height = config.rectangle
    crop = frame[y : y + height, x : x + width]
    suffix = "" if args.variant is None else f"__{args.variant}"
    destination = (
        Path(args.templates).resolve()
        / "markers"
        / f"{args.marker}__{args.state}{suffix}.png"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not args.overwrite:
        raise FileExistsError(f"marker template already exists: {destination}")
    if not cv2.imwrite(str(destination), crop):
        raise RuntimeError(f"could not write marker template: {destination}")
    record_template_metadata(
        args.templates,
        destination,
        label=f"{args.marker}:{args.state}",
        orientation=0,
        target_region=f"markers.{args.marker}",
        source_image=args.source_reference or str(Path(args.image).resolve()),
        crop=(x, y, width, height),
        image_size=(crop.shape[1], crop.shape[0]),
    )
    print(destination)


if __name__ == "__main__":
    main()
