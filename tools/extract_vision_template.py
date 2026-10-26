#!/usr/bin/env python3
"""Extract one user-supplied tile/empty template from a calibrated screenshot slot."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from mahjong_ai.perception import LayoutConfig
from mahjong_ai.schema import TILE_TO_INDEX

HONOR_ALIASES = {f"z{value}" for value in range(1, 8)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--layout", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--slot", type=int, required=True)
    parser.add_argument(
        "--label",
        required=True,
        help="internal tile kind (m1..z7), m5_red/p5_red/s5_red, or empty",
    )
    parser.add_argument(
        "--variant",
        help="optional safe suffix such as bottom, left, right, dark; enables multiple templates",
    )
    parser.add_argument("--templates", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    kind = args.label.removesuffix("_red")
    if args.label != "empty" and kind not in TILE_TO_INDEX and kind not in HONOR_ALIASES:
        parser.error("label must be m1..m9, p1..p9, s1..s9, z1..z7, *_red, or empty")
    if args.label.endswith("_red") and kind not in {"m5", "p5", "s5"}:
        parser.error("only m5, p5 and s5 may use the _red suffix")
    if args.variant is not None and (
        not args.variant
        or any(not (character.isalnum() or character in "-_") for character in args.variant)
    ):
        parser.error("variant may contain only letters, digits, '-' and '_'")
    layout = LayoutConfig.load(args.layout)
    if args.region not in layout.slots:
        parser.error(f"unknown region: {args.region}")
    slots = layout.slots[args.region]
    if not 0 <= args.slot < len(slots):
        parser.error(f"slot must be in 0..{len(slots) - 1}")

    frame = cv2.imread(str(Path(args.image)), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"cannot read image: {args.image}")
    if frame.shape[1] != layout.image_width or frame.shape[0] != layout.image_height:
        raise ValueError("image dimensions do not match the layout")
    x, y, width, height = slots[args.slot]
    crop = frame[y : y + height, x : x + width]
    suffix = "" if args.variant is None else f"__{args.variant}"
    destination = Path(args.templates).resolve() / f"{args.label}{suffix}.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not args.overwrite:
        raise FileExistsError(f"template already exists: {destination}")
    if not cv2.imwrite(str(destination), crop):
        raise RuntimeError(f"could not write template: {destination}")
    print(destination)


if __name__ == "__main__":
    main()
