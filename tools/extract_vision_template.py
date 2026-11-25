#!/usr/bin/env python3
"""Extract one user-supplied tile, empty, or shared visual template."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from mahjong_ai.perception import LayoutConfig, rotate_slot_crop
from mahjong_ai.perception.profile import record_template_metadata
from mahjong_ai.schema import TILE_TO_INDEX

HONOR_ALIASES = {f"z{value}" for value in range(1, 8)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--layout")
    parser.add_argument("--region")
    parser.add_argument(
        "--crop",
        nargs=4,
        type=int,
        metavar=("X", "Y", "WIDTH", "HEIGHT"),
        help="direct crop for a reference/template sheet; does not require --layout",
    )
    parser.add_argument(
        "--rotation",
        type=int,
        choices=(0, 90, 180, 270),
        default=0,
        help="clockwise normalization applied to a direct --crop",
    )
    parser.add_argument(
        "--target-region",
        default="direct-crop",
        help="metadata target name used with --crop",
    )
    parser.add_argument("--slot", type=int)
    parser.add_argument(
        "--candidate",
        type=int,
        help="slot_variants candidate index for --region/--slot",
    )
    parser.add_argument("--meld-pattern", help="meld_layouts pattern name")
    parser.add_argument("--meld-slot", type=int, help="slot index inside --meld-pattern")
    parser.add_argument(
        "--label",
        required=True,
        help="internal tile kind, red five, empty, or shared visual label back",
    )
    parser.add_argument(
        "--variant",
        help="optional safe suffix such as bottom, left, right, dark; enables multiple templates",
    )
    parser.add_argument("--templates", required=True)
    parser.add_argument(
        "--source-reference",
        help="original local path or permitted source URL recorded as provenance",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    kind = args.label.removesuffix("_red")
    if args.label not in {"empty", "back"} and kind not in TILE_TO_INDEX and kind not in HONOR_ALIASES:
        parser.error("label must be a tile, red five, empty, or back")
    if args.label.endswith("_red") and kind not in {"m5", "p5", "s5"}:
        parser.error("only m5, p5 and s5 may use the _red suffix")
    if args.variant is not None and (
        not args.variant
        or any(not (character.isalnum() or character in "-_") for character in args.variant)
    ):
        parser.error("variant may contain only letters, digits, '-' and '_'")
    using_direct_crop = args.crop is not None
    if using_direct_crop and any(
        value is not None
        for value in (args.slot, args.candidate, args.meld_pattern, args.meld_slot)
    ):
        parser.error("--crop cannot be combined with slot or meld selection")
    if not using_direct_crop and (args.layout is None or args.region is None):
        parser.error("--layout and --region are required unless --crop is used")

    layout = None if using_direct_crop else LayoutConfig.load(args.layout)
    using_meld = args.meld_pattern is not None or args.meld_slot is not None
    if using_direct_crop:
        rectangle = tuple(args.crop)
        rotation = args.rotation
        target_region = args.target_region
    elif using_meld:
        assert layout is not None and args.region is not None
        if args.meld_pattern is None or args.meld_slot is None:
            parser.error("--meld-pattern and --meld-slot must be used together")
        patterns = layout.meld_layouts.get(args.region, ())
        pattern = next((value for value in patterns if value.name == args.meld_pattern), None)
        if pattern is None:
            parser.error(f"unknown meld pattern: {args.region}.{args.meld_pattern}")
        if not 0 <= args.meld_slot < len(pattern.slots):
            parser.error(f"meld slot must be in 0..{len(pattern.slots) - 1}")
        selected = pattern.slots[args.meld_slot]
        rectangle = selected.rectangle
        rotation = selected.rotation
        target_region = f"{args.region}.{pattern.name}[{args.meld_slot}]"
        if args.label == "back" and selected.role != "back":
            parser.error("the selected meld slot is not configured as a back")
    else:
        assert layout is not None and args.region is not None
        if args.slot is None:
            parser.error("--slot is required unless --meld-pattern is used")
        if args.region not in layout.slots:
            parser.error(f"unknown region: {args.region}")
        slots = layout.slots[args.region]
        if not 0 <= args.slot < len(slots):
            parser.error(f"slot must be in 0..{len(slots) - 1}")
        if args.candidate is None:
            rectangle = slots[args.slot]
            rotation = layout.rotations.get(args.region, 0)
        else:
            variants = layout.slot_variants.get(args.region)
            if variants is None or not 0 <= args.candidate < len(variants[args.slot]):
                parser.error("candidate does not exist for this logical slot")
            selected_variant = variants[args.slot][args.candidate]
            rectangle = selected_variant.rectangle
            rotation = selected_variant.rotation
        target_region = f"{args.region}[{args.slot}]"

    frame = cv2.imread(str(Path(args.image)), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"cannot read image: {args.image}")
    if layout is not None and (
        frame.shape[1] != layout.image_width or frame.shape[0] != layout.image_height
    ):
        raise ValueError("image dimensions do not match the layout")
    x, y, width, height = rectangle
    if (
        min(x, y) < 0
        or width <= 0
        or height <= 0
        or x + width > frame.shape[1]
        or y + height > frame.shape[0]
    ):
        raise ValueError("crop lies outside the input image")
    crop = rotate_slot_crop(
        frame[y : y + height, x : x + width],
        rotation,
    )
    suffix = "" if args.variant is None else f"__{args.variant}"
    template_root = Path(args.templates).resolve()
    destination = (
        template_root / "visuals" / f"{args.label}{suffix}.png"
        if args.label == "back"
        else template_root / f"{args.label}{suffix}.png"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not args.overwrite:
        raise FileExistsError(f"template already exists: {destination}")
    if not cv2.imwrite(str(destination), crop):
        raise RuntimeError(f"could not write template: {destination}")
    record_template_metadata(
        args.templates,
        destination,
        label=args.label,
        orientation=rotation,
        target_region=target_region,
        source_image=args.source_reference or str(Path(args.image).resolve()),
        crop=(x, y, width, height),
        image_size=(crop.shape[1], crop.shape[0]),
    )
    print(destination)


if __name__ == "__main__":
    main()
