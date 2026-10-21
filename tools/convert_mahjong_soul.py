#!/usr/bin/env python3
"""Convert a user-owned, locally decoded Mahjong Soul replay to a Bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mahjong_ai.data import create_mahjong_soul_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="decoded Mahjong Soul JSON")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--authorization",
        choices=("user_owned", "explicit_permission", "open_license"),
        default="user_owned",
    )
    parser.add_argument(
        "--acquisition",
        choices=("user_export", "manual_conversion", "authorized_adapter"),
        default="user_export",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--reference")
    source.add_argument("--share-url")
    args = parser.parse_args()

    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")
    raw = json.loads(args.input.read_text(encoding="utf-8"))
    bundle = create_mahjong_soul_bundle(
        raw,
        authorization=args.authorization,
        acquisition=args.acquisition,
        reference=args.reference,
        share_url=args.share_url,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(bundle['logs'])} rounds to {args.output}")


if __name__ == "__main__":
    main()
