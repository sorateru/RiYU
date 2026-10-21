#!/usr/bin/env python3
"""Create a validated provenance bundle from normalized replay round logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mahjong_ai.data import create_replay_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path, help="normalized round JSON files")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--platform", choices=("mahjong_soul", "tenhou", "local", "other"), required=True
    )
    parser.add_argument(
        "--acquisition",
        choices=("user_export", "manual_conversion", "authorized_adapter", "generated"),
        default="user_export",
    )
    parser.add_argument(
        "--authorization",
        choices=("user_owned", "explicit_permission", "open_license", "generated"),
        required=True,
    )
    parser.add_argument("--converter-name", required=True)
    parser.add_argument("--converter-version", required=True)
    parser.add_argument("--reference")
    parser.add_argument(
        "--share-url",
        help="optional source URL; only a SHA-256 reference is exposed by the analysis API",
    )
    args = parser.parse_args()

    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")
    documents = [json.loads(path.read_text(encoding="utf-8")) for path in args.logs]
    bundle = create_replay_bundle(
        documents,
        platform=args.platform,
        acquisition=args.acquisition,
        authorization=args.authorization,
        converter_name=args.converter_name,
        converter_version=args.converter_version,
        reference=args.reference,
        share_url=args.share_url,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(documents)} rounds to {args.output}")


if __name__ == "__main__":
    main()
