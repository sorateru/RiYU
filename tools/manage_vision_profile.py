#!/usr/bin/env python3
"""Check, export, or import a local RiYU Vision profile."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mahjong_ai.perception import VisionAdapterConfig
from mahjong_ai.perception.profile import (
    export_vision_profile,
    import_vision_profile,
    inspect_vision_profile,
    prune_template_metadata,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check")
    check.add_argument("config")
    export = commands.add_parser("export")
    export.add_argument("config")
    export.add_argument("output")
    import_ = commands.add_parser("import")
    import_.add_argument("bundle")
    import_.add_argument("destination")
    prune = commands.add_parser("prune-manifest")
    prune.add_argument("templates")
    args = parser.parse_args()

    if args.command == "check":
        config = VisionAdapterConfig.load(args.config)
        report = inspect_vision_profile(config.layout_path, config.template_directory)
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        raise SystemExit(0 if report.complete else 2)
    if args.command == "export":
        print(export_vision_profile(args.config, args.output))
        return
    if args.command == "prune-manifest":
        path, count = prune_template_metadata(args.templates)
        print(json.dumps({"manifest": str(path), "removed": count}, ensure_ascii=False))
        return
    config_path = import_vision_profile(args.bundle, args.destination)
    print(config_path)


if __name__ == "__main__":
    main()
