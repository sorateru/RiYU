#!/usr/bin/env python3
"""Parse a saved screenshot; this tool has no screen-capture capability."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from mahjong_ai.perception import LayoutConfig, ScreenshotParser, TemplateTileClassifier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--layout", required=True)
    parser.add_argument("--templates", required=True)
    parser.add_argument("--base-state")
    args = parser.parse_args()
    base_state = None
    if args.base_state:
        base_state = json.loads(Path(args.base_state).read_text(encoding="utf-8"))
    result = ScreenshotParser(
        LayoutConfig.load(args.layout), TemplateTileClassifier(args.templates)
    ).parse(args.image, base_state)
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
