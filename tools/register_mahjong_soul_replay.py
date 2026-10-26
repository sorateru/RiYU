#!/usr/bin/env python3
"""Register a decoded Mahjong Soul replay for later share-URL lookup."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mahjong_ai.data import LocalReplayURLCache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", help="official HTTPS share URL containing paipu=...")
    parser.add_argument("decoded", type=Path, help="user-owned decoded replay JSON")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/replay-url-cache"),
        help="local cache used by the analysis server",
    )
    args = parser.parse_args()
    raw = json.loads(args.decoded.read_text(encoding="utf-8"))
    destination = LocalReplayURLCache(args.cache_dir).register(args.url, raw)
    print(f"registered replay URL as {destination.name}")


if __name__ == "__main__":
    main()
