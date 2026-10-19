#!/usr/bin/env python3
"""Run a seat-balanced two-checkpoint east-south hanchan arena."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mahjong_ai.evaluation.arena import evaluate_checkpoints


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_a")
    parser.add_argument("checkpoint_b")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed-blocks", type=int, default=64)
    parser.add_argument("--base-seed", type=int, default=20_261_019)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error("output already exists")
    result = evaluate_checkpoints(
        args.checkpoint_a,
        args.checkpoint_b,
        seed_blocks=args.seed_blocks,
        base_seed=args.base_seed,
        workers=args.workers,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"summary": result["summary"], "paired": result["paired"], "verdict": result["verdict"]}, ensure_ascii=False, sort_keys=True))
    print(f"saved arena report to {output}")


if __name__ == "__main__":
    main()
