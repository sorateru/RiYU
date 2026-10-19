#!/usr/bin/env python3
"""Generate deterministic local self-play trajectories in parallel shards."""

from __future__ import annotations

import argparse

from mahjong_ai.training.selfplay import (
    ParallelSelfPlayConfig,
    ParallelSelfPlayRunner,
    verify_selfplay_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--workers", type=int, default=0, help="0 selects the available CPU count")
    parser.add_argument("--shard-rounds", type=int, default=64)
    parser.add_argument("--output", required=True, help="new or empty output directory")
    args = parser.parse_args()
    round_state = {
        "wind": "east",
        "hand_number": 1,
        "honba": 0,
        "riichi_sticks": 0,
        "dealer": 0,
    }
    config = ParallelSelfPlayConfig(
        rounds=args.rounds,
        seed=args.seed,
        workers=args.workers,
        shard_rounds=args.shard_rounds,
    )
    manifest = ParallelSelfPlayRunner(config, round_state).run(args.output)
    verify_selfplay_dataset(args.output)
    print(
        f"saved {manifest['transitions']} transitions from {manifest['rounds']} rounds "
        f"to {len(manifest['shards'])} shard(s) in {args.output}"
    )


if __name__ == "__main__":
    main()
