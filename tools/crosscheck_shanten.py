"""Differential shanten check against MahjongRepository/mahjong.

This is a development verification tool, not a runtime dependency.
"""

from __future__ import annotations

import argparse
import json
import random

from mahjong.shanten import Shanten

from mahjong_ai import calculate_shanten_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-per-meld-count", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=20_261_018)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    reference = Shanten()
    wall = [tile for tile in range(34) for _ in range(4)]
    checked = 0
    for fixed_melds in range(5):
        for sample_index in range(args.samples_per_meld_count):
            tile_count = 14 - fixed_melds * 3 - (sample_index % 2)
            hand = rng.sample(wall, tile_count)
            counts = [0] * 34
            for tile in hand:
                counts[tile] += 1

            expected = reference.calculate_shanten(counts)
            actual = json.loads(
                calculate_shanten_json(json.dumps(counts), fixed_melds)
            )["minimum"]
            if actual != expected:
                raise AssertionError(
                    f"fixed={fixed_melds} counts={counts} "
                    f"expected={expected} actual={actual}"
                )
            checked += 1

    print(f"cross-check passed: {checked} random hands")


if __name__ == "__main__":
    main()
