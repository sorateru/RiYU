#!/usr/bin/env python3
"""Isolated, one-factor-at-a-time native Parquet measurements."""

import argparse
import json
from pathlib import Path

from compare_riyu_arrow import manifest, measure


GRID = {
    "workers-1": ["--workers", "1"], "workers-2": ["--workers", "2"],
    "workers-4": [], "workers-6": ["--workers", "6"],
    "arrow-batch-1024": ["--batch", "1024"], "arrow-batch-8192": ["--batch", "8192"],
    "row-group-2048": ["--row-group", "2048"],
    "zstd-1": ["--zstd", "1"], "zstd-6": ["--zstd", "6"],
    "checkpoint-16": ["--checkpoint", "16"], "checkpoint-64": ["--checkpoint", "64"],
    "profile-C": ["--profile", "C"], "shard-32": ["--shard-size", "32"],
    "legacy-shanten": ["--no-shanten-cache"],
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("database", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--limit", type=int, default=32)
    p.add_argument("--binary", type=Path, default=Path("target/release/riyu-data-lake"))
    p.add_argument("--authorization-reference", required=True)
    a = p.parse_args()
    a.output.mkdir(mode=0o700)
    metrics = {}
    for key, variation in GRID.items():
        root = a.output / key
        cmd = [str(a.binary), str(a.database), str(root), "--limit", str(a.limit),
               "--authorization-reference", a.authorization_reference]
        metric = measure(cmd + variation)
        data = manifest(root)
        metric.update({
            "source_hash_s": data["source_hash_us"] / 1_000_000,
            "bytes": sum(s["bytes"] for s in data["shards"]),
            "shard_bytes": [s["bytes"] for s in data["shards"]],
            "process_s": sum(s["footer"]["process_us"] for s in data["shards"]) / 1_000_000,
            "arrow_s": sum(f["arrow_us"] for s in data["shards"] for f in s["footer"]["files"].values()) / 1_000_000,
            "encode_write_s": sum(f["encode_write_us"] for s in data["shards"] for f in s["footer"]["files"].values()) / 1_000_000,
        })
        metrics[key] = metric
        (a.output / "metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
