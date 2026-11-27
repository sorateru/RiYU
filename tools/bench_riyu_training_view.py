#!/usr/bin/env python3
"""Compare C and D on identical decisions without fitting any model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from mahjong_ai.data.lake_loader import ENCODER, model_features
from mahjong_ai.data.lake_training import LakeTrainingLoader


def run(c, d, work, max_shards):
    work.mkdir(mode=0o700)
    loader = LakeTrainingLoader(c, source="checkpoints", max_shards=max_shards)
    snapshots = work / "snapshots"
    start = perf_counter()
    loader.write_snapshot_cache(snapshots)
    derived_seconds = perf_counter() - start
    result = {}
    for name, current in (
        ("C_from_round_start", LakeTrainingLoader(c, source="replay", max_shards=max_shards)),
        ("C_from_checkpoint", loader),
        ("C_derived", LakeTrainingLoader(c, source="derived", snapshot_cache=snapshots, max_shards=max_shards)),
        ("D_direct", LakeTrainingLoader(d, source="snapshots", max_shards=max_shards)),
    ):
        start = perf_counter()
        digest = hashlib.sha256()
        size = 0
        for batch in current.batches():
            for observation in batch.inputs.snapshots:
                digest.update(json.dumps(observation, sort_keys=True, separators=(",", ":")).encode())
            for code in batch.inputs.action_codes:
                digest.update(code)
            size += len(batch.inputs.snapshots)
        seconds = perf_counter() - start
        result[name] = {"rows": size, "seconds": seconds, "rows_per_second": size / seconds,
                        "input_digest": digest.hexdigest()}
    if len({(r["rows"], r["input_digest"]) for r in result.values()}) != 1:
        raise ValueError("C and D do not contain the same input rows")

    start = perf_counter()
    features = np.concatenate([
        batch.encode(lambda inputs: np.stack([model_features(s) for s in inputs.snapshots]))
        for batch in loader.batches()
    ])
    tensor = work / "diagnostic.npy"
    np.save(tensor, features, allow_pickle=False)
    (work / "tensor.json").write_text(json.dumps({
        **loader.cache_scope, "encoder": ENCODER, "split": loader.split,
        "shards": [shard[0] for shard in loader.shards],
    }, sort_keys=True))
    tensor_seconds = perf_counter() - start
    start = perf_counter()
    restored = np.load(tensor, mmap_mode="r", allow_pickle=False)
    if not np.array_equal(features, restored):
        raise ValueError("tensor cache changed input arrays")
    read_seconds = perf_counter() - start
    return {
        "view": result,
        "snapshot_cache": {"seconds": derived_seconds, "bytes": sum(p.stat().st_size for p in snapshots.iterdir())},
        "tensor_cache": {"encode_write_seconds": tensor_seconds, "read_scan_seconds": read_seconds,
                         "bytes": tensor.stat().st_size, "rows": len(restored), "encoder": ENCODER},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("c", type=Path)
    parser.add_argument("d", type=Path)
    parser.add_argument("work", type=Path)
    parser.add_argument("--max-shards", type=int, default=1)
    args = parser.parse_args()
    if args.max_shards < 1:
        parser.error("--max-shards must be positive")
    print(json.dumps(run(args.c, args.d, args.work, args.max_shards), indent=2))
