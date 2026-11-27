#!/usr/bin/env python3
"""Benchmark snapshot loading and NumPy/PyTorch handoff without training."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import resource
import sys
import tempfile
from time import perf_counter

import numpy as np
import torch

from mahjong_ai.data.lake_loader import PolicyLoader


def benchmark(root, shards, batch_size):
    torch.set_num_threads(1)
    accelerators = [x for x, available in (
        ("mps", torch.backends.mps.is_available()), ("cuda", torch.cuda.is_available())
    ) if available]
    results, totals = {}, set()
    with tempfile.TemporaryDirectory(prefix="riyu-loader-") as scratch:
        cache = Path(scratch) / "tensors"
        t = perf_counter()
        PolicyLoader(root, max_shards=shards).create_feature_cache(cache)
        creation = perf_counter() - t
        cache_bytes = sum(p.stat().st_size for p in cache.iterdir())
        for strategy in ("shard", "rowgroup", "buffer", "cache"):
            setup = perf_counter()
            loader = PolicyLoader(root, max_shards=shards, feature_cache=cache if strategy == "cache" else None)
            setup = perf_counter() - setup
            steps, transfer = [], {x: 0.0 for x in accelerators}
            bridge = 0.0
            started = tick = perf_counter()
            for batch in loader.batches(strategy=strategy, size=batch_size):
                steps.append(perf_counter() - tick)
                start = perf_counter()
                features = torch.from_numpy(batch.inputs.features)
                target = torch.from_numpy(batch.targets.chosen_index)
                bridge += perf_counter() - start
                for device in accelerators:
                    t = perf_counter()
                    features.to(device)
                    target.to(device)
                    if device == "mps":
                        torch.mps.synchronize()
                    else:
                        torch.cuda.synchronize()
                    transfer[device] += perf_counter() - t
                tick = perf_counter()
            elapsed = perf_counter() - started
            stats = asdict(loader.metrics)
            totals.add(stats["emitted"])
            results[strategy] = {
                **stats, "samples_s": stats["emitted"] / elapsed,
                "batch_p50_ms": float(np.median(steps) * 1000), "batch_p95_ms": float(np.percentile(steps, 95) * 1000),
                "adjacent_same_match_fraction": stats["adjacent_same_match"] / stats["emitted"],
                "setup_s": setup, "wall_s": elapsed, "numpy_torch_s": bridge, "device_transfer_s": transfer,
            }
    if len(totals) != 1:
        raise ValueError("policy split sample counts differ by loader strategy")
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {"results": results, "cache_build_s": creation, "cache_bytes": cache_bytes,
            "batch_size": batch_size, "shard_count": shards, "available_accelerators": accelerators,
            "peak_rss_bytes": usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024)}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("release", type=Path)
    p.add_argument("--shards", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=256)
    a = p.parse_args()
    print(json.dumps(benchmark(a.release, a.shards, a.batch_size), indent=2))
