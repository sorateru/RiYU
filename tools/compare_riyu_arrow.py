#!/usr/bin/env python3
"""Compare the direct Parquet, IPC and JSONL routes on identical source records."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import pyarrow.parquet as pq

from pack_riyu_lake import SCHEMAS


def measure(command):
    start = perf_counter()
    task = subprocess.Popen(command)
    _, status, usage = os.wait4(task.pid, 0)
    task.returncode = os.waitstatus_to_exitcode(status)
    if task.returncode:
        raise subprocess.CalledProcessError(task.returncode, command)
    return {
        "wall_s": perf_counter() - start,
        "cpu_s": usage.ru_utime + usage.ru_stime,
        "rss_bytes": usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024),
        "input_blocks": usage.ru_inblock,
        "output_blocks": usage.ru_oublock,
    }


def manifest(root):
    return json.loads((root / "manifest.json").read_text())


def compare(direct, releases):
    counts = {}
    for shard in manifest(direct)["shards"]:
        for table, info in shard["footer"]["files"].items():
            paths = [direct / shard["file"] / (table + ".parquet")]
            paths.extend(root / table / f"part-{shard['offset']:08}.parquet" for root in releases)
            files = [pq.ParquetFile(p) for p in paths]
            reference, *others = [f.read() for f in files]
            if reference.num_rows != info["rows"] or not reference.schema.equals(SCHEMAS[table]):
                raise ValueError(f"bad direct schema or count for {table}")
            if any(f.schema_arrow.metadata != files[0].schema_arrow.metadata for f in files[1:]):
                raise ValueError(f"schema metadata mismatch for {table}")
            if not all(reference.equals(t) for t in others):
                # Arrow/Parquet implementations may call a list child item or element.
                # The shared logical schema and value sequence still have to match.
                if not all(reference.schema.equals(t.schema, check_metadata=False)
                           and reference.to_pylist() == t.to_pylist() for t in others):
                    bad_columns = [name for name in reference.schema.names
                                   if any(not reference[name].equals(t[name]) for t in others)]
                    raise ValueError(f"logical mismatch at {shard['offset']} {table}: {bad_columns}")
            counts[table] = counts.get(table, 0) + reference.num_rows
    for root in releases:
        actual = {k: s["rows"] for k, s in manifest(root)["stats"].items()}
        if counts != actual:
            raise ValueError("unexpected/missing comparison rows")
    return counts


def run(args):
    args.output.mkdir(mode=0o700)
    metrics, final_paths = {}, []
    for mode in ("parquet", "ipc", "jsonl"):
        root = args.output / mode
        print(f"running {mode}", flush=True)
        phase = measure([
            str(args.binary), str(args.database), str(root), "--format", mode,
            "--limit", str(args.limit), "--shard-size", str(args.shard_size),
            "--workers", str(args.workers), "--authorization-reference", args.authorization_reference,
        ])
        raw = manifest(root)
        intermediate = sum(s["bytes"] for s in raw["shards"])
        stages = {}
        for shard in raw["shards"]:
            for stage, elapsed in shard["footer"].get("timings", {}).items():
                stages[stage] = stages.get(stage, 0) + elapsed
        packed = None
        if mode == "parquet":
            final_bytes = intermediate
            intermediate = 0
        else:
            release = args.output / f"{mode}-parquet"
            cmd = [sys.executable, str(Path(__file__).with_name("pack_riyu_lake.py")), str(release), str(root)]
            if mode == "ipc":
                cmd.append("--from-ipc")
            packed = measure(cmd)
            final_paths.append(release)
            final_bytes = sum(entry["bytes"] for entry in manifest(release)["stats"].values())
        metrics[mode] = {"rust": phase, "pack": packed, "intermediate_bytes": intermediate,
                         "final_bytes": final_bytes, "worker_timings": stages}
        (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    metrics["equal_rows"] = compare(args.output / "parquet", final_paths)
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--binary", type=Path, default=Path("target/release/riyu-data-lake"))
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--shard-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--authorization-reference", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
