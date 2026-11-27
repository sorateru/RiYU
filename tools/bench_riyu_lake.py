#!/usr/bin/env python3
"""Local, aggregate-only benchmark for restricted RiYU pilot artifacts."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import resource
import tempfile
import subprocess
from time import perf_counter

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from pack_riyu_lake import packed_counts, semantic_hand, verified_rows


def records(spool):
    manifest = json.loads((spool / "manifest.json").read_text())
    for shard in manifest["shards"]:
        yield from verified_rows(spool / shard["file"], shard, manifest["build"])


def check_win_transfers(round_row, events):
    """Cross-check seat payments using factual hand values, independent of Rust scoring."""
    wins = [events[i] for i in round_row["wins"]]
    if not wins:
        return "draw"
    if any(w["responsibility"] is not None for w in wins):
        return "pao_skipped"
    start = next(e for e in events if e["round"] == round_row["round_index"] and e["type"] == "start")
    dealer = start["dealer"]
    loser = wins[0]["from"]
    nearest = min(wins, key=lambda w: (w["seat"] - loser) % 4)["seat"]
    honba, sticks = (int(n) for n in wins[0]["attrs"]["ba"].split(","))
    for win in wins:
        seat = win["seat"]
        expected = [0] * 4
        points = win["points"]
        if loser == seat:
            if seat == dealer:
                if points % 3:
                    return "dealer_tsumo_payment_mismatch"
                payments = [points // 3] * 4
            else:
                # With rounding to the next 100, the dealer component is
                # twice the child component or 100 less.
                child_payment = (points + 399) // 400 * 100
                dealer_payment = points - 2 * child_payment
                if dealer_payment not in (2 * child_payment, 2 * child_payment - 100):
                    return "child_tsumo_payment_mismatch"
                payments = [dealer_payment if i == dealer else child_payment for i in range(4)]
            for i, amount in enumerate(payments):
                if i != seat:
                    expected[i] -= amount + 100 * honba
            expected[seat] = -sum(expected)
        else:
            paid = points + (300 * honba if seat == nearest else 0)
            expected[loser] = -paid
            expected[seat] = paid
        if seat == nearest:
            expected[seat] += 1000 * sticks
        if expected != win["scores"]["delta"]:
            return "individual_transfer_mismatch"
    return "verified"


def aggregate(spool):
    statuses, diagnostics, reasons, policy, timings, rule_codes, transfers, call_kinds = (
        Counter() for _ in range(8)
    )
    restored = []
    matches = rounds = decisions = scored = skipped = 0
    manifest = json.loads((spool / "manifest.json").read_text())
    for row in records(spool):
        matches += 1
        statuses[row["status"]] += 1
        match = row.get("canonical")
        if match:
            rule_codes[match["raw_rule"]] += 1
        if row["status"] != "replayed":
            if row.get("diagnostic"):
                diagnostics[row["diagnostic"].split(": ")[-1].split(" {")[0]] += 1
            timings["parse_us"] += row.get("parse_us", 0)
            continue
        replay = row["replay"]
        source_events = match["events"]
        call_kinds.update(e["meld"]["kind"] for e in source_events if e["type"] == "call")
        timings.update(replay["timings"])
        restored.extend(replay.get("restore", []))
        for r in replay["rounds"]:
            transfers[check_win_transfers(r, source_events)] += 1
            rounds += 1
            reasons[r["reason"]] += 1
            if len(r["wins"]) > 1:
                reasons["double_ron"] += 1
            scored += r["scored_wins"]
            skipped += r["score_checks_skipped"]
            decisions += len(r["decisions"])
            policy.update((d["choice_status"], d["legal_status"]) for d in r["decisions"])
    restore = {}
    if restored:
        for interval, latency in zip(("event_only", "K16", "K32", "K64"), np.asarray(restored).T):
            restore[interval] = {
                "n": len(latency), "p50_us": np.median(latency) / 1000,
                "p95_us": np.percentile(latency, 95) / 1000,
            }
    return {
        "matches": matches, "rounds": rounds, "decisions": decisions, "status": statuses,
        "rule_codes": rule_codes, "diagnostics": diagnostics, "round_reasons": reasons,
        "score_checks": {"compared": scored, "skipped": skipped},
        "transfer_checks": transfers, "calls": call_kinds,
        "policy": {"/".join(k): v for k, v in policy.items()}, "sum_worker_timings": timings,
        "restore": restore, "source_hash_us": manifest["source_hash_us"],
        "wall_phases_us": {
            phase: sum(s["footer"][phase] for s in manifest["shards"])
            for phase in ("read_us", "process_us", "write_us")
        },
        "spool_bytes": sum(s["bytes"] for s in manifest["shards"]),
    }


def bench_hands(spool, matches, levels):
    source = []
    for index, row in enumerate(records(spool)):
        if index >= matches:
            break
        if row["status"] != "replayed":
            continue
        for r in row["replay"]["rounds"]:
            for decision in r["decisions"]:
                for h in decision["opponents"]:
                    source.append(((row["canonical"]["id"], r["round_index"], h["seat"]), h))
    if not source:
        raise ValueError("no accepted hands in sample")
    kinds = {
        "uint8_34": pa.binary(34), "packed_3bit": pa.binary(13),
        "tile_codes": pa.binary(), "physical_ids": pa.binary(), "delta_tile_codes": pa.binary(),
    }
    outputs = {}
    with tempfile.TemporaryDirectory(prefix="riyu-hand-bench-") as folder:
        for variant, arrow_type in kinds.items():
            t = perf_counter()
            previous = {}
            payloads, reds, drawn, round_start = [], [], [], []
            for key, hand in source:
                s = semantic_hand(hand)
                tiles = s["concealed"]
                fresh = key not in previous
                round_start.append(fresh)
                if variant in ("uint8_34", "packed_3bit"):
                    counts = [0] * 34
                    for x in tiles:
                        counts[x // 2] += 1
                    payload = bytes(counts) if variant == "uint8_34" else packed_counts(counts)
                elif variant == "tile_codes":
                    payload = tiles
                elif variant == "physical_ids":
                    payload = bytes(t["id"] for t in hand["concealed"])
                else:
                    before, after = Counter(previous.get(key, b"")), Counter(tiles)
                    removed = bytes((before - after).elements())
                    added = bytes((after - before).elements())
                    payload = bytes([len(removed)]) + removed + added
                    previous[key] = tiles
                payloads.append(payload)
                reds.append(s["red_mask"])
                drawn.append(s["drawn"])
            table = pa.table({
                "counts_or_tiles": pa.array(payloads, type=arrow_type),
                "concealed_red_mask": pa.array(reds, type=pa.uint8()),
                "drawn": pa.array(drawn, type=pa.uint8()),
                # Delta needs an explicit stream key. All variants carry the
                # same key so the comparison charges it equally.
                "stream": pa.array([f"{key[0]}:{key[1]}:{key[2]}" for key, _ in source]),
                "stream_start": pa.array(round_start),
            })
            convert_s = perf_counter() - t
            for level in levels:
                path = Path(folder) / f"{variant}-{level}.parquet"
                t = perf_counter()
                pq.write_table(table, path, compression="zstd", compression_level=level, row_group_size=8192)
                write_s = perf_counter() - t
                t = perf_counter()
                restored = pq.read_table(path)
                read_s = perf_counter() - t
                if restored["concealed_red_mask"].to_pylist() != reds or restored["drawn"].to_pylist() != drawn:
                    raise ValueError(f"red/draw roundtrip mismatch: {variant}")
                t = perf_counter()
                if variant == "uint8_34":
                    array = np.frombuffer(restored.column(0).combine_chunks().buffers()[1], dtype=np.uint8).reshape(-1, 34)
                elif variant == "packed_3bit":
                    encoded = np.frombuffer(restored.column(0).combine_chunks().buffers()[1], dtype=np.uint8).reshape(-1, 13)
                    padded = np.pad(encoded, ((0, 0), (0, 1))).astype(np.uint16)
                    positions = np.arange(34) * 3
                    start, shift = positions // 8, positions % 8
                    array = (((padded[:, start] | (padded[:, start + 1] << 8)) >> shift) & 7).astype(np.uint8)
                else:
                    array = np.zeros((len(source), 34), dtype=np.uint8)
                    previous = {}
                    for i, (payload, key) in enumerate(zip(
                        restored.column(0).to_pylist(), restored["stream"].to_pylist()
                    )):
                        if variant == "delta_tile_codes":
                            changes = Counter(previous.get(key, b""))
                            count = payload[0]
                            changes.subtract(payload[1:count+1])
                            changes.update(payload[count+1:])
                            payload = bytes(changes.elements())
                            previous[key] = payload
                        for tile in payload:
                            array[i, tile // (4 if variant == "physical_ids" else 2)] += 1
                decode_s = perf_counter() - t
                t = perf_counter()
                tensor_input = np.ascontiguousarray(array, dtype=np.float32)
                batch_s = perf_counter() - t
                if variant == "uint8_34" and level == levels[0]:
                    reference = tensor_input
                elif not np.array_equal(reference, tensor_input):
                    raise ValueError(f"roundtrip mismatch: {variant}")
                outputs[f"{variant}/zstd{level}"] = {
                    "rows": len(source), "payload_bytes": sum(map(len, payloads)),
                    "arrow_bytes": table.nbytes, "parquet_bytes": path.stat().st_size,
                    "to_arrow_s": convert_s, "write_s": write_s, "read_s": read_s,
                    "decode_s": decode_s, "to_float32_s": batch_s,
                }
    return outputs


def bench_storage(release):
    manifest = json.loads((release / "manifest.json").read_text())
    paths = sorted((release / "inputs").glob("*/*.parquet"))
    seq = {}
    for path in paths:
        key = path.parent.name
        values = seq.setdefault(key, {"rows": 0, "sec": 0.0})
        t = perf_counter()
        for batch in pq.ParquetFile(path).iter_batches(batch_size=4096):
            values["rows"] += batch.num_rows
        values["sec"] += perf_counter() - t
    result = {
        "input_sizes": {k: v["bytes"] for k, v in manifest["stats"].items() if k.startswith("inputs/")},
        "sequential_read": seq,
    }
    snapshots = list((release / "inputs" / "snapshots").glob("*.parquet"))
    if snapshots:
        data = pq.read_table(snapshots, columns=["snapshot"]).column(0).combine_chunks()
        rng = random.Random(20261126)
        batches = [[rng.randrange(len(data)) for _ in range(256)] for _ in range(60)]
        times = []
        for batch in batches:
            t = perf_counter()
            [json.loads(s) for s in data.take(pa.array(batch)).to_pylist()]
            times.append(1000 * (perf_counter() - t))
        result["in_memory_snapshot_batch"] = {"batch_size": 256, "p50_ms": np.median(times), "p95_ms": np.percentile(times, 95)}
    return result


def bench_lookup(releases, batch_size):
    """Time the same seeded batches through the public lookup and hash check."""
    from mahjong_ai.data.lake_view import input_batches, restore_observation

    roots = [p / "inputs" for p in releases]
    decisions = [row for batch in input_batches(roots[0], "decisions") for row in batch.to_pylist()]
    rng = random.Random(20261126)
    batches = [rng.sample(decisions, batch_size) for _ in range(5)]
    expected = None
    results = {}
    for release, root in zip(releases, roots):
        milliseconds, snapshots = [], []
        for batch in batches:
            start = perf_counter()
            snapshots.extend(restore_observation(root, d) for d in batch)
            milliseconds.append(1000 * (perf_counter() - start))
        if expected is None:
            expected = snapshots
        elif expected != snapshots:
            raise ValueError(f"storage profile changed the recovered observation: {release.name}")
        results[release.name] = {"batch_size": batch_size, "batches": len(batches),
                                 "p50_ms": np.median(milliseconds), "p95_ms": np.percentile(milliseconds, 95)}
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spool", type=Path)
    parser.add_argument("--hand-matches", type=int, default=0)
    parser.add_argument("--release", action="append", type=Path, default=[])
    parser.add_argument("--lookup-batch-size", type=int, default=0)
    args = parser.parse_args()
    report = {"aggregate": aggregate(args.spool)}
    if args.hand_matches:
        report["hand_encodings"] = bench_hands(args.spool, args.hand_matches, (1, 3, 6))
    if args.release:
        report["storage"] = {p.name: bench_storage(p) for p in args.release}
    if args.lookup_batch_size:
        if not args.release or args.lookup_batch_size < 1:
            parser.error("lookup requires a positive size and at least one release")
        report["lookup"] = bench_lookup(args.release, args.lookup_batch_size)
    print(json.dumps(report, indent=2))


def timed_command(command):
    """Wrap one child process and record getrusage peak RSS without host sysctl."""
    import sys

    start = perf_counter()
    subprocess.run(command, check=True)
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return {
        "wall_s": perf_counter() - start,
        "user_s": usage.ru_utime,
        "system_s": usage.ru_stime,
        "peak_rss_bytes": usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024),
    }


if __name__ == "__main__":
    main()
