#!/usr/bin/env python3
"""Pack verified Rust pilot shards into physically separated Arrow/Parquet views."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import tempfile
from time import perf_counter

import pyarrow as pa
import pyarrow.parquet as pq


CONTRACT = json.loads((Path(__file__).resolve().parents[1] /
                       "schemas/riyu-data-lake-pilot-v1.json").read_text())


def arrow_type(spec):
    if isinstance(spec, str):
        if spec.startswith("b") and spec[1:].isdigit():
            return pa.binary(int(spec[1:]))
        return pa.type_for_alias(spec)
    kind, item = spec
    if kind == "list":
        return pa.list_(arrow_type(item))
    if kind == "struct":
        return pa.struct(arrow_fields(item))
    raise ValueError(f"unknown Arrow type {spec}")


def arrow_fields(fields):
    return [(key, arrow_type(spec)) for key, spec in fields]


SCHEMAS = {table: pa.schema(arrow_fields(spec["fields"])) for table, spec in CONTRACT.items()}


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def packed_counts(counts):
    bits = sum(int(value) << (3 * index) for index, value in enumerate(counts))
    return bits.to_bytes(13, "little")


def semantic_hand(hand):
    concealed = bytes(sorted(2 * t["kind"] + t["red"] for t in hand["concealed"]))
    red_mask = sum(1 << (t["kind"] // 9) for t in hand["concealed"] if t["red"])
    drawn = hand["drawn"]
    return {
        "seat": hand["seat"], "concealed": concealed, "red_mask": red_mask,
        "drawn": 2 * drawn["kind"] + drawn["red"] if drawn else None,
    }


def belief_full(hand):
    semantic = semantic_hand(hand)
    counts = bytearray(34)
    for code in semantic.pop("concealed"):
        counts[code // 2] += 1
    return {**semantic, "concealed_counts": bytes(counts)}


def local_actions_arrow(local):
    def hand(profile):
        return {**profile, "concealed": bytes(profile["concealed"])} if profile is not None else None

    candidates = []
    for item in local["candidates"]:
        axis = item["successor_set_hash"]
        successors = [{
            **child, "id": bytes.fromhex(child["id"]), "semantic": bytes(child["semantic"]),
            "after_discard": hand(child["after_discard"]),
        } for child in item["successors"]]
        candidates.append({
            **item, "after": hand(item["after"]), "successors": successors,
            "successor_set_hash": bytes.fromhex(axis) if axis is not None else None,
            "visible_win_score": encode(item["visible_win_score"]) if item["visible_win_score"] is not None else None,
        })
    return {"before": hand(local["before"]), "candidates": candidates}


def convert_row(row, rows, seen, action_registry, profile, checkpoint_interval):
    match = row.get("canonical", {})
    match_id = bytes.fromhex(match["id"]) if match else None
    if match_id is not None:
        if match_id in seen:
            raise ValueError("duplicate match across ranges")
        seen.add(match_id)
    rows["restricted/matches"].append({
        **match, "match_id": match_id, "source_locator": match.get("locator"),
        "root_attrs": encode(match["root_attrs"]) if "root_attrs" in match else None,
        "payload_sha256": bytes.fromhex(match["payload_sha256"]) if match else None,
        "locator_hash": bytes.fromhex(row["locator_hash"]) if "locator_hash" in row else None,
        "status": row["status"], "diagnostic": row.get("diagnostic"),
    })
    for event in match.get("events", ()):
        scores = event.get("scores")
        dora = event.get("dora")
        rows["restricted/events"].append({
            **event, "match_id": match_id, "attrs": encode(event["attrs"]),
            "score_change": scores if isinstance(scores, dict) else None,
            "score_vector": scores if isinstance(scores, list) else None,
            "initial_dora": dora if isinstance(dora, dict) else None,
            "dora": dora if isinstance(dora, list) else None,
            "rating": event.get("ratings"),
        })
    if row["status"] != "replayed":
        return
    bucket = int.from_bytes(match_id[:8], "big") % 100
    rows["governance/split"].append({
        "match_id": match_id, "split": "train" if bucket < 80 else "validation" if bucket < 90 else "final_test",
    })
    rows["targets/match"].append({
        "match_id": match_id, "final_scores": match["final_scores"], "final_ranks": match["final_ranks"],
    })
    by_ordinal = {e["source_ordinal"]: e for e in match["events"]}

    def opponent_action(ordinal):
        if ordinal is None:
            return None
        event = by_ordinal[ordinal]
        tile = event.get("tile")
        return {
            "source_ordinal": ordinal, "type": event["type"],
            "tile_code": 2 * tile["kind"] + tile["red"] if tile else None,
            "meld_kind": event.get("meld", {}).get("kind"),
        }

    for r in row["replay"]["rounds"]:
        prefix = {"match_id": match_id, "round": r["round_index"]}
        rows["targets/round"].append({
            **prefix, **r, "terminal_observations": [encode(s) for s in r["terminal_observations"]],
        })
        for ordinal in r["wins"]:
            event = by_ordinal[ordinal]
            rows["targets/wins"].append({**prefix, **event, "score_change": event["scores"]})
        if profile != "B":
            for seat, events in enumerate(r["safe_events"]):
                rows["inputs/events"].extend({
                    **prefix, "seat": seat, "seq": e["sequence"], "event": encode(e),
                } for e in events)
        if profile in ("C", "D"):
            for seat, checkpoints in enumerate(r["checkpoints16"]):
                rows["inputs/checkpoints"].extend({
                    **prefix, "seat": seat, "seq": c["sequence"], "state": encode(c["state"]),
                } for c in checkpoints if c["sequence"] % checkpoint_interval == 0)
        for d in r["decisions"]:
            key = {"decision_id": bytes.fromhex(d["id"])}
            axis = {"action_set_hash": bytes.fromhex(d["action_set_hash"])}
            actions = [
                {"id": bytes.fromhex(a["id"]), "semantic": bytes(a["semantic"])} for a in d["actions"]
            ]
            if len({a["id"] for a in actions}) != len(actions):
                raise ValueError("duplicate semantic ActionId")
            if hashlib.sha256(encode(d["actions"])).digest() != axis["action_set_hash"]:
                raise ValueError("invalid action axis digest")
            for action in actions:
                if action_registry.setdefault(action["id"], action["semantic"]) != action["semantic"]:
                    raise ValueError("semantic ActionId collision")
            for value in [d["attack"]["candidates"], d["defense"]]:
                indices = [c["action_index"] for c in value]
                if indices != sorted(set(indices)) or any(not 0 <= i < len(actions) for i in indices):
                    raise ValueError("teacher axis does not match the legal action set")
            if d["choice"] is not None and not 0 <= d["choice"] < len(actions):
                raise ValueError("selected action outside action axis")
            if d["choice_status"] == "censored" and d["choice"] is not None:
                raise ValueError("censored window carries a choice")
            rows["inputs/decisions"].append({
                **prefix, **key, "rule_code": match["raw_rule"],
                **{k: d[k] for k in ("actor", "observation_seq", "window")},
                **{k: bytes.fromhex(d[k]) for k in ("state_hash", "history_hash", "action_set_hash")}, "actions": actions,
            })
            if profile in ("B", "D"):
                rows["inputs/snapshots"].append({**key, "snapshot": encode(d["observation"])})
            rows["restricted/execution"].append({**key, "physical_ids": d["execution"]})
            rows["targets/policy"].append({
                **key, **d, **axis, "trainable": d["choice"] is not None and d["legal_status"] == "complete",
            })
            rows["targets/attack"].append({**key, **axis, **d["attack"]})
            rows["targets/defense"].append({**key, **axis, "candidates": d["defense"]})
            if d.get("local_actions") is not None:
                rows["targets/local_actions"].append({**key, **axis, **local_actions_arrow(d["local_actions"])})
            rows["targets/belief_lite"].append({
                **key, "opponents": d["opponents"],
                "next_opponent_action": [opponent_action(a) for a in d["next_opponent_action"]],
            })
            rows["targets/belief_full"].append({
                **key, "hands": [belief_full(h) for h in d["opponents"]],
            })


def verified_rows(path, descriptor, build):
    with path.open("rb") as stream:
        header = json.loads(next(stream))
        if header != {"type": "header", "build": build, "offset": descriptor["offset"],
                      "requested": descriptor["footer"]["rows"]}:
            raise ValueError("shard header mismatch")
        digest, count, statuses = hashlib.sha256(), 0, Counter()
        for line in stream:
            row = json.loads(line)
            if row["type"] == "footer":
                if (row != descriptor["footer"] or digest.hexdigest() != row["digest"]
                        or count != row["rows"] or statuses != row["statuses"] or stream.read(1)):
                    raise ValueError("shard checksum or count mismatch")
                return
            digest.update(line)
            count += 1
            statuses[row["status"]] += 1
            yield row
    raise ValueError("missing shard footer")


def run(spools, output, profile, interval, zstd, row_group, *, remove_verified_jsonl=False):
    if output.exists():
        raise FileExistsError(f"choose a new output path: {output}")
    if profile in ("C", "D"):
        for spool in spools:
            if not json.loads((spool / "manifest.json").read_text())["build"]["checkpoints"]:
                raise ValueError("this profile requires checkpoints in every Rust spool")
    output.parent.mkdir(parents=True, exist_ok=True)
    seen, action_registry, ranges, builds, totals = set(), {}, [], [], Counter()
    verified_sources = []
    with tempfile.TemporaryDirectory(prefix=".riyu-pack-", dir=output.parent) as work:
        staging = Path(work) / "release"
        staging.mkdir(mode=0o700)
        previous = None
        for spool in spools:
            manifest = json.loads((spool / "manifest.json").read_text())
            build = manifest["build"]
            plan = [(i, min(build["shard_size"], build["offset"] + build["limit"] - i))
                    for i in range(build["offset"], build["offset"] + build["limit"], build["shard_size"])]
            if [(s["offset"], s["footer"]["rows"]) for s in manifest["shards"]] != plan:
                raise ValueError("manifest does not cover its declared input range")
            common = {k: v for k, v in build.items() if k not in ("offset", "limit", "shard_size")}
            if previous is not None and common != previous:
                raise ValueError("incompatible source/configuration across spools")
            previous = common
            for descriptor in manifest["shards"]:
                offset = descriptor["offset"]
                end = offset + descriptor["footer"]["rows"]
                if any(offset < b and a < end for a, b in ranges):
                    raise ValueError("overlapping shard ranges")
                ranges.append((offset, end))
                folder = defaultdict(list)
                spool_file = spool / descriptor["file"]
                if remove_verified_jsonl and (
                    spool_file.is_symlink() or descriptor["file"] !=
                    f"shard-{offset:08}-{descriptor['footer']['rows']:04}.jsonl"
                ):
                    raise ValueError("unexpected JSONL filename for cleanup")
                for row in verified_rows(spool_file, descriptor, build):
                    convert_row(row, folder, seen, action_registry, profile, interval)
                if remove_verified_jsonl:
                    verified_sources.append((spool_file, spool_file.stat()))
                for table, rows in folder.items():
                    if not rows:
                        continue
                    domain = staging / table
                    domain.mkdir(mode=0o700, parents=True, exist_ok=True)
                    path = domain / f"part-{offset:08}.parquet"
                    t = perf_counter()
                    schema = SCHEMAS[table].with_metadata({
                        b"riyu.contract": b"riyu-data-lake-pilot/1",
                        b"riyu.role": table.split("/")[0].encode(),
                        b"riyu.origin": table_origin(table),
                    })
                    pq.write_table(pa.Table.from_pylist(rows, schema), path, compression="zstd",
                                   compression_level=zstd, row_group_size=row_group)
                    if remove_verified_jsonl and pq.ParquetFile(path).metadata.num_rows != len(rows):
                        raise ValueError("Parquet readback failed before cleanup")
                    totals[(table, "seconds")] += perf_counter() - t
                    totals[(table, "bytes")] += path.stat().st_size
                    totals[(table, "rows")] += len(rows)
            builds.append(build)
        stats = defaultdict(dict)
        for (table, metric), value in totals.items():
            stats[table][metric] = value
        (staging / "manifest.json").write_text(json.dumps({
            "contract": "riyu-data-lake-pilot/1", "sources": builds, "ranges": sorted(ranges),
            "profile": profile, "checkpoint_interval": interval, "zstd": zstd,
            "row_group_size": row_group, "stats": stats, "redistribution_allowed": False,
        }, indent=2))
        staging.rename(output)
    if remove_verified_jsonl:
        for path, stat in verified_sources:
            if path.is_symlink() or path.stat() != stat:
                raise ValueError("verified JSONL changed before cleanup")
        # The committed release and original SQLite remain available. Keep
        # the input manifest as a record of the released range and checksums.
        for path, _ in verified_sources:
            path.unlink()


def table_origin(table):
    return CONTRACT[table]["origin"].encode()


def run_ipc(spools, output, profile, interval, zstd, row_group):
    """Read committed IPC batches, bounding buffered rows to one row group."""
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    ranges, seen, sources = [], set(), []
    stats = defaultdict(Counter)
    with tempfile.TemporaryDirectory(prefix=".riyu-ipc-", dir=output.parent) as work:
        staging = Path(work) / "release"
        staging.mkdir(mode=0o700)
        previous = None
        for root in spools:
            manifest = json.loads((root / "manifest.json").read_text())
            build = manifest["build"]
            if (build.get("format"), build["profile"], build["checkpoint"]) != ("ipc", profile, interval):
                raise ValueError("IPC was written with another profile or checkpoint interval")
            common = {k: v for k, v in build.items() if k not in ("offset", "limit", "shard_size")}
            if previous is not None and common != previous:
                raise ValueError("incompatible source/configuration across IPC spools")
            previous = common
            sources.append(build)
            plan = [(i, min(build["shard_size"], build["offset"] + build["limit"] - i))
                    for i in range(build["offset"], build["offset"] + build["limit"], build["shard_size"])]
            if [(s["offset"], s["footer"]["rows"]) for s in manifest["shards"]] != plan:
                raise ValueError("IPC manifest range is incomplete")
            for shard in manifest["shards"]:
                offset = shard["offset"]
                footer = shard["footer"]
                end = offset + footer["rows"]
                if any(offset < b and a < end for a, b in ranges) or seen.intersection(footer["ids"]):
                    raise ValueError("IPC range or match overlap")
                ranges.append((offset, end))
                seen.update(footer["ids"])
                for table, info in footer["files"].items():
                    if table not in SCHEMAS:
                        raise ValueError("unknown IPC table")
                    incoming = root / shard["file"] / (table + ".ipc")
                    with incoming.open("rb") as stream:
                        if hashlib.file_digest(stream, "sha256").hexdigest() != info["sha256"]:
                            raise ValueError(f"invalid IPC checksum for {table}")
                    destination = staging / table / f"part-{offset:08}.parquet"
                    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                    count = 0
                    t = perf_counter()
                    with pa.memory_map(str(incoming)) as memory:
                        reader = pa.ipc.open_file(memory)
                        if not reader.schema.equals(SCHEMAS[table]):
                            raise ValueError(f"IPC schema mismatch for {table}")
                        with pq.ParquetWriter(destination, reader.schema, compression="zstd",
                                              compression_level=zstd) as writer:
                            group, pending = [], 0
                            for index in range(reader.num_record_batches):
                                batch = reader.get_batch(index)
                                count += batch.num_rows
                                while batch.num_rows:
                                    take = min(row_group - pending, batch.num_rows)
                                    group.append(batch.slice(0, take))
                                    pending += take
                                    batch = batch.slice(take)
                                    if pending == row_group:
                                        writer.write_table(pa.Table.from_batches(group), row_group_size=row_group)
                                        group, pending = [], 0
                            if group:
                                writer.write_table(pa.Table.from_batches(group), row_group_size=row_group)
                    if count != info["rows"]:
                        raise ValueError(f"IPC row count mismatch for {table}")
                    stats[table].update(rows=count, bytes=destination.stat().st_size, seconds=perf_counter() - t)
        (staging / "manifest.json").write_text(json.dumps({
            "contract": "riyu-data-lake-pilot/1", "sources": sources, "ranges": sorted(ranges),
            "profile": profile, "checkpoint_interval": interval, "zstd": zstd,
            "row_group_size": row_group, "stats": stats, "redistribution_allowed": False,
        }, indent=2))
        staging.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("spool", nargs="+", type=Path)
    parser.add_argument("--profile", choices=list("ABCD"), default="D")
    parser.add_argument("--checkpoint-interval", type=int, choices=[16, 32, 64], default=32)
    parser.add_argument("--zstd", type=int, choices=[1, 3, 6], default=3)
    parser.add_argument("--row-group", type=int, default=8192)
    parser.add_argument("--from-ipc", action="store_true")
    parser.add_argument("--remove-verified-jsonl", action="store_true",
                        help="remove the checked JSONL shards after the final release is published")
    args = parser.parse_args()
    if args.row_group < 1:
        parser.error("--row-group must be positive")
    os.umask(0o077)
    if args.from_ipc:
        if args.remove_verified_jsonl:
            parser.error("JSONL cleanup does not apply to IPC input")
        run_ipc(args.spool, args.output, args.profile, args.checkpoint_interval, args.zstd, args.row_group)
    else:
        run(args.spool, args.output, args.profile, args.checkpoint_interval, args.zstd, args.row_group,
            remove_verified_jsonl=args.remove_verified_jsonl)


if __name__ == "__main__":
    main()
