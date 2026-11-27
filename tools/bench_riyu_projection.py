#!/usr/bin/env python3
"""Compare observer-projected streams and typed snapshots on committed shards."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import tempfile
from time import perf_counter

import pyarrow as pa
import pyarrow.parquet as pq

from pack_riyu_lake import encode


TILE = pa.struct([("kind", pa.string()), ("red", pa.bool_())])
MELD = pa.struct([
    ("kind", pa.string()), ("tiles", pa.list_(TILE)), ("called_tile", TILE), ("from", pa.uint8()),
])
RIVER = pa.struct([
    ("tile", TILE), ("tsumogiri", pa.bool_()), ("tsumogiri_known", pa.bool_()),
    ("riichi_declaration", pa.bool_()), ("called_by", pa.uint8()),
])
PLAYER = pa.struct([
    ("discards", pa.list_(RIVER)), ("melds", pa.list_(MELD)), ("riichi", pa.string()),
])
TYPED = pa.schema([
    ("decision_id", pa.binary(16)), ("seat", pa.uint8()),
    ("round", pa.struct([("wind", pa.string()), ("hand_number", pa.uint8()),
                         ("dealer", pa.uint8()), ("honba", pa.uint16()), ("riichi_sticks", pa.uint16())])),
    ("scores", pa.list_(pa.int32())), ("hand", pa.list_(TILE)), ("drawn", TILE),
    ("players", pa.list_(PLAYER)), ("dora", pa.list_(TILE)), ("ippatsu", pa.list_(pa.bool_())),
    ("phase", pa.struct([("phase", pa.string()), ("player", pa.uint8()), ("discarder", pa.uint8()),
                         ("declarer", pa.uint8()), ("tile", TILE), ("ankan", pa.bool_()),
                         ("reason", pa.string())])),
    ("pending_kan", pa.struct([("player", pa.uint8()), ("meld", MELD), ("replace_meld_index", pa.uint8())])),
    ("remaining", pa.uint16()), ("turn", pa.uint16()),
])


def clean(obj, field):
    """The typed variant restores absent tagged-union fields on read."""
    if obj is None:
        return None
    if pa.types.is_struct(field):
        return {f.name: clean(obj[f.name], f.type) for f in field
                if obj[f.name] is not None or f.name in {"called_by", "called_tile", "from", "replace_meld_index"}}
    if pa.types.is_list(field):
        return [clean(v, field.value_type) for v in obj]
    return obj


def stats(out, label, table):
    path = out / (label + ".parquet")
    t = perf_counter()
    pq.write_table(table, path, compression="zstd", compression_level=3, row_group_size=8192)
    writing = perf_counter() - t
    t = perf_counter()
    recovered = pq.ParquetFile(path).read()
    read_s = perf_counter() - t
    return {"bytes": path.stat().st_size, "rows": table.num_rows, "write_s": writing, "read_s": read_s}, recovered


def run(release, limit):
    manifests = json.loads((release / "manifest.json").read_text())["shards"][:limit]
    events = pa.concat_tables([pq.ParquetFile(release / s["file"] / "inputs/events.parquet").read() for s in manifests])
    snapshots = pa.concat_tables([pq.ParquetFile(release / s["file"] / "inputs/snapshots.parquet").read() for s in manifests])
    tally = {}
    with tempfile.TemporaryDirectory(prefix="riyu-view-") as tmp:
        out = Path(tmp)
        tally["observer_events"], _ = stats(out, "observer-events", events)
        tally["json_snapshots"], _ = stats(out, "json-snapshots", snapshots)
        t = perf_counter()
        payloads = {r["decision_id"]: json.loads(r["snapshot"]) for r in snapshots.to_pylist()}
        rows = [{"decision_id": key, **snapshot} for key, snapshot in payloads.items()]
        typed = pa.Table.from_pylist(rows, TYPED)
        tally["typed_snapshots"], recovered = stats(out, "typed-snapshots", typed)
        tally["typed_snapshots"]["build_s"] = perf_counter() - t - tally["typed_snapshots"]["read_s"] - tally["typed_snapshots"]["write_s"]
        for restored in recovered.to_pylist():
            key = restored.pop("decision_id")
            observation = {k: clean(v, TYPED.field(k).type) for k, v in restored.items()}
            assert observation == payloads[key], "typed snapshot lost an observation field"
        t = perf_counter()
        public, private, expected = {}, [], []
        for row in events.to_pylist():
            key = (row["match_id"], row["round"], row["seq"])
            value = json.loads(row["event"])
            expected.append((key, row["seat"], row["event"]))
            if value["type"] == "round_start":
                private.append((*key, row["seat"], encode(value.pop("initial_hand"))))
            elif value["type"] == "draw" and value["tile"] is not None:
                assert value["player"] == row["seat"]
                private.append((*key, row["seat"], encode(value["tile"])))
                value["tile"] = None
            payload = encode(value)
            if public.setdefault(key, payload) != payload:
                raise ValueError("event differs across observers after masking private fields")
        public_rows = [{"match_id": k[0], "round": k[1], "seq": k[2], "event": v} for k, v in public.items()]
        private_rows = [{"match_id": m, "round": r, "seq": q, "seat": s, "delta": d} for m, r, q, s, d in private]
        event_schema = events.schema.remove_metadata()
        tally["projection_build_s"] = perf_counter() - t
        tally["shared_events"], p = stats(out, "shared", pa.Table.from_pylist(public_rows, event_schema.remove(2)))
        tally["private_events"], x = stats(out, "private", pa.Table.from_pylist(private_rows, event_schema.set(4, pa.field("delta", pa.binary()))))
        t = perf_counter()
        observable = {(r["match_id"], r["round"], r["seq"]): r["event"] for r in p.to_pylist()}
        secret = {(r["match_id"], r["round"], r["seq"], r["seat"]): r["delta"] for r in x.to_pylist()}
        for key, seat, expected_event in expected:
            event = json.loads(observable[key])
            if (private_delta := secret.get((*key, seat))) is not None:
                field = "initial_hand" if event["type"] == "round_start" else "tile"
                event[field] = json.loads(private_delta)
            assert encode(event) == expected_event
        tally["projection_recombine_s"] = perf_counter() - t
    return tally


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("release", type=Path)
    p.add_argument("--shards", type=int, default=4)
    a = p.parse_args()
    print(json.dumps(run(a.release, a.shards), indent=2))
