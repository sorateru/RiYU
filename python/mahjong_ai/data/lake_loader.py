"""Batched RiYU reader with explicit model input and policy target types."""

from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
from time import perf_counter
from typing import Iterator

import numpy as np
import pyarrow.parquet as pq


TILES = tuple(f"{suit}{rank}" for suit in "mps" for rank in range(1, 10)) + (
    "east", "south", "west", "north", "white", "green", "red",
)
TILE_INDEX = {tile: i for i, tile in enumerate(TILES)}
ENCODER = "riyu-observation-counts/1"
INPUT_FIELDS = frozenset({"hand", "drawn", "dora", "scores", "remaining"})
TABLES = frozenset({
    "governance/split", "inputs/decisions", "inputs/snapshots", "inputs/events",
    "inputs/checkpoints", "targets/policy", "targets/attack", "targets/defense",
    "targets/belief_lite", "targets/belief_full", "targets/round", "targets/match",
    "targets/wins", "targets/local_actions",
})


@dataclass(frozen=True)
class ModelInput:
    features: np.ndarray
    legal_action_codes: tuple[tuple[bytes, ...], ...]


@dataclass(frozen=True)
class PolicyTarget:
    chosen_index: np.ndarray


@dataclass(frozen=True)
class TrainingBatch:
    inputs: ModelInput
    targets: PolicyTarget


@dataclass
class LoaderMetrics:
    read_s: float = 0.0
    encoding_s: float = 0.0
    rows_read: int = 0
    emitted: int = 0
    adjacent_same_match: int = 0


def model_features(snapshot: dict) -> np.ndarray:
    """A diagnostic encoder reading only listed observation keys."""
    out = np.zeros(109, dtype=np.float32)
    for tile in snapshot["hand"]:
        out[TILE_INDEX[tile["kind"]]] += 1
        out[108] += bool(tile["red"])
    if (drawn := snapshot["drawn"]) is not None:
        out[34 + TILE_INDEX[drawn["kind"]]] = 1
        out[108] += bool(drawn["red"])
    for dora in snapshot["dora"]:
        out[68 + TILE_INDEX[dora["kind"]]] += 1
    out[102:106] = np.asarray(snapshot["scores"], dtype=np.float32) / 100_000
    out[106] = snapshot["remaining"] / 70
    out[107] = len(snapshot["hand"]) / 14
    return out


def release_shards(root: Path, *, require_snapshots=True) -> list[tuple[str, bool]]:
    manifest = json.loads((root / "manifest.json").read_text())
    table = "snapshots" if require_snapshots else "decisions"
    if manifest.get("build", {}).get("format") == "parquet":
        return [(entry["file"], True) for entry in manifest["shards"]
                if f"inputs/{table}" in entry["footer"]["files"]]
    return [(f"part-{offset:08}", False) for offset, _ in manifest["ranges"]
            if (root / "inputs" / table / f"part-{offset:08}.parquet").is_file()]


def parquet_table(root, shard, table):
    if table not in TABLES:
        raise ValueError("unknown train-loader table")
    name, direct = shard
    if Path(name).name != name:
        raise ValueError("invalid shard name")
    path = root / name / (table + ".parquet") if direct else root / table / (name + ".parquet")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("table path escapes release root")
    f = pq.ParquetFile(path)
    metadata = f.schema_arrow.metadata or {}
    if metadata.get(b"riyu.role") != table.split("/")[0].encode() or (
        metadata.get(b"riyu.contract") != b"riyu-data-lake-pilot/1"
    ):
        raise ValueError(f"wrong Arrow metadata for {table}")
    return f


class PolicyLoader:
    def __init__(self, root: Path, split="train", *, max_shards=None, feature_cache=None):
        if split not in ("train", "validation", "final_test"):
            raise ValueError("invalid split")
        self.root, self.split = root, split
        self.shards = release_shards(root)
        if not self.shards:
            raise ValueError("policy loader needs a snapshot profile (B or D)")
        self.shards = self.shards[:max_shards]
        self.metrics = LoaderMetrics()
        self.feature_cache = feature_cache
        self.cache_scope = None
        if feature_cache is not None:
            self.cache_scope = json.loads((feature_cache / "manifest.json").read_text())
            if self.cache_scope != self.cache_contract():
                raise ValueError("tensor cache belongs to a different view, split or encoder")
        self._matches = {}
        for shard in release_shards(root):
            for row in parquet_table(root, shard, "governance/split").read().to_pylist():
                if row["match_id"] in self._matches:
                    raise ValueError("duplicate split assignment")
                self._matches[row["match_id"]] = row["split"]

    def cache_contract(self):
        return {
            "encoder": ENCODER, "split": self.split,
            "release": hashlib.sha256((self.root / "manifest.json").read_bytes()).hexdigest(),
            "shards": [name for name, _ in self.shards],
        }

    def _units(self, strategy, rng):
        shards = list(self.shards)
        rng.shuffle(shards)
        if strategy == "shard":
            return [(s, None) for s in shards]
        units = []
        for shard in shards:
            f = parquet_table(self.root, shard, "inputs/decisions")
            units.extend((shard, group) for group in range(f.num_row_groups))
        rng.shuffle(units)
        return units

    def _read_unit(self, unit, *, snapshots=True):
        shard, group = unit
        t = perf_counter()
        entries = {
            "inputs/decisions": ["decision_id", "match_id", "actions"],
            "targets/policy": ["decision_id", "choice", "trainable"],
        }
        if snapshots:
            entries["inputs/snapshots"] = ["decision_id", "snapshot"]
        tables = []
        for name, fields in entries.items():
            p = parquet_table(self.root, shard, name)
            a = p.read(columns=fields) if group is None else p.read_row_group(group, columns=fields)
            tables.append(a)
        ids = [t["decision_id"].combine_chunks() for t in tables]
        if not all(ids[0].equals(other) for other in ids[1:]):
            raise ValueError("unaligned decision, snapshot or target rows")
        read_s = perf_counter() - t
        decision, policy = [t.to_pylist() for t in tables[:2]]
        payloads = tables[2]["snapshot"].to_pylist() if snapshots else [None] * len(decision)
        records = []
        for d, s, p in zip(decision, payloads, policy, strict=True):
            match_id = d["match_id"]
            if match_id not in self._matches:
                raise ValueError("decision has no split")
            if not p["trainable"] or self._matches[match_id] != self.split:
                continue
            chosen = p["choice"]
            if chosen is None or not 0 <= chosen < len(d["actions"]):
                raise ValueError("trainable policy target is outside legal axis")
            codes = tuple(action["semantic"] for action in d["actions"])
            records.append((match_id, d["decision_id"], s, codes, chosen))
        return records, read_s, len(decision)

    def _collections(self, strategy, rng, buffer_rows, readers):
        units = self._units(strategy, rng)
        if strategy in ("shard", "rowgroup"):
            # Bound outstanding reads to one wave; Executor.map on all
            # row groups would otherwise retain outputs ahead of the consumer.
            if strategy == "shard":
                readers = 1
            with ThreadPoolExecutor(max_workers=readers) as pool:
                for i in range(0, len(units), readers):
                    for rows, seconds, read in pool.map(self._read_unit, units[i:i+readers]):
                        self.metrics.read_s += seconds
                        self.metrics.rows_read += read
                        rng.shuffle(rows)
                        yield rows
            return
        buffer = []
        with ThreadPoolExecutor(max_workers=readers) as pool:
            for i in range(0, len(units), readers):
                wave = []
                for records, seconds, read in pool.map(self._read_unit, units[i:i+readers]):
                    self.metrics.read_s += seconds
                    self.metrics.rows_read += read
                    if records:
                        wave.append(deque(records))
                while wave:
                    wave = [x for x in wave if x]
                    for records in wave:
                        buffer.append(records.popleft())
                    if len(buffer) >= buffer_rows:
                        rng.shuffle(buffer)
                        keep = len(buffer) // 2
                        yield buffer[keep:]
                        buffer = buffer[:keep]
        rng.shuffle(buffer)
        yield buffer

    def create_feature_cache(self, path: Path):
        path.mkdir(mode=0o700)
        for shard in self.shards:
            records, _, _ = self._read_unit((shard, None))
            shape = (len(records), 109)
            features = np.empty(shape, dtype=np.float32)
            ids = np.empty(len(records), dtype="V16")
            for i, (_, decision_id, snapshot, _, _) in enumerate(records):
                features[i] = model_features(json.loads(snapshot))
                ids[i] = decision_id
            np.save(path / (shard[0] + ".npy"), features, allow_pickle=False)
            np.save(path / (shard[0] + ".ids.npy"), ids, allow_pickle=False)
        (path / "manifest.json").write_text(json.dumps(self.cache_contract()))

    def batches(self, *, strategy="buffer", size=256, seed=7, buffer_rows=4096, readers=4) -> Iterator[TrainingBatch]:
        if strategy not in ("shard", "rowgroup", "buffer", "cache") or min(size, readers) < 1 or buffer_rows < size:
            raise ValueError("invalid batch settings")
        if strategy == "cache" and self.feature_cache is None:
            raise ValueError("cache strategy requires a feature cache")
        self.metrics = LoaderMetrics()
        rng = random.Random(seed)
        pending, last_match = [], None
        if strategy == "cache":
            cache_units = self._units("shard", rng)

            def cached():
                for shard, _ in cache_units:
                    rows, read_s, read = self._read_unit((shard, None), snapshots=False)
                    t = perf_counter()
                    arrays = np.load(self.feature_cache / (shard[0] + ".npy"), mmap_mode="r", allow_pickle=False)
                    ids = np.load(self.feature_cache / (shard[0] + ".ids.npy"), allow_pickle=False)
                    if len(rows) != len(arrays) or arrays.shape[1:] != (109,) or arrays.dtype != np.float32:
                        raise ValueError("invalid cache shape")
                    if [bytes(v) for v in ids] != [r[1] for r in rows]:
                        raise ValueError("cache ID order mismatch")
                    self.metrics.read_s += read_s + perf_counter() - t
                    self.metrics.rows_read += read
                    rows = [(r[0], r[1], arrays[i], r[3], r[4]) for i, r in enumerate(rows)]
                    rng.shuffle(rows)
                    yield rows
            collections = cached()
        else:
            collections = self._collections(strategy, rng, buffer_rows, readers)
        for collection in collections:
            for record in collection:
                match_id = record[0]
                self.metrics.adjacent_same_match += match_id == last_match
                last_match = match_id
                pending.append(record)
                if len(pending) == size:
                    yield self._encode(pending)
                    pending = []
        if pending:
            yield self._encode(pending)

    def _encode(self, records):
        t = perf_counter()
        features = np.stack([value if isinstance(value, np.ndarray) else model_features(json.loads(value))
                             for _, _, value, _, _ in records])
        inputs = ModelInput(features, tuple(r[3] for r in records))
        targets = PolicyTarget(np.array([r[4] for r in records], dtype=np.int64))
        self.metrics.encoding_s += perf_counter() - t
        self.metrics.emitted += len(records)
        return TrainingBatch(inputs, targets)
