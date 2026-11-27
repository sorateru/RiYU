"""Model-independent observations, ragged action axes and projected teachers."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from mahjong_ai import ObservationReplayer
from .lake_loader import parquet_table, release_shards
from .lake_view import pilot_rules


TARGET_KEYS = {
    **{n: ("decision_id", "action_set_hash") for n in ("policy", "attack", "defense", "local_actions")},
    **{n: ("decision_id",) for n in ("belief_lite", "belief_full")},
    "round": ("match_id", "round"),
    "match": ("match_id",),
    "wins": ("match_id", "round"),
}
SPARSE_TARGETS = frozenset({"local_actions", "wins"})


@dataclass(frozen=True)
class ObservationBatch:
    snapshots: tuple[dict, ...]
    action_codes: tuple[bytes, ...]
    action_ids: tuple[bytes, ...]
    action_offsets: tuple[int, ...]
    history: tuple[tuple[dict, ...], ...] | None


@dataclass(frozen=True)
class LakeBatch:
    inputs: ObservationBatch
    targets: Mapping[str, tuple]

    def encode(self, encoder: Callable[[ObservationBatch], object]) -> object:
        return encoder(self.inputs)


def _sha(data):
    return hashlib.sha256(data).digest()


def _key(row, columns):
    return tuple(row[c] for c in columns)


def _snapshot_map(root, shard, decisions, *, checkpoint, events=None):
    stream_key = ("match_id", "round", "seat")
    if events is None:
        events = _streams(parquet_table(root, shard, "inputs/events").read().to_pylist(), "event")
    checkpoints = {}
    if checkpoint:
        file = parquet_table(root, shard, "inputs/checkpoints")
        checkpoints = _streams(file.read().to_pylist(), "state")
    groups = defaultdict(list)
    for d in decisions:
        groups[(d["match_id"], d["round"], d["actor"])].append(d)
    result = {}
    for key, rows in groups.items():
        stream = events[key]
        first = min(d["observation_seq"] for d in rows)
        prior = [seq for seq in checkpoints.get(key, ()) if seq <= first]
        anchor = max(prior, default=-1)
        if anchor < 0:
            replay = ObservationReplayer(key[2], json.dumps(pilot_rules(rows[0]["rule_code"])))
        else:
            replay = ObservationReplayer.from_state_json(checkpoints[key][anchor].decode())
        cursor = anchor + 1
        for d in sorted(rows, key=lambda d: d["observation_seq"]):
            sequence = d["observation_seq"]
            while cursor <= sequence:
                event = json.loads(stream[cursor])
                if event["sequence"] != cursor:
                    raise ValueError(f"bad projected event sequence at {cursor}")
                event["sequence"] = replay.next_event_sequence()
                replay.apply_event_json(json.dumps(event, separators=(",", ":")))
                cursor += 1
            payload = replay.lake_snapshot_json().encode()
            if _sha(payload) != d["state_hash"]:
                raise ValueError("replayed snapshot does not match state hash")
            result[d["decision_id"]] = payload
    return result


def _streams(rows, payload):
    streams = defaultdict(dict)
    for row in rows:
        key = _key(row, ("match_id", "round", "seat"))
        seq = row["seq"]
        if seq in streams[key]:
            raise ValueError("duplicate projected sequence")
        streams[key][seq] = row[payload]
    return streams


class LakeTrainingLoader:
    """Read one shard at a time. Encoders receive only an ObservationBatch.

    Source may be native snapshots, projected replay, checkpoint replay, or an
    input-only snapshot cache derived from this exact release. Target columns
    are opt-in and retain the root action indices and nested successor axes.
    """

    def __init__(self, root: Path, split="train", *, source="auto", snapshot_cache=None, max_shards=None):
        if split not in ("train", "validation", "final_test"):
            raise ValueError("invalid split")
        if source not in ("auto", "snapshots", "replay", "checkpoints", "derived"):
            raise ValueError("invalid observation source")
        self.root, self.split, self.source = Path(root), split, source
        self.shards = release_shards(self.root, require_snapshots=False)[:max_shards]
        self.direct_snapshots = set(release_shards(self.root))
        self.snapshot_cache = Path(snapshot_cache) if snapshot_cache is not None else None
        self.cache_scope = {
            "format": "riyu-derived-snapshots/1",
            "release": _sha((self.root / "manifest.json").read_bytes()).hex(),
        }
        if source == "derived" and (
            self.snapshot_cache is None
            or json.loads((self.snapshot_cache / "manifest.json").read_text()) != self.cache_scope
        ):
            raise ValueError("derived snapshots belong to a different release or format")

    def _decisions(self, shard, *, all_splits=False):
        decisions = parquet_table(self.root, shard, "inputs/decisions").read().to_pylist()
        if all_splits:
            return decisions
        matches = parquet_table(self.root, shard, "governance/split").read().to_pylist()
        splits = {row["match_id"]: row["split"] for row in matches}
        if len(splits) != len(matches):
            raise ValueError("duplicate split assignment in shard")
        return [d for d in decisions if splits[d["match_id"]] == self.split]

    def _snapshots(self, shard, decisions, events):
        mode = self.source
        if mode == "auto":
            mode = "snapshots" if shard in self.direct_snapshots else "replay"
        if mode in ("snapshots", "derived"):
            if mode == "snapshots":
                file = parquet_table(self.root, shard, "inputs/snapshots")
            else:
                file = pq.ParquetFile(self.snapshot_cache / (shard[0] + ".parquet"))
            data = file.read(columns=["decision_id", "snapshot"]).to_pylist()
            snapshots = {r["decision_id"]: r["snapshot"] for r in data}
            if len(snapshots) != len(data):
                raise ValueError("duplicate snapshot key")
            for d in decisions:
                if _sha(snapshots[d["decision_id"]]) != d["state_hash"]:
                    raise ValueError("snapshot does not match state hash")
            return snapshots
        return _snapshot_map(self.root, shard, decisions, checkpoint=mode == "checkpoints", events=events)

    def write_snapshot_cache(self, path: Path):
        """Materialize input snapshots from C once, aside from the authoritative release."""
        path = Path(path)
        path.mkdir(mode=0o700)
        fields = pa.schema([("decision_id", pa.binary(16)), ("snapshot", pa.binary())], metadata={
            b"riyu.role": b"inputs", b"riyu.contract": b"riyu-data-lake-pilot/1",
        })
        for shard in self.shards:
            decisions = self._decisions(shard, all_splits=True)
            snapshots = _snapshot_map(self.root, shard, decisions, checkpoint=True)
            rows = [{"decision_id": d["decision_id"], "snapshot": snapshots[d["decision_id"]]} for d in decisions]
            pq.write_table(pa.Table.from_pylist(rows, schema=fields), path / (shard[0] + ".parquet"),
                           compression="zstd", compression_level=3)
        (path / "manifest.json").write_text(json.dumps(self.cache_scope, sort_keys=True))

    def _targets(self, shard, name, columns):
        keys = TARGET_KEYS[name]
        try:
            f = parquet_table(self.root, shard, f"targets/{name}")
        except FileNotFoundError:
            if name in SPARSE_TARGETS:
                return {}
            raise
        if columns is not None:
            columns = list(dict.fromkeys((*keys, *columns)))
            unknown = set(columns) - set(f.schema_arrow.names)
            if unknown:
                raise ValueError(f"unknown columns for {name}: {sorted(unknown)}")
        rows = f.read(columns=columns).to_pylist()
        if name == "wins":
            by_key = defaultdict(list)
            for row in rows:
                by_key[_key(row, keys)].append(row)
            return by_key
        by_key = {_key(row, keys): row for row in rows}
        if len(by_key) != len(rows):
            raise ValueError(f"duplicated target keys in {name}")
        return by_key

    def batches(self, *, size=256, history=False,
                targets: Mapping[str, Sequence[str] | None] | None = None) -> Iterator[LakeBatch]:
        targets = targets or {}
        if size < 1 or targets.keys() - TARGET_KEYS.keys():
            raise ValueError("invalid batch size or target")
        for shard in self.shards:
            decisions = self._decisions(shard)
            if not decisions:
                continue
            streams = None
            if history:
                streams = _streams(parquet_table(self.root, shard, "inputs/events").read().to_pylist(), "event")
            snapshots = self._snapshots(shard, decisions, streams)
            projected = {n: self._targets(shard, n, c) for n, c in targets.items()}
            for first in range(0, len(decisions), size):
                rows = decisions[first:first + size]
                histories = [] if history else None
                offsets, codes, ids = [0], [], []
                for d in rows:
                    action_ids = [a["id"] for a in d["actions"]]
                    if len(action_ids) != len(set(action_ids)):
                        raise ValueError("duplicated candidate ID")
                    ids.extend(action_ids)
                    codes.extend(a["semantic"] for a in d["actions"])
                    offsets.append(len(ids))
                    if histories is not None:
                        stream = streams[(d["match_id"], d["round"], d["actor"])]
                        prefix = tuple(json.loads(stream[i]) for i in range(d["observation_seq"] + 1))
                        if any(event["sequence"] != i for i, event in enumerate(prefix)):
                            raise ValueError("observation history is out of sequence")
                        histories.append(prefix)
                inputs = ObservationBatch(
                    tuple(json.loads(snapshots[d["decision_id"]]) for d in rows),
                    tuple(codes), tuple(ids), tuple(offsets),
                    tuple(histories) if history else None,
                )
                teachers = {}
                for name, index in projected.items():
                    keys = [_key(d, TARGET_KEYS[name]) for d in rows]
                    if name == "wins":
                        teachers[name] = tuple(tuple(index.get(key, ())) for key in keys)
                    elif name == "local_actions":
                        teachers[name] = tuple(index.get(key) for key in keys)
                    else:
                        try:
                            teachers[name] = tuple(index[key] for key in keys)
                        except KeyError as error:
                            raise ValueError(f"missing or misaligned {name} target") from error
                yield LakeBatch(inputs, teachers)
