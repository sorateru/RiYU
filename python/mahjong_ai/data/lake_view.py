"""Model-side opening of RiYU's safe Parquet view."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


_INPUT_TABLES = frozenset({"decisions", "snapshots", "events", "checkpoints"})


def _files(input_root, table):
    import pyarrow.parquet as pq

    if table not in _INPUT_TABLES:
        raise ValueError(f"unknown input table {table!r}")
    root = input_root.resolve(strict=True)
    manifest_path = root / "manifest.json"
    paths = sorted((root / table).glob("*.parquet"))
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("build", {}).get("format") == "parquet":
            paths = [
                root / entry["file"] / "inputs" / (table + ".parquet")
                for entry in manifest["shards"] if f"inputs/{table}" in entry["footer"]["files"]
            ]
    for path in paths:
        if not path.resolve().is_relative_to(root):
            raise ValueError("input path resolves outside the mounted input tree")
        file = pq.ParquetFile(path)
        metadata = file.schema_arrow.metadata or {}
        if (metadata.get(b"riyu.role") != b"inputs"
                or metadata.get(b"riyu.contract") != b"riyu-data-lake-pilot/1"):
            raise ValueError(f"file is outside the supported input contract: {path.name}")
        yield path, file


def input_batches(input_root: Path, table: str, batch_size: int = 4096):
    """Yield Arrow batches with only the mounted input subtree available."""
    for _, file in _files(input_root, table):
        yield from file.iter_batches(batch_size=batch_size)


def feature_cache_key(*, state_hash: bytes, history_hash: bytes | None, action_set_hash: bytes,
                      encoder_version: str, settings: dict, uses_history: bool) -> str:
    """Return a key for a versioned encoder; chronological encoders require history."""
    if uses_history and history_hash is None:
        raise ValueError("chronological encoder requires an observation history hash")
    for value in (state_hash, action_set_hash, history_hash):
        if value is not None and len(value) != 32:
            raise ValueError("RiYU input hashes must be 32 bytes")
    record = {
        "contract": 1, "state": state_hash.hex(), "actions": action_set_hash.hex(),
        "history": history_hash.hex() if uses_history else None,
        "encoder": encoder_version, "settings": settings,
    }
    return hashlib.sha256(json.dumps(
        record, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def feature_payload_hash(cache_key: str, payload: bytes) -> str:
    """Hash canonical feature bytes under their encoder and input cache key."""
    key = bytes.fromhex(cache_key)
    if len(key) != 32:
        raise ValueError("expected a SHA-256 feature cache key")
    return hashlib.sha256(b"riyu-feature-payload-v1\0" + key + payload).hexdigest()


def restore_observation(input_root: Path, decision: dict, *, use_checkpoints: bool = True) -> dict:
    """Get the exact observation using only files in the input subtree."""
    import pyarrow.parquet as pq
    from mahjong_ai import apply_event_json, replay_events_json

    def read(table, filters):
        paths = [str(path) for path, _ in _files(input_root, table)]
        return pq.read_table(paths, filters=filters).to_pylist() if paths else []

    def verified(snapshot):
        encoded = json.dumps(snapshot, sort_keys=True, separators=(",", ":")).encode()
        if hashlib.sha256(encoded).digest() != decision["state_hash"]:
            raise ValueError("restored observation mismatch; check input and Rust binding versions")
        return snapshot

    materialized = read("snapshots", [("decision_id", "=", decision["decision_id"])])
    if materialized:
        if len(materialized) != 1:
            raise ValueError("duplicate observation snapshot")
        return verified(json.loads(materialized[0]["snapshot"]))
    actor, limit = decision["actor"], decision["observation_seq"]
    keys = [("match_id", "=", decision["match_id"]), ("round", "=", decision["round"]),
            ("seat", "=", actor), ("seq", "<=", limit)]
    checkpoints = read("checkpoints", keys) if use_checkpoints else []
    checkpoint = max(checkpoints, key=lambda c: c["seq"]) if checkpoints else None
    start = checkpoint["seq"] + 1 if checkpoint else 0
    rows = sorted(read("events", keys + [("seq", ">=", start)]), key=lambda e: e["seq"])
    if [e["seq"] for e in rows] != list(range(start, limit + 1)):
        raise ValueError("projected event range is incomplete")
    if checkpoint:
        state_json = checkpoint["state"].decode()
        for sequence, row in enumerate(rows, 1):
            event = json.loads(row["event"])
            event["sequence"] = sequence
            state_json = apply_event_json(state_json, json.dumps(event))
    else:
        rule = decision["rule_code"]
        if rule not in (169, 225):
            raise ValueError("unsupported pilot rule profile")
        rules = {
            "player_count": 4, "starting_score": 25000, "target_score": 30000,
            "red_fives": [1, 1, 1], "open_tanyao": True, "double_wind_pair_fu": True,
            "kiriage_mangan": False, "kazoe_limit": "yakuman", "allow_double_ron": True,
            "triple_ron_is_draw": True, "double_yakuman": False,
            "match_length": "east_south" if rule == 169 else "east_only",
            "allow_kokushi_ankan_chankan": False,
        }
        state_json = replay_events_json(actor, "[" + ",".join(row["event"].decode() for row in rows) + "]",
                                        json.dumps(rules))
    state = json.loads(state_json)
    kinds = {kind: i for i, kind in enumerate(
        [f"{suit}{rank}" for suit in "mps" for rank in range(1, 10)]
        + ["east", "south", "west", "north", "white", "green", "red"]
    )}
    hand = sorted(state["hand"], key=lambda tile: (kinds[tile["kind"]], tile["red"]))
    return verified({
        "seat": actor, "round": state["round"], "scores": state["scores"], "hand": hand,
        "drawn": state["drawn_tile"], "players": state["players"], "dora": state["dora_indicators"],
        "ippatsu": state["ippatsu_eligible"], "phase": state["phase"],
        "remaining": state["remaining_tiles"], "turn": state["turn"], "pending_kan": state["pending_kan"],
    })
