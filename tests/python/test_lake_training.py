import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from mahjong_ai import ObservationReplayer

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from mahjong_ai.data.lake_loader import model_features
from mahjong_ai.data.lake_training import LakeTrainingLoader, ObservationBatch
from mahjong_ai.data.lake_view import pilot_rules


def write(root, name, rows):
    path = root / name / "part-00000000.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    schema_metadata = {
        b"riyu.contract": b"riyu-data-lake-pilot/1", b"riyu.role": name.split("/")[0].encode(),
    }
    pq.write_table(pa.Table.from_pylist(rows).replace_schema_metadata(schema_metadata), path)


def build_release(root: Path):
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps({"ranges": [[0, 2]]}))
    view = {"decision_id": b"d" * 16, "match_id": b"m" * 16, "round": 0, "actor": 0,
            "rule_code": 169, "action_set_hash": b"a" * 32,
            "actions": [{"id": b"q" * 8, "semantic": b"\x01\x00\x00\x00"},
                        {"id": b"r" * 8, "semantic": b"\x01\x00\x04\x01"}]}
    def tile(kind):
        return {"kind": kind, "red": False}

    events = [
        {"sequence": 0, "type": "round_start",
         "round": {"wind": "east", "hand_number": 1, "honba": 0, "riichi_sticks": 0, "dealer": 0},
         "scores": [25000] * 4,
         "initial_hand": [tile(f"m{n}") for n in range(1, 10)] + [tile("p1")] * 3 + [tile("s1")],
         "dora_indicator": tile("east"), "remaining_tiles": 70},
        {"sequence": 1, "type": "draw", "player": 0, "tile": tile("m3"), "rinshan": False},
        {"sequence": 2, "type": "discard", "player": 0, "tile": tile("m3"), "tsumogiri": True,
         "tsumogiri_known": True, "riichi_declaration": False},
        {"sequence": 3, "type": "draw", "player": 1, "tile": None, "rinshan": False},
    ]
    r = ObservationReplayer(0, json.dumps(pilot_rules(169)))
    r.apply_event_json(json.dumps(events[0]))
    key = {"match_id": view["match_id"], "round": 0, "seat": 0}
    write(root, "inputs/checkpoints", [{**key, "seq": 0, "state": r.state_json().encode()}])
    r.apply_event_json(json.dumps(events[1]))
    payload = r.lake_snapshot_json().encode()
    view.update(observation_seq=1, state_hash=hashlib.sha256(payload).digest())
    write(root, "inputs/events", [{**key, "seq": e["sequence"], "event": json.dumps(e).encode()} for e in events])
    write(root, "inputs/decisions", [view])
    write(root, "governance/split", [{"match_id": view["match_id"], "split": "train"}])
    return view, payload


def test_derived_and_direct_match_checkpoint_replay_before_future_events(tmp_path):
    root = tmp_path / "release"
    decision, payload = build_release(root)
    loader = LakeTrainingLoader(root, source="checkpoints")
    cache = tmp_path / "snapshots"
    loader.write_snapshot_cache(cache)
    write(root, "inputs/snapshots", [{"decision_id": decision["decision_id"], "snapshot": payload}])
    replayed = next(loader.batches(history=True, size=1))
    for kwargs in ({"source": "replay"}, {"source": "snapshots"},
                   {"source": "derived", "snapshot_cache": cache}):
        assert replayed == next(LakeTrainingLoader(root, **kwargs).batches(history=True, size=1))
    assert replayed.inputs.action_offsets == (0, 2)
    assert replayed.inputs.snapshots[0]["drawn"]["kind"] == "m3"
    assert [e["sequence"] for e in replayed.inputs.history[0]] == [0, 1]
    assert not list(LakeTrainingLoader(root, split="final_test").batches())
    (root / "manifest.json").write_text(json.dumps({"ranges": [[0, 2]], "changed": True}))
    with pytest.raises(ValueError, match="different release"):
        LakeTrainingLoader(root, source="derived", snapshot_cache=cache)


def test_encoder_has_no_access_to_targets_restricted_or_future(tmp_path):
    root = tmp_path / "release"
    decision, _ = build_release(root)
    key = {c: decision[c] for c in ("decision_id", "action_set_hash")}
    request = {"policy": ["choice", "trainable"], "belief_full": ["hands"], "match": ["final_scores"]}
    loader = LakeTrainingLoader(root, source="replay")

    def encode(inputs):
        assert isinstance(inputs, ObservationBatch)
        assert tuple(vars(inputs)) == ("snapshots", "action_codes", "action_ids", "action_offsets", "history")
        assert len(inputs.history[0]) == 2
        return np.stack([model_features(state) for state in inputs.snapshots])

    without_targets = next(loader.batches(history=True))
    variants = []
    for i in (0, 1):
        write(root, "restricted/execution", [{**key, "physical_ids": [30+i]}])
        write(root, "targets/policy", [{**key, "choice": i, "trainable": True, "evidence_ordinal": 100+i}])
        write(root, "targets/belief_full", [{"decision_id": key["decision_id"], "hands": [100+i]}])
        write(root, "targets/match", [{"match_id": decision["match_id"], "final_scores": [i] * 4}])
        variants.append(next(loader.batches(targets=request, history=True)))
    assert variants[0].targets != variants[1].targets
    assert "evidence_ordinal" not in variants[0].targets["policy"][0]
    for batch in variants:
        assert batch.inputs == without_targets.inputs
        np.testing.assert_array_equal(batch.encode(encode), without_targets.encode(encode))


def test_misaligned_projected_action_axis_is_rejected(tmp_path):
    root = tmp_path / "release"
    decision, _ = build_release(root)
    write(root, "targets/policy", [{"decision_id": decision["decision_id"], "action_set_hash": b"z" * 32,
                                    "choice": 1, "trainable": True}])
    loader = LakeTrainingLoader(root, source="checkpoints")
    with pytest.raises(ValueError, match="misaligned policy"):
        next(loader.batches(targets={"policy": ["choice"]}))
