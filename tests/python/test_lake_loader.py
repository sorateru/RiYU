from collections import Counter
import json

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from mahjong_ai.data.lake_loader import ModelInput, PolicyLoader, PolicyTarget, model_features


def write(root, table, rows):
    path = root / table / "part-00000000.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {b"riyu.contract": b"riyu-data-lake-pilot/1", b"riyu.role": table.split("/")[0].encode()}
    pq.write_table(pa.Table.from_pylist(rows).replace_schema_metadata(metadata), path, row_group_size=2)


def snapshot(remaining=40):
    return {"hand": [{"kind": "m1", "red": False}], "drawn": None, "dora": [],
            "scores": [25000] * 4, "remaining": remaining}


def test_model_features_ignore_private_and_future_fields():
    visible = snapshot()
    hidden = {"opponent_hands": [0, 2], "seed": "not visible", "next_action": 1,
              "final_scores": [10000, 40000, 30000, 20000], "rank": 4}
    first = model_features({**visible, **hidden})
    second = model_features({**visible, **hidden, "opponent_hands": [8, 9], "rank": 1})
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, model_features(snapshot(remaining=32)))


def test_training_loader_split_order_and_cache(tmp_path):
    root = tmp_path / "release"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps({"ranges": [[0, 2]]}))
    train, test = b"a" * 16, b"b" * 16
    ids = [i.to_bytes(16, "little") for i in range(5)]
    write(root, "governance/split", [
        {"match_id": train, "split": "train"}, {"match_id": test, "split": "final_test"},
    ])
    write(root, "inputs/decisions", [
        {"decision_id": decision_id, "match_id": train if i < 4 else test,
         "actions": [{"semantic": b"\x01\x00"}, {"semantic": b"\x01\x01"}]}
        for i, decision_id in enumerate(ids)
    ])
    write(root, "inputs/snapshots", [
        {"decision_id": decision_id, "snapshot": json.dumps(snapshot(40+i)).encode()}
        for i, decision_id in enumerate(ids)
    ])
    write(root, "targets/policy", [
        {"decision_id": decision_id, "choice": (i % 2) if i != 3 else None, "trainable": i != 3}
        for i, decision_id in enumerate(ids)
    ])
    cache = tmp_path / "cache"
    PolicyLoader(root).create_feature_cache(cache)
    loader = PolicyLoader(root, feature_cache=cache)
    for strategy in ("shard", "rowgroup", "buffer", "cache"):
        batches = list(loader.batches(strategy=strategy, size=2))
        assert all(isinstance(b.inputs, ModelInput) and isinstance(b.targets, PolicyTarget) for b in batches)
        assert sum(len(b.inputs.features) for b in batches) == 3
        assert Counter(i for b in batches for i in b.targets.chosen_index) == {0: 2, 1: 1}
        assert sorted(round(70*value) for b in batches for value in b.inputs.features[:, 106]) == [40, 41, 42]
    with pytest.raises(ValueError, match="split"):
        PolicyLoader(root, split="final_test", feature_cache=cache)
