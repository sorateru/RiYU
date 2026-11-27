from pathlib import Path

import pytest

from mahjong_ai.data.lake_view import feature_cache_key, feature_payload_hash, input_batches


def test_chronological_encoder_cache_uses_visible_history():
    args = dict(state_hash=b"s" * 32, action_set_hash=b"a" * 32,
                encoder_version="sequence-test/1", settings={"window": 64})
    a, b = b"x" * 32, b"y" * 32
    assert feature_cache_key(**args, history_hash=a, uses_history=True) != (
        feature_cache_key(**args, history_hash=b, uses_history=True)
    )
    assert feature_cache_key(**args, history_hash=a, uses_history=False) == (
        feature_cache_key(**args, history_hash=b, uses_history=False)
    )
    with pytest.raises(ValueError, match="history"):
        feature_cache_key(**args, history_hash=None, uses_history=True)


def test_input_guard_rejects_targets_by_name(tmp_path: Path):
    pytest.importorskip("pyarrow")
    with pytest.raises(ValueError, match="unknown input table"):
        list(input_batches(tmp_path, "../targets/belief_full"))


def test_input_guard_rejects_symlink_that_escapes(tmp_path: Path):
    pytest.importorskip("pyarrow")
    inputs = tmp_path / "inputs"
    (inputs / "events").mkdir(parents=True)
    external = tmp_path / "hidden.parquet"
    external.touch()
    (inputs / "events" / "part.parquet").symlink_to(external)
    with pytest.raises(ValueError, match="outside"):
        list(input_batches(inputs, "events"))


def test_feature_payload_hash_binds_encoder_key_and_bytes():
    payload = bytes([0, 1])
    one = "01" * 32
    two = "02" * 32
    assert feature_payload_hash(one, payload) != feature_payload_hash(two, payload)
    assert feature_payload_hash(one, payload) != feature_payload_hash(one, bytes([1, 0]))


def test_input_guard_checks_parquet_role(tmp_path: Path):
    arrow = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    table = arrow.table({"hidden": [1]}).replace_schema_metadata({
        b"riyu.role": b"targets", b"riyu.contract": b"riyu-data-lake-pilot/1",
    })
    inputs = tmp_path / "events"
    inputs.mkdir()
    parquet.write_table(table, inputs / "part.parquet")
    with pytest.raises(ValueError, match="supported input contract"):
        list(input_batches(tmp_path, "events"))
