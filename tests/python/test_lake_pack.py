import hashlib
import json

import pytest

pytest.importorskip("pyarrow")

from tools.pack_riyu_lake import run


def test_cleanup_only_removes_verified_intermediate(tmp_path):
    root = tmp_path / "spool"
    root.mkdir()
    source = root / "unrelated.db"
    source.write_bytes(b"stand-in for a raw archive")
    build = {"offset": 0, "limit": 1, "shard_size": 1, "checkpoints": True}
    header = {"type": "header", "build": build, "offset": 0, "requested": 1}
    # Synthetic failure rows have only a locator digest and an error status.
    payload = (json.dumps({
        "type": "match", "status": "normalize_rejected", "locator_hash": "00"*32,
    }) + "\n").encode()
    footer = {"type": "footer", "rows": 1, "statuses": {"normalize_rejected": 1},
              "digest": hashlib.sha256(payload).hexdigest()}
    shard = root / "shard-00000000-0001.jsonl"
    content = (json.dumps(header) + "\n").encode() + payload + (json.dumps(footer) + "\n").encode()
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"build": build, "shards": [
        {"file": shard.name, "footer": footer, "offset": 0},
    ]}))

    shard.write_bytes(content + b"unverified tail")
    with pytest.raises(ValueError, match="checksum or count"):
        run([root], tmp_path / "failed", "D", 32, 3, 8192, remove_verified_jsonl=True)
    assert shard.exists() and not (tmp_path / "failed").exists()

    shard.write_bytes(content)
    release = tmp_path / "release"
    run([root], release, "D", 32, 3, 8192, remove_verified_jsonl=True)
    assert not shard.exists()
    assert source.exists() and manifest.exists()
    assert (release / "restricted/matches/part-00000000.parquet").exists()
