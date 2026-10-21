from __future__ import annotations

import json
import sys

import pytest

from mahjong_ai.data import (
    BUNDLE_FORMAT,
    ReplayBundleError,
    create_replay_bundle,
    parse_replay_bundle,
)
from tools.create_replay_bundle import main as create_bundle_main


def _event_log() -> dict:
    round_start = {
        "sequence": 0,
        "type": "round_start",
        "round": {
            "wind": "east",
            "hand_number": 1,
            "honba": 0,
            "riichi_sticks": 0,
            "dealer": 0,
        },
        "scores": [25_000] * 4,
        "initial_hand": [],
        "dora_indicator": {"kind": "m1", "red": False},
        "remaining_tiles": 70,
    }
    return {
        "format": "mahjong-ai-event-log/v1",
        "log_id": "fixture-round",
        "rules": None,
        "observations": {str(seat): [round_start] for seat in range(4)},
        "result": {"final_scores": [25_000] * 4},
        "provenance": {"source": "fixture", "converter_version": "test-v1"},
    }


def test_replay_bundle_preserves_provenance_without_exposing_share_url() -> None:
    share_url = "https://game.mahjongsoul.com/?paipu=private-fixture"
    raw = create_replay_bundle(
        [_event_log()],
        platform="mahjong_soul",
        acquisition="user_export",
        authorization="user_owned",
        converter_name="fixture-converter",
        converter_version="1.0",
        share_url=share_url,
    )
    assert raw["format"] == BUNDLE_FORMAT
    parsed = parse_replay_bundle(raw)
    summary = parsed.source.public_summary()
    assert summary["platform"] == "mahjong_soul"
    assert str(summary["reference"]).startswith("url-sha256:")
    assert share_url not in str(summary)
    assert parsed.logs[0].log_id == "fixture-round"


@pytest.mark.parametrize(
    "source_update, message",
    [
        ({"platform": "unknown"}, "platform"),
        ({"authorization": "generated"}, "generated authorization"),
        ({"share_url": "http://example.test/replay"}, "HTTPS URL"),
        ({"share_url": "https://user:pass@example.test/replay"}, "without credentials"),
        ({"share_url": "https://example.test/replay"}, "unexpected host"),
        ({"unexpected": True}, "unknown fields"),
    ],
)
def test_replay_bundle_rejects_invalid_source(source_update: dict, message: str) -> None:
    raw = {
        "format": BUNDLE_FORMAT,
        "source": {
            "platform": "mahjong_soul",
            "acquisition": "user_export",
            "authorization": "user_owned",
            "converter": {"name": "fixture", "version": "1"},
            **source_update,
        },
        "logs": [_event_log()],
    }
    with pytest.raises(ReplayBundleError, match=message):
        parse_replay_bundle(raw)


def test_replay_bundle_runs_event_log_hidden_information_validation() -> None:
    document = _event_log()
    document["observations"]["1"].append(
        {
            "sequence": 1,
            "type": "draw",
            "player": 0,
            "tile": {"kind": "m2", "red": False},
        }
    )
    raw = {
        "format": BUNDLE_FORMAT,
        "source": {
            "platform": "mahjong_soul",
            "acquisition": "user_export",
            "authorization": "user_owned",
            "converter": {"name": "fixture", "version": "1"},
        },
        "logs": [document],
    }
    with pytest.raises(ValueError, match="leaks a tile"):
        parse_replay_bundle(raw)


def test_create_replay_bundle_cli_writes_valid_bundle(tmp_path, monkeypatch) -> None:
    source = tmp_path / "round.json"
    output = tmp_path / "bundle.json"
    source.write_text(json.dumps(_event_log()), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "create_replay_bundle.py",
            str(source),
            "--output",
            str(output),
            "--platform",
            "mahjong_soul",
            "--authorization",
            "user_owned",
            "--converter-name",
            "fixture",
            "--converter-version",
            "1",
        ],
    )
    create_bundle_main()
    parsed = parse_replay_bundle(json.loads(output.read_text(encoding="utf-8")))
    assert len(parsed.logs) == 1
    assert parsed.source.platform == "mahjong_soul"
