from __future__ import annotations

import json
import sys

import pytest
from fastapi.testclient import TestClient

from mahjong_ai.apps.reviewer import create_app
from mahjong_ai.data import (
    MAHJONG_SOUL_DECODED_FORMAT,
    LocalReplayURLCache,
    MahjongSoulConversionError,
    ReplayURLCacheError,
    ReplayURLNotFound,
    build_decision_records,
    convert_decoded_match,
    create_mahjong_soul_bundle,
    mahjong_soul_replay_id,
    parse_replay_bundle,
)
from tools.convert_mahjong_soul import main as convert_main
from mahjong_ai.schema import action_key


def _hands() -> list[list[str]]:
    hand = ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "1p", "2p", "3p", "7s", "1z", "5z"]
    return [list(hand) for _ in range(4)]


def _start(**updates: object) -> dict:
    data: dict = {
        "chang": 0,
        "ju": 0,
        "ben": 0,
        "liqibang": 0,
        "scores": [25_000, 25_000, 25_000, 25_000],
        "hands": _hands(),
        "dora": "4z",
    }
    data.update(updates)
    return {"name": ".lq.RecordNewRound", "data": data}


def _decoded(*records: dict, final_scores: list[int] | None = None) -> dict:
    return {
        "format": MAHJONG_SOUL_DECODED_FORMAT,
        "records": list(records),
        "final_scores": final_scores or [25_000] * 4,
    }


def test_decoded_match_projects_draws_and_builds_legal_decision_record() -> None:
    raw = _decoded(
        _start(),
        {"name": "RecordDealTile", "data": {"seat": 0, "tile": "9p"}},
        {
            "name": "RecordDiscardTile",
            "data": {"seat": 0, "tile": "9p", "moqie": True},
        },
        {"name": "RecordNoTile", "data": {"delta_scores": [0, 0, 0, 0]}},
    )
    documents = convert_decoded_match(raw)
    assert len(documents) == 1
    own_draw = documents[0]["observations"]["0"][1]
    opponent_draw = documents[0]["observations"]["1"][1]
    assert own_draw["tile"] == {"kind": "p9", "red": False}
    assert opponent_draw["tile"] is None

    bundle = parse_replay_bundle(create_mahjong_soul_bundle(raw))
    records = build_decision_records(bundle.logs[0])
    assert any(json.loads(record.expert_action)["type"] == "discard" for record in records)


def test_alias_records_convert_calls_riichi_kan_dora_and_terminal() -> None:
    raw = _decoded(
        _start(),
        {"type": "draw", "player": 0, "tile": "0m"},
        {
            "type": "discard",
            "player": 0,
            "tile": "5mr",
            "tsumogiri": True,
            "riichi": True,
        },
        {
            "name": "RecordChiPengGang",
            "data": {
                "seat": 1,
                "type": 1,
                "tiles": ["5m", "5m", "5mr"],
                "froms": [1, 1, 0],
            },
        },
        {
            "name": "RecordAnGangAddGang",
            "data": {"seat": 2, "type": 3, "tiles": ["9s", "9s", "9s", "9s"]},
        },
        {"name": "RecordRevealTile", "data": {"tile": "7z"}},
        {
            "name": "RecordHule",
            "data": {
                "hules": [{"seat": 3, "from": 2}],
                "delta_scores": [0, 0, -8000, 8000],
            },
        },
        final_scores=[25_000, 25_000, 17_000, 33_000],
    )
    events = convert_decoded_match(raw)[0]["observations"]["0"]
    assert [event["type"] for event in events] == [
        "round_start",
        "draw",
        "riichi",
        "discard",
        "riichi",
        "pon",
        "kan",
        "dora_reveal",
        "ron",
    ]
    assert events[1]["tile"] == {"kind": "m5", "red": True}
    assert events[5]["meld"]["from"] == 0
    assert events[6]["meld"]["kind"] == "ankan"
    assert events[-1]["score_deltas"] == [0, 0, -8000, 8000]


def test_converter_rejects_partial_hands_and_ambiguous_calls() -> None:
    partial = _start()
    partial["data"].pop("hands")
    partial["data"]["tiles0"] = _hands()[0]
    with pytest.raises(MahjongSoulConversionError, match="all four"):
        convert_decoded_match(_decoded(partial))

    ambiguous = _decoded(
        _start(),
        {"type": "call", "seat": 1, "call_type": "chi", "tiles": ["1m", "2m", "3m"]},
        {"type": "draw_game", "reason": "exhaustive", "score_deltas": [0, 0, 0, 0]},
    )
    with pytest.raises(MahjongSoulConversionError, match="unambiguous source"):
        convert_decoded_match(ambiguous)


def test_converter_cli_writes_bundle_without_raw_share_url(tmp_path, monkeypatch) -> None:
    source = tmp_path / "decoded.json"
    output = tmp_path / "bundle.json"
    share_url = "https://game.mahjongsoul.com/?paipu=private-fixture"
    source.write_text(
        json.dumps(
            _decoded(
                _start(),
                {
                    "type": "draw_game",
                    "reason": "exhaustive",
                    "score_deltas": [0, 0, 0, 0],
                },
            )
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "convert_mahjong_soul.py",
            str(source),
            "--output",
            str(output),
            "--share-url",
            share_url,
        ],
    )
    convert_main()
    saved = json.loads(output.read_text(encoding="utf-8"))
    parsed = parse_replay_bundle(saved)
    assert parsed.source.platform == "mahjong_soul"
    assert share_url not in str(parsed.source.public_summary())


def test_web_api_accepts_decoded_mahjong_soul_directly() -> None:
    raw = _decoded(
        _start(),
        {"type": "draw", "player": 0, "tile": "9p"},
        {"type": "discard", "player": 0, "tile": "9p", "tsumogiri": True},
        {"type": "draw_game", "reason": "exhaustive", "score_deltas": [0, 0, 0, 0]},
    )
    client = TestClient(create_app())
    response = client.post(
        "/api/replay-analysis",
        json={"decoded_mahjong_soul": raw, "observer": 0},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["source"]["platform"] == "mahjong_soul"
    assert body["decisionCount"] == 1
    assert MAHJONG_SOUL_DECODED_FORMAT in client.get("/api/health").json()[
        "replay_import"
    ]["formats"]


@pytest.mark.parametrize(
    "url",
    [
        "http://game.mahjongsoul.com/?paipu=abcdefgh",
        "https://mahjongsoul.com.evil.test/?paipu=abcdefgh",
        "https://user:pass@game.mahjongsoul.com/?paipu=abcdefgh",
        "https://game.mahjongsoul.com:8443/?paipu=abcdefgh",
        "https://game.mahjongsoul.com/?paipu=abc",
        "https://game.mahjongsoul.com/?paipu=abcdefgh&paipu=ijklmnop",
    ],
)
def test_share_url_parser_rejects_unsafe_or_ambiguous_urls(url: str) -> None:
    with pytest.raises(ReplayURLCacheError):
        mahjong_soul_replay_id(url)


def test_registered_share_url_runs_through_web_analysis(tmp_path) -> None:
    url = "https://mahjongsoul.game.yo-star.com/?paipu=260130-fixture-12345678"
    raw = _decoded(
        _start(),
        {"type": "draw", "player": 0, "tile": "9p"},
        {"type": "discard", "player": 0, "tile": "9p", "tsumogiri": True},
        {"type": "draw_game", "reason": "exhaustive", "score_deltas": [0, 0, 0, 0]},
    )
    cache = LocalReplayURLCache(tmp_path)
    stored = cache.register(url, raw)
    assert stored.name != "260130-fixture-12345678.json"
    assert mahjong_soul_replay_id(url) == "260130-fixture-12345678"
    with pytest.raises(ReplayURLCacheError, match="already registered"):
        cache.register(url, raw)

    client = TestClient(create_app(replay_cache_dir=tmp_path))
    response = client.post("/api/replay-analysis", json={"url": url, "observer": 0})
    assert response.status_code == 200
    body = response.json()
    assert body["decisionCount"] == 1
    assert body["source"]["platform"] == "mahjong_soul"
    assert body["source"]["reference"].startswith("url-sha256:")
    assert url not in json.dumps(body)


def test_unregistered_share_url_returns_not_found(tmp_path) -> None:
    url = "https://game.mahjongsoul.com/?paipu=260130-missing-12345678"
    with pytest.raises(ReplayURLNotFound):
        LocalReplayURLCache(tmp_path).resolve(url)
    response = TestClient(create_app(replay_cache_dir=tmp_path)).post(
        "/api/replay-analysis", json={"url": url, "observer": 0}
    )
    assert response.status_code == 404
    assert "ローカル登録" in response.json()["detail"]


def test_extension_import_opens_short_lived_review_url(tmp_path) -> None:
    raw = _decoded(
        _start(),
        {"type": "draw", "player": 0, "tile": "9p"},
        {"type": "discard", "player": 0, "tile": "9p", "tsumogiri": True},
        {"type": "draw_game", "reason": "exhaustive", "score_deltas": [0, 0, 0, 0]},
    )
    client = TestClient(create_app(replay_cache_dir=tmp_path))
    response = client.post(
        "/api/extension-import",
        headers={"origin": "chrome-extension://abcdefghijklmnopabcdefghijklmnop"},
        json={
            "decoded_mahjong_soul": raw,
            "observer": 0,
            "share_url": "https://game.mahjongsoul.com/?paipu=extension-fixture-1234",
        },
    )
    assert response.status_code == 200
    review_url = response.json()["review_url"]
    token = review_url.rsplit("=", 1)[-1]
    imported = client.get(f"/api/extension-import/{token}")
    assert imported.status_code == 200
    assert imported.json()["decisionCount"] == 1
    assert imported.json()["source"]["acquisition"] == "authorized_adapter"
    assert "extension-fixture-1234" not in imported.text


@pytest.mark.parametrize(
    "origin",
    [
        "https://evil.example",
        "chrome-extension://evil",
        "chrome-extension://abcdefghijklmnopabcdefghijklmnop.evil.example",
    ],
)
def test_extension_import_rejects_invalid_origins(tmp_path, origin: str) -> None:
    response = TestClient(create_app(replay_cache_dir=tmp_path)).post(
        "/api/extension-import",
        headers={"origin": origin},
        json={
            "decoded_mahjong_soul": {},
            "observer": 0,
            "share_url": "https://game.mahjongsoul.com/?paipu=extension-fixture-1234",
        },
    )
    assert response.status_code == 403


def test_extension_import_requires_active_official_replay_url(tmp_path) -> None:
    response = TestClient(create_app(replay_cache_dir=tmp_path)).post(
        "/api/extension-import",
        headers={"origin": "chrome-extension://abcdefghijklmnopabcdefghijklmnop"},
        json={
            "decoded_mahjong_soul": _decoded(_start()),
            "observer": 0,
            "share_url": "https://game.mahjongsoul.com/",
        },
    )
    assert response.status_code == 422
    assert "paipu" in response.json()["detail"]


def test_real_shape_accepts_dealer_14_tiles_nested_draw_scores_and_kan_codes() -> None:
    hands = _hands()
    hands[0].append("9p")
    raw = _decoded(
        _start(hands=hands),
        {"name": "RecordDiscardTile", "data": {"seat": 0, "tile": "9p", "moqie": True}},
        {
            "name": "RecordAnGangAddGang",
            "data": {"seat": 1, "type": 3, "tiles": "9s", "doras": ["4z", "7z"]},
        },
        {
            "name": "RecordNoTile",
            "data": {
                "scores": [
                    {"delta_scores": [1500, -1500, 0, 0]},
                    {"delta_scores": [0, 0, 0, 0]},
                ]
            },
        },
        final_scores=[26_500, 23_500, 25_000, 25_000],
    )
    events = convert_decoded_match(raw)[0]["observations"]["0"]
    assert events[1] == {
        "sequence": 1,
        "type": "draw",
        "player": 0,
        "tile": {"kind": "p9", "red": False},
        "rinshan": False,
    }
    assert any(event["type"] == "dora_reveal" for event in events)
    assert any(
        event["type"] == "kan" and event["meld"]["kind"] == "ankan"
        for event in events
    )
    assert events[-1]["score_deltas"] == [1500, -1500, 0, 0]


def test_real_protobuf_shape_accepts_doras_empty_repeats_and_abortive_draw() -> None:
    hands = _hands()
    hands[0].append("9p")
    raw = _decoded(
        _start(hands=hands, dora=None, doras=["4z"]),
        {
            "name": "RecordDiscardTile",
            "data": {"seat": 0, "tile": "9p", "moqie": True, "doras": []},
        },
        {"name": "RecordLiuJu", "data": {"type": 1, "doras": []}},
    )
    # Emulate JSON.stringify omitting East-1 protobuf scalar defaults.
    for key in ("chang", "ju", "ben", "liqibang"):
        raw["records"][0]["data"].pop(key, None)

    document = convert_decoded_match(raw)[0]
    assert document["observations"]["0"][0]["round"]["dealer"] == 0
    assert document["observations"]["0"][-1] == {
        "sequence": 3,
        "type": "draw_game",
        "reason": "nine_terminals",
        "score_deltas": [0, 0, 0, 0],
    }


def test_action_identity_treats_red_five_consumption_as_a_multiset() -> None:
    normal = {"kind": "s5", "red": False}
    red = {"kind": "s5", "red": True}
    base = {"type": "pon", "called_tile": normal, "from": 0}
    assert action_key({**base, "consume": [red, normal]}) == action_key(
        {**base, "consume": [normal, red]}
    )
