from __future__ import annotations

from copy import deepcopy

import pytest
from fastapi.testclient import TestClient

from mahjong_ai.adapters import (
    MAJSOUL_EVENT_PROFILE,
    MahjongSoulAdapterError,
    map_majsoul_event,
    parse_majsoul_event,
)
from mahjong_ai.analysis import (
    MAJSOUL_SOURCE_SERVICE,
    REALTIME_PROTOCOL,
    RealtimeStateError,
    RealtimeStateManager,
)
from mahjong_ai.apps.reviewer import create_app


BASE_HAND = [
    "1m",
    "2m",
    "3m",
    "4m",
    "5m",
    "6m",
    "7m",
    "1p",
    "2p",
    "3p",
    "7s",
    "1z",
    "5z",
]


def source_event(
    sequence: int,
    name: str,
    data: dict,
    *,
    round_id: str = "east-1-0",
    event_id: str | None = None,
) -> dict:
    return {
        "event_id": event_id or f"event-{sequence}",
        "sequence": sequence,
        "round_id": round_id,
        "name": name,
        "data": data,
    }


def round_start(
    *,
    dealer: int = 0,
    hand: list[str] | None = None,
    sequence: int = 0,
    round_id: str = "east-1-0",
    hand_number: int = 1,
    scores: list[int] | None = None,
) -> dict:
    return source_event(
        sequence,
        "round_start",
        {
            "wind": "east",
            "hand_number": hand_number,
            "honba": 0,
            "riichi_sticks": 0,
            "dealer": dealer,
            "scores": scores or [25_000] * 4,
            "initial_hand": hand or BASE_HAND,
            "dora_indicator": "4z",
            "remaining_tiles": 70,
        },
        round_id=round_id,
    )


def action_types(result) -> set[str]:
    assert result.decision is not None
    return {action["type"] for action in result.decision.legal_actions}


def test_majsoul_adapter_maps_tiles_and_rejects_hidden_opponent_draw() -> None:
    parsed = parse_majsoul_event(round_start(hand=[*BASE_HAND[:-1], "0p"]))
    mapped = map_majsoul_event(parsed, game_sequence=0, observer=0)
    assert mapped["initial_hand"][-1] == {"kind": "p5", "red": True}
    assert mapped["dora_indicator"] == {"kind": "north", "red": False}

    hidden = parse_majsoul_event(
        source_event(1, "draw", {"player": 1, "tile": "9m", "rinshan": False})
    )
    with pytest.raises(MahjongSoulAdapterError, match="must be null"):
        map_majsoul_event(hidden, game_sequence=1, observer=0)


def test_state_manager_detects_self_turn_duplicate_gap_and_assertion_mismatch() -> None:
    manager = RealtimeStateManager(observer=0)
    assert manager.apply(round_start()).decision is None
    draw = source_event(1, "draw", {"player": 0, "tile": "9p", "rinshan": False})
    result = manager.apply(draw)
    assert result.decision is not None
    assert "discard" in action_types(result)
    assert result.state["drawn_tile"] == {"kind": "p9", "red": False}
    assert manager.apply(draw).status == "duplicate"

    conflicting = deepcopy(draw)
    conflicting["data"]["tile"] = "8p"
    with pytest.raises(RealtimeStateError, match="reused"):
        manager.apply(conflicting)
    with pytest.raises(RealtimeStateError, match="expected source sequence 2"):
        manager.apply(
            source_event(3, "discard", {"player": 0, "tile": "9p", "tsumogiri": True})
        )
    check = manager.validate_assertion({"hand_count": 13, "turn": 99})
    assert check["valid"] is False
    assert set(check["mismatches"]) == {"hand_count", "turn"}


def test_state_manager_handles_chi_and_pon_decisions() -> None:
    chi = RealtimeStateManager(observer=0)
    chi.apply(round_start(dealer=3))
    chi.apply(source_event(1, "draw", {"player": 3, "tile": None, "rinshan": False}))
    response = chi.apply(
        source_event(2, "discard", {"player": 3, "tile": "3m", "tsumogiri": True})
    )
    assert "chi" in action_types(response)
    after_chi = chi.apply(
        source_event(
            3,
            "chi",
            {"player": 0, "tiles": ["1m", "2m", "3m"], "called_tile": "3m", "from": 3},
        )
    )
    assert after_chi.state["players"][0]["melds"][0]["kind"] == "chi"
    assert after_chi.decision is not None

    pon_hand = ["5m", "5m", *BASE_HAND[2:]]
    pon = RealtimeStateManager(observer=0)
    pon.apply(round_start(dealer=1, hand=pon_hand))
    pon.apply(source_event(1, "draw", {"player": 1, "tile": None, "rinshan": False}))
    response = pon.apply(
        source_event(2, "discard", {"player": 1, "tile": "5m", "tsumogiri": True})
    )
    assert "pon" in action_types(response)
    after_pon = pon.apply(
        source_event(
            3,
            "pon",
            {"player": 0, "tiles": ["5m", "5m", "5m"], "called_tile": "5m", "from": 1},
        )
    )
    assert after_pon.state["players"][0]["melds"][0]["kind"] == "pon"


def test_state_manager_handles_riichi_and_concealed_kan() -> None:
    riichi = RealtimeStateManager(observer=0)
    riichi.apply(round_start())
    riichi.apply(source_event(1, "draw", {"player": 0, "tile": "9p", "rinshan": False}))
    riichi.apply(source_event(2, "riichi", {"player": 0, "step": "declared"}))
    riichi.apply(
        source_event(
            3,
            "discard",
            {"player": 0, "tile": "9p", "tsumogiri": True, "riichi_declaration": True},
        )
    )
    accepted = riichi.apply(source_event(4, "riichi", {"player": 0, "step": "accepted"}))
    assert accepted.state["players"][0]["riichi"] == "double_accepted"
    assert accepted.state["scores"][0] == 24_000

    kan_hand = ["1m", "1m", "1m", "1m", "2m", "3m", "4m", "2p", "3p", "4p", "2s", "3s", "4s"]
    kan = RealtimeStateManager(observer=0)
    kan.apply(round_start(hand=kan_hand))
    kan.apply(source_event(1, "draw", {"player": 0, "tile": "9p", "rinshan": False}))
    declared = kan.apply(
        source_event(
            2,
            "kan",
            {"player": 0, "kind": "ankan", "tiles": ["1m"] * 4, "called_tile": None, "from": None},
        )
    )
    assert declared.state["pending_kan"]["meld"]["kind"] == "ankan"
    kan.apply(source_event(3, "dora_reveal", {"tile": "6s"}))
    completed = kan.apply(
        source_event(4, "draw", {"player": 0, "tile": "9s", "rinshan": True})
    )
    assert completed.state["pending_kan"] is None
    assert completed.state["players"][0]["melds"][0]["kind"] == "ankan"


def test_state_manager_handles_win_draw_round_switch_and_snapshot_restore() -> None:
    win = RealtimeStateManager(observer=0)
    win.apply(round_start(dealer=3))
    win.apply(source_event(1, "draw", {"player": 3, "tile": None, "rinshan": False}))
    win.apply(source_event(2, "discard", {"player": 3, "tile": "9m", "tsumogiri": True}))
    result = win.apply(
        source_event(3, "ron", {"winners": [0], "loser": 3, "score_deltas": [8_000, 0, 0, -8_000]})
    )
    assert result.state["phase"]["phase"] == "awaiting_round_end"
    result = win.apply(
        source_event(4, "round_end", {"reason": "ron", "scores": [33_000, 25_000, 25_000, 17_000]})
    )
    assert result.state["phase"]["phase"] == "ended"

    next_round = win.apply(
        round_start(
            sequence=5,
            round_id="east-2-0",
            hand_number=2,
            dealer=0,
            scores=[33_000, 25_000, 25_000, 17_000],
        )
    )
    assert next_round.state["round"]["hand_number"] == 2

    draw = RealtimeStateManager(observer=0)
    draw.apply(round_start(dealer=1))
    draw_result = draw.apply(
        source_event(1, "draw_game", {"reason": "exhaustive", "score_deltas": [0, 0, 0, 0]})
    )
    draw.apply(
        source_event(2, "round_end", {"reason": "draw", "scores": [25_000] * 4})
    )
    restored = RealtimeStateManager(observer=0)
    snapshot = restored.restore(
        draw_result.state,
        next_source_sequence=2,
        round_id="east-1-0",
    )
    assert snapshot.state["phase"]["phase"] == "awaiting_round_end"
    assert restored.next_source_sequence == 2


def test_event_websocket_updates_state_and_infers_only_at_decision() -> None:
    class StaticAnalyzer:
        def __init__(self) -> None:
            self.calls = 0

        def analyze(self, _state):
            self.calls += 1
            return {
                "legal_actions": {"status": "exact", "value": []},
                "shanten": {"status": "exact", "value": {"minimum": 2}},
                "discard_efficiency": {"status": "exact", "value": []},
                "model_policy": {"status": "unavailable", "value": None},
            }

    analyzer = StaticAnalyzer()
    client = TestClient(create_app(engine=analyzer))
    with client.websocket_connect("/ws/realtime/events/source") as source:
        source.send_json(
            {
                "type": "hello",
                "protocol": REALTIME_PROTOCOL,
                "session_id": "event-stream-test",
                "observer": 0,
                "source_kind": "offline_replay",
                "source_service": MAJSOUL_SOURCE_SERVICE,
                "source_profile": MAJSOUL_EVENT_PROFILE,
            }
        )
        assert source.receive_json()["type"] == "ready"
        with client.websocket_connect("/ws/realtime/view/event-stream-test") as viewer:
            assert viewer.receive_json()["type"] == "subscribed"
            source.send_json({"type": "event", "event": round_start()})
            state_update = source.receive_json()
            assert state_update["type"] == "state_update"
            assert viewer.receive_json()["state"]["round_id"] == "east-1-0"
            assert analyzer.calls == 0

            source.send_json(
                {
                    "type": "event",
                    "event": source_event(
                        1, "draw", {"player": 0, "tile": "9p", "rinshan": False}
                    ),
                }
            )
            decision = source.receive_json()
            viewer_decision = viewer.receive_json()
            assert decision["type"] == "analysis"
            assert decision["decision"]["kind"] == "self_turn"
            assert decision["state"]["decision_required"] is True
            assert viewer_decision["source_sequence"] == 1
            assert analyzer.calls == 1
