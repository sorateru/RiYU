import json

import mahjong_ai


def test_replay_empty_event_stream_returns_valid_initial_state() -> None:
    state = json.loads(mahjong_ai.replay_events_json(0, "[]"))

    assert state["schema_version"] == 1
    assert state["information"] == {
        "observer": 0,
        "observer_hand_visible": True,
        "opponent_hands_visible": False,
        "wall_order_visible": False,
    }
    assert state["phase"] == {"phase": "not_started"}
    assert state["event_history"] == []


def test_rejects_invalid_observer() -> None:
    try:
        mahjong_ai.replay_events_json(4, "[]")
    except ValueError as error:
        assert "expected 0..=3" in str(error)
    else:
        raise AssertionError("invalid observer was accepted")


def test_analysis_never_presents_missing_model_values_as_probabilities() -> None:
    state_json = mahjong_ai.replay_events_json(0, "[]")
    analysis = json.loads(mahjong_ai.analyze_state_json(state_json))

    assert analysis["shanten"]["status"] == "unavailable"
    assert analysis["legal_actions"] == {
        "status": "exact",
        "value": [],
        "reason": None,
    }
    assert analysis["model_policy"]["status"] == "unavailable"
    assert analysis["model_policy"]["value"] is None
    assert analysis["estimated_win_rate"]["status"] == "unavailable"


def test_legal_actions_binding_matches_full_analysis() -> None:
    state_json = mahjong_ai.replay_events_json(0, "[]")
    analysis = json.loads(mahjong_ai.analyze_state_json(state_json))
    legal = json.loads(mahjong_ai.legal_actions_json(state_json))
    assert legal == analysis["legal_actions"]


def test_stateful_replayer_matches_stateless_replay() -> None:
    replayer = mahjong_ai.ObservationReplayer(0)
    assert json.loads(replayer.state_json()) == json.loads(
        mahjong_ai.replay_events_json(0, "[]")
    )
    assert json.loads(replayer.legal_actions_json()) == json.loads(
        mahjong_ai.legal_actions_json(replayer.state_json())
    )
    assert replayer.next_event_sequence() == 0


def test_hypothetical_hand_scoring_uses_authoritative_rust_rules() -> None:
    concealed = [
        {"kind": kind, "red": False}
        for kind in (
            "m1", "m2", "m3", "m4", "m5", "m6", "p7", "p8", "p9",
            "s2", "s3", "s4", "p5", "p5",
        )
    ]
    context = {
        "method": "tsumo",
        "seat_wind": "south",
        "round_wind": "east",
        "riichi": "none",
        "ippatsu": False,
        "rinshan": False,
        "chankan": False,
        "haitei": False,
        "houtei": False,
        "tenhou": False,
        "chiihou": False,
        "dora_indicators": [],
        "ura_dora_indicators": [],
        "honba": 0,
        "riichi_sticks": 0,
    }
    result = json.loads(
        mahjong_ai.score_hand_json(
            json.dumps(concealed),
            "[]",
            json.dumps({"kind": "s4", "red": False}),
            json.dumps(context),
        )
    )
    assert result["fu"]["rounded_total"] == 20
    assert result["payment"]["total_payment"] == 1_500
