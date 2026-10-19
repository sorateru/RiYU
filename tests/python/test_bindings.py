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
