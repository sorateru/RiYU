from __future__ import annotations

import json

import pytest
import torch

from mahjong_ai import FourPlayerSimulator
from mahjong_ai.ai.decision import ExpectedValuePolicy
from mahjong_ai.ai.model import MahjongNetwork, ModelConfig
from mahjong_ai.ai.opponent import probability_metrics
from mahjong_ai.ai.search import (
    BeliefRolloutSearch,
    ImitationDiscardLikelihood,
    RolloutConfig,
    SearchCandidate,
    StatefulParticleFilter,
    _Outcome,
    _Particle,
    _exact_hand_value,
)
from mahjong_ai.schema import canonical_json
from mahjong_ai.schema import TILE_TYPES


def _round() -> dict[str, object]:
    return {
        "wind": "east",
        "hand_number": 1,
        "honba": 0,
        "riichi_sticks": 0,
        "dealer": 0,
    }


def _observation(seed: int = 17) -> tuple[dict, list[dict]]:
    simulator = FourPlayerSimulator(seed, canonical_json(_round()), [25_000] * 4)
    simulator.draw()
    return (
        json.loads(simulator.observation_json(0)),
        json.loads(simulator.legal_actions_json(0)),
    )


def _counts(*tiles: int) -> tuple[int, ...]:
    result = [0] * 34
    for tile in tiles:
        result[tile] += 1
    return tuple(result)


def test_discard_likelihood_reweights_hidden_hand_particles() -> None:
    state, _ = _observation()
    state["players"][1]["discards"].append(
        {
            "tile": {"kind": "m1", "red": False},
            "tsumogiri": False,
            "riichi_declaration": False,
            "called_by": None,
        }
    )
    empty = _counts()
    likely = _Particle((empty, _counts(0), empty, empty), (frozenset(),) * 4, ())
    unlikely = _Particle((empty, _counts(1), empty, empty), (frozenset(),) * 4, ())

    def action_likelihood(state, player, hand, discard):  # noqa: ANN001
        assert player == 1
        assert discard["tile"]["kind"] == "m1"
        return 0.9 if hand[0] else 0.1

    search = BeliefRolloutSearch(
        RolloutConfig(samples=2, depth=1, max_candidates=1),
        action_likelihood=action_likelihood,
    )
    weights = search._posterior_weights(state, [likely, unlikely])
    assert weights == pytest.approx([0.9, 0.1])


def test_imitation_policy_scores_only_the_acting_players_particle_hand() -> None:
    state, _ = _observation()
    discard = {
        "tile": {"kind": "m1", "red": False},
        "tsumogiri": False,
        "riichi_declaration": False,
        "called_by": None,
    }
    model = MahjongNetwork(
        ModelConfig(channels=8, residual_blocks=1, representation_dim=16, action_dim=8)
    ).eval()
    likelihood = ImitationDiscardLikelihood(model)
    hand = _counts(1, 1, 2, 2, 3, 3, 10, 11, 12, 18, 19, 20, 27)

    probability = likelihood(state, 1, hand, discard)

    assert 0.0 <= probability <= 1.0
    pre_discard = tuple(
        amount + extra
        for amount, extra in zip(hand, _counts(4), strict=True)
    )
    choice, red = likelihood.choose_discard(
        state, 1, pre_discard, frozenset({4})
    )
    assert 0 <= choice < 34
    assert isinstance(red, bool)


def test_candidates_share_one_weighted_particle_population(monkeypatch) -> None:
    state, _ = _observation()
    empty = _counts()
    particles = [
        _Particle((empty,) * 4, (frozenset(),) * 4, (index,))
        for index in range(3)
    ]
    created: list[_Particle] = []
    search = BeliefRolloutSearch(RolloutConfig(samples=3, depth=1, max_candidates=2))

    def sample(state, beliefs, rng):  # noqa: ANN001
        particle = particles[len(created)]
        created.append(particle)
        return particle

    seen: dict[str, list[int]] = {"m1": [], "m2": []}

    def rollout(state, candidate, particle, beliefs):  # noqa: ANN001
        kind = candidate.action["tile"]["kind"]
        seen[kind].append(particle.wall[0])
        return _Outcome("unresolved", float(particle.wall[0]))

    monkeypatch.setattr(search, "_sample_particle", sample)
    monkeypatch.setattr(search, "_rollout", rollout)
    candidates = [
        SearchCandidate(
            {"type": "discard", "tile": {"kind": kind, "red": False}, "tsumogiri": False},
            state,
            None,
            0.0,
        )
        for kind in ("m1", "m2")
    ]
    estimates = search.evaluate(state, candidates)

    assert len(created) == 3
    assert seen["m1"] == seen["m2"] == [0, 1, 2]
    assert all(estimate.expected_points == pytest.approx(1.0) for estimate in estimates.values())
    assert all(estimate.effective_samples == pytest.approx(3.0) for estimate in estimates.values())


def test_stateful_filter_advances_public_events_and_exports_posterior() -> None:
    state, legal = _observation()
    particle_filter = StatefulParticleFilter(
        state,
        config=RolloutConfig(samples=8, depth=1, max_candidates=1),
    )
    initial = particle_filter.posterior()
    discard = dict(legal[0])
    discard_event = {
        "sequence": initial.event_sequence + 1,
        "type": "discard",
        "player": 0,
        "tile": discard["tile"],
        "tsumogiri": discard["tsumogiri"],
        "riichi_declaration": False,
    }
    after_discard = json.loads(canonical_json(state))
    after_discard["next_event_sequence"] += 1
    diagnostics = particle_filter.observe(after_discard, discard_event)

    assert diagnostics.event_sequence == initial.event_sequence + 1
    assert diagnostics.impossible_particles == 0
    assert particle_filter.posterior().event_sequence == diagnostics.event_sequence
    assert sum(particle_filter.posterior().weights) == pytest.approx(1.0)

    hidden_draw = {
        "sequence": diagnostics.event_sequence + 1,
        "type": "draw",
        "player": 1,
        "tile": None,
        "rinshan": False,
    }
    after_draw = json.loads(canonical_json(after_discard))
    after_draw["next_event_sequence"] += 1
    draw_diagnostics = particle_filter.observe(after_draw, hidden_draw)
    assert draw_diagnostics.event_sequence == hidden_draw["sequence"]
    assert particle_filter.posterior().effective_samples > 0.0


def test_stateful_filter_rejects_leaked_opponent_draw() -> None:
    state, _ = _observation()
    particle_filter = StatefulParticleFilter(
        state,
        config=RolloutConfig(samples=4, depth=1, max_candidates=1),
    )
    event = {
        "sequence": particle_filter.posterior().event_sequence + 1,
        "type": "draw",
        "player": 2,
        "tile": {"kind": "m1", "red": False},
        "rinshan": False,
    }
    next_state = json.loads(canonical_json(state))
    next_state["next_event_sequence"] += 1
    with pytest.raises(ValueError, match="leaked"):
        particle_filter.observe(next_state, event)


def test_dora_reveal_conditions_dead_wall_without_consuming_live_wall() -> None:
    state, _ = _observation()
    particle_filter = StatefulParticleFilter(
        state,
        config=RolloutConfig(samples=8, depth=1, max_candidates=1),
    )
    before = particle_filter.posterior()
    source = before.particles[0]
    assert source.dead_wall
    value = source.dead_wall[0]
    red = source.dead_wall_reds[0]
    event = {
        "sequence": before.event_sequence + 1,
        "type": "dora_reveal",
        "indicator": {"kind": TILE_TYPES[value], "red": red},
    }
    next_state = json.loads(canonical_json(state))
    next_state["next_event_sequence"] += 1
    particle_filter.observe(next_state, event)
    after = particle_filter.posterior()
    assert all(
        len(particle.wall) == len(before.particles[0].wall)
        for particle in after.particles
    )


def test_policy_and_value_are_identical_when_only_world_secrets_change() -> None:
    state, legal = _observation()
    world_a = {
        "observation": state,
        "private_hands": [[0] * 34, [1] * 34, [2] * 34],
        "wall": list(range(34)),
    }
    world_b = {
        "observation": json.loads(canonical_json(state)),
        "private_hands": [[3] * 34, [2] * 34, [1] * 34],
        "wall": list(reversed(range(34))),
    }
    torch.manual_seed(9)
    model = MahjongNetwork(
        ModelConfig(channels=8, residual_blocks=1, representation_dim=16, action_dim=8)
    ).eval()
    policy = ExpectedValuePolicy(model, enable_rollout=False)

    first = policy.rank(world_a["observation"], legal)
    second = policy.rank(world_b["observation"], legal)

    assert [item.action for item in first] == [item.action for item in second]
    assert [item.policy_score for item in first] == [item.policy_score for item in second]
    assert [item.model_final_score for item in first] == [
        item.model_final_score for item in second
    ]


def test_search_rejects_complete_world_even_if_observation_fields_exist() -> None:
    state, _ = _observation()
    secret_state = replace_dict(state, private_hands=[[], [], [], []])
    with pytest.raises(ValueError, match="observable state only"):
        BeliefRolloutSearch().evaluate(secret_state, [])


def test_probability_metrics_include_calibration_error() -> None:
    logits = torch.tensor([-2.0, 0.0, 2.0, 3.0])
    targets = torch.tensor([0.0, 0.0, 1.0, 1.0])
    metrics = probability_metrics(logits, targets, calibration_bins=4)
    assert set(metrics) == {"brier", "log_loss", "calibration_error"}
    assert 0.0 <= float(metrics["calibration_error"]) <= 1.0


def test_particle_hand_value_uses_exact_yaku_fu_and_payment() -> None:
    state, _ = _observation()
    state["players"][1]["riichi"] = "accepted"
    counts = _counts(0, 1, 2, 3, 4, 5, 15, 16, 17, 19, 20, 13, 13)
    value = _exact_hand_value(
        state,
        1,
        counts,
        frozenset(),
        {"kind": "s4", "red": False},
        method="ron",
    )
    assert value is not None
    assert value >= 1_000.0


def test_rollout_beam_keeps_riichi_and_dama_counterfactual_together() -> None:
    tile = {"kind": "m5", "red": False}
    candidates = [
        SearchCandidate(
            {"type": "riichi", "discard": tile, "tsumogiri": False}, {}, None, 100.0
        ),
        SearchCandidate(
            {"type": "discard", "tile": {"kind": "p1", "red": False}, "tsumogiri": False},
            {}, None, 90.0,
        ),
        SearchCandidate(
            {"type": "discard", "tile": tile, "tsumogiri": False}, {}, None, 10.0
        ),
    ]
    search = BeliefRolloutSearch(RolloutConfig(samples=1, depth=1, max_candidates=2))
    selected = search._select_counterfactual_candidates(candidates)
    assert [candidate.action["type"] for candidate in selected] == ["riichi", "discard"]
    assert selected[1].action["tile"] == tile


def test_rank_utility_weight_is_strongest_in_south_four() -> None:
    model = MahjongNetwork(
        ModelConfig(channels=8, residual_blocks=1, representation_dim=16, action_dim=8)
    )
    policy = ExpectedValuePolicy(model, enable_rollout=False)
    state, _ = _observation()
    east_one = policy._rank_phase_weight(state)
    state["round"] = {**state["round"], "wind": "south", "hand_number": 4}
    south_four = policy._rank_phase_weight(state)
    assert east_one == pytest.approx(0.25)
    assert south_four == pytest.approx(1.0)


def replace_dict(source: dict, **updates) -> dict:  # noqa: ANN003
    result = dict(source)
    result.update(updates)
    return result
