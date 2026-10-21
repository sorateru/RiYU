"""Hybrid action evaluation using learned value plus exact tile efficiency.

The learned score head estimates final points from the observable successor
state. Exact shanten and live ukeire are then used as a conservative,
auditable correction. The result is an estimate, never presented as an exact
point expectation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Any, Callable

import numpy as np
import torch

from mahjong_ai import analyze_state_json, apply_event_json
from mahjong_ai.schema import canonical_json

from .encoder import FeatureEncoder
from .model import MahjongNetwork
from .search import (
    BeliefRolloutSearch,
    DiscardLikelihood,
    OpponentBelief,
    ParticlePosterior,
    RolloutConfig,
    SearchCandidate,
)


@dataclass(frozen=True)
class ActionValue:
    action: dict[str, Any]
    estimated_action_value: float
    model_final_score: float
    tile_efficiency_adjustment: float
    policy_prior_adjustment: float
    shanten: int | None
    ukeire_count: int | None
    policy_score: float
    followup_discard: dict[str, Any] | None = None
    terminal_win: bool = False
    rollout_expected_points: float | None = None
    rollout_adjustment: float = 0.0
    win_probability: float | None = None
    deal_in_probability: float | None = None
    opponent_win_probability: float | None = None
    horizontal_transfer_probability: float | None = None
    exhaustive_draw_probability: float | None = None
    unresolved_probability: float | None = None
    expected_win_points: float | None = None
    expected_deal_in_loss: float | None = None
    deal_in_probability_by_opponent: tuple[float, ...] | None = None
    expected_deal_in_loss_by_opponent: tuple[float, ...] | None = None
    model_deal_in_probability_by_opponent: tuple[float, ...] | None = None
    model_expected_deal_in_loss_by_opponent: tuple[float, ...] | None = None
    model_outcome_probabilities: tuple[float, ...] | None = None
    model_risk_adjustment: float = 0.0
    rank_probabilities: tuple[float, float, float, float] | None = None
    rank_utility: float | None = None
    rank_utility_adjustment: float = 0.0
    rollout_samples: int = 0
    rollout_effective_samples: float = 0.0
    rollout_depth: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "estimated_action_value": self.estimated_action_value,
            "model_final_score": self.model_final_score,
            "tile_efficiency_adjustment": self.tile_efficiency_adjustment,
            "policy_prior_adjustment": self.policy_prior_adjustment,
            "shanten": self.shanten,
            "ukeire_count": self.ukeire_count,
            "policy_score": self.policy_score,
            "followup_discard": self.followup_discard,
            "terminal_win": self.terminal_win,
            "rollout_expected_points": self.rollout_expected_points,
            "rollout_adjustment": self.rollout_adjustment,
            "win_probability": self.win_probability,
            "deal_in_probability": self.deal_in_probability,
            "opponent_win_probability": self.opponent_win_probability,
            "horizontal_transfer_probability": self.horizontal_transfer_probability,
            "exhaustive_draw_probability": self.exhaustive_draw_probability,
            "unresolved_probability": self.unresolved_probability,
            "expected_win_points": self.expected_win_points,
            "expected_deal_in_loss": self.expected_deal_in_loss,
            "deal_in_probability_by_opponent": self.deal_in_probability_by_opponent,
            "expected_deal_in_loss_by_opponent": self.expected_deal_in_loss_by_opponent,
            "model_deal_in_probability_by_opponent": (
                self.model_deal_in_probability_by_opponent
            ),
            "model_expected_deal_in_loss_by_opponent": (
                self.model_expected_deal_in_loss_by_opponent
            ),
            "model_outcome_probabilities": self.model_outcome_probabilities,
            "model_risk_adjustment": self.model_risk_adjustment,
            "rank_probabilities": self.rank_probabilities,
            "rank_utility": self.rank_utility,
            "rank_utility_adjustment": self.rank_utility_adjustment,
            "rollout_samples": self.rollout_samples,
            "rollout_effective_samples": self.rollout_effective_samples,
            "rollout_depth": self.rollout_depth,
        }


class ExpectedValuePolicy:
    """Rank legal actions by one-step value with exact efficiency support."""

    SHANTEN_POINT_PENALTY = 1_500.0
    UKEIRE_POINT_VALUE = 30.0
    POLICY_PRIOR_POINT_SCALE = 1_000.0
    # The current Tenhou adapter intentionally omitted kan labels. Until kan
    # outcomes are supervised, avoid letting an untrained logit trigger one.
    UNSUPERVISED_KAN_PENALTY = 100_000.0
    TERMINAL_WIN_BONUS = 100_000.0
    # Centered negative expected-rank utility. Its zero point is arbitrary;
    # only candidate-to-current differences enter action comparison.
    RANK_UTILITIES = (1.5, 0.5, -0.5, -1.5)
    RANK_UTILITY_POINT_SCALE = 8_000.0

    def __init__(
        self,
        model: MahjongNetwork,
        *,
        device: str = "cpu",
        rollout_config: RolloutConfig | None = None,
        enable_rollout: bool = True,
        belief_predictor: Callable[[dict[str, Any]], tuple[OpponentBelief, ...]] | None = None,
        action_likelihood: DiscardLikelihood | None = None,
        risk_predictor: Callable[
            [dict[str, Any], list[dict[str, Any]]], dict[str, dict[str, Any]]
        ]
        | None = None,
    ) -> None:
        self.model = model
        self.device = device
        self.encoder = FeatureEncoder()
        self.rollout = BeliefRolloutSearch(
            rollout_config or RolloutConfig(),
            belief_predictor=belief_predictor,
            action_likelihood=action_likelihood,
        )
        self.enable_rollout = enable_rollout
        self.risk_predictor = risk_predictor

    def choose(
        self,
        state: dict[str, Any],
        legal: list[dict[str, Any]],
        *,
        belief_posterior: ParticlePosterior | None = None,
    ) -> int:
        ranked = self.rank(state, legal, belief_posterior=belief_posterior)
        best_key = canonical_json(ranked[0].action)
        return next(index for index, action in enumerate(legal) if canonical_json(action) == best_key)

    def rank(
        self,
        state: dict[str, Any],
        legal: list[dict[str, Any]],
        *,
        belief_posterior: ParticlePosterior | None = None,
    ) -> list[ActionValue]:
        if not legal:
            return []
        current_analysis = json.loads(analyze_state_json(canonical_json(state)))
        policy_scores = self._policy_scores(state, legal)
        risk_estimates = self.risk_predictor(state, legal) if self.risk_predictor else {}
        current_score = float(state["scores"][int(state["information"]["observer"])])

        successors: list[dict[str, Any] | None] = []
        metrics: list[tuple[int | None, int | None, dict[str, Any] | None, bool]] = []
        ron_available = any(candidate["type"] == "ron" for candidate in legal)
        for action in legal:
            terminal = action["type"] in {"ron", "tsumo"}
            if terminal:
                successors.append(None)
                metrics.append((-1, 0, None, True))
                continue
            successor = self._successor(state, action, ron_available=ron_available)
            followup = None
            if action["type"] in {"chi", "pon"}:
                successor, followup = self._best_followup_discard(successor)
            successors.append(successor)
            shanten, ukeire = self._efficiency(action, current_analysis, successor)
            metrics.append((shanten, ukeire, followup, False))

        scored_states = [candidate for candidate in successors if candidate is not None]
        current_prediction = self._state_predictions([state])[0]
        current_rank_utility = self._rank_utility(current_prediction[1])
        learned_predictions = iter(self._state_predictions(scored_states))
        rank_weight = self._rank_phase_weight(state)
        values: list[ActionValue] = []
        for action, successor, policy_score, metric in zip(
            legal, successors, policy_scores, metrics, strict=True
        ):
            shanten, ukeire, followup, terminal = metric
            if terminal:
                model_score = current_score + self.TERMINAL_WIN_BONUS
                adjustment = 0.0
                policy_adjustment = 0.0
                risk_adjustment = 0.0
                rank_probabilities = None
                rank_utility = None
                rank_adjustment = 0.0
            else:
                model_score, rank_probabilities = next(learned_predictions)
                rank_utility = self._rank_utility(rank_probabilities)
                rank_adjustment = (
                    self.RANK_UTILITY_POINT_SCALE
                    * rank_weight
                    * (rank_utility - current_rank_utility)
                )
                adjustment = self._efficiency_adjustment(shanten, ukeire)
                # The supervised policy supplies the missing consequences that
                # raw shanten cannot express: yaku loss, defense, placement and
                # the option value of staying closed. Treat it as a Bayesian
                # prior in point-like utility units, not a calibrated payout.
                policy_adjustment = self.POLICY_PRIOR_POINT_SCALE * float(
                    np.log(max(policy_score, 1e-6))
                )
                if action["type"] == "kan":
                    policy_adjustment -= self.UNSUPERVISED_KAN_PENALTY
                risk = risk_estimates.get(canonical_json(action))
                risk_adjustment = 0.0
                if risk is not None and bool(risk.get("calibrated", False)):
                    risk_adjustment = -sum(
                        float(probability) * float(loss)
                        for probability, loss in zip(
                            risk["deal_in_probability_by_opponent"],
                            risk["expected_deal_in_loss_by_opponent"],
                            strict=True,
                        )
                    )
            risk = risk_estimates.get(canonical_json(action))
            values.append(
                ActionValue(
                    action=action,
                    estimated_action_value=(
                        model_score
                        + adjustment
                        + policy_adjustment
                        + risk_adjustment
                        + rank_adjustment
                    ),
                    model_final_score=model_score,
                    tile_efficiency_adjustment=adjustment,
                    policy_prior_adjustment=policy_adjustment,
                    shanten=shanten,
                    ukeire_count=ukeire,
                    policy_score=policy_score,
                    followup_discard=followup,
                    terminal_win=terminal,
                    model_deal_in_probability_by_opponent=(
                        tuple(risk["deal_in_probability_by_opponent"])
                        if risk is not None
                        else None
                    ),
                    model_expected_deal_in_loss_by_opponent=(
                        tuple(risk["expected_deal_in_loss_by_opponent"])
                        if risk is not None
                        else None
                    ),
                    model_outcome_probabilities=(
                        tuple(risk["outcome_probabilities"])
                        if risk is not None
                        else None
                    ),
                    model_risk_adjustment=risk_adjustment,
                    rank_probabilities=rank_probabilities,
                    rank_utility=rank_utility,
                    rank_utility_adjustment=rank_adjustment,
                )
            )
        if self.enable_rollout:
            candidates = [
                SearchCandidate(
                    action=value.action,
                    successor=successor,
                    followup_discard=value.followup_discard,
                    base_value=value.estimated_action_value,
                )
                for value, successor in zip(values, successors, strict=True)
                if successor is not None and value.action["type"] != "kan"
            ]
            try:
                estimates = self.rollout.evaluate(
                    state, candidates, posterior=belief_posterior
                )
            except (ValueError, RuntimeError):
                # Search is an optional stochastic refinement.  A malformed or
                # exhausted belief sample must never remove the deterministic
                # one-step policy fallback.
                estimates = {}
            values = [
                replace(
                    value,
                    estimated_action_value=value.estimated_action_value
                    + estimate.utility_adjustment,
                    rollout_expected_points=estimate.expected_points,
                    rollout_adjustment=estimate.utility_adjustment,
                    win_probability=estimate.win_probability,
                    deal_in_probability=estimate.deal_in_probability,
                    opponent_win_probability=estimate.opponent_win_probability,
                    horizontal_transfer_probability=estimate.horizontal_transfer_probability,
                    exhaustive_draw_probability=estimate.exhaustive_draw_probability,
                    unresolved_probability=estimate.unresolved_probability,
                    expected_win_points=estimate.expected_win_points,
                    expected_deal_in_loss=estimate.expected_deal_in_loss,
                    deal_in_probability_by_opponent=estimate.deal_in_probability_by_opponent,
                    expected_deal_in_loss_by_opponent=(
                        estimate.expected_deal_in_loss_by_opponent
                    ),
                    rollout_samples=estimate.samples,
                    rollout_effective_samples=estimate.effective_samples,
                    rollout_depth=estimate.depth,
                )
                if (estimate := estimates.get(canonical_json(value.action))) is not None
                else value
                for value in values
            ]
        return sorted(
            values,
            key=lambda item: (
                item.estimated_action_value,
                -(item.deal_in_probability or 0.0),
                item.win_probability or 0.0,
                -(item.shanten if item.shanten is not None else 99),
                item.ukeire_count if item.ukeire_count is not None else -1,
                item.policy_score,
            ),
            reverse=True,
        )

    def _policy_scores(self, state: dict[str, Any], legal: list[dict[str, Any]]) -> list[float]:
        encoded = self.encoder.encode(state, legal)
        planes = torch.from_numpy(encoded.planes).unsqueeze(0).to(self.device)
        actions = torch.from_numpy(encoded.action_features).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits = self.model(planes, actions)["policy_logits"][0]
            return torch.softmax(logits, dim=-1).cpu().tolist()

    def _state_predictions(
        self, states: list[dict[str, Any]]
    ) -> list[tuple[float, tuple[float, float, float, float]]]:
        if not states:
            return []
        dummy = [{"type": "pass"}]
        encoded = [self.encoder.encode(state, dummy) for state in states]
        planes = torch.from_numpy(np.stack([item.planes for item in encoded])).to(self.device)
        actions = torch.from_numpy(np.stack([item.action_features for item in encoded])).to(
            self.device
        )
        with torch.inference_mode():
            output = self.model(planes, actions)
            normalized = output["score_value"].cpu().numpy()
            ranks = torch.softmax(output["rank_logits"], dim=-1).cpu().numpy()
        return [
            (
                25_000.0 + 10_000.0 * float(score),
                tuple(float(value) for value in rank),
            )
            for score, rank in zip(normalized, ranks, strict=True)
        ]

    def _rank_utility(self, probabilities: tuple[float, ...]) -> float:
        if len(probabilities) != 4:
            raise ValueError("rank distribution must contain four probabilities")
        return sum(
            probability * utility
            for probability, utility in zip(
                probabilities, self.RANK_UTILITIES, strict=True
            )
        )

    def _rank_phase_weight(self, state: dict[str, Any]) -> float:
        """Increase placement sensitivity from East 1 toward South 4.

        The rank head already conditions on scores, dealer, honba, sticks and
        round context through observable planes. This factor only determines
        how strongly its candidate-to-current utility difference competes with
        short-horizon point EV.
        """

        round_state = state.get("round") or {}
        wind = str(round_state.get("wind", "east"))
        hand = min(max(int(round_state.get("hand_number", 1)), 1), 4)
        wind_offset = {"east": 0, "south": 4, "west": 8, "north": 8}.get(wind, 0)
        progress = min(wind_offset + hand - 1, 7) / 7.0
        return 0.25 + 0.75 * progress

    def _successor(
        self,
        state: dict[str, Any],
        action: dict[str, Any],
        *,
        ron_available: bool = False,
    ) -> dict[str, Any]:
        observer = int(state["information"]["observer"])
        state_json = canonical_json(state)

        def apply(event: dict[str, Any]) -> None:
            nonlocal state_json
            event = dict(event)
            event["sequence"] = json.loads(state_json)["next_event_sequence"]
            state_json = apply_event_json(state_json, canonical_json(event))

        action_type = action["type"]
        if action_type == "discard":
            apply(
                {
                    "type": "discard",
                    "player": observer,
                    "tile": action["tile"],
                    "tsumogiri": action["tsumogiri"],
                    "riichi_declaration": False,
                }
            )
        elif action_type == "riichi":
            apply({"type": "riichi", "player": observer, "step": "declared"})
            apply(
                {
                    "type": "discard",
                    "player": observer,
                    "tile": action["discard"],
                    "tsumogiri": action["tsumogiri"],
                    "riichi_declaration": True,
                }
            )
            apply({"type": "riichi", "player": observer, "step": "accepted"})
        elif action_type in {"chi", "pon"}:
            apply(
                {
                    "type": action_type,
                    "player": observer,
                    "meld": {
                        "kind": action_type,
                        "tiles": [action["called_tile"], *action["consume"]],
                        "called_tile": action["called_tile"],
                        "from": action["from"],
                    },
                }
            )
        elif action_type == "kan":
            apply({"type": "kan", "player": observer, "meld": self._kan_meld(state, action)})
        elif action_type == "pass":
            apply(
                {
                    "type": "pass",
                    "player": observer,
                    "ron_available": ron_available,
                }
            )
        else:
            return state
        return json.loads(state_json)

    def _kan_meld(self, state: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
        nested = action["action"]
        kind = nested["kind"]
        if kind == "daiminkan":
            return {
                "kind": kind,
                "tiles": [nested["called_tile"], *nested["consume"]],
                "called_tile": nested["called_tile"],
                "from": nested["from"],
            }
        if kind == "ankan":
            return {"kind": kind, "tiles": nested["tiles"], "called_tile": None, "from": None}
        observer = int(state["information"]["observer"])
        old = state["players"][observer]["melds"][int(nested["meld_index"])]
        return {
            "kind": "shouminkan",
            "tiles": [*old["tiles"], nested["tile"]],
            "called_tile": old["called_tile"],
            "from": old["from"],
        }

    def _best_followup_discard(
        self, state: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        analysis = json.loads(analyze_state_json(canonical_json(state)))
        candidates = analysis["discard_efficiency"].get("value") or []
        legal = analysis["legal_actions"].get("value") or []
        if not candidates:
            return state, None
        tile = candidates[0]["discard"]
        action = next(
            (
                candidate
                for candidate in legal
                if candidate["type"] == "discard" and candidate["tile"] == tile
            ),
            None,
        )
        if action is None:
            return state, None
        return self._successor(state, action), action

    def _efficiency(
        self,
        action: dict[str, Any],
        current_analysis: dict[str, Any],
        successor: dict[str, Any],
    ) -> tuple[int | None, int | None]:
        if action["type"] in {"discard", "riichi"}:
            tile = action.get("tile") or action.get("discard")
            for candidate in current_analysis["discard_efficiency"].get("value") or []:
                if candidate["discard"] == tile:
                    return int(candidate["shanten"]["minimum"]), int(candidate["ukeire_count"])
        analysis = json.loads(analyze_state_json(canonical_json(successor)))
        shanten = analysis["shanten"].get("value")
        ukeire = analysis["ukeire"].get("value")
        return (
            int(shanten["minimum"]) if shanten is not None else None,
            sum(int(item["remaining"]) for item in ukeire) if ukeire is not None else None,
        )

    def _efficiency_adjustment(self, shanten: int | None, ukeire: int | None) -> float:
        if shanten is None:
            return 0.0
        return -self.SHANTEN_POINT_PENALTY * max(shanten, 0) + self.UKEIRE_POINT_VALUE * (
            ukeire or 0
        )
