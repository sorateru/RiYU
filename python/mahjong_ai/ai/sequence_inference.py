"""Validated sequence-model inference adapter for the belief rollout search."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from mahjong_ai import analyze_state_json
from mahjong_ai.schema import canonical_json, tile_index

from .encoder import FeatureEncoder
from .inference import ModelMetadata
from .search import OpponentBelief
from .sequence import (
    SEQUENCE_ENCODER_VERSION,
    SequenceFeatureEncoder,
    SequenceMahjongNetwork,
    SequenceModelConfig,
    public_event_tokens,
)


class SequenceBeliefPredictor:
    """Turn V2 auxiliary heads into priors accepted by ``BeliefRolloutSearch``."""

    def __init__(
        self,
        model: SequenceMahjongNetwork,
        metadata: ModelMetadata,
        *,
        device: str = "cpu",
        require_validated: bool = True,
    ) -> None:
        expected = f"{FeatureEncoder.version}+{SEQUENCE_ENCODER_VERSION}"
        if metadata.encoder_version != expected:
            raise ValueError("checkpoint sequence encoder version does not match runtime")
        if not metadata.trained:
            raise ValueError("sequence belief checkpoint is not trained")
        if require_validated and not metadata.validated:
            raise ValueError("sequence belief checkpoint is not approved by evaluation")
        self.model = model.to(device).eval()
        self.metadata = metadata
        self.device = device
        self.state_encoder = FeatureEncoder()
        self.sequence_encoder = SequenceFeatureEncoder(model.config.maximum_events)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: str = "cpu",
        require_validated: bool = True,
    ) -> "SequenceBeliefPredictor":
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        if checkpoint.get("model_family") != "sequence-transformer-v2":
            raise ValueError("checkpoint is not sequence-transformer-v2")
        model = SequenceMahjongNetwork(SequenceModelConfig(**checkpoint["model_config"]))
        model.load_state_dict(checkpoint["model_state"])
        return cls(
            model,
            ModelMetadata(**checkpoint["metadata"]),
            device=device,
            require_validated=require_validated,
        )

    def __call__(self, state: dict[str, Any]) -> tuple[OpponentBelief, ...]:
        encoded = self.state_encoder.encode(state, [{"type": "pass"}])
        sequence = self.sequence_encoder.encode_tokens(public_event_tokens(state))

        def tensor(value: Any) -> torch.Tensor:
            return torch.from_numpy(value).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            output = self.model(
                tensor(encoded.planes),
                tensor(encoded.action_features),
                tensor(sequence.event_types),
                tensor(sequence.players),
                tensor(sequence.tiles),
                tensor(sequence.flags),
                tensor(sequence.valid_mask),
                danger_features=tensor(encoded.danger_features[:, 1:, :]),
            )
            tenpai = torch.sigmoid(output["tenpai_logits"])[0].cpu().tolist()
            waits = torch.sigmoid(output["wait_logits"])[0].cpu().tolist()
            deltas = (output["round_delta_values"][0] * 10_000.0).cpu().tolist()

        observer = int(state["information"]["observer"])
        dealer = int((state.get("round") or {}).get("dealer", -1))
        result: list[OpponentBelief] = []
        for player in range(4):
            relative = (player - observer) % 4
            public = state["players"][player]
            safe = frozenset(tile_index(item["tile"]) for item in public["discards"])
            if player == observer:
                result.append(OpponentBelief(player, 0.0, 0.0, safe, None))
                continue
            melds = len(public["melds"])
            riichi = public["riichi"] in {
                "declared", "double_declared", "accepted", "double_accepted"
            }
            baseline = 7_000.0 if riichi else 3_000.0 + 850.0 * melds
            if player == dealer:
                baseline *= 1.35
            predicted_gain = max(0.0, float(deltas[relative]))
            value = max(1_500.0, 0.7 * baseline + 0.3 * predicted_gain)
            opponent_index = relative - 1
            probability = 0.995 if riichi else min(
                max(float(tenpai[opponent_index]), 0.01), 0.98
            )
            result.append(
                OpponentBelief(
                    player,
                    probability,
                    value,
                    safe,
                    tuple(
                        probability * float(wait_probability)
                        for wait_probability in waits[opponent_index]
                    ),
                )
            )
        return tuple(result)

    def action_risks(
        self, state: dict[str, Any], legal: list[dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        """Return candidate-specific opponent danger and terminal outcome priors."""

        if not self.model.config.decision_heads:
            return {}
        encoded = self.state_encoder.encode(state, legal)
        sequence = self.sequence_encoder.encode_tokens(public_event_tokens(state))

        def tensor(value: Any) -> torch.Tensor:
            return torch.from_numpy(value).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            output = self.model(
                tensor(encoded.planes),
                tensor(encoded.action_features),
                tensor(sequence.event_types),
                tensor(sequence.players),
                tensor(sequence.tiles),
                tensor(sequence.flags),
                tensor(sequence.valid_mask),
                danger_features=tensor(encoded.danger_features[:, 1:, :]),
            )
            danger_logits = output["deal_in_logits"][0]
            scales = self.metadata.risk_logit_scale_by_opponent
            biases = self.metadata.risk_logit_bias_by_opponent
            if scales is not None or biases is not None:
                if scales is None or biases is None or len(scales) != 4 or len(biases) != 4:
                    raise ValueError("risk calibration needs four scales and four biases")
                scale_tensor = torch.tensor(scales[1:], device=danger_logits.device)
                bias_tensor = torch.tensor(biases[1:], device=danger_logits.device)
                danger_logits = danger_logits * scale_tensor + bias_tensor
            dangers = torch.sigmoid(danger_logits).cpu()
            score_probabilities = torch.softmax(
                output["deal_in_point_logits"], dim=-1
            )[0].cpu()
            outcomes = torch.softmax(output["action_outcome_logits"], dim=-1)[0].cpu()
        centers = torch.tensor(
            [1_000.0, 2_900.0, 4_550.0, 6_450.0, 7_850.0, 10_000.0, 15_000.0, 24_000.0]
        )
        expected_losses = (score_probabilities * centers).sum(dim=-1).tolist()
        observer = int(state["information"]["observer"])
        result: dict[str, dict[str, Any]] = {}
        for action_index, action in enumerate(legal):
            absolute_danger = [0.0] * 4
            absolute_loss = [0.0] * 4
            for relative in range(1, 4):
                player = (observer + relative) % 4
                absolute_danger[player] = float(dangers[action_index, relative - 1])
                absolute_loss[player] = float(expected_losses[action_index][relative - 1])
            result[canonical_json(action)] = {
                "calibrated": self.metadata.risk_calibrated,
                "deal_in_probability_by_opponent": tuple(absolute_danger),
                "expected_deal_in_loss_by_opponent": tuple(absolute_loss),
                "outcome_probabilities": tuple(float(value) for value in outcomes[action_index]),
            }
        return result

    def evaluate_state(self, state: dict[str, Any]) -> dict[str, Any]:
        """Evaluate one observable state using the frozen ModelOutput V1 contract."""

        exact = json.loads(analyze_state_json(canonical_json(state)))
        if not self.metadata.trained or not self.metadata.validated:
            exact["model_policy"] = {
                "status": "unavailable",
                "value": None,
                "reason": "sequence model is not trained and evaluation-approved",
            }
            return exact
        legal = exact["legal_actions"].get("value") or []
        if not legal:
            return exact
        encoded = self.state_encoder.encode(state, legal)
        sequence = self.sequence_encoder.encode_tokens(public_event_tokens(state))

        def tensor(value: Any) -> torch.Tensor:
            return torch.from_numpy(value).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            output = self.model(
                tensor(encoded.planes),
                tensor(encoded.action_features),
                tensor(sequence.event_types),
                tensor(sequence.players),
                tensor(sequence.tiles),
                tensor(sequence.flags),
                tensor(sequence.valid_mask),
                action_mask=torch.ones(
                    1, len(legal), dtype=torch.bool, device=self.device
                ),
                danger_features=tensor(encoded.danger_features[:, 1:, :]),
            )
            policy = torch.softmax(output.policy_logits[0], dim=-1).cpu()
            rank = torch.softmax(output.rank_logits[0], dim=-1).cpu()
            final_score = float(output.final_score[0].cpu()) * 10_000.0 + 25_000.0
            round_delta = (output.round_delta[0] * 10_000.0).cpu()
            tenpai = torch.sigmoid(output.tenpai_logits[0]).cpu()
            waits = torch.sigmoid(output.wait_logits[0]).cpu()
            shanten = output.shanten_logits[0].argmax(dim=-1).cpu() - 1
            future_draw = torch.sigmoid(output.future_draw_logits[0]).cpu()

            danger = torch.zeros(len(legal), 3)
            expected_loss = torch.zeros(len(legal), 3)
            outcome = torch.zeros(len(legal), 5)
            if self.model.config.decision_heads:
                danger_logits = output.deal_in_logits[0]
                scales = self.metadata.risk_logit_scale_by_opponent
                biases = self.metadata.risk_logit_bias_by_opponent
                if scales is not None or biases is not None:
                    if scales is None or biases is None or len(scales) != 4 or len(biases) != 4:
                        raise ValueError("risk calibration needs four scales and four biases")
                    danger_logits = (
                        danger_logits * torch.tensor(scales[1:], device=self.device)
                        + torch.tensor(biases[1:], device=self.device)
                    )
                danger = torch.sigmoid(danger_logits).cpu()
                point_probability = torch.softmax(
                    output.deal_in_point_logits[0], dim=-1
                ).cpu()
                centers = torch.tensor(
                    [1_000.0, 2_900.0, 4_550.0, 6_450.0, 7_850.0, 10_000.0, 15_000.0, 24_000.0]
                )
                expected_loss = (point_probability * centers).sum(dim=-1)
                outcome = torch.softmax(output.round_result_logits[0], dim=-1).cpu()

        efficiency_by_tile = {
            canonical_json(candidate["discard"]): candidate
            for candidate in exact.get("discard_efficiency", {}).get("value") or []
        }
        observer = int(state["information"]["observer"])
        rank_probabilities = tuple(float(value) for value in rank)
        ranking: list[dict[str, Any]] = []
        for index, action in enumerate(legal):
            tile = action.get("tile") or action.get("discard")
            efficiency = efficiency_by_tile.get(canonical_json(tile)) if tile else None
            absolute_danger = [0.0] * 4
            absolute_loss = [0.0] * 4
            for relative in range(1, 4):
                player = (observer + relative) % 4
                absolute_danger[player] = float(danger[index, relative - 1])
                absolute_loss[player] = float(expected_loss[index, relative - 1])
            risk_ev = sum(
                absolute_danger[player] * absolute_loss[player]
                for player in range(4)
            )
            ranking.append(
                {
                    "action": action,
                    "estimated_action_value": final_score - risk_ev,
                    "model_final_score": final_score,
                    "policy_score": float(policy[index]),
                    "shanten": (
                        int(efficiency["shanten"]["minimum"])
                        if efficiency is not None
                        else None
                    ),
                    "ukeire_count": (
                        int(efficiency["ukeire_count"])
                        if efficiency is not None
                        else None
                    ),
                    "expected_deal_in_loss": risk_ev,
                    "model_deal_in_probability_by_opponent": tuple(absolute_danger),
                    "model_expected_deal_in_loss_by_opponent": tuple(absolute_loss),
                    "model_outcome_probabilities": tuple(float(value) for value in outcome[index]),
                    "rank_probabilities": rank_probabilities,
                }
            )
        ranking.sort(key=lambda item: float(item["policy_score"]), reverse=True)
        calibration = "calibrated" if self.metadata.policy_calibrated else "uncalibrated"
        exact["model_policy"] = {
            "status": "model_estimate",
            "value": ranking,
            "reason": (
                "Decision V3 hierarchical policy with exact shanten/ukeire and "
                f"candidate-specific opponent risk; {calibration}; "
                f"model_id={self.metadata.model_id}"
            ),
        }
        exact["estimated_rank_value"] = {
            "status": "model_estimate",
            "value": sum((index + 1) * float(value) for index, value in enumerate(rank)),
            "reason": "softmax rank head output",
        }
        exact["sequence_model_predictions"] = {
            "status": "model_estimate",
            "value": {
                "final_score": final_score,
                "round_delta_by_relative_player": [float(value) for value in round_delta],
                "opponent_tenpai": [float(value) for value in tenpai],
                "opponent_shanten": [int(value) for value in shanten],
                "opponent_wait": [[float(value) for value in row] for row in waits],
                "future_draw": [float(value) for value in future_draw],
            },
            "reason": "observable-only ModelOutput V1 heads",
        }
        return exact

    def analyze(self, state: dict[str, Any]) -> dict[str, Any]:
        """Compatibility alias used by the reviewer service."""

        return self.evaluate_state(state)
