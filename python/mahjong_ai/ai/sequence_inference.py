"""Validated sequence-model inference adapter for the belief rollout search."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

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
                danger_features=tensor(encoded.danger_features),
            )
            tenpai = torch.sigmoid(output["opponent_tenpai_logits"])[0].cpu().tolist()
            waits = torch.sigmoid(output["opponent_wait_logits"])[0].cpu().tolist()
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
            probability = 0.995 if riichi else min(max(float(tenpai[relative]), 0.01), 0.98)
            result.append(
                OpponentBelief(
                    player,
                    probability,
                    value,
                    safe,
                    tuple(
                        probability * float(wait_probability)
                        for wait_probability in waits[relative]
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
                danger_features=tensor(encoded.danger_features),
            )
            danger_logits = output["deal_in_logits"][0]
            scales = self.metadata.risk_logit_scale_by_opponent
            biases = self.metadata.risk_logit_bias_by_opponent
            if scales is not None or biases is not None:
                if scales is None or biases is None or len(scales) != 4 or len(biases) != 4:
                    raise ValueError("risk calibration needs four scales and four biases")
                scale_tensor = torch.tensor(scales, device=danger_logits.device)
                bias_tensor = torch.tensor(biases, device=danger_logits.device)
                danger_logits = danger_logits * scale_tensor + bias_tensor
            dangers = torch.sigmoid(danger_logits).cpu()
            score_probabilities = torch.softmax(
                output["deal_in_score_logits"], dim=-1
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
                absolute_danger[player] = float(dangers[action_index, relative])
                absolute_loss[player] = float(expected_losses[relative])
            result[canonical_json(action)] = {
                "calibrated": self.metadata.risk_calibrated,
                "deal_in_probability_by_opponent": tuple(absolute_danger),
                "expected_deal_in_loss_by_opponent": tuple(absolute_loss),
                "outcome_probabilities": tuple(float(value) for value in outcomes[action_index]),
            }
        return result
