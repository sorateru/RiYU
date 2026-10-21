"""Inference API that refuses to present unvalidated weights as advice."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mahjong_ai import analyze_state_json
from mahjong_ai.schema import canonical_json

from .encoder import ENCODER_VERSION, FeatureEncoder
from .decision import ExpectedValuePolicy
from .model import MahjongNetwork, ModelConfig
from .search import ImitationDiscardLikelihood


@dataclass(frozen=True)
class ModelMetadata:
    model_id: str
    encoder_version: str
    trained: bool
    validated: bool
    policy_calibrated: bool
    dataset_fingerprint: str | None = None
    # Kept separate from policy calibration: a well calibrated action
    # distribution does not imply that rare deal-in probabilities are
    # calibrated.  Old checkpoints omit this field and safely default to
    # diagnostics-only risk predictions.
    risk_calibrated: bool = False
    risk_logit_scale_by_opponent: tuple[float, float, float, float] | None = None
    risk_logit_bias_by_opponent: tuple[float, float, float, float] | None = None


class InferenceEngine:
    def __init__(self, model: MahjongNetwork, metadata: ModelMetadata, device: str = "cpu") -> None:
        if metadata.encoder_version != ENCODER_VERSION:
            raise ValueError("checkpoint encoder version does not match runtime encoder")
        self.model = model.to(device).eval()
        self.metadata = metadata
        self.device = device
        self.encoder = FeatureEncoder()
        self.expected_value = ExpectedValuePolicy(
            self.model,
            device=device,
            action_likelihood=ImitationDiscardLikelihood(self.model, device=device),
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        device: str = "cpu",
        belief_checkpoint: str | Path | None = None,
    ) -> "InferenceEngine":
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        config = ModelConfig(**checkpoint["model_config"])
        model = MahjongNetwork(config)
        model.load_state_dict(checkpoint["model_state"])
        engine = cls(model, ModelMetadata(**checkpoint["metadata"]), device)
        if belief_checkpoint is not None:
            from .sequence_inference import SequenceBeliefPredictor

            predictor = SequenceBeliefPredictor.load(
                belief_checkpoint, device=device, require_validated=True
            )
            engine.expected_value = ExpectedValuePolicy(
                engine.model,
                device=device,
                belief_predictor=predictor,
                action_likelihood=ImitationDiscardLikelihood(engine.model, device=device),
                risk_predictor=predictor.action_risks,
            )
        return engine

    def analyze(self, state: dict[str, Any]) -> dict[str, Any]:
        exact = json.loads(analyze_state_json(canonical_json(state)))
        if not self.metadata.trained or not self.metadata.validated:
            reason = "モデルは学習済みかつ評価済みとして承認されていません"
            exact["model_policy"] = {"status": "unavailable", "value": None, "reason": reason}
            return exact
        legal = exact["legal_actions"]["value"] or []
        if not legal:
            return exact
        encoded = self.encoder.encode(state, legal)
        planes = torch.from_numpy(encoded.planes).unsqueeze(0).to(self.device)
        actions = torch.from_numpy(encoded.action_features).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            output = self.model(planes, actions)
            logits = output["policy_logits"][0]
            scores = torch.softmax(logits, dim=-1).cpu().numpy()
            rank_scores = torch.softmax(output["rank_logits"][0], dim=-1).cpu().numpy()
        ranking = [item.to_dict() for item in self.expected_value.rank(state, legal)]
        label = "calibrated" if self.metadata.policy_calibrated else "uncalibrated"
        exact["model_policy"] = {
            "status": "model_estimate",
            "value": ranking,
            "reason": (
                "successor-state final-score estimate, exact shanten/ukeire, and "
                "observable-only opponent-hand/wall belief rollouts; "
                f"{label}; model_id={self.metadata.model_id}"
            ),
        }
        exact["estimated_rank_value"] = {
            "status": "model_estimate",
            "value": float(np.dot(rank_scores, np.arange(1, 5))),
            "reason": "softmax rank head output; calibration is reported separately",
        }
        return exact


def save_checkpoint(
    path: str | Path,
    model: MahjongNetwork,
    metadata: ModelMetadata,
    optimizer_state: dict[str, Any] | None = None,
    epoch: int = 0,
    training_state: dict[str, Any] | None = None,
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": model.config.to_dict(),
            "metadata": asdict(metadata),
            "optimizer_state": optimizer_state,
            "epoch": epoch,
            "training_state": training_state,
        },
        path,
    )
