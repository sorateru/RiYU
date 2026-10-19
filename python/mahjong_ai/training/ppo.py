"""On-policy self-play training for the dynamic-action network."""

from __future__ import annotations

import json
import hashlib
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from mahjong_ai.ai.encoder import ACTION_FEATURES, FeatureEncoder
from mahjong_ai.ai.inference import ModelMetadata, save_checkpoint
from mahjong_ai.ai.model import MahjongNetwork, ModelConfig

from .selfplay import (
    ModelSelfPlayConfig,
    ParallelModelSelfPlayCollector,
    Transition,
    clipped_ppo_loss,
)


@dataclass(frozen=True)
class PPOTrainerConfig:
    seed: int = 20_261_019
    iterations: int = 10
    rounds_per_iteration: int = 8
    update_epochs: int = 4
    batch_size: int = 128
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    clip_ratio: float = 0.2
    value_loss_weight: float = 0.5
    entropy_weight: float = 0.01
    max_grad_norm: float = 1.0
    temperature: float = 1.0
    collection_workers: int = 0
    device: str = "cpu"

    def validate(self) -> None:
        integer_values = {
            "iterations": self.iterations,
            "rounds_per_iteration": self.rounds_per_iteration,
            "update_epochs": self.update_epochs,
            "batch_size": self.batch_size,
        }
        for name, value in integer_values.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 < self.clip_ratio < 1.0:
            raise ValueError("clip_ratio must be between zero and one")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if self.collection_workers < 0:
            raise ValueError("collection_workers must be zero (auto) or positive")


class PPOTrainer:
    """Collect on-policy rounds and update policy/score-value heads."""

    def __init__(
        self,
        model: MahjongNetwork,
        config: PPOTrainerConfig = PPOTrainerConfig(),
        start_iteration: int = 0,
        lineage: dict[str, Any] | None = None,
    ) -> None:
        config.validate()
        if start_iteration < 0:
            raise ValueError("start_iteration must not be negative")
        self.config = config
        self.model = model.to(config.device)
        self.start_iteration = start_iteration
        self.last_collection_backend = "not_started"
        self.last_collection_workers = 0
        self.lineage = dict(lineage or {})
        self.encoder = FeatureEncoder()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

    @classmethod
    def resume(cls, path: str | Path, config: PPOTrainerConfig) -> "PPOTrainer":
        checkpoint = torch.load(path, map_location=config.device, weights_only=False)
        model = MahjongNetwork(ModelConfig(**checkpoint["model_config"]))
        model.load_state_dict(checkpoint["model_state"])
        training_state = checkpoint.get("training_state") or {}
        if training_state.get("algorithm") != "selfplay_ppo_v1":
            raise ValueError("--resume requires a self-play PPO checkpoint")
        trainer = cls(
            model,
            config,
            start_iteration=int(checkpoint.get("epoch", 0)),
            lineage=training_state.get("lineage"),
        )
        optimizer_state = checkpoint.get("optimizer_state")
        if optimizer_state is not None:
            trainer.optimizer.load_state_dict(optimizer_state)
        return trainer

    @classmethod
    def warm_start(cls, path: str | Path, config: PPOTrainerConfig) -> "PPOTrainer":
        """Start PPO with model weights from a supervised checkpoint."""

        checkpoint_path = Path(path)
        checkpoint = torch.load(checkpoint_path, map_location=config.device, weights_only=False)
        model = MahjongNetwork(ModelConfig(**checkpoint["model_config"]))
        model.load_state_dict(checkpoint["model_state"])
        digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        return cls(
            model,
            config,
            start_iteration=0,
            lineage={
                "warm_start_checkpoint_sha256": digest,
                "warm_start_metadata": checkpoint.get("metadata"),
            },
        )

    def fit(
        self,
        checkpoint_path: str | Path,
        model_id: str,
        round_state: dict[str, Any],
    ) -> list[dict[str, float]]:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        history: list[dict[str, float]] = []
        for offset in range(1, self.config.iterations + 1):
            iteration = self.start_iteration + offset
            transitions = self._collect_iteration(iteration, round_state)
            metrics = self._update_iteration(iteration, transitions)
            metrics["iteration"] = float(iteration)
            metrics["rounds"] = float(self.config.rounds_per_iteration)
            metrics["transitions"] = float(len(transitions))
            metrics["collection_workers"] = float(self.last_collection_workers)
            terminal_by_player = {
                (transition.round_index, transition.player): transition.reward
                for transition in transitions
            }
            terminal_returns = np.asarray(list(terminal_by_player.values()), dtype=np.float32)
            metrics["mean_terminal_return"] = float(terminal_returns.mean())
            metrics["mean_absolute_return"] = float(np.abs(terminal_returns).mean())
            metrics["terminal_return_std"] = float(terminal_returns.std())
            metrics["nonzero_return_rate"] = float(np.count_nonzero(terminal_returns) / len(terminal_returns))
            history.append(metrics)
            fingerprint_source = json.dumps(
                {
                    "selfplay": {
                        "format": "selfplay-ppo-v1",
                        "seed": self.config.seed,
                        "iteration": iteration,
                        "rounds": self.config.rounds_per_iteration,
                    },
                    "lineage": self.lineage,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            fingerprint = hashlib.sha256(fingerprint_source.encode()).hexdigest()
            save_checkpoint(
                checkpoint_path,
                self.model,
                ModelMetadata(
                    model_id=model_id,
                    encoder_version=FeatureEncoder.version,
                    trained=True,
                    validated=False,
                    policy_calibrated=False,
                    dataset_fingerprint=fingerprint,
                ),
                self.optimizer.state_dict(),
                iteration,
                training_state={
                    "algorithm": "selfplay_ppo_v1",
                    "config": asdict(self.config),
                    "collection_backend": self.last_collection_backend,
                    "last_metrics": metrics,
                    "lineage": self.lineage,
                },
            )
        self.start_iteration += self.config.iterations
        return history

    def _collect_iteration(
        self,
        iteration: int,
        round_state: dict[str, Any],
    ) -> list[Transition]:
        self.model.eval()
        first_round = (iteration - 1) * self.config.rounds_per_iteration
        collection = ParallelModelSelfPlayCollector(
            ModelSelfPlayConfig(
                rounds=self.config.rounds_per_iteration,
                seed=self.config.seed,
                workers=self.config.collection_workers,
                temperature=self.config.temperature,
                first_round_index=first_round,
            ),
            round_state,
        ).collect(self.model)
        self.last_collection_backend = collection.execution_backend
        self.last_collection_workers = collection.workers_used
        transitions = collection.transitions
        if not transitions:
            raise RuntimeError("PPO collection produced no transitions")
        return transitions

    def _update_iteration(
        self,
        iteration: int,
        transitions: list[Transition],
    ) -> dict[str, float]:
        old_values = np.asarray([item.old_value for item in transitions], dtype=np.float32)
        returns = np.asarray([item.reward for item in transitions], dtype=np.float32)
        advantages = returns - old_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        generator = torch.Generator(device="cpu").manual_seed(self.config.seed + iteration)
        totals = {
            "loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
        }
        examples = 0
        self.model.train()
        for _ in range(self.config.update_epochs):
            order = torch.randperm(len(transitions), generator=generator).tolist()
            for start in range(0, len(order), self.config.batch_size):
                indices = order[start : start + self.config.batch_size]
                batch = self._collate(
                    [transitions[index] for index in indices],
                    advantages[indices],
                    returns[indices],
                )
                output = self.model(batch["planes"], batch["actions"], batch["mask"])
                logits = output["policy_logits"] / self.config.temperature
                log_probabilities = F.log_softmax(logits, dim=-1)
                selected = log_probabilities.gather(1, batch["action_index"].unsqueeze(1)).squeeze(1)
                policy_loss = clipped_ppo_loss(
                    selected,
                    batch["old_log_probability"],
                    batch["advantage"],
                    self.config.clip_ratio,
                )
                value_loss = F.mse_loss(output["score_value"], batch["return"])
                probabilities = log_probabilities.exp()
                entropy = -(probabilities * log_probabilities).sum(dim=-1).mean()
                loss = (
                    policy_loss
                    + self.config.value_loss_weight * value_loss
                    - self.config.entropy_weight * entropy
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    log_ratio = selected - batch["old_log_probability"]
                    ratio = log_ratio.exp()
                    count = len(indices)
                    totals["loss"] += float(loss) * count
                    totals["policy_loss"] += float(policy_loss) * count
                    totals["value_loss"] += float(value_loss) * count
                    totals["entropy"] += float(entropy) * count
                    totals["approx_kl"] += float(((ratio - 1.0) - log_ratio).mean()) * count
                    totals["clip_fraction"] += float(
                        ((ratio - 1.0).abs() > self.config.clip_ratio).float().mean()
                    ) * count
                    examples += count
        return {name: value / examples for name, value in totals.items()}

    def _collate(
        self,
        transitions: list[Transition],
        advantages: np.ndarray,
        returns: np.ndarray,
    ) -> dict[str, torch.Tensor]:
        encoded = [
            self.encoder.encode(json.loads(item.observation), json.loads(item.legal_actions))
            for item in transitions
        ]
        max_actions = max(item.action_features.shape[0] for item in encoded)
        action_features = np.zeros(
            (len(encoded), max_actions, ACTION_FEATURES),
            dtype=np.float32,
        )
        mask = np.zeros((len(encoded), max_actions), dtype=np.bool_)
        for index, item in enumerate(encoded):
            count = item.action_features.shape[0]
            action_features[index, :count] = item.action_features
            mask[index, :count] = True
        device = self.config.device
        return {
            "planes": torch.from_numpy(np.stack([item.planes for item in encoded])).to(device),
            "actions": torch.from_numpy(action_features).to(device),
            "mask": torch.from_numpy(mask).to(device),
            "action_index": torch.tensor(
                [item.action_index for item in transitions],
                dtype=torch.long,
                device=device,
            ),
            "old_log_probability": torch.tensor(
                [item.old_log_probability for item in transitions],
                dtype=torch.float32,
                device=device,
            ),
            "advantage": torch.from_numpy(advantages).to(device),
            "return": torch.from_numpy(returns).to(device),
        }
