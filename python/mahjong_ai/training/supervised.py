"""Reproducible supervised policy/value training."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pyarrow.parquet as pq
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from mahjong_ai.ai.encoder import ACTION_FEATURES, FeatureEncoder
from mahjong_ai.ai.inference import ModelMetadata, save_checkpoint
from mahjong_ai.ai.model import MahjongNetwork
from mahjong_ai.schema import action_key


@dataclass(frozen=True)
class TrainerConfig:
    seed: int = 20261018
    epochs: int = 3
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    value_loss_weight: float = 0.5
    score_loss_weight: float = 0.1
    device: str = "cpu"
    num_workers: int = 0


class ParquetDecisionDataset(Dataset[dict[str, Any]]):
    def __init__(self, paths: str | Path | Sequence[str | Path]) -> None:
        if isinstance(paths, (str, Path)):
            paths = [paths]
        tables = [pq.read_table(path) for path in paths]
        import pyarrow as pa

        self.rows = pa.concat_tables(tables).to_pylist()
        self.encoder = FeatureEncoder()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        state = json.loads(row["observation"])
        legal = json.loads(row["legal_actions"])
        expert = json.loads(row["expert_action"])
        encoded = self.encoder.encode(state, legal)
        keys = [action_key(action) for action in legal]
        try:
            target = keys.index(action_key(expert))
        except ValueError as error:
            raise ValueError(f"row {index}: expert action absent from legal actions") from error
        return {
            "planes": encoded.planes,
            "actions": encoded.action_features,
            "policy_target": target,
            "rank_target": int(row["final_rank"]) - 1,
            "score_target": (float(row["final_score"]) - 25_000.0) / 10_000.0,
            "log_id": row["log_id"],
        }


def collate_decisions(batch: list[dict[str, Any]]) -> dict[str, Tensor]:
    max_actions = max(item["actions"].shape[0] for item in batch)
    actions = np.zeros((len(batch), max_actions, ACTION_FEATURES), dtype=np.float32)
    mask = np.zeros((len(batch), max_actions), dtype=np.bool_)
    for index, item in enumerate(batch):
        count = item["actions"].shape[0]
        actions[index, :count] = item["actions"]
        mask[index, :count] = True
    return {
        "planes": torch.from_numpy(np.stack([item["planes"] for item in batch])),
        "actions": torch.from_numpy(actions),
        "mask": torch.from_numpy(mask),
        "policy_target": torch.tensor([item["policy_target"] for item in batch]),
        "rank_target": torch.tensor([item["rank_target"] for item in batch]),
        "score_target": torch.tensor([item["score_target"] for item in batch], dtype=torch.float32),
    }


def split_by_log(dataset: ParquetDecisionDataset, validation_fraction: float = 0.1) -> tuple[Subset, Subset]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    train: list[int] = []
    validation: list[int] = []
    threshold = int(validation_fraction * 10_000)
    for index, row in enumerate(dataset.rows):
        bucket = int.from_bytes(hashlib.sha256(row["log_id"].encode()).digest()[:4], "big") % 10_000
        (validation if bucket < threshold else train).append(index)
    if not train or not validation:
        raise ValueError("split needs multiple log_ids distributed across train and validation")
    return Subset(dataset, train), Subset(dataset, validation)


class Trainer:
    def __init__(self, model: MahjongNetwork, config: TrainerConfig = TrainerConfig()) -> None:
        self.model = model.to(config.device)
        self.config = config
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

    def fit(
        self,
        train_set: Dataset,
        validation_set: Dataset,
        checkpoint_path: str | Path,
        model_id: str,
        dataset_fingerprint: str,
        use_mlflow: bool = False,
    ) -> list[dict[str, float]]:
        train_loader = DataLoader(
            train_set,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=collate_decisions,
        )
        validation_loader = DataLoader(
            validation_set,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=collate_decisions,
        )
        history: list[dict[str, float]] = []
        run = _MlflowRun(self.config, enabled=use_mlflow)
        with run:
            for epoch in range(1, self.config.epochs + 1):
                train_metrics = self._run_epoch(train_loader, training=True)
                val_metrics = self._run_epoch(validation_loader, training=False)
                metrics = {f"train_{key}": value for key, value in train_metrics.items()}
                metrics.update({f"val_{key}": value for key, value in val_metrics.items()})
                metrics["epoch"] = float(epoch)
                history.append(metrics)
                run.log(metrics, epoch)
                save_checkpoint(
                    checkpoint_path,
                    self.model,
                    ModelMetadata(
                        model_id=model_id,
                        encoder_version=FeatureEncoder.version,
                        trained=True,
                        validated=False,
                        policy_calibrated=False,
                        dataset_fingerprint=dataset_fingerprint,
                    ),
                    self.optimizer.state_dict(),
                    epoch,
                )
        return history

    def _run_epoch(self, loader: DataLoader, training: bool) -> dict[str, float]:
        self.model.train(training)
        totals = {"loss": 0.0, "policy_accuracy": 0.0, "rank_mae": 0.0, "score_mse": 0.0}
        examples = 0
        for batch in loader:
            batch = {key: value.to(self.config.device) for key, value in batch.items()}
            with torch.set_grad_enabled(training):
                output = self.model(batch["planes"], batch["actions"], batch["mask"])
                policy_loss = F.cross_entropy(output["policy_logits"], batch["policy_target"])
                rank_loss = F.cross_entropy(output["rank_logits"], batch["rank_target"])
                score_loss = F.mse_loss(output["score_value"], batch["score_target"])
                loss = policy_loss + self.config.value_loss_weight * rank_loss + self.config.score_loss_weight * score_loss
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    self.optimizer.step()
            count = batch["planes"].shape[0]
            examples += count
            totals["loss"] += float(loss.detach()) * count
            totals["policy_accuracy"] += float((output["policy_logits"].argmax(1) == batch["policy_target"]).sum())
            totals["rank_mae"] += float((output["rank_logits"].argmax(1) - batch["rank_target"]).abs().sum())
            totals["score_mse"] += float(
                ((output["score_value"] - batch["score_target"]) ** 2).sum().detach()
            )
        if examples == 0:
            raise ValueError("empty data loader")
        return {key: value / examples for key, value in totals.items()}


class _MlflowRun:
    def __init__(self, config: TrainerConfig, enabled: bool) -> None:
        self.config = config
        self.enabled = enabled
        self.mlflow: Any = None

    def __enter__(self) -> "_MlflowRun":
        if self.enabled:
            import mlflow

            self.mlflow = mlflow
            mlflow.start_run()
            mlflow.log_params(self.config.__dict__)
        return self

    def log(self, metrics: dict[str, float], step: int) -> None:
        if self.mlflow is not None:
            self.mlflow.log_metrics(metrics, step=step)

    def __exit__(self, *_: object) -> None:
        if self.mlflow is not None:
            self.mlflow.end_run()
