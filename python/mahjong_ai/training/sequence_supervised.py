"""Supervised multi-task training for observable sequence decision/v2 shards."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pyarrow.parquet as pq
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from mahjong_ai.ai.encoder import ACTION_FEATURES, FeatureEncoder
from mahjong_ai.ai.inference import ModelMetadata
from mahjong_ai.ai.sequence import (
    MAX_SEQUENCE_EVENTS,
    SEQUENCE_ENCODER_VERSION,
    SequenceFeatureEncoder,
    SequenceMahjongNetwork,
)
from mahjong_ai.schema import action_key


@dataclass(frozen=True)
class SequenceTrainerConfig:
    seed: int = 20261019
    epochs: int = 3
    batch_size: int = 48
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    rank_loss_weight: float = 0.5
    score_loss_weight: float = 0.1
    round_delta_loss_weight: float = 0.15
    tenpai_loss_weight: float = 0.2
    wait_loss_weight: float = 0.1
    shanten_loss_weight: float = 0.1
    future_draw_loss_weight: float = 0.05
    gradient_clip_norm: float = 1.0
    device: str = "cpu"
    num_workers: int = 0


class ParquetSequenceDataset(Dataset[dict[str, Any]]):
    def __init__(self, paths: str | Path | Sequence[str | Path]) -> None:
        if isinstance(paths, (str, Path)):
            paths = [paths]
        import pyarrow as pa

        tables = [pq.read_table(path) for path in paths]
        self.rows = pa.concat_tables(tables).to_pylist()
        self.state_encoder = FeatureEncoder()
        self.sequence_encoder = SequenceFeatureEncoder()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        state = json.loads(row["observation"])
        legal = json.loads(row["legal_actions"])
        expert = json.loads(row["expert_action"])
        encoded = self.state_encoder.encode(state, legal)
        sequence = self.sequence_encoder.encode_tokens(json.loads(row["public_events"]))
        keys = [action_key(action) for action in legal]
        try:
            policy_target = keys.index(action_key(expert))
        except ValueError as error:
            raise ValueError(f"row {index}: expert action absent from legal actions") from error

        observer = int(row["observer"])
        rotate = lambda values: [values[(observer + relative) % 4] for relative in range(4)]
        shanten = rotate(json.loads(row["all_player_shanten"]))
        tenpai = rotate(json.loads(row["opponent_tenpai"]))
        waits = rotate(json.loads(row["opponent_waits"]))
        deltas = rotate(json.loads(row["all_round_score_deltas"]))
        wait_target = np.zeros((4, 34), dtype=np.float32)
        for relative, tiles in enumerate(waits):
            wait_target[relative, [int(tile) for tile in tiles]] = 1.0
        future_target = np.zeros(34, dtype=np.float32)
        future_target[[int(tile) for tile in json.loads(row["future_draws"])]] = 1.0
        return {
            "planes": encoded.planes,
            "actions": encoded.action_features,
            "event_types": sequence.event_types,
            "players": sequence.players,
            "tiles": sequence.tiles,
            "event_flags": sequence.flags,
            "event_mask": sequence.valid_mask,
            "policy_target": policy_target,
            "rank_target": int(row["final_rank"]) - 1,
            "score_target": (float(row["final_score"]) - 25_000.0) / 10_000.0,
            "round_delta_target": np.asarray(deltas, dtype=np.float32) / 10_000.0,
            "tenpai_target": np.asarray(tenpai, dtype=np.float32),
            "wait_target": wait_target,
            "shanten_target": np.asarray(
                [min(max(int(value), -1), 6) + 1 for value in shanten], dtype=np.int64
            ),
            "future_target": future_target,
            "log_id": row["log_id"],
        }


def collate_sequence_decisions(batch: list[dict[str, Any]]) -> dict[str, Tensor]:
    max_actions = max(item["actions"].shape[0] for item in batch)
    actions = np.zeros((len(batch), max_actions, ACTION_FEATURES), dtype=np.float32)
    action_mask = np.zeros((len(batch), max_actions), dtype=np.bool_)
    for index, item in enumerate(batch):
        count = item["actions"].shape[0]
        actions[index, :count] = item["actions"]
        action_mask[index, :count] = True
    stack = lambda key: torch.from_numpy(np.stack([item[key] for item in batch]))
    return {
        "planes": stack("planes"),
        "actions": torch.from_numpy(actions),
        "action_mask": torch.from_numpy(action_mask),
        "event_types": stack("event_types"),
        "players": stack("players"),
        "tiles": stack("tiles"),
        "event_flags": stack("event_flags"),
        "event_mask": stack("event_mask"),
        "policy_target": torch.tensor([item["policy_target"] for item in batch]),
        "rank_target": torch.tensor([item["rank_target"] for item in batch]),
        "score_target": torch.tensor([item["score_target"] for item in batch]),
        "round_delta_target": stack("round_delta_target"),
        "tenpai_target": stack("tenpai_target"),
        "wait_target": stack("wait_target"),
        "shanten_target": stack("shanten_target"),
        "future_target": stack("future_target"),
    }


def split_sequence_by_log(
    dataset: ParquetSequenceDataset, validation_fraction: float = 0.1
) -> tuple[Subset, Subset]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    train: list[int] = []
    validation: list[int] = []
    threshold = int(validation_fraction * 10_000)
    for index, row in enumerate(dataset.rows):
        # A converted round has a ``-rNN`` suffix. Keep every round from the
        # same hanchan in one side of the split to prevent player/style and
        # final-score leakage across train and validation.
        group_id = str(row["log_id"]).rsplit("-r", 1)[0]
        bucket = int.from_bytes(
            hashlib.sha256(group_id.encode()).digest()[:4], "big"
        ) % 10_000
        (validation if bucket < threshold else train).append(index)
    if not train or not validation:
        raise ValueError("split needs multiple log_ids distributed across train and validation")
    return Subset(dataset, train), Subset(dataset, validation)


class SequenceTrainer:
    def __init__(
        self,
        model: SequenceMahjongNetwork,
        config: SequenceTrainerConfig = SequenceTrainerConfig(),
    ) -> None:
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
    ) -> list[dict[str, float]]:
        train_loader = DataLoader(
            train_set, batch_size=self.config.batch_size, shuffle=True,
            num_workers=self.config.num_workers, collate_fn=collate_sequence_decisions,
        )
        validation_loader = DataLoader(
            validation_set, batch_size=self.config.batch_size, shuffle=False,
            num_workers=self.config.num_workers, collate_fn=collate_sequence_decisions,
        )
        history: list[dict[str, float]] = []
        for epoch in range(1, self.config.epochs + 1):
            train_metrics = self._run_epoch(train_loader, training=True)
            validation_metrics = self._run_epoch(validation_loader, training=False)
            metrics = {f"train_{key}": value for key, value in train_metrics.items()}
            metrics.update({f"val_{key}": value for key, value in validation_metrics.items()})
            metrics["epoch"] = float(epoch)
            history.append(metrics)
            self._save(checkpoint_path, model_id, dataset_fingerprint, epoch)
        return history

    def _forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        return self.model(
            batch["planes"], batch["actions"], batch["event_types"],
            batch["players"], batch["tiles"], batch["event_flags"],
            batch["event_mask"], batch["action_mask"],
        )

    def _run_epoch(self, loader: DataLoader, training: bool) -> dict[str, float]:
        self.model.train(training)
        totals = {"loss": 0.0, "policy_accuracy": 0.0, "tenpai_accuracy": 0.0}
        examples = 0
        opponent_mask = torch.tensor([False, True, True, True], device=self.config.device)
        for raw_batch in loader:
            batch = {key: value.to(self.config.device) for key, value in raw_batch.items()}
            with torch.set_grad_enabled(training):
                output = self._forward(batch)
                policy_loss = F.cross_entropy(output["policy_logits"], batch["policy_target"])
                rank_loss = F.cross_entropy(output["rank_logits"], batch["rank_target"])
                score_loss = F.mse_loss(output["score_value"], batch["score_target"])
                delta_loss = F.smooth_l1_loss(
                    output["round_delta_values"], batch["round_delta_target"]
                )
                tenpai_loss = F.binary_cross_entropy_with_logits(
                    output["opponent_tenpai_logits"][:, opponent_mask],
                    batch["tenpai_target"][:, opponent_mask],
                )
                wait_loss = F.binary_cross_entropy_with_logits(
                    output["opponent_wait_logits"][:, opponent_mask],
                    batch["wait_target"][:, opponent_mask],
                )
                shanten_loss = F.cross_entropy(
                    output["opponent_shanten_logits"][:, opponent_mask].reshape(
                        -1, self.model.config.shanten_classes
                    ),
                    batch["shanten_target"][:, opponent_mask].reshape(-1),
                )
                future_loss = F.binary_cross_entropy_with_logits(
                    output["future_draw_logits"], batch["future_target"]
                )
                cfg = self.config
                loss = (
                    policy_loss + cfg.rank_loss_weight * rank_loss
                    + cfg.score_loss_weight * score_loss
                    + cfg.round_delta_loss_weight * delta_loss
                    + cfg.tenpai_loss_weight * tenpai_loss
                    + cfg.wait_loss_weight * wait_loss
                    + cfg.shanten_loss_weight * shanten_loss
                    + cfg.future_draw_loss_weight * future_loss
                )
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), cfg.gradient_clip_norm
                    )
                    self.optimizer.step()
            count = batch["planes"].shape[0]
            examples += count
            totals["loss"] += float(loss.detach()) * count
            totals["policy_accuracy"] += float(
                (output["policy_logits"].argmax(1) == batch["policy_target"]).sum()
            )
            predicted_tenpai = output["opponent_tenpai_logits"][:, opponent_mask] >= 0
            totals["tenpai_accuracy"] += float(
                (predicted_tenpai == batch["tenpai_target"][:, opponent_mask].bool()).float().mean()
            ) * count
        if examples == 0:
            raise ValueError("empty data loader")
        return {key: value / examples for key, value in totals.items()}

    def _save(
        self, path: str | Path, model_id: str, dataset_fingerprint: str, epoch: int
    ) -> None:
        metadata = ModelMetadata(
            model_id=model_id,
            encoder_version=f"{FeatureEncoder.version}+{SEQUENCE_ENCODER_VERSION}",
            trained=True,
            validated=False,
            policy_calibrated=False,
            dataset_fingerprint=dataset_fingerprint,
        )
        torch.save(
            {
                "model_family": "sequence-transformer-v2",
                "model_state": self.model.state_dict(),
                "model_config": self.model.config.to_dict(),
                "metadata": asdict(metadata),
                "optimizer_state": self.optimizer.state_dict(),
                "epoch": epoch,
                "training_config": asdict(self.config),
            },
            path,
        )
