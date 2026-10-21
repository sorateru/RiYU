"""Supervised multi-task training for observable sequence decision/v2/v3 shards."""

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

from mahjong_ai.ai.encoder import ACTION_FEATURES, DANGER_FEATURES, FeatureEncoder
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
    tenpai_positive_weight: float = 1.0
    wait_positive_weight: float = 10.0
    shanten_loss_weight: float = 0.1
    future_draw_loss_weight: float = 0.05
    deal_in_loss_weight: float = 0.2
    deal_in_score_loss_weight: float = 0.1
    outcome_loss_weight: float = 0.2
    deal_in_positive_weight: float = 20.0
    gradient_clip_norm: float = 1.0
    device: str = "cpu"
    num_workers: int = 0


class ParquetSequenceDataset(Dataset[dict[str, Any]]):
    def __init__(self, paths: str | Path | Sequence[str | Path]) -> None:
        if isinstance(paths, (str, Path)):
            paths = [paths]
        import pyarrow as pa

        tables = [pq.read_table(path) for path in paths]
        required_decision_labels = {
            "expert_deal_in_by_player",
            "expert_deal_in_points_by_player",
            "expert_outcome_target",
        }
        self.has_decision_labels = all(
            required_decision_labels.issubset(table.column_names) for table in tables
        )
        self.rows = pa.concat_tables(tables, promote_options="default").to_pylist()
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
        deal_in = rotate(
            json.loads(
                row.get("expert_deal_in_by_player")
                or "[false,false,false,false]"
            )
        )
        raw_points_by_player = row.get("expert_deal_in_points_by_player")
        if raw_points_by_player:
            deal_in_points_by_player = rotate(json.loads(raw_points_by_player))
        else:
            legacy_points = int(row.get("expert_deal_in_points") or 0)
            deal_in_points_by_player = [legacy_points if target else 0 for target in deal_in]
        score_edges = np.asarray(
            [2_000, 3_900, 5_200, 7_700, 8_000, 12_000, 18_000],
            dtype=np.int64,
        )
        return {
            "planes": encoded.planes,
            "actions": encoded.action_features,
            "danger_features": encoded.danger_features,
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
            "deal_in_target": np.asarray(deal_in, dtype=np.float32),
            "deal_in_score_target": np.searchsorted(
                score_edges, np.asarray(deal_in_points_by_player, dtype=np.int64)
            ).astype(np.int64),
            "deal_in_score_mask": np.asarray(deal_in, dtype=np.float32),
            "action_outcome_target": int(
                row["expert_outcome_target"]
                if row.get("expert_outcome_target") is not None
                else 4
            ),
            "log_id": row["log_id"],
        }


def collate_sequence_decisions(batch: list[dict[str, Any]]) -> dict[str, Tensor]:
    max_actions = max(item["actions"].shape[0] for item in batch)
    actions = np.zeros((len(batch), max_actions, ACTION_FEATURES), dtype=np.float32)
    danger_features = np.zeros(
        (len(batch), max_actions, 4, DANGER_FEATURES), dtype=np.float32
    )
    action_mask = np.zeros((len(batch), max_actions), dtype=np.bool_)
    for index, item in enumerate(batch):
        count = item["actions"].shape[0]
        actions[index, :count] = item["actions"]
        danger_features[index, :count] = item["danger_features"]
        action_mask[index, :count] = True
    stack = lambda key: torch.from_numpy(np.stack([item[key] for item in batch]))
    return {
        "planes": stack("planes"),
        "actions": torch.from_numpy(actions),
        "danger_features": torch.from_numpy(danger_features),
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
        "deal_in_target": stack("deal_in_target"),
        "deal_in_score_target": stack("deal_in_score_target"),
        "deal_in_score_mask": stack("deal_in_score_mask"),
        "action_outcome_target": torch.tensor(
            [item["action_outcome_target"] for item in batch]
        ),
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


def split_sequence_train_validation_holdout(
    dataset: ParquetSequenceDataset,
    validation_fraction: float = 0.1,
    holdout_fraction: float = 0.1,
) -> tuple[Subset, Subset, Subset]:
    """Deterministic hanchan-level split with a sealed final holdout.

    The holdout occupies the first hash interval, validation the next one, and
    training the remainder.  Callers must never use the returned holdout for
    checkpoint selection, early stopping, calibration fitting, or tuning.
    """

    if validation_fraction <= 0.0 or holdout_fraction <= 0.0:
        raise ValueError("validation and holdout fractions must be positive")
    if validation_fraction + holdout_fraction >= 1.0:
        raise ValueError("validation plus holdout fraction must be below one")
    holdout_threshold = int(holdout_fraction * 1_000_000)
    validation_threshold = holdout_threshold + int(validation_fraction * 1_000_000)
    partitions: tuple[list[int], list[int], list[int]] = ([], [], [])
    for index, row in enumerate(dataset.rows):
        group_id = str(row["log_id"]).rsplit("-r", 1)[0]
        bucket = int.from_bytes(
            hashlib.sha256(("final-split-v1:" + group_id).encode()).digest()[:8],
            "big",
        ) % 1_000_000
        if bucket < holdout_threshold:
            partitions[2].append(index)
        elif bucket < validation_threshold:
            partitions[1].append(index)
        else:
            partitions[0].append(index)
    if any(not indexes for indexes in partitions):
        raise ValueError("three-way split needs hanchan in every partition")
    return (
        Subset(dataset, partitions[0]),
        Subset(dataset, partitions[1]),
        Subset(dataset, partitions[2]),
    )


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
        best_validation_loss = float("inf")
        checkpoint_path = Path(checkpoint_path)
        last_checkpoint_path = checkpoint_path.with_name(
            f"{checkpoint_path.stem}-last{checkpoint_path.suffix}"
        )
        for epoch in range(1, self.config.epochs + 1):
            train_metrics = self._run_epoch(train_loader, training=True)
            validation_metrics = self._run_epoch(validation_loader, training=False)
            metrics = {f"train_{key}": value for key, value in train_metrics.items()}
            metrics.update({f"val_{key}": value for key, value in validation_metrics.items()})
            metrics["epoch"] = float(epoch)
            history.append(metrics)
            validation_loss = validation_metrics["loss"]
            self._save(
                last_checkpoint_path,
                model_id,
                dataset_fingerprint,
                epoch,
                validation_loss=validation_loss,
                selected_as_best=False,
            )
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                self._save(
                    checkpoint_path,
                    model_id,
                    dataset_fingerprint,
                    epoch,
                    validation_loss=validation_loss,
                    selected_as_best=True,
                )
        return history

    def _forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        return self.model(
            batch["planes"], batch["actions"], batch["event_types"],
            batch["players"], batch["tiles"], batch["event_flags"],
            batch["event_mask"], batch["action_mask"],
            batch["danger_features"],
        )

    def _run_epoch(self, loader: DataLoader, training: bool) -> dict[str, float]:
        self.model.train(training)
        totals = {
            "loss": 0.0,
            "policy_accuracy": 0.0,
            "tenpai_accuracy": 0.0,
            "deal_in_brier": 0.0,
            "outcome_accuracy": 0.0,
        }
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
                    pos_weight=torch.tensor(
                        self.config.tenpai_positive_weight,
                        device=self.config.device,
                    ),
                )
                wait_logits = output["opponent_wait_logits"][:, opponent_mask]
                wait_targets = batch["wait_target"][:, opponent_mask]
                # Wait shape is meaningful conditional on an actually tenpai
                # hand. Training all 34 negatives for non-tenpai rows teaches
                # the degenerate all-zero predictor; the separate tenpai head
                # already learns whether this conditional distribution applies.
                wait_rows = wait_targets.sum(dim=-1) > 0
                if bool(wait_rows.any()):
                    wait_loss = F.binary_cross_entropy_with_logits(
                        wait_logits[wait_rows],
                        wait_targets[wait_rows],
                        pos_weight=torch.tensor(
                            self.config.wait_positive_weight,
                            device=self.config.device,
                        ),
                    )
                else:
                    wait_loss = wait_logits.sum() * 0.0
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
                selected_danger: Tensor | None = None
                selected_outcome: Tensor | None = None
                if self.model.config.decision_heads:
                    batch_indexes = torch.arange(
                        batch["planes"].shape[0], device=self.config.device
                    )
                    selected_danger = output["deal_in_logits"][
                        batch_indexes, batch["policy_target"]
                    ]
                    danger_loss = F.binary_cross_entropy_with_logits(
                        selected_danger[:, opponent_mask],
                        batch["deal_in_target"][:, opponent_mask],
                        pos_weight=torch.tensor(
                            cfg.deal_in_positive_weight, device=self.config.device
                        ),
                    )
                    positive = batch["deal_in_target"].bool()
                    if bool(positive.any()):
                        deal_in_score_loss = F.cross_entropy(
                            output["deal_in_score_logits"][positive],
                            batch["deal_in_score_target"][positive],
                        )
                    else:
                        deal_in_score_loss = output["deal_in_score_logits"].sum() * 0.0
                    selected_outcome = output["action_outcome_logits"][
                        batch_indexes, batch["policy_target"]
                    ]
                    outcome_loss = F.cross_entropy(
                        selected_outcome, batch["action_outcome_target"]
                    )
                    loss = (
                        loss
                        + cfg.deal_in_loss_weight * danger_loss
                        + cfg.deal_in_score_loss_weight * deal_in_score_loss
                        + cfg.outcome_loss_weight * outcome_loss
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
            if selected_danger is not None and selected_outcome is not None:
                danger_probabilities = torch.sigmoid(
                    selected_danger[:, opponent_mask].detach()
                )
                totals["deal_in_brier"] += float(
                    (
                        danger_probabilities
                        - batch["deal_in_target"][:, opponent_mask]
                    )
                    .square()
                    .mean()
                ) * count
                totals["outcome_accuracy"] += float(
                    (
                        selected_outcome.detach().argmax(1)
                        == batch["action_outcome_target"]
                    ).sum()
                )
        if examples == 0:
            raise ValueError("empty data loader")
        return {key: value / examples for key, value in totals.items()}

    def _save(
        self,
        path: str | Path,
        model_id: str,
        dataset_fingerprint: str,
        epoch: int,
        *,
        validation_loss: float,
        selected_as_best: bool,
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
                "selection": {
                    "validation_loss": validation_loss,
                    "selected_as_best": selected_as_best,
                },
            },
            path,
        )
