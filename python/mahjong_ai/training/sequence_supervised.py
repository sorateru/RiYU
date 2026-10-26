"""Supervised multi-task training for observable sequence decision/v2-v4 shards."""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

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
    MODEL_OUTPUT_VERSION,
    SEQUENCE_ARCHITECTURE_VERSION,
    SEQUENCE_ENCODER_VERSION,
    ROUND_VALUE_BIN_EDGES,
    RuleFeatureEncoder,
    MahjongModelOutput,
    SequenceFeatureEncoder,
    SequenceMahjongNetwork,
)
from mahjong_ai.data.sequence_dataset import (
    SEQUENCE_DATASET_FORMAT,
    SEQUENCE_LABEL_VERSION,
)
from mahjong_ai.schema import action_key


ROUND_VALUE_EDGES = np.asarray(ROUND_VALUE_BIN_EDGES, dtype=np.float32)


class HanchanUtility(Protocol):
    def __call__(self, final_rank: int, final_score: int) -> float: ...


@dataclass(frozen=True)
class RankScoreUtility:
    """Replaceable utility mapping used by supervised and future offline RL."""

    rank_values: tuple[float, float, float, float] = (1.0, 0.35, -0.35, -1.0)
    score_weight: float = 0.0
    score_center: float = 25_000.0
    score_scale: float = 10_000.0

    def __post_init__(self) -> None:
        if len(self.rank_values) != 4:
            raise ValueError("rank_values must contain one utility for each rank")
        if self.score_scale <= 0:
            raise ValueError("score_scale must be positive")

    def __call__(self, final_rank: int, final_score: int) -> float:
        if not 1 <= final_rank <= 4:
            raise ValueError("final_rank must be between one and four")
        return self.rank_values[final_rank - 1] + self.score_weight * (
            (final_score - self.score_center) / self.score_scale
        )


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
    response_loss_weight: float = 0.2
    call_type_loss_weight: float = 0.15
    riichi_loss_weight: float = 0.15
    state_round_value_loss_weight: float = 0.15
    state_hanchan_value_loss_weight: float = 0.2
    action_round_value_loss_weight: float = 0.3
    action_hanchan_value_loss_weight: float = 0.3
    action_round_distribution_loss_weight: float = 0.2
    opponent_discard_loss_weight: float = 0.1
    opponent_response_loss_weight: float = 0.1
    opponent_call_type_loss_weight: float = 0.05
    opponent_tile_count_loss_weight: float = 0.05
    belief_consistency_loss_weight: float = 0.05
    deal_in_positive_weight: float = 20.0
    auto_positive_weights: bool = True
    maximum_positive_weight: float = 30.0
    policy_label_smoothing: float = 0.02
    gradient_clip_norm: float = 1.0
    scheduler_t_max_epochs: int = 50
    minimum_learning_rate_ratio: float = 0.05
    hanchan_utility_version: str = "rank-score-utility-v1"
    hanchan_rank_utilities: tuple[float, float, float, float] = (
        1.0,
        0.35,
        -0.35,
        -1.0,
    )
    hanchan_score_weight: float = 0.0
    device: str = "cpu"
    num_workers: int = 0


def estimate_sequence_positive_weights(
    dataset: Dataset,
    *,
    fallback: dict[str, float] | None = None,
    maximum: float = 30.0,
) -> dict[str, float]:
    """Estimate bounded negative/positive ratios from training rows only.

    Validation and final-holdout labels are intentionally never inspected.
    Wait weights are conditional on tenpai rows, matching the conditional wait
    loss. Missing legacy decision labels retain the configured fallback.
    """

    if maximum < 1.0:
        raise ValueError("maximum positive weight must be at least one")
    defaults = fallback or {"tenpai": 1.0, "wait": 10.0, "deal_in": 20.0}
    counts = {
        "tenpai": [0, 0],
        "wait": [0, 0],
        "deal_in": [0, 0],
    }
    labelled_deal_in_rows = 0
    for row in _raw_rows(dataset):
        is_v3 = row.get("factual_deal_in_by_action") is not None
        observer = int(row["observer"])
        if is_v3:
            tenpai = json.loads(row["opponent_tenpai"])
            waits = json.loads(row["opponent_wait"])
            opponent_indexes = range(3)
        else:
            tenpai = json.loads(row["opponent_tenpai"])
            waits = json.loads(row["opponent_waits"])
            opponent_indexes = [seat for seat in range(4) if seat != observer]
        for seat in opponent_indexes:
            counts["tenpai"][1] += 1
            counts["tenpai"][0] += int(bool(tenpai[seat]))
            tiles = waits[seat]
            if is_v3:
                if bool(tenpai[seat]):
                    counts["wait"][0] += sum(bool(value) for value in tiles)
                    counts["wait"][1] += 34
            elif tiles:
                counts["wait"][0] += len(set(int(tile) for tile in tiles))
                counts["wait"][1] += 34

        if is_v3:
            factual = json.loads(row["factual_deal_in_by_action"])
            factual_mask = json.loads(row["factual_deal_in_mask"])
            counterfactual = json.loads(row["counterfactual_deal_in_by_action"])
            counterfactual_mask = json.loads(row["counterfactual_deal_in_mask"])
            labelled_deal_in_rows += 1
            for action in range(len(factual)):
                for seat in range(3):
                    if counterfactual_mask[action][seat]:
                        counts["deal_in"][1] += 1
                        counts["deal_in"][0] += int(counterfactual[action][seat])
                    elif factual_mask[action][seat]:
                        counts["deal_in"][1] += 1
                        counts["deal_in"][0] += int(factual[action][seat])
        else:
            raw_deal_in = row.get("expert_deal_in_by_player")
            if raw_deal_in is not None:
                labelled_deal_in_rows += 1
                deal_in = json.loads(raw_deal_in)
                for seat in [seat for seat in range(4) if seat != observer]:
                    counts["deal_in"][1] += 1
                    counts["deal_in"][0] += int(bool(deal_in[seat]))

    result: dict[str, float] = {}
    for name, (positives, total) in counts.items():
        if name == "deal_in" and labelled_deal_in_rows == 0:
            result[name] = float(defaults[name])
            continue
        negatives = total - positives
        if positives <= 0 or negatives <= 0:
            result[name] = float(defaults[name])
        else:
            result[name] = min(max(float(negatives) / float(positives), 1.0), maximum)
    return result


def _raw_rows(dataset: Dataset) -> list[dict[str, Any]]:
    if isinstance(dataset, Subset):
        parent = dataset.dataset
        rows = getattr(parent, "rows", None)
        if rows is not None:
            return [rows[int(index)] for index in dataset.indices]
    rows = getattr(dataset, "rows", None)
    if rows is None:
        raise TypeError("positive-weight estimation requires a dataset with raw rows")
    return list(rows)


def masked_policy_cross_entropy(
    logits: Tensor,
    targets: Tensor,
    action_mask: Tensor,
    label_smoothing: float,
) -> Tensor:
    """Cross entropy whose smoothing mass covers legal actions only."""

    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError("label_smoothing must be in [0, 1)")
    if logits.shape != action_mask.shape:
        raise ValueError("action mask must match policy logits")
    if not bool(action_mask.gather(1, targets.unsqueeze(1)).all()):
        raise ValueError("policy target points to a masked action")
    log_probabilities = F.log_softmax(logits, dim=-1)
    negative_log_likelihood = -log_probabilities.gather(
        1, targets.unsqueeze(1)
    ).squeeze(1)
    legal_log_probabilities = log_probabilities.masked_fill(~action_mask, 0.0)
    smooth = -legal_log_probabilities.sum(dim=-1) / action_mask.sum(
        dim=-1
    ).clamp_min(1)
    return (
        (1.0 - label_smoothing) * negative_log_likelihood
        + label_smoothing * smooth
    ).mean()


def masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    """Mean over explicitly valid labels; an empty mask returns differentiable zero."""

    valid = mask.to(values.dtype)
    if valid.shape != values.shape:
        valid = valid.expand_as(values)
    return (values * valid).sum() / valid.sum().clamp_min(1.0)


def masked_cross_entropy(logits: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
    valid = mask
    if valid.shape != targets.shape:
        valid = valid.expand_as(targets)
    # Missing categorical labels may use an out-of-range sentinel such as -1.
    # Replace only masked cells before CE; the mask still removes their loss.
    safe_targets = torch.where(valid, targets, torch.zeros_like(targets))
    losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        safe_targets.reshape(-1),
        reduction="none",
    ).reshape(targets.shape)
    return masked_mean(losses, valid)


def masked_binary_cross_entropy(
    logits: Tensor,
    targets: Tensor,
    mask: Tensor,
    *,
    positive_weight: float | None = None,
) -> Tensor:
    pos_weight = (
        torch.tensor(positive_weight, device=logits.device, dtype=logits.dtype)
        if positive_weight is not None
        else None
    )
    losses = F.binary_cross_entropy_with_logits(
        logits, targets.to(logits.dtype), pos_weight=pos_weight, reduction="none"
    )
    return masked_mean(losses, mask)


def belief_consistency_loss(
    tenpai_logits: Tensor,
    wait_logits: Tensor,
    shanten_logits: Tensor,
    mask: Tensor,
) -> Tensor:
    """Softly align tenpai, ready-shanten and existence-of-wait beliefs."""

    tenpai_probability = torch.sigmoid(tenpai_logits)
    shanten_probability = torch.softmax(shanten_logits, dim=-1)[..., :2].sum(dim=-1)
    # A max-based soft existence score avoids the strong independence
    # assumption that would make 34 uncertain tiles imply near-certain waits.
    any_wait_probability = torch.sigmoid(wait_logits).amax(dim=-1)
    disagreement = (
        (tenpai_probability - shanten_probability).square()
        + (tenpai_probability - any_wait_probability).square()
        + (shanten_probability - any_wait_probability).square()
    ) / 3.0
    return masked_mean(disagreement, mask)


@dataclass(frozen=True)
class OfflineRLSignals:
    state_round_value: Tensor
    action_round_value: Tensor
    round_advantage: Tensor
    state_hanchan_value: Tensor
    action_hanchan_value: Tensor
    hanchan_advantage: Tensor
    action_mask: Tensor
    behavior_action: Tensor
    behavior_action_mask: Tensor


def build_offline_rl_signals(
    output: MahjongModelOutput, batch: Mapping[str, Tensor]
) -> OfflineRLSignals:
    """Stable bridge for future AWR/IQL trainers without changing the model."""

    if output.action_round_value is None or output.action_hanchan_value is None:
        raise ValueError("offline RL signals require action value heads")
    return OfflineRLSignals(
        state_round_value=output.state_round_value,
        action_round_value=output.action_round_value,
        round_advantage=output.action_round_value
        - output.state_round_value.unsqueeze(1),
        state_hanchan_value=output.state_hanchan_value,
        action_hanchan_value=output.action_hanchan_value,
        hanchan_advantage=output.action_hanchan_value
        - output.state_hanchan_value.unsqueeze(1),
        action_mask=batch["action_mask"],
        behavior_action=batch["policy_target"],
        behavior_action_mask=batch["policy_target_mask"],
    )


def merge_factual_counterfactual(
    factual: np.ndarray,
    factual_mask: np.ndarray,
    counterfactual: np.ndarray,
    counterfactual_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge only at the trainer boundary while preserving disjoint provenance."""

    if factual.shape != counterfactual.shape or factual_mask.shape != counterfactual_mask.shape:
        raise ValueError("factual and counterfactual label shapes must match")
    if factual.shape != factual_mask.shape:
        raise ValueError("label and mask shapes must match")
    if np.any(factual_mask & counterfactual_mask):
        raise ValueError("factual and counterfactual masks must not overlap")
    return np.where(counterfactual_mask, counterfactual, factual), (
        factual_mask | counterfactual_mask
    )


class ParquetSequenceDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        paths: str | Path | Sequence[str | Path],
        *,
        hanchan_utility: HanchanUtility | None = None,
    ) -> None:
        if isinstance(paths, (str, Path)):
            paths = [paths]
        import pyarrow as pa

        tables = [pq.read_table(path) for path in paths]
        required_decision_labels = {
            "factual_deal_in_by_action",
            "factual_deal_in_points_by_action",
            "factual_deal_in_mask",
            "counterfactual_deal_in_by_action",
            "counterfactual_deal_in_points_by_action",
            "counterfactual_deal_in_mask",
            "round_result",
        }
        self.has_decision_labels = all(
            required_decision_labels.issubset(table.column_names) for table in tables
        )
        required_v4_labels = {
            "factual_action_round_delta",
            "counterfactual_action_round_delta",
            "factual_round_result_by_action",
            "counterfactual_round_result_by_action",
            "opponent_next_discard",
            "opponent_tile_count",
        }
        self.has_v4_labels = all(
            required_v4_labels.issubset(table.column_names) for table in tables
        )
        self.rows = pa.concat_tables(tables, promote_options="default").to_pylist()
        self.state_encoder = FeatureEncoder()
        self.sequence_encoder = SequenceFeatureEncoder()
        self.rule_encoder = RuleFeatureEncoder()
        self.hanchan_utility = hanchan_utility or RankScoreUtility()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        state = json.loads(row["observation"])
        legal = json.loads(row["legal_actions"])
        expert = json.loads(row.get("chosen_action") or row["expert_action"])
        encoded = self.state_encoder.encode(state, legal)
        sequence = self.sequence_encoder.encode_tokens(json.loads(row["public_events"]))
        rule_features = self.rule_encoder.encode(state)
        keys = [action_key(action) for action in legal]
        try:
            policy_target = keys.index(action_key(expert))
        except ValueError as error:
            raise ValueError(f"row {index}: expert action absent from legal actions") from error

        observer = int(row["observer"])
        rotate = lambda values: [values[(observer + relative) % 4] for relative in range(4)]
        is_v3 = row.get("factual_deal_in_by_action") is not None
        is_v4 = row.get("factual_action_round_delta") is not None
        if is_v3:
            shanten = json.loads(row["opponent_shanten"])
            tenpai = json.loads(row["opponent_tenpai"])
            waits = json.loads(row["opponent_wait"])
            deltas = json.loads(row["round_delta"])
            future_draws = json.loads(row["future_draw"])
            label_masks = json.loads(row["label_masks"])
        else:
            shanten = rotate(json.loads(row["all_player_shanten"]))[1:]
            tenpai = rotate(json.loads(row["opponent_tenpai"]))[1:]
            waits = rotate(json.loads(row["opponent_waits"]))[1:]
            deltas = rotate(json.loads(row["all_round_score_deltas"]))
            future_draws = json.loads(row["future_draws"])
            label_masks = {
                "policy": True,
                "rank": True,
                "final_score": True,
                "round_delta": [True] * 4,
                "opponent_state": [True] * 3,
                "future_draw": True,
                "round_result": row.get("expert_outcome_target") is not None,
            }
        if is_v3:
            wait_target = np.asarray(waits, dtype=np.float32)
            future_target = np.asarray(future_draws, dtype=np.float32)
            if wait_target.shape != (3, 34) or future_target.shape != (34,):
                raise ValueError(f"row {index}: invalid sequence tile-label axes")
        else:
            wait_target = np.zeros((3, 34), dtype=np.float32)
            for relative, waiting_tiles in enumerate(waits):
                wait_target[relative, [int(tile) for tile in waiting_tiles]] = 1.0
            future_target = np.zeros(34, dtype=np.float32)
            future_target[[int(tile) for tile in future_draws]] = 1.0
        action_count = len(legal)
        if is_v3:
            factual_deal = np.asarray(
                json.loads(row["factual_deal_in_by_action"]), dtype=np.float32
            )
            factual_points = np.asarray(
                json.loads(row["factual_deal_in_points_by_action"]), dtype=np.int64
            )
            factual_mask = np.asarray(
                json.loads(row["factual_deal_in_mask"]), dtype=np.bool_
            )
            counterfactual_deal = np.asarray(
                json.loads(row["counterfactual_deal_in_by_action"]), dtype=np.float32
            )
            counterfactual_points = np.asarray(
                json.loads(row["counterfactual_deal_in_points_by_action"]), dtype=np.int64
            )
            counterfactual_mask = np.asarray(
                json.loads(row["counterfactual_deal_in_mask"]), dtype=np.bool_
            )
            deal_in, deal_in_mask = merge_factual_counterfactual(
                factual_deal, factual_mask, counterfactual_deal, counterfactual_mask
            )
            deal_in_points, _ = merge_factual_counterfactual(
                factual_points, factual_mask, counterfactual_points, counterfactual_mask
            )
            selected_outcome_target = (
                "self_win", "deal_in", "opponent_tsumo", "horizontal", "draw"
            ).index(str(row["round_result"]))
        else:
            deal_in = np.zeros((action_count, 3), dtype=np.float32)
            deal_in_points = np.zeros((action_count, 3), dtype=np.int64)
            deal_in_mask = np.zeros((action_count, 3), dtype=np.bool_)
            if self.has_decision_labels and expert["type"] in {"discard", "riichi"}:
                deal_in[policy_target] = rotate(
                    json.loads(row["expert_deal_in_by_player"])
                )[1:]
                deal_in_points[policy_target] = rotate(
                    json.loads(row["expert_deal_in_points_by_player"])
                )[1:]
                deal_in_mask[policy_target] = True
            selected_outcome_target = int(row.get("expert_outcome_target") or 4)

        action_outcome_target = np.zeros(action_count, dtype=np.int64)
        action_outcome_mask = np.zeros(action_count, dtype=np.bool_)
        action_round_value_target = np.zeros(action_count, dtype=np.float32)
        action_round_value_mask = np.zeros(action_count, dtype=np.bool_)
        action_hanchan_value_target = np.zeros(action_count, dtype=np.float32)
        action_hanchan_value_mask = np.zeros(action_count, dtype=np.bool_)
        if is_v4:
            factual_outcome = np.asarray(
                json.loads(row["factual_round_result_by_action"]), dtype=np.int64
            )
            factual_outcome_mask = np.asarray(
                json.loads(row["factual_round_result_mask"]), dtype=np.bool_
            )
            counterfactual_outcome = np.asarray(
                json.loads(row["counterfactual_round_result_by_action"]), dtype=np.int64
            )
            counterfactual_outcome_mask = np.asarray(
                json.loads(row["counterfactual_round_result_mask"]), dtype=np.bool_
            )
            action_outcome_target, action_outcome_mask = merge_factual_counterfactual(
                factual_outcome,
                factual_outcome_mask,
                counterfactual_outcome,
                counterfactual_outcome_mask,
            )

            factual_round = np.asarray(
                json.loads(row["factual_action_round_delta"]), dtype=np.float32
            )
            factual_round_mask = np.asarray(
                json.loads(row["factual_action_round_delta_mask"]), dtype=np.bool_
            )
            counterfactual_round = np.asarray(
                json.loads(row["counterfactual_action_round_delta"]), dtype=np.float32
            )
            counterfactual_round_mask = np.asarray(
                json.loads(row["counterfactual_action_round_delta_mask"]), dtype=np.bool_
            )
            action_round_value_target, action_round_value_mask = (
                merge_factual_counterfactual(
                    factual_round,
                    factual_round_mask,
                    counterfactual_round,
                    counterfactual_round_mask,
                )
            )
            action_round_value_target = action_round_value_target / 10_000.0

            factual_rank = np.asarray(
                json.loads(row["factual_action_final_rank"]), dtype=np.int64
            )
            factual_score = np.asarray(
                json.loads(row["factual_action_final_score"]), dtype=np.int64
            )
            factual_hanchan_mask = np.asarray(
                json.loads(row["factual_action_hanchan_mask"]), dtype=np.bool_
            )
            counterfactual_rank = np.asarray(
                json.loads(row["counterfactual_action_final_rank"]), dtype=np.int64
            )
            counterfactual_score = np.asarray(
                json.loads(row["counterfactual_action_final_score"]), dtype=np.int64
            )
            counterfactual_hanchan_mask = np.asarray(
                json.loads(row["counterfactual_action_hanchan_mask"]), dtype=np.bool_
            )
            combined_rank, action_hanchan_value_mask = merge_factual_counterfactual(
                factual_rank,
                factual_hanchan_mask,
                counterfactual_rank,
                counterfactual_hanchan_mask,
            )
            combined_score, _ = merge_factual_counterfactual(
                factual_score,
                factual_hanchan_mask,
                counterfactual_score,
                counterfactual_hanchan_mask,
            )
            for action_index in range(action_count):
                if action_hanchan_value_mask[action_index]:
                    action_hanchan_value_target[action_index] = self.hanchan_utility(
                        int(combined_rank[action_index]),
                        int(combined_score[action_index]),
                    )
        else:
            action_outcome_target[policy_target] = selected_outcome_target
            action_outcome_mask[policy_target] = bool(
                label_masks.get("round_result", True)
            )
            action_round_value_target[policy_target] = float(deltas[0]) / 10_000.0
            action_round_value_mask[policy_target] = bool(
                label_masks.get("round_delta", [True] * 4)[0]
            )
            action_hanchan_value_target[policy_target] = self.hanchan_utility(
                int(row["final_rank"]), int(row["final_score"])
            )
            action_hanchan_value_mask[policy_target] = bool(
                label_masks.get("rank", True)
            )

        if is_v4:
            action_round_factual_mask = factual_round_mask
            action_round_counterfactual_mask = counterfactual_round_mask
            action_hanchan_factual_mask = factual_hanchan_mask
            action_hanchan_counterfactual_mask = counterfactual_hanchan_mask
            outcome_factual_mask = factual_outcome_mask
            outcome_counterfactual_mask = counterfactual_outcome_mask
        else:
            action_round_factual_mask = action_round_value_mask.copy()
            action_round_counterfactual_mask = np.zeros(action_count, dtype=np.bool_)
            action_hanchan_factual_mask = action_hanchan_value_mask.copy()
            action_hanchan_counterfactual_mask = np.zeros(action_count, dtype=np.bool_)
            outcome_factual_mask = action_outcome_mask.copy()
            outcome_counterfactual_mask = np.zeros(action_count, dtype=np.bool_)

        action_round_distribution_target = np.searchsorted(
            ROUND_VALUE_EDGES,
            action_round_value_target * 10_000.0,
        ).astype(np.int64)

        if is_v4:
            opponent_next_discard = np.asarray(
                json.loads(row["opponent_next_discard"]), dtype=np.int64
            )
            opponent_next_discard_mask = np.asarray(
                json.loads(row["opponent_next_discard_mask"]), dtype=np.bool_
            )
            opponent_next_response = np.asarray(
                json.loads(row["opponent_next_response"]), dtype=np.int64
            )
            opponent_next_response_mask = np.asarray(
                json.loads(row["opponent_next_response_mask"]), dtype=np.bool_
            )
            opponent_next_call_type = np.asarray(
                json.loads(row["opponent_next_call_type"]), dtype=np.int64
            )
            opponent_next_call_type_mask = np.asarray(
                json.loads(row["opponent_next_call_type_mask"]), dtype=np.bool_
            )
            opponent_tile_count = np.asarray(
                json.loads(row["opponent_tile_count"]), dtype=np.int64
            )
            opponent_tile_count_mask = np.asarray(
                json.loads(row["opponent_tile_count_mask"]), dtype=np.bool_
            )[:, None]
        else:
            opponent_next_discard = np.zeros(3, dtype=np.int64)
            opponent_next_discard_mask = np.zeros(3, dtype=np.bool_)
            opponent_next_response = np.zeros(3, dtype=np.int64)
            opponent_next_response_mask = np.zeros(3, dtype=np.bool_)
            opponent_next_call_type = np.zeros(3, dtype=np.int64)
            opponent_next_call_type_mask = np.zeros(3, dtype=np.bool_)
            opponent_tile_count = np.zeros((3, 34), dtype=np.int64)
            opponent_tile_count_mask = np.zeros((3, 1), dtype=np.bool_)
        score_edges = np.asarray(
            [2_000, 3_900, 5_200, 7_700, 8_000, 12_000, 18_000],
            dtype=np.int64,
        )
        response_types = {"pass": 0, "chi": 1, "pon": 1, "kan": 1, "ron": 2}
        call_types = {"chi": 0, "pon": 1, "kan": 2}
        riichi_mask = np.asarray(
            [action["type"] == "riichi" for action in legal], dtype=np.bool_
        )
        riichi_target = np.zeros(action_count, dtype=np.float32)
        if expert["type"] == "riichi":
            riichi_target[policy_target] = 1.0
        opponent_state_mask = np.asarray(
            label_masks.get("opponent_state", [True] * 3), dtype=np.bool_
        )
        return {
            "planes": encoded.planes,
            "rule_features": rule_features,
            "actions": encoded.action_features,
            "danger_features": encoded.danger_features[:, 1:, :],
            "event_types": sequence.event_types,
            "players": sequence.players,
            "tiles": sequence.tiles,
            "event_flags": sequence.flags,
            "event_mask": sequence.valid_mask,
            "memory_features": sequence.memory_features,
            "policy_target": policy_target,
            "policy_target_mask": bool(label_masks.get("policy", True)),
            "response_target": response_types.get(expert["type"], 0),
            "response_target_mask": expert["type"] in response_types and any(
                action["type"] == "pass" for action in legal
            ),
            "call_type_target": call_types.get(expert["type"], 0),
            "call_type_target_mask": expert["type"] in call_types,
            "riichi_target": riichi_target,
            "riichi_target_mask": riichi_mask,
            "rank_target": int(row["final_rank"]) - 1,
            "rank_target_mask": bool(label_masks.get("rank", True)),
            "score_target": (float(row["final_score"]) - 25_000.0) / 10_000.0,
            "score_target_mask": bool(label_masks.get("final_score", True)),
            "round_delta_target": np.asarray(deltas, dtype=np.float32) / 10_000.0,
            "round_delta_target_mask": np.asarray(
                label_masks.get("round_delta", [True] * 4), dtype=np.bool_
            ),
            "state_round_value_target": float(deltas[0]) / 10_000.0,
            "state_round_value_mask": bool(
                label_masks.get("round_delta", [True] * 4)[0]
            ),
            "state_hanchan_value_target": self.hanchan_utility(
                int(row["final_rank"]), int(row["final_score"])
            ),
            "state_hanchan_value_mask": bool(label_masks.get("rank", True)),
            "tenpai_target": np.asarray(tenpai, dtype=np.float32),
            "tenpai_target_mask": opponent_state_mask,
            "wait_target": wait_target,
            "wait_target_mask": opponent_state_mask[:, None] & (
                wait_target > 0
            ).any(axis=-1, keepdims=True),
            "shanten_target": np.asarray(
                [min(max(int(value), -1), 6) + 1 for value in shanten], dtype=np.int64
            ),
            "shanten_target_mask": opponent_state_mask,
            "future_target": future_target,
            "future_target_mask": bool(label_masks.get("future_draw", True)),
            "deal_in_target": deal_in,
            "deal_in_target_mask": deal_in_mask,
            "deal_in_score_target": np.searchsorted(score_edges, deal_in_points).astype(
                np.int64
            ),
            "deal_in_score_mask": deal_in_mask & deal_in.astype(np.bool_),
            "action_outcome_target": action_outcome_target,
            "action_outcome_target_mask": action_outcome_mask,
            "outcome_factual_mask": outcome_factual_mask,
            "outcome_counterfactual_mask": outcome_counterfactual_mask,
            "action_round_value_target": action_round_value_target,
            "action_round_value_mask": action_round_value_mask,
            "action_round_factual_mask": action_round_factual_mask,
            "action_round_counterfactual_mask": action_round_counterfactual_mask,
            "action_hanchan_value_target": action_hanchan_value_target,
            "action_hanchan_value_mask": action_hanchan_value_mask,
            "action_hanchan_factual_mask": action_hanchan_factual_mask,
            "action_hanchan_counterfactual_mask": action_hanchan_counterfactual_mask,
            "action_round_distribution_target": action_round_distribution_target,
            "action_round_distribution_mask": action_round_value_mask,
            "opponent_next_discard_target": opponent_next_discard,
            "opponent_next_discard_mask": opponent_next_discard_mask,
            "opponent_next_response_target": opponent_next_response,
            "opponent_next_response_mask": opponent_next_response_mask,
            "opponent_next_call_type_target": opponent_next_call_type,
            "opponent_next_call_type_mask": opponent_next_call_type_mask,
            "opponent_tile_count_target": opponent_tile_count,
            "opponent_tile_count_mask": opponent_tile_count_mask,
            "log_id": row["log_id"],
        }


def collate_sequence_decisions(batch: list[dict[str, Any]]) -> dict[str, Tensor]:
    max_actions = max(item["actions"].shape[0] for item in batch)
    actions = np.zeros((len(batch), max_actions, ACTION_FEATURES), dtype=np.float32)
    danger_features = np.zeros(
        (len(batch), max_actions, 3, DANGER_FEATURES), dtype=np.float32
    )
    action_mask = np.zeros((len(batch), max_actions), dtype=np.bool_)
    riichi_target = np.zeros((len(batch), max_actions), dtype=np.float32)
    riichi_target_mask = np.zeros((len(batch), max_actions), dtype=np.bool_)
    deal_in_target = np.zeros((len(batch), max_actions, 3), dtype=np.float32)
    deal_in_target_mask = np.zeros((len(batch), max_actions, 3), dtype=np.bool_)
    deal_in_score_target = np.zeros((len(batch), max_actions, 3), dtype=np.int64)
    deal_in_score_mask = np.zeros((len(batch), max_actions, 3), dtype=np.bool_)
    action_outcome_target = np.zeros((len(batch), max_actions), dtype=np.int64)
    action_outcome_target_mask = np.zeros((len(batch), max_actions), dtype=np.bool_)
    outcome_factual_mask = np.zeros((len(batch), max_actions), dtype=np.bool_)
    outcome_counterfactual_mask = np.zeros((len(batch), max_actions), dtype=np.bool_)
    action_round_value_target = np.zeros((len(batch), max_actions), dtype=np.float32)
    action_round_value_mask = np.zeros((len(batch), max_actions), dtype=np.bool_)
    action_round_factual_mask = np.zeros((len(batch), max_actions), dtype=np.bool_)
    action_round_counterfactual_mask = np.zeros((len(batch), max_actions), dtype=np.bool_)
    action_hanchan_value_target = np.zeros((len(batch), max_actions), dtype=np.float32)
    action_hanchan_value_mask = np.zeros((len(batch), max_actions), dtype=np.bool_)
    action_hanchan_factual_mask = np.zeros((len(batch), max_actions), dtype=np.bool_)
    action_hanchan_counterfactual_mask = np.zeros((len(batch), max_actions), dtype=np.bool_)
    action_round_distribution_target = np.zeros((len(batch), max_actions), dtype=np.int64)
    action_round_distribution_mask = np.zeros((len(batch), max_actions), dtype=np.bool_)
    for index, item in enumerate(batch):
        count = item["actions"].shape[0]
        actions[index, :count] = item["actions"]
        danger_features[index, :count] = item["danger_features"]
        action_mask[index, :count] = True
        riichi_target[index, :count] = item["riichi_target"]
        riichi_target_mask[index, :count] = item["riichi_target_mask"]
        deal_in_target[index, :count] = item["deal_in_target"]
        deal_in_target_mask[index, :count] = item["deal_in_target_mask"]
        deal_in_score_target[index, :count] = item["deal_in_score_target"]
        deal_in_score_mask[index, :count] = item["deal_in_score_mask"]
        action_outcome_target[index, :count] = item["action_outcome_target"]
        action_outcome_target_mask[index, :count] = item["action_outcome_target_mask"]
        outcome_factual_mask[index, :count] = item["outcome_factual_mask"]
        outcome_counterfactual_mask[index, :count] = item["outcome_counterfactual_mask"]
        action_round_value_target[index, :count] = item["action_round_value_target"]
        action_round_value_mask[index, :count] = item["action_round_value_mask"]
        action_round_factual_mask[index, :count] = item["action_round_factual_mask"]
        action_round_counterfactual_mask[index, :count] = item["action_round_counterfactual_mask"]
        action_hanchan_value_target[index, :count] = item["action_hanchan_value_target"]
        action_hanchan_value_mask[index, :count] = item["action_hanchan_value_mask"]
        action_hanchan_factual_mask[index, :count] = item["action_hanchan_factual_mask"]
        action_hanchan_counterfactual_mask[index, :count] = item["action_hanchan_counterfactual_mask"]
        action_round_distribution_target[index, :count] = item[
            "action_round_distribution_target"
        ]
        action_round_distribution_mask[index, :count] = item[
            "action_round_distribution_mask"
        ]
    stack = lambda key: torch.from_numpy(np.stack([item[key] for item in batch]))
    return {
        "planes": stack("planes"),
        "rule_features": stack("rule_features"),
        "actions": torch.from_numpy(actions),
        "danger_features": torch.from_numpy(danger_features),
        "action_mask": torch.from_numpy(action_mask),
        "event_types": stack("event_types"),
        "players": stack("players"),
        "tiles": stack("tiles"),
        "event_flags": stack("event_flags"),
        "event_mask": stack("event_mask"),
        "memory_features": stack("memory_features"),
        "policy_target": torch.tensor([item["policy_target"] for item in batch]),
        "policy_target_mask": torch.tensor(
            [item["policy_target_mask"] for item in batch], dtype=torch.bool
        ),
        "response_target": torch.tensor([item["response_target"] for item in batch]),
        "response_target_mask": torch.tensor(
            [item["response_target_mask"] for item in batch], dtype=torch.bool
        ),
        "call_type_target": torch.tensor([item["call_type_target"] for item in batch]),
        "call_type_target_mask": torch.tensor(
            [item["call_type_target_mask"] for item in batch], dtype=torch.bool
        ),
        "riichi_target": torch.from_numpy(riichi_target),
        "riichi_target_mask": torch.from_numpy(riichi_target_mask),
        "rank_target": torch.tensor([item["rank_target"] for item in batch]),
        "rank_target_mask": torch.tensor(
            [item["rank_target_mask"] for item in batch], dtype=torch.bool
        ),
        "score_target": torch.tensor([item["score_target"] for item in batch]),
        "score_target_mask": torch.tensor(
            [item["score_target_mask"] for item in batch], dtype=torch.bool
        ),
        "round_delta_target": stack("round_delta_target"),
        "round_delta_target_mask": stack("round_delta_target_mask"),
        "state_round_value_target": torch.tensor(
            [item["state_round_value_target"] for item in batch]
        ),
        "state_round_value_mask": torch.tensor(
            [item["state_round_value_mask"] for item in batch], dtype=torch.bool
        ),
        "state_hanchan_value_target": torch.tensor(
            [item["state_hanchan_value_target"] for item in batch]
        ),
        "state_hanchan_value_mask": torch.tensor(
            [item["state_hanchan_value_mask"] for item in batch], dtype=torch.bool
        ),
        "tenpai_target": stack("tenpai_target"),
        "tenpai_target_mask": stack("tenpai_target_mask"),
        "wait_target": stack("wait_target"),
        "wait_target_mask": stack("wait_target_mask"),
        "shanten_target": stack("shanten_target"),
        "shanten_target_mask": stack("shanten_target_mask"),
        "future_target": stack("future_target"),
        "future_target_mask": torch.tensor(
            [item["future_target_mask"] for item in batch], dtype=torch.bool
        ),
        "deal_in_target": torch.from_numpy(deal_in_target),
        "deal_in_target_mask": torch.from_numpy(deal_in_target_mask),
        "deal_in_score_target": torch.from_numpy(deal_in_score_target),
        "deal_in_score_mask": torch.from_numpy(deal_in_score_mask),
        "action_outcome_target": torch.from_numpy(action_outcome_target),
        "action_outcome_target_mask": torch.from_numpy(action_outcome_target_mask),
        "outcome_factual_mask": torch.from_numpy(outcome_factual_mask),
        "outcome_counterfactual_mask": torch.from_numpy(outcome_counterfactual_mask),
        "action_round_value_target": torch.from_numpy(action_round_value_target),
        "action_round_value_mask": torch.from_numpy(action_round_value_mask),
        "action_round_factual_mask": torch.from_numpy(action_round_factual_mask),
        "action_round_counterfactual_mask": torch.from_numpy(
            action_round_counterfactual_mask
        ),
        "action_hanchan_value_target": torch.from_numpy(action_hanchan_value_target),
        "action_hanchan_value_mask": torch.from_numpy(action_hanchan_value_mask),
        "action_hanchan_factual_mask": torch.from_numpy(action_hanchan_factual_mask),
        "action_hanchan_counterfactual_mask": torch.from_numpy(
            action_hanchan_counterfactual_mask
        ),
        "action_round_distribution_target": torch.from_numpy(
            action_round_distribution_target
        ),
        "action_round_distribution_mask": torch.from_numpy(
            action_round_distribution_mask
        ),
        "opponent_next_discard_target": stack("opponent_next_discard_target"),
        "opponent_next_discard_mask": stack("opponent_next_discard_mask"),
        "opponent_next_response_target": stack("opponent_next_response_target"),
        "opponent_next_response_mask": stack("opponent_next_response_mask"),
        "opponent_next_call_type_target": stack("opponent_next_call_type_target"),
        "opponent_next_call_type_mask": stack("opponent_next_call_type_mask"),
        "opponent_tile_count_target": stack("opponent_tile_count_target"),
        "opponent_tile_count_mask": stack("opponent_tile_count_mask"),
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
        if config.scheduler_t_max_epochs <= 0:
            raise ValueError("scheduler_t_max_epochs must be positive")
        if not 0.0 <= config.minimum_learning_rate_ratio <= 1.0:
            raise ValueError("minimum_learning_rate_ratio must be in [0, 1]")
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.scheduler_t_max_epochs,
            eta_min=config.learning_rate * config.minimum_learning_rate_ratio,
        )
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        self.effective_positive_weights = {
            "tenpai": config.tenpai_positive_weight,
            "wait": config.wait_positive_weight,
            "deal_in": config.deal_in_positive_weight,
        }
        self.global_step = 0
        self.dataset_manifest_hashes: dict[str, str] = {}
        self.best_validation_loss = float("inf")

    def restore_training_state(
        self, checkpoint: dict[str, Any], *, restore_optimizer: bool = True
    ) -> int:
        """Restore continuation state and return the completed epoch count."""

        if restore_optimizer:
            optimizer_state = checkpoint.get("optimizer_state")
            if optimizer_state is not None:
                self.optimizer.load_state_dict(optimizer_state)
            scheduler_state = checkpoint.get("scheduler_state")
            if scheduler_state is not None:
                self.scheduler.load_state_dict(scheduler_state)
        rng = checkpoint.get("rng_state")
        if isinstance(rng, dict):
            torch_state = rng.get("torch")
            python_state = rng.get("python")
            numpy_state = rng.get("numpy")
            if torch_state is not None:
                torch.set_rng_state(torch_state.cpu())
            if python_state is not None:
                random.setstate(python_state)
            if numpy_state is not None:
                np.random.set_state(numpy_state)
        self.global_step = int(checkpoint.get("global_step", 0))
        if restore_optimizer:
            selection = checkpoint.get("selection") or {}
            self.best_validation_loss = float(
                selection.get("validation_loss", float("inf"))
            )
        return int(checkpoint.get("epoch", 0))

    def fit(
        self,
        train_set: Dataset,
        validation_set: Dataset,
        checkpoint_path: str | Path,
        model_id: str,
        dataset_fingerprint: str,
        *,
        start_epoch: int = 0,
        dataset_manifest_hashes: Mapping[str, str] | None = None,
    ) -> list[dict[str, float]]:
        if start_epoch < 0:
            raise ValueError("start_epoch cannot be negative")
        self.dataset_manifest_hashes = dict(dataset_manifest_hashes or {})
        if self.config.auto_positive_weights:
            self.effective_positive_weights = estimate_sequence_positive_weights(
                train_set,
                fallback=self.effective_positive_weights,
                maximum=self.config.maximum_positive_weight,
            )
        validation_loader = DataLoader(
            validation_set, batch_size=self.config.batch_size, shuffle=False,
            num_workers=self.config.num_workers, collate_fn=collate_sequence_decisions,
        )
        history: list[dict[str, float]] = []
        best_validation_loss = self.best_validation_loss
        checkpoint_path = Path(checkpoint_path)
        last_checkpoint_path = checkpoint_path.with_name(
            f"{checkpoint_path.stem}-last{checkpoint_path.suffix}"
        )
        for epoch in range(start_epoch + 1, start_epoch + self.config.epochs + 1):
            # Epoch-local seeding makes an interrupted/resumed run use the exact
            # same shuffle as an uninterrupted run for the same epoch number.
            train_loader = DataLoader(
                train_set,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=self.config.num_workers,
                collate_fn=collate_sequence_decisions,
                generator=torch.Generator().manual_seed(self.config.seed + epoch),
            )
            train_metrics = self._run_epoch(train_loader, training=True)
            validation_metrics = self._run_epoch(validation_loader, training=False)
            learning_rate = float(self.optimizer.param_groups[0]["lr"])
            self.scheduler.step()
            metrics = {f"train_{key}": value for key, value in train_metrics.items()}
            metrics.update({f"val_{key}": value for key, value in validation_metrics.items()})
            metrics["epoch"] = float(epoch)
            metrics["learning_rate"] = learning_rate
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
                self.best_validation_loss = validation_loss
                self._save(
                    checkpoint_path,
                    model_id,
                    dataset_fingerprint,
                    epoch,
                    validation_loss=validation_loss,
                    selected_as_best=True,
                )
        return history

    def _forward(self, batch: dict[str, Tensor]) -> MahjongModelOutput:
        return self.model(
            batch["planes"], batch["actions"], batch["event_types"],
            batch["players"], batch["tiles"], batch["event_flags"],
            batch["event_mask"], batch["action_mask"],
            batch["danger_features"],
            batch["memory_features"],
            batch["rule_features"],
        )

    def offline_rl_signals(
        self, output: MahjongModelOutput, batch: Mapping[str, Tensor]
    ) -> OfflineRLSignals:
        return build_offline_rl_signals(output, batch)

    def _run_epoch(self, loader: DataLoader, training: bool) -> dict[str, float]:
        self.model.train(training)
        totals = {
            "loss": 0.0,
            "policy_accuracy": 0.0,
            "policy_valid": 0.0,
            "tenpai_accuracy": 0.0,
            "tenpai_valid": 0.0,
            "tenpai_true_positive": 0.0,
            "tenpai_predicted_positive": 0.0,
            "tenpai_actual_positive": 0.0,
            "wait_top1_hit": 0.0,
            "wait_positive_rows": 0.0,
            "deal_in_brier": 0.0,
            "deal_in_valid": 0.0,
            "deal_in_true_positive": 0.0,
            "deal_in_predicted_positive": 0.0,
            "deal_in_actual_positive": 0.0,
            "outcome_accuracy": 0.0,
            "outcome_valid": 0.0,
            **{f"loss_{name}": 0.0 for name in (
                "policy", "rank", "score", "round", "tenpai", "wait",
                "shanten", "draw", "deal", "dealpoint", "result",
                "response", "call_type", "riichi", "state_round_value",
                "state_hanchan_value", "action_round_value",
                "action_hanchan_value", "action_round_distribution",
                "opponent_discard", "opponent_response", "opponent_call_type",
                "opponent_tile_count", "belief_consistency",
            )},
            **{f"gradient_norm_{name}": 0.0 for name in (
                "shared", "policy", "value", "belief", "risk", "hierarchy",
                "action_value", "opponent_action", "rule_memory"
            )},
        }
        examples = 0
        for raw_batch in loader:
            batch = {key: value.to(self.config.device) for key, value in raw_batch.items()}
            with torch.set_grad_enabled(training):
                output = self._forward(batch)
                policy_rows = batch["policy_target_mask"]
                policy_loss = (
                    masked_policy_cross_entropy(
                        output["policy_logits"][policy_rows],
                        batch["policy_target"][policy_rows],
                        batch["action_mask"][policy_rows],
                        self.config.policy_label_smoothing,
                    )
                    if bool(policy_rows.any())
                    else output["policy_logits"][:, :1].sum() * 0.0
                )
                rank_loss = masked_cross_entropy(
                    output["rank_logits"], batch["rank_target"], batch["rank_target_mask"]
                )
                score_loss = masked_mean(
                    F.smooth_l1_loss(
                        output["final_score"], batch["score_target"], reduction="none"
                    ),
                    batch["score_target_mask"],
                )
                delta_loss = masked_mean(
                    F.smooth_l1_loss(
                        output["round_delta"],
                        batch["round_delta_target"],
                        reduction="none",
                    ),
                    batch["round_delta_target_mask"],
                )
                tenpai_loss = masked_binary_cross_entropy(
                    output["tenpai_logits"],
                    batch["tenpai_target"],
                    batch["tenpai_target_mask"],
                    positive_weight=self.effective_positive_weights["tenpai"],
                )
                wait_logits = output["wait_logits"]
                wait_targets = batch["wait_target"]
                wait_loss = masked_binary_cross_entropy(
                    wait_logits,
                    wait_targets,
                    batch["wait_target_mask"],
                    positive_weight=self.effective_positive_weights["wait"],
                )
                shanten_loss = masked_cross_entropy(
                    output["shanten_logits"],
                    batch["shanten_target"],
                    batch["shanten_target_mask"],
                )
                future_loss = masked_binary_cross_entropy(
                    output["future_draw_logits"],
                    batch["future_target"],
                    batch["future_target_mask"].unsqueeze(-1),
                )
                state_round_value_loss = masked_mean(
                    F.smooth_l1_loss(
                        output["state_round_value"],
                        batch["state_round_value_target"],
                        reduction="none",
                    ),
                    batch["state_round_value_mask"],
                )
                state_hanchan_value_loss = masked_mean(
                    F.smooth_l1_loss(
                        output["state_hanchan_value"],
                        batch["state_hanchan_value_target"],
                        reduction="none",
                    ),
                    batch["state_hanchan_value_mask"],
                )
                consistency_loss = belief_consistency_loss(
                    output["tenpai_logits"],
                    output["wait_logits"],
                    output["shanten_logits"],
                    batch["tenpai_target_mask"] & batch["shanten_target_mask"],
                )
                cfg = self.config
                zero = output["policy_logits"][:, :1].sum() * 0.0
                danger_loss = deal_in_score_loss = outcome_loss = zero
                response_loss = call_type_loss = riichi_loss = zero
                action_round_value_loss = action_hanchan_value_loss = zero
                action_round_distribution_loss = zero
                opponent_discard_loss = opponent_response_loss = zero
                opponent_call_type_loss = opponent_tile_count_loss = zero
                loss = (
                    policy_loss + cfg.rank_loss_weight * rank_loss
                    + cfg.score_loss_weight * score_loss
                    + cfg.round_delta_loss_weight * delta_loss
                    + cfg.tenpai_loss_weight * tenpai_loss
                    + cfg.wait_loss_weight * wait_loss
                    + cfg.shanten_loss_weight * shanten_loss
                    + cfg.future_draw_loss_weight * future_loss
                    + cfg.state_round_value_loss_weight * state_round_value_loss
                    + cfg.state_hanchan_value_loss_weight * state_hanchan_value_loss
                    + cfg.belief_consistency_loss_weight * consistency_loss
                )
                selected_danger: Tensor | None = None
                if self.model.config.decision_heads:
                    batch_indexes = torch.arange(
                        batch["planes"].shape[0], device=self.config.device
                    )
                    danger_loss = masked_binary_cross_entropy(
                        output["deal_in_logits"],
                        batch["deal_in_target"],
                        batch["deal_in_target_mask"],
                        positive_weight=self.effective_positive_weights["deal_in"],
                    )
                    deal_in_score_loss = masked_cross_entropy(
                        output["deal_in_point_logits"],
                        batch["deal_in_score_target"],
                        batch["deal_in_score_mask"],
                    )
                    selected_danger = output["deal_in_logits"][
                        batch_indexes, batch["policy_target"]
                    ]
                    outcome_loss = masked_cross_entropy(
                        output["round_result_logits"],
                        batch["action_outcome_target"],
                        batch["action_outcome_target_mask"],
                    )
                    response_loss = masked_cross_entropy(
                        output["response_logits"],
                        batch["response_target"],
                        batch["response_target_mask"],
                    )
                    call_type_loss = masked_cross_entropy(
                        output["call_type_logits"],
                        batch["call_type_target"],
                        batch["call_type_target_mask"],
                    )
                    riichi_loss = masked_binary_cross_entropy(
                        output["riichi_logits"],
                        batch["riichi_target"],
                        batch["riichi_target_mask"],
                    )
                    action_round_value_loss = masked_mean(
                        F.smooth_l1_loss(
                            output["action_round_value"],
                            batch["action_round_value_target"],
                            reduction="none",
                        ),
                        batch["action_round_value_mask"],
                    )
                    action_hanchan_value_loss = masked_mean(
                        F.smooth_l1_loss(
                            output["action_hanchan_value"],
                            batch["action_hanchan_value_target"],
                            reduction="none",
                        ),
                        batch["action_hanchan_value_mask"],
                    )
                    action_round_distribution_loss = masked_cross_entropy(
                        output["action_round_distribution_logits"],
                        batch["action_round_distribution_target"],
                        batch["action_round_distribution_mask"],
                    )
                    opponent_discard_loss = masked_cross_entropy(
                        output["opponent_next_discard_logits"],
                        batch["opponent_next_discard_target"],
                        batch["opponent_next_discard_mask"],
                    )
                    opponent_response_loss = masked_cross_entropy(
                        output["opponent_response_logits"],
                        batch["opponent_next_response_target"],
                        batch["opponent_next_response_mask"],
                    )
                    opponent_call_type_loss = masked_cross_entropy(
                        output["opponent_call_type_logits"],
                        batch["opponent_next_call_type_target"],
                        batch["opponent_next_call_type_mask"],
                    )
                    opponent_tile_count_loss = masked_cross_entropy(
                        output["opponent_tile_count_logits"],
                        batch["opponent_tile_count_target"],
                        batch["opponent_tile_count_mask"],
                    )
                    loss = (
                        loss
                        + cfg.deal_in_loss_weight * danger_loss
                        + cfg.deal_in_score_loss_weight * deal_in_score_loss
                        + cfg.outcome_loss_weight * outcome_loss
                        + cfg.response_loss_weight * response_loss
                        + cfg.call_type_loss_weight * call_type_loss
                        + cfg.riichi_loss_weight * riichi_loss
                        + cfg.action_round_value_loss_weight * action_round_value_loss
                        + cfg.action_hanchan_value_loss_weight * action_hanchan_value_loss
                        + cfg.action_round_distribution_loss_weight
                        * action_round_distribution_loss
                        + cfg.opponent_discard_loss_weight * opponent_discard_loss
                        + cfg.opponent_response_loss_weight * opponent_response_loss
                        + cfg.opponent_call_type_loss_weight * opponent_call_type_loss
                        + cfg.opponent_tile_count_loss_weight * opponent_tile_count_loss
                    )
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    gradient_norms = _gradient_norms(self.model)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), cfg.gradient_clip_norm
                    )
                    self.optimizer.step()
                    self.global_step += 1
                else:
                    gradient_norms = {
                        name: 0.0
                        for name in (
                            "shared", "policy", "value", "belief", "risk", "hierarchy",
                            "action_value", "opponent_action", "rule_memory",
                        )
                    }
            count = batch["planes"].shape[0]
            examples += count
            totals["loss"] += float(loss.detach()) * count
            head_losses = {
                "policy": policy_loss,
                "rank": rank_loss,
                "score": score_loss,
                "round": delta_loss,
                "tenpai": tenpai_loss,
                "wait": wait_loss,
                "shanten": shanten_loss,
                "draw": future_loss,
                "deal": danger_loss,
                "dealpoint": deal_in_score_loss,
                "result": outcome_loss,
                "response": response_loss,
                "call_type": call_type_loss,
                "riichi": riichi_loss,
                "state_round_value": state_round_value_loss,
                "state_hanchan_value": state_hanchan_value_loss,
                "action_round_value": action_round_value_loss,
                "action_hanchan_value": action_hanchan_value_loss,
                "action_round_distribution": action_round_distribution_loss,
                "opponent_discard": opponent_discard_loss,
                "opponent_response": opponent_response_loss,
                "opponent_call_type": opponent_call_type_loss,
                "opponent_tile_count": opponent_tile_count_loss,
                "belief_consistency": consistency_loss,
            }
            for name, value in head_losses.items():
                totals[f"loss_{name}"] += float(value.detach()) * count
            for name, value in gradient_norms.items():
                totals[f"gradient_norm_{name}"] += value * count
            policy_valid = batch["policy_target_mask"]
            totals["policy_accuracy"] += float(
                ((output["policy_logits"].argmax(1) == batch["policy_target"]) & policy_valid).sum()
            )
            totals["policy_valid"] += float(policy_valid.sum())
            predicted_tenpai = output["tenpai_logits"] >= 0
            actual_tenpai = batch["tenpai_target"].bool()
            tenpai_valid = batch["tenpai_target_mask"]
            totals["tenpai_accuracy"] += float(
                ((predicted_tenpai == actual_tenpai) & tenpai_valid).sum()
            )
            totals["tenpai_valid"] += float(tenpai_valid.sum())
            totals["tenpai_true_positive"] += float(
                (predicted_tenpai & actual_tenpai & tenpai_valid).sum()
            )
            totals["tenpai_predicted_positive"] += float(
                (predicted_tenpai & tenpai_valid).sum()
            )
            totals["tenpai_actual_positive"] += float(
                (actual_tenpai & tenpai_valid).sum()
            )
            wait_valid_rows = batch["wait_target_mask"].any(dim=-1)
            positive_wait_rows = (wait_targets.sum(dim=-1) > 0) & wait_valid_rows
            if bool(positive_wait_rows.any()):
                top_wait = wait_logits.argmax(dim=-1)
                top_hit = wait_targets.gather(
                    -1, top_wait.unsqueeze(-1)
                ).squeeze(-1).bool()
                totals["wait_top1_hit"] += float(
                    (top_hit & positive_wait_rows).sum()
                )
                totals["wait_positive_rows"] += float(positive_wait_rows.sum())
            if selected_danger is not None:
                batch_indexes = torch.arange(count, device=self.config.device)
                danger_probabilities = torch.sigmoid(selected_danger.detach())
                danger_targets = batch["deal_in_target"][
                    batch_indexes, batch["policy_target"]
                ].bool()
                danger_valid = batch["deal_in_target_mask"][
                    batch_indexes, batch["policy_target"]
                ]
                danger_predictions = danger_probabilities >= 0.5
                totals["deal_in_brier"] += float(
                    ((danger_probabilities - danger_targets.to(danger_probabilities.dtype)).square()
                     * danger_valid).sum()
                )
                totals["deal_in_valid"] += float(danger_valid.sum())
                totals["deal_in_true_positive"] += float(
                    (danger_predictions & danger_targets & danger_valid).sum()
                )
                totals["deal_in_predicted_positive"] += float(
                    (danger_predictions & danger_valid).sum()
                )
                totals["deal_in_actual_positive"] += float(
                    (danger_targets & danger_valid).sum()
                )
                outcome_valid = batch["action_outcome_target_mask"]
                outcome_prediction = output["round_result_logits"].detach().argmax(-1)
                totals["outcome_accuracy"] += float(
                    ((outcome_prediction == batch["action_outcome_target"]) & outcome_valid).sum()
                )
                totals["outcome_valid"] += float(outcome_valid.sum())
        if examples == 0:
            raise ValueError("empty data loader")
        metrics = {
            "loss": totals["loss"] / examples,
            "policy_accuracy": totals["policy_accuracy"] / max(totals["policy_valid"], 1.0),
            "tenpai_accuracy": totals["tenpai_accuracy"] / max(totals["tenpai_valid"], 1.0),
            "deal_in_brier": totals["deal_in_brier"] / max(totals["deal_in_valid"], 1.0),
            "outcome_accuracy": totals["outcome_accuracy"] / max(totals["outcome_valid"], 1.0),
        }
        metrics.update(
            {
                "tenpai_precision": totals["tenpai_true_positive"]
                / max(totals["tenpai_predicted_positive"], 1.0),
                "tenpai_recall": totals["tenpai_true_positive"]
                / max(totals["tenpai_actual_positive"], 1.0),
                "wait_top1_hit_rate": totals["wait_top1_hit"]
                / max(totals["wait_positive_rows"], 1.0),
                "deal_in_precision": totals["deal_in_true_positive"]
                / max(totals["deal_in_predicted_positive"], 1.0),
                "deal_in_recall": totals["deal_in_true_positive"]
                / max(totals["deal_in_actual_positive"], 1.0),
            }
        )
        metrics.update(
            {
                key: value / examples
                for key, value in totals.items()
                if key.startswith("loss_") or key.startswith("gradient_norm_")
            }
        )
        return metrics

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
                "scheduler_state": self.scheduler.state_dict(),
                "global_step": self.global_step,
                "architecture_version": SEQUENCE_ARCHITECTURE_VERSION,
                "model_output_version": MODEL_OUTPUT_VERSION,
                "decision_schema_version": SEQUENCE_DATASET_FORMAT,
                "dataset_version": SEQUENCE_DATASET_FORMAT,
                "feature_version": f"{FeatureEncoder.version}+{SEQUENCE_ENCODER_VERSION}",
                "label_version": SEQUENCE_LABEL_VERSION,
                "git_commit": _git_commit(),
                "seed": self.config.seed,
                "hyperparameters": asdict(self.config),
                "dataset_manifest_hashes": dict(self.dataset_manifest_hashes),
                "rng_state": {
                    "torch": torch.get_rng_state(),
                    "python": random.getstate(),
                    "numpy": np.random.get_state(),
                },
                "epoch": epoch,
                "training_config": asdict(self.config),
                "effective_positive_weights": dict(self.effective_positive_weights),
                "selection": {
                    "validation_loss": validation_loss,
                    "selected_as_best": selected_as_best,
                },
            },
            path,
        )


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _gradient_norms(model: SequenceMahjongNetwork) -> dict[str, float]:
    groups = {
        "policy": ("action_encoder", "policy_query", "policy_bias"),
        "action_value": (
            "action_value_trunk", "action_round_value_head",
            "action_hanchan_value_head", "action_round_distribution_head",
        ),
        "opponent_action": (
            "opponent_next_discard_head", "opponent_response_head",
            "opponent_call_type_head", "opponent_tile_count_head",
        ),
        "rule_memory": ("rule_projection", "memory_projection", "rule_gate", "memory_gate"),
        "value": (
            "rank_head", "score_head", "round_delta_head", "action_outcome_head",
            "state_round_value_head", "state_hanchan_value_head",
        ),
        "belief": (
            "opponent_tenpai", "opponent_wait", "opponent_shanten",
            "opponent_projection", "opponent_temporal", "opponent_event",
            "opponent_seat_embedding", "future_draw_head",
        ),
        "risk": ("deal_in_head", "deal_in_point_head"),
        "hierarchy": ("response_head", "call_type_head", "riichi_head"),
    }
    squares = {name: 0.0 for name in ("shared", *groups)}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        group = next(
            (key for key, prefixes in groups.items() if name.startswith(prefixes)),
            "shared",
        )
        squares[group] += float(parameter.grad.detach().square().sum())
    return {name: value**0.5 for name, value in squares.items()}
