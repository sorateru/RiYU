#!/usr/bin/env python3
"""Evaluate a sequence checkpoint without silently opening the final holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from mahjong_ai.ai import SequenceMahjongNetwork, SequenceModelConfig
from mahjong_ai.ai.opponent import expected_calibration_error
from mahjong_ai.training import (
    ParquetSequenceDataset,
    collate_sequence_decisions,
    split_sequence_train_validation_holdout,
)


SCORE_CENTERS = torch.tensor(
    [1_000.0, 2_900.0, 4_550.0, 6_450.0, 7_850.0, 10_000.0, 15_000.0, 24_000.0]
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("datasets", nargs="+")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--holdout-fraction", type=float, default=0.1)
    parser.add_argument(
        "--final-holdout",
        action="store_true",
        help="evaluate the sealed final split instead of validation",
    )
    parser.add_argument(
        "--acknowledge-final-evaluation",
        action="store_true",
        help="required with --final-holdout to prevent accidental repeated tuning",
    )
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("batch size must be positive")
    if args.final_holdout and not args.acknowledge_final_evaluation:
        parser.error("--final-holdout requires --acknowledge-final-evaluation")
    if args.acknowledge_final_evaluation and not args.final_holdout:
        parser.error("--acknowledge-final-evaluation only applies to --final-holdout")

    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    if checkpoint.get("model_family") != "sequence-transformer-v2":
        raise ValueError("checkpoint is not sequence-transformer-v2")
    model = SequenceMahjongNetwork(SequenceModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state"])
    model.to(args.device).eval()
    dataset = ParquetSequenceDataset(args.datasets)
    if model.config.decision_heads and not dataset.has_decision_labels:
        raise ValueError("decision-head evaluation requires newly converted labelled shards")
    _, validation, holdout = split_sequence_train_validation_holdout(
        dataset, args.validation_fraction, args.holdout_fraction
    )
    selected = holdout if args.final_holdout else validation
    split_name = "final_holdout" if args.final_holdout else "validation"
    loader = DataLoader(
        selected,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_sequence_decisions,
    )
    totals = {
        "examples": 0.0,
        "policy_correct": 0.0,
        "policy_nll": 0.0,
        "rank_absolute_error": 0.0,
        "rank_nll": 0.0,
        "rank_brier": 0.0,
        "score_absolute_error_points": 0.0,
        "round_delta_absolute_error_points": 0.0,
        "tenpai_correct": 0.0,
        "tenpai_true_positive": 0.0,
        "tenpai_true_negative": 0.0,
        "tenpai_predicted_positive": 0.0,
        "tenpai_actual_positive": 0.0,
        "tenpai_brier": 0.0,
        "wait_brier": 0.0,
        "wait_true_positive": 0.0,
        "wait_predicted_positive": 0.0,
        "wait_actual_positive": 0.0,
        "wait_top1_hit": 0.0,
        "wait_tenpai_rows": 0.0,
        "shanten_correct": 0.0,
        "shanten_absolute_error": 0.0,
        "future_top5_hit": 0.0,
        "future_actual_positive": 0.0,
        "deal_in_score_correct": 0.0,
        "deal_in_score_absolute_error_points": 0.0,
        "deal_in_positive_cells": 0.0,
        "outcome_correct": 0.0,
        "outcome_nll": 0.0,
        "outcome_brier": 0.0,
    }
    danger_probabilities: list[torch.Tensor] = []
    danger_targets: list[torch.Tensor] = []
    danger_probabilities_by_opponent: list[list[torch.Tensor]] = [[], [], []]
    danger_targets_by_opponent: list[list[torch.Tensor]] = [[], [], []]
    rank_probabilities_all: list[torch.Tensor] = []
    rank_targets_all: list[torch.Tensor] = []
    outcome_probabilities_all: list[torch.Tensor] = []
    outcome_targets_all: list[torch.Tensor] = []
    opponent = torch.tensor([False, True, True, True], device=args.device)
    centers = SCORE_CENTERS.to(args.device)
    with torch.inference_mode():
        for raw in loader:
            batch = {key: value.to(args.device) for key, value in raw.items()}
            output = model(
                batch["planes"], batch["actions"], batch["event_types"],
                batch["players"], batch["tiles"], batch["event_flags"],
                batch["event_mask"], batch["action_mask"],
                batch["danger_features"],
            )
            count = batch["planes"].shape[0]
            totals["examples"] += count
            totals["policy_correct"] += float(
                (output["policy_logits"].argmax(1) == batch["policy_target"]).sum()
            )
            totals["policy_nll"] += float(
                F.cross_entropy(output["policy_logits"], batch["policy_target"], reduction="sum")
            )
            rank_probability = torch.softmax(output["rank_logits"], dim=-1)
            rank_probabilities_all.append(rank_probability.cpu())
            rank_targets_all.append(batch["rank_target"].cpu())
            rank_one_hot = F.one_hot(batch["rank_target"], 4).float()
            totals["rank_absolute_error"] += float(
                (output["rank_logits"].argmax(1) - batch["rank_target"]).abs().sum()
            )
            totals["rank_nll"] += float(
                F.cross_entropy(output["rank_logits"], batch["rank_target"], reduction="sum")
            )
            totals["rank_brier"] += float((rank_probability - rank_one_hot).square().sum())
            totals["score_absolute_error_points"] += float(
                (output["score_value"] - batch["score_target"]).abs().sum() * 10_000.0
            )
            totals["round_delta_absolute_error_points"] += float(
                (output["round_delta_values"] - batch["round_delta_target"]).abs().sum()
                * 10_000.0
            )

            tenpai_probability = torch.sigmoid(output["opponent_tenpai_logits"][:, opponent])
            tenpai_target = batch["tenpai_target"][:, opponent]
            tenpai_prediction = tenpai_probability >= 0.5
            totals["tenpai_correct"] += float((tenpai_prediction == tenpai_target.bool()).sum())
            totals["tenpai_true_positive"] += float((tenpai_prediction & tenpai_target.bool()).sum())
            totals["tenpai_true_negative"] += float(((~tenpai_prediction) & (~tenpai_target.bool())).sum())
            totals["tenpai_predicted_positive"] += float(tenpai_prediction.sum())
            totals["tenpai_actual_positive"] += float(tenpai_target.sum())
            totals["tenpai_brier"] += float((tenpai_probability - tenpai_target).square().sum())

            wait_probability = torch.sigmoid(output["opponent_wait_logits"][:, opponent])
            wait_target = batch["wait_target"][:, opponent]
            totals["wait_brier"] += float((wait_probability - wait_target).square().sum())
            wait_prediction = wait_probability >= 0.5
            totals["wait_true_positive"] += float((wait_prediction & wait_target.bool()).sum())
            totals["wait_predicted_positive"] += float(wait_prediction.sum())
            totals["wait_actual_positive"] += float(wait_target.sum())
            has_wait = wait_target.sum(dim=-1) > 0
            top_wait = wait_probability.argmax(dim=-1)
            top_hit = wait_target.gather(-1, top_wait.unsqueeze(-1)).squeeze(-1).bool()
            totals["wait_top1_hit"] += float((top_hit & has_wait).sum())
            totals["wait_tenpai_rows"] += float(has_wait.sum())

            shanten_prediction = output["opponent_shanten_logits"][:, opponent].argmax(-1)
            shanten_target = batch["shanten_target"][:, opponent]
            totals["shanten_correct"] += float((shanten_prediction == shanten_target).sum())
            totals["shanten_absolute_error"] += float((shanten_prediction - shanten_target).abs().sum())

            future_top5 = output["future_draw_logits"].topk(5, dim=-1).indices
            future_target = batch["future_target"]
            totals["future_top5_hit"] += float(future_target.gather(-1, future_top5).sum())
            totals["future_actual_positive"] += float(future_target.sum())

            if model.config.decision_heads:
                indexes = torch.arange(count, device=args.device)
                selected_danger_logits = output["deal_in_logits"][
                    indexes, batch["policy_target"]
                ]
                metadata = checkpoint["metadata"]
                scales = metadata.get("risk_logit_scale_by_opponent")
                biases = metadata.get("risk_logit_bias_by_opponent")
                if scales is not None or biases is not None:
                    if scales is None or biases is None or len(scales) != 4 or len(biases) != 4:
                        raise ValueError("checkpoint risk calibration is malformed")
                    selected_danger_logits = (
                        selected_danger_logits
                        * torch.tensor(scales, device=args.device)
                        + torch.tensor(biases, device=args.device)
                    )
                selected_danger = torch.sigmoid(selected_danger_logits[:, opponent])
                selected_target = batch["deal_in_target"][:, opponent]
                danger_probabilities.append(selected_danger.cpu())
                danger_targets.append(selected_target.cpu())
                for relative_index in range(3):
                    danger_probabilities_by_opponent[relative_index].append(
                        selected_danger[:, relative_index].cpu()
                    )
                    danger_targets_by_opponent[relative_index].append(
                        selected_target[:, relative_index].cpu()
                    )
                positive = batch["deal_in_target"].bool()
                if bool(positive.any()):
                    score_logits = output["deal_in_score_logits"][positive]
                    score_targets = batch["deal_in_score_target"][positive]
                    score_prediction = score_logits.argmax(-1)
                    expected_points = (torch.softmax(score_logits, -1) * centers).sum(-1)
                    target_points = centers[score_targets]
                    totals["deal_in_score_correct"] += float((score_prediction == score_targets).sum())
                    totals["deal_in_score_absolute_error_points"] += float(
                        (expected_points - target_points).abs().sum()
                    )
                    totals["deal_in_positive_cells"] += score_targets.numel()
                selected_outcome = output["action_outcome_logits"][indexes, batch["policy_target"]]
                outcome_probability = torch.softmax(selected_outcome, -1)
                outcome_probabilities_all.append(outcome_probability.cpu())
                outcome_targets_all.append(batch["action_outcome_target"].cpu())
                outcome_one_hot = F.one_hot(batch["action_outcome_target"], 5).float()
                totals["outcome_correct"] += float(
                    (selected_outcome.argmax(-1) == batch["action_outcome_target"]).sum()
                )
                totals["outcome_nll"] += float(
                    F.cross_entropy(selected_outcome, batch["action_outcome_target"], reduction="sum")
                )
                totals["outcome_brier"] += float(
                    (outcome_probability - outcome_one_hot).square().sum()
                )

    examples = totals["examples"]
    opponents = examples * 3.0
    wait_cells = opponents * 34.0
    result: dict[str, object] = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        "checkpoint_epoch": checkpoint["epoch"],
        "selection": checkpoint.get("selection"),
        "evaluated_split": split_name,
        "examples": int(examples),
        "policy_accuracy": totals["policy_correct"] / examples,
        "policy_nll": totals["policy_nll"] / examples,
        "rank_mae": totals["rank_absolute_error"] / examples,
        "rank_nll": totals["rank_nll"] / examples,
        "rank_brier": totals["rank_brier"] / examples,
        "rank_ece_15": _multiclass_ece(
            torch.cat(rank_probabilities_all), torch.cat(rank_targets_all), 15
        ),
        "score_mae_points": totals["score_absolute_error_points"] / examples,
        "round_delta_mae_points": totals["round_delta_absolute_error_points"] / (examples * 4.0),
        "tenpai_accuracy": totals["tenpai_correct"] / opponents,
        "tenpai_positive_rate": totals["tenpai_actual_positive"] / opponents,
        "tenpai_precision_at_0_5": totals["tenpai_true_positive"] / max(totals["tenpai_predicted_positive"], 1.0),
        "tenpai_recall_at_0_5": totals["tenpai_true_positive"] / max(totals["tenpai_actual_positive"], 1.0),
        "tenpai_specificity_at_0_5": totals["tenpai_true_negative"] / max(opponents - totals["tenpai_actual_positive"], 1.0),
        "tenpai_brier": totals["tenpai_brier"] / opponents,
        "wait_brier": totals["wait_brier"] / wait_cells,
        "wait_precision_at_0_5": totals["wait_true_positive"] / max(totals["wait_predicted_positive"], 1.0),
        "wait_recall_at_0_5": totals["wait_true_positive"] / max(totals["wait_actual_positive"], 1.0),
        "wait_top1_hit_rate_when_wait_exists": totals["wait_top1_hit"] / max(totals["wait_tenpai_rows"], 1.0),
        "shanten_accuracy": totals["shanten_correct"] / opponents,
        "shanten_mae": totals["shanten_absolute_error"] / opponents,
        "future_draw_top5_recall": totals["future_top5_hit"] / max(totals["future_actual_positive"], 1.0),
    }
    if model.config.decision_heads:
        probabilities = torch.cat(danger_probabilities)
        targets = torch.cat(danger_targets)
        result["deal_in"] = _binary_metrics(probabilities, targets)
        result["deal_in"]["calibration_applied"] = bool(
            checkpoint["metadata"].get("risk_calibrated", False)
        )
        result["deal_in_by_relative_opponent"] = {
            str(relative): _binary_metrics(
                torch.cat(danger_probabilities_by_opponent[relative - 1]),
                torch.cat(danger_targets_by_opponent[relative - 1]),
            )
            for relative in range(1, 4)
        }
        positive_cells = totals["deal_in_positive_cells"]
        result["deal_in_score"] = {
            "positive_cells": int(positive_cells),
            "bin_accuracy": totals["deal_in_score_correct"] / max(positive_cells, 1.0),
            "expected_points_mae": totals["deal_in_score_absolute_error_points"] / max(positive_cells, 1.0),
        }
        result["action_outcome"] = {
            "accuracy": totals["outcome_correct"] / examples,
            "nll": totals["outcome_nll"] / examples,
            "brier": totals["outcome_brier"] / examples,
            "ece_15": _multiclass_ece(
                torch.cat(outcome_probabilities_all),
                torch.cat(outcome_targets_all),
                15,
            ),
        }
    output_path = Path(args.output) if args.output else Path(args.checkpoint).with_suffix(
        f".{split_name}.evaluation.json"
    )
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


def _binary_metrics(probabilities: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    probabilities = probabilities.float().reshape(-1).clamp(1e-7, 1.0 - 1e-7)
    targets = targets.float().reshape(-1)
    return {
        "examples": float(targets.numel()),
        "positive_rate": float(targets.mean()),
        "brier": float((probabilities - targets).square().mean()),
        "log_loss": float(F.binary_cross_entropy(probabilities, targets)),
        "ece_15": float(expected_calibration_error(probabilities, targets, 15)),
    }


def _multiclass_ece(
    probabilities: torch.Tensor, targets: torch.Tensor, bins: int
) -> float:
    confidence, prediction = probabilities.float().max(dim=-1)
    correctness = (prediction == targets.reshape(-1)).float()
    return float(expected_calibration_error(confidence, correctness, bins))


if __name__ == "__main__":
    main()
