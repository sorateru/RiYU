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

from mahjong_ai.ai import (
    MODEL_OUTPUT_VERSION,
    SEQUENCE_ARCHITECTURE_VERSION,
    SequenceMahjongNetwork,
    SequenceModelConfig,
    load_sequence_state_dict_compatible,
)
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
    model_config = dict(checkpoint["model_config"])
    has_v2_outputs = checkpoint.get("model_output_version") == MODEL_OUTPUT_VERSION
    model_config["architecture_version"] = SEQUENCE_ARCHITECTURE_VERSION
    model = SequenceMahjongNetwork(SequenceModelConfig(**model_config))
    load_sequence_state_dict_compatible(model, checkpoint["model_state"])
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
        "policy_cells": 0.0,
        "rank_cells": 0.0,
        "score_cells": 0.0,
        "round_delta_cells": 0.0,
        "tenpai_cells": 0.0,
        "wait_cells": 0.0,
        "shanten_cells": 0.0,
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
        "outcome_cells": 0.0,
        "action_round_absolute_error_points": 0.0,
        "action_round_cells": 0.0,
        "action_hanchan_absolute_error": 0.0,
        "action_hanchan_cells": 0.0,
        "action_round_distribution_correct": 0.0,
        "opponent_discard_correct": 0.0,
        "opponent_discard_cells": 0.0,
        "opponent_response_correct": 0.0,
        "opponent_response_cells": 0.0,
    }
    danger_probabilities: list[torch.Tensor] = []
    danger_targets: list[torch.Tensor] = []
    danger_probabilities_by_opponent: list[list[torch.Tensor]] = [[], [], []]
    danger_targets_by_opponent: list[list[torch.Tensor]] = [[], [], []]
    rank_probabilities_all: list[torch.Tensor] = []
    rank_targets_all: list[torch.Tensor] = []
    outcome_probabilities_all: list[torch.Tensor] = []
    outcome_targets_all: list[torch.Tensor] = []
    centers = SCORE_CENTERS.to(args.device)
    with torch.inference_mode():
        for raw in loader:
            batch = {key: value.to(args.device) for key, value in raw.items()}
            output = model(
                batch["planes"], batch["actions"], batch["event_types"],
                batch["players"], batch["tiles"], batch["event_flags"],
                batch["event_mask"], batch["action_mask"],
                batch["danger_features"],
                batch["memory_features"],
                batch["rule_features"],
            )
            count = batch["planes"].shape[0]
            totals["examples"] += count
            policy_mask = batch["policy_target_mask"]
            totals["policy_correct"] += float(
                ((output["policy_logits"].argmax(1) == batch["policy_target"]) & policy_mask).sum()
            )
            policy_losses = F.cross_entropy(
                output["policy_logits"], batch["policy_target"], reduction="none"
            )
            totals["policy_nll"] += float(policy_losses[policy_mask].sum())
            totals["policy_cells"] += float(policy_mask.sum())
            rank_probability = torch.softmax(output["rank_logits"], dim=-1)
            rank_mask = batch["rank_target_mask"]
            rank_probabilities_all.append(rank_probability[rank_mask].cpu())
            rank_targets_all.append(batch["rank_target"][rank_mask].cpu())
            rank_one_hot = F.one_hot(batch["rank_target"], 4).float()
            totals["rank_absolute_error"] += float(
                ((output["rank_logits"].argmax(1) - batch["rank_target"]).abs() * rank_mask).sum()
            )
            rank_losses = F.cross_entropy(
                output["rank_logits"], batch["rank_target"], reduction="none"
            )
            totals["rank_nll"] += float(rank_losses[rank_mask].sum())
            totals["rank_brier"] += float(
                ((rank_probability - rank_one_hot).square() * rank_mask.unsqueeze(-1)).sum()
            )
            totals["rank_cells"] += float(rank_mask.sum())
            score_mask = batch["score_target_mask"]
            totals["score_absolute_error_points"] += float(
                ((output["score_value"] - batch["score_target"]).abs() * score_mask).sum()
                * 10_000.0
            )
            totals["score_cells"] += float(score_mask.sum())
            delta_mask = batch["round_delta_target_mask"]
            totals["round_delta_absolute_error_points"] += float(
                (
                    (output["round_delta_values"] - batch["round_delta_target"]).abs()
                    * delta_mask
                ).sum()
                * 10_000.0
            )
            totals["round_delta_cells"] += float(delta_mask.sum())

            tenpai_probability = torch.sigmoid(output["tenpai_logits"])
            tenpai_target = batch["tenpai_target"]
            tenpai_mask = batch["tenpai_target_mask"]
            tenpai_prediction = tenpai_probability >= 0.5
            totals["tenpai_correct"] += float(((tenpai_prediction == tenpai_target.bool()) & tenpai_mask).sum())
            totals["tenpai_true_positive"] += float((tenpai_prediction & tenpai_target.bool() & tenpai_mask).sum())
            totals["tenpai_true_negative"] += float(((~tenpai_prediction) & (~tenpai_target.bool()) & tenpai_mask).sum())
            totals["tenpai_predicted_positive"] += float((tenpai_prediction & tenpai_mask).sum())
            totals["tenpai_actual_positive"] += float((tenpai_target.bool() & tenpai_mask).sum())
            totals["tenpai_brier"] += float(((tenpai_probability - tenpai_target).square() * tenpai_mask).sum())
            totals["tenpai_cells"] += float(tenpai_mask.sum())

            wait_probability = torch.sigmoid(output["wait_logits"])
            wait_target = batch["wait_target"]
            wait_mask = batch["wait_target_mask"].expand_as(wait_target)
            totals["wait_brier"] += float(((wait_probability - wait_target).square() * wait_mask).sum())
            totals["wait_cells"] += float(wait_mask.sum())
            wait_prediction = wait_probability >= 0.5
            totals["wait_true_positive"] += float((wait_prediction & wait_target.bool() & wait_mask).sum())
            totals["wait_predicted_positive"] += float((wait_prediction & wait_mask).sum())
            totals["wait_actual_positive"] += float((wait_target.bool() & wait_mask).sum())
            valid_wait_row = batch["wait_target_mask"].any(dim=-1)
            has_wait = (wait_target.sum(dim=-1) > 0) & valid_wait_row
            top_wait = wait_probability.argmax(dim=-1)
            top_hit = wait_target.gather(-1, top_wait.unsqueeze(-1)).squeeze(-1).bool()
            totals["wait_top1_hit"] += float((top_hit & has_wait).sum())
            totals["wait_tenpai_rows"] += float(has_wait.sum())

            shanten_prediction = output["shanten_logits"].argmax(-1)
            shanten_target = batch["shanten_target"]
            shanten_mask = batch["shanten_target_mask"]
            totals["shanten_correct"] += float(((shanten_prediction == shanten_target) & shanten_mask).sum())
            totals["shanten_absolute_error"] += float(((shanten_prediction - shanten_target).abs() * shanten_mask).sum())
            totals["shanten_cells"] += float(shanten_mask.sum())

            future_top5 = output["future_draw_logits"].topk(5, dim=-1).indices
            future_target = batch["future_target"]
            future_mask = batch["future_target_mask"]
            totals["future_top5_hit"] += float(
                (future_target.gather(-1, future_top5) * future_mask.unsqueeze(-1)).sum()
            )
            totals["future_actual_positive"] += float(
                (future_target * future_mask.unsqueeze(-1)).sum()
            )

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
                        * torch.tensor(scales[1:], device=args.device)
                        + torch.tensor(biases[1:], device=args.device)
                    )
                selected_danger = torch.sigmoid(selected_danger_logits)
                selected_target = batch["deal_in_target"][indexes, batch["policy_target"]]
                selected_mask = batch["deal_in_target_mask"][indexes, batch["policy_target"]]
                danger_probabilities.append(selected_danger[selected_mask].cpu())
                danger_targets.append(selected_target[selected_mask].cpu())
                for relative_index in range(3):
                    relative_mask = selected_mask[:, relative_index]
                    danger_probabilities_by_opponent[relative_index].append(
                        selected_danger[relative_mask, relative_index].cpu()
                    )
                    danger_targets_by_opponent[relative_index].append(
                        selected_target[relative_mask, relative_index].cpu()
                    )
                positive = batch["deal_in_score_mask"]
                if bool(positive.any()):
                    score_logits = output["deal_in_point_logits"][positive]
                    score_targets = batch["deal_in_score_target"][positive]
                    score_prediction = score_logits.argmax(-1)
                    expected_points = (torch.softmax(score_logits, -1) * centers).sum(-1)
                    target_points = centers[score_targets]
                    totals["deal_in_score_correct"] += float((score_prediction == score_targets).sum())
                    totals["deal_in_score_absolute_error_points"] += float(
                        (expected_points - target_points).abs().sum()
                    )
                    totals["deal_in_positive_cells"] += score_targets.numel()
                outcome_logits = output["action_outcome_logits"]
                outcome_probability = torch.softmax(outcome_logits, -1)
                outcome_mask = batch["action_outcome_target_mask"]
                outcome_probabilities_all.append(outcome_probability[outcome_mask].cpu())
                outcome_targets_all.append(batch["action_outcome_target"][outcome_mask].cpu())
                outcome_one_hot = F.one_hot(batch["action_outcome_target"], 5).float()
                totals["outcome_correct"] += float(
                    ((outcome_logits.argmax(-1) == batch["action_outcome_target"]) & outcome_mask).sum()
                )
                outcome_losses = F.cross_entropy(
                    outcome_logits.transpose(1, 2),
                    batch["action_outcome_target"],
                    reduction="none",
                )
                totals["outcome_nll"] += float(outcome_losses[outcome_mask].sum())
                totals["outcome_brier"] += float(
                    ((outcome_probability - outcome_one_hot).square() * outcome_mask.unsqueeze(-1)).sum()
                )
                totals["outcome_cells"] += float(outcome_mask.sum())
                if has_v2_outputs:
                    action_round_mask = batch["action_round_value_mask"]
                    totals["action_round_absolute_error_points"] += float(
                        (
                            (output["action_round_value"] - batch["action_round_value_target"]).abs()
                            * action_round_mask
                        ).sum()
                        * 10_000.0
                    )
                    totals["action_round_cells"] += float(action_round_mask.sum())
                    action_hanchan_mask = batch["action_hanchan_value_mask"]
                    totals["action_hanchan_absolute_error"] += float(
                        (
                            (output["action_hanchan_value"] - batch["action_hanchan_value_target"]).abs()
                            * action_hanchan_mask
                        ).sum()
                    )
                    totals["action_hanchan_cells"] += float(action_hanchan_mask.sum())
                    totals["action_round_distribution_correct"] += float(
                        (
                            output["action_round_distribution_logits"].argmax(-1)
                            == batch["action_round_distribution_target"]
                        )[batch["action_round_distribution_mask"]].sum()
                    )
                    opponent_discard_mask = batch["opponent_next_discard_mask"]
                    totals["opponent_discard_correct"] += float(
                        (
                            output["opponent_next_discard_logits"].argmax(-1)
                            == batch["opponent_next_discard_target"]
                        )[opponent_discard_mask].sum()
                    )
                    totals["opponent_discard_cells"] += float(opponent_discard_mask.sum())
                    opponent_response_mask = batch["opponent_next_response_mask"]
                    totals["opponent_response_correct"] += float(
                        (
                            output["opponent_response_logits"].argmax(-1)
                            == batch["opponent_next_response_target"]
                        )[opponent_response_mask].sum()
                    )
                    totals["opponent_response_cells"] += float(opponent_response_mask.sum())

    examples = totals["examples"]
    policy_cells = max(totals["policy_cells"], 1.0)
    rank_cells = max(totals["rank_cells"], 1.0)
    score_cells = max(totals["score_cells"], 1.0)
    delta_cells = max(totals["round_delta_cells"], 1.0)
    tenpai_cells = max(totals["tenpai_cells"], 1.0)
    wait_cells = max(totals["wait_cells"], 1.0)
    shanten_cells = max(totals["shanten_cells"], 1.0)
    result: dict[str, object] = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        "checkpoint_epoch": checkpoint["epoch"],
        "selection": checkpoint.get("selection"),
        "evaluated_split": split_name,
        "examples": int(examples),
        "valid_labels": {
            "policy": int(totals["policy_cells"]),
            "rank": int(totals["rank_cells"]),
            "score": int(totals["score_cells"]),
            "round_delta": int(totals["round_delta_cells"]),
            "tenpai": int(totals["tenpai_cells"]),
            "wait": int(totals["wait_cells"]),
            "shanten": int(totals["shanten_cells"]),
            "outcome": int(totals["outcome_cells"]),
        },
        "policy_accuracy": totals["policy_correct"] / policy_cells,
        "policy_nll": totals["policy_nll"] / policy_cells,
        "rank_mae": totals["rank_absolute_error"] / rank_cells,
        "rank_nll": totals["rank_nll"] / rank_cells,
        "rank_brier": totals["rank_brier"] / rank_cells,
        "rank_ece_15": _multiclass_ece(
            torch.cat(rank_probabilities_all), torch.cat(rank_targets_all), 15
        ),
        "score_mae_points": totals["score_absolute_error_points"] / score_cells,
        "round_delta_mae_points": totals["round_delta_absolute_error_points"] / delta_cells,
        "tenpai_accuracy": totals["tenpai_correct"] / tenpai_cells,
        "tenpai_positive_rate": totals["tenpai_actual_positive"] / tenpai_cells,
        "tenpai_precision_at_0_5": totals["tenpai_true_positive"] / max(totals["tenpai_predicted_positive"], 1.0),
        "tenpai_recall_at_0_5": totals["tenpai_true_positive"] / max(totals["tenpai_actual_positive"], 1.0),
        "tenpai_specificity_at_0_5": totals["tenpai_true_negative"] / max(tenpai_cells - totals["tenpai_actual_positive"], 1.0),
        "tenpai_brier": totals["tenpai_brier"] / tenpai_cells,
        "wait_brier": totals["wait_brier"] / wait_cells,
        "wait_precision_at_0_5": totals["wait_true_positive"] / max(totals["wait_predicted_positive"], 1.0),
        "wait_recall_at_0_5": totals["wait_true_positive"] / max(totals["wait_actual_positive"], 1.0),
        "wait_top1_hit_rate_when_wait_exists": totals["wait_top1_hit"] / max(totals["wait_tenpai_rows"], 1.0),
        "shanten_accuracy": totals["shanten_correct"] / shanten_cells,
        "shanten_mae": totals["shanten_absolute_error"] / shanten_cells,
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
            "valid_labels": int(totals["outcome_cells"]),
            "accuracy": totals["outcome_correct"] / max(totals["outcome_cells"], 1.0),
            "nll": totals["outcome_nll"] / max(totals["outcome_cells"], 1.0),
            "brier": totals["outcome_brier"] / max(totals["outcome_cells"], 1.0),
            "ece_15": _multiclass_ece(
                torch.cat(outcome_probabilities_all),
                torch.cat(outcome_targets_all),
                15,
            ),
        }
        if has_v2_outputs:
            result["action_value"] = {
                "round_valid_labels": int(totals["action_round_cells"]),
                "round_mae_points": totals["action_round_absolute_error_points"]
                / max(totals["action_round_cells"], 1.0),
                "round_distribution_accuracy": totals[
                    "action_round_distribution_correct"
                ]
                / max(totals["action_round_cells"], 1.0),
                "hanchan_valid_labels": int(totals["action_hanchan_cells"]),
                "hanchan_utility_mae": totals["action_hanchan_absolute_error"]
                / max(totals["action_hanchan_cells"], 1.0),
            }
            result["opponent_action"] = {
                "discard_valid_labels": int(totals["opponent_discard_cells"]),
                "discard_accuracy": totals["opponent_discard_correct"]
                / max(totals["opponent_discard_cells"], 1.0),
                "response_valid_labels": int(totals["opponent_response_cells"]),
                "response_accuracy": totals["opponent_response_correct"]
                / max(totals["opponent_response_cells"], 1.0),
            }
    output_path = Path(args.output) if args.output else Path(args.checkpoint).with_suffix(
        f".{split_name}.evaluation.json"
    )
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


def _binary_metrics(
    probabilities: torch.Tensor, targets: torch.Tensor
) -> dict[str, float | int | None]:
    probabilities = probabilities.float().reshape(-1).clamp(1e-7, 1.0 - 1e-7)
    targets = targets.float().reshape(-1)
    if targets.numel() == 0:
        return {
            "examples": 0,
            "positive_rate": None,
            "brier": None,
            "log_loss": None,
            "ece_15": None,
        }
    return {
        "examples": int(targets.numel()),
        "positive_rate": float(targets.mean()),
        "brier": float((probabilities - targets).square().mean()),
        "log_loss": float(F.binary_cross_entropy(probabilities, targets)),
        "ece_15": float(expected_calibration_error(probabilities, targets, 15)),
    }


def _multiclass_ece(
    probabilities: torch.Tensor, targets: torch.Tensor, bins: int
) -> float | None:
    if targets.numel() == 0:
        return None
    confidence, prediction = probabilities.float().max(dim=-1)
    correctness = (prediction == targets.reshape(-1)).float()
    return float(expected_calibration_error(confidence, correctness, bins))


if __name__ == "__main__":
    main()
