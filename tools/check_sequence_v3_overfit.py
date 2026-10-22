#!/usr/bin/env python3
"""Intentional small-set overfit gate before expensive Decision V4 training."""

from __future__ import annotations

import argparse
import json

import torch
from torch.utils.data import DataLoader, Subset

from mahjong_ai.ai import SequenceMahjongNetwork, SequenceModelConfig
from mahjong_ai.training import (
    ParquetSequenceDataset,
    SequenceTrainer,
    SequenceTrainerConfig,
    collate_sequence_decisions,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="+")
    parser.add_argument("--decisions", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--minimum-policy-accuracy", type=float, default=0.98)
    parser.add_argument("--maximum-loss-ratio", type=float, default=0.25)
    parser.add_argument(
        "--require-complete-label-coverage",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="fail unless every frozen V1 head has at least one valid label",
    )
    args = parser.parse_args()
    if not 100 <= args.decisions <= 1_000:
        parser.error("--decisions must be between 100 and 1000")
    if not 0 < args.maximum_loss_ratio < 1:
        parser.error("--maximum-loss-ratio must be between zero and one")
    if not 0 < args.minimum_policy_accuracy <= 1:
        parser.error("--minimum-policy-accuracy must be in (0, 1]")
    dataset = ParquetSequenceDataset(args.datasets)
    if not dataset.has_decision_labels:
        raise ValueError("overfit gate requires frozen candidate labels")
    if not dataset.has_v4_labels:
        raise ValueError("full overfit gate requires Decision V4 action-value labels")
    generator = torch.Generator().manual_seed(20261019)
    indexes = torch.randperm(len(dataset), generator=generator)[
        : min(args.decisions, len(dataset))
    ].tolist()
    selected = Subset(dataset, indexes)
    if len(selected) < 100:
        raise ValueError("overfit gate requires at least 100 decisions")
    coverage = _label_coverage(selected)
    missing = sorted(name for name, count in coverage.items() if count == 0)
    if args.require_complete_label_coverage and missing:
        raise ValueError(
            "overfit subset has no valid labels for: " + ", ".join(missing)
        )
    model = SequenceMahjongNetwork(
        SequenceModelConfig(
            snapshot_channels=16,
            snapshot_blocks=1,
            snapshot_dim=48,
            event_dim=32,
            transformer_heads=4,
            transformer_layers=1,
            transformer_feedforward=96,
            fused_dim=64,
            action_dim=32,
            decision_heads=True,
            opponent_dim=32,
            opponent_temporal_latents=True,
        )
    )
    trainer = SequenceTrainer(
        model,
        SequenceTrainerConfig(
            epochs=1,
            batch_size=args.batch_size,
            learning_rate=1e-3,
            weight_decay=0.0,
            auto_positive_weights=False,
            policy_label_smoothing=0.0,
            device=args.device,
        ),
    )
    loader = DataLoader(
        selected,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_sequence_decisions,
        generator=torch.Generator().manual_seed(trainer.config.seed),
    )
    initial = trainer._run_epoch(loader, training=False)
    final = initial
    for _ in range(args.epochs):
        final = trainer._run_epoch(loader, training=True)
    report = {
        "decisions": len(selected),
        "epochs": args.epochs,
        "label_coverage": coverage,
        "initial": initial,
        "final": final,
        "loss_ratio": final["loss"] / max(initial["loss"], 1e-12),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if final["policy_accuracy"] < args.minimum_policy_accuracy:
        raise SystemExit("overfit gate failed: policy did not memorize the small set")
    if report["loss_ratio"] > args.maximum_loss_ratio:
        raise SystemExit("overfit gate failed: total loss did not fall far enough")
    loss_to_coverage = {
        "policy": "policy",
        "rank": "rank",
        "score": "score",
        "round": "round",
        "tenpai": "tenpai",
        "wait": "wait",
        "shanten": "shanten",
        "draw": "draw",
        "deal": "deal",
        "dealpoint": "dealpoint",
        "result": "result",
        "response": "response",
        "call_type": "call_type",
        "riichi": "riichi",
        "state_round_value": "state_round_value",
        "state_hanchan_value": "state_hanchan_value",
        "action_round_value": "action_round_value",
        "action_hanchan_value": "action_hanchan_value",
        "action_round_distribution": "action_round_distribution",
        "opponent_discard": "opponent_discard",
        "opponent_response": "opponent_response",
        "opponent_call_type": "opponent_call_type",
        "opponent_tile_count": "opponent_tile_count",
    }
    for name, coverage_name in loss_to_coverage.items():
        if coverage[coverage_name] == 0:
            continue
        initial_loss = initial[f"loss_{name}"]
        final_loss = final[f"loss_{name}"]
        if initial_loss > 0 and final_loss >= initial_loss:
            raise SystemExit(f"overfit gate failed: {name} loss did not improve")


def _label_coverage(dataset: Subset) -> dict[str, int]:
    coverage = {
        name: 0
        for name in (
            "policy", "rank", "score", "round", "tenpai", "wait", "shanten",
            "draw", "deal", "dealpoint", "result", "response", "call_type", "riichi",
            "state_round_value", "state_hanchan_value", "action_round_value",
            "action_hanchan_value", "action_round_distribution", "opponent_discard",
            "opponent_response", "opponent_call_type", "opponent_tile_count",
        )
    }
    for item in dataset:
        coverage["policy"] += int(item["policy_target_mask"])
        coverage["rank"] += int(item["rank_target_mask"])
        coverage["score"] += int(item["score_target_mask"])
        coverage["round"] += int(item["round_delta_target_mask"].sum())
        coverage["tenpai"] += int(item["tenpai_target_mask"].sum())
        coverage["wait"] += int(item["wait_target_mask"].sum())
        coverage["shanten"] += int(item["shanten_target_mask"].sum())
        coverage["draw"] += int(item["future_target_mask"])
        coverage["deal"] += int(item["deal_in_target_mask"].sum())
        coverage["dealpoint"] += int(item["deal_in_score_mask"].sum())
        coverage["result"] += int(item["action_outcome_target_mask"])
        coverage["response"] += int(item["response_target_mask"])
        coverage["call_type"] += int(item["call_type_target_mask"])
        coverage["riichi"] += int(item["riichi_target_mask"].sum())
        coverage["state_round_value"] += int(item["state_round_value_mask"])
        coverage["state_hanchan_value"] += int(item["state_hanchan_value_mask"])
        coverage["action_round_value"] += int(item["action_round_value_mask"].sum())
        coverage["action_hanchan_value"] += int(item["action_hanchan_value_mask"].sum())
        coverage["action_round_distribution"] += int(
            item["action_round_distribution_mask"].sum()
        )
        coverage["opponent_discard"] += int(item["opponent_next_discard_mask"].sum())
        coverage["opponent_response"] += int(item["opponent_next_response_mask"].sum())
        coverage["opponent_call_type"] += int(item["opponent_next_call_type_mask"].sum())
        coverage["opponent_tile_count"] += int(item["opponent_tile_count_mask"].sum())
    return coverage


if __name__ == "__main__":
    main()
