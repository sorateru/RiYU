#!/usr/bin/env python3
"""Report held-out policy accuracy by expert action type."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch

from mahjong_ai.ai import MahjongNetwork, ModelConfig
from mahjong_ai.training.supervised import (
    ParquetDecisionDataset,
    collate_decisions,
    split_by_log,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("datasets", nargs="+")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = MahjongNetwork(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    dataset = ParquetDecisionDataset([Path(value) for value in args.datasets])
    _, validation = split_by_log(dataset)
    totals: Counter[str] = Counter()
    exact: Counter[str] = Counter()
    same_type: Counter[str] = Counter()
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    indices = list(validation.indices)
    for start in range(0, len(indices), args.batch_size):
        selected = indices[start : start + args.batch_size]
        batch = collate_decisions([dataset[index] for index in selected])
        with torch.inference_mode():
            predicted = model(batch["planes"], batch["actions"], batch["mask"])[
                "policy_logits"
            ].argmax(1)
        for offset, index in enumerate(selected):
            row = dataset.rows[index]
            expert = json.loads(row["expert_action"])
            legal = json.loads(row["legal_actions"])
            target = int(batch["policy_target"][offset])
            guess = int(predicted[offset])
            target_type = expert["type"]
            predicted_type = legal[guess]["type"]
            totals[target_type] += 1
            exact[target_type] += int(guess == target)
            same_type[target_type] += int(predicted_type == target_type)
            confusion[target_type][predicted_type] += 1
    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "validation_decisions": len(indices),
        "overall_exact_accuracy": sum(exact.values()) / sum(totals.values()),
        "by_expert_action": {
            action: {
                "count": totals[action],
                "exact_accuracy": exact[action] / totals[action],
                "action_type_accuracy": same_type[action] / totals[action],
                "predicted_types": dict(sorted(confusion[action].items())),
            }
            for action in sorted(totals)
        },
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
