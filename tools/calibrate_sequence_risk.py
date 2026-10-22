#!/usr/bin/env python3
"""Fit opponent-specific deal-in calibration on validation, never holdout."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from mahjong_ai.ai import (
    SEQUENCE_ARCHITECTURE_VERSION,
    SequenceMahjongNetwork,
    SequenceModelConfig,
    load_sequence_state_dict_compatible,
)
from mahjong_ai.ai.calibration import fit_platt_scaling
from mahjong_ai.training import (
    ParquetSequenceDataset,
    collate_sequence_decisions,
    split_sequence_train_validation_holdout,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("datasets", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--holdout-fraction", type=float, default=0.1)
    args = parser.parse_args()
    source = Path(args.checkpoint).resolve()
    output_path = Path(args.output).resolve()
    if source == output_path:
        parser.error("output must differ from the source checkpoint")

    checkpoint = torch.load(source, map_location=args.device, weights_only=False)
    if checkpoint.get("model_family") != "sequence-transformer-v2":
        raise ValueError("checkpoint is not sequence-transformer-v2")
    raw_config = dict(checkpoint["model_config"])
    raw_config["architecture_version"] = SEQUENCE_ARCHITECTURE_VERSION
    config = SequenceModelConfig(**raw_config)
    if not config.decision_heads:
        raise ValueError("checkpoint has no decision heads to calibrate")
    if checkpoint.get("metadata", {}).get("risk_calibrated", False):
        raise ValueError("checkpoint risk head is already calibrated")
    model = SequenceMahjongNetwork(config)
    load_sequence_state_dict_compatible(model, checkpoint["model_state"])
    model.to(args.device).eval()
    dataset = ParquetSequenceDataset(args.datasets)
    if not dataset.has_decision_labels:
        raise ValueError("risk calibration requires newly converted decision labels")
    _, validation, _ = split_sequence_train_validation_holdout(
        dataset, args.validation_fraction, args.holdout_fraction
    )
    loader = DataLoader(
        validation, batch_size=args.batch_size, shuffle=False, num_workers=0,
        collate_fn=collate_sequence_decisions,
    )
    logits_by_relative: list[list[torch.Tensor]] = [[], [], []]
    targets_by_relative: list[list[torch.Tensor]] = [[], [], []]
    with torch.inference_mode():
        for raw in loader:
            batch = {key: value.to(args.device) for key, value in raw.items()}
            predicted = model(
                batch["planes"], batch["actions"], batch["event_types"],
                batch["players"], batch["tiles"], batch["event_flags"],
                batch["event_mask"], batch["action_mask"],
                batch["danger_features"],
                batch["memory_features"],
                batch["rule_features"],
            )
            indexes = torch.arange(batch["planes"].shape[0], device=args.device)
            selected = predicted["deal_in_logits"][indexes, batch["policy_target"]]
            selected_target = batch["deal_in_target"][indexes, batch["policy_target"]]
            selected_mask = batch["deal_in_target_mask"][indexes, batch["policy_target"]]
            for relative in range(3):
                valid = selected_mask[:, relative]
                logits_by_relative[relative].append(selected[valid, relative].cpu())
                targets_by_relative[relative].append(selected_target[valid, relative].cpu())

    results = [
        fit_platt_scaling(torch.cat(logits), torch.cat(targets))
        for logits, targets in zip(logits_by_relative, targets_by_relative, strict=True)
    ]
    metadata = dict(checkpoint["metadata"])
    scales = [1.0] + [result.scale for result in results]
    biases = [0.0] + [result.bias for result in results]
    metadata.update(
        {
            "risk_calibrated": True,
            "risk_logit_scale_by_opponent": scales,
            "risk_logit_bias_by_opponent": biases,
        }
    )
    checkpoint["metadata"] = metadata
    checkpoint["risk_calibration"] = {
        "method": "opponent_specific_monotone_platt",
        "fitted_split": "validation",
        "holdout_opened": False,
        "created_at": time.time(),
        "source_checkpoint": str(source),
        "relative_opponents": {
            str(relative): asdict(result)
            for relative, result in enumerate(results, start=1)
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)
    report_path = output_path.with_suffix(".calibration.json")
    report_path.write_text(
        json.dumps(checkpoint["risk_calibration"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(checkpoint["risk_calibration"], sort_keys=True))


if __name__ == "__main__":
    main()
