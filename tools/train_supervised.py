#!/usr/bin/env python3
"""Train a baseline dynamic-action policy/value network."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from mahjong_ai.ai import MahjongNetwork, ModelConfig
from mahjong_ai.training import (
    ParquetDecisionDataset,
    Trainer,
    TrainerConfig,
    split_train_validation_holdout,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--warm-start")
    parser.add_argument("--mlflow", action="store_true")
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--holdout-fraction", type=float, default=0.1)
    args = parser.parse_args()
    dataset_paths = [Path(value) for value in args.datasets]
    digest = hashlib.sha256()
    for dataset_path in sorted(dataset_paths):
        digest.update(str(dataset_path.resolve()).encode())
        digest.update(dataset_path.read_bytes())
    fingerprint = digest.hexdigest()
    dataset = ParquetDecisionDataset(dataset_paths)
    train, validation, holdout = split_train_validation_holdout(
        dataset, args.validation_fraction, args.holdout_fraction
    )
    if args.warm_start:
        warm = torch.load(args.warm_start, map_location=args.device, weights_only=False)
        model = MahjongNetwork(ModelConfig(**warm["model_config"]))
        model.load_state_dict(warm["model_state"])
    else:
        model = MahjongNetwork(ModelConfig())
    trainer = Trainer(
        model,
        TrainerConfig(epochs=args.epochs, batch_size=args.batch_size, device=args.device),
    )
    history = trainer.fit(
        train,
        validation,
        args.checkpoint,
        model_id=Path(args.checkpoint).stem,
        dataset_fingerprint=fingerprint,
        use_mlflow=args.mlflow,
    )
    print(history[-1])
    summary = {
        "completed_at": time.time(),
        "datasets": [str(path.resolve()) for path in dataset_paths],
        "dataset_fingerprint": fingerprint,
        "decisions": len(dataset),
        "train_decisions": len(train),
        "validation_decisions": len(validation),
        "sealed_holdout_decisions": len(holdout),
        "holdout_policy": "never evaluated during training or checkpoint selection",
        "epochs": args.epochs,
        "warm_start": str(Path(args.warm_start).resolve()) if args.warm_start else None,
        "history": history,
    }
    Path(args.checkpoint).with_suffix(".training.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("checkpoint remains unapproved until an explicit arena/calibration review")


if __name__ == "__main__":
    main()
