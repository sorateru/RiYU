#!/usr/bin/env python3
"""Train the observable sequence Transformer on decision/v2 shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from mahjong_ai.ai import SequenceMahjongNetwork, SequenceModelConfig
from mahjong_ai.training import (
    ParquetSequenceDataset,
    SequenceTrainer,
    SequenceTrainerConfig,
    split_sequence_by_log,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="+")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--warm-start")
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument(
        "--preset", choices=("tiny", "base"), default="base",
        help="tiny is intended only for pipeline smoke tests",
    )
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        parser.error("epochs and batch size must be positive")
    dataset_paths = [Path(value) for value in args.datasets]
    digest = hashlib.sha256()
    for dataset_path in sorted(dataset_paths):
        digest.update(str(dataset_path.resolve()).encode())
        digest.update(dataset_path.read_bytes())
    fingerprint = digest.hexdigest()
    dataset = ParquetSequenceDataset(dataset_paths)
    train, validation = split_sequence_by_log(dataset, args.validation_fraction)
    if args.warm_start:
        warm = torch.load(args.warm_start, map_location=args.device, weights_only=False)
        if warm.get("model_family") != "sequence-transformer-v2":
            raise ValueError("warm-start checkpoint is not sequence-transformer-v2")
        model = SequenceMahjongNetwork(SequenceModelConfig(**warm["model_config"]))
        model.load_state_dict(warm["model_state"])
    elif args.preset == "tiny":
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
            )
        )
    else:
        model = SequenceMahjongNetwork(SequenceModelConfig())
    trainer = SequenceTrainer(
        model,
        SequenceTrainerConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device,
        ),
    )
    history = trainer.fit(
        train,
        validation,
        args.checkpoint,
        model_id=Path(args.checkpoint).stem,
        dataset_fingerprint=fingerprint,
    )
    summary = {
        "completed_at": time.time(),
        "model_family": "sequence-transformer-v2",
        "preset": args.preset,
        "datasets": [str(path.resolve()) for path in dataset_paths],
        "dataset_fingerprint": fingerprint,
        "decisions": len(dataset),
        "epochs": args.epochs,
        "warm_start": str(Path(args.warm_start).resolve()) if args.warm_start else None,
        "history": history,
    }
    Path(args.checkpoint).with_suffix(".training.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(history[-1], ensure_ascii=False, sort_keys=True))
    print("checkpoint remains unapproved until an explicit arena/calibration review")


if __name__ == "__main__":
    main()
