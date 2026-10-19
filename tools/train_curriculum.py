#!/usr/bin/env python3
"""Train from verified human decisions, then continue with self-play PPO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mahjong_ai.ai import MahjongNetwork, ModelConfig
from mahjong_ai.data import load_human_dataset_manifest
from mahjong_ai.training import (
    PPOTrainer,
    PPOTrainerConfig,
    ParquetDecisionDataset,
    Trainer,
    TrainerConfig,
    split_by_log,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--supervised-epochs", type=int, default=5)
    parser.add_argument("--ppo-iterations", type=int, default=10)
    parser.add_argument("--rounds-per-iteration", type=int, default=32)
    parser.add_argument("--ppo-update-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--ppo-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    if checkpoint.exists():
        parser.error("checkpoint already exists; choose a new curriculum output path")
    manifest = load_human_dataset_manifest(args.manifest)
    dataset = ParquetDecisionDataset(manifest.files)
    train_set, validation_set = split_by_log(dataset)
    torch.manual_seed(args.seed)
    supervised = Trainer(
        MahjongNetwork(ModelConfig()),
        TrainerConfig(
            seed=args.seed,
            epochs=args.supervised_epochs,
            batch_size=args.batch_size,
            device=args.device,
        ),
    )
    supervised_history = supervised.fit(
        train_set,
        validation_set,
        checkpoint,
        checkpoint.stem,
        manifest.fingerprint,
    )

    ppo_config = PPOTrainerConfig(
        seed=args.seed,
        iterations=args.ppo_iterations,
        rounds_per_iteration=args.rounds_per_iteration,
        update_epochs=args.ppo_update_epochs,
        batch_size=args.batch_size,
        collection_workers=args.ppo_workers,
        device=args.device,
    )
    ppo = PPOTrainer.warm_start(checkpoint, ppo_config)
    ppo_history = ppo.fit(
        checkpoint,
        checkpoint.stem,
        {
            "wind": "east",
            "hand_number": 1,
            "honba": 0,
            "riichi_sticks": 0,
            "dealer": 0,
        },
    )
    report = {
        "human_dataset": {
            "dataset_id": manifest.dataset_id,
            "source": manifest.source,
            "rights_basis": manifest.rights_basis,
            "authorization_reference": manifest.authorization_reference,
            "redistribution_allowed": manifest.redistribution_allowed,
            "fingerprint": manifest.fingerprint,
        },
        "supervised": supervised_history,
        "selfplay_ppo": ppo_history,
        "checkpoint": str(checkpoint),
        "validated": False,
    }
    report_path = checkpoint.with_suffix(".curriculum.json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(ppo_history[-1], ensure_ascii=False, sort_keys=True))
    print(f"saved curriculum checkpoint to {checkpoint}")
    print("checkpoint remains unapproved until arena and calibration review")


if __name__ == "__main__":
    main()
