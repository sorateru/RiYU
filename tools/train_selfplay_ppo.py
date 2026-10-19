#!/usr/bin/env python3
"""Train a dynamic-action model with local simulator self-play PPO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mahjong_ai.ai import MahjongNetwork, ModelConfig
from mahjong_ai.training import PPOTrainer, PPOTrainerConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--warm-start", action="store_true")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--rounds-per-iteration", type=int, default=8)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=0, help="0 selects workers from rounds per iteration")
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--history")
    parser.add_argument("--small-model", action="store_true", help="use a smoke-test-sized model")
    args = parser.parse_args()

    config = PPOTrainerConfig(
        seed=args.seed,
        iterations=args.iterations,
        rounds_per_iteration=args.rounds_per_iteration,
        update_epochs=args.update_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        collection_workers=args.workers,
        device=args.device,
    )
    checkpoint = Path(args.checkpoint)
    history_path = Path(args.history) if args.history else checkpoint.with_suffix(".history.json")
    if args.resume and args.warm_start:
        parser.error("--resume and --warm-start are mutually exclusive")
    if args.resume:
        if not checkpoint.exists():
            parser.error("--resume requires an existing checkpoint")
        trainer = PPOTrainer.resume(checkpoint, config)
    elif args.warm_start:
        if not checkpoint.exists():
            parser.error("--warm-start requires an existing supervised checkpoint")
        trainer = PPOTrainer.warm_start(checkpoint, config)
    else:
        if checkpoint.exists():
            parser.error("checkpoint already exists; use --resume or choose a new path")
        torch.manual_seed(args.seed)
        model_config = (
            ModelConfig(channels=8, residual_blocks=1, representation_dim=32, action_dim=16)
            if args.small_model
            else ModelConfig()
        )
        trainer = PPOTrainer(MahjongNetwork(model_config), config)

    round_state = {
        "wind": "east",
        "hand_number": 1,
        "honba": 0,
        "riichi_sticks": 0,
        "dealer": 0,
    }
    history = trainer.fit(checkpoint, checkpoint.stem, round_state)
    previous_history: list[dict[str, float]] = []
    if (args.resume or args.warm_start) and history_path.exists():
        loaded_history = json.loads(history_path.read_text(encoding="utf-8"))
        if not isinstance(loaded_history, list):
            raise ValueError("history file must contain a JSON list")
        previous_history = loaded_history
    history_path.write_text(
        json.dumps(previous_history + history, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(history[-1], ensure_ascii=False, sort_keys=True))
    print(f"saved training checkpoint to {checkpoint}")
    print("checkpoint remains unapproved until arena and calibration review")


if __name__ == "__main__":
    main()
