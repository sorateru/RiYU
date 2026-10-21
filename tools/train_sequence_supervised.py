#!/usr/bin/env python3
"""Train the observable sequence Transformer on decision/v2 or labelled v3 shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from mahjong_ai.ai import MahjongNetwork, ModelConfig, SequenceMahjongNetwork, SequenceModelConfig
from mahjong_ai.ai import MODEL_OUTPUT_VERSION, SEQUENCE_ARCHITECTURE_VERSION
from mahjong_ai.data import SEQUENCE_DATASET_FORMAT, SEQUENCE_LABEL_VERSION
from mahjong_ai.training import (
    ParquetSequenceDataset,
    SequenceTrainer,
    SequenceTrainerConfig,
    split_sequence_train_validation_holdout,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="+")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--scheduler-t-max-epochs", type=int, default=50)
    parser.add_argument("--minimum-learning-rate-ratio", type=float, default=0.05)
    parser.add_argument(
        "--auto-positive-weights",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="estimate bounded tenpai/wait/deal-in class weights from training only",
    )
    parser.add_argument("--maximum-positive-weight", type=float, default=30.0)
    parser.add_argument("--policy-label-smoothing", type=float, default=0.02)
    parser.add_argument("--warm-start")
    parser.add_argument(
        "--legacy-warm-start",
        help="initialize snapshot/policy/value layers from an observable-planes-v1 checkpoint",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--holdout-fraction", type=float, default=0.1)
    parser.add_argument(
        "--decision-heads",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="train opponent-specific danger, loss-bin, and action-outcome heads",
    )
    parser.add_argument(
        "--opponent-temporal-latents",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "pool each relative player's contextual public events into shared "
            "tenpai/wait/shanten/risk latents"
        ),
    )
    parser.add_argument(
        "--preset", choices=("tiny", "base"), default="base",
        help="tiny is intended only for pipeline smoke tests",
    )
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        parser.error("epochs and batch size must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        parser.error("learning rate must be positive and weight decay cannot be negative")
    if args.scheduler_t_max_epochs <= 0:
        parser.error("scheduler t-max epochs must be positive")
    if not 0 <= args.minimum_learning_rate_ratio <= 1:
        parser.error("minimum learning rate ratio must be in [0, 1]")
    if args.maximum_positive_weight < 1:
        parser.error("maximum positive weight must be at least one")
    if not 0 <= args.policy_label_smoothing < 1:
        parser.error("policy label smoothing must be in [0, 1)")
    if args.warm_start and args.legacy_warm_start:
        parser.error("--warm-start and --legacy-warm-start are mutually exclusive")
    if args.opponent_temporal_latents and not args.decision_heads:
        parser.error("--opponent-temporal-latents requires --decision-heads")
    # Model construction happens before SequenceTrainer seeds its optimizer.
    # Seed here as well so newly initialized Transformer heads are reproducible.
    torch.manual_seed(args.seed)
    dataset_paths = [Path(value) for value in args.datasets]
    digest = hashlib.sha256()
    artifact_hashes: dict[str, str] = {}
    for dataset_path in sorted(dataset_paths):
        resolved = dataset_path.resolve()
        payload = dataset_path.read_bytes()
        digest.update(str(resolved).encode())
        digest.update(payload)
        artifact_hashes[str(resolved)] = hashlib.sha256(payload).hexdigest()
    for manifest_path in sorted({path.resolve().parent / "manifest.json" for path in dataset_paths}):
        if manifest_path.is_file():
            artifact_hashes[str(manifest_path)] = hashlib.sha256(
                manifest_path.read_bytes()
            ).hexdigest()
    fingerprint = digest.hexdigest()
    dataset = ParquetSequenceDataset(dataset_paths)
    if args.decision_heads and not dataset.has_decision_labels:
        raise ValueError(
            "--decision-heads requires newly converted shards with opponent danger, "
            "per-opponent loss, and terminal outcome labels"
        )
    train, validation, holdout = split_sequence_train_validation_holdout(
        dataset, args.validation_fraction, args.holdout_fraction
    )
    warm = None
    resume_optimizer = False
    if args.warm_start:
        warm = torch.load(args.warm_start, map_location=args.device, weights_only=False)
        if warm.get("model_family") != "sequence-transformer-v2":
            raise ValueError("warm-start checkpoint is not sequence-transformer-v2")
        warm_config = dict(warm["model_config"])
        original_decision_heads = bool(warm_config.get("decision_heads", False))
        original_temporal_latents = bool(
            warm_config.get("opponent_temporal_latents", False)
        )
        warm_config["decision_heads"] = args.decision_heads
        warm_config["opponent_temporal_latents"] = args.opponent_temporal_latents
        model = SequenceMahjongNetwork(SequenceModelConfig(**warm_config))
        incompatible = model.load_state_dict(warm["model_state"], strict=False)
        unexpected = [
            key
            for key in incompatible.unexpected_keys
            if not key.startswith("deal_in_score_head")
        ]
        forbidden_missing = [
            key
            for key in incompatible.missing_keys
            if not key.startswith(
                (
                    "opponent_seat_embedding",
                    "opponent_projection",
                    "opponent_event_query",
                    "opponent_event_projection",
                    "opponent_temporal_fusion",
                    "opponent_temporal_norm",
                    "opponent_tenpai_from_latent",
                    "opponent_wait_from_latent",
                    "opponent_shanten_from_latent",
                    "response_head",
                    "call_type_head",
                    "riichi_head",
                    "deal_in_head",
                    "deal_in_point_head",
                    "outcome_action_projection",
                    "action_outcome_head",
                )
            )
        ]
        if unexpected or forbidden_missing:
            raise ValueError(
                f"warm-start mismatch: missing={forbidden_missing}, unexpected={unexpected}"
            )
        resume_optimizer = (
            original_decision_heads == args.decision_heads
            and original_temporal_latents == args.opponent_temporal_latents
        )
    elif args.legacy_warm_start:
        legacy_checkpoint = torch.load(
            args.legacy_warm_start, map_location=args.device, weights_only=False
        )
        legacy_config = ModelConfig(**legacy_checkpoint["model_config"])
        legacy = MahjongNetwork(legacy_config)
        legacy.load_state_dict(legacy_checkpoint["model_state"])
        model = SequenceMahjongNetwork(
            SequenceModelConfig(
                snapshot_channels=legacy_config.channels,
                snapshot_blocks=legacy_config.residual_blocks,
                snapshot_dim=legacy_config.representation_dim,
                fused_dim=legacy_config.representation_dim,
                action_dim=legacy_config.action_dim,
                decision_heads=args.decision_heads,
                opponent_temporal_latents=args.opponent_temporal_latents,
            )
        )
        _transfer_legacy_weights(model, legacy)
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
                decision_heads=args.decision_heads,
                opponent_temporal_latents=args.opponent_temporal_latents,
            )
        )
    else:
        model = SequenceMahjongNetwork(
            SequenceModelConfig(
                decision_heads=args.decision_heads,
                opponent_temporal_latents=args.opponent_temporal_latents,
            )
        )
    trainer = SequenceTrainer(
        model,
        SequenceTrainerConfig(
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            auto_positive_weights=args.auto_positive_weights,
            maximum_positive_weight=args.maximum_positive_weight,
            policy_label_smoothing=args.policy_label_smoothing,
            scheduler_t_max_epochs=args.scheduler_t_max_epochs,
            minimum_learning_rate_ratio=args.minimum_learning_rate_ratio,
            device=args.device,
        ),
    )
    start_epoch = 0
    if warm is not None:
        start_epoch = trainer.restore_training_state(
            warm, restore_optimizer=resume_optimizer
        )
    history = trainer.fit(
        train,
        validation,
        args.checkpoint,
        model_id=Path(args.checkpoint).stem,
        dataset_fingerprint=fingerprint,
        start_epoch=start_epoch,
        dataset_manifest_hashes=artifact_hashes,
    )
    summary = {
        "completed_at": time.time(),
        "model_family": "sequence-transformer-v2",
        "architecture_version": SEQUENCE_ARCHITECTURE_VERSION,
        "model_output_version": MODEL_OUTPUT_VERSION,
        "decision_schema_version": SEQUENCE_DATASET_FORMAT,
        "dataset_version": SEQUENCE_DATASET_FORMAT,
        "label_version": SEQUENCE_LABEL_VERSION,
        "preset": args.preset,
        "decision_heads": args.decision_heads,
        "opponent_temporal_latents": args.opponent_temporal_latents,
        "datasets": [str(path.resolve()) for path in dataset_paths],
        "dataset_fingerprint": fingerprint,
        "dataset_manifest_hashes": artifact_hashes,
        "decisions": len(dataset),
        "train_decisions": len(train),
        "validation_decisions": len(validation),
        "sealed_holdout_decisions": len(holdout),
        "holdout_policy": "never evaluated during training or checkpoint selection",
        "epochs": args.epochs,
        "seed": args.seed,
        "start_epoch": start_epoch,
        "effective_positive_weights": trainer.effective_positive_weights,
        "optimizer_resumed": bool(
            warm is not None
            and resume_optimizer
            and warm.get("optimizer_state") is not None
        ),
        "warm_start": str(Path(args.warm_start).resolve()) if args.warm_start else None,
        "legacy_warm_start": (
            str(Path(args.legacy_warm_start).resolve())
            if args.legacy_warm_start
            else None
        ),
        "history": history,
    }
    Path(args.checkpoint).with_suffix(".training.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(history[-1], ensure_ascii=False, sort_keys=True))
    print("checkpoint remains unapproved until an explicit arena/calibration review")


def _transfer_legacy_weights(
    model: SequenceMahjongNetwork, legacy: MahjongNetwork
) -> None:
    """Preserve the V1 snapshot policy while allowing event features to grow in."""

    model.snapshot_stem.load_state_dict(legacy.stem.state_dict())
    model.snapshot_backbone.load_state_dict(legacy.backbone.state_dict())
    model.snapshot_representation.load_state_dict(legacy.representation.state_dict())
    model.action_encoder.load_state_dict(legacy.action_encoder.state_dict())
    model.policy_query.load_state_dict(legacy.policy_query.state_dict())
    model.policy_bias.load_state_dict(legacy.policy_bias.state_dict())
    model.rank_head.load_state_dict(legacy.rank_head.state_dict())
    model.score_head.load_state_dict(legacy.score_head.state_dict())
    fusion = model.fusion[0]
    if not isinstance(fusion, torch.nn.Linear):
        raise TypeError("unexpected fusion layer")
    if model.config.fused_dim != model.config.snapshot_dim:
        raise ValueError("legacy transfer requires fused_dim == snapshot_dim")
    with torch.no_grad():
        fusion.weight.zero_()
        fusion.bias.zero_()
        fusion.weight[:, : model.config.snapshot_dim].copy_(
            torch.eye(model.config.snapshot_dim)
        )


if __name__ == "__main__":
    main()
