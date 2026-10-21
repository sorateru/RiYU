#!/usr/bin/env python3
"""Promote a sequence checkpoint only after sealed-holdout and paired-Arena gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--holdout-report", required=True)
    parser.add_argument("--arena-report", required=True)
    parser.add_argument("--candidate", choices=("a", "b"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-deal-in-ece", type=float, default=0.05)
    parser.add_argument("--max-rank-ece", type=float, default=0.10)
    parser.add_argument("--minimum-deal-ins", type=int, default=100)
    parser.add_argument("--minimum-seed-blocks", type=int, default=64)
    args = parser.parse_args()

    source = Path(args.checkpoint).resolve()
    output = Path(args.output).resolve()
    if output == source or output.exists():
        parser.error("output must be a new path different from the source checkpoint")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    if checkpoint.get("model_family") != "sequence-transformer-v2":
        raise ValueError("only sequence-transformer-v2 checkpoints can be promoted")
    if not checkpoint["model_config"].get("decision_heads", False):
        raise ValueError("promotion requires opponent danger/outcome decision heads")
    metadata = dict(checkpoint["metadata"])
    if not metadata.get("risk_calibrated", False):
        raise ValueError("risk head must be calibrated on validation before holdout")

    holdout_path = Path(args.holdout_report).resolve()
    holdout = json.loads(holdout_path.read_text(encoding="utf-8"))
    if holdout.get("evaluated_split") != "final_holdout":
        raise ValueError("evaluation report is not the sealed final holdout")
    if holdout.get("checkpoint_sha256") != digest:
        raise ValueError("holdout report checkpoint hash does not match candidate")
    deal_in = holdout.get("deal_in") or {}
    if not deal_in.get("calibration_applied", False):
        raise ValueError("holdout report did not apply risk calibration")
    _bounded_metric(deal_in, "ece_15", args.max_deal_in_ece)
    for relative, metrics in (holdout.get("deal_in_by_relative_opponent") or {}).items():
        _bounded_metric(metrics, "ece_15", args.max_deal_in_ece)
        positives = float(metrics.get("examples", 0.0)) * float(
            metrics.get("positive_rate", 0.0)
        )
        if positives < args.minimum_deal_ins:
            raise ValueError(
                f"relative opponent {relative} has only {positives:.0f} deal-ins; "
                f"need {args.minimum_deal_ins}"
            )
    rank_ece = float(holdout.get("rank_ece_15", math.inf))
    if not math.isfinite(rank_ece) or rank_ece > args.max_rank_ece:
        raise ValueError(f"rank ECE {rank_ece:.6f} exceeds gate {args.max_rank_ece}")

    arena_path = Path(args.arena_report).resolve()
    arena = json.loads(arena_path.read_text(encoding="utf-8"))
    if int(arena.get("config", {}).get("seed_blocks", 0)) < args.minimum_seed_blocks:
        raise ValueError("arena report has too few independent seed blocks")
    expected_verdict = f"{args.candidate}_better"
    if arena.get("verdict") != expected_verdict:
        raise ValueError(f"paired Arena verdict must be {expected_verdict}")
    identity = arena.get("models", {}).get(args.candidate, {})
    belief_identity = identity.get("belief_checkpoint") or {}
    if belief_identity.get("sha256") != digest:
        raise ValueError("Arena candidate did not use this sequence belief checkpoint")

    metadata["validated"] = True
    checkpoint["metadata"] = metadata
    checkpoint["promotion"] = {
        "promoted_at": time.time(),
        "source_checkpoint": str(source),
        "source_sha256": digest,
        "holdout_report": str(holdout_path),
        "arena_report": str(arena_path),
        "candidate": args.candidate,
        "gates": {
            "max_deal_in_ece": args.max_deal_in_ece,
            "max_rank_ece": args.max_rank_ece,
            "minimum_deal_ins_per_relative_opponent": args.minimum_deal_ins,
            "minimum_seed_blocks": args.minimum_seed_blocks,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    print(json.dumps(checkpoint["promotion"], sort_keys=True))


def _bounded_metric(metrics: dict[str, object], key: str, maximum: float) -> None:
    value = float(metrics.get(key, math.inf))
    if not math.isfinite(value) or value > maximum:
        raise ValueError(f"{key} {value:.6f} exceeds gate {maximum}")


if __name__ == "__main__":
    main()
