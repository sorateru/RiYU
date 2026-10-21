#!/usr/bin/env python3
"""Combine 200k/500k/1M validation and paired-Arena runs into a learning curve."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", help="JSON array of scaling-run entries")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--plateau-rank-improvement", type=float, default=0.01,
        help="flag adjacent data scales improving average rank by less than this",
    )
    args = parser.parse_args()
    entries = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if not isinstance(entries, list) or len(entries) < 2:
        raise ValueError("scaling manifest needs at least two run entries")
    rows = sorted((_load_entry(entry) for entry in entries), key=lambda row: row["decisions"])
    if len({row["decisions"] for row in rows}) != len(rows):
        raise ValueError("decision scales must be unique")
    for index, row in enumerate(rows):
        if index == 0:
            row["incremental"] = None
            continue
        previous = rows[index - 1]
        rank_improvement = previous["average_rank"] - row["average_rank"]
        row["incremental"] = {
            "additional_decisions": row["decisions"] - previous["decisions"],
            "policy_accuracy_change": row["policy_accuracy"] - previous["policy_accuracy"],
            "average_rank_improvement": rank_improvement,
            "average_score_delta_change": (
                row["average_score_delta"] - previous["average_score_delta"]
            ),
            "plateau_flag": rank_improvement < args.plateau_rank_improvement,
        }
    report = {
        "format": "mahjong-ai-scaling-curve/v1",
        "rows": rows,
        "interpretation": (
            "A plateau flag means Arena rank did not improve enough at the next data scale; "
            "inspect model/label/search design rather than assuming more data is sufficient."
        ),
    }
    output = Path(args.output)
    if output.exists():
        parser.error("output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


def _load_entry(entry: dict[str, Any]) -> dict[str, Any]:
    validation = json.loads(Path(entry["validation_report"]).read_text(encoding="utf-8"))
    arena = json.loads(Path(entry["arena_report"]).read_text(encoding="utf-8"))
    candidate = str(entry.get("candidate", "a"))
    summary = arena["summary"][candidate]
    decisions = int(entry["decisions"])
    if decisions <= 0:
        raise ValueError("decisions must be positive")
    return {
        "label": str(entry.get("label", decisions)),
        "decisions": decisions,
        "validation_report": str(Path(entry["validation_report"]).resolve()),
        "arena_report": str(Path(entry["arena_report"]).resolve()),
        "policy_accuracy": float(validation["policy_accuracy"]),
        "policy_nll": float(validation["policy_nll"]),
        "average_rank": float(summary["average_rank"]),
        "average_rank_95ci": summary["average_rank_95ci_unblocked"],
        "average_score_delta": float(summary["average_score_delta"]),
        "win_rate": float(summary["win_rate_per_seat_hand"]),
        "deal_in_rate": float(summary["deal_in_rate_per_seat_hand"]),
        "seed_blocks": int(arena["config"]["seed_blocks"]),
    }


if __name__ == "__main__":
    main()
