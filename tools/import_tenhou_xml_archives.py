#!/usr/bin/env python3
"""Stream user-provided Tenhou XML bundles into sequence decision/v3 shards."""

from __future__ import annotations

import argparse
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from mahjong_ai.data import (
    build_sequence_decision_records,
    create_human_dataset_manifest,
    iter_tenhou_archive_xml,
    validate_tenhou_archive,
    write_sequence_parquet,
)
from mahjong_ai.data.tenhou import convert_document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", nargs="+")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-decisions", type=int, default=1_000_000)
    parser.add_argument("--max-decisions-per-archive", type=int, default=0)
    parser.add_argument("--max-games-per-archive", type=int, default=0)
    parser.add_argument("--shard-decisions", type=int, default=25_000)
    parser.add_argument("--future-draw-horizon", type=int, default=12)
    parser.add_argument("--authorization-reference", required=True)
    args = parser.parse_args()
    if args.max_decisions <= 0 or args.shard_decisions <= 0:
        parser.error("decision limits must be positive")
    if args.max_decisions_per_archive < 0 or args.max_games_per_archive < 0:
        parser.error("per-archive limits must be non-negative")
    if args.future_draw_horizon <= 0:
        parser.error("future draw horizon must be positive")

    archives = [Path(value).resolve() for value in args.archives]
    if len(set(archives)) != len(archives):
        parser.error("duplicate input archives are not allowed")
    for archive in archives:
        validate_tenhou_archive(archive)

    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        parser.error("output directory must be absent or empty")
    output.mkdir(parents=True, exist_ok=True)

    pending = []
    shards: list[Path] = []
    stats: dict = {
        "format": "mahjong-ai-sequence-decision/v3",
        "archives": [],
        "games_seen": 0,
        "rounds_seen": 0,
        "rounds_accepted": 0,
        "rounds_rejected": 0,
        "decisions": 0,
        "errors": {},
        "started_at": time.time(),
    }
    stop_all = False
    for archive in archives:
        archive_stats = {
            "name": archive.name,
            "bytes": archive.stat().st_size,
            "games_seen": 0,
            "decisions": 0,
        }
        stats["archives"].append(archive_stats)
        try:
            documents = iter_tenhou_archive_xml(archive)
            for raw_log_id, payload in documents:
                archive_stats["games_seen"] += 1
                stats["games_seen"] += 1
                try:
                    rounds = convert_document(
                        ET.fromstring(payload),
                        raw_log_id,
                        source="Tenhou Phoenix user-provided XML archive",
                    )
                except Exception as error:
                    _record_error(stats, error)
                    if _archive_limit_reached(args, archive_stats):
                        break
                    continue
                for converted in rounds:
                    stats["rounds_seen"] += 1
                    try:
                        records = build_sequence_decision_records(
                            converted.log,
                            future_draw_horizon=args.future_draw_horizon,
                        )
                    except Exception as error:
                        stats["rounds_rejected"] += 1
                        _record_error(stats, error)
                        continue
                    remaining_total = args.max_decisions - stats["decisions"]
                    remaining_archive = (
                        args.max_decisions_per_archive - archive_stats["decisions"]
                        if args.max_decisions_per_archive
                        else remaining_total
                    )
                    records = records[: min(remaining_total, remaining_archive)]
                    if records:
                        pending.extend(records)
                        count = len(records)
                        stats["decisions"] += count
                        archive_stats["decisions"] += count
                        stats["rounds_accepted"] += 1
                    while len(pending) >= args.shard_decisions:
                        shard = output / f"part-{len(shards):05d}.parquet"
                        write_sequence_parquet(pending[: args.shard_decisions], shard)
                        del pending[: args.shard_decisions]
                        shards.append(shard)
                        _write_progress(output, stats, complete=False)
                    if stats["decisions"] >= args.max_decisions:
                        stop_all = True
                        break
                    if _archive_limit_reached(args, archive_stats):
                        break
                if stop_all or _archive_limit_reached(args, archive_stats):
                    break
        except Exception as error:
            _record_error(stats, error)
            archive_stats["fatal_error"] = f"{type(error).__name__}: {str(error)[:300]}"
        if stop_all:
            break

    if pending:
        shard = output / f"part-{len(shards):05d}.parquet"
        write_sequence_parquet(pending, shard)
        shards.append(shard)
    if not shards:
        raise RuntimeError("no valid sequence decisions were converted")
    create_human_dataset_manifest(
        output / "manifest.json",
        [str(path) for path in shards],
        dataset_id="tenhou-phoenix-user-xml-sequence-v3",
        source="Tenhou Phoenix user-provided XML archive",
        rights_basis="explicit_permission",
        authorization_reference=args.authorization_reference,
        redistribution_allowed=False,
    )
    _write_progress(output, stats, complete=True)
    print(json.dumps(stats, ensure_ascii=False, sort_keys=True))


def _archive_limit_reached(args: argparse.Namespace, stats: dict) -> bool:
    return bool(
        (args.max_games_per_archive and stats["games_seen"] >= args.max_games_per_archive)
        or (
            args.max_decisions_per_archive
            and stats["decisions"] >= args.max_decisions_per_archive
        )
    )


def _record_error(stats: dict, error: Exception) -> None:
    name = f"{type(error).__name__}: {str(error)[:180]}"
    stats["errors"][name] = stats["errors"].get(name, 0) + 1


def _write_progress(output: Path, stats: dict, *, complete: bool) -> None:
    report = dict(stats)
    report["complete"] = complete
    report["elapsed_seconds"] = time.time() - stats["started_at"]
    (output / "conversion-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
