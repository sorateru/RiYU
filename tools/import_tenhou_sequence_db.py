#!/usr/bin/env python3
"""Convert an offline Tenhou SQLite archive to sequence decision/v4 shards."""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

from mahjong_ai.data import (
    SEQUENCE_DATASET_FORMAT,
    build_sequence_decision_records,
    create_human_dataset_manifest,
    write_sequence_parquet,
)
from mahjong_ai.data.tenhou import convert_document, decode_archive_xml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-decisions", type=int, default=300_000)
    parser.add_argument("--shard-decisions", type=int, default=25_000)
    parser.add_argument("--max-games", type=int, default=0)
    parser.add_argument("--offset-games", type=int, default=0)
    parser.add_argument("--future-draw-horizon", type=int, default=12)
    parser.add_argument("--authorization-reference", required=True)
    args = parser.parse_args()
    if args.max_decisions <= 0 or args.shard_decisions <= 0:
        parser.error("decision limits must be positive")
    if args.offset_games < 0 or args.future_draw_horizon <= 0:
        parser.error("offset must be non-negative and horizon must be positive")
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        parser.error("output directory must be absent or empty")
    output.mkdir(parents=True, exist_ok=True)

    database = Path(args.database).resolve()
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    cursor = connection.execute(
        "SELECT log_id, log_content FROM logs ORDER BY log_id LIMIT -1 OFFSET ?",
        (args.offset_games,),
    )
    pending = []
    shards: list[Path] = []
    stats: dict = {
        "format": SEQUENCE_DATASET_FORMAT,
        "games_seen": 0,
        "rounds_seen": 0,
        "rounds_accepted": 0,
        "rounds_rejected": 0,
        "decisions": 0,
        "errors": {},
        "started_at": time.time(),
    }
    try:
        for raw_log_id, blob in cursor:
            stats["games_seen"] += 1
            try:
                rounds = convert_document(decode_archive_xml(blob), str(raw_log_id))
            except Exception as error:
                _record_error(stats, error)
                if args.max_games and stats["games_seen"] >= args.max_games:
                    break
                continue
            for converted in rounds:
                stats["rounds_seen"] += 1
                try:
                    records = build_sequence_decision_records(
                        converted.log, future_draw_horizon=args.future_draw_horizon
                    )
                except Exception as error:
                    stats["rounds_rejected"] += 1
                    _record_error(stats, error)
                    continue
                if not records:
                    continue
                remaining = args.max_decisions - stats["decisions"]
                records = records[:remaining]
                pending.extend(records)
                stats["decisions"] += len(records)
                stats["rounds_accepted"] += 1
                while len(pending) >= args.shard_decisions:
                    shard = output / f"part-{len(shards):05d}.parquet"
                    write_sequence_parquet(pending[: args.shard_decisions], shard)
                    del pending[: args.shard_decisions]
                    shards.append(shard)
                    _write_progress(output, stats, complete=False)
                if stats["decisions"] >= args.max_decisions:
                    break
            if stats["decisions"] >= args.max_decisions:
                break
            if args.max_games and stats["games_seen"] >= args.max_games:
                break
    finally:
        connection.close()
    if pending:
        shard = output / f"part-{len(shards):05d}.parquet"
        write_sequence_parquet(pending, shard)
        shards.append(shard)
    if not shards:
        raise RuntimeError("no valid sequence decisions were converted")
    create_human_dataset_manifest(
        output / "manifest.json",
        [str(path) for path in shards],
        dataset_id="tenhou-phoenix-anonymized-sequence-v4",
        source="Tenhou Phoenix offline archive (top room)",
        rights_basis="explicit_permission",
        authorization_reference=args.authorization_reference,
        redistribution_allowed=False,
    )
    _write_progress(output, stats, complete=True)
    print(json.dumps(stats, ensure_ascii=False, sort_keys=True))


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
