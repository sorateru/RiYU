#!/usr/bin/env python3
"""Wait for conversion shards, merge provenance, and train the human policy."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

from mahjong_ai.ai import MahjongNetwork, ModelConfig
from mahjong_ai.data import create_human_dataset_manifest, load_human_dataset_manifest
from mahjong_ai.training import ParquetDecisionDataset, Trainer, TrainerConfig, split_by_log


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--deadline", required=True, help="local ISO timestamp")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--authorization-reference", required=True)
    args = parser.parse_args()
    deadline = datetime.fromisoformat(args.deadline).timestamp()
    sources = [Path(value) for value in args.inputs]
    while not all((source / "manifest.json").exists() for source in sources):
        if time.time() >= deadline:
            raise TimeoutError("conversion did not complete before the overnight deadline")
        time.sleep(30)

    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("merged output directory must be absent or empty")
    output.mkdir(parents=True, exist_ok=True)
    merged_files: list[str] = []
    reports = []
    for source_index, source in enumerate(sources):
        manifest = load_human_dataset_manifest(source / "manifest.json")
        report = json.loads((source / "conversion-report.json").read_text(encoding="utf-8"))
        if report.get("complete") is not True:
            raise ValueError(f"incomplete conversion report in {source}")
        reports.append(report)
        for file_index, path in enumerate(manifest.files):
            destination = output / f"source-{source_index:02d}-part-{file_index:05d}.parquet"
            shutil.copy2(path, destination)
            merged_files.append(str(destination))
    create_human_dataset_manifest(
        output / "manifest.json",
        merged_files,
        dataset_id="tenhou-phoenix-2020-anonymized-overnight-v1",
        source="Tenhou Phoenix offline archive (top room)",
        rights_basis="explicit_permission",
        authorization_reference=args.authorization_reference,
        redistribution_allowed=False,
    )
    manifest = load_human_dataset_manifest(output / "manifest.json")
    dataset = ParquetDecisionDataset(manifest.files)
    train_set, validation_set = split_by_log(dataset)
    trainer = Trainer(
        MahjongNetwork(ModelConfig()),
        TrainerConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            device="cpu",
            num_workers=0,
        ),
    )
    checkpoint = Path(args.checkpoint)
    history = trainer.fit(
        train_set,
        validation_set,
        checkpoint,
        checkpoint.stem,
        manifest.fingerprint,
    )
    summary = {
        "completed_at": time.time(),
        "conversion_reports": reports,
        "dataset_fingerprint": manifest.fingerprint,
        "decisions": len(dataset),
        "epochs": args.epochs,
        "history": history,
    }
    checkpoint.with_suffix(".human-training.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
