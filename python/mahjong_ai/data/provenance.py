"""Verifiable provenance gate for human-play decision datasets."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

HUMAN_DATASET_FORMAT = "mahjong-ai-human-dataset/v1"
ALLOWED_RIGHTS_BASES = frozenset({"user_owned", "explicit_permission", "open_license"})
SOURCES_REQUIRING_EXPLICIT_PERMISSION = ("tenhou", "天鳳", "mahjong soul", "雀魂")
SUPPORTED_DECISION_FORMATS = frozenset(
    {b"decision/v1", b"mahjong-ai-sequence-decision/v2"}
)


@dataclass(frozen=True)
class HumanDatasetManifest:
    dataset_id: str
    source: str
    rights_basis: str
    authorization_reference: str
    redistribution_allowed: bool
    files: tuple[Path, ...]
    fingerprint: str


def load_human_dataset_manifest(path: str | Path) -> HumanDatasetManifest:
    manifest_path = Path(path)
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read human dataset manifest: {error}") from error
    if raw.get("format") != HUMAN_DATASET_FORMAT:
        raise ValueError(f"expected human dataset format {HUMAN_DATASET_FORMAT}")
    dataset_id = raw.get("dataset_id")
    source = raw.get("source")
    rights_basis = raw.get("rights_basis")
    reference = raw.get("authorization_reference")
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise ValueError("human dataset manifest requires dataset_id")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("human dataset manifest requires source")
    if rights_basis not in ALLOWED_RIGHTS_BASES:
        raise ValueError("rights_basis must be user_owned, explicit_permission, or open_license")
    normalized_source = source.casefold() if isinstance(source, str) else ""
    if (
        any(name in normalized_source for name in SOURCES_REQUIRING_EXPLICIT_PERMISSION)
        and rights_basis != "explicit_permission"
    ):
        raise ValueError("this service source requires an explicit_permission manifest")
    if not isinstance(reference, str) or len(reference.strip()) < 3:
        raise ValueError("authorization_reference must identify the permission or license")
    if raw.get("machine_learning_allowed") is not True:
        raise ValueError("manifest must explicitly confirm machine_learning_allowed=true")
    if raw.get("contains_personal_identifiers") is not False:
        raise ValueError("human datasets must be anonymized before training")
    entries = raw.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("human dataset manifest requires at least one file")

    files: list[Path] = []
    verified_entries: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("human dataset file entries must be objects")
        relative = entry.get("path")
        expected_hash = entry.get("sha256")
        if not isinstance(relative, str) or Path(relative).name != relative:
            raise ValueError("human dataset paths must be plain filenames beside the manifest")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise ValueError(f"human dataset file {relative} has an invalid SHA-256")
        file_path = manifest_path.parent / relative
        try:
            actual_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
            metadata = pq.read_metadata(file_path)
        except Exception as error:
            raise ValueError(f"cannot read human dataset file {relative}: {error}") from error
        if actual_hash != expected_hash:
            raise ValueError(f"human dataset hash mismatch: {relative}")
        schema_metadata = metadata.metadata or {}
        dataset_format = schema_metadata.get(b"mahjong_ai_dataset")
        if dataset_format not in SUPPORTED_DECISION_FORMATS:
            raise ValueError(f"human dataset file {relative} has an unsupported decision format")
        files.append(file_path)
        verified_entries.append(
            {
                "path": relative,
                "sha256": actual_hash,
                "rows": metadata.num_rows,
                "decision_format": dataset_format.decode(),
            }
        )

    fingerprint_source = {
        "dataset_id": dataset_id.strip(),
        "source": source.strip(),
        "rights_basis": rights_basis,
        "authorization_reference": reference.strip(),
        "files": verified_entries,
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_source, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return HumanDatasetManifest(
        dataset_id=dataset_id.strip(),
        source=source.strip(),
        rights_basis=rights_basis,
        authorization_reference=reference.strip(),
        redistribution_allowed=raw.get("redistribution_allowed") is True,
        files=tuple(files),
        fingerprint=fingerprint,
    )


def create_human_dataset_manifest(
    output: str | Path,
    dataset_files: list[str | Path],
    *,
    dataset_id: str,
    source: str,
    rights_basis: str,
    authorization_reference: str,
    redistribution_allowed: bool = False,
) -> dict[str, Any]:
    output_path = Path(output)
    if rights_basis not in ALLOWED_RIGHTS_BASES:
        raise ValueError("unsupported rights_basis")
    entries: list[dict[str, str]] = []
    for value in dataset_files:
        file_path = Path(value)
        if file_path.parent.resolve() != output_path.parent.resolve():
            raise ValueError("dataset files and manifest must be in the same directory")
        entries.append(
            {
                "path": file_path.name,
                "sha256": hashlib.sha256(file_path.read_bytes()).hexdigest(),
            }
        )
    manifest = {
        "format": HUMAN_DATASET_FORMAT,
        "dataset_id": dataset_id,
        "source": source,
        "rights_basis": rights_basis,
        "authorization_reference": authorization_reference,
        "machine_learning_allowed": True,
        "redistribution_allowed": redistribution_allowed,
        "contains_personal_identifiers": False,
        "files": entries,
    }
    output_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    load_human_dataset_manifest(output_path)
    return manifest
