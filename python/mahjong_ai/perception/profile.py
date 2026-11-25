"""Validation and portable, local-only bundles for Vision Adapter profiles."""

from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import datetime, timezone
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from mahjong_ai.schema import TILE_TO_INDEX

from .parser import LayoutConfig

PROFILE_FORMAT = "riyu-vision-profile/v1"
TEMPLATE_MANIFEST_FORMAT = "riyu-vision-template-manifest/v1"
RED_FIVES = ("m5_red", "p5_red", "s5_red")
HONOR_FILENAMES = {
    "z1": "east",
    "z2": "south",
    "z3": "west",
    "z4": "north",
    "z5": "white",
    "z6": "green",
    "z7": "red",
}


@dataclass(frozen=True)
class ProfileIssue:
    severity: str
    code: str
    detail: str


@dataclass(frozen=True)
class VisionProfileReport:
    complete: bool
    tile_labels: tuple[str, ...]
    missing_tile_labels: tuple[str, ...]
    issues: tuple[ProfileIssue, ...]
    capabilities: dict[str, bool]
    template_metadata: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "complete": self.complete,
            "tile_labels": list(self.tile_labels),
            "missing_tile_labels": list(self.missing_tile_labels),
            "issues": [asdict(issue) for issue in self.issues],
            "capabilities": self.capabilities,
            "template_metadata": self.template_metadata,
        }


def inspect_vision_profile(
    layout_path: str | Path,
    template_directory: str | Path,
) -> VisionProfileReport:
    layout = LayoutConfig.load(layout_path)
    root = Path(template_directory)
    labels: set[str] = set()
    png_files = list(root.rglob("*.png"))
    for path in root.glob("*.png"):
        label = path.stem.split("__", 1)[0]
        if label == "empty":
            continue
        if label in HONOR_FILENAMES:
            label = HONOR_FILENAMES[label]
        if label.removesuffix("_red") in TILE_TO_INDEX:
            labels.add(label)
    required_tiles = set(TILE_TO_INDEX) | set(RED_FIVES)
    missing_tiles = sorted(required_tiles - labels)
    issues: list[ProfileIssue] = []
    if missing_tiles:
        issues.append(
            ProfileIssue(
                "error",
                "tile_templates_incomplete",
                "missing tile labels: " + ", ".join(missing_tiles),
            )
        )

    expected_slots = {"hand": 13, "drawn_tile": 1, "dora_indicators": 5}
    expected_slots.update({f"river_{player}": 18 for player in range(4)})
    for name, count in expected_slots.items():
        observed = len(layout.slots.get(name, ()))
        if observed < count:
            issues.append(
                ProfileIssue(
                    "error",
                    "slot_coverage_incomplete",
                    f"{name} has {observed} slots; expected at least {count}",
                )
            )

    meld_regions = {
        f"meld_{player}_{group}"
        for player in range(4)
        for group in range(4)
    }
    configured_melds = set(layout.slots) | set(layout.meld_layouts) | set(layout.meld_scans)
    missing_melds = sorted(meld_regions - configured_melds)
    if missing_melds:
        issues.append(
            ProfileIssue(
                "error",
                "meld_regions_incomplete",
                "missing meld regions: " + ", ".join(missing_melds),
            )
        )
    short_melds = sorted(
        name
        for name in meld_regions
        & set(layout.slots)
        - set(layout.meld_layouts)
        - set(layout.meld_scans)
        if len(layout.slots[name]) < 4
    )
    if short_melds:
        issues.append(
            ProfileIssue(
                "error",
                "meld_slot_coverage_incomplete",
                "meld regions need four slots for kan: " + ", ".join(short_melds),
            )
        )

    marker_groups = {
        "winds": {f"wind_{wind}" for wind in ("east", "south", "west", "north")},
        "dealers": {f"dealer_{player}" for player in range(4)},
        "riichi": {
            marker
            for player in range(4)
            for marker in (f"riichi_{player}", f"double_riichi_{player}")
        },
        "results": {f"result_{kind}" for kind in ("ron", "tsumo", "draw", "abortive_draw")},
        "winners": {f"winner_{player}" for player in range(4)},
        "losers": {f"loser_{player}" for player in range(4)},
        "draw_reasons": {
            f"draw_reason_{reason}"
            for reason in (
                "nine_terminals",
                "four_winds",
                "four_riichi",
                "four_kan",
                "triple_ron",
            )
        },
    }
    marker_root = root / "markers"
    marker_files = {
        path.stem.split("__", 1)[0]
        for path in marker_root.glob("*.png")
        if "__on" in path.stem
    }
    marker_capabilities: dict[str, bool] = {}
    for group, required in marker_groups.items():
        configured = required <= set(layout.markers)
        templated = required <= marker_files
        marker_capabilities[group] = configured and templated

    missing_meld_geometry = sorted(
        meld_regions - set(layout.meld_layouts) - set(layout.meld_scans)
    )
    if missing_meld_geometry:
        issues.append(
            ProfileIssue(
                "error",
                "meld_geometry_incomplete",
                "meld resync needs candidate geometry for: "
                + ", ".join(missing_meld_geometry),
            )
        )
    needs_back = any(
        slot.role == "back"
        for patterns in layout.meld_layouts.values()
        for pattern in patterns
        for slot in pattern.slots
    ) or bool(layout.meld_scans)
    has_back = any((root / "visuals").glob("back*.png"))
    if needs_back and not has_back:
        issues.append(
            ProfileIssue(
                "error",
                "meld_back_template_missing",
                "ankan geometry is configured but templates/visuals/back__*.png is missing",
            )
        )

    horizontal_rivers = {
        name
        for name, values in layout.slot_variants.items()
        if name.startswith("river_")
        and values
        and all(any(candidate.orientation == "horizontal" for candidate in group) for group in values)
    }
    wind_ocr = "wind" in layout.text_fields
    dealer_ocr = "dealer" in layout.text_fields
    score_delta_ocr = all(
        f"delta_{player}" in layout.text_fields for player in range(4)
    )
    round_identity = (marker_capabilities["winds"] or wind_ocr) and (
        marker_capabilities["dealers"] or dealer_ocr
    )
    riichi_detection = marker_capabilities["riichi"] or len(horizontal_rivers) == 4
    # Result kind, winner and loser are derived from stable score deltas plus
    # the current GameState phase.  Only abortive-draw reasons remain semantic
    # UI templates because they are not always recoverable from public state.
    terminal_detection = score_delta_ocr and marker_capabilities["draw_reasons"]

    if not (marker_capabilities["winds"] or wind_ocr):
        issues.append(
            ProfileIssue(
                "error",
                "wind_evidence_incomplete",
                "configure OCR text field wind or all four wind marker templates",
            )
        )
    if not (marker_capabilities["dealers"] or dealer_ocr):
        issues.append(
            ProfileIssue(
                "error",
                "dealer_evidence_incomplete",
                "configure OCR integer field dealer or all four dealer marker templates",
            )
        )
    if not riichi_detection:
        issues.append(
            ProfileIssue(
                "error",
                "riichi_evidence_incomplete",
                "configure vertical/horizontal slot_variants for every river slot; "
                "riichi marker templates are only a fallback",
            )
        )
    if not score_delta_ocr:
        issues.append(
            ProfileIssue(
                "error",
                "result_score_deltas_incomplete",
                "configure OCR fields delta_0..delta_3 to infer ron, tsumo, draw, winner and loser",
            )
        )
    if not marker_capabilities["draw_reasons"]:
        missing = sorted(
            marker_groups["draw_reasons"] - (set(layout.markers) & marker_files)
        )
        issues.append(
            ProfileIssue(
                "error",
                "draw_reasons_markers_incomplete",
                "abortive draws still need exact reason templates: " + ", ".join(missing),
            )
        )

    anchor_files = {
        path.stem.split("__", 1)[0]
        for path in (root / "anchors").glob("*.png")
    }
    required_anchors = {
        name for name, config in layout.scene_anchors.items() if config.required
    }
    scene_guard = bool(required_anchors) and required_anchors <= anchor_files
    if not scene_guard:
        issues.append(
            ProfileIssue(
                "error",
                "scene_guard_incomplete",
                "configure at least one required scene anchor and create its local template",
            )
        )

    manifest_path = root / "template_manifest.json"
    manifest_entries: dict[str, Any] = {}
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("format") != TEMPLATE_MANIFEST_FORMAT:
                raise ValueError("unsupported format")
            raw_entries = manifest.get("templates")
            if not isinstance(raw_entries, dict):
                raise ValueError("templates is not an object")
            manifest_entries = raw_entries
        except (ValueError, json.JSONDecodeError) as error:
            issues.append(
                ProfileIssue("error", "template_manifest_invalid", str(error))
            )
    relative_pngs = {path.relative_to(root).as_posix() for path in png_files}
    missing_metadata = sorted(relative_pngs - set(manifest_entries))
    stale_metadata = sorted(set(manifest_entries) - relative_pngs)
    if missing_metadata:
        issues.append(
            ProfileIssue(
                "warning",
                "template_metadata_incomplete",
                f"{len(missing_metadata)} template files have no provenance metadata",
            )
        )
    if stale_metadata:
        issues.append(
            ProfileIssue(
                "warning",
                "template_metadata_stale",
                f"{len(stale_metadata)} metadata entries have no template file",
            )
        )

    capabilities = {
        "all_tiles": not missing_tiles,
        "all_rivers": all(
            len(layout.slots.get(f"river_{player}", ())) >= 18 for player in range(4)
        ),
        "all_melds": not missing_melds and not short_melds,
        "midround_meld_resync": (
            not missing_melds
            and not short_melds
            and not missing_meld_geometry
            and (not needs_back or has_back)
            and riichi_detection
        ),
        "automatic_round_identity": round_identity,
        "riichi_detection": riichi_detection,
        "score_delta_terminal_inference": score_delta_ocr,
        "winner_loser_inference": score_delta_ocr,
        "terminal_detection": terminal_detection,
        "scene_guard": scene_guard,
    }
    return VisionProfileReport(
        complete=not any(issue.severity == "error" for issue in issues),
        tile_labels=tuple(sorted(labels)),
        missing_tile_labels=tuple(missing_tiles),
        issues=tuple(issues),
        capabilities=capabilities,
        template_metadata={
            "template_files": len(relative_pngs),
            "registered": len(relative_pngs & set(manifest_entries)),
            "missing": len(missing_metadata),
            "stale": len(stale_metadata),
        },
    )


def record_template_metadata(
    template_directory: str | Path,
    template_path: str | Path,
    *,
    label: str,
    orientation: int,
    target_region: str,
    source_image: str,
    crop: tuple[int, int, int, int],
    image_size: tuple[int, int],
) -> Path:
    """Atomically add/update local provenance for an extracted template."""

    root = Path(template_directory).resolve()
    template = Path(template_path).resolve()
    relative = template.relative_to(root).as_posix()
    manifest_path = root / "template_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") != TEMPLATE_MANIFEST_FORMAT:
            raise ValueError("unsupported vision template manifest format")
    else:
        manifest = {"format": TEMPLATE_MANIFEST_FORMAT, "templates": {}}
    entries = manifest.setdefault("templates", {})
    entries[relative] = {
        "label": label,
        "orientation": orientation,
        "target_region": target_region,
        "source_image": source_image,
        "crop": list(crop),
        "image_size": list(image_size),
        "registered_at": datetime.now(timezone.utc).isoformat(),
    }
    temporary = manifest_path.with_name(f".{manifest_path.name}.tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest_path)
    return manifest_path


def prune_template_metadata(template_directory: str | Path) -> tuple[Path, int]:
    """Remove provenance entries whose local template was archived or deleted."""

    root = Path(template_directory).resolve()
    manifest_path = root / "template_manifest.json"
    if not manifest_path.is_file():
        return manifest_path, 0
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != TEMPLATE_MANIFEST_FORMAT:
        raise ValueError("unsupported vision template manifest format")
    entries = manifest.get("templates")
    if not isinstance(entries, dict):
        raise ValueError("vision template manifest templates is not an object")
    existing = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*.png")
    }
    stale = set(entries) - existing
    for relative in stale:
        del entries[relative]
    temporary = manifest_path.with_name(f".{manifest_path.name}.tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest_path)
    return manifest_path, len(stale)


def export_vision_profile(
    config_path: str | Path,
    destination: str | Path,
) -> Path:
    """Bundle a user's own config/layout/templates without publishing them."""

    config_path = Path(config_path).resolve()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    base = config_path.parent
    layout_path = _resolve(base, raw["layout"])
    template_root = _resolve(base, raw["templates"])
    bundled_config = dict(raw)
    bundled_config["layout"] = "layout.json"
    bundled_config["templates"] = "templates"
    files: dict[str, bytes] = {
        "config.json": json.dumps(
            bundled_config, ensure_ascii=False, indent=2, sort_keys=True
        ).encode("utf-8"),
        "layout.json": layout_path.read_bytes(),
    }
    for path in sorted(template_root.rglob("*.png")):
        files[f"templates/{path.relative_to(template_root).as_posix()}"] = path.read_bytes()
    metadata = template_root / "template_manifest.json"
    if metadata.is_file():
        files["templates/template_manifest.json"] = metadata.read_bytes()
    manifest = {
        "format": PROFILE_FORMAT,
        "files": {
            name: hashlib.sha256(value).hexdigest() for name, value in files.items()
        },
    }
    files["manifest.json"] = json.dumps(
        manifest, ensure_ascii=False, indent=2, sort_keys=True
    ).encode("utf-8")
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, value in files.items():
            archive.writestr(name, value)
    return destination


def import_vision_profile(bundle: str | Path, destination: str | Path) -> Path:
    bundle = Path(bundle).resolve()
    destination = Path(destination).resolve()
    if destination.exists():
        if not destination.is_dir() or any(destination.iterdir()):
            raise ValueError("profile destination must be an empty directory")
    with zipfile.ZipFile(bundle) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("profile contains duplicate paths")
        for name in names:
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("profile contains an unsafe path")
        if "manifest.json" not in names:
            raise ValueError("profile has no manifest.json")
        manifest = json.loads(archive.read("manifest.json"))
        if manifest.get("format") != PROFILE_FORMAT:
            raise ValueError("unsupported Vision profile format")
        manifest_files = manifest.get("files")
        if not isinstance(manifest_files, dict) or not manifest_files:
            raise ValueError("profile manifest has no files")
        archived_files = {name for name in names if not name.endswith("/")}
        expected_files = set(manifest_files) | {"manifest.json"}
        if archived_files != expected_files:
            raise ValueError("profile contains unmanifested or unexpected files")
        for name, digest in manifest_files.items():
            if name not in names:
                raise ValueError(f"profile file is missing: {name}")
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise ValueError(f"profile checksum mismatch: {name}")
        destination.mkdir(parents=True, exist_ok=True)
        for name in names:
            if name.endswith("/"):
                continue
            output = destination.joinpath(*PurePosixPath(name).parts)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(archive.read(name))
    return destination / "config.json"


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()
