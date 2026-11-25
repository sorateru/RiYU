#!/usr/bin/env python3
"""Benchmark saved Vision frames from slots through events and synchronization."""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2

from mahjong_ai.analysis import RealtimeStateError, RealtimeStateManager
from mahjong_ai.perception import (
    StableObservation,
    VisionAdapterConfig,
    VisionAdapterRuntime,
    VisionDeltaError,
)

FORMAT = "riyu-vision-benchmark/v1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("frames", help="PNG file or directory")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--annotations", help="optional riyu-vision-benchmark/v1 JSON")
    parser.add_argument("--output", help="write the JSON report atomically")
    parser.add_argument("--debug-dir", help="save failed frames, ROIs, and diagnostics")
    parser.add_argument("--quiet", action="store_true", help="suppress stdout JSON")
    args = parser.parse_args()
    if not 1 <= args.repeat <= 100:
        parser.error("--repeat must be in 1..=100")
    if args.annotations and args.repeat != 1:
        parser.error("annotated event benchmarks require --repeat 1")

    source = Path(args.frames).resolve()
    paths = [source] if source.is_file() else sorted(source.glob("*.png"))
    if not paths:
        parser.error("no PNG frames found")
    annotations = _load_annotations(args.annotations)
    expected_by_file = {
        str(value["file"]): value for value in annotations.get("frames", [])
    }
    runtime = VisionAdapterRuntime(VisionAdapterConfig.load(args.config))
    _initialize_manager(runtime, annotations)
    debug_root = None if not args.debug_dir else Path(args.debug_dir).resolve()
    if debug_root is not None:
        debug_root.mkdir(parents=True, exist_ok=True)

    timings: list[float] = []
    alignment_scales: list[float] = []
    alignment_rotations: list[float] = []
    alignment_offsets: list[float] = []
    scene_misses = 0
    unknown_total = 0
    tile_classifications = tile_total = tile_correct = tile_errors = tile_unknown = 0
    meld_total = meld_correct = meld_unknown = 0
    orientation_total = orientation_correct = 0
    ocr_total = ocr_correct = 0
    marker_total = marker_correct = 0
    expected_event_total = event_correct = false_events = missed_events = 0
    desync_count = resync_success = resync_failure = 0
    round_ok: dict[str, bool] = defaultdict(lambda: True)
    game_ok: dict[str, bool] = defaultdict(lambda: True)
    failures: list[dict[str, Any]] = []
    primed = False

    for _ in range(args.repeat):
        for path in paths:
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError(f"cannot read image: {path}")
            expected = expected_by_file.get(path.name, {})
            round_id = str(expected.get("round_id", "unlabelled-round"))
            game_id = str(expected.get("game_id", "unlabelled-game"))
            started = time.perf_counter()
            result = runtime.parser.parse(frame)
            timings.append((time.perf_counter() - started) * 1_000)
            scene_misses += int(not result.scene_found)
            if result.alignment_scale is not None:
                alignment_scales.append(result.alignment_scale)
            if result.alignment_rotation_degrees is not None:
                alignment_rotations.append(result.alignment_rotation_degrees)
            if result.alignment_offset is not None:
                alignment_offsets.append(
                    (result.alignment_offset[0] ** 2 + result.alignment_offset[1] ** 2) ** 0.5
                )
            unknown_total += len(result.unknown_slots)
            frame_ok = result.scene_found and not result.unknown_slots

            diagnostics: dict[tuple[str, int], dict[str, Any]] = {}
            for diagnostic in result.slot_diagnostics:
                key = (diagnostic["region"], diagnostic["index"])
                if key not in diagnostics or diagnostic.get("selected"):
                    diagnostics[key] = diagnostic
            tile_classifications += sum(
                len(values) for values in result.slot_states.values()
            )
            if result.unknown_slots:
                for diagnostic in result.slot_diagnostics:
                    if diagnostic.get("reason") is not None:
                        failures.append(_slot_failure(path, diagnostic, "labelled ground truth unavailable"))
            for region, labels in (expected.get("tiles") or {}).items():
                for index, label in enumerate(labels):
                    if not isinstance(label, str):
                        continue
                    tile_total += 1
                    diagnostic = diagnostics.get((region, index), {})
                    actual = diagnostic.get("label")
                    if actual == label:
                        tile_correct += 1
                    elif actual is None:
                        tile_unknown += 1
                        frame_ok = False
                        failures.append(_slot_failure(path, diagnostic, label))
                    else:
                        tile_errors += 1
                        frame_ok = False
                        failures.append(_slot_failure(path, diagnostic, label))

            for region, expected_meld in (expected.get("melds") or {}).items():
                meld_total += 1
                actual_meld = _benchmark_meld(result.melds.get(region))
                if actual_meld == expected_meld:
                    meld_correct += 1
                else:
                    frame_ok = False
                    if actual_meld is None:
                        meld_unknown += 1
                    failures.append(
                        {
                            "frame": path.name,
                            "kind": "meld_geometry",
                            "region": region,
                            "expected": expected_meld,
                            "actual": actual_meld,
                            "candidates": [
                                value
                                for value in result.meld_diagnostics
                                if value.get("region") == region
                            ],
                        }
                    )

            for region, expected_orientations in (
                expected.get("slot_orientations") or {}
            ).items():
                actual_states = result.slot_states.get(region, ())
                for index, expected_orientation in enumerate(expected_orientations):
                    if not isinstance(expected_orientation, str):
                        continue
                    orientation_total += 1
                    actual_orientation = (
                        actual_states[index].get("orientation")
                        if index < len(actual_states)
                        else None
                    )
                    if actual_orientation == expected_orientation:
                        orientation_correct += 1
                    else:
                        frame_ok = False
                        failures.append(
                            {
                                "frame": path.name,
                                "kind": "slot_orientation",
                                "region": region,
                                "slot": index,
                                "expected": expected_orientation,
                                "actual": actual_orientation,
                            }
                        )

            for name, expected_value in (expected.get("metadata") or {}).items():
                ocr_total += 1
                if result.metadata.get(name) == expected_value:
                    ocr_correct += 1
                else:
                    frame_ok = False
                    failures.append(
                        {
                            "frame": path.name,
                            "kind": "ocr",
                            "field": name,
                            "expected": expected_value,
                            "actual": result.metadata.get(name),
                            "confidence": result.metadata_confidences.get(name),
                        }
                    )
            for name, expected_value in (expected.get("markers") or {}).items():
                marker_total += 1
                if result.markers.get(name) is expected_value:
                    marker_correct += 1
                else:
                    frame_ok = False
                    failures.append(
                        {
                            "frame": path.name,
                            "kind": "marker",
                            "field": name,
                            "expected": expected_value,
                            "actual": result.markers.get(name),
                            "confidence": result.marker_confidences.get(name),
                        }
                    )

            stable = runtime.filter.push(result)
            expected_events = list(expected.get("events") or [])
            expected_event_total += len(expected_events)
            detected: list[Any] = []
            rejection: str | None = None
            if stable is not None and runtime.manager.round_id is not None:
                if not primed:
                    runtime.detector.prime(stable)
                    primed = True
                else:
                    try:
                        detected = runtime.detector.detect(stable, runtime.manager.state)
                    except VisionDeltaError as error:
                        desync_count += 1
                        rejection = str(error)
                        frame_ok = False
            actual_events = [
                {"name": event.name, "data": event.data} for event in detected
            ]
            correct_here = sum(
                actual == wanted
                for actual, wanted in zip(actual_events, expected_events)
            )
            event_correct += correct_here
            false_events += max(0, len(actual_events) - correct_here)
            missed_events += max(0, len(expected_events) - correct_here)
            if correct_here != len(actual_events) or correct_here != len(expected_events):
                frame_ok = False
                if actual_events or expected_events or rejection:
                    failures.append(
                        {
                            "frame": path.name,
                            "kind": "event",
                            "expected": expected_events,
                            "actual": actual_events,
                            "game_state_rejection": rejection,
                        }
                    )

            if detected:
                trial = copy.deepcopy(runtime.manager)
                try:
                    for event in detected:
                        trial.apply(_event_envelope(trial, runtime.current_round_id, event.name, event.data))
                except RealtimeStateError as error:
                    desync_count += 1
                    false_events += len(detected)
                    frame_ok = False
                    failures.append(
                        {
                            "frame": path.name,
                            "kind": "game_state_rejection",
                            "events": actual_events,
                            "game_state_rejection": f"{error.code}: {error}",
                        }
                    )

            for wanted in expected_events:
                try:
                    runtime.manager.apply(
                        _event_envelope(
                            runtime.manager,
                            runtime.current_round_id,
                            str(wanted["name"]),
                            dict(wanted.get("data") or {}),
                        )
                    )
                except RealtimeStateError as error:
                    frame_ok = False
                    failures.append(
                        {
                            "frame": path.name,
                            "kind": "invalid_benchmark_event",
                            "game_state_rejection": f"{error.code}: {error}",
                        }
                    )

            if "resync_expected" in expected and stable is not None:
                can_restore = runtime._build_midround_restore_plan(stable).state is not None
                if can_restore == bool(expected["resync_expected"]):
                    resync_success += int(can_restore)
                else:
                    resync_failure += 1
                    frame_ok = False
            if not frame_ok:
                round_ok[round_id] = False
                game_ok[game_id] = False
                _save_failure_artifacts(debug_root, path, frame, result, failures, runtime)
            else:
                round_ok[round_id] = round_ok[round_id]
                game_ok[game_id] = game_ok[game_id]

    ordered = sorted(timings)
    p95 = ordered[min(len(ordered) - 1, round((len(ordered) - 1) * 0.95))]
    report = {
        "format": FORMAT,
        "frames": len(timings),
        "unique_frames": len(paths),
        "mean_ms": statistics.fmean(timings),
        "p95_ms": p95,
        "effective_fps": 1_000 / statistics.fmean(timings),
        "scene_misses": scene_misses,
        "alignment_scale_range": (
            [min(alignment_scales), max(alignment_scales)] if alignment_scales else None
        ),
        "alignment_rotation_range_degrees": (
            [min(alignment_rotations), max(alignment_rotations)]
            if alignment_rotations
            else None
        ),
        "alignment_max_offset_pixels": max(alignment_offsets) if alignment_offsets else None,
        "tile_slot_count": tile_classifications,
        "tile_labelled_count": tile_total,
        "tile_correct_count": tile_correct,
        "tile_error_count": tile_errors,
        "unknown_count": unknown_total,
        "tile_accuracy": _rate(tile_correct, tile_total),
        "meld_layout_count": meld_total,
        "meld_layout_correct_count": meld_correct,
        "meld_layout_unknown_count": meld_unknown,
        "meld_layout_accuracy": _rate(meld_correct, meld_total),
        "slot_orientation_count": orientation_total,
        "slot_orientation_correct_count": orientation_correct,
        "slot_orientation_accuracy": _rate(orientation_correct, orientation_total),
        "ocr_field_count": ocr_total,
        "ocr_accuracy": _rate(ocr_correct, ocr_total),
        "marker_count": marker_total,
        "ui_marker_accuracy": _rate(marker_correct, marker_total),
        "event_count": expected_event_total,
        "event_correct_count": event_correct,
        "event_accuracy": _rate(event_correct, expected_event_total),
        "false_event_count": false_events,
        "missed_event_count": missed_events,
        "false_event_rate": _rate(false_events, event_correct + false_events),
        "desync_count": desync_count,
        "resync_success_count": resync_success,
        "resync_failure_count": resync_failure,
        "round_count": len(round_ok) if annotations else 0,
        "round_full_sync_rate": _rate(sum(round_ok.values()), len(round_ok)) if annotations else None,
        "game_count": len(game_ok) if annotations else 0,
        "game_full_sync_rate": _rate(sum(game_ok.values()), len(game_ok)) if annotations else None,
        "failure_count": len(failures),
        "failures": failures,
        "profile": runtime.profile_report.to_dict(),
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    if not args.quiet:
        print(encoded)
    if args.output:
        destination = Path(args.output).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        temporary.write_text(encoded + "\n", encoding="utf-8")
        temporary.replace(destination)
    complete = (
        scene_misses == 0
        and unknown_total == 0
        and tile_errors == 0
        and tile_unknown == 0
        and meld_unknown == 0
        and meld_correct == meld_total
        and orientation_correct == orientation_total
        and false_events == 0
        and missed_events == 0
        and desync_count == 0
    )
    raise SystemExit(0 if complete else 2)


def _load_annotations(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("format") != FORMAT or not isinstance(raw.get("frames"), list):
        raise ValueError(f"annotations must use {FORMAT}")
    return raw


def _initialize_manager(runtime: VisionAdapterRuntime, annotations: dict[str, Any]) -> None:
    events = annotations.get("initial_events") or []
    if not events:
        return
    manager = RealtimeStateManager(runtime.config.observer)
    for event in events:
        manager.apply(event)
    runtime.manager = manager
    runtime.current_round_id = str(events[-1]["round_id"])


def _event_envelope(
    manager: RealtimeStateManager, round_id: str, name: str, data: dict[str, Any]
) -> dict[str, Any]:
    sequence = manager.next_source_sequence
    return {
        "event_id": f"benchmark-{sequence}-{name}",
        "sequence": sequence,
        "round_id": round_id,
        "name": name,
        "data": data,
    }


def _rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _benchmark_meld(value: dict[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return {
        "kind": value.get("kind"),
        "tiles": [
            f"{tile['kind']}{'_red' if tile.get('red') else ''}"
            for tile in value.get("tiles", ())
        ],
        "called_index": value.get("called_index"),
        "from": value.get("from"),
    }


def _slot_failure(path: Path, diagnostic: dict[str, Any], expected: str) -> dict[str, Any]:
    return {
        "frame": path.name,
        "kind": "tile",
        "region": diagnostic.get("region"),
        "slot": diagnostic.get("index"),
        "expected": expected,
        "actual": diagnostic.get("label"),
        "candidate_1": (diagnostic.get("candidates") or [None])[0],
        "candidate_2": (diagnostic.get("candidates") or [None, None])[1]
        if len(diagnostic.get("candidates") or []) > 1
        else None,
        "template_difference": diagnostic.get("template_difference"),
        "reason": diagnostic.get("reason"),
    }


def _save_failure_artifacts(
    root: Path | None,
    path: Path,
    frame: Any,
    result: Any,
    failures: list[dict[str, Any]],
    runtime: VisionAdapterRuntime,
) -> None:
    if root is None:
        return
    destination = root / path.stem
    destination.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination / "frame.png"), frame)
    aligned = frame
    if result.scene_found and runtime.parser.scene_aligner is not None:
        # Re-run the read-only alignment so saved ROIs match reference layout.
        # This path is benchmark/debug-only and never runs during normal play.
        alignment = runtime.parser.scene_aligner.align(frame)
        if alignment.found:
            aligned = alignment.frame
    for diagnostic in result.slot_diagnostics:
        if diagnostic.get("reason") is None and diagnostic.get("selected") is not False:
            continue
        x, y, width, height = diagnostic["rectangle"]
        crop = aligned[y : y + height, x : x + width]
        cv2.imwrite(
            str(destination / f"{diagnostic['region']}-{diagnostic['index']}.png"),
            crop,
        )
    for diagnostic in result.meld_diagnostics:
        if diagnostic.get("mode") == "right-to-left-scan":
            x, y, width, height = diagnostic["scan_rectangle"]
            crop = aligned[y : y + height, x : x + width]
            cv2.imwrite(
                str(destination / f"{diagnostic['region']}-scan.png"),
                crop,
            )
            continue
        if diagnostic.get("rejection") is None:
            continue
        for index, slot in enumerate(diagnostic.get("slots") or []):
            x, y, width, height = slot["rectangle"]
            crop = aligned[y : y + height, x : x + width]
            cv2.imwrite(
                str(
                    destination
                    / f"{diagnostic['region']}-{diagnostic['pattern']}-{index}.png"
                ),
                crop,
            )
    relevant = [failure for failure in failures if failure.get("frame") == path.name]
    (destination / "diagnostics.json").write_text(
        json.dumps(
            {
                "failures": relevant,
                "slot_diagnostics": list(result.slot_diagnostics),
                "meld_diagnostics": list(result.meld_diagnostics),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
