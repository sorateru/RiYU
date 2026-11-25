"""Detect the configured game scene and normalize small window/zoom changes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from .parser import LayoutConfig, SceneAnchorConfig


@dataclass(frozen=True)
class SceneAlignment:
    found: bool
    frame: np.ndarray
    confidence: float | None = None
    offset: tuple[float, float] | None = None
    scale: float | None = None
    rotation_degrees: float | None = None
    matrix: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None
    reason: str | None = None


@dataclass(frozen=True)
class _AnchorMatch:
    name: str
    confidence: float
    template_scale: float
    reference_center: tuple[float, float]
    observed_center: tuple[float, float]


class SceneAligner:
    """Match user-created anchor crops and warp the frame to layout coordinates.

    Anchor images live below ``templates/anchors`` and use
    ``<anchor-name>__<variant>.png``. They are deliberately kept in the local
    profile: no service-owned artwork needs to be distributed with RiYU.
    """

    def __init__(self, layout: "LayoutConfig", template_directory: str | Path) -> None:
        self.layout = layout
        root = Path(template_directory) / "anchors"
        self.templates: dict[str, tuple[np.ndarray, ...]] = {}
        for name in layout.scene_anchors:
            values: list[np.ndarray] = []
            for path in sorted(root.glob(f"{name}*.png")):
                if path.stem.split("__", 1)[0] != name:
                    continue
                image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if image is not None and min(image.shape[:2]) >= 4:
                    values.append(image)
            self.templates[name] = tuple(values)

    def align(self, frame: np.ndarray) -> SceneAlignment:
        if not self.layout.scene_anchors:
            return SceneAlignment(True, frame, scale=1.0, offset=(0.0, 0.0))
        matches: list[_AnchorMatch] = []
        failures: list[str] = []
        for name, config in self.layout.scene_anchors.items():
            templates = self.templates.get(name, ())
            if not templates:
                if config.required:
                    failures.append(f"{name}:anchor_template_missing")
                continue
            match = self._match(name, frame, config, templates)
            if match is None:
                if config.required:
                    failures.append(f"{name}:anchor_not_found")
                continue
            matches.append(match)
        if failures or not matches:
            return SceneAlignment(
                False,
                frame,
                confidence=0.0,
                reason=failures[0] if failures else "no_scene_anchor_matched",
            )

        confidence = min(value.confidence for value in matches)
        transform = self._transform(matches)
        if transform is None:
            return SceneAlignment(
                False,
                frame,
                confidence=confidence,
                reason="scene_anchors_geometrically_inconsistent",
            )
        linear = transform[:, :2]
        scale = float(np.sqrt(abs(np.linalg.det(linear))))
        offset_x = float(transform[0, 2])
        offset_y = float(transform[1, 2])
        rotation = float(np.degrees(np.arctan2(transform[1, 0], transform[0, 0])))
        matrix = tuple(tuple(float(item) for item in row) for row in transform)
        if not any(config.align for config in self.layout.scene_anchors.values()):
            return SceneAlignment(
                True,
                frame,
                confidence=confidence,
                offset=(offset_x, offset_y),
                scale=scale,
                rotation_degrees=rotation,
                matrix=matrix,  # type: ignore[arg-type]
            )
        # Observed coordinates are ``scale * reference + offset``. Warp the
        # observed image back into the immutable reference-layout coordinate
        # system used by all tile and annotation rectangles.
        inverse = cv2.invertAffineTransform(transform)
        aligned = cv2.warpAffine(
            frame,
            inverse,
            (self.layout.image_width, self.layout.image_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        return SceneAlignment(
            True,
            aligned,
            confidence=confidence,
            offset=(offset_x, offset_y),
            scale=scale,
            rotation_degrees=rotation,
            matrix=matrix,  # type: ignore[arg-type]
        )

    @staticmethod
    def _transform(matches: list[_AnchorMatch]) -> np.ndarray | None:
        """Estimate reference-to-observed similarity transform conservatively."""

        if len(matches) == 1:
            match = matches[0]
            scale = match.template_scale
            return np.asarray(
                [
                    [scale, 0.0, match.observed_center[0] - scale * match.reference_center[0]],
                    [0.0, scale, match.observed_center[1] - scale * match.reference_center[1]],
                ],
                dtype=np.float32,
            )
        reference = np.asarray([value.reference_center for value in matches], dtype=np.float32)
        observed = np.asarray([value.observed_center for value in matches], dtype=np.float32)
        method = cv2.RANSAC if len(matches) >= 3 else cv2.LMEDS
        transform, inliers = cv2.estimateAffinePartial2D(
            reference,
            observed,
            method=method,
            ransacReprojThreshold=4.0,
            maxIters=2_000,
            confidence=0.995,
            refineIters=10,
        )
        if transform is None:
            return None
        projected = cv2.transform(reference.reshape(-1, 1, 2), transform).reshape(-1, 2)
        errors = np.linalg.norm(projected - observed, axis=1)
        # A wrong anchor is more dangerous than losing alignment. All required
        # anchors have already been required to match, so reject a transform
        # which cannot explain every accepted anchor within a small UI margin.
        if float(np.max(errors)) > 6.0:
            return None
        if inliers is not None and not bool(np.all(inliers)):
            return None
        return transform.astype(np.float32)

    @staticmethod
    def _match(
        name: str,
        frame: np.ndarray,
        config: "SceneAnchorConfig",
        templates: tuple[np.ndarray, ...],
    ) -> _AnchorMatch | None:
        x, y, width, height = config.search_rectangle or (
            0,
            0,
            frame.shape[1],
            frame.shape[0],
        )
        search = cv2.cvtColor(frame[y : y + height, x : x + width], cv2.COLOR_BGR2GRAY)
        reference_x, reference_y, reference_width, reference_height = config.rectangle
        best: tuple[float, float, float, float] | None = None
        scale = config.minimum_scale
        while scale <= config.maximum_scale + 1e-6:
            for template in templates:
                resized = cv2.resize(
                    template,
                    None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC,
                )
                if resized.shape[0] > search.shape[0] or resized.shape[1] > search.shape[1]:
                    continue
                values = cv2.matchTemplate(search, resized, cv2.TM_CCOEFF_NORMED)
                _minimum, score, _minimum_location, location = cv2.minMaxLoc(values)
                if best is None or score > best[0]:
                    observed_x = float(x + location[0])
                    observed_y = float(y + location[1])
                    best = (
                        float(score),
                        scale,
                        observed_x + resized.shape[1] / 2.0,
                        observed_y + resized.shape[0] / 2.0,
                    )
            scale += config.scale_step
        if best is None or best[0] < config.minimum_confidence:
            return None
        return _AnchorMatch(
            name,
            best[0],
            best[1],
            (reference_x + reference_width / 2.0, reference_y + reference_height / 2.0),
            (best[2], best[3]),
        )
