"""Fast marker templates and one-process OCR for public screen annotations."""

from __future__ import annotations

import csv
import io
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .parser import LayoutConfig, TextFieldConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnnotationResult:
    metadata: dict[str, int | str]
    metadata_confidences: dict[str, float]
    markers: dict[str, bool | None]
    marker_confidences: dict[str, float]
    required_unknown: tuple[str, ...]


@dataclass(frozen=True)
class MarkerClassification:
    value: bool | None
    confidence: float
    on_confidence: float
    off_confidence: float
    reason: str | None = None


class MarkerTemplateBank:
    """Classify configured boolean UI signals from user-provided on/off crops."""

    def __init__(
        self,
        template_directory: str | Path,
        marker_names: set[str],
        *,
        minimum_confidence: float = 0.72,
        minimum_margin: float = 0.05,
        absence_confidence: float = 0.38,
    ) -> None:
        if not 0 <= absence_confidence < minimum_confidence <= 1:
            raise ValueError("marker confidence thresholds are invalid")
        if not 0 <= minimum_margin <= 1:
            raise ValueError("marker minimum_margin must be in 0..=1")
        self.minimum_confidence = minimum_confidence
        self.minimum_margin = minimum_margin
        self.absence_confidence = absence_confidence
        root = Path(template_directory) / "markers"
        self.templates: dict[str, tuple[list[np.ndarray], list[np.ndarray]]] = {}
        for name in marker_names:
            on = self._load(root, f"{name}__on*.png")
            off = self._load(root, f"{name}__off*.png")
            self.templates[name] = (on, off)

    def classify(self, name: str, crop: np.ndarray) -> MarkerClassification:
        on_templates, off_templates = self.templates[name]
        if not on_templates:
            return MarkerClassification(
                None, 0.0, 0.0, 0.0, "marker_template_missing"
            )
        query = self._prepare(crop)
        on_score = max(self._similarity(query, value) for value in on_templates)
        off_score = (
            max(self._similarity(query, value) for value in off_templates)
            if off_templates
            else 0.0
        )
        best = max(on_score, off_score)
        if off_templates:
            if best < self.minimum_confidence:
                return MarkerClassification(
                    None, best, on_score, off_score, "confidence_below_threshold"
                )
            if abs(on_score - off_score) < self.minimum_margin:
                return MarkerClassification(
                    None, best, on_score, off_score, "ambiguous_templates"
                )
            return MarkerClassification(
                on_score > off_score,
                best,
                on_score,
                off_score,
            )
        if on_score >= self.minimum_confidence:
            return MarkerClassification(True, on_score, on_score, 0.0)
        if on_score <= self.absence_confidence:
            return MarkerClassification(False, 1.0 - on_score, on_score, 0.0)
        return MarkerClassification(
            None,
            max(on_score, 1.0 - on_score),
            on_score,
            0.0,
            "marker_transition_uncertain",
        )

    @classmethod
    def _load(cls, root: Path, pattern: str) -> list[np.ndarray]:
        output: list[np.ndarray] = []
        for path in sorted(root.glob(pattern)):
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is not None:
                output.append(cls._prepare(image))
        return output

    @staticmethod
    def _prepare(image: np.ndarray) -> np.ndarray:
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
        return cv2.equalizeHist(resized).astype(np.float32) / 255.0

    @staticmethod
    def _similarity(left: np.ndarray, right: np.ndarray) -> float:
        left_centered = left - float(np.mean(left))
        right_centered = right - float(np.mean(right))
        denominator = float(
            np.linalg.norm(left_centered) * np.linalg.norm(right_centered)
        )
        if denominator <= 1e-12:
            return 1.0 if np.array_equal(left, right) else 0.0
        correlation = float(np.sum(left_centered * right_centered)) / denominator
        return min(1.0, max(0.0, (correlation + 1.0) / 2.0))


class TesseractFieldRecognizer:
    """Read all configured fields in a single Tesseract process.

    Crops are normalized and stacked into separate rows, avoiding one process
    launch per score/round field and preventing unrelated screen text from
    entering the result.
    """

    def __init__(
        self,
        fields: dict[str, TextFieldConfig],
        *,
        executable: str | Path | None = None,
        minimum_confidence: float = 0.55,
    ) -> None:
        self.fields = fields
        self.minimum_confidence = minimum_confidence
        found = str(executable) if executable is not None else shutil.which("tesseract")
        if fields and not found:
            raise ValueError(
                "layout has text_fields but tesseract is unavailable; install tesseract or remove text_fields"
            )
        self.executable = found

    def recognize(
        self, frame: np.ndarray
    ) -> tuple[dict[str, int | str], dict[str, float], tuple[str, ...]]:
        if not self.fields:
            return {}, {}, ()
        composite, rows = self._composite(frame)
        values, confidences, unknown = self._recognize_composite(
            composite,
            rows,
            self.fields,
            page_segmentation_mode=6,
        )
        score_names = [f"score_{player}" for player in range(4)]
        if all(name in values for name in score_names):
            least_confident = min(score_names, key=lambda name: confidences[name])
            if confidences[least_confident] < 0.85:
                # Vertical seat text can be read differently when Tesseract
                # considers the surrounding rows as one paragraph.  Re-read
                # only the uncertain score as a single line; this keeps the
                # common path at one OCR process and prevents a low-confidence
                # digit substitution from entering round metadata.
                fallback = TesseractFieldRecognizer(
                    {least_confident: self.fields[least_confident]},
                    executable=self.executable,
                    minimum_confidence=self.minimum_confidence,
                )
                fallback_composite, fallback_rows = fallback._composite(frame)
                minimum_y, maximum_y = fallback_rows[least_confident]
                line = fallback_composite[minimum_y:maximum_y]
                line_width = max(320, line.shape[1])
                single_line = np.full(
                    (line.shape[0], line_width), 255, dtype=np.uint8
                )
                single_line[:, : line.shape[1]] = line
                fallback_values, fallback_confidences, _ = (
                    fallback._recognize_composite(
                        single_line,
                        {least_confident: (0, line.shape[0])},
                        fallback.fields,
                        page_segmentation_mode=7,
                    )
                )
                if least_confident in fallback_values:
                    values[least_confident] = fallback_values[least_confident]
                    confidences[least_confident] = fallback_confidences[least_confident]
        return values, confidences, unknown

    def _recognize_composite(
        self,
        composite: np.ndarray,
        rows: dict[str, tuple[int, int]],
        fields: dict[str, TextFieldConfig],
        *,
        page_segmentation_mode: int,
    ) -> tuple[dict[str, int | str], dict[str, float], tuple[str, ...]]:
        success, encoded = cv2.imencode(".png", composite)
        if not success:
            raise RuntimeError("could not encode OCR input")
        process = subprocess.run(
            [
                str(self.executable),
                "stdin",
                "stdout",
                "--psm",
                str(page_segmentation_mode),
                "-l",
                "eng",
                "-c",
                "tessedit_char_whitelist=0123456789-+ESWN",
                "tsv",
            ],
            input=encoded.tobytes(),
            capture_output=True,
            timeout=5,
            check=False,
        )
        if process.returncode != 0:
            detail = process.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"tesseract failed: {detail or process.returncode}")
        tokens: dict[str, list[tuple[int, str, float]]] = {
            name: [] for name in fields
        }
        reader = csv.DictReader(
            io.StringIO(process.stdout.decode("utf-8", errors="replace")),
            delimiter="\t",
        )
        for token in reader:
            text = (token.get("text") or "").strip()
            try:
                confidence = float(token.get("conf") or -1) / 100.0
                left = int(token.get("left") or 0)
                top = int(token.get("top") or 0)
                width = int(token.get("width") or 0)
                height = int(token.get("height") or 0)
            except ValueError:
                continue
            if not text or confidence < 0:
                continue
            center_y = top + height // 2
            for name, (minimum_y, maximum_y) in rows.items():
                if minimum_y <= center_y < maximum_y:
                    tokens[name].append((left + width // 2, text, confidence))
                    break
        values: dict[str, int | str] = {}
        confidences: dict[str, float] = {}
        unknown: list[str] = []
        for name, config in fields.items():
            pieces = sorted(tokens[name])
            raw = "".join(piece[1] for piece in pieces)
            confidence = min((piece[2] for piece in pieces), default=0.0)
            value = self._parse_value(raw, config)
            if value is None or confidence < self.minimum_confidence:
                if config.required:
                    unknown.append(f"text_fields.{name}:unreadable")
                continue
            values[name] = value
            confidences[name] = confidence
        return values, confidences, tuple(unknown)

    def _composite(
        self, frame: np.ndarray
    ) -> tuple[np.ndarray, dict[str, tuple[int, int]]]:
        scale = 3
        gap = 28
        prepared: list[tuple[str, np.ndarray]] = []
        for name, config in self.fields.items():
            x, y, width, height = config.rectangle
            crop = frame[y : y + height, x : x + width]
            crop = _rotate(crop, config.rotation)
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
            resized = cv2.resize(
                gray,
                (gray.shape[1] * scale, gray.shape[0] * scale),
                interpolation=cv2.INTER_CUBIC,
            )
            _, binary = cv2.threshold(
                resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            border = np.concatenate(
                (binary[0], binary[-1], binary[:, 0], binary[:, -1])
            )
            if float(np.median(border)) < 127:
                binary = cv2.bitwise_not(binary)
            binary = _remove_border_components(binary)
            prepared.append((name, binary))
        width = max(value.shape[1] for _, value in prepared) + 40
        height = sum(value.shape[0] + gap for _, value in prepared) + gap
        composite = np.full((height, width), 255, dtype=np.uint8)
        rows: dict[str, tuple[int, int]] = {}
        cursor = gap
        for name, value in prepared:
            composite[cursor : cursor + value.shape[0], 20 : 20 + value.shape[1]] = value
            rows[name] = (cursor, cursor + value.shape[0])
            cursor += value.shape[0] + gap
        return composite, rows

    @staticmethod
    def _parse_value(raw: str, config: TextFieldConfig) -> int | str | None:
        normalized = raw.translate(str.maketrans("０１２３４５６７８９", "0123456789"))
        if config.value_type == "text":
            return normalized.strip() or None
        match = re.search(r"[-+]?\d[\d,]*", normalized)
        if match is None:
            return None
        value = int(match.group(0).replace(",", ""))
        if config.minimum is not None and value < config.minimum:
            return None
        if config.maximum is not None and value > config.maximum:
            return None
        return value


class ScreenAnnotationParser:
    def __init__(
        self,
        layout: LayoutConfig,
        template_directory: str | Path,
        *,
        tesseract_executable: str | Path | None = None,
        minimum_confidence: float = 0.72,
        minimum_margin: float = 0.05,
        ocr_minimum_confidence: float = 0.55,
    ) -> None:
        self.layout = layout
        self.markers = (
            MarkerTemplateBank(
                template_directory,
                set(layout.markers),
                minimum_confidence=minimum_confidence,
                minimum_margin=minimum_margin,
            )
            if layout.markers
            else None
        )
        self.ocr = TesseractFieldRecognizer(
            layout.text_fields,
            executable=tesseract_executable,
            minimum_confidence=ocr_minimum_confidence,
        )

    def parse(self, frame: np.ndarray) -> AnnotationResult:
        try:
            metadata, metadata_confidences, required_unknown = self.ocr.recognize(frame)
        except (OSError, RuntimeError, subprocess.SubprocessError) as error:
            logger.warning("screen OCR failed for this frame: %s", error)
            metadata = {}
            metadata_confidences = {}
            required_unknown = tuple(
                f"text_fields.{name}:ocr_error"
                for name, config in self.layout.text_fields.items()
                if config.required
            )
        marker_values: dict[str, bool | None] = {}
        marker_confidences: dict[str, float] = {}
        unknown = list(required_unknown)
        if self.markers is not None:
            for name, config in self.layout.markers.items():
                x, y, width, height = config.rectangle
                result = self.markers.classify(
                    name, frame[y : y + height, x : x + width]
                )
                marker_values[name] = result.value
                marker_confidences[name] = result.confidence
                if config.required and result.value is None:
                    unknown.append(f"markers.{name}:{result.reason}")
        return AnnotationResult(
            metadata,
            metadata_confidences,
            marker_values,
            marker_confidences,
            tuple(unknown),
        )


def _rotate(image: np.ndarray, degrees: int) -> np.ndarray:
    if degrees == 0:
        return image
    if degrees == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if degrees == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)


def _remove_border_components(binary: np.ndarray) -> np.ndarray:
    """Remove table-frame ornaments that touch an OCR crop boundary.

    Mahjong Soul draws the score digits inside a decorative frame.  Those
    border fragments are frequently interpreted as an extra ``2`` or minus
    sign.  OCR crops intentionally leave padding around their glyphs, so a
    connected dark component touching the crop edge is UI chrome rather than
    text and can be discarded deterministically.
    """

    foreground = (binary < 128).astype(np.uint8)
    component_count, labels = cv2.connectedComponents(foreground, connectivity=8)
    output = binary.copy()
    border_labels = set(int(value) for value in labels[0])
    border_labels.update(int(value) for value in labels[-1])
    border_labels.update(int(value) for value in labels[:, 0])
    border_labels.update(int(value) for value in labels[:, -1])
    border_labels.discard(0)
    for label in border_labels:
        output[labels == label] = 255
    return output
