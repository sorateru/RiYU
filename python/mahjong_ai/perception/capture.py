"""Read-only frame sources for the Vision Adapter."""

from __future__ import annotations

import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol

import cv2
import numpy as np


class FrameSource(Protocol):
    def capture(self) -> np.ndarray: ...


@dataclass(frozen=True)
class CaptureRegion:
    left: int
    top: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.left < 0 or self.top < 0 or self.width <= 0 or self.height <= 0:
            raise ValueError("capture region must have non-negative origin and positive size")

    @property
    def macos_argument(self) -> str:
        return f"{self.left},{self.top},{self.width},{self.height}"


class MacOSScreenCapture:
    """Capture a configured screen rectangle through macOS screencapture.

    macOS prompts the user for Screen Recording permission. The adapter cannot
    grant or bypass that permission and never sends mouse/keyboard input.
    """

    def __init__(
        self,
        region: CaptureRegion,
        executable: str | Path = "/usr/sbin/screencapture",
    ) -> None:
        self.region = region
        self.executable = Path(executable)
        if not self.executable.is_file():
            raise ValueError(f"screencapture executable does not exist: {self.executable}")

    def capture(self) -> np.ndarray:
        with tempfile.NamedTemporaryFile(suffix=".png") as handle:
            process = subprocess.run(
                [
                    str(self.executable),
                    "-x",
                    "-R",
                    self.region.macos_argument,
                    handle.name,
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if process.returncode != 0:
                detail = process.stderr.strip() or "screen capture was denied"
                raise RuntimeError(
                    f"macOS screen capture failed: {detail}. Enable Screen Recording permission for RiYU."
                )
            frame = cv2.imread(handle.name, cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(
                "macOS returned no readable frame; verify Screen Recording permission and capture bounds"
            )
        scale_x = frame.shape[1] / self.region.width
        scale_y = frame.shape[0] / self.region.height
        if abs(scale_x - scale_y) > 0.01 or not 1.0 <= scale_x <= 4.0:
            raise RuntimeError(
                f"captured {frame.shape[1]}x{frame.shape[0]} for logical region "
                f"{self.region.width}x{self.region.height}; display scaling is inconsistent"
            )
        if frame.shape[1] != self.region.width or frame.shape[0] != self.region.height:
            frame = cv2.resize(
                frame,
                (self.region.width, self.region.height),
                interpolation=cv2.INTER_AREA,
            )
        return frame


class MacOSWindowCapture:
    """Capture one explicitly authorized macOS window by numeric window id.

    ``content_region`` is measured in the PNG returned by ``screencapture``.
    Keeping the id and crop explicit avoids accessibility automation and makes
    the operation auditable. The selected window may be behind the RiYU UI,
    while switching the selected Safari window away from the game is still
    rejected by the scene guard.
    """

    def __init__(
        self,
        window_id: int,
        content_region: CaptureRegion,
        output_size: tuple[int, int],
        executable: str | Path = "/usr/sbin/screencapture",
    ) -> None:
        if window_id <= 0:
            raise ValueError("window_id must be positive")
        if min(output_size) <= 0:
            raise ValueError("window capture output size must be positive")
        self.window_id = window_id
        self.content_region = content_region
        self.output_size = output_size
        self.executable = Path(executable)
        if not self.executable.is_file():
            raise ValueError(f"screencapture executable does not exist: {self.executable}")

    def capture(self) -> np.ndarray:
        with tempfile.NamedTemporaryFile(suffix=".png") as handle:
            process = subprocess.run(
                [
                    str(self.executable),
                    "-x",
                    "-o",
                    f"-l{self.window_id}",
                    handle.name,
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if process.returncode != 0:
                detail = process.stderr.strip() or "window capture was denied or the id expired"
                raise RuntimeError(f"macOS window capture failed: {detail}")
            frame = cv2.imread(handle.name, cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError("macOS returned no readable window frame")
        region = self.content_region
        if region.left + region.width > frame.shape[1] or region.top + region.height > frame.shape[0]:
            raise RuntimeError(
                f"window content crop {region.macos_argument} is outside captured "
                f"{frame.shape[1]}x{frame.shape[0]}; recalibrate the Safari window profile"
            )
        content = frame[
            region.top : region.top + region.height,
            region.left : region.left + region.width,
        ]
        if (content.shape[1], content.shape[0]) != self.output_size:
            content = cv2.resize(content, self.output_size, interpolation=cv2.INTER_AREA)
        return content


def front_safari_window_id(
    executable: str | Path = "/usr/bin/osascript",
) -> int:
    """Return Safari's front window id without reading page content or sending input."""

    executable = Path(executable)
    if not executable.is_file():
        raise ValueError(f"osascript executable does not exist: {executable}")
    process = subprocess.run(
        [
            str(executable),
            "-e",
            'tell application "Safari" to get id of front window',
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    if process.returncode != 0:
        detail = process.stderr.strip() or "Safari has no front window"
        raise RuntimeError(f"could not resolve Safari window id: {detail}")
    try:
        value = int(process.stdout.strip())
    except ValueError as error:
        raise RuntimeError("Safari returned an invalid window id") from error
    if value <= 0:
        raise RuntimeError("Safari returned an invalid window id")
    return value


class DirectoryFrameSource:
    """Deterministic source used for saved-frame replay and tests."""

    def __init__(self, directory: str | Path) -> None:
        self.paths = tuple(sorted(Path(directory).glob("*.png")))
        if not self.paths:
            raise ValueError("frame directory contains no PNG files")
        self._index = 0

    def capture(self) -> np.ndarray:
        path = self.paths[min(self._index, len(self.paths) - 1)]
        self._index += 1
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"cannot read frame: {path}")
        return frame


def frame_stream(source: FrameSource, frames_per_second: float) -> Iterator[np.ndarray]:
    if not 0.2 <= frames_per_second <= 10.0:
        raise ValueError("frames_per_second must be between 0.2 and 10")
    interval = 1.0 / frames_per_second
    while True:
        started = time.monotonic()
        yield source.capture()
        remaining = interval - (time.monotonic() - started)
        if remaining > 0:
            time.sleep(remaining)
