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
        if frame.shape[1] != self.region.width or frame.shape[0] != self.region.height:
            raise RuntimeError(
                f"captured {frame.shape[1]}x{frame.shape[0]}, expected "
                f"{self.region.width}x{self.region.height}"
            )
        return frame


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
