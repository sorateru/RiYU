"""Directory-fed lab for saved frames and local simulator output only."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator

from .parser import PerceptionResult, ScreenshotParser


class OfflineFrameLab:
    def __init__(self, parser: ScreenshotParser, input_directory: str | Path) -> None:
        self.parser = parser
        self.input_directory = Path(input_directory).resolve()
        if not self.input_directory.is_dir():
            raise ValueError("offline frame input must be an existing directory")
        self._processed: set[Path] = set()

    def scan(self, base_state: dict | None = None) -> list[PerceptionResult]:
        results: list[PerceptionResult] = []
        for path in sorted(self.input_directory.glob("*.png")):
            resolved = path.resolve()
            if resolved in self._processed:
                continue
            self._processed.add(resolved)
            results.append(self.parser.parse(resolved, base_state))
        return results

    def watch(
        self,
        base_state: dict | None = None,
        poll_interval: float = 0.25,
    ) -> Iterator[PerceptionResult]:
        if poll_interval < 0.05:
            raise ValueError("poll_interval must be at least 50 ms")
        while True:
            yield from self.scan(base_state)
            time.sleep(poll_interval)
