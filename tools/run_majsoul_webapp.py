#!/usr/bin/env python3
"""Start the local RiYU web app and Mahjong Soul Vision Adapter together."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path
from urllib.parse import urlparse

from mahjong_ai.perception import VisionAdapterConfig


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Vision Adapter JSON config")
    parser.add_argument("--checkpoint", help="validated analysis checkpoint")
    parser.add_argument("--belief-checkpoint", help="validated belief checkpoint")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-open", action="store_true", help="do not open the web app")
    parser.add_argument(
        "--log-level",
        choices=("debug", "info", "warning", "error"),
        default="info",
    )
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = VisionAdapterConfig.load(config_path)
    parsed = urlparse(config.websocket_url)
    port = parsed.port or (443 if parsed.scheme == "wss" else 80)
    if parsed.scheme != "ws" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        parser.error("one-command launcher requires a local ws:// Vision Adapter URL")
    if parsed.path != "/ws/realtime/events/source":
        parser.error("config websocket_url must use /ws/realtime/events/source")
    ui_index = ROOT / "apps" / "reviewer" / "dist" / "index.html"
    if not ui_index.is_file():
        parser.error(
            "web app is not built; run `cd apps/reviewer && npm run build` first"
        )

    environment = os.environ.copy()
    python_path = str(ROOT / "python")
    current_python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        python_path
        if not current_python_path
        else os.pathsep.join((python_path, current_python_path))
    )
    server_command = [
        sys.executable,
        str(ROOT / "tools" / "serve_analysis.py"),
        "--port",
        str(port),
        "--device",
        args.device,
        "--log-level",
        args.log_level,
    ]
    if args.checkpoint:
        server_command.extend(("--checkpoint", str(Path(args.checkpoint).resolve())))
    if args.belief_checkpoint:
        server_command.extend(
            ("--belief-checkpoint", str(Path(args.belief_checkpoint).resolve()))
        )

    server = subprocess.Popen(server_command, cwd=ROOT, env=environment)
    adapter: subprocess.Popen[bytes] | None = None
    stopping = False

    def stop(_signum: int | None = None, _frame: object | None = None) -> None:
        nonlocal stopping
        if stopping:
            return
        stopping = True
        if adapter is not None and adapter.poll() is None:
            adapter.terminate()
        if server.poll() is None:
            server.terminate()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        _wait_for_server(port, server)
        web_url = f"http://127.0.0.1:{port}/"
        if not args.no_open:
            webbrowser.open(web_url)
        print(f"RiYU web app: {web_url}", flush=True)
        print("Stop both processes with Ctrl+C.", flush=True)
        adapter = subprocess.Popen(
            [
                sys.executable,
                str(ROOT / "tools" / "run_vision_adapter.py"),
                str(config_path),
                "--log-level",
                args.log_level.upper(),
            ],
            cwd=ROOT,
            env=environment,
        )
        while True:
            if stopping:
                return
            adapter_status = adapter.poll()
            server_status = server.poll()
            if adapter_status is not None:
                if adapter_status != 0:
                    raise SystemExit(f"Vision Adapter exited with status {adapter_status}")
                return
            if server_status is not None:
                raise SystemExit(f"analysis server exited with status {server_status}")
            time.sleep(0.25)
    finally:
        stop()
        _wait_for_exit(adapter)
        _wait_for_exit(server)


def _wait_for_server(port: int, process: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + 30.0
    health_url = f"http://127.0.0.1:{port}/health"
    while time.monotonic() < deadline:
        status = process.poll()
        if status is not None:
            raise SystemExit(f"analysis server exited with status {status}")
        try:
            with urllib.request.urlopen(health_url, timeout=0.5) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(0.1)
    raise SystemExit(f"analysis server did not become ready at {health_url}")


def _wait_for_exit(process: subprocess.Popen[bytes] | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


if __name__ == "__main__":
    main()
