#!/usr/bin/env python3
"""Start the local RiYU web app and Mahjong Soul Vision Adapter together."""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path
from urllib.parse import urlparse

from mahjong_ai.perception import VisionAdapterConfig, VisionAdapterRuntime
from mahjong_ai.perception.profile import inspect_vision_profile


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Vision Adapter JSON config")
    parser.add_argument("--checkpoint", help="validated analysis checkpoint")
    parser.add_argument("--belief-checkpoint", help="validated belief checkpoint")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-open", action="store_true", help="do not open the web app")
    parser.add_argument(
        "--require-complete-profile",
        action="store_true",
        help="refuse startup unless every production Vision capability is calibrated",
    )
    parser.add_argument(
        "--save-latest-frame",
        help="local-only PNG overwritten periodically for capture diagnostics",
    )
    parser.add_argument(
        "--skip-capture-preflight",
        action="store_true",
        help="skip the initial permission/scene probe",
    )
    parser.add_argument(
        "--log-level",
        choices=("debug", "info", "warning", "error"),
        default="info",
    )
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = VisionAdapterConfig.load(config_path)
    profile = inspect_vision_profile(config.layout_path, config.template_directory)
    if not profile.complete:
        details = "\n".join(f"  - {issue.detail}" for issue in profile.issues)
        message = "Vision profile is usable only for its calibrated subset:\n" + details
        if args.require_complete_profile:
            parser.error(message)
        print(f"WARNING: {message}", file=sys.stderr, flush=True)
    if config.capture_region is None:
        parser.error(
            "the integrated realtime launcher requires macos_region or macos_window; "
            "use serve_analysis.py and run_vision_adapter.py separately for directory replay"
        )
    parsed = urlparse(config.websocket_url)
    port = parsed.port or (443 if parsed.scheme == "wss" else 80)
    if parsed.scheme != "ws" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        parser.error("one-command launcher requires a local ws:// Vision Adapter URL")
    if parsed.path != "/ws/realtime/events/source":
        parser.error("config websocket_url must use /ws/realtime/events/source")
    existing = _server_health(port)
    if existing is not None:
        parser.error(
            f"RiYU is already running on port {port}. Stop the existing launcher with "
            "Ctrl+C before starting another one."
        )
    if not _port_is_available(port):
        parser.error(f"port {port} is already used by another application")
    if not args.skip_capture_preflight:
        try:
            probe = VisionAdapterRuntime(config)
            frame = probe.source.capture()
            perception = probe.parser.parse(frame)
        except (OSError, RuntimeError, ValueError) as error:
            parser.error(f"capture preflight failed: {error}")
        if perception.scene_found:
            print(
                "Capture preflight: Mahjong Soul scene found "
                f"(unknown slots={len(perception.unknown_slots)}).",
                flush=True,
            )
        else:
            print(
                "WARNING: capture permission works, but the Mahjong Soul game scene "
                "is not currently visible in the configured source.",
                file=sys.stderr,
                flush=True,
            )
    web_dist = ROOT / "apps" / "web" / "dist"
    ui_index = web_dist / "index.html"
    if not ui_index.is_file():
        parser.error(
            "web app is not built; run `npm run build --prefix apps/web` first"
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
        "--web-dist",
        str(web_dist),
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
        web_url = f"http://127.0.0.1:{port}/#realtime"
        if not args.no_open:
            webbrowser.open(web_url)
        print(f"RiYU web app: {web_url}", flush=True)
        print("Stop both processes with Ctrl+C.", flush=True)
        adapter_command = [
            sys.executable,
            str(ROOT / "tools" / "run_vision_adapter.py"),
            str(config_path),
            "--log-level",
            args.log_level.upper(),
        ]
        if args.save_latest_frame:
            adapter_command.extend(
                ("--save-latest-frame", str(Path(args.save_latest_frame).resolve()))
            )
        adapter = subprocess.Popen(
            adapter_command,
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


def _server_health(port: int) -> dict[str, object] | None:
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/health", timeout=0.4
        ) as response:
            if response.status != 200:
                return None
            value = json.loads(response.read())
            return (
                value
                if isinstance(value, dict) and value.get("status") == "ok"
                else None
            )
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None


def _port_is_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


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
