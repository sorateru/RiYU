"""Safe URL-to-replay lookup through a local, user-controlled cache.

No request is sent to Mahjong Soul.  A share URL is reduced to an opaque
replay identifier, hashed, and matched to decoded JSON that the user has
explicitly registered on this machine.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from .mahjong_soul import convert_decoded_match, create_mahjong_soul_bundle

MAX_CACHED_REPLAY_BYTES = 16 * 1024 * 1024
_MAHJONG_SOUL_HOSTS = frozenset({"mahjongsoul.com", "yo-star.com"})
_REPLAY_ID = re.compile(r"^[A-Za-z0-9_-]{8,256}$")


class ReplayURLCacheError(ValueError):
    """Base error for URL parsing and local cache operations."""


class ReplayURLNotFound(ReplayURLCacheError):
    """The URL is valid but has not been registered locally."""


def mahjong_soul_replay_id(url: str) -> str:
    if not isinstance(url, str) or len(url) > 4096:
        raise ReplayURLCacheError("replay URL must be a string of at most 4096 characters")
    parts = urlsplit(url.strip())
    if parts.scheme != "https" or not parts.hostname or parts.username or parts.password:
        raise ReplayURLCacheError(
            "Mahjong Soul replay URL must be HTTPS and contain no credentials"
        )
    hostname = parts.hostname.lower()
    if not any(
        hostname == suffix or hostname.endswith(f".{suffix}")
        for suffix in _MAHJONG_SOUL_HOSTS
    ):
        raise ReplayURLCacheError("Mahjong Soul replay URL has an unexpected host")
    if parts.port not in {None, 443}:
        raise ReplayURLCacheError("Mahjong Soul replay URL must use the default HTTPS port")

    queries = [parse_qs(parts.query, keep_blank_values=True)]
    if parts.fragment:
        queries.append(parse_qs(parts.fragment.lstrip("?"), keep_blank_values=True))
    values = [value for query in queries for value in query.get("paipu", [])]
    if len(values) != 1 or not _REPLAY_ID.fullmatch(values[0]):
        raise ReplayURLCacheError("Mahjong Soul replay URL must contain one valid paipu ID")
    return values[0]


def replay_cache_key(url: str) -> str:
    replay_id = mahjong_soul_replay_id(url)
    return hashlib.sha256(f"mahjong_soul:{replay_id}".encode()).hexdigest()


class LocalReplayURLCache:
    """Resolve share URLs to explicitly registered decoded replay documents."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory).expanduser().resolve()

    def register(self, url: str, decoded: dict[str, Any]) -> Path:
        # Full conversion is the admission check.  Invalid or incomplete data
        # is never persisted into the URL cache.
        convert_decoded_match(decoded)
        encoded = (json.dumps(decoded, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )
        if len(encoded) > MAX_CACHED_REPLAY_BYTES:
            raise ReplayURLCacheError(
                f"decoded replay exceeds {MAX_CACHED_REPLAY_BYTES} bytes"
            )
        self.directory.mkdir(parents=True, exist_ok=True)
        destination = self.directory / f"{replay_cache_key(url)}.json"
        if destination.exists():
            raise ReplayURLCacheError(f"replay URL is already registered: {destination}")

        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".replay-", suffix=".tmp", dir=self.directory
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
        return destination

    def resolve(self, url: str) -> dict[str, Any]:
        path = self.directory / f"{replay_cache_key(url)}.json"
        if not path.is_file():
            raise ReplayURLNotFound(
                "この共有URLはローカル登録されていません。"
                " register_mahjong_soul_replay.py で所有するdecode済み牌譜を登録してください。"
            )
        if path.stat().st_size > MAX_CACHED_REPLAY_BYTES:
            raise ReplayURLCacheError("cached replay exceeds the size limit")
        try:
            decoded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ReplayURLCacheError("cached replay is unreadable or invalid JSON") from error
        if not isinstance(decoded, dict):
            raise ReplayURLCacheError("cached replay must contain a JSON object")
        return create_mahjong_soul_bundle(decoded, share_url=url)
