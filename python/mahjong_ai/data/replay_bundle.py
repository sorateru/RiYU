"""Validated container for offline replay imports and their provenance."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from .log import EventLog, parse_event_log

BUNDLE_FORMAT = "mahjong-ai-replay-bundle/v1"
MAX_BUNDLE_LOGS = 64
MAX_SOURCE_REFERENCE_LENGTH = 256
MAX_SOURCE_URL_LENGTH = 4096

SUPPORTED_PLATFORMS = frozenset({"mahjong_soul", "tenhou", "local", "other"})
SUPPORTED_ACQUISITIONS = frozenset(
    {"user_export", "manual_conversion", "authorized_adapter", "generated"}
)
SUPPORTED_AUTHORIZATIONS = frozenset(
    {"user_owned", "explicit_permission", "open_license", "generated"}
)


class ReplayBundleError(ValueError):
    """Raised when an offline replay bundle does not satisfy the import contract."""


@dataclass(frozen=True)
class ReplaySource:
    platform: str
    acquisition: str
    authorization: str
    converter_name: str
    converter_version: str
    reference: str | None = None

    def public_summary(self) -> dict[str, str | None]:
        return {
            "platform": self.platform,
            "acquisition": self.acquisition,
            "authorization": self.authorization,
            "converterName": self.converter_name,
            "converterVersion": self.converter_version,
            "reference": self.reference,
        }


@dataclass(frozen=True)
class ReplayBundle:
    source: ReplaySource
    logs: tuple[EventLog, ...]


def parse_replay_bundle(raw: dict[str, Any]) -> ReplayBundle:
    if not isinstance(raw, dict):
        raise ReplayBundleError("replay bundle must be a JSON object")
    if raw.get("format") != BUNDLE_FORMAT:
        raise ReplayBundleError(f"expected format {BUNDLE_FORMAT!r}")
    _reject_unknown(raw, {"format", "source", "logs"}, "bundle")

    source_raw = raw.get("source")
    if not isinstance(source_raw, dict):
        raise ReplayBundleError("bundle.source must be a JSON object")
    source = _parse_source(source_raw)

    documents = raw.get("logs")
    if not isinstance(documents, list) or not documents:
        raise ReplayBundleError("bundle.logs must be a non-empty array")
    if len(documents) > MAX_BUNDLE_LOGS:
        raise ReplayBundleError(f"bundle exceeds the {MAX_BUNDLE_LOGS}-round limit")

    logs: list[EventLog] = []
    for index, document in enumerate(documents):
        if not isinstance(document, dict):
            raise ReplayBundleError(f"bundle.logs[{index}] must be a JSON object")
        logs.append(parse_event_log(document, default_log_id=f"bundle-round-{index + 1}"))
    return ReplayBundle(source=source, logs=tuple(logs))


def create_replay_bundle(
    documents: list[dict[str, Any]],
    *,
    platform: str,
    acquisition: str,
    authorization: str,
    converter_name: str,
    converter_version: str,
    reference: str | None = None,
    share_url: str | None = None,
) -> dict[str, Any]:
    raw: dict[str, Any] = {
        "format": BUNDLE_FORMAT,
        "source": {
            "platform": platform,
            "acquisition": acquisition,
            "authorization": authorization,
            "converter": {"name": converter_name, "version": converter_version},
        },
        "logs": documents,
    }
    if reference is not None:
        raw["source"]["reference"] = reference
    if share_url is not None:
        raw["source"]["share_url"] = share_url
    # Run the exact same checks used by the API before returning the document.
    parse_replay_bundle(raw)
    return raw


def _parse_source(raw: dict[str, Any]) -> ReplaySource:
    _reject_unknown(
        raw,
        {
            "platform",
            "acquisition",
            "authorization",
            "converter",
            "reference",
            "share_url",
        },
        "bundle.source",
    )
    platform = _enum_field(raw, "platform", SUPPORTED_PLATFORMS)
    acquisition = _enum_field(raw, "acquisition", SUPPORTED_ACQUISITIONS)
    authorization = _enum_field(raw, "authorization", SUPPORTED_AUTHORIZATIONS)

    converter = raw.get("converter")
    if not isinstance(converter, dict):
        raise ReplayBundleError("bundle.source.converter must be a JSON object")
    _reject_unknown(converter, {"name", "version"}, "bundle.source.converter")
    converter_name = _bounded_text(converter.get("name"), "converter.name", 80)
    converter_version = _bounded_text(converter.get("version"), "converter.version", 80)

    reference_raw = raw.get("reference")
    reference = None
    if reference_raw is not None:
        reference = _bounded_text(
            reference_raw, "source.reference", MAX_SOURCE_REFERENCE_LENGTH
        )

    share_url = raw.get("share_url")
    if share_url is not None:
        share_url = _bounded_text(share_url, "source.share_url", MAX_SOURCE_URL_LENGTH)
        parts = urlsplit(share_url)
        if parts.scheme != "https" or not parts.hostname or parts.username or parts.password:
            raise ReplayBundleError("source.share_url must be an HTTPS URL without credentials")
        hostname = parts.hostname.lower()
        if platform == "mahjong_soul" and not _host_matches(
            hostname, {"mahjongsoul.com", "yo-star.com"}
        ):
            raise ReplayBundleError("Mahjong Soul share URL has an unexpected host")
        if platform == "tenhou" and not _host_matches(hostname, {"tenhou.net"}):
            raise ReplayBundleError("Tenhou share URL has an unexpected host")
        url_reference = f"url-sha256:{hashlib.sha256(share_url.encode()).hexdigest()[:24]}"
        if reference is not None and reference != url_reference:
            raise ReplayBundleError(
                "source.reference must be omitted when source.share_url is provided"
            )
        reference = url_reference

    if platform in {"mahjong_soul", "tenhou"} and authorization == "generated":
        raise ReplayBundleError("external replay platforms cannot use generated authorization")

    return ReplaySource(
        platform=platform,
        acquisition=acquisition,
        authorization=authorization,
        converter_name=converter_name,
        converter_version=converter_version,
        reference=reference,
    )


def _enum_field(raw: dict[str, Any], name: str, allowed: frozenset[str]) -> str:
    value = raw.get(name)
    if not isinstance(value, str) or value not in allowed:
        raise ReplayBundleError(f"source.{name} must be one of {sorted(allowed)}")
    return value


def _bounded_text(value: Any, name: str, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReplayBundleError(f"{name} must be a non-empty string")
    normalized = value.strip()
    if len(normalized) > maximum:
        raise ReplayBundleError(f"{name} exceeds {maximum} characters")
    return normalized


def _reject_unknown(raw: dict[str, Any], allowed: set[str], name: str) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ReplayBundleError(f"{name} contains unknown fields: {unknown}")


def _host_matches(hostname: str, allowed_suffixes: set[str]) -> bool:
    return any(
        hostname == suffix or hostname.endswith(f".{suffix}")
        for suffix in allowed_suffixes
    )
