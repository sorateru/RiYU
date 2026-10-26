"""Stateful, service-independent streaming analysis.

This module intentionally accepts only normalized observable ``GameState``
snapshots.  It contains no screen capture, network interception, browser
injection, or input automation.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from mahjong_ai import validate_state_json
from mahjong_ai.schema import canonical_json

from .explanation import explain_analysis

REALTIME_PROTOCOL = "mahjong-ai-realtime/v1"
SOURCE_KINDS = frozenset({"local_simulator", "offline_replay", "authorized_external_adapter"})
EXTERNAL_AUTHORIZATION = "source_operator_confirms_authorized"
MAJSOUL_SOURCE_SERVICE = "majsoul"
MAJSOUL_REALTIME_PROFILE = "majsoul-normalized-observation/v1"
SUPPORTED_SOURCE_SERVICES = frozenset({MAJSOUL_SOURCE_SERVICE})


class RealtimeProtocolError(ValueError):
    """A client-visible protocol validation error."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class SessionHello:
    session_id: str
    observer: int
    source_kind: str
    source_service: str | None = None
    source_profile: str | None = None
    authorization_reference: str | None = None


class RealtimeAnalysisSession:
    """Validate and analyze one ordered stream of observable snapshots."""

    def __init__(self, analyzer: Callable[[dict[str, Any]], dict[str, Any]]) -> None:
        self._analyzer = analyzer
        self._hello: SessionHello | None = None
        self._last_sequence: int | None = None
        self._last_fingerprint: str | None = None
        self._last_summary: dict[str, Any] | None = None

    @property
    def is_open(self) -> bool:
        return self._hello is not None

    @property
    def hello(self) -> SessionHello:
        return self._require_open()

    def open(self, message: dict[str, Any]) -> dict[str, Any]:
        if self._hello is not None:
            raise RealtimeProtocolError("already_initialized", "hello has already been accepted")
        if message.get("type") != "hello":
            raise RealtimeProtocolError("handshake_required", "the first message must be hello")
        if message.get("protocol") != REALTIME_PROTOCOL:
            raise RealtimeProtocolError("unsupported_protocol", f"protocol must be {REALTIME_PROTOCOL}")

        session_id = message.get("session_id")
        if not isinstance(session_id, str) or not session_id.strip() or len(session_id) > 128:
            raise RealtimeProtocolError("invalid_session_id", "session_id must contain 1 to 128 characters")
        observer = message.get("observer")
        if not isinstance(observer, int) or isinstance(observer, bool) or not 0 <= observer <= 3:
            raise RealtimeProtocolError("invalid_observer", "observer must be an integer from 0 to 3")
        source_kind = message.get("source_kind")
        if source_kind not in SOURCE_KINDS:
            raise RealtimeProtocolError(
                "invalid_source_kind",
                "source_kind must be local_simulator, offline_replay, or authorized_external_adapter",
            )
        if (
            source_kind == "authorized_external_adapter"
            and message.get("authorization") != EXTERNAL_AUTHORIZATION
        ):
            raise RealtimeProtocolError(
                "source_not_authorized",
                "external adapters require an explicit source-operator authorization assertion",
            )

        source_service = message.get("source_service")
        source_profile = message.get("source_profile")
        authorization_reference = message.get("authorization_reference")
        if source_service is not None:
            if source_service not in SUPPORTED_SOURCE_SERVICES:
                raise RealtimeProtocolError(
                    "unsupported_source_service",
                    "source_service is not supported by this runtime",
                )
            if source_service == MAJSOUL_SOURCE_SERVICE:
                if source_kind not in {"offline_replay", "authorized_external_adapter"}:
                    raise RealtimeProtocolError(
                        "invalid_source_kind",
                        "majsoul streams must be offline_replay or authorized_external_adapter",
                    )
                if source_profile != MAJSOUL_REALTIME_PROFILE:
                    raise RealtimeProtocolError(
                        "unsupported_source_profile",
                        f"majsoul source_profile must be {MAJSOUL_REALTIME_PROFILE}",
                    )
                if source_kind == "authorized_external_adapter":
                    if (
                        not isinstance(authorization_reference, str)
                        or not authorization_reference.strip()
                        or len(authorization_reference) > 256
                    ):
                        raise RealtimeProtocolError(
                            "authorization_reference_required",
                            "authorized majsoul streams require a permission reference",
                        )
                    authorization_reference = authorization_reference.strip()
        elif source_profile is not None or authorization_reference is not None:
            raise RealtimeProtocolError(
                "source_service_required",
                "source_profile and authorization_reference require source_service",
            )

        self._hello = SessionHello(
            session_id.strip(),
            observer,
            source_kind,
            source_service,
            source_profile,
            authorization_reference,
        )
        return {
            "type": "ready",
            "protocol": REALTIME_PROTOCOL,
            "session_id": self._hello.session_id,
            "observer": observer,
            "source_kind": source_kind,
            "source_service": source_service,
            "source_profile": source_profile,
            "input_contract": "normalized_observable_game_state_only",
            "acquisition_contract": "no_capture_no_interception_no_automation",
        }

    def analyze(self, message: dict[str, Any]) -> dict[str, Any]:
        hello = self._require_open()
        if message.get("type") != "analyze":
            raise RealtimeProtocolError("invalid_message_type", "expected an analyze message")
        sequence = message.get("sequence")
        if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 0:
            raise RealtimeProtocolError("invalid_sequence", "sequence must be a non-negative integer")
        if self._last_sequence is not None and sequence <= self._last_sequence:
            raise RealtimeProtocolError(
                "sequence_not_monotonic",
                f"sequence must be greater than {self._last_sequence}",
            )
        state = message.get("state")
        if not isinstance(state, dict):
            raise RealtimeProtocolError("invalid_state", "state must be a JSON object")

        try:
            normalized = json.loads(validate_state_json(canonical_json(state)))
        except (TypeError, ValueError) as error:
            raise RealtimeProtocolError("invalid_state", str(error)) from error
        self._validate_observation_scope(normalized, hello.observer)

        started = time.perf_counter()
        received_at = time.time()
        try:
            result = self._analyzer(normalized)
        except (TypeError, ValueError) as error:
            raise RealtimeProtocolError("analysis_failed", str(error)) from error
        result["explanations"] = explain_analysis(result)
        result["safety_scope"] = "normalized_authorized_source_only"
        fingerprint = hashlib.sha256(canonical_json(normalized).encode("utf-8")).hexdigest()
        summary = _analysis_summary(result)
        result["realtime"] = {
            "source_service": hello.source_service,
            "source_profile": hello.source_profile,
            "received_at_unix": received_at,
            "analysis_ms": (time.perf_counter() - started) * 1_000.0,
            "state_fingerprint": fingerprint,
            "state_changed": fingerprint != self._last_fingerprint,
            "changes": _summary_changes(self._last_summary, summary),
        }
        self._last_sequence = sequence
        self._last_fingerprint = fingerprint
        self._last_summary = summary
        return {
            "type": "analysis",
            "protocol": REALTIME_PROTOCOL,
            "session_id": hello.session_id,
            "sequence": sequence,
            "result": result,
        }

    def _require_open(self) -> SessionHello:
        if self._hello is None:
            raise RealtimeProtocolError("handshake_required", "hello must be accepted before analysis")
        return self._hello

    @staticmethod
    def _validate_observation_scope(state: dict[str, Any], observer: int) -> None:
        information = state.get("information")
        if not isinstance(information, dict):
            raise RealtimeProtocolError("invalid_state", "state.information is required")
        if information.get("observer") != observer:
            raise RealtimeProtocolError(
                "observer_mismatch",
                "state observer does not match the session observer",
            )
        if information.get("observer_hand_visible") is not True:
            raise RealtimeProtocolError("invalid_visibility", "the observer hand must be visible")
        if information.get("opponent_hands_visible") is not False:
            raise RealtimeProtocolError("secret_information_rejected", "opponent hands must not be visible")
        if information.get("wall_order_visible") is not False:
            raise RealtimeProtocolError("secret_information_rejected", "wall order must not be visible")


def _analysis_summary(result: dict[str, Any]) -> dict[str, Any]:
    shanten_value = (result.get("shanten") or {}).get("value") or {}
    policy = (result.get("model_policy") or {}).get("value") or []
    efficiency = (result.get("discard_efficiency") or {}).get("value") or []
    best = policy[0].get("action") if policy else None
    if best is None and efficiency:
        best = {"type": "discard", "tile": efficiency[0].get("discard")}
    return {
        "shanten": shanten_value.get("minimum"),
        "best_action": best,
        "legal_actions": len((result.get("legal_actions") or {}).get("value") or []),
    }


def _summary_changes(
    previous: dict[str, Any] | None, current: dict[str, Any]
) -> dict[str, bool]:
    if previous is None:
        return {"initial": True, "shanten": True, "best_action": True, "legal_actions": True}
    return {
        "initial": False,
        "shanten": previous.get("shanten") != current.get("shanten"),
        "best_action": previous.get("best_action") != current.get("best_action"),
        "legal_actions": previous.get("legal_actions") != current.get("legal_actions"),
    }
