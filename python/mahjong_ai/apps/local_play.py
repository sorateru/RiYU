"""Local human-versus-AI hanchan sessions, separate from analysis streams."""

from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path
from typing import Any

import torch

from mahjong_ai import FourPlayerSimulator
from mahjong_ai.ai import ExpectedValuePolicy, MahjongNetwork, ModelConfig
from mahjong_ai.evaluation.arena import (
    HandResult,
    _derive_hand_seed,
    _is_tenpai,
    _next_round,
    _ranks,
    _terminal_result,
)
from mahjong_ai.schema import canonical_json


class LocalPlayError(ValueError):
    """A safe, user-facing local-play error."""


class LocalPlayUnavailable(LocalPlayError):
    """Raised when no local opponent model was configured."""


class _GreedyPolicy:
    def __init__(self, model: MahjongNetwork, device: str) -> None:
        self.model = model
        self.device = device
        self.expected_value = ExpectedValuePolicy(model, device=device)

    def choose(self, state: dict[str, Any], legal: list[dict[str, Any]]) -> dict[str, Any]:
        if not legal:
            raise LocalPlayError("AI received an empty legal-action set")
        return legal[self.expected_value.choose(state, legal)]


class LocalPlaySession:
    """One isolated east-south match with one human and three AI seats."""

    def __init__(
        self,
        model: MahjongNetwork,
        model_id: str,
        *,
        seed: int,
        human_seat: int,
        device: str,
    ) -> None:
        if human_seat not in range(4):
            raise LocalPlayError("human_seat must be between 0 and 3")
        self.session_id = uuid.uuid4().hex
        self.seed = seed
        self.human_seat = human_seat
        self.model_id = model_id
        self.device = device
        self.policy = _GreedyPolicy(model, device)
        self.lock = threading.RLock()
        self.round_state: dict[str, Any] = {
            "wind": "east",
            "hand_number": 1,
            "honba": 0,
            "riichi_sticks": 0,
            "dealer": 0,
        }
        self.scores: tuple[int, int, int, int] = (25_000, 25_000, 25_000, 25_000)
        self.hand_index = 0
        self.match_ended = False
        self.last_hand: dict[str, Any] | None = None
        self.simulator = self._new_simulator()
        self._advance_until_human()

    def act(self, action: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            if self.match_ended:
                raise LocalPlayError("the hanchan has already ended")
            state = self._observation(self.human_seat)
            legal = self._human_legal_actions(state)
            selected = next((candidate for candidate in legal if candidate == action), None)
            if selected is None:
                raise LocalPlayError("action is not legal in the current position")

            phase_type = state["phase"]["phase"]
            if phase_type == "awaiting_discard":
                self.simulator.act_json(self.human_seat, canonical_json(selected))
            elif phase_type in {"awaiting_response", "awaiting_kan_response"}:
                choices = self._ai_response_choices(state)
                choices.append((self.human_seat, selected))
                self.simulator.resolve_responses_json(canonical_json(choices))
                self._draw_after_unclaimed_response()
            else:
                raise LocalPlayError("the session is not waiting for a human action")
            self._advance_until_human()
            return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            observation = self._observation(self.human_seat)
            legal = [] if self.match_ended else self._human_legal_actions(observation)
            return {
                "format": "mahjong-ai-local-play/v1",
                "session_id": self.session_id,
                "mode": "local_human_vs_ai",
                "analysis_attached": False,
                "opponent_model_id": self.model_id,
                "human_seat": self.human_seat,
                "seed": self.seed,
                "hand_index": self.hand_index,
                "status": "match_ended" if self.match_ended else "awaiting_human_action",
                "round": observation.get("round") or self.round_state,
                "scores": list(self.scores),
                "ranks": list(_ranks(self.scores)) if self.match_ended else None,
                "observation": observation,
                "legal_actions": legal,
                "last_hand": self.last_hand,
            }

    def _advance_until_human(self) -> None:
        for _ in range(2_000):
            states = [self._observation(player) for player in range(4)]
            phase = states[0]["phase"]
            phase_type = phase["phase"]
            if phase_type == "ended":
                self._finish_hand(states)
                if self.match_ended:
                    return
                states = [self._observation(player) for player in range(4)]
                phase = states[0]["phase"]
                phase_type = phase["phase"]
            if phase_type in {"awaiting_draw", "awaiting_rinshan_draw"}:
                self.simulator.draw()
                continue
            if phase_type == "awaiting_discard":
                player = int(phase["player"])
                if player == self.human_seat:
                    return
                legal = json.loads(self.simulator.legal_actions_json(player))
                action = self.policy.choose(states[player], legal)
                self.simulator.act_json(player, canonical_json(action))
                continue
            if phase_type in {"awaiting_response", "awaiting_kan_response"}:
                human_legal = self._human_legal_actions(states[self.human_seat])
                if any(action["type"] != "pass" for action in human_legal):
                    return
                choices = self._ai_response_choices(states[0])
                self.simulator.resolve_responses_json(canonical_json(choices))
                self._draw_after_unclaimed_response()
                continue
            raise RuntimeError(f"unsupported local-play simulator phase: {phase_type}")
        raise RuntimeError("local-play session exceeded its automatic action limit")

    def _ai_response_choices(self, state: dict[str, Any]) -> list[tuple[int, dict[str, Any]]]:
        phase = state["phase"]
        source = int(phase.get("discarder", phase.get("declarer")))
        choices: list[tuple[int, dict[str, Any]]] = []
        for player in range(4):
            if player in {source, self.human_seat}:
                continue
            legal = json.loads(self.simulator.legal_actions_json(player))
            if legal:
                choices.append((player, self.policy.choose(self._observation(player), legal)))
        return choices

    def _human_legal_actions(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        phase = state["phase"]
        phase_type = phase["phase"]
        if phase_type == "awaiting_discard" and int(phase["player"]) == self.human_seat:
            return json.loads(self.simulator.legal_actions_json(self.human_seat))
        if phase_type in {"awaiting_response", "awaiting_kan_response"}:
            source = int(phase.get("discarder", phase.get("declarer")))
            if source != self.human_seat:
                return json.loads(self.simulator.legal_actions_json(self.human_seat))
        return []

    def _draw_after_unclaimed_response(self) -> None:
        next_phase = self._observation(0)["phase"]["phase"]
        if next_phase in {"awaiting_response", "awaiting_kan_response"}:
            self.simulator.draw()

    def _finish_hand(self, states: list[dict[str, Any]]) -> None:
        final_scores = tuple(int(value) for value in states[0]["scores"])
        result_type, winners, loser = _terminal_result(states[0])
        dealer = int(self.round_state["dealer"])
        dealer_tenpai = _is_tenpai(states[dealer]) if result_type == "exhaustive" else None
        result = HandResult(
            seed=_derive_hand_seed(self.seed, self.hand_index),
            round_state=dict(states[0]["round"]),
            final_scores=final_scores,  # type: ignore[arg-type]
            result_type=result_type,
            winners=winners,
            loser=loser,
            dealer_tenpai=dealer_tenpai,
            open_meld_counts=tuple(
                len(states[0]["players"][seat]["melds"]) for seat in range(4)
            ),
        )
        self.scores = final_scores  # type: ignore[assignment]
        self.last_hand = {
            "round": dict(self.round_state),
            "result_type": result_type,
            "winners": list(winners),
            "loser": loser,
            "scores": list(final_scores),
        }
        next_round = _next_round(self.round_state, result)
        if next_round is None:
            self.match_ended = True
            return
        self.hand_index += 1
        if self.hand_index >= 64:
            raise RuntimeError("local hanchan exceeded 64 hands")
        self.round_state = next_round
        self.simulator = self._new_simulator()

    def _new_simulator(self) -> FourPlayerSimulator:
        return FourPlayerSimulator(
            _derive_hand_seed(self.seed, self.hand_index),
            canonical_json(self.round_state),
            list(self.scores),
        )

    def _observation(self, player: int) -> dict[str, Any]:
        return json.loads(self.simulator.observation_json(player))


class LocalPlayManager:
    """Own in-memory local-play sessions; never reads an external game client."""

    def __init__(
        self,
        model: MahjongNetwork | None = None,
        *,
        model_id: str = "unconfigured",
        device: str = "cpu",
        maximum_sessions: int = 8,
    ) -> None:
        self.model = model.to(device).eval() if model is not None else None
        self.model_id = model_id
        self.device = device
        self.maximum_sessions = maximum_sessions
        self.sessions: dict[str, LocalPlaySession] = {}
        self.lock = threading.RLock()

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str = "cpu") -> "LocalPlayManager":
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = MahjongNetwork(ModelConfig(**checkpoint["model_config"]))
        model.load_state_dict(checkpoint["model_state"])
        metadata = checkpoint.get("metadata") or {}
        return cls(
            model,
            model_id=str(metadata.get("model_id") or Path(path).stem),
            device=device,
        )

    @property
    def available(self) -> bool:
        return self.model is not None

    def start(self, *, seed: int, human_seat: int) -> dict[str, Any]:
        with self.lock:
            if self.model is None:
                raise LocalPlayUnavailable("local-play AI checkpoint is not loaded")
            if not 0 <= seed < 2**64:
                raise LocalPlayError("seed must be between 0 and 2^64 - 1")
            if len(self.sessions) >= self.maximum_sessions:
                raise LocalPlayError("too many local-play sessions are open")
            session = LocalPlaySession(
                self.model,
                self.model_id,
                seed=seed,
                human_seat=human_seat,
                device=self.device,
            )
            self.sessions[session.session_id] = session
            return session.snapshot()

    def get(self, session_id: str) -> LocalPlaySession:
        try:
            return self.sessions[session_id]
        except KeyError as error:
            raise LocalPlayError("local-play session was not found") from error

    def close(self, session_id: str) -> None:
        with self.lock:
            if self.sessions.pop(session_id, None) is None:
                raise LocalPlayError("local-play session was not found")
