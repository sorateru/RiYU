"""Deterministic, seat-balanced two-model hanchan arena evaluation."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mahjong_ai import FourPlayerSimulator, analyze_state_json, calculate_shanten_json
from mahjong_ai.ai import (
    ExpectedValuePolicy,
    FeatureEncoder,
    ImitationDiscardLikelihood,
    MahjongNetwork,
    ModelConfig,
    RolloutConfig,
    SequenceBeliefPredictor,
)
from mahjong_ai.schema import canonical_json, tile_index

from .metrics import confidence_interval


_SEAT_PATTERNS: tuple[tuple[str, str, str, str], ...] = (
    ("a", "a", "b", "b"),
    ("b", "a", "a", "b"),
    ("b", "b", "a", "a"),
    ("a", "b", "b", "a"),
)
_STARTING_SCORE = 25_000
_TARGET_SCORE = 30_000
_MAX_HANDS_PER_MATCH = 64
_WIND_ORDER = ("east", "south", "west", "north")
_POLICY_MODES = {
    "tile_efficiency",
    "imitation",
    "hybrid_no_search",
    "hybrid_search",
    "hybrid_search_high_accuracy",
}


@dataclass(frozen=True)
class HandResult:
    seed: int
    round_state: dict[str, Any]
    initial_scores: tuple[int, int, int, int]
    final_scores: tuple[int, int, int, int]
    result_type: str
    winners: tuple[int, ...]
    loser: int | None
    dealer_tenpai: bool | None
    open_meld_counts: tuple[int, int, int, int]


@dataclass(frozen=True)
class MatchResult:
    block: int
    seed: int
    pattern: tuple[str, str, str, str]
    final_scores: tuple[int, int, int, int]
    ranks: tuple[int, int, int, int]
    hand_results: tuple[HandResult, ...]
    action_counts: dict[str, dict[str, int]]


class GreedyModelPolicy:
    def __init__(
        self,
        model: MahjongNetwork,
        label: str,
        mode: str,
        belief_predictor: SequenceBeliefPredictor | None = None,
    ) -> None:
        if mode not in _POLICY_MODES:
            raise ValueError(f"unknown arena policy mode: {mode}")
        self.model = model
        self.label = label
        self.mode = mode
        self.encoder = FeatureEncoder()
        self.expected_value = ExpectedValuePolicy(
            model,
            enable_rollout=mode in {"hybrid_search", "hybrid_search_high_accuracy"},
            rollout_config=(
                RolloutConfig.high_accuracy()
                if mode == "hybrid_search_high_accuracy"
                else None
            ),
            belief_predictor=belief_predictor,
            action_likelihood=ImitationDiscardLikelihood(model),
            risk_predictor=(
                belief_predictor.action_risks if belief_predictor is not None else None
            ),
        )

    def choose(self, state: dict[str, Any], legal: list[dict[str, Any]]) -> int:
        terminal = next(
            (index for index, action in enumerate(legal) if action["type"] in {"ron", "tsumo"}),
            None,
        )
        if terminal is not None:
            return terminal
        if self.mode == "tile_efficiency":
            analysis = json.loads(analyze_state_json(canonical_json(state)))
            for efficient in analysis["discard_efficiency"].get("value") or []:
                for index, action in enumerate(legal):
                    if action["type"] == "discard" and action["tile"] == efficient["discard"]:
                        return index
            return next(
                (index for index, action in enumerate(legal) if action["type"] == "pass"),
                0,
            )
        if self.mode == "imitation":
            encoded = self.encoder.encode(state, legal)
            with torch.inference_mode():
                output = self.model(
                    torch.from_numpy(encoded.planes).unsqueeze(0),
                    torch.from_numpy(encoded.action_features).unsqueeze(0),
                )
            return int(output["policy_logits"][0].argmax())
        return self.expected_value.choose(state, legal)


def evaluate_checkpoints(
    checkpoint_a: str | Path,
    checkpoint_b: str | Path,
    *,
    seed_blocks: int,
    base_seed: int = 20_261_019,
    workers: int = 1,
    policy_a: str = "hybrid_search",
    policy_b: str = "hybrid_search",
    belief_checkpoint_a: str | Path | None = None,
    belief_checkpoint_b: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate two checkpoints over seat-balanced east-south matches."""
    if seed_blocks <= 0:
        raise ValueError("seed_blocks must be positive")
    if workers <= 0:
        raise ValueError("workers must be positive")
    if policy_a not in _POLICY_MODES or policy_b not in _POLICY_MODES:
        raise ValueError(f"policy modes must be one of {sorted(_POLICY_MODES)}")
    path_a = Path(checkpoint_a).resolve()
    path_b = Path(checkpoint_b).resolve()
    belief_paths = {
        "a": Path(belief_checkpoint_a).resolve() if belief_checkpoint_a else None,
        "b": Path(belief_checkpoint_b).resolve() if belief_checkpoint_b else None,
    }
    identities = {
        "a": _checkpoint_identity(path_a, belief_paths["a"]),
        "b": _checkpoint_identity(path_b, belief_paths["b"]),
    }
    tasks = [(block, base_seed + block) for block in range(seed_blocks)]
    worker_count = min(workers, seed_blocks)
    initargs = (
        str(path_a),
        str(path_b),
        policy_a,
        policy_b,
        str(belief_paths["a"]) if belief_paths["a"] else None,
        str(belief_paths["b"]) if belief_paths["b"] else None,
    )
    if worker_count == 1:
        _initialize_arena_worker(*initargs)
        blocks = list(map(_play_seed_block, tasks))
        backend = "inline"
    else:
        context = multiprocessing.get_context("spawn")
        try:
            executor: ProcessPoolExecutor | ThreadPoolExecutor = ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=context,
                initializer=_initialize_arena_worker,
                initargs=initargs,
            )
            backend = "process"
        except (NotImplementedError, PermissionError):
            executor = ThreadPoolExecutor(
                max_workers=worker_count,
                initializer=_initialize_arena_worker,
                initargs=initargs,
            )
            backend = "thread_fallback"
        with executor:
            blocks = list(executor.map(_play_seed_block, tasks, chunksize=1))

    matches = [match for block in blocks for match in block]
    total_hands = sum(len(match.hand_results) for match in matches)
    model_summaries = {label: _summarize_model(matches, label) for label in ("a", "b")}
    paired = _paired_summary(blocks)
    rank_ci = paired["average_rank_difference_a_minus_b_95ci"]
    if len(blocks) < 2:
        verdict = "inconclusive"
    elif rank_ci[1] < 0:
        verdict = "a_better"
    elif rank_ci[0] > 0:
        verdict = "b_better"
    else:
        verdict = "inconclusive"
    return {
        "format": "mahjong-ai-arena/v2",
        "models": identities,
        "config": {
            "seed_blocks": seed_blocks,
            "matches": len(matches),
            "hands": total_hands,
            "base_seed": base_seed,
            "seat_patterns": [list(pattern) for pattern in _SEAT_PATTERNS],
            "policies": {"a": policy_a, "b": policy_b},
            "workers": worker_count,
            "execution_backend": backend,
            "unit": "east_south_hanchan",
            "starting_score": _STARTING_SCORE,
            "target_score": _TARGET_SCORE,
            "west_extension": True,
            "agari_yame": True,
            "tenpai_yame": True,
            "bankruptcy_ends_match": True,
        },
        "summary": model_summaries,
        "paired": paired,
        "verdict": verdict,
        "matches": [asdict(match) for match in matches],
    }


def _checkpoint_identity(
    path: Path, belief_path: Path | None = None
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    identity = {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "metadata": checkpoint["metadata"],
        "model_config": checkpoint["model_config"],
    }
    if belief_path is not None:
        belief = torch.load(belief_path, map_location="cpu", weights_only=False)
        identity["belief_checkpoint"] = {
            "path": str(belief_path),
            "sha256": hashlib.sha256(belief_path.read_bytes()).hexdigest(),
            "metadata": belief["metadata"],
            "model_config": belief["model_config"],
        }
    return identity


_ARENA_MODELS: dict[str, MahjongNetwork] = {}
_ARENA_POLICY_MODES: dict[str, str] = {}
_ARENA_BELIEFS: dict[str, SequenceBeliefPredictor | None] = {}


def _initialize_arena_worker(
    path_a: str,
    path_b: str,
    policy_a: str,
    policy_b: str,
    belief_path_a: str | None,
    belief_path_b: str | None,
) -> None:
    global _ARENA_MODELS, _ARENA_POLICY_MODES, _ARENA_BELIEFS
    torch.set_num_threads(1)
    _ARENA_MODELS = {"a": _load_model(path_a), "b": _load_model(path_b)}
    _ARENA_POLICY_MODES = {"a": policy_a, "b": policy_b}
    _ARENA_BELIEFS = {
        "a": SequenceBeliefPredictor.load(
            belief_path_a, require_validated=False
        ) if belief_path_a else None,
        "b": SequenceBeliefPredictor.load(
            belief_path_b, require_validated=False
        ) if belief_path_b else None,
    }


def _load_model(path: str) -> MahjongNetwork:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = MahjongNetwork(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state"])
    return model.eval()


def _play_seed_block(task: tuple[int, int]) -> list[MatchResult]:
    block, seed = task
    if set(_ARENA_MODELS) != {"a", "b"}:
        raise RuntimeError("arena worker is not initialized")
    return [_play_match(block, seed, pattern) for pattern in _SEAT_PATTERNS]


def _play_match(
    block: int,
    seed: int,
    pattern: tuple[str, str, str, str],
) -> MatchResult:
    round_state: dict[str, Any] = {
        "wind": "east",
        "hand_number": 1,
        "honba": 0,
        "riichi_sticks": 0,
        "dealer": 0,
    }
    scores = (_STARTING_SCORE,) * 4
    policies = [
        GreedyModelPolicy(
            _ARENA_MODELS[label],
            label,
            _ARENA_POLICY_MODES[label],
            _ARENA_BELIEFS[label],
        )
        for label in pattern
    ]
    action_counts = {"a": Counter(), "b": Counter()}
    hand_results: list[HandResult] = []
    for hand_index in range(_MAX_HANDS_PER_MATCH):
        hand = _play_hand(
            _derive_hand_seed(seed, hand_index),
            round_state,
            scores,
            pattern,
            policies,
            action_counts,
        )
        hand_results.append(hand)
        scores = hand.final_scores
        next_state = _next_round(round_state, hand)
        if next_state is None:
            return MatchResult(
                block=block,
                seed=seed,
                pattern=pattern,
                final_scores=scores,
                ranks=_ranks(scores),
                hand_results=tuple(hand_results),
                action_counts={label: dict(counts) for label, counts in action_counts.items()},
            )
        round_state = next_state
    raise RuntimeError(f"arena hanchan exceeded {_MAX_HANDS_PER_MATCH} hands")


def _derive_hand_seed(match_seed: int, hand_index: int) -> int:
    payload = f"mahjong-ai-hanchan-v1:{match_seed}:{hand_index}".encode("ascii")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def _play_hand(
    seed: int,
    round_state: dict[str, Any],
    scores: tuple[int, int, int, int],
    pattern: tuple[str, str, str, str],
    policies: list[GreedyModelPolicy],
    action_counts: dict[str, Counter[str]],
) -> HandResult:
    simulator = FourPlayerSimulator(seed, canonical_json(round_state), list(scores))
    for _ in range(400):
        states = [json.loads(simulator.observation_json(player)) for player in range(4)]
        phase = states[0]["phase"]
        phase_type = phase["phase"]
        if phase_type == "ended":
            final_scores = tuple(int(value) for value in states[0]["scores"])
            result_type, winners, loser = _terminal_result(states[0])
            dealer_tenpai = (
                _is_tenpai(states[int(round_state["dealer"])])
                if result_type == "exhaustive"
                else None
            )
            terminal_round = dict(states[0]["round"])
            return HandResult(
                seed=seed,
                round_state=terminal_round,
                initial_scores=scores,
                final_scores=final_scores,  # type: ignore[arg-type]
                result_type=result_type,
                winners=winners,
                loser=loser,
                dealer_tenpai=dealer_tenpai,
                open_meld_counts=tuple(
                    len(states[0]["players"][seat]["melds"]) for seat in range(4)
                ),
            )
        if phase_type in {"awaiting_draw", "awaiting_rinshan_draw"}:
            simulator.draw()
            continue
        if phase_type == "awaiting_discard":
            player = int(phase["player"])
            legal = json.loads(simulator.legal_actions_json(player))
            index = policies[player].choose(states[player], legal)
            action = legal[index]
            action_counts[pattern[player]][action["type"]] += 1
            simulator.act_json(player, canonical_json(action))
            continue
        if phase_type in {"awaiting_response", "awaiting_kan_response"}:
            source = int(phase.get("discarder", phase.get("declarer")))
            choices: list[tuple[int, dict[str, Any]]] = []
            for player in range(4):
                if player == source:
                    continue
                legal = json.loads(simulator.legal_actions_json(player))
                if not legal:
                    continue
                index = policies[player].choose(states[player], legal)
                action = legal[index]
                choices.append((player, action))
            before_events = len(states[0]["event_history"])
            simulator.resolve_responses_json(canonical_json(choices))
            after_state = json.loads(simulator.observation_json(0))
            resolved_events = after_state["event_history"][before_events:]
            for player, action in choices:
                action_type = action["type"]
                executed = action_type == "pass"
                if action_type == "ron":
                    executed = any(
                        event["type"] == "ron" and player in event["winners"]
                        for event in resolved_events
                    )
                elif action_type in {"chi", "pon", "kan"}:
                    executed = any(
                        event["type"] == action_type and event["player"] == player
                        for event in resolved_events
                    )
                if executed:
                    action_counts[pattern[player]][action_type] += 1
            next_phase = after_state["phase"]["phase"]
            if next_phase in {"awaiting_response", "awaiting_kan_response"}:
                simulator.draw()
            continue
        raise RuntimeError(f"unsupported simulator phase: {phase_type}")
    raise RuntimeError("arena hand exceeded 400 simulator decisions")


def _terminal_result(state: dict[str, Any]) -> tuple[str, tuple[int, ...], int | None]:
    for event in reversed(state["event_history"]):
        if event["type"] == "ron":
            return "ron", tuple(int(value) for value in event["winners"]), int(event["loser"])
        if event["type"] == "tsumo":
            return "tsumo", (int(event["winner"]),), None
        if event["type"] == "draw_game":
            return str(event["reason"]), (), None
    raise RuntimeError("ended arena hand has no result event")


def _is_tenpai(state: dict[str, Any]) -> bool:
    counts = [0] * 34
    for tile in state["hand"]:
        counts[tile_index(tile)] += 1
    if state.get("drawn_tile") is not None:
        counts[tile_index(state["drawn_tile"])] += 1
    observer = int(state["information"]["observer"])
    fixed_melds = len(state["players"][observer]["melds"])
    shanten = json.loads(calculate_shanten_json(canonical_json(counts), fixed_melds))
    return int(shanten["minimum"]) == 0


def _next_round(current: dict[str, Any], result: HandResult) -> dict[str, Any] | None:
    scores = result.final_scores
    if any(score < 0 for score in scores):
        return None

    dealer = int(current["dealer"])
    won = result.result_type in {"ron", "tsumo"}
    if won:
        dealer_continues = dealer in result.winners
    elif result.result_type == "exhaustive":
        if result.dealer_tenpai is None:
            raise RuntimeError("exhaustive draw is missing dealer tenpai state")
        dealer_continues = result.dealer_tenpai
    else:
        dealer_continues = True

    wind = str(current["wind"])
    hand_number = int(current["hand_number"])
    final_scheduled = wind == "south" and hand_number == 4
    extension_round = wind in {"west", "north"}
    top_score = max(scores)
    if extension_round and top_score >= _TARGET_SCORE:
        return None
    if wind == "west" and hand_number == 4:
        return None
    dealer_is_top = scores[dealer] == top_score
    if (
        final_scheduled
        and dealer_continues
        and dealer_is_top
        and scores[dealer] >= _TARGET_SCORE
        and (won or result.result_type == "exhaustive")
    ):
        return None

    terminal_round = result.round_state
    riichi_sticks = int(terminal_round["riichi_sticks"])
    if dealer_continues:
        return {
            "wind": wind,
            "hand_number": hand_number,
            "honba": int(current["honba"]) + 1,
            "riichi_sticks": 0 if won else riichi_sticks,
            "dealer": dealer,
        }

    if final_scheduled and top_score >= _TARGET_SCORE:
        return None
    next_wind, next_hand = _advance_hand(wind, hand_number)
    return {
        "wind": next_wind,
        "hand_number": next_hand,
        "honba": 0 if won else int(current["honba"]) + 1,
        "riichi_sticks": 0 if won else riichi_sticks,
        "dealer": (dealer + 1) % 4,
    }


def _advance_hand(wind: str, hand_number: int) -> tuple[str, int]:
    if hand_number < 4:
        return wind, hand_number + 1
    index = _WIND_ORDER.index(wind)
    if index + 1 >= len(_WIND_ORDER):
        raise RuntimeError("cannot advance beyond north 4")
    return _WIND_ORDER[index + 1], 1


def _ranks(scores: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    order = sorted(range(4), key=lambda seat: (-scores[seat], seat))
    ranks = [0] * 4
    for rank, seat in enumerate(order, start=1):
        ranks[seat] = rank
    return tuple(ranks)  # type: ignore[return-value]


def _summarize_model(matches: list[MatchResult], label: str) -> dict[str, Any]:
    ranks: list[int] = []
    deltas: list[int] = []
    wins = 0
    deal_ins = 0
    seat_hand_exposures = 0
    furo_hand_exposures = 0
    open_melds = 0
    hand_score_deltas: list[int] = []
    win_points: list[int] = []
    deal_in_points: list[int] = []
    actions: Counter[str] = Counter()
    for match in matches:
        seats = [seat for seat, seat_label in enumerate(match.pattern) if seat_label == label]
        ranks.extend(match.ranks[seat] for seat in seats)
        deltas.extend(match.final_scores[seat] - _STARTING_SCORE for seat in seats)
        for hand in match.hand_results:
            seat_hand_exposures += len(seats)
            wins += sum(seat in hand.winners for seat in seats)
            deal_ins += sum(hand.loser == seat for seat in seats)
            for seat in seats:
                delta = hand.final_scores[seat] - hand.initial_scores[seat]
                hand_score_deltas.append(delta)
                if seat in hand.winners:
                    win_points.append(max(0, delta))
                if hand.loser == seat:
                    deal_in_points.append(max(0, -delta))
                meld_count = hand.open_meld_counts[seat]
                open_melds += meld_count
                furo_hand_exposures += int(meld_count > 0)
        actions.update(match.action_counts[label])
    exposures = len(ranks)
    return {
        "seat_match_results": exposures,
        "seat_hand_results": seat_hand_exposures,
        "furo_rate_per_seat_hand": furo_hand_exposures / seat_hand_exposures,
        "average_open_melds_per_seat_hand": open_melds / seat_hand_exposures,
        "average_rank": float(np.mean(ranks)),
        "average_rank_95ci_unblocked": confidence_interval([float(value) for value in ranks]),
        "first_rate": ranks.count(1) / exposures,
        "second_rate": ranks.count(2) / exposures,
        "third_rate": ranks.count(3) / exposures,
        "fourth_rate": ranks.count(4) / exposures,
        "average_final_score": float(np.mean([delta + _STARTING_SCORE for delta in deltas])),
        "average_score_delta": float(np.mean(deltas)),
        "win_rate_per_seat_hand": wins / seat_hand_exposures,
        "deal_in_rate_per_seat_hand": deal_ins / seat_hand_exposures,
        "riichi_rate_per_seat_hand": actions["riichi"] / seat_hand_exposures,
        "average_win_points": float(np.mean(win_points)) if win_points else 0.0,
        "average_deal_in_points": (
            float(np.mean(deal_in_points)) if deal_in_points else 0.0
        ),
        "average_hand_score_delta": float(np.mean(hand_score_deltas)),
        "action_counts": dict(sorted(actions.items())),
    }


def _paired_summary(blocks: list[list[MatchResult]]) -> dict[str, Any]:
    score_differences: list[float] = []
    rank_differences: list[float] = []
    for block in blocks:
        by_label: dict[str, dict[str, list[float]]] = {
            "a": {"score": [], "rank": []},
            "b": {"score": [], "rank": []},
        }
        for match in block:
            for seat, label in enumerate(match.pattern):
                by_label[label]["score"].append(float(match.final_scores[seat] - _STARTING_SCORE))
                by_label[label]["rank"].append(float(match.ranks[seat]))
        score_differences.append(
            float(np.mean(by_label["a"]["score"]) - np.mean(by_label["b"]["score"]))
        )
        rank_differences.append(
            float(np.mean(by_label["a"]["rank"]) - np.mean(by_label["b"]["rank"]))
        )
    return {
        "blocks": len(blocks),
        "average_score_delta_difference_a_minus_b": float(np.mean(score_differences)),
        "average_score_delta_difference_a_minus_b_95ci": confidence_interval(score_differences),
        "average_rank_difference_a_minus_b": float(np.mean(rank_differences)),
        "average_rank_difference_a_minus_b_95ci": confidence_interval(rank_differences),
        "block_score_differences": score_differences,
        "block_rank_differences": rank_differences,
    }
