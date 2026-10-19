"""Observable-information belief sampling and short-horizon rollouts.

This module never accepts complete simulator state.  It determinizes only the
tiles that are unknown in a normal player observation, assigns plausible
concealed hands to the opponents, and treats the rest as a sampled live wall.
The resulting values are probability estimates, not exact mahjong odds.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Iterable

from mahjong_ai import calculate_shanten_json
from mahjong_ai.schema import canonical_json, tile_index


@dataclass(frozen=True)
class RolloutConfig:
    samples: int = 12
    depth: int = 3
    max_candidates: int = 4
    utility_scale: float = 0.65
    seed_salt: str = "belief-rollout-v1"


@dataclass(frozen=True)
class OpponentBelief:
    player: int
    tenpai_probability: float
    expected_hand_value: float
    safe_tiles: frozenset[int]
    wait_probabilities: tuple[float, ...] | None = None


@dataclass(frozen=True)
class SearchCandidate:
    action: dict[str, Any]
    successor: dict[str, Any]
    followup_discard: dict[str, Any] | None
    base_value: float


@dataclass(frozen=True)
class RolloutEstimate:
    expected_points: float
    utility_adjustment: float
    win_probability: float
    deal_in_probability: float
    opponent_win_probability: float
    samples: int
    depth: int


@dataclass(frozen=True)
class _Particle:
    hands: tuple[tuple[int, ...], ...]
    waits: tuple[frozenset[int], ...]
    wall: tuple[int, ...]


class BeliefRolloutSearch:
    """Estimate action outcomes from public information only."""

    def __init__(
        self,
        config: RolloutConfig = RolloutConfig(),
        belief_predictor: Callable[[dict[str, Any]], tuple[OpponentBelief, ...]] | None = None,
    ) -> None:
        if config.samples <= 0 or config.depth <= 0 or config.max_candidates <= 0:
            raise ValueError("rollout samples, depth and max_candidates must be positive")
        self.config = config
        self.belief_predictor = belief_predictor

    def evaluate(
        self, state: dict[str, Any], candidates: list[SearchCandidate]
    ) -> dict[str, RolloutEstimate]:
        self._validate_observation(state)
        if not candidates:
            return {}
        selected = sorted(candidates, key=lambda item: item.base_value, reverse=True)[
            : self.config.max_candidates
        ]
        observer = int(state["information"]["observer"])
        beliefs = self.opponent_beliefs(state)
        seed = int.from_bytes(
            hashlib.sha256(
                (self.config.seed_salt + canonical_json(state)).encode("utf-8")
            ).digest()[:8],
            "big",
        )
        sampler = random.Random(seed)
        particles = [self._sample_particle(state, beliefs, sampler) for _ in range(self.config.samples)]

        estimates: dict[str, RolloutEstimate] = {}
        for candidate in selected:
            wins = deal_ins = opponent_wins = 0
            points = 0.0
            for particle in particles:
                outcome, delta = self._rollout(
                    state,
                    candidate,
                    particle,
                    beliefs,
                )
                points += delta
                wins += int(outcome == "win")
                deal_ins += int(outcome == "deal_in")
                opponent_wins += int(outcome == "opponent_win")
            count = len(particles)
            expected = points / count
            estimates[canonical_json(candidate.action)] = RolloutEstimate(
                expected_points=expected,
                utility_adjustment=expected * self.config.utility_scale,
                win_probability=wins / count,
                deal_in_probability=deal_ins / count,
                opponent_win_probability=opponent_wins / count,
                samples=count,
                depth=self.config.depth,
            )
        return estimates

    def opponent_beliefs(self, state: dict[str, Any]) -> tuple[OpponentBelief, ...]:
        if self.belief_predictor is not None:
            predicted = self.belief_predictor(state)
            if len(predicted) != 4 or any(item.player != player for player, item in enumerate(predicted)):
                raise ValueError("belief predictor must return players 0..3 in seat order")
            return predicted
        observer = int(state["information"]["observer"])
        round_state = state.get("round") or {}
        dealer = int(round_state.get("dealer", 0))
        own_turns = float(state.get("turn", 0)) / 4.0
        beliefs: list[OpponentBelief] = []
        for player, public in enumerate(state["players"]):
            safe = frozenset(tile_index(item["tile"]) for item in public["discards"])
            if player == observer:
                beliefs.append(OpponentBelief(player, 0.0, 0.0, safe))
                continue
            melds = len(public["melds"])
            riichi = public["riichi"] in {
                "declared",
                "double_declared",
                "accepted",
                "double_accepted",
            }
            if riichi:
                tenpai = 0.995
                value = 7_000.0
            else:
                logit = -4.2 + 0.34 * own_turns + 0.62 * melds
                tenpai = min(0.98, 1.0 / (1.0 + math.exp(-logit)))
                value = 3_000.0 + 850.0 * melds
            if player == dealer:
                value *= 1.35
            value += 300.0 * float(round_state.get("honba", 0))
            beliefs.append(OpponentBelief(player, tenpai, value, safe))
        return tuple(beliefs)

    def _sample_particle(
        self,
        state: dict[str, Any],
        beliefs: tuple[OpponentBelief, ...],
        rng: random.Random,
    ) -> _Particle:
        observer = int(state["information"]["observer"])
        visible = _visible_counts(state)
        pool = [4 - count for count in visible]
        if any(count < 0 for count in pool):
            raise ValueError("observable state contains more than four copies of a tile")

        hands: list[tuple[int, ...]] = [tuple([0] * 34) for _ in range(4)]
        waits: list[frozenset[int]] = [frozenset() for _ in range(4)]
        order = [player for player in range(4) if player != observer]
        rng.shuffle(order)
        for player in order:
            public = state["players"][player]
            fixed_melds = len(public["melds"])
            size = max(1, 13 - 3 * fixed_melds)
            belief = beliefs[player]
            if rng.random() < belief.tenpai_probability:
                target = 0
            else:
                target = 1 if state.get("turn", 0) >= 32 else 2
            hand, pool = _conditioned_hand_with_wait_prior(
                pool, size, fixed_melds, target, _tile_weights(public),
                belief.wait_probabilities, rng,
            )
            hands[player] = hand
            if _shanten(hand, fixed_melds) == 0:
                waits[player] = frozenset(_winning_tiles(hand, fixed_melds))

        remaining = [tile for tile, count in enumerate(pool) for _ in range(count)]
        rng.shuffle(remaining)
        live_count = min(int(state.get("remaining_tiles", 0)), len(remaining))
        wall = tuple(remaining[:live_count])
        return _Particle(tuple(hands), tuple(waits), wall)

    def _rollout(
        self,
        state: dict[str, Any],
        candidate: SearchCandidate,
        particle: _Particle,
        beliefs: tuple[OpponentBelief, ...],
    ) -> tuple[str, float]:
        observer = int(state["information"]["observer"])
        successor = candidate.successor
        hand = list(_observer_counts(successor))
        fixed_melds = len(successor["players"][observer]["melds"])
        initial_shanten = _shanten(tuple(hand), fixed_melds)
        wall = list(particle.wall)

        first_discard = _action_discard(candidate.action, candidate.followup_discard)
        if first_discard is not None:
            winners = _ron_winners(first_discard, observer, particle, beliefs)
            if winners:
                return "deal_in", -sum(beliefs[player].expected_hand_value for player in winners)

        turn_order = [((observer + offset) % 4) for offset in range(1, 4)]
        current_shanten = initial_shanten
        for _ in range(self.config.depth):
            for player in turn_order:
                if not wall:
                    break
                opponent_draw = wall.pop()
                if opponent_draw in particle.waits[player]:
                    return "opponent_win", -beliefs[player].expected_hand_value / 3.0
            if not wall:
                break
            drawn = wall.pop()
            hand[drawn] += 1
            if _shanten(tuple(hand), fixed_melds) == -1:
                return "win", _own_win_value(state, candidate, observer)
            discarded, after = _best_discard(tuple(hand), fixed_melds)
            hand = list(after)
            current_shanten = _shanten(after, fixed_melds)
            winners = _ron_winners(discarded, observer, particle, beliefs)
            if winners:
                return "deal_in", -sum(beliefs[player].expected_hand_value for player in winners)

        progress = max(0, initial_shanten - current_shanten) * 250.0
        return "no_result", progress

    @staticmethod
    def _validate_observation(state: dict[str, Any]) -> None:
        information = state.get("information") or {}
        if information.get("opponent_hands_visible", False):
            raise ValueError("belief search rejects visible opponent hands")
        if information.get("wall_order_visible", False):
            raise ValueError("belief search rejects visible wall order")
        if "private_hands" in state or "wall" in state:
            raise ValueError("belief search accepts observable state only")


def _visible_counts(state: dict[str, Any]) -> list[int]:
    counts = [0] * 34

    def add(tiles: Iterable[dict[str, Any]]) -> None:
        for tile in tiles:
            counts[tile_index(tile)] += 1

    add(state.get("hand") or [])
    if state.get("drawn_tile") is not None:
        add([state["drawn_tile"]])
    add(state.get("dora_indicators") or [])
    for public in state["players"]:
        add(item["tile"] for item in public["discards"] if item.get("called_by") is None)
        for meld in public["melds"]:
            add(meld["tiles"])
    return counts


def _observer_counts(state: dict[str, Any]) -> tuple[int, ...]:
    counts = [0] * 34
    for tile in state.get("hand") or []:
        counts[tile_index(tile)] += 1
    if state.get("drawn_tile") is not None:
        counts[tile_index(state["drawn_tile"])] += 1
    return tuple(counts)


def _tile_weights(public: dict[str, Any]) -> tuple[float, ...]:
    weights = [1.0] * 34
    discarded = {tile_index(item["tile"]) for item in public["discards"]}
    for tile in discarded:
        weights[tile] *= 0.45
    suit_melds = [0, 0, 0]
    for meld in public["melds"]:
        for tile in meld["tiles"]:
            index = tile_index(tile)
            if index < 27:
                suit_melds[index // 9] += 1
    if suit_melds and max(suit_melds) >= 6:
        preferred = max(range(3), key=suit_melds.__getitem__)
        for index in range(27):
            weights[index] *= 1.8 if index // 9 == preferred else 0.38
        for index in range(27, 34):
            weights[index] *= 1.25
    return tuple(weights)


def _conditioned_hand(
    pool: list[int],
    size: int,
    fixed_melds: int,
    target_shanten: int,
    weights: tuple[float, ...],
    rng: random.Random,
) -> tuple[tuple[int, ...], list[int]]:
    available = list(pool)
    hand = [0] * 34
    for _ in range(size):
        tile = _weighted_pick(available, weights, rng)
        available[tile] -= 1
        hand[tile] += 1

    current = _shanten(tuple(hand), fixed_melds)
    for _ in range(48):
        if current == target_shanten:
            break
        removable = [tile for tile, count in enumerate(hand) if count]
        additions = [tile for tile, count in enumerate(available) if count]
        if not removable or not additions:
            break
        best: tuple[int, int, int] | None = None
        for _ in range(10):
            removed = rng.choice(removable)
            added = rng.choice(additions)
            if removed == added:
                continue
            proposal = list(hand)
            proposal[removed] -= 1
            proposal[added] += 1
            shanten = _shanten(tuple(proposal), fixed_melds)
            distance = abs(shanten - target_shanten)
            if best is None or distance < best[0]:
                best = (distance, removed, added)
        if best is None:
            break
        new_distance, removed, added = best
        old_distance = abs(current - target_shanten)
        if new_distance > old_distance or (new_distance == old_distance and rng.random() > 0.15):
            continue
        hand[removed] -= 1
        hand[added] += 1
        available[removed] += 1
        available[added] -= 1
        current = _shanten(tuple(hand), fixed_melds)
    return tuple(hand), available


def _conditioned_hand_with_wait_prior(
    pool: list[int],
    size: int,
    fixed_melds: int,
    target_shanten: int,
    weights: tuple[float, ...],
    wait_probabilities: tuple[float, ...] | None,
    rng: random.Random,
) -> tuple[tuple[int, ...], list[int]]:
    """Choose among sampled hands using learned wait probabilities when available."""

    trials = 3 if wait_probabilities is not None and target_shanten == 0 else 1
    best: tuple[tuple[float, float], tuple[int, ...], list[int]] | None = None
    for _ in range(trials):
        hand, remaining = _conditioned_hand(
            pool, size, fixed_melds, target_shanten, weights, rng
        )
        shanten = _shanten(hand, fixed_melds)
        wait_score = 0.0
        if wait_probabilities is not None and shanten == 0:
            waits = _winning_tiles(hand, fixed_melds)
            if waits:
                wait_score = sum(wait_probabilities[tile] for tile in waits) / len(waits)
        key = (-float(abs(shanten - target_shanten)), wait_score)
        if best is None or key > best[0]:
            best = (key, hand, remaining)
    if best is None:
        raise ValueError("failed to sample a concealed hand")
    return best[1], best[2]


def _weighted_pick(pool: list[int], weights: tuple[float, ...], rng: random.Random) -> int:
    total = sum(count * weights[index] for index, count in enumerate(pool))
    if total <= 0:
        raise ValueError("hidden tile pool is empty")
    threshold = rng.random() * total
    for index, count in enumerate(pool):
        threshold -= count * weights[index]
        if threshold <= 0 and count:
            return index
    return next(index for index, count in enumerate(pool) if count)


@lru_cache(maxsize=250_000)
def _shanten(counts: tuple[int, ...], fixed_melds: int) -> int:
    result = json.loads(calculate_shanten_json(json.dumps(counts), fixed_melds))
    return int(result["minimum"])


def _winning_tiles(hand: tuple[int, ...], fixed_melds: int) -> list[int]:
    output: list[int] = []
    for tile in range(34):
        if hand[tile] >= 4:
            continue
        candidate = list(hand)
        candidate[tile] += 1
        if _shanten(tuple(candidate), fixed_melds) == -1:
            output.append(tile)
    return output


def _best_discard(hand: tuple[int, ...], fixed_melds: int) -> tuple[int, tuple[int, ...]]:
    best: tuple[tuple[float, ...], int, tuple[int, ...]] | None = None
    for tile, count in enumerate(hand):
        if not count:
            continue
        after = list(hand)
        after[tile] -= 1
        result = tuple(after)
        key = (float(_shanten(result, fixed_melds)), -_shape_score(result), float(tile))
        if best is None or key < best[0]:
            best = (key, tile, result)
    if best is None:
        raise ValueError("cannot discard from an empty hand")
    return best[1], best[2]


def _shape_score(hand: tuple[int, ...]) -> float:
    score = 0.0
    for tile, count in enumerate(hand):
        score += min(count, 2) * 1.4 + max(0, count - 2) * 0.8
        if tile < 27:
            rank = tile % 9
            if rank < 8:
                score += min(count, hand[tile + 1]) * 0.7
            if rank < 7:
                score += min(count, hand[tile + 2]) * 0.35
    return score


def _action_discard(
    action: dict[str, Any], followup: dict[str, Any] | None
) -> int | None:
    if action["type"] == "discard":
        return tile_index(action["tile"])
    if action["type"] == "riichi":
        return tile_index(action["discard"])
    if followup is not None:
        return tile_index(followup["tile"])
    return None


def _ron_winners(
    tile: int,
    observer: int,
    particle: _Particle,
    beliefs: tuple[OpponentBelief, ...],
) -> list[int]:
    return [
        player
        for player in range(4)
        if player != observer
        and tile not in beliefs[player].safe_tiles
        and tile in particle.waits[player]
    ]


def _own_win_value(
    state: dict[str, Any], candidate: SearchCandidate, observer: int
) -> float:
    successor = candidate.successor
    open_melds = sum(
        meld["kind"] != "ankan" for meld in successor["players"][observer]["melds"]
    )
    riichi = candidate.action["type"] == "riichi" or successor["players"][observer][
        "riichi"
    ] in {"declared", "double_declared", "accepted", "double_accepted"}
    value = 7_000.0 if riichi else (5_200.0 if open_melds == 0 else 3_900.0)
    round_state = state.get("round") or {}
    if int(round_state.get("dealer", -1)) == observer:
        value *= 1.35
    value += 300.0 * float(round_state.get("honba", 0))
    value += 1_000.0 * float(round_state.get("riichi_sticks", 0))
    return value
