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
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Iterable, Protocol

import numpy as np
import torch
from mahjong_ai import calculate_shanten_json, score_hand_json
from mahjong_ai.schema import TILE_TYPES, canonical_json, tile_index

from .encoder import FeatureEncoder
from .model import MahjongNetwork


@dataclass(frozen=True)
class RolloutConfig:
    samples: int = 12
    depth: int = 3
    max_candidates: int = 4
    utility_scale: float = 0.65
    seed_salt: str = "belief-rollout-v1"
    likelihood_floor: float = 1e-5
    imitation_continuation: bool = False

    @classmethod
    def high_accuracy(cls) -> "RolloutConfig":
        """Offline-quality preset; intentionally expensive and heat intensive."""

        return cls(
            samples=48,
            depth=6,
            max_candidates=8,
            utility_scale=0.65,
            seed_salt="belief-rollout-high-accuracy-v1",
            likelihood_floor=1e-5,
            imitation_continuation=True,
        )


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
    horizontal_transfer_probability: float
    exhaustive_draw_probability: float
    unresolved_probability: float
    expected_win_points: float
    expected_deal_in_loss: float
    deal_in_probability_by_opponent: tuple[float, ...]
    expected_deal_in_loss_by_opponent: tuple[float, ...]
    samples: int
    effective_samples: float
    depth: int


@dataclass(frozen=True)
class _Particle:
    hands: tuple[tuple[int, ...], ...]
    waits: tuple[frozenset[int], ...]
    wall: tuple[int, ...]
    hand_reds: tuple[frozenset[int], ...] = (
        frozenset(),
        frozenset(),
        frozenset(),
        frozenset(),
    )
    wall_reds: tuple[bool, ...] = ()
    dead_wall: tuple[int, ...] = ()
    dead_wall_reds: tuple[bool, ...] = ()


@dataclass(frozen=True)
class ParticlePosterior:
    """Immutable weighted hidden-world population for candidate comparison."""

    particles: tuple[_Particle, ...]
    weights: tuple[float, ...]
    event_sequence: int
    effective_samples: float

    def __post_init__(self) -> None:
        if not self.particles or len(self.particles) != len(self.weights):
            raise ValueError("posterior particles and weights must be non-empty and aligned")
        if any(not math.isfinite(weight) or weight < 0.0 for weight in self.weights):
            raise ValueError("posterior weights must be finite and non-negative")
        if not math.isclose(sum(self.weights), 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("posterior weights must sum to one")


@dataclass(frozen=True)
class BeliefUpdateDiagnostics:
    event_sequence: int
    effective_samples: float
    entropy: float
    resampled: bool
    impossible_particles: int
    recovered: bool


@dataclass(frozen=True)
class _Outcome:
    kind: str
    point_delta: float
    winners: tuple[int, ...] = ()
    winner_values: tuple[tuple[int, float], ...] = ()


class DiscardLikelihood(Protocol):
    """Approximate P(observed discard | sampled concealed hand, public state)."""

    def __call__(
        self,
        state: dict[str, Any],
        player: int,
        concealed_hand: tuple[int, ...],
        discard: dict[str, Any],
    ) -> float: ...


class ImitationDiscardLikelihood:
    """Use the observable policy network as an opponent-action likelihood.

    A particle hand is exposed only as the acting opponent's own hand in a
    temporary observation.  No other sampled hand or sampled wall tile enters
    the policy model.  For a hand-built posterior we only score the latest
    observed discard per opponent; applying all historical actions to the
    current hand would count mutually inconsistent hand snapshots.
    """

    def __init__(self, model: MahjongNetwork, *, device: str = "cpu") -> None:
        self.model = model
        self.device = device
        self.encoder = FeatureEncoder()

    def __call__(
        self,
        state: dict[str, Any],
        player: int,
        concealed_hand: tuple[int, ...],
        discard: dict[str, Any],
    ) -> float:
        observed_tile = tile_index(discard["tile"])
        reconstructed = list(concealed_hand)
        reconstructed[observed_tile] += 1
        if reconstructed[observed_tile] > 4:
            return 0.0

        hypothetical = deepcopy(state)
        public_discards = hypothetical["players"][player]["discards"]
        if public_discards:
            latest = public_discards[-1]
            if tile_index(latest["tile"]) == observed_tile:
                # ``state`` may be the post-event observation. The action
                # being scored must not already be visible in its own input.
                public_discards.pop()
        history = hypothetical.get("event_history") or []
        if history and history[-1].get("type") == "discard":
            latest_event = history[-1]
            if (
                int(latest_event.get("player", -1)) == player
                and tile_index(latest_event["tile"]) == observed_tile
            ):
                history.pop()
        hypothetical["information"] = {
            "observer": player,
            "observer_hand_visible": True,
            "opponent_hands_visible": False,
            "wall_order_visible": False,
        }
        hypothetical["hand"] = [
            {"kind": TILE_TYPES[index], "red": False}
            for index, count in enumerate(reconstructed)
            for _ in range(count)
        ]
        hypothetical["drawn_tile"] = None
        hypothetical["phase"] = {"phase": "awaiting_discard", "player": player}
        legal = [
            {
                "type": "discard",
                "tile": {"kind": TILE_TYPES[index], "red": False},
                "tsumogiri": False,
            }
            for index, count in enumerate(reconstructed)
            if count
        ]
        encoded = self.encoder.encode(hypothetical, legal)
        planes = torch.from_numpy(encoded.planes).unsqueeze(0).to(self.device)
        actions = torch.from_numpy(encoded.action_features).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits = self.model(planes, actions)["policy_logits"][0]
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
        matching = [
            index
            for index, action in enumerate(legal)
            if tile_index(action["tile"]) == observed_tile
        ]
        return float(np.asarray(probabilities)[matching].sum()) if matching else 0.0

    def choose_discard(
        self,
        state: dict[str, Any],
        player: int,
        concealed_hand: tuple[int, ...],
        red_tiles: frozenset[int],
    ) -> tuple[int, bool]:
        """Choose a rollout discard in one batched imitation-policy call."""

        hypothetical = deepcopy(state)
        hypothetical["information"] = {
            "observer": player,
            "observer_hand_visible": True,
            "opponent_hands_visible": False,
            "wall_order_visible": False,
        }
        physical: list[dict[str, Any]] = []
        legal: list[dict[str, Any]] = []
        for index, count in enumerate(concealed_hand):
            red_count = int(index in red_tiles)
            physical.extend(
                {"kind": TILE_TYPES[index], "red": copy < red_count}
                for copy in range(count)
            )
            if count - red_count > 0:
                legal.append(
                    {
                        "type": "discard",
                        "tile": {"kind": TILE_TYPES[index], "red": False},
                        "tsumogiri": False,
                    }
                )
            if red_count:
                legal.append(
                    {
                        "type": "discard",
                        "tile": {"kind": TILE_TYPES[index], "red": True},
                        "tsumogiri": False,
                    }
                )
        if not legal:
            raise ValueError("rollout policy cannot discard from an empty hand")
        hypothetical["hand"] = physical
        hypothetical["drawn_tile"] = None
        hypothetical["phase"] = {"phase": "awaiting_discard", "player": player}
        encoded = self.encoder.encode(hypothetical, legal)
        with torch.inference_mode():
            logits = self.model(
                torch.from_numpy(encoded.planes).unsqueeze(0).to(self.device),
                torch.from_numpy(encoded.action_features).unsqueeze(0).to(self.device),
            )["policy_logits"][0]
        action = legal[int(logits.argmax())]
        return tile_index(action["tile"]), bool(action["tile"].get("red", False))


class StatefulParticleFilter:
    """Sequential P(H|O) approximation driven only by public observations.

    Hidden opponent draws are sampled inside each particle.  Known observer
    draws and public melds condition the population by physical consistency.
    Opponent discards additionally use the imitation-policy likelihood.  The
    filter never accepts complete simulator state or a true wall order.
    """

    def __init__(
        self,
        state: dict[str, Any],
        *,
        config: RolloutConfig = RolloutConfig(),
        belief_predictor: Callable[[dict[str, Any]], tuple[OpponentBelief, ...]] | None = None,
        action_likelihood: DiscardLikelihood | None = None,
        resample_ess_ratio: float = 0.5,
    ) -> None:
        if not 0.0 < resample_ess_ratio <= 1.0:
            raise ValueError("resample_ess_ratio must be in (0, 1]")
        BeliefRolloutSearch._validate_observation(state)
        self.config = config
        self.belief_predictor = belief_predictor
        self.action_likelihood = action_likelihood
        self.resample_ess_ratio = resample_ess_ratio
        self.observer = int(state["information"]["observer"])
        seed_material = config.seed_salt + ":stateful:" + canonical_json(state)
        self._rng = random.Random(
            int.from_bytes(hashlib.sha256(seed_material.encode()).digest()[:8], "big")
        )
        self._particles: list[_Particle] = []
        self._weights: list[float] = []
        self._pending_draws: list[list[int | None]] = []
        self._event_sequence = int(state.get("next_event_sequence", 0)) - 1
        self._initialize(state)

    def _initialize(self, state: dict[str, Any]) -> None:
        search = BeliefRolloutSearch(
            self.config,
            belief_predictor=self.belief_predictor,
            action_likelihood=self.action_likelihood,
        )
        beliefs = search.opponent_beliefs(state)
        self._particles = [
            search._sample_particle(state, beliefs, self._rng)
            for _ in range(self.config.samples)
        ]
        self._weights = search._posterior_weights(state, self._particles)
        self._pending_draws = [[None] * 4 for _ in self._particles]

    def posterior(self) -> ParticlePosterior:
        effective = _effective_sample_size(self._weights)
        return ParticlePosterior(
            tuple(self._particles),
            tuple(self._weights),
            self._event_sequence,
            effective,
        )

    def observe(
        self, state: dict[str, Any], event: dict[str, Any]
    ) -> BeliefUpdateDiagnostics:
        BeliefRolloutSearch._validate_observation(state)
        if int(state["information"]["observer"]) != self.observer:
            raise ValueError("particle filter observer cannot change")
        sequence = int(event.get("sequence", self._event_sequence + 1))
        if sequence != self._event_sequence + 1:
            raise ValueError("belief events must be contiguous")
        event_type = str(event.get("type"))
        updated: list[_Particle] = []
        raw_weights: list[float] = []
        pending: list[list[int | None]] = []
        impossible = 0
        for index, (particle, prior) in enumerate(
            zip(self._particles, self._weights, strict=True)
        ):
            next_particle, likelihood, next_pending = self._transition_particle(
                particle, self._pending_draws[index], state, event
            )
            if next_particle is None or likelihood <= 0.0:
                impossible += 1
                continue
            updated.append(next_particle)
            raw_weights.append(prior * likelihood)
            pending.append(next_pending)

        recovered = False
        if not updated or sum(raw_weights) <= 0.0:
            self._initialize(state)
            recovered = True
        else:
            total = sum(raw_weights)
            self._particles = updated
            self._weights = [weight / total for weight in raw_weights]
            self._pending_draws = pending
        self._event_sequence = sequence

        effective = _effective_sample_size(self._weights)
        resampled = False
        if not recovered and effective < self.resample_ess_ratio * self.config.samples:
            self._systematic_resample()
            effective = _effective_sample_size(self._weights)
            resampled = True
        entropy = -sum(
            weight * math.log(weight) for weight in self._weights if weight > 0.0
        )
        return BeliefUpdateDiagnostics(
            event_sequence=sequence,
            effective_samples=effective,
            entropy=entropy,
            resampled=resampled,
            impossible_particles=impossible,
            recovered=recovered,
        )

    def _transition_particle(
        self,
        particle: _Particle,
        pending_draws: list[int | None],
        state: dict[str, Any],
        event: dict[str, Any],
    ) -> tuple[_Particle | None, float, list[int | None]]:
        hands = [list(hand) for hand in particle.hands]
        wall = list(particle.wall)
        hand_reds = [set(tiles) for tiles in particle.hand_reds]
        wall_reds = (
            list(particle.wall_reds)
            if len(particle.wall_reds) == len(wall)
            else [False] * len(wall)
        )
        dead_wall = list(particle.dead_wall)
        dead_wall_reds = (
            list(particle.dead_wall_reds)
            if len(particle.dead_wall_reds) == len(dead_wall)
            else [False] * len(dead_wall)
        )
        pending = list(pending_draws)
        likelihood = 1.0
        event_type = str(event.get("type"))
        player_value = event.get("player")
        player = int(player_value) if player_value is not None else None

        if event_type == "draw" and player is not None:
            tile = event.get("tile")
            if player == self.observer:
                if tile is None:
                    return None, 0.0, pending
                tile_value = tile_index(tile)
                red = bool(tile.get("red", False))
                matching = [
                    index
                    for index, candidate in enumerate(wall)
                    if candidate == tile_value and wall_reds[index] == red
                ]
                if not matching:
                    return None, 0.0, pending
                likelihood *= len(matching) / len(wall)
                chosen = matching[0]
                wall.pop(chosen)
                wall_reds.pop(chosen)
            else:
                if tile is not None:
                    raise ValueError("opponent draw tile leaked into public belief update")
                if not wall:
                    return None, 0.0, pending
                drawn_index = self._rng.randrange(len(wall))
                drawn = wall.pop(drawn_index)
                drawn_red = wall_reds.pop(drawn_index)
                hands[player][drawn] += 1
                if drawn_red:
                    hand_reds[player].add(drawn)
                pending[player] = drawn
        elif event_type == "discard" and player is not None:
            discarded = tile_index(event["tile"])
            if player != self.observer:
                if hands[player][discarded] <= 0:
                    return None, 0.0, pending
                if (
                    event.get("tsumogiri_known", True)
                    and bool(event.get("tsumogiri", False))
                    and pending[player] != discarded
                ):
                    return None, 0.0, pending
                discarded_red = bool(event["tile"].get("red", False))
                if discarded_red and discarded not in hand_reds[player]:
                    return None, 0.0, pending
                if (
                    not discarded_red
                    and discarded in hand_reds[player]
                    and hands[player][discarded] == 1
                ):
                    return None, 0.0, pending
                hands[player][discarded] -= 1
                if discarded_red:
                    hand_reds[player].remove(discarded)
                if self.action_likelihood is not None:
                    action_probability = float(
                        self.action_likelihood(
                            state, player, tuple(hands[player]), event
                        )
                    )
                    if not math.isfinite(action_probability) or action_probability < 0.0:
                        raise ValueError("discard likelihood must be finite and non-negative")
                    likelihood *= max(action_probability, self.config.likelihood_floor)
                pending[player] = None
        elif event_type in {"chi", "pon", "kan"} and player is not None:
            if player != self.observer:
                meld = event["meld"]
                meld_kind = str(meld["kind"])
                consumed_tiles = (
                    [meld["tiles"][-1]]
                    if meld_kind == "shouminkan"
                    else list(meld["tiles"])
                )
                if meld_kind != "shouminkan" and meld.get("called_tile") is not None:
                    called_tile = meld["called_tile"]
                    consumed_tiles.remove(called_tile)
                for consumed_tile in consumed_tiles:
                    tile_value = tile_index(consumed_tile)
                    if hands[player][tile_value] <= 0:
                        return None, 0.0, pending
                    red = bool(consumed_tile.get("red", False))
                    if red and tile_value not in hand_reds[player]:
                        return None, 0.0, pending
                    if (
                        not red
                        and tile_value in hand_reds[player]
                        and hands[player][tile_value] == 1
                    ):
                        return None, 0.0, pending
                    hands[player][tile_value] -= 1
                    if red:
                        hand_reds[player].remove(tile_value)
                pending[player] = None
        elif event_type == "dora_reveal":
            indicator = event.get(
                "tile", event.get("indicator", event.get("dora_indicator"))
            )
            if indicator is not None:
                tile_value = tile_index(indicator)
                red = bool(indicator.get("red", False))
                matching = [
                    index
                    for index, candidate in enumerate(dead_wall)
                    if candidate == tile_value and dead_wall_reds[index] == red
                ]
                if not matching:
                    return None, 0.0, pending
                likelihood *= len(matching) / len(dead_wall)
                chosen = matching[0]
                dead_wall.pop(chosen)
                dead_wall_reds.pop(chosen)

        waits = list(particle.waits)
        for seat in range(4):
            if seat == self.observer:
                continue
            fixed_melds = len(state["players"][seat]["melds"])
            concealed = tuple(hands[seat])
            waits[seat] = frozenset(
                _winning_tiles(concealed, fixed_melds)
                if _shanten(concealed, fixed_melds) == 0
                else ()
            )
        return (
            _Particle(
                tuple(tuple(hand) for hand in hands),
                tuple(waits),
                tuple(wall),
                tuple(frozenset(tiles) for tiles in hand_reds),
                tuple(wall_reds),
                tuple(dead_wall),
                tuple(dead_wall_reds),
            ),
            likelihood,
            pending,
        )

    def _systematic_resample(self) -> None:
        count = self.config.samples
        cumulative: list[float] = []
        running = 0.0
        for weight in self._weights:
            running += weight
            cumulative.append(running)
        start = self._rng.random() / count
        indexes: list[int] = []
        cursor = 0
        for offset in range(count):
            target = start + offset / count
            while cursor + 1 < len(cumulative) and target > cumulative[cursor]:
                cursor += 1
            indexes.append(cursor)
        self._particles = [self._particles[index] for index in indexes]
        self._pending_draws = [list(self._pending_draws[index]) for index in indexes]
        self._weights = [1.0 / count] * count


class BeliefRolloutSearch:
    """Estimate action outcomes from public information only."""

    def __init__(
        self,
        config: RolloutConfig = RolloutConfig(),
        belief_predictor: Callable[[dict[str, Any]], tuple[OpponentBelief, ...]] | None = None,
        action_likelihood: DiscardLikelihood | None = None,
    ) -> None:
        if config.samples <= 0 or config.depth <= 0 or config.max_candidates <= 0:
            raise ValueError("rollout samples, depth and max_candidates must be positive")
        if not 0.0 < config.likelihood_floor <= 1.0:
            raise ValueError("likelihood_floor must be in (0, 1]")
        self.config = config
        self.belief_predictor = belief_predictor
        self.action_likelihood = action_likelihood

    def evaluate(
        self,
        state: dict[str, Any],
        candidates: list[SearchCandidate],
        *,
        posterior: ParticlePosterior | None = None,
    ) -> dict[str, RolloutEstimate]:
        self._validate_observation(state)
        if not candidates:
            return {}
        selected = self._select_counterfactual_candidates(candidates)
        observer = int(state["information"]["observer"])
        beliefs = self.opponent_beliefs(state)
        seed = int.from_bytes(
            hashlib.sha256(
                (self.config.seed_salt + canonical_json(state)).encode("utf-8")
            ).digest()[:8],
            "big",
        )
        if posterior is None:
            sampler = random.Random(seed)
            particles = [
                self._sample_particle(state, beliefs, sampler)
                for _ in range(self.config.samples)
            ]
            weights = self._posterior_weights(state, particles)
            effective_samples = _effective_sample_size(weights)
        else:
            expected_sequence = int(state.get("next_event_sequence", 0)) - 1
            if posterior.event_sequence != expected_sequence:
                raise ValueError("particle posterior is stale for this observation")
            particles = list(posterior.particles)
            weights = list(posterior.weights)
            effective_samples = posterior.effective_samples

        estimates: dict[str, RolloutEstimate] = {}
        for candidate in selected:
            probabilities = {
                "win": 0.0,
                "deal_in": 0.0,
                "opponent_win": 0.0,
                "horizontal_transfer": 0.0,
                "exhaustive_draw": 0.0,
                "unresolved": 0.0,
            }
            deal_in_by_opponent = [0.0] * 4
            loss_by_opponent = [0.0] * 4
            win_points = deal_in_loss = 0.0
            points = 0.0
            for particle, weight in zip(particles, weights, strict=True):
                outcome = self._rollout(
                    state,
                    candidate,
                    particle,
                    beliefs,
                )
                points += weight * outcome.point_delta
                probabilities[outcome.kind] += weight
                if outcome.kind == "win":
                    win_points += weight * outcome.point_delta
                elif outcome.kind == "deal_in":
                    deal_in_loss += weight * -outcome.point_delta
                    for winner in outcome.winners:
                        deal_in_by_opponent[winner] += weight
                    for winner, loss in outcome.winner_values:
                        loss_by_opponent[winner] += weight * loss
            count = len(particles)
            expected = points
            expected_win_points = (
                win_points / probabilities["win"] if probabilities["win"] else 0.0
            )
            expected_deal_in_loss = (
                deal_in_loss / probabilities["deal_in"]
                if probabilities["deal_in"]
                else 0.0
            )
            conditional_losses = tuple(
                loss_by_opponent[player] / deal_in_by_opponent[player]
                if deal_in_by_opponent[player]
                else 0.0
                for player in range(4)
            )
            estimates[canonical_json(candidate.action)] = RolloutEstimate(
                expected_points=expected,
                utility_adjustment=expected * self.config.utility_scale,
                win_probability=probabilities["win"],
                deal_in_probability=probabilities["deal_in"],
                opponent_win_probability=probabilities["opponent_win"],
                horizontal_transfer_probability=probabilities["horizontal_transfer"],
                exhaustive_draw_probability=probabilities["exhaustive_draw"],
                unresolved_probability=probabilities["unresolved"],
                expected_win_points=expected_win_points,
                expected_deal_in_loss=expected_deal_in_loss,
                deal_in_probability_by_opponent=tuple(deal_in_by_opponent),
                expected_deal_in_loss_by_opponent=conditional_losses,
                samples=count,
                effective_samples=effective_samples,
                depth=self.config.depth,
            )
        return estimates

    def _select_counterfactual_candidates(
        self, candidates: list[SearchCandidate]
    ) -> list[SearchCandidate]:
        """Keep the strongest candidate and its strategic alternative together.

        Common random numbers only help a riichi/dama or call/pass decision if
        both alternatives reach rollout. The pairing is therefore performed
        before filling the remaining beam by base value.
        """

        ordered = sorted(candidates, key=lambda item: item.base_value, reverse=True)
        selected: list[SearchCandidate] = []
        selected_keys: set[str] = set()

        def add(candidate: SearchCandidate) -> None:
            key = canonical_json(candidate.action)
            if key not in selected_keys and len(selected) < self.config.max_candidates:
                selected.append(candidate)
                selected_keys.add(key)

        for candidate in ordered:
            if len(selected) >= self.config.max_candidates:
                break
            add(candidate)
            counterpart = _counterfactual_counterpart(candidate, ordered)
            if counterpart is not None:
                add(counterpart)
        return selected

    def _posterior_weights(
        self, state: dict[str, Any], particles: list[_Particle]
    ) -> list[float]:
        if not particles:
            raise ValueError("belief posterior needs at least one particle")
        if self.action_likelihood is None:
            return [1.0 / len(particles)] * len(particles)
        observer = int(state["information"]["observer"])
        log_weights: list[float] = []
        for particle in particles:
            log_weight = 0.0
            for player, public in enumerate(state["players"]):
                if player == observer or not public["discards"]:
                    continue
                likelihood = float(
                    self.action_likelihood(
                        state,
                        player,
                        particle.hands[player],
                        public["discards"][-1],
                    )
                )
                if not math.isfinite(likelihood) or likelihood < 0.0:
                    raise ValueError("discard likelihood must be finite and non-negative")
                log_weight += math.log(max(likelihood, self.config.likelihood_floor))
            log_weights.append(log_weight)
        maximum = max(log_weights)
        unnormalized = [math.exp(value - maximum) for value in log_weights]
        total = sum(unnormalized)
        if total <= 0.0 or not math.isfinite(total):
            return [1.0 / len(particles)] * len(particles)
        return [value / total for value in unnormalized]

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
            phase = state.get("phase") or {}
            if (
                phase.get("phase") == "awaiting_discard"
                and int(phase.get("player", -1)) == player
            ):
                size += 1
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
        hand_reds, remaining_reds = _assign_hidden_red_fives(
            state, hands, remaining, rng
        )
        return _Particle(
            tuple(hands),
            tuple(waits),
            wall,
            hand_reds,
            tuple(remaining_reds[:live_count]),
            tuple(remaining[live_count:]),
            tuple(remaining_reds[live_count:]),
        )

    def _rollout(
        self,
        state: dict[str, Any],
        candidate: SearchCandidate,
        particle: _Particle,
        beliefs: tuple[OpponentBelief, ...],
    ) -> _Outcome:
        observer = int(state["information"]["observer"])
        successor = candidate.successor
        hands = [list(values) for values in particle.hands]
        hands[observer] = list(_observer_counts(successor))
        hand_reds = [set(values) for values in particle.hand_reds]
        hand_reds[observer] = set(_observer_reds(successor))
        hand = hands[observer]
        current_waits = list(particle.waits)
        fixed_melds = len(successor["players"][observer]["melds"])
        initial_shanten = _shanten(tuple(hand), fixed_melds)
        current_waits[observer] = frozenset(
            _winning_tiles(tuple(hand), fixed_melds)
            if initial_shanten == 0
            else []
        )
        wall = list(particle.wall)
        wall_reds = (
            list(particle.wall_reds)
            if len(particle.wall_reds) == len(wall)
            else [False] * len(wall)
        )

        first_discard_tile = _action_discard_tile(
            candidate.action, candidate.followup_discard
        )
        if first_discard_tile is not None:
            winner_values = _ron_winner_values(
                successor,
                first_discard_tile,
                observer,
                current_waits,
                tuple(tuple(values) for values in hands),
                tuple(frozenset(values) for values in hand_reds),
                beliefs,
            )
            if winner_values:
                return _Outcome(
                    "deal_in",
                    -sum(winner_values.values()),
                    tuple(winner_values),
                    tuple(winner_values.items()),
                )

        phase = state.get("phase") or {}
        if candidate.action["type"] == "pass" and phase.get("phase") in {
            "awaiting_response", "awaiting_kan_response"
        }:
            source = int(phase.get("discarder", phase.get("declarer", observer)))
            first_draw_player = (source + 1) % 4
        else:
            first_draw_player = (observer + 1) % 4
        turn_order = [((first_draw_player + offset) % 4) for offset in range(4)]
        current_shanten = initial_shanten
        for _ in range(self.config.depth):
            for player in turn_order:
                if not wall:
                    break
                drawn = wall.pop()
                drawn_red = wall_reds.pop()
                winning_tile = {"kind": TILE_TYPES[drawn], "red": drawn_red}
                if drawn in current_waits[player]:
                    exact_value = _exact_hand_value(
                        successor,
                        player,
                        tuple(hands[player]),
                        frozenset(hand_reds[player]),
                        winning_tile,
                        method="tsumo",
                    )
                    if exact_value is not None:
                        if player == observer:
                            gain = _exact_win_gain(
                                successor,
                                player,
                                tuple(hands[player]),
                                frozenset(hand_reds[player]),
                                winning_tile,
                                method="tsumo",
                            )
                            return _Outcome("win", gain, (observer,))
                        return _Outcome(
                            "opponent_win", -exact_value, (player,)
                        )
                hands[player][drawn] += 1
                if drawn_red:
                    hand_reds[player].add(drawn)
                public = successor["players"][player]
                locked = public["riichi"] in {
                    "declared", "double_declared", "accepted", "double_accepted"
                }
                fixed = len(public["melds"])
                if locked:
                    discarded_index = drawn
                    after = list(hands[player])
                    after[drawn] -= 1
                    discard_red = drawn_red
                else:
                    chooser = (
                        getattr(self.action_likelihood, "choose_discard", None)
                        if self.config.imitation_continuation
                        else None
                    )
                    if chooser is not None:
                        discarded_index, discard_red = chooser(
                            successor,
                            player,
                            tuple(hands[player]),
                            frozenset(hand_reds[player]),
                        )
                        after = list(hands[player])
                        after[discarded_index] -= 1
                    else:
                        discarded_index, after_tuple = _best_discard(
                            tuple(hands[player]), fixed
                        )
                        after = list(after_tuple)
                        discard_red = (
                            discarded_index in hand_reds[player]
                            and hands[player][discarded_index] == 1
                        )
                if discard_red:
                    hand_reds[player].remove(discarded_index)
                hands[player] = after
                if player == observer:
                    hand = hands[observer]
                    current_shanten = _shanten(tuple(hand), fixed_melds)
                player_shanten = _shanten(tuple(after), fixed)
                current_waits[player] = frozenset(
                    _winning_tiles(tuple(after), fixed) if player_shanten == 0 else []
                )
                discarded = {
                    "kind": TILE_TYPES[discarded_index],
                    "red": discard_red,
                }
                winner_values = _ron_winner_values(
                    successor,
                    discarded,
                    player,
                    current_waits,
                    tuple(tuple(values) for values in hands),
                    tuple(frozenset(values) for values in hand_reds),
                    beliefs,
                )
                if player == observer and winner_values:
                    return _Outcome(
                        "deal_in",
                        -sum(winner_values.values()),
                        tuple(winner_values),
                        tuple(winner_values.items()),
                    )
                if observer in winner_values:
                    gain = _exact_win_gain(
                        successor,
                        observer,
                        tuple(hands[observer]),
                        frozenset(hand_reds[observer]),
                        discarded,
                        method="ron",
                    )
                    return _Outcome(
                        "win", gain, (observer,)
                    )
                if winner_values:
                    return _Outcome(
                        "horizontal_transfer", 0.0, tuple(winner_values)
                    )

        progress = max(0, initial_shanten - current_shanten) * 250.0
        if not wall:
            tenpai_players = [
                player
                for player in range(4)
                if _shanten(
                    tuple(hands[player]),
                    len(successor["players"][player]["melds"]),
                ) == 0
            ]
            if not tenpai_players or len(tenpai_players) == 4:
                draw_delta = 0.0
            elif observer in tenpai_players:
                draw_delta = 3_000.0 / len(tenpai_players)
            else:
                draw_delta = -3_000.0 / (4 - len(tenpai_players))
            return _Outcome("exhaustive_draw", draw_delta)
        return _Outcome("unresolved", progress)

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


def _assign_hidden_red_fives(
    state: dict[str, Any],
    hands: list[tuple[int, ...]],
    remaining: list[int],
    rng: random.Random,
) -> tuple[tuple[frozenset[int], ...], list[bool]]:
    hand_reds: list[set[int]] = [set() for _ in range(4)]
    remaining_reds = [False] * len(remaining)
    visible_reds = _visible_red_counts(state)
    configured = list((state.get("rules") or {}).get("red_fives", [1, 1, 1]))
    for suit, tile_value in enumerate((4, 13, 22)):
        hidden_reds = max(0, int(configured[suit]) - visible_reds[suit])
        locations: list[tuple[str, int]] = []
        for player, hand in enumerate(hands):
            locations.extend(("hand", player) for _ in range(hand[tile_value]))
        locations.extend(
            ("remaining", index)
            for index, value in enumerate(remaining)
            if value == tile_value
        )
        if hidden_reds > len(locations):
            raise ValueError("red-five inventory exceeds hidden physical copies")
        for location, index in rng.sample(locations, hidden_reds):
            if location == "hand":
                hand_reds[index].add(tile_value)
            else:
                remaining_reds[index] = True
    return tuple(frozenset(values) for values in hand_reds), remaining_reds


def _visible_red_counts(state: dict[str, Any]) -> list[int]:
    counts = [0, 0, 0]

    def add(tile: dict[str, Any]) -> None:
        if not tile.get("red", False):
            return
        index = tile_index(tile)
        if index in (4, 13, 22):
            counts[index // 9] += 1

    for tile in state.get("hand") or []:
        add(tile)
    if state.get("drawn_tile") is not None:
        add(state["drawn_tile"])
    for tile in state.get("dora_indicators") or []:
        add(tile)
    for public in state["players"]:
        for discard in public["discards"]:
            if discard.get("called_by") is None:
                add(discard["tile"])
        for meld in public["melds"]:
            for tile in meld["tiles"]:
                add(tile)
    return counts


def _observer_counts(state: dict[str, Any]) -> tuple[int, ...]:
    counts = [0] * 34
    for tile in state.get("hand") or []:
        counts[tile_index(tile)] += 1
    if state.get("drawn_tile") is not None:
        counts[tile_index(state["drawn_tile"])] += 1
    return tuple(counts)


def _observer_reds(state: dict[str, Any]) -> frozenset[int]:
    tiles = list(state.get("hand") or [])
    if state.get("drawn_tile") is not None:
        tiles.append(state["drawn_tile"])
    return frozenset(tile_index(tile) for tile in tiles if tile.get("red", False))


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


def _effective_sample_size(weights: Iterable[float]) -> float:
    squared = sum(float(weight) ** 2 for weight in weights)
    if squared <= 0.0 or not math.isfinite(squared):
        raise ValueError("particle weights do not define an effective sample size")
    return 1.0 / squared


@lru_cache(maxsize=250_000)
def _shanten(counts: tuple[int, ...], fixed_melds: int) -> int:
    result = json.loads(calculate_shanten_json(json.dumps(counts), fixed_melds))
    return int(result["minimum"])


@lru_cache(maxsize=250_000)
def _winning_tiles(hand: tuple[int, ...], fixed_melds: int) -> tuple[int, ...]:
    output: list[int] = []
    for tile in range(34):
        if hand[tile] >= 4:
            continue
        candidate = list(hand)
        candidate[tile] += 1
        if _shanten(tuple(candidate), fixed_melds) == -1:
            output.append(tile)
    return tuple(output)


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


def _action_discard_tile(
    action: dict[str, Any], followup: dict[str, Any] | None
) -> dict[str, Any] | None:
    if action["type"] == "discard":
        return action["tile"]
    if action["type"] == "riichi":
        return action["discard"]
    if followup is not None:
        return followup["tile"]
    return None


def _counterfactual_counterpart(
    candidate: SearchCandidate, ordered: list[SearchCandidate]
) -> SearchCandidate | None:
    action = candidate.action
    action_type = action["type"]
    if action_type in {"riichi", "discard"}:
        tile = action.get("discard", action.get("tile"))
        wanted = "discard" if action_type == "riichi" else "riichi"
        return next(
            (
                other
                for other in ordered
                if other.action["type"] == wanted
                and other.action.get("discard", other.action.get("tile")) == tile
            ),
            None,
        )
    if action_type in {"chi", "pon", "kan"}:
        return next(
            (other for other in ordered if other.action["type"] == "pass"), None
        )
    if action_type == "pass":
        return next(
            (
                other
                for other in ordered
                if other.action["type"] in {"chi", "pon", "kan"}
            ),
            None,
        )
    return None


def _ron_winner_values(
    state: dict[str, Any],
    tile: dict[str, Any],
    discarder: int,
    waits: list[frozenset[int]],
    hands: tuple[tuple[int, ...], ...],
    hand_reds: tuple[frozenset[int], ...],
    beliefs: tuple[OpponentBelief, ...],
) -> dict[int, float]:
    tile_value = tile_index(tile)
    result: dict[int, float] = {}
    observer = int(state["information"]["observer"])
    observer_furiten = state.get("observer_furiten") or {}
    for player in range(4):
        if (
            player == discarder
            or tile_value in beliefs[player].safe_tiles
            or tile_value not in waits[player]
            or (
                player == observer
                and bool(
                    observer_furiten.get("temporary", False)
                    or observer_furiten.get("riichi", False)
                )
            )
        ):
            continue
        value = _exact_hand_value(
            state,
            player,
            hands[player],
            hand_reds[player],
            tile,
            method="ron",
        )
        if value is not None:
            result[player] = value
    return result


def _exact_hand_value(
    state: dict[str, Any],
    player: int,
    concealed_counts: tuple[int, ...],
    concealed_reds: frozenset[int],
    winning_tile: dict[str, Any],
    *,
    method: str,
) -> float | None:
    scored = _score_exact_hand(
        state,
        player,
        concealed_counts,
        concealed_reds,
        winning_tile,
        method=method,
    )
    if scored is None:
        return None
    payment = scored["payment"]
    if method == "ron":
        return float(payment["ron_payment"])
    observer = int(state["information"]["observer"])
    if player == int((state.get("round") or {}).get("dealer", -1)):
        return float(payment["tsumo_from_non_dealer"])
    if observer == int((state.get("round") or {}).get("dealer", -1)):
        return float(payment["tsumo_from_dealer"])
    return float(payment["tsumo_from_non_dealer"])


def _exact_win_gain(
    state: dict[str, Any],
    player: int,
    concealed_counts: tuple[int, ...],
    concealed_reds: frozenset[int],
    winning_tile: dict[str, Any],
    *,
    method: str,
) -> float:
    scored = _score_exact_hand(
        state,
        player,
        concealed_counts,
        concealed_reds,
        winning_tile,
        method=method,
    )
    if scored is None:
        raise ValueError("winning rollout shape unexpectedly has no yaku")
    return float(scored["payment"]["total_received"])


def _score_exact_hand(
    state: dict[str, Any],
    player: int,
    concealed_counts: tuple[int, ...],
    concealed_reds: frozenset[int],
    winning_tile: dict[str, Any],
    *,
    method: str,
) -> dict[str, Any] | None:
    concealed: list[dict[str, Any]] = []
    for index, count in enumerate(concealed_counts):
        red_count = int(index in concealed_reds)
        concealed.extend(
            {"kind": TILE_TYPES[index], "red": copy < red_count}
            for copy in range(count)
        )
    concealed.append(winning_tile)
    round_state = state.get("round") or {}
    dealer = int(round_state.get("dealer", 0))
    seat_wind = ("east", "south", "west", "north")[(player - dealer) % 4]
    riichi_state = str(state["players"][player]["riichi"])
    if riichi_state in {"double_declared", "double_accepted"}:
        riichi = "double_riichi"
    elif riichi_state in {"declared", "accepted"}:
        riichi = "riichi"
    else:
        riichi = "none"
    context = {
        "method": method,
        "seat_wind": seat_wind,
        "round_wind": str(round_state.get("wind", "east")),
        "riichi": riichi,
        "ippatsu": bool(state.get("ippatsu_eligible", [False] * 4)[player]),
        "rinshan": False,
        "chankan": False,
        "haitei": method == "tsumo" and int(state.get("remaining_tiles", 1)) == 0,
        "houtei": method == "ron" and int(state.get("remaining_tiles", 1)) == 0,
        "tenhou": False,
        "chiihou": False,
        "dora_indicators": [tile["kind"] for tile in state.get("dora_indicators", [])],
        "ura_dora_indicators": [],
        "honba": int(round_state.get("honba", 0)),
        "riichi_sticks": int(round_state.get("riichi_sticks", 0)),
    }
    try:
        scored = json.loads(
            score_hand_json(
                canonical_json(concealed),
                canonical_json(state["players"][player]["melds"]),
                canonical_json(winning_tile),
                canonical_json(context),
                canonical_json(state.get("rules") or {}),
            )
        )
    except ValueError as error:
        if "no yaku" in str(error):
            return None
        raise
    return scored


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
