"""Local Rust-simulator self-play, replay buffer, and PPO primitives."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import random
from collections import deque
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch import Tensor

from mahjong_ai import FourPlayerSimulator
from mahjong_ai.ai.encoder import FeatureEncoder
from mahjong_ai.ai.model import MahjongNetwork, ModelConfig
from mahjong_ai.schema import canonical_json


@dataclass(frozen=True)
class PolicyDecision:
    action_index: int
    log_probability: float
    value: float


class Policy(Protocol):
    def decide(
        self,
        observation: dict[str, Any],
        legal_actions: list[dict[str, Any]],
    ) -> PolicyDecision: ...


class RandomPolicy:
    def __init__(self, seed: int) -> None:
        self.random = random.Random(seed)

    def choose(self, observation: dict[str, Any], legal_actions: list[dict[str, Any]]) -> int:
        return self.decide(observation, legal_actions).action_index

    def decide(
        self,
        observation: dict[str, Any],
        legal_actions: list[dict[str, Any]],
    ) -> PolicyDecision:
        del observation
        index = self.random.randrange(len(legal_actions))
        return PolicyDecision(index, -float(np.log(len(legal_actions))), 0.0)


class TorchPolicy:
    """Sample legal actions from a dynamic-action MahjongNetwork."""

    def __init__(
        self,
        model: MahjongNetwork,
        seed: int,
        device: str = "cpu",
        temperature: float = 1.0,
    ) -> None:
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        self.model = model
        self.device = device
        self.temperature = temperature
        self.encoder = FeatureEncoder()
        self.generator = torch.Generator(device="cpu").manual_seed(seed)

    def choose(self, observation: dict[str, Any], legal_actions: list[dict[str, Any]]) -> int:
        return self.decide(observation, legal_actions).action_index

    def decide(
        self,
        observation: dict[str, Any],
        legal_actions: list[dict[str, Any]],
    ) -> PolicyDecision:
        if not legal_actions:
            raise ValueError("policy cannot choose from an empty legal-action set")
        encoded = self.encoder.encode(observation, legal_actions)
        planes = torch.from_numpy(encoded.planes).unsqueeze(0).to(self.device)
        actions = torch.from_numpy(encoded.action_features).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.inference_mode():
            output = self.model(planes, actions)
            logits = output["policy_logits"][0] / self.temperature
            probabilities = torch.softmax(logits, dim=-1).cpu()
            index = int(torch.multinomial(probabilities, 1, generator=self.generator).item())
            log_probability = float(torch.log(probabilities[index].clamp_min(1e-12)))
            value = float(output["score_value"][0].cpu())
        return PolicyDecision(index, log_probability, value)


@dataclass
class Transition:
    observation: str
    legal_actions: str
    action_index: int
    player: int
    reward: float = 0.0
    done: bool = False
    round_index: int = 0
    round_seed: int = 0
    decision_index: int = 0
    old_log_probability: float = 0.0
    old_value: float = 0.0


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._items: deque[Transition] = deque(maxlen=capacity)

    def extend(self, transitions: list[Transition]) -> None:
        self._items.extend(transitions)

    def sample(self, count: int, seed: int) -> list[Transition]:
        if count > len(self._items):
            raise ValueError("sample exceeds replay buffer size")
        return random.Random(seed).sample(list(self._items), count)

    def save(self, path: str | Path) -> None:
        if not self._items:
            raise ValueError("cannot save empty replay buffer")
        table = pa.Table.from_pylist([asdict(item) for item in self._items])
        pq.write_table(table, path, compression="zstd")

    def __len__(self) -> int:
        return len(self._items)


class SelfPlayWorker:
    def __init__(self, policies: list[Policy]) -> None:
        if len(policies) != 4:
            raise ValueError("self-play needs four policies")
        self.policies = policies

    def play_round(
        self,
        seed: int,
        round_state: dict[str, Any],
        round_index: int = 0,
    ) -> list[Transition]:
        simulator = FourPlayerSimulator(seed, canonical_json(round_state), [25_000] * 4)
        initial_scores = [25_000] * 4
        transitions: list[Transition] = []
        for _ in range(400):
            states = [json.loads(simulator.observation_json(player)) for player in range(4)]
            phase = states[0]["phase"]
            phase_type = phase["phase"]
            if phase_type == "ended":
                final_scores = states[0]["scores"]
                for transition in transitions:
                    transition.reward = (final_scores[transition.player] - initial_scores[transition.player]) / 10_000.0
                    transition.done = True
                return transitions
            if phase_type in {"awaiting_draw", "awaiting_rinshan_draw"}:
                simulator.draw()
                continue
            if phase_type == "awaiting_discard":
                player = int(phase["player"])
                legal = json.loads(simulator.legal_actions_json(player))
                decision = self.policies[player].decide(states[player], legal)
                index = decision.action_index
                transitions.append(
                    Transition(
                        canonical_json(states[player]),
                        canonical_json(legal),
                        index,
                        player,
                        round_index=round_index,
                        round_seed=seed,
                        decision_index=len(transitions),
                        old_log_probability=decision.log_probability,
                        old_value=decision.value,
                    )
                )
                simulator.act_json(player, canonical_json(legal[index]))
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
                    decision = self.policies[player].decide(states[player], legal)
                    index = decision.action_index
                    choices.append((player, legal[index]))
                    transitions.append(
                        Transition(
                            canonical_json(states[player]),
                            canonical_json(legal),
                            index,
                            player,
                            round_index=round_index,
                            round_seed=seed,
                            decision_index=len(transitions),
                            old_log_probability=decision.log_probability,
                            old_value=decision.value,
                        )
                    )
                simulator.resolve_responses_json(canonical_json(choices))
                next_phase = json.loads(simulator.observation_json(0))["phase"]["phase"]
                if next_phase in {"awaiting_response", "awaiting_kan_response"}:
                    simulator.draw()
                continue
            raise RuntimeError(f"unsupported simulator phase: {phase_type}")
        raise RuntimeError("round exceeded 400 simulator decisions")


@dataclass(frozen=True)
class ParallelSelfPlayConfig:
    rounds: int
    seed: int = 20_261_018
    workers: int = 0
    shard_rounds: int = 64

    def validate(self) -> None:
        if self.rounds <= 0:
            raise ValueError("rounds must be positive")
        if self.workers < 0:
            raise ValueError("workers must be zero (auto) or positive")
        if self.shard_rounds <= 0:
            raise ValueError("shard_rounds must be positive")

    def resolved_workers(self) -> int:
        available = os.cpu_count() or 1
        requested = self.workers or available
        return max(1, min(requested, self.rounds))


class ParallelSelfPlayRunner:
    """Generate deterministic random-policy trajectories in Parquet shards."""

    FORMAT = "mahjong-ai-selfplay/v1"

    def __init__(self, config: ParallelSelfPlayConfig, round_state: dict[str, Any]) -> None:
        config.validate()
        self.config = config
        self.round_state = round_state

    def run(self, output_directory: str | Path) -> dict[str, Any]:
        output = Path(output_directory)
        output.mkdir(parents=True, exist_ok=True)
        if (output / "manifest.json").exists() or any(output.glob("part-*.parquet")):
            raise ValueError("output directory already contains a self-play dataset")

        tasks = [
            (round_index, self.config.seed + round_index, self.round_state)
            for round_index in range(self.config.rounds)
        ]
        workers = self.config.resolved_workers()
        if workers == 1:
            results = map(_play_random_round, tasks)
            manifest = self._consume(results, output, workers, "inline")
        else:
            context = multiprocessing.get_context("spawn")
            try:
                executor: ProcessPoolExecutor | ThreadPoolExecutor = ProcessPoolExecutor(
                    max_workers=workers,
                    mp_context=context,
                )
                backend = "process"
            except (NotImplementedError, PermissionError):
                executor = ThreadPoolExecutor(max_workers=workers)
                backend = "thread_fallback"
            with executor:
                results = executor.map(_play_random_round, tasks, chunksize=1)
                manifest = self._consume(results, output, workers, backend)

        temporary_manifest = output / "manifest.json.tmp"
        temporary_manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_manifest.replace(output / "manifest.json")
        return manifest

    def _consume(
        self,
        results: Iterable[tuple[int, list[Transition]]],
        output: Path,
        workers: int,
        execution_backend: str,
    ) -> dict[str, Any]:
        shard_rows: list[Transition] = []
        shard_round_count = 0
        shards: list[dict[str, Any]] = []
        transition_count = 0
        for round_index, transitions in results:
            if not transitions:
                raise RuntimeError(f"self-play round {round_index} produced no transitions")
            shard_rows.extend(transitions)
            shard_round_count += 1
            transition_count += len(transitions)
            if shard_round_count == self.config.shard_rounds:
                shards.append(_write_selfplay_shard(output, len(shards), shard_rows, shard_round_count))
                shard_rows = []
                shard_round_count = 0
        if shard_rows:
            shards.append(_write_selfplay_shard(output, len(shards), shard_rows, shard_round_count))
        return {
            "format": self.FORMAT,
            "generator": "parallel_random_policy_v1",
            "base_seed": self.config.seed,
            "rounds": self.config.rounds,
            "transitions": transition_count,
            "workers_used": workers,
            "execution_backend": execution_backend,
            "shard_rounds": self.config.shard_rounds,
            "round_state": self.round_state,
            "shards": shards,
        }


@dataclass(frozen=True)
class ModelSelfPlayConfig:
    rounds: int
    seed: int = 20_261_019
    workers: int = 0
    temperature: float = 1.0
    first_round_index: int = 0

    def validate(self) -> None:
        if self.rounds <= 0:
            raise ValueError("rounds must be positive")
        if self.workers < 0:
            raise ValueError("workers must be zero (auto) or positive")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if self.first_round_index < 0:
            raise ValueError("first_round_index must not be negative")

    def resolved_workers(self) -> int:
        available = os.cpu_count() or 1
        requested = self.workers or available
        return max(1, min(requested, self.rounds))


@dataclass(frozen=True)
class SelfPlayCollection:
    transitions: list[Transition]
    execution_backend: str
    workers_used: int


class ParallelModelSelfPlayCollector:
    """Collect ordered on-policy rounds from an immutable model snapshot."""

    def __init__(self, config: ModelSelfPlayConfig, round_state: dict[str, Any]) -> None:
        config.validate()
        self.config = config
        self.round_state = round_state

    def collect(self, model: MahjongNetwork) -> SelfPlayCollection:
        workers = self.config.resolved_workers()
        state_dict = {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
        }
        initargs = (model.config.to_dict(), state_dict, self.config.temperature)
        tasks = [
            (
                self.config.first_round_index + offset,
                self.config.seed + self.config.first_round_index + offset,
                self.round_state,
            )
            for offset in range(self.config.rounds)
        ]
        if workers == 1:
            _initialize_model_worker(*initargs)
            results = map(_play_model_round, tasks)
            backend = "inline"
        else:
            context = multiprocessing.get_context("spawn")
            try:
                executor: ProcessPoolExecutor | ThreadPoolExecutor = ProcessPoolExecutor(
                    max_workers=workers,
                    mp_context=context,
                    initializer=_initialize_model_worker,
                    initargs=initargs,
                )
                backend = "process"
            except (NotImplementedError, PermissionError):
                executor = ThreadPoolExecutor(
                    max_workers=workers,
                    initializer=_initialize_model_worker,
                    initargs=initargs,
                )
                backend = "thread_fallback"
            with executor:
                results = executor.map(_play_model_round, tasks, chunksize=1)
                ordered = list(results)
            results = iter(ordered)

        transitions: list[Transition] = []
        for round_index, round_transitions in results:
            if not round_transitions:
                raise RuntimeError(f"model self-play round {round_index} produced no transitions")
            transitions.extend(round_transitions)
        return SelfPlayCollection(transitions, backend, workers)


_MODEL_WORKER_MODEL: MahjongNetwork | None = None
_MODEL_WORKER_TEMPERATURE = 1.0


def _initialize_model_worker(
    model_config: dict[str, Any],
    state_dict: dict[str, Tensor],
    temperature: float,
) -> None:
    global _MODEL_WORKER_MODEL, _MODEL_WORKER_TEMPERATURE
    model = MahjongNetwork(ModelConfig(**model_config))
    model.load_state_dict(state_dict)
    _MODEL_WORKER_MODEL = model.eval()
    _MODEL_WORKER_TEMPERATURE = temperature


def _play_model_round(task: tuple[int, int, dict[str, Any]]) -> tuple[int, list[Transition]]:
    if _MODEL_WORKER_MODEL is None:
        raise RuntimeError("model self-play worker was not initialized")
    round_index, seed, round_state = task
    policies = [
        TorchPolicy(
            _MODEL_WORKER_MODEL,
            seed=seed * 4 + seat,
            temperature=_MODEL_WORKER_TEMPERATURE,
        )
        for seat in range(4)
    ]
    transitions = SelfPlayWorker(policies).play_round(seed, round_state, round_index=round_index)
    return round_index, transitions


def _play_random_round(task: tuple[int, int, dict[str, Any]]) -> tuple[int, list[Transition]]:
    round_index, seed, round_state = task
    policies = [RandomPolicy(seed * 4 + seat) for seat in range(4)]
    transitions = SelfPlayWorker(policies).play_round(seed, round_state, round_index=round_index)
    return round_index, transitions


def _write_selfplay_shard(
    output: Path,
    shard_index: int,
    transitions: list[Transition],
    round_count: int,
) -> dict[str, Any]:
    filename = f"part-{shard_index:05d}.parquet"
    destination = output / filename
    temporary = output / f"{filename}.tmp"
    table = pa.Table.from_pylist([asdict(item) for item in transitions])
    pq.write_table(table, temporary, compression="zstd")
    temporary.replace(destination)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    return {
        "path": filename,
        "rounds": round_count,
        "transitions": len(transitions),
        "first_round_index": transitions[0].round_index,
        "last_round_index": transitions[-1].round_index,
        "sha256": digest,
    }


def verify_selfplay_dataset(directory: str | Path) -> dict[str, Any]:
    """Verify manifest, shard hashes, row counts, and contiguous round ranges."""

    root = Path(directory)
    manifest_path = root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read self-play manifest: {error}") from error
    if manifest.get("format") != ParallelSelfPlayRunner.FORMAT:
        raise ValueError("unsupported self-play dataset format")
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("self-play manifest has no shards")

    transition_count = 0
    expected_first_round = 0
    for shard in shards:
        relative = shard.get("path")
        if not isinstance(relative, str) or Path(relative).name != relative:
            raise ValueError("self-play shard path must be a plain filename")
        path = root / relative
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as error:
            raise ValueError(f"cannot read self-play shard {relative}: {error}") from error
        if digest != shard.get("sha256"):
            raise ValueError(f"self-play shard hash mismatch: {relative}")
        try:
            rows = pq.read_metadata(path).num_rows
        except Exception as error:
            raise ValueError(f"cannot parse self-play shard {relative}: {error}") from error
        if rows != shard.get("transitions"):
            raise ValueError(f"self-play shard row count mismatch: {relative}")
        if shard.get("first_round_index") != expected_first_round:
            raise ValueError(f"self-play shard round range is not contiguous: {relative}")
        expected_first_round = int(shard["last_round_index"]) + 1
        transition_count += rows

    if transition_count != manifest.get("transitions"):
        raise ValueError("self-play manifest transition total does not match shards")
    if expected_first_round != manifest.get("rounds"):
        raise ValueError("self-play manifest round total does not match shards")
    return manifest


def generalized_advantage_estimate(
    rewards: Tensor,
    values: Tensor,
    dones: Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Tensor:
    if rewards.shape != dones.shape or values.shape[-1] != rewards.shape[-1] + 1:
        raise ValueError("GAE expects values to include one bootstrap step")
    advantages = torch.zeros_like(rewards)
    running = torch.zeros_like(rewards[..., 0])
    for step in range(rewards.shape[-1] - 1, -1, -1):
        alive = 1.0 - dones[..., step]
        delta = rewards[..., step] + gamma * values[..., step + 1] * alive - values[..., step]
        running = delta + gamma * gae_lambda * alive * running
        advantages[..., step] = running
    return advantages


def clipped_ppo_loss(
    new_log_probability: Tensor,
    old_log_probability: Tensor,
    advantages: Tensor,
    clip_ratio: float = 0.2,
) -> Tensor:
    ratio = (new_log_probability - old_log_probability).exp()
    unclipped = ratio * advantages
    clipped = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
    return -torch.minimum(unclipped, clipped).mean()
