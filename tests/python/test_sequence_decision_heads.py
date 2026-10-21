from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, Subset

from mahjong_ai import FourPlayerSimulator
from mahjong_ai.ai.calibration import fit_platt_scaling
from mahjong_ai.ai.encoder import ACTION_FEATURES, ENCODER_PLANES
from mahjong_ai.ai.inference import ModelMetadata
from mahjong_ai.ai.sequence import (
    EVENT_FLAG_FEATURES,
    MahjongModelOutput,
    SequenceMahjongNetwork,
    SequenceModelConfig,
    compose_hierarchical_policy_logits,
)
from mahjong_ai.ai.sequence_inference import SequenceBeliefPredictor
from mahjong_ai.schema import ACTION_TYPES
from mahjong_ai.data.log import EventLog
from mahjong_ai.data.sequence_dataset import (
    SequenceDecisionRecord,
    _outcome_target,
    _terminal_deal_in_labels,
    validate_sequence_decision_record,
)
from mahjong_ai.training.sequence_supervised import (
    ParquetSequenceDataset,
    SequenceTrainer,
    SequenceTrainerConfig,
    collate_sequence_decisions,
    estimate_sequence_positive_weights,
    masked_binary_cross_entropy,
    masked_cross_entropy,
    masked_policy_cross_entropy,
)


def test_decision_heads_have_candidate_and_opponent_axes() -> None:
    config = SequenceModelConfig(
        snapshot_channels=8,
        snapshot_blocks=1,
        snapshot_dim=16,
        event_dim=16,
        transformer_heads=4,
        transformer_layers=1,
        transformer_feedforward=32,
        fused_dim=24,
        action_dim=12,
        maximum_events=8,
        decision_heads=True,
        opponent_dim=10,
        opponent_temporal_latents=True,
    )
    model = SequenceMahjongNetwork(config).eval()
    batch, actions, events = 2, 5, 8
    event_mask = torch.zeros(batch, events, dtype=torch.bool)
    event_mask[:, -2:] = True
    players = torch.zeros(batch, events, dtype=torch.long)
    players[0, -2:] = torch.tensor([2, 2])
    players[1, -2:] = torch.tensor([3, 4])
    action_mask = torch.ones(batch, actions, dtype=torch.bool)
    action_mask[0, -1] = False
    with torch.inference_mode():
        output = model(
            torch.zeros(batch, ENCODER_PLANES, 34),
            torch.zeros(batch, actions, ACTION_FEATURES),
            torch.zeros(batch, events, dtype=torch.long),
            players,
            torch.zeros(batch, events, dtype=torch.long),
            torch.zeros(batch, events, EVENT_FLAG_FEATURES),
            event_mask,
            action_mask,
        )
    assert output["deal_in_logits"].shape == (batch, actions, 3)
    assert output["deal_in_point_logits"].shape == (batch, actions, 3, 8)
    assert output["action_outcome_logits"].shape == (batch, actions, 5)
    assert output["response_logits"].shape == (batch, 3)
    assert output["call_type_logits"].shape == (batch, 3)
    assert output["riichi_logits"].shape == (batch, actions)
    assert output["opponent_latents"].shape == (batch, 3, 10)
    assert output["opponent_event_counts"].tolist() == [
        [2.0, 0.0, 0.0],
        [0.0, 1.0, 1.0],
    ]
    assert isinstance(output, MahjongModelOutput)
    assert output.tenpai_logits.shape == (batch, 3)
    assert output.wait_logits.shape == (batch, 3, 34)


def test_empty_missing_label_masks_are_finite_differentiable_zero() -> None:
    binary_logits = torch.randn(2, 3, requires_grad=True)
    binary = masked_binary_cross_entropy(
        binary_logits,
        torch.zeros_like(binary_logits),
        torch.zeros_like(binary_logits, dtype=torch.bool),
    )
    class_logits = torch.randn(2, 4, requires_grad=True)
    categorical = masked_cross_entropy(
        class_logits,
        torch.zeros(2, dtype=torch.long),
        torch.zeros(2, dtype=torch.bool),
    )
    loss = binary + categorical
    assert float(loss.detach()) == 0.0
    loss.backward()
    assert binary_logits.grad is not None
    assert class_logits.grad is not None
    assert float(binary_logits.grad.abs().sum()) == 0.0
    assert float(class_logits.grad.abs().sum()) == 0.0


def test_hierarchical_response_and_call_heads_change_final_policy() -> None:
    action_names = ("pass", "chi", "pon", "kan", "ron")
    features = torch.zeros(1, len(action_names), ACTION_FEATURES)
    for index, name in enumerate(action_names):
        features[0, index, ACTION_TYPES.index(name)] = 1.0
    logits = compose_hierarchical_policy_logits(
        torch.zeros(1, len(action_names)),
        features,
        response_logits=torch.tensor([[-4.0, 4.0, -4.0]]),
        call_type_logits=torch.tensor([[-4.0, 4.0, -4.0]]),
        riichi_logits=None,
        action_mask=torch.ones(1, len(action_names), dtype=torch.bool),
    )
    assert logits.argmax(dim=-1).item() == action_names.index("pon")


def test_decision_v3_validator_enforces_frozen_tile_and_action_axes() -> None:
    action = {"type": "discard", "tile": {"kind": "m1", "red": False}}
    record = SequenceDecisionRecord(
        log_id="fixture-r1",
        observer=0,
        event_sequence=1,
        observation="{}",
        public_events="[]",
        legal_actions=json.dumps([action]),
        chosen_action=json.dumps(action),
        chosen_action_index=0,
        final_rank=1,
        final_score=30_000,
        round_score_delta=5_000,
        round_delta="[5000,-5000,0,0]",
        opponent_shanten="[0,1,2]",
        opponent_tenpai="[true,false,false]",
        opponent_wait=json.dumps([[False] * 34 for _ in range(3)]),
        future_draw=json.dumps([False] * 34),
        round_result="draw",
        factual_deal_in_by_action="[[false,false,false]]",
        factual_deal_in_points_by_action="[[0,0,0]]",
        factual_deal_in_mask="[[true,true,true]]",
        counterfactual_deal_in_by_action="[[false,false,false]]",
        counterfactual_deal_in_points_by_action="[[0,0,0]]",
        counterfactual_deal_in_mask="[[false,false,false]]",
        label_masks="{}",
        source="fixture",
        converter_version="fixture",
        state_schema_version=1,
    )
    validate_sequence_decision_record(record)
    with pytest.raises(ValueError, match="34 tile"):
        validate_sequence_decision_record(replace(record, future_draw="[]"))


def test_hierarchical_riichi_head_penalizes_unlikely_riichi_candidate() -> None:
    features = torch.zeros(1, 2, ACTION_FEATURES)
    features[0, 0, ACTION_TYPES.index("discard")] = 1.0
    features[0, 1, ACTION_TYPES.index("riichi")] = 1.0
    logits = compose_hierarchical_policy_logits(
        torch.zeros(1, 2),
        features,
        response_logits=None,
        call_type_logits=None,
        riichi_logits=torch.tensor([[0.0, -8.0]]),
        action_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    assert logits.argmax(dim=-1).item() == 0


def test_sequence_evaluate_state_returns_web_ready_candidate_contract() -> None:
    simulator = FourPlayerSimulator(
        11,
        json.dumps(
            {
                "wind": "east",
                "hand_number": 1,
                "honba": 0,
                "riichi_sticks": 0,
                "dealer": 0,
            }
        ),
        [25_000] * 4,
    )
    simulator.draw()
    state = json.loads(simulator.observation_json(0))
    predictor = SequenceBeliefPredictor(
        _tiny_sequence_model(temporal=True),
        ModelMetadata(
            "sequence-fixture",
            "observable-planes-v1+observable-events-v2",
            trained=True,
            validated=True,
            policy_calibrated=False,
        ),
    )
    analysis = predictor.evaluate_state(state)
    candidates = analysis["model_policy"]["value"]
    assert analysis["model_policy"]["status"] == "model_estimate"
    assert len(candidates) == len(analysis["legal_actions"]["value"])
    assert abs(sum(item["policy_score"] for item in candidates) - 1.0) < 1e-5
    assert len(candidates[0]["model_deal_in_probability_by_opponent"]) == 4
    assert len(analysis["sequence_model_predictions"]["value"]["opponent_wait"]) == 3


def test_temporal_opponent_heads_backpropagate_into_public_event_encoder() -> None:
    model = SequenceMahjongNetwork(
        SequenceModelConfig(
            snapshot_channels=8,
            snapshot_blocks=1,
            snapshot_dim=16,
            event_dim=16,
            transformer_heads=4,
            transformer_layers=1,
            transformer_feedforward=32,
            fused_dim=24,
            action_dim=12,
            maximum_events=8,
            decision_heads=True,
            opponent_dim=10,
            opponent_temporal_latents=True,
            dropout=0.0,
        )
    )
    event_mask = torch.zeros(1, 8, dtype=torch.bool)
    event_mask[:, -4:] = True
    players = torch.zeros(1, 8, dtype=torch.long)
    players[:, -4:] = torch.tensor([[2, 3, 2, 4]])
    output = model(
        torch.zeros(1, ENCODER_PLANES, 34),
        torch.zeros(1, 2, ACTION_FEATURES),
        torch.tensor([[0, 0, 0, 0, 3, 3, 4, 3]]),
        players,
        torch.tensor([[0, 0, 0, 0, 2, 8, 0, 12]]),
        torch.zeros(1, 8, EVENT_FLAG_FEATURES),
        event_mask,
        torch.ones(1, 2, dtype=torch.bool),
    )
    (
        output["tenpai_logits"].sum()
        + output["wait_logits"].sum()
        + output["shanten_logits"].sum()
    ).backward()
    assert model.event_type_embedding.weight.grad is not None
    assert float(model.event_type_embedding.weight.grad.abs().sum()) > 0.0
    assert model.opponent_event_projection[0].weight.grad is not None


def test_legacy_sequence_config_keeps_global_opponent_heads() -> None:
    model = _tiny_sequence_model()
    assert model.config.opponent_temporal_latents is False
    assert not hasattr(model, "opponent_event_projection")


def test_temporal_opponent_model_can_warm_start_from_legacy_state() -> None:
    legacy = _tiny_sequence_model(temporal=False)
    temporal = _tiny_sequence_model(temporal=True)
    incompatible = temporal.load_state_dict(legacy.state_dict(), strict=False)
    assert incompatible.unexpected_keys == []
    allowed = (
        "opponent_event_query",
        "opponent_event_projection",
        "opponent_temporal_fusion",
        "opponent_temporal_norm",
        "opponent_tenpai_from_latent",
        "opponent_wait_from_latent",
        "opponent_shanten_from_latent",
        "response_head",
        "call_type_head",
        "riichi_head",
        "deal_in_point_head",
    )
    assert incompatible.missing_keys
    assert all(key.startswith(allowed) for key in incompatible.missing_keys)


def test_platt_scaling_improves_biased_synthetic_probabilities() -> None:
    generator = torch.Generator().manual_seed(20261020)
    logits = torch.linspace(-4.0, 4.0, 5_000)
    true_probability = torch.sigmoid(logits * 0.55 - 1.2)
    targets = torch.bernoulli(true_probability, generator=generator)
    result = fit_platt_scaling(logits, targets)
    assert result.scale > 0.0
    assert result.positives > 0
    assert result.after["log_loss"] < result.before["log_loss"]
    assert result.after["brier"] < result.before["brier"]


def test_old_metadata_defaults_risk_to_uncalibrated() -> None:
    metadata = ModelMetadata(
        **{
            "model_id": "old",
            "encoder_version": "fixture",
            "trained": True,
            "validated": True,
            "policy_calibrated": True,
        }
    )
    assert metadata.risk_calibrated is False
    assert metadata.risk_logit_scale_by_opponent is None


def test_terminal_labels_distinguish_loser_from_horizontal_ron() -> None:
    events = [
        {"type": "round_start", "round": {"honba": 0}},
        {"type": "discard", "player": 2},
        {"type": "ron", "winners": [1], "loser": 2},
    ]
    log = EventLog(
        log_id="labels-r1",
        rules=None,
        observations={player: events for player in range(4)},
        final_scores=(25_000, 33_000, 17_000, 25_000),
        source="test",
        converter_version="test",
        round_score_deltas=(0, 8_000, -8_000, 0),
    )
    discard_index, winners, points = _terminal_deal_in_labels(log)
    assert discard_index == 1
    assert winners == {1}
    assert points == [0, 8_000, 0, 0]
    assert _outcome_target(log, 1) == 0
    assert _outcome_target(log, 2) == 1
    assert _outcome_target(log, 0) == 3


def _tiny_sequence_model(
    *, decision_heads: bool = True, temporal: bool = False
) -> SequenceMahjongNetwork:
    return SequenceMahjongNetwork(
        SequenceModelConfig(
            snapshot_channels=8,
            snapshot_blocks=1,
            snapshot_dim=16,
            event_dim=16,
            transformer_heads=4,
            transformer_layers=1,
            transformer_feedforward=32,
            fused_dim=24,
            action_dim=12,
            decision_heads=decision_heads,
            opponent_dim=10,
            opponent_temporal_latents=temporal,
        )
    )


def test_decision_heads_training_step_accepts_legacy_sequence_rows() -> None:
    root = Path(__file__).resolve().parents[2]
    dataset = ParquetSequenceDataset(
        root / "artifacts/smoke/sequence-v2-data/part-00000.parquet"
    )
    loader = DataLoader(
        Subset(dataset, list(range(4))),
        batch_size=4,
        collate_fn=collate_sequence_decisions,
    )
    model = _tiny_sequence_model(temporal=True)
    trainer = SequenceTrainer(
        model, SequenceTrainerConfig(epochs=1, batch_size=4, learning_rate=1e-3)
    )
    metrics = trainer._run_epoch(loader, training=True)
    assert all(math.isfinite(value) for value in metrics.values())


def test_fit_records_effective_weights_and_continues_epoch_numbers(tmp_path) -> None:
    root = Path(__file__).resolve().parents[2]
    dataset = ParquetSequenceDataset(
        root / "artifacts/smoke/sequence-v2-data/part-00000.parquet"
    )
    trainer = SequenceTrainer(
        _tiny_sequence_model(decision_heads=False),
        SequenceTrainerConfig(epochs=1, batch_size=4, learning_rate=1e-3),
    )
    checkpoint = tmp_path / "continued.pt"
    history = trainer.fit(
        Subset(dataset, list(range(8))),
        Subset(dataset, list(range(8, 12))),
        checkpoint,
        model_id="continued",
        dataset_fingerprint="fixture",
        start_epoch=3,
        dataset_manifest_hashes={"manifest.json": "abc123"},
    )
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert history[0]["epoch"] == 4.0
    assert saved["epoch"] == 4
    assert saved["effective_positive_weights"] == trainer.effective_positive_weights
    assert saved["optimizer_state"]["state"]
    assert set(saved["rng_state"]) == {"torch", "python", "numpy"}
    assert saved["architecture_version"] == "sequence-multitask-v1"
    assert saved["model_output_version"] == "mahjong-model-output/v1"
    assert saved["decision_schema_version"] == "mahjong-ai-sequence-decision/v3"
    assert saved["dataset_version"] == "mahjong-ai-sequence-decision/v3"
    assert saved["label_version"] == "sequence-labels-v1"
    assert saved["global_step"] > 0
    assert saved["scheduler_state"] is not None
    assert saved["dataset_manifest_hashes"] == {"manifest.json": "abc123"}
    assert "feature_version" in saved
    assert "git_commit" in saved
    resumed = SequenceTrainer(
        _tiny_sequence_model(decision_heads=False),
        SequenceTrainerConfig(epochs=1, batch_size=4, learning_rate=1e-3),
    )
    assert resumed.restore_training_state(saved) == 4
    assert resumed.optimizer.state_dict()["state"]
    assert resumed.scheduler.state_dict() == saved["scheduler_state"]


def test_sequence_training_is_reproducible_for_the_same_seed() -> None:
    root = Path(__file__).resolve().parents[2]
    dataset = ParquetSequenceDataset(
        root / "artifacts/smoke/sequence-v2-data/part-00000.parquet"
    )
    loader = DataLoader(
        Subset(dataset, list(range(4))),
        batch_size=4,
        collate_fn=collate_sequence_decisions,
    )

    def run() -> tuple[dict[str, float], dict[str, torch.Tensor]]:
        torch.manual_seed(991)
        trainer = SequenceTrainer(
            _tiny_sequence_model(decision_heads=False),
            SequenceTrainerConfig(
                seed=77,
                epochs=1,
                batch_size=4,
                learning_rate=1e-3,
                auto_positive_weights=False,
            ),
        )
        metrics = trainer._run_epoch(loader, training=True)
        return metrics, {
            key: value.detach().clone() for key, value in trainer.model.state_dict().items()
        }

    first_metrics, first_state = run()
    second_metrics, second_state = run()
    assert first_metrics == second_metrics
    assert first_state.keys() == second_state.keys()
    assert all(torch.equal(first_state[key], second_state[key]) for key in first_state)


def test_positive_weights_use_training_rows_and_bound_rare_labels() -> None:
    class Rows(torch.utils.data.Dataset):
        def __init__(self) -> None:
            self.rows = [
                {
                    "observer": 0,
                    "opponent_tenpai": "[false,true,false,false]",
                    "opponent_waits": "[[],[0,1],[],[]]",
                    "expert_deal_in_by_player": "[false,false,true,false]",
                },
                {
                    "observer": 1,
                    "opponent_tenpai": "[false,false,false,false]",
                    "opponent_waits": "[[],[],[],[]]",
                    "expert_deal_in_by_player": "[false,false,false,false]",
                },
            ]

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int):
            return self.rows[index]

    weights = estimate_sequence_positive_weights(Rows(), maximum=12.0)
    assert weights["tenpai"] == 5.0
    assert weights["wait"] == 12.0
    assert weights["deal_in"] == 5.0


def test_positive_weights_do_not_read_rows_outside_training_subset() -> None:
    class Rows(torch.utils.data.Dataset):
        def __init__(self) -> None:
            self.rows = [
                {
                    "observer": 0,
                    "opponent_tenpai": "[false,true,false,false]",
                    "opponent_waits": "[[],[0],[],[]]",
                    "expert_deal_in_by_player": None,
                },
                {
                    "observer": 0,
                    "opponent_tenpai": "[false,false,false,false]",
                    "opponent_waits": "[[],[],[],[]]",
                    "expert_deal_in_by_player": "[false,true,true,true]",
                },
            ]

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int):
            return self.rows[index]

    weights = estimate_sequence_positive_weights(
        Subset(Rows(), [0]),
        fallback={"tenpai": 1.0, "wait": 10.0, "deal_in": 7.0},
    )
    assert weights["deal_in"] == 7.0


def test_policy_smoothing_never_assigns_mass_to_padding_actions() -> None:
    logits = torch.tensor(
        [[2.0, 0.0, torch.finfo(torch.float32).min]], requires_grad=True
    )
    mask = torch.tensor([[True, True, False]])
    loss = masked_policy_cross_entropy(
        logits, torch.tensor([0]), mask, label_smoothing=0.1
    )
    assert math.isfinite(float(loss.detach()))
    loss.backward()
    assert logits.grad is not None
    assert float(logits.grad[0, 2]) == 0.0
    two_action_expected = 0.9 * -torch.log_softmax(logits[:, :2], -1)[0, 0]
    two_action_expected += 0.1 * -torch.log_softmax(logits[:, :2], -1).mean()
    assert torch.allclose(loss, two_action_expected)
