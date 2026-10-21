from __future__ import annotations

import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from mahjong_ai.ai.calibration import fit_platt_scaling
from mahjong_ai.ai.encoder import ACTION_FEATURES, ENCODER_PLANES
from mahjong_ai.ai.inference import ModelMetadata
from mahjong_ai.ai.sequence import EVENT_FLAG_FEATURES, SequenceMahjongNetwork, SequenceModelConfig
from mahjong_ai.data.log import EventLog
from mahjong_ai.data.sequence_dataset import _outcome_target, _terminal_deal_in_labels
from mahjong_ai.training.sequence_supervised import (
    ParquetSequenceDataset,
    SequenceTrainer,
    SequenceTrainerConfig,
    collate_sequence_decisions,
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
    )
    model = SequenceMahjongNetwork(config).eval()
    batch, actions, events = 2, 5, 8
    event_mask = torch.zeros(batch, events, dtype=torch.bool)
    event_mask[:, -2:] = True
    action_mask = torch.ones(batch, actions, dtype=torch.bool)
    action_mask[0, -1] = False
    with torch.inference_mode():
        output = model(
            torch.zeros(batch, ENCODER_PLANES, 34),
            torch.zeros(batch, actions, ACTION_FEATURES),
            torch.zeros(batch, events, dtype=torch.long),
            torch.zeros(batch, events, dtype=torch.long),
            torch.zeros(batch, events, dtype=torch.long),
            torch.zeros(batch, events, EVENT_FLAG_FEATURES),
            event_mask,
            action_mask,
        )
    assert output["deal_in_logits"].shape == (batch, actions, 4)
    assert output["deal_in_score_logits"].shape == (batch, 4, 8)
    assert output["action_outcome_logits"].shape == (batch, actions, 5)
    assert output["opponent_latents"].shape == (batch, 4, 10)


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
            decision_heads=True,
            opponent_dim=10,
        )
    )
    trainer = SequenceTrainer(
        model, SequenceTrainerConfig(epochs=1, batch_size=4, learning_rate=1e-3)
    )
    metrics = trainer._run_epoch(loader, training=True)
    assert all(math.isfinite(value) for value in metrics.values())
