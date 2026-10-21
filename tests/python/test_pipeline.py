from __future__ import annotations

import json
from dataclasses import replace

import cv2
import numpy as np
import pyarrow.parquet as pq
import pytest
import torch

from mahjong_ai import FourPlayerSimulator
from mahjong_ai.ai.encoder import ACTION_FEATURES, ENCODER_PLANES, FeatureEncoder
from mahjong_ai.ai.inference import InferenceEngine, ModelMetadata, save_checkpoint
from mahjong_ai.ai.model import MahjongNetwork, ModelConfig
from mahjong_ai.analysis import EXTERNAL_AUTHORIZATION, REALTIME_PROTOCOL
from mahjong_ai.apps.reviewer import create_app
from mahjong_ai.apps.local_play import LocalPlayManager
from mahjong_ai.data.dataset import build_decision_records, write_parquet
from mahjong_ai.data.log import EventLogError, load_event_log
from mahjong_ai.data.tenhou import _kind_index, convert_document
from mahjong_ai.data.tenhou_archive import (
    iter_tenhou_archive_xml,
    validate_tenhou_archive,
)
from mahjong_ai.evaluation import evaluate_checkpoints
from mahjong_ai.data.provenance import (
    create_human_dataset_manifest,
    load_human_dataset_manifest,
)
from mahjong_ai.perception import LayoutConfig, ScreenshotParser, TemplateTileClassifier
from mahjong_ai.training.oracle import OracleBatch, OracleTeacher, distillation_loss
from mahjong_ai.training.ppo import PPOTrainer, PPOTrainerConfig
from mahjong_ai.training.selfplay import (
    ModelSelfPlayConfig,
    ParallelModelSelfPlayCollector,
    ParallelSelfPlayConfig,
    ParallelSelfPlayRunner,
    RandomPolicy,
    SelfPlayWorker,
    verify_selfplay_dataset,
)
from mahjong_ai.training.supervised import ParquetDecisionDataset, Trainer, TrainerConfig
from fastapi.testclient import TestClient


def _round() -> dict:
    return {"wind": "east", "hand_number": 1, "honba": 0, "riichi_sticks": 0, "dealer": 0}


def _started_simulator(seed: int = 1):
    simulator = FourPlayerSimulator(seed, json.dumps(_round()), [25_000] * 4)
    simulator.draw()
    state = json.loads(simulator.observation_json(0))
    legal = json.loads(simulator.legal_actions_json(0))
    return simulator, state, legal


def test_encoder_and_dynamic_policy_model_shapes() -> None:
    _, state, legal = _started_simulator()
    encoded = FeatureEncoder().encode(state, legal)
    assert encoded.planes.shape == (ENCODER_PLANES, 34)
    assert encoded.action_features.shape == (len(legal), ACTION_FEATURES)
    model = MahjongNetwork(ModelConfig(channels=8, residual_blocks=1, representation_dim=16, action_dim=8))
    output = model(
        torch.from_numpy(encoded.planes).unsqueeze(0),
        torch.from_numpy(encoded.action_features).unsqueeze(0),
    )
    assert output["policy_logits"].shape == (1, len(legal))
    assert output["rank_logits"].shape == (1, 4)


def test_observable_encoder_rejects_oracle_fields() -> None:
    _, state, legal = _started_simulator()
    state["private_hands"] = [[], [], [], []]
    with pytest.raises(ValueError, match="complete-state"):
        FeatureEncoder().encode(state, legal)


def test_unvalidated_model_is_not_presented_as_advice() -> None:
    _, state, _ = _started_simulator()
    model = MahjongNetwork(ModelConfig(channels=8, residual_blocks=1, representation_dim=16, action_dim=8))
    engine = InferenceEngine(
        model,
        ModelMetadata("test", FeatureEncoder.version, trained=True, validated=False, policy_calibrated=False),
    )
    analysis = engine.analyze(state)
    assert analysis["model_policy"]["status"] == "unavailable"


def test_reviewer_api_returns_exact_ranked_discards() -> None:
    _, state, _ = _started_simulator()
    response = TestClient(create_app()).post("/analyze", json={"state": state})
    assert response.status_code == 200
    body = response.json()
    assert body["shanten"]["status"] == "exact"
    assert body["discard_efficiency"]["value"]
    assert body["model_policy"]["status"] == "unavailable"
    assert body["safety_scope"] == "offline_saved_log_or_local_simulator_only"

    capabilities = TestClient(create_app()).get("/api/health").json()["replay_import"]
    assert "mahjong-ai-replay-bundle/v1" in capabilities["formats"]
    assert capabilities["direct_mahjong_soul_fetch"] is False


def test_local_hanchan_play_is_separate_from_analysis_and_hides_opponent_hands() -> None:
    torch.manual_seed(41)
    model = MahjongNetwork(
        ModelConfig(channels=8, residual_blocks=1, representation_dim=16, action_dim=8)
    )
    manager = LocalPlayManager(model, model_id="fixture-opponents")
    client = TestClient(create_app(local_play=manager))
    assert client.get("/health").json()["local_play_available"] is True

    started = client.post(
        "/local-play/sessions",
        json={"human_seat": 0, "seed": 9},
    )
    assert started.status_code == 200
    game = started.json()
    assert game["mode"] == "local_human_vs_ai"
    assert game["analysis_attached"] is False
    assert game["status"] == "awaiting_human_action"
    assert game["round"]["wind"] == "east"
    assert game["round"]["hand_number"] == 1
    assert game["legal_actions"]
    assert game["observation"]["information"]["opponent_hands_visible"] is False

    acted = client.post(
        f"/local-play/sessions/{game['session_id']}/actions",
        json={"action": game["legal_actions"][0]},
    )
    assert acted.status_code == 200
    advanced = acted.json()
    assert advanced["session_id"] == game["session_id"]
    assert advanced["analysis_attached"] is False
    for event in advanced["observation"]["event_history"]:
        if event["type"] == "draw" and event["player"] != 0:
            assert event["tile"] is None

    closed = client.delete(f"/local-play/sessions/{game['session_id']}")
    assert closed.status_code == 204
    assert client.get(f"/local-play/sessions/{game['session_id']}").status_code == 404


def test_realtime_lab_requires_handshake_and_orders_snapshots() -> None:
    _, state, _ = _started_simulator()
    with TestClient(create_app()).websocket_connect("/ws/lab") as websocket:
        websocket.send_json({"type": "analyze", "sequence": 0, "state": state})
        assert websocket.receive_json()["code"] == "handshake_required"
        websocket.send_json(
            {
                "type": "hello",
                "protocol": REALTIME_PROTOCOL,
                "session_id": "sim-test",
                "observer": 0,
                "source_kind": "local_simulator",
            }
        )
        assert websocket.receive_json()["type"] == "ready"
        websocket.send_json({"type": "analyze", "sequence": 10, "state": state})
        response = websocket.receive_json()
        assert response["type"] == "analysis"
        assert response["sequence"] == 10
        assert response["result"]["discard_efficiency"]["value"]
        assert response["result"]["safety_scope"] == "normalized_authorized_source_only"
        websocket.send_json({"type": "analyze", "sequence": 10, "state": state})
        assert websocket.receive_json()["code"] == "sequence_not_monotonic"


def test_realtime_lab_rejects_unconfirmed_adapter_and_observer_mismatch() -> None:
    _, state, _ = _started_simulator()
    client = TestClient(create_app())
    with client.websocket_connect("/ws/lab") as websocket:
        websocket.send_json(
            {
                "type": "hello",
                "protocol": REALTIME_PROTOCOL,
                "session_id": "adapter-test",
                "observer": 1,
                "source_kind": "authorized_external_adapter",
            }
        )
        assert websocket.receive_json()["code"] == "source_not_authorized"
        websocket.send_json(
            {
                "type": "hello",
                "protocol": REALTIME_PROTOCOL,
                "session_id": "adapter-test",
                "observer": 1,
                "source_kind": "authorized_external_adapter",
                "authorization": EXTERNAL_AUTHORIZATION,
            }
        )
        assert websocket.receive_json()["type"] == "ready"
        websocket.send_json({"type": "analyze", "sequence": 0, "state": state})
        assert websocket.receive_json()["code"] == "observer_mismatch"


def test_oracle_interface_is_explicit_and_distillation_detaches_teacher() -> None:
    teacher = OracleTeacher(ENCODER_PLANES, 8, hidden=16)
    policy, value = teacher(OracleBatch(torch.zeros(2, ENCODER_PLANES, 34), torch.zeros(2, 8, 34)))
    student_policy = torch.zeros_like(policy, requires_grad=True)
    student_value = torch.zeros_like(value, requires_grad=True)
    loss = distillation_loss(student_policy, student_value, policy, value)
    loss.backward()
    assert student_policy.grad is not None


def test_offline_template_parser_reports_tiles_and_confidence(tmp_path) -> None:
    templates = tmp_path / "templates"
    templates.mkdir()
    m1 = np.full((60, 40), 255, np.uint8)
    p1 = np.full((60, 40), 255, np.uint8)
    cv2.rectangle(m1, (8, 8), (31, 51), 0, 4)
    cv2.circle(p1, (20, 30), 15, 0, 4)
    cv2.imwrite(str(templates / "m1.png"), m1)
    cv2.imwrite(str(templates / "p1.png"), p1)
    frame = np.concatenate((m1, p1), axis=1)
    parser = ScreenshotParser(
        LayoutConfig(80, 60, {"hand": ((0, 0, 40, 60), (40, 0, 40, 60))}),
        TemplateTileClassifier(templates, minimum_confidence=0.8, minimum_margin=0.01),
    )
    result = parser.parse(frame)
    assert [tile["kind"] for tile in result.recognized["hand"]] == ["m1", "p1"]
    assert result.unknown_slots == ()
    assert result.state is None


def test_selfplay_worker_runs_against_rust_simulator() -> None:
    worker = SelfPlayWorker([RandomPolicy(seed) for seed in range(4)])
    transitions = worker.play_round(7, _round())
    assert transitions
    assert all(transition.done for transition in transitions)
    assert all(np.isfinite(transition.reward) for transition in transitions)


def test_parallel_model_selfplay_is_ordered_and_reproducible() -> None:
    torch.manual_seed(31)
    model = MahjongNetwork(
        ModelConfig(channels=8, residual_blocks=1, representation_dim=16, action_dim=8)
    )
    config = ModelSelfPlayConfig(rounds=2, seed=31, workers=2, first_round_index=5)
    first = ParallelModelSelfPlayCollector(config, _round()).collect(model)
    second = ParallelModelSelfPlayCollector(config, _round()).collect(model)
    assert first.workers_used == 2
    assert first.execution_backend in {"process", "thread_fallback"}
    assert [item.round_index for item in first.transitions] == sorted(
        item.round_index for item in first.transitions
    )
    assert [(item.action_index, item.old_log_probability) for item in first.transitions] == [
        (item.action_index, item.old_log_probability) for item in second.transitions
    ]
    assert {item.round_index for item in first.transitions} == {5, 6}


def test_parallel_selfplay_is_sharded_ordered_and_reproducible(tmp_path) -> None:
    config = ParallelSelfPlayConfig(rounds=4, seed=91, workers=2, shard_rounds=2)
    first = ParallelSelfPlayRunner(config, _round()).run(tmp_path / "first")
    second = ParallelSelfPlayRunner(config, _round()).run(tmp_path / "second")
    assert first["rounds"] == 4
    assert first["workers_used"] == 2
    assert len(first["shards"]) == 2
    assert first["transitions"] > 0
    assert [item["first_round_index"] for item in first["shards"]] == [0, 2]
    assert [item["last_round_index"] for item in first["shards"]] == [1, 3]
    assert [item["sha256"] for item in first["shards"]] == [
        item["sha256"] for item in second["shards"]
    ]
    table = pq.read_table(tmp_path / "first" / "part-00000.parquet")
    assert table.column("round_index").to_pylist() == sorted(table.column("round_index").to_pylist())
    assert set(table.column("round_seed").to_pylist()) == {91, 92}
    assert verify_selfplay_dataset(tmp_path / "first")["transitions"] == first["transitions"]


def test_selfplay_verifier_detects_tampered_shard(tmp_path) -> None:
    output = tmp_path / "dataset"
    ParallelSelfPlayRunner(
        ParallelSelfPlayConfig(rounds=1, seed=7, workers=1, shard_rounds=1),
        _round(),
    ).run(output)
    shard = output / "part-00000.parquet"
    shard.write_bytes(shard.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_selfplay_dataset(output)


def test_selfplay_ppo_updates_and_saves_resumable_checkpoint(tmp_path) -> None:
    torch.manual_seed(17)
    model = MahjongNetwork(
        ModelConfig(channels=8, residual_blocks=1, representation_dim=16, action_dim=8)
    )
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    config = PPOTrainerConfig(
        seed=17,
        iterations=1,
        rounds_per_iteration=1,
        update_epochs=1,
        batch_size=128,
        learning_rate=1e-3,
    )
    checkpoint = tmp_path / "ppo.pt"
    history = PPOTrainer(model, config).fit(checkpoint, "ppo-smoke", _round())
    assert checkpoint.exists()
    assert history[0]["transitions"] > 0
    assert all(np.isfinite(value) for value in history[0].values())
    assert any(
        not torch.equal(before[name], parameter.detach())
        for name, parameter in model.named_parameters()
    )
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert saved["metadata"]["trained"] is True
    assert saved["metadata"]["validated"] is False
    assert saved["training_state"]["algorithm"] == "selfplay_ppo_v1"
    resumed = PPOTrainer.resume(checkpoint, config)
    assert resumed.start_iteration == 1


def test_event_log_dataset_parquet_and_training_smoke(tmp_path) -> None:
    hand = [
        {"kind": kind, "red": False}
        for kind in ["m1", "m2", "m3", "m4", "m5", "m6", "m7", "p1", "p2", "p3", "s7", "east", "white"]
    ]
    events_by_observer = {}
    for observer in range(4):
        events_by_observer[str(observer)] = [
            {
                "sequence": 0,
                "type": "round_start",
                "round": _round(),
                "scores": [25_000] * 4,
                "initial_hand": hand,
                "dora_indicator": {"kind": "north", "red": False},
                "remaining_tiles": 70,
            },
            {
                "sequence": 1,
                "type": "draw",
                "player": 0,
                "tile": {"kind": "p9", "red": False} if observer == 0 else None,
                "rinshan": False,
            },
            {
                "sequence": 2,
                "type": "discard",
                "player": 0,
                "tile": {"kind": "p9", "red": False},
                "tsumogiri": True,
                "riichi_declaration": False,
            },
        ]
    document = {
        "format": "mahjong-ai-event-log/v1",
        "log_id": "fixture-a",
        "observations": events_by_observer,
        "result": {"final_scores": [30_000, 25_000, 24_000, 21_000]},
        "provenance": {"source": "synthetic_test", "converter_version": "test-v1"},
    }
    path = tmp_path / "log.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    log = load_event_log(path)
    records = build_decision_records(log)
    assert len(records) == 1
    records = [replace(records[0], log_id=f"fixture-{index}") for index in range(4)]
    parquet = tmp_path / "decisions.parquet"
    write_parquet(records, parquet)
    dataset = ParquetDecisionDataset(parquet)
    model = MahjongNetwork(ModelConfig(channels=8, residual_blocks=1, representation_dim=16, action_dim=8))
    trainer = Trainer(model, TrainerConfig(epochs=1, batch_size=2, learning_rate=1e-3))
    checkpoint = tmp_path / "model.pt"
    history = trainer.fit(
        torch.utils.data.Subset(dataset, [0, 1]),
        torch.utils.data.Subset(dataset, [2, 3]),
        checkpoint,
        "smoke",
        "fixture",
    )
    assert checkpoint.exists()
    assert history[0]["val_loss"] >= 0
    manifest_path = tmp_path / "human-manifest.json"
    create_human_dataset_manifest(
        manifest_path,
        [parquet],
        dataset_id="consented-fixture",
        source="synthetic_human_fixture",
        rights_basis="explicit_permission",
        authorization_reference="test-consent-record",
    )
    manifest = load_human_dataset_manifest(manifest_path)
    assert manifest.files == (parquet,)
    assert len(manifest.fingerprint) == 64
    warm_started = PPOTrainer.warm_start(
        checkpoint,
        PPOTrainerConfig(
            seed=3,
            iterations=1,
            rounds_per_iteration=1,
            update_epochs=1,
            batch_size=128,
        ),
    )
    assert warm_started.lineage["warm_start_metadata"]["dataset_fingerprint"] == "fixture"


def test_tenhou_offline_converter_anonymizes_and_builds_expert_discard() -> None:
    import xml.etree.ElementTree as ET

    root = ET.fromstring(
        '<mjloggm><GO type="169"/><INIT seed="0,0,0,1,2,100" ten="250,250,250,250" '
        'oya="0" hai0="0,1,2,3,4,5,6,7,8,9,10,11,12" '
        'hai1="20,21,22,23,24,25,26,27,28,29,30,31,32" '
        'hai2="40,41,42,43,44,45,46,47,48,49,50,51,53" '
        'hai3="60,61,62,63,64,65,66,67,68,69,70,71,72"/>'
        '<T52/><D52/><AGARI owari="260,10,250,0,240,-10,250,0"/></mjloggm>'
    )
    rounds = convert_document(root, "raw-identifying-log-id")
    assert len(rounds) == 1
    assert "raw-identifying" not in rounds[0].log.log_id
    records = build_decision_records(rounds[0].log)
    assert len(records) == 1
    assert json.loads(records[0].expert_action) == {
        "type": "discard",
        "tile": {"kind": "p5", "red": True},
        "tsumogiri": True,
    }


def test_web_replay_api_analyzes_uploaded_normalized_log() -> None:
    import xml.etree.ElementTree as ET

    root = ET.fromstring(
        '<mjloggm><GO type="169"/><INIT seed="0,0,0,1,2,100" ten="250,250,250,250" '
        'oya="0" hai0="0,1,2,3,4,5,6,7,8,9,10,11,12" '
        'hai1="20,21,22,23,24,25,26,27,28,29,30,31,32" '
        'hai2="40,41,42,43,44,45,46,47,48,49,50,51,53" '
        'hai3="60,61,62,63,64,65,66,67,68,69,70,71,72"/>'
        '<T52/><D52/><AGARI owari="260,10,250,0,240,-10,250,0"/></mjloggm>'
    )
    log = convert_document(root, "web-upload-fixture")[0].log
    document = {
        "format": "mahjong-ai-event-log/v1",
        "log_id": log.log_id,
        "rules": log.rules,
        "observations": {str(seat): events for seat, events in log.observations.items()},
        "result": {
            "final_scores": list(log.final_scores),
            "round_score_deltas": list(log.round_score_deltas),
        },
        "provenance": {"source": log.source, "converter_version": log.converter_version},
    }
    client = TestClient(create_app())
    response = client.post(
        "/api/replay-analysis",
        json={"log": document, "observer": 0},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["format"] == "riyu-replay-analysis/v1"
    assert body["analysisMode"] == "exact_tile_efficiency"
    assert body["rounds"][0]["turns"][0]["candidates"]
    assert body["rounds"][0]["turns"][0]["candidates"][0]["dealInRisk"] is None

    collection = client.post(
        "/api/replay-analysis",
        json={"logs": [document, {**document, "log_id": "web-upload-fixture-2"}], "observer": 0},
    )
    assert collection.status_code == 200
    collection_body = collection.json()
    assert len(collection_body["rounds"]) == 2
    assert collection_body["sourceLogIds"] == [
        log.log_id,
        "web-upload-fixture-2",
    ]

    bundle = {
        "format": "mahjong-ai-replay-bundle/v1",
        "source": {
            "platform": "mahjong_soul",
            "acquisition": "user_export",
            "authorization": "user_owned",
            "converter": {"name": "test-adapter", "version": "1"},
            "share_url": "https://game.mahjongsoul.com/?paipu=fixture-secret",
        },
        "logs": [document],
    }
    bundled = client.post(
        "/api/replay-analysis",
        json={"bundle": bundle, "observer": 0},
    )
    assert bundled.status_code == 200
    bundled_body = bundled.json()
    assert bundled_body["source"]["platform"] == "mahjong_soul"
    assert bundled_body["source"]["reference"].startswith("url-sha256:")
    assert "fixture-secret" not in str(bundled_body)
    assert bundled_body["decisionCount"] == 1
    assert bundled_body["rounds"][0]["honba"] == 0


def test_web_replay_api_requires_exactly_one_input_shape() -> None:
    client = TestClient(create_app())
    assert client.post("/api/replay-analysis", json={"observer": 0}).status_code == 422
    assert client.post(
        "/api/replay-analysis",
        json={"log": {}, "logs": [], "observer": 0},
    ).status_code == 422
    too_large = client.post(
        "/api/replay-analysis",
        content=b"{}",
        headers={"content-type": "application/json", "content-length": str(16 * 1024 * 1024 + 1)},
    )
    assert too_large.status_code == 413


def test_tenhou_nested_archive_streams_xml_without_extracting(tmp_path) -> None:
    import io
    import lzma
    import tarfile
    import zipfile

    xml = b'<mjloggm><GO type="169"/></mjloggm>'
    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w") as bundle:
        info = tarfile.TarInfo("2017/fixture.mjlog")
        info.size = len(xml)
        bundle.addfile(info, io.BytesIO(xml))
        readme = b"not xml"
        info = tarfile.TarInfo("README")
        info.size = len(readme)
        bundle.addfile(info, io.BytesIO(readme))
    archive = tmp_path / "logs.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as outer:
        outer.writestr("xml/2017.tar.xz", lzma.compress(tar_buffer.getvalue()))

    validate_tenhou_archive(archive)
    documents = list(iter_tenhou_archive_xml(archive))
    assert len(documents) == 1
    assert documents[0][0].endswith(":2017/fixture.mjlog")
    assert documents[0][1] == xml


def test_tenhou_archive_rejects_incomplete_zip(tmp_path) -> None:
    archive = tmp_path / "partial.zip"
    archive.write_bytes(b"PK\x03\x04incomplete")
    with pytest.raises(ValueError, match="incomplete or invalid ZIP"):
        validate_tenhou_archive(archive)


def test_tenhou_honor_kind_index_is_not_parsed_as_souzu() -> None:
    assert _kind_index({"kind": "south", "red": False}) == 28


def test_seat_balanced_arena_is_zero_for_identical_models(tmp_path) -> None:
    torch.manual_seed(7)
    model = MahjongNetwork(
        ModelConfig(channels=8, residual_blocks=1, representation_dim=16, action_dim=8)
    )
    checkpoint = tmp_path / "same.pt"
    save_checkpoint(
        checkpoint,
        model,
        ModelMetadata(
            model_id="same",
            encoder_version=FeatureEncoder.version,
            trained=True,
            validated=False,
            policy_calibrated=False,
        ),
    )
    result = evaluate_checkpoints(checkpoint, checkpoint, seed_blocks=1, base_seed=11, workers=1)
    assert result["config"]["matches"] == 4
    assert result["config"]["hands"] >= 32
    assert result["config"]["unit"] == "east_south_hanchan"
    assert result["paired"]["average_score_delta_difference_a_minus_b"] == 0.0
    assert result["paired"]["average_rank_difference_a_minus_b"] == 0.0
    assert result["verdict"] == "inconclusive"


def test_human_dataset_manifest_rejects_missing_ml_rights(tmp_path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "format": "mahjong-ai-human-dataset/v1",
                "dataset_id": "unsafe",
                "source": "unknown",
                "rights_basis": "open_license",
                "authorization_reference": "missing-ml-scope",
                "machine_learning_allowed": False,
                "contains_personal_identifiers": False,
                "files": [{"path": "missing.parquet", "sha256": "0" * 64}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="machine_learning_allowed"):
        load_human_dataset_manifest(manifest)


def test_human_dataset_manifest_requires_explicit_permission_for_service_logs(tmp_path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "format": "mahjong-ai-human-dataset/v1",
                "dataset_id": "restricted-source",
                "source": "Tenhou export",
                "rights_basis": "user_owned",
                "authorization_reference": "self-created account",
                "machine_learning_allowed": True,
                "contains_personal_identifiers": False,
                "files": [{"path": "missing.parquet", "sha256": "0" * 64}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="explicit_permission"):
        load_human_dataset_manifest(manifest)


def test_event_log_rejects_secret_opponent_draw(tmp_path) -> None:
    document = {
        "format": "mahjong-ai-event-log/v1",
        "observations": {str(i): [] for i in range(4)},
        "result": {"final_scores": [25_000] * 4},
    }
    document["observations"]["0"] = [
        {"sequence": 0, "type": "draw", "player": 1, "tile": {"kind": "m1", "red": False}}
    ]
    path = tmp_path / "leak.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(EventLogError, match="leaks"):
        load_event_log(path)
