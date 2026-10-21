"""Observable discard-sequence encoder and multi-task Transformer model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from .encoder import ACTION_FEATURES, DANGER_FEATURES, ENCODER_PLANES
from .model import ResidualBlock
from mahjong_ai.schema import tile_index


SEQUENCE_ENCODER_VERSION = "observable-events-v2"
MAX_SEQUENCE_EVENTS = 96
EVENT_FLAG_FEATURES = 6
EVENT_TYPES = (
    "unknown",
    "round_start",
    "draw",
    "discard",
    "riichi",
    "chi",
    "pon",
    "kan",
    "dora_reveal",
    "pass",
    "ron",
    "tsumo",
    "draw_game",
    "round_end",
)


@dataclass(frozen=True)
class EncodedEventSequence:
    event_types: np.ndarray
    players: np.ndarray
    tiles: np.ndarray
    flags: np.ndarray
    valid_mask: np.ndarray


class SequenceFeatureEncoder:
    """Encode only public events; zero is reserved for padding/unknown values."""

    version = SEQUENCE_ENCODER_VERSION

    def __init__(self, maximum_events: int = MAX_SEQUENCE_EVENTS) -> None:
        if maximum_events <= 0:
            raise ValueError("maximum_events must be positive")
        self.maximum_events = maximum_events

    def encode_tokens(self, events: list[dict[str, Any]]) -> EncodedEventSequence:
        event_types = np.zeros(self.maximum_events, dtype=np.int64)
        players = np.zeros(self.maximum_events, dtype=np.int64)
        tiles = np.zeros(self.maximum_events, dtype=np.int64)
        flags = np.zeros((self.maximum_events, EVENT_FLAG_FEATURES), dtype=np.float32)
        valid = np.zeros(self.maximum_events, dtype=np.bool_)
        clipped = events[-self.maximum_events :]
        offset = self.maximum_events - len(clipped)
        for output_index, event in enumerate(clipped, start=offset):
            event_type = str(event.get("type", "unknown"))
            try:
                event_types[output_index] = EVENT_TYPES.index(event_type)
            except ValueError:
                event_types[output_index] = 0
            relative = event.get("relative_player")
            if relative is not None and 0 <= int(relative) < 4:
                players[output_index] = int(relative) + 1
            tile = event.get("tile")
            if tile is not None and 0 <= int(tile) < 34:
                tiles[output_index] = int(tile) + 1
            flags[output_index, 0] = float(bool(event.get("tsumogiri", False)))
            flags[output_index, 1] = float(bool(event.get("riichi_declaration", False)))
            flags[output_index, 2] = float(bool(event.get("rinshan", False)))
            meld_kind = event.get("meld_kind")
            flags[output_index, 3] = float(meld_kind == "chi")
            flags[output_index, 4] = float(meld_kind == "pon")
            flags[output_index, 5] = float(meld_kind in {"ankan", "daiminkan", "shouminkan"})
            valid[output_index] = True
        return EncodedEventSequence(event_types, players, tiles, flags, valid)


def public_event_tokens(
    state: dict[str, Any], maximum: int = MAX_SEQUENCE_EVENTS
) -> list[dict[str, Any]]:
    """Project runtime event history into the same observable token contract as V2."""

    observer = int(state["information"]["observer"])
    tokens: list[dict[str, Any]] = []
    for event in state.get("event_history", [])[-maximum:]:
        token: dict[str, Any] = {"type": event["type"]}
        player = event.get("player", event.get("winner", event.get("loser")))
        if player is not None:
            token["relative_player"] = (int(player) - observer) % 4
        tile = event.get("tile") or event.get("dora_indicator")
        if tile is not None:
            token["tile"] = tile_index(tile)
        token["tsumogiri"] = bool(event.get("tsumogiri", False))
        token["riichi_declaration"] = bool(event.get("riichi_declaration", False))
        token["rinshan"] = bool(event.get("rinshan", False))
        if event["type"] in {"chi", "pon", "kan"}:
            token["meld_kind"] = event["meld"]["kind"]
            token["meld_tiles"] = [tile_index(tile) for tile in event["meld"]["tiles"]]
        tokens.append(token)
    return tokens


@dataclass(frozen=True)
class SequenceModelConfig:
    snapshot_channels: int = 64
    snapshot_blocks: int = 3
    snapshot_dim: int = 192
    event_dim: int = 128
    transformer_heads: int = 4
    transformer_layers: int = 3
    transformer_feedforward: int = 384
    dropout: float = 0.1
    fused_dim: int = 256
    action_dim: int = 128
    maximum_events: int = MAX_SEQUENCE_EVENTS
    shanten_classes: int = 8
    decision_heads: bool = False
    opponent_dim: int = 128
    deal_in_score_bins: int = 8
    danger_features: int = DANGER_FEATURES

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class SequenceMahjongNetwork(nn.Module):
    """Policy/value model augmented with chronological public information.

    The opponent and future-draw heads are training-time oracle targets.  Their
    inputs remain strictly observable, so their predictions can safely guide a
    belief sampler at inference time.
    """

    def __init__(self, config: SequenceModelConfig = SequenceModelConfig()) -> None:
        super().__init__()
        if config.event_dim % config.transformer_heads:
            raise ValueError("event_dim must be divisible by transformer_heads")
        self.config = config
        self.snapshot_stem = nn.Sequential(
            nn.Conv1d(ENCODER_PLANES, config.snapshot_channels, 3, padding=1, bias=False),
            nn.BatchNorm1d(config.snapshot_channels),
            nn.ReLU(),
        )
        self.snapshot_backbone = nn.Sequential(
            *(ResidualBlock(config.snapshot_channels) for _ in range(config.snapshot_blocks))
        )
        self.snapshot_representation = nn.Sequential(
            nn.Flatten(),
            nn.Linear(config.snapshot_channels * 34, config.snapshot_dim),
            nn.ReLU(),
        )

        self.event_type_embedding = nn.Embedding(len(EVENT_TYPES), config.event_dim)
        self.player_embedding = nn.Embedding(5, config.event_dim)
        self.tile_embedding = nn.Embedding(35, config.event_dim)
        self.position_embedding = nn.Embedding(config.maximum_events, config.event_dim)
        self.flag_projection = nn.Linear(EVENT_FLAG_FEATURES, config.event_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=config.event_dim,
            nhead=config.transformer_heads,
            dim_feedforward=config.transformer_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.event_transformer = nn.TransformerEncoder(layer, config.transformer_layers)
        self.event_norm = nn.LayerNorm(config.event_dim)
        self.fusion = nn.Sequential(
            nn.Linear(config.snapshot_dim + config.event_dim, config.fused_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )

        self.action_encoder = nn.Sequential(
            nn.Linear(ACTION_FEATURES, config.action_dim),
            nn.ReLU(),
            nn.Linear(config.action_dim, config.action_dim),
        )
        self.policy_query = nn.Linear(config.fused_dim, config.action_dim)
        self.policy_bias = nn.Sequential(
            nn.Linear(ACTION_FEATURES, 64), nn.ReLU(), nn.Linear(64, 1)
        )
        self.rank_head = nn.Linear(config.fused_dim, 4)
        self.score_head = nn.Linear(config.fused_dim, 1)
        self.round_delta_head = nn.Linear(config.fused_dim, 4)
        self.opponent_tenpai_head = nn.Linear(config.fused_dim, 4)
        self.opponent_wait_head = nn.Linear(config.fused_dim, 4 * 34)
        self.opponent_shanten_head = nn.Linear(
            config.fused_dim, 4 * config.shanten_classes
        )
        self.future_draw_head = nn.Linear(config.fused_dim, 34)
        if config.decision_heads:
            self.opponent_seat_embedding = nn.Embedding(4, config.opponent_dim)
            self.opponent_projection = nn.Sequential(
                nn.Linear(config.fused_dim + config.opponent_dim, config.opponent_dim),
                nn.GELU(),
                nn.LayerNorm(config.opponent_dim),
            )
            self.deal_in_head = nn.Sequential(
                nn.Linear(
                    config.action_dim + config.opponent_dim + config.danger_features,
                    config.opponent_dim,
                ),
                nn.GELU(),
                nn.Linear(config.opponent_dim, 1),
            )
            self.deal_in_score_head = nn.Linear(
                config.opponent_dim, config.deal_in_score_bins
            )
            self.outcome_action_projection = nn.Linear(
                config.action_dim, config.fused_dim
            )
            self.action_outcome_head = nn.Sequential(
                nn.Linear(config.fused_dim * 2, config.fused_dim),
                nn.GELU(),
                nn.Linear(config.fused_dim, 5),
            )

    def forward(
        self,
        planes: Tensor,
        action_features: Tensor,
        event_types: Tensor,
        players: Tensor,
        tiles: Tensor,
        event_flags: Tensor,
        event_mask: Tensor,
        action_mask: Tensor | None = None,
        danger_features: Tensor | None = None,
    ) -> dict[str, Tensor]:
        snapshot = self.snapshot_representation(
            self.snapshot_backbone(self.snapshot_stem(planes))
        )
        positions = torch.arange(
            event_types.shape[1], device=event_types.device
        ).unsqueeze(0)
        events = (
            self.event_type_embedding(event_types)
            + self.player_embedding(players)
            + self.tile_embedding(tiles)
            + self.position_embedding(positions)
            + self.flag_projection(event_flags)
        )
        transformed = self.event_transformer(events, src_key_padding_mask=~event_mask)
        transformed = self.event_norm(transformed)
        weights = event_mask.unsqueeze(-1).to(transformed.dtype)
        sequence = (transformed * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        representation = self.fusion(torch.cat((snapshot, sequence), dim=1))

        action_embedding = self.action_encoder(action_features)
        query = self.policy_query(representation).unsqueeze(1)
        logits = (query * action_embedding).sum(dim=-1) / self.config.action_dim**0.5
        logits = logits + self.policy_bias(action_features).squeeze(-1)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, torch.finfo(logits.dtype).min)
        batch = planes.shape[0]
        output = {
            "policy_logits": logits,
            "rank_logits": self.rank_head(representation),
            "score_value": self.score_head(representation).squeeze(-1),
            "round_delta_values": self.round_delta_head(representation),
            "opponent_tenpai_logits": self.opponent_tenpai_head(representation),
            "opponent_wait_logits": self.opponent_wait_head(representation).view(batch, 4, 34),
            "opponent_shanten_logits": self.opponent_shanten_head(representation).view(
                batch, 4, self.config.shanten_classes
            ),
            "future_draw_logits": self.future_draw_head(representation),
        }
        if self.config.decision_heads:
            seats = torch.arange(4, device=planes.device)
            seat_features = self.opponent_seat_embedding(seats).unsqueeze(0).expand(
                batch, -1, -1
            )
            repeated = representation.unsqueeze(1).expand(-1, 4, -1)
            opponents = self.opponent_projection(
                torch.cat((repeated, seat_features), dim=-1)
            )
            if danger_features is None:
                danger_features = torch.zeros(
                    batch,
                    action_embedding.shape[1],
                    4,
                    self.config.danger_features,
                    device=planes.device,
                    dtype=planes.dtype,
                )
            if danger_features.shape != (
                batch, action_embedding.shape[1], 4, self.config.danger_features
            ):
                raise ValueError("danger_features shape does not match candidates/opponents")
            action_for_opponents = action_embedding.unsqueeze(2).expand(-1, -1, 4, -1)
            opponent_for_actions = opponents.unsqueeze(1).expand(
                -1, action_embedding.shape[1], -1, -1
            )
            output["deal_in_logits"] = self.deal_in_head(
                torch.cat(
                    (action_for_opponents, opponent_for_actions, danger_features), dim=-1
                )
            ).squeeze(-1)
            output["deal_in_score_logits"] = self.deal_in_score_head(opponents)
            outcome_action = self.outcome_action_projection(action_embedding)
            outcome_state = representation.unsqueeze(1).expand(-1, action_embedding.shape[1], -1)
            output["action_outcome_logits"] = self.action_outcome_head(
                torch.cat((outcome_state, outcome_action), dim=-1)
            )
            output["opponent_latents"] = opponents
        return output
