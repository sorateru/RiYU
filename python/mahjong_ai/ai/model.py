"""Residual policy/value network with dynamic legal-action scoring."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import torch
from torch import Tensor, nn

from .encoder import ACTION_FEATURES, ENCODER_PLANES


@dataclass(frozen=True)
class ModelConfig:
    channels: int = 96
    residual_blocks: int = 4
    representation_dim: int = 256
    action_dim: int = 128
    auxiliary_heads: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.activation = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(x + self.layers(x))


class MahjongNetwork(nn.Module):
    def __init__(self, config: ModelConfig = ModelConfig()) -> None:
        super().__init__()
        self.config = config
        self.stem = nn.Sequential(
            nn.Conv1d(ENCODER_PLANES, config.channels, 3, padding=1, bias=False),
            nn.BatchNorm1d(config.channels),
            nn.ReLU(),
        )
        self.backbone = nn.Sequential(
            *(ResidualBlock(config.channels) for _ in range(config.residual_blocks))
        )
        self.representation = nn.Sequential(
            nn.Flatten(),
            nn.Linear(config.channels * 34, config.representation_dim),
            nn.ReLU(),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(ACTION_FEATURES, config.action_dim), nn.ReLU(), nn.Linear(config.action_dim, config.action_dim)
        )
        self.policy_query = nn.Linear(config.representation_dim, config.action_dim)
        self.policy_bias = nn.Sequential(nn.Linear(ACTION_FEATURES, 64), nn.ReLU(), nn.Linear(64, 1))
        self.rank_head = nn.Linear(config.representation_dim, 4)
        self.score_head = nn.Linear(config.representation_dim, 1)
        self.auxiliary = nn.ModuleDict(
            {name: nn.Linear(config.representation_dim, 1) for name in config.auxiliary_heads}
        )

    def forward(
        self,
        planes: Tensor,
        action_features: Tensor,
        action_mask: Tensor | None = None,
    ) -> dict[str, Tensor | dict[str, Tensor]]:
        representation = self.representation(self.backbone(self.stem(planes)))
        action_embedding = self.action_encoder(action_features)
        query = self.policy_query(representation).unsqueeze(1)
        logits = (query * action_embedding).sum(dim=-1) / self.config.action_dim**0.5
        logits = logits + self.policy_bias(action_features).squeeze(-1)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, torch.finfo(logits.dtype).min)
        return {
            "policy_logits": logits,
            "rank_logits": self.rank_head(representation),
            "score_value": self.score_head(representation).squeeze(-1),
            "auxiliary": {name: head(representation).squeeze(-1) for name, head in self.auxiliary.items()},
        }
