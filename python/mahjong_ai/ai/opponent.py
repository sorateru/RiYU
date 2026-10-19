"""Observable-only opponent model and calibrated probability metrics."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .encoder import ENCODER_PLANES


class OpponentModel(nn.Module):
    def __init__(self, hidden: int = 128, score_bins: int = 6) -> None:
        super().__init__()
        self.score_bins = score_bins
        self.encoder = nn.Sequential(
            nn.Conv1d(ENCODER_PLANES, hidden, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, 3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(hidden * 34, hidden),
            nn.ReLU(),
        )
        self.tenpai = nn.Linear(hidden, 4)
        self.waits = nn.Linear(hidden, 4 * 34)
        self.score = nn.Linear(hidden, 4 * score_bins)

    def forward(self, observable_planes: Tensor) -> dict[str, Tensor]:
        representation = self.encoder(observable_planes)
        return {
            "tenpai_logits": self.tenpai(representation),
            "wait_logits": self.waits(representation).reshape(-1, 4, 34),
            "score_logits": self.score(representation).reshape(-1, 4, self.score_bins),
        }


def probability_metrics(logits: Tensor, targets: Tensor) -> dict[str, Tensor]:
    probabilities = torch.sigmoid(logits)
    return {
        "brier": ((probabilities - targets) ** 2).mean(),
        "log_loss": F.binary_cross_entropy_with_logits(logits, targets),
    }
