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


def probability_metrics(
    logits: Tensor, targets: Tensor, calibration_bins: int = 15
) -> dict[str, Tensor]:
    if calibration_bins <= 0:
        raise ValueError("calibration_bins must be positive")
    probabilities = torch.sigmoid(logits)
    return {
        "brier": ((probabilities - targets) ** 2).mean(),
        "log_loss": F.binary_cross_entropy_with_logits(logits, targets),
        "calibration_error": expected_calibration_error(
            probabilities, targets, calibration_bins
        ),
    }


def expected_calibration_error(
    probabilities: Tensor, targets: Tensor, bins: int = 15
) -> Tensor:
    """Weighted absolute calibration gap for binary probability estimates."""

    if bins <= 0:
        raise ValueError("bins must be positive")
    if probabilities.shape != targets.shape:
        raise ValueError("probabilities and targets must have the same shape")
    flattened_probabilities = probabilities.float().reshape(-1)
    flattened_targets = targets.float().reshape(-1)
    if flattened_probabilities.numel() == 0:
        raise ValueError("calibration error needs observations")
    if torch.any((flattened_probabilities < 0.0) | (flattened_probabilities > 1.0)):
        raise ValueError("probabilities must be in [0, 1]")
    boundaries = torch.linspace(
        0.0,
        1.0,
        bins + 1,
        device=flattened_probabilities.device,
        dtype=flattened_probabilities.dtype,
    )
    error = torch.zeros((), device=probabilities.device, dtype=torch.float32)
    for index in range(bins):
        lower = boundaries[index]
        upper = boundaries[index + 1]
        mask = (flattened_probabilities >= lower) & (
            flattened_probabilities <= upper
            if index == bins - 1
            else flattened_probabilities < upper
        )
        count = mask.sum()
        if count:
            confidence = flattened_probabilities[mask].mean()
            frequency = flattened_targets[mask].mean()
            error = error + count.float() / flattened_probabilities.numel() * torch.abs(
                confidence - frequency
            )
    return error
