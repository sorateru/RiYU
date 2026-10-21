"""Post-hoc calibration fitted only on the development validation split."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from .opponent import expected_calibration_error


@dataclass(frozen=True)
class BinaryCalibrationResult:
    scale: float
    bias: float
    examples: int
    positives: int
    before: dict[str, float]
    after: dict[str, float]


def binary_probability_metrics(logits: Tensor, targets: Tensor) -> dict[str, float]:
    logits = logits.detach().double().reshape(-1)
    targets = targets.detach().double().reshape(-1)
    if logits.shape != targets.shape or not logits.numel():
        raise ValueError("binary metrics need equally shaped non-empty tensors")
    probabilities = torch.sigmoid(logits)
    return {
        "brier": float((probabilities - targets).square().mean()),
        "log_loss": float(F.binary_cross_entropy_with_logits(logits, targets)),
        "ece_15": float(
            expected_calibration_error(probabilities.float(), targets.float(), 15)
        ),
    }


def fit_platt_scaling(logits: Tensor, targets: Tensor) -> BinaryCalibrationResult:
    """Fit monotone ``sigmoid(scale * logit + bias)`` calibration.

    A positive scale is enforced so calibration cannot reverse action ordering.
    Both classes are required; this prevents a rare-event validation shard from
    being incorrectly marked calibrated when it contains no deal-ins.
    """

    logits = logits.detach().double().cpu().reshape(-1)
    targets = targets.detach().double().cpu().reshape(-1)
    if logits.shape != targets.shape or not logits.numel():
        raise ValueError("calibration needs equally shaped non-empty tensors")
    positives = int(targets.sum().item())
    if positives == 0 or positives == targets.numel():
        raise ValueError("calibration needs both positive and negative examples")
    before = binary_probability_metrics(logits, targets)
    log_scale = torch.zeros((), dtype=torch.float64, requires_grad=True)
    bias = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        (log_scale, bias), lr=0.25, max_iter=100, tolerance_grad=1e-10,
        tolerance_change=1e-12, line_search_fn="strong_wolfe",
    )

    def closure() -> Tensor:
        optimizer.zero_grad(set_to_none=True)
        scale = log_scale.clamp(-4.0, 4.0).exp()
        calibrated = logits * scale + bias.clamp(-12.0, 12.0)
        # Weak regularization stabilizes very rare validation positives while
        # remaining negligible relative to a normally sized dataset.
        loss = F.binary_cross_entropy_with_logits(calibrated, targets)
        loss = loss + 1e-6 * (log_scale.square() + bias.square())
        loss.backward()
        return loss

    optimizer.step(closure)
    scale_value = float(log_scale.detach().clamp(-4.0, 4.0).exp())
    bias_value = float(bias.detach().clamp(-12.0, 12.0))
    after = binary_probability_metrics(logits * scale_value + bias_value, targets)
    return BinaryCalibrationResult(
        scale=scale_value,
        bias=bias_value,
        examples=targets.numel(),
        positives=positives,
        before=before,
        after=after,
    )
