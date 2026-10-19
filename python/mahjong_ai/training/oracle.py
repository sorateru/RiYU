"""Type-separated oracle teacher and observable-student distillation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class OracleBatch:
    observable_planes: Tensor
    private_planes: Tensor


class OracleTeacher(nn.Module):
    def __init__(self, observable_planes: int, private_planes: int, hidden: int = 256) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear((observable_planes + private_planes) * 34, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.policy = nn.Linear(hidden, 128)
        self.value = nn.Linear(hidden, 4)

    def forward(self, batch: OracleBatch) -> tuple[Tensor, Tensor]:
        merged = torch.cat((batch.observable_planes, batch.private_planes), dim=1)
        representation = self.network(merged)
        return self.policy(representation), self.value(representation)


def distillation_loss(
    student_policy_logits: Tensor,
    student_value_logits: Tensor,
    teacher_policy_logits: Tensor,
    teacher_value_logits: Tensor,
    temperature: float = 2.0,
) -> Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    teacher_policy = F.softmax(teacher_policy_logits.detach() / temperature, dim=-1)
    teacher_value = F.softmax(teacher_value_logits.detach() / temperature, dim=-1)
    policy_loss = F.kl_div(
        F.log_softmax(student_policy_logits / temperature, dim=-1),
        teacher_policy,
        reduction="batchmean",
    )
    value_loss = F.kl_div(
        F.log_softmax(student_value_logits / temperature, dim=-1),
        teacher_value,
        reduction="batchmean",
    )
    return (policy_loss + value_loss) * temperature**2
