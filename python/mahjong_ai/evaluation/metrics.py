"""Aggregate arena and prediction metrics with uncertainty."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class ArenaMetrics:
    ranks: list[int] = field(default_factory=list)
    rank_points: list[float] = field(default_factory=list)
    wins: int = 0
    deal_ins: int = 0
    hands: int = 0
    win_points: list[int] = field(default_factory=list)
    deal_in_points: list[int] = field(default_factory=list)

    def summary(self) -> dict[str, float | int | tuple[float, float]]:
        if not self.ranks:
            raise ValueError("arena metrics are empty")
        return {
            "games": len(self.ranks),
            "average_rank": sum(self.ranks) / len(self.ranks),
            "average_rank_95ci": confidence_interval([float(rank) for rank in self.ranks]),
            "first_rate": self.ranks.count(1) / len(self.ranks),
            "fourth_rate": self.ranks.count(4) / len(self.ranks),
            "average_rank_points": _mean(self.rank_points),
            "win_rate": self.wins / self.hands if self.hands else 0.0,
            "deal_in_rate": self.deal_ins / self.hands if self.hands else 0.0,
            "average_win_points": _mean([float(x) for x in self.win_points]),
            "average_deal_in_points": _mean([float(x) for x in self.deal_in_points]),
        }


def confidence_interval(values: list[float], z: float = 1.96) -> tuple[float, float]:
    if not values:
        raise ValueError("confidence interval needs observations")
    mean = sum(values) / len(values)
    if len(values) == 1:
        return (mean, mean)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    half_width = z * math.sqrt(variance / len(values))
    return (mean - half_width, mean + half_width)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
