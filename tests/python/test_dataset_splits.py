from __future__ import annotations

from torch.utils.data import Dataset, Subset

from mahjong_ai.training.sequence_supervised import (
    split_sequence_train_validation_holdout,
)
from mahjong_ai.training.supervised import split_train_validation_holdout


class _Rows(Dataset[dict[str, str]]):
    def __init__(self) -> None:
        self.rows = [
            {"log_id": f"game-{game:04d}-r{round_index:02d}"}
            for game in range(200)
            for round_index in range(1, 5)
        ]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, str]:
        return self.rows[index]


def _groups(dataset: _Rows, subset: Subset) -> set[str]:
    return {
        dataset.rows[index]["log_id"].rsplit("-r", 1)[0]
        for index in subset.indices
    }


def test_three_way_splits_keep_hanchan_out_of_final_holdout_selection() -> None:
    dataset = _Rows()
    for split in (
        split_train_validation_holdout,
        split_sequence_train_validation_holdout,
    ):
        train, validation, holdout = split(dataset, 0.15, 0.15)
        groups = [_groups(dataset, subset) for subset in (train, validation, holdout)]
        assert all(groups)
        assert groups[0].isdisjoint(groups[1])
        assert groups[0].isdisjoint(groups[2])
        assert groups[1].isdisjoint(groups[2])
        assert sum(len(subset) for subset in (train, validation, holdout)) == len(dataset)
