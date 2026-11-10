from __future__ import annotations

import torch

from ._base import AtomicDataset


class AnomalyScenarioDataset(AtomicDataset):
    """Whether an item's scenario is an anomaly scenario.

    Each item is a scalar ``BoolTensor``, ``True`` for scenarios under
    ``test/anomaly``. Unlike the other datasets, it has no time dimension. These
    are the labels :class:`~carlanomaly.ScenarioEvaluator` uses.
    """

    modality = "anomaly_scenario"

    def __getitem__(self, idx: int) -> torch.Tensor:
        rec, _ = self._index[idx]
        item = torch.tensor(rec.split == "test_anomaly")
        return self._apply_transform(item)
