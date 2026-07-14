from __future__ import annotations

import torch

from ._base import AtomicDataset


class AnomalyScenarioDataset(AtomicDataset):
    """Scenario-level anomaly labels (one label per scenario).

    Returns a scalar ``BoolTensor``: ``True`` iff the item's scenario is an
    anomaly scenario.  The label is derived from the scenario's location in
    the dataset layout (``test/anomaly/`` vs ``test/normal/``); no label file
    is read.  Unlike the per-timestep datasets, the returned tensor has no
    time dimension: the label applies to the scenario as a whole.
    """

    modality = "anomaly_scenario"

    def __getitem__(self, idx: int) -> torch.Tensor:
        rec, _ = self._index[idx]
        item = torch.tensor(rec.split == "test_anomaly")
        return self._apply_transform(item)
