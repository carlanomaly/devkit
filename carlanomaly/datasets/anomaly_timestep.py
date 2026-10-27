from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from ._base import AtomicDataset


class AnomalyTimestepDataset(AtomicDataset):
    """Timestep-level anomaly labels (one label per timestep).

    Returns a ``BoolTensor (T,)``.  Labels are stored on disk as
    ``anomaly-observation.feather``.  For the train split (where no label
    file exists), returns all-False.
    """

    modality = "anomaly_timestep"

    def __getitem__(self, idx: int) -> torch.Tensor:
        rec, _ = self._index[idx]
        timesteps = self._index.timesteps_for(idx)

        if self._is_train:
            item = torch.zeros(len(timesteps), dtype=torch.bool)
        else:
            cache = self._get_label_cache(rec)
            item = torch.from_numpy(cache[timesteps])

        return self._apply_transform(item)

    def _get_label_cache(self, rec) -> np.ndarray:
        self._ensure_cache()
        key = str(rec.path)
        if key not in self._feather_cache:
            self._feather_cache[key] = {}
        if "_timestep" not in self._feather_cache[key]:
            df = pd.read_feather(rec.path / "anomaly-observation.feather")
            self._feather_cache[key]["_timestep"] = df["anomaly"].values.astype(bool)
        return self._feather_cache[key]["_timestep"]
