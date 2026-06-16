from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from ._base import AtomicDataset


class AnomalyObservationDataset(AtomicDataset):
    """Per-frame observation-level anomaly labels.

    Returns a ``BoolTensor (T,)``.  For the train split (where no label
    file exists), returns all-False.
    """

    modality = "anomaly_obs"

    def __getitem__(self, idx: int) -> torch.Tensor:
        rec, _ = self._index[idx]
        frames = self._index.frames_for(idx)

        if self._is_train:
            item = torch.zeros(len(frames), dtype=torch.bool)
        else:
            cache = self._get_obs_cache(rec)
            item = torch.from_numpy(cache[frames])

        return self._apply_transform(item)

    def _get_obs_cache(self, rec) -> np.ndarray:
        self._ensure_cache()
        key = str(rec.path)
        if key not in self._feather_cache:
            self._feather_cache[key] = {}
        if "_obs" not in self._feather_cache[key]:
            df = pd.read_feather(rec.path / "anomaly-observation.feather")
            self._feather_cache[key]["_obs"] = df["anomaly"].values.astype(bool)
        return self._feather_cache[key]["_obs"]
