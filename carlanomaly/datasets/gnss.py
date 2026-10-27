from __future__ import annotations

import torch

from ._base import AtomicDataset


class GNSSDataset(AtomicDataset):
    """Per-timestep GNSS position.

    Returns a ``FloatTensor (T, 3)`` with columns: altitude, latitude, longitude.
    """

    modality = "gnss"

    def __getitem__(self, idx: int) -> torch.Tensor:
        rec, _ = self._index[idx]
        timesteps = self._index.timesteps_for(idx)
        arr = self._read_feather_cached(rec, "gnss")
        item = torch.from_numpy(arr[timesteps])  # (T, 3)
        return self._apply_transform(item)
