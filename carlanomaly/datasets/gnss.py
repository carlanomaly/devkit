from __future__ import annotations

import torch

from ._base import AtomicDataset


class GNSSDataset(AtomicDataset):
    """Per-frame GNSS position.

    Returns a ``FloatTensor (T, 3)`` with columns: altitude, latitude, longitude.
    """

    modality = "gnss"

    def __getitem__(self, idx: int) -> torch.Tensor:
        rec, _ = self._index[idx]
        frames = self._index.frames_for(idx)
        arr = self._read_feather_cached(rec, "gnss")
        item = torch.from_numpy(arr[frames])  # (T, 3)
        return self._apply_transform(item)
