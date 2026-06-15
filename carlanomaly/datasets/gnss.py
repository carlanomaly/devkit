from __future__ import annotations

from typing import Callable, Optional

import torch

from ..index import ScenarioIndex
from ._base import AtomicDataset


class GNSSDataset(AtomicDataset):
    """Per-frame GNSS position.

    Returns a ``FloatTensor (T, 3)`` with columns: altitude, latitude, longitude.
    """

    def __init__(
        self,
        index: ScenarioIndex,
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(index, transform)

    def __getitem__(self, idx: int) -> torch.Tensor:
        rec, _ = self._index[idx]
        frames = self._index.frames_for(idx)
        arr = self._read_feather_cached(rec, "gnss")
        item = torch.from_numpy(arr[frames])  # (T, 3)
        return self._apply_transform(item)
