from __future__ import annotations

from typing import Callable, Optional

import torch

from ..index import ScenarioIndex
from ._base import AtomicDataset


class ActionsDataset(AtomicDataset):
    """Per-frame ego-vehicle control actions.

    Returns a ``FloatTensor (T, 7)`` with columns: throttle, steer, brake,
    hand_brake, reverse, manual_gear_shift, gear.  Boolean columns are cast
    to float.
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
        arr = self._read_feather_cached(rec, "actions")
        item = torch.from_numpy(arr[frames])  # (T, 7)
        return self._apply_transform(item)
