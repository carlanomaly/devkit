from __future__ import annotations

import torch

from ._base import AtomicDataset


class ActionsDataset(AtomicDataset):
    """Per-timestep ego-vehicle control actions.

    Returns a ``FloatTensor (T, 7)`` with columns: throttle, steer, brake,
    hand_brake, reverse, manual_gear_shift, gear.  Boolean columns are cast
    to float.
    """

    modality = "actions"

    def __getitem__(self, idx: int) -> torch.Tensor:
        rec, _ = self._index[idx]
        timesteps = self._index.timesteps_for(idx)
        arr = self._read_feather_cached(rec, "actions")
        item = torch.from_numpy(arr[timesteps])  # (T, 7)
        return self._apply_transform(item)
