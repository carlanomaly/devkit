from __future__ import annotations

import torch

from ._base import AtomicDataset


class ActionsDataset(AtomicDataset):
    """Control commands of the ego vehicle's autopilot.

    Each item is a ``FloatTensor (T, 7)`` with the columns ``throttle`` and
    ``brake`` (0 to 1), ``steer`` (-1 to 1), ``hand_brake``, ``reverse``,
    ``manual_gear_shift`` (0 or 1) and ``gear``. The command of timestep ``t``
    is the one applied after the sensors recorded it.
    """

    modality = "actions"

    def __getitem__(self, idx: int) -> torch.Tensor:
        rec, _ = self._index[idx]
        timesteps = self._index.timesteps_for(idx)
        arr = self._read_feather_cached(rec, "actions")
        item = torch.from_numpy(arr[timesteps])  # (T, 7)
        return self._apply_transform(item)
