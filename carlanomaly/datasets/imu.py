from __future__ import annotations

import torch

from ._base import AtomicDataset


class IMUDataset(AtomicDataset):
    """IMU readings of the ego vehicle.

    Each item is a ``FloatTensor (T, 7)`` with the columns

    - ``acceleration_x``, ``acceleration_y``, ``acceleration_z``: in m/s²;
    - ``compass``: heading in radians, clockwise from north;
    - ``longitude_x``, ``longitude_y``, ``longitude_z``: despite their name,
      the angular velocity around x, y and z in rad/s (the gyroscope).
    """

    modality = "imu"

    def __getitem__(self, idx: int) -> torch.Tensor:
        rec, _ = self._index[idx]
        timesteps = self._index.timesteps_for(idx)
        arr = self._read_feather_cached(rec, "imu")
        item = torch.from_numpy(arr[timesteps])  # (T, 7)
        return self._apply_transform(item)
