from __future__ import annotations

from typing import List

import pandas as pd

from ._base import AtomicDataset


class PointCloudDataset(AtomicDataset):
    """Per-timestep LiDAR point clouds.

    Returns a ``List[pd.DataFrame]`` of length T.  Each DataFrame has
    columns: x, y, z, cos_inc_angle, object_id, object_tag.

    With ``download=True`` the ``lidar`` part (plus ``base``) is fetched
    automatically.
    """

    modality = "pointcloud"

    def __getitem__(self, idx: int) -> List[pd.DataFrame]:
        rec, _ = self._index[idx]
        timesteps = self._index.timesteps_for(idx)
        clouds = []
        for f in timesteps:
            path = rec.path / "pointclouds" / f"{f:06d}.feather"
            clouds.append(pd.read_feather(path))
        return self._apply_transform(clouds)
