from __future__ import annotations

from typing import List

import pandas as pd

from ._base import AtomicDataset


class PointCloudDataset(AtomicDataset):
    """LiDAR point clouds.

    Each item is a list of ``T`` DataFrames, one per timestep, with one row per
    point (about 100,000) and the columns

    - ``x``, ``y``, ``z``: position in metres in the sensor frame (x forward,
      y left, z up); the LiDAR is mounted with the front camera;
    - ``cos_inc_angle``: cosine of the angle at which the ray hits the surface;
    - ``object_id``: the object the point belongs to, 0 for static map geometry;
    - ``object_tag``: its semantic class, with the ids of
      :class:`SegmentationDataset`.

    Point clouds differ in size, so batch them with
    :func:`carlanomaly_collate_fn`.
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
