from __future__ import annotations

from typing import Callable, List, Optional

import pandas as pd

from ..index import ScenarioIndex
from ._base import AtomicDataset


class PointCloudDataset(AtomicDataset):
    """Per-frame LiDAR point clouds.

    Returns a ``List[pd.DataFrame]`` of length T.  Each DataFrame has
    columns: x, y, z, cos_inc_angle, object_id, object_tag.
    """

    def __init__(
        self,
        index: ScenarioIndex,
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(index, transform)

    def __getitem__(self, idx: int) -> List[pd.DataFrame]:
        rec, _ = self._index[idx]
        frames = self._index.frames_for(idx)
        clouds = []
        for f in frames:
            path = rec.path / "pointclouds" / f"{f:06d}.feather"
            clouds.append(pd.read_feather(path))
        return self._apply_transform(clouds)
