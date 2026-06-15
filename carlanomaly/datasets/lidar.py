from __future__ import annotations

from typing import Callable, Dict, Optional

from torch.utils.data import Dataset

from ..index import ScenarioIndex
from .anomaly_lidar import AnomalyLiDARDataset
from .pointcloud import PointCloudDataset


class LiDARDataset(Dataset):
    """Composite dataset for LiDAR point clouds with anomaly labels.

    Wraps :class:`PointCloudDataset` and :class:`AnomalyLiDARDataset`.

    Returns a dict with keys ``'points'`` (``List[DataFrame]``) and
    ``'anomaly_lidar'`` (``List[BoolTensor]``).
    """

    def __init__(
        self,
        index: ScenarioIndex,
        transform: Optional[Callable] = None,
    ) -> None:
        self._index = index
        self.transform = transform

        self.pointcloud = PointCloudDataset(index)
        self.anomaly_lidar = AnomalyLiDARDataset(index)

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict:
        item = {
            "points": self.pointcloud[idx],
            "anomaly_lidar": self.anomaly_lidar[idx],
        }
        if self.transform is not None:
            item = self.transform(item)
        return item
