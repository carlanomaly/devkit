from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import torch
from torch.utils.data import Dataset

from ..index import ScenarioIndex
from ._base import PathLike, check_parts_for, ensure_parts_for, required_parts, resolve_index
from .anomaly_lidar import AnomalyLiDARDataset
from .pointcloud import PointCloudDataset


class LiDARDataset(Dataset):
    """LiDAR point clouds with their anomaly labels and identifiers.

    Each item is a dict with the keys

    - ``"points"``: the item of :class:`PointCloudDataset`;
    - ``"anomaly_lidar"``: the item of :class:`AnomalyLiDARDataset`;
    - ``"scenario_id"``, ``"timesteps"``: the identifiers the evaluators need.

    Batch it with :func:`carlanomaly_collate_fn`.
    """

    def __init__(
        self,
        root: Optional[PathLike] = None,
        split: str = "train",
        *,
        transform: Optional[Callable] = None,
        index: Optional[ScenarioIndex] = None,
        download: bool = False,
        **index_kwargs: Any,
    ) -> None:
        parts = index_kwargs.pop("parts", None) or required_parts(
            [("pointcloud", None), ("anomaly_lidar", None)]
        )
        if download:
            ensure_parts_for(root, split, index, parts,
                             verify=index_kwargs.get("download_verify", True))
        elif index is None:
            check_parts_for(root, split, parts)
        index = resolve_index(root, split, index=index, download=False, **index_kwargs)
        self._index = index
        self.transform = transform

        # Parts already fetched above; children must not re-download.
        self.pointcloud = PointCloudDataset(index=index, download=False)
        self.anomaly_lidar = AnomalyLiDARDataset(index=index, download=False)

    @property
    def index(self) -> ScenarioIndex:
        """The :class:`ScenarioIndex` this dataset is built on (shareable)."""
        return self._index

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict:
        rec, _ = self._index[idx]
        item = {
            "scenario_id": str(rec.path),
            "timesteps": torch.tensor(self._index.timesteps_for(idx), dtype=torch.long),
            "points": self.pointcloud[idx],
            "anomaly_lidar": self.anomaly_lidar[idx],
        }
        if self.transform is not None:
            item = self.transform(item)
        return item
