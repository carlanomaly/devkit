from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from torch.utils.data import Dataset

from ..index import ScenarioIndex
from ._base import PathLike, required_parts, resolve_index
from .anomaly_lidar import AnomalyLiDARDataset
from .pointcloud import PointCloudDataset


class LiDARDataset(Dataset):
    """Composite dataset for LiDAR point clouds with anomaly labels.

    Wraps :class:`PointCloudDataset` and :class:`AnomalyLiDARDataset`.

    Returns a dict with keys ``'points'`` (``List[DataFrame]``) and
    ``'anomaly_lidar'`` (``List[BoolTensor]``).

    Parameters
    ----------
    root:
        Path to the CarlAnomaly dataset directory.
    split:
        ``'train'``, ``'test_normal'``, ``'test_anomaly'``, or ``'test'``.
    transform:
        Optional transform applied to the merged output dict.
    download:
        If ``True``, fetch the archive parts this composite needs (``base`` +
        ``lidar``) into ``root`` before loading.

    Additional keyword arguments (``clip_len``, ``stride``, ``parts``, ...)
    are forwarded to :class:`~carlanomaly.index.ScenarioIndex`.
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
        if download and index is None:
            index_kwargs.setdefault(
                "parts", required_parts([("pointcloud", None), ("anomaly_lidar", None)])
            )
        index = resolve_index(root, split, index=index, download=download, **index_kwargs)
        self._index = index
        self.transform = transform

        self.pointcloud = PointCloudDataset(index=index)
        self.anomaly_lidar = AnomalyLiDARDataset(index=index)

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
