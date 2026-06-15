from __future__ import annotations

from typing import Callable, Dict, Optional

from torch.utils.data import Dataset

from ..index import ScenarioIndex
from .anomaly_seg import AnomalySegmentationDataset
from .depth import DepthDataset
from .rgb import RGBDataset
from .segmentation import SegmentationDataset


class CameraDataset(Dataset):
    """Composite dataset for a single camera direction.

    Wraps :class:`RGBDataset`, :class:`DepthDataset`,
    :class:`SegmentationDataset`, and :class:`AnomalySegmentationDataset`.

    Returns a dict with keys ``'rgb'``, ``'depth'``, ``'segmentation'``
    (sub-dict with ``'semantic'`` and ``'instance'``), and ``'anomaly_mask'``.
    """

    def __init__(
        self,
        index: ScenarioIndex,
        direction: str = "front",
        transform: Optional[Callable] = None,
    ) -> None:
        self._index = index
        self.direction = direction
        self.transform = transform

        self.rgb = RGBDataset(index, direction)
        self.depth = DepthDataset(index, direction)
        self.segmentation = SegmentationDataset(index, direction)
        self.anomaly_seg = AnomalySegmentationDataset(index, direction)

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict:
        item = {
            "rgb": self.rgb[idx],
            "depth": self.depth[idx],
            "segmentation": self.segmentation[idx],
            "anomaly_mask": self.anomaly_seg[idx],
        }
        if self.transform is not None:
            item = self.transform(item)
        return item
