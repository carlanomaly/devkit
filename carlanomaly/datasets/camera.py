from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from torch.utils.data import Dataset

from ..index import ScenarioIndex
from ._base import PathLike, required_parts, resolve_index
from .anomaly_seg import AnomalySegmentationDataset
from .depth import DepthDataset
from .rgb import RGBDataset
from .segmentation import SegmentationDataset

#: Modalities a single-camera composite reads (used for ``download`` parts).
_CAMERA_MODALITIES = ("rgb", "depth", "segmentation", "anomaly_seg")


class CameraDataset(Dataset):
    """Composite dataset for a single camera direction.

    Wraps :class:`RGBDataset`, :class:`DepthDataset`,
    :class:`SegmentationDataset`, and :class:`AnomalySegmentationDataset`.

    Returns a dict with keys ``'rgb'``, ``'depth'``, ``'segmentation'``
    (sub-dict with ``'semantic'`` and ``'instance'``), and ``'anomaly_mask'``.

    Parameters
    ----------
    root:
        Path to the CarlAnomaly dataset directory.
    split:
        ``'train'``, ``'test_normal'``, ``'test_anomaly'``, or ``'test'``.
    direction:
        Camera direction (``'front'``, ``'left'``, ``'right'``, ``'rear'``).
    transform:
        Optional transform applied to the merged output dict.
    download:
        If ``True``, fetch the archive parts this composite needs (``base`` +
        ``depth``, plus ``camera-extended`` for non-front directions) into
        ``root`` before loading.

    Additional keyword arguments (``clip_len``, ``stride``, ``parts``, ...)
    are forwarded to :class:`~carlanomaly.index.ScenarioIndex`.
    """

    def __init__(
        self,
        root: Optional[PathLike] = None,
        split: str = "train",
        direction: str = "front",
        *,
        transform: Optional[Callable] = None,
        index: Optional[ScenarioIndex] = None,
        download: bool = False,
        **index_kwargs: Any,
    ) -> None:
        if download and index is None:
            index_kwargs.setdefault(
                "parts", required_parts((m, direction) for m in _CAMERA_MODALITIES)
            )
        index = resolve_index(root, split, index=index, download=download, **index_kwargs)
        self._index = index
        self.direction = direction
        self.transform = transform

        self.rgb = RGBDataset(direction=direction, index=index)
        self.depth = DepthDataset(direction=direction, index=index)
        self.segmentation = SegmentationDataset(direction=direction, index=index)
        self.anomaly_seg = AnomalySegmentationDataset(direction=direction, index=index)

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
