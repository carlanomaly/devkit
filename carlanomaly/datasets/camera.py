from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import torch
from torch.utils.data import Dataset

from ..index import ScenarioIndex
from ._base import PathLike, check_parts_for, ensure_parts_for, required_parts, resolve_index
from .anomaly_seg import AnomalySegmentationDataset
from .depth import DepthDataset
from .rgb import RGBDataset
from .segmentation import SegmentationDataset

#: Modalities a single-camera composite reads (used for ``download`` parts).
_CAMERA_MODALITIES = ("rgb", "depth", "segmentation", "anomaly_seg")


class CameraDataset(Dataset):
    """Everything one camera records, with its anomaly mask and identifiers.

    Each item is a dict with the keys

    - ``"rgb"``, ``"depth"``, ``"segmentation"``, ``"anomaly_mask"``: the items
      of :class:`RGBDataset`, :class:`DepthDataset`,
      :class:`SegmentationDataset` and :class:`AnomalySegmentationDataset`;
    - ``"scenario_id"``, ``"timesteps"``: the identifiers the evaluators need.

    Needs the ``depth`` part in addition to the camera's own.

    Example
    -------
    >>> cam = CameraDataset("/data/carlanomaly", split="test", direction="front")
    >>> sorted(cam[0])
    ['anomaly_mask', 'depth', 'rgb', 'scenario_id', 'segmentation', 'timesteps']
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
        parts = index_kwargs.pop("parts", None) or required_parts(
            (m, direction) for m in _CAMERA_MODALITIES
        )
        if download:
            ensure_parts_for(root, split, index, parts,
                             verify=index_kwargs.get("download_verify", True))
        elif index is None:
            check_parts_for(root, split, parts)
        index = resolve_index(root, split, index=index, download=False, **index_kwargs)
        self._index = index
        self.direction = direction
        self.transform = transform

        # Parts already fetched above; children must not re-download.
        self.rgb = RGBDataset(direction=direction, index=index, download=False)
        self.depth = DepthDataset(direction=direction, index=index, download=False)
        self.segmentation = SegmentationDataset(direction=direction, index=index, download=False)
        self.anomaly_seg = AnomalySegmentationDataset(direction=direction, index=index, download=False)

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
            "rgb": self.rgb[idx],
            "depth": self.depth[idx],
            "segmentation": self.segmentation[idx],
            "anomaly_mask": self.anomaly_seg[idx],
        }
        if self.transform is not None:
            item = self.transform(item)
        return item
