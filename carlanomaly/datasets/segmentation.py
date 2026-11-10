from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
from PIL import Image

from ..index import ScenarioIndex
from ._base import AtomicDataset, PathLike


class SegmentationDataset(AtomicDataset):
    """Semantic and instance segmentation from one camera.

    Each item is a dict of two ``LongTensor (T, 1080, 1920)``:

    - ``"semantic"``: the CARLA class of each pixel, e.g. 1 road, 7 traffic
      light, 14 car (`class table <https://carlanomaly.de/format/#segmentation>`_).
    - ``"instance"``: the object each pixel belongs to. The ids match the
      ``object_id`` of the LiDAR points (:class:`PointCloudDataset`).

    The segmentation is rendered by the simulator, so it is exact. Use it to
    train a segmentation model on the train split, a common basis for
    anomaly detectors.
    """

    modality = "segmentation"

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
        super().__init__(
            root, split, direction=direction, transform=transform,
            index=index, download=download, **index_kwargs,
        )

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec, _ = self._index[idx]
        timesteps = self._index.timesteps_for(idx)
        semantic_list, instance_list = [], []
        for f in timesteps:
            path = rec.path / f"segmentation-{self.direction}" / f"{f:06d}.png"
            arr = np.array(Image.open(path).convert("RGBA"))  # (H, W, 4)
            r = arr[:, :, 0].astype(np.int64)
            r[r > 28] = 0  # rendering artifacts at object boundaries
            semantic_list.append(torch.from_numpy(r))
            # B is the high byte, G the low byte
            inst = arr[:, :, 2].astype(np.int64) * 256 + arr[:, :, 1].astype(np.int64)
            instance_list.append(torch.from_numpy(inst))
        item = {
            "semantic": torch.stack(semantic_list),   # (T, H, W)
            "instance": torch.stack(instance_list),   # (T, H, W)
        }
        return self._apply_transform(item)
