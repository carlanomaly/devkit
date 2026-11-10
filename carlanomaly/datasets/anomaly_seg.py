from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
import torch
from PIL import Image

from ..index import ScenarioIndex
from ._base import AtomicDataset, PathLike


class AnomalySegmentationDataset(AtomicDataset):
    """Anomaly masks for one camera: which pixels show the anomaly.

    Each item is a ``BoolTensor (T, 1080, 1920)``, ``True`` on the pixels of the
    anomalous object. Timesteps without a visible anomaly, normal scenarios, the
    train split, and anomaly types without masks give all-``False`` masks.

    These are the labels :class:`~carlanomaly.PixelEvaluator` uses.
    """

    modality = "anomaly_seg"

    # Native resolution used when no mask file exists.
    _DEFAULT_H = 1080
    _DEFAULT_W = 1920

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

    def __getitem__(self, idx: int) -> torch.Tensor:
        rec, _ = self._index[idx]
        timesteps = self._index.timesteps_for(idx)
        masks = []
        for f in timesteps:
            if self._is_train:
                masks.append(torch.zeros(
                    self._DEFAULT_H, self._DEFAULT_W, dtype=torch.bool
                ))
            else:
                path = rec.path / f"anomaly-{self.direction}" / f"{f:06d}.png"
                if path.exists():
                    arr = np.array(Image.open(path).convert("L"))
                    masks.append(torch.from_numpy(arr > 0))
                else:
                    masks.append(torch.zeros(
                        self._DEFAULT_H, self._DEFAULT_W, dtype=torch.bool
                    ))
        item = torch.stack(masks)  # (T, H, W)
        return self._apply_transform(item)
