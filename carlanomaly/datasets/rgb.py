from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
import torch
from PIL import Image

from ..index import ScenarioIndex
from ._base import AtomicDataset, PathLike


class RGBDataset(AtomicDataset):
    """RGB images from one camera.

    Each item is a ``FloatTensor (T, 3, 1080, 1920)`` with channels in RGB order
    and values in ``[0, 1]``.

    Example
    -------
    >>> rgb = RGBDataset("/data/carlanomaly", split="train", direction="left")
    >>> rgb[0].shape
    torch.Size([1, 3, 1080, 1920])
    """

    modality = "rgb"

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
        images = []
        for f in timesteps:
            path = rec.path / f"rgb-{self.direction}" / f"{f:06d}.jpg"
            img = Image.open(path).convert("RGB")
            arr = np.array(img, dtype=np.float32) / 255.0  # (H, W, 3)
            images.append(torch.from_numpy(arr))
        item = torch.stack(images).permute(0, 3, 1, 2)  # (T, 3, H, W)
        return self._apply_transform(item)
