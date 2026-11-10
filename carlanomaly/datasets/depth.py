from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
import torch
from PIL import Image

from ..index import ScenarioIndex
from ._base import AtomicDataset, PathLike


class DepthDataset(AtomicDataset):
    """Depth maps from one camera.

    Each item is a ``FloatTensor (T, 1, 1080, 1920)`` in ``[0, 1]``: the
    distance along the camera's viewing axis in CARLA's logarithmic encoding,
    stored with 8 bits. To convert it to metres::

        metres = 1000 * torch.exp(5.70378 * (depth - 1))

    The encoding covers 3.3 m to 1000 m, with steps of about 2 % of the
    distance.
    """

    modality = "depth"

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
            path = rec.path / f"depth-{self.direction}" / f"{f:06d}.png"
            img = Image.open(path).convert("L")
            arr = np.array(img, dtype=np.float32) / 255.0  # (H, W)
            images.append(torch.from_numpy(arr))
        item = torch.stack(images).unsqueeze(1)  # (T, 1, H, W)
        return self._apply_transform(item)
