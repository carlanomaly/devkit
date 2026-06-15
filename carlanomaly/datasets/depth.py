from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image

from ..index import ScenarioIndex
from ._base import AtomicDataset


class DepthDataset(AtomicDataset):
    """Per-frame log-encoded depth maps from a single camera.

    Returns a ``FloatTensor (T, 1, H, W)`` in ``[0, 1]`` (uint8 / 255).
    """

    def __init__(
        self,
        index: ScenarioIndex,
        direction: str = "front",
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(index, transform)
        self.direction = direction

    def __getitem__(self, idx: int) -> torch.Tensor:
        rec, _ = self._index[idx]
        frames = self._index.frames_for(idx)
        images = []
        for f in frames:
            path = rec.path / f"depth-{self.direction}" / f"{f:06d}.png"
            img = Image.open(path).convert("L")
            arr = np.array(img, dtype=np.float32) / 255.0  # (H, W)
            images.append(torch.from_numpy(arr))
        item = torch.stack(images).unsqueeze(1)  # (T, 1, H, W)
        return self._apply_transform(item)
