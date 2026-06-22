from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
import torch
from PIL import Image

from ..index import ScenarioIndex
from ._base import AtomicDataset, PathLike


class RGBDataset(AtomicDataset):
    """Per-frame RGB images from a single camera.

    Returns a ``FloatTensor (T, 3, H, W)`` in ``[0, 1]``.

    The required archive parts are fetched into ``root`` automatically (``front``
    lives in ``base``; other directions add ``camera-extended``); pass
    ``download=False`` to skip.  Remaining keyword arguments (``clip_len``,
    ``stride``, ``parts``, ...) are forwarded to
    :class:`~carlanomaly.index.ScenarioIndex`.
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
        download: bool = True,
        **index_kwargs: Any,
    ) -> None:
        super().__init__(
            root, split, direction=direction, transform=transform,
            index=index, download=download, **index_kwargs,
        )

    def __getitem__(self, idx: int) -> torch.Tensor:
        rec, _ = self._index[idx]
        frames = self._index.frames_for(idx)
        images = []
        for f in frames:
            path = rec.path / f"rgb-{self.direction}" / f"{f:06d}.jpg"
            img = Image.open(path).convert("RGB")
            arr = np.array(img, dtype=np.float32) / 255.0  # (H, W, 3)
            images.append(torch.from_numpy(arr))
        item = torch.stack(images).permute(0, 3, 1, 2)  # (T, 3, H, W)
        return self._apply_transform(item)
