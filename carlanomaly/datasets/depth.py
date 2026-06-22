from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
import torch
from PIL import Image

from ..index import ScenarioIndex
from ._base import AtomicDataset, PathLike


class DepthDataset(AtomicDataset):
    """Per-frame log-encoded depth maps from a single camera.

    Returns a ``FloatTensor (T, 1, H, W)`` in ``[0, 1]`` (uint8 / 255).

    Depth maps live in the ``depth`` part; it (plus the ``base`` part) is fetched
    into ``root`` automatically; pass ``download=False`` to skip.  Remaining
    keyword arguments (``clip_len``, ``stride``, ``parts``, ...) are forwarded to
    :class:`~carlanomaly.index.ScenarioIndex`.
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
            path = rec.path / f"depth-{self.direction}" / f"{f:06d}.png"
            img = Image.open(path).convert("L")
            arr = np.array(img, dtype=np.float32) / 255.0  # (H, W)
            images.append(torch.from_numpy(arr))
        item = torch.stack(images).unsqueeze(1)  # (T, 1, H, W)
        return self._apply_transform(item)
