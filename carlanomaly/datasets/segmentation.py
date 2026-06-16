from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
from PIL import Image

from ..index import ScenarioIndex
from ._base import AtomicDataset, PathLike


class SegmentationDataset(AtomicDataset):
    """Per-frame instance segmentation masks from a single camera.

    Returns a dict with:
        ``'semantic'``: ``LongTensor (T, H, W)`` of CARLA class ids (1-28,
            non-contiguous).
        ``'instance'``: ``LongTensor (T, H, W)`` of instance ids
            (green * 256 + blue channel of the RGBA mask).

    With ``download=True`` the required archive parts are fetched into ``root``
    automatically (``front`` lives in ``base``; other directions add
    ``camera-extended``).  Remaining keyword arguments (``clip_len``,
    ``stride``, ``parts``, ...) are forwarded to
    :class:`~carlanomaly.index.ScenarioIndex`.
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
        frames = self._index.frames_for(idx)
        semantic_list, instance_list = [], []
        for f in frames:
            path = rec.path / f"segmentation-{self.direction}" / f"{f:06d}.png"
            arr = np.array(Image.open(path).convert("RGBA"))  # (H, W, 4)
            r = arr[:, :, 0].astype(np.int64)
            r[r > 28] = 0  # rendering artifacts at object boundaries
            semantic_list.append(torch.from_numpy(r))
            inst = arr[:, :, 1].astype(np.int64) * 256 + arr[:, :, 2].astype(np.int64)
            instance_list.append(torch.from_numpy(inst))
        item = {
            "semantic": torch.stack(semantic_list),   # (T, H, W)
            "instance": torch.stack(instance_list),   # (T, H, W)
        }
        return self._apply_transform(item)
