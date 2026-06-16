from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
import torch
from PIL import Image

from ..index import ScenarioIndex
from ._base import AtomicDataset, PathLike


class AnomalySegmentationDataset(AtomicDataset):
    """Per-frame pixel-level anomaly masks from a single camera.

    Returns a ``BoolTensor (T, H, W)``.  For the train split (where no
    anomaly masks exist), returns all-False tensors.

    With ``download=True`` the required archive parts are fetched into ``root``
    automatically (``front`` lives in ``base``; other directions add
    ``camera-extended``).  Remaining keyword arguments (``clip_len``,
    ``stride``, ``parts``, ...) are forwarded to
    :class:`~carlanomaly.index.ScenarioIndex`.
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
        frames = self._index.frames_for(idx)
        masks = []
        for f in frames:
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
