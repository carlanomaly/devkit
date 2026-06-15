from __future__ import annotations

from typing import Callable, List, Optional

import pandas as pd
import torch

from ..index import ScenarioIndex
from ._base import AtomicDataset


class AnomalyLiDARDataset(AtomicDataset):
    """Per-frame, per-point LiDAR anomaly labels.

    Returns a ``List[BoolTensor]`` of length T.  Each tensor has shape
    ``(N_points,)`` matching the corresponding point cloud.  For the train
    split, returns all-False tensors sized to match the actual point cloud.
    """

    def __init__(
        self,
        index: ScenarioIndex,
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(index, transform)
        self._is_train = index.split == "train"

    def __getitem__(self, idx: int) -> List[torch.Tensor]:
        rec, _ = self._index[idx]
        frames = self._index.frames_for(idx)
        labels = []
        for f in frames:
            if self._is_train:
                pc_path = rec.path / "pointclouds" / f"{f:06d}.feather"
                n_points = len(pd.read_feather(pc_path))
                labels.append(torch.zeros(n_points, dtype=torch.bool))
            else:
                path = rec.path / "anomaly-lidar" / f"{f:06d}.feather"
                if path.exists():
                    df = pd.read_feather(path)
                    labels.append(
                        torch.from_numpy(df["anomaly"].values.astype(bool))
                    )
                else:
                    pc_path = rec.path / "pointclouds" / f"{f:06d}.feather"
                    n_points = len(pd.read_feather(pc_path))
                    labels.append(torch.zeros(n_points, dtype=torch.bool))
        return self._apply_transform(labels)
