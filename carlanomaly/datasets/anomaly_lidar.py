from __future__ import annotations

from typing import List

import pandas as pd
import torch

from ._base import AtomicDataset


class AnomalyLiDARDataset(AtomicDataset):
    """Anomaly labels for the LiDAR points.

    Each item is a list of ``T`` ``BoolTensor (N,)``, one per point cloud,
    ``True`` for points on the anomalous object. They follow the row order of
    :class:`PointCloudDataset`. Without an anomaly, all points are ``False``.

    These are the labels :class:`~carlanomaly.PointEvaluator` uses.
    """

    modality = "anomaly_lidar"

    def __getitem__(self, idx: int) -> List[torch.Tensor]:
        rec, _ = self._index[idx]
        timesteps = self._index.timesteps_for(idx)
        labels = []
        for f in timesteps:
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
