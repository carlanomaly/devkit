from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

import pandas as pd
import torch
from torch.utils.data import Dataset

from ..index import CAMERAS, ScenarioIndex
from .actions import ActionsDataset
from .anomaly_obs import AnomalyObservationDataset
from .camera import CameraDataset
from .collisions import CollisionsDataset
from .gnss import GNSSDataset
from .imu import IMUDataset
from .lidar import LiDARDataset
from .weather import WeatherDataset


class CarlAnomalyDataset(Dataset):
    """Joint dataset wrapping all modalities and all cameras.

    Returns a dict with prefixed camera keys (``front_rgb``, ``left_depth``,
    etc.) plus tabular, LiDAR, and anomaly-observation entries.

    Use :func:`carlanomaly_collate_fn` as the DataLoader ``collate_fn``.

    Parameters
    ----------
    index:
        Shared :class:`ScenarioIndex`.
    cameras:
        Camera directions to include.  Defaults to all four.
    transform:
        Optional transform applied to the merged output dict.
    """

    def __init__(
        self,
        index: ScenarioIndex,
        cameras: Sequence[str] = CAMERAS,
        transform: Optional[Callable] = None,
    ) -> None:
        self._index = index
        self.cameras = list(cameras)
        self.transform = transform

        self._camera_datasets: Dict[str, CameraDataset] = {
            cam: CameraDataset(index, direction=cam) for cam in self.cameras
        }
        self._lidar = LiDARDataset(index)
        self._weather = WeatherDataset(index)
        self._gnss = GNSSDataset(index)
        self._imu = IMUDataset(index)
        self._actions = ActionsDataset(index)
        self._collisions = CollisionsDataset(index)
        self._anomaly_obs = AnomalyObservationDataset(index)

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec, _ = self._index[idx]

        item: Dict[str, Any] = {
            "scenario_id": str(rec.path),
            "anomaly_type": rec.anomaly_type,
            "town": rec.town,
            "frame_indices": torch.tensor(
                self._index.frames_for(idx), dtype=torch.long
            ),
        }

        for cam, ds in self._camera_datasets.items():
            cam_data = ds[idx]
            for key, value in cam_data.items():
                item[f"{cam}_{key}"] = value

        lidar_data = self._lidar[idx]
        item["points"] = lidar_data["points"]
        item["anomaly_lidar"] = lidar_data["anomaly_lidar"]

        item["weather"] = self._weather[idx]
        item["gnss"] = self._gnss[idx]
        item["imu"] = self._imu[idx]
        item["actions"] = self._actions[idx]
        item["collisions"] = self._collisions[idx]
        item["anomaly_observation"] = self._anomaly_obs[idx]

        if self.transform is not None:
            item = self.transform(item)
        return item


def carlanomaly_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function for :class:`CarlAnomalyDataset`.

    - Stacks :class:`torch.Tensor` values along a new batch dimension.
    - Keeps ``List``, ``str``, ``None``, and ``pd.DataFrame`` values as
      plain Python lists.
    - Recurses into dicts.
    """
    if not batch:
        return {}
    keys = batch[0].keys()
    collated: Dict[str, Any] = {}
    for k in keys:
        values = [item[k] for item in batch]
        collated[k] = _collate_values(values)
    return collated


def _collate_values(values: List[Any]) -> Any:
    first = values[0]
    if isinstance(first, torch.Tensor):
        return torch.stack(values)
    if isinstance(first, dict):
        keys = first.keys()
        return {k: _collate_values([v[k] for v in values]) for k in keys}
    # Lists (point clouds, collisions, lidar labels), strings, None, etc.
    return values
