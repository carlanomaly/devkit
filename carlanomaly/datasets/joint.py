from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

import pandas as pd
import torch
from torch.utils.data import Dataset

from ..index import CAMERAS, ScenarioIndex
from ._base import PathLike, ensure_parts_for, required_parts, resolve_index
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
    root:
        Path to the CarlAnomaly dataset directory.
    split:
        ``'train'``, ``'test_normal'``, ``'test_anomaly'``, or ``'test'``.
    cameras:
        Camera directions to include.  Defaults to all four.
    transform:
        Optional transform applied to the merged output dict.
    download:
        If ``True``, fetch every archive part this joint dataset needs
        (``base``, ``depth``, ``lidar``, plus ``camera-extended`` when any
        non-front camera is included) into ``root`` before loading.

    Additional keyword arguments (``clip_len``, ``stride``, ``parts``, ...)
    are forwarded to :class:`~carlanomaly.index.ScenarioIndex`.
    """

    def __init__(
        self,
        root: Optional[PathLike] = None,
        split: str = "train",
        cameras: Sequence[str] = CAMERAS,
        *,
        transform: Optional[Callable] = None,
        index: Optional[ScenarioIndex] = None,
        download: bool = True,
        **index_kwargs: Any,
    ) -> None:
        if download:
            parts = index_kwargs.pop("parts", None)
            if parts is None:
                specs = [(m, cam) for cam in cameras
                         for m in ("rgb", "depth", "segmentation", "anomaly_seg")]
                specs += [(m, None) for m in ("pointcloud", "anomaly_lidar", "weather",
                                              "gnss", "imu", "actions", "collisions",
                                              "anomaly_obs")]
                parts = required_parts(specs)
            ensure_parts_for(root, split, index, parts,
                             verify=index_kwargs.get("download_verify", True))
        index = resolve_index(root, split, index=index, download=False, **index_kwargs)
        self._index = index
        self.cameras = list(cameras)
        self.transform = transform

        # Parts already fetched above; children must not re-download.
        self._camera_datasets: Dict[str, CameraDataset] = {
            cam: CameraDataset(direction=cam, index=index, download=False) for cam in self.cameras
        }
        self._lidar = LiDARDataset(index=index, download=False)
        self._weather = WeatherDataset(index=index, download=False)
        self._gnss = GNSSDataset(index=index, download=False)
        self._imu = IMUDataset(index=index, download=False)
        self._actions = ActionsDataset(index=index, download=False)
        self._collisions = CollisionsDataset(index=index, download=False)
        self._anomaly_obs = AnomalyObservationDataset(index=index, download=False)

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
