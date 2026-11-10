from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

import pandas as pd
import torch
from torch.utils.data import Dataset

from ..index import CAMERAS, ScenarioIndex
from ._base import PathLike, check_parts_for, ensure_parts_for, required_parts, resolve_index
from .actions import ActionsDataset
from .anomaly_timestep import AnomalyTimestepDataset
from .camera import CameraDataset
from .collisions import CollisionsDataset
from .gnss import GNSSDataset
from .imu import IMUDataset
from .lidar import LiDARDataset
from .weather import WeatherDataset


class CarlAnomalyDataset(Dataset):
    """Every sensor, every label and the vehicle data in one item.

    Use it for detectors that combine many inputs; it needs every part except
    ``kitti``. Each item is a dict with the keys

    - ``"{camera}_{key}"`` for each camera in ``cameras`` and each key of
      :class:`CameraDataset`, e.g. ``"front_rgb"`` or ``"left_anomaly_mask"``;
    - ``"points"``, ``"anomaly_lidar"``: as in :class:`LiDARDataset`;
    - ``"weather"``, ``"gnss"``, ``"imu"``, ``"actions"``, ``"collisions"``,
      ``"anomaly_timestep"``: the items of the matching datasets;
    - ``"scenario_id"``, ``"timesteps"``, ``"anomaly_type"`` (``None`` for
      normal scenarios) and ``"town"``.

    Batch it with :func:`carlanomaly_collate_fn`.

    Parameters
    ----------
    cameras:
        The cameras to include. Default: all four. Fewer cameras load faster.
    """

    def __init__(
        self,
        root: Optional[PathLike] = None,
        split: str = "train",
        cameras: Sequence[str] = CAMERAS,
        *,
        transform: Optional[Callable] = None,
        index: Optional[ScenarioIndex] = None,
        download: bool = False,
        **index_kwargs: Any,
    ) -> None:
        parts = index_kwargs.pop("parts", None)
        if parts is None:
            specs = [(m, cam) for cam in cameras
                     for m in ("rgb", "depth", "segmentation", "anomaly_seg")]
            specs += [(m, None) for m in ("pointcloud", "anomaly_lidar", "weather",
                                          "gnss", "imu", "actions", "collisions",
                                          "anomaly_timestep")]
            parts = required_parts(specs)
        if download:
            ensure_parts_for(root, split, index, parts,
                             verify=index_kwargs.get("download_verify", True))
        elif index is None:
            check_parts_for(root, split, parts)
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
        self._anomaly_timestep = AnomalyTimestepDataset(index=index, download=False)

    @property
    def index(self) -> ScenarioIndex:
        """The :class:`ScenarioIndex` this dataset is built on (shareable)."""
        return self._index

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec, _ = self._index[idx]

        item: Dict[str, Any] = {
            "scenario_id": str(rec.path),
            "anomaly_type": rec.anomaly_type,
            "town": rec.town,
            "timesteps": torch.tensor(
                self._index.timesteps_for(idx), dtype=torch.long
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
        item["anomaly_timestep"] = self._anomaly_timestep[idx]

        if self.transform is not None:
            item = self.transform(item)
        return item


def carlanomaly_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function for items that hold lists, DataFrames or strings.

    Pass it as ``collate_fn`` to a ``DataLoader`` for datasets whose items vary
    in size: point clouds, LiDAR labels, collisions and the combined datasets.
    Tensors are stacked along a new batch dimension, dicts are batched key by
    key, and everything else becomes a list with one entry per item.

    Example
    -------
    >>> loader = DataLoader(LiDARDataset(root, split="test"), batch_size=8,
    ...                     collate_fn=carlanomaly_collate_fn)
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
