from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch

from ..index import SENSORS
from ._base import AtomicDataset


class AnomalySensorDataset(AtomicDataset):
    """Sensor-level anomaly labels for one sensor (one label per timestep).

    Returns a ``BoolTensor (T,)`` marking whether the anomaly is visible in
    this specific sensor at each timestep.  Labels are stored on disk as
    ``anomaly-{sensor}/sensor.feather``.  For the train split (where no label
    file exists), returns all-False.

    Parameters
    ----------
    sensor:
        Any sensor name (``"front"``, ``"left"``, ``"right"``, ``"rear"``,
        ``"lidar"``).
    """

    modality = "anomaly_sensor"

    def __init__(self, *args: Any, sensor: str = "front", **kwargs: Any) -> None:
        if sensor not in SENSORS:
            raise ValueError(
                f"AnomalySensorDataset sensor must be one of {SENSORS}, got {sensor!r}"
            )
        self.sensor = sensor
        # The sensor doubles as the download "direction": front labels are in
        # `base`, other cameras in `camera-extended`, lidar in `lidar`.
        kwargs["direction"] = sensor
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx: int) -> torch.Tensor:
        rec, _ = self._index[idx]
        timesteps = self._index.timesteps_for(idx)

        if self._is_train:
            item = torch.zeros(len(timesteps), dtype=torch.bool)
        else:
            cache = self._get_label_cache(rec)
            item = torch.from_numpy(cache[timesteps])

        return self._apply_transform(item)

    def _get_label_cache(self, rec) -> np.ndarray:
        self._ensure_cache()
        key = str(rec.path)
        cache_name = f"_sensor_{self.sensor}"
        if key not in self._feather_cache:
            self._feather_cache[key] = {}
        if cache_name not in self._feather_cache[key]:
            df = pd.read_feather(rec.path / f"anomaly-{self.sensor}" / "sensor.feather")
            self._feather_cache[key][cache_name] = df["anomaly"].values.astype(bool)
        return self._feather_cache[key][cache_name]
