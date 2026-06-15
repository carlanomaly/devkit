from __future__ import annotations

from typing import Callable, Optional

import torch

from ..index import ScenarioIndex
from ._base import AtomicDataset


class WeatherDataset(AtomicDataset):
    """Per-frame weather parameters.

    Returns a ``FloatTensor (T, 14)`` with columns: cloudiness, precipitation,
    sun_altitude_angle, sun_azimuth_angle, fog_density, fog_distance,
    fog_falloff, precipitation_deposits, wind_intensity, wetness,
    scattering_intensity, mie_scattering_scale, rayleigh_scattering_scale,
    dust_storm.
    """

    def __init__(
        self,
        index: ScenarioIndex,
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(index, transform)

    def __getitem__(self, idx: int) -> torch.Tensor:
        rec, start = self._index[idx]
        frames = self._index.frames_for(idx)
        arr = self._read_feather_cached(rec, "weather")
        item = torch.from_numpy(arr[frames])  # (T, 14)
        return self._apply_transform(item)
