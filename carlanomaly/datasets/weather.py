from __future__ import annotations

import torch

from ._base import AtomicDataset


class WeatherDataset(AtomicDataset):
    """Weather and lighting, as set in the simulator.

    Each item is a ``FloatTensor (T, 14)`` with the columns ``cloudiness``,
    ``precipitation``, ``sun_altitude_angle``, ``sun_azimuth_angle``,
    ``fog_density``, ``fog_distance``, ``fog_falloff``,
    ``precipitation_deposits``, ``wind_intensity``, ``wetness``,
    ``scattering_intensity``, ``mie_scattering_scale``,
    ``rayleigh_scattering_scale`` and ``dust_storm``. Most are on CARLA's 0 to
    100 scale; the sun angles are in degrees and ``fog_distance`` in metres.

    The weather is constant within a training scenario; it changes abruptly in
    the ``change-weather`` anomaly scenarios.
    """

    modality = "weather"

    def __getitem__(self, idx: int) -> torch.Tensor:
        rec, start = self._index[idx]
        timesteps = self._index.timesteps_for(idx)
        arr = self._read_feather_cached(rec, "weather")
        item = torch.from_numpy(arr[timesteps])  # (T, 14)
        return self._apply_transform(item)
