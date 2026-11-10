from __future__ import annotations

from typing import List

import pandas as pd

from ._base import AtomicDataset


class CollisionsDataset(AtomicDataset):
    """Collisions of the ego vehicle.

    Each item is a list of ``T`` DataFrames, one per timestep, with one row per
    collision (usually none) and the columns ``frame`` (the timestep), ``ego_id``, ``ego_type``,
    ``other_id``, ``other_type`` (the actor hit) and ``normal_impulse_x``,
    ``normal_impulse_y``, ``normal_impulse_z``, ``normal_impulse_norm`` (the
    impulse of the collision).

    Batch it with :func:`carlanomaly_collate_fn`.
    """

    modality = "collisions"

    def __getitem__(self, idx: int) -> List[pd.DataFrame]:
        rec, _ = self._index[idx]
        timesteps = self._index.timesteps_for(idx)
        df = self._read_feather_cached(rec, "collisions", as_numpy=False)
        item = []
        for f in timesteps:
            if "frame" in df.columns and len(df) > 0:
                item.append(df[df["frame"] == f].reset_index(drop=True))
            else:
                item.append(pd.DataFrame())
        return self._apply_transform(item)
