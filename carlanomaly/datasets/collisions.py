from __future__ import annotations

from typing import Callable, List, Optional

import pandas as pd

from ..index import ScenarioIndex
from ._base import AtomicDataset


class CollisionsDataset(AtomicDataset):
    """Per-frame collision events (sparse).

    Returns a ``List[pd.DataFrame]`` of length T.  Each DataFrame contains
    zero or more rows with columns: ego_id, ego_type, other_id, other_type,
    normal_impulse_x, normal_impulse_y, normal_impulse_z, normal_impulse_norm.
    Most frames have no collisions (empty DataFrame).
    """

    def __init__(
        self,
        index: ScenarioIndex,
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(index, transform)

    def __getitem__(self, idx: int) -> List[pd.DataFrame]:
        rec, _ = self._index[idx]
        frames = self._index.frames_for(idx)
        df = self._read_feather_cached(rec, "collisions", as_numpy=False)
        item = []
        for f in frames:
            if "frame" in df.columns and len(df) > 0:
                item.append(df[df["frame"] == f].reset_index(drop=True))
            else:
                item.append(pd.DataFrame())
        return self._apply_transform(item)
