from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from ..index import ScenarioIndex, ScenarioRecord


class _CachedFeatherMixin:
    """Shared caching for per-scenario feather files."""

    _feather_cache: Dict[str, Dict[str, np.ndarray]]

    def _ensure_cache(self) -> None:
        if not hasattr(self, "_feather_cache"):
            self._feather_cache = {}

    def _read_feather_cached(
        self, rec: ScenarioRecord, name: str, *, as_numpy: bool = True
    ) -> Any:
        self._ensure_cache()
        key = str(rec.path)
        if key not in self._feather_cache:
            self._feather_cache[key] = {}
        if name not in self._feather_cache[key]:
            path = rec.path / f"{name}.feather"
            df = pd.read_feather(path)
            if as_numpy:
                cols = [c for c in df.columns if c != "frame"]
                self._feather_cache[key][name] = df[cols].values.astype(np.float32)
            else:
                self._feather_cache[key][name] = df
        return self._feather_cache[key][name]


class AtomicDataset(Dataset, _CachedFeatherMixin):
    """Base class for all atomic CarlAnomaly datasets."""

    def __init__(
        self,
        index: ScenarioIndex,
        transform: Optional[Callable] = None,
    ) -> None:
        self._index = index
        self.transform = transform

    def __len__(self) -> int:
        return len(self._index)

    def _apply_transform(self, item: Any) -> Any:
        if self.transform is not None:
            return self.transform(item)
        return item
