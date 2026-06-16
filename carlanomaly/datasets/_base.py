from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, ClassVar, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from ..download import part_for
from ..index import ScenarioIndex, ScenarioRecord

#: Anything accepted as a filesystem path for the dataset root.
PathLike = Union[str, Path]


def required_parts(specs: Iterable[Tuple[str, Optional[str]]]) -> List[str]:
    """Archive parts needed to serve the given ``(modality, direction)`` specs.

    ``"base"`` is always included first: it holds the front camera and the
    per-scenario feather files that :class:`ScenarioIndex` discovery relies on,
    so it is required no matter which modality is requested.  The remaining
    parts are resolved via :func:`carlanomaly.download.part_for` and returned in
    a stable, de-duplicated order.
    """
    parts = {"base"}
    for modality, direction in specs:
        parts.add(part_for(modality, direction))
    return ["base"] + sorted(parts - {"base"})


def resolve_index(
    root: Optional[PathLike],
    split: str,
    *,
    index: Optional[ScenarioIndex] = None,
    **index_kwargs: Any,
) -> ScenarioIndex:
    """Return ``index`` if given, otherwise build one from ``root``/``split``.

    Composite datasets use this to build a single :class:`ScenarioIndex` and
    share it with their sub-datasets (so the filesystem is scanned once).
    """
    if index is not None:
        return index
    if root is None:
        raise ValueError("either `root` or `index` must be provided")
    return ScenarioIndex(root=root, split=split, **index_kwargs)


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
    """Base class for all atomic CarlAnomaly datasets.

    Subclasses point the loader at a dataset ``root`` directory and a
    ``split``; the underlying scenario index is built internally.

    With ``download=True`` the dataset fetches the archive parts it needs into
    ``root`` before loading (idempotent: already-present parts are skipped).
    The parts are selected automatically from the subclass's :attr:`modality`
    (and ``direction`` where applicable), so e.g. a LiDAR dataset pulls the
    ``lidar`` part and a left-camera dataset pulls ``camera-extended``.  Pass an
    explicit ``parts`` to override the auto-selection.
    """

    #: Dataset-registry modality key (see ``carlanomaly.download``).  Drives the
    #: ``download=True`` part auto-selection; ``None`` disables it.
    modality: ClassVar[Optional[str]] = None

    def __init__(
        self,
        root: Optional[PathLike] = None,
        split: str = "train",
        *,
        clip_len: int = 1,
        stride: Optional[int] = None,
        anomaly_types: Optional[Sequence[str]] = None,
        towns: Optional[Sequence[str]] = None,
        download: bool = False,
        parts: Optional[Sequence[str]] = None,
        download_verify: bool = True,
        direction: Optional[str] = None,
        transform: Optional[Callable] = None,
        index: Optional[ScenarioIndex] = None,
    ) -> None:
        self.direction = direction
        if download and parts is None and self.modality is not None:
            parts = required_parts([(self.modality, direction)])
        self._index = resolve_index(
            root,
            split,
            index=index,
            clip_len=clip_len,
            stride=stride,
            anomaly_types=anomaly_types,
            towns=towns,
            download=download,
            parts=parts,
            download_verify=download_verify,
        )
        self.transform = transform

    @property
    def _is_train(self) -> bool:
        return self._index.split == "train"

    def __len__(self) -> int:
        return len(self._index)

    def _apply_transform(self, item: Any) -> Any:
        if self.transform is not None:
            return self.transform(item)
        return item
