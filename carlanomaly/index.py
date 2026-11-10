from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union


CAMERAS = ("front", "left", "right", "rear")

SENSORS = CAMERAS + ("lidar",)

ANOMALY_TYPES = (
    "change-weather",
    "running-pedestrian",
    "spawn-props",
    "steer-driver",
    "street-light-flicker",
    "traffic-light-flicker",
    "traffic-light-off",
    "traffic-light-yellow-blinking",
    "vanish-actor",
)


@dataclass(frozen=True)
class ScenarioRecord:
    """One scenario of the dataset.

    ``path`` is its directory, ``n_timesteps`` its length and ``split`` one
    of ``"train"``, ``"test_normal"`` and ``"test_anomaly"``.  ``anomaly_type``
    is ``None`` for normal scenarios.
    """

    path: Path
    n_timesteps: int
    split: str  # 'train' | 'test_normal' | 'test_anomaly'
    town: Optional[str]
    anomaly_type: Optional[str]


class ScenarioIndex:
    """The scenarios of a split and the clips cut from them.

    Item ``i`` of every dataset built on the same index is the same clip.
    The arguments are the :ref:`common dataset arguments <dataset-arguments>`;
    with ``download=True`` the index downloads ``parts`` (default:
    ``["base"]``).
    """

    def __init__(
        self,
        root: Union[str, Path],
        split: str = "train",
        clip_len: int = 1,
        stride: Optional[int] = None,
        anomaly_types: Optional[Sequence[str]] = None,
        towns: Optional[Sequence[str]] = None,
        download: bool = False,
        parts: Optional[Sequence[str]] = None,
        download_verify: bool = True,
    ) -> None:
        valid_splits = ("train", "test_normal", "test_anomaly", "test")
        if split not in valid_splits:
            raise ValueError(f"split must be one of {valid_splits}, got {split!r}")

        self.root = Path(root)
        self.split = split
        self.clip_len = clip_len
        self.stride = stride if stride is not None else clip_len
        self.anomaly_types = set(anomaly_types) if anomaly_types else None
        self.towns = set(towns) if towns else None

        if download:
            from .download import ensure_parts, splits_for

            ensure_parts(
                self.root,
                list(parts) if parts else ["base"],
                splits_for(split),
                verify=download_verify,
            )

        self._records: List[ScenarioRecord] = self._discover()
        self._index: List[Tuple[int, int]] = self._build_index()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def records(self) -> List[ScenarioRecord]:
        """All scenarios of the split."""
        return self._records

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Tuple[ScenarioRecord, int]:
        """Return the scenario of item ``idx`` and the clip's first timestep."""
        rec_idx, timestep_start = self._index[idx]
        return self._records[rec_idx], timestep_start

    def timesteps_for(self, idx: int) -> List[int]:
        """Return the timesteps of item ``idx``."""
        _, start = self._index[idx]
        return list(range(start, start + self.clip_len))

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def _discover(self) -> List[ScenarioRecord]:
        records: List[ScenarioRecord] = []
        if self.split in ("train",):
            records += self._scan_train()
        if self.split in ("test_normal", "test"):
            records += self._scan_test_normal()
        if self.split in ("test_anomaly", "test"):
            records += self._scan_test_anomaly()
        if not records:
            # usually the dataset is not there yet: name the part to fetch
            from .download import require_parts, splits_for

            require_parts(self.root, ["base"], splits_for(self.split))
            raise RuntimeError(
                f"No scenarios found for split={self.split!r} under {self.root}"
            )
        return records

    def _scan_train(self) -> List[ScenarioRecord]:
        train_root = self.root / "train"
        if not train_root.exists():
            return []
        records = []
        for town_dir in sorted(train_root.iterdir()):
            if not town_dir.is_dir():
                continue
            if self.towns and town_dir.name not in self.towns:
                continue
            for sc_dir in sorted(town_dir.iterdir()):
                if not sc_dir.is_dir():
                    continue
                n = self._count_timesteps(sc_dir)
                if n > 0:
                    records.append(ScenarioRecord(
                        path=sc_dir,
                        n_timesteps=n,
                        split="train",
                        town=town_dir.name,
                        anomaly_type=None,
                    ))
        return records

    def _scan_test_normal(self) -> List[ScenarioRecord]:
        normal_root = self.root / "test" / "normal"
        if not normal_root.exists():
            return []
        records = []
        for obs in sorted(normal_root.rglob("anomaly-observation.feather")):
            sc_dir = obs.parent
            # path: test/normal/{town}/scenario-N
            town = sc_dir.parent.name
            if self.towns and town not in self.towns:
                continue
            n = self._count_timesteps(sc_dir)
            if n > 0:
                records.append(ScenarioRecord(
                    path=sc_dir,
                    n_timesteps=n,
                    split="test_normal",
                    town=town,
                    anomaly_type=None,
                ))
        return records

    def _scan_test_anomaly(self) -> List[ScenarioRecord]:
        anomaly_root = self.root / "test" / "anomaly"
        if not anomaly_root.exists():
            return []
        records = []
        for obs in sorted(anomaly_root.rglob("anomaly-observation.feather")):
            sc_dir = obs.parent
            # path: test/anomaly/{town}/{anomaly_type}/scenario-N
            atype = sc_dir.parent.name
            town = sc_dir.parent.parent.name
            if self.anomaly_types and atype not in self.anomaly_types:
                continue
            if self.towns and town not in self.towns:
                continue
            n = self._count_timesteps(sc_dir)
            if n > 0:
                records.append(ScenarioRecord(
                    path=sc_dir,
                    n_timesteps=n,
                    split="test_anomaly",
                    town=town,
                    anomaly_type=atype,
                ))
        return records

    # ------------------------------------------------------------------
    # Windowing
    # ------------------------------------------------------------------

    def _build_index(self) -> List[Tuple[int, int]]:
        index = []
        for rec_idx, rec in enumerate(self._records):
            max_start = rec.n_timesteps - self.clip_len
            if max_start < 0:
                continue
            for start in range(0, max_start + 1, self.stride):
                index.append((rec_idx, start))
        return index

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _count_timesteps(sc_dir: Path) -> int:
        rgb_dir = sc_dir / "rgb-front"
        if not rgb_dir.exists():
            return 0
        return len(list(rgb_dir.glob("*.jpg")))

    def __repr__(self) -> str:
        return (
            f"ScenarioIndex(split={self.split!r}, "
            f"scenarios={len(self._records)}, "
            f"items={len(self._index)}, "
            f"clip_len={self.clip_len}, "
            f"stride={self.stride})"
        )
