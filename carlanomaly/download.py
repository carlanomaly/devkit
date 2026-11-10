"""Auto-download and extraction of CarlAnomaly dataset shards.

The dataset is published as modular ``tar.gz`` archives ("parts"), each split
into a ``train`` and a ``test`` archive, hosted at :data:`DATA_BASE_URL`.  Every
archive stores dataset-root-relative paths, so extracting it at the dataset root
reconstructs the ``train/<town>/scenario-N/...`` / ``test/...`` layout that
:class:`carlanomaly.index.ScenarioIndex` expects.

The high-level entry point is :func:`ensure_parts`, which is idempotent: it
downloads (with HTTP-range resume), verifies the SHA-256 checksum, extracts, and
writes a marker file so subsequent calls are no-ops.  A part that was downloaded
and extracted by hand (no marker) is recognised from its directories and not
downloaded again (see :func:`part_present`).  Datasets call it when constructed
with ``download=True``; it can also be used directly or via the
``carlanomaly-download`` command.

Only the Python standard library is required.  ``tqdm`` is used for progress bars
if importable, otherwise a plain textual fallback is printed.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import tarfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union
from urllib.error import HTTPError
from urllib.request import Request, urlopen

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

#: Base URL the archives are served from.  Override with the
#: ``CARLANOMALY_BASE_URL`` environment variable (no trailing slash).
DATA_BASE_URL = os.environ.get(
    "CARLANOMALY_BASE_URL", "https://data.carlanomaly.de/v1"
).rstrip("/")

#: The parts and what each contains.
PARTS: Dict[str, str] = {
    "base": "Front camera, GNSS, IMU, weather, control inputs, collisions; "
            "test: front anomaly masks and timestep labels",
    "camera-extended": "Left, right and rear cameras; test: their anomaly masks",
    "lidar": "LiDAR point clouds; test: their anomaly labels",
    "depth": "Depth maps of all four cameras",
    "kitti": "LiDAR labels in KITTI format",
}

ALL_PARTS: Tuple[str, ...] = tuple(PARTS)
SPLITS: Tuple[str, ...] = ("train", "test")

#: A directory each part adds to every scenario; used to recognise parts that
#: were extracted without the downloader (and so have no marker file).
PART_PROBES: Dict[str, str] = {
    "base": "rgb-front",
    "camera-extended": "rgb-left",
    "lidar": "pointclouds",
    "depth": "depth-front",
    "kitti": "kitti-front",
}

#: SHA-256 checksums keyed by ``(part, split)``, from the ``SHA256SUMS.txt``
#: published next to the archives (updated 2026-11-01, archives now ship the
#: dataset ``LICENSE``).
SHA256: Dict[Tuple[str, str], str] = {
    ("base", "test"): "943601ee9b00470fe7aa002063208bb2c066bdad3f647a6f2c1ce8ed6b1925e3",
    ("base", "train"): "7ede2b700dc6340d4c5b2b1dbc746cb9d8ad09a68a6f12a91c8dc1170b4fe844",
    ("camera-extended", "test"): "5f65d34e6b814b76e3501a4fac5eea887687168da7daf063411077dd2bd97f82",
    ("camera-extended", "train"): "930f0b698944d03dd540cac0413dc978d93d1c581b20491ec944e171f9de69f3",
    ("depth", "test"): "d527b6f8299d2931da2e48861e7456459b575b9f27f07e030a37d0e5f681cc88",
    ("depth", "train"): "f36ee07c65021563e6078ef1729f34d9c840fac6aa91dbac530fe8493dcdcc48",
    ("kitti", "test"): "ee34446704c26a70b0876f74dd61f6bb1fdb6f6e603902ea8a2d07e053f0c967",
    ("kitti", "train"): "7806387792524969c466e7c5bb19c70daa3b4833d986c8d104a344eee8c46d90",
    ("lidar", "test"): "c9af4f61abca1b146fbbe280aaab9f7a6383426476eab6e6b94257678fd7a0e4",
    ("lidar", "train"): "f9d159632da18c3fa615fdb8fe1dfab30805aae20f0feb33131f44ae5ac7c677",
}

#: Where archives + extraction markers live, relative to the dataset root.
DOWNLOAD_SUBDIR = ".carlanomaly_download"

_CHUNK = 1 << 20  # 1 MiB

# --------------------------------------------------------------------------
# Mapping helpers
# --------------------------------------------------------------------------

# Modality (dataset) -> part, for the cameras' "front" direction and tabular
# sensors.  Non-front camera directions resolve to "camera-extended" instead;
# that special case is handled in :func:`part_for`.
_MODALITY_PART: Dict[str, str] = {
    "rgb": "base",
    "segmentation": "base",
    "anomaly_seg": "base",
    "depth": "depth",
    "pointcloud": "lidar",
    "lidar": "lidar",
    "anomaly_lidar": "lidar",
    "gnss": "base",
    "imu": "base",
    "weather": "base",
    "actions": "base",
    "collisions": "base",
    "anomaly_timestep": "base",
    "anomaly_sensor": "base",
    "anomaly_scenario": "base",
    "kitti": "kitti",
}

# Camera modalities whose "front" direction is in `base` but whose other
# directions live in `camera-extended`.
_CAMERA_MODALITIES = {"rgb", "segmentation", "anomaly_seg", "anomaly_sensor"}


def part_for(modality: str, direction: Optional[str] = None) -> str:
    """Return the part that holds a modality.

    Parameters
    ----------
    modality:
        A dataset's ``modality``, e.g. ``"rgb"``, ``"depth"`` or
        ``"pointcloud"``.
    direction:
        The camera, for camera data: the front camera is in ``base``, the
        others in ``camera-extended``.
    """
    if modality not in _MODALITY_PART:
        raise ValueError(
            f"unknown modality {modality!r}; expected one of {sorted(_MODALITY_PART)}"
        )
    if modality == "anomaly_sensor" and direction == "lidar":
        return "lidar"
    if modality in _CAMERA_MODALITIES and direction is not None and direction != "front":
        return "camera-extended"
    return _MODALITY_PART[modality]


def splits_for(index_split: str) -> Tuple[str, ...]:
    """Map a :class:`ScenarioIndex` split to the archive split(s) it needs.

    ``train`` needs the ``train`` archives; every ``test*`` split (including the
    combined ``test``) is served by the single ``test`` archive.
    """
    return ("train",) if index_split == "train" else ("test",)


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

def archive_name(part: str, split: str) -> str:
    return f"carlanomaly-{part}-{split}.tar.gz"


def archive_url(part: str, split: str, base_url: Optional[str] = None) -> str:
    return f"{(base_url or DATA_BASE_URL).rstrip('/')}/{archive_name(part, split)}"


def _download_dir(root: Path) -> Path:
    return root / DOWNLOAD_SUBDIR


def _marker_path(root: Path, part: str, split: str) -> Path:
    return _download_dir(root) / f"{part}-{split}.done"


def _scenario_dirs(root: Path, split: str) -> List[Path]:
    """Scenario directories of one archive split (``train`` or ``test``)."""
    if split == "train":
        return sorted(root.glob("train/*/scenario-*"))
    return sorted(root.glob("test/normal/*/scenario-*")) + sorted(root.glob("test/anomaly/*/*/scenario-*"))


def part_present(root: Union[str, Path], part: str, split: str) -> bool:
    """Whether a part's ``"train"`` or ``"test"`` archive is extracted in ``root``.

    Recognises parts downloaded by :func:`~carlanomaly.ensure_parts` and parts extracted by
    hand.
    """
    root = Path(root)
    if _marker_path(root, part, split).exists():
        return True
    scenarios = _scenario_dirs(root, split)
    return bool(scenarios) and all((sc / PART_PROBES[part]).is_dir() for sc in scenarios)


def require_parts(root: Union[str, Path], parts: Sequence[str], splits: Sequence[str]) -> None:
    """Raise :class:`FileNotFoundError` if any of ``parts`` is missing from ``root``.

    The message names the missing parts and how to download them.
    """
    missing = [(p, sp) for p in parts for sp in splits if not part_present(root, p, sp)]
    if not missing:
        return
    names = sorted({p for p, _ in missing}, key=list(ALL_PARTS).index)
    split_names = sorted({sp for _, sp in missing}, key=list(SPLITS).index)
    listed = ", ".join(f"{p} ({sp})" for p, sp in missing)
    raise FileNotFoundError(
        f"CarlAnomaly data missing under {root}: {listed}.\n"
        f"Download it by passing download=True, or run:\n"
        f"  carlanomaly-download --root {root} --parts {' '.join(names)} --splits {' '.join(split_names)}\n"
        f"or fetch the archives listed at https://carlanomaly.de/download/ and extract them into {root}."
    )


# --------------------------------------------------------------------------
# Download / verify / extract
# --------------------------------------------------------------------------

def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or unit == "TiB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TiB"


def _download(url: str, part_path: Path, *, progress: bool) -> str:
    """Download ``url`` to ``part_path`` with resume; return the SHA-256 hex.

    If a partial ``.part`` file already exists, an HTTP ``Range`` request resumes
    from where it stopped (seeding the hash with the existing bytes).  A server
    that ignores the range (responding ``200``) triggers a clean restart.
    """
    hasher = hashlib.sha256()
    resume = part_path.stat().st_size if part_path.exists() else 0
    if resume:
        with open(part_path, "rb") as f:
            for chunk in iter(lambda: f.read(_CHUNK), b""):
                hasher.update(chunk)

    req = Request(url)
    if resume:
        req.add_header("Range", f"bytes={resume}-")

    try:
        resp = urlopen(req, timeout=60)
    except HTTPError as e:
        if e.code == 416 and resume:
            # Range Not Satisfiable: file already fully downloaded.
            log.info("%s already complete", part_path.name)
            return hasher.hexdigest()
        raise

    with resp:
        if resume and resp.status == 200:
            # Server ignored the range header; start over.
            log.info("server ignored resume; restarting download")
            hasher = hashlib.sha256()
            resume = 0
        mode = "ab" if resume else "wb"

        clen = resp.headers.get("Content-Length")
        total = (resume + int(clen)) if clen is not None else None

        bar = _progress_bar(total, resume, part_path.name) if progress else None
        downloaded = resume
        with open(part_path, mode) as out:
            while True:
                chunk = resp.read(_CHUNK)
                if not chunk:
                    break
                out.write(chunk)
                hasher.update(chunk)
                downloaded += len(chunk)
                if bar is not None:
                    bar.update(len(chunk))
        if bar is not None:
            bar.close()
        elif progress and total:
            print(f"  downloaded {_fmt_bytes(downloaded)} / {_fmt_bytes(total)}")

    return hasher.hexdigest()


def _progress_bar(total: Optional[int], initial: int, desc: str):
    """Return a tqdm bar, or a tiny stderr-printing fallback shim."""
    try:
        from tqdm import tqdm
    except Exception:
        return _FallbackBar(total, initial, desc)
    return tqdm(
        total=total, initial=initial, unit="B", unit_scale=True,
        unit_divisor=1024, desc=desc,
    )


class _FallbackBar:
    """Minimal progress reporter when tqdm is unavailable."""

    def __init__(self, total: Optional[int], initial: int, desc: str) -> None:
        self._total = total
        self._n = initial
        self._desc = desc
        self._last_pct = -1
        self._print()

    def update(self, n: int) -> None:
        self._n += n
        self._print()

    def _print(self) -> None:
        if not self._total:
            return
        pct = int(self._n * 100 / self._total)
        if pct != self._last_pct and pct % 2 == 0:
            self._last_pct = pct
            print(
                f"  {self._desc}: {pct:3d}%  "
                f"({_fmt_bytes(self._n)} / {_fmt_bytes(self._total)})",
                flush=True,
            )

    def close(self) -> None:
        if self._total:
            print(f"  {self._desc}: done ({_fmt_bytes(self._n)})", flush=True)


#: Files that every archive carries at the dataset root (the dataset licence).
#: Only the first copy is extracted: later parts skip them, so an existing copy
#: that cannot be overwritten (e.g. one created by another user in a shared
#: dataset directory) does not stop a part from extracting.
SHARED_ROOT_FILES = ("LICENSE",)


def _safe_extract(tar_path: Path, dest: Path) -> None:
    """Extract ``tar_path`` into ``dest``, guarding against path traversal."""
    dest = dest.resolve()

    def already_there(member: tarfile.TarInfo) -> bool:
        # lexists: a dangling symlink counts as present and is left alone
        return member.name in SHARED_ROOT_FILES and os.path.lexists(dest / member.name)

    with tarfile.open(tar_path, "r:gz") as tar:
        if hasattr(tarfile, "data_filter"):
            # Python >= 3.12 (and security releases of older versions): vetted
            # extraction filter; returning None skips the member.
            def data_filter(member: tarfile.TarInfo, path: str) -> Optional[tarfile.TarInfo]:
                return None if already_there(member) else tarfile.data_filter(member, path)

            tar.extractall(dest, filter=data_filter)
        else:
            prefix = str(dest) + os.sep
            members = [m for m in tar.getmembers() if not already_there(m)]
            for member in members:
                target = (dest / member.name).resolve()
                if target != dest and not str(target).startswith(prefix):
                    raise RuntimeError(
                        f"refusing to extract {member.name!r}: escapes {dest}"
                    )
            tar.extractall(dest, members=members)


def _ensure_one(
    root: Path,
    part: str,
    split: str,
    *,
    verify: bool,
    progress: bool,
    keep_archive: bool,
    base_url: Optional[str],
) -> None:
    marker = _marker_path(root, part, split)
    if marker.exists():
        log.info("%s-%s already present (marker exists); skipping", part, split)
        return
    if part_present(root, part, split):
        # extracted by hand: record it so later calls skip the directory scan
        log.info("%s-%s found on disk without a marker; skipping download", part, split)
        try:
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(f"part={part}\nsplit={split}\nsource=found on disk\n")
        except OSError:  # read-only dataset copy: fine, the scan just repeats next time
            pass
        return

    ddir = _download_dir(root)
    ddir.mkdir(parents=True, exist_ok=True)

    name = archive_name(part, split)
    url = archive_url(part, split, base_url)
    part_path = ddir / (name + ".part")
    final_path = ddir / name

    log.info("downloading %s", url)
    computed = _download(url, part_path, progress=progress)

    expected = SHA256.get((part, split))
    if verify:
        if expected is None:
            log.warning(
                "no published checksum for %s; skipping integrity check", name
            )
        elif computed != expected:
            raise RuntimeError(
                f"checksum mismatch for {name}:\n"
                f"  expected {expected}\n  got      {computed}\n"
                f"The partial file was kept at {part_path} for inspection."
            )
        else:
            log.info("checksum OK for %s", name)

    os.replace(part_path, final_path)
    log.info("extracting %s into %s", name, root)
    _safe_extract(final_path, root)

    marker.write_text(
        f"part={part}\nsplit={split}\nsha256={computed}\nsource={url}\n"
    )
    if not keep_archive:
        final_path.unlink()
    log.info("done: %s-%s", part, split)


def ensure_parts(
    root: Union[str, Path],
    parts: Union[str, Sequence[str]],
    splits: Optional[Union[str, Sequence[str]]] = None,
    *,
    verify: bool = True,
    progress: bool = True,
    keep_archive: bool = False,
    base_url: Optional[str] = None,
) -> None:
    """Download and extract dataset parts into ``root``, skipping those present.

    Interrupted downloads resume.

    Parameters
    ----------
    root:
        The dataset directory.
    parts:
        One or more part names (see :data:`PARTS`).
    splits:
        ``"train"``, ``"test"`` or both (default).
    verify:
        Check each archive's SHA-256 checksum.
    progress:
        Show a progress bar.
    keep_archive:
        Keep the ``.tar.gz`` files after extracting them.
    base_url:
        Download from this URL instead of the default.
    """
    root = Path(root)
    part_list = [parts] if isinstance(parts, str) else list(parts)
    if splits is None:
        split_list = list(SPLITS)
    elif isinstance(splits, str):
        split_list = [splits]
    else:
        split_list = list(splits)

    for part in part_list:
        if part not in ALL_PARTS:
            raise ValueError(
                f"unknown part {part!r}; expected one of {list(ALL_PARTS)}"
            )
    for split in split_list:
        if split not in SPLITS:
            raise ValueError(f"unknown split {split!r}; expected one of {list(SPLITS)}")

    for part in part_list:
        for split in split_list:
            _ensure_one(
                root, part, split,
                verify=verify, progress=progress,
                keep_archive=keep_archive, base_url=base_url,
            )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="carlanomaly-download",
        description="Download and extract CarlAnomaly dataset shards.",
    )
    parser.add_argument(
        "--root", type=Path,
        help="dataset root directory (archives extract here); "
             "required unless --list is given",
    )
    parser.add_argument(
        "--parts", nargs="+", default=["base"], metavar="PART",
        help=f"parts to fetch (default: base). Choices: {', '.join(ALL_PARTS)}",
    )
    parser.add_argument(
        "--splits", nargs="+", default=list(SPLITS), choices=list(SPLITS),
        help="splits to fetch (default: train test)",
    )
    parser.add_argument(
        "--no-verify", action="store_true", help="skip checksum verification"
    )
    parser.add_argument(
        "--keep-archive", action="store_true",
        help="keep the .tar.gz after extraction",
    )
    parser.add_argument(
        "--base-url", default=None, help="override the download base URL"
    )
    parser.add_argument(
        "--list", action="store_true", help="list available parts and exit"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.list:
        for part, desc in PARTS.items():
            print(f"{part:18s} {desc}")
        return 0

    if args.root is None:
        parser.error("--root is required (unless --list is given)")

    ensure_parts(
        args.root, args.parts, args.splits,
        verify=not args.no_verify,
        keep_archive=args.keep_archive,
        base_url=args.base_url,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
