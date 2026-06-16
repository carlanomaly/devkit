"""Auto-download and extraction of CarlAnomaly dataset shards.

The dataset is published as modular ``tar.gz`` archives ("parts"), each split
into a ``train`` and a ``test`` archive, hosted at :data:`DATA_BASE_URL`.  Every
archive stores dataset-root-relative paths, so extracting it at the dataset root
reconstructs the ``train/<town>/scenario-N/...`` / ``test/...`` layout that
:class:`carlanomaly.index.ScenarioIndex` expects.

The high-level entry point is :func:`ensure_parts`, which is idempotent: it
downloads (with HTTP-range resume), verifies the SHA-256 checksum, extracts, and
writes a marker file so subsequent calls are no-ops.  It is invoked automatically
when a :class:`ScenarioIndex` is constructed with ``download=True``; it can also
be used directly or via ``python -m carlanomaly.download``.

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
from typing import Dict, Optional, Sequence, Tuple, Union
from urllib.error import HTTPError
from urllib.request import Request, urlopen

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

#: Base URL the archives are served from.  Override with the
#: ``CARLANOMALY_BASE_URL`` environment variable (no trailing slash).
DATA_BASE_URL = os.environ.get(
    "CARLANOMALY_BASE_URL", "https://data.kkirchheim.de/carlanomaly/v1"
).rstrip("/")

#: All downloadable parts, with a short human-readable description of the
#: directories each one contributes to a scenario (see ``package_dataset.sh``).
PARTS: Dict[str, str] = {
    "base": "Front camera + segmentation, GNSS/IMU/weather/actions/collisions, "
            "and (test only) front anomaly masks + observation labels.",
    "camera-extended": "Left/right/rear RGB + segmentation, and (test only) "
                       "their anomaly masks.",
    "lidar": "LiDAR point clouds, and (test only) per-point anomaly labels.",
    "depth": "Depth maps for all four cameras.",
    "kitti": "LiDAR labels in KITTI format.",
    "carla-recordings": "Raw CARLA simulator recordings (sim.log).",
}

ALL_PARTS: Tuple[str, ...] = tuple(PARTS)
SPLITS: Tuple[str, ...] = ("train", "test")

#: SHA-256 checksums keyed by ``(part, split)``.  Sourced from ``shasums.md``.
#: Note: ``("base", "train")`` is intentionally absent: no checksum has been
#: published for it, so that archive is downloaded but not verified.
SHA256: Dict[Tuple[str, str], str] = {
    ("base", "test"): "267e48f2249deb0269ad950aa81bca57dc02e3bbdf2d73acc172af267b18254a",
    ("camera-extended", "test"): "6daa1afc80df1766af4265822279f8f73f9c380f89a3902f311297c5aff1f621",
    ("camera-extended", "train"): "11f5d0ecf124cc42a143acd3757c0ece938b6c29fb77754a5143d8cc61cc9b51",
    ("carla-recordings", "test"): "25d039cfe5cef942378132019c70b9a05f56b14a2ccf58873f0b76f5aebb0a2c",
    ("carla-recordings", "train"): "4b5d9843a2ef811a11c4e628d805547444b68cfedabf208a66ed0dfe255a5a58",
    ("depth", "test"): "3d596f30830d6b7b25208fc51a2f2b86df15538d53d791020db5759cc1fc48d5",
    ("depth", "train"): "19be433364424e051f462a888c1471b7b55b5001160f3778206695973bf26906",
    ("kitti", "test"): "3f9a16a97e81c2279cb6e1a6172b065b3009399336574596ed2ebbc863badccc",
    ("kitti", "train"): "49838ef473164965d59f4878b056e8abe915c809f61f48aea80fbacad3a734d1",
    ("lidar", "test"): "fd7ebe33e824ab20005f8ee4ce7773cd1cc3c66d32e965fe4e91fb2b13b6386c",
    ("lidar", "train"): "0d6b8f27deaca746a48da5c92d8c3b06a4d4f032a7a3245cdba2db77e2f8d998",
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
    "anomaly_obs": "base",
    "kitti": "kitti",
    "carla_recordings": "carla-recordings",
}

# Camera modalities whose "front" direction is in `base` but whose other
# directions live in `camera-extended`.
_CAMERA_MODALITIES = {"rgb", "segmentation", "anomaly_seg"}


def part_for(modality: str, direction: Optional[str] = None) -> str:
    """Return the archive part that contains a given modality/direction.

    Depth maps are bundled together regardless of camera direction; the RGB,
    segmentation and pixel-anomaly modalities split by direction: ``front`` is
    in ``base``, the other directions are in ``camera-extended``.
    """
    if modality not in _MODALITY_PART:
        raise ValueError(
            f"unknown modality {modality!r}; expected one of {sorted(_MODALITY_PART)}"
        )
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


def _safe_extract(tar_path: Path, dest: Path) -> None:
    """Extract ``tar_path`` into ``dest``, guarding against path traversal."""
    dest = dest.resolve()
    with tarfile.open(tar_path, "r:gz") as tar:
        if hasattr(tarfile, "data_filter"):
            # Python >= 3.12: vetted extraction filter.
            tar.extractall(dest, filter="data")
        else:
            prefix = str(dest) + os.sep
            for member in tar.getmembers():
                target = (dest / member.name).resolve()
                if target != dest and not str(target).startswith(prefix):
                    raise RuntimeError(
                        f"refusing to extract {member.name!r}: escapes {dest}"
                    )
            tar.extractall(dest)


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
    """Download and extract dataset parts into ``root`` (idempotent).

    Parameters
    ----------
    root:
        Dataset root (archives extract to ``root/{train,test}/...``).
    parts:
        One part name or a sequence of them (see :data:`PARTS`).
    splits:
        ``'train'``, ``'test'``, or both.  Defaults to both.
    verify:
        Verify the SHA-256 checksum after download.  When no checksum is
        published for an archive, a warning is logged and the file is kept.
    progress:
        Show a download progress bar (tqdm if available, else textual).
    keep_archive:
        Keep the downloaded ``.tar.gz`` after extraction (default: delete it).
    base_url:
        Override :data:`DATA_BASE_URL` for this call.
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
