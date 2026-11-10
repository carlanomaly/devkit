"""Tests for archive-part selection, detection and the missing-data error.

No network access: :func:`carlanomaly.download.ensure_parts` is replaced with a
recorder and scenario discovery is stubbed, so the wiring tests only check that
each dataset asks for the correct parts.  The detection tests build a tiny fake
dataset layout under ``tmp_path``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import carlanomaly.download as dl
from carlanomaly.datasets import (
    AnomalyScenarioDataset,
    AnomalySensorDataset,
    CameraDataset,
    CarlAnomalyDataset,
    DepthDataset,
    LiDARDataset,
    PointCloudDataset,
    RGBDataset,
    WeatherDataset,
)
from carlanomaly.datasets._base import required_parts
from carlanomaly.index import ScenarioIndex, ScenarioRecord


# ---------------------------------------------------------------------------
# required_parts — pure
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "specs, expected",
    [
        ([("rgb", "front")], ["base"]),
        ([("rgb", "left")], ["base", "camera-extended"]),
        ([("segmentation", "rear")], ["base", "camera-extended"]),
        ([("depth", "front")], ["base", "depth"]),
        ([("pointcloud", None)], ["base", "lidar"]),
        ([("weather", None)], ["base"]),
        ([("anomaly_sensor", "front")], ["base"]),
        ([("anomaly_sensor", "rear")], ["base", "camera-extended"]),
        ([("anomaly_sensor", "lidar")], ["base", "lidar"]),
        ([("anomaly_scenario", None)], ["base"]),
        (
            [("rgb", "left"), ("depth", "front"), ("pointcloud", None)],
            ["base", "camera-extended", "depth", "lidar"],
        ),
    ],
)
def test_required_parts(specs, expected):
    assert required_parts(specs) == expected


def test_required_parts_always_includes_base_first():
    parts = required_parts([("pointcloud", None), ("depth", "front")])
    assert parts[0] == "base"
    assert parts == sorted(set(parts), key=lambda p: (p != "base", p))


# ---------------------------------------------------------------------------
# Dataset wiring — download triggers the right parts (recorded, not fetched)
# ---------------------------------------------------------------------------


@pytest.fixture
def record_download(monkeypatch):
    """Capture ensure_parts(...) calls and stub scenario discovery."""
    calls = []

    def fake_ensure_parts(root, parts, splits, *, verify=True, **kw):
        calls.append({"root": Path(root), "parts": list(parts),
                      "splits": tuple(splits), "verify": verify})

    monkeypatch.setattr(dl, "ensure_parts", fake_ensure_parts)
    monkeypatch.setattr(
        ScenarioIndex, "_discover",
        lambda self: [ScenarioRecord(path=Path("/x/s0"), n_timesteps=1, split="train",
                                     town=None, anomaly_type=None)],
    )
    return calls


def test_rgb_front_downloads_base_only(record_download):
    RGBDataset(root="/data", split="train", direction="front", download=True)
    assert record_download[0]["parts"] == ["base"]
    assert record_download[0]["splits"] == ("train",)


def test_rgb_left_downloads_camera_extended(record_download):
    RGBDataset(root="/data", split="train", direction="left", download=True)
    assert record_download[0]["parts"] == ["base", "camera-extended"]


def test_depth_downloads_depth_part(record_download):
    DepthDataset(root="/data", split="train", download=True)
    assert record_download[0]["parts"] == ["base", "depth"]


def test_pointcloud_downloads_lidar_part(record_download):
    PointCloudDataset(root="/data", split="train", download=True)
    assert record_download[0]["parts"] == ["base", "lidar"]


def test_weather_downloads_base_only(record_download):
    WeatherDataset(root="/data", split="train", download=True)
    assert record_download[0]["parts"] == ["base"]


def test_anomaly_sensor_lidar_downloads_lidar_part(record_download):
    AnomalySensorDataset(root="/data", split="train", sensor="lidar", download=True)
    assert record_download[0]["parts"] == ["base", "lidar"]


def test_anomaly_sensor_left_downloads_camera_extended(record_download):
    AnomalySensorDataset(root="/data", split="train", sensor="left", download=True)
    assert record_download[0]["parts"] == ["base", "camera-extended"]


def test_anomaly_scenario_downloads_base_only(record_download):
    AnomalyScenarioDataset(root="/data", split="train", download=True)
    assert record_download[0]["parts"] == ["base"]


def test_explicit_parts_override_auto_selection(record_download):
    RGBDataset(root="/data", split="train", direction="left", download=True,
               parts=["base"])
    assert record_download[0]["parts"] == ["base"]


def test_download_is_off_by_default(record_download):
    # Nothing on disk and no download=True: nothing is fetched, and the error
    # names the missing parts and the command that fetches them.
    with pytest.raises(FileNotFoundError) as err:
        RGBDataset(root="/data", split="train", direction="left")
    assert record_download == []
    msg = str(err.value)
    assert "base (train), camera-extended (train)" in msg
    assert "carlanomaly-download --root /data --parts base camera-extended --splits train" in msg


def test_composite_without_download_checks_its_parts(record_download):
    with pytest.raises(FileNotFoundError, match="lidar"):
        LiDARDataset(root="/data", split="test")
    assert record_download == []


def test_shared_index_dataset_downloads_its_modality_part(record_download):
    # A dataset attached to a shared index still pulls its modality part
    # (the index's discovery only needs `base`).
    idx = ScenarioIndex(root="/data", split="train", download=False)
    record_download.clear()
    PointCloudDataset(index=idx, download=True)
    assert record_download[0]["parts"] == ["base", "lidar"]
    assert record_download[0]["splits"] == ("train",)
    assert record_download[0]["root"] == Path("/data")


def test_camera_composite_downloads_union(record_download):
    CameraDataset(root="/data", split="train", direction="left", download=True)
    assert record_download[0]["parts"] == ["base", "camera-extended", "depth"]


def test_lidar_composite_downloads_lidar(record_download):
    LiDARDataset(root="/data", split="train", download=True)
    assert record_download[0]["parts"] == ["base", "lidar"]


def test_joint_composite_downloads_everything(record_download):
    CarlAnomalyDataset(root="/data", split="train", download=True)
    assert record_download[0]["parts"] == ["base", "camera-extended", "depth", "lidar"]


def test_joint_front_only_skips_camera_extended(record_download):
    CarlAnomalyDataset(root="/data", split="train", cameras=["front"], download=True)
    assert record_download[0]["parts"] == ["base", "depth", "lidar"]


# ---------------------------------------------------------------------------
# Detection of parts that were extracted by hand (no marker file)
# ---------------------------------------------------------------------------


def _fake_layout(root: Path, dirs: list, scenarios: int = 2) -> None:
    """train/Town01/scenario-<i>/<dir>/000000.jpg for every scenario and dir."""
    for i in range(scenarios):
        for d in dirs:
            (root / "train" / "Town01" / f"scenario-{i}" / d).mkdir(parents=True)
            (root / "train" / "Town01" / f"scenario-{i}" / d / "000000.jpg").touch()


def test_part_present_from_directories(tmp_path):
    _fake_layout(tmp_path, ["rgb-front", "pointclouds"])
    assert dl.part_present(tmp_path, "base", "train")
    assert dl.part_present(tmp_path, "lidar", "train")
    assert not dl.part_present(tmp_path, "depth", "train")
    assert not dl.part_present(tmp_path, "base", "test")


def test_part_present_requires_every_scenario(tmp_path):
    # an interrupted manual extraction of the lidar archive
    _fake_layout(tmp_path, ["rgb-front"])
    (tmp_path / "train" / "Town01" / "scenario-0" / "pointclouds").mkdir()
    assert not dl.part_present(tmp_path, "lidar", "train")


def test_part_present_from_marker(tmp_path):
    marker = tmp_path / dl.DOWNLOAD_SUBDIR / "depth-test.done"
    marker.parent.mkdir()
    marker.touch()
    assert dl.part_present(tmp_path, "depth", "test")


def test_ensure_parts_skips_parts_found_on_disk(tmp_path, monkeypatch):
    _fake_layout(tmp_path, ["rgb-front"])

    def no_network(*args, **kwargs):
        raise AssertionError("must not download a part that is already extracted")

    monkeypatch.setattr(dl, "_download", no_network)
    dl.ensure_parts(tmp_path, "base", "train")
    assert (tmp_path / dl.DOWNLOAD_SUBDIR / "base-train.done").exists()


def test_dataset_without_download_loads_extracted_data(tmp_path):
    _fake_layout(tmp_path, ["rgb-front"])
    ds = RGBDataset(tmp_path, split="train")
    assert len(ds.index.records) == 2


def test_empty_index_names_the_missing_base_part(tmp_path):
    with pytest.raises(FileNotFoundError, match=r"base \(test\)"):
        ScenarioIndex(tmp_path, split="test_normal")


# ---------------------------------------------------------------------------
# _safe_extract — the dataset LICENSE shipped in every archive
# ---------------------------------------------------------------------------


def _archive_with_license(path: Path, license_text: str = "CC BY 4.0") -> Path:
    """A part archive like the published ones: LICENSE first, then dataset files."""
    import io
    import tarfile

    with tarfile.open(path, "w:gz") as tar:
        for name, data in [
            ("LICENSE", license_text.encode()),
            ("train/Town01/scenario-0/rgb-front/000000.jpg", b"jpg"),
        ]:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return path


@pytest.fixture(params=["data_filter", "manual_check"])
def extract_path(request, monkeypatch):
    """Run each test with the tarfile data filter and with the fallback for Pythons without it."""
    import tarfile

    if request.param == "manual_check":
        monkeypatch.delattr(tarfile, "data_filter")
    return request.param


def test_extract_writes_license_into_empty_root(tmp_path, extract_path):
    archive = _archive_with_license(tmp_path / "part.tar.gz")
    root = tmp_path / "data"
    root.mkdir()
    dl._safe_extract(archive, root)
    assert (root / "LICENSE").read_text() == "CC BY 4.0"
    assert (root / "train/Town01/scenario-0/rgb-front/000000.jpg").exists()


def test_extract_keeps_existing_license(tmp_path, extract_path):
    # a LICENSE that cannot be overwritten, e.g. created by another user in a
    # shared dataset directory, must not stop the part from extracting
    archive = _archive_with_license(tmp_path / "part.tar.gz", "new text")
    root = tmp_path / "data"
    root.mkdir()
    (root / "LICENSE").write_text("from an earlier part")
    (root / "LICENSE").chmod(0o444)
    dl._safe_extract(archive, root)
    assert (root / "LICENSE").read_text() == "from an earlier part"
    assert (root / "train/Town01/scenario-0/rgb-front/000000.jpg").exists()
