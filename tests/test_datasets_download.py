"""Tests for automatic archive-part selection when ``download=True``.

No network or disk access: :func:`carlanomaly.download.ensure_parts` is replaced
with a recorder and scenario discovery is stubbed, so we only check that each
dataset asks for the correct parts.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import carlanomaly.download as dl
from carlanomaly.datasets import (
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
        lambda self: [ScenarioRecord(path=Path("/x/s0"), n_frames=1, split="train",
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


def test_explicit_parts_override_auto_selection(record_download):
    RGBDataset(root="/data", split="train", direction="left", download=True,
               parts=["base"])
    assert record_download[0]["parts"] == ["base"]


def test_no_download_means_no_fetch(record_download):
    RGBDataset(root="/data", split="train", direction="left", download=False)
    assert record_download == []


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
