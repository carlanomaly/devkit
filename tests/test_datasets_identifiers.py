"""Tests for :class:`WithIdentifiers` and the label datasets that need no files.

No disk access: scenario discovery is stubbed with in-memory records.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import Dataset

from carlanomaly.datasets import AnomalyScenarioDataset, AnomalySensorDataset, WithIdentifiers
from carlanomaly.index import ScenarioIndex, ScenarioRecord


@pytest.fixture
def stub_discovery(monkeypatch):
    """Stub discovery with one normal and one anomaly test scenario (3 timesteps each)."""
    records = [
        ScenarioRecord(path=Path("/x/test/normal/Town01/scenario-0"), n_timesteps=3,
                       split="test_normal", town="Town01", anomaly_type=None),
        ScenarioRecord(path=Path("/x/test/anomaly/Town01/vanish-actor/scenario-1"),
                       n_timesteps=3, split="test_anomaly", town="Town01",
                       anomaly_type="vanish-actor"),
    ]
    monkeypatch.setattr(ScenarioIndex, "_discover", lambda self: list(records))
    return records


# ---------------------------------------------------------------------------
# WithIdentifiers
# ---------------------------------------------------------------------------


def test_with_identifiers_items(stub_discovery):
    ds = AnomalyScenarioDataset(root="/x", split="test", download=False)
    wrapped = WithIdentifiers(ds)

    assert len(wrapped) == len(ds) == 6  # 2 scenarios x 3 timesteps (clip_len=1)

    item = wrapped[4]  # second scenario, timestep 1
    assert set(item) == {"data", "scenario_id", "timesteps"}
    assert item["scenario_id"] == str(stub_discovery[1].path)
    assert torch.equal(item["timesteps"], torch.tensor([1]))
    assert torch.equal(item["data"], ds[4])


def test_with_identifiers_respects_clip_len(stub_discovery):
    ds = AnomalyScenarioDataset(root="/x", split="test", download=False, clip_len=3)
    wrapped = WithIdentifiers(ds)
    assert len(wrapped) == 2  # one clip per scenario
    assert torch.equal(wrapped[0]["timesteps"], torch.tensor([0, 1, 2]))


def test_with_identifiers_rejects_foreign_datasets():
    class Plain(Dataset):
        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return torch.zeros(1)

    with pytest.raises(TypeError, match="ScenarioIndex"):
        WithIdentifiers(Plain())


# ---------------------------------------------------------------------------
# AnomalyScenarioDataset / AnomalySensorDataset
# ---------------------------------------------------------------------------


def test_anomaly_scenario_labels_from_split(stub_discovery):
    ds = AnomalyScenarioDataset(root="/x", split="test", download=False)
    labels = [bool(ds[i]) for i in range(len(ds))]
    assert labels == [False] * 3 + [True] * 3  # normal scenario first, then anomaly


def test_anomaly_scenario_label_is_scalar(stub_discovery):
    ds = AnomalyScenarioDataset(root="/x", split="test", download=False)
    assert ds[0].shape == ()
    assert ds[0].dtype == torch.bool


def test_anomaly_sensor_rejects_unknown_sensor():
    with pytest.raises(ValueError, match="sensor"):
        AnomalySensorDataset(root="/x", split="test", sensor="radar", download=False)


def test_anomaly_sensor_train_is_all_false(monkeypatch):
    monkeypatch.setattr(
        ScenarioIndex, "_discover",
        lambda self: [ScenarioRecord(path=Path("/x/train/Town01/scenario-0"),
                                     n_timesteps=2, split="train", town="Town01",
                                     anomaly_type=None)],
    )
    ds = AnomalySensorDataset(root="/x", split="train", sensor="lidar",
                              download=False, clip_len=2)
    assert torch.equal(ds[0], torch.zeros(2, dtype=torch.bool))
