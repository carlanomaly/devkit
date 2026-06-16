"""Unit tests for the metric computation in :mod:`carlanomaly.evaluator`.

These tests verify metric *correctness* only and never touch disk: ground-truth
labels are supplied in memory (directly to the pure helpers, via path strings, or
by monkeypatching the private label loaders). No scenario data is required.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision

from carlanomaly.evaluator import (
    PixelEvaluator,
    PointEvaluator,
    ScenarioEvaluator,
    SensorEvaluator,
    _bin_indices,
    _fpr_at_tpr,
    _pooled_metrics,
    _resolve_binning,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _histograms(scores: torch.Tensor, labels: torch.Tensor, n_bins: int = 65536,
                transform=None, score_range=None):
    """Bin scores/labels into positive/negative histograms (test mirror of the worker)."""
    tf, lo, hi = _resolve_binning(transform, score_range)
    idx, clamped = _bin_indices(scores, tf, lo, hi, n_bins)
    pos = torch.bincount(idx[labels.bool()], minlength=n_bins)
    neg = torch.bincount(idx[~labels.bool()], minlength=n_bins)
    return pos, neg, clamped


def _make_scores_labels(n: int, seed: int):
    """Imbalanced binary problem with unbounded, partly-negative scores."""
    g = torch.Generator().manual_seed(seed)
    scores = torch.randn(n, generator=g) * 5.0
    labels = (torch.rand(n, generator=g) < torch.sigmoid(scores - 2.0)).long()
    return scores, labels


# ---------------------------------------------------------------------------
# Pooled metrics vs exact reference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_pooled_matches_torchmetrics(seed):
    scores, labels = _make_scores_labels(200_000, seed)
    pos, neg, _ = _histograms(scores, labels)
    m = _pooled_metrics(pos, neg)

    ref_auroc = BinaryAUROC()(scores, labels).item()
    ref_ap = BinaryAveragePrecision()(scores, labels).item()
    ref_fpr = _fpr_at_tpr(scores, labels.bool())

    # 65536 asinh bins => agreement well below 1e-3.
    assert m["auroc"] == pytest.approx(ref_auroc, abs=1e-3)
    assert m["aupr"] == pytest.approx(ref_ap, abs=1e-3)
    assert m["fpr95"] == pytest.approx(ref_fpr, abs=1e-3)


def test_pooled_perfect_and_chance():
    # Perfectly separable: all positives above all negatives.
    scores = torch.tensor([0.1, 0.2, 0.8, 0.9])
    labels = torch.tensor([0, 0, 1, 1])
    pos, neg, _ = _histograms(scores, labels, score_range=(0.0, 1.0))
    m = _pooled_metrics(pos, neg)
    assert m["auroc"] == pytest.approx(1.0, abs=1e-3)
    assert m["aupr"] == pytest.approx(1.0, abs=1e-3)
    assert m["fpr95"] == pytest.approx(0.0, abs=1e-3)


def test_pooled_transform_invariance():
    """A strictly monotonic transform must not change rank-based metrics."""
    scores, labels = _make_scores_labels(100_000, seed=3)
    # Identity over an explicit range (scores standardised into [0,1] by rank-safe affine).
    s01 = torch.sigmoid(scores)  # strictly monotonic squash into (0,1)
    pos_a, neg_a, _ = _histograms(s01, labels, score_range=(0.0, 1.0))
    pos_b, neg_b, _ = _histograms(scores, labels)  # default asinh on raw scores
    ma, mb = _pooled_metrics(pos_a, neg_a), _pooled_metrics(pos_b, neg_b)
    assert ma["auroc"] == pytest.approx(mb["auroc"], abs=2e-3)
    assert ma["aupr"] == pytest.approx(mb["aupr"], abs=2e-3)


def test_clamping_flagged_when_range_too_narrow():
    scores = torch.tensor([-50.0, -1.0, 0.0, 1.0, 50.0])
    labels = torch.tensor([0, 0, 0, 1, 1])
    _, _, clamped = _histograms(scores, labels, score_range=(-2.0, 2.0))
    assert clamped.sum().item() == 2  # the two |x|=50 values
    # Default asinh range clamps nothing of this magnitude.
    _, _, clamped_default = _histograms(scores, labels)
    assert clamped_default.sum().item() == 0


def test_pooled_degenerate_single_class_is_nan():
    pos = torch.zeros(16, dtype=torch.int64)
    neg = torch.ones(16, dtype=torch.int64)
    m = _pooled_metrics(pos, neg)
    assert all(np.isnan(m[k]) for k in ("auroc", "aupr", "fpr95"))


# ---------------------------------------------------------------------------
# The core regression: pooling removes the positive-frame selection bias
# ---------------------------------------------------------------------------


def test_pooling_penalises_traffic_light_shortcut():
    """A detector that scores *all* traffic lights high (and cannot tell a broken
    light from an ordinary one) should look great under per-frame macro-averaging
    over anomaly frames, but poor under pooled evaluation that also sees the many
    high-scored ordinary lights in normal frames.
    """
    HIGH, LOW = 0.9, 0.1

    # Anomaly frame: 1 broken light (label 1) + few ordinary lights + lots of bg.
    anom_scores = torch.tensor([HIGH] + [HIGH] * 9 + [LOW] * 990)
    anom_labels = torch.tensor([1] + [0] * 9 + [0] * 990)

    # Normal frames: dominated by ordinary traffic lights, all scored HIGH, label 0.
    norm_scores = torch.full((5 * 1000,), HIGH)
    norm_labels = torch.zeros(5 * 1000, dtype=torch.long)

    # Per-frame macro over anomaly-containing frames (the biased metric).
    macro = BinaryAUROC()(anom_scores, anom_labels).item()

    # Pooled over all evaluated pixels (the fix).
    all_scores = torch.cat([anom_scores, norm_scores])
    all_labels = torch.cat([anom_labels, norm_labels])
    pos, neg, _ = _histograms(all_scores, all_labels, score_range=(0.0, 1.0))
    pooled = _pooled_metrics(pos, neg)["auroc"]

    assert macro > 0.95, f"macro should be fooled, got {macro}"
    assert pooled < 0.7, f"pooled should expose the shortcut, got {pooled}"


# ---------------------------------------------------------------------------
# Evaluator classes, end-to-end but disk-free (labels injected)
# ---------------------------------------------------------------------------


ANOM = "/data/test/anomaly/Town01/broken_light/0000"
NORM = "/data/test/normal/Town01/0000"


def test_pixel_evaluator_pools_across_frames(monkeypatch):
    HIGH, LOW = 0.9, 0.1
    n = 1000

    masks = {
        (ANOM, 0): torch.tensor([True] + [False] * (n - 1)),
        (NORM, 0): torch.zeros(4 * n, dtype=torch.bool),
    }
    scores = {
        (ANOM, 0): torch.tensor([HIGH] * 10 + [LOW] * (n - 10)),
        (NORM, 0): torch.full((4 * n,), HIGH),  # normal frames full of high-scored lights
    }

    ev = PixelEvaluator(sensor="front", num_workers=2, max_inflight=4,
                        score_range=(0.0, 1.0))
    monkeypatch.setattr(ev, "_load_mask", lambda sid, fid: masks[(sid, fid)])

    for (sid, fid), s in scores.items():
        ev.update(s.unsqueeze(0), [sid], [fid])
    res = ev.compute()

    assert res["n_pixels"] == 5 * n
    assert res["n_positive"] == 1
    assert res["clamped_fraction"] == pytest.approx(0.0)
    assert res["auroc"] < 0.7  # shortcut exposed by including the normal frame
    # Only the anomaly scenario has both classes -> contributes to scenario macro.
    assert res["scenario_macro"]["n_scenarios"] == 1
    assert "broken_light" in res["by_type"]
    assert res["by_type"]["broken_light"]["n_positive"] == 1


def test_pixel_evaluator_empty_compute():
    ev = PixelEvaluator(sensor="front")
    res = ev.compute()
    assert res["n_pixels"] == 0
    assert np.isnan(res["auroc"])


def test_point_evaluator_variable_length(monkeypatch):
    labels = {
        (ANOM, 0): torch.tensor([True, False, False, False]),
        (ANOM, 1): torch.tensor([False, True, False]),  # different length
    }
    scores = {
        (ANOM, 0): torch.tensor([5.0, 0.0, -1.0, -2.0]),
        (ANOM, 1): torch.tensor([0.0, 9.0, 1.0]),
    }

    ev = PointEvaluator(num_workers=2, max_inflight=4)
    monkeypatch.setattr(ev, "_load_labels", lambda sid, fid, n: labels[(sid, fid)])

    for (sid, fid), s in scores.items():
        ev.update([s], [sid], [fid])
    res = ev.compute()
    assert res["n_pixels"] == 7
    assert res["n_positive"] == 2
    assert res["auroc"] == pytest.approx(1.0, abs=1e-3)  # positives are the top scores


# ---------------------------------------------------------------------------
# Frame / scenario tiers
# ---------------------------------------------------------------------------


def test_scenario_evaluator_label_from_path():
    ev = ScenarioEvaluator()
    ev.update(0.9, "/data/test/anomaly/Town01/broken_light/0000")
    ev.update(0.8, "/data/test/anomaly/Town02/broken_light/0001")
    ev.update(0.2, "/data/test/normal/Town01/0000")
    ev.update(0.1, "/data/test/normal/Town02/0001")
    res = ev.compute()
    assert res["n_scenarios"] == 4
    assert res["auroc"] == pytest.approx(1.0)
    assert res["by_type"]["broken_light"]["n_scenarios"] == 2


def test_sensor_evaluator_frame_auroc(monkeypatch):
    # Two scenarios, three frames each; positive on the last frame of the anomaly scenario.
    label_arrays = {
        ANOM: np.array([False, False, True]),
        NORM: np.array([False, False, False]),
    }
    ev = SensorEvaluator(sensor="front")
    monkeypatch.setattr(ev, "_labels_for_scenario", lambda sid: label_arrays[sid])

    ev.update([0.1, 0.2, 0.9], [ANOM, ANOM, ANOM], [0, 1, 2])
    ev.update([0.05, 0.1, 0.15], [NORM, NORM, NORM], [0, 1, 2])
    res = ev.compute()
    assert res["n_frames"] == 6
    assert res["auroc"] == pytest.approx(1.0)  # the one positive frame is top-scored
