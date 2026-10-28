"""Unit tests for the metric computation in :mod:`carlanomaly.evaluator`.

These tests verify metric *correctness* only and never touch disk: ground-truth
labels are supplied in memory (directly to the pure helpers, via path strings, or
by monkeypatching the private label loaders). No scenario data is required.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from pytorch_ood.metrics import functional as ood_metrics

from carlanomaly.evaluator import (
    PixelEvaluator,
    PointEvaluator,
    ScenarioEvaluator,
    SensorEvaluator,
    _bin_indices,
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


def _ood_labels(labels: torch.Tensor) -> torch.Tensor:
    """Binary anomaly labels -> pytorch-ood convention (label < 0 is the positive class)."""
    return torch.where(labels.bool(), -1, 0)


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
def test_pooled_matches_exact(seed):
    scores, labels = _make_scores_labels(200_000, seed)
    pos, neg, _ = _histograms(scores, labels)
    m = _pooled_metrics(pos, neg)

    ood_labels = _ood_labels(labels)
    ref_auroc = ood_metrics.auroc(scores, ood_labels).item()
    ref_ap = ood_metrics.aupr(scores, ood_labels, positive="ood").item()
    ref_fpr = ood_metrics.fpr_at_tpr(scores, ood_labels, tpr=0.95).item()

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
# The core regression: pooling removes the positive-timestep selection bias
# ---------------------------------------------------------------------------


def test_pooling_penalises_traffic_light_shortcut():
    """A detector that scores *all* traffic lights high (and cannot tell a broken
    light from an ordinary one) should look great under per-timestep macro-averaging
    over anomalous timesteps, but poor under pooled evaluation that also sees the many
    high-scored ordinary lights in normal timesteps.
    """
    HIGH, LOW = 0.9, 0.1

    # Anomalous timestep: 1 broken light (label 1) + few ordinary lights + lots of bg.
    anom_scores = torch.tensor([HIGH] + [HIGH] * 9 + [LOW] * 990)
    anom_labels = torch.tensor([1] + [0] * 9 + [0] * 990)

    # Normal timesteps: dominated by ordinary traffic lights, all scored HIGH, label 0.
    norm_scores = torch.full((5 * 1000,), HIGH)
    norm_labels = torch.zeros(5 * 1000, dtype=torch.long)

    # Per-timestep macro over anomaly-containing timesteps (the biased metric).
    macro = ood_metrics.auroc(anom_scores, _ood_labels(anom_labels)).item()

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


def test_pixel_evaluator_pools_across_timesteps(monkeypatch):
    HIGH, LOW = 0.9, 0.1
    n = 1000

    masks = {
        (ANOM, 0): torch.tensor([True] + [False] * (n - 1)),
        (NORM, 0): torch.zeros(4 * n, dtype=torch.bool),
    }
    scores = {
        (ANOM, 0): torch.tensor([HIGH] * 10 + [LOW] * (n - 10)),
        (NORM, 0): torch.full((4 * n,), HIGH),  # normal timesteps full of high-scored lights
    }

    ev = PixelEvaluator(sensor="front", num_workers=2, max_inflight=4,
                        score_range=(0.0, 1.0))
    monkeypatch.setattr(ev, "_load_mask", lambda sid, fid: masks[(sid, fid)])

    for (sid, fid), s in scores.items():
        ev.update(s.unsqueeze(0), [sid], [fid])
    res = ev.compute()

    assert res["n_samples"] == 5 * n
    assert res["n_positive"] == 1
    assert res["clamped_fraction"] == pytest.approx(0.0)
    assert res["auroc"] < 0.7  # shortcut exposed by including the normal timestep
    # Only the anomaly scenario has both classes -> contributes to scenario macro.
    assert res["scenario_macro"]["n_scenarios"] == 1
    assert "broken_light" in res["by_type"]
    assert res["by_type"]["broken_light"]["n_positive"] == 1


def test_pixel_evaluator_empty_compute():
    ev = PixelEvaluator(sensor="front")
    res = ev.compute()
    assert res["n_samples"] == 0
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
    assert res["n_samples"] == 7
    assert res["n_positive"] == 2
    assert res["auroc"] == pytest.approx(1.0, abs=1e-3)  # positives are the top scores


# ---------------------------------------------------------------------------
# Sensor / timestep / scenario levels
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
    # Perfect separation: AUPR is 1.0 and FPR95 is 0.0 too.
    assert res["aupr"] == pytest.approx(1.0)
    assert res["fpr95"] == pytest.approx(0.0)
    assert res["by_type"]["broken_light"]["n_scenarios"] == 2
    assert res["by_type"]["broken_light"]["aupr"] == pytest.approx(1.0)


def test_sensor_evaluator_auroc(monkeypatch):
    # Two scenarios, three timesteps each; positive on the last timestep of the anomaly scenario.
    label_arrays = {
        ANOM: np.array([False, False, True]),
        NORM: np.array([False, False, False]),
    }
    ev = SensorEvaluator(sensor="front")
    monkeypatch.setattr(ev, "_labels_for_scenario", lambda sid: label_arrays[sid])

    ev.update([0.1, 0.2, 0.9], [ANOM, ANOM, ANOM], [0, 1, 2])
    ev.update([0.05, 0.1, 0.15], [NORM, NORM, NORM], [0, 1, 2])
    res = ev.compute()
    assert res["n_timesteps"] == 6
    assert res["auroc"] == pytest.approx(1.0)  # the one positive timestep is top-scored
    # Perfect separation here too.
    assert res["aupr"] == pytest.approx(1.0)
    assert res["fpr95"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Raw-score metrics are exact for any score range
# ---------------------------------------------------------------------------


def test_scenario_evaluator_large_scores_not_saturated():
    """Regression: torchmetrics sigmoids scores outside [0, 1], so in float32 every
    score above ~17 tied at 1.0 and AUROC came back 0.5 for a perfect ranking."""
    ev = ScenarioEvaluator()
    for score, sid in [
        (20.0, NORM),
        (25.0, "/data/test/normal/Town02/0001"),
        (30.0, ANOM),
        (35.0, "/data/test/anomaly/Town02/broken_light/0001"),
        (40.0, "/data/test/anomaly/Town03/broken_light/0002"),
    ]:
        ev.update(score, sid)
    res = ev.compute()
    assert res["auroc"] == pytest.approx(1.0)
    assert res["aupr"] == pytest.approx(1.0)
    assert res["fpr95"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    "shift, scale", [(0.0, 1.0), (1e3, 1.0), (0.0, -1e-9), (0.0, 1e6)]
)
def test_timestep_metrics_invariant_to_score_range(monkeypatch, shift, scale):
    """Rank metrics must not change under shifting or rescaling the scores, including
    into tiny negative values (which a float32 sigmoid squashes onto 0.5) and huge
    ones (which it saturates). ``scale < 0`` flips the order, so that case is
    compared against the negated scores."""
    g = torch.Generator().manual_seed(0)
    raw = torch.randn(400, generator=g).double()
    labels = (raw + 0.8 * torch.randn(400, generator=g).double()) > 1.0
    sids = [ANOM if i < 200 else NORM for i in range(400)]
    fids = [i % 200 for i in range(400)]
    label_arrays = {ANOM: labels[:200].numpy(), NORM: labels[200:].numpy()}

    def run(scores):
        ev = SensorEvaluator(sensor="front")
        monkeypatch.setattr(ev, "_labels_for_scenario", lambda sid: label_arrays[sid])
        ev.update(scores.tolist(), sids, fids)
        return ev.compute()

    ref = run(raw if scale > 0 else -raw)
    res = run(raw * scale + shift)
    for k in ("auroc", "aupr", "fpr95"):
        assert res[k] == pytest.approx(ref[k], abs=1e-12)
