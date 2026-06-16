"""Multi-level evaluators for the CarlAnomaly benchmark.

Evaluation is split into four independent tiers, each a pure metric
calculator that loads its own ground-truth labels from disk.  The caller
only ever provides anomaly *scores* plus identifiers (``scenario_id`` and
``frame_id``); reducing pixel/point scores to frame scores and fusing
multiple sensors is the caller's responsibility.

Identifiers
-----------
``scenario_id`` is the filesystem path to the scenario directory (the value
the datasets expose under the same key).  Evaluators use it both to locate a
scenario's label files and to parse its anomaly type and town from the path,
so it must follow the on-disk layout, e.g.
``.../test/anomaly/{town}/{anomaly_type}/{run}`` or
``.../test/normal/{town}/{run}``.  ``frame_id`` is the integer frame number,
used to index per-frame labels and to build per-frame label paths
(``{frame_id:06d}.png`` / ``.feather``).

Tiers
-----
- :class:`PixelEvaluator` / :class:`PointEvaluator`: spatial metrics (AUROC,
  AUPR, FPR@95TPR) pooled over every evaluated pixel/point in the split
  (anomalous frames, anomaly-free frames, and normal scenarios alike) using
  bounded-memory streaming histograms.
- :class:`SensorEvaluator`: frame-level AUROC against sensor-specific labels
  (``anomaly-{sensor}/sensor.feather``).
- :class:`ObservationEvaluator`: frame-level AUROC against observation-level
  labels (``anomaly-observation.feather``).
- :class:`ScenarioEvaluator`: one score per scenario; label inferred from the
  scenario path.
"""

from __future__ import annotations

import math
import re
import threading
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchmetrics.classification import BinaryAUROC

from .index import CAMERAS

SENSORS = CAMERAS + ("lidar",)

# Native camera resolution, used when a mask file is missing.
_MASK_H, _MASK_W = 1080, 1920

# Spatial-tier async defaults: worker threads load labels + compute metrics off
# the main loop; max_inflight bounds host RAM held by queued score tensors.
_DEFAULT_WORKERS = 32
_DEFAULT_MAX_INFLIGHT = 32


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _fpr_at_tpr(scores: torch.Tensor, labels: torch.Tensor, tpr_target: float = 0.95) -> float:
    """Compute FPR at a given TPR threshold."""
    pos = labels.bool()
    neg = ~pos
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")

    thresholds = torch.sort(scores[pos], descending=True).values
    idx = max(0, int(tpr_target * pos.sum().item()) - 1)
    threshold = thresholds[min(idx, len(thresholds) - 1)]
    fpr = (scores[neg] >= threshold).float().mean().item()
    return fpr


def _parse_anomaly_type(scenario_id: str) -> Optional[str]:
    """Extract anomaly type from a scenario path string.

    Expected patterns:
        .../test/anomaly/{town}/{anomaly_type}/...
        .../test/normal/{town}/...
    """
    match = re.search(r"/test/anomaly/[^/]+/([^/]+)/", scenario_id)
    if match:
        return match.group(1)
    return None


def _parse_town(scenario_id: str) -> Optional[str]:
    match = re.search(r"/(Town\w+)/", scenario_id)
    if match:
        return match.group(1)
    return None


def _mean(values: Sequence[float]) -> float:
    valid = [v for v in values if v == v]  # filter NaN
    return sum(valid) / len(valid) if valid else float("nan")


def _auroc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """AUROC over a flat list of scores/labels; NaN if labels are degenerate."""
    if not scores or all(labels) or not any(labels):
        return float("nan")
    metric = BinaryAUROC()
    metric.update(
        torch.as_tensor(scores, dtype=torch.float32),
        torch.as_tensor(labels, dtype=torch.long),
    )
    return metric.compute().item()


def _as_float_list(scores: Any) -> List[float]:
    """Coerce a batch of scalar scores (tensor/array/sequence) to a float list.

    Rejects non-scalar elements so that spatial data (e.g. pixel maps) passed
    to a frame-level evaluator fails loudly rather than being silently reduced.
    """
    if isinstance(scores, torch.Tensor):
        t = scores.detach().cpu().float()
        if t.ndim != 1:
            raise ValueError(
                f"expected a 1-D batch of scalar scores, got tensor with shape "
                f"{tuple(t.shape)}; reduce spatial scores to per-frame scalars first"
            )
        return t.tolist()
    if isinstance(scores, np.ndarray):
        if scores.ndim != 1:
            raise ValueError(
                f"expected a 1-D batch of scalar scores, got array with shape {scores.shape}"
            )
        return [float(v) for v in scores]
    out: List[float] = []
    for v in scores:
        if isinstance(v, torch.Tensor):
            if v.numel() != 1:
                raise ValueError("each score must be a scalar, got a non-scalar tensor")
            out.append(float(v.item()))
        else:
            out.append(float(v))
    return out


# ---------------------------------------------------------------------------
# Tier 1: Pixel / Point: pooled spatial metrics over the whole split
# ---------------------------------------------------------------------------

#: |score| beyond this saturates the default ``asinh`` histogram range.
_ASINH_MAX_SCORE = 1e6
#: Bins for the global / per-type histograms (the primary, high-resolution view).
_DEFAULT_N_BINS = 65536
#: Bins for per-scenario histograms (a coarser secondary diagnostic).
_DEFAULT_SCENARIO_BINS = 4096

#: Anything accepted as a monotonic score transform applied before binning.
ScoreTransform = Callable[[torch.Tensor], torch.Tensor]


def _resolve_binning(
    score_transform: Optional[ScoreTransform],
    score_range: Optional[Tuple[float, float]],
) -> Tuple[ScoreTransform, float, float]:
    """Return ``(transform, lo, hi)`` describing the histogram binning.

    With neither argument given, scores are binned over ``asinh(score)`` across
    a wide range, so unbounded scores (MaxLogit, MSE) need no per-model tuning.
    ``asinh`` is linear near 0 and logarithmic in the tails, so (unlike a
    sigmoid) it never saturates and preserves tail resolution where FPR95 lives.
    AUROC/AUPR/FPR95 are rank-based, so any strictly monotonic transform leaves
    their true values unchanged; binning only trades within-bin resolution.
    """
    if score_transform is None and score_range is None:
        m = math.asinh(_ASINH_MAX_SCORE)
        return torch.asinh, -m, m
    if score_transform is None:
        assert score_range is not None
        lo, hi = score_range
        return (lambda x: x), float(lo), float(hi)
    lo, hi = score_range if score_range is not None else (0.0, 1.0)
    return score_transform, float(lo), float(hi)


def _bin_indices(
    scores: torch.Tensor,
    transform: ScoreTransform,
    lo: float,
    hi: float,
    n_bins: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Map scores to ``[0, n_bins)`` bin indices; flag values outside ``[lo, hi]``."""
    t = transform(scores)
    idx = ((t - lo) / (hi - lo) * n_bins).floor().long()
    clamped = (idx < 0) | (idx >= n_bins)
    return idx.clamp_(0, n_bins - 1), clamped


def _pooled_metrics(h_pos: torch.Tensor, h_neg: torch.Tensor) -> Dict[str, float]:
    """AUROC / AUPR / FPR@95TPR from positive/negative score histograms.

    Bins are ordered low→high score. Returns NaN for a degenerate
    (single-class) histogram, mirroring :func:`_auroc`.
    """
    h_pos = h_pos.double()
    h_neg = h_neg.double()
    n_pos = h_pos.sum()
    n_neg = h_neg.sum()
    if n_pos == 0 or n_neg == 0:
        return {"auroc": float("nan"), "aupr": float("nan"), "fpr95": float("nan")}

    # Reverse-cumulative: tp[b] = #positives in bins >= b (i.e. threshold at bin b).
    tp = torch.flip(torch.cumsum(torch.flip(h_pos, [0]), 0), [0])
    fp = torch.flip(torch.cumsum(torch.flip(h_neg, [0]), 0), [0])
    tpr = tp / n_pos
    fpr = fp / n_neg

    # ROC: sweep threshold from above the top bin (0, 0) down to below bin 0 (1, 1).
    zero = torch.zeros(1, dtype=torch.double)
    tpr_curve = torch.cat([zero, torch.flip(tpr, [0])])
    fpr_curve = torch.cat([zero, torch.flip(fpr, [0])])
    auroc = torch.trapz(tpr_curve, fpr_curve).item()

    # AUPR as step-interpolated average precision (torchmetrics/sklearn convention).
    precision = tp / (tp + fp).clamp(min=1.0)
    rec_f = torch.flip(tpr, [0])
    prec_f = torch.flip(precision, [0])
    rec_prev = torch.cat([zero, rec_f[:-1]])
    aupr = ((rec_f - rec_prev) * prec_f).sum().item()

    # FPR at the highest threshold still reaching TPR >= 0.95 (bin 0 always does).
    b_star = int((tpr >= 0.95).nonzero().max().item())
    fpr95 = (fp[b_star] / n_neg).item()

    return {"auroc": auroc, "aupr": aupr, "fpr95": fpr95}


class _SpatialEvaluator:
    """Pooled per-element spatial metrics over the entire evaluated split.

    Unlike a per-frame macro-average, every pixel/point (from anomalous frames,
    anomaly-free frames, and normal scenarios alike) enters a single pooled
    binary problem.  This removes the selection bias of conditioning the negative
    distribution on "the frame contains an anomaly" (see ``plan.md`` §3).

    Scores are accumulated into bounded-memory score histograms; label loading
    (PNG/feather decode) and binning run on a pool of worker threads so they
    overlap the GPU's next forward pass.  ``update()`` submits per-frame tasks
    and ``compute()`` joins them, draining every outstanding future before
    aggregating.  Three views are derived from one set of histograms:

    - **global** pooled AUROC/AUPR/FPR95 (the primary metric);
    - **by_type** pooled metrics, grouped by anomaly type;
    - **scenario_macro**: per-scenario metrics averaged over scenarios that
      contain positives (an equal-per-scenario secondary diagnostic, §5).

    See :class:`PixelEvaluator` for the constructor parameters.
    """

    def __init__(
        self,
        num_workers: int = _DEFAULT_WORKERS,
        max_inflight: int = _DEFAULT_MAX_INFLIGHT,
        *,
        n_bins: int = _DEFAULT_N_BINS,
        scenario_bins: int = _DEFAULT_SCENARIO_BINS,
        score_transform: Optional[ScoreTransform] = None,
        score_range: Optional[Tuple[float, float]] = None,
    ) -> None:
        if n_bins % scenario_bins != 0:
            raise ValueError(
                f"n_bins ({n_bins}) must be a multiple of scenario_bins ({scenario_bins})"
            )
        self._n_bins = n_bins
        self._scenario_bins = scenario_bins
        self._transform, self._lo, self._hi = _resolve_binning(score_transform, score_range)
        # key -> [pos_hist, neg_hist]; type key None == normal scenarios.
        self._by_type: Dict[Optional[str], List[torch.Tensor]] = {}
        self._by_scenario: Dict[str, List[torch.Tensor]] = {}
        self._n_clamped = 0
        self._n_total = 0
        self._executor = ThreadPoolExecutor(max_workers=num_workers)
        self._futures: List[Future] = []
        self._sem = threading.Semaphore(max_inflight)

    def _submit(
        self,
        scores: torch.Tensor,
        load_labels: Callable[[], torch.Tensor],
        scenario_id: str,
    ) -> None:
        """Queue one frame's label-load + histogram binning on a worker thread.

        ``scores`` stays on its source device (the model's GPU); the worker
        decodes the label file on the CPU (parallel I/O, GIL released), bins both
        classes, and returns small ``n_bins``-length count tensors.  Blocks if
        ``max_inflight`` frames are already queued (backpressure), bounding both
        host work and retained GPU score tensors.  Every frame contributes,
        including frames with no positive labels (they are pure-negative pools).
        """
        self._sem.acquire()
        atype = _parse_anomaly_type(scenario_id)
        n_bins = self._n_bins

        def task() -> Tuple[torch.Tensor, torch.Tensor, int, Optional[str], str]:
            try:
                labels = load_labels().reshape(-1).bool().to(scores.device)
                idx, clamped = _bin_indices(scores, self._transform, self._lo, self._hi, n_bins)
                pos = torch.bincount(idx[labels], minlength=n_bins).cpu()
                neg = torch.bincount(idx[~labels], minlength=n_bins).cpu()
                return pos, neg, int(clamped.sum().item()), atype, scenario_id
            finally:
                self._sem.release()

        self._futures.append(self._executor.submit(task))

    @staticmethod
    def _accumulate(
        store: Dict[Any, List[torch.Tensor]], key: Any,
        pos: torch.Tensor, neg: torch.Tensor, n_bins: int,
    ) -> None:
        if key not in store:
            store[key] = [
                torch.zeros(n_bins, dtype=torch.int64),
                torch.zeros(n_bins, dtype=torch.int64),
            ]
        store[key][0].add_(pos)
        store[key][1].add_(neg)

    def _drain(self) -> None:
        """Wait for every queued frame, accumulating counts in submission order.

        ``Future.result()`` re-raises any exception a worker hit (e.g. a corrupt
        mask), so failures surface here rather than being silently dropped.
        """
        group = self._n_bins // self._scenario_bins
        for fut in self._futures:
            pos, neg, n_clamped, atype, sid = fut.result()
            self._accumulate(self._by_type, atype, pos, neg, self._n_bins)
            # Coarsen to scenario resolution by summing adjacent bin groups.
            ds_pos = pos.view(self._scenario_bins, group).sum(1)
            ds_neg = neg.view(self._scenario_bins, group).sum(1)
            self._accumulate(self._by_scenario, sid, ds_pos, ds_neg, self._scenario_bins)
            self._n_clamped += n_clamped
            self._n_total += int(pos.sum().item() + neg.sum().item())
        self._futures.clear()

    def compute(self) -> Dict[str, Any]:
        self._drain()
        if not self._by_type:
            return {
                "auroc": float("nan"), "aupr": float("nan"), "fpr95": float("nan"),
                "n_pixels": 0, "n_positive": 0, "clamped_fraction": float("nan"),
            }

        g_pos = torch.zeros(self._n_bins, dtype=torch.int64)
        g_neg = torch.zeros(self._n_bins, dtype=torch.int64)
        for pos, neg in self._by_type.values():
            g_pos += pos
            g_neg += neg
        results: Dict[str, Any] = _pooled_metrics(g_pos, g_neg)
        results["n_pixels"] = int(g_pos.sum().item() + g_neg.sum().item())
        results["n_positive"] = int(g_pos.sum().item())
        results["clamped_fraction"] = (
            self._n_clamped / self._n_total if self._n_total else float("nan")
        )

        # Scenario-macro: average over scenarios that contain both classes.
        s_auroc, s_aupr, s_fpr95 = [], [], []
        for pos, neg in self._by_scenario.values():
            m = _pooled_metrics(pos, neg)
            if m["auroc"] == m["auroc"]:  # not NaN -> scenario has positives and negatives
                s_auroc.append(m["auroc"])
                s_aupr.append(m["aupr"])
                s_fpr95.append(m["fpr95"])
        results["scenario_macro"] = {
            "auroc": _mean(s_auroc), "aupr": _mean(s_aupr),
            "fpr95": _mean(s_fpr95), "n_scenarios": len(s_auroc),
        }

        by_type: Dict[str, Dict[str, float]] = {}
        for t in sorted(k for k in self._by_type if k is not None):
            pos, neg = self._by_type[t]
            m = _pooled_metrics(pos, neg)
            m["n_positive"] = int(pos.sum().item())
            by_type[t] = m
        if by_type:
            results["by_type"] = by_type
        return results

    def reset(self) -> None:
        for fut in self._futures:  # let outstanding workers finish before clearing
            fut.result()
        self._futures.clear()
        self._by_type.clear()
        self._by_scenario.clear()
        self._n_clamped = 0
        self._n_total = 0


class PixelEvaluator(_SpatialEvaluator):
    """Per-pixel anomaly metrics for one camera sensor, pooled over the split.

    Loads ground-truth masks from
    ``{scenario_id}/anomaly-{sensor}/{frame_id:06d}.png`` (any non-zero pixel is
    anomalous); a missing mask counts as all-negative.

    Parameters
    ----------
    sensor:
        Camera direction (``"front"``, ``"left"``, ``"right"``, ``"rear"``).
    num_workers:
        Worker threads that decode masks and bin scores off the main loop.
    max_inflight:
        Maximum frames queued at once; bounds retained host and GPU memory.
    n_bins:
        Bins for the global and per-type score histograms.
    scenario_bins:
        Bins for the per-scenario histograms; must divide ``n_bins``.
    score_transform:
        Strictly monotonic map applied to scores before binning; defaults to
        ``asinh``, which handles unbounded scores without tuning.
    score_range:
        ``(lo, hi)`` covered by the histogram after ``score_transform``.
    """

    def __init__(
        self,
        sensor: str,
        num_workers: int = _DEFAULT_WORKERS,
        max_inflight: int = _DEFAULT_MAX_INFLIGHT,
        *,
        n_bins: int = _DEFAULT_N_BINS,
        scenario_bins: int = _DEFAULT_SCENARIO_BINS,
        score_transform: Optional[ScoreTransform] = None,
        score_range: Optional[Tuple[float, float]] = None,
    ) -> None:
        super().__init__(
            num_workers=num_workers, max_inflight=max_inflight,
            n_bins=n_bins, scenario_bins=scenario_bins,
            score_transform=score_transform, score_range=score_range,
        )
        if sensor not in CAMERAS:
            raise ValueError(f"PixelEvaluator sensor must be one of {CAMERAS}, got {sensor!r}")
        self.sensor = sensor

    def update(
        self,
        pixel_scores: torch.Tensor,
        scenario_ids: Sequence[str],
        frame_ids: Sequence[int],
    ) -> None:
        """Accumulate one batch of per-pixel scores.

        Scores stay on their source device (the model's GPU); each frame's
        mask decode + GPU metric computation is handed to a worker thread.  Each
        frame's score slice is cloned so the full batch tensor can be freed
        immediately, bounding retained GPU memory to ``max_inflight`` frames.

        Parameters
        ----------
        pixel_scores:
            ``FloatTensor (B, H, W)`` of per-pixel anomaly scores.
        scenario_ids:
            length-B sequence of scenario path strings (see module docstring).
        frame_ids:
            length-B sequence of integer frame numbers.
        """
        ps = pixel_scores.detach().float()
        for i in range(ps.shape[0]):
            sid = scenario_ids[i]
            fid = int(frame_ids[i])
            self._submit(
                ps[i].reshape(-1).clone(),
                lambda sid=sid, fid=fid: self._load_mask(sid, fid).reshape(-1),
                sid,
            )

    def _load_mask(self, scenario_id: str, frame_id: int) -> torch.Tensor:
        path = Path(scenario_id) / f"anomaly-{self.sensor}" / f"{frame_id:06d}.png"
        if path.exists():
            arr = np.array(Image.open(path).convert("L"))
            return torch.from_numpy(arr > 0)
        return torch.zeros(_MASK_H, _MASK_W, dtype=torch.bool)


class PointEvaluator(_SpatialEvaluator):
    """Per-point anomaly metrics for LiDAR, pooled over the split.

    Loads ground-truth labels from
    ``{scenario_id}/anomaly-lidar/{frame_id:06d}.feather`` (``anomaly`` column)
    and handles variable-length point clouds.  Constructor parameters
    (``num_workers``, ``max_inflight``, ``n_bins``, ``scenario_bins``,
    ``score_transform``, ``score_range``) match :class:`PixelEvaluator`; this
    evaluator takes no ``sensor`` argument.
    """

    def update(
        self,
        point_scores: Sequence[torch.Tensor],
        scenario_ids: Sequence[str],
        frame_ids: Sequence[int],
    ) -> None:
        """Accumulate one batch of per-point scores.

        Parameters
        ----------
        point_scores:
            length-B sequence of ``FloatTensor (N_i,)`` per-point scores.
        scenario_ids:
            length-B sequence of scenario path strings (see module docstring).
        frame_ids:
            length-B sequence of integer frame numbers.
        """
        for i, scores in enumerate(point_scores):
            s = scores.detach().float().reshape(-1).clone()
            sid = scenario_ids[i]
            fid = int(frame_ids[i])
            n = s.shape[0]
            self._submit(
                s,
                lambda sid=sid, fid=fid, n=n: self._load_labels(sid, fid, n),
                sid,
            )

    def _load_labels(self, scenario_id: str, frame_id: int, n_points: int) -> torch.Tensor:
        path = Path(scenario_id) / "anomaly-lidar" / f"{frame_id:06d}.feather"
        if not path.exists():
            return torch.zeros(n_points, dtype=torch.bool)
        df = pd.read_feather(path)
        labels = torch.from_numpy(df["anomaly"].values.astype(bool))
        if labels.shape[0] != n_points:
            raise ValueError(
                f"point score count ({n_points}) does not match label count "
                f"({labels.shape[0]}) for {path}"
            )
        return labels


# ---------------------------------------------------------------------------
# Tiers 2 & 3: Sensor / Observation: per-frame AUROC
# ---------------------------------------------------------------------------


class _FrameLevelEvaluator:
    """Shared logic for per-frame AUROC with per-frame ground-truth labels.

    Labels are loaded once per scenario (a full per-scenario boolean array,
    indexed by ``frame_id``) and cached.
    """

    def __init__(self) -> None:
        self._scores: List[float] = []
        self._labels: List[bool] = []
        self._types: List[Optional[str]] = []
        self._scenario_ids: List[str] = []
        self._frame_ids: List[int] = []
        self._label_cache: Dict[str, np.ndarray] = {}

    # Subclasses provide the path to the per-scenario label feather.
    def _label_path(self, scenario_id: str) -> Path:
        raise NotImplementedError

    def update(
        self,
        scores: Any,
        scenario_ids: Sequence[str],
        frame_ids: Sequence[int],
    ) -> None:
        """Accumulate one batch of scalar per-frame scores.

        Parameters
        ----------
        scores:
            ``FloatTensor (B,)`` or length-B sequence of scalar scores.
        scenario_ids:
            length-B sequence of scenario path strings.
        frame_ids:
            length-B sequence of frame numbers.
        """
        score_list = _as_float_list(scores)
        for score, sid, fid in zip(score_list, scenario_ids, frame_ids):
            labels = self._labels_for_scenario(sid)
            self._scores.append(score)
            self._labels.append(bool(labels[int(fid)]))
            self._types.append(_parse_anomaly_type(sid))
            self._scenario_ids.append(sid)
            self._frame_ids.append(int(fid))

    def _labels_for_scenario(self, scenario_id: str) -> np.ndarray:
        if scenario_id not in self._label_cache:
            df = pd.read_feather(self._label_path(scenario_id))
            self._label_cache[scenario_id] = df["anomaly"].values.astype(bool)
        return self._label_cache[scenario_id]

    def max_per_scenario(self) -> Dict[str, float]:
        """Reduce accumulated frame scores to one max score per scenario."""
        out: Dict[str, float] = {}
        for score, sid in zip(self._scores, self._scenario_ids):
            if sid not in out or score > out[sid]:
                out[sid] = score
        return out

    def compute(self) -> Dict[str, Any]:
        results: Dict[str, Any] = {
            "auroc": _auroc(self._scores, self._labels),
            "n_frames": len(self._scores),
        }
        by_type = self._compute_by_type()
        if by_type:
            results["by_type"] = by_type
        return results

    def _compute_by_type(self) -> Dict[str, Dict[str, float]]:
        normal_s = [s for s, t in zip(self._scores, self._types) if t is None]
        normal_l = [l for l, t in zip(self._labels, self._types) if t is None]
        grouped_s: Dict[str, List[float]] = defaultdict(list)
        grouped_l: Dict[str, List[bool]] = defaultdict(list)
        for s, l, t in zip(self._scores, self._labels, self._types):
            if t is not None:
                grouped_s[t].append(s)
                grouped_l[t].append(l)
        out: Dict[str, Dict[str, float]] = {}
        for t in sorted(grouped_s):
            combined_s = grouped_s[t] + normal_s
            combined_l = grouped_l[t] + normal_l
            out[t] = {
                "auroc": _auroc(combined_s, combined_l),
                "n_frames": len(grouped_s[t]),
            }
        return out

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            "scenario_id": self._scenario_ids,
            "frame_id": self._frame_ids,
            "score": self._scores,
            "label": self._labels,
            "anomaly_type": self._types,
            "town": [_parse_town(s) for s in self._scenario_ids],
        })

    def reset(self) -> None:
        self._scores.clear()
        self._labels.clear()
        self._types.clear()
        self._scenario_ids.clear()
        self._frame_ids.clear()
        self._label_cache.clear()


class SensorEvaluator(_FrameLevelEvaluator):
    """Per-frame AUROC against sensor-specific labels.

    A frame is positive only if the anomaly is visible in this specific
    sensor.  Labels come from ``{scenario_id}/anomaly-{sensor}/sensor.feather``.

    Parameters
    ----------
    sensor:
        Any sensor name (``"front"``, ``"left"``, ``"right"``, ``"rear"``,
        ``"lidar"``).
    """

    def __init__(self, sensor: str) -> None:
        super().__init__()
        if sensor not in SENSORS:
            raise ValueError(f"SensorEvaluator sensor must be one of {SENSORS}, got {sensor!r}")
        self.sensor = sensor

    def _label_path(self, scenario_id: str) -> Path:
        return Path(scenario_id) / f"anomaly-{self.sensor}" / "sensor.feather"


class ObservationEvaluator(_FrameLevelEvaluator):
    """Per-frame AUROC against observation-level labels.

    The observation label marks whether the scene is anomalous at a given
    timestep, independent of any single sensor.  It can fire when an anomaly is
    visible to one or more sensors, when the sensors disagree, or for purely
    semantic anomalies that no sensor localises (e.g. anomalous weather).
    Labels come from ``{scenario_id}/anomaly-observation.feather``.  Accepts
    fused multi-sensor scores; a scenario with 30 frames contributes 30 scores.
    """

    def _label_path(self, scenario_id: str) -> Path:
        return Path(scenario_id) / "anomaly-observation.feather"


# ---------------------------------------------------------------------------
# Tier 4: Scenario: one score per scenario
# ---------------------------------------------------------------------------


class ScenarioEvaluator:
    """Scenario-level AUROC: one score per scenario.

    The label is inferred from the scenario path (``/test/anomaly/`` is
    positive, ``/test/normal/`` is negative).  Each :meth:`update` takes one
    scalar score for the whole scenario; how that score is produced is up to the
    caller and need not involve frame-level evaluation (e.g. a max over frame
    scores, or a model that scores whole scenarios directly).
    """

    def __init__(self) -> None:
        self._scores: List[float] = []
        self._labels: List[bool] = []
        self._types: List[Optional[str]] = []
        self._scenario_ids: List[str] = []

    def update(self, score: float, scenario_id: str) -> None:
        """Accumulate one scenario's score; label inferred from path."""
        self._scores.append(float(score))
        self._labels.append("/test/anomaly/" in scenario_id)
        self._types.append(_parse_anomaly_type(scenario_id))
        self._scenario_ids.append(scenario_id)

    def compute(self) -> Dict[str, Any]:
        results: Dict[str, Any] = {
            "auroc": _auroc(self._scores, self._labels),
            "n_scenarios": len(self._scores),
        }
        normal_s = [s for s, t in zip(self._scores, self._types) if t is None]
        normal_l = [l for l, t in zip(self._labels, self._types) if t is None]
        grouped_s: Dict[str, List[float]] = defaultdict(list)
        grouped_l: Dict[str, List[bool]] = defaultdict(list)
        for s, l, t in zip(self._scores, self._labels, self._types):
            if t is not None:
                grouped_s[t].append(s)
                grouped_l[t].append(l)
        by_type: Dict[str, Dict[str, float]] = {}
        for t in sorted(grouped_s):
            by_type[t] = {
                "auroc": _auroc(grouped_s[t] + normal_s, grouped_l[t] + normal_l),
                "n_scenarios": len(grouped_s[t]),
            }
        if by_type:
            results["by_type"] = by_type
        return results

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            "scenario_id": self._scenario_ids,
            "score": self._scores,
            "label": self._labels,
            "anomaly_type": self._types,
            "town": [_parse_town(s) for s in self._scenario_ids],
        })

    def reset(self) -> None:
        self._scores.clear()
        self._labels.clear()
        self._types.clear()
        self._scenario_ids.clear()
