"""Multi-level evaluators for the CarlAnomaly benchmark.

Evaluation happens at four independent levels — **sample**, **sensor**,
**timestep**, and **scenario** — each served by a pure metric calculator
that loads its own ground-truth labels from disk.  The caller only ever
provides anomaly *scores* plus identifiers (``scenario_id`` and
``timestep_id``); reducing sample scores to higher-level scores and fusing
multiple sensors is the caller's responsibility.

Identifiers
-----------
``scenario_id`` is the filesystem path to the scenario directory (the value
the datasets expose under the same key).  Evaluators use it both to locate a
scenario's label files and to parse its anomaly type and town from the path,
so it must follow the on-disk layout, e.g.
``.../test/anomaly/{town}/{anomaly_type}/{run}`` or
``.../test/normal/{town}/{run}``.  ``timestep_id`` is the integer timestep
number, used to index per-timestep labels and to build per-timestep label
paths (``{timestep_id:06d}.png`` / ``.feather``).

Levels
------
- :class:`PixelEvaluator` / :class:`PointEvaluator` (sample level): metrics
  (AUROC, AUPR, FPR@95TPR) pooled over every evaluated pixel/point in the
  split (anomalous timesteps, anomaly-free timesteps, and normal scenarios
  alike) using bounded-memory streaming histograms.
- :class:`SensorEvaluator` (sensor level): per-timestep AUROC/AUPR/FPR@95TPR
  against sensor-specific labels (``anomaly-{sensor}/sensor.feather``).
- :class:`TimestepEvaluator` (timestep level): per-timestep
  AUROC/AUPR/FPR@95TPR against timestep-level labels (stored on disk as
  ``anomaly-observation.feather``).
- :class:`ScenarioEvaluator` (scenario level): one score per scenario;
  AUROC/AUPR/FPR@95TPR, label inferred from the scenario path.
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
from pytorch_ood.metrics import functional as ood_metrics

from .index import CAMERAS, SENSORS

# Native camera resolution, used when a mask file is missing.
_MASK_H, _MASK_W = 1080, 1920

# Sample-level async defaults: worker threads load labels + compute metrics off
# the main loop; max_inflight bounds host RAM held by queued score tensors.
_DEFAULT_WORKERS = 32
_DEFAULT_MAX_INFLIGHT = 32


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


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


def _binary_metrics(scores: Sequence[float], labels: Sequence[bool]) -> Dict[str, float]:
    """AUROC/AUPR/FPR@95TPR over a flat list of scores/labels.

    Shared by the timestep-level (:class:`SensorEvaluator`/:class:`TimestepEvaluator`)
    and scenario-level (:class:`ScenarioEvaluator`) tiers, computed exactly with
    pytorch-ood on the raw per-sample scores, since these tiers' sample counts
    are small enough to not need bounded-memory binning.
    :func:`_pooled_metrics` uses the same definitions (trapezoidal AUPR with
    anomalies as the positive class, FPR at the highest threshold still
    reaching TPR >= 0.95). NaN for a degenerate (single-class) problem.

    Scores are used as given, in float64: torchmetrics' binary metrics instead
    apply a sigmoid to scores outside [0, 1], which in float32 collapses large
    (or tiny) scores into ties and changes the ranking.
    """
    if not scores or all(labels) or not any(labels):
        return {"auroc": float("nan"), "aupr": float("nan"), "fpr95": float("nan")}
    s = torch.as_tensor(scores, dtype=torch.float64)
    # pytorch-ood convention: label < 0 marks the positive (anomalous) class.
    l = torch.where(torch.as_tensor(labels, dtype=torch.bool), -1, 0)
    return {
        "auroc": ood_metrics.auroc(s, l).item(),
        "aupr": ood_metrics.aupr(s, l, positive="ood").item(),
        "fpr95": ood_metrics.fpr_at_tpr(s, l, tpr=0.95).item(),
    }


def _as_float_list(scores: Any) -> List[float]:
    """Coerce a batch of scalar scores (tensor/array/sequence) to a float list.

    Rejects non-scalar elements so that sample-level data (e.g. pixel maps)
    passed to a per-timestep evaluator fails loudly rather than being silently
    reduced.
    """
    if isinstance(scores, torch.Tensor):
        t = scores.detach().cpu().float()
        if t.ndim != 1:
            raise ValueError(
                f"expected a 1-D batch of scalar scores, got tensor with shape "
                f"{tuple(t.shape)}; reduce sample scores to per-timestep scalars first"
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
# Sample level: pixel/point metrics pooled over the whole split
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
    (single-class) histogram, mirroring :func:`_binary_metrics`.
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

    # AUPR by the trapezoidal rule, as pytorch-ood (and :func:`_binary_metrics`):
    # one PR point per occupied bin (each bin is one threshold), ordered by
    # decreasing recall, closed at (recall 0, precision 1).
    occupied = (h_pos + h_neg) > 0
    tp_o, fp_o = tp[occupied], fp[occupied]
    one = torch.ones(1, dtype=torch.double)
    precision = torch.cat([tp_o / (tp_o + fp_o), one])
    recall = torch.cat([tp_o / n_pos, zero])
    aupr = -torch.trapz(precision, recall).item()

    # FPR at the highest threshold still reaching TPR >= 0.95 (bin 0 always does).
    b_star = int((tpr >= 0.95).nonzero().max().item())
    fpr95 = (fp[b_star] / n_neg).item()

    return {"auroc": auroc, "aupr": aupr, "fpr95": fpr95}


class _SampleLevelEvaluator:
    """Pooled per-sample (pixel/point) metrics over the entire evaluated split.

    Unlike a per-timestep macro-average, every pixel/point (from anomalous
    timesteps, anomaly-free timesteps, and normal scenarios alike) enters a
    single pooled binary problem.  This removes the selection bias of
    conditioning the negative distribution on "the timestep contains an
    anomaly": a detector cannot score high merely by ranking a difficult class
    (e.g. all traffic lights) above unrelated background, because normal
    instances of that class from the rest of the split enter the negatives.

    Scores are accumulated into bounded-memory score histograms; label loading
    (PNG/feather decode) and binning run on a pool of worker threads so they
    overlap the GPU's next forward pass.  ``update()`` submits per-timestep
    tasks and ``compute()`` joins them, draining every outstanding future
    before aggregating.  Three views are derived from one set of histograms:

    - **global** pooled AUROC/AUPR/FPR95 (the primary metric);
    - **by_type** pooled metrics, grouped by anomaly type;
    - **scenario_macro**: per-scenario metrics averaged over scenarios that
      contain positives (an equal-per-scenario secondary diagnostic).

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
        """Queue one timestep's label-load + histogram binning on a worker thread.

        ``scores`` stays on its source device (the model's GPU); the worker
        decodes the label file on the CPU (parallel I/O, GIL released), bins both
        classes, and returns small ``n_bins``-length count tensors.  Blocks if
        ``max_inflight`` timesteps are already queued (backpressure), bounding
        both host work and retained GPU score tensors.  Every timestep
        contributes, including those with no positive labels (they are
        pure-negative pools).
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
        """Wait for every queued timestep, accumulating counts in submission order.

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
                "n_samples": 0, "n_positive": 0, "clamped_fraction": float("nan"),
            }

        g_pos = torch.zeros(self._n_bins, dtype=torch.int64)
        g_neg = torch.zeros(self._n_bins, dtype=torch.int64)
        for pos, neg in self._by_type.values():
            g_pos += pos
            g_neg += neg
        results: Dict[str, Any] = _pooled_metrics(g_pos, g_neg)
        results["n_samples"] = int(g_pos.sum().item() + g_neg.sum().item())
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


class PixelEvaluator(_SampleLevelEvaluator):
    """Sample-level metrics over per-pixel scores of one camera, pooled over the split.

    Loads ground-truth masks from
    ``{scenario_id}/anomaly-{sensor}/{timestep_id:06d}.png`` (any non-zero pixel
    is anomalous); a missing mask counts as all-negative.

    Parameters
    ----------
    sensor:
        Camera direction (``"front"``, ``"left"``, ``"right"``, ``"rear"``).
    num_workers:
        Worker threads that decode masks and bin scores off the main loop.
    max_inflight:
        Maximum timesteps queued at once; bounds retained host and GPU memory.
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
        timestep_ids: Sequence[int],
    ) -> None:
        """Accumulate one batch of per-pixel scores.

        Scores stay on their source device (the model's GPU); each timestep's
        mask decode + GPU metric computation is handed to a worker thread.  Each
        timestep's score slice is cloned so the full batch tensor can be freed
        immediately, bounding retained GPU memory to ``max_inflight`` timesteps.

        Parameters
        ----------
        pixel_scores:
            ``FloatTensor (B, H, W)`` of per-pixel anomaly scores.
        scenario_ids:
            length-B sequence of scenario path strings (see module docstring).
        timestep_ids:
            length-B sequence of integer timestep numbers.
        """
        ps = pixel_scores.detach().float()
        for i in range(ps.shape[0]):
            sid = scenario_ids[i]
            tid = int(timestep_ids[i])
            self._submit(
                ps[i].reshape(-1).clone(),
                lambda sid=sid, tid=tid: self._load_mask(sid, tid).reshape(-1),
                sid,
            )

    def _load_mask(self, scenario_id: str, timestep_id: int) -> torch.Tensor:
        path = Path(scenario_id) / f"anomaly-{self.sensor}" / f"{timestep_id:06d}.png"
        if path.exists():
            arr = np.array(Image.open(path).convert("L"))
            return torch.from_numpy(arr > 0)
        return torch.zeros(_MASK_H, _MASK_W, dtype=torch.bool)


class PointEvaluator(_SampleLevelEvaluator):
    """Sample-level metrics over per-point LiDAR scores, pooled over the split.

    Loads ground-truth labels from
    ``{scenario_id}/anomaly-lidar/{timestep_id:06d}.feather`` (``anomaly`` column)
    and handles variable-length point clouds.  Constructor parameters
    (``num_workers``, ``max_inflight``, ``n_bins``, ``scenario_bins``,
    ``score_transform``, ``score_range``) match :class:`PixelEvaluator`; this
    evaluator takes no ``sensor`` argument.
    """

    def update(
        self,
        point_scores: Sequence[torch.Tensor],
        scenario_ids: Sequence[str],
        timestep_ids: Sequence[int],
    ) -> None:
        """Accumulate one batch of per-point scores.

        Parameters
        ----------
        point_scores:
            length-B sequence of ``FloatTensor (N_i,)`` per-point scores.
        scenario_ids:
            length-B sequence of scenario path strings (see module docstring).
        timestep_ids:
            length-B sequence of integer timestep numbers.
        """
        for i, scores in enumerate(point_scores):
            s = scores.detach().float().reshape(-1).clone()
            sid = scenario_ids[i]
            tid = int(timestep_ids[i])
            n = s.shape[0]
            self._submit(
                s,
                lambda sid=sid, tid=tid, n=n: self._load_labels(sid, tid, n),
                sid,
            )

    def _load_labels(self, scenario_id: str, timestep_id: int, n_points: int) -> torch.Tensor:
        path = Path(scenario_id) / "anomaly-lidar" / f"{timestep_id:06d}.feather"
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
# Sensor & timestep levels: per-timestep AUROC
# ---------------------------------------------------------------------------


class _PerTimestepEvaluator:
    """Shared logic for per-timestep AUROC with per-timestep ground-truth labels.

    Labels are loaded once per scenario (a full per-scenario boolean array,
    indexed by ``timestep_id``) and cached.
    """

    def __init__(self) -> None:
        self._scores: List[float] = []
        self._labels: List[bool] = []
        self._types: List[Optional[str]] = []
        self._scenario_ids: List[str] = []
        self._timestep_ids: List[int] = []
        self._label_cache: Dict[str, np.ndarray] = {}

    # Subclasses provide the path to the per-scenario label feather.
    def _label_path(self, scenario_id: str) -> Path:
        raise NotImplementedError

    def update(
        self,
        scores: Any,
        scenario_ids: Sequence[str],
        timestep_ids: Sequence[int],
    ) -> None:
        """Accumulate one batch of scalar per-timestep scores.

        Parameters
        ----------
        scores:
            ``FloatTensor (B,)`` or length-B sequence of scalar scores.
        scenario_ids:
            length-B sequence of scenario path strings.
        timestep_ids:
            length-B sequence of timestep numbers.
        """
        score_list = _as_float_list(scores)
        for score, sid, tid in zip(score_list, scenario_ids, timestep_ids):
            labels = self._labels_for_scenario(sid)
            self._scores.append(score)
            self._labels.append(bool(labels[int(tid)]))
            self._types.append(_parse_anomaly_type(sid))
            self._scenario_ids.append(sid)
            self._timestep_ids.append(int(tid))

    def _labels_for_scenario(self, scenario_id: str) -> np.ndarray:
        if scenario_id not in self._label_cache:
            df = pd.read_feather(self._label_path(scenario_id))
            self._label_cache[scenario_id] = df["anomaly"].values.astype(bool)
        return self._label_cache[scenario_id]

    def max_per_scenario(self) -> Dict[str, float]:
        """Reduce accumulated timestep scores to one max score per scenario."""
        out: Dict[str, float] = {}
        for score, sid in zip(self._scores, self._scenario_ids):
            if sid not in out or score > out[sid]:
                out[sid] = score
        return out

    def compute(self) -> Dict[str, Any]:
        results: Dict[str, Any] = {
            **_binary_metrics(self._scores, self._labels),
            "n_timesteps": len(self._scores),
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
                **_binary_metrics(combined_s, combined_l),
                "n_timesteps": len(grouped_s[t]),
            }
        return out

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            "scenario_id": self._scenario_ids,
            "timestep_id": self._timestep_ids,
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
        self._timestep_ids.clear()
        self._label_cache.clear()


class SensorEvaluator(_PerTimestepEvaluator):
    """Sensor-level AUROC: one score per sensor reading, against sensor-specific labels.

    A sensor reading (this sensor at one timestep) is positive only if the
    anomaly is visible in this specific sensor.  Labels come from
    ``{scenario_id}/anomaly-{sensor}/sensor.feather``.

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


class TimestepEvaluator(_PerTimestepEvaluator):
    """Timestep-level AUROC against timestep-level labels.

    The timestep label marks whether the scene is anomalous at a given
    timestep — all synchronized sensor readings treated as one multimodal
    observation, independent of any single sensor.  It can fire when an
    anomaly is visible to one or more sensors, when the sensors disagree, or
    for purely semantic anomalies that no sensor localises (e.g. anomalous
    weather).  Labels are stored on disk as
    ``{scenario_id}/anomaly-observation.feather``.  Accepts fused multi-sensor
    scores; a scenario with 30 timesteps contributes 30 scores.
    """

    def _label_path(self, scenario_id: str) -> Path:
        return Path(scenario_id) / "anomaly-observation.feather"


# ---------------------------------------------------------------------------
# Scenario level: one score per scenario
# ---------------------------------------------------------------------------


class ScenarioEvaluator:
    """Scenario-level AUROC: one score per scenario.

    The label is inferred from the scenario path (``/test/anomaly/`` is
    positive, ``/test/normal/`` is negative).  Each :meth:`update` takes one
    scalar score for the whole scenario; how that score is produced is up to the
    caller and need not involve timestep-level evaluation (e.g. a max over
    timestep scores, or a model that scores whole scenarios directly).
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
            **_binary_metrics(self._scores, self._labels),
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
                **_binary_metrics(grouped_s[t] + normal_s, grouped_l[t] + normal_l),
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
