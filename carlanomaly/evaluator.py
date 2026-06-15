"""Multi-level evaluators for the CarlAnomaly benchmark.

Evaluation is split into four independent tiers, each a pure metric
calculator that loads its own ground-truth labels from disk.  The caller
only ever provides anomaly *scores* plus identifiers (``scenario_id`` and
``frame_id``); all reduction (pixel/point -> frame) and multi-sensor fusion
is the caller's responsibility.

Tiers
-----
- :class:`PixelEvaluator` / :class:`PointEvaluator` — spatial metrics within
  a frame (AUROC, AUPR, FPR@95TPR), computed per frame then averaged.
- :class:`SensorEvaluator` — frame-level AUROC against sensor-specific labels
  (``anomaly-{sensor}/sensor.feather``).
- :class:`ObservationEvaluator` — frame-level AUROC against scenario-level
  labels (``anomaly-observation.feather``).
- :class:`ScenarioEvaluator` — one score per scenario; label inferred from the
  scenario path.

Anomaly type is parsed from the ``scenario_id`` path so per-type breakdowns
require no extra argument.
"""

from __future__ import annotations

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
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision

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


def _spatial_metrics(scores: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    """AUROC, AUPR, FPR@95TPR for a single frame's flattened elements.

    Runs on whatever device ``scores`` lives on (kept on-GPU for the large
    per-frame sorts) — ``labels`` is moved to match.
    """
    device = scores.device
    labels = labels.to(device)
    labels_long = labels.long()
    auroc = BinaryAUROC().to(device)
    auroc.update(scores, labels_long)
    aupr = BinaryAveragePrecision().to(device)
    aupr.update(scores, labels_long)
    return {
        "auroc": auroc.compute().item(),
        "aupr": aupr.compute().item(),
        "fpr95": _fpr_at_tpr(scores, labels),
    }


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
                f"{tuple(t.shape)} — reduce spatial scores to per-frame scalars first"
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
# Tier 1: Pixel / Point — spatial metrics within a frame
# ---------------------------------------------------------------------------


_SpatialResult = Optional[Tuple[float, float, float, Optional[str]]]


class _SpatialEvaluator:
    """Shared logic for per-frame spatial metrics, averaged across frames.

    Label loading (PNG/feather decode) and metric computation run on a pool of
    worker threads so they overlap the GPU's next forward pass instead of
    blocking it.  ``update()`` only does a one-shot GPU->CPU copy of the batch's
    scores and submits per-frame tasks; ``compute()`` is the join point — it
    drains every outstanding future before aggregating.  No caller-side
    lifecycle call is required: once drained the worker threads sit idle and are
    reaped automatically at interpreter exit.

    Only frames with at least one positive label contribute.
    """

    def __init__(
        self,
        num_workers: int = _DEFAULT_WORKERS,
        max_inflight: int = _DEFAULT_MAX_INFLIGHT,
    ) -> None:
        self._aurocs: List[float] = []
        self._auprs: List[float] = []
        self._fpr95s: List[float] = []
        self._types: List[Optional[str]] = []
        self._executor = ThreadPoolExecutor(max_workers=num_workers)
        self._futures: List[Future] = []
        self._sem = threading.Semaphore(max_inflight)
        # Dedicated stream for worker metric ops so they don't share the
        # default stream with the model forward — keeps main-thread syncs from
        # waiting behind queued metric work.  Created lazily on first CUDA frame.
        self._metric_stream: Optional[torch.cuda.Stream] = None

    def _submit(
        self,
        scores: torch.Tensor,
        load_labels: Callable[[], torch.Tensor],
        scenario_id: str,
    ) -> None:
        """Queue one frame's label-load + metric computation on a worker thread.

        ``scores`` stays on its source device (the model's GPU); the worker
        decodes the label file on the CPU (parallel I/O, GIL released) and runs
        the metric on that device — ``_spatial_metrics`` uploads the labels to
        match.  Blocks if ``max_inflight`` frames are already queued
        (backpressure), bounding both host work and retained GPU score tensors.

        For CUDA scores the metric runs on a dedicated stream.  ``scores`` was
        produced on the caller's (default) stream, so we record an event here and
        have the metric stream wait on it — this is the cross-stream handoff that
        keeps the read correctly ordered after the producing clone.  The worker's
        ``_spatial_metrics`` ends in ``.item()`` syncs, so all metric ops finish
        before the closure drops its ``scores`` ref — no ``record_stream`` needed.
        """
        self._sem.acquire()
        atype = _parse_anomaly_type(scenario_id)

        event: Optional[torch.cuda.Event] = None
        if scores.is_cuda:
            if self._metric_stream is None:
                self._metric_stream = torch.cuda.Stream(device=scores.device)
            event = torch.cuda.Event()
            event.record()  # captures the producing op on the caller's stream

        def task() -> _SpatialResult:
            try:
                labels = load_labels()
                if not labels.any():
                    return None
                if event is not None:
                    with torch.cuda.stream(self._metric_stream):
                        self._metric_stream.wait_event(event)
                        m = _spatial_metrics(scores, labels)
                else:
                    m = _spatial_metrics(scores, labels)
                return m["auroc"], m["aupr"], m["fpr95"], atype
            finally:
                self._sem.release()

        self._futures.append(self._executor.submit(task))

    def _drain(self) -> None:
        """Wait for every queued frame, collecting results in submission order.

        ``Future.result()`` re-raises any exception a worker hit (e.g. a corrupt
        mask), so failures surface here rather than being silently dropped.
        """
        for fut in self._futures:
            r = fut.result()
            if r is None:
                continue
            auroc, aupr, fpr95, atype = r
            self._aurocs.append(auroc)
            self._auprs.append(aupr)
            self._fpr95s.append(fpr95)
            self._types.append(atype)
        self._futures.clear()

    def compute(self) -> Dict[str, Any]:
        self._drain()
        results: Dict[str, Any] = {
            "auroc": _mean(self._aurocs),
            "aupr": _mean(self._auprs),
            "fpr95": _mean(self._fpr95s),
            "n_frames": len(self._aurocs),
        }
        by_type: Dict[str, Dict[str, float]] = {}
        grouped: Dict[str, List[int]] = defaultdict(list)
        for i, t in enumerate(self._types):
            if t is not None:
                grouped[t].append(i)
        for t in sorted(grouped):
            idxs = grouped[t]
            by_type[t] = {
                "auroc": _mean([self._aurocs[i] for i in idxs]),
                "aupr": _mean([self._auprs[i] for i in idxs]),
                "fpr95": _mean([self._fpr95s[i] for i in idxs]),
                "n_frames": len(idxs),
            }
        if by_type:
            results["by_type"] = by_type
        return results

    def reset(self) -> None:
        for fut in self._futures:  # let outstanding workers finish before clearing
            fut.result()
        self._futures.clear()
        self._aurocs.clear()
        self._auprs.clear()
        self._fpr95s.clear()
        self._types.clear()


class PixelEvaluator(_SpatialEvaluator):
    """Per-pixel anomaly metrics for one camera sensor.

    Loads ground-truth masks from ``{scenario_id}/anomaly-{sensor}/{frame:06d}.png``.

    Parameters
    ----------
    sensor:
        Camera direction (``"front"``, ``"left"``, ``"right"``, ``"rear"``).
    """

    def __init__(
        self,
        sensor: str,
        num_workers: int = _DEFAULT_WORKERS,
        max_inflight: int = _DEFAULT_MAX_INFLIGHT,
    ) -> None:
        super().__init__(num_workers=num_workers, max_inflight=max_inflight)
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
            ``FloatTensor (B, H, W)`` — per-pixel anomaly scores.
        scenario_ids:
            length-B sequence of scenario path strings.
        frame_ids:
            length-B sequence of frame numbers.
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
    """Per-point anomaly metrics for LiDAR.

    Loads ground-truth labels from
    ``{scenario_id}/anomaly-lidar/{frame:06d}.feather`` (``anomaly`` column).
    Handles variable-length point clouds.
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
            length-B sequence of ``FloatTensor (N_i,)`` — per-point scores.
        scenario_ids:
            length-B sequence of scenario path strings.
        frame_ids:
            length-B sequence of frame numbers.
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
# Tiers 2 & 3: Sensor / Observation — per-frame AUROC
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

    A frame is positive if any anomaly is present in the scenario at that
    timestep, regardless of which sensor sees it.  Labels come from
    ``{scenario_id}/anomaly-observation.feather``.  Accepts fused
    multi-sensor scores.  A scenario with 30 frames contributes 30 scores.
    """

    def _label_path(self, scenario_id: str) -> Path:
        return Path(scenario_id) / "anomaly-observation.feather"


# ---------------------------------------------------------------------------
# Tier 4: Scenario — one score per scenario
# ---------------------------------------------------------------------------


class ScenarioEvaluator:
    """Scenario-level AUROC — one score per scenario.

    The label is inferred from the scenario path: ``/test/anomaly/`` is
    positive, ``/test/normal/`` is negative.  The caller decides how to reduce
    frame scores to a single scenario score (e.g. max).
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
