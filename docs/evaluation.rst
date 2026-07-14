Evaluating your model
=====================

CarlAnomaly evaluates anomaly detectors at four levels, from fine-grained
localisation to whole-scenario detection:

.. list-table::
   :header-rows: 1
   :widths: 14 30 34 22

   * - Level
     - One score per …
     - Question answered
     - Evaluator
   * - Sample
     - pixel / LiDAR point
     - *Where* in the sensor data is the anomaly?
     - :class:`~carlanomaly.PixelEvaluator`, :class:`~carlanomaly.PointEvaluator`
   * - Sensor
     - sensor reading (one sensor, one timestep)
     - Does this sensor see an anomaly?
     - :class:`~carlanomaly.SensorEvaluator`
   * - Timestep
     - timestep (all sensors jointly)
     - Is the scene anomalous right now?
     - :class:`~carlanomaly.TimestepEvaluator`
   * - Scenario
     - scenario
     - Did anything anomalous happen in this recording?
     - :class:`~carlanomaly.ScenarioEvaluator`

Your model only produces anomaly **scores** (higher = more anomalous); every
evaluator loads its own ground-truth labels from disk.  All evaluators are
streaming accumulators, so the test split never has to fit in memory::

   evaluator = ...
   for batch in loader:
       scores = model(batch)
       evaluator.update(scores, scenario_ids, timestep_ids)
   results = evaluator.compute()

Clips, timesteps, and identifiers
---------------------------------

Two conventions connect the datasets to the evaluators:

**Evaluate one timestep at a time.**  Datasets serve *clips* of ``clip_len``
consecutive timesteps and always return a leading time dimension ``T`` —
that is what temporal models train on.  Evaluation, however, is
per-timestep.  Use the defaults ``clip_len=1, stride=1`` for the test split,
so that every timestep appears exactly once as its own item; ``T`` is then a
singleton dimension, which is why the examples below index ``[:, 0]`` — it
selects the clip's single timestep, it does not skip anything.  A temporal
model that consumes longer clips should assign its score to the timestep it
describes (e.g. a future-frame-prediction model scoring the last timestep of
each clip would pass ``batch["timesteps"][:, -1]``).

**Every score carries identifiers.**  Evaluators identify data by
``scenario_id`` (the scenario directory's path, as a string) and
``timestep_id`` (integer).  :class:`~carlanomaly.datasets.CameraDataset`,
:class:`~carlanomaly.datasets.LiDARDataset`, and
:class:`~carlanomaly.datasets.CarlAnomalyDataset` include them in every item
under the ``"scenario_id"`` and ``"timesteps"`` keys.  For the single-modality
datasets (which return bare tensors), wrap them in
:class:`~carlanomaly.datasets.WithIdentifiers`::

   from torch.utils.data import DataLoader
   from carlanomaly.datasets import RGBDataset, WithIdentifiers

   rgb = WithIdentifiers(RGBDataset(root="/data/carlanomaly", split="test"))
   rgb[0]   # {"data": FloatTensor (T, 3, H, W), "scenario_id": str,
            #  "timesteps": LongTensor (T,)}

   loader = DataLoader(rgb, batch_size=8)  # batches: data (B, T, 3, H, W),
                                           # scenario_id list[str], timesteps (B, T)

Sample level
------------

Provide per-pixel score maps (or per-point score vectors) for each timestep.
Pixels and points from the *entire* split — anomalous timesteps, anomaly-free
timesteps, and normal scenarios alike — are pooled into one binary problem, so
false positives on normal data count against you::

   import torch
   from carlanomaly import PixelEvaluator

   pixel_eval = PixelEvaluator(sensor="front")

   with torch.no_grad():
       for batch in loader:
           images = batch["data"][:, 0].to(device)   # (B, 3, H, W)
           scores = model(images)                    # (B, H, W) pixel scores
           pixel_eval.update(scores, batch["scenario_id"],
                             batch["timesteps"][:, 0])

   results = pixel_eval.compute()
   # {'auroc': ..., 'aupr': ..., 'fpr95': ..., 'n_samples': ...,
   #  'n_positive': ..., 'clamped_fraction': ...,
   #  'scenario_macro': {...}, 'by_type': {...}}

:class:`~carlanomaly.PointEvaluator` works the same way for LiDAR, taking a
list of variable-length per-point score tensors per batch.

Scores are accumulated in bounded-memory histograms.  AUROC/AUPR/FPR95 are
rank-based, so the default ``asinh`` binning handles unbounded scores
(MaxLogit, MSE, …) without tuning; ``clamped_fraction`` reports how many
scores fell outside the histogram range (it should be ~0).  Pass
``score_transform``/``score_range`` if your scores need a custom binning.

Sensor and timestep level
-------------------------

Reduce sample scores to one scalar per timestep — the benchmark baselines use
a simple ``max`` — or let your model produce timestep scores directly.
:class:`~carlanomaly.SensorEvaluator` scores a single sensor against
sensor-specific visibility labels; :class:`~carlanomaly.TimestepEvaluator`
scores the whole (possibly fused, multimodal) scene::

   from carlanomaly import SensorEvaluator, TimestepEvaluator

   sensor_eval = SensorEvaluator(sensor="front")
   timestep_eval = TimestepEvaluator()

   for batch in loader:
       images = batch["data"][:, 0].to(device)            # (B, 3, H, W)
       pixel_scores = model(images)                       # (B, H, W)
       timestep_scores = pixel_scores.amax(dim=(1, 2))    # (B,) max-reduce

       sensor_eval.update(timestep_scores, batch["scenario_id"],
                          batch["timesteps"][:, 0])
       timestep_eval.update(timestep_scores, batch["scenario_id"],
                            batch["timesteps"][:, 0])

   print(sensor_eval.compute())    # {'auroc': ..., 'n_timesteps': ..., 'by_type': {...}}
   print(timestep_eval.compute())

Both evaluators offer :meth:`to_dataframe` to export the raw
``(scenario_id, timestep_id, score, label, anomaly_type, town)`` table for
your own analysis or plots.

Scenario level
--------------

One scalar per scenario; the label is inferred from the scenario path.
:meth:`max_per_scenario` reduces already-accumulated timestep scores for
you::

   from carlanomaly import ScenarioEvaluator

   scenario_eval = ScenarioEvaluator()
   for scenario_id, score in timestep_eval.max_per_scenario().items():
       scenario_eval.update(score, scenario_id)

   print(scenario_eval.compute())  # {'auroc': ..., 'n_scenarios': ..., 'by_type': {...}}

Reading the results
-------------------

- ``auroc`` / ``aupr`` / ``fpr95`` — metrics pooled over everything the
  evaluator has seen.  Evaluate the **full test split** (normal *and* anomaly
  scenarios); otherwise the negatives are unrepresentative and the numbers are
  not comparable to the benchmark.
- ``by_type`` — the same metric per anomaly type, each computed against all
  normal data (sub-scenario levels report ``n_timesteps``; the scenario level
  reports ``n_scenarios``).
- ``scenario_macro`` (sample level only) — per-scenario metrics averaged over
  scenarios that contain positives; a robustness diagnostic that weights every
  anomalous scenario equally instead of by its number of pixels.

The ground-truth labels behind each level are also available as datasets for
inspection and custom analyses:
:class:`~carlanomaly.datasets.AnomalySegmentationDataset` and
:class:`~carlanomaly.datasets.AnomalyLiDARDataset` (sample),
:class:`~carlanomaly.datasets.AnomalySensorDataset` (sensor),
:class:`~carlanomaly.datasets.AnomalyTimestepDataset` (timestep), and
:class:`~carlanomaly.datasets.AnomalyScenarioDataset` (scenario).
