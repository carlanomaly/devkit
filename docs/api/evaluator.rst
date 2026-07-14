Evaluators
==========

Evaluation happens at four levels — **sample**, **sensor**, **timestep**, and
**scenario** — matching the benchmark protocol.  Each evaluator is a streaming
accumulator: call :meth:`update` once per batch, then :meth:`compute` at the
end for the aggregated metrics.  See :doc:`/evaluation` for a guided tour with
code examples.

Every evaluator identifies data by ``scenario_id`` (the filesystem path to the
scenario directory) and — below the scenario level — ``timestep_id`` (an
integer timestep number).  The path must follow the on-disk layout, e.g.
``.../test/anomaly/{town}/{anomaly_type}/...`` or
``.../test/normal/{town}/...``: evaluators read it both to locate label files
and to parse the anomaly type and town for per-type breakdowns.

.. currentmodule:: carlanomaly

Sample level
------------

Per-pixel (camera) and per-point (LiDAR) scores.  Every evaluated sample in
the split — from anomalous timesteps, anomaly-free timesteps, and normal
scenarios alike — is pooled into a single binary problem, instead of averaging
per-timestep scores over anomaly-containing timesteps.  This removes the
selection bias that lets a detector score a whole class (e.g. traffic lights)
high without separating anomalous instances from normal ones.  Metrics use
bounded-memory streaming histograms; :meth:`compute` returns global pooled
AUROC/AUPR/FPR95 plus a per-anomaly-type breakdown and a per-scenario
macro-average diagnostic.

.. autoclass:: PixelEvaluator
   :members: update, compute, reset
   :inherited-members:

.. autoclass:: PointEvaluator
   :members: update, compute, reset
   :inherited-members:

Sensor level
------------

One score per sensor reading (one sensor at one timestep), evaluated against
labels that mark whether the anomaly is visible in that specific sensor.

.. autoclass:: SensorEvaluator
   :members: update, compute, reset, to_dataframe, max_per_scenario
   :inherited-members:

Timestep level
--------------

One score per timestep, treating all synchronized sensor readings as one
multimodal observation.  This is where fused multi-sensor scores are
evaluated.

.. autoclass:: TimestepEvaluator
   :members: update, compute, reset, to_dataframe, max_per_scenario
   :inherited-members:

Scenario level
--------------

One score per scenario; the label is inferred from the scenario path.

.. autoclass:: ScenarioEvaluator
   :members: update, compute, reset, to_dataframe
