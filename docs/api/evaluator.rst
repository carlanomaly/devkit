Evaluators
==========

Evaluation is split into four independent tiers.  Each evaluator is a
streaming accumulator: call :meth:`update` once per batch, then
:meth:`compute` at the end for the aggregated metrics.

Every evaluator identifies data by ``scenario_id`` (the filesystem path to the
scenario directory) and ``frame_id`` (an integer frame number).  The path must
follow the on-disk layout, e.g. ``.../test/anomaly/{town}/{anomaly_type}/...``
or ``.../test/normal/{town}/...``: evaluators read it both to locate label
files and to parse the anomaly type and town for per-type breakdowns.

.. currentmodule:: carlanomaly

Spatial: pixel and point
------------------------

The spatial evaluators pool every evaluated pixel/point in the split
(anomalous frames, anomaly-free frames, and normal scenarios) into a single
binary problem, instead of averaging per-frame scores over anomaly-containing
frames.  This removes the selection bias that lets a detector score a whole
class (e.g. traffic lights) high without separating anomalous instances from
normal ones.  Metrics use bounded-memory streaming histograms; ``compute()``
returns global pooled AUROC/AUPR/FPR95 plus a per-anomaly-type breakdown and a
per-scenario macro-average diagnostic.

.. autoclass:: PixelEvaluator
   :members: update, compute, reset

.. autoclass:: PointEvaluator
   :members: update, compute, reset

Frame-level
-----------

.. autoclass:: SensorEvaluator
   :members: update, compute, reset, to_dataframe, max_per_scenario

.. autoclass:: ObservationEvaluator
   :members: update, compute, reset, to_dataframe, max_per_scenario

Scenario-level
--------------

.. autoclass:: ScenarioEvaluator
   :members: update, compute, reset, to_dataframe
