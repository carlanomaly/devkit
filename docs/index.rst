CarlAnomaly DevKit
==================

Data-loading and evaluation framework for the
`CarlAnomaly <https://carlanomaly.github.io>`_ benchmark: multimodal anomaly
detection for autonomous driving built on CARLA simulator data.

.. code-block:: bash

   pip install git+https://github.com/carlanomaly/devkit

Requires Python ≥ 3.10 and PyTorch.

Quick start
-----------

Every dataset takes the dataset ``root`` and a ``split``, downloads the
archive parts it needs on first use, and serves clips as tensors:

.. code-block:: python

   from carlanomaly.datasets import RGBDataset

   rgb = RGBDataset(root="/data/carlanomaly", split="test", direction="front")
   clip = rgb[0]  # FloatTensor (T, 3, H, W) in [0, 1]

Datasets built from the same ``root``/``split`` (and ``clip_len``/``stride``)
align index-for-index: ``rgb[i]`` and ``lidar[i]`` refer to the same scenario
and timestep window.

To score a model against the benchmark, feed its anomaly scores to a
streaming evaluator — ground-truth labels are loaded automatically.  The
:class:`~carlanomaly.datasets.WithIdentifiers` wrapper attaches the scenario
and timestep identifiers the evaluators expect:

.. code-block:: python

   from torch.utils.data import DataLoader
   from carlanomaly import PixelEvaluator
   from carlanomaly.datasets import RGBDataset, WithIdentifiers

   rgb = WithIdentifiers(RGBDataset(root="/data/carlanomaly", split="test"))
   loader = DataLoader(rgb, batch_size=8)

   evaluator = PixelEvaluator(sensor="front")
   for batch in loader:
       scores = model(batch["data"][:, 0])  # (B, H, W) per-pixel anomaly scores
       evaluator.update(scores, batch["scenario_id"], batch["timesteps"][:, 0])

   print(evaluator.compute())  # pooled AUROC / AUPR / FPR95 + breakdowns

Evaluation is per-timestep: with the default ``clip_len=1`` each item is a
single timestep, and ``[:, 0]`` selects it from the clip dimension ``T`` that
every dataset returns.  See :doc:`evaluation` for the full protocol and a
runnable example for every level.

Evaluation levels
-----------------

Models are evaluated at four levels, matching the paper:

- **Sample** — per-pixel / per-point anomaly localisation
  (:class:`~carlanomaly.PixelEvaluator`, :class:`~carlanomaly.PointEvaluator`).
- **Sensor** — one score per sensor reading
  (:class:`~carlanomaly.SensorEvaluator`).
- **Timestep** — one score per timestep, all sensors treated as one multimodal
  observation (:class:`~carlanomaly.TimestepEvaluator`).
- **Scenario** — one score per driving scenario
  (:class:`~carlanomaly.ScenarioEvaluator`).

Package overview
----------------

- **Datasets** (:mod:`carlanomaly.datasets`) — one class per modality (RGB,
  segmentation, depth, point clouds, anomaly labels, IMU/GNSS/weather/…), plus
  composite datasets (:class:`~carlanomaly.datasets.CameraDataset`,
  :class:`~carlanomaly.datasets.LiDARDataset`, and
  :class:`~carlanomaly.datasets.CarlAnomalyDataset` for everything).
- **Evaluators** (:mod:`carlanomaly.evaluator`) — streaming metric
  accumulators for the four levels above; ground-truth labels are loaded
  automatically.
- **Download** (:mod:`carlanomaly.download`) — the dataset ships as modular
  archive parts; datasets fetch what they need with ``download=True``, or use
  the ``carlanomaly-download`` CLI.

.. toctree::
   :maxdepth: 2
   :caption: Guide

   evaluation

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/datasets
   api/evaluator
   api/download
   api/advanced
