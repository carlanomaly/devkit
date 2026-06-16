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

.. code-block:: python

   from carlanomaly.datasets import RGBDataset

   # Point a loader at the directory you downloaded the dataset into.
   rgb = RGBDataset(root="/path/to/carlanomaly", split="test", direction="front")
   frame = rgb[0]  # FloatTensor (T, 3, H, W) in [0, 1]

Every dataset takes the dataset ``root`` and a ``split`` and serves all
scenarios found there.  Datasets built from the same ``root``/``split`` (and
``clip_len``/``stride``) align index-for-index: ``rgb[i]`` and ``lidar[i]``
refer to the same scenario and frame window.

Package overview
----------------

- **Atomic datasets** (:mod:`carlanomaly.datasets`): one per modality (RGB,
  segmentation, depth, point clouds, anomaly masks, IMU/GNSS/weather/...).  Each
  takes ``root`` + ``split`` and discovers scenarios internally.

- **Composite datasets**: :class:`~carlanomaly.datasets.CameraDataset`,
  :class:`~carlanomaly.datasets.LiDARDataset`, and
  :class:`~carlanomaly.datasets.CarlAnomalyDataset` (everything).

- **Evaluators** (:mod:`carlanomaly.evaluator`): streaming evaluation
  (update per batch, compute at the end) at the pixel, point, sensor,
  observation, and scenario tiers.

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/core
   api/datasets
   api/evaluator
   api/download
