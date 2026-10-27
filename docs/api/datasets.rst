Datasets
========

Every dataset takes the dataset ``root`` and a ``split`` and discovers its
scenarios internally.  Datasets built from the same ``root``/``split`` (and
``clip_len``/``stride``) align index-for-index: ``dataset_a[i]`` and
``dataset_b[i]`` refer to the same scenario and timestep window.  Every
dataset returns a time dimension ``T`` even when ``clip_len=1``.

.. currentmodule:: carlanomaly.datasets

Camera
------

.. autoclass:: RGBDataset
   :members:
   :special-members: __len__, __getitem__

.. autoclass:: SegmentationDataset
   :members:
   :special-members: __len__, __getitem__

.. autoclass:: DepthDataset
   :members:
   :special-members: __len__, __getitem__

LiDAR
-----

.. autoclass:: PointCloudDataset
   :members:
   :special-members: __len__, __getitem__

Anomaly labels
--------------

Ground-truth labels for every evaluation level: per-pixel masks and per-point
labels (sample level), per-sensor-reading labels (sensor level), per-timestep
labels (timestep level), and per-scenario labels (scenario level).  The
evaluators load these labels themselves; these datasets exist for inspection,
visualisation, and custom evaluation.

.. autoclass:: AnomalySegmentationDataset
   :members:
   :special-members: __len__, __getitem__

.. autoclass:: AnomalyLiDARDataset
   :members:
   :special-members: __len__, __getitem__

.. autoclass:: AnomalySensorDataset
   :members:
   :special-members: __len__, __getitem__

.. autoclass:: AnomalyTimestepDataset
   :members:
   :special-members: __len__, __getitem__

.. autoclass:: AnomalyScenarioDataset
   :members:
   :special-members: __len__, __getitem__

Tabular
-------

.. autoclass:: WeatherDataset
   :members:
   :special-members: __len__, __getitem__

.. autoclass:: GNSSDataset
   :members:
   :special-members: __len__, __getitem__

.. autoclass:: IMUDataset
   :members:
   :special-members: __len__, __getitem__

.. autoclass:: ActionsDataset
   :members:
   :special-members: __len__, __getitem__

.. autoclass:: CollisionsDataset
   :members:
   :special-members: __len__, __getitem__

Composite
---------

Composite datasets return dicts that already include the evaluator
identifiers ``scenario_id`` and ``timesteps``.

.. autoclass:: CameraDataset
   :members:
   :special-members: __len__, __getitem__

.. autoclass:: LiDARDataset
   :members:
   :special-members: __len__, __getitem__

.. autoclass:: CarlAnomalyDataset
   :members:
   :special-members: __len__, __getitem__

.. autofunction:: carlanomaly_collate_fn

Utilities
---------

.. autoclass:: WithIdentifiers
   :members:
   :special-members: __len__, __getitem__
