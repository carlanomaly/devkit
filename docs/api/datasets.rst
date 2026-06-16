Datasets
========

Every dataset takes the dataset ``root`` and a ``split`` and discovers its
scenarios internally.  Datasets built from the same ``root``/``split`` (and
``clip_len``/``stride``) align index-for-index: ``dataset_a[i]`` and
``dataset_b[i]`` refer to the same scenario and frame window.  Every dataset
returns a time dimension ``T`` even when ``clip_len=1``.

.. currentmodule:: carlanomaly.datasets

Atomic: image
--------------

.. autoclass:: RGBDataset
   :members:
   :special-members: __len__, __getitem__

.. autoclass:: SegmentationDataset
   :members:
   :special-members: __len__, __getitem__

.. autoclass:: DepthDataset
   :members:
   :special-members: __len__, __getitem__

.. autoclass:: AnomalySegmentationDataset
   :members:
   :special-members: __len__, __getitem__

Atomic: LiDAR
--------------

.. autoclass:: PointCloudDataset
   :members:
   :special-members: __len__, __getitem__

.. autoclass:: AnomalyLiDARDataset
   :members:
   :special-members: __len__, __getitem__

.. autoclass:: AnomalyObservationDataset
   :members:
   :special-members: __len__, __getitem__

Atomic: tabular
----------------

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
