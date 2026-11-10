Sensor data
===========

.. currentmodule:: carlanomaly.datasets

Camera datasets take ``direction``: ``"front"`` (default), ``"left"``,
``"right"`` or ``"rear"``. Images are 1080 × 1920 pixels. All datasets also
take the :ref:`common arguments <dataset-arguments>`.

.. autoclass:: RGBDataset(root, split="train", direction="front", **kwargs)

.. autoclass:: SegmentationDataset(root, split="train", direction="front", **kwargs)

.. autoclass:: DepthDataset(root, split="train", direction="front", **kwargs)

.. autoclass:: PointCloudDataset(root, split="train", **kwargs)
