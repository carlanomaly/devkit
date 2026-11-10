Combined datasets
=================

.. currentmodule:: carlanomaly.datasets

These datasets return a dict per item that includes the identifiers the
evaluators need, ``scenario_id`` and ``timesteps``. All take the
:ref:`common arguments <dataset-arguments>`.

.. autoclass:: CameraDataset(root, split="train", direction="front", **kwargs)

.. autoclass:: LiDARDataset(root, split="train", **kwargs)

.. autoclass:: CarlAnomalyDataset(root, split="train", cameras=CAMERAS, **kwargs)

.. autofunction:: carlanomaly_collate_fn

.. autoclass:: WithIdentifiers(dataset)
