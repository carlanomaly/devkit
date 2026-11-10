Vehicle and weather
===================

.. currentmodule:: carlanomaly.datasets

Tables recorded once per timestep, all in the ``base`` part. All datasets
take the :ref:`common arguments <dataset-arguments>`.

.. autoclass:: GNSSDataset(root, split="train", **kwargs)

.. autoclass:: IMUDataset(root, split="train", **kwargs)

.. autoclass:: ActionsDataset(root, split="train", **kwargs)

.. autoclass:: CollisionsDataset(root, split="train", **kwargs)

.. autoclass:: WeatherDataset(root, split="train", **kwargs)
