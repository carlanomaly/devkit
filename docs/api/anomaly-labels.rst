Anomaly labels
==============

.. currentmodule:: carlanomaly.datasets

The ground truth of each evaluation level. The evaluators load it
themselves; use these datasets to look at it or for your own analysis. In the
train split, which has no anomalies, every label is ``False``. All datasets
take the :ref:`common arguments <dataset-arguments>`.

Sample level
------------

.. autoclass:: AnomalySegmentationDataset(root, split="train", direction="front", **kwargs)

.. autoclass:: AnomalyLiDARDataset(root, split="train", **kwargs)

Sensor level
------------

.. autoclass:: AnomalySensorDataset(root, split="train", *, sensor="front", **kwargs)

Timestep level
--------------

.. autoclass:: AnomalyTimestepDataset(root, split="train", **kwargs)

Scenario level
--------------

.. autoclass:: AnomalyScenarioDataset(root, split="train", **kwargs)
