Datasets
========

.. currentmodule:: carlanomaly.datasets

All datasets take the :ref:`common arguments <dataset-arguments>`. Their
items have a leading time dimension ``T`` (see :doc:`/loading-data`), except
those of :class:`AnomalyScenarioDataset`.

Sensor data
-----------

See :doc:`sensor-data`.

.. autosummary::
   :nosignatures:

   RGBDataset
   SegmentationDataset
   DepthDataset
   PointCloudDataset

Anomaly labels
--------------

The evaluators load the labels themselves; these datasets are for looking at
them. See :doc:`anomaly-labels`.

.. autosummary::
   :nosignatures:

   AnomalySegmentationDataset
   AnomalyLiDARDataset
   AnomalySensorDataset
   AnomalyTimestepDataset
   AnomalyScenarioDataset

Vehicle and weather
-------------------

See :doc:`vehicle-data`.

.. autosummary::
   :nosignatures:

   GNSSDataset
   IMUDataset
   ActionsDataset
   CollisionsDataset
   WeatherDataset

Combined datasets
-----------------

See :doc:`combined`.

.. autosummary::
   :nosignatures:

   CameraDataset
   LiDARDataset
   CarlAnomalyDataset
   WithIdentifiers
   carlanomaly_collate_fn

.. toctree::
   :hidden:

   sensor-data
   anomaly-labels
   vehicle-data
   combined
