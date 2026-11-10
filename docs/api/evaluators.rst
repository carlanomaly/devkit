Evaluators
==========

.. currentmodule:: carlanomaly

One evaluator per evaluation level. All work the same way: ``update`` once
per batch, ``compute`` at the end. :doc:`/evaluation` shows them in use,
:doc:`/metrics` explains the results.

``scenario_ids`` are the ``scenario_id`` values the datasets return: paths
into the dataset directory, from which the evaluators find the labels.

Sample level
------------

.. autoclass:: PixelEvaluator
   :members: update, compute, reset
   :inherited-members:

.. autoclass:: PointEvaluator
   :members: update, compute, reset
   :inherited-members:

Sensor level
------------

.. autoclass:: SensorEvaluator

   .. automethod:: update
   .. automethod:: compute
   .. automethod:: max_per_scenario
   .. automethod:: time_to_detection
   .. automethod:: to_dataframe
   .. automethod:: reset

Timestep level
--------------

.. autoclass:: TimestepEvaluator

   .. automethod:: update
   .. automethod:: compute
   .. automethod:: max_per_scenario
   .. automethod:: time_to_detection
   .. automethod:: to_dataframe
   .. automethod:: reset

Scenario level
--------------

.. autoclass:: ScenarioEvaluator
   :members: update, compute, to_dataframe, reset
