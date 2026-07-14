Advanced
========

.. currentmodule:: carlanomaly

You normally do not need anything on this page: datasets and evaluators
cover the benchmark workflow end to end.

Scenario index
--------------

Datasets build a :class:`ScenarioIndex` internally from the ``root``/``split``
you pass them, so you normally never construct one yourself (for the
evaluator identifiers, use
:class:`~carlanomaly.datasets.WithIdentifiers` or a composite dataset — see
:doc:`/evaluation`).  It is documented here because the keyword arguments
datasets forward to it (``clip_len``, ``stride``, ``anomaly_types``,
``towns``, ``download``, ``parts``) are defined on its constructor.

.. autoclass:: ScenarioIndex
   :members:
   :special-members: __len__, __getitem__

Constants
---------

.. autodata:: CAMERAS
   :annotation: = ("front", "left", "right", "rear")

.. autodata:: ANOMALY_TYPES
