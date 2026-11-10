Scenario index and constants
============================

.. currentmodule:: carlanomaly

Every dataset finds its scenarios and builds its items through a
:class:`ScenarioIndex`. You only need it to share one index between datasets
(the ``index`` argument) or to look up which scenario an item belongs to.

.. autoclass:: ScenarioIndex
   :members: records, timesteps_for
   :special-members: __getitem__

.. autoclass:: ScenarioRecord

.. data:: CAMERAS
   :value: ("front", "left", "right", "rear")

   The cameras, as passed to ``direction`` and ``sensor``.

.. data:: ANOMALY_TYPES

   The anomaly types, as passed to ``anomaly_types``: ``"change-weather"``,
   ``"running-pedestrian"``, ``"spawn-props"``, ``"steer-driver"``,
   ``"street-light-flicker"``, ``"traffic-light-flicker"``,
   ``"traffic-light-off"``, ``"traffic-light-yellow-blinking"`` and
   ``"vanish-actor"``.
