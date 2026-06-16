Core constants
==============

.. currentmodule:: carlanomaly

.. autodata:: CAMERAS
   :annotation: = ("front", "left", "right", "rear")

.. autodata:: ANOMALY_TYPES

Advanced: scenario index
-------------------------

Datasets build a :class:`ScenarioIndex` internally from the ``root``/``split``
you pass them, so you normally never construct one yourself.  It is documented
here because the keyword arguments datasets forward to it (``clip_len``,
``stride``, ``anomaly_types``, ``towns``, ``download``, ``parts``) are defined
on its constructor.

.. autoclass:: ScenarioIndex
   :members:
   :special-members: __len__, __getitem__
