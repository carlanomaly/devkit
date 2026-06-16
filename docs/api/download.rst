Download utilities
==================

.. currentmodule:: carlanomaly

.. autodata:: carlanomaly.download.PARTS

.. autofunction:: ensure_parts

.. autofunction:: part_for

CLI
---

The ``carlanomaly-download`` command-line tool wraps :func:`ensure_parts`:

.. code-block:: bash

   # List available parts
   carlanomaly-download --list

   # Download the base part (front camera + tabular + test labels)
   carlanomaly-download --root /data/carlanomaly --parts base

   # Download LiDAR data for the test split only
   carlanomaly-download --root /data/carlanomaly --parts lidar --splits test
