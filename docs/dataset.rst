The dataset
===========

This page explains the terms the rest of the documentation uses. The
`dataset page <https://carlanomaly.de/dataset/>`_ on the website has more
detail and pictures, and the `data format page <https://carlanomaly.de/format/>`_
the file formats for reading the data without this package.

Scenarios and timesteps
-----------------------

A **scenario** is one 30-second drive of the ego vehicle through one of six
CARLA towns (``Town01`` to ``Town05``, ``Town10HD``), in fixed or changing
weather and lighting. Each scenario has its own directory.

All sensors record at the same simulator tick, ten times per second. One such
tick is a **timestep**, numbered from 0 within its scenario. Files of the same
timestep belong together: the front image, the point cloud and the weather
row of timestep 232 describe the same moment.

Sensors
-------

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Sensor
     - Records per timestep
   * - 4 cameras
     - ``front``, ``left``, ``right``, ``rear``: an RGB image, a depth map and
       a semantic and instance segmentation, each 1920 × 1080 pixels with a
       105° horizontal field of view.
   * - LiDAR
     - About 100,000 points with position, semantic class and instance id
       (64 channels, 120 m range).
   * - GNSS, IMU
     - Position; acceleration, angular velocity and compass heading.
   * - Collisions
     - Every collision of the ego vehicle, with the other actor and the
       impulse.
   * - Environment
     - The 14 weather parameters and the autopilot's control commands.

A **sensor reading** is what one sensor records at one timestep: one camera
image or one point cloud.

Splits
------

``train``
   Normal driving only. Detectors learn what normal looks like from it.
``test``
   Normal scenarios (``test/normal``) and anomaly scenarios
   (``test/anomaly``). Benchmark results are computed on all of it.

Each anomaly scenario contains one type of anomaly. The anomaly does not last
the whole scenario: the timesteps before it starts are labelled normal.

Anomaly types
-------------

.. list-table::
   :header-rows: 1
   :widths: 32 18 50

   * - Type
     - Masks
     - What happens
   * - ``spawn-props``
     - yes
     - Debris suddenly appears on the road in front of the ego vehicle.
   * - ``traffic-light-off``
     - yes
     - Traffic lights are switched off, which never happens in training.
   * - ``traffic-light-flicker``
     - yes
     - Traffic lights alternate erratically between red and green.
   * - ``traffic-light-yellow-blinking``
     - yes
     - Traffic lights keep blinking yellow.
   * - ``steer-driver``
     - yes
     - Another car steers abruptly for no apparent reason.
   * - ``running-pedestrian``
     - yes
     - A pedestrian moves at unusually high speed.
   * - ``vanish-actor``
     - no
     - A car disappears without physical cause.
   * - ``change-weather``
     - no
     - The weather changes abruptly within a few timesteps.
   * - ``street-light-flicker``
     - no
     - Street lights flicker, changing the illumination at night.

The names are the directory names under ``test/anomaly/<town>/`` and the keys
of the ``by_type`` results. Types with masks mark the anomalous object in
every camera image and point cloud that shows it. The others have no object to point
at, so they have no masks and only count at the timestep and scenario levels.

.. _evaluation-levels:

Labels and evaluation levels
----------------------------

The benchmark asks four questions, from the finest to the coarsest. Each has
its own labels, and :doc:`evaluation` has an evaluator for each:

.. list-table::
   :header-rows: 1
   :widths: 14 30 56

   * - Level
     - Question
     - Label
   * - Sample
     - Which pixels or points show the anomaly?
     - A mask per camera image, a flag per LiDAR point.
   * - Sensor
     - Does this sensor see an anomaly now?
     - One flag per timestep and sensor: the anomaly is visible to it.
   * - Timestep
     - Is the scene anomalous now?
     - One flag per timestep, set while the anomaly is going on. It is set
       whenever a sensor flag is, but also without one: the types without
       masks set only this flag and the scenario flag.
   * - Scenario
     - Did anything anomalous happen?
     - One flag per scenario: it is under ``test/anomaly``.

All labels come from the simulator's state, not from manual annotation. A
mask covers the anomalous object as CARLA renders it, which can differ from
what a person would outline.
