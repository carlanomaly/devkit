Evaluating a detector
=====================

Your detector produces anomaly **scores**: numbers where higher means more
anomalous. An evaluator collects them over the test split, compares them with
the ground truth, and reports how well they separate anomalous from normal
data. It loads the ground truth from the dataset itself, so your evaluation
code only handles scores.

There is one evaluator per :ref:`evaluation level <evaluation-levels>`:

.. list-table::
   :header-rows: 1
   :widths: 14 32 54

   * - Level
     - Your detector gives one score per
     - Evaluator
   * - Sample
     - pixel of a camera image, or LiDAR point
     - :class:`~carlanomaly.PixelEvaluator`, :class:`~carlanomaly.PointEvaluator`
   * - Sensor
     - sensor reading (one camera image or point cloud)
     - :class:`~carlanomaly.SensorEvaluator`
   * - Timestep
     - timestep, using any or all sensors
     - :class:`~carlanomaly.TimestepEvaluator`
   * - Scenario
     - scenario
     - :class:`~carlanomaly.ScenarioEvaluator`

You can evaluate at one level or at all of them. A detector that scores pixels
can report every level, by reducing its pixel scores step by step, as in the
:ref:`complete example <all-levels>` below.

How evaluators work
-------------------

Every evaluator is used the same way:

.. code-block:: python

   evaluator = SomeEvaluator(...)
   for batch in loader:
       scores = ...                       # your detector
       evaluator.update(scores, scenario_ids, timestep_ids)
   results = evaluator.compute()          # dict of metrics, see Metrics

``update`` keeps only what the metrics need: a histogram at the sample level,
one number per score at the other levels. ``compute`` can be
called at any time; ``reset`` starts over.

Two rules make results comparable with the benchmark and the paper:

- **Evaluate the whole** ``test`` **split.** The metrics measure how well
  scores separate anomalous from normal data, so they need both. Leaving out
  normal scenarios hides false alarms; leaving out anomaly types changes the
  mix of positives.
- **Do not tune on the test split.** Choose thresholds, normalisation
  statistics and hyperparameters on the training split.

Identifiers
-----------

``update`` takes, next to the scores, the scenario and timestep each score
belongs to. The evaluator uses them to find the matching label, and to report
results per anomaly type.

The single-sensor datasets return bare tensors, without identifiers. Wrap them
in :class:`~carlanomaly.datasets.WithIdentifiers` to get both:

.. code-block:: python

   from torch.utils.data import DataLoader
   from carlanomaly.datasets import RGBDataset, WithIdentifiers

   test = WithIdentifiers(RGBDataset("/data/carlanomaly", split="test"))
   loader = DataLoader(test, batch_size=8, num_workers=8)

   batch = next(iter(loader))
   batch["data"]         # (8, 1, 3, 1080, 1920): the RGB images
   batch["scenario_id"]  # 8 strings: the scenario directory of each item
   batch["timesteps"]    # (8, 1): the timesteps of each item

The combined datasets (:class:`~carlanomaly.datasets.CameraDataset`,
:class:`~carlanomaly.datasets.LiDARDataset`,
:class:`~carlanomaly.datasets.CarlAnomalyDataset`) include ``scenario_id``
and ``timesteps`` without the wrapper.

A ``scenario_id`` is the path of the scenario's directory. Pass it on
unchanged: the evaluators read the labels from that directory and the anomaly
type from the path.

**One score per timestep.** ``batch["timesteps"]`` has a column per timestep
of the clip. With the default ``clip_len=1`` there is one, and
``batch["timesteps"][:, 0]`` gives the timestep of each item, as in the
examples below. A model that reads longer clips scores one timestep per clip,
usually the last; pass that one, ``batch["timesteps"][:, -1]``, and use
``stride=1`` so that every timestep gets a score. The first ``clip_len - 1``
timesteps of each scenario then have no score and are left out.

Sample level
------------

At the sample level your detector marks *where* the anomaly is: a score for
every pixel of a camera image or every LiDAR point.

.. code-block:: python

   import torch
   from carlanomaly import PixelEvaluator

   pixel_eval = PixelEvaluator(sensor="front")

   with torch.no_grad():
       for batch in loader:
           images = batch["data"][:, 0].to("cuda")  # (B, 3, 1080, 1920)
           scores = model(images)                   # (B, 1080, 1920)
           pixel_eval.update(scores, batch["scenario_id"], batch["timesteps"][:, 0])

   results = pixel_eval.compute()

Score maps must have the camera's full resolution, 1080 × 1920; upsample them
if your model works on smaller images. They can stay on the GPU.

The evaluator puts every pixel of the test split into one pool: anomalous
pixels as positives, and all other pixels, from anomaly scenarios and normal
scenarios alike, as negatives. A detector therefore cannot do well by giving
high scores to a whole class of objects. If it flags every traffic light,
the many normal traffic lights in normal scenarios count as false alarms. The
anomaly types without masks contribute only negatives here.

For the LiDAR, :class:`~carlanomaly.PointEvaluator` takes one score per point,
as a list with one tensor per item (point clouds differ in size):

.. code-block:: python

   from torch.utils.data import DataLoader
   from carlanomaly import PointEvaluator
   from carlanomaly.datasets import LiDARDataset, carlanomaly_collate_fn

   lidar = LiDARDataset("/data/carlanomaly", split="test")
   loader = DataLoader(lidar, batch_size=8, collate_fn=carlanomaly_collate_fn)
   point_eval = PointEvaluator()

   for batch in loader:
       clouds = [clip[0] for clip in batch["points"]]  # B DataFrames
       scores = [model(cloud) for cloud in clouds]     # B tensors of shape (N,)
       point_eval.update(scores, batch["scenario_id"], batch["timesteps"][:, 0])

The scores of a point cloud must be in the order of its rows.

Sensor level
------------

At the sensor level your detector gives one score per sensor reading: does
this camera image, or this point cloud, show an anomaly? A timestep counts as
anomalous for a sensor while the anomaly is visible to that sensor: debris
appearing ahead of the ego vehicle shows in the front camera, but usually not
in the rear one.

A detector that scores pixels gets a sensor score by reducing its score map,
for example to its maximum:

.. code-block:: python

   from carlanomaly import SensorEvaluator

   sensor_eval = SensorEvaluator(sensor="front")

   for batch in loader:
       pixel_scores = model(batch["data"][:, 0].to("cuda"))  # (B, 1080, 1920)
       scores = pixel_scores.amax(dim=(1, 2))               # (B,)
       sensor_eval.update(scores, batch["scenario_id"], batch["timesteps"][:, 0])

``sensor`` is the camera or ``"lidar"``. Evaluate each sensor with its own
evaluator.

Timestep level
--------------

At the timestep level your detector gives one score per timestep for the
whole scene. A timestep counts as anomalous while the anomaly is going on,
whether or not a sensor shows it. This is the level for anomalies without a
location, such as a sudden change of weather, and for detectors that combine
several sensors or the vehicle data.

A simple way to combine sensors is to bring their scores to a common scale
and take the maximum:

.. code-block:: python

   from carlanomaly import TimestepEvaluator

   timestep_eval = TimestepEvaluator()

   # one batch of a CarlAnomalyDataset loader; mean and std estimated on the
   # training split
   camera_scores = camera_model(batch["front_rgb"][:, 0]).amax(dim=(1, 2))  # (B,)
   lidar_scores = lidar_model(batch["points"])                              # (B,)
   fused = torch.maximum((camera_scores - cam_mean) / cam_std,
                         (lidar_scores - lidar_mean) / lidar_std)
   timestep_eval.update(fused, batch["scenario_id"], batch["timesteps"][:, 0])

With a single sensor, the timestep evaluator takes the same scores as the
sensor evaluator; the two differ only in their labels.

How quickly a detector notices an anomaly is a separate question from how
well it ranks timesteps. :meth:`~carlanomaly.TimestepEvaluator.time_to_detection`
answers it from the same scores, at a fixed false-alarm rate on the normal
scenarios (see :ref:`time-to-detection`):

.. code-block:: python

   timestep_eval.time_to_detection(max_false_alarm_rate=0.01)
   # {'ttd': 17.1, 'detection_rate': 0.16, 'false_alarm_rate': 0.009, ...}

Scenario level
--------------

At the scenario level your detector gives one score per scenario: did
anything anomalous happen in these 30 seconds? Scenarios under
``test/anomaly`` count as anomalous, those under ``test/normal`` as normal.

``update`` takes one scenario at a time. The simplest score is the highest
timestep score, which :meth:`~carlanomaly.TimestepEvaluator.max_per_scenario`
computes from what a timestep or sensor evaluator has collected:

.. code-block:: python

   from carlanomaly import ScenarioEvaluator

   scenario_eval = ScenarioEvaluator()
   for scenario_id, score in timestep_eval.max_per_scenario().items():
       scenario_eval.update(score, scenario_id)

   results = scenario_eval.compute()

A detector that looks at whole scenarios can pass its own scores instead.

.. _all-levels:

All levels at once
------------------

One pass over the test split can feed all levels. This evaluates a detector
that scores the pixels of the front camera:

.. code-block:: python

   import torch
   from torch.utils.data import DataLoader
   from carlanomaly import (PixelEvaluator, ScenarioEvaluator, SensorEvaluator,
                            TimestepEvaluator)
   from carlanomaly.datasets import RGBDataset, WithIdentifiers

   test = WithIdentifiers(RGBDataset("/data/carlanomaly", split="test"))
   loader = DataLoader(test, batch_size=8, num_workers=8)

   pixel_eval = PixelEvaluator(sensor="front")
   sensor_eval = SensorEvaluator(sensor="front")
   timestep_eval = TimestepEvaluator()

   with torch.no_grad():
       for batch in loader:
           ids, timesteps = batch["scenario_id"], batch["timesteps"][:, 0]
           pixel_scores = model(batch["data"][:, 0].to("cuda"))  # (B, 1080, 1920)
           frame_scores = pixel_scores.amax(dim=(1, 2))         # (B,)

           pixel_eval.update(pixel_scores, ids, timesteps)
           sensor_eval.update(frame_scores, ids, timesteps)
           timestep_eval.update(frame_scores, ids, timesteps)

   scenario_eval = ScenarioEvaluator()
   for scenario_id, score in timestep_eval.max_per_scenario().items():
       scenario_eval.update(score, scenario_id)

   results = {
       "sample": pixel_eval.compute(),
       "sensor": sensor_eval.compute(),
       "timestep": timestep_eval.compute(),
       "scenario": scenario_eval.compute(),
   }

To analyse the scores yourself, for example to plot them over time,
``to_dataframe()`` on the sensor, timestep and scenario evaluators returns
every score with its label, anomaly type and town.

Next, :doc:`metrics` explains the numbers in ``results``.
