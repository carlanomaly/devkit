Loading data
============

The package has one dataset class per kind of data: one for RGB images, one
for point clouds, one for each kind of label, and so on. All of them are
PyTorch datasets and are created the same way, from the dataset directory and
a split:

.. code-block:: python

   from carlanomaly.datasets import RGBDataset

   rgb = RGBDataset("/data/carlanomaly", split="train", direction="front")
   len(rgb)  # number of items: here one per timestep of every training scenario
   rgb[0]    # FloatTensor (1, 3, 1080, 1920): the first timestep of the first scenario

Creating a dataset scans ``root`` for the scenarios of the split; it does not
load any data. Items are read from disk when you index them.

Which dataset for which data
----------------------------

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Class
     - One item
   * - :class:`~carlanomaly.datasets.RGBDataset`
     - ``FloatTensor (T, 3, 1080, 1920)``, RGB in ``[0, 1]``
   * - :class:`~carlanomaly.datasets.SegmentationDataset`
     - dict of ``LongTensor (T, 1080, 1920)``: semantic class and instance id
   * - :class:`~carlanomaly.datasets.DepthDataset`
     - ``FloatTensor (T, 1, 1080, 1920)``, log-encoded depth
   * - :class:`~carlanomaly.datasets.PointCloudDataset`
     - list of ``T`` DataFrames, one row per LiDAR point
   * - :class:`~carlanomaly.datasets.GNSSDataset`,
       :class:`~carlanomaly.datasets.IMUDataset`,
       :class:`~carlanomaly.datasets.ActionsDataset`,
       :class:`~carlanomaly.datasets.WeatherDataset`
     - ``FloatTensor (T, columns)``
   * - :class:`~carlanomaly.datasets.CollisionsDataset`
     - list of ``T`` DataFrames, one row per collision
   * - :class:`~carlanomaly.datasets.AnomalySegmentationDataset`,
       :class:`~carlanomaly.datasets.AnomalyLiDARDataset`,
       :class:`~carlanomaly.datasets.AnomalySensorDataset`,
       :class:`~carlanomaly.datasets.AnomalyTimestepDataset`,
       :class:`~carlanomaly.datasets.AnomalyScenarioDataset`
     - the labels of one :ref:`evaluation level <evaluation-levels>`
   * - :class:`~carlanomaly.datasets.CameraDataset`,
       :class:`~carlanomaly.datasets.LiDARDataset`,
       :class:`~carlanomaly.datasets.CarlAnomalyDataset`
     - a dict combining several of the above

The camera classes take ``direction`` (``"front"``, ``"left"``, ``"right"``
or ``"rear"``) to choose the camera. The :doc:`API reference
<api/datasets>` describes every class, including units and encodings.

Splits
------

``split`` selects the scenarios (see :doc:`dataset`):

``"train"``
   Normal driving only. Train your detector on this.
``"test"``
   All test scenarios, normal and anomalous. Evaluate on this.
``"test_normal"``, ``"test_anomaly"``
   One half of the test split each, for looking at the data. Do not evaluate
   on one of them alone: without the other half, a detector cannot be told
   apart from one that flags everything (or nothing).

Clips and the time dimension
----------------------------

An item is a **clip**: ``clip_len`` consecutive timesteps of one scenario.
That is why every item has a leading time dimension ``T``, of length
``clip_len``, even with the default ``clip_len=1``. Models that look at
single frames index it away with ``[:, 0]`` after batching; models that look
at sequences, such as frame predictors, get their sequence without extra
bookkeeping.

``stride`` is the number of timesteps from the start of one clip to the start
of the next. It defaults to ``clip_len``, so clips do not overlap:

.. code-block:: python

   rgb = RGBDataset("/data/carlanomaly", split="train", clip_len=4)
   rgb[0]  # (4, 3, 1080, 1920): timesteps 0-3 of the first scenario
   rgb[1]  # timesteps 4-7

   rgb = RGBDataset("/data/carlanomaly", split="train", clip_len=4, stride=1)
   rgb[1]  # timesteps 1-4

Clips never span two scenarios. Timesteps at the end of a scenario that do
not fill a whole clip are left out.

For evaluation, keep ``clip_len=1`` and ``stride=1`` (the defaults) unless your
model needs longer clips: the evaluators expect one score per timestep, and
these settings visit every timestep exactly once. :doc:`evaluation` shows what
to do with longer clips.

Combining sensors
-----------------

Datasets created with the same ``root``, ``split``, ``clip_len``, ``stride``
and filters have the same items in the same order: ``rgb[i]`` and
``lidar[i]`` hold the same timesteps of the same scenario. You can therefore
combine sensors by indexing several datasets with the same ``i``.

For the common combinations, the combined datasets do this for you and return
a dict:

- :class:`~carlanomaly.datasets.CameraDataset`: RGB, depth, segmentation and
  anomaly mask of one camera.
- :class:`~carlanomaly.datasets.LiDARDataset`: point clouds and their anomaly
  labels.
- :class:`~carlanomaly.datasets.CarlAnomalyDataset`: everything, from all
  cameras.

.. code-block:: python

   from carlanomaly.datasets import CameraDataset

   cam = CameraDataset("/data/carlanomaly", split="test", direction="front")
   item = cam[0]
   item["rgb"]           # (1, 3, 1080, 1920)
   item["anomaly_mask"]  # (1, 1080, 1920), True where the anomaly is
   item["scenario_id"]   # "/data/carlanomaly/test/normal/Town01/scenario-1"

Loading fewer scenarios
-----------------------

``towns`` and ``anomaly_types`` restrict a dataset to some of the scenarios,
for example to look at one anomaly type or to hold out a town for validation:

.. code-block:: python

   flicker = RGBDataset("/data/carlanomaly", split="test_anomaly",
                        anomaly_types=["traffic-light-flicker"])
   val = RGBDataset("/data/carlanomaly", split="train", towns=["Town10HD"])

``anomaly_types`` only filters anomaly scenarios; normal scenarios stay in.
:data:`~carlanomaly.ANOMALY_TYPES` lists the names.

Transforms
----------

``transform`` is a function applied to every item before it is returned, for
example to downscale images for training:

.. code-block:: python

   import torch.nn.functional as F

   rgb = RGBDataset("/data/carlanomaly", split="train",
                    transform=lambda x: F.interpolate(x, size=(540, 960)))

The pixel evaluator needs scores at the full 1080 × 1920 resolution, so
upsample your score maps again before evaluating.

Batching
--------

Datasets whose items have a fixed shape (images, tables, labels per timestep)
work with a plain ``DataLoader``. Point clouds, LiDAR labels and collisions
vary in size from timestep to timestep, so the default collate function
cannot stack them. Pass :func:`~carlanomaly.datasets.carlanomaly_collate_fn`
instead, which turns them into lists:

.. code-block:: python

   from torch.utils.data import DataLoader
   from carlanomaly.datasets import LiDARDataset, carlanomaly_collate_fn

   loader = DataLoader(LiDARDataset("/data/carlanomaly", split="test"),
                       batch_size=8, num_workers=8, collate_fn=carlanomaly_collate_fn)
   batch = next(iter(loader))
   batch["points"]  # list of 8 items, each a list of T DataFrames

Images are decoded on the fly and are large, so use several
``num_workers`` to keep a GPU busy.

.. _dataset-arguments:

Arguments
---------

Every dataset takes these arguments. The API pages list the ones specific to
a class, such as ``direction`` or ``sensor``.

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Argument
     - Meaning
   * - ``root``
     - The dataset directory (see :doc:`install`). A relative path works
       as long as the working directory stays the same until evaluation.
   * - ``split``
     - ``"train"`` (default), ``"test"``, ``"test_normal"`` or
       ``"test_anomaly"``.
   * - ``clip_len``
     - Timesteps per item. Default: 1.
   * - ``stride``
     - Timesteps from the start of one item to the start of the next.
       Default: ``clip_len``.
   * - ``towns``
     - Only load scenarios from these towns, e.g. ``["Town01"]``.
   * - ``anomaly_types``
     - Only load anomaly scenarios of these types. Normal scenarios are kept.
   * - ``transform``
     - Function applied to every item.
   * - ``download``
     - Download the parts this dataset needs if they are not in ``root``.
       Default: ``False``.
   * - ``parts``
     - Download these parts instead of the ones the dataset needs.
   * - ``download_verify``
     - Check the checksums of downloaded archives. Default: ``True``.
   * - ``index``
     - A :class:`~carlanomaly.ScenarioIndex` to share with another dataset.
       It replaces ``root``, ``split``, ``clip_len``, ``stride`` and the
       filters. Rarely needed.
