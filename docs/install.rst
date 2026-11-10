Installation and data
=====================

Install the package
-------------------

.. code-block:: bash

   pip install git+https://github.com/carlanomaly/devkit

The package needs Python 3.10 or newer. It installs PyTorch, pandas, pyarrow
and Pillow if they are missing; install PyTorch yourself first if you need a
particular CUDA build.

Dataset parts
-------------

The full dataset is about 1.4 TB, but few projects need all of it. It is
published in **parts**, grouped by sensor, so that you only download what you
use. Each part has one archive for the **train** split, which holds normal
driving only, and one for the **test** split, which holds normal and anomaly
scenarios (see :doc:`dataset`):

.. list-table::
   :header-rows: 1
   :widths: 18 46 12 12

   * - Part
     - Contents
     - Train
     - Test
   * - ``base``
     - Front camera (RGB, segmentation, anomaly masks), GNSS, IMU, weather,
       control inputs, collisions, timestep labels
     - 137 GB
     - 86 GB
   * - ``camera-extended``
     - Left, right and rear cameras (RGB, segmentation, anomaly masks)
     - 427 GB
     - 267 GB
   * - ``lidar``
     - LiDAR point clouds and their anomaly labels
     - 227 GB
     - 138 GB
   * - ``depth``
     - Depth maps of all four cameras
     - 74 GB
     - 48 GB
   * - ``kitti``
     - 3D bounding boxes in KITTI format (no dataset class in this package)
     - 22 GB
     - 12 GB

Every dataset class needs ``base``, because the package finds the scenarios
and their timesteps through files in it. What else you need depends on the sensors you
use:

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - To work with
     - Download
   * - the front camera, vehicle data, weather
     - ``base``
   * - all four cameras
     - ``base``, ``camera-extended``
   * - the LiDAR
     - ``base``, ``lidar``
   * - depth maps, or :class:`~carlanomaly.datasets.CameraDataset`
     - ``base``, ``depth`` (plus ``camera-extended`` for other cameras than
       the front one)
   * - :class:`~carlanomaly.datasets.CarlAnomalyDataset`
     - everything except ``kitti``

For evaluation alone, the test archives are enough; you only need the train
archives to train a detector.

Downloading
-----------

There are three ways to get the parts. All of them put the data into one
directory, the ``root`` that you later pass to every dataset.

**From a dataset.** Datasets do not download anything unless you ask them to.
With ``download=True``, a dataset fetches the parts it needs for its split
that are not in ``root`` yet:

.. code-block:: python

   from carlanomaly.datasets import RGBDataset

   rgb = RGBDataset("/data/carlanomaly", split="test", download=True)  # base, test: 86 GB

Without ``download=True``, a missing part raises a ``FileNotFoundError``
that names the part and the command to download it. This is deliberate: a
typo in ``root`` should not start a download of several hundred gigabytes.

**From the command line**, for example on a server before training:

.. code-block:: bash

   carlanomaly-download --list                                       # show the parts
   carlanomaly-download --root /data/carlanomaly --parts base        # train and test
   carlanomaly-download --root /data/carlanomaly --parts lidar depth --splits test

**By hand.** Download the archives from the
`website <https://carlanomaly.de/download/>`_ and extract them into
``root``. The package recognises extracted parts and does not download them
again.

Both the dataset and the command line do the same thing for each archive:

1. download it into ``root/.carlanomaly_download/``, resuming an interrupted
   download where it stopped;
2. check its SHA-256 checksum;
3. extract it into ``root`` and delete the archive (``--keep-archive`` keeps
   it).

While an archive is being extracted it is on disk twice, so leave room for
the largest archive on top of the extracted data. A part that is already
there is skipped, so running the same command again is safe.

Directory layout
----------------

After extracting, ``root`` looks like this:

.. code-block:: text

   root/
     train/<town>/scenario-<n>/
     test/normal/<town>/scenario-<n>/
     test/anomaly/<town>/<anomaly type>/scenario-<n>/

Every part adds its own subdirectories (``rgb-front/``, ``pointclouds/``,
...) to the same scenario directories. You do not need to know them to use
this package; the `data format page <https://carlanomaly.de/format/>`_
describes them for reading the files directly.
