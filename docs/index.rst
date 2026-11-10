CarlAnomaly DevKit
==================

`CarlAnomaly <https://carlanomaly.de>`_ is a benchmark for anomaly detection
in autonomous driving. It consists of 30-second driving scenarios recorded in
the CARLA simulator with four cameras, a LiDAR and vehicle sensors. The
training split contains only normal driving; the test split adds scenarios
with nine kinds of anomalies, from debris on the road to traffic lights that
blink yellow or a car that vanishes. A detector is trained on normal data and
has to tell where, when and whether something anomalous happens.

This package, ``carlanomaly``, does the bookkeeping around that:

- **Loading**: every sensor and every label is a PyTorch ``Dataset``, so the
  data plugs into a ``DataLoader`` and your training code.
- **Evaluation**: one evaluator per benchmark level computes AUROC, AUPR and
  FPR95 from your detector's scores. It loads the ground truth itself, so you
  never handle the labels.
- **Download**: the dataset is several hundred gigabytes, split into parts.
  The package fetches the parts you need, and only when you ask it to.

.. code-block:: bash

   pip install git+https://github.com/carlanomaly/devkit

Example
-------

This scores every pixel of the front camera on the test split and reports how
well the scores separate anomalous from normal pixels:

.. code-block:: python

   import torch
   from torch.utils.data import DataLoader
   from carlanomaly import PixelEvaluator
   from carlanomaly.datasets import RGBDataset, WithIdentifiers

   # The front camera's test images, each with the scenario and timestep it
   # belongs to. Fetches the base part (86 GB) on first use.
   test = WithIdentifiers(RGBDataset("/data/carlanomaly", split="test", download=True))
   evaluator = PixelEvaluator(sensor="front")

   with torch.no_grad():
       for batch in DataLoader(test, batch_size=8, num_workers=8):
           images = batch["data"][:, 0]          # (B, 3, 1080, 1920)
           scores = model(images)                # (B, 1080, 1920), higher = more anomalous
           evaluator.update(scores, batch["scenario_id"], batch["timesteps"][:, 0])

   results = evaluator.compute()
   print(results["auroc"], results["fpr95"])

``model`` stands for your detector. The ``[:, 0]`` drops the time dimension
that every dataset returns; :doc:`loading-data` explains why it is there.

Where to go next
----------------

- :doc:`install`: install the package and get the parts of the dataset you
  need.
- :doc:`dataset`: what is in a scenario, the anomaly types, and how they are
  labelled.
- :doc:`loading-data`: splits, clips, and the dataset classes for each sensor.
- :doc:`evaluation`: evaluate a detector at the four benchmark levels.
- :doc:`metrics`: what the reported numbers mean and how to compare them.
- The `baselines <https://github.com/carlanomaly/baselines>`_ repository has
  complete training and evaluation scripts for the detectors in the paper.

.. toctree::
   :hidden:
   :caption: Guide

   install
   dataset
   loading-data
   evaluation
   metrics

.. toctree::
   :hidden:
   :caption: API reference

   api/datasets
   api/evaluators
   api/download
   api/scenario-index
