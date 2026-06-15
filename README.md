# CarlAnomaly — Dev Kit

Data-loading and evaluation framework for the **[CarlAnomaly](https://carlanomaly.github.io)**
benchmark — multimodal anomaly detection for autonomous driving, built on CARLA
simulator data.

This package provides the tools to load the dataset and to evaluate your own
model against the benchmark. The dataset itself is available from the
[project website](https://carlanomaly.github.io/download/).

## Installation

```bash
pip install git+https://github.com/carlanomaly/devkit
```

Requires Python ≥ 3.10 and PyTorch.

## Overview

- **`ScenarioIndex`** — discovers scenario directories for a split and builds a
  flat sliding-window index of `(ScenarioRecord, frame_start)` tuples with
  configurable `clip_len` and `stride`. All datasets that share an index align
  on `dataset[i]`.
- **Atomic datasets** (`carlanomaly.datasets`) — one per modality (RGB,
  segmentation, depth, point clouds, anomaly masks, IMU/GNSS/weather/…). Each
  takes a `ScenarioIndex` and an optional transform and returns a time
  dimension `T`.
- **Composite datasets** — `CameraDataset`, `LiDARDataset`, and
  `CarlAnomalyDataset` (everything). The joint dataset uses
  `carlanomaly_collate_fn` to handle variable-length point clouds and
  collisions.
- **Evaluators** (`carlanomaly.evaluator`) — streaming evaluation (update per
  batch, compute at the end) at the **pixel**, **point**, **sensor**,
  **observation**, and **scenario** tiers.

## Quick start

```python
from carlanomaly import ScenarioIndex
from carlanomaly.datasets.rgb import RGBDataset

index = ScenarioIndex(root="/path/to/carlanomaly", split="test", clip_len=1)
rgb = RGBDataset(index, camera="front")
frame = rgb[0]  # FloatTensor (T, 3, H, W) in [0, 1]
```

See the [baseline models](https://github.com/carlanomaly/baselines) for complete
training and evaluation examples.

## License

MIT — see [LICENSE](LICENSE).
