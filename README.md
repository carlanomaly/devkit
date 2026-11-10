# CarlAnomaly — Dev Kit

Data-loading and evaluation framework for the **[CarlAnomaly](https://carlanomaly.de)**
benchmark — multimodal anomaly detection for autonomous driving, built on CARLA
simulator data.

This package provides the tools to load the dataset and to evaluate your own
model against the benchmark. The dataset itself is available from the
[project website](https://carlanomaly.de/download/). Datasets read it
from a local directory; pass `download=True` to have them fetch missing archive
parts on first use, or use the `carlanomaly-download` command. Archives you
downloaded and extracted yourself are recognised and not fetched again.

## Installation

```bash
pip install git+https://github.com/carlanomaly/devkit
```

Requires Python ≥ 3.10 and PyTorch.

## Quick start

```python
from carlanomaly.datasets import RGBDataset

# download=True fetches the part it needs (here base, test: 86 GB) once
rgb = RGBDataset(root="/data/carlanomaly", split="test", direction="front", download=True)
clip = rgb[0]  # FloatTensor (T, 3, H, W) in [0, 1]
```

Without `download=True`, a dataset whose archive parts are missing raises a
`FileNotFoundError` that names the parts and how to fetch them.

Every dataset takes the dataset `root` and a `split` and discovers its
scenarios internally. Datasets built from the same `root`/`split` (and
`clip_len`/`stride`) align index-for-index: `rgb[i]` and `lidar[i]` refer to
the same scenario and timestep window.

## Overview

- **Datasets** (`carlanomaly.datasets`) — one class per modality (RGB,
  segmentation, depth, point clouds, anomaly labels, IMU/GNSS/weather/…), plus
  composite datasets (`CameraDataset`, `LiDARDataset`, and `CarlAnomalyDataset`
  for everything). Each returns a time dimension `T`.
- **Evaluators** (`carlanomaly.evaluator`) — streaming metric accumulators
  (update per batch, compute at the end) for the benchmark's four evaluation
  levels:

  | Level    | One score per …                | Evaluator                          |
  |----------|--------------------------------|------------------------------------|
  | Sample   | pixel / LiDAR point            | `PixelEvaluator`, `PointEvaluator` |
  | Sensor   | sensor reading                 | `SensorEvaluator`                  |
  | Timestep | timestep (all sensors jointly) | `TimestepEvaluator`                |
  | Scenario | driving scenario               | `ScenarioEvaluator`                |

See the [documentation](https://docs.carlanomaly.de) for the full evaluation
protocol, and the [baseline models](https://github.com/carlanomaly/baselines)
for complete training and evaluation examples.

## License

MIT — see [LICENSE](LICENSE). The CarlAnomaly dataset itself is released under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
