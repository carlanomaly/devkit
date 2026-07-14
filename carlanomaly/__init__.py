"""CarlAnomaly: modular data loading and evaluation for the CarlAnomaly benchmark."""

from .download import PARTS, ensure_parts, part_for
from .evaluator import (
    PixelEvaluator,
    PointEvaluator,
    ScenarioEvaluator,
    SensorEvaluator,
    TimestepEvaluator,
)
from .index import ANOMALY_TYPES, CAMERAS, ScenarioIndex, ScenarioRecord

__all__ = [
    "ANOMALY_TYPES",
    "CAMERAS",
    "PARTS",
    "PixelEvaluator",
    "PointEvaluator",
    "ScenarioEvaluator",
    "ScenarioIndex",
    "ScenarioRecord",
    "SensorEvaluator",
    "TimestepEvaluator",
    "ensure_parts",
    "part_for",
]
