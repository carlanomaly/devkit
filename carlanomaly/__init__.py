"""CarlAnomaly — modular data loading and evaluation for the CarlAnomaly benchmark."""

from .evaluator import (
    ObservationEvaluator,
    PixelEvaluator,
    PointEvaluator,
    ScenarioEvaluator,
    SensorEvaluator,
)
from .index import ANOMALY_TYPES, CAMERAS, ScenarioIndex, ScenarioRecord

__all__ = [
    "ANOMALY_TYPES",
    "CAMERAS",
    "ObservationEvaluator",
    "PixelEvaluator",
    "PointEvaluator",
    "ScenarioEvaluator",
    "ScenarioIndex",
    "ScenarioRecord",
    "SensorEvaluator",
]
