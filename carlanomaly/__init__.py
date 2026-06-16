"""CarlAnomaly: modular data loading and evaluation for the CarlAnomaly benchmark."""

from .download import PARTS, ensure_parts, part_for
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
    "PARTS",
    "ObservationEvaluator",
    "PixelEvaluator",
    "PointEvaluator",
    "ScenarioEvaluator",
    "ScenarioIndex",
    "ScenarioRecord",
    "SensorEvaluator",
    "ensure_parts",
    "part_for",
]
