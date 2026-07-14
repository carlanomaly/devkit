"""Atomic and composite CarlAnomaly datasets."""

from ._base import WithIdentifiers
from .actions import ActionsDataset
from .anomaly_lidar import AnomalyLiDARDataset
from .anomaly_scenario import AnomalyScenarioDataset
from .anomaly_seg import AnomalySegmentationDataset
from .anomaly_sensor import AnomalySensorDataset
from .anomaly_timestep import AnomalyTimestepDataset
from .camera import CameraDataset
from .collisions import CollisionsDataset
from .depth import DepthDataset
from .gnss import GNSSDataset
from .imu import IMUDataset
from .joint import CarlAnomalyDataset, carlanomaly_collate_fn
from .lidar import LiDARDataset
from .pointcloud import PointCloudDataset
from .rgb import RGBDataset
from .segmentation import SegmentationDataset
from .weather import WeatherDataset

__all__ = [
    # Atomic: tabular
    "WeatherDataset",
    "GNSSDataset",
    "IMUDataset",
    "ActionsDataset",
    "CollisionsDataset",
    # Atomic: image
    "RGBDataset",
    "DepthDataset",
    "SegmentationDataset",
    "AnomalySegmentationDataset",
    # Atomic: LiDAR
    "PointCloudDataset",
    # Anomaly labels (one per evaluation level)
    "AnomalyLiDARDataset",
    "AnomalyScenarioDataset",
    "AnomalySensorDataset",
    "AnomalyTimestepDataset",
    # Composite / utilities
    "CameraDataset",
    "LiDARDataset",
    "CarlAnomalyDataset",
    "WithIdentifiers",
    "carlanomaly_collate_fn",
]
