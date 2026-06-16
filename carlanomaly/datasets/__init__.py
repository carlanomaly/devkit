"""Atomic and composite CarlAnomaly datasets."""

from .actions import ActionsDataset
from .anomaly_lidar import AnomalyLiDARDataset
from .anomaly_obs import AnomalyObservationDataset
from .anomaly_seg import AnomalySegmentationDataset
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
    # Atomic: LiDAR / labels
    "PointCloudDataset",
    "AnomalyLiDARDataset",
    "AnomalyObservationDataset",
    # Composite
    "CameraDataset",
    "LiDARDataset",
    "CarlAnomalyDataset",
    "carlanomaly_collate_fn",
]
