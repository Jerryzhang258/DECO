"""Single place that resolves the LeRobotDataset import across lerobot package versions.

Older lerobot releases (and the vendored copy inside VB-VLA) expose these under
`lerobot.common.datasets.lerobot_dataset`; newer releases moved this to
`lerobot.datasets.lerobot_dataset`. Everything else in this repo should import
LeRobotDataset/LeRobotDatasetMetadata from here instead of picking one path directly.
"""

try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
except ImportError:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

__all__ = ["LeRobotDataset", "LeRobotDatasetMetadata"]
