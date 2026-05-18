"""
Grasp Data Collection Configuration
====================================
Settings for scripted grasp trajectories, table/object placement,
camera, and episode recording.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class GraspDataCollectionConfig:
    """Configuration for scripted graps data collection."""

    # ========================================================================
    # Episode Settings
    # ========================================================================
    num_episodes: int = 10
    max_episode_steps: int = 2000  # ~10s at 200Hz

    # ========================================================================
    # Table Setup
    # ========================================================================
    table_pos: Tuple[float, float, float] = (0.5, 0.0, 0.8)
    table_size: Tuple[float, float, float] = (0.6, 0.8, 0.02)

    # ========================================================================
    # Object Settings
    # ========================================================================
    object_type: str = "cube"
    # Object spawn on table surface: table_pos[2] + table_size[2]/2 + margin
    object_pos: Tuple[float, float, float] = (0.355, -0.15, 0.87)

    # ========================================================================
    # Camera Settings
    # ========================================================================
    camera_enabled: bool = True
    camera_height: int = 400
    camera_width: int = 640
    camera_pos: Tuple[float, float, float] = (1.0, -0.5, 1.2)
    camera_target: Tuple[float, float, float] = (0.5, 0.0, 0.4)

    # ========================================================================
    # Data Saving
    # ========================================================================
    output_dir: str = "/workspace/tudor_unitree_isaaclab/grasp_data"
    save_tactile_rgb: bool = False
    save_camera_images: bool = True
    save_every_n_steps: int = 1

    # ========================================================================
    # Grasp Variation (for multi-episode runs)
    # ========================================================================
    randomize_object_pos: bool = False
    object_pos_noise: float = 0.005 # +/- 5 mm in x, y

    def __pos_init__(self):
        if self.object_type not in ("cube", "nut"):
            raise ValueError(f"object_type must be 'cube' or 'nut', got {self.object_type}")
        if self.num_episodes < 1: 
            raise ValueError(f"num_episodes must be >= 1, got {self.num_episodes}")
