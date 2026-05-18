"""
Hand USD Configuration
======================
Defines hand-specific constants for RH56E2 hand model.
Supports both left and right hands with the same USD structure.

Usage:
    # Right hand (default)
    right_hand = HandConfig()

    # Left hand
    left_hand = HandConfig(hand_side="left")
"""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class HandConfig:
    """Configuration for RH56E2 hand model (left or right)."""

    # ========================================================================
    # USD File Configuration
    # ========================================================================
    usd_base_path: str = "/workspace/tudor_unitree_isaaclab/RH56E2_USD/RH56E2_R_2026_1_5"
    usd_filename: str = "RH56E2_R_2026_1_5.usd"
    usd_root_prim: str = "RH56E2_R_2026_1_5"  # Change for left hand when instantiating

    # Hand side: "right" or "left"
    # This affects all link name prefixes automatically
    hand_side: str = "right"  # Change to "left" when instantiating

    # ========================================================================
    # Finger Structure Definition
    # ========================================================================
    # Fingers that have tactile sensors (in order)
    finger_names: tuple = ("index", "middle", "little", "thumb")

    # Number of regions (sensors) per finger
    # Thumb has 4, others have 3
    regions_per_finger: Dict[str, int] = field(default_factory=lambda: {
        "index": 3,
        "middle": 3,
        "little": 3,
        "thumb": 4,
    })

    # Region names in order from base to tip
    # (You'll use these to index tactile_array_sizes in SensorConfig)
    region_names: tuple = ("proximal", "middle", "distal", "tip")

    # ========================================================================
    # Compliant Contact Properties (Physics)
    # ========================================================================
    # Applied to elastomer links to make them deformable
    compliant_contact_stiffness: float = 100.0
    compliant_contact_damping: float = 10.0

    # Additional physics properties
    max_depenetration_velocity: float = 5.0
    contact_offset: float = 0.001
    rest_offset: float = -0.0005
    disable_gravity: bool = True

    # Solver iterations
    solver_position_iteration_count: int = 12
    solver_velocity_iteration_count: int = 1

    # ========================================================================
    # Camera Configuration
    # ========================================================================
    camera_name_convention: str = "tactile_sensor_camera"  # Camera prim name under each sensor

    # ========================================================================
    # Helper Methods (Auto-generate link names based on hand_side)
    # ========================================================================

    #TODO: Modify paths such that they are not hardcoded
    def __post_init__(self):
        if self.hand_side == "left":
            self.usd_base_path = "/workspace/tudor_unitree_isaaclab/RH56E2_USD/RH56E2_L_2026_1_5"
            self.usd_filename = "RH56E2_L_2026_1_5.usd"
            self.usd_root_prim = "RH56E2_L_2026_1_5"
        else:
            self.usd_base_path = "/workspace/tudor_unitree_isaaclab/RH56E2_USD/RH56E2_R_2026_1_5"
            self.usd_filename = "RH56E2_R_2026_1_5.usd"
            self.usd_root_prim = "RH56E2_R_2026_1_5"

    @property
    def usd_full_path(self) -> str:
        """Returns complete USD file path."""
        return f"{self.usd_base_path}/{self.usd_filename}"

    def get_palm_link_name(self) -> str:
        """
        Returns palm force sensor link name.
        """
        return f"{self.hand_side}_palm_force_sensor"

    def get_finger_link_name(self, finger: str, region_index: int) -> str:
        """
        Returns force sensor link name for a specific finger region.

        Args:
            finger: "index", "middle", "little", or "thumb"
            region_index: 0-based index (0=proximal, 1=middle, 2=distal, 3=tip)

        Returns:
            Link name like "right_index_force_sensor_1"

        Note: USD uses 1-based sensor numbering (1, 2, 3, ...)
        """
        sensor_number = region_index + 1
        return f"{self.hand_side}_{finger}_force_sensor_{sensor_number}"

    def get_location_key(self, finger: str, region_index: int) -> str:
        """
        Returns standardized location key for a finger region.

        Args:
            finger: "index", "middle", "little", or "thumb"
            region_index: 0-based index

        Returns:
            Location key like "index_proximal", "thumb_tip", etc.
        """
        region_name = self.region_names[region_index]
        return f"{finger}_{region_name}"

    def get_all_sensor_locations(self) -> Dict[str, str]:
        """
        Returns mapping of all sensor locations to their USD link names.

        Returns:
            {
                "palm": "right_palm_force_sensor",
                "index_proximal": "right_index_force_sensor_1",
                "index_middle": "right_index_force_sensor_2",
                "index_distal": "right_index_force_sensor_3",
                "thumb_proximal": "right_thumb_force_sensor_1",
                "thumb_middle": "right_thumb_force_sensor_2",
                "thumb_distal": "right_thumb_force_sensor_3",
                "thumb_tip": "right_thumb_force_sensor_4",
                ...
            }
        """
        locations = {}

        # Add palm
        locations["palm"] = self.get_palm_link_name()

        # Add all finger regions
        for finger in self.finger_names:
            num_regions = self.regions_per_finger[finger]
            for region_idx in range(num_regions):
                location_key = self.get_location_key(finger, region_idx)
                link_name = self.get_finger_link_name(finger, region_idx)
                locations[location_key] = link_name

        return locations

    def get_physics_material_path(self, location: str) -> str:
        """
        Returns physics material prim path for a sensor location.

        Args:
            location: "palm" or "{finger}_{region}" (e.g., "index_proximal")

        Returns:
            Material path like "elastomer_right_index_tip"

        Note: Physics materials are applied per-finger (to the tip),
              not per-region, so all regions of a finger share the same material.
        """
        if location == "palm":
            return f"elastomer_{self.hand_side}_palm_tip"

        # Extract finger name from location (e.g., "index_proximal" -> "index")
        finger = location.split("_")[0]
        return f"elastomer_{self.hand_side}_{finger}_tip"

