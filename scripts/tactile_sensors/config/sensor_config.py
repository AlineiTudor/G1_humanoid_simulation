"""
Sensor Configuration
====================
Defines tactile sensor placement, parameters, and enabling/disabling.

Design:
- Default parameters apply to ALL sensors
- Per-location overrides for specific customization
- Tactile array sizes defined per location (fill in actual values)
- Easy enable/disable of palm and individual fingers
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, Any, TYPE_CHECKING

# Lazy import to avoid triggering Isaac Lab before SimulationApp
if TYPE_CHECKING:
    from isaaclab_assets.sensors import GELSIGHT_R15_CFG


@dataclass
class SensorConfig:
    """Configuration for tactile sensors on the hand."""

    # ========================================================================
    # Sensor Naming
    # ========================================================================
    # Prim name appended to force sensor links (e.g., "right_palm_force_sensor/tactile_sensor").
    # Set to "" if the hand USD has no dedicated sensor sub-prim  the force
    # sensor link itself (which has RigidBodyAPI) will be used as the sensor body.
    sensor_prim_name: str = ""

    # ========================================================================
    # Sensor Enable/Disable Flags
    # ========================================================================
    # Global enable (set to False to disable ALL sensors)
    sensors_enabled: bool = True

    # Per-location enable flags
    # Set any to False to disable that specific sensor
    palm_enabled: bool = True
    index_enabled: bool = True
    middle_enabled: bool = True
    little_enabled: bool = True
    thumb_enabled: bool = True
    ring_enabled: bool = True
    # ========================================================================
    # Tactile Array Sizes (Number of Taxels)
    # ========================================================================
    # Format: (rows, columns) for each sensor location

    # Palm sensor
    palm_array_size: Tuple[int, int] = (8, 14)

    # Index finger (3 regions: proximal, middle, distal)
    index_proximal_array_size: Tuple[int, int] = (10, 8)  
    index_middle_array_size: Tuple[int, int] = (12, 8)    
    index_distal_array_size: Tuple[int, int] = (3, 3)    

    # Middle finger (3 regions)
    middle_proximal_array_size: Tuple[int, int] = (10, 8)  
    middle_middle_array_size: Tuple[int, int] = (12, 8)    
    middle_distal_array_size: Tuple[int, int] = (3, 3)    

    # Little finger (3 regions)
    little_proximal_array_size: Tuple[int, int] = (10, 8)  
    little_middle_array_size: Tuple[int, int] = (12, 8)    
    little_distal_array_size: Tuple[int, int] = (3, 3)    

    # Thumb (4 regions: proximal, middle, distal, tip)
    thumb_proximal_array_size: Tuple[int, int] = (12, 8)  
    thumb_middle_array_size: Tuple[int, int] = (3, 3)   
    thumb_distal_array_size: Tuple[int, int] = (12, 8)   
    thumb_tip_array_size: Tuple[int, int] = (3, 3)  

    # Ring (3 regions)
    ring_proximal_array_size: Tuple[int, int] = (10, 8)  
    ring_middle_array_size: Tuple[int, int] = (12, 8)    
    ring_distal_array_size: Tuple[int, int] = (3, 3)    

    # ========================================================================
    # Default Sensor Parameters (Applied to ALL sensors)
    # ========================================================================
    # These values are used unless overridden in per_location_overrides

    # Rendering configuration
    # IMPORTANT: Set to None here to avoid Isaac Lab imports before SimulationApp
    # Use get_default_render_cfg() or set explicitly when creating SensorConfig
    render_cfg: Optional[Any] = None

    # Tactile surface margin (gap between taxels and edge)
    tactile_margin: float = 0.003

    # Enable camera-based tactile sensing (depth images)
    enable_camera_tactile: bool = True

    # Enable force field computation (requires SDF collision meshes on contact objects)
    enable_force_field: bool = True

    # Force field physics parameters
    normal_contact_stiffness: float = 20.0
    friction_coefficient: float = 2.0
    tangential_stiffness: float = 0.2

    # Update frequency (0.0 = every step)
    update_period: float = 0.0

    # History length (0 = no history, just current frame)
    history_length: int = 0

    # Tactile normal offset (meters). Pushes tactile sensing points outward
    # along the surface normal so they extend past parent link collision
    # geometry. Needed for multi-body hands (RH56E2) where force sensor
    # meshes are recessed behind parent collision surfaces. Set to 0.0
    # to disable. Typical values: 0.005-0.02 for small objects, more for
    # larger scaled objects.
    tactile_normal_offset: float = 0.005

    # Contact object path pattern (use ENV_REGEX_NS for multi-env)
    # MUST point to the contact object's prim path for force field to work.
    # The object must have an SDF collision mesh (USD mesh, not procedural).
    # Set to None only when no contact object exists (disables force field).
    contact_object_prim_path_expr: Optional[str] = "{ENV_REGEX_NS}/grasp_object"

    # ========================================================================
    # Per-Location Parameter Overrides (Optional)
    # ========================================================================
    # Use this to override default parameters for specific locations
    # Format: {"location_key": {"param_name": value, ...}, ...}
    #
    # Example:
    #   per_location_overrides = {
    #       "palm": {"normal_contact_stiffness": 5.0},  # Stiffer palm
    #       "thumb_tip": {"friction_coefficient": 3.0},  # Grippier thumb
    #   }
    per_location_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # ========================================================================
    # Helper Methods
    # ========================================================================

    @staticmethod
    def get_default_render_cfg():
        """
        Get default GelSight R15 render config.

        This is a static method to allow lazy importing of Isaac Lab modules.
        Only call this AFTER SimulationApp has been instantiated!

        Returns:
            GELSIGHT_R15_CFG instance
        """
        from isaaclab_assets.sensors import GELSIGHT_R15_CFG
        return GELSIGHT_R15_CFG

    def get_tactile_array_size(self, location: str) -> Tuple[int, int]:
        """
        Returns the tactile array size for a specific sensor location.

        Args:
            location: "palm" or "{finger}_{region}" (e.g., "index_proximal")

        Returns:
            (rows, cols) tuple

        Raises:
            ValueError: If location is unknown or array size not configured
        """
        # Map location strings to attribute names
        attr_name = f"{location}_array_size"

        if not hasattr(self, attr_name):
            raise ValueError(f"Unknown sensor location: {location}")

        array_size = getattr(self, attr_name)

        # Check if size is configured
        if not array_size or len(array_size) != 2:
            raise ValueError(
                f"Tactile array size not configured for location '{location}'. "
                f"Please set {attr_name} in SensorConfig."
            )

        return array_size

    def is_location_enabled(self, location: str) -> bool:
        """
        Checks if a sensor location is enabled.

        Args:
            location: "palm" or "{finger}_{region}" (e.g., "index_proximal")

        Returns:
            True if sensor is enabled, False otherwise
        """
        # Check global enable flag
        if not self.sensors_enabled:
            return False

        # Check per-location enable flag
        if location == "palm":
            return self.palm_enabled

        # Extract finger name from location (e.g., "index_proximal" -> "index")
        finger = location.split("_")[0]

        # Map finger to enable attribute
        enable_attr = f"{finger}_enabled"

        if not hasattr(self, enable_attr):
            return False

        return getattr(self, enable_attr)

    def get_sensor_parameters(self, location: str) -> Dict[str, Any]:
        """
        Returns complete sensor parameters for a specific location.

        This combines:
        1. Default parameters (from class attributes)
        2. Tactile array size (location-specific)
        3. Per-location overrides (if any)

        Args:
            location: "palm" or "{finger}_{region}"

        Returns:
            Dictionary of sensor parameters ready for VisuoTactileSensorCfg

        Note:
            If render_cfg is None, automatically uses GELSIGHT_R15_CFG default.
            This requires Isaac Lab to be imported (call after SimulationApp init).
        """
        # Use default render config if not set
        render_cfg = self.render_cfg
        if render_cfg is None:
            render_cfg = self.get_default_render_cfg()

        # Start with default parameters
        params = {
            "render_cfg": render_cfg,
            "tactile_array_size": self.get_tactile_array_size(location),
            "tactile_margin": self.tactile_margin,
            "enable_camera_tactile": self.enable_camera_tactile,
            "enable_force_field": self.enable_force_field,
            "normal_contact_stiffness": self.normal_contact_stiffness,
            "friction_coefficient": self.friction_coefficient,
            "tangential_stiffness": self.tangential_stiffness,
            "update_period": self.update_period,
            "history_length": self.history_length,
            "contact_object_prim_path_expr": self.contact_object_prim_path_expr,
        }

        # Apply per-location overrides (if any)
        if location in self.per_location_overrides:
            params.update(self.per_location_overrides[location])

        return params

    def get_all_enabled_locations(self, hand_config) -> list[str]:
        """
        Returns list of all enabled sensor locations.

        Args:
            hand_config: HandConfig instance (to get all possible locations)

        Returns:
            List of enabled location keys (e.g., ["palm", "index_proximal", ...])
        """
        all_locations = hand_config.get_all_sensor_locations().keys()
        return [loc for loc in all_locations if self.is_location_enabled(loc)]
