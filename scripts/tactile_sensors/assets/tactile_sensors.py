"""
Tactile Sensor Asset Configurations
"""

from isaaclab.sensors import TiledCameraCfg
from isaaclab_contrib.sensors.tacsl_sensor import VisuoTactileSensorCfg
from typing import Dict

from ..config.sensor_config import SensorConfig
from ..config.hand_config import HandConfig

def create_tactile_sensor_cfg(
        hand_cfg: HandConfig,
        sensor_cfg: SensorConfig,
        location: str,
        prim_path_base: str = "{ENV_REGEX_NS}/hand",
) -> VisuoTactileSensorCfg:
    """
    Creates VisuoTactileSensorCfg fora a specific sensor location.

    Args: 
        hand_cfg: HandConfig instance
        sensor_cfg: SensorConfig instance
        location: Sensor location (e.g. "palm", "index_proximal" etc.)
        prim_path_base: Base prim path for the hand
    
    Return: 
        VisuoTactileSensorCfg ready to add to InteractiveSceneCfg

    Example:
        palm_sensor_cfg = create_tactile_sensor_cfg(
            hand_cfg, sensor_cfg, "palm"
            )
    """

    # Get sensor parameters
    params = sensor_cfg.get_sensor_parameters(location = location)

    # Get USD link name for this location
    link_name = hand_cfg.get_all_sensor_locations()[location]

    # Build sensor prim path
    # Format: {ENV_REGEX_NS}/Robot/{link_name}/tactile_sensor
    sensor_prim_path = f"{prim_path_base}/{link_name}/{sensor_cfg.sensor_prim_name}"

    # Build camera prim path (if only camera is enabled)
    camera_cfg = None
    if params["enable_camera_tactile"]:
        # Camera is under elastomer link
        # E.G. right_pallm_force_sensor/elastomer_right_palm_tip/tactile_sensor_camera
        elastomer_name = hand_cfg.get_physics_material_path(location = location)
        camera_prim_path = f"{prim_path_base}/{link_name}/{elastomer_name}/{hand_cfg.camera_name_convention}"

        camera_cfg = TiledCameraCfg(
            prim_path = camera_prim_path,
            height = params["render_cfg"].image_height,
            width = params["render_cfg"].image_width,
            data_types = ["distance_to_image_plane"],
            spawn = None, # Camera should exist in USD
        )

    # Create VisuoTactileSensorCfg
    tactile_sensor_cfg = VisuoTactileSensorCfg(
        prim_path = sensor_prim_path,
        
        # Render Configuration
        render_cfg = params["render_cfg"],

        # Enable flags 
        enable_camera_tactile = params["enable_camera_tactile"],
        enable_force_field = params["enable_force_field"],

        # Tactile array
        tactile_array_size = params["tactile_array_size"],
        tactile_margin = params["tactile_margin"],

        # Camera configuration
        camera_cfg = camera_cfg,

        # Force field parameters (if enabled)
        contact_object_prim_path_expr = params["contact_object_prim_path_expr"],
        normal_contact_stiffness = params["normal_contact_stiffness"],
        friction_coefficient = params["friction_coefficient"],
        tangential_stiffness = params["tangential_stiffness"],

        # Update settings
        update_period = params["update_period"],
        history_length = params["history_length"]
    )

    return tactile_sensor_cfg

def create_all_tactile_sensors(
        hand_cfg: HandConfig,
        sensor_cfg: SensorConfig,
        prim_path_base: str = "{ENV_REGEX_NS}/hand",
) -> Dict[str, VisuoTactileSensorCfg]:
    """
    Creates VisuoTactileSensorCfg for all enabled sensor locations.

    Args:
        hand_cfg: HandConfig instance
        sensor_cfg: SensorConfig instance
        prim_path_base: Base prim path for the hand
    
    Return:
        Dictionary mapping location to VisuoTactileSensorCfg

    Example:
        sensors = create_all_tactile_sensors(hand_cfg, sensor_cfg)
        # Returns: {
        #       "palm": VisuoTactileSensorCfg(...),
        #       "index_proximal": VisuoTactileSensorCfg(...),
        #       ...}
    """

    sensors = {}

    # Get all enabled tactile sensors from hand
    enabled_locations = sensor_cfg.get_all_enabled_locations(hand_config = hand_cfg)

    # Create sensor config for each enabled location
    for location in enabled_locations:
        sensors["location"] = create_tactile_sensor_cfg(hand_cfg = hand_cfg, 
                                                        sensor_cfg = sensor_cfg, 
                                                        location = location,
                                                        prim_path_base = prim_path_base)
    return sensors