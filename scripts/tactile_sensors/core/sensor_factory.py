"""
Sensor Factory for Tactile Sensor Creation
"""

from typing import Dict, List
from isaaclab_contrib.sensors.tacsl_sensor import VisuoTactileSensorCfg

from ..config.hand_config import HandConfig
from ..config.sensor_config import SensorConfig
from ..assets.tactile_sensors import create_tactile_sensor_cfg

class SensorFactory:
    """
    Factory for creating tactile sensor configurations.

    Usage:
        factory = SensorFactory(hand_cfg, sensor_cfg)
        sensors = factory.create_sensors_for_hand("right_hand")
    """

    def __init__(self, hand_cfg: HandConfig, sensor_cfg: SensorConfig):
        """
        Initialize SensorFactory.
        Args: 
            hand_cfg: HandConfig instance
            sensor_cfg: SensorConfig instance
        """
        self.hand_cfg = hand_cfg
        self.sensor_cfg = sensor_cfg
    
    def create_sensors_for_hand(
            self,
            hand_name: str,
            prim_path_base: str = "{ENV_REGEX_NS}/hand",
        ) -> Dict[str, VisuoTactileSensorCfg]:
        """
        Creates all enabled sensors for a single hand. 
        This is a bit different from assets.tactile_sensors.create_all_tactile_sensors as the dictionary key
        now containes the hand_name prefix in order to distinguish between multiple hands:
            - create_all_tactile_sensors returns {"palm": VisuoTactilesensorCfg(...), ...}
            compared to:
            - create_sensors_for_hand which returns {"right_hand_palm": VisuoTactileSensorCfg(...), ...}

        Args:
            hand_name: Name of the hand to put sensors on (e.g. "right_hand")
            prim_path_base: Prim path base for the hand 
        Returns:
            {
            "right_hand_palm": VisuoTactileSensorCfg(...),
            "right_hand_index_proximal": VisuoTactileSensorCfg(...),
            ...}
        """
        sensors = {}

        # Get enabled locations
        enabled_locations = self.sensor_cfg.get_all_enabled_locations(self.hand_cfg)
        for location in enabled_locations:
            sensor_name = f"{hand_name}_{location}"

            sensor_config = create_tactile_sensor_cfg(
                self.hand_cfg, 
                self.sensor_cfg,
                location,
                prim_path_base
            )
            
            sensors[sensor_name] = sensor_config

        return sensors
    
    def create_sensors_for_multiple_hands(
            self,
            hand_names: List[str], #e.g. ["right_hand", "left_hand"]
            prim_path_pattern: str = "{ENV_REGEX_NS}/{hand_name}"
    ) -> Dict[str, VisuoTactileSensorCfg]:
        """
        Creates sensors for multiple hands.

        Returns:
            {
            "right_hand_palm": ...,
            "right_hand_index_proximal": ...,
            "left_hand_palm": ...,
            ... 
            }
        """

        all_sensors = {}
        for hand_name in hand_names:
            # Build prim path for this hand
            prim_path_base = prim_path_pattern.format(hand_name = hand_name)

            # Create sensors for this hand
            hand_sensors = self.create_sensors_for_hand(hand_name, prim_path_base)

            # Add to combined dict
            all_sensors.update(hand_sensors)
        
        return all_sensors
    
    def get_sensor_count(self) -> int:
        """
        Returns total number of enabled sensors
        """
        enabled = self.sensor_cfg.get_all_enabled_locations(self.hand_cfg)
        return len(enabled)