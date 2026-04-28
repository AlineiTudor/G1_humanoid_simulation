"""
Hand builder for complete hand assembly
"""

from typing import Dict, Any, Tuple, List
from isaaclab.assets import ArticulationCfg

from ..config.hand_config import HandConfig
from ..config.sensor_config import SensorConfig
from ..assets.rh56e2_hand import create_hand_articulation_cfg
from .sensor_factory import SensorFactory

class HandBuilder:
    """
    Builds complete hand configurations.

    Usage: 
        builder = HandBuilder(hand_cfg, sensor_cfg)
        hand_bundle = builder.build_hand("right_hand", init_pos=(0.0, 0.0, 0.4))
    """
    def __init__(self, hand_cfg: HandConfig, sensor_cfg: SensorConfig):
        self.hand_cfg = hand_cfg
        self.sensor_cfg = sensor_cfg
        self.sensor_factory = SensorFactory(hand_cfg, sensor_cfg)
    
    def build_hand(
            self,
            hand_name: str,
            prim_path: str = "{ENV_REGEX_NS}/hand",
            init_pos: Tuple[float, float, float] = (0.0, 0.0, 0.4),
            init_rot: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    ) -> Dict[str, Any]:
        """
        Builds a complete hand configuration.

        Returns:
            {
            "articulation": ArticulationCfg(...),
            "sensors": {
                "hand_name_palm": VisuoTactileSensorCfg(...),
                "hand_name_index_proximal": VisuoTactileSensorCfg(...),
                ...
                }
            }
        """
        # Create a hand articulation
        hand_articulation = create_hand_articulation_cfg(self.hand_cfg, prim_path, init_pos, init_rot)

        # Create sensors for this hand
        sensors = self.sensor_factory.create_sensors_for_hand(
            hand_name = hand_name,
            prim_path_base = prim_path,
        )

        return{
            "articulation": hand_articulation,
            "sensors": sensors
        }
    
    def build_multiple_hands(
            self,
            hand_names: List[str],
            positions: List[Tuple[float, float, float]],
            rotations: List[Tuple[float, float, float, float]],
            prim_path_pattern: str = "{ENV_REGEX_NS}/{hand_name}",
    ) -> Dict[str, Any]:
        """
        Builds multiple hands (of the same type: two right hands with same hand config) with different positions and rotations.

        Return: 
            {
                "articulations": {
                    "right_hand_1": ArticulationCfg(...),
                    "right_hand_2": ArticulationCfg(...),
                },
                "sensors": {
                    "right_hand_1_palm": VisuoTactileSensorCfg(...),
                    "left_hand_1_palm": VisuoTactileSensorCfg(...),
                }
            }
        """

        # Validate input
        if len(positions)!=len(rotations) or len(hand_names)!=len(positions):
            raise ValueError(f"Number of positions ({len(positions)}), rotations ({len(rotations)}) and hand_hanmes ({len(hand_names)}) must be the same.")

        hand_articulations = {}
        all_sensors={}

        for hand_name, pos, rot in zip(hand_names, positions, rotations):
            prim_path = prim_path_pattern.format(hand_name = hand_name)

            hand_bundle = self.build_hand(hand_name, prim_path, pos, rot)
            hand_articulations[hand_name] = hand_bundle["articulation"]
            all_sensors.update(hand_bundle["sensor"])
        
        return {
            "articulations": hand_articulations,
            "sensors": all_sensors
            }