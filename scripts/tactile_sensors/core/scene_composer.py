"""
Scene Composer for Multi-Hand Tactile Scenes
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from typing import List, Tuple

from ..config.hand_config import HandConfig
from ..config.sensor_config import SensorConfig
from ..config.simulation_config import SimulationConfig
from .hand_builder import HandBuilder

# Module-level storage for configs during scene construction
# This is a workaround for @configclass not allowing non-field attributes
_SCENE_BUILD_CONFIGS = {}



@configclass
class TactileHandSceneCfg(InteractiveSceneCfg):
    """
    Scene configuration with RH56E2 hands and tactile sensors.

    Dynamically creates hand and sensor configs based on SimulationConfig
    """

    # =======================================================
    # Static Scene Elements
    # =======================================================
    ground = AssetBaseCfg(
        prim_path = "/World/defaultGroundPlane",
        spawn = sim_utils.GroundPlaneCfg(),
    )

    dome_light = AssetBaseCfg(
        prim_path = "/World/Light",
        spawn = sim_utils.DomeLightCfg(intensity = 3000.0, color = (0.75, 0.75, 0.75)),
    )

    # =======================================================
    # Dynamic Elements (Hand & Sensors)
    # =======================================================
    def __post_init__(self):
        """
        Dynamically add hands and sensors to scene or spawn robot with hands depending on SimulationConfig spawn_mode.
        """
        # Get configs from module-level storage (workaround for @configclass limitations)
        # This allows us to pass configs without making them part of the config schema
        scene_id = id(self)
        if scene_id not in _SCENE_BUILD_CONFIGS:
            raise RuntimeError(
                "Scene configs not found! Use create_tactile_scene() factory function "
                "instead of direct instantiation."
            )

        sim_cfg, sensor_cfg = _SCENE_BUILD_CONFIGS.pop(scene_id)

        if sim_cfg.spawn_mode == "standalone":
            self._build_standalone_hands(sim_cfg, sensor_cfg)
        else:
            self._build_robot_with_hands(sim_cfg, sensor_cfg)

    def _build_robot_with_hands(self, sim_cfg, sensor_cfg):
        """ Spawn G1 robot which already has hands referenced in USD. """
        from ..assets.g1_robot import create_robot_articulation_cfg
        from ..core.sensor_factory import SensorFactory

        # One articulation for the entire robot
        robot_prim_path = f"{{ENV_REGEX_NS}}/{sim_cfg.robot_prim_name}"
        robot_cfg = create_robot_articulation_cfg(
            robot_usd_path = sim_cfg.robot_usd_path,
            prim_path = robot_prim_path,
            source_prim_path = sim_cfg.robot_source_prim_path,
            fix_root_link = sim_cfg.fix_root_link,
        )
        setattr(self, "robot", robot_cfg)

        # Adding sensors to each hand, using the correct mount path
        for hand_type in sim_cfg.hand_types:
            hand_cfg = HandConfig(hand_side=hand_type)
            sensor_factory = SensorFactory(hand_cfg, sensor_cfg)

            # Sensor links are under the hand mount point, not directly under Robot
            # e.g., Robot/left_wrist_yaw_link/left_hand_palm_link/{link_name}/...
            hand_mount = sim_cfg.robot_hand_mount_paths.get(hand_type, "")
            hand_base_path = f"{robot_prim_path}/{hand_mount}" if hand_mount else robot_prim_path

            sensors = sensor_factory.create_sensors_for_hand(
                hand_name=f"{hand_type}_hand",
                prim_path_base = hand_base_path
            )

            for sensor_name, sensor_cfg_item in sensors.items():
                setattr(self, sensor_name, sensor_cfg_item)

    def _build_standalone_hands(self, sim_cfg, sensor_cfg):
        """ Spawn hands as independent articulations """
        # Create hand builders (one for left, one for right)
        hand_builders = {
            "left": HandBuilder(HandConfig(hand_side="left"), sensor_cfg=sensor_cfg),
            "right": HandBuilder(HandConfig(hand_side="right"), sensor_cfg=sensor_cfg)
        }

        # Get hand positions from simulation config
        positions = sim_cfg.get_hand_positions()
        # Set initial rotation for each hand
        rotations = [(1.0, 0.0, 0.0, 0.0)] * len(positions)

        # Adding hands in scene
        for i, hand_type in enumerate(sim_cfg.hand_types):
            hand_name = f"{hand_type}_hand"
            builder = hand_builders[hand_type]

            # Build hand (articulation + sensors)
            # For standalone hands: spawn directly under environment
            # For robot-attached hands: would use {ENV_REGEX_NS}/Robot/{arm_link}/{hand_name}
            hand_bundle = builder.build_hand(
                hand_name = hand_name,
                prim_path = f"{{ENV_REGEX_NS}}/{hand_name}",
                init_pos = positions[i],
                init_rot = rotations[i]
            )

            # Add articulation to scene
            setattr(self, hand_name, hand_bundle["articulation"])

            # Add Sensors to scene
            for sensor_name, sensor_cfg in hand_bundle["sensors"].items():
                setattr(self, sensor_name, sensor_cfg)


# Factory function for scene creation
def create_tactile_scene(
        sim_cfg: SimulationConfig,
        sensor_cfg: SensorConfig,
) -> TactileHandSceneCfg:
    """
    Creates a TactileHandSceneCfg with given configurations.

    Args:
        sim_cfg: SimulationConfig
        sensor_cfg: SensorConfig

    Returns:
        TactileHandSceneCfg ready to pass to InteractiveScene

    Usage:
        scene_cfg = create_tactile_scene(sim_cfg, sensor_cfg)
        scene = InteractiveScene(scene_cfg)

    Example - Single humanoid with 2 hands:
        sim_cfg = SimulationConfig(
            num_envs=1,  # Single robot
            hand_types=("left", "right"),  # 2 hands on same robot
        )

    Example - Parallel training with 4 humanoids:
        sim_cfg = SimulationConfig(
            num_envs=4,  # 4 parallel robots
            hand_types=("left", "right"),  # Each robot has 2 hands
        )
    """
    # Store configs in module-level dict BEFORE creating scene object
    # This is a workaround for @configclass not allowing non-schema attributes
    # The scene object will retrieve them in __post_init__ using its id()

    # Create scene config (this will trigger __post_init__)
    scene_cfg = TactileHandSceneCfg.__new__(TactileHandSceneCfg)

    # Store configs using the actual scene object's id
    _SCENE_BUILD_CONFIGS[id(scene_cfg)] = (sim_cfg, sensor_cfg)

    # Initialize the scene config (this calls __post_init__)
    TactileHandSceneCfg.__init__(
        scene_cfg,
        num_envs = sim_cfg.num_envs,
        env_spacing = sim_cfg.env_spacing,
    )

    return scene_cfg

