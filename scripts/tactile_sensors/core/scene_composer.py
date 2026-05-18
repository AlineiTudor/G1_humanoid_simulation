"""
Scene Composer for Multi-Hand Tactile Scenes
"""

import math
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
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
_GRASP_BUILD_CONFIGS = {}



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


# ======================================================================
# Grasp Scene  extends TactileHandSceneCfg with table + object + camera
# ======================================================================

def _look_at_quat(eye, target):
    """Compute quaternion (w,x,y,z) for camera looking from eye toward target."""
    forward = [target[i] - eye[i] for i in range(3)]
    norm = math.sqrt(sum(f * f for f in forward))
    forward = [f / norm for f in forward]

    # Camera convention: -Z is forward, Y is up
    # Compute rotation from -Z axis to forward direction
    world_up = [0.0, 0.0, 1.0]

    # Right = forward x up
    right = [
        forward[1] * world_up[2] - forward[2] * world_up[1],
        forward[2] * world_up[0] - forward[0] * world_up[2],
        forward[0] * world_up[1] - forward[1] * world_up[0],
    ]
    r_norm = math.sqrt(sum(r * r for r in right))
    if r_norm < 1e-6:
        right = [1.0, 0.0, 0.0]
    else:
        right = [r / r_norm for r in right]

    # Recompute up = right x forward
    up = [
        right[1] * forward[2] - right[2] * forward[1],
        right[2] * forward[0] - right[0] * forward[2],
        right[0] * forward[1] - right[1] * forward[0],
    ]

    # Build rotation matrix (row-major) and convert to quaternion
    # For ROS convention camera: X=right, Y=down, Z=forward
    m00, m01, m02 = right[0], -up[0], forward[0]
    m10, m11, m12 = right[1], -up[1], forward[1]
    m20, m21, m22 = right[2], -up[2], forward[2]

    tr = m00 + m11 + m22
    if tr > 0:
        s = 0.5 / math.sqrt(tr + 1.0)
        w = 0.25 / s
        x = (m21 - m12) * s
        y = (m02 - m20) * s
        z = (m10 - m01) * s
    elif m00 > m11 and m00 > m22:
        s = 2.0 * math.sqrt(1.0 + m00 - m11 - m22)
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = 2.0 * math.sqrt(1.0 + m11 - m00 - m22)
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = 2.0 * math.sqrt(1.0 + m22 - m00 - m11)
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s

    return (w, x, y, z)


@configclass
class GraspSceneCfg(TactileHandSceneCfg):
    """Scene with robot, tactile sensors, table, graspable object, and camera."""

    def __post_init__(self):
        super().__post_init__()

        grasp_cfg = _GRASP_BUILD_CONFIGS.pop(id(self), None)
        if grasp_cfg is not None:
            self._build_grasp_scene(grasp_cfg)

    def _build_grasp_scene(self, grasp_cfg):
        """Add table, object, and camera to the scene."""
        from isaaclab.sensors import CameraCfg

        # --- Table (static kinematic body) ---
        table = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            spawn=sim_utils.CuboidCfg(
                size=grasp_cfg.table_size,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.6, 0.4, 0.2),
                ),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=grasp_cfg.table_pos,
            ),
        )
        setattr(self, "table", table)

        # --- Graspable object ---
        from ..assets.contact_objects import CUBE_CFG, NUT_CFG

        if grasp_cfg.object_type == "cube":
            obj_cfg = CUBE_CFG.copy()
        else:
            obj_cfg = NUT_CFG.copy()

        obj_cfg.prim_path = "{ENV_REGEX_NS}/grasp_object"
        obj_cfg.init_state = RigidObjectCfg.InitialStateCfg(
            pos=grasp_cfg.object_pos,
            rot=(1.0, 0.0, 0.0, 0.0),
        )
        setattr(self, "grasp_object", obj_cfg)

        # --- Camera ---
        if grasp_cfg.camera_enabled:
            cam_rot = _look_at_quat(grasp_cfg.camera_pos, grasp_cfg.camera_target)
            camera = CameraCfg(
                prim_path="{ENV_REGEX_NS}/scene_camera",
                update_period=0.0,
                height=grasp_cfg.camera_height,
                width=grasp_cfg.camera_width,
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=12.0,
                    focus_distance=400.0,
                    horizontal_aperture=20.0,
                    clipping_range=(0.1, 100.0),
                ),
                offset=CameraCfg.OffsetCfg(
                    pos=grasp_cfg.camera_pos,
                    rot=cam_rot,
                    convention="world",
                ),
            )
            setattr(self, "scene_camera", camera)


def create_grasp_scene(
    sim_cfg: SimulationConfig,
    sensor_cfg: SensorConfig,
    grasp_cfg,
) -> GraspSceneCfg:
    """Creates a GraspSceneCfg with table, object, camera, robot, and sensors.

    Args:
        sim_cfg: Simulation configuration.
        sensor_cfg: Sensor configuration.
        grasp_cfg: GraspDataCollectionConfig with table/object/camera settings.

    Returns:
        GraspSceneCfg ready to pass to InteractiveScene.
    """
    scene_cfg = GraspSceneCfg.__new__(GraspSceneCfg)

    _SCENE_BUILD_CONFIGS[id(scene_cfg)] = (sim_cfg, sensor_cfg)
    _GRASP_BUILD_CONFIGS[id(scene_cfg)] = grasp_cfg

    GraspSceneCfg.__init__(
        scene_cfg,
        num_envs=sim_cfg.num_envs,
        env_spacing=sim_cfg.env_spacing,
    )

    return scene_cfg
