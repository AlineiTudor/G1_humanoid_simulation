"""
RH56E2 Assets
======================
Create ArticulationCfg for the RH56E2 hand with compliant contact properties

Usage:
    - Takes HandConfig as input
    - Returns ArticulationCfg ready for InteractiveSceneCfg
    - Configures compliant contact for elastomer links
    - Sets up actuators, physics properties
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from typing import Tuple 

from ..config.hand_config import HandConfig


def _get_elastomer_paths(hand_cfg: HandConfig) -> list[str]:
    """
    Generate list of relative paths to collision mesh prims for compliant contact.

    These paths point to the tactile_pad meshes which have CollisionAPI/MeshCollisionAPI.
    We bind materials to the PARENT force sensor links, and the material properties will
    PROPAGATE to the child collision meshes.

    Args:
        hand_cfg: HandConfig instance

    Returns:
        List of relative paths like:
        ["right_palm_force_sensor",
         "right_index_force_sensor_1",
         ...]

    Example USD structure:
        /World/envs/env_0/right_hand/                   ? Articulation root
          +-- right_palm_force_sensor/                  ? Bind material HERE
          +-- right_index_force_sensor_1/
          ...
    """
    collision_paths = []

    # Get all sensor locations from hand config
    all_locations = hand_cfg.get_all_sensor_locations()

    # For each location, build the path to its tactile_pad collision mesh
    for location, link_name in all_locations.items():
        # Build relative path: force_sensor_link/tactile_sensor/tactile_pad
        # This is the actual collision mesh that needs the compliant material
        # Example: "right_palm_force_sensor/tactile_sensor/tactile_pad"
        #relative_path = f"{link_name}/tactile_sensor/tactile_pad"

        collision_paths.append(link_name)

    return collision_paths


def create_hand_articulation_cfg(
        hand_cfg: HandConfig,
        prim_path: str = "{ENV_REGEX_NS}/hand",
        init_pos: Tuple[float, float, float] = (0.0, 0.0, 0.4),
        init_rot: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
) -> ArticulationCfg:
    """
    Creates ArticulationCfg for RH56E2 hand with compliant contact.

    Args:
        hand_cfg: HandConfig instance
        prim_path: USD prim path (for multi-environment use {ENV_REGEX_NS})
        init_pos: Initial position (x, y, z)
        init_rot: Initial rotation quaternion (w, x, y, z)

    Return:
        ArticulationCfg ready to add to InteractiveSceneCfg
    """

    # Build list of all elastomer prim paths for compliant contact
    # These paths are RELATIVE to the articulation root (the hand prim)
    elastomer_paths = _get_elastomer_paths(hand_cfg)

    # Create USD spawner with compliant contact
    usd_spawn_cfg = sim_utils.UsdFileWithCompliantContactCfg(
        usd_path = hand_cfg.usd_full_path,

        # Compliant contact for elastomer links
        compliant_contact_stiffness = hand_cfg.compliant_contact_stiffness,
        compliant_contact_damping = hand_cfg.compliant_contact_damping,

        # Apply compliant contact to all elastomer links
        # physics_material_prim_path accepts list[str] - perfect for multiple elastomers!
        # Paths are relative to articulation root
        physics_material_prim_path = elastomer_paths,

        # Rigid body properties
        rigid_props = sim_utils.RigidBodyPropertiesCfg(
            disable_gravity = hand_cfg.disable_gravity,
            max_depenetration_velocity = hand_cfg.max_depenetration_velocity,
        ),

        # Collision properties
        collision_props = sim_utils.CollisionPropertiesCfg(
            contact_offset = hand_cfg.contact_offset,
            rest_offset = hand_cfg.rest_offset,
        ),

        # Articulation properties (solver settings)
        articulation_props = sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions = False,
            solver_position_iteration_count = hand_cfg.solver_position_iteration_count,
            solver_velocity_iteration_count = hand_cfg.solver_velocity_iteration_count,
        ),
    )

    # Create ArticulationCfg
    hand_articulation_cfg = ArticulationCfg(
        prim_path = prim_path,
        spawn = usd_spawn_cfg,

        # Initial state
        init_state = ArticulationCfg.InitialStateCfg(
            pos = init_pos,
            rot = init_rot
        ),

        # Actuators (free motion for now - zero stiffness/damping)
        actuators = {
            "hand_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness = 0.0,
                damping = 0.0
            ),
        },
    )

    return hand_articulation_cfg

# Pre-configured right-hand instance for quick usage
RH56E2_R_HAND_CFG = create_hand_articulation_cfg(HandConfig())