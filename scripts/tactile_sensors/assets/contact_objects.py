"""
Contact Objects for tactile sensor testing
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

# Build a simple cube
CUBE_CFG = RigidObjectCfg(
    prim_path = "{ENV_REGEX_NS}/cube",
    spawn = sim_utils.CuboidCfg(
        size = (0.05, 0.05, 0.05),
        rigid_props = sim_utils.RigidBodyPropertiesCfg(
            disable_gravity = False,
        ),
        collision_props = sim_utils.CollisionPropertiesCfg(),
        mass_props = sim_utils.MassPropertiesCfg(mass = 1.1),
        visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color = (0.8, 0.2, 0.2)),
    ),
    init_state = RigidObjectCfg.InitialStateCfg(
        pos = (0.0, 0.0, 0.5),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
)

NUT_CFG = RigidObjectCfg(
    prim_path = "{ENV_REGEX_NS}/nut",
    spawn = sim_utils.UsdFileCfg(
        usd_path = f"{ISAACLAB_NUCLEUS_DIR}/Factory/factory_nut_m16.usd",
        scale = (1.0, 1.0, 1.0),  # SDF queries use unscaled mesh; scale>1 breaks force field
        rigid_props = sim_utils.RigidBodyPropertiesCfg(
            disable_gravity = False,
            solver_position_iteration_count = 12,
            solver_velocity_iteration_count = 1,
            max_angular_velocity = 180.0,
        ),
        mass_props = sim_utils.MassPropertiesCfg(mass=0.1),
        collision_props = sim_utils.CollisionPropertiesCfg(
            contact_offset = 0.001,
            rest_offset = -0.0005,
        ),
        articulation_props = sim_utils.ArticulationRootPropertiesCfg(
            articulation_enabled = False,
        ),
    ),
    init_state = RigidObjectCfg.InitialStateCfg(
        pos = (0.0, 0.0, 0.5),
        rot = (1.0, 0.0, 0.0, 0.0),
    ),
)