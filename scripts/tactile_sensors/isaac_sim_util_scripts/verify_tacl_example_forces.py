"""
Verify that the tacl_sensor.py example produces non-zero force readings.

This is a standalone script that runs the same scene as tacl_sensor.py
(GelSight finger + nut contact object) with force field enabled,
pushes the nut into the finger, and prints force values each step.

Run with:
    python verify_tacl_example_forces.py --enable_cameras

If forces are non-zero here but zero in the G1 setup, the issue is
in the G1 configuration (not in the sensor implementation itself).
"""

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Verify tacl_sensor example forces")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from isaaclab_contrib.sensors.tacsl_sensor import VisuoTactileSensorCfg
from isaaclab_assets.sensors import GELSIGHT_R15_CFG

from pxr import Usd, UsdGeom, UsdPhysics


@configclass
class VerifySceneCfg(InteractiveSceneCfg):
    """Minimal scene: GelSight finger + nut + force field sensor."""

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileWithCompliantContactCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/TacSL/gelsight_r15_finger/gelsight_r15_finger.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            compliant_contact_stiffness=100.0,
            compliant_contact_damping=10.0,
            physics_material_prim_path="elastomer",
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=1,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.001, rest_offset=-0.0005
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),
            rot=(math.sqrt(2) / 2, -math.sqrt(2) / 2, 0.0, 0.0),
            joint_pos={},
            joint_vel={},
        ),
        actuators={},
    )

    contact_object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/contact_object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Factory/factory_nut_m16.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=1,
                max_angular_velocity=180.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.005, rest_offset=0
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0 + 0.06776, 0.498),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    tactile_sensor = VisuoTactileSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/elastomer/tactile_sensor",
        history_length=0,
        debug_vis=False,
        render_cfg=GELSIGHT_R15_CFG,
        enable_camera_tactile=False,
        enable_force_field=True,
        tactile_array_size=(20, 25),
        tactile_margin=0.003,
        contact_object_prim_path_expr="{ENV_REGEX_NS}/contact_object",
        normal_contact_stiffness=1.0,
        friction_coefficient=2.0,
        tangential_stiffness=0.1,
        trimesh_vis_tactile_points=False,
    )


def run_diagnostics(stage):
    """Print diagnostic info about the scene before stepping."""
    print("\n" + "=" * 60)
    print("SCENE DIAGNOSTICS")
    print("=" * 60)

    # Check elastomer structure
    elast_path = "/World/envs/env_0/Robot/elastomer"
    elast_prim = stage.GetPrimAtPath(elast_path)
    if elast_prim.IsValid():
        print(f"\n[Elastomer] {elast_path}")
        print(f"  RigidBodyAPI: {elast_prim.HasAPI(UsdPhysics.RigidBodyAPI)}")

        for prim in Usd.PrimRange(elast_prim):
            if prim.IsA(UsdGeom.Mesh):
                has_col = prim.HasAPI(UsdPhysics.CollisionAPI)
                usd_mesh = UsdGeom.Mesh(prim)
                pts = np.array(usd_mesh.GetPointsAttr().Get())
                label = "COLLISION" if has_col else "VISUAL"
                print(f"  [{label}] {prim.GetPath()} ({len(pts)} verts)")

                # Check intermediate transform
                mesh_world = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
                    Usd.TimeCode.Default()
                )
                rb_world = UsdGeom.Xformable(elast_prim).ComputeLocalToWorldTransform(
                    Usd.TimeCode.Default()
                )
                rel = mesh_world * rb_world.GetInverse()
                translate = rel.ExtractTranslation()
                trans_mag = sum(t**2 for t in translate) ** 0.5
                if trans_mag < 1e-5:
                    print(f"    Mesh-to-RB transform: IDENTITY (OK)")
                else:
                    print(f"    Mesh-to-RB transform: OFFSET ({translate[0]:.6f}, {translate[1]:.6f}, {translate[2]:.6f})")

    # Check contact object
    co_path = "/World/envs/env_0/contact_object"
    co_prim = stage.GetPrimAtPath(co_path)
    if co_prim.IsValid():
        print(f"\n[Contact Object] {co_path}")
        for prim in Usd.PrimRange(co_prim):
            if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                approx = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
                print(f"  SDF mesh: {prim.GetPath()} (approx={approx})")

    print("=" * 60)


def main():
    sim_cfg = sim_utils.SimulationCfg(
        dt=0.005,
        device=args_cli.device,
        physx=sim_utils.PhysxCfg(gpu_collision_stack_size=2**30),
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[0.5, 0.6, 1.0], target=[-0.1, 0.1, 0.5])

    scene_cfg = VerifySceneCfg(num_envs=args_cli.num_envs, env_spacing=0.2)
    scene = InteractiveScene(scene_cfg)

    sim.reset()

    # Run diagnostics
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    run_diagnostics(stage)

    # Get initial render (required by sensor)
    scene["tactile_sensor"].get_initial_render()

    # Push nut downward onto the finger
    force_tensor = torch.zeros(scene.num_envs, 1, 3, device=sim.device)
    torque_tensor = torch.zeros(scene.num_envs, 1, 3, device=sim.device)
    force_tensor[:, 0, 2] = -1.0  # push down

    sim_dt = sim.get_physics_dt()
    max_force_seen = 0.0

    print("\n" + "=" * 60)
    print("STEPPING SIMULATION  checking force values")
    print("=" * 60)

    for step in range(200):
        # Apply force to push nut onto sensor
        if step > 10:
            scene["contact_object"].permanent_wrench_composer.set_forces_and_torques(
                force_tensor, torque_tensor
            )

        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

        # Read force data
        data = scene["tactile_sensor"].data
        if data.tactile_normal_force is not None:
            f_max = data.tactile_normal_force.max().item()
            f_mean = data.tactile_normal_force.mean().item()
            max_force_seen = max(max_force_seen, f_max)

            depth = data.penetration_depth
            d_max = depth.max().item() if depth is not None else 0.0

            if step % 10 == 0 or f_max > 0:
                print(
                    f"  step={step:4d}  "
                    f"force_max={f_max:10.6f}  "
                    f"force_mean={f_mean:10.6f}  "
                    f"depth_max={d_max:10.6f}  "
                    f"{'*** NON-ZERO ***' if f_max > 1e-6 else ''}"
                )

        if not simulation_app.is_running():
            break

    print(f"\n{'=' * 60}")
    print(f"RESULT: max force seen across all steps: {max_force_seen:.6f}")
    if max_force_seen > 1e-6:
        print("SUCCESS: tacl_sensor example produces non-zero forces!")
        print("The sensor implementation works correctly.")
        print("If your G1 setup still shows zero, the issue is in the G1 configuration.")
    else:
        print("FAILURE: forces are zero even in the example!")
        print("This suggests a sensor implementation issue or environment problem.")
    print("=" * 60)


if __name__ == "__main__":
    main()
    simulation_app.close()
