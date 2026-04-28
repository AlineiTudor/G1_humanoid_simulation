#!/usr/bin/env python3
"""Test dual hands in single environment."""

import argparse
import sys
from pathlib import Path

# Add the package parent directory to Python path
# This works regardless of where the script is run from
SCRIPT_DIR = Path(__file__).resolve().parent  # .../RH56E2_tactile_proj/scripts
PROJECT_ROOT = SCRIPT_DIR.parent  # .../RH56E2_tactile_proj
PACKAGE_PARENT = PROJECT_ROOT.parent  # .../IsaacLab_code

# Add package parent to sys.path if not already there
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Now import Isaac Lab modules
from tactile_sensors.config import SimulationConfig, SensorConfig
from tactile_sensors.core import create_tactile_scene
from isaaclab.scene import InteractiveScene
import isaaclab.sim as sim_utils

def main():
    # Config: 1 environment with 2 hands
    sim_cfg = SimulationConfig(
        spawn_mode = "g1_robot",
        hand_types=("left","right"),
        num_envs=1,
        robot_usd_path="/workspace/tudor_unitree_isaaclab/isaaclab_tactile_proj/rh56e2_hands_tactile_sensors/gelsight_tactile_sensors/scripts/tactile_sensors/resources/g1_with_rh56e2_hands/G1/g1_rh56e2.usd",
    )

    sensor_cfg = SensorConfig(palm_enabled=True,
                              index_enabled=True,
                              middle_enabled=True,
                              thumb_enabled=True,
                              little_enabled=True,
                              ring_enabled=True
                              )

    # Create simulation context (MUST be created before InteractiveScene)
    sim_context = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(
            dt=sim_cfg.physics_dt,
            render_interval=int(sim_cfg.rendering_dt / sim_cfg.physics_dt) if sim_cfg.rendering_dt > 0 else 1,
            device=sim_cfg.device,
            gravity=(0.0, 0.0, -9.81),
            physx=sim_utils.PhysxCfg(
                solver_type=1,  # TGS solver
                enable_stabilization=True,
                gpu_max_rigid_contact_count=2**23,
                gpu_max_rigid_patch_count=2**21,
                gpu_found_lost_pairs_capacity=2**21,
                gpu_found_lost_aggregate_pairs_capacity=2**20,
                gpu_total_aggregate_pairs_capacity=2**20,
                gpu_collision_stack_size=sim_cfg.gpu_collision_stack_size,
            ),
        )
    )

    # Set main camera view
    sim_context.set_camera_view(eye=sim_cfg.camera_eye, target=sim_cfg.camera_target)

    # # =============================================
    # # DEBUG: Pre-scene diagnostics
    # # =============================================
    # import os
    # print(f"\n{'='*60}")
    # print(f"PRE-SCENE DIAGNOSTICS")
    # print(f"{'='*60}")
    # print(f"  spawn_mode:     {sim_cfg.spawn_mode}")
    # print(f"  hand_types:     {sim_cfg.hand_types}")
    # print(f"  num_envs:       {sim_cfg.num_envs}")
    # if hasattr(sim_cfg, 'robot_usd_path'):
    #     print(f"  robot_usd_path: {sim_cfg.robot_usd_path}")
    #     print(f"  file exists:    {os.path.exists(sim_cfg.robot_usd_path)}")
    # if hasattr(sim_cfg, 'robot_prim_name'):
    #     print(f"  robot_prim_name: {sim_cfg.robot_prim_name}")
    # print(f"{'='*60}\n")

    # # DEBUG: Load the robot USD directly to verify it works
    # from tactile_sensors.utils.debug_prims import (
    #     print_stage_hierarchy, find_articulation_roots, load_usd_directly
    # )
    # if hasattr(sim_cfg, 'robot_usd_path'):
    #     load_usd_directly(sim_cfg.robot_usd_path)

    # Create scene
    scene_cfg = create_tactile_scene(sim_cfg, sensor_cfg)
    scene = InteractiveScene(scene_cfg)

    # # DEBUG: Print prim hierarchy after scene creation, before reset
    # print_stage_hierarchy("/World/envs/env_0", max_depth=5)
    # find_articulation_roots("/World/envs/env_0")

    # NOTE: Hand ArticulationRootAPI removal is now done at USD level
    # in _fix_usd_before_load(), called from create_robot_articulation_cfg().

    # DEBUG: Verify rigid body properties are applied to hand bodies
    import omni.usd
    from pxr import Usd, UsdPhysics
    stage = omni.usd.get_context().get_stage()
    for body_name in ["pelvis", "left_wrist_yaw_link", "l_base_link",
                      "left_index_1", "left_index_force_sensor_1"]:
        for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/envs/env_0/Robot")):
            if prim.GetName() == body_name and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                max_lin = prim.GetAttribute("physxRigidBody:maxLinearVelocity").Get()
                max_ang = prim.GetAttribute("physxRigidBody:maxAngularVelocity").Get()
                max_dep = prim.GetAttribute("physxRigidBody:maxDepenetrationVelocity").Get()
                max_imp = prim.GetAttribute("physxRigidBody:maxContactImpulse").Get()
                lin_damp = prim.GetAttribute("physxRigidBody:linearDamping").Get()
                print(f"  {body_name}: maxLinVel={max_lin}, maxAngVel={max_ang}, "
                      f"maxDepen={max_dep}, maxImpulse={max_imp}, linDamp={lin_damp}")
                break

    # Simulation loop
    sim_context.reset()
    scene.reset()

    print("\nStarting simulation loop...")
    step_count = 0

    robot = scene["robot"]

    # Diagnostics: understand what PhysX sees
    print(f"\n{'='*60}")
    print(f"ARTICULATION DIAGNOSTICS")
    print(f"{'='*60}")
    print(f"  num_bodies:  {robot.num_bodies}")
    print(f"  num_joints:  {robot.num_joints}")
    print(f"  joint_names: {robot.joint_names}")
    print(f"  default_joint_pos shape: {robot.data.default_joint_pos.shape}")
    print(f"  default_joint_pos: {robot.data.default_joint_pos}")
    print(f"  current joint_pos: {robot.data.joint_pos}")
    print(f"  body_names:  {robot.body_names}")
    print(f"{'='*60}\n")

    # Investigate the joints scope - why are body joints missing?
    import omni.usd
    from pxr import Usd, UsdPhysics
    stage = omni.usd.get_context().get_stage()
    joints_scope = stage.GetPrimAtPath("/World/envs/env_0/Robot/joints")
    if joints_scope.IsValid():
        print(f"\n--- Joints under /Robot/joints ---")
        for child in joints_scope.GetChildren():
            is_joint = child.IsA(UsdPhysics.Joint)
            joint_type = child.GetTypeName()
            # Check body0/body1 relationships
            body0_rel = child.GetRelationship("physics:body0")
            body1_rel = child.GetRelationship("physics:body1")
            body0_targets = body0_rel.GetTargets() if body0_rel else []
            body1_targets = body1_rel.GetTargets() if body1_rel else []
            print(f"  {child.GetName()} [{joint_type}] isJoint={is_joint}")
            print(f"    body0: {body0_targets}")
            print(f"    body1: {body1_targets}")
        print()
    else:
        print("  WARNING: /Robot/joints scope not found!")
        print("  Searching for all joint prims under Robot...")
        robot_prim = stage.GetPrimAtPath("/World/envs/env_0/Robot")
        joint_count = 0
        for prim in Usd.PrimRange(robot_prim):
            if prim.IsA(UsdPhysics.Joint):
                print(f"    {prim.GetPath()} [{prim.GetTypeName()}]")
                joint_count += 1
        print(f"  Total joints found: {joint_count}\n")

    # =============================================
    # DEBUG: Diagnose why hands appear at origin
    # =============================================
    print(f"\n{'='*60}")
    print(f"HAND PLACEMENT DIAGNOSTICS")
    print(f"{'='*60}")

    # 1) Check if hand bodies are in the articulation
    hand_bodies = [b for b in robot.body_names if "hand" in b.lower() or "finger" in b.lower() or "thumb" in b.lower()]
    print(f"\n  [1] Hand-related bodies in articulation ({len(hand_bodies)}):")
    for b in hand_bodies:
        print(f"      {b}")
    if not hand_bodies:
        print(f"      NONE  hands are NOT part of the articulation!")
        print(f"      All body_names: {robot.body_names}")

    hand_joints = [j for j in robot.joint_names if "hand" in j.lower() or "finger" in j.lower() or "thumb" in j.lower()]
    print(f"\n  [2] Hand-related joints in articulation ({len(hand_joints)}):")
    for j in hand_joints:
        print(f"      {j}")
    if not hand_joints:
        print(f"      NONE  hand joints not discovered!")

    # 2) Check FixedJoint body0/body1 targets
    from pxr import UsdGeom
    robot_path = "/World/envs/env_0/Robot"
    print(f"\n  [3] FixedJoint diagnostics:")
    for side in ["left", "right"]:
        hand_path = f"{robot_path}/{side}_hand"
        hand_prim = stage.GetPrimAtPath(hand_path)
        if not hand_prim.IsValid():
            print(f"      {side}_hand: prim NOT FOUND at {hand_path}")
            continue
        print(f"      {side}_hand: prim exists at {hand_path}")

        # Check transform on hand container
        xformable = UsdGeom.Xformable(hand_prim)
        xform_ops = xformable.GetOrderedXformOps()
        print(f"        xform ops: {[str(op.GetOpName()) for op in xform_ops]}")
        xform_cache = UsdGeom.XformCache()
        local_xform = xform_cache.GetLocalToWorldTransform(hand_prim)
        print(f"        world transform: {local_xform}")

        # Look for FixedJoint (root_joint) under hand
        for child in Usd.PrimRange(hand_prim):
            if child.IsA(UsdPhysics.Joint):
                body0_rel = child.GetRelationship("physics:body0")
                body1_rel = child.GetRelationship("physics:body1")
                body0_targets = body0_rel.GetTargets() if body0_rel else []
                body1_targets = body1_rel.GetTargets() if body1_rel else []
                print(f"        Joint: {child.GetPath()} [{child.GetTypeName()}]")
                print(f"          body0 targets: {body0_targets}")
                print(f"          body1 targets: {body1_targets}")
                # Check if targets resolve
                for t in body0_targets:
                    resolved = stage.GetPrimAtPath(t)
                    print(f"          body0 '{t}' resolves: {resolved.IsValid() if resolved else False}")
                for t in body1_targets:
                    resolved = stage.GetPrimAtPath(t)
                    print(f"          body1 '{t}' resolves: {resolved.IsValid() if resolved else False}")

    # 3) Check for ResetXformStack on hand rigid body prims
    print(f"\n  [4] ResetXformStack check on hand prims:")
    for side in ["left", "right"]:
        hand_path = f"{robot_path}/{side}_hand"
        hand_prim = stage.GetPrimAtPath(hand_path)
        if not hand_prim.IsValid():
            continue
        reset_count = 0
        for prim in Usd.PrimRange(hand_prim):
            xformable = UsdGeom.Xformable(prim)
            if xformable:
                ops = xformable.GetOrderedXformOps()
                has_reset = xformable.GetResetXformStack()
                if has_reset:
                    has_rigid = prim.HasAPI(UsdPhysics.RigidBodyAPI)
                    print(f"      RESET_XFORM: {prim.GetPath()} (RigidBody={has_rigid})")
                    reset_count += 1
        if reset_count == 0:
            print(f"      {side}_hand: no ResetXformStack found (good)")

    print(f"{'='*60}\n")

    joint_targets = robot.data.default_joint_pos.clone()

    while simulation_app.is_running():
        # Slowly wave all joints using a sine wave
        # import math
        # joint_targets[:, :] = robot.data.default_joint_pos + 3.0 * math.sin(step_count * 0.02)
        # robot.set_joint_position_target(joint_targets)

        # Write data to simulation
        scene.write_data_to_sim()

        # Step simulation
        sim_context.step()

        # Update scene
        scene.update(sim_context.get_physics_dt())

        # Print status every 200 steps
        step_count += 1
        if step_count % 200 == 0:
            print(f"Step {step_count} | "
                  f"target[0:3]={joint_targets[0, :3].tolist()} | "
                  f"actual[0:3]={robot.data.joint_pos[0, :3].tolist()}")

if __name__ == "__main__":
    main()
    simulation_app.close()
