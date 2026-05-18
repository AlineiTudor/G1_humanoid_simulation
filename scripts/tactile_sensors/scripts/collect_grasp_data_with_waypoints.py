#!/usr/bin/env python3
"""Scripted grasp data collection for G1 + RH56E2 tactile hands.

Runs a scripted grasp trajectory (reach ? grasp ? lift) while recording
joint states, tactile sensor data, camera images, and object pose to HDF5.

Usage:
    python collect_grasp_data.py --num_episodes 5 --output_dir grasp_data
"""

import argparse
import sys
import os
from pathlib import Path

# Path setup (same pattern as test_g1_rh56e2_spawn.py)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PACKAGE_PARENT = PROJECT_ROOT.parent

if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Scripted grasp data collection")
parser.add_argument("--num_episodes", type=int, default=5,
                    help="Number of grasp episodes to record")
parser.add_argument("--output_dir", type=str, default=None,
                    help="Output directory for HDF5 files (default: from GraspDataCollectionConfig)")
parser.add_argument("--object_type", type=str, default="cube",
                    choices=["cube", "nut"], help="Object to grasp")
parser.add_argument("--no_camera", action="store_true",
                    help="Disable camera recording")
parser.add_argument("--save_tactile_rgb", action="store_true",
                    help="Save tactile RGB images (large)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# === All Isaac Lab imports AFTER AppLauncher ===
import torch

from tactile_sensors.config import SimulationConfig, SensorConfig
from tactile_sensors.config.grasp_config import GraspDataCollectionConfig
from tactile_sensors.core.scene_composer import create_grasp_scene
from tactile_sensors.control.grasp_controller import (
    GraspController, get_default_cube_grasp_waypoints,
)
from tactile_sensors.data.collectors import GraspEpisodeCollector
from tactile_sensors.data.savers import save_episode_hdf5, save_metadata

from isaaclab.scene import InteractiveScene
import isaaclab.sim as sim_utils


def main():
    # =========================================================
    # 1. Configuration
    # =========================================================
    sim_cfg = SimulationConfig(
        spawn_mode="g1_robot",
        hand_types=("left", "right"),
        num_envs=1,
        fix_root_link=True,
        robot_usd_path="/workspace/tudor_unitree_isaaclab/isaaclab_tactile_proj/rh56e2_hands_tactile_sensors/gelsight_tactile_sensors/scripts/tactile_sensors/resources/g1_with_rh56e2_hands/G1/g1_rh56e2.usd",
    )

    # NOTE: VisuoTactileSensors disabled for initial testing.
    # They require camera prims (tactile_sensor_camera) inside the USD's
    # elastomer links, which don't exist in the current hand USDs.
    # Enable once the USD prim structure is set up or ContactSensors are used.
    sensor_cfg = SensorConfig(
        palm_enabled=True,
        index_enabled=True,
        middle_enabled=True,
        thumb_enabled=True,
        little_enabled=True,
        ring_enabled=True,
    )

    grasp_kwargs = dict(
        num_episodes=args.num_episodes,
        object_type=args.object_type,
        camera_enabled=not args.no_camera,
        save_tactile_rgb=args.save_tactile_rgb,
    )
    if args.output_dir is not None:
        grasp_kwargs["output_dir"] = args.output_dir
    grasp_cfg = GraspDataCollectionConfig(**grasp_kwargs)

    # =========================================================
    # 2. Simulation Context
    # =========================================================
    sim_context = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(
            dt=sim_cfg.physics_dt,
            render_interval=1,
            device=sim_cfg.device,
            gravity=(0.0, 0.0, -9.81),
            physx=sim_utils.PhysxCfg(
                solver_type=1,
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
    sim_context.set_camera_view(eye=(1.5, -0.5, 1.2), target=(0.5, 0.0, 0.4))

    # =========================================================
    # 3. Scene Creation (robot + sensors + table + object + camera)
    # =========================================================
    scene_cfg = create_grasp_scene(sim_cfg, sensor_cfg, grasp_cfg)
    scene = InteractiveScene(scene_cfg)

    # =========================================================
    # 4. Reset
    # =========================================================
    sim_context.reset()
    scene.reset()

    # =========================================================
    # 5. Robot Diagnostics
    # =========================================================
    robot = scene["robot"]
    print(f"\n{'='*60}")
    print(f"ROBOT DIAGNOSTICS")
    print(f"{'='*60}")
    print(f"  num_joints:  {robot.num_joints}")
    print(f"  joint_names: {robot.joint_names}")
    print(f"  num_bodies:  {robot.num_bodies}")

    # Print joint limits and effort limits for hand joints
    limits = robot.data.soft_joint_pos_limits[0]  # (num_joints, 2)
    effort_limits = robot.data.joint_effort_limits[0] if hasattr(robot.data, "joint_effort_limits") else None
    print(f"\n  Hand joint properties (name: [pos_lower, pos_upper] effort_limit):")
    for i, name in enumerate(robot.joint_names):
        if not any(f in name for f in ("index", "middle", "ring", "little", "thumb")):
            continue
        lo, hi = limits[i, 0].item(), limits[i, 1].item()
        eff = effort_limits[i].item() if effort_limits is not None else "N/A"
        print(f"    {name:40s}  [{lo:+.3f}, {hi:+.3f}]  effort={eff}")
    print(f"{'='*60}\n")

    # Find tactile sensor names in scene
    tactile_sensor_names = []
    for key in scene.keys():
        if "hand" in key and any(
            part in key for part in ("palm", "index", "middle", "thumb", "little", "ring")
        ):
            tactile_sensor_names.append(key)

    print(f"Tactile sensors found: {len(tactile_sensor_names)}")
    for name in tactile_sensor_names[:5]:
        print(f"  {name}")
    if len(tactile_sensor_names) > 5:
        print(f"  ... and {len(tactile_sensor_names) - 5} more")

    # =========================================================
    # 5b. Initialize Tactile Sensors (baseline render)
    # =========================================================
    if tactile_sensor_names:
        # Step a few times so the renderer produces valid images
        for _ in range(5):
            scene.write_data_to_sim()
            sim_context.step()
            scene.update(sim_context.get_physics_dt())

        # Capture nominal (no-contact) tactile images
        for name in tactile_sensor_names:
            sensor = scene[name]
            if hasattr(sensor, "get_initial_render"):
                sensor.get_initial_render()
        print("Tactile sensors initialized (baseline render captured)")

    # =========================================================
    # 6. Output Directory
    # =========================================================
    os.makedirs(grasp_cfg.output_dir, exist_ok=True)
    save_metadata(
        {
            "sim_cfg": {k: v for k, v in vars(sim_cfg).items() if not k.startswith("_")},
            "grasp_cfg": {k: v for k, v in vars(grasp_cfg).items() if not k.startswith("_")},
            "joint_names": list(robot.joint_names),
            "tactile_sensors": tactile_sensor_names,
        },
        os.path.join(grasp_cfg.output_dir, "experiment_config.json"),
    )

    # =========================================================
    # 7. Episode Loop
    # =========================================================
    for episode_idx in range(grasp_cfg.num_episodes):
        if not simulation_app.is_running():
            break

        print(f"\n{'='*60}")
        print(f"EPISODE {episode_idx + 1}/{grasp_cfg.num_episodes}")
        print(f"{'='*60}")

        # --- Reset robot and object ---
        # Reset robot to default state
        root_state = robot.data.default_root_state.clone()
        root_state[:, :3] += scene.env_origins
        robot.write_root_state_to_sim(root_state)

        joint_default = robot.data.default_joint_pos.clone()
        robot.write_joint_state_to_sim(joint_default, torch.zeros_like(joint_default))

        # Reset object
        obj = scene["grasp_object"]
        obj_state = obj.data.default_root_state.clone()
        obj_state[:, :3] += scene.env_origins
        if grasp_cfg.randomize_object_pos:
            noise = torch.randn(1, 3, device=sim_cfg.device) * grasp_cfg.object_pos_noise
            noise[:, 2] = 0.0
            obj_state[:, :3] += noise
        obj.write_root_state_to_sim(obj_state)

        scene.reset()

        # Let physics settle after reset (prevents first-episode instability).
        # Actively drive joints to default position during settle so actuators
        # pull the robot into the correct pose before the trajectory starts.
        settle_target = robot.data.default_joint_pos.clone()
        robot.set_joint_position_target(settle_target)
        for _ in range(50):
            scene.write_data_to_sim()
            sim_context.step()
            scene.update(sim_context.get_physics_dt())

        # --- Create controller ---
        waypoints = get_default_cube_grasp_waypoints()
        controller = GraspController(
            robot=robot,
            waypoints=waypoints,
            physics_dt=sim_cfg.physics_dt,
        )
        print(f"  Trajectory: {controller.total_steps} steps "
              f"({controller.total_steps * sim_cfg.physics_dt:.1f}s)")

        # --- Create episode collector ---
        camera_name = "scene_camera" if grasp_cfg.camera_enabled else None
        collector = GraspEpisodeCollector(
            scene=scene,
            robot_name="robot",
            object_name="grasp_object",
            camera_name=camera_name,
            tactile_sensor_names=tactile_sensor_names,
            save_tactile_rgb=grasp_cfg.save_tactile_rgb,
        )

        # --- Run episode ---
        sim_time = 0.0
        for step in range(controller.total_steps):
            if not simulation_app.is_running():
                break

            # Get and apply action
            action = controller.get_action(step)
            phase = controller.get_phase(step)
            robot.set_joint_position_target(action)

            # Step simulation
            scene.write_data_to_sim()
            sim_context.step()
            scene.update(sim_context.get_physics_dt())
            sim_time += sim_cfg.physics_dt

            # Collect data
            if step % grasp_cfg.save_every_n_steps == 0:
                collector.collect_step(step, sim_time, action, phase)

            # Progress  print on phase transitions + finger diagnostics
            phase_name = controller.get_phase_name(step)
            if step == 0 or phase_name != controller.get_phase_name(max(0, step - 1)):
                obj_pos = obj.data.root_pos_w[0].detach().cpu().numpy()
                print(f"  Step {step:5d}/{controller.total_steps} | "
                      f"Phase: {phase_name:10s} | "
                      f"Obj pos: ({obj_pos[0]:.3f}, {obj_pos[1]:.3f}, {obj_pos[2]:.3f})")

                # Show target vs actual for right hand finger joints
                actual = robot.data.joint_pos[0].detach().cpu()
                target = action[0].detach().cpu()
                finger_joints = [n for n in robot.joint_names if "right_" in n and
                                 any(f in n for f in ("index", "middle", "ring", "little", "thumb"))]
                for jn in finger_joints:
                    idx = controller.name_to_idx[jn]
                    t, a = target[idx].item(), actual[idx].item()
                    if abs(t) > 0.01 or abs(a) > 0.01:
                        print(f"    {jn:35s}  target={t:+.3f}  actual={a:+.3f}  error={t-a:+.3f}")

        # --- Save episode ---
        episode_data = collector.get_episode_data()
        episode_path = os.path.join(
            grasp_cfg.output_dir,
            f"episode_{episode_idx:04d}.hdf5",
        )
        save_episode_hdf5(
            episode_data,
            episode_path,
            metadata={
                "episode_idx": episode_idx,
                "total_steps": controller.total_steps,
                "object_type": grasp_cfg.object_type,
            },
            joint_names=list(robot.joint_names),
        )
        print(f"  Saved: {episode_path} "
              f"({episode_data['joint_pos'].shape[0]} steps)")

    print(f"\nData collection complete. "
          f"{grasp_cfg.num_episodes} episodes saved to {grasp_cfg.output_dir}/")


if __name__ == "__main__":
    main()
    simulation_app.close()
