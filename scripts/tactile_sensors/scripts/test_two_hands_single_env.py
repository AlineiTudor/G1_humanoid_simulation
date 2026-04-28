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
        num_envs=2,
        hand_types=("left","right"),
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

    # Create scene
    scene_cfg = create_tactile_scene(sim_cfg, sensor_cfg)
    scene = InteractiveScene(scene_cfg)

    print(f"Created {sim_cfg.num_envs} environments")
    print(f"Hands per env: {sim_cfg.num_hands_per_env}")
    print(f"Total hands: {sim_cfg.total_hands}")
    print(f"Hand types: {sim_cfg.hand_types}")
    print("\nCompliant contact materials applied automatically during hand spawning")

    # Simulation loop
    sim_context.reset()
    scene.reset()

    print("\nStarting simulation loop...")
    step_count = 0

    while simulation_app.is_running():
        # Write data to simulation
        scene.write_data_to_sim()

        # Step simulation
        sim_context.step()

        # Update scene
        scene.update(sim_context.get_physics_dt())

        # Print status every 100 steps
        step_count += 1
        if step_count % 100 == 0:
            print(f"Step {step_count}")

if __name__ == "__main__":
    main()
    simulation_app.close()
