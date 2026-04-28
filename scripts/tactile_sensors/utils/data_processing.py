#!/usr/bin/env python3
"""Test dual hands in single environment."""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Now import Isaac Lab modules
from RH56E2_tactile_proj.config import SimulationConfig, SensorConfig
from RH56E2_tactile_proj.core import create_tactile_scene
from isaaclab.scene import InteractiveScene
import isaaclab.sim as sim_utils

def main():
    # Config: 1 environment with 2 hands
    sim_cfg = SimulationConfig(
        num_envs=1,
        hand_types=("left", "right"),
    )
    sensor_cfg = SensorConfig(palm_enabled=True)
    
    # Create scene
    scene_cfg = create_tactile_scene(sim_cfg, sensor_cfg)
    scene = InteractiveScene(scene_cfg)
    
    print(f"Created {sim_cfg.num_envs} environments")
    print(f"Hands per env: {sim_cfg.num_hands_per_env}")
    print(f"Total hands: {sim_cfg.total_hands}")
    print(f"Hand types: {sim_cfg.hand_types}")
    
    # Simulation loop
    scene.reset()
    while simulation_app.is_running():
        scene.write_data_to_sim()
        sim_utils.SimulationContext.instance().step()
        scene.update(sim_cfg.physics_dt)

if __name__ == "__main__":
    main()
    simulation_app.close()
