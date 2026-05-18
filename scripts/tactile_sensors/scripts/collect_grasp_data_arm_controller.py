#!/usr/bin/env python3
"""Keyboard-controlled joint teleop with data recording for G1 + RH56E2.

Select joints with Tab/Shift+Tab, nudge position targets with Up/Down arrows,
and record episodes to HDF5 with R key.

Usage:
    python collect_grasp_data_keyboard.py --output_dir keyboard_data
"""

import argparse
import sys
import os
import time
import threading
from pathlib import Path

# Path setup (same pattern as collect_grasp_data_with_waypoints.py)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PACKAGE_PARENT = PROJECT_ROOT.parent

if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Keyboard joint teleop with data recording")
parser.add_argument("--output_dir", type=str, default=None,
                    help="Output directory for HDF5 files")
parser.add_argument("--object_type", type=str, default="cube",
                    choices=["cube", "nut"], help="Object to grasp")
parser.add_argument("--no_camera", action="store_true",
                    help="Disable camera recording")
parser.add_argument("--save_tactile_rgb", action="store_true",
                    help="Save tactile RGB images (large)")
parser.add_argument("--step_size", type=float, default=0.02,
                    help="Initial position increment per keypress (rad)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# === All Isaac Lab imports AFTER AppLauncher ===
import torch
from pynput import keyboard

from tactile_sensors.config import SimulationConfig, SensorConfig
from tactile_sensors.config.grasp_config import GraspDataCollectionConfig
from tactile_sensors.core.scene_composer import create_grasp_scene
from tactile_sensors.data.collectors import GraspEpisodeCollector
from tactile_sensors.data.savers import save_episode_hdf5, save_metadata

from isaaclab.scene import InteractiveScene
import isaaclab.sim as sim_utils


# =====================================================================
# KeyboardJointController
# =====================================================================

STEP_SIZES = [0.005, 0.01, 0.02, 0.05, 0.1]


class KeyboardJointController:
    """Controls individual robot joints via keyboard input.

    Keys:
        Right / Left arrow  next / previous joint
        Up / Down arrow     increment / decrement selected joint position
        +/= / -/_           increase / decrease step size
        R                   toggle recording
        Backspace           reset all joints to default
        Q / Esc             quit
    """

    def __init__(self, joint_names, default_pos, joint_limits, device, step_size=0.02):
        """
        Args:
            joint_names: list of joint name strings.
            default_pos: tensor of shape (num_joints,) with default positions.
            joint_limits: tensor of shape (num_joints, 2) with [lower, upper].
            device: torch device.
            step_size: initial position increment per keypress (rad).
        """
        self.joint_names = list(joint_names)
        self.num_joints = len(self.joint_names)
        self.device = device

        self.default_pos = default_pos.clone().detach().cpu()
        self.joint_limits = joint_limits.clone().detach().cpu()
        self.targets = default_pos.clone().detach().cpu()

        # State
        self.selected_joint = 0
        self.step_size = step_size
        self._step_idx = STEP_SIZES.index(step_size) if step_size in STEP_SIZES else 2
        self.recording = False
        self.quit_requested = False

        # Events for the main loop to react to
        self._recording_just_toggled = False
        self._reset_requested = False
        self._display_dirty = True

        self._lock = threading.Lock()

        # Start pynput listener
        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._listener.daemon = True
        self._listener.start()

        self._print_help()

    def _print_help(self):
        print("\n" + "=" * 60)
        print("KEYBOARD JOINT TELEOP")
        print("=" * 60)
        print("  Right / Left       next / previous joint")
        print("  Up / Down          increase / decrease joint position")
        print("  + / -              increase / decrease step size")
        print("  R                  toggle recording")
        print("  Backspace          reset all joints to default")
        print("  Q / Esc            quit")
        print("=" * 60 + "\n")

    def _on_press(self, key):
        with self._lock:
            # --- Special keys ---
            if key == keyboard.Key.right:
                self.selected_joint = (self.selected_joint + 1) % self.num_joints
                self._display_dirty = True
                return

            if key == keyboard.Key.left:
                self.selected_joint = (self.selected_joint - 1) % self.num_joints
                self._display_dirty = True
                return

            if key == keyboard.Key.up:
                idx = self.selected_joint
                self.targets[idx] = min(
                    self.targets[idx] + self.step_size,
                    self.joint_limits[idx, 1].item(),
                )
                self._display_dirty = True
                return

            if key == keyboard.Key.down:
                idx = self.selected_joint
                self.targets[idx] = max(
                    self.targets[idx] - self.step_size,
                    self.joint_limits[idx, 0].item(),
                )
                self._display_dirty = True
                return

            if key == keyboard.Key.backspace:
                self.targets = self.default_pos.clone()
                self._reset_requested = True
                self._display_dirty = True
                return

            if key == keyboard.Key.esc:
                self.quit_requested = True
                return False

            # --- Character keys ---
            try:
                ch = key.char.lower() if hasattr(key, 'char') and key.char else None
            except AttributeError:
                return

            if ch is None:
                return

            if ch == 'q':
                self.quit_requested = True
                return False

            if ch == 'r':
                self.recording = not self.recording
                self._recording_just_toggled = True
                self._display_dirty = True

            elif ch in ('+', '='):
                self._step_idx = min(self._step_idx + 1, len(STEP_SIZES) - 1)
                self.step_size = STEP_SIZES[self._step_idx]
                self._display_dirty = True

            elif ch in ('-', '_'):
                self._step_idx = max(self._step_idx - 1, 0)
                self.step_size = STEP_SIZES[self._step_idx]
                self._display_dirty = True

    def _on_release(self, _key):
        pass

    def get_targets(self):
        """Returns current joint targets as a (1, num_joints) tensor on device."""
        with self._lock:
            return self.targets.clone().unsqueeze(0).to(self.device)

    def poll_events(self):
        """Returns (recording_toggled, reset_requested, display_dirty) and clears flags."""
        with self._lock:
            toggled = self._recording_just_toggled
            reset = self._reset_requested
            dirty = self._display_dirty
            self._recording_just_toggled = False
            self._reset_requested = False
            self._display_dirty = False
            return toggled, reset, dirty

    def get_status_line(self, actual_pos=None, sim_step=0):
        """Returns a formatted status string for terminal display."""
        with self._lock:
            idx = self.selected_joint
            name = self.joint_names[idx]
            target = self.targets[idx].item()
            lo = self.joint_limits[idx, 0].item()
            hi = self.joint_limits[idx, 1].item()
            actual = actual_pos[idx].item() if actual_pos is not None else float('nan')
            rec = "ON" if self.recording else "OFF"
            return (
                f"[Joint {idx+1}/{self.num_joints}] {name:40s}  "
                f"target={target:+.4f}  actual={actual:+.4f}  "
                f"limits=[{lo:+.3f}, {hi:+.3f}]  step={self.step_size}\n"
                f"[Recording: {rec}]  [Sim step: {sim_step}]"
            )

    def stop(self):
        if hasattr(self, '_listener'):
            self._listener.stop()


# =====================================================================
# Main
# =====================================================================

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

    sensor_cfg = SensorConfig(
        palm_enabled=True,
        index_enabled=True,
        middle_enabled=True,
        thumb_enabled=True,
        little_enabled=True,
        ring_enabled=True,
    )

    grasp_kwargs = dict(
        num_episodes=999,  # unlimited  controlled by user
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
        for _ in range(5):
            scene.write_data_to_sim()
            sim_context.step()
            scene.update(sim_context.get_physics_dt())

        for name in tactile_sensor_names:
            sensor = scene[name]
            if hasattr(sensor, "get_initial_render"):
                sensor.get_initial_render()
        print("Tactile sensors initialized (baseline render captured)")

    # =========================================================
    # 5c. Let physics settle
    # =========================================================
    settle_target = robot.data.default_joint_pos.clone()
    robot.set_joint_position_target(settle_target)
    for _ in range(50):
        scene.write_data_to_sim()
        sim_context.step()
        scene.update(sim_context.get_physics_dt())
    print("Physics settled.")

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
            "control_mode": "keyboard_teleop",
        },
        os.path.join(grasp_cfg.output_dir, "experiment_config.json"),
    )

    # =========================================================
    # 7. Keyboard Controller
    # =========================================================
    controller = KeyboardJointController(
        joint_names=robot.joint_names,
        default_pos=robot.data.default_joint_pos[0],
        joint_limits=robot.data.soft_joint_pos_limits[0],
        device=robot.device,
        step_size=args.step_size,
    )

    # =========================================================
    # 8. Main Loop
    # =========================================================
    episode_idx = 0
    sim_step = 0
    recording_step = 0
    collector = None
    last_display_time = 0.0

    camera_name = "scene_camera" if grasp_cfg.camera_enabled else None

    try:
        while simulation_app.is_running() and not controller.quit_requested:

            robot_view = Articulation(robot)
            arm_controller = ArticulationController()
            arm_controller.initialize(robot_view)
            current_positions = robot_view.get_joint_positions()
            print(f"Current joint positions: {current_positions}")
            #TODO: Here i would need IK to determine each joint position depending on the wrist effector desired pose
            #Then i would have to apply the Articulation controller knowing each joint posistion (Rotations in this case).
            #The controller only takes each joint position individually, not the end effector 3D pose. That's why we need IK.
            #See https://docs.isaacsim.omniverse.nvidia.com/5.1.0/robot_simulation/articulation_controller.html


            # --- Poll keyboard events ---
            toggled, reset, dirty = controller.poll_events()

            # Handle recording toggle
            if toggled:
                if controller.recording:
                    # Started recording
                    collector = GraspEpisodeCollector(
                        scene=scene,
                        robot_name="robot",
                        object_name="grasp_object",
                        camera_name=camera_name,
                        tactile_sensor_names=tactile_sensor_names,
                        save_tactile_rgb=grasp_cfg.save_tactile_rgb,
                    )
                    recording_step = 0
                    print(f"\n>>> RECORDING STARTED (episode {episode_idx}) <<<\n")
                else:
                    # Stopped recording  save episode
                    if collector is not None:
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
                                "total_steps": recording_step,
                                "object_type": grasp_cfg.object_type,
                                "control_mode": "keyboard_teleop",
                            },
                            joint_names=list(robot.joint_names),
                        )
                        print(f"\n>>> RECORDING STOPPED  saved {episode_path} "
                              f"({recording_step} steps) <<<\n")
                        episode_idx += 1
                        collector = None

            # Handle reset
            if reset:
                root_state = robot.data.default_root_state.clone()
                root_state[:, :3] += scene.env_origins
                robot.write_root_state_to_sim(root_state)
                robot.write_joint_state_to_sim(
                    robot.data.default_joint_pos.clone(),
                    torch.zeros_like(robot.data.default_joint_pos),
                )

                obj = scene["grasp_object"]
                obj_state = obj.data.default_root_state.clone()
                obj_state[:, :3] += scene.env_origins
                obj.write_root_state_to_sim(obj_state)

                scene.reset()
                # Re-settle after reset
                settle_target = robot.data.default_joint_pos.clone()
                robot.set_joint_position_target(settle_target)
                for _ in range(20):
                    scene.write_data_to_sim()
                    sim_context.step()
                    scene.update(sim_context.get_physics_dt())
                print("  [Reset to default pose]")

            # --- Apply joint targets ---
            targets = controller.get_targets()
            robot.set_joint_position_target(targets)

            # --- Step simulation ---
            scene.write_data_to_sim()
            sim_context.step()
            scene.update(sim_context.get_physics_dt())
            sim_step += 1

            # --- Collect data if recording ---
            if controller.recording and collector is not None:
                if recording_step % grasp_cfg.save_every_n_steps == 0:
                    collector.collect_step(
                        step=recording_step,
                        sim_time=sim_step * sim_cfg.physics_dt,
                        action=targets,
                        phase=0,  # no phase distinction in keyboard mode
                    )
                recording_step += 1

            # --- Terminal HUD (throttled to ~2 Hz) ---
            now = time.monotonic()
            if dirty or (now - last_display_time > 0.5):
                actual_pos = robot.data.joint_pos[0].detach().cpu()
                status = controller.get_status_line(actual_pos, sim_step)
                # Clear previous lines and reprint
                print(f"\r\033[2K{status}", end="\033[F\033[2K\r", flush=True)
                # Print on two lines
                lines = status.split('\n')
                print(f"\r\033[2K{lines[0]}")
                print(f"\r\033[2K{lines[1]}", end="", flush=True)
                last_display_time = now

    except KeyboardInterrupt:
        print("\n\nInterrupted by Ctrl+C")

    # Save any in-progress recording
    if controller.recording and collector is not None:
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
                "total_steps": recording_step,
                "object_type": grasp_cfg.object_type,
                "control_mode": "keyboard_teleop",
            },
            joint_names=list(robot.joint_names),
        )
        print(f"\nAuto-saved in-progress recording: {episode_path} "
              f"({recording_step} steps)")
        episode_idx += 1

    controller.stop()
    print(f"\nSession complete. {episode_idx} episodes saved to {grasp_cfg.output_dir}/")


if __name__ == "__main__":
    main()
    simulation_app.close()
