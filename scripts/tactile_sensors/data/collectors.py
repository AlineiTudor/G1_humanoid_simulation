"""
Data Collectors  per-step sensor/state recording
"""

from typing import List, Dict, Any, Optional
import numpy as np
import torch

from isaaclab.scene import InteractiveScene


class TactileDataCollector:
    """Collects tactile sensor data during simulation."""

    def __init__(self, scene: InteractiveScene, sensor_names: List[str]):
        self.scene = scene
        self.sensor_names = sensor_names
        self.buffer = []

    def collect_step(self, step: int):
        """Collect data for current step."""
        step_data = {"step": step}

        for sensor_name in self.sensor_names:
            sensor = self.scene[sensor_name]
            step_data[sensor_name] = {
                "rgb": sensor.data.tactile_rgb_image,
                "depth": sensor.data.tactile_depth_image,
                "normal_force": sensor.data.tactile_normal_force,
                "shear_force": sensor.data.tactile_shear_force,
            }

        self.buffer.append(step_data)

    def get_buffer(self) -> List[Dict]:
        """Returns collected data."""
        return self.buffer

    def clear_buffer(self):
        """Clears collected data."""
        self.buffer = []


class GraspEpisodeCollector:
    """Collects all data for a single grasp episode.

    Records joint states, actions, object pose, tactile sensors,
    and camera images at each timestep.
    """

    def __init__(
        self,
        scene: InteractiveScene,
        robot_name: str = "robot",
        object_name: str = "grasp_object",
        camera_name: Optional[str] = "scene_camera",
        tactile_sensor_names: Optional[List[str]] = None,
        save_tactile_rgb: bool = False,
    ):
        self.scene = scene
        self.robot = scene[robot_name]
        self.object = scene[object_name]
        try:
            self.camera = scene[camera_name] if camera_name else None
        except KeyError:
            self.camera = None
        self.tactile_sensors = {}
        if tactile_sensor_names:
            for name in tactile_sensor_names:
                try:
                    self.tactile_sensors[name] = scene[name]
                except KeyError:
                    pass
        self.save_tactile_rgb = save_tactile_rgb
        self.reset()

    def reset(self):
        """Clear buffers for new episode."""
        self.timestamps = []
        self.joint_positions = []
        self.joint_velocities = []
        self.joint_torques = []
        self.actions_applied = []
        self.object_positions = []
        self.object_rotations = []
        self.grasp_phases = []
        self.tactile_data = {name: [] for name in self.tactile_sensors}
        self.camera_frames = []

    def collect_step(
        self,
        step: int,
        sim_time: float,
        action: torch.Tensor,
        phase: int,
    ):
        """Collect all data for current timestep.

        Args:
            step: Simulation step index.
            sim_time: Current simulation time in seconds.
            action: Joint position targets tensor, shape (num_envs, num_joints).
            phase: Current grasp phase index.
        """
        self.timestamps.append(sim_time)
        self.grasp_phases.append(phase)

        # Joint state (env 0)
        self.joint_positions.append(
            self.robot.data.joint_pos[0].detach().cpu().numpy().copy()
        )
        self.joint_velocities.append(
            self.robot.data.joint_vel[0].detach().cpu().numpy().copy()
        )
        # Applied torque  may not be available before first step
        if hasattr(self.robot.data, "applied_torque") and self.robot.data.applied_torque is not None:
            self.joint_torques.append(
                self.robot.data.applied_torque[0].detach().cpu().numpy().copy()
            )
        else:
            self.joint_torques.append(
                np.zeros_like(self.joint_positions[-1])
            )

        # Action
        self.actions_applied.append(
            action[0].detach().cpu().numpy().copy()
        )

        # Object pose (env 0)
        obj_pos = self.object.data.root_pos_w[0]
        obj_rot = self.object.data.root_quat_w[0]
        self.object_positions.append(obj_pos.detach().cpu().numpy().copy())
        self.object_rotations.append(obj_rot.detach().cpu().numpy().copy())

        # Tactile sensors
        for name, sensor in self.tactile_sensors.items():
            sensor_step = {}
            data = sensor.data
            if data.tactile_normal_force is not None:
                sensor_step["normal_force"] = (
                    data.tactile_normal_force[0].detach().cpu().numpy().copy()
                )
            if data.tactile_shear_force is not None:
                sensor_step["shear_force"] = (
                    data.tactile_shear_force[0].detach().cpu().numpy().copy()
                )
            if self.save_tactile_rgb and data.tactile_rgb_image is not None:
                sensor_step["rgb"] = (
                    data.tactile_rgb_image[0].detach().cpu().numpy().copy()
                )
            self.tactile_data[name].append(sensor_step)

        # Camera
        if self.camera is not None:
            rgb = self.camera.data.output.get("rgb")
            if rgb is not None:
                self.camera_frames.append(
                    rgb[0].detach().cpu().numpy().copy()
                )

    def get_episode_data(self) -> Dict[str, Any]:
        """Package all collected data into a dict for saving."""
        # Stack tactile data per sensor
        tactile_out = {}
        for name, steps in self.tactile_data.items():
            if not steps or not steps[0]:
                continue
            sensor_out = {}
            for key in steps[0].keys():
                arrays = [s[key] for s in steps if key in s]
                if arrays:
                    sensor_out[key] = np.stack(arrays)
            tactile_out[name] = sensor_out

        return {
            "timestamps": np.array(self.timestamps),
            "joint_pos": np.stack(self.joint_positions),
            "joint_vel": np.stack(self.joint_velocities),
            "joint_torque": np.stack(self.joint_torques),
            "actions": np.stack(self.actions_applied),
            "object_pos": np.stack(self.object_positions),
            "object_rot": np.stack(self.object_rotations),
            "grasp_phase": np.array(self.grasp_phases),
            "tactile": tactile_out,
            "camera_rgb": np.stack(self.camera_frames) if self.camera_frames else None,
        }
