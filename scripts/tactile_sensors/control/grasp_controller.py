"""
Grasp Controller  Waypoint interpolation for scripted grasps

Mimic joints (_2_, _3_, _4_ finger joints) are handled by PhysX
MimicJointAPI. Only master joints (_1_ joints) are set in waypoints;
PhysX overrides the position targets for mimic joints during the
physics step and the drive (from the actuator group) enforces them.
"""

from dataclasses import dataclass, field
from typing import Dict, List
import torch


@dataclass
class GraspWaypoint:
    """A single target pose in a grasp trajectory.

    Only joints listed in joint_targets are changed  all others
    carry forward from the previous waypoint (or default pose).
    """
    joint_targets: Dict[str, float]
    duration_s: float
    phase_name: str = ""


class GraspController:
    """Interpolates between joint position waypoints over time.

    Only set targets for master joints. Mimic joints keep their default
    target (0.0)  PhysX MimicJointAPI overrides these during the
    physics step with gearing * master_pos.

    Args:
        robot: Isaac Lab Articulation reference (for joint names and defaults).
        waypoints: Ordered list of GraspWaypoints.
        physics_dt: Simulation timestep in seconds.
    """

    def __init__(
        self,
        robot,
        waypoints: List[GraspWaypoint],
        physics_dt: float,
    ):
        self.robot = robot
        self.waypoints = waypoints
        self.physics_dt = physics_dt
        self.device = robot.device

        # Build name -> index mapping
        self.joint_names = list(robot.joint_names)
        self.name_to_idx = {name: i for i, name in enumerate(self.joint_names)}
        self.num_joints = len(self.joint_names)

        self._build_trajectory()

    def _build_trajectory(self):
        """Convert waypoints into a dense trajectory tensor.

        For each waypoint, expands duration_s into N interpolation steps.
        Joints not mentioned in a waypoint keep their value from the
        previous waypoint. Mimic joints keep their default position 
        PhysX MimicJointAPI drives them from the master joint.
        """
        default_pos = self.robot.data.default_joint_pos[0].detach().cpu()

        # Resolve each waypoint into a full joint target vector
        full_targets = []
        current = default_pos.clone()

        for wp in self.waypoints:
            target = current.clone()
            for joint_name, value in wp.joint_targets.items():
                if joint_name not in self.name_to_idx:
                    print(f"  [GraspController] WARNING: joint '{joint_name}' "
                          f"not found in robot. Available: {self.joint_names}")
                    continue
                target[self.name_to_idx[joint_name]] = value
            full_targets.append(target)
            current = target.clone()

        # Interpolate between consecutive targets
        trajectory_steps = []
        phase_ids = []

        prev_target = default_pos.clone()

        for phase_idx, (wp, target) in enumerate(zip(self.waypoints, full_targets)):
            n_steps = max(1, int(wp.duration_s / self.physics_dt))
            for step in range(n_steps):
                alpha = step / n_steps
                interp = prev_target + alpha * (target - prev_target)
                trajectory_steps.append(interp)
                phase_ids.append(phase_idx)
            prev_target = target.clone()

        # Hold final pose for a short period
        hold_steps = int(0.5 / self.physics_dt)
        for _ in range(hold_steps):
            trajectory_steps.append(prev_target.clone())
            phase_ids.append(len(self.waypoints) - 1)

        self.trajectory = torch.stack(trajectory_steps).to(self.device)
        self.phase_ids = phase_ids

    def get_action(self, step: int) -> torch.Tensor:
        """Returns joint position target for the given step.

        Returns:
            Tensor of shape (1, num_joints).
        """
        idx = min(step, len(self.trajectory) - 1)
        return self.trajectory[idx].unsqueeze(0)

    def get_phase(self, step: int) -> int:
        """Returns current grasp phase index."""
        idx = min(step, len(self.phase_ids) - 1)
        return self.phase_ids[idx]

    def get_phase_name(self, step: int) -> str:
        """Returns current grasp phase name."""
        phase = self.get_phase(step)
        if phase < len(self.waypoints):
            return self.waypoints[phase].phase_name
        return "done"

    @property
    def total_steps(self) -> int:
        return len(self.trajectory)


def get_default_cube_grasp_waypoints() -> List[GraspWaypoint]:
    """Default right-hand grasp trajectory for a cube on a table.

    Only set master joints (_1_joint)  mimic joints (_2_, _3_, _4_)
    are driven automatically by PhysX MimicJointAPI.

    Joint naming convention (from robot.joint_names):
        right_shoulder_pitch_joint, right_shoulder_roll_joint, etc.
        right_index_1_joint (master), right_index_2_joint (mimic), etc.
    """
    time_factor = 4.0  # 1.0 = normal speed
    return [
        GraspWaypoint(
            phase_name="home",
            duration_s=0.5 / time_factor,
            joint_targets={
                # Right arm: relaxed at side
                "right_shoulder_pitch_joint": 0.0,
                "right_shoulder_roll_joint": 0.0,
                "right_shoulder_yaw_joint": 0.0,
                "right_elbow_joint": 0.0,
                "right_wrist_roll_joint": 0.0,
                "right_wrist_pitch_joint": 0.0,
                "right_wrist_yaw_joint": 0.0,
                # Right hand: open (only master joints)
                "right_index_1_joint": 0.0,
                "right_middle_1_joint": 0.0,
                "right_ring_1_joint": 0.0,
                "right_little_1_joint": 0.0,
                "right_thumb_1_joint": 0.0,
            },
        ),
        GraspWaypoint(
            phase_name="pregrasp",
            duration_s=1.0 / time_factor,
            joint_targets={
                # Arm reaches forward and down toward table
                "right_shoulder_pitch_joint": -1.0,
                "right_shoulder_roll_joint": -0.3,
                "right_elbow_joint": 1.2,
                "right_wrist_pitch_joint": -0.5,
            },
        ),
        GraspWaypoint(
            phase_name="approach",
            duration_s=1.0 / time_factor,
            joint_targets={
                # Lower hand closer to object
                "right_shoulder_pitch_joint": -1.3,
                "right_elbow_joint": 1.5,
                "right_wrist_pitch_joint": -0.8,
            },
        ),
        GraspWaypoint(
            phase_name="grasp",
            duration_s=1.0 / time_factor,
            joint_targets={
                # Close master joints  mimic driven by PhysX
                "right_index_1_joint": 1.0,
                "right_middle_1_joint": 1.0,
                "right_ring_1_joint": 1.0,
                "right_little_1_joint": 1.0,
                "right_thumb_1_joint": 1.0,
                "right_thumb_2_joint": 0.7,
            },
        ),
        GraspWaypoint(
            phase_name="hold",
            duration_s=0.5 / time_factor,
            joint_targets={
                # Nothing changes  let fingers settle
            },
        ),
        GraspWaypoint(
            phase_name="lift",
            duration_s=1.0 / time_factor,
            joint_targets={
                # Raise arm  fingers stay closed (inherited)
                "right_shoulder_pitch_joint": -0.5,
                "right_elbow_joint": 0.8,
            },
        ),
    ]
