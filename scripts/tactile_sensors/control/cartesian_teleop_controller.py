"""
Cartesian Teleop Controller  6-DOF IK keyboard control with grasp
===================================================================
Keyboard-driven Cartesian end-effector control using damped
least-squares (DLS) differential IK, plus a 0-1 grasp slider.

Only master finger joints are commanded; mimic joints (_2/_3/_4)
are handled by PhysX MimicJointAPI.
"""

import threading
import torch
from pynput import keyboard


# =====================================================================
# Quaternion utilities (w,x,y,z convention  matches IsaacLab)
# =====================================================================

def quat_multiply(q1, q2):
    """Hamilton product of two quaternions (w,x,y,z)."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dim=-1)


def quat_conjugate(q):
    """Conjugate (inverse for unit quaternion), (w,x,y,z)."""
    return torch.stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]], dim=-1)


def quat_from_axis_angle(axis, angle, device=None):
    """Quaternion from axis (3,) and scalar angle. Returns (w,x,y,z)."""
    half = angle * 0.5
    s = torch.sin(torch.tensor(half, device=device))
    c = torch.cos(torch.tensor(half, device=device))
    ax = torch.as_tensor(axis, dtype=torch.float32, device=device)
    return torch.stack([c, ax[0]*s, ax[1]*s, ax[2]*s])


def quat_error_to_axis_angle(q_desired, q_current):
    """Orientation error as axis-angle vector (3,).

    Returns the rotation that takes q_current ? q_desired.
    """
    q_err = quat_multiply(q_desired, quat_conjugate(q_current))
    # Shortest path
    if q_err[0] < 0:
        q_err = -q_err
    xyz = q_err[1:4]
    sin_half = torch.norm(xyz)
    if sin_half < 1e-8:
        return torch.zeros(3, device=q_desired.device)
    axis = xyz / sin_half
    angle = 2.0 * torch.atan2(sin_half, q_err[0])
    return axis * angle


# =====================================================================
# Step size presets
# =====================================================================

POS_STEP_SIZES = [0.001, 0.002, 0.005, 0.01, 0.02]
ROT_STEP_SIZES = [0.005, 0.01, 0.02, 0.05, 0.1]

# Default closed positions for finger master joints
FINGER_CLOSED = {
    "index_1_joint": 1.4,
    "middle_1_joint": 1.4,
    "ring_1_joint": 1.4,
    "little_1_joint": 1.4,
    "thumb_1_joint": 1.0,
    "thumb_2_joint": 0.5,
}

# Arm joint names in kinematic order (shoulder ? wrist)
ARM_JOINT_NAMES = [
    "{side}_shoulder_pitch_joint",
    "{side}_shoulder_roll_joint",
    "{side}_shoulder_yaw_joint",
    "{side}_elbow_joint",
    "{side}_wrist_roll_joint",
    "{side}_wrist_pitch_joint",
    "{side}_wrist_yaw_joint",
]

EE_BODY_NAME = "{side}_wrist_yaw_link"


# =====================================================================
# CartesianTeleopController
# =====================================================================

class CartesianTeleopController:
    """6-DOF Cartesian keyboard teleop with DLS IK for G1 + RH56E2.

    Keys:
        W/S, A/D, E/C     translate EE in world X/Y/Z
        Up/Down            pitch
        Left/Right         yaw
        Z/X                roll
        G / R              close / open hand
        Tab                switch active arm
        P                  toggle recording
        +/- (=/_ also)     increase / decrease step size
        Backspace          reset robot and object
        Esc                quit
    """

    def __init__(
        self,
        robot,
        device,
        pos_step=0.005,
        rot_step=0.02,
        damping_lambda=0.05,
        grasp_smoothing=0.05,
        max_dq=0.2,
    ):
        self.device = device
        self.num_joints = robot.num_joints

        # --- Build index maps ---
        self._joint_names = list(robot.joint_names)
        self._body_names = list(robot.body_names)
        joint_idx = {n: i for i, n in enumerate(self._joint_names)}
        body_idx = {n: i for i, n in enumerate(self._body_names)}

        self._arm_joint_indices = {}
        self._ee_body_indices = {}
        self._hand_joint_indices = {}
        self._hand_closed_vals = {}

        for side in ("left", "right"):
            # Arm joints
            arm_names = [t.format(side=side) for t in ARM_JOINT_NAMES]
            self._arm_joint_indices[side] = [joint_idx[n] for n in arm_names]

            # EE body
            ee_name = EE_BODY_NAME.format(side=side)
            if ee_name not in body_idx:
                raise RuntimeError(
                    f"EE body '{ee_name}' not found. Available: {self._body_names}"
                )
            self._ee_body_indices[side] = body_idx[ee_name]

            # Hand master joints
            h_indices = []
            h_closed = []
            for suffix, closed_val in FINGER_CLOSED.items():
                name = f"{side}_{suffix}"
                if name in joint_idx:
                    h_indices.append(joint_idx[name])
                    h_closed.append(closed_val)
            self._hand_joint_indices[side] = h_indices
            self._hand_closed_vals[side] = h_closed

        # --- Joint limits ---
        self._joint_limits = robot.data.soft_joint_pos_limits[0].clone().cpu()
        self._default_pos = robot.data.default_joint_pos[0].clone().cpu()

        # --- Jacobian body offset (fixed base removes root body row) ---
        jac = robot.root_physx_view.get_jacobians()
        if jac.shape[1] == robot.num_bodies - 1:
            self._jac_body_offset = 1
        elif jac.shape[1] == robot.num_bodies:
            self._jac_body_offset = 0
        else:
            raise RuntimeError(
                f"Unexpected Jacobian shape {jac.shape}, "
                f"num_bodies={robot.num_bodies}"
            )
        print(f"  [IK] Jacobian shape: {jac.shape}, body offset: {self._jac_body_offset}")

        # --- IK parameters ---
        self.damping_lambda = damping_lambda
        self.max_dq = max_dq
        self.grasp_smoothing = grasp_smoothing

        # --- Keyboard state (protected by lock) ---
        self._lock = threading.Lock()

        self.active_arm = "right"
        self.desired_pos = {"left": None, "right": None}
        self.desired_quat = {"left": None, "right": None}
        self.grasp_target = {"left": 0.0, "right": 0.0}
        self.grasp_current = {"left": 0.0, "right": 0.0}

        self.pos_step = pos_step
        self.rot_step = rot_step
        self._step_idx = (
            POS_STEP_SIZES.index(pos_step) if pos_step in POS_STEP_SIZES else 2
        )

        self.recording = False
        self.quit_requested = False
        self._recording_just_toggled = False
        self._reset_requested = False
        self._display_dirty = True

        # --- Start keyboard listener ---
        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._listener.daemon = True
        self._listener.start()

        self._print_help()

    # -----------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------

    def initialize_desired_poses(self, robot):
        """Read current EE poses from robot state. Call after physics settle."""
        with self._lock:
            for side in ("left", "right"):
                idx = self._ee_body_indices[side]
                self.desired_pos[side] = (
                    robot.data.body_pos_w[0, idx].detach().clone().cpu()
                )
                self.desired_quat[side] = (
                    robot.data.body_quat_w[0, idx].detach().clone().cpu()
                )

    # -----------------------------------------------------------------
    # Keyboard handler
    # -----------------------------------------------------------------

    def _print_help(self):
        print("\n" + "=" * 60)
        print("CARTESIAN TELEOP (6-DOF IK)")
        print("=" * 60)
        print("  W/S            EE forward / backward  (X)")
        print("  A/D            EE left / right         (Y)")
        print("  E/C            EE up / down            (Z)")
        print("  Up/Down        EE pitch")
        print("  Left/Right     EE yaw")
        print("  Z/X            EE roll")
        print("  G / R          close / open hand")
        print("  Tab            switch active arm")
        print("  P              toggle recording")
        print("  + / -          increase / decrease step size")
        print("  Backspace      reset")
        print("  Esc            quit")
        print("=" * 60 + "\n")

    def _on_press(self, key):
        with self._lock:
            arm = self.active_arm
            dp = self.desired_pos[arm]
            dq = self.desired_quat[arm]
            dev = dp.device if dp is not None else "cpu"

            # --- Arrow keys: orientation ---
            if key == keyboard.Key.up:
                delta = quat_from_axis_angle([0, 1, 0], self.rot_step, dev)
                self.desired_quat[arm] = quat_multiply(dq, delta)
                self._display_dirty = True
                return
            if key == keyboard.Key.down:
                delta = quat_from_axis_angle([0, 1, 0], -self.rot_step, dev)
                self.desired_quat[arm] = quat_multiply(dq, delta)
                self._display_dirty = True
                return
            if key == keyboard.Key.left:
                delta = quat_from_axis_angle([0, 0, 1], self.rot_step, dev)
                self.desired_quat[arm] = quat_multiply(dq, delta)
                self._display_dirty = True
                return
            if key == keyboard.Key.right:
                delta = quat_from_axis_angle([0, 0, 1], -self.rot_step, dev)
                self.desired_quat[arm] = quat_multiply(dq, delta)
                self._display_dirty = True
                return

            if key == keyboard.Key.tab:
                self.active_arm = "left" if self.active_arm == "right" else "right"
                self._display_dirty = True
                return

            if key == keyboard.Key.backspace:
                self._reset_requested = True
                self._display_dirty = True
                return

            if key == keyboard.Key.esc:
                self.quit_requested = True
                return False

            # --- Character keys ---
            try:
                ch = key.char.lower() if hasattr(key, "char") and key.char else None
            except AttributeError:
                return
            if ch is None:
                return

            # Position
            if ch == "w":
                dp[0] += self.pos_step
                self._display_dirty = True
            elif ch == "s":
                dp[0] -= self.pos_step
                self._display_dirty = True
            elif ch == "a":
                dp[1] += self.pos_step
                self._display_dirty = True
            elif ch == "d":
                dp[1] -= self.pos_step
                self._display_dirty = True
            elif ch == "e":
                dp[2] += self.pos_step
                self._display_dirty = True
            elif ch == "c":
                dp[2] -= self.pos_step
                self._display_dirty = True

            # Orientation (roll)
            elif ch == "z":
                delta = quat_from_axis_angle([1, 0, 0], self.rot_step, dev)
                self.desired_quat[arm] = quat_multiply(dq, delta)
                self._display_dirty = True
            elif ch == "x":
                delta = quat_from_axis_angle([1, 0, 0], -self.rot_step, dev)
                self.desired_quat[arm] = quat_multiply(dq, delta)
                self._display_dirty = True

            # Grasp
            elif ch == "g":
                self.grasp_target[arm] = 1.0
                self._display_dirty = True
            elif ch == "r":
                self.grasp_target[arm] = 0.0
                self._display_dirty = True

            # Recording
            elif ch == "p":
                self.recording = not self.recording
                self._recording_just_toggled = True
                self._display_dirty = True

            # Step size
            elif ch in ("+", "="):
                self._step_idx = min(self._step_idx + 1, len(POS_STEP_SIZES) - 1)
                self.pos_step = POS_STEP_SIZES[self._step_idx]
                self.rot_step = ROT_STEP_SIZES[self._step_idx]
                self._display_dirty = True
            elif ch in ("-", "_"):
                self._step_idx = max(self._step_idx - 1, 0)
                self.pos_step = POS_STEP_SIZES[self._step_idx]
                self.rot_step = ROT_STEP_SIZES[self._step_idx]
                self._display_dirty = True

    def _on_release(self, _key):
        pass

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def poll_events(self):
        """Returns (recording_toggled, reset_requested, display_dirty) and clears."""
        with self._lock:
            toggled = self._recording_just_toggled
            reset = self._reset_requested
            dirty = self._display_dirty
            self._recording_just_toggled = False
            self._reset_requested = False
            self._display_dirty = False
            return toggled, reset, dirty

    def compute_ik_targets(self, robot):
        """Compute full (1, num_joints) target tensor using DLS IK.

        - Active arm: IK-computed joint positions
        - Active hand: grasp-interpolated finger positions
        - Inactive arm/hand: current joint positions (frozen)
        - Legs/waist: default positions
        """
        with self._lock:
            current_joints = robot.data.joint_pos[0].detach().clone()
            targets = current_joints.clone()

            # Keep legs/waist at defaults
            default_gpu = self._default_pos.to(self.device)
            non_arm_non_hand = set(range(self.num_joints))
            for side in ("left", "right"):
                non_arm_non_hand -= set(self._arm_joint_indices[side])
                non_arm_non_hand -= set(self._hand_joint_indices[side])
            for idx in non_arm_non_hand:
                targets[idx] = default_gpu[idx]

            # --- Active arm: IK ---
            arm = self.active_arm
            arm_idx = torch.tensor(
                self._arm_joint_indices[arm], device=self.device, dtype=torch.long
            )
            ee_body_idx = self._ee_body_indices[arm]
            jac_body_idx = ee_body_idx - self._jac_body_offset

            # Current EE pose
            cur_pos = robot.data.body_pos_w[0, ee_body_idx]
            cur_quat = robot.data.body_quat_w[0, ee_body_idx]

            # Desired EE pose (on CPU, move to device)
            des_pos = self.desired_pos[arm].to(self.device)
            des_quat = self.desired_quat[arm].to(self.device)

            # 6D pose error
            pos_error = des_pos - cur_pos
            # Clamp position error magnitude
            pos_norm = torch.norm(pos_error)
            if pos_norm > 0.1:
                pos_error = pos_error * (0.1 / pos_norm)

            ori_error = quat_error_to_axis_angle(des_quat, cur_quat)
            # Clamp orientation error magnitude
            ori_norm = torch.norm(ori_error)
            if ori_norm > 0.5:
                ori_error = ori_error * (0.5 / ori_norm)

            error_6d = torch.cat([pos_error, ori_error])

            # Jacobian sub-matrix: (6, 7)
            full_jac = robot.root_physx_view.get_jacobians()
            J = full_jac[0, jac_body_idx, :, :][:, arm_idx]  # (6, 7)

            # DLS solve
            lam = self.damping_lambda
            JtJ = J.T @ J
            damped = JtJ + lam * lam * torch.eye(7, device=self.device)
            delta_q = torch.linalg.solve(damped, J.T @ error_6d)

            # Clamp delta
            delta_q = torch.clamp(delta_q, -self.max_dq, self.max_dq)

            # Apply to arm joints
            arm_current = current_joints[arm_idx]
            arm_new = arm_current + delta_q
            limits = self._joint_limits.to(self.device)
            arm_new = torch.clamp(
                arm_new, limits[arm_idx, 0], limits[arm_idx, 1]
            )
            targets[arm_idx] = arm_new

            # --- Active hand: grasp ---
            self.grasp_current[arm] += self.grasp_smoothing * (
                self.grasp_target[arm] - self.grasp_current[arm]
            )
            g = self.grasp_current[arm]
            for ji, cv in zip(
                self._hand_joint_indices[arm], self._hand_closed_vals[arm]
            ):
                targets[ji] = g * cv

            # Inactive hand: also apply its current grasp value
            other = "left" if arm == "right" else "right"
            self.grasp_current[other] += self.grasp_smoothing * (
                self.grasp_target[other] - self.grasp_current[other]
            )
            g_other = self.grasp_current[other]
            for ji, cv in zip(
                self._hand_joint_indices[other], self._hand_closed_vals[other]
            ):
                targets[ji] = g_other * cv

            return targets.unsqueeze(0)  # (1, num_joints)

    def get_status_line(self, sim_step=0):
        """Formatted status string for terminal HUD."""
        with self._lock:
            arm = self.active_arm
            dp = self.desired_pos[arm]
            g = self.grasp_current[arm]
            gt = self.grasp_target[arm]
            rec = "ON" if self.recording else "OFF"
            pos_str = (
                f"({dp[0]:+.3f}, {dp[1]:+.3f}, {dp[2]:+.3f})"
                if dp is not None
                else "(n/a)"
            )
            return (
                f"[Arm: {arm.upper()}]  EE desired: {pos_str}  "
                f"Grasp: {g:.2f} (target {gt:.0f})  "
                f"Step: pos={self.pos_step} rot={self.rot_step}\n"
                f"[Recording: {rec}]  [Sim step: {sim_step}]"
            )

    def stop(self):
        if hasattr(self, "_listener"):
            self._listener.stop()
