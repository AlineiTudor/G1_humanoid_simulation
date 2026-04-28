"""
Simulation Configuration
========================
Defines physics, rendering, and environment settings for Isaac Lab simulation.

Usage:
    # Default config (2 hands, standard physics)
    sim_cfg = SimulationConfig()

    # Custom config (1 hand, faster timestep)
    sim_cfg = SimulationConfig(
        num_hands=1,
        hand_types=("right",),
        physics_dt=0.01  # 100 Hz instead of 200 Hz
    )
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class SimulationConfig:
    """Configuration for Isaac Lab simulation parameters."""

    # ========================================================================
    # Environment Configuration
    # ========================================================================
    # Number of parallel simulation environments (for parallel training)
    # - Single robot: num_envs = 1
    # - Parallel training with N robots: num_envs = N
    num_envs: int = 1

    # =======================================================================
    # Spawn Mode Configuration
    # =======================================================================
    # "standalone" = spawn hands only
    # "g1_robot" = spawn G1 humanoid robot with RH56E2 hands attached. USD must
    #               contain hands a priori
    spawn_mode: str = "standalone"

    # Robot USD path
    robot_usd_path: str = "/workspace/tudor_unitree_isaaclab/isaaclab_tactile_proj/" \
                            "rh56e2_hands_tactile_sensors/gelsight_tactile_sensors/scripts/tactile_sensors/" \
                            "resources/g1_with_rh56e2_hands/G1/g1_rh56e2.usd"

    # Source prim path inside the robot USD file
    # This is the root prim of the robot in the USD (NOT the scene prim path).
    # Needed when the USD's defaultPrim is wrong (e.g., set to /World instead of the robot).
    robot_source_prim_path: str = "/g1_29dof_with_hand_rev_1_0"

    # Robot prim name in scene
    robot_prim_name = "Robot"

    # Path from robot root to each hand's mount point in the USD hierarchy.
    # The hand's force sensor links (e.g., left_palm_force_sensor) are children
    # of this mount prim. Derived from the robot USD's kinematic tree.
    robot_hand_mount_paths: Dict[str, str] = field(default_factory=lambda: {
        "left": "left_hand",
        "right": "right_hand",
    })

    # Types of hands to spawn in EACH environment
    # For humanoid robot: ("left", "right") spawns both hands on same robot
    # For single hand testing: ("right",) or ("left",)
    hand_types: Tuple[str, ...] = ("left", "right")

    # Distance between ENVIRONMENTS in scene (meters)
    # Used for parallel training visualization (env_0, env_1, env_2, ...)
    env_spacing: float = 2.0

    # Distance between hands WITHIN same environment (meters)
    # Only used for standalone hand testing (not needed for humanoid robot)
    hand_spacing: float = 0.5

    # ========================================================================
    # Physics Configuration
    # ========================================================================
    # Physics timestep (seconds)
    # Smaller = more accurate but slower
    # 0.005 = 200 Hz (good for contact-rich tactile sensing)
    # 0.01 = 100 Hz (faster, less accurate)
    physics_dt: float = 0.005

    # Rendering timestep (seconds)
    # Can be slower than physics (e.g., render every 2nd physics step)
    # Set to 0 to render at physics rate
    rendering_dt: float = 0.0

    # GPU collision stack size (bytes)
    # Increase for complex contact scenarios (many tactile sensors!)
    # 2^28 = 268 MB (recommended for tactile sensing)
    gpu_collision_stack_size: int = 2**28

    # PhysX solver iterations
    # Higher = more accurate contacts but slower
    solver_position_iteration_count: int = 12
    solver_velocity_iteration_count: int = 1

    # Fix the robot's root link (pelvis) in place
    # True = pelvis pinned to spawn position (good for hand/tactile testing)
    # False = robot is free to move (needed for locomotion, whole-body control)
    fix_root_link: bool = False

    # Enable GPU dynamics
    # Set to False if you have GPU memory issues
    enable_gpu_dynamics: bool = True

    # ========================================================================
    # Device Configuration
    # ========================================================================
    # Compute device ("cuda:0", "cuda:1", "cpu")
    device: str = "cuda:0"

    # ========================================================================
    # Visualization & Data Saving
    # ========================================================================
    # Enable real-time visualization (GUI)
    # Set to False for headless training
    enable_visualization: bool = True

    # Save tactile sensor data to disk
    save_tactile_data: bool = False

    # Save tactile visualizations (RGB images, force field plots)
    save_tactile_viz: bool = False

    # Output directory for saved data
    output_dir: str = "output"

    # Save frequency (steps)
    # E.g., save_every_n_steps=10 means save every 10th step
    save_every_n_steps: int = 10

    # ========================================================================
    # Camera Configuration (for visualization only, NOT tactile cameras)
    # ========================================================================
    # Initial camera position (eye) and target (lookat)
    camera_eye: Tuple[float, float, float] = (1.2, 0.0, 0.6)
    camera_target: Tuple[float, float, float] = (0.0, 0.0, 0.4)

    # ========================================================================
    # Validation & Helper Methods
    # ========================================================================

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Check is spawn_mode is valid
        valid_modes = {"standalone", "g1_robot"}
        if self.spawn_mode not in valid_modes:
            raise ValueError(
                f"Invalid spawn_mode '{self.spawn_mode}'. Must be one of {valid_modes}"
            )
        # Check that hand_types contains valid values
        valid_hand_types = {"left", "right"}
        for hand_type in self.hand_types:
            if hand_type not in valid_hand_types:
                raise ValueError(
                    f"Invalid hand type '{hand_type}'. Must be 'left' or 'right'. "
                    f"Got hand_types={self.hand_types}"
                )

        # Check that hand_types has at least one hand
        if len(self.hand_types) == 0:
            raise ValueError("hand_types must contain at least one hand type")

        # Check num_envs is positive
        if self.num_envs < 1:
            raise ValueError(f"num_envs must be at least 1, got {self.num_envs}")

        # Check device format
        valid_devices = {"cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"}
        if self.device not in valid_devices and not self.device.startswith("cuda:"):
            raise ValueError(
                f"Invalid device '{self.device}'. Must be 'cpu' or 'cuda:N' where N is GPU index."
            )

    def get_hand_positions(self) -> list[Tuple[float, float, float]]:
        """
        Returns initial positions for each hand WITHIN a single environment.

        For humanoid robot: Returns positions for left/right hands on the robot.
        For standalone testing: Returns spaced positions along X-axis.

        Returns:
            List of (x, y, z) positions, one per hand in hand_types

        Examples:
            - hand_types=("left", "right"): returns [(-0.25, 0, 0.4), (0.25, 0, 0.4)]
            - hand_types=("right",): returns [(0.0, 0, 0.4)]
        """

        if self.spawn_mode == "g1_robot":
            return None

        positions = []
        num_hands = len(self.hand_types)

        # Center hands around origin, spaced by hand_spacing
        start_x = -(num_hands - 1) * self.hand_spacing / 2.0

        for i in range(num_hands):
            x = start_x + i * self.hand_spacing
            y = 0.0
            z = 0.4  # 40cm above ground
            positions.append((x, y, z))

        return positions

    @property
    def num_hands_per_env(self) -> int:
        """Returns number of hands in each environment."""
        return len(self.hand_types)

    @property
    def total_hands(self) -> int:
        """Returns total number of hands across all environments."""
        return self.num_envs * len(self.hand_types)

    def should_save_data(self) -> bool:
        """Returns True if any data saving is enabled."""
        return self.save_tactile_data or self.save_tactile_viz

