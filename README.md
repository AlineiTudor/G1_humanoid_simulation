# RH56E2 Tactile Sensor Project

Simulation framework for the **Unitree G1 humanoid robot** with **RH56E2 dexterous hands** and dense **GelSight-style tactile sensors**, built on [NVIDIA Isaac Lab](https://github.com/isaac-sim/IsaacLab).

The project enables:

- Spawning the G1 robot with left/right RH56E2 hands (or standalone hands)
- Dense visuotactile sensing on palm and finger regions (force field + camera-based)
- Interactive grasp data collection via keyboard joint teleop or 6-DOF Cartesian IK teleop
- Scripted grasp trajectory recording with automatic HDF5 data saving
- Contact object interaction with SDF-based force computation

<!-- TODO: Add a hero image or GIF of the simulation running -->
![Simulation Overview](docs/assets/hero.gif)

---

## Table of Contents

- [Prerequisites \& Docker Setup](#prerequisites--docker-setup)
- [Repository Structure](#repository-structure)
- [Scripts Reference](#scripts-reference)
- [Configuration Parameters](#configuration-parameters)
- [How to Launch Scripts](#how-to-launch-scripts)
- [Developer Guide](#developer-guide)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites & Docker Setup

This project runs inside a Docker container on a remote PC with an **NVIDIA GPU**. Two scripts in `docker/` handle the full setup.

### System Requirements

| Requirement | Minimum |
|---|---|
| NVIDIA GPU driver | >= 525 |
| Docker Engine | Installed with `nvidia-container-toolkit` |
| Free disk space | >= 50 GB |
| OS | Linux (tested on Ubuntu) |
| Other | `git`, `python3` |

### Step 1  Build the Docker Image

Run the setup script to clone Isaac Lab and build the Docker image:

```bash
cd docker/
./1_setup_isaaclab_docker.sh
```

This script will:

1. **Check prerequisites**  verifies NVIDIA driver version (525+), Docker, NVIDIA Container Toolkit, available disk space (50 GB+), git, and Python3
2. **Create workspace**  sets up `$HOME/isaaclab_docker_workspace`
3. **Clone Isaac Lab**  pulls the official repository
4. **Build the Docker image**  runs `./docker/container.py build` (~2045 min, ~10 GB download)
5. **Verify**  confirms the image was created

#### Variables to configure (before running)

| Variable | Default | Description |
|---|---|---|
| `WORKSPACE_DIR` | `$HOME/isaaclab_docker_workspace` | Where Isaac Lab is cloned |
| `DOCKER_PROFILE` | `"base"` | **Change to `"ros2"`** if you need ROS2 support |
| `IMAGE_SUFFIX` | `"tactile"` | Tag suffix for the built image |

> **Important:** Set `DOCKER_PROFILE="ros2"` before running if you plan to use `2_run_isaaclab_ros2_unitree.sh` (which expects a ROS2 image).

The output image will be named: `isaac-lab-{DOCKER_PROFILE}-{IMAGE_SUFFIX}:latest`
(e.g., `isaac-lab-ros2-tactile:latest`).

### Step 2  Clone Unitree Assets

Before the first container launch, clone the Unitree simulation assets:

```bash
git clone https://github.com/unitreerobotics/unitree_sim_isaaclab.git ~/tudor_unitree_isaaclab/unitree_sim_isaaclab
```

### Step 3  Launch the Container

```bash
cd docker/
./2_run_isaaclab_ros2_unitree.sh
```

- **First run:** builds and starts the container with all volume mounts.
- **Subsequent runs:** starts the container directly (the image is already built).

#### Variables to configure (before first launch)

Open `2_run_isaaclab_ros2_unitree.sh` and update these variables to match your remote PC:

| Variable | Default | What to change |
|---|---|---|
| `IMAGE_NAME` | `isaac-lab-ros2-tactile:latest` | Must match the image built in Step 1 |
| `PROJECT_DIR_1` | `/home/analog/tudor_unitree_isaaclab` | Path to your **main project directory** on the host |
| `PROJECT_DIR_2` | `/home/analog/develop` | Path to your **secondary dev directory** on the host (optional) |
| `UNITREE_REPO_DIR` | `$HOME/tudor_unitree_isaaclab/unitree_sim_isaaclab` | Path to the cloned **Unitree assets** repo |

These host directories are mounted into the container at:

| Host Path | Container Path |
|---|---|
| `PROJECT_DIR_1` | `/workspace/tudor_unitree_isaaclab` |
| `PROJECT_DIR_2` | `/workspace/develop` |
| `UNITREE_REPO_DIR` | `/workspace/unitree_assets` |

The script also mounts X11, Vulkan, and GLVND directories for GPU rendering, sets up `DISPLAY` forwarding, and auto-sources ROS2 Humble inside the container.

> **Note:** If `PROJECT_DIR_1` or `PROJECT_DIR_2` don't exist, the script warns but continues. `UNITREE_REPO_DIR` is required  the script will exit if it's missing.

---

## Repository Structure

```
tactile_sensors/
+-- config/                              # Dataclass-based configuration
¦   +-- hand_config.py                   # Hand USD structure, physics properties
¦   +-- sensor_config.py                 # Tactile sensor parameters (enable/disable, array sizes)
¦   +-- simulation_config.py             # Physics, rendering, environment settings
¦   +-- grasp_config.py                  # Grasp data collection settings
¦
+-- assets/                              # Isaac Lab asset configurations
¦   +-- rh56e2_hand.py                   # ArticulationCfg for RH56E2 hand
¦   +-- tactile_sensors.py               # VisuoTactileSensorCfg creation
¦   +-- contact_objects.py               # Test objects (cube, nut)
¦   +-- g1_robot.py                      # G1 humanoid ArticulationCfg + USD fixes
¦
+-- core/                                # Scene building logic
¦   +-- sensor_factory.py                # Factory for creating multiple sensors
¦   +-- hand_builder.py                  # Hand + sensor assembly
¦   +-- scene_composer.py                # Full scene composition (TactileHandSceneCfg)
¦   +-- sensor_init.py                   # Post-spawn sensor initialization (offsets)
¦
+-- control/                             # Motion control
¦   +-- grasp_controller.py              # Waypoint-based grasp trajectory interpolation
¦   +-- cartesian_teleop_controller.py   # 6-DOF IK keyboard teleop + grasp slider
¦
+-- data/                                # Data collection and storage
¦   +-- collectors.py                    # GraspEpisodeCollector (per-step recording)
¦   +-- savers.py                        # HDF5/numpy/JSON saving
¦
+-- utils/                               # Helper utilities
¦   +-- visualization.py                 # Tactile data visualization
¦   +-- data_processing.py               # Data filtering and analysis
¦   +-- logging.py                       # Structured logging
¦   +-- debug_prims.py                   # USD hierarchy diagnostics
¦
+-- resources/                           # USD assets (robot, hands)
¦   +-- g1_with_rh56e2_hands/            # G1 humanoid + RH56E2 hands
¦   ¦   +-- G1/g1_rh56e2.usd
¦   +-- RH56E2_R_.../RH56E2_R_...usd     # Right hand USD
¦   +-- RH56E2_L_.../RH56E2_L_...usd     # Left hand USD
¦
+-- scripts/                             # ** User-facing entry points **
¦   +-- collect_grasp_data_ik_teleop.py  # IK-based 6-DOF teleoperation
¦   +-- collect_grasp_data_keyboard.py   # Joint-level keyboard control
¦   +-- collect_grasp_data_with_waypoints.py  # Scripted grasp trajectories
¦   +-- test_g1_rh56e2_spawn.py          # G1 robot + hands spawn test
¦   +-- test_two_hands_single_env.py     # Standalone dual hands test
¦   +-- debug_usd_structure.py           # USD prim inspection utility
¦
+-- isaac_sim_util_scripts/              # Developer diagnostic scripts
¦   +-- check_sensor_recording_data.py
¦   +-- check_tacl_sensor_forces.py
¦   +-- diagnose_sensor_vs_collision.py
¦   +-- diagnose_tactile_transforms.py
¦   +-- diagnose_contact_offsets.py
¦   +-- ...
¦
+-- logs/                                # Development logs (diagnostic outputs)
```

**Key directories for users:**

- **`scripts/`**  all scripts launched directly by the user
- **`config/`**  all tunable parameters (physics, sensors, grasp collection)

**Key directories for developers:**

- **`assets/`**  how robots/sensors/objects are spawned in Isaac Lab
- **`core/`**  how scenes are composed from assets
- **`control/`**  motion controllers (IK, waypoints)

---

## Scripts Reference

All scripts live in `scripts/` and are the primary user entry points. Each script follows the Isaac Lab pattern: parse arguments, create `AppLauncher`, then import simulation modules.

### `collect_grasp_data_ik_teleop.py`

**6-DOF Cartesian IK teleoperation** with grasp control and data recording. Move the arm end-effector in Cartesian space using keyboard controls, close/open fingers, and record episodes to HDF5.

<!-- TODO: Add GIF showing IK teleop in action -->
![IK Teleop Demo](docs/assets/ik_teleop.gif)

| Argument | Type | Default | Description |
|---|---|---|---|
| `--output_dir` | str | from config | Output directory for HDF5 episode files |
| `--object_type` | str | `nut` | Object to grasp: `nut` (SDF mesh, supports force field) or `cube` |
| `--no_camera` | flag |  | Disable scene camera recording |
| `--save_tactile_rgb` | flag |  | Save tactile RGB images (large files) |
| `--pos_step` | float | `0.005` | Position increment per keypress (meters) |
| `--rot_step` | float | `0.02` | Orientation increment per keypress (radians) |
| `--damping` | float | `0.05` | DLS damping factor (lambda) for IK solver |
| `--enable_cameras` | flag |  | Isaac Lab flag to enable camera rendering |

**Keyboard controls:**

| Key | Action |
|---|---|
| `W` / `S` | Move end-effector forward / backward (X) |
| `A` / `D` | Move end-effector left / right (Y) |
| `Up` / `Down` | Move end-effector up / down (Z) |
| `J` / `L` | Rotate around X axis |
| `I` / `K` | Rotate around Y axis |
| `U` / `O` | Rotate around Z axis |
| `G` | Close grasp (gradual) |
| `R (key)` | Open grasp (gradual) |
| `Tab` | Switch active arm (left / right) |
| `+` / `-` | Increase / decrease step size |
| `Space` | Toggle recording on/off |
| `Backspace` | Reset robot and object to default pose |
| `Q` / `Esc` | Quit |

**Features:**
- Real-time force readout from tactile sensors in terminal HUD
- Auto-saves in-progress recording on exit (Ctrl+C safe)
- Detailed tactile sensor diagnostics every 200 steps
- Runs a force-push pipeline test at startup to validate sensor setup

---

### `collect_grasp_data_keyboard.py`

**Joint-level keyboard teleop** with data recording. Select individual joints and nudge their position targets directly.

| Argument | Type | Default | Description |
|---|---|---|---|
| `--output_dir` | str | from config | Output directory for HDF5 episode files |
| `--object_type` | str | `nut` | Object to grasp: `nut` or `cube` |
| `--no_camera` | flag |  | Disable scene camera |
| `--save_tactile_rgb` | flag |  | Save tactile RGB images |
| `--step_size` | float | `0.02` | Initial position increment per keypress (radians) |

**Keyboard controls:**

| Key | Action |
|---|---|
| `Right` / `Left` | Next / previous joint |
| `Up` / `Down` | Increment / decrement joint position |
| `+` / `-` | Increase / decrease step size |
| `R` | Toggle recording |
| `Backspace` | Reset all joints to default |
| `Q` / `Esc` | Quit |

**Step sizes cycle through:** 0.005, 0.01, 0.02, 0.05, 0.1 rad

---

### `collect_grasp_data_with_waypoints.py`

**Scripted grasp trajectory collection.** Runs a predefined reach ? grasp ? lift trajectory automatically and records data to HDF5.

| Argument | Type | Default | Description |
|---|---|---|---|
| `--num_episodes` | int | `5` | Number of grasp episodes to record |
| `--output_dir` | str | from config | Output directory for HDF5 episode files |
| `--object_type` | str | `nut` | Object to grasp: `nut` or `cube` |
| `--no_camera` | flag |  | Disable camera recording |
| `--save_tactile_rgb` | flag |  | Save tactile RGB images |

This script is useful for **batch data collection** without manual intervention.

---

### `test_g1_rh56e2_spawn.py`

**Spawn test** for the G1 humanoid robot with both RH56E2 hands attached. Verifies that the robot USD loads correctly, articulation joints are discovered, and hands are properly connected to the robot's kinematic tree.

No custom arguments beyond Isaac Lab's standard `AppLauncher` flags.

Prints detailed diagnostics: joint names, body names, hand placement, articulation roots, and fixed joint connectivity.

---

### `test_two_hands_single_env.py`

**Standalone dual hands test.** Spawns left and right RH56E2 hands as independent articulations (no robot body) in 2 environments. Useful for testing hand-only scenarios.

No custom arguments beyond Isaac Lab's standard `AppLauncher` flags.

---

### `debug_usd_structure.py`

**USD prim inspection utility.** Spawns a single hand and prints the full USD hierarchy, focusing on elastomer prims, collision APIs, and mesh collision properties. Useful for debugging sensor prim path issues.

---

## Configuration Parameters

Configuration is defined via Python dataclasses in `config/`. Users typically don't need to modify these files directly  the scripts instantiate configs with sensible defaults, and key parameters are exposed as CLI arguments. However, for advanced use or custom scripts, these are the main knobs.

### `config/simulation_config.py`  SimulationConfig

| Parameter | Default | Description |
|---|---|---|
| `num_envs` | `1` | Number of parallel simulation environments |
| `spawn_mode` | `"standalone"` | `"standalone"` (hands only) or `"g1_robot"` (full humanoid) |
| `robot_usd_path` | (see file) | Path to the G1 robot USD inside the container |
| `hand_types` | `("left", "right")` | Which hands to spawn |
| `physics_dt` | `0.005` | Physics timestep (200 Hz) |
| `fix_root_link` | `True` | Pin robot pelvis in place (for tabletop manipulation) |
| `device` | `"cuda:0"` | Simulation device |

### `config/sensor_config.py`  SensorConfig

| Parameter | Default | Description |
|---|---|---|
| `sensors_enabled` | `True` | Global sensor enable |
| `palm_enabled` | `True` | Enable palm tactile sensor |
| `index_enabled` | `True` | Enable index finger sensors (3 regions) |
| `middle_enabled`, `thumb_enabled`, `little_enabled`, `ring_enabled` | `True` | Per-finger enable flags |
| `tactile_normal_offset` | `0.005` | Offset sensing points outward past collision geometry (meters) |
| `normal_contact_stiffness` | `2.0` | Force = stiffness * penetration_depth |
| `friction_coefficient` | `2.0` | Tangential friction |
| `contact_object_prim_path_expr` | `"{ENV_REGEX_NS}/grasp_object"` | Prim path pattern for the contact object |

**Tactile array sizes** (rows, cols) define the resolution of each sensing region. Defaults:
- Palm: (8, 14) = 112 taxels
- Index proximal: (10, 8), middle: (12, 8), distal: (3, 3)
- Similar for other fingers

### `config/grasp_config.py`  GraspDataCollectionConfig

| Parameter | Default | Description |
|---|---|---|
| `num_episodes` | `10` | Number of grasp episodes |
| `object_type` | `"cube"` | `"cube"` or `"nut"`  **use `nut` for force field** |
| `object_pos` | `(0.5, 0.0, 0.39)` | Object spawn position |
| `table_pos` / `table_size` | `(0.5, 0.0, 0.35)` / `(0.6, 0.8, 0.02)` | Table placement |
| `output_dir` | `/workspace/tudor_unitree_isaaclab/grasp_data` | Where HDF5 files are saved |
| `save_tactile_rgb` | `False` | Save tactile camera RGB (large files) |
| `save_every_n_steps` | `1` | Data saving frequency (1 = every physics step) |
| `randomize_object_pos` | `False` | Add noise to object position per episode |
| `camera_enabled` | `True` | Enable scene camera |

### `config/hand_config.py`  HandConfig

| Parameter | Default | Description |
|---|---|---|
| `hand_side` | `"right"` | `"right"` or `"left"` |
| `compliant_contact_stiffness` | `100.0` | Elastomer stiffness for deformable contact |
| `compliant_contact_damping` | `10.0` | Elastomer damping |
| `finger_names` | `("index", "middle", "little", "thumb")` | Fingers present on the hand |

### USD Resource Paths

All USD models (robot, hands) are loaded from the `resources/` directory inside the package. Paths are resolved relative to the package root, so they work regardless of where the package is mounted in the container.

| Resource | Location | Used by |
|---|---|---|
| G1 robot with RH56E2 hands | `resources/g1_with_rh56e2_hands/G1/g1_rh56e2.usd` | `SimulationConfig.robot_usd_path` |
| Right hand USD | `resources/RH56E2_R_.../RH56E2_R_...usd` | `HandConfig` (auto-selected) |
| Left hand USD | `resources/RH56E2_L_.../RH56E2_L_...usd` | `HandConfig` (auto-selected) |

The `output_dir` for grasp data defaults to `/workspace/tudor_unitree_isaaclab/grasp_data` and can be overridden via `--output_dir` at launch time.

---

## How to Launch Scripts

All scripts are launched from the **Isaac Lab root directory** inside the container using `isaaclab.sh`:

```bash
./isaaclab.sh -p <path_to_script> [script_args] [isaaclab_args]
```

### Examples

**IK Teleop** (most common workflow):

```bash
./isaaclab.sh -p /workspace/tudor_unitree_isaaclab/isaaclab_tactile_proj/rh56e2_hands_tactile_sensors/gelsight_tactile_sensors/scripts/tactile_sensors/scripts/collect_grasp_data_ik_teleop.py \
  --enable_cameras \
  --object_type nut \
  --no_camera
```

**Keyboard Joint Teleop:**

```bash
./isaaclab.sh -p /workspace/tudor_unitree_isaaclab/isaaclab_tactile_proj/rh56e2_hands_tactile_sensors/gelsight_tactile_sensors/scripts/tactile_sensors/scripts/collect_grasp_data_keyboard.py \
  --enable_cameras \
  --object_type nut \
  --output_dir /workspace/tudor_unitree_isaaclab/grasp_data/keyboard_sessions
```

**Scripted Waypoint Collection:**

```bash
./isaaclab.sh -p /workspace/tudor_unitree_isaaclab/isaaclab_tactile_proj/rh56e2_hands_tactile_sensors/gelsight_tactile_sensors/scripts/tactile_sensors/scripts/collect_grasp_data_with_waypoints.py \
  --enable_cameras \
  --object_type nut \
  --num_episodes 20 \
  --output_dir /workspace/tudor_unitree_isaaclab/grasp_data/waypoint_runs
```

**Spawn Test (verify robot loads correctly):**

```bash
./isaaclab.sh -p /workspace/tudor_unitree_isaaclab/isaaclab_tactile_proj/rh56e2_hands_tactile_sensors/gelsight_tactile_sensors/scripts/tactile_sensors/scripts/test_g1_rh56e2_spawn.py \
  --enable_cameras
```

### Notes on Launch Arguments

- `--enable_cameras` is an **Isaac Lab flag** (not script-specific) that enables camera rendering in the simulation. Required for tactile camera sensors.
- `--no_camera` is a **script flag** that disables the *scene overview camera* recording (not the tactile sensors). Use it to reduce overhead when you don't need scene camera images.
- Script-specific arguments (like `--object_type`, `--output_dir`) come before Isaac Lab arguments, but argument order is generally flexible.

---

## Developer Guide

### Architecture Overview

The codebase follows a layered architecture:

```
config/    ?  User-facing parameters (WHAT to simulate)
assets/    ?  Isaac Lab spawn configurations (HOW to create entities)
core/      ?  Scene composition (HOW to combine entities into a scene)
control/   ?  Motion controllers (HOW to move the robot)
data/      ?  Episode recording & saving (WHAT to record)
scripts/   ?  Entry points that wire everything together
```

Each layer depends only on the layers above it. This makes it straightforward to swap components (e.g., different controller, different sensor config) without touching unrelated code.

### Creating a Custom Script

1. **Copy an existing script** as a template (e.g., `collect_grasp_data_ik_teleop.py`)

2. **Follow the Isaac Lab import pattern**  all Isaac Lab and `tactile_sensors` imports must happen AFTER `AppLauncher`:

```python
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
# add your args
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# NOW import Isaac Lab and tactile_sensors modules
from tactile_sensors.config import SimulationConfig, SensorConfig
from tactile_sensors.core.scene_composer import create_grasp_scene
# ...
```

3. **Configure the simulation** by instantiating config dataclasses:

```python
sim_cfg = SimulationConfig(
    spawn_mode="g1_robot",
    hand_types=("left", "right"),
    num_envs=1,
    fix_root_link=True,
)
sensor_cfg = SensorConfig(palm_enabled=True, index_enabled=True, ...)
grasp_cfg = GraspDataCollectionConfig(object_type="nut", ...)
```

4. **Create the scene** using the scene composer:

```python
scene_cfg = create_grasp_scene(sim_cfg, sensor_cfg, grasp_cfg)
scene = InteractiveScene(scene_cfg)
```

5. **Initialize sensors** (required for tactile cameras):

```python
# Step a few times for renderer warm-up
for _ in range(5):
    scene.write_data_to_sim()
    sim_context.step()
    scene.update(sim_context.get_physics_dt())

# Capture baseline tactile images
for name in tactile_sensor_names:
    sensor = scene[name]
    if hasattr(sensor, "get_initial_render"):
        sensor.get_initial_render()

# Enable force field updates
scene.cfg.lazy_sensor_update = False

# Apply normal offset to push sensing points past collision geometry
from tactile_sensors.core.sensor_init import apply_offset_to_all_sensors
apply_offset_to_all_sensors(scene, tactile_sensor_names, sensor_cfg.tactile_normal_offset)
```

6. **Run the simulation loop:**

```python
while simulation_app.is_running():
    robot.set_joint_position_target(targets)
    scene.write_data_to_sim()
    sim_context.step()
    scene.update(sim_context.get_physics_dt())

    # Access tactile data
    for name in tactile_sensor_names:
        sensor_data = scene[name].data
        normal_force = sensor_data.tactile_normal_force  # (num_envs, num_taxels)
        depth = sensor_data.penetration_depth
```

### Building RL on Top of This Framework

The project provides the building blocks for reinforcement learning with tactile feedback:

**Observation space components available:**
- `robot.data.joint_pos`  joint positions (num_envs, num_joints)
- `robot.data.joint_vel`  joint velocities
- `sensor.data.tactile_normal_force`  per-taxel normal force
- `sensor.data.penetration_depth`  per-taxel penetration
- `obj.data.root_pos_w`, `obj.data.root_quat_w`  object pose

**Action space:**
- Joint position targets via `robot.set_joint_position_target()`
- Cartesian IK targets via `CartesianTeleopController` (for Cartesian action spaces)

**To build an RL environment:**

1. Create a new script or Isaac Lab environment class
2. Use `create_grasp_scene()` for the scene setup
3. Define `_get_observations()` pulling from robot/sensor/object data
4. Define `_compute_rewards()` using tactile force readings, object pose, etc.
5. Use `num_envs > 1` in `SimulationConfig` for parallel rollouts

**Important considerations:**
- The `nut` object type uses SDF meshes required for the force field  always use `object_type="nut"` for force-based rewards
- Tactile sensors add computational overhead  start with palm-only (`palm_enabled=True`, others `False`) for faster prototyping
- Physics at 200 Hz (dt=0.005) is recommended for stable tactile contact

### Key Implementation Details

**Force field requirements:**
- Contact objects **must** have SDF collision meshes (USD meshes, not procedural shapes)
- The `cube` object is procedural (CuboidCfg) and does **not** support force field
- The `nut` object (factory_nut_m16.usd) has SDF meshes and works with the force field
- Compliant contact material allows controlled penetration (~1mm) for tactile overlap

**Sensor prim path structure:**
```
{ENV_REGEX_NS}/Robot/{side}_hand/{side}_{finger}_force_sensor_{N}/elastomer_{side}_{finger}_tip
```

**Mimic joints:**
The RH56E2 hand uses PhysX MimicJointAPI  only master joints (e.g., `index_1_joint`) need to be commanded. Follower joints (`_2`, `_3`, `_4`) follow automatically.

---

## Troubleshooting

### Common Issues

**"Image not found" when running `2_run_isaaclab_ros2_unitree.sh`**
- Make sure you built the image with `DOCKER_PROFILE="ros2"` in `1_setup_isaaclab_docker.sh`
- Check image exists: `docker images | grep isaac-lab-ros2-tactile`

**Force field shows zero forces**
- Use `--object_type nut` (the cube doesn't have SDF meshes)
- Check that `contact_object_prim_path_expr` matches the actual object prim path
- Verify `tactile_normal_offset` is > 0 (default 0.005m) to push sensing points past collision geometry

**"Tactile sensor camera not found"**
- Ensure the hand USD contains camera prims at the expected paths under each elastomer link
- Use `debug_usd_structure.py` to inspect the actual USD hierarchy

**Hands appear at world origin instead of on the robot**
- Check that the robot USD has valid FixedJoints connecting wrist links to hand roots
- Run `test_g1_rh56e2_spawn.py` for detailed hand placement diagnostics

**Simulation crashes with GPU memory errors**
- Reduce `gpu_collision_stack_size` in `SimulationConfig`
- Disable some finger sensors to reduce taxel count
- Run with fewer environments (`num_envs=1`)

---

<!-- TODO: Add more screenshots/GIFs as needed -->

<!-- Example placeholders: -->
<!-- ![Grasp Data Collection](docs/assets/grasp_collection.gif) -->
<!-- ![Tactile Force Visualization](docs/assets/tactile_forces.png) -->
<!-- ![Robot Spawn Test](docs/assets/g1_spawn.png) -->
