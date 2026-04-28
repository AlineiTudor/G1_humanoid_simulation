# RH56E2 Dexterous Hand USD Files

Universal Scene Description (USD) files for the RH56E2 dexterous robotic hands with integrated force sensors.

## Overview

This directory contains the complete USD representation of the RH56E2 dexterous hands (both left and right), designed for integration with humanoid robots like the Unitree G1-29DOF in Isaac Sim.

## Directory Structure

```
Rh56E2_USD/
├── README.md                           # This file
├── RH56DFTP_L/                        # Left Hand
│   ├── RH56DFTP_L.usd                # Main USD file (8KB)
│   └── configuration/
│       ├── RH56DFTP_L_base.usd       # Visual geometry & meshes (28MB)
│       ├── RH56DFTP_L_physics.usd    # Physics properties (16KB)
│       ├── RH56DFTP_L_robot.usd      # Joint definitions (4KB)
│       └── RH56DFTP_L_sensor.usd     # Sensor configuration (4KB)
└── RH56DFTP_R/                        # Right Hand (mirror structure)
    ├── RH56DFTP_R.usd
    └── configuration/
        ├── RH56DFTP_R_base.usd
        ├── RH56DFTP_R_physics.usd
        ├── RH56DFTP_R_robot.usd
        └── RH56DFTP_R_sensor.usd
```

## Hand Specifications

### Degrees of Freedom (DOF)
- **Per Hand**: 12 DOF
- **Both Hands**: 24 DOF total
- **Fingers**: Index, middle, ring, little (3-4 joints each)
- **Thumb**: 4-5 joints
- **Palm**: Base link with mounting interface

### Force Sensors (Tactile Feedback)
- **17 sensors per hand** (34 total for both hands)
  - 3 sensors per finger (index, middle, ring, little) = 12 sensors
  - 4 sensors for thumb = 4 sensors
  - 1 palm sensor = 1 sensor

### Physical Properties
- **Coordinate System**: Z-up (standard for Isaac Sim)
- **Units**: Meters
- **Total Size**: ~28MB per hand (mostly geometry)

## USD Layer Architecture

Each hand uses a **modular layered USD structure** for flexibility and maintainability:

### 1. Main USD File (`RH56DFTP_*.usd`)
- Entry point that references all configuration layers
- Defines variants (base, joints, sensors)
- Lightweight (8KB)

### 2. Base Configuration (`*_base.usd`)
- **Largest file** (~28MB) containing all visual geometry
- STL meshes for each link
- Collision geometry (convex hulls)
- Material definitions
- Link hierarchy

### 3. Robot Configuration (`*_robot.usd`)
- Joint definitions (revolute joints)
- Joint properties:
  - Position limits (upper/lower)
  - Velocity limits
  - Effort (force/torque) limits
  - Damping and stiffness coefficients
  - Drive parameters
- Mimic joint relationships
- IsaacRobotAPI metadata

### 4. Physics Configuration (`*_physics.usd`)
- RigidBody properties for each link
- Mass and inertia tensors
- Center of mass locations
- PhysicsScene parameters:
  - Solver iterations
  - Collision detection settings (CCD)
  - GPU dynamics configuration
- Articulation root settings
- Break forces/torques

### 5. Sensor Configuration (`*_sensor.usd`)
- Force sensor mounting points
- Sensor metadata
- Integration with IsaacSensor API

## Key Components

### Link Structure

**Left Hand:**
- `l_base_link` - Mounting interface (connects to wrist)
- `left_index_1`, `left_index_2`, `left_index_3`, `left_index_4`
- `left_middle_1`, `left_middle_2`, `left_middle_3`, `left_middle_4`
- `left_ring_1`, `left_ring_2`, `left_ring_3`, `left_ring_4`
- `left_little_1`, `left_little_2`, `left_little_3`, `left_little_4`
- `left_thumb_1`, `left_thumb_2`, `left_thumb_3`, `left_thumb_4`
- `left_plam_casing` - Palm base structure
- `left_*_force_sensor_*` - 17 force sensor links

**Right Hand:**
- Mirror structure with `r_base_link` and `right_*` prefix

### Force Sensor Links

Each finger has 3 force sensors, thumb has 4, plus 1 palm sensor:
```
left_index_force_sensor_1
left_index_force_sensor_2
left_index_force_sensor_3
left_middle_force_sensor_1
...
left_thumb_force_sensor_1
left_thumb_force_sensor_2
left_thumb_force_sensor_3
left_thumb_force_sensor_4
left_plam_force_sensor
```

## Integration with G1-29DOF Robot

### Mounting Points

The hands are designed to attach to the G1 robot's wrist joints:

**Left Hand:**
- `l_base_link` → connects to `left_wrist_yaw_link`

**Right Hand:**
- `r_base_link` → connects to `right_wrist_yaw_link`

### Joint Indices (in merged G1 model)

When integrated with G1-29DOF, the hand joints occupy these indices:

```python
# Right hand joints: 34, 44, 36, 46, 37, 47, 35, 45, 38, 48, 50, 52
# Left hand joints:  29, 39, 31, 41, 32, 42, 30, 40, 33, 43, 49, 51
```

### Force Sensor Pattern Matching

Use these regex patterns in Isaac Lab's `ContactSensorCfg`:

```python
# Left hand force sensors
prim_path="/World/envs/env_.*/Robot/left_rh56e2_hand/left_.*_force_sensor.*"

# Right hand force sensors
prim_path="/World/envs/env_.*/Robot/right_rh56e2_hand/right_.*_force_sensor.*"
```

## Usage

### Viewing in Isaac Sim

```bash
# View left hand
isaacsim RH56DFTP_L/RH56DFTP_L.usd

# View right hand
isaacsim RH56DFTP_R/RH56DFTP_R.usd
```

### Loading in Isaac Lab

```python
from omni.isaac.lab.assets import ArticulationCfg

left_hand_cfg = ArticulationCfg(
    prim_path="/World/Robot/left_hand",
    spawn=sim_utils.UsdFileCfg(
        usd_path="RH56E2_USD/Rh56E2_USD/RH56DFTP_L/RH56DFTP_L.usd",
    ),
)
```

### Merging with G1 Robot

Use the provided build script to create a complete G1 + RH56E2 robot:

```bash
# From the unitree_sim_isaaclab directory
./build_rh56e2_usd.sh
```

This script:
1. Loads the G1-29DOF body USD
2. Removes any existing hand geometry
3. References the RH56E2 left and right hand USDs
4. Creates fixed joints connecting hands to wrists
5. Exports a merged USD file (~40MB)

See `../../BUILD_RH56E2_USD_INSTRUCTIONS.md` for detailed integration steps.

## Technical Details

### USD Format
- **Format**: USDC (binary USD)
- **Schema**: PhysicsRigidBodyAPI, PhysicsArticulationRootAPI
- **Extensions**: IsaacRobotAPI, IsaacSensorAPI

### Physics Properties
- **Solver Type**: TGS (Temporal Gauss-Seidel)
- **Collision Detection**: Continuous Collision Detection (CCD) enabled
- **Stabilization**: Enabled for improved joint stability
- **GPU Acceleration**: Configured for GPU dynamics

### Joint Properties
- **Type**: Revolute joints (1 DOF rotation)
- **Actuators**: Position-controlled with PD gains
- **Mimic Joints**: Some joints mimic others for synchronized movement
- **Limits**: Carefully tuned to match physical hardware

## Benefits of Layered USD Structure

1. **Modularity**: Modify physics without touching geometry
2. **Reusability**: Share base geometry across different configurations
3. **Performance**: Load only necessary layers for specific use cases
4. **Maintainability**: Edit specific aspects independently
5. **Version Control**: Track changes to individual components
6. **Composition**: Easy integration with other USD assets

## File Sizes Breakdown

| Component | Left Hand | Right Hand | Purpose |
|-----------|-----------|------------|---------|
| Main USD | 8 KB | 8 KB | Entry point, layer composition |
| Base (geometry) | 28 MB | 28 MB | STL meshes, visual appearance |
| Physics | 16 KB | 16 KB | Mass, inertia, collision |
| Robot | 4 KB | 4 KB | Joint definitions, limits |
| Sensor | 4 KB | 4 KB | Force sensor mounting |
| **Total** | **~28 MB** | **~28 MB** | **~56 MB for both hands** |

## Troubleshooting

### Issue: Hands not visible in simulation
- **Check**: USD file paths are correct
- **Check**: File sizes match expected values (28MB for base)
- **Solution**: Verify all configuration files are present

### Issue: Joints not moving
- **Check**: Joint indices in your control code
- **Check**: Joint limits and drive parameters
- **Solution**: Use `verify_rh56e2_joints.sh` to test

### Issue: Force sensors not detected
- **Check**: Regex pattern in `ContactSensorCfg`
- **Check**: Force sensor links exist in USD
- **Solution**: Verify with `verify.py` script

### Issue: Hands detached from wrists
- **Check**: Fixed joint connections in merged USD
- **Check**: Base link names match (`l_base_link`, `r_base_link`)
- **Solution**: Rebuild merged USD with `build_rh56e2_usd.sh`

## Related Documentation

- `../../BUILD_RH56E2_USD_INSTRUCTIONS.md` - Integration guide
- `../../RH56E2_SETUP_GUIDE.md` - Complete setup instructions
- `../../RH56E2_INTEGRATION_SUMMARY.md` - Technical overview
- `../../RH56E2_READY_TO_RUN.md` - Quick start guide

## Version Information

- **Hand Model**: RH56E2 (2026-1-5 version)
- **USD Schema**: PhysX 5.x compatible
- **Isaac Sim**: Compatible with Isaac Sim 2023.1.0+
- **IsaacLab**: Compatible with IsaacLab v1.0.0+

## License

These USD files are provided for use with the Unitree G1-29DOF robot in Isaac Sim simulations.

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review related documentation files
3. Verify USD structure with Isaac Sim USD viewer
4. Check joint and sensor configurations in code

---

**Last Updated**: 2026-03-17
**Created For**: Unitree G1-29DOF + RH56E2 Integration
**Environment**: NVIDIA Isaac Sim / Isaac Lab
