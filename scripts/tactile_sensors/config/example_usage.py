"""
Example Usage of HandConfig and SensorConfig
=============================================
Demonstrates how to use the configuration classes.

IMPORTANT: This script runs STANDALONE without Isaac Lab imports.
To use get_sensor_parameters() (which requires Isaac Lab), see the
example at the bottom of this file.

Run with:
    cd /path/to/RH56E2_tactile_proj/config
    python example_usage.py
"""

from hand_config import HandConfig
from sensor_config import SensorConfig


def example_basic_usage():
    """Basic usage: Create configs for right and left hands."""

    # Create config for RIGHT hand (default)
    right_hand = HandConfig()
    print(f"Right hand USD: {right_hand.usd_full_path}")
    print(f"Palm link: {right_hand.get_palm_link_name()}")

    # Create config for LEFT hand (override hand_side)
    left_hand = HandConfig(hand_side="left")
    print(f"Left hand palm link: {left_hand.get_palm_link_name()}")

    # Create sensor config (same for both hands)
    sensors = SensorConfig()
    print(f"Palm enabled: {sensors.palm_enabled}")
    print(f"Palm array size: {sensors.palm_array_size}")


def example_get_all_locations():
    """Show all sensor locations for a hand."""

    hand = HandConfig()

    # Get all possible sensor locations with their USD link names
    locations = hand.get_all_sensor_locations()

    print("\nAll sensor locations:")
    for location, link_name in locations.items():
        print(f"  {location:25s} -> {link_name}")


def example_sensor_parameters():
    """Get sensor parameters for specific locations."""

    hand = HandConfig()
    sensors = SensorConfig()

    # Show sensor config attributes (without calling get_sensor_parameters)
    # Note: get_sensor_parameters() requires Isaac Lab to be imported
    print(f"\nSensor configuration:")
    print(f"  Palm array size: {sensors.palm_array_size}")
    print(f"  Tactile margin: {sensors.tactile_margin}")
    print(f"  Enable camera: {sensors.enable_camera_tactile}")
    print(f"  Enable force field: {sensors.enable_force_field}")
    print(f"  Normal contact stiffness: {sensors.normal_contact_stiffness}")

    # Example: Override parameters for specific location
    sensors_custom = SensorConfig(
        per_location_overrides={
            "palm": {"normal_contact_stiffness": 5.0},  # Stiffer palm
            "thumb_tip": {"friction_coefficient": 3.0},  # Grippier thumb tip
        }
    )

    print(f"\nCustom sensor config with overrides:")
    print(f"  Per-location overrides: {sensors_custom.per_location_overrides}")
    print(f"  Palm override stiffness: {sensors_custom.per_location_overrides['palm']['normal_contact_stiffness']}")


def example_enabled_locations():
    """Get only enabled sensor locations."""

    hand = HandConfig()

    # Sensor config with some fingers disabled
    sensors = SensorConfig(
        palm_enabled=True,
        index_enabled=True,
        middle_enabled=False,  # Disabled
        little_enabled=False,  # Disabled
        thumb_enabled=True,
    )

    enabled = sensors.get_all_enabled_locations(hand)
    print(f"\nEnabled sensor locations ({len(enabled)} total):")
    for loc in enabled:
        print(f"  - {loc}")


def example_physics_material_paths():
    """Show physics material paths for each location."""

    hand = HandConfig()
    locations = hand.get_all_sensor_locations().keys()

    print("\nPhysics material paths:")
    for location in locations:
        material_path = hand.get_physics_material_path(location)
        print(f"  {location:25s} -> {material_path}")


def example_isaac_lab_usage():
    """
    Example of using configs inside an Isaac Lab script.

    This shows how to properly use get_sensor_parameters() which requires
    Isaac Lab to be imported AFTER SimulationApp is instantiated.

    COPY THIS PATTERN to your scripts!
    """
    print("\n" + "=" * 70)
    print("Isaac Lab Script Usage Pattern")
    print("=" * 70)
    print("""
# In your Isaac Lab script:

import argparse
from isaaclab.app import AppLauncher

# Parse arguments FIRST
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Launch app BEFORE any Isaac Lab imports
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# NOW it's safe to import configs and use get_sensor_parameters()
from RH56E2_tactile_proj.config import HandConfig, SensorConfig

hand_cfg = HandConfig(hand_side="right")
sensor_cfg = SensorConfig()

# This works now because SimulationApp is running
palm_params = sensor_cfg.get_sensor_parameters("palm")
print(f"Palm render config: {palm_params['render_cfg']}")  # Uses GELSIGHT_R15_CFG

# Create your scene, run simulation, etc...
# ...

# Always close at the end
simulation_app.close()
    """)


if __name__ == "__main__":
    print("=" * 70)
    print("HandConfig and SensorConfig Usage Examples")
    print("=" * 70)

    example_basic_usage()
    example_get_all_locations()
    example_sensor_parameters()
    example_enabled_locations()
    example_physics_material_paths()
    example_isaac_lab_usage()

    print("\n" + "=" * 70)
    print("Done!")
