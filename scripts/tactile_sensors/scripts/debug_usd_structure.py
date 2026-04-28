#!/usr/bin/env python3
"""Debug USD structure to see what prims exist and their properties."""

import argparse
import sys
from pathlib import Path

# Add package to Python path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PACKAGE_PARENT = PROJECT_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Now import after AppLauncher
from tactile_sensors.config import HandConfig, SimulationConfig, SensorConfig
from tactile_sensors.core import create_tactile_scene
from isaaclab.scene import InteractiveScene
import isaaclab.sim as sim_utils
import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics


def print_prim_info(prim, indent=0):
    """Print information about a prim."""
    prefix = "  " * indent
    prim_type = prim.GetTypeName()

    # Check if it has collision
    has_collision = UsdPhysics.CollisionAPI(prim)
    has_mesh_collision = UsdPhysics.MeshCollisionAPI(prim)

    print(f"{prefix}{prim.GetName()} [{prim_type}]")
    if has_collision:
        print(f"{prefix}  ? Has CollisionAPI")
    if has_mesh_collision:
        print(f"{prefix}  ? Has MeshCollisionAPI")


def explore_hand_structure(hand_prim_path: str):
    """Explore and print the structure of a hand prim."""
    stage = omni.usd.get_context().get_stage()
    hand_prim = stage.GetPrimAtPath(hand_prim_path)

    if not hand_prim.IsValid():
        print(f"ERROR: Hand prim not found at {hand_prim_path}")
        return

    print(f"\n{'='*80}")
    print(f"Hand structure at: {hand_prim_path}")
    print(f"{'='*80}\n")

    # Look for elastomer prims
    print("Searching for elastomer prims...")
    elastomer_prims = []

    for prim in Usd.PrimRange(hand_prim):
        prim_name = prim.GetName()
        if "elastomer" in prim_name.lower():
            elastomer_prims.append(prim)
            # Get path relative to hand root
            full_path = prim.GetPath()
            relative_path = str(full_path).replace(hand_prim_path + "/", "")

            print(f"\nFound elastomer: {prim_name}")
            print(f"  Full path: {full_path}")
            print(f"  Relative path: {relative_path}")
            print(f"  Type: {prim.GetTypeName()}")

            # Check for collision APIs
            has_collision = UsdPhysics.CollisionAPI(prim)
            has_mesh_collision = UsdPhysics.MeshCollisionAPI(prim)

            print(f"  Has CollisionAPI: {bool(has_collision)}")
            print(f"  Has MeshCollisionAPI: {bool(has_mesh_collision)}")

            # Check children
            children = prim.GetChildren()
            if children:
                print(f"  Children:")
                for child in children:
                    child_collision = UsdPhysics.CollisionAPI(child)
                    child_mesh_collision = UsdPhysics.MeshCollisionAPI(child)
                    collision_marker = " [COLLISION]" if child_collision or child_mesh_collision else ""
                    print(f"    - {child.GetName()} [{child.GetTypeName()}]{collision_marker}")

    # Also search for ALL collision prims
    print("\n" + "="*80)
    print("Searching for ALL collision prims in hand...")
    print("="*80)

    for prim in Usd.PrimRange(hand_prim):
        has_collision = UsdPhysics.CollisionAPI(prim)
        has_mesh_collision = UsdPhysics.MeshCollisionAPI(prim)

        if has_collision or has_mesh_collision:
            full_path = prim.GetPath()
            relative_path = str(full_path).replace(hand_prim_path + "/", "")
            prim_name = prim.GetName()

            print(f"\n? {prim_name}")
            print(f"  Path: {relative_path}")
            print(f"  Type: {prim.GetTypeName()}")
            print(f"  CollisionAPI: {bool(has_collision)}")
            print(f"  MeshCollisionAPI: {bool(has_mesh_collision)}")

            # Check if it's under an elastomer
            if "elastomer" in str(full_path).lower():
                print(f"  ? This is under an elastomer prim!")

    if not elastomer_prims:
        print("\n??  No elastomer prims found!")
        print("\nShowing first few levels of hand structure:")
        print_prim_hierarchy(hand_prim, max_depth=3)


def print_prim_hierarchy(prim, depth=0, max_depth=3):
    """Print prim hierarchy up to max_depth."""
    if depth > max_depth:
        return

    indent = "  " * depth
    prim_type = prim.GetTypeName()
    print(f"{indent}{prim.GetName()} [{prim_type}]")

    for child in prim.GetChildren():
        print_prim_hierarchy(child, depth + 1, max_depth)


def main():
    # Create minimal config
    sim_cfg = SimulationConfig(
        num_envs=1,
        hand_types=("right",),
    )
    sensor_cfg = SensorConfig(palm_enabled=True)

    # Create simulation context
    sim_context = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(
            dt=0.01,
            device="cuda:0",
        )
    )

    # Create scene
    scene_cfg = create_tactile_scene(sim_cfg, sensor_cfg)
    scene = InteractiveScene(scene_cfg)

    # Explore the hand structure
    hand_prim_path = "/World/envs/env_0/right_hand"
    explore_hand_structure(hand_prim_path)

    print("\n" + "="*80)
    print("Debug complete. Press Ctrl+C to exit.")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
    simulation_app.close()
