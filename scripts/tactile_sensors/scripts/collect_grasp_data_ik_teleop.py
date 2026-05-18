#!/usr/bin/env python3
"""Cartesian IK teleop with grasp control and data recording for G1 + RH56E2.

Move the arm end-effector in 6-DOF via keyboard (WASD + arrows),
close/open the hand with G/R, switch arms with Tab, and record to HDF5.

Usage:
    python collect_grasp_data_ik_teleop.py --output_dir ik_teleop_data
"""

import argparse
import sys
import os
import time
from pathlib import Path

# Path setup (same pattern as collect_grasp_data_with_waypoints.py)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PACKAGE_PARENT = PROJECT_ROOT.parent

if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Cartesian IK teleop with data recording")
parser.add_argument("--output_dir", type=str, default=None,
                    help="Output directory for HDF5 files")
parser.add_argument("--object_type", type=str, default="nut",
                    choices=["cube", "nut"],
                    help="Object to grasp (nut=SDF mesh, required for force field)")
parser.add_argument("--no_camera", action="store_true",
                    help="Disable camera recording")
parser.add_argument("--save_tactile_rgb", action="store_true",
                    help="Save tactile RGB images (large)")
parser.add_argument("--pos_step", type=float, default=0.005,
                    help="Position increment per keypress (meters)")
parser.add_argument("--rot_step", type=float, default=0.02,
                    help="Orientation increment per keypress (radians)")
parser.add_argument("--damping", type=float, default=0.05,
                    help="DLS damping factor (lambda)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# === All Isaac Lab imports AFTER AppLauncher ===
import torch

from tactile_sensors.config import SimulationConfig, SensorConfig
from tactile_sensors.config.grasp_config import GraspDataCollectionConfig
from tactile_sensors.core.scene_composer import create_grasp_scene
from tactile_sensors.data.collectors import GraspEpisodeCollector
from tactile_sensors.data.savers import save_episode_hdf5, save_metadata
from tactile_sensors.control.cartesian_teleop_controller import CartesianTeleopController
from tactile_sensors.core.sensor_init import apply_offset_to_all_sensors

from isaaclab.scene import InteractiveScene
import isaaclab.sim as sim_utils
from isaaclab.utils import math as math_utils
import isaacsim.core.utils.torch as torch_utils


def diagnose_tactile_setup(scene, sensor_names):
    """Validate that tactile sensor prim paths exist in the composed scene.

    Prints the actual USD hierarchy so we can verify sensor prim paths
    are correct and the force field is properly initialized.
    Matches diagnostic format from working tacl_sensor.py for comparison.
    """
    import omni.usd
    from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema

    stage = omni.usd.get_context().get_stage()
    print(f"\n{'='*60}")
    print("TACTILE SENSOR PIPELINE DIAGNOSTIC (RH56E2)")
    print(f"{'='*60}")

    import re
    for name in sensor_names[:3]:  # check first 3 sensors
        sensor = scene[name]
        cfg = sensor.cfg
        raw_path = cfg.prim_path
        resolved = re.sub(r"/World/envs/env_\.\*", "/World/envs/env_0", raw_path)
        resolved = resolved.replace("{ENV_REGEX_NS}", "/World/envs/env_0")

        print(f"\n  --- [{name}] ---")

        # [1] Parent prim (sensor body)
        _parent = sensor._parent_prims[0]
        print(f"\n  [1] Sensor parent prim:")
        print(f"      Path: {_parent.GetPath()}")
        print(f"      HasRigidBodyAPI: {_parent.HasAPI(UsdPhysics.RigidBodyAPI)}")
        print(f"      HasCollisionAPI: {_parent.HasAPI(UsdPhysics.CollisionAPI)}")

        # [2] Tactile points (body-local frame)
        if hasattr(sensor, '_tactile_pos_local') and sensor._tactile_pos_local is not None:
            _pts = sensor._tactile_pos_local
            print(f"\n  [2] Tactile points (body-local frame):")
            print(f"      Count: {_pts.shape[0]}")
            print(f"      X range: [{_pts[:,0].min():.6f}, {_pts[:,0].max():.6f}]")
            print(f"      Y range: [{_pts[:,1].min():.6f}, {_pts[:,1].max():.6f}]")
            print(f"      Z range: [{_pts[:,2].min():.6f}, {_pts[:,2].max():.6f}]")
            _ranges = _pts.max(0).values - _pts.min(0).values
            _slim = _ranges.argmin().item()
            print(f"      Slim axis: {_slim} (range={_ranges[_slim]:.6f}m)")
            print(f"      First pt: ({_pts[0,0]:.6f}, {_pts[0,1]:.6f}, {_pts[0,2]:.6f})")
        else:
            print(f"\n  [2] Tactile points: NOT GENERATED")

        # [3] Contact object SDF
        if hasattr(sensor, '_contact_object_sdf_view') and sensor._contact_object_sdf_view is not None:
            print(f"\n  [3] Contact object SDF view: CREATED")
        else:
            print(f"\n  [3] Contact object SDF view: NONE")

        # [4] Elastomer body view
        if hasattr(sensor, '_elastomer_body_view') and sensor._elastomer_body_view is not None:
            _el_tr = sensor._elastomer_body_view.get_transforms()
            print(f"\n  [4] Elastomer body view:")
            print(f"      Pos (env0): ({_el_tr[0,0]:.4f}, {_el_tr[0,1]:.4f}, {_el_tr[0,2]:.4f})")
        else:
            print(f"\n  [4] Elastomer body view: NONE")

        # [5] Contact object body view
        if hasattr(sensor, '_contact_object_body_view') and sensor._contact_object_body_view is not None:
            _co_tr = sensor._contact_object_body_view.get_transforms()
            print(f"\n  [5] Contact object body view:")
            print(f"      Pos (env0): ({_co_tr[0,0]:.4f}, {_co_tr[0,1]:.4f}, {_co_tr[0,2]:.4f})")
        else:
            print(f"\n  [5] Contact object body view: NONE")

        # [6] Compliant contact material
        _parent_path = _parent.GetPath().pathString
        _comp_mat_path = f"{_parent_path}/compliant_material"
        _comp_prim = stage.GetPrimAtPath(_comp_mat_path)
        print(f"\n  [6] Compliant contact material:")
        print(f"      Path: {_comp_mat_path}")
        print(f"      Exists: {_comp_prim.IsValid() if _comp_prim else False}")
        if _comp_prim and _comp_prim.IsValid():
            _physx_mat = PhysxSchema.PhysxMaterialAPI(_comp_prim)
            if _physx_mat:
                _stiff = _physx_mat.GetCompliantContactStiffnessAttr()
                _damp = _physx_mat.GetCompliantContactDampingAttr()
                print(f"      Stiffness: {_stiff.Get() if _stiff else 'N/A'}")
                print(f"      Damping: {_damp.Get() if _damp else 'N/A'}")

        # [7] Collision/visual mesh prims under sensor parent
        print(f"\n  [7] Mesh prims under sensor parent:")
        for _prim in Usd.PrimRange(_parent):
            _has_col = _prim.HasAPI(UsdPhysics.CollisionAPI)
            _has_mesh_col = _prim.HasAPI(UsdPhysics.MeshCollisionAPI)
            _is_mesh = _prim.IsA(UsdGeom.Mesh)
            if _is_mesh:
                _approx = "N/A"
                if _has_mesh_col:
                    _approx = UsdPhysics.MeshCollisionAPI(_prim).GetApproximationAttr().Get()
                _label = "COLLISION" if _has_col else "VISUAL"
                print(f"      [{_label}] {_prim.GetPath()} approx={_approx}")

        # [8] Check contact object SDF mesh
        co = cfg.contact_object_prim_path_expr
        if co:
            co_resolved = re.sub(r"/World/envs/env_\.\*", "/World/envs/env_0", co)
            co_resolved = co_resolved.replace("{ENV_REGEX_NS}", "/World/envs/env_0")
            co_prim = stage.GetPrimAtPath(co_resolved)
            print(f"\n  [8] Contact object:")
            print(f"      Path: {co_resolved}  exists={co_prim.IsValid()}")
            if co_prim.IsValid():
                # Find SDF mesh
                for _cp in Usd.PrimRange(co_prim):
                    if _cp.HasAPI(UsdPhysics.MeshCollisionAPI):
                        _approx = UsdPhysics.MeshCollisionAPI(_cp).GetApproximationAttr().Get()
                        print(f"      SDF mesh: {_cp.GetPath()} approx={_approx}")

        # [9] Sensor force data
        data = sensor.data
        if data.tactile_normal_force is not None:
            print(f"\n  [9] Force data:")
            print(f"      Shape: {tuple(data.tactile_normal_force.shape)}")
            print(f"      Max: {data.tactile_normal_force.max().item():.6f}")
        else:
            print(f"\n  [9] Force data: None (force field NOT initialized)")

    print(f"\n{'='*60}\n")

    # 2. Show actual prim hierarchy under Robot for env 0
    for hand_name in ["right_hand", "left_hand"]:
        hand_path = f"/World/envs/env_0/Robot/{hand_name}"
        hand_prim = stage.GetPrimAtPath(hand_path)
        print(f"\n  Hierarchy under {hand_path}:")
        if hand_prim.IsValid():
            count = 0
            for p in Usd.PrimRange(hand_prim):
                pname = p.GetName()
                # Only show force_sensor and tactile prims, not everything
                if any(k in pname for k in ("force_sensor", "tactile", "elastomer", "palm")):
                    depth = str(p.GetPath()).count("/") - hand_path.count("/")
                    indent = "      " + "  " * depth
                    apis = ""
                    if p.HasAPI(UsdPhysics.RigidBodyAPI):
                        apis += " [RigidBody]"
                    if p.HasAPI(UsdPhysics.CollisionAPI):
                        apis += " [Collision]"
                    print(f"{indent}{pname}{apis}")
                count += 1
                if count > 200:
                    print(f"      ... (truncated)")
                    break
        else:
            print(f"    NOT FOUND!")
            # List direct children of Robot
            robot_path = "/World/envs/env_0/Robot"
            robot_prim = stage.GetPrimAtPath(robot_path)
            if robot_prim.IsValid():
                children = [c.GetName() for c in robot_prim.GetChildren()]
                print(f"    Robot children: {children}")
            print(f"\n    >>> TIP: if '{hand_name}' doesn't exist, the sensor")
            print(f"    >>> prim paths are WRONG. Check robot_hand_mount_paths")
            print(f"    >>> in SimulationConfig.")

    print(f"\n{'='*60}\n")


def run_force_push_test(scene, sensor_names, sim_context):
    """Verify tactile pipeline by pushing the nut onto a sensor's face.

    Three-stage test:
    1. SDF geometry probe  characterise the nut mesh shape in the SDF frame
    2. Wrench-based push  physically push the nut toward the sensor (like the
       working tacsl demo) and check for forces after several steps
    3. Auto-update check  verify scene.update() triggers force field via
       the lazy_sensor_update fix

    Returns True if forces were detected.
    """
    import omni.usd
    from pxr import Usd, UsdGeom, UsdPhysics

    # Pick a test sensor  prefer right palm (largest pad)
    test_name = next(
        (n for n in sensor_names if "right" in n and "palm" in n),
        next((n for n in sensor_names if "palm" in n), sensor_names[0]),
    )

    sensor = scene[test_name]
    obj = scene["grasp_object"]
    device = sensor._device

    print(f"\n{'='*60}")
    print(f"FORCE-PUSH TEST: {test_name}")
    print(f"{'='*60}")

    # --- 0. Basic sensor state check ---
    print(f"  enable_force_field:    {sensor.cfg.enable_force_field}")
    print(f"  enable_camera_tactile: {sensor.cfg.enable_camera_tactile}")
    tp_local = sensor._tactile_pos_local
    print(f"  _tactile_pos_local:    {tp_local.shape if tp_local is not None else 'None'}")
    print(f"  _elastomer_body_view:  {sensor._elastomer_body_view is not None}")
    print(f"  _contact_obj_sdf:      {sensor._contact_object_sdf_view is not None}")

    # Check if sensor was auto-updated during physics settle
    tp_w_raw = sensor._data.tactile_points_pos_w
    auto_updated = not (tp_w_raw == 0).all().item()
    print(f"  Auto-updated during settle: {auto_updated}")

    if not auto_updated:
        print(f"  >>> WARNING: Sensor was NOT updated during scene.update()!")
        print(f"  >>> This means lazy_sensor_update=True is blocking force field.")

    # --- 1. SDF geometry probe ---
    # The nut M16 is a hexagonal cylinder WITH A HOLE through the center.
    # If we naively teleport the nut body origin to the tactile centroid,
    # the tactile points may end up inside the hole (SDF > 0, no contact).
    # Probe the SDF in 3D to understand the mesh geometry.
    #
    # IMPORTANT: get_sdf_and_gradients requires EXACTLY num_query_points
    # per env (fixed at SDF view creation time). We must pad all probes.
    n_pts = sensor.num_tactile_points  # e.g. 112
    print(f"\n  --- SDF geometry probe (n_pts={n_pts}) ---")

    def query_sdf(points_3d):
        """Query SDF, padding to n_pts. points_3d: (K, 3) tensor, K <= n_pts."""
        padded = torch.zeros(1, n_pts, 3, device=device)
        k = min(points_3d.shape[0], n_pts)
        padded[0, :k] = points_3d[:k]
        result = sensor._contact_object_sdf_view.get_sdf_and_gradients(padded)
        return result[0, :k, -1]  # SDF values for the real points

    # Probe SDF at origin
    sdf_at_origin = query_sdf(torch.zeros(1, 3, device=device)).item()
    print(f"  SDF at (0,0,0): {sdf_at_origin:.6f} "
          f"{'INSIDE' if sdf_at_origin < 0 else 'OUTSIDE (likely in the hole!)'}")

    # Probe along each axis to find mesh extent and hole
    n_probe = min(31, n_pts)
    for axis_name, axis_idx in [("X", 0), ("Y", 1), ("Z", 2)]:
        pts = torch.zeros(n_probe, 3, device=device)
        vals = torch.linspace(-0.02, 0.02, n_probe, device=device)
        pts[:, axis_idx] = vals
        sdf_vals = query_sdf(pts)
        min_sdf = sdf_vals.min().item()
        min_idx = sdf_vals.argmin().item()
        n_inside = (sdf_vals < 0).sum().item()
        print(f"  {axis_name}-axis: min_sdf={min_sdf:.6f} at {axis_name}={vals[min_idx]:.4f}, "
              f"inside_count={n_inside}/{n_probe}")

    # The Z-axis probe minimum tells us where the nut center is along Z.
    # Probe along Z to find it precisely.
    n_z = min(31, n_pts)
    pts_z = torch.zeros(n_z, 3, device=device)
    z_vals = torch.linspace(-0.02, 0.04, n_z, device=device)
    pts_z[:, 2] = z_vals
    sdf_z = query_sdf(pts_z)
    z_center_idx = sdf_z.argmin().item()
    z_center = z_vals[z_center_idx].item()
    print(f"  Nut mesh center Z ~= {z_center:.4f} (SDF={sdf_z[z_center_idx]:.6f})")

    # Probe a ring at the nut wall AT THE CORRECT Z
    n_ring = min(16, n_pts)
    pts_ring = torch.zeros(n_ring, 3, device=device)
    angles = torch.linspace(0, 2 * 3.14159, n_ring + 1, device=device)[:n_ring]
    ring_r = 0.010  # 10mm radius  should be in the nut wall
    pts_ring[:, 0] = ring_r * torch.cos(angles)
    pts_ring[:, 1] = ring_r * torch.sin(angles)
    pts_ring[:, 2] = z_center  # AT the nut center Z, not Z=0
    sdf_ring_vals = query_sdf(pts_ring)
    n_ring_inside = (sdf_ring_vals < 0).sum().item()
    print(f"  Ring (r=10mm, Z={z_center:.4f}): inside={n_ring_inside}/{n_ring}, "
          f"SDF range=[{sdf_ring_vals.min():.6f}, {sdf_ring_vals.max():.6f}]")

    # If ring not inside, try different radii to find the wall
    if n_ring_inside == 0:
        print(f"  Radial probe at Z={z_center:.4f}:")
        n_rad = min(21, n_pts)
        pts_rad = torch.zeros(n_rad, 3, device=device)
        r_vals = torch.linspace(0.0, 0.020, n_rad, device=device)
        pts_rad[:, 0] = r_vals  # probe along +X
        pts_rad[:, 2] = z_center
        sdf_rad = query_sdf(pts_rad)
        for i in range(n_rad):
            marker = " INSIDE" if sdf_rad[i] < 0 else ""
            print(f"    r={r_vals[i]*1000:.1f}mm: SDF={sdf_rad[i]:.6f}{marker}")

    # --- 1b. USD transform check: rigid body vs SDF mesh prim ---
    stage = omni.usd.get_context().get_stage()
    co_prim_path = "/World/envs/env_0/grasp_object"
    co_prim = stage.GetPrimAtPath(co_prim_path)
    if co_prim.IsValid():
        rb_world = UsdGeom.Xformable(co_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        # Find SDF mesh prim
        for prim in Usd.PrimRange(co_prim):
            if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                approx = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
                if approx == "sdf":
                    mesh_world = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                    rel = mesh_world * rb_world.GetInverse()
                    translate = rel.ExtractTranslation()
                    trans_mag = sum(t**2 for t in translate) ** 0.5
                    print(f"\n  USD mesh-to-body transform:")
                    print(f"    SDF mesh: {prim.GetPath()}")
                    if trans_mag < 1e-5:
                        print(f"    Offset: IDENTITY (mesh at body origin)")
                    else:
                        print(f"    Offset: ({translate[0]:.6f}, {translate[1]:.6f}, {translate[2]:.6f})")
                        print(f"    Magnitude: {trans_mag:.6f}m")

    # --- 2. Direct SDF overlap test (NO PHYSICS) ---
    # Previous test failed because sim_context.step() triggers PhysX collision
    # resolution, which pushes the nut away from the palm before we query SDF.
    # Fix: compute the transform analytically using the INTENDED nut position,
    # bypassing physics entirely. This isolates "does the SDF pipeline work?"
    # from "does the collision geometry allow overlap?"
    print(f"\n  --- Direct SDF overlap test (no physics step) ---")

    # Get sensor surface info (force an update to get current world positions)
    sensor._update_force_field(slice(None))
    tp_w = sensor._data.tactile_points_pos_w[0]  # (N, 3) world
    tp_centroid = tp_w.mean(0)

    el_tr = sensor._elastomer_body_view.get_transforms()
    el_pos = el_tr[0, :3]
    print(f"  Sensor body:   ({el_pos[0]:.4f}, {el_pos[1]:.4f}, {el_pos[2]:.4f})")
    print(f"  Tac centroid:  ({tp_centroid[0]:.4f}, {tp_centroid[1]:.4f}, {tp_centroid[2]:.4f})")

    # Compute tactile surface normal
    surface_dir = tp_centroid - el_pos
    surface_dist = surface_dir.norm()
    surface_dir = surface_dir / surface_dist
    print(f"  Surface normal:({surface_dir[0]:.4f}, {surface_dir[1]:.4f}, {surface_dir[2]:.4f})")
    print(f"  Surface dist:  {surface_dist:.4f}m")

    # Intended nut position: place nut so its wall passes through the tactile centroid.
    # Nut hole axis = world Z (identity quat), wall ring at r=8-12mm.
    # Offset nut body origin by 10mm in X so the wall at X=-10mm overlaps centroid.
    # Shift Z by -z_center to align mesh center with tactile points.
    nut_pos_intended = tp_centroid.clone()
    nut_pos_intended[0] += 0.010   # offset 10mm in X
    nut_pos_intended[2] -= z_center  # compensate for mesh Z offset
    nut_quat_intended = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)

    print(f"  Intended nut pos: ({nut_pos_intended[0]:.4f}, {nut_pos_intended[1]:.4f}, {nut_pos_intended[2]:.4f})")

    # Analytically transform tactile world points to intended nut body-local
    # (no physics step involved  pure math)
    co_q_inv, co_p_inv = torch_utils.tf_inverse(
        nut_quat_intended.unsqueeze(0), nut_pos_intended.unsqueeze(0)
    )
    tp_nut_local = torch_utils.tf_apply(
        co_q_inv.expand(n_pts, -1), co_p_inv.expand(n_pts, -1), tp_w
    )

    print(f"\n  Tactile pts in intended nut body-local:")
    print(f"    X: [{tp_nut_local[:,0].min():.4f}, {tp_nut_local[:,0].max():.4f}]")
    print(f"    Y: [{tp_nut_local[:,1].min():.4f}, {tp_nut_local[:,1].max():.4f}]")
    print(f"    Z: [{tp_nut_local[:,2].min():.4f}, {tp_nut_local[:,2].max():.4f}]")

    # Query SDF at these positions
    tp_query = tp_nut_local.unsqueeze(0)  # (1, N, 3)
    sdf_at_pts = sensor._contact_object_sdf_view.get_sdf_and_gradients(tp_query)
    sdf_vals = sdf_at_pts[0, :, -1]
    n_inside = (sdf_vals < 0).sum().item()
    print(f"    SDF: [{sdf_vals.min():.6f}, {sdf_vals.max():.6f}]")
    print(f"    Inside mesh: {n_inside}/{n_pts}")

    # Radial distances from nut axis (Z axis in body-local)
    r_vals = torch.sqrt(tp_nut_local[:, 0]**2 + tp_nut_local[:, 1]**2)
    print(f"    Radial dist from nut axis: [{r_vals.min():.4f}, {r_vals.max():.4f}]")
    # Count points in wall region
    in_wall = ((r_vals >= 0.008) & (r_vals <= 0.012)).sum().item()
    in_z = ((tp_nut_local[:, 2] >= 0.007) & (tp_nut_local[:, 2] <= 0.017)).sum().item()
    print(f"    In nut wall radially (8-12mm): {in_wall}/{n_pts}")
    print(f"    In nut Z range (7-17mm):       {in_z}/{n_pts}")
    print(f"    (Nut: inner r~8mm, outer r~12mm, Z~7-17mm)")

    if n_inside > 0:
        # Compute depth and force for the inside points
        depth_vals = torch.clamp(-sdf_vals, min=0)
        max_depth = depth_vals.max().item()
        force_est = max_depth * sensor.cfg.normal_contact_stiffness
        print(f"\n  >>> SUCCESS: {n_inside} taxels inside nut wall!")
        print(f"  >>> max_depth={max_depth:.6f}m  est_force={force_est:.6f}N")
        print(f"  >>> SDF pipeline works. Collision geometry blocks contact during sim.")
    else:
        print(f"\n  >>> FAIL: 0 taxels inside even without physics.")
        print(f"  >>> Check nut geometry or offset value.")

    detected = n_inside > 0

    # --- 3. Collision gap measurement ---
    # Teleport nut to the intended position, step physics, and measure how far
    # collision resolution pushes it. This tells us the "collision gap".
    print(f"\n  --- Collision gap measurement ---")
    orig_state = obj.data.root_state_w.clone()

    new_state = orig_state.clone()
    new_state[0, :3] = nut_pos_intended
    new_state[0, 3:7] = torch.tensor([1, 0, 0, 0], device=device, dtype=torch.float)
    new_state[0, 7:] = 0
    obj.write_root_state_to_sim(new_state)
    scene.write_data_to_sim()
    sim_context.step()
    scene.update(sim_context.get_physics_dt())

    # Read where the nut actually ended up after collision resolution
    co_tr_after = sensor._contact_object_body_view.get_transforms()
    nut_pos_after = co_tr_after[0, :3]
    displacement = (nut_pos_after - nut_pos_intended).norm().item()
    dz = (nut_pos_after[2] - nut_pos_intended[2]).item()
    print(f"  Intended: ({nut_pos_intended[0]:.4f}, {nut_pos_intended[1]:.4f}, {nut_pos_intended[2]:.4f})")
    print(f"  After physics: ({nut_pos_after[0]:.4f}, {nut_pos_after[1]:.4f}, {nut_pos_after[2]:.4f})")
    print(f"  Displacement: {displacement*1000:.1f}mm (dZ={dz*1000:.1f}mm)")
    print(f"  This is how far collision pushes the nut away.")
    if displacement > 0.001:
        print(f"  Suggested minimum offset: {(surface_dist.item() + displacement)*1000:.0f}mm")

    # --- 4. Reset ---
    obj.write_root_state_to_sim(orig_state)
    for _ in range(5):
        scene.write_data_to_sim()
        sim_context.step()
        scene.update(sim_context.get_physics_dt())

    print(f"\n  Nut reset to original position.")
    print(f"{'='*60}\n")
    return detected


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
        num_episodes=999,
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
    print("ROBOT DIAGNOSTICS")
    print(f"{'='*60}")
    print(f"  num_joints:  {robot.num_joints}")
    print(f"  joint_names: {robot.joint_names}")
    print(f"  num_bodies:  {robot.num_bodies}")
    print(f"  body_names:  {robot.body_names}")
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

        # NOW enable non-lazy sensor updates so force field runs every step.
        # Must be after get_initial_render()  camera tactile crashes if
        # _update_buffers_impl runs before _nominal_tactile is set.
        scene.cfg.lazy_sensor_update = False
        print("Switched to non-lazy sensor updates (force field active)")

        # Apply tactile normal offset to push sensing points past parent collision
        if sensor_cfg.tactile_normal_offset > 0:
            apply_offset_to_all_sensors(scene, tactile_sensor_names, sensor_cfg.tactile_normal_offset)

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
    # 5d. Tactile Sensor Diagnostics
    # =========================================================
    if tactile_sensor_names:
        diagnose_tactile_setup(scene, tactile_sensor_names)
    
    # =========================================================
    # 5e. Force-push pipeline test
    # =========================================================
    if tactile_sensor_names:
        run_force_push_test(scene, tactile_sensor_names, sim_context)

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
            "control_mode": "cartesian_ik_teleop",
        },
        os.path.join(grasp_cfg.output_dir, "experiment_config.json"),
    )

    # =========================================================
    # 7. Cartesian Teleop Controller
    # =========================================================
    controller = CartesianTeleopController(
        robot=robot,
        device=robot.device,
        pos_step=args.pos_step,
        rot_step=args.rot_step,
        damping_lambda=args.damping,
    )
    controller.initialize_desired_poses(robot)

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
            # --- Poll keyboard events ---
            toggled, reset, dirty = controller.poll_events()

            # Handle recording toggle
            if toggled:
                if controller.recording:
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
                                "control_mode": "cartesian_ik_teleop",
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
                settle_target = robot.data.default_joint_pos.clone()
                robot.set_joint_position_target(settle_target)
                for _ in range(20):
                    scene.write_data_to_sim()
                    sim_context.step()
                    scene.update(sim_context.get_physics_dt())

                # Re-initialize desired poses from the settled state
                controller.initialize_desired_poses(robot)
                controller.grasp_target = {"left": 0.0, "right": 0.0}
                controller.grasp_current = {"left": 0.0, "right": 0.0}
                print("  [Reset to default pose]")

            # --- Compute IK targets and apply ---
            targets = controller.compute_ik_targets(robot)
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
                        phase=0,
                    )
                recording_step += 1

            # --- Terminal HUD (throttled to ~2 Hz) ---
            now = time.monotonic()
            if dirty or (now - last_display_time > 0.5):
                status = controller.get_status_line(sim_step)

                # Live force readout from tactile sensors
                force_line = ""
                if tactile_sensor_names:
                    max_forces = []
                    for tname in tactile_sensor_names:
                        tdata = scene[tname].data
                        if tdata.tactile_normal_force is not None:
                            mf = tdata.tactile_normal_force.max().item()
                            max_forces.append((tname.split("_", 2)[-1], mf))
                    # Show top 3 non-zero or first 3
                    max_forces.sort(key=lambda x: -x[1])
                    top = max_forces[:3]
                    parts = [f"{n}={v:.5f}" for n, v in top]
                    force_line = f"  Forces: {', '.join(parts)}" if parts else ""

                lines = status.split("\n")
                print(f"\r\033[2K{lines[0]}{force_line}")
                print(f"\r\033[2K{lines[1]}", end="", flush=True)
                last_display_time = now

            # --- Deep tactile debug (every 200 steps) ---
            if tactile_sensor_names and sim_step % 200 == 0 and sim_step > 0:
                print(f"\n\n{'='*60}")
                print(f"TACTILE DEBUG  step={sim_step}")
                print(f"{'='*60}")

                # Get nut position from any sensor's contact object view
                _co_pos = None
                for _tn in tactile_sensor_names:
                    _s = scene[_tn]
                    if _s._contact_object_body_view is not None:
                        _co_pos = _s._contact_object_body_view.get_transforms()[0, :3]
                        break
                if _co_pos is not None:
                    print(f"  Nut world pos: ({_co_pos[0]:.4f}, {_co_pos[1]:.4f}, {_co_pos[2]:.4f})")

                # Show ALL right-hand sensor positions + distance to nut
                print(f"\n  Right-hand sensor distances to nut:")
                for _tn in tactile_sensor_names:
                    if "right" not in _tn:
                        continue
                    _s = scene[_tn]
                    if _s._elastomer_body_view is None:
                        continue
                    _el_pos = _s._elastomer_body_view.get_transforms()[0, :3]
                    _d = (_co_pos - _el_pos).norm().item() if _co_pos is not None else -1
                    _short = _tn.replace("right_hand_", "R_")
                    # Also show tactile pts centroid if available
                    _tp = _s._data.tactile_points_pos_w
                    _tp_info = ""
                    if _tp is not None and _tp.numel() > 0:
                        _tp_ctr = _tp[0].mean(0)
                        _tp_d = (_co_pos - _tp_ctr).norm().item() if _co_pos is not None else -1
                        _tp_info = f"  pts_d={_tp_d:.3f}"
                    print(f"    {_short:30s} body_d={_d:.3f}m  pos=({_el_pos[0]:.3f},{_el_pos[1]:.3f},{_el_pos[2]:.3f}){_tp_info}")

                # Show which ROBOT BODIES are closest to the nut
                if _co_pos is not None:
                    _body_pos = robot.data.body_pos_w[0]  # (num_bodies, 3)
                    _body_dists = (_body_pos - _co_pos.unsqueeze(0)).norm(dim=-1)
                    _sorted_idx = _body_dists.argsort()
                    print(f"\n  Closest robot bodies to nut (top 10):")
                    for _i in range(min(10, len(_sorted_idx))):
                        _idx = _sorted_idx[_i].item()
                        _bname = robot.body_names[_idx]
                        _bpos = _body_pos[_idx]
                        _bd = _body_dists[_idx].item()
                        _is_sensor = "force_sensor" in _bname
                        _marker = " <<< SENSOR" if _is_sensor else ""
                        print(f"    {_bname:40s} d={_bd:.3f}m  pos=({_bpos[0]:.3f},{_bpos[1]:.3f},{_bpos[2]:.3f}){_marker}")

                # --- SDF diagnostic: query known test points ---
                _diag_sensor = None
                for _tn in tactile_sensor_names:
                    if "right" in _tn and "palm" in _tn:
                        _diag_sensor = scene[_tn]
                        break
                if _diag_sensor is not None and _diag_sensor._contact_object_sdf_view is not None:
                    _n_pts = _diag_sensor.num_tactile_points
                    # Get nut body-local frame
                    _co_tr = _diag_sensor._contact_object_body_view.get_transforms()
                    _co_p = _co_tr[0, :3]
                    _co_q = math_utils.convert_quat(_co_tr[0, 3:7], to="wxyz")
                    print(f"\n  SDF DIAGNOSTIC (R_palm, {_n_pts} pts):")

                    # Test points in nut body-local frame
                    _test_pts_local = torch.zeros(1, _n_pts, 3, device=_diag_sensor._device)
                    # Point 0: body origin (0,0,0)
                    # Point 1: mesh-native center (0,0,0.016)
                    # Point 2: 4x-scaled center (0,0,0.065)
                    # Point 3: known inside at native scale (0,0,0.015)
                    if _n_pts > 3:
                        _test_pts_local[0, 1] = torch.tensor([0, 0, 0.016], device=_diag_sensor._device)
                        _test_pts_local[0, 2] = torch.tensor([0, 0, 0.065], device=_diag_sensor._device)
                        _test_pts_local[0, 3] = torch.tensor([0, 0, 0.015], device=_diag_sensor._device)
                    _test_sdf = _diag_sensor._contact_object_sdf_view.get_sdf_and_gradients(_test_pts_local)
                    _test_sdf_vals = _test_sdf[0, :4, -1]
                    print(f"    SDF at body-local (0,0,0):     {_test_sdf_vals[0]:.6f}")
                    if _n_pts > 3:
                        print(f"    SDF at body-local (0,0,0.016): {_test_sdf_vals[1]:.6f}  (mesh-native center)")
                        print(f"    SDF at body-local (0,0,0.065): {_test_sdf_vals[2]:.6f}  (4x-scaled center)")
                        print(f"    SDF at body-local (0,0,0.015): {_test_sdf_vals[3]:.6f}  (mesh-native inside)")

                    # Now query with actual tactile points in body-local
                    _tp_w = _diag_sensor._data.tactile_points_pos_w[0]  # (N,3) world
                    # Transform to nut body-local
                    _co_q_inv, _co_p_inv = torch_utils.tf_inverse(_co_q.unsqueeze(0), _co_p.unsqueeze(0))
                    _tp_local = torch_utils.tf_apply(
                        _co_q_inv.expand(_n_pts, -1), _co_p_inv.expand(_n_pts, -1), _tp_w
                    )
                    _tp_query = _tp_local.unsqueeze(0)  # (1, N, 3)
                    _tp_sdf = _diag_sensor._contact_object_sdf_view.get_sdf_and_gradients(_tp_query)
                    _tp_sdf_vals = _tp_sdf[0, :, -1]
                    _neg_count = (_tp_sdf_vals < 0).sum().item()

                    # Geometry analysis: where are tactile pts relative to nut?
                    _r_vals = torch.sqrt(_tp_local[:, 0]**2 + _tp_local[:, 1]**2)
                    _in_wall = ((_r_vals >= 0.008) & (_r_vals <= 0.012)).sum().item()
                    _z_min = _tp_local[:, 2].min().item()
                    _z_max = _tp_local[:, 2].max().item()
                    _z_mean = _tp_local[:, 2].mean().item()

                    # Nut extent: Z ~0.007 to ~0.017 (10mm tall, center at 0.012)
                    print(f"\n    Tactile pts in nut body-local:")
                    print(f"      Z: [{_z_min:.4f}, {_z_max:.4f}] mean={_z_mean:.4f}")
                    print(f"      Nut Z extent: [0.007, 0.017]  (need Z in this range)")
                    if _z_mean > 0.017:
                        print(f"      >>> Points {(_z_mean-0.017)*1000:.1f}mm ABOVE nut top -- need more offset")
                    elif _z_mean < 0.007:
                        print(f"      >>> Points {(0.007-_z_mean)*1000:.1f}mm BELOW nut bottom")
                    else:
                        print(f"      >>> Z is within nut range")
                    print(f"      Radial: [{_r_vals.min():.4f}, {_r_vals.max():.4f}]  in wall(8-12mm): {_in_wall}/{_n_pts}")
                    print(f"    SDF: min={_tp_sdf_vals.min():.6f} max={_tp_sdf_vals.max():.6f}")
                    print(f"    Inside nut: {_neg_count}/{_n_pts}")

                    # Direct readout of what the pipeline actually computed
                    _pf = _diag_sensor._data.tactile_normal_force
                    _pd = _diag_sensor._data.penetration_depth
                    print(f"\n    PIPELINE OUTPUT (what sensor actually computed):")
                    print(f"      tactile_normal_force: shape={tuple(_pf.shape)} "
                          f"max={_pf.max().item():.8f} sum={_pf.sum().item():.8f}")
                    print(f"      penetration_depth:    shape={tuple(_pd.shape)} "
                          f"max={_pd.max().item():.8f} sum={_pd.sum().item():.8f}")
                    # Check if the pipeline's world points match our diagnostic
                    _pipe_w = _diag_sensor._data.tactile_points_pos_w
                    if _pipe_w is not None and _pipe_w.numel() > 0:
                        print(f"      tactile_points_pos_w: shape={tuple(_pipe_w.shape)} "
                              f"Z range=[{_pipe_w[0,:,2].min():.4f}, {_pipe_w[0,:,2].max():.4f}]")
                    # Check _tactile_pos_expanded (offset should be applied)
                    _tpe = _diag_sensor._tactile_pos_expanded
                    print(f"      _tactile_pos_expanded: shape={tuple(_tpe.shape)} "
                          f"Z range=[{_tpe[0,:,2].min():.4f}, {_tpe[0,:,2].max():.4f}] "
                          f"(offset applied? Z should be ~+0.003 not ~-0.001)")

                print(f"{'='*60}\n")

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
                "control_mode": "cartesian_ik_teleop",
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
