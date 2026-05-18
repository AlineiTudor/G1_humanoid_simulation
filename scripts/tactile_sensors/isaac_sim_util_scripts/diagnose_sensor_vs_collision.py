"""
Diagnostic: Force sensor mesh positions vs parent link collision surfaces.

Shows where the force sensor tactile points are relative to the collision
geometry that actually contacts objects. Run in Isaac Sim Script Editor
while the scene is loaded.

Key question: Can the nut ever overlap with force sensor meshes,
or does parent link collision block it?
"""

import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics, Gf
import numpy as np

stage = omni.usd.get_context().get_stage()

# ============================================================
# Check the right hand force sensor links
# ============================================================
hand_path = "/World/envs/env_0/Robot/right_hand"
hand_prim = stage.GetPrimAtPath(hand_path)

if not hand_prim.IsValid():
    print(f"Hand not found at {hand_path}")
else:
    print("=" * 70)
    print("FORCE SENSOR vs PARENT LINK COLLISION ANALYSIS")
    print("=" * 70)

    # Find all force sensor links
    force_sensor_links = []
    for prim in Usd.PrimRange(hand_prim):
        name = prim.GetName()
        if "force_sensor" in name and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            force_sensor_links.append(prim)

    for fs_prim in force_sensor_links[:5]:  # first 5 for brevity
        fs_path = str(fs_prim.GetPath())
        fs_name = fs_prim.GetName()
        print(f"\n{'-'*60}")
        print(f"Force Sensor: {fs_name}")
        print(f"  Path: {fs_path}")

        # Get force sensor world transform
        fs_world = UsdGeom.Xformable(fs_prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        fs_pos = fs_world.ExtractTranslation()
        print(f"  World pos: ({fs_pos[0]:.4f}, {fs_pos[1]:.4f}, {fs_pos[2]:.4f})")

        # Find ALL meshes under this force sensor
        print(f"\n  Meshes under force sensor:")
        for child in Usd.PrimRange(fs_prim):
            if child.IsA(UsdGeom.Mesh):
                mesh = UsdGeom.Mesh(child)
                pts_raw = mesh.GetPointsAttr().Get()
                if pts_raw is None or len(pts_raw) == 0:
                    print(f"    {child.GetName()}: EMPTY")
                    continue

                pts = np.array(pts_raw)
                has_col = child.HasAPI(UsdPhysics.CollisionAPI)
                is_instance = child.IsInstance()
                label = "COL" if has_col else "VIS"
                if is_instance:
                    label += "+INST"

                # Transform mesh points to world
                mesh_world = UsdGeom.Xformable(child).ComputeLocalToWorldTransform(
                    Usd.TimeCode.Default()
                )

                pts_world = []
                for pt in pts:
                    p = Gf.Vec4d(float(pt[0]), float(pt[1]), float(pt[2]), 1.0)
                    pw = p * mesh_world
                    pts_world.append([pw[0], pw[1], pw[2]])
                pts_world = np.array(pts_world)

                print(f"    [{label}] {child.GetName()} ({len(pts)} verts)")
                print(f"      mesh-local: min=({pts.min(0)[0]:.4f},{pts.min(0)[1]:.4f},{pts.min(0)[2]:.4f})")
                print(f"                  max=({pts.max(0)[0]:.4f},{pts.max(0)[1]:.4f},{pts.max(0)[2]:.4f})")
                print(f"      world:      min=({pts_world.min(0)[0]:.4f},{pts_world.min(0)[1]:.4f},{pts_world.min(0)[2]:.4f})")
                print(f"                  max=({pts_world.max(0)[0]:.4f},{pts_world.max(0)[1]:.4f},{pts_world.max(0)[2]:.4f})")
                span = pts.max(0) - pts.min(0)
                print(f"      span: ({span[0]*1000:.1f}mm, {span[1]*1000:.1f}mm, {span[2]*1000:.1f}mm)")

        # Find the PARENT link (the joint parent that has collision)
        parent_prim = fs_prim.GetParent()
        # Walk up until we find a prim with collision children
        while parent_prim.IsValid() and str(parent_prim.GetPath()) != hand_path:
            has_collision_child = False
            for child in Usd.PrimRange(parent_prim):
                if child.HasAPI(UsdPhysics.CollisionAPI) and child.IsA(UsdGeom.Mesh):
                    has_collision_child = True
                    break
            if has_collision_child and parent_prim != fs_prim:
                break
            parent_prim = parent_prim.GetParent()

        if parent_prim.IsValid() and str(parent_prim.GetPath()) != hand_path:
            print(f"\n  Parent with collision: {parent_prim.GetName()}")
            parent_pos = UsdGeom.Xformable(parent_prim).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default()
            ).ExtractTranslation()
            print(f"  Parent world pos: ({parent_pos[0]:.4f}, {parent_pos[1]:.4f}, {parent_pos[2]:.4f})")

            # Find collision meshes in parent
            print(f"  Parent collision meshes:")
            for child in Usd.PrimRange(parent_prim):
                if child == fs_prim or str(child.GetPath()).startswith(fs_path):
                    continue  # skip the force sensor itself
                if child.HasAPI(UsdPhysics.CollisionAPI) and child.IsA(UsdGeom.Mesh):
                    mesh = UsdGeom.Mesh(child)
                    pts_raw = mesh.GetPointsAttr().Get()
                    if pts_raw is None or len(pts_raw) == 0:
                        continue
                    pts = np.array(pts_raw)
                    mesh_world = UsdGeom.Xformable(child).ComputeLocalToWorldTransform(
                        Usd.TimeCode.Default()
                    )
                    pts_world = []
                    for pt in pts:
                        p = Gf.Vec4d(float(pt[0]), float(pt[1]), float(pt[2]), 1.0)
                        pw = p * mesh_world
                        pts_world.append([pw[0], pw[1], pw[2]])
                    pts_world = np.array(pts_world)

                    print(f"    {child.GetName()} ({len(pts)} verts)")
                    print(f"      world: min=({pts_world.min(0)[0]:.4f},{pts_world.min(0)[1]:.4f},{pts_world.min(0)[2]:.4f})")
                    print(f"             max=({pts_world.max(0)[0]:.4f},{pts_world.max(0)[1]:.4f},{pts_world.max(0)[2]:.4f})")

    # Also check contact object
    co_path = "/World/envs/env_0/grasp_object"
    co_prim = stage.GetPrimAtPath(co_path)
    if co_prim.IsValid():
        print(f"\n{'-'*60}")
        print(f"CONTACT OBJECT: {co_path}")
        co_pos = UsdGeom.Xformable(co_prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        ).ExtractTranslation()
        print(f"  World pos: ({co_pos[0]:.4f}, {co_pos[1]:.4f}, {co_pos[2]:.4f})")

        for child in Usd.PrimRange(co_prim):
            if child.IsA(UsdGeom.Mesh):
                mesh = UsdGeom.Mesh(child)
                pts_raw = mesh.GetPointsAttr().Get()
                if pts_raw is None or len(pts_raw) == 0:
                    continue
                pts = np.array(pts_raw)
                has_sdf = False
                if child.HasAPI(UsdPhysics.MeshCollisionAPI):
                    approx = UsdPhysics.MeshCollisionAPI(child).GetApproximationAttr().Get()
                    has_sdf = approx == "sdf"

                # Check for scale on the parent
                xform_ops = UsdGeom.Xformable(co_prim).GetOrderedXformOps()
                scale_str = ""
                for op in xform_ops:
                    if "scale" in str(op.GetOpName()).lower():
                        scale_str = f"  SCALE={op.Get()}"

                mesh_world = UsdGeom.Xformable(child).ComputeLocalToWorldTransform(
                    Usd.TimeCode.Default()
                )
                pts_world = []
                for pt in pts:
                    p = Gf.Vec4d(float(pt[0]), float(pt[1]), float(pt[2]), 1.0)
                    pw = p * mesh_world
                    pts_world.append([pw[0], pw[1], pw[2]])
                pts_world = np.array(pts_world)

                print(f"  {child.GetName()} ({len(pts)} verts) SDF={has_sdf}{scale_str}")
                print(f"    mesh-local: min=({pts.min(0)[0]:.6f},{pts.min(0)[1]:.6f},{pts.min(0)[2]:.6f})")
                print(f"                max=({pts.max(0)[0]:.6f},{pts.max(0)[1]:.6f},{pts.max(0)[2]:.6f})")
                print(f"    world:      min=({pts_world.min(0)[0]:.4f},{pts_world.min(0)[1]:.4f},{pts_world.min(0)[2]:.4f})")
                print(f"                max=({pts_world.max(0)[0]:.4f},{pts_world.max(0)[1]:.4f},{pts_world.max(0)[2]:.4f})")
                span_w = pts_world.max(0) - pts_world.min(0)
                print(f"    world span: ({span_w[0]*1000:.1f}mm, {span_w[1]*1000:.1f}mm, {span_w[2]*1000:.1f}mm)")

    print(f"\n{'='*70}")
    print("ANALYSIS")
    print("=" * 70)
    print("""
For SDF force field to work, tactile points (from force sensor visual mesh)
must OVERLAP with the contact object mesh (SDF < 0 = inside).

Check:
  1. Is the force sensor mesh surface at the same location as the
     collision surface that contacts the nut? (compare world bounds)
  2. Does the parent link's collision geometry BLOCK the nut from
     reaching the force sensor surface?
  3. How thick is the force sensor mesh vs how far it extends outward?

If the force sensor is recessed behind parent collision, the nut can
never reach it ? forces always zero. Fix: extend tactile points outward
OR remove blocking parent collision near sensor pads.
""")
    print("=" * 70)
