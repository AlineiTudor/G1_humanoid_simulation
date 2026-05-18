"""
Diagnostic script to check if tacl_sensor example produces non-zero forces.

Run INSIDE Isaac Sim script editor while the tacl_sensor.py example is loaded
and running (with --use_tactile_ff --contact_object_type nut).

Alternatively, paste this into the Isaac Sim Script Editor console after
opening the tacl_sensor example scene.

This script inspects the USD stage to verify:
1. The elastomer prim exists and has a visual mesh
2. The contact object exists and has an SDF mesh
3. The sensor body views are valid
"""

import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics

stage = omni.usd.get_context().get_stage()

print("=" * 60)
print("TACL_SENSOR DIAGNOSTICS")
print("=" * 60)

# --- 1. Check elastomer structure ---
elastomer_path = "/World/envs/env_0/Robot/elastomer"
elastomer_prim = stage.GetPrimAtPath(elastomer_path)
print(f"\n[Elastomer] path: {elastomer_path}")
print(f"  exists: {elastomer_prim.IsValid()}")

if elastomer_prim.IsValid():
    print(f"  type: {elastomer_prim.GetTypeName()}")
    has_rb = elastomer_prim.HasAPI(UsdPhysics.RigidBodyAPI)
    print(f"  RigidBodyAPI: {has_rb}")

    # Find visual meshes (no CollisionAPI)
    visual_meshes = []
    collision_meshes = []
    for desc in Usd.PrimRange(elastomer_prim):
        if desc.IsA(UsdGeom.Mesh):
            if desc.HasAPI(UsdPhysics.CollisionAPI):
                collision_meshes.append(desc)
            else:
                visual_meshes.append(desc)

    print(f"  visual meshes: {len(visual_meshes)}")
    for m in visual_meshes:
        pts = UsdGeom.Mesh(m).GetPointsAttr().Get()
        n_pts = len(pts) if pts else 0
        print(f"    {m.GetPath()} ({n_pts} vertices)")

    print(f"  collision meshes: {len(collision_meshes)}")
    for m in collision_meshes:
        print(f"    {m.GetPath()}")
        if m.HasAPI(UsdPhysics.MeshCollisionAPI):
            approx = UsdPhysics.MeshCollisionAPI(m).GetApproximationAttr().Get()
            print(f"      MeshCollisionAPI approx: {approx}")

    # Check for compliant material
    for desc in Usd.PrimRange(elastomer_prim):
        if "compliant" in desc.GetName().lower() or "material" in desc.GetName().lower():
            print(f"  material prim: {desc.GetPath()} (type: {desc.GetTypeName()})")

    # Print children hierarchy
    print(f"\n  Hierarchy under {elastomer_path}:")
    for desc in Usd.PrimRange(elastomer_prim):
        depth = len(str(desc.GetPath()).split("/")) - len(elastomer_path.split("/"))
        indent = "    " + "  " * depth
        apis = []
        if desc.HasAPI(UsdPhysics.RigidBodyAPI):
            apis.append("RigidBody")
        if desc.HasAPI(UsdPhysics.CollisionAPI):
            apis.append("Collision")
        if desc.HasAPI(UsdPhysics.MeshCollisionAPI):
            approx = UsdPhysics.MeshCollisionAPI(desc).GetApproximationAttr().Get()
            apis.append(f"MeshCollision({approx})")
        api_str = f" [{', '.join(apis)}]" if apis else ""
        print(f"{indent}{desc.GetName()} ({desc.GetTypeName()}){api_str}")


# --- 2. Check contact object ---
contact_paths = [
    "/World/envs/env_0/contact_object",
    "/World/envs/env_0/ContactObject",
]

for co_path in contact_paths:
    co_prim = stage.GetPrimAtPath(co_path)
    if co_prim.IsValid():
        print(f"\n[Contact Object] path: {co_path}")
        print(f"  type: {co_prim.GetTypeName()}")
        has_rb = co_prim.HasAPI(UsdPhysics.RigidBodyAPI)
        print(f"  RigidBodyAPI: {has_rb}")

        # Find SDF meshes
        sdf_meshes = []
        for desc in Usd.PrimRange(co_prim):
            if desc.HasAPI(UsdPhysics.MeshCollisionAPI):
                approx = UsdPhysics.MeshCollisionAPI(desc).GetApproximationAttr().Get()
                sdf_meshes.append((desc, approx))
                print(f"  MeshCollisionAPI prim: {desc.GetPath()}")
                print(f"    approximation: {approx}")
                print(f"    is SDF: {approx == 'sdf'}")

        if not sdf_meshes:
            print("  WARNING: No MeshCollisionAPI prims found!")
            print("  Force field mode requires SDF mesh on contact object.")

        # Check collision children
        for desc in Usd.PrimRange(co_prim):
            if desc.HasAPI(UsdPhysics.CollisionAPI):
                print(f"  CollisionAPI prim: {desc.GetPath()} ({desc.GetTypeName()})")

        break
else:
    print("\n[Contact Object] NOT FOUND at any expected path")
    print("  Searched:", contact_paths)

# --- 3. Check G1 robot force sensor structure (if present) ---
g1_palm_path = "/World/envs/env_0/Robot/right_hand/right_palm_force_sensor"
g1_palm = stage.GetPrimAtPath(g1_palm_path)
if g1_palm.IsValid():
    print(f"\n[G1 Force Sensor] path: {g1_palm_path}")
    print(f"  RigidBodyAPI: {g1_palm.HasAPI(UsdPhysics.RigidBodyAPI)}")

    # Check elastomer child
    elastomer_child = stage.GetPrimAtPath(f"{g1_palm_path}/elastomer_right_palm_tip")
    if elastomer_child.IsValid():
        print(f"  elastomer child exists: True")
        # Check for visual mesh under parent (what the sensor will search)
        for desc in Usd.PrimRange(g1_palm):
            if desc.IsA(UsdGeom.Mesh) and not desc.HasAPI(UsdPhysics.CollisionAPI):
                pts = UsdGeom.Mesh(desc).GetPointsAttr().Get()
                n_pts = len(pts) if pts else 0
                print(f"  visual mesh: {desc.GetPath()} ({n_pts} verts)")
                break
        else:
            print("  WARNING: No visual mesh found under force sensor!")
            print("  The sensor cannot generate tactile points without a mesh.")

    # Check grasp object for SDF
    grasp_obj = stage.GetPrimAtPath("/World/envs/env_0/grasp_object")
    if grasp_obj.IsValid():
        print(f"\n[Grasp Object] path: /World/envs/env_0/grasp_object")
        print(f"  type: {grasp_obj.GetTypeName()}")
        found_sdf = False
        for desc in Usd.PrimRange(grasp_obj):
            if desc.HasAPI(UsdPhysics.MeshCollisionAPI):
                approx = UsdPhysics.MeshCollisionAPI(desc).GetApproximationAttr().Get()
                print(f"  MeshCollisionAPI: {desc.GetPath()} (approx={approx})")
                if approx == "sdf":
                    found_sdf = True
            if desc.HasAPI(UsdPhysics.CollisionAPI):
                print(f"  CollisionAPI: {desc.GetPath()} ({desc.GetTypeName()})")
        if not found_sdf:
            print("  *** NO SDF MESH FOUND ***")
            print("  Force field mode REQUIRES an SDF mesh on the contact object.")
            print("  Use --object_type nut (not cube). CuboidCfg creates Shape Prims")
            print("  which cannot generate SDF meshes.")

print("\n" + "=" * 60)
print("DIAGNOSTICS COMPLETE")
print("=" * 60)
