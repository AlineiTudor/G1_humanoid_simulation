"""
Diagnostic script for Isaac Sim Script Editor.
Checks the transform chain between RigidBody parents and visual meshes
used by VisuoTactileSensor for tactile point generation.

The sensor reads mesh vertices via GetPointsAttr() (mesh-local frame)
and transforms them using the RigidBody's world pose. If there are
intermediate Xform transforms between the RigidBody and the mesh,
the tactile points will be placed at WRONG world-space positions,
causing zero SDF penetration and zero forces.

Run this in the Isaac Sim Script Editor while the simulation is loaded
(after scene creation, before or during stepping).

Works with both:
  - G1 + RH56E2 setup
  - tacl_sensor.py example
"""

import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics, Gf
import numpy as np

stage = omni.usd.get_context().get_stage()

# ============================================================
# CONFIG: paths to check (add/modify as needed)
# ============================================================
# G1 force sensor links (right hand subset)
g1_rb_paths = [
    "/World/envs/env_0/Robot/right_hand/right_palm_force_sensor",
    "/World/envs/env_0/Robot/right_hand/right_index_force_sensor_1",
    "/World/envs/env_0/Robot/right_hand/right_thumb_force_sensor_1",
    "/World/envs/env_0/Robot/left_hand/left_palm_force_sensor",
]

# tacl_sensor example
example_rb_paths = [
    "/World/envs/env_0/Robot/elastomer",
]

# Contact objects
contact_paths = [
    "/World/envs/env_0/grasp_object",
    "/World/envs/env_0/contact_object",
]

all_rb_paths = g1_rb_paths + example_rb_paths


def check_intermediate_transforms(rb_prim, mesh_prim):
    """Check if there's a non-identity transform between a mesh and its RigidBody ancestor."""
    rb_xform = UsdGeom.Xformable(rb_prim)
    mesh_xform = UsdGeom.Xformable(mesh_prim)

    time = Usd.TimeCode.Default()
    rb_to_world = rb_xform.ComputeLocalToWorldTransform(time)
    mesh_to_world = mesh_xform.ComputeLocalToWorldTransform(time)

    world_to_rb = rb_to_world.GetInverse()
    mesh_to_rb = mesh_to_world * world_to_rb

    # Extract translation
    translate = mesh_to_rb.ExtractTranslation()

    # Extract rotation (compare to identity)
    rot = mesh_to_rb.ExtractRotationMatrix()
    identity_rot = Gf.Matrix3d(1)
    rot_diff = max(
        abs(rot[i][j] - identity_rot[i][j])
        for i in range(3) for j in range(3)
    )

    trans_mag = (translate[0]**2 + translate[1]**2 + translate[2]**2) ** 0.5
    is_identity = trans_mag < 1e-5 and rot_diff < 1e-5

    return is_identity, translate, rot_diff, mesh_to_rb


def transform_points_to_rb_frame(points_np, mesh_to_rb_matrix):
    """Transform mesh-local points to rigid body local frame."""
    transformed = []
    for pt in points_np:
        p4 = Gf.Vec4d(float(pt[0]), float(pt[1]), float(pt[2]), 1.0)
        # USD convention: point * matrix (row-vector on the left)
        # Gf.Matrix4d * Gf.Vec4d does column-vector multiplication
        # so we need to use the transpose, or just multiply manually
        row = [
            p4[0] * mesh_to_rb_matrix[0][0] + p4[1] * mesh_to_rb_matrix[1][0] + p4[2] * mesh_to_rb_matrix[2][0] + p4[3] * mesh_to_rb_matrix[3][0],
            p4[0] * mesh_to_rb_matrix[0][1] + p4[1] * mesh_to_rb_matrix[1][1] + p4[2] * mesh_to_rb_matrix[2][1] + p4[3] * mesh_to_rb_matrix[3][1],
            p4[0] * mesh_to_rb_matrix[0][2] + p4[1] * mesh_to_rb_matrix[1][2] + p4[2] * mesh_to_rb_matrix[2][2] + p4[3] * mesh_to_rb_matrix[3][2],
        ]
        transformed.append(row)
    return np.array(transformed)


print("=" * 70)
print("TACTILE SENSOR TRANSFORM DIAGNOSTIC")
print("=" * 70)

# ============================================================
# 1. Check RigidBody ? Mesh transforms
# ============================================================
for rb_path in all_rb_paths:
    rb_prim = stage.GetPrimAtPath(rb_path)
    if not rb_prim.IsValid():
        continue

    print(f"\n[RigidBody] {rb_path}")
    has_rb_api = rb_prim.HasAPI(UsdPhysics.RigidBodyAPI)
    print(f"  HasRigidBodyAPI: {has_rb_api}")

    rb_world = UsdGeom.Xformable(rb_prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    rb_pos = rb_world.ExtractTranslation()
    print(f"  World position: ({rb_pos[0]:.4f}, {rb_pos[1]:.4f}, {rb_pos[2]:.4f})")

    # Find ALL descendant meshes
    visual_meshes = []
    collision_meshes = []
    for prim in Usd.PrimRange(rb_prim):
        if prim.IsA(UsdGeom.Mesh):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                collision_meshes.append(prim)
            else:
                visual_meshes.append(prim)

    print(f"  Visual meshes (no CollisionAPI): {len(visual_meshes)}")
    print(f"  Collision meshes (has CollisionAPI): {len(collision_meshes)}")

    # Check transforms for visual meshes (these are what the sensor uses)
    all_meshes = visual_meshes + collision_meshes
    for mesh_prim in all_meshes:
        mesh_path = str(mesh_prim.GetPath())
        is_visual = mesh_prim not in collision_meshes
        label = "VISUAL" if is_visual else "COLLISION"

        usd_mesh = UsdGeom.Mesh(mesh_prim)
        pts = np.array(usd_mesh.GetPointsAttr().Get())
        if len(pts) == 0:
            print(f"\n  [{label}] {mesh_path}  EMPTY MESH")
            continue

        is_id, translate, rot_diff, mesh_to_rb = check_intermediate_transforms(
            rb_prim, mesh_prim
        )

        print(f"\n  [{label}] {mesh_path}")
        print(f"    Vertices: {len(pts)}")
        print(f"    Mesh-local bounds:")
        print(f"      min: ({pts.min(0)[0]:.6f}, {pts.min(0)[1]:.6f}, {pts.min(0)[2]:.6f})")
        print(f"      max: ({pts.max(0)[0]:.6f}, {pts.max(0)[1]:.6f}, {pts.max(0)[2]:.6f})")
        print(f"      size: ({(pts.max(0)-pts.min(0))[0]:.6f}, {(pts.max(0)-pts.min(0))[1]:.6f}, {(pts.max(0)-pts.min(0))[2]:.6f})")

        if is_id:
            print(f"    Transform to RB: IDENTITY (OK  sensor will place points correctly)")
        else:
            print(f"    Transform to RB: *** NON-IDENTITY *** (PROBLEM!)")
            print(f"      Translation offset: ({translate[0]:.6f}, {translate[1]:.6f}, {translate[2]:.6f})")
            print(f"      Rotation diff from identity: {rot_diff:.6f}")

            # Show what the points look like in the RB frame
            pts_rb = transform_points_to_rb_frame(pts, mesh_to_rb)
            print(f"    RB-local bounds (where sensor SHOULD place points):")
            print(f"      min: ({pts_rb.min(0)[0]:.6f}, {pts_rb.min(0)[1]:.6f}, {pts_rb.min(0)[2]:.6f})")
            print(f"      max: ({pts_rb.max(0)[0]:.6f}, {pts_rb.max(0)[1]:.6f}, {pts_rb.max(0)[2]:.6f})")

        # Print intermediate prim transforms
        print(f"    Intermediate prims (RB ? mesh):")
        current = mesh_prim
        chain = []
        while current and current.GetPath() != rb_prim.GetPath():
            chain.append(current)
            current = current.GetParent()
        chain.reverse()

        for prim in chain:
            xformable = UsdGeom.Xformable(prim)
            ops = xformable.GetOrderedXformOps() if xformable else []
            is_instance = prim.IsInstance()
            prim_type = prim.GetTypeName()
            has_ops = len(ops) > 0
            detail = f"type={prim_type}"
            if is_instance:
                detail += ", INSTANCED"
            if has_ops:
                op_strs = []
                for op in ops:
                    op_strs.append(f"{op.GetOpName()}={op.Get()}")
                detail += f", xformOps=[{', '.join(op_strs)}]"
            else:
                detail += ", no xformOps"
            print(f"      {prim.GetPath()} ({detail})")


# ============================================================
# 2. Check contact object SDF mesh
# ============================================================
print(f"\n{'=' * 70}")
print("CONTACT OBJECT CHECK")
print("=" * 70)

for co_path in contact_paths:
    co_prim = stage.GetPrimAtPath(co_path)
    if not co_prim.IsValid():
        continue

    print(f"\n[Contact Object] {co_path}")
    co_type = co_prim.GetTypeName()
    has_rb = co_prim.HasAPI(UsdPhysics.RigidBodyAPI)
    print(f"  Type: {co_type}, RigidBodyAPI: {has_rb}")

    co_world = UsdGeom.Xformable(co_prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    co_pos = co_world.ExtractTranslation()
    print(f"  World position: ({co_pos[0]:.4f}, {co_pos[1]:.4f}, {co_pos[2]:.4f})")

    # Find SDF meshes
    sdf_found = False
    for prim in Usd.PrimRange(co_prim):
        if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            approx = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
            is_sdf = approx == "sdf"
            print(f"  MeshCollisionAPI: {prim.GetPath()}")
            print(f"    approximation: {approx} {'(SDF OK)' if is_sdf else '(NOT SDF!)'}")
            sdf_found = is_sdf or sdf_found

        if prim.HasAPI(UsdPhysics.CollisionAPI) and prim.IsA(UsdGeom.Mesh):
            usd_mesh = UsdGeom.Mesh(prim)
            pts = np.array(usd_mesh.GetPointsAttr().Get())
            if len(pts) > 0:
                mesh_world = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
                    Usd.TimeCode.Default()
                )
                print(f"  Collision mesh: {prim.GetPath()} ({len(pts)} verts)")

    if not sdf_found:
        rb_prim_parent = None
        for prim in Usd.PrimRange(co_prim):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                rb_prim_parent = prim
                break
        if rb_prim_parent is None:
            for prim in Usd.PrimRange(co_prim):
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    approx_attr = prim.GetAttribute("physics:approximation")
                    approx = approx_attr.Get() if approx_attr else None
                    print(f"  CollisionAPI prim: {prim.GetPath()}, approx={approx}, type={prim.GetTypeName()}")

        print(f"  WARNING: No SDF mesh found! Force field requires SDF collision mesh.")
        print(f"  If using a CuboidCfg (Shape Prim), switch to a USD mesh object (e.g., nut).")


# ============================================================
# 3. Summary & recommendation
# ============================================================
print(f"\n{'=' * 70}")
print("SUMMARY")
print("=" * 70)
print("""
The VisuoTactileSensor reads mesh vertices via GetPointsAttr() which
returns points in the MESH's local frame. It then transforms them
using the RigidBody's world pose (get_transforms()).

If there are intermediate Xform transforms between the RigidBody
and the mesh prim, the tactile points will be placed at WRONG
world-space positions. This means:
  - SDF queries return positive values (outside the contact object)
  - Forces are always zero

Fix: account for the mesh-to-RigidBody relative transform, OR
bake the intermediate transforms into the mesh vertices.
""")
print("=" * 70)
