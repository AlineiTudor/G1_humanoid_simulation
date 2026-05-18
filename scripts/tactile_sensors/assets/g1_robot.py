""" G1 Humanoid Robot with RH56E2 Hands Asset Configuration"""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg


def _fix_usd_before_load(
    usd_path: str,
    source_prim_path: str,
    hand_mount_paths: dict = None,
):
    """
    Fix USD file issues before Isaac Lab loads it.

    Applies two permanent fixes (idempotent, saved to disk):
    1. Ensures defaultPrim points to the robot root.
    2. Removes extra ArticulationRootAPI from hand sub-trees, keeping
       only the body's articulation root (e.g., on pelvis). Isaac Lab
       requires exactly one ArticulationRootAPI per ArticulationCfg.

    Args:
        usd_path: Path to the USD file.
        source_prim_path: The prim path inside the USD that contains
            the actual robot (e.g., "/g1_29dof_with_hand_rev_1_0").
        hand_mount_paths: Dict mapping hand side to mount path from
            robot root. Used to identify hand sub-trees whose
            ArticulationRootAPI should be removed.
    """
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
    try:
        from pxr import PhysxSchema
    except ImportError:
        PhysxSchema = None

    if hand_mount_paths is None:
        hand_mount_paths = {
            "left": "left_hand",
            "right": "right_hand",
        }

    robot_root = source_prim_path.strip("/")

    # --- Fix 1: defaultPrim (Sdf layer level) ---
    layer = Sdf.Layer.FindOrOpen(usd_path)
    if layer is None:
        raise FileNotFoundError(f"Cannot open USD layer: {usd_path}")

    modified = False
    if layer.defaultPrim != robot_root:
        print(f"  [g1_robot] Fixing USD default prim: "
              f"'{layer.defaultPrim}' -> '{robot_root}'")
        layer.defaultPrim = robot_root
        modified = True
    else:
        print(f"  [g1_robot] USD default prim already correct: '{robot_root}'")

    if modified:
        layer.Save()
        print(f"  [g1_robot] Saved defaultPrim fix to: {usd_path}")

    # --- Fix 2: Remove hand ArticulationRootAPI (composed stage level) ---
    # Must use Usd.Stage.Open() because the ArticulationRootAPI comes from
    # referenced hand USD files, not from the top-level layer. Sdf.Layer
    # only sees the top layer's own prim specs.
    stage = Usd.Stage.Open(usd_path)
    modified = False

    for side, mount in hand_mount_paths.items():
        hand_path = f"/{robot_root}/{mount}"
        hand_prim = stage.GetPrimAtPath(hand_path)
        if not hand_prim.IsValid():
            continue

        for prim in Usd.PrimRange(hand_prim):
            # Fix 2a: Remove hand ArticulationRootAPI
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                print(f"  [g1_robot] Removing ArticulationRootAPI from: {prim.GetPath()}")
                prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
                modified = True

            prim_name = prim.GetName()
            prim_path_str = str(prim.GetPath())

            # Fix 2b: Ensure force sensor collision prims have CollisionAPI
            # with convexHull approximation. The sensor finds visual meshes
            # from the elastomer sub-prim, not /collisions, so collision
            # must stay on /collisions for physical contact to work.
            # Fix 3 (below) sets tight contact_offset/rest_offset so the
            # nut can get close enough for SDF overlap despite collision.
            if ("force_sensor" in prim_path_str
                    and "/collisions" in prim_path_str
                    and prim.IsA(UsdGeom.Mesh)):
                if not prim.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI.Apply(prim)
                    print(f"  [g1_robot] Restored CollisionAPI on force sensor: {prim.GetPath()}")
                    modified = True
                if not prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                    UsdPhysics.MeshCollisionAPI.Apply(prim)
                    print(f"  [g1_robot] Applied MeshCollisionAPI on: {prim.GetPath()}")
                    modified = True
                mesh_col_api = UsdPhysics.MeshCollisionAPI(prim)
                current = mesh_col_api.GetApproximationAttr().Get()
                if current != "convexHull":
                    mesh_col_api.GetApproximationAttr().Set("convexHull")
                    print(f"  [g1_robot] Set convexHull on force sensor: {prim.GetPath()}")
                    modified = True

            # Fix 2c: Set convexHull on non-force-sensor collision prims
            # that already have CollisionAPI (body links, etc).
            if (prim_name == "collisions"
                    and "force_sensor" not in prim_path_str
                    and prim.HasAPI(UsdPhysics.CollisionAPI)):
                if not prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                    UsdPhysics.MeshCollisionAPI.Apply(prim)
                mesh_col_api = UsdPhysics.MeshCollisionAPI(prim)
                current = mesh_col_api.GetApproximationAttr().Get()
                if current != "convexHull":
                    mesh_col_api.GetApproximationAttr().Set("convexHull")
                    print(f"  [g1_robot] Set convexHull on: {prim.GetPath()}")
                    modified = True

            # Fix 2d: Increase mimic joint damping ratio.
            # USD default is 0.005 (very underdamped ? wobble).
            # Set to 1.0 (critically damped) for rigid gear-like behavior.
            # Per Isaac Sim docs: mimic joints should NOT have their own
            # drive  PhysX copies the drive from the reference joint.
            # Also remove any joint drive on mimic joints so PhysX
            # MimicJointAPI handles them natively.
            damping_attr = prim.GetAttribute("physxMimicJoint:rotX:dampingRatio")
            if damping_attr and damping_attr.IsValid():
                current_damping = damping_attr.Get()
                if current_damping is not None and current_damping < 0.5:
                    damping_attr.Set(1.0)
                    print(f"  [g1_robot] Set mimic dampingRatio=1.0 on: {prim.GetPath()}")
                    modified = True
                # Remove any joint drive on this mimic joint (per Isaac Sim docs:
                # "Remove or set all values to zero in the joint drive")
                if prim.HasAPI(UsdPhysics.DriveAPI):
                    prim.RemoveAPI(UsdPhysics.DriveAPI)
                    print(f"  [g1_robot] Removed DriveAPI from mimic: {prim.GetPath()}")
                    modified = True

    # --- Fix 3: Set contact_offset / rest_offset on ALL collision prims ---
    # Isaac Lab's collision_props uses @apply_nested at runtime, but it
    # FAILS because collision sub-prims are instanced (URDF-to-USD creates
    # instanceable visual/collision sub-prims). Setting these in the source
    # USD before load bypasses the instancing problem  the authored values
    # are part of the prototype and inherited by all instances.
    robot_prim = stage.GetPrimAtPath(f"/{robot_root}")
    if robot_prim.IsValid():
        contact_offset = 0.001
        rest_offset = -0.0005
        fix3_count = 0
        for prim in Usd.PrimRange(robot_prim):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            # Apply PhysxCollisionAPI schema so PhysX recognises the attrs
            if PhysxSchema is not None:
                if not prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
                    PhysxSchema.PhysxCollisionAPI.Apply(prim)
            # Set contact offset
            co_attr = prim.GetAttribute("physxCollision:contactOffset")
            if not co_attr or not co_attr.IsValid():
                co_attr = prim.CreateAttribute(
                    "physxCollision:contactOffset",
                    Sdf.ValueTypeNames.Float
                )
            if co_attr.Get() != contact_offset:
                co_attr.Set(contact_offset)
                fix3_count += 1
            # Set rest offset
            ro_attr = prim.GetAttribute("physxCollision:restOffset")
            if not ro_attr or not ro_attr.IsValid():
                ro_attr = prim.CreateAttribute(
                    "physxCollision:restOffset",
                    Sdf.ValueTypeNames.Float
                )
            if ro_attr.Get() != rest_offset:
                ro_attr.Set(rest_offset)
                fix3_count += 1
        if fix3_count > 0:
            print(f"  [g1_robot] Set contact_offset={contact_offset}, "
                  f"rest_offset={rest_offset} on collision prims "
                  f"({fix3_count} attrs updated)")
            modified = True

    # --- Fix 4: Create & bind compliant contact material on force sensor prims ---
    #
    # WHY THIS IS NEEDED:
    # The TacSL force field pipeline requires tactile points to OVERLAP with the
    # contact object mesh (SDF < 0) to produce forces. PhysX collision normally
    # prevents any overlap. Compliant contact material changes PhysX behavior:
    # instead of hard rigid collision, it uses implicit springs (stiffness + damping)
    # that allow controlled penetration through the collision surface. This small
    # penetration (~1mm) is enough for tactile points to enter the contact object's
    # SDF field and register forces.
    #
    # WHY IT MUST BE DONE HERE (in the source USD):
    # Isaac Lab's runtime bind_physics_material() uses @apply_nested, which SKIPS
    # instanced prims. The G1 robot's URDF-to-USD conversion creates instanceable
    # collision sub-prims, so the runtime binding silently fails. The compliant
    # material IS created (verified in diagnostic logs), but never BOUND to the
    # collision geometry  PhysX ignores it.
    #
    # By creating and binding the material here in the source USD (before Isaac Lab
    # instances it), the binding becomes part of the prototype and applies to all
    # instances automatically.
    #
    # REFERENCE: This replicates what spawn_rigid_body_material() + bind_physics_material()
    # do in the GelSight finger example (where it works because the USD is not instanced).
    # See: isaaclab/sim/spawners/materials/physics_materials.py (material creation)
    #      isaaclab/sim/utils/prims.py (material binding)
    if robot_prim.IsValid() and PhysxSchema is not None:
        # Step 1: Create a single shared compliant contact material prim.
        # One material for all force sensors  they share the same stiffness/damping.
        mat_path = f"/{robot_root}/force_sensor_compliant_material"
        mat_prim = stage.GetPrimAtPath(mat_path)
        if not mat_prim.IsValid():
            mat_prim = UsdShade.Material.Define(stage, mat_path).GetPrim()

        # Step 2: Apply physics material APIs and set compliant contact properties.
        # UsdPhysics.MaterialAPI: standard rigid body material (friction, restitution).
        # PhysxSchema.PhysxMaterialAPI: PhysX-specific extensions including compliant
        # contact springs. Both are needed  PhysX reads from PhysxMaterialAPI.
        if not mat_prim.HasAPI(UsdPhysics.MaterialAPI):
            UsdPhysics.MaterialAPI.Apply(mat_prim)
        if not mat_prim.HasAPI(PhysxSchema.PhysxMaterialAPI):
            PhysxSchema.PhysxMaterialAPI.Apply(mat_prim)

        physx_mat = PhysxSchema.PhysxMaterialAPI(mat_prim)
        # compliantContactStiffness: spring constant for the implicit contact springs.
        # Higher = stiffer (closer to rigid). Must be > 0 to enable compliant mode.
        # 100.0 matches the GelSight example that produces forces successfully.
        physx_mat.GetCompliantContactStiffnessAttr().Set(100.0)
        # compliantContactDamping: viscous damping for the contact springs.
        # Prevents oscillation. 10.0 matches the GelSight example.
        physx_mat.GetCompliantContactDampingAttr().Set(10.0)

        # Step 3: Bind the material to every force sensor collision mesh.
        # Detection pattern matches Fix 2b: prims whose path contains "force_sensor"
        # AND "/collisions" AND that are Mesh prims (the actual collision geometry).
        # The binding uses materialPurpose="physics" so it only affects PhysX collision
        # behavior, not visual rendering.
        material = UsdShade.Material(mat_prim)
        fix4_count = 0
        for prim in Usd.PrimRange(robot_prim):
            prim_path_str = str(prim.GetPath())
            if ("force_sensor" in prim_path_str
                    and "/collisions" in prim_path_str
                    and prim.IsA(UsdGeom.Mesh)):
                binding_api = UsdShade.MaterialBindingAPI.Apply(prim)
                binding_api.Bind(
                    material,
                    bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                    materialPurpose="physics",
                )
                fix4_count += 1

        if fix4_count > 0:
            print(f"  [g1_robot] Bound compliant contact material "
                  f"(stiffness=100, damping=10) to {fix4_count} force sensor "
                  f"collision prims")
            modified = True

    if modified:
        stage.GetRootLayer().Save()
        print(f"  [g1_robot] Saved USD fixes to: {usd_path}")
    else:
        print(f"  [g1_robot] Hand USD already clean (no fixes needed)")


def fix_robot_articulation_roots(
    num_envs: int,
    robot_prim_name: str = "Robot",
    hand_mount_paths: dict = None,
):
    """
    Fix the articulation hierarchy for G1 robot with attached hands.

    Does two things:
    1. Removes extra ArticulationRootAPI from hand root_joint prims, keeping
       the body's articulation root (e.g., on pelvis). Isaac Lab requires
       exactly one ArticulationRootAPI per ArticulationCfg.
    2. Adds ResetXformStack to hand rigid bodies that are nested under body
       rigid bodies, preventing PhysX "missing xformstack reset" errors.

    Must be called AFTER InteractiveScene(scene_cfg) but BEFORE sim_context.reset().

    Args:
        num_envs: Number of environments.
        robot_prim_name: Name of the robot prim under each environment.
        hand_mount_paths: Dict mapping hand side to mount path from robot root,
            e.g. {"left": "left_wrist_yaw_link/left_hand_palm_link", ...}.
            Used to identify which sub-trees are hands (vs body).
    """
    import omni.usd
    from pxr import Usd, UsdPhysics

    if hand_mount_paths is None:
        hand_mount_paths = {
            "left": "left_wrist_yaw_link/left_hand_palm_link",
            "right": "right_wrist_yaw_link/right_hand_palm_link",
        }

    stage = omni.usd.get_context().get_stage()

    for env_idx in range(num_envs):
        robot_path = f"/World/envs/env_{env_idx}/{robot_prim_name}"
        robot_prim = stage.GetPrimAtPath(robot_path)

        if not robot_prim.IsValid():
            print(f"  WARNING: Robot not found at {robot_path}")
            continue

        # --- Step 1: Find and manage ArticulationRootAPI ---
        # Find ALL prims with ArticulationRootAPI
        art_roots = []
        for prim in Usd.PrimRange(robot_prim):
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                art_roots.append(prim)

        print(f"  [g1_robot] env_{env_idx}: found {len(art_roots)} ArticulationRootAPI(s):")
        for ar in art_roots:
            print(f"    {ar.GetPath()}")

        # Determine which roots are from hands (under hand mount paths)
        hand_root_paths = set()
        for side, mount in hand_mount_paths.items():
            hand_subtree = f"{robot_path}/{mount}"
            hand_root_paths.add(hand_subtree)

        for ar in art_roots:
            ar_path_str = str(ar.GetPath())
            # Check if this root is under a hand mount path
            is_hand_root = any(ar_path_str.startswith(hp) for hp in hand_root_paths)
            if is_hand_root:
                print(f"  [g1_robot] Removing hand ArticulationRootAPI: {ar.GetPath()}")
                ar.RemoveAPI(UsdPhysics.ArticulationRootAPI)

        # Verify exactly one remains
        remaining = [p for p in Usd.PrimRange(robot_prim)
                     if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
        if len(remaining) == 1:
            print(f"  [g1_robot] Articulation root: {remaining[0].GetPath()}")
        elif len(remaining) == 0:
            # Fallback: no body root found, apply to robot prim
            print(f"  [g1_robot] WARNING: No articulation root found, applying to {robot_path}")
            UsdPhysics.ArticulationRootAPI.Apply(robot_prim)
        else:
            print(f"  [g1_robot] WARNING: {len(remaining)} articulation roots remain!")

        # NOTE: Hand rigid bodies are nested under body rigid bodies
        # (e.g., l_base_link under left_wrist_yaw_link). This causes
        # "missing xformstack reset" warnings from PhysX. These are
        # cosmetic  adding ResetXformStack would disconnect the hands
        # from the body transform hierarchy, placing them at the origin.
        # The warnings can be safely ignored.


def create_robot_articulation_cfg(
        robot_usd_path: str,
        prim_path: str = "{ENV_REGEX_NS}/Robot",
        source_prim_path: str = "/g1_29dof_with_hand_rev_1_0",
        fix_root_link: bool = True,
) -> ArticulationCfg:
    """Creates Articulation for G1 robot with hands already attached.

    Args:
        robot_usd_path: Path to the robot USD file.
        prim_path: Scene prim path where the robot will be spawned.
        source_prim_path: Root prim path inside the USD file.
            Needed when the USD's defaultPrim doesn't point to the robot.
        fix_root_link: If True, pin the pelvis to its spawn position.
    """
    # Fix USD issues (defaultPrim + hand ArticulationRootAPI) before Isaac Lab loads it
    _fix_usd_before_load(robot_usd_path, source_prim_path)

    # Build list of force sensor link paths for compliant contact.
    # Paths must be RELATIVE to the articulation root (Robot/).
    # Bind the material to the force sensor link PARENTS  the compliant
    # contact material propagates from parent to child collision prims.
    # (Cannot bind to collision meshes directly because they are instanced.)
    force_sensor_links = []
    for side in ("left", "right"):
        hand_mount = f"{side}_hand"
        # Palm force sensor link
        force_sensor_links.append(f"{hand_mount}/{side}_palm_force_sensor")
        # Fingers: index, middle, little have 3 regions; thumb has 4
        for finger, n_regions in [("index", 3), ("middle", 3), ("little", 3), ("thumb", 4)]:
            for region in range(1, n_regions + 1):
                force_sensor_links.append(f"{hand_mount}/{side}_{finger}_force_sensor_{region}")

    return ArticulationCfg(
        prim_path = prim_path,
        spawn = sim_utils.UsdFileWithCompliantContactCfg(
            usd_path = robot_usd_path,
            # Compliant contact for tactile sensor elastomer links
            compliant_contact_stiffness = 100.0,
            compliant_contact_damping = 10.0,
            physics_material_prim_path = force_sensor_links,
            rigid_props = sim_utils.RigidBodyPropertiesCfg(
                disable_gravity = False,
                max_depenetration_velocity = 1.0,
                max_linear_velocity = 10.0,
                max_angular_velocity = 20.0,
                max_contact_impulse = 1.0,    # cap contact force impulse
                linear_damping = 1.0,         # drag on linear motion
                angular_damping = 1.0,        # drag on rotation
            ),
            # NOTE: collision_props is NOT set here because it fails at
            # runtime  all collision sub-prims are instanced, so Isaac
            # Lab's modify_collision_properties cannot reach them. Instead,
            # Fix 3 in _fix_usd_before_load() sets contact_offset and
            # rest_offset directly in the source USD before instancing.
            articulation_props = sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions = True,
                solver_position_iteration_count = 12,
                solver_velocity_iteration_count = 1,
                fix_root_link = fix_root_link,
            ),
        ),
        init_state = ArticulationCfg.InitialStateCfg(
            pos = (0.0, 0.0, 0.78) # just above ground (minimize free-fall)
        ),
        actuators = {
            "leg_joints": ImplicitActuatorCfg(
                joint_names_expr = [".*_hip_.*", ".*_knee_.*", ".*_ankle_.*"],
                stiffness = 5000.0,
                damping = 100.0,
            ),
            "torso_arm_joints": ImplicitActuatorCfg(
                joint_names_expr = ["waist_.*",
                                    ".*_shoulder_.*", ".*_elbow_.*",
                                    ".*_wrist_.*"],
                stiffness = 200.0,
                damping = 20.0,
            ),
            # All hand joints (master + mimic) in one group.
            # Matching the reference unitree_sim_isaaclab pattern:
            # PhysX MimicJointAPI overrides position targets for mimic
            # joints during the physics step. The drive (stiffness/damping)
            # then enforces the mimic-computed target. effort_limit must
            # be > 0 (USD has 0.0 for mimic joints from URDF convention).
            "hand_joints": ImplicitActuatorCfg(
                joint_names_expr = [".*_index_.*_joint", ".*_middle_.*_joint",
                                    ".*_ring_.*_joint", ".*_little_.*_joint",
                                    ".*_thumb_.*_joint"],
                effort_limit = 300.0,
                velocity_limit = 100.0,
                stiffness = 100.0,
                damping = 10.0,
                armature = 0.1,
            ),
        },
    )

