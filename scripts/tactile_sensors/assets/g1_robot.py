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
    from pxr import Sdf, Usd, UsdPhysics

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

            # Fix 2b: Set collision prims to convexHull approximation.
            # Only fix prims that ALREADY have CollisionAPI  never apply
            # CollisionAPI to new prims (force sensor "collisions" scopes
            # are not meant to be collision objects).
            prim_name = prim.GetName()
            parent = prim.GetParent()
            parent_name = parent.GetName() if parent else ""

            # Fix 2b: Remove CollisionAPI from force sensor prims
            # (not meant to be collision objects).
            if "force_sensor" in parent_name and prim.HasAPI(UsdPhysics.CollisionAPI):
                prim.RemoveAPI(UsdPhysics.CollisionAPI)
                print(f"  [g1_robot] Removed CollisionAPI from: {prim.GetPath()}")
                modified = True
            # Fix 2c: Set convexHull on collision prims that already
            # have CollisionAPI. PhysX requires convexHull for dynamic
            # rigid body collisions.
            elif prim_name == "collisions" and prim.HasAPI(UsdPhysics.CollisionAPI):
                approx_attr = prim.GetAttribute("physics:approximation")
                current = approx_attr.Get() if approx_attr and approx_attr.HasValue() else None
                if current != "convexHull":
                    prim.CreateAttribute(
                        "physics:approximation", Sdf.ValueTypeNames.Token
                    ).Set("convexHull")
                    print(f"  [g1_robot] Set convexHull on: {prim.GetPath()}")
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

    return ArticulationCfg(
        prim_path = prim_path,
        spawn = sim_utils.UsdFileCfg(
            usd_path = robot_usd_path,
            rigid_props = sim_utils.RigidBodyPropertiesCfg(
                disable_gravity = False,
                max_depenetration_velocity = 1.0,
                max_linear_velocity = 10.0,
                max_angular_velocity = 20.0,
                max_contact_impulse = 1.0,    # cap contact force impulse
                linear_damping = 1.0,         # drag on linear motion
                angular_damping = 1.0,        # drag on rotation
            ),
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
            "hand_joints": ImplicitActuatorCfg(
                joint_names_expr = [".*_index_.*", ".*_middle_.*", ".*_ring_.*",
                                    ".*_little_.*", ".*_thumb_.*"],
                stiffness = 50.0,
                damping = 20.0,
            ),
        },
    )

