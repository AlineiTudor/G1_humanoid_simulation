"""
Debug utility to print USD prim hierarchy.

Usage in test scripts:
    # After creating scene, BEFORE sim_context.reset()
    scene = InteractiveScene(scene_cfg)

    from RH56E2_tactile_proj.utils.debug_prims import print_stage_hierarchy
    print_stage_hierarchy("/World/envs/env_0", max_depth=4)

    sim_context.reset()  # This is where errors happen
"""

import omni.usd
from pxr import Usd, UsdPhysics


def print_stage_hierarchy(
    root_path: str = "/World",
    max_depth: int = 4,
    show_apis: bool = True,
):
    """
    Print the USD prim hierarchy starting from root_path.

    Args:
        root_path: Starting prim path (e.g., "/World/envs/env_0" or "/World/envs/env_0/Robot")
        max_depth: Maximum depth to traverse
        show_apis: Show physics APIs on each prim
    """
    stage = omni.usd.get_context().get_stage()
    root_prim = stage.GetPrimAtPath(root_path)

    if not root_prim.IsValid():
        print(f"ERROR: Prim not found at '{root_path}'")
        print(f"\nAvailable top-level prims:")
        for prim in stage.GetPseudoRoot().GetChildren():
            print(f"  {prim.GetPath()}")
        return

    print(f"\n{'='*80}")
    print(f"USD Hierarchy: {root_path} (max_depth={max_depth})")
    print(f"{'='*80}\n")

    # Also count total prims for sanity check
    total = sum(1 for _ in Usd.PrimRange(root_prim))

    _print_prim_tree(root_prim, 0, max_depth, show_apis)

    print(f"\nTotal prims under {root_path}: {total}")
    print(f"{'='*80}\n")


def _print_prim_tree(prim, depth, max_depth, show_apis):
    """Recursively print prim tree."""
    if depth > max_depth:
        return

    indent = "  " * depth
    prim_type = prim.GetTypeName() or "(no type)"
    name = prim.GetName()

    # Build API markers
    markers = []
    if show_apis:
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            markers.append("ARTICULATION_ROOT")
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            markers.append("RIGID_BODY")
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            markers.append("COLLISION")
        if prim.IsA(UsdPhysics.Joint):
            markers.append("JOINT")
        if prim.IsInstance():
            markers.append("INSTANCED")

    marker_str = f"  <{'|'.join(markers)}>" if markers else ""

    print(f"{indent}{name} [{prim_type}]{marker_str}")

    for child in prim.GetChildren():
        _print_prim_tree(child, depth + 1, max_depth, show_apis)


def load_usd_directly(usd_path: str, target_prim: str = "/TestRobot"):
    """
    Load a USD file directly into the stage (bypassing Isaac Lab)
    to verify the file is valid and see its contents.

    Args:
        usd_path: Path to the USD file
        target_prim: Where to load it in the stage
    """
    import os
    print(f"\n{'='*80}")
    print(f"Direct USD Load Test")
    print(f"{'='*80}\n")
    print(f"  USD path: {usd_path}")
    print(f"  File exists: {os.path.exists(usd_path)}")

    if not os.path.exists(usd_path):
        print(f"\n  ERROR: File does NOT exist at '{usd_path}'!")
        # Try to find similar files nearby
        parent_dir = os.path.dirname(usd_path)
        if os.path.exists(parent_dir):
            print(f"  Files in {parent_dir}:")
            for f in os.listdir(parent_dir):
                print(f"    {f}")
        else:
            print(f"  Parent directory '{parent_dir}' also does not exist!")
        return

    file_size = os.path.getsize(usd_path)
    print(f"  File size: {file_size} bytes")

    # =========================================================
    # Step 1: Open the USD as its own stage to inspect internals
    # =========================================================
    print(f"\n  --- Inspecting USD file internals ---")
    try:
        file_stage = Usd.Stage.Open(usd_path)
        if file_stage is None:
            print(f"  ERROR: Could not open USD file as stage!")
            return

        # Check default prim
        default_prim = file_stage.GetDefaultPrim()
        if default_prim and default_prim.IsValid():
            print(f"  Default prim: {default_prim.GetPath()} [{default_prim.GetTypeName()}]")
        else:
            print(f"  WARNING: No default prim set!")
            print(f"  This is likely why AddReference loads 0 children.")
            print(f"  USD references use the default prim as the root.")

        # List root prims
        root_prims = list(file_stage.GetPseudoRoot().GetChildren())
        print(f"  Root prims ({len(root_prims)}):")
        for rp in root_prims:
            total_sub = sum(1 for _ in Usd.PrimRange(rp))
            print(f"    {rp.GetPath()} [{rp.GetTypeName()}] ({total_sub} prims)")

        # Check sublayers
        root_layer = file_stage.GetRootLayer()
        sublayers = root_layer.subLayerPaths
        if sublayers:
            print(f"\n  Sublayers ({len(sublayers)}):")
            for sl in sublayers:
                resolved = root_layer.ComputeAbsolutePath(sl)
                exists = os.path.exists(resolved) if resolved else False
                print(f"    {sl}")
                print(f"      resolved: {resolved}")
                print(f"      exists:   {exists}")
        else:
            print(f"\n  No sublayers.")

        # Check references on root prims
        print(f"\n  References on root prims:")
        for rp in root_prims:
            refs_and_layers = rp.GetMetadata("references")
            if refs_and_layers:
                print(f"    {rp.GetPath()}: has references")
                # Try to get reference details
                prim_spec = root_layer.GetPrimAtPath(rp.GetPath())
                if prim_spec:
                    ref_list = prim_spec.referenceList
                    for ref_item in ref_list.prependedItems:
                        ref_path = ref_item.assetPath
                        resolved_ref = root_layer.ComputeAbsolutePath(ref_path) if ref_path else ""
                        ref_exists = os.path.exists(resolved_ref) if resolved_ref else False
                        print(f"      ref: {ref_path}")
                        print(f"        resolved: {resolved_ref}")
                        print(f"        exists:   {ref_exists}")
                        if ref_item.primPath:
                            print(f"        primPath: {ref_item.primPath}")
            else:
                print(f"    {rp.GetPath()}: no references")

        # Total prim count in file
        total_in_file = sum(1 for _ in Usd.PrimRange(file_stage.GetPseudoRoot()))
        print(f"\n  Total prims in USD file: {total_in_file}")

        # Show hierarchy of file contents
        if total_in_file > 1:
            print(f"\n  File hierarchy:")
            for rp in root_prims:
                _print_prim_tree(rp, 2, 3, True)

    except Exception as e:
        print(f"  ERROR inspecting USD file: {e}")

    # =========================================================
    # Step 2: Load via AddReference into current stage
    # =========================================================
    print(f"\n  --- Loading via AddReference ---")
    stage = omni.usd.get_context().get_stage()
    prim = stage.DefinePrim(target_prim, "Xform")
    prim.GetReferences().AddReference(usd_path)

    # Count what loaded
    total = sum(1 for _ in Usd.PrimRange(prim))
    print(f"  Prims loaded via AddReference: {total}")

    if total <= 1:
        print(f"\n  WARNING: AddReference loaded no children!")
        print(f"  Trying AddReference with explicit prim path...")

        # If no default prim, try referencing each root prim explicitly
        try:
            file_stage = Usd.Stage.Open(usd_path)
            root_prims = list(file_stage.GetPseudoRoot().GetChildren())
            for rp in root_prims:
                test_path = f"{target_prim}_test_{rp.GetName()}"
                test_prim = stage.DefinePrim(test_path, "Xform")
                test_prim.GetReferences().AddReference(
                    usd_path, primPath=rp.GetPath()
                )
                test_total = sum(1 for _ in Usd.PrimRange(test_prim))
                print(f"    AddReference(primPath={rp.GetPath()}): {test_total} prims")
                if test_total > 1:
                    print(f"    SUCCESS! Use this prim path in your config.")
                    _print_prim_tree(test_prim, 3, 3, True)
        except Exception as e:
            print(f"    ERROR: {e}")
    else:
        print(f"\n  USD loaded successfully! Showing hierarchy:")
        _print_prim_tree(prim, 1, 4, True)

    print(f"\n{'='*80}\n")


def find_articulation_roots(root_path: str = "/World"):
    """Find and print all prims with ArticulationRootAPI."""
    stage = omni.usd.get_context().get_stage()
    root_prim = stage.GetPrimAtPath(root_path)

    if not root_prim.IsValid():
        print(f"ERROR: Prim not found at '{root_path}'")
        return

    print(f"\n{'='*80}")
    print(f"Searching for ArticulationRootAPI under: {root_path}")
    print(f"{'='*80}\n")

    found = False
    for prim in Usd.PrimRange(root_prim):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            print(f"  FOUND: {prim.GetPath()} [{prim.GetTypeName()}]")
            found = True

    if not found:
        print("  No prims with ArticulationRootAPI found!")
        print("\n  Prims with RigidBodyAPI (potential articulation roots):")
        for prim in Usd.PrimRange(root_prim):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                print(f"    {prim.GetPath()} [{prim.GetTypeName()}]")
