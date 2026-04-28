from pxr import UsdGeom
import omni.usd

stage = omni.usd.get_context().get_stage()
xform_cache = UsdGeom.XformCache()

# For each hand: get palm_link world transform, apply to hand prim
for side, palm_path, hand_path in [
    ("left",
     "/g1_29dof_with_hand_rev_1_0/left_wrist_yaw_link/left_hand_palm_link",
     "/g1_29dof_with_hand_rev_1_0/left_hand"),
    ("right",
     "/g1_29dof_with_hand_rev_1_0/right_wrist_yaw_link/right_hand_palm_link",
     "/g1_29dof_with_hand_rev_1_0/right_hand"),
]:
    palm = stage.GetPrimAtPath(palm_path)
    world_tf = xform_cache.GetLocalToWorldTransform(palm)

    hand = stage.GetPrimAtPath(hand_path)
    xformable = UsdGeom.Xformable(hand)
    xformable.ClearXformOpOrder()
    xformable.AddTransformOp().Set(world_tf)

    print(f"{side}: applied world transform from {palm_path}")
