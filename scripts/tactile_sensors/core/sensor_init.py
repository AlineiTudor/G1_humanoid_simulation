"""
Post-initialization helpers for tactile sensors.

Apply adjustments that can't be done through config alone, such as
offsetting tactile points outward past parent link collision surfaces.
"""

import torch


def apply_tactile_normal_offset(sensor, offset_meters: float):
    """Push tactile sensing points outward along the surface normal.

    The VisuoTactileSensor generates tactile points ON the sensor mesh surface.
    For multi-body hands (RH56E2), the sensor mesh is a 2mm-thick pad recessed
    behind parent link collision geometry. During grasping, parent collision
    stops the contact object before it overlaps with the sensor mesh, so SDF
    queries always return positive (outside) and forces are zero.

    This function offsets all tactile points along the outward-facing normal
    (the "tip direction"  same as the sensor's internal raycasting direction)
    so they extend past the parent collision into the contact zone.

    Args:
        sensor: A VisuoTactileSensor instance (already initialized).
        offset_meters: Distance to push points outward (meters).
            Typical values: 0.005-0.02 for small objects.
    """
    if offset_meters == 0.0:
        return

    if not hasattr(sensor, "_tactile_pos_local") or sensor._tactile_pos_local is None:
        return

    pts = sensor._tactile_pos_local  # (N, 3)
    if pts.numel() == 0:
        return

    # Find the slim axis (thinnest dimension  the surface normal axis)
    ranges = pts.max(0).values - pts.min(0).values
    slim_axis = ranges.argmin().item()

    # Determine the outward (tip) direction along the slim axis.
    #
    # Two cases based on how far the points are from the body origin:
    #
    # CASE A  Points far from origin (GelSight finger, COM Z  +0.068):
    #   The COM sign reliably indicates the outward direction. Push in the
    #   same direction as COM (further from origin = further outward).
    #
    # CASE B  Points essentially at origin (G1 force sensor pads, COM Z  -0.0015):
    #   The slim axis range is < 1mm (effectively a 2D sheet at the body origin).
    #   The COM sign is noise  1.5mm is below floating-point reliability for
    #   direction determination. In this case, push AWAY from COM (through the
    #   origin to the other side). Why: the tactile points sit on one face of a
    #   thin pad centered at the body origin. Pushing through the origin moves
    #   them past the collision surface on the opposite face, into the zone where
    #   contact objects overlap. This works because the offset distance (5mm) is
    #   much larger than the COM distance from origin (1.5mm).
    slim_range = ranges[slim_axis].item()
    com = pts.mean(0)[slim_axis].item()

    if slim_range > 0.001:
        # Case A: large range ? COM sign is reliable
        tip_sign = 1.0 if com >= 0 else -1.0
    else:
        # Case B: flat pad at origin ? flip sign (push through origin)
        tip_sign = -1.0 if com >= 0 else 1.0

    # Apply the offset along the slim axis in the tip direction
    offset = torch.zeros(3, device=pts.device)
    offset[slim_axis] = tip_sign * offset_meters
    sensor._tactile_pos_local = pts + offset.unsqueeze(0)

    # Update the pre-expanded tensor used during simulation
    sensor._tactile_pos_expanded = sensor._tactile_pos_local.unsqueeze(0).expand(
        sensor._num_envs, -1, -1
    )


def apply_offset_to_all_sensors(scene, sensor_names, offset_meters: float):
    """Apply tactile normal offset to all named sensors in the scene.

    Args:
        scene: InteractiveScene instance.
        sensor_names: List of sensor name keys in the scene.
        offset_meters: Distance to push points outward (meters).
    """
    if offset_meters == 0.0:
        return

    count = 0
    for name in sensor_names:
        sensor = scene[name]
        if hasattr(sensor, "_tactile_pos_local") and sensor._tactile_pos_local is not None:
            apply_tactile_normal_offset(sensor, offset_meters)
            count += 1

    if count > 0:
        print(f"  Applied tactile normal offset ({offset_meters*1000:.1f}mm) to {count} sensors")
