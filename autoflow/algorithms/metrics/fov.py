"""Physical FOV contact of the measured cross-section's outer contour."""

import numpy as np


def measure_fov_boundary_contact(region, plane, shape, spacing, origin, threshold):
    """Count contour segments on actual image box faces, not vessel walls.

    Use the largest enclosed boundary loop for the connected ROI so interior
    holes cannot dilute its contact fraction. Isolated tangent points add no
    length. The tolerance is numerical, not a voxel-wide exclusion margin.
    """
    spacing, origin = np.asarray(spacing, dtype=float), np.asarray(origin, dtype=float)
    lower, upper = origin, origin + np.asarray(shape, dtype=float) * spacing
    tolerance = max(1e-5, float(np.max(spacing)) * 1e-4)
    record = {"enabled": threshold is not None, "basis": "temporal_union",
              "max_fraction": threshold, "tolerance_mm": tolerance,
              "fov_bounds_mm": np.column_stack((lower, upper)).tolist(),
              "perimeter_mm": None, "contact_length_mm": None, "contact_fraction": None,
              "faces": [], "status": "disabled" if threshold is None else "no_roi"}
    if threshold is None or region is None or region.n_cells == 0:
        return record
    try:
        surface = region.extract_surface(algorithm="dataset_surface")
    except TypeError:  # PyVista versions before the algorithm keyword.
        surface = region.extract_surface()
    surface = surface.clean()
    edges = surface.extract_feature_edges(boundary_edges=True, feature_edges=False,
                                         manifold_edges=False, non_manifold_edges=False).strip()
    points, lines = np.asarray(edges.points, dtype=float), np.asarray(edges.lines, dtype=int)
    normal = np.array(plane.normal, dtype=float, copy=True)
    normal /= np.linalg.norm(normal) + 1e-12
    contours, cursor = [], 0
    while cursor < len(lines):
        count = int(lines[cursor])
        ids = lines[cursor + 1:cursor + 1 + count]
        cursor += count + 1
        if count < 4:
            continue
        contour = points[ids]
        if np.linalg.norm(contour[0] - contour[-1]) > tolerance:
            continue
        centered = contour - contour[0]
        area = abs(float(np.sum(np.cross(centered[:-1], centered[1:]) @ normal))) / 2
        contours.append((area, contour))
    if not contours:
        record["status"] = "contour_unavailable"
        return record
    contour = max(contours, key=lambda item: item[0])[1]
    start, end = contour[:-1], contour[1:]
    lengths = np.linalg.norm(end - start, axis=1)
    perimeter = float(np.sum(lengths))
    touched = np.zeros(len(lengths), dtype=bool)
    for axis, axis_name in enumerate(("x", "y", "z")):
        for side, boundary in (("min", lower[axis]), ("max", upper[axis])):
            contact = (np.abs(start[:, axis] - boundary) <= tolerance) & (np.abs(end[:, axis] - boundary) <= tolerance)
            length = float(np.sum(lengths[contact]))
            if length > tolerance:
                record["faces"].append({"face": f"{axis_name}_{side}", "contact_length_mm": length})
            touched |= contact
    length = float(np.sum(lengths[touched]))
    fraction = length / perimeter if perimeter > tolerance else None
    record.update({"perimeter_mm": perimeter, "contact_length_mm": length, "contact_fraction": fraction,
                   "status": "contour_unavailable" if fraction is None else
                   "excluded" if fraction + 1e-12 >= float(threshold) and length > tolerance else
                   "below_threshold" if length > tolerance else "clear"})
    return record
