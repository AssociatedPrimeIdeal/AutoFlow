"""Label-preserving cleanup and display surfaces for the paper figures."""
from __future__ import annotations

import numpy as np


def clean_grouped_labels(labels, spacing, groups):
    """Use the same per-phase anatomical-group CC rule as PipelineEngine.

    Input labels are never merged in the returned array. Grouping only decides
    connected support; retained voxels keep their original integer label.
    """
    from autoflow.algorithms.preprocess import filter_connected_components

    source = np.asarray(labels)
    if source.ndim not in (3, 4):
        raise ValueError(f"Expected XYZ or XYZT labels, got {source.shape}")
    temporal = source[..., None] if source.ndim == 3 else source
    retained = np.zeros(temporal.shape, dtype=bool)
    configured = set()
    for group in groups.values():
        ids = [int(x) for x in group["labels"]]
        configured.update(ids)
        params = group["params"]
        for t in range(temporal.shape[3]):
            mask = np.isin(temporal[..., t], ids)
            if params.get("remove_small_cc", True):
                mask = filter_connected_components(
                    mask, spacing, mode=params.get("cc_filter_mode", "hybrid"),
                    min_volume_mm3=float(params.get("min_cc_volume_mm3", 50)),
                    rel_min_ratio=float(params.get("cc_rel_min_ratio", .1)),
                    connectivity=int(params.get("connectivity", 1)),
                )
            retained[..., t] |= mask
    # A figure preset must not silently discard labels absent from its groups.
    unconfigured = set(int(x) for x in np.unique(temporal) if x) - configured
    if unconfigured:
        raise ValueError(f"Labels missing from cleanup groups: {sorted(unconfigured)}")
    cleaned = np.where(retained, temporal, 0).astype(source.dtype)
    assert np.all((cleaned == 0) | (cleaned == temporal))
    return cleaned[..., 0] if source.ndim == 3 else cleaned


def labeled_surface(labels, spacing, origin, smooth_iterations=60,
                    taubin_iterations=80, pass_band=.08):
    """Create the reference figure's display mesh; leave voxel labels intact."""
    from scipy.spatial import cKDTree
    from autoflow.algorithms.surfaces import build_cell_mask_surface

    labels = np.asarray(labels)
    points = np.argwhere(labels > 0)
    if not len(points):
        return None
    spacing, origin = np.asarray(spacing), np.asarray(origin)
    mesh = build_cell_mask_surface(
        labels > 0, spacing, origin - .5 * spacing,
        smooth_iter=int(smooth_iterations),
    ).triangulate()
    if taubin_iterations:
        mesh = mesh.smooth_taubin(n_iter=int(taubin_iterations), pass_band=pass_band)
    _, nearest = cKDTree(points * spacing + origin).query(mesh.points)
    mesh["Label"] = labels[tuple(points[nearest].T)]
    return mesh


def change_statistics(before, after):
    before, after = np.asarray(before), np.asarray(after)
    if before.shape != after.shape or np.any((after != 0) & (after != before)):
        raise ValueError("Cleanup may only remove source labels on the same grid")
    before = before[..., None] if before.ndim == 3 else before
    after = after[..., None] if after.ndim == 3 else after
    rows = []
    for t in range(before.shape[3]):
        n0, n1 = int(np.count_nonzero(before[..., t])), int(np.count_nonzero(after[..., t]))
        rows.append(dict(phase=t, before_voxels=n0, after_voxels=n1,
                         removed_voxels=n0-n1, removed_percent=100*(n0-n1)/max(n0, 1)))
    return rows
