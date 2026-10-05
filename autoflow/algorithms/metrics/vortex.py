"""Velocity-gradient vortex identifiers and swirling strength."""

import numpy as np
from scipy.ndimage import binary_erosion, gaussian_filter, generate_binary_structure

from ._common import _ensure_flow5d, _ensure_mask4d


def compute_vortex_metrics(mask4d, flow, spacing, *, smoothing_sigma=0.0,
                           support_erosion_iters=1):
    """Compute velocity-gradient vortex descriptors inside an eroded lumen mask.

    ``flow`` is the normalized XYZTV velocity field in cm/s and ``spacing`` is
    in mm.  Calculations are carried out in SI units so vorticity and swirling
    strength are returned in s^-1 and Q is returned in s^-2.  The public
    arrays retain the input shape, while ``vortex_support_mask`` records the
    voxels for which all results are considered valid.
    """
    mask4d = _ensure_mask4d(mask4d)
    flow = _ensure_flow5d(flow)
    if flow.shape[:3] != mask4d.shape[:3]:
        raise ValueError(
            f"flow spatial shape {flow.shape[:3]} does not match mask {mask4d.shape[:3]}"
        )
    if flow.shape[3] != mask4d.shape[3]:
        raise ValueError(f"flow time dimension {flow.shape[3]} does not match mask {mask4d.shape[3]}")

    spacing = np.asarray(spacing, dtype=float).reshape(3)
    if not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError('Vortex spacing must be finite and positive (mm)')
    if not np.isfinite(smoothing_sigma):
        raise ValueError('Vortex smoothing sigma must be finite')

    spatial_shape = tuple(int(value) for value in flow.shape[:3])
    output_shape = flow.shape[:4]
    vorticity = np.zeros(flow.shape, dtype=np.float32)
    vorticity_magnitude = np.zeros(output_shape, dtype=np.float32)
    q_criterion = np.zeros(output_shape, dtype=np.float32)
    swirling_strength = np.zeros(output_shape, dtype=np.float32)
    support = np.zeros(output_shape, dtype=bool)

    # A one-voxel halo is required for the central spatial-difference stencil.
    # Crop to the vessel extent for predictable memory use on large images.
    union_mask = np.any(mask4d, axis=3)
    occupied = np.where(union_mask)
    if occupied[0].size == 0:
        return {
            "vorticity_array": vorticity,
            "vorticity_magnitude": vorticity_magnitude,
            "vorticity_magnitude_peak": np.zeros(spatial_shape, dtype=np.float32),
            "q_criterion_array": q_criterion,
            "q_criterion_peak": np.zeros(spatial_shape, dtype=np.float32),
            "swirling_strength_array": swirling_strength,
            "swirling_strength_peak": np.zeros(spatial_shape, dtype=np.float32),
            "vortex_support_mask": support.astype(np.uint8),
        }
    spatial_slices = tuple(
        slice(max(int(np.min(axis_values)) - 1, 0), min(int(np.max(axis_values)) + 2, spatial_shape[axis]))
        for axis, axis_values in enumerate(occupied)
    )
    work_mask = np.asarray(mask4d[spatial_slices + (slice(None),)], dtype=bool)
    work_flow = np.asarray(flow[spatial_slices + (slice(None), slice(None))], dtype=np.float32)
    if any(size < 3 for size in work_flow.shape[:3]):
        return {
            "vorticity_array": vorticity,
            "vorticity_magnitude": vorticity_magnitude,
            "vorticity_magnitude_peak": np.zeros(spatial_shape, dtype=np.float32),
            "q_criterion_array": q_criterion,
            "q_criterion_peak": np.zeros(spatial_shape, dtype=np.float32),
            "swirling_strength_array": swirling_strength,
            "swirling_strength_peak": np.zeros(spatial_shape, dtype=np.float32),
            "vortex_support_mask": support.astype(np.uint8),
        }

    # Convert cm/s to m/s before differentiating over meter spacing.
    velocity = work_flow / 100.0
    finite = np.all(np.isfinite(work_flow), axis=-1)
    valid_samples = work_mask & finite
    mask_float = valid_samples.astype(np.float32)
    velocity = np.where(valid_samples[..., None], velocity, 0.0)
    sigma = max(float(smoothing_sigma), 0.0)
    if sigma > 0.0:
        # Normalize the filtered field by filtered mask weights so a zero
        # background does not artificially damp values near the valid support.
        weights = gaussian_filter(mask_float, sigma=(sigma, sigma, sigma, 0.0), mode="nearest")
        weights = np.maximum(weights, np.float32(1e-6))
        for component in range(3):
            velocity[..., component] = (
                gaussian_filter(velocity[..., component], sigma=(sigma, sigma, sigma, 0.0), mode="nearest")
                / weights
            )
        velocity *= mask_float[..., None]

    # Central differences always need finite lumen neighbours, even when
    # optional extra erosion is disabled.
    support_work = np.zeros(work_mask.shape, dtype=bool)
    for tidx in range(work_mask.shape[3]):
        support_work[..., tidx] = binary_erosion(
            valid_samples[..., tidx], structure=generate_binary_structure(3, 1), border_value=0)
    erosion_iters = max(int(support_erosion_iters), 0)
    if erosion_iters > 0:
        structure = np.ones((3, 3, 3), dtype=bool)
        for tidx in range(work_mask.shape[3]):
            support_work[..., tidx] &= binary_erosion(
                work_mask[..., tidx],
                structure=structure,
                iterations=erosion_iters,
                border_value=0,
            )
    support_inner = support_work[1:-1, 1:-1, 1:-1, :]

    spacing_m = np.asarray(spacing, dtype=float).reshape(3) / 1000.0
    dx, dy, dz = [float(max(value, 1e-12)) for value in spacing_m]
    # The descriptors outside support are zero by definition. Gather the
    # same central-difference neighbours only for valid voxels so background
    # tensors, eigensolves, and their large intermediate arrays are avoided.
    support_indices = np.nonzero(support_inner)
    centers = tuple(index + 1 for index in support_indices[:3]) + (support_indices[3],)
    jacobian = np.empty((len(support_indices[0]), 3, 3), dtype=np.float32)
    spacings = (dx, dy, dz)
    for axis, step in enumerate(spacings):
        before = list(centers)
        after = list(centers)
        before[axis] = before[axis] - 1
        after[axis] = after[axis] + 1
        for component in range(3):
            jacobian[..., component, axis] = (
                velocity[tuple(after) + (component,)]
                - velocity[tuple(before) + (component,)]
            ) / (2.0 * step)

    vort_inner = np.empty(jacobian.shape[:-2] + (3,), dtype=np.float32)
    vort_inner[..., 0] = jacobian[..., 2, 1] - jacobian[..., 1, 2]
    vort_inner[..., 1] = jacobian[..., 0, 2] - jacobian[..., 2, 0]
    vort_inner[..., 2] = jacobian[..., 1, 0] - jacobian[..., 0, 1]
    vortmag_inner = np.sqrt(np.sum(np.square(vort_inner, dtype=np.float32), axis=-1)).astype(np.float32)

    strain = 0.5 * (jacobian + np.swapaxes(jacobian, -1, -2))
    rotation = 0.5 * (jacobian - np.swapaxes(jacobian, -1, -2))
    q_inner = 0.5 * (
        np.sum(np.square(rotation, dtype=np.float32), axis=(-2, -1))
        - np.sum(np.square(strain, dtype=np.float32), axis=(-2, -1))
    )

    # λci is the positive imaginary part of the complex-conjugate eigenvalue
    # pair of the local velocity-gradient tensor.  It is zero for pure shear.
    eigvals = np.linalg.eigvals(jacobian)
    lambda_ci_inner = np.max(np.abs(np.imag(eigvals)), axis=1).astype(np.float32)

    output_slices = tuple(slice(part.start + 1, part.stop - 1) for part in spatial_slices) + (slice(None),)
    vorticity[output_slices][support_inner] = vort_inner
    vorticity_magnitude[output_slices][support_inner] = vortmag_inner
    q_criterion[output_slices][support_inner] = q_inner
    swirling_strength[output_slices][support_inner] = lambda_ci_inner
    support[output_slices] = support_inner
    return {
        "vorticity_array": vorticity,
        "vorticity_magnitude": vorticity_magnitude,
        "vorticity_magnitude_peak": np.max(vorticity_magnitude, axis=3).astype(np.float32),
        "q_criterion_array": q_criterion,
        "q_criterion_peak": np.max(q_criterion, axis=3).astype(np.float32),
        "swirling_strength_array": swirling_strength,
        "swirling_strength_peak": np.max(swirling_strength, axis=3).astype(np.float32),
        "vortex_support_mask": support.astype(np.uint8),
    }
