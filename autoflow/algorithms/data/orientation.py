"""Spatial-axis and velocity-component reorientation for normalized inputs."""

import numpy as np


def _axis_pair(a):
    mp = {
        "LR": ("LR", "RL"), "RL": ("LR", "RL"),
        "AP": ("AP", "PA"), "PA": ("AP", "PA"),
        "HF": ("HF", "FH"), "FH": ("HF", "FH"),
    }
    a = a.upper()
    if a not in mp:
        raise ValueError(a)
    return mp[a][0]


def _need_flip(curr_label, target_label):
    c, t = curr_label.upper(), target_label.upper()
    if _axis_pair(c) != _axis_pair(t):
        raise ValueError(f"{c} vs {t}")
    return c != t


def _permute_spatial(arr, curr_order, target_order, spatial_axes=(0, 1, 2)):
    curr_order = [x.upper() for x in curr_order]
    target_order = [x.upper() for x in target_order]
    cb = [_axis_pair(x) for x in curr_order]
    tb = [_axis_pair(x) for x in target_order]
    src_pos = [cb.index(x) for x in tb]
    axes = list(range(arr.ndim))
    new_spatial = [spatial_axes[p] for p in src_pos]
    for k, ax in enumerate(spatial_axes):
        axes[ax] = new_spatial[k]
    return np.transpose(arr, axes), src_pos


def _flip_axes(arr, axes_to_flip):
    for ax in axes_to_flip:
        arr = np.flip(arr, axis=ax)
    return arr


def reorient(mag, flow, segmask, venc, resolution, spatial_order, venc_order,
             target_spatial_order, target_venc_order, return_velocity=False, normalize_mag=True):
    spatial_order = [s.upper() for s in spatial_order]
    venc_order = [v.upper() for v in venc_order]
    target_spatial_order = [s.upper() for s in target_spatial_order]
    target_venc_order = [v.upper() for v in target_venc_order]
    resolution = np.asarray(resolution, dtype=np.float32)
    venc = np.asarray(venc, dtype=np.float32)
    if venc.ndim == 0:
        venc = np.full(3, float(venc), dtype=np.float32)

    mag_r, src_pos_mag = _permute_spatial(mag, spatial_order, target_spatial_order, spatial_axes=(0, 1, 2))
    flow_r, src_pos_flow = _permute_spatial(flow, spatial_order, target_spatial_order, spatial_axes=(0, 1, 2))
    seg_r, src_pos_seg = _permute_spatial(segmask, spatial_order, target_spatial_order, spatial_axes=(0, 1, 2))

    cb = [_axis_pair(s) for s in spatial_order]
    tb = [_axis_pair(s) for s in target_spatial_order]
    res_perm = np.array([cb.index(x) for x in tb], dtype=int)
    resolution_r = resolution[res_perm]

    flip_mag = [i for i in range(3) if _need_flip(spatial_order[src_pos_mag[i]], target_spatial_order[i])]
    flip_flow = [i for i in range(3) if _need_flip(spatial_order[src_pos_flow[i]], target_spatial_order[i])]
    flip_seg = [i for i in range(3) if _need_flip(spatial_order[src_pos_seg[i]], target_spatial_order[i])]
    mag_r = _flip_axes(mag_r, flip_mag)
    flow_r = _flip_axes(flow_r, flip_flow)
    seg_r = _flip_axes(seg_r, flip_seg)

    vb = [_axis_pair(v) for v in venc_order]
    tb2 = [_axis_pair(v) for v in target_venc_order]
    comp_perm = np.array([vb.index(x) for x in tb2], dtype=int)

    flow_r = flow_r[..., comp_perm]
    venc_r = venc[comp_perm]

    sign3 = np.array([(-1.0 if _need_flip(venc_order[comp_perm[i]], target_venc_order[i]) else 1.0)
                      for i in range(3)], dtype=np.float32)
    sign_shape = (1,) * (flow_r.ndim - 1) + (int(sign3.shape[0]),)
    flow_r = flow_r * sign3.reshape(sign_shape)

    if return_velocity:
        venc_shape = (1,) * (flow_r.ndim - 1) + (int(venc_r.shape[0]),)
        flow_r = (flow_r / np.pi) * venc_r.reshape(venc_shape)

    if normalize_mag:
        mag_max = np.max(np.abs(mag_r))
        if mag_max > 0:
            mag_r = mag_r / mag_max

    return flow_r, mag_r, seg_r, venc_r, resolution_r


def _reorient_spatial_only(arr, spatial_order, target_spatial_order):
    arr_r, src_pos = _permute_spatial(arr, spatial_order, target_spatial_order, spatial_axes=(0, 1, 2))
    flip_axes = [i for i in range(3) if _need_flip(spatial_order[src_pos[i]], target_spatial_order[i])]
    return _flip_axes(arr_r, flip_axes)


def _compute_spatial_bbox(mask, pad=0):
    m = np.asarray(mask, dtype=bool)
    if m.ndim > 3:
        m = np.any(m, axis=tuple(range(3, m.ndim)))
    if not np.any(m):
        return tuple(slice(0, int(m.shape[i])) for i in range(3))
    idx = np.argwhere(m)
    lo = np.maximum(idx.min(axis=0) - int(pad), 0)
    hi = np.minimum(idx.max(axis=0) + int(pad) + 1, np.array(m.shape[:3], dtype=int))
    return tuple(slice(int(lo[i]), int(hi[i])) for i in range(3))


def _target_bbox_to_source_slices(shape_raw, spatial_order, target_spatial_order, bbox_target):
    spatial_order = [str(x).upper() for x in spatial_order]
    target_spatial_order = [str(x).upper() for x in target_spatial_order]
    cb = [_axis_pair(s) for s in spatial_order]
    tb = [_axis_pair(s) for s in target_spatial_order]
    src_pos = [cb.index(x) for x in tb]
    out = [slice(0, int(shape_raw[i])) for i in range(3)]
    for target_axis, raw_axis in enumerate(src_pos):
        s = int(bbox_target[target_axis].start)
        e = int(bbox_target[target_axis].stop)
        if _need_flip(spatial_order[raw_axis], target_spatial_order[target_axis]):
            out[raw_axis] = slice(int(shape_raw[raw_axis]) - e, int(shape_raw[raw_axis]) - s)
        else:
            out[raw_axis] = slice(s, e)
    return tuple(out)


def _reorient_component_abs(arr, spatial_order, target_spatial_order, venc_order, target_venc_order):
    spatial_order = [s.upper() for s in spatial_order]
    venc_order = [v.upper() for v in venc_order]
    target_spatial_order = [s.upper() for s in target_spatial_order]
    target_venc_order = [v.upper() for v in target_venc_order]

    arr_r, src_pos = _permute_spatial(arr, spatial_order, target_spatial_order, spatial_axes=(0, 1, 2))
    flip_axes = [i for i in range(3) if _need_flip(spatial_order[src_pos[i]], target_spatial_order[i])]
    arr_r = _flip_axes(arr_r, flip_axes)

    vb = [_axis_pair(v) for v in venc_order]
    tb = [_axis_pair(v) for v in target_venc_order]
    comp_perm = np.array([vb.index(x) for x in tb], dtype=int)

    return arr_r[..., comp_perm]


def _reorient_component_signed(arr, spatial_order, target_spatial_order, venc_order, target_venc_order):
    """Reorient a directional phase field into the normalized velocity axes."""
    spatial_order = [s.upper() for s in spatial_order]
    venc_order = [v.upper() for v in venc_order]
    target_venc_order = [v.upper() for v in target_venc_order]
    arr_r = _reorient_component_abs(
        arr,
        spatial_order=spatial_order,
        target_spatial_order=target_spatial_order,
        venc_order=venc_order,
        target_venc_order=target_venc_order,
    )
    vb = [_axis_pair(v) for v in venc_order]
    tb = [_axis_pair(v) for v in target_venc_order]
    comp_perm = np.array([vb.index(axis) for axis in tb], dtype=int)
    sign = np.array(
        [(-1.0 if _need_flip(venc_order[comp_perm[i]], target_venc_order[i]) else 1.0) for i in range(3)],
        dtype=np.float32,
    )
    return np.asarray(arr_r, dtype=np.float32) * sign.reshape((1,) * (arr_r.ndim - 1) + (3,))


def _reorient_real_valued_fields(
    *,
    mag,
    flow,
    segmask,
    sigma,
    tke_array,
    venc,
    resolution,
    spatial_order,
    venc_order,
    target_spatial_order,
    target_venc_order,
    return_velocity=False,
):
    segmask_for_reorient = segmask if segmask is not None else np.zeros(np.asarray(mag).shape, dtype=np.int16)
    flow_r, mag_r, seg_r, venc_r, resolution_r = reorient(
        mag,
        flow,
        segmask_for_reorient,
        venc=venc,
        resolution=resolution,
        spatial_order=spatial_order,
        venc_order=venc_order,
        target_spatial_order=target_spatial_order,
        target_venc_order=target_venc_order,
        return_velocity=return_velocity,
        normalize_mag=False,
    )
    sigma_r = None
    if sigma is not None:
        sigma_r = _reorient_component_abs(
            sigma,
            spatial_order=spatial_order,
            target_spatial_order=target_spatial_order,
            venc_order=venc_order,
            target_venc_order=target_venc_order,
        ).astype(np.float32)
    tke_r = None
    if tke_array is not None:
        tke_r = _reorient_spatial_only(
            tke_array,
            spatial_order=spatial_order,
            target_spatial_order=target_spatial_order,
        ).astype(np.float32)
    return flow_r, mag_r, (seg_r if segmask is not None else None), venc_r, resolution_r, sigma_r, tke_r
