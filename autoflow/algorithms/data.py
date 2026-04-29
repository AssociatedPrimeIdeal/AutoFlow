import h5py
import numpy as np

from .metrics import compute_tke_array_from_sigma


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
             target_spatial_order, target_venc_order, return_velocity=False):
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
    flow_r = flow_r * sign3.reshape((1, 1, 1, 1, -1))

    if return_velocity:
        flow_r = (flow_r / np.pi) * venc_r.reshape((1, 1, 1, 1, -1))

    mag_max = np.max(np.abs(mag_r))
    if mag_max > 0:
        mag_r = mag_r / mag_max

    return flow_r, mag_r, seg_r, venc_r, resolution_r


def _ensure_flow_mag_time_and_segmask(flow, mag, segmask):
    flow = np.asarray(flow)
    mag = np.asarray(mag)
    segmask = np.asarray(segmask)
    if flow.ndim == 4 and flow.shape[-1] == 3:
        flow = flow[..., np.newaxis, :]
    if flow.ndim != 5 or flow.shape[-1] != 3:
        raise ValueError(f"flow must be XYZTV or XYZV with 3 components, got {flow.shape}")
    nt = int(flow.shape[3])
    if mag.ndim == 3:
        mag = np.repeat(mag[..., np.newaxis], nt, axis=3)
    elif mag.ndim == 4 and mag.shape[3] == 1 and nt > 1:
        mag = np.repeat(mag, nt, axis=3)
    elif mag.ndim != 4:
        raise ValueError(f"mag must be XYZT or XYZ, got {mag.shape}")
    if mag.shape[3] != nt:
        if mag.shape[3] == 1:
            mag = np.repeat(mag, nt, axis=3)
        else:
            raise ValueError(f"mag time dimension {mag.shape[3]} does not match flow {nt}")
    if segmask.ndim == 3:
        segmask = np.repeat(segmask[..., np.newaxis], nt, axis=3)
    elif segmask.ndim == 4 and segmask.shape[3] == 1 and nt > 1:
        segmask = np.repeat(segmask, nt, axis=3)
    elif segmask.ndim != 4:
        raise ValueError(f"segmask must be XYZT or XYZ, got {segmask.shape}")
    if segmask.shape[3] != nt:
        if segmask.shape[3] == 1:
            segmask = np.repeat(segmask, nt, axis=3)
        else:
            raise ValueError(f"segmask time dimension {segmask.shape[3]} does not match flow {nt}")
    return (
        np.ascontiguousarray(flow, dtype=np.float32),
        np.ascontiguousarray(mag, dtype=np.float32),
        np.ascontiguousarray(segmask, dtype=np.int16),
    )


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


def _sigma_from_complex(img_complex, venc):
    venc = np.asarray(venc, dtype=np.float32)
    if venc.ndim == 0:
        venc = np.full(3, float(venc), dtype=np.float32)
    ref = np.abs(img_complex[..., 0]).astype(np.float32)
    enc = np.abs(img_complex[..., 1:4]).astype(np.float32)
    kv = np.pi / venc.reshape((1, 1, 1, 1, 3))
    ratio = ref[..., None] / np.clip(enc, 1e-12, None)
    ratio = np.clip(ratio, 1.0, None)
    sigma = np.sqrt(2.0 * np.log(ratio)) / kv
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=0.0, neginf=0.0)
    return sigma.astype(np.float32)


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


def load_h5_data(path):
    target_spatial_order = ("LR", "AP", "FH")
    target_venc_order = ("LR", "AP", "FH")
    with h5py.File(path, "r") as g:
        if "img_complex" not in g or "segmask" not in g:
            raise ValueError(f"h5 must contain img_complex and segmask: {path}")
        VENC = g["VENC"][:] if "VENC" in g else np.array([150, 150, 150], dtype=float)
        resolution = g["Resolution"][:] if "Resolution" in g else np.array([1, 1, 1], dtype=float)
        origin = np.array([0.0, 0.0, 0.0], dtype=float)
        rr = float(g["RR"][()]) if "RR" in g else 1000.0
        spatial_order = g["SpatialOrder"][:].astype(str) if "SpatialOrder" in g else np.array(["FH", "AP", "LR"])
        venc_order = g["VENCOrder"][:].astype(str) if "VENCOrder" in g else np.array(["FH", "AP", "LR"])

        segmask_ds = g["segmask"]
        segmask_full = segmask_ds[:].astype(np.int16)
        src_slices = _compute_spatial_bbox(segmask_full, pad=2)
        segmask = segmask_ds[src_slices + (slice(None),) * (segmask_ds.ndim - 3)].astype(np.int16)
        del segmask_full

        img_ds = g["img_complex"]
        img_complex = np.asarray(img_ds[src_slices + (slice(None),) * (img_ds.ndim - 3)])

        mag = np.abs(img_complex[..., 0]).astype(np.float32)
        flow_raw = np.angle(img_complex[..., 1:4] * np.conj(img_complex[..., 0][..., None])).astype(np.float32)
        sigma_raw = _sigma_from_complex(img_complex, VENC)

        flow, mag_out, seg_r, venc_new, res_new = reorient(
            mag, flow_raw, segmask, venc=VENC, resolution=resolution,
            spatial_order=spatial_order, venc_order=venc_order,
            target_spatial_order=target_spatial_order,
            target_venc_order=target_venc_order,
            return_velocity=True,
        )
        sigma = _reorient_component_abs(
            sigma_raw,
            spatial_order=spatial_order,
            target_spatial_order=target_spatial_order,
            venc_order=venc_order,
            target_venc_order=target_venc_order,
        ).astype(np.float32)

        flow, mag_out, seg_r = _ensure_flow_mag_time_and_segmask(flow, mag_out, seg_r)
        tke_array = compute_tke_array_from_sigma(sigma, rho=1060.0)

        return {
            "flow": flow,
            "mag": mag_out,
            "segmask": seg_r,
            "resolution": np.asarray(res_new, dtype=float),
            "origin": origin,
            "venc": np.asarray(venc_new, dtype=float),
            "rr": float(rr),
            "sigma": sigma,
            "tke_array": tke_array,
        }
