import numpy as np
from scipy.signal import savgol_filter


def _vector_orientation_text(vec):
    vec = np.asarray(vec, dtype=float).reshape(3)
    n = np.linalg.norm(vec)
    if n <= 1e-12:
        return ""
    v = vec / n
    axis_names = ["LR", "AP", "HF"]
    idx = int(np.argmax(np.abs(v)))
    sign = "+" if v[idx] >= 0 else "-"
    return f"{axis_names[idx]}{sign}"


def _path_cumulative_distance(path_points):
    pts = np.asarray(path_points, dtype=float).reshape(-1, 3)
    if len(pts) == 0:
        return np.zeros(0, dtype=float)
    if len(pts) == 1:
        return np.zeros(1, dtype=float)
    seglens = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seglens)])


def _path_point_at_distance(path_points, distance):
    pts = np.asarray(path_points, dtype=float).reshape(-1, 3)
    cum = _path_cumulative_distance(pts)
    if len(pts) == 0:
        return np.zeros(3, dtype=float), 0, 0.0, cum
    if len(pts) == 1 or len(cum) <= 1 or float(cum[-1]) <= 1e-12:
        return pts[0].copy(), 0, 0.0, cum
    d = float(np.clip(float(distance), 0.0, float(cum[-1])))
    j = int(np.searchsorted(cum, d, side="right") - 1)
    j = min(max(0, j), len(pts) - 2)
    seg = pts[j + 1] - pts[j]
    seglen = float(np.linalg.norm(seg))
    if seglen <= 1e-12:
        alpha = 0.0
        point = pts[j].copy()
    else:
        alpha = float(np.clip((d - float(cum[j])) / seglen, 0.0, 1.0))
        point = pts[j] + alpha * seg
    return point.astype(float), int(j), float(alpha), cum


def _path_tangent_from_segment(path_points, seg_idx):
    pts = np.asarray(path_points, dtype=float).reshape(-1, 3)
    N = len(pts)
    if N < 2:
        return np.zeros(3, dtype=float), "degenerate", 0, 0
    seg_idx = min(max(0, int(seg_idx)), N - 2)
    i0 = max(0, seg_idx - 1)
    i1 = min(N - 1, seg_idx + 2)
    if i0 >= i1:
        i0 = max(0, seg_idx)
        i1 = min(N - 1, seg_idx + 1)
        if i0 >= i1:
            return np.zeros(3, dtype=float), "degenerate", i0, i1
    t = pts[i1] - pts[i0]
    n = float(np.linalg.norm(t))
    if n < 1e-12:
        return np.zeros(3, dtype=float), "degenerate", i0, i1
    source = "local" if (seg_idx > 0 and seg_idx < N - 2) else "endpoint"
    return (t / n).astype(float), source, int(i0), int(i1)


def _project_point_to_path(path_points, query_point):
    pts = np.asarray(path_points, dtype=float).reshape(-1, 3)
    q = np.asarray(query_point, dtype=float).reshape(3)
    cum = _path_cumulative_distance(pts)
    if len(pts) == 0:
        return np.zeros(3, dtype=float), 0, 0.0, 0.0, float("inf"), cum
    if len(pts) == 1:
        err = float(np.linalg.norm(pts[0] - q))
        return pts[0].copy(), 0, 0.0, 0.0, err, cum

    seg = pts[1:] - pts[:-1]
    seglen = np.linalg.norm(seg, axis=1)
    seglen2 = seglen * seglen
    valid = seglen2 > 1e-12
    if np.any(valid):
        base = pts[:-1][valid]
        direction = seg[valid]
        alpha = np.sum((q.reshape(1, 3) - base) * direction, axis=1) / seglen2[valid]
        alpha = np.clip(alpha, 0.0, 1.0)
        proj = base + alpha[:, None] * direction
        d2 = np.sum((proj - q.reshape(1, 3)) ** 2, axis=1)
        best_local = int(np.argmin(d2))
        seg_ids = np.nonzero(valid)[0]
        seg_idx = int(seg_ids[best_local])
        a = float(alpha[best_local])
        point = proj[best_local]
        distance = float(cum[seg_idx] + a * seglen[seg_idx])
        err = float(np.linalg.norm(point - q))
        return point.astype(float), int(seg_idx), float(a), distance, err, cum

    idx = int(np.argmin(np.linalg.norm(pts - q.reshape(1, 3), axis=1)))
    point = pts[idx].copy()
    if idx >= len(pts) - 1:
        seg_idx = max(0, len(pts) - 2)
        a = 1.0
    else:
        seg_idx = idx
        a = 0.0
    distance = float(cum[idx]) if len(cum) > idx else 0.0
    err = float(np.linalg.norm(point - q))
    return point.astype(float), int(seg_idx), float(a), distance, err, cum


def _determine_plane_forward(plane, path_points, path_info, eps=0.1):
    center = np.asarray(getattr(plane, "center", [0.0, 0.0, 0.0]), dtype=float).reshape(3)
    normal = np.asarray(getattr(plane, "normal", [1.0, 0.0, 0.0]), dtype=float).reshape(3)
    n_mag = float(np.linalg.norm(normal))
    if n_mag > 1e-12:
        normal = normal / n_mag
    else:
        normal = np.zeros(3, dtype=float)

    pp = np.asarray(path_points, dtype=float).reshape(-1, 3) if path_points is not None else np.empty((0, 3), dtype=float)
    local_t = np.zeros(3, dtype=float)
    local_src = "none"
    cos_local = 0.0

    if len(pp) >= 2:
        seglens = np.linalg.norm(np.diff(pp, axis=0), axis=1)
        pos = seglens[seglens > 1e-12]
        seg_step = float(np.median(pos)) if len(pos) else 0.0
        tol = max(1.0, 2.0 * seg_step)

        try:
            qd = float(getattr(plane, "distance", None))
        except (TypeError, ValueError):
            qd = None

        tan_d = np.zeros(3, dtype=float)
        cos_d = 0.0
        err_d = float("inf")
        src_d = "none"
        if qd is not None and np.isfinite(qd):
            pt_d, seg_d, _alpha_d, _cum_d = _path_point_at_distance(pp, qd)
            err_d = float(np.linalg.norm(pt_d - center))
            tan_d, src_d, _i0_d, _i1_d = _path_tangent_from_segment(pp, seg_d)
            cos_d = float(np.dot(normal, tan_d)) if np.linalg.norm(tan_d) > 0 else 0.0
        else:
            qd = None

        _pt_p, seg_p, _alpha_p, _dist_p, err_p, _cum_p = _project_point_to_path(pp, center)
        tan_p, src_p, _i0_p, _i1_p = _path_tangent_from_segment(pp, seg_p)
        cos_p = float(np.dot(normal, tan_p)) if np.linalg.norm(tan_p) > 0 else 0.0

        if qd is not None and np.isfinite(err_d) and err_d <= max(tol, err_p + tol):
            local_t = tan_d
            local_src = str(src_d) if src_d != "degenerate" else "none"
            cos_local = float(cos_d)
        else:
            local_t = tan_p
            local_src = str(src_p) if src_p != "degenerate" else "none"
            cos_local = float(cos_p)

    fallback = np.zeros(3, dtype=float)
    if path_info is not None and 0 <= int(getattr(plane, "path_index", -1)) < len(path_info):
        info = path_info[int(getattr(plane, "path_index", -1))]
        sp = np.asarray(info.get("start_point", [0.0, 0.0, 0.0]), dtype=float).reshape(3)
        ep = np.asarray(info.get("end_point", [0.0, 0.0, 0.0]), dtype=float).reshape(3)
        fallback = ep - sp
    if np.linalg.norm(fallback) < 1e-12 and len(pp) >= 2:
        fallback = pp[-1] - pp[0]
    fb_mag = float(np.linalg.norm(fallback))
    if fb_mag > 1e-12:
        fb_unit = fallback / fb_mag
        cos_fb = float(np.dot(normal, fb_unit))
    else:
        fb_unit = np.zeros(3, dtype=float)
        cos_fb = 0.0

    if local_src != "none" and abs(cos_local) >= float(eps):
        return 1 if cos_local >= 0 else -1, "flow_tangent", local_t, float(cos_local)
    if fb_mag > 1e-12:
        if local_src != "none":
            return 1 if cos_fb >= 0 else -1, "geometry_fallback", local_t, float(cos_local)
        return 1 if cos_fb >= 0 else -1, "geometry_fallback", fb_unit, float(cos_fb)
    return 1, "none", local_t, float(cos_local)


def smooth_path_savgol(points, window=15, polyorder=3):
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    n = len(pts)
    if n <= polyorder:
        return pts.copy()
    w = int(window)
    if w % 2 == 0:
        w += 1
    w = min(w, n if n % 2 == 1 else max(1, n - 1))
    if w <= polyorder:
        return pts.copy()
    out = np.zeros_like(pts, dtype=float)
    for dim in range(3):
        out[:, dim] = savgol_filter(pts[:, dim], w, polyorder)
    out[0] = pts[0]
    out[-1] = pts[-1]
    return out


def inter_points(points, time=100):
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(pts) <= 2:
        return pts
    x = np.arange(len(pts))
    fine_x = np.linspace(0, len(pts) - 1, num=max(len(pts) * time, len(pts)))
    return np.stack([np.interp(fine_x, x, pts[:, k]) for k in range(3)], axis=1)
