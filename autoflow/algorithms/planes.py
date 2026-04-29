import numpy as np

from ..models import PlaneData
from .paths import inter_points, smooth_path_savgol


def generate_planes_from_paths(
    paths,
    cross_section_distance=20.0,
    start_distance=5.0,
    end_distance=0.0,
    smoothing_window=15,
    smoothing_polyorder=3,
    inter_time=9,
    use_center_plane=True,
):
    planes = []
    smooth_paths = []

    for path_i, path in enumerate(paths):
        path = inter_points(path, time=inter_time)
        if len(path) < 2:
            smooth_paths.append(path)
            continue

        path_smooth = smooth_path_savgol(path, window=smoothing_window, polyorder=smoothing_polyorder)
        smooth_paths.append(path_smooth)
        if len(path_smooth) < 2:
            continue

        seglens = np.linalg.norm(np.diff(path_smooth, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seglens)])
        total = cum[-1]

        if use_center_plane:
            target = total * 0.5
            j = int(np.searchsorted(cum, target, side="right") - 1)
            j = min(max(0, j), len(path_smooth) - 2)
            seg = path_smooth[j + 1] - path_smooth[j]
            seglen = np.linalg.norm(seg) + 1e-12
            alpha = (target - cum[j]) / seglen
            center = path_smooth[j] + alpha * seg
            i0 = max(0, j - 1)
            i1 = min(len(path_smooth) - 1, j + 2)
            tangent = path_smooth[i1] - path_smooth[i0]
            normal = tangent / (np.linalg.norm(tangent) + 1e-12)
            planes.append(
                PlaneData(
                    center=center,
                    normal=normal,
                    label=path_i + 1,
                    path_index=path_i,
                    distance=float(target),
                )
            )
            continue

        effective = total - float(end_distance)

        if effective <= float(start_distance):
            mid = len(path_smooth) // 2
            i0 = max(0, mid - 1)
            i1 = min(len(path_smooth) - 1, mid + 1)
            d = path_smooth[i1] - path_smooth[i0]
            n = d / (np.linalg.norm(d) + 1e-12)
            planes.append(
                PlaneData(
                    center=path_smooth[mid],
                    normal=n,
                    label=path_i + 1,
                    path_index=path_i,
                    distance=float(cum[mid]),
                )
            )
            continue

        planes_before = len(planes)
        for target in np.arange(float(start_distance), effective + 1e-8, float(cross_section_distance)):
            j = int(np.searchsorted(cum, target, side="right") - 1)
            j = min(max(0, j), len(path_smooth) - 2)
            seg = path_smooth[j + 1] - path_smooth[j]
            seglen = np.linalg.norm(seg) + 1e-12
            alpha = (target - cum[j]) / seglen
            center = path_smooth[j] + alpha * seg
            i0 = max(0, j - 1)
            i1 = min(len(path_smooth) - 1, j + 2)
            tangent = path_smooth[i1] - path_smooth[i0]
            normal = tangent / (np.linalg.norm(tangent) + 1e-12)
            planes.append(
                PlaneData(
                    center=center,
                    normal=normal,
                    label=path_i + 1,
                    path_index=path_i,
                    distance=float(target),
                )
            )

        if len(planes) == planes_before and float(cross_section_distance) > total > 0:
            target_mid = total * 0.5
            j = int(np.searchsorted(cum, target_mid, side="right") - 1)
            j = min(max(0, j), len(path_smooth) - 2)
            seg = path_smooth[j + 1] - path_smooth[j]
            seglen = np.linalg.norm(seg) + 1e-12
            alpha = (target_mid - cum[j]) / seglen
            center = path_smooth[j] + alpha * seg
            i0 = max(0, j - 1)
            i1 = min(len(path_smooth) - 1, j + 2)
            tangent = path_smooth[i1] - path_smooth[i0]
            normal = tangent / (np.linalg.norm(tangent) + 1e-12)
            planes.append(
                PlaneData(
                    center=center,
                    normal=normal,
                    label=path_i + 1,
                    path_index=path_i,
                    distance=float(target_mid),
                )
            )

    return planes, smooth_paths
