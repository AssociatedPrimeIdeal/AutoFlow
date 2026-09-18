from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d


def _clean_points(points, minimum_distance=1e-6):
    data = np.asarray(points, dtype=float).reshape(-1, 2)
    data = data[np.all(np.isfinite(data), axis=1)]
    if len(data) < 2:
        return data
    keep = np.r_[True, np.linalg.norm(np.diff(data, axis=0), axis=1) > float(minimum_distance)]
    return data[keep]


def _resample(points, spacing, *, closed=False, max_points=384):
    data = _clean_points(points)
    if len(data) < 2:
        return data
    if closed:
        data = np.vstack((data, data[0]))
    lengths = np.linalg.norm(np.diff(data, axis=0), axis=1)
    cumulative = np.r_[0.0, np.cumsum(lengths)]
    total = float(cumulative[-1])
    if total <= 1e-8:
        return data[:1]
    count = int(np.clip(np.ceil(total / max(float(spacing), 1e-3)) + (0 if closed else 1), 8, max_points))
    samples = np.linspace(0.0, total, count, endpoint=not closed)
    out = np.column_stack(
        (
            np.interp(samples, cumulative, data[:, 0]),
            np.interp(samples, cumulative, data[:, 1]),
        )
    )
    return _clean_points(out)


def _smooth(points, sigma, *, closed=False):
    data = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(data) < 5 or float(sigma) <= 0:
        return data.copy()
    mode = "wrap" if closed else "nearest"
    return np.column_stack(
        (
            gaussian_filter1d(data[:, 0], float(sigma), mode=mode),
            gaussian_filter1d(data[:, 1], float(sigma), mode=mode),
        )
    )


def polygon_area(points):
    data = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(data) < 3:
        return 0.0
    return 0.5 * float(
        np.sum(data[:, 0] * np.roll(data[:, 1], -1) - data[:, 1] * np.roll(data[:, 0], -1))
    )


def _cross(one, two):
    return float(one[0] * two[1] - one[1] * two[0])


def _segment_intersection(a, b, c, d, tolerance=1e-9):
    r = b - a
    s = d - c
    denominator = _cross(r, s)
    if abs(denominator) <= tolerance:
        return None
    offset = c - a
    t = _cross(offset, s) / denominator
    u = _cross(offset, r) / denominator
    if -tolerance <= t <= 1.0 + tolerance and -tolerance <= u <= 1.0 + tolerance:
        return float(np.clip(t, 0.0, 1.0)), float(np.clip(u, 0.0, 1.0)), a + np.clip(t, 0.0, 1.0) * r
    return None


def _self_intersects(points):
    data = np.asarray(points, dtype=float).reshape(-1, 2)
    count = len(data)
    if count < 4:
        return False
    # Test all non-adjacent edge pairs in one NumPy operation. The previous
    # Python double loop made validation quadratic in interpreter time for
    # otherwise small contours (384 samples meant ~73k segment pairs).
    first, second = np.triu_indices(count, k=2)
    keep = ~((first == 0) & (second == count - 1))
    first = first[keep]
    second = second[keep]
    if len(first) == 0:
        return False
    a = data[first]
    b = data[(first + 1) % count]
    c = data[second]
    d = data[(second + 1) % count]
    r = b - a
    s = d - c
    denominator = r[:, 0] * s[:, 1] - r[:, 1] * s[:, 0]
    valid = np.abs(denominator) > 1e-9
    if not np.any(valid):
        return False
    offset = c - a
    denominator = denominator[valid]
    r = r[valid]
    s = s[valid]
    offset = offset[valid]
    t = (offset[:, 0] * s[:, 1] - offset[:, 1] * s[:, 0]) / denominator
    u = (offset[:, 0] * r[:, 1] - offset[:, 1] * r[:, 0]) / denominator
    return bool(np.any(
        (t >= -1e-9) & (t <= 1.0 + 1e-9)
        & (u >= -1e-9) & (u <= 1.0 + 1e-9)
    ))


def _nearest_boundary_point(contour, point):
    best_distance = np.inf
    best_point = None
    best_index = 0
    count = len(contour)
    for index in range(count):
        start = contour[index]
        end = contour[(index + 1) % count]
        segment = end - start
        denom = float(np.dot(segment, segment))
        fraction = 0.0 if denom <= 1e-12 else float(np.clip(np.dot(point - start, segment) / denom, 0.0, 1.0))
        projected = start + fraction * segment
        distance = float(np.linalg.norm(point - projected))
        if distance < best_distance:
            best_distance = distance
            best_point = projected
            best_index = index if fraction < 0.5 else (index + 1) % count
    return best_distance, np.asarray(best_point, dtype=float), int(best_index)


def _nearest_boundary_point_excluding(contour, point, exclude_index, exclude_radius=2):
    """Nearest boundary projection while avoiding a local attachment arc."""
    data = np.asarray(contour, dtype=float).reshape(-1, 2)
    count = len(data)
    if count == 0:
        return np.inf, np.zeros(2, dtype=float), 0
    starts = data
    ends = np.roll(data, -1, axis=0)
    segment = ends - starts
    denom = np.einsum("ij,ij->i", segment, segment)
    fraction = np.divide(
        np.einsum("ij,ij->i", np.asarray(point, dtype=float).reshape(1, 2) - starts, segment),
        denom,
        out=np.zeros(count, dtype=float),
        where=denom > 1e-12,
    )
    fraction = np.clip(fraction, 0.0, 1.0)
    projected = starts + fraction[:, None] * segment
    distances = np.linalg.norm(projected - np.asarray(point, dtype=float).reshape(1, 2), axis=1)
    radius = max(0, int(exclude_radius))
    if 0 <= int(exclude_index) < count and radius:
        indices = np.arange(count)
        cyclic_distance = np.minimum(
            np.abs(indices - int(exclude_index)),
            count - np.abs(indices - int(exclude_index)),
        )
        distances[cyclic_distance <= radius] = np.inf
    index = int(np.argmin(distances))
    if not np.isfinite(distances[index]):
        return _nearest_boundary_point(data, point)
    boundary_index = index if fraction[index] < 0.5 else (index + 1) % count
    return float(distances[index]), projected[index], int(boundary_index)


def _stroke_intersections(stroke, contour, merge_distance):
    drawn = np.asarray(stroke, dtype=float).reshape(-1, 2)
    boundary = np.asarray(contour, dtype=float).reshape(-1, 2)
    if len(drawn) < 2 or len(boundary) < 2:
        return []
    starts = drawn[:-1]
    r = drawn[1:] - starts
    boundary_starts = boundary
    s = np.roll(boundary, -1, axis=0) - boundary
    denominator = r[:, None, 0] * s[None, :, 1] - r[:, None, 1] * s[None, :, 0]
    valid_denominator = np.abs(denominator) > 1e-9
    if not np.any(valid_denominator):
        return []
    offset = boundary_starts[None, :, :] - starts[:, None, :]
    t = np.divide(
        offset[:, :, 0] * s[None, :, 1] - offset[:, :, 1] * s[None, :, 0],
        denominator,
        out=np.zeros_like(denominator),
        where=valid_denominator,
    )
    u = np.divide(
        offset[:, :, 0] * r[:, None, 1] - offset[:, :, 1] * r[:, None, 0],
        denominator,
        out=np.zeros_like(denominator),
        where=valid_denominator,
    )
    valid = valid_denominator & (t >= -1e-9) & (t <= 1.0 + 1e-9) & (u >= -1e-9) & (u <= 1.0 + 1e-9)
    stroke_indices, contour_indices = np.nonzero(valid)
    if len(stroke_indices) == 0:
        return []
    fractions = np.clip(t[stroke_indices, contour_indices], 0.0, 1.0)
    points = starts[stroke_indices] + fractions[:, None] * r[stroke_indices]
    parameters = stroke_indices.astype(float) + fractions
    order = np.argsort(parameters)
    hits = []
    for position in order:
        parameter = float(parameters[position])
        point = np.asarray(points[position], dtype=float)
        contour_index = int(contour_indices[position])
        boundary_fraction = float(u[stroke_indices[position], contour_index])
        boundary_index = contour_index if boundary_fraction < 0.5 else (contour_index + 1) % len(boundary)
        if any(abs(parameter - old[0]) < 0.1 or np.linalg.norm(point - old[1]) < merge_distance for old in hits):
            continue
        hits.append((parameter, point, int(boundary_index)))
    return hits


def _trim_stroke(stroke, start_parameter, start_point, end_parameter, end_point):
    first_segment = int(np.floor(start_parameter))
    last_segment = int(np.floor(end_parameter))
    interior_start = first_segment + 1
    interior_end = min(last_segment + 1, len(stroke))
    interior = stroke[interior_start:interior_end]
    return _clean_points(np.vstack((start_point, interior, end_point)))


def _arc(contour, start, end, direction):
    indices = [int(start)]
    current = int(start)
    for _ in range(len(contour) + 1):
        if current == int(end):
            break
        current = (current + int(direction)) % len(contour)
        indices.append(current)
    return contour[np.asarray(indices, dtype=int)]


def _smooth_junctions(points, junctions, radius=3):
    data = np.asarray(points, dtype=float).copy()
    count = len(data)
    if count < 8:
        return data
    for _ in range(2):
        old = data.copy()
        for junction in junctions:
            for offset in range(-int(radius), int(radius) + 1):
                index = (int(junction) + offset) % count
                weight = 1.0 - abs(offset) / float(radius + 1)
                averaged = 0.25 * old[(index - 1) % count] + 0.5 * old[index] + 0.25 * old[(index + 1) % count]
                data[index] = (1.0 - 0.65 * weight) * old[index] + 0.65 * weight * averaged
    return data


def new_contour_from_stroke(stroke, spacing, *, validate=True):
    data = _resample(stroke, spacing, closed=False)
    if len(data) < 8:
        return None, "Draw a longer closed contour"
    data = _smooth(data, 1.2, closed=False)
    closed = _resample(data, spacing, closed=True)
    closed = _smooth(closed, 1.0, closed=True)
    if len(closed) < 8 or abs(polygon_area(closed)) < max(float(spacing) ** 2 * 4.0, 1e-3):
        return None, "The drawn contour is too small"
    if validate and _self_intersects(closed):
        return None, "Contour was not changed: the new boundary self-intersects"
    return closed, "New contour"


def replace_contour_segment(contour, stroke, spacing, snap_distance, *, validate=True):
    boundary = _resample(contour, spacing, closed=True)
    drawn = _resample(stroke, spacing, closed=False)
    if len(boundary) < 8 or len(drawn) < 3:
        return None, "Draw a longer stroke across the boundary"

    merge_distance = max(float(spacing) * 0.75, 1e-3)
    hits = _stroke_intersections(drawn, boundary, merge_distance)
    if len(hits) > 2:
        return None, "Contour was not changed: the stroke crosses the boundary more than twice"

    start_snap = _nearest_boundary_point(boundary, drawn[0])
    end_snap = _nearest_boundary_point(boundary, drawn[-1])
    attachments = list(hits)
    if len(attachments) == 0:
        if start_snap[0] > snap_distance and end_snap[0] > snap_distance:
            return None, "Contour was not changed: the stroke must meet or approach the boundary"
        attachments = [
            (0.0, start_snap[1], start_snap[2]),
            (float(len(drawn) - 1), end_snap[1], end_snap[2]),
        ]
    elif len(attachments) == 1:
        hit = attachments[0]
        candidates = []
        exclude_radius = max(2, int(np.ceil(float(snap_distance) / max(float(spacing), 1e-3))))
        if hit[0] > 0.5:
            # If the start is not close enough to count as a hit, project it
            # to the nearest boundary point and use the straight segment from
            # the drawn endpoint to that projection as the closing join.
            start_attachment = start_snap if start_snap[0] <= snap_distance else _nearest_boundary_point_excluding(
                boundary, drawn[0], hit[2], exclude_radius
            )
            candidates.append((0.0, start_attachment[1], start_attachment[2]))
        if hit[0] < len(drawn) - 1.5:
            end_attachment = end_snap if end_snap[0] <= snap_distance else _nearest_boundary_point_excluding(
                boundary, drawn[-1], hit[2], exclude_radius
            )
            candidates.append((float(len(drawn) - 1), end_attachment[1], end_attachment[2]))
        if len(candidates) != 1:
            return None, "Contour was not changed: the stroke needs two unambiguous joins"
        attachments.append(candidates[0])
        attachments.sort(key=lambda item: item[0])

    first, second = attachments[0], attachments[1]
    if second[0] - first[0] < 1.0:
        return None, "Contour was not changed: the boundary joins are too close"
    replacement = _trim_stroke(drawn, first[0], first[1], second[0], second[1])
    replacement = _smooth(replacement, 1.0, closed=False)
    replacement[0] = first[1]
    replacement[-1] = second[1]

    first_index = int(first[2])
    second_index = int(second[2])
    forward = _arc(boundary, first_index, second_index, 1)
    backward = _arc(boundary, first_index, second_index, -1)
    forward_length = float(np.sum(np.linalg.norm(np.diff(forward, axis=0), axis=1)))
    backward_length = float(np.sum(np.linalg.norm(np.diff(backward, axis=0), axis=1)))
    if min(forward_length, backward_length) < 2.0 * float(spacing):
        return None, "Contour was not changed: the replaced boundary segment is too short"

    # The shorter old arc is the local section being replaced. The other arc
    # is retained verbatim apart from a few samples blended at each join.
    keep_direction = 1 if forward_length <= backward_length else -1
    kept = _arc(boundary, second_index, first_index, keep_direction)
    candidate = _clean_points(np.vstack((replacement, kept[1:-1])))
    candidate = _smooth_junctions(candidate, (0, len(replacement) - 1))
    if len(candidate) < 8 or abs(polygon_area(candidate)) < max(float(spacing) ** 2 * 4.0, 1e-3):
        return None, "Contour was not changed: the result is too small"
    if validate and _self_intersects(candidate):
        return None, "Contour was not changed: the replacement would self-intersect"
    return candidate, "Local boundary replaced"
