"""Automatic inflow/outflow roots of each label's directed path graph."""

from itertools import combinations

import networkx as nx
import numpy as np

from ..plane_rules import normalize_label_plane_rules
from .paths import inter_points, smooth_path_savgol
from .planes import _build_plane, _junction_anchor_endpoint, _label_runs, _sample_path_labels, _target_distances


def build_label_plane_candidates(paths, segmentation_labels, label_map, *,
                                 spacing=(1, 1, 1), origin=(0, 0, 0), path_info=None,
                                 smoothing_window=15, smoothing_polyorder=2, inter_time=10):
    """Preserve graph identity and every contiguous label-contained segment."""
    if np.asarray(segmentation_labels).ndim != 3:
        raise ValueError("Label plane placement requires a 3-D majority-voted segmentation")
    smooth_paths, candidates = [], {}
    infos = list(path_info or [])
    tolerance = 2 * float(np.max(np.asarray(spacing, dtype=float)))
    for path_index, raw in enumerate(paths):
        points = np.asarray(raw, dtype=float).reshape(-1, 3)
        if len(points) < 2 or not np.all(np.isfinite(points)):
            smooth_paths.append(np.empty((0, 3), dtype=float))
            continue
        dense = inter_points(points, time=inter_time)
        smooth = smooth_path_savgol(dense, window=smoothing_window, polyorder=smoothing_polyorder)
        original_labels = _sample_path_labels(dense, segmentation_labels, spacing, origin)
        smoothed_labels = _sample_path_labels(smooth, segmentation_labels, spacing, origin)
        restored = (original_labels > 0) & (original_labels != smoothed_labels)
        # Keep smoothing from moving a known source-label sample into another
        # label/background. The graph and source segmentation remain untouched.
        smooth[restored] = dense[restored]
        smooth_paths.append(smooth)
        cumulative = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(smooth, axis=0), axis=1))]
        _, runs = _label_runs(smooth, segmentation_labels, spacing, origin, neighborhood=0)
        endpoint_labels = _sample_path_labels(points[[0, -1]], segmentation_labels, spacing, origin)
        info = infos[path_index] if path_index < len(infos) else {}
        segment_counts = {}
        for run_index, (label, lo, hi, _length) in enumerate(runs):
            if hi <= lo or cumulative[hi] - cumulative[lo] <= 1e-8:
                continue
            segment_index = segment_counts.get(label, 0)
            segment_counts[label] = segment_index + 1
            touches_start = (run_index == 0 and int(endpoint_labels[0]) == label
                             and cumulative[lo] <= tolerance)
            touches_end = (run_index == len(runs) - 1 and int(endpoint_labels[1]) == label
                           and cumulative[-1] - cumulative[hi] <= tolerance)
            start_node = int(info.get("start_node", -1)) if touches_start else -1
            end_node = int(info.get("end_node", -1)) if touches_end else -1
            candidates.setdefault(int(label), []).append({
                "path_index": path_index, "segment_index": segment_index,
                "sample_start": int(lo), "sample_end": int(hi),
                "start_distance_mm": float(cumulative[lo]), "end_distance_mm": float(cumulative[hi]),
                "length_mm": float(cumulative[hi] - cumulative[lo]),
                "smoothing_label_restored_samples": int(np.count_nonzero(restored)),
                "start_point": smooth[lo].tolist(), "end_point": smooth[hi].tolist(),
                "group_name": str(info.get("group_name", "")),
                "start_node": start_node, "end_node": end_node,
                "touches_path_start": bool(touches_start), "touches_path_end": bool(touches_end),
                "junction_nodes": sorted({node for node in (start_node, end_node) if node >= 0}),
                "adjacent_labels": [int(runs[i][0]) for i in (run_index - 1, run_index + 1)
                                    if 0 <= i < len(runs)],
            })
    return smooth_paths, candidates


def _flow_direction(points, flow, mask, spacing, origin):
    """Signed tangent alignment integrated over path length and cardiac phases."""
    result = {"flow_sign": 1, "flow_confidence": None, "flow_alignment": None,
              "flow_direction_status": "flow_unavailable"}
    if flow is None:
        return result
    velocity = np.asarray(flow)
    if velocity.ndim == 4 and velocity.shape[-1] == 3:
        velocity = velocity[..., None, :]
    if velocity.ndim != 5 or velocity.shape[-1] != 3:
        raise ValueError("Main-path direction requires flow with shape X,Y,Z,T,3")
    voxels = np.rint((points - np.asarray(origin)) / np.asarray(spacing)).astype(int)
    inside = np.all((voxels >= 0) & (voxels < np.asarray(velocity.shape[:3])), axis=1)
    voxels = np.clip(voxels, 0, np.asarray(velocity.shape[:3]) - 1)
    vectors = np.asarray(velocity[tuple(voxels.T)], dtype=float)
    valid = inside[:, None] & np.all(np.isfinite(vectors), axis=-1)
    if mask is not None:
        sampled = np.asarray(mask)[tuple(voxels.T)]
        if sampled.ndim == 1:
            sampled = sampled[:, None]
        valid &= np.broadcast_to(sampled > 0, valid.shape)
    dl = np.diff(points, axis=0)
    middle = .5 * (vectors[:-1] + vectors[1:])
    pairs = valid[:-1] & valid[1:]
    dot = np.sum(middle * dl[:, None, :], axis=-1)
    denominator = np.sum(np.where(pairs, np.linalg.norm(middle, axis=-1)
                                   * np.linalg.norm(dl, axis=1)[:, None], 0.0))
    numerator = np.sum(np.where(pairs, dot, 0.0))
    alignment = float(numerator / denominator) if denominator > 1e-12 else 0.0
    reliable = denominator > 1e-12 and abs(alignment) >= .05
    result.update(flow_sign=-1 if alignment < 0 else 1,
                  flow_confidence=abs(alignment), flow_alignment=alignment,
                  flow_direction_status="reliable" if reliable else "flow_direction_unreliable")
    return result


def _endpoint_key(candidate, side, raw):
    if not candidate["touches_path_" + side]:
        return None
    group = candidate["group_name"]
    node = candidate[side + "_node"]
    if node >= 0:
        return group, "node", node
    point = np.asarray(raw[0 if side == "start" else -1], dtype=float)
    return group, "point", *np.round(point, decimals=6).tolist()


def _select_main_paths(options, paths, smooth_paths, side, flow, mask, spacing, origin):
    """All directed sources/sinks; deterministic per-component fallback for bad flow."""
    keys, by_path, gap_links = {}, {}, []
    for index, candidate in enumerate(options):
        points = smooth_paths[candidate["path_index"]][candidate["sample_start"]:candidate["sample_end"] + 1]
        candidate.update(_flow_direction(points, flow, mask, spacing, origin))
        keys[index] = [_endpoint_key(candidate, endpoint, paths[candidate["path_index"]])
                       for endpoint in ("start", "end")]
        by_path.setdefault(candidate["path_index"], []).append(index)
    # Smoothing can briefly leave a label. Connect only small background gaps
    # on the SAME path; never bridge an intervening anatomical label.
    max_gap = 2 * float(np.max(np.asarray(spacing)))
    for path_index, indices in by_path.items():
        indices.sort(key=lambda i: options[i]["start_distance_mm"])
        for first, second in zip(indices, indices[1:]):
            a, b = options[first], options[second]
            gap = b["start_distance_mm"] - a["end_distance_mm"]
            if (a["adjacent_labels"] and a["adjacent_labels"][-1] == a["segmentation_label"]
                    and 0 <= gap <= max_gap):
                key = (a["group_name"], "gap", path_index, first, second)
                keys[first][1] = keys[second][0] = key
                gap_links.append({"path_index": path_index, "from_segment": a["segment_index"],
                                  "to_segment": b["segment_index"], "gap_mm": float(gap)})
    incident = {}
    for index, (start, end) in keys.items():
        if options[index]["flow_sign"] < 0:
            start, end = end, start
        for role, key in (("outgoing", start), ("incoming", end)):
            if key is not None:
                incident.setdefault(key, {"incoming": [], "outgoing": []})[role].append(index)
    directed, undirected = nx.DiGraph(), nx.Graph()
    directed.add_nodes_from(range(len(options)))
    undirected.add_nodes_from(range(len(options)))
    for arms in incident.values():
        undirected.add_edges_from(combinations(sorted(set(arms["incoming"] + arms["outgoing"])), 2))
        directed.add_edges_from((parent, child) for parent in arms["incoming"]
                                for child in arms["outgoing"] if parent != child)
    selected, warnings = [], []
    for component_index, nodes in enumerate(nx.connected_components(undirected)):
        component = sorted(nodes)
        uncertain = any(options[i]["flow_direction_status"] != "reliable" for i in component)
        roots = [i for i in component if (directed.in_degree(i) if side == "inflow" else directed.out_degree(i)) == 0]
        fallback = len(component) > 1 and (uncertain or not roots)
        if fallback:
            roots = [min(component, key=lambda i: (-round(options[i]["length_mm"], 6), options[i]["path_index"],
                                                  options[i]["segment_index"]))]
            causes = sorted({options[i]["flow_direction_status"] for i in component
                             if options[i]["flow_direction_status"] != "reliable"})
            warnings.extend(causes or ["directed_cycle_no_root"])
        for index in component:
            candidate = options[index]
            candidate.update(component_index=component_index,
                             upstream_segments=[{"path_index": options[i]["path_index"],
                                                 "segment_index": options[i]["segment_index"]}
                                                for i in sorted(directed.predecessors(index))],
                             downstream_segments=[{"path_index": options[i]["path_index"],
                                                   "segment_index": options[i]["segment_index"]}
                                                  for i in sorted(directed.successors(index))],
                             selected_main_path=index in roots)
            candidate["selection_reason"] = ("geometry_fallback_longest" if fallback and index in roots else
                "single_segment" if len(component) == 1 else
                "inflow_root" if side == "inflow" and index in roots else
                "outflow_root" if index in roots else
                "geometry_fallback_not_longest" if fallback else
                "has_same_label_upstream" if side == "inflow" else "has_same_label_downstream")
        selected.extend(options[i] for i in roots)
    return selected, sorted(set(warnings)), gap_links, nx.number_connected_components(undirected)


def generate_label_planes(paths, segmentation_labels, label_map, rules, *,
                          flow_xyzt3=None, segmask_binary_4d=None, fork_points=None,
                          branch_labels_3d=None, filter_defaults=None, **kwargs):
    """Apply the configured plane count independently to every inflow/outflow root."""
    rules = normalize_label_plane_rules(rules)
    smooth_paths, candidates = build_label_plane_candidates(paths, segmentation_labels, label_map, **kwargs)
    # Root identity and direction come from the unsmoothed graph. Smoothing is
    # exclusively placement geometry and cannot manufacture additional roots.
    topology_paths, topology_candidates = build_label_plane_candidates(
        paths, segmentation_labels, label_map, **dict(kwargs, smoothing_window=1))
    from .landmark_planes import ConnectionGraph, generate_landmark_label_planes
    connection_graph = ConnectionGraph(topology_paths, topology_candidates, kwargs.get("path_info"), label_map)
    present = set(int(x) for x in np.unique(segmentation_labels) if int(x) > 0)
    planes, labels_qc = [], []
    path_qc = [{"segmentation_filter": True, "requested_plane_count": 0,
                "label_rules": [], "original_path_length_mm": float(np.sum(
                    np.linalg.norm(np.diff(path, axis=0), axis=1))) if len(path) > 1 else 0.0}
               for path in smooth_paths]
    resolved_labels = set()
    for name, rule in rules.items():
        value = dict(label_map or {}).get(name)
        if value is None:
            try:
                value = int(name)
            except ValueError:
                value = 0
        value = int(value)
        if value > 0:
            if value in resolved_labels:
                raise ValueError(f"Multiple plane rules resolve to segmentation label {value}")
            resolved_labels.add(value)
        placement_paths = smooth_paths
        label_candidates, label_topology_candidates = candidates, topology_candidates
        label_topology_paths = topology_paths
        if any(rule[key] is not None for key in ("smoothing_window", "smoothing_polyorder", "inter_time")):
            geometry_kwargs = dict(kwargs)
            density = rule["inter_time"] or kwargs.get("inter_time", 10)
            window = (rule["smoothing_window"] if rule["smoothing_window"] is not None else
                      kwargs.get("smoothing_window", 15) / kwargs.get("inter_time", 10))
            geometry_kwargs.update(inter_time=density, smoothing_window=max(1, int(round(window * density))),
                smoothing_polyorder=rule["smoothing_polyorder"] if rule["smoothing_polyorder"] is not None
                else kwargs.get("smoothing_polyorder", 2))
            placement_paths, label_candidates = build_label_plane_candidates(
                paths, segmentation_labels, label_map, **geometry_kwargs)
            label_topology_paths, label_topology_candidates = build_label_plane_candidates(
                paths, segmentation_labels, label_map, **dict(geometry_kwargs, smoothing_window=1))
        options = label_candidates.get(value, [])
        topology_options = label_topology_candidates.get(value, [])
        for candidate in options + topology_options:
            candidate["segmentation_label"] = value
        qc = {"label_name": name, "segmentation_label": value,
              "plane_count_per_main_path": int(rule["plane_count"]), "main_path": rule.get("main_path", "landmark"),
              "main_path_count": 0, "selected_main_paths": [],
              "requested_plane_count": 0, "actual_plane_count": 0,
              "rule": dict(rule), "candidates": options, "topology_candidates": topology_options,
              "status": "label_absent"}
        labels_qc.append(qc)
        if rule["plane_count"] == 0:
            qc.update(status="disabled", reason="disabled_by_plane_config")
            continue
        if value not in present:
            continue
        if not topology_options:
            qc.update(status="no_label_path", reason="no_label_path")
            continue
        if rule["plane_mode"] == "landmark":
            context = connection_graph
            if label_topology_paths is not topology_paths:
                context = ConnectionGraph(label_topology_paths, label_topology_candidates, kwargs.get("path_info"), label_map)
            semantic_name = next((str(key) for key, label in dict(label_map or {}).items() if int(label) == value), name)
            planes.extend(generate_landmark_label_planes(context, semantic_name, value, rule, placement_paths,
                segmentation_labels, kwargs.get("spacing", (1, 1, 1)), qc, path_qc,
                origin=kwargs.get("origin", (0, 0, 0)), mask4d=segmask_binary_4d,
                branch_labels=branch_labels_3d, filter_defaults=filter_defaults))
            continue
        selected, warnings, gaps, components = _select_main_paths(
            topology_options, paths, label_topology_paths, rule["main_path"], flow_xyzt3, segmask_binary_4d,
            kwargs.get("spacing", (1, 1, 1)), kwargs.get("origin", (0, 0, 0)))
        qc.update(status="selected", main_path_count=len(selected), component_count=components,
                  requested_plane_count=len(selected) * max(0, rule["plane_count"]),
                  selection_source="automatic_with_fallback" if warnings else "flow_topology",
                  selection_warnings=warnings, topology_gap_links=gaps)
        for candidate in options:
            candidate.update(selected_main_path=False, selection_reason="not_on_selected_main_path")
        for main_index, root in enumerate(selected):
            index = root["path_index"]
            smooth = placement_paths[index]
            cumulative = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(smooth, axis=0), axis=1))]
            eligible = [(candidate, max(root["sample_start"], candidate["sample_start"]),
                         min(root["sample_end"], candidate["sample_end"]))
                        for candidate in options if candidate["path_index"] == index]
            eligible = [(candidate, lo, hi) for candidate, lo, hi in eligible if hi > lo]
            # Normally support-preserving smoothing leaves one full run. If
            # source background remains, use the longest contained overlap.
            if not eligible:
                qc.setdefault("omitted_positions", []).append({"reason": "no_smoothed_label_segment",
                    "path_index": index, "main_path_index": main_index,
                    "center_local_mm": root["start_point"], "fraction": None})
                qc["selected_main_paths"].append(dict(root, main_path_index=main_index,
                    requested_plane_count=max(0, rule["plane_count"]), actual_plane_count=0))
                continue
            chosen, lo, hi = min(eligible, key=lambda item: (-round(cumulative[item[2]] - cumulative[item[1]], 6),
                                                            item[0]["segment_index"]))
            chosen.update(selected_main_path=True, selection_reason=root["selection_reason"])
            record = dict(chosen, main_path_index=main_index, requested_plane_count=rule["plane_count"],
                actual_plane_count=0, sample_start=lo, sample_end=hi,
                start_distance_mm=float(cumulative[lo]), end_distance_mm=float(cumulative[hi]),
                start_point=smooth[lo].tolist(), end_point=smooth[hi].tolist(),
                length_mm=float(cumulative[hi] - cumulative[lo]), topology_segment_index=root["segment_index"],
                topology_length_mm=root["length_mm"])
            for field in ("flow_sign", "flow_confidence", "flow_alignment", "flow_direction_status",
                          "component_index", "upstream_segments", "downstream_segments"):
                record[field] = root[field]
            qc["selected_main_paths"].append(record)
            path_qc[index]["label_rules"].append({"label_name": name, "segmentation_label": value,
                "segment_index": chosen["segment_index"], "main_path_index": main_index, "main_path": rule["main_path"]})
            oriented_points = smooth[lo:hi + 1]
            if record["flow_sign"] < 0:
                oriented_points = oriented_points[::-1]
            junction = _junction_anchor_endpoint(oriented_points, fork_points, rule["anchor"])
            distances = _target_distances(record["length_mm"],
                **{key: rule[key] for key in ("plane_mode", "plane_count", "cross_section_distance",
                    "start_distance", "end_distance", "anchor", "anchor_offset_mm", "direction",
                    "spacing_mode", "spacing_ratio")}, junction_endpoint=junction)
            record["requested_plane_count"] = len(distances) if rule["plane_count"] == -1 else rule["plane_count"]
            if rule["plane_count"] == -1:
                qc["requested_plane_count"] += len(distances)
            path_qc[index]["requested_plane_count"] += record["requested_plane_count"]
            for distance in distances:
                local_distance = record["length_mm"] - distance if record["flow_sign"] < 0 else distance
                plane = _build_plane(smooth, cumulative, record["start_distance_mm"] + local_distance, index)
                plane.segmentation_label = value
                plane._label_plane_rule = rule
                plane._main_path_index = main_index
                plane._main_path_fraction = float(distance / record["length_mm"])
                if record["flow_sign"] < 0:
                    plane.normal = -plane.normal
                voxel = np.rint((plane.center - np.asarray(kwargs.get("origin", (0, 0, 0))))
                                / np.asarray(kwargs.get("spacing", (1, 1, 1)))).astype(int)
                volume = np.asarray(segmentation_labels)
                if np.any(voxel < 0) or np.any(voxel >= volume.shape) or int(volume[tuple(voxel)]) != value:
                    qc.setdefault("omitted_positions", []).append({"fraction": float(distance / record["length_mm"]),
                        "reason": "center_outside_label", "path_index": int(index),
                        "segment_index": chosen["segment_index"], "main_path_index": main_index,
                        "center_local_mm": plane.center.tolist(), "normal": plane.normal.tolist(),
                        "distance_mm": float(plane.distance)})
                    continue
                planes.append(plane)
                record["actual_plane_count"] += 1
        qc["actual_plane_count"] = sum(p.segmentation_label == value for p in planes)
    names = {int(value): str(name) for name, value in dict(label_map or {}).items()}
    for value in sorted(present - resolved_labels):
        labels_qc.append({"label_name": names.get(value, f"label_{value}"),
                          "segmentation_label": int(value), "requested_plane_count": 0,
                          "actual_plane_count": 0, "status": "not_configured",
                          "reason": "not_in_plane_config", "candidates": candidates.get(value, [])})
    return planes, smooth_paths, path_qc, labels_qc
