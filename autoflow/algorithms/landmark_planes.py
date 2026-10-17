"""Place planes on explicit labeled arms of the physical path multigraph.

Landmark failures are omissions, never an invitation to pick a flow root.
Flow can orient a plane normal; it cannot replace a missing anatomical anchor.
"""

from itertools import product

import networkx as nx
import numpy as np

from .planes import _build_plane, _sample_path_labels


PATTERNS = {
    "pulmonary_split": ("MPA", "LPA", "RPA"),
    "portal_confluence": ("SMV", "SV", "PV"),
    "vertebral_confluence": ("LVA", "RVA", "BA"),
    "pca_split": ("BA", "LPCA", "RPCA"),
    "left_carotid_split": ("LICA", "LMCA", "ACA"),
    "right_carotid_split": ("RICA", "RMCA", "ACA"),
    "sinus_confluence": ("SSS", "StrS", "LTS", "RTS"),
}


class ConnectionGraph:
    def __init__(self, paths, candidates, infos, label_map):
        self.paths, self.infos, self.labels = paths, list(infos or []), dict(label_map or {})
        self.graph, self.edges, self.spans = nx.MultiGraph(), {}, {}
        self.cums = [np.r_[0., np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))] for p in paths]
        for label, rows in candidates.items():
            for row in rows:
                self.spans.setdefault((row["path_index"], int(label)), []).append(row)
        for index, points in enumerate(paths):
            if len(points) < 2:
                continue
            info = self.infos[index] if index < len(self.infos) else {}
            group = str(info.get("group_name", ""))
            endpoints = []
            for side, point in (("start", points[0]), ("end", points[-1])):
                node = int(info.get(side + "_node", -1))
                endpoints.append((group, "node", node) if node >= 0 else (group, "point", *np.round(point, 6)))
            u, v = endpoints
            self.edges[index] = (u, v)
            self.graph.add_edge(u, v, key=index, path_index=index, weight=float(self.cums[index][-1]))
        self.components = {}
        for component, nodes in enumerate(nx.connected_components(self.graph)):
            for node in nodes:
                self.components[node] = component
        self.cache = {}

    def observed_connections(self):
        """All actual cross-label arms; labels are never anatomy templates."""
        if "observed_connections" not in self.cache:
            from .label_connections import build_observed_connections
            candidates = {}
            for (index, value), spans in self.spans.items():
                candidates.setdefault(value, []).extend(spans)
            self.cache["observed_connections"] = build_observed_connections(self.paths, candidates, self.infos)
        return self.cache["observed_connections"]

    def choice_component(self, choice):
        return self.components[self.edges[choice["chain"][0]["path_index"]][0]]

    def interval_choices(self, name, spec):
        """Choose own-label intervals, independently in physical components."""
        value = int(self.labels.get(name, int(name) if str(name).isdigit() else 0))
        by_component = {}
        for (index, label), spans in self.spans.items():
            if label != value or index not in self.edges:
                continue
            component = self.components[self.edges[index][0]]
            hierarchy = self.infos[index].get("hierarchy", {}) if index < len(self.infos) else {}
            for span in spans:
                by_component.setdefault(component, []).append((span, hierarchy))
        selections, omissions = [], []
        for component, rows in sorted(by_component.items()):
            if spec.get("_components") is not None and component not in spec["_components"]:
                continue
            level = None if spec["anchor"] == "label_longest" else spec.get("level", 1) - 1
            neighbours = set()
            if spec.get("when_connected_to"):
                for event in self.observed_connections()[0]:
                    for span, _ in event["selected"].get(value, []):
                        if self.components[self.edges[span["path_index"]][0]] == component:
                            neighbours.update(event["labels"])
                required = {int(self.labels.get(token, int(token) if token.isdigit() else 0))
                            for token in spec["when_connected_to"]}
                if not required <= neighbours:
                    level = spec["otherwise_level"] - 1
            eligible = [(span, hierarchy) for span, hierarchy in rows if level is None or
                        hierarchy.get("label_local_levels", {}).get(str(value)) == level]
            if not eligible:
                omissions.append("label_level_unavailable")
                continue
            eligible.sort(key=lambda row: (-round(row[0]["length_mm"], 6), row[0]["path_index"], row[0]["segment_index"]))
            policy = "longest" if level is None else spec.get("path_selection", "unique")
            ambiguous = (policy == "unique" and len(eligible) > 1) or (
                policy == "longest" and len(eligible) > 1 and
                abs(eligible[0][0]["length_mm"] - eligible[1][0]["length_mm"]) < 1e-6)
            if ambiguous:
                omissions.append("ambiguous_label_level" if level is not None else "ambiguous_label_interval")
                continue
            for span, hierarchy in (eligible if policy == "all" else eligible[:1]):
                selections.append({"chain": [{"path_index": span["path_index"], "span": span,
                                               "reverse": hierarchy.get("flow_sign") == -1}],
                                   "anchor_nodes": [], "boundary_path_indices": [], "path_selection": policy,
                                   "paths_at_level": sorted({row[0]["path_index"] for row in eligible}),
                                   "label_level": None if level is None else level + 1,
                                   "component_index": component, "connected_labels": sorted(neighbours),
                                   "conditional_level": bool(spec.get("when_connected_to"))})
        return selections, (omissions[0] if omissions else "" if selections else "no_label_path")

    def connection_choices(self, name, spec):
        value = int(self.labels.get(name, int(name) if str(name).isdigit() else 0))
        selections = []
        for event in self.observed_connections()[0]:
            for span, reverse in event["selected"].get(value, []):
                selections.append({"chain": [{"path_index": span["path_index"], "span": span, "reverse": reverse}],
                    "anchor_nodes": [{"group_name": event["group_name"], "node_id": event["node_id"],
                                      "point_mm": event["point_mm"]}],
                    "boundary_path_indices": sorted({row["path_index"] for row in event["boundary_ports"]}),
                    "connection_id": event["event_id"], "connection_kind": event["kind"],
                    "connected_labels": event["labels"], "connection_side": bool(reverse)})
        connected = {self.choice_component(choice) for choice in selections}
        fallback = spec.get("fallback")
        reason = ""
        components = {self.components[self.edges[index][0]] for index, label in self.spans
                      if label == value and index in self.edges}
        if fallback and components - connected:
            choices, reason = self.resolve(name, dict(fallback, _components=components - connected))
            for choice in choices:
                if self.choice_component(choice) not in connected:
                    selections.append(dict(choice, placement_spec=fallback, connection_fallback=True))
        return selections, reason if reason else ("" if selections else "label_connection_not_found")

    def boundary(self, nodes):
        nodes = set(nodes)
        result = []
        for index, (u, v) in self.edges.items():
            if (u in nodes) != (v in nodes):
                result.append((index, u if u in nodes else v))
        return result

    def intervals(self, index, name):
        return self.spans.get((index, int(self.labels.get(name, 0))), [])

    def arm(self, index, node, name):
        rows = self.intervals(index, name)
        if not rows:
            return None
        start = self.edges[index][0] == node
        length = self.cums[index][-1]
        distance = lambda row: row["start_distance_mm"] if start else length - row["end_distance_mm"]
        row = min(rows, key=distance)
        return {"path_index": index, "node": node, "span": row,
                "reverse": not start, "distance_to_label_mm": float(distance(row))}

    @staticmethod
    def node_record(node):
        return {"group_name": node[0], "node_id": int(node[2]) if node[1] == "node" else None,
                "point_mm": list(node[2:]) if node[1] == "point" else None}

    def match(self, anchor, target):
        key = (anchor, target if anchor == "aortic_origin" else "")
        if key in self.cache:
            return self.cache[key]
        names = PATTERNS.get(anchor)
        roles = {name: (name,) for name in names or ()}
        if anchor == "aortic_origin":
            roles = {target: (target,), "aorta_1": ("AAO", "ARCH", "DAO"),
                     "aorta_2": ("AAO", "ARCH", "DAO")}
        if not roles:
            return [], "unknown_landmark"
        junctions = {node for node in self.graph if self.graph.degree(node) >= 3}
        regions = [(node,) for node in junctions]
        # An anatomical confluence can occupy two neighboring physical forks.
        # Only an internal path containing related labels may extend a region.
        related = {name for options in roles.values() for name in options}
        for index, (u, v) in self.edges.items():
            if u != v and u in junctions and v in junctions and any(self.intervals(index, name) for name in related):
                regions.append((u, v))
        matches = []
        for region in regions:
            boundary = self.boundary(region)
            if len(boundary) < len(roles):
                continue
            options = []
            for role, labels in roles.items():
                arms = []
                for index, node in boundary:
                    rows = [self.arm(index, node, name) for name in labels]
                    rows = [row for row in rows if row is not None]
                    if rows:
                        arms.append(min(rows, key=lambda row: row["distance_to_label_mm"]))
                options.append(arms)
            if any(not choices for choices in options):
                continue
            assignments = {}
            for combination in product(*options):
                indices = [arm["path_index"] for arm in combination]
                if len(set(indices)) != len(indices):
                    continue
                # The two unnamed aorta arms are interchangeable.
                signature = tuple(sorted(indices[-2:])) if anchor == "aortic_origin" else ()
                signature = ((indices[0],) + signature) if anchor == "aortic_origin" else tuple(indices)
                cost = sum(arm["distance_to_label_mm"] for arm in combination)
                assignments.setdefault(signature, (cost, dict(zip(roles, combination))))
            if not assignments:
                continue
            best = sorted(assignments.values(), key=lambda row: row[0])
            ambiguous = len(best) > 1 and abs(best[0][0] - best[1][0]) < 1e-6
            internal = sum(self.cums[i][-1] for i, (u, v) in self.edges.items() if u in region and v in region)
            matches.append({"nodes": region, "arms": best[0][1], "boundary": boundary,
                            "score": (len(region), float(internal), best[0][0]), "ambiguous": ambiguous,
                            "component": self.components[region[0]]})
        # Pick at most one anatomical landmark per physical connected component.
        # Equally supported alternatives are reported, never broken by path ID.
        selected, ambiguous = [], False
        for component in sorted({row["component"] for row in matches}):
            rows = sorted((row for row in matches if row["component"] == component), key=lambda row: row["score"])
            best = rows[0]
            tied = len(rows) > 1 and np.allclose(best["score"], rows[1]["score"], rtol=0, atol=1e-6)
            if best["ambiguous"] or tied:
                ambiguous = True
            else:
                selected.append(best)
        result = selected, ("ambiguous_landmark" if ambiguous else "landmark_not_found" if not selected else "")
        self.cache[key] = result
        return result

    def portal_route(self):
        if "portal_route" in self.cache:
            return self.cache["portal_route"]
        matches, reason = self.match("portal_confluence", "PV")
        routes = []
        for match in matches:
            arm = match["arms"]["PV"]
            chain, seen = [], set(match["nodes"])
            while arm is not None:
                index, node = arm["path_index"], arm["node"]
                if len(self.intervals(index, "PV")) != 1:
                    reason = "portal_interval_not_contiguous"
                    break
                if index in {row["path_index"] for row in chain}:
                    reason = "portal_route_cycle"
                    break
                chain.append(arm)
                u, v = self.edges[index]
                far = v if node == u else u
                if far in seen:
                    reason = "portal_route_cycle"
                    break
                seen.add(far)
                next_arms = [self.arm(i, far, "PV") for i, n in self.boundary((far,)) if i != index]
                next_arms = [row for row in next_arms if row is not None]
                if len(next_arms) >= 2:
                    terminal = self.arm(index, far, "PV")
                    routes.append({"match": match, "chain": chain, "terminal": terminal,
                                   "split_node": far})
                    break
                if len(next_arms) != 1 or self.graph.degree(far) != 2:
                    reason = "pv_first_split_not_found"
                    break
                arm = next_arms[0]
        result = routes, reason if not routes else ""
        self.cache["portal_route"] = result
        return result

    def resolve(self, name, spec):
        anchor = spec["anchor"]
        if anchor == "label_connections":
            return self.connection_choices(name, spec)
        if anchor == "label_longest" or (anchor == "label_level" and
                (spec.get("path_selection") == "longest" or spec.get("when_connected_to"))):
            return self.interval_choices(name, spec)
        if anchor in {"label_level", "label_terminal"}:
            by_component = {}
            for index, (u, v) in self.edges.items():
                if not self.intervals(index, name):
                    continue
                hierarchy = self.infos[index].get("hierarchy", {}) if index < len(self.infos) else {}
                level = hierarchy.get("label_local_levels", {}).get(str(self.labels.get(name, 0)))
                if level is not None:
                    by_component.setdefault(self.components[u], []).append((index, level, hierarchy))
            choices, ambiguous = [], False
            for rows in by_component.values():
                level = max(row[1] for row in rows) if anchor == "label_terminal" else spec.get("level", 1) - 1
                eligible = [row for row in rows if row[1] == level]
                policy = spec.get("path_selection", "unique")
                if policy == "longest" and eligible:
                    def length(row):
                        return max((span["length_mm"] for span in self.intervals(row[0], name)), default=0.)
                    eligible.sort(key=length, reverse=True)
                    if len(eligible) > 1 and abs(length(eligible[0]) - length(eligible[1])) < 1e-6:
                        ambiguous = True
                        continue
                    eligible = eligible[:1]
                elif policy == "unique" and len(eligible) != 1:
                    ambiguous |= len(eligible) > 1
                    continue
                for index, _, hierarchy in eligible:
                    spans = self.intervals(index, name)
                    if len(spans) != 1:
                        ambiguous = True
                        continue
                    reverse = hierarchy.get("flow_sign") == -1
                    choices.append({"chain": [{"path_index": index, "span": spans[0], "reverse": reverse}],
                                    "anchor_nodes": [], "boundary_path_indices": [], "label_level": int(level) + 1,
                                    "path_selection": policy,
                                    "paths_at_level": sorted(row[0] for row in rows if row[1] == level)})
            return choices, "ambiguous_label_level" if ambiguous else "label_level_unavailable" if not choices else ""
        if anchor in {"pv_first_split", "portal_trunk"}:
            routes, reason = self.portal_route()
            return [{"chain": [row["terminal"]] if anchor == "pv_first_split" else row["chain"],
                     "anchor_nodes": [self.node_record(row["split_node"])] if anchor == "pv_first_split" else
                         [self.node_record(node) for node in row["match"]["nodes"]],
                     "boundary_path_indices": [i for i, n in self.boundary((row["split_node"],))],
                     "route_path_indices": [arm["path_index"] for arm in row["chain"]]} for row in routes], reason
        if anchor == "arch_route":
            return self.arch_route(name)
        matches, reason = self.match(anchor, name)
        choices = []
        for match in matches:
            arm = match["arms"].get(name)
            if arm is None:
                continue
            choices.append({"chain": [arm], "anchor_nodes": [self.node_record(node) for node in match["nodes"]],
                            "boundary_path_indices": sorted(i for i, n in match["boundary"]),
                            "matched_arms": {key: value["path_index"] for key, value in match["arms"].items()}})
        return choices, reason or ("target_label_not_on_landmark" if not choices else "")

    def arch_route(self, name):
        # Annotation transitions may lie inside a raw edge. Keep the full edge
        # multigraph and find a unique simple route between AAO and DAO spans.
        active = {i for i in self.edges if any(self.intervals(i, label) for label in ("AAO", "ARCH", "DAO"))}
        starts = [i for i in active if self.intervals(i, "AAO")]
        ends = [i for i in active if self.intervals(i, "DAO")]
        routes = {}
        adjacency = {}
        for index in active:
            for node in self.edges[index]:
                adjacency.setdefault(node, []).append(index)
        def walk(index, chain, visited_nodes):
            if len(routes) > 1:
                return
            if index in ends:
                if any(self.intervals(i, name) for i in chain):
                    routes[tuple(chain)] = chain
                return
            for node in self.edges[index]:
                if node in visited_nodes:
                    continue
                for other in adjacency[node]:
                    if other not in chain:
                        walk(other, chain + [other], visited_nodes | {node})
        for index in starts:
            walk(index, [index], set())
        if len(routes) != 1:
            return [], "ambiguous_arch_route" if routes else "arch_route_not_found"
        chain = next(iter(routes.values()))
        arms = []
        for position, index in enumerate(chain):
            spans = self.intervals(index, name)
            if not spans:
                continue
            if len(spans) != 1:
                return [], "ambiguous_arch_interval"
            if position + 1 < len(chain):
                far = next(iter(set(self.edges[index]) & set(self.edges[chain[position + 1]])))
                reverse = self.edges[index][0] == far
            elif position:
                near = next(iter(set(self.edges[index]) & set(self.edges[chain[position - 1]])))
                reverse = self.edges[index][1] == near
            else:
                reverse = False
            arms.append({"path_index": index, "span": spans[0], "reverse": reverse})
        return [{"chain": arms, "anchor_nodes": [], "boundary_path_indices": [], "route_path_indices": chain}], ""


def generate_landmark_label_planes(context, name, value, rule, smooth_paths, segmentation_labels, spacing,
                                   qc, path_qc, *, origin=(0, 0, 0), mask4d=None, branch_labels=None, filter_defaults=None):
    """Resolve each declarative position, recording all omissions and provenance."""
    output, seen = [], set()
    anchors = {spec["anchor"] for spec in rule["placements"]}
    source = ("observed_label_connections" if "label_connections" in anchors else
              "label_path_properties" if anchors <= {"label_level", "label_longest", "label_terminal"}
              else "anatomical_landmarks")
    qc.update(status="selected", selection_source=source, selection_warnings=[],
              main_path="landmark", selected_main_paths=[], main_path_count=0, placements=[])
    filter_defaults = dict(filter_defaults or {})
    support_cache = {}
    for placement_index, spec in enumerate(rule["placements"]):
        selections, reason = context.resolve(name, spec)
        count = len(spec["fractions"]) if spec["position"] == "fractions" else 1
        placement_qc = {"placement_index": placement_index, "spec": spec, "status": "resolved" if selections else "omitted",
                        "reason": reason, "resolved_count": len(selections)}
        qc["placements"].append(placement_qc)
        if not selections:
            qc["requested_plane_count"] += count
        if reason:
            qc["selection_warnings"].append(reason)
        if not selections:
            qc.setdefault("omitted_positions", []).append({"reason": reason, "placement_index": placement_index,
                "anchor": spec["anchor"], "requested_plane_count": count})
            continue
        requested_spec = spec
        for selection in selections:
            spec = selection.get("placement_spec", requested_spec)
            count = len(spec["fractions"]) if spec["position"] == "fractions" else 1
            qc["requested_plane_count"] += count
            spans = []
            for arm in selection["chain"]:
                index, source = arm["path_index"], arm["span"]
                smooth = smooth_paths[index]
                cum = np.r_[0., np.cumsum(np.linalg.norm(np.diff(smooth, axis=0), axis=1))]
                lo, hi = source["sample_start"], source["sample_end"]
                if hi >= len(smooth) or hi <= lo:
                    continue
                spans.append(dict(arm, cumulative=cum, lo=lo, hi=hi, length=float(cum[hi] - cum[lo])))
            total = sum(arm["length"] for arm in spans)
            if not spans or total <= 1e-8:
                qc.setdefault("omitted_positions", []).append({"reason": "empty_landmark_interval", "placement_index": placement_index})
                continue
            short = spec["position"] == "near" and total < 2 * float(spec.get("offset_mm", 5.))
            distances = [total * fraction for fraction in spec["fractions"]] if spec["position"] == "fractions" else [
                total / 2 if short else float(spec.get("offset_mm", 5.))]
            for nominal in distances:
                # Search from the nominal position into the SAME label interval.
                arm, local = spans[-1], nominal
                for candidate in spans:
                    arm = candidate
                    if local <= candidate["length"] + 1e-8:
                        break
                    local -= candidate["length"]
                index = arm["path_index"]
                arm["span"].update(selected_main_path=True, selection_reason=spec["anchor"])
                main_index = len(qc["selected_main_paths"])
                record = dict(arm["span"], main_path_index=main_index, requested_plane_count=1, actual_plane_count=0,
                              selection_reason=spec["anchor"])
                qc["selected_main_paths"].append(record)
                path_qc[index]["requested_plane_count"] += 1
                path_qc[index]["label_rules"].append({"label_name": name, "segmentation_label": value,
                    "main_path_index": main_index, "placement_index": placement_index, "anchor": spec["anchor"]})
                step = .5 * float(np.min(spacing))
                trials = [local]
                if spec["position"] == "near":
                    trials.extend(np.arange(local + step, arm["length"] - .5 * step, step).tolist())
                else:
                    # Fractions keep their nominal coordinate unless geometry
                    # rejects it; search nearest alternatives without crossing a fork.
                    for delta in np.arange(step, arm["length"], step):
                        trials.extend(x for x in (local - delta, local + delta) if 0 < x < arm["length"])
                accepted, failure = None, "center_outside_label"
                for distance in trials:
                    target = arm["cumulative"][arm["hi"]] - distance if arm["reverse"] else arm["cumulative"][arm["lo"]] + distance
                    plane = _build_plane(smooth_paths[index], arm["cumulative"], target, index)
                    if int(_sample_path_labels(np.asarray([plane.center]), segmentation_labels, spacing, origin)[0]) != value:
                        continue
                    plane.segmentation_label = value
                    hierarchy = context.infos[index].get("hierarchy", {}) if index < len(context.infos) else {}
                    if hierarchy.get("flow_sign") == -1:
                        plane.normal = -plane.normal
                    if mask4d is not None:
                        from .metrics.sampling import filter_planes_by_branch_support
                        enabled = rule.get("fov_boundary_filter")
                        enabled = filter_defaults.get("fov_boundary_filter", True) if enabled is None else enabled
                        fraction = rule.get("fov_boundary_max_fraction")
                        fraction = filter_defaults.get("fov_boundary_max_fraction", .25) if fraction is None else fraction
                        check = rule.get("segmentation_filter")
                        check = filter_defaults.get("segmentation_filter", True) if check is None else check
                        kept, reports = filter_planes_by_branch_support([plane], mask4d, spacing, origin,
                            branch_labels_3d=branch_labels, segmentation_labels_3d=segmentation_labels,
                            support_cache=support_cache, check_branch_support=check,
                            fov_boundary_max_fraction=fraction if enabled else None)
                        if not kept or (check and reports[0].get("max_cell_count", 1) == 0):
                            failure = reports[0].get("reason") or "no_label_section"
                            continue
                    signature = (index, round(float(target), 6))
                    duplicate = signature in seen
                    if requested_spec["anchor"] == "label_connections":
                        duplicate = any(old_index == index and abs(old_distance - target) < step
                                        for old_index, old_distance in seen)
                    if duplicate:
                        failure = "duplicate_landmark_position"
                        break
                    seen.add(signature)
                    plane.placement = {"schema": "autoflow.landmark_placement.v1", "anchor": spec["anchor"],
                        "placement_index": placement_index, "position": spec["position"],
                        "anchor_nodes": selection["anchor_nodes"], "boundary_path_indices": selection["boundary_path_indices"],
                        "matched_arms": selection.get("matched_arms", {}), "route_path_indices": selection.get("route_path_indices", []),
                        "label_interval_mm": [float(arm["cumulative"][arm["lo"]]), float(arm["cumulative"][arm["hi"]])],
                        "nominal_distance_in_interval_mm": float(local), "distance_in_interval_mm": float(distance),
                        "search_shift_mm": float(distance - local), "short_interval_midpoint": bool(short),
                        "direction_status": hierarchy.get("direction_status", "unavailable"),
                        "label_level": selection.get("label_level")}
                    if "path_selection" in selection:
                        plane.placement.update(path_selection=selection["path_selection"],
                                               paths_at_level=selection["paths_at_level"])
                    if requested_spec["anchor"] == "label_connections" or requested_spec.get("when_connected_to"):
                        plane.placement.update(schema="autoflow.label_connection_placement.v1",
                            requested_anchor=requested_spec["anchor"], connection_fallback=bool(selection.get("connection_fallback")),
                            connection_id=selection.get("connection_id"), connection_kind=selection.get("connection_kind"),
                            connected_labels=selection.get("connected_labels", []), near_end=bool(arm["reverse"]))
                    if spec["position"] == "fractions":
                        plane.placement["fraction"] = float(nominal / total)
                    plane._label_plane_rule, plane._main_path_index = rule, main_index
                    plane._main_path_fraction = float(distance / arm["length"])
                    accepted = plane
                    break
                if accepted is None:
                    qc.setdefault("omitted_positions", []).append({"reason": failure, "placement_index": placement_index,
                        "path_index": index, "main_path_index": main_index, "anchor": spec["anchor"]})
                else:
                    output.append(accepted)
                    record["actual_plane_count"] = 1
                    record["placement"] = accepted.placement
    qc["main_path_count"] = len(qc["selected_main_paths"])
    qc["actual_plane_count"] = len(output)
    qc["selection_warnings"] = sorted(set(qc["selection_warnings"]))
    return output
