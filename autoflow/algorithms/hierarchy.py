"""Connected subgroup topology, multi-source depth and label-relative relationships.

The physical graph is immutable. Directed cycles are condensed only in this
analysis; parent references and every original graph edge remain available.
"""

from collections import defaultdict

import networkx as nx
import numpy as np

from .graph import graph_to_networkx
from .paths import inter_points


def graph_edge_chains(graph):
    """Cover each undirected graph edge once, including isolated closed rings."""
    network = graph_to_networkx(graph)
    junctions = {int(node) for node, degree in network.degree() if degree != 2}
    visited, chains = set(), []

    def walk(start, neighbor):
        chain = [int(start), int(neighbor)]
        visited.add(tuple(sorted((int(start), int(neighbor)))))
        previous, current = start, neighbor
        while current not in junctions:
            following = sorted(node for node in network.neighbors(current) if node != previous)
            if not following:
                break
            nxt = following[0]
            edge = tuple(sorted((int(current), int(nxt))))
            if edge in visited:
                break
            visited.add(edge)
            chain.append(int(nxt))
            previous, current = current, nxt
        return chain

    for start in sorted(junctions):
        for neighbor in sorted(network.neighbors(start)):
            if tuple(sorted((int(start), int(neighbor)))) not in visited:
                chains.append(walk(start, neighbor))
    # Components with only degree-2 nodes have no endpoint/junction seed.
    for left, right in sorted(tuple(sorted(map(int, edge))) for edge in network.edges()):
        if (left, right) not in visited:
            chains.append(walk(left, right))
    return chains


def _direction(points, flow, mask, spacing, flow_spacing=None):
    if flow is None:
        return {"direction_status": "flow_unavailable", "direction_confidence": None, "flow_sign": None}
    from .label_planes import _flow_direction
    score = _flow_direction(inter_points(points, time=3), flow, mask,
                            spacing if flow_spacing is None else flow_spacing, (0, 0, 0), mask_spacing=spacing)
    reliable = score["flow_direction_status"] == "reliable"
    return {"direction_status": "reliable" if reliable else "flow_direction_unreliable",
            "direction_confidence": score["flow_confidence"],
            "flow_sign": score["flow_sign"] if reliable else None}


def hierarchy_path_order(report):
    """Order path ids by component and ascending depth, unresolved paths last."""
    return [int(record["path_index"]) for record in sorted(report["paths"], key=lambda record: (
        int(record["subgroup_index"]), record["subgroup_level"] is None,
        record["subgroup_level"] if record["subgroup_level"] is not None else 0,
        int(record["path_index"]),
    ))]


def _rank(network, inlet_paths, blocked=(), first_level=1):
    """Rank the SCC DAG; no inlet or unknown upstream means unknown depth."""
    components = sorted((sorted(nodes) for nodes in nx.strongly_connected_components(network)),
                        key=lambda nodes: nodes[0])
    membership = {node: index for index, nodes in enumerate(components) for node in nodes}
    dag = nx.DiGraph()
    dag.add_nodes_from(range(len(components)))
    dag.add_edges_from((membership[a], membership[b]) for a, b in network.edges()
                       if membership[a] != membership[b])
    levels, roots, loop_ids = {}, {}, {}
    for index, nodes in enumerate(components):
        if len(nodes) > 1 or any(network.has_edge(node, node) for node in nodes):
            loop_ids[index] = f"loop_{min(nodes)}"
    for index in nx.lexicographical_topological_sort(dag):
        parents = sorted(dag.predecessors(index))
        anchors = set(components[index]) & set(inlet_paths)
        roots[index] = anchors | set().union(*(roots[parent] for parent in parents))
        if set(components[index]) & set(blocked) or not roots[index] or any(levels[parent] is None for parent in parents):
            levels[index] = None
        else:
            levels[index] = max((levels[parent] + 1 for parent in parents), default=first_level)
    return {node: {"level": levels[membership[node]], "root_path_indices": sorted(roots[membership[node]]),
                   "loop_id": loop_ids.get(membership[node]), "block_index": membership[node]}
            for node in network}, components


def analyze_group_graph(graph, node_chains, *, flow=None, mask=None, spacing=(1, 1, 1),
                        segmentation_labels=None, label_map=None, group_name="", path_indices=None,
                        direction_paths=None, flow_spacing=None):
    """Annotate compressed graph arms before final centerline path materialization."""
    indices = list(range(len(node_chains))) if path_indices is None else list(map(int, path_indices))
    network = graph_to_networkx(graph)
    used_nodes = {int(node) for chain in node_chains for node in chain}
    network = network.subgraph(used_nodes).copy()
    points = np.asarray(graph.points, dtype=float)
    degrees = dict(network.degree())
    bridges = {tuple(sorted(map(int, edge))) for edge in nx.bridges(network)}
    components = sorted((sorted(map(int, nodes)) for nodes in nx.connected_components(network)),
                        key=lambda nodes: nodes[0])
    component_of = {node: index for index, nodes in enumerate(components) for node in nodes}
    subgroup_ids = {index: f"{group_name}:subgroup_{index + 1}" for index in range(len(components))}
    names = {int(value): str(name) for name, value in dict(label_map or {}).items()}
    incoming, outgoing, unknown = defaultdict(list), defaultdict(list), defaultdict(list)
    records = {}
    for local_index, (path_index, nodes) in enumerate(zip(indices, node_chains)):
        curve = points[np.asarray(nodes, dtype=int)]
        sample = curve if direction_paths is None else direction_paths[local_index]
        direction = _direction(np.asarray(sample, dtype=float), flow, mask, spacing, flow_spacing)
        component = component_of[int(nodes[0])]
        record = {"path_index": path_index, "group_name": str(group_name),
                  "component_index": component, "subgroup_index": component,
                  "subgroup_id": subgroup_ids[component],
                  "start_node": int(nodes[0]), "end_node": int(nodes[-1]),
                  "length_mm": float(np.linalg.norm(np.diff(curve, axis=0), axis=1).sum()),
                  "closed": bool(nodes[0] == nodes[-1]),
                  "topological_cycle": any(tuple(sorted((int(a), int(b)))) not in bridges
                                           for a, b in zip(nodes, nodes[1:])),
                  **direction}
        source, target = int(nodes[0]), int(nodes[-1])
        if direction["flow_sign"] == -1:
            source, target = target, source
        record.update(source_node=source if direction["flow_sign"] is not None else None,
                      target_node=target if direction["flow_sign"] is not None else None,
                      is_inlet=direction["flow_sign"] is not None and degrees[source] == 1,
                      is_outlet=direction["flow_sign"] is not None and degrees[target] == 1)
        if direction["flow_sign"] is None:
            unknown[source].append(path_index)
            unknown[target].append(path_index)
        else:
            outgoing[source].append(path_index)
            incoming[target].append(path_index)
        membership = []
        if segmentation_labels is not None:
            from .planes import _sample_path_labels
            dense = inter_points(curve, time=3)
            values = _sample_path_labels(dense, segmentation_labels, spacing, (0, 0, 0))
            lengths = np.linalg.norm(np.diff(dense, axis=0), axis=1)
            for value in sorted(int(value) for value in np.unique(values) if int(value) > 0):
                length = float(lengths[values[:-1] == value].sum())
                membership.append({"label": value, "label_name": names.get(value, f"Label {value}"),
                                   "length_mm": length})
        record["labels"] = membership
        record["label_names"] = [item["label_name"] for item in membership]
        records[path_index] = record
    directed = nx.DiGraph()
    directed.add_nodes_from(indices)
    junctions, blocked = [], set()
    for node in sorted(set(incoming) | set(outgoing) | set(unknown)):
        ins, outs, uncertain = sorted(incoming[node]), sorted(outgoing[node]), sorted(set(unknown[node]))
        directed.add_edges_from((parent, child) for parent in ins for child in outs if parent != child)
        if uncertain:
            blocked.update(outs)
        kind = ("undirected" if uncertain else "split" if len(ins) == 1 and len(outs) > 1 else
                "merge" if len(ins) > 1 and len(outs) == 1 else
                "mixed" if len(ins) > 1 and len(outs) > 1 else
                "inlet" if not ins and len(outs) == 1 else
                "outlet" if len(ins) == 1 and not outs else "direction_conflict")
        junctions.append({"node": node, "point_mm": points[node].tolist(), "degree": degrees[node],
                          "subgroup_index": component_of[node], "subgroup_id": subgroup_ids[component_of[node]],
                          "kind": kind, "incoming_path_indices": ins, "outgoing_path_indices": outs,
                          "undirected_path_indices": uncertain})
    for path_index, record in records.items():
        if record["closed"]:
            directed.add_edge(path_index, path_index)
        if record["direction_status"] != "reliable":
            blocked.add(path_index)
    # Each physical connected component is an independent grading domain.
    # A component can have multiple inlets, or no reliable inlet at all.
    rank, blocks = {}, []
    for component in range(len(components)):
        members = {i for i in indices if records[i]["subgroup_index"] == component}
        local_rank, local_blocks = _rank(directed.subgraph(members),
            [i for i in sorted(members) if records[i]["is_inlet"]], blocked & members)
        for item in local_rank.values():
            item["block_index"] += len(blocks)
        rank.update(local_rank)
        blocks.extend(local_blocks)
    loops = []
    for nodes in blocks:
        item = rank[nodes[0]]
        if item["loop_id"]:
            loop_id = f"{group_name}:{item['loop_id']}"
            for node in nodes:
                rank[node]["loop_id"] = loop_id
            loops.append({"loop_id": loop_id, "path_indices": nodes, "group_level": item["level"],
                          "subgroup_level": item["level"], "subgroup_id": records[nodes[0]]["subgroup_id"],
                          "subgroup_index": records[nodes[0]]["subgroup_index"],
                          "parent_path_indices": sorted({parent for node in nodes for parent in directed.predecessors(node)} - set(nodes)),
                          "child_path_indices": sorted({child for node in nodes for child in directed.successors(node)} - set(nodes))})
    for path_index, record in records.items():
        parents = sorted(directed.predecessors(path_index))
        children = sorted(directed.successors(path_index))
        level = rank[path_index]["level"]
        candidates = [parent for parent in parents if rank[parent]["block_index"] != rank[path_index]["block_index"]]
        display_parent = min(candidates, key=lambda parent: (-(rank[parent]["level"] or 0), parent)) if candidates else None
        record.update(group_level=level, subgroup_level=level, root_path_indices=rank[path_index]["root_path_indices"],
                      loop_id=rank[path_index]["loop_id"], parent_path_indices=parents,
                      child_path_indices=children, display_parent_path_index=display_parent,
                      reference_parent_path_indices=[parent for parent in parents if parent != display_parent],
                      level_status="unoriented" if record["direction_status"] != "reliable" else
                      "unrooted_loop" if level is None and rank[path_index]["loop_id"] else
                      "upstream_unresolved" if level is None else "ranked",
                      label_local_levels={})
    # Local label depth is an annotation of this same connectivity. It never
    # clips graph arms, creates an inlet, or changes the group-level depth.
    label_values = sorted({item["label"] for record in records.values() for item in record["labels"]})
    for value in label_values:
        members = {i for i, record in records.items() if any(item["label"] == value for item in record["labels"])}
        local = directed.subgraph(members).copy()
        local_roots = [i for i in members if local.in_degree(i) == 0 and records[i]["direction_status"] == "reliable"]
        local_rank, _ = _rank(local, local_roots, blocked & members, first_level=0)
        for i in members:
            records[i]["label_local_levels"][str(value)] = local_rank[i]["level"]
    subgroups = []
    for component, nodes in enumerate(components):
        members = sorted(i for i in indices if records[i]["subgroup_index"] == component)
        physical = network.subgraph(nodes)
        grades = [records[i]["subgroup_level"] for i in members if records[i]["subgroup_level"] is not None]
        subgroups.append({"subgroup_id": subgroup_ids[component], "subgroup_index": component,
            "group_name": str(group_name), "path_indices": members,
            "label_names": sorted({name for i in members for name in records[i]["label_names"]}),
            "node_count": physical.number_of_nodes(), "edge_count": physical.number_of_edges(),
            "inlet_path_indices": [i for i in members if records[i]["is_inlet"]],
            "outlet_path_indices": [i for i in members if records[i]["is_outlet"]],
            "loop_ids": [loop["loop_id"] for loop in loops if loop["subgroup_index"] == component],
            "undirected_cycle_rank": physical.number_of_edges() - physical.number_of_nodes() + 1,
            "ranked_path_count": len(grades), "unresolved_path_count": len(members) - len(grades),
            "maximum_level": max(grades, default=None)})
    return {"group_name": str(group_name), "grading_scope": "connected_subgroup", "subgroups": subgroups,
            "paths": list(records.values()), "junctions": junctions,
            "loops": loops, "inlet_path_indices": [i for i in indices if records[i]["is_inlet"]],
            "outlet_path_indices": [i for i in indices if records[i]["is_outlet"]],
            "component_count": nx.number_connected_components(network),
            "undirected_cycle_rank": network.number_of_edges() - network.number_of_nodes() + nx.number_connected_components(network)}




def build_workspace_hierarchy(workspace):
    """Analyze the existing physical vessel groups without auxiliary graphs."""
    ws = workspace
    fine_labels = getattr(ws, 'segmentation_analysis_labels_3d', None)
    geometry_spacing = ws.resolution if fine_labels is None else ws.segmentation_spacing
    geometry_labels = ws.segmask_labels_3d if fine_labels is None else fine_labels
    groups, all_paths = [], []
    for group_name in ws.group_order:
        state = ws.multilabel_groups[group_name]
        indices = [int(item["path_index"]) for item in state.get("path_info", [])]
        # Group chains use global node ids while their graph stores local ids.
        # Use the global graph so compacted/unused node ids need no inferred offset.
        report = analyze_group_graph(ws.graph, state.get("centerline_node_paths", []), flow=ws.flow_raw,
            mask=state.get("geometry_binary", state.get("segmask_binary")), spacing=geometry_spacing, segmentation_labels=geometry_labels,
            label_map=ws.skeleton_params.label_map, group_name=group_name, path_indices=indices, flow_spacing=ws.resolution)
        report["graph_connectivity"] = ws.skeleton_params.params_for_group(group_name).graph_connectivity
        report["cycle_cleanup"] = state.get("cycle_cleanup", {})
        report["junction_cluster_cleanup"] = state.get("junction_cluster_cleanup", {})
        groups.append(report)
        all_paths.extend(report["paths"])
    by_index = {record["path_index"]: record for record in all_paths}
    for info in ws.path_info:
        if int(info.get("path_index", -1)) in by_index:
            info["hierarchy"] = by_index[int(info["path_index"])]
    return {"schema": "autoflow.graph_hierarchy.v1", "status": "available" if all_paths else "empty",
            "grading_scope": "connected_subgroup",
            "graph_connectivity": int(ws.skeleton_params.graph_connectivity),
            "path_numbering": "group order; subgroup order; ascending level; unresolved last; ties keep graph traversal order",
            "level_rule": "within each connected subgroup: inlet=1; downstream=1+max(upstream); directed cycles condensed",
            "label_local_level_rule": "first label-bearing arm=0; each subsequent label-bearing junction adds 1",
            "direction_confidence_threshold": .05, "groups": groups, "paths": all_paths}
