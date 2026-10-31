"""Transactional routing of terminal blobs and repeated label contacts."""
import networkx as nx
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from autoflow.core.models import GraphData
from autoflow.algorithms.graph import graph_to_networkx, _sample_graph_labels
from autoflow.task_control import check_cancelled

def rank(g):
    return g.number_of_edges() - len(g) + nx.number_connected_components(g) if len(g) else 0

def simplify_remaining_junctions(graph, labels, spacing, old_report, flow=None, flow_spacing=None, *, origin=(0, 0, 0), flow_origin=(0, 0, 0), min_cycle_rank=2):
    points = np.asarray(graph.points)
    spacing = np.asarray(spacing)
    origin = np.asarray(origin)
    net = graph_to_networkx(graph)
    original = net.copy()
    lookup = {tuple(p): i for i, p in enumerate(points)}
    values = _sample_graph_labels(points, labels, spacing, origin)
    candidates = [c for c in old_report.get('clusters', []) if c['cycle_rank_before'] >= max(2, min_cycle_rank) and c.get('reason') not in {'coincident_graph_points', 'extent_exceeds_limit'}]
    candidates = [c for c in candidates if rank(net.subgraph([lookup[tuple(p)] for p in c['points_mm'] if tuple(p) in lookup])) > 0]
    if not candidates:
        return (graph, dict(schema='autoflow.local_junction_routes.v1', before_cycles=rank(net), after_cycles=rank(net), removed_edges=0, removed_nodes=0, checks={}, clusters=[], removed_edge_points_mm=[]))
    key = lambda n: tuple(points[n])
    edge = lambda a, b: tuple(sorted((int(a), int(b))))
    original_edges = {edge(a, b) for a, b in net.edges()}
    old_ends = {n for n, d in net.degree() if d == 1}
    lo = np.maximum(np.floor((points.min(0) - origin) / spacing).astype(int) - 3, 0)
    hi = np.minimum(np.ceil((points.max(0) - origin) / spacing).astype(int) + 4, labels.shape)
    crop = labels[tuple((slice(a, b) for a, b in zip(lo, hi)))]
    edt = ndimage.distance_transform_edt(np.pad(crop > 0, 1), sampling=spacing)
    ijk = np.clip(np.rint((points - origin) / spacing - lo + 1).astype(int), 0, np.asarray(edt.shape) - 1)
    radii = edt[tuple(ijk.T)]
    velocity = None
    if flow is not None:
        field = np.asarray(flow)
        native_spacing = np.asarray(flow_spacing if flow_spacing is not None else spacing, dtype=float)
        if field.ndim != 5 or field.shape[-1] != 3:
            raise ValueError('Junction flow must have shape (X, Y, Z, T, 3)')
        if native_spacing.shape != (3,) or not np.all(np.isfinite(native_spacing)) or np.any(native_spacing <= 0):
            raise ValueError('Junction flow spacing must contain three positive finite values')
        vox = np.clip(np.rint((points - np.asarray(flow_origin)) / native_spacing).astype(int), 0, np.asarray(field.shape[:3]) - 1)
        samples = field[tuple(vox.T)]
        finite = np.isfinite(samples)
        counts = finite.sum(1)
        velocity = np.divide(np.where(finite, samples, 0).sum(1), counts, out=np.zeros((len(points), 3)), where=counts > 0)

    def metrics(a, b):
        v = points[b] - points[a]
        length = float(np.linalg.norm(v))
        radius = float(0.5 * (radii[a] + radii[b]))
        alignment = None
        if velocity is not None:
            f = 0.5 * (velocity[a] + velocity[b])
            den = np.linalg.norm(f) * length
            if den > 1e-10:
                alignment = float(abs(np.dot(f, v)) / den)
        cost = length / (0.5 * spacing.min() + radius) ** 2 * (1 + 0.5 * (1 - (alignment if alignment is not None else 1.0)))
        return dict(length_mm=length, radius_mm=radius, absolute_mean_flow_alignment=alignment, cost=float(cost))

    def weight(a, b, _):
        return metrics(a, b)['cost']
    original_components = list(nx.connected_components(original))
    label_components = {int(v): list(nx.connected_components(original.subgraph(np.flatnonzero(values == v)))) for v in np.unique(values) if v > 0}
    records = []
    all_scope = set()
    protected = set(old_ends)

    def validate(scope):
        retained = {edge(a, b) for a, b in net.edges()}
        used = set(net)
        return dict(no_new_edges=retained <= original_edges, original_endpoints_preserved=old_ends <= used, boundary_ports_preserved=protected <= used, outside_region_edges_preserved=original_edges - scope <= retained, component_connectivity_preserved=all((bool(c & used) and nx.is_connected(net.subgraph(c & used)) for c in original_components)), label_component_connectivity_preserved=all((bool(c & used) and nx.is_connected(net.subgraph(c & used)) for comps in label_components.values() for c in comps)))
    for cluster in candidates:
        check_cancelled()
        nodes = {lookup[tuple(p)] for p in cluster['points_mm'] if tuple(p) in lookup and lookup[tuple(p)] in net}
        if len(nodes) < 3 or rank(net.subgraph(nodes)) == 0:
            continue
        previous = net.copy()
        previous_protected = set(protected)
        previous_scope = set(all_scope)
        all_scope.update((edge(a, b) for a, b in original.subgraph(nodes).edges()))
        ports = {n for n in nodes if any((x not in nodes for x in net[n]))}
        protected.update(ports)
        rec = dict(cluster_id=cluster['cluster_id'], input_nodes=len(nodes), before_cycles=rank(net.subgraph(nodes)), ports_mm=[points[n].tolist() for n in sorted(ports, key=key)], single_ports=[], contacts=[])
        for value in sorted(set(values[list(nodes)])):
            if value <= 0:
                continue
            local = net.subgraph([n for n in nodes if values[n] == value]).copy()
            for comp in list(nx.connected_components(local)):
                if rank(local.subgraph(comp)) < 2:
                    continue
                required = {n for n in comp if any((x not in comp for x in net[n]))} | comp & old_ends
                if len(required) != 1:
                    continue
                port = next(iter(required))
                part = nx.Graph()
                part.add_nodes_from(sorted(comp, key=key))
                part.add_edges_from(sorted(net.subgraph(comp).edges(), key=lambda e: tuple(sorted((key(e[0]), key(e[1]))))))
                distal = max(comp, key=lambda n: (float(np.linalg.norm(points[n] - points[port])), float(radii[n]), key(n)))
                route = nx.shortest_path(part, port, distal, weight=weight)
                keep = {edge(a, b) for a, b in zip(route, route[1:])}
                cut = {edge(a, b) for a, b in part.edges()} - keep
                net.remove_edges_from(cut)
                net.remove_nodes_from([n for n in comp if net.degree(n) == 0 and n not in required])
                protected.add(distal)
                rec['single_ports'].append(dict(label=int(value), before_cycles=rank(part), after_cycles=0, port_mm=points[port].tolist(), distal_mm=points[distal].tolist(), kept_path_mm=points[route].tolist(), removed_edges=len(cut)))
        nodes &= set(net)
        pairs = {tuple(sorted((int(values[a]), int(values[b])))) for a, b in net.subgraph(nodes).edges() if values[a] > 0 and values[b] > 0 and (values[a] != values[b])}
        for pair in sorted(pairs):
            check_cancelled()
            pidx = np.asarray(sorted(nodes))
            ll = np.maximum(np.floor((points[pidx].min(0) - origin) / spacing).astype(int) - 2, 0)
            hh = np.minimum(np.ceil((points[pidx].max(0) - origin) / spacing).astype(int) + 3, labels.shape)
            lab = labels[tuple((slice(a, b) for a, b in zip(ll, hh)))]
            a, b = pair
            aa = lab == a
            bb = lab == b
            faces = ndimage.generate_binary_structure(3, 1)
            contact = aa & ndimage.binary_dilation(bb, structure=faces) | bb & ndimage.binary_dilation(aa, structure=faces)
            cc, ncc = ndimage.label(contact, structure=np.ones((3, 3, 3)))
            coords = np.argwhere(cc > 0)
            if not len(coords):
                continue
            tree = cKDTree((coords + ll) * spacing + origin)
            patches = {}
            for u, v in list(net.subgraph(nodes).edges()):
                if tuple(sorted((int(values[u]), int(values[v])))) != pair:
                    continue
                distance, index = tree.query(0.5 * (points[u] + points[v]))
                if distance > 2 * spacing.min():
                    continue
                cid = int(cc[tuple(coords[index])])
                patches.setdefault(cid, []).append(edge(u, v))
            for cid, edges in sorted(patches.items()):
                if len(edges) < 2:
                    continue
                ordered = sorted(edges, key=lambda e: (metrics(*e)['cost'], tuple(sorted((key(e[0]), key(e[1]))))))
                cut = []
                for u, v in ordered[::-1]:
                    retained = [e for e in edges if net.has_edge(*e)]
                    if len(retained) <= 1:
                        break
                    net.remove_edge(u, v)
                    if nx.has_path(net.subgraph(nodes), u, v):
                        cut.append([u, v])
                    else:
                        net.add_edge(u, v)
                kept = [e for e in edges if net.has_edge(*e)]
                protected.update((n for e in kept for n in e))
                rec['contacts'].append(dict(labels=list(pair), contact_component=cid, before_edges=len(edges), after_edges=len(kept), removed_edges=len(cut), kept=[dict(points_mm=points[list(e)].tolist(), **metrics(*e)) for e in kept]))
        while set(previous.edges()) != set(net.edges()) or set(previous) != set(net):
            leaves = [n for n in nodes if n in net and net.degree(n) <= 1 and (n not in protected) and (values[n] > 0)]
            if not leaves:
                break
            net.remove_nodes_from(leaves)
        rec['checks'] = validate(all_scope)
        if not all(rec['checks'].values()):
            net = previous
            protected = previous_protected
            all_scope = previous_scope
            rec.update(status='retained', reason='connectivity_check_failed')
        else:
            rec['status'] = 'simplified' if {edge(a, b) for a, b in previous.edges()} != {edge(a, b) for a, b in net.edges()} else 'unchanged'
        rec.update(after_cycles=rank(net.subgraph(nodes & set(net))), remaining_nodes=len(nodes & set(net)), removed_edges=len({edge(a, b) for a, b in previous.edges()} - {edge(a, b) for a, b in net.edges()}))
        records.append(rec)
    retained = {edge(a, b) for a, b in net.edges()}
    used = set(net)
    removed = original_edges - retained
    checks = validate(all_scope)
    if not all(checks.values()):
        raise RuntimeError('Junction cleanup changed a protected endpoint or component')
    ids = sorted(net)
    index = {n: i for i, n in enumerate(ids)}
    out = GraphData(points=points[ids].copy(), edges=np.asarray([[index[a], index[b]] for a, b in sorted(retained)], int).reshape(-1, 2))
    report = dict(schema='autoflow.local_junction_routes.v1', stage='after_label_routing_before_flow', before_cycles=rank(original), after_cycles=rank(net), removed_edges=len(removed), removed_nodes=len(points) - len(ids), checks=checks, clusters=records, removed_edge_points_mm=[points[[a, b]].tolist() for a, b in sorted(removed)], cost='length/(0.5*min_spacing+mean_radius)^2 * (1+0.5*(1-abs_mean_flow_alignment))', scope='existing junction clusters only')
    return (graph if not removed else out, report)
