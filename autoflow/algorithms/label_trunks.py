"""Continuous, radius-supported label routes for approximate clinical targets.

Graph paths are numerical pieces, not measurement segments. Rejoin only their
existing same-label endpoint identities; never bridge a mask gap or a label.
"""

from collections import deque

import networkx as nx
import numpy as np

from ..task_control import check_cancelled

COMPETING_SUPPORT_RATIO = .9
MAX_COMPETING_NORMAL_ANGLE_DEG = 30.
MIN_REFERENCE_EXTENT_VOXELS = 2.


REFERENCE_LABELS = {
    'MPA': ('LPA', 'RPA'), 'LPA': ('MPA',), 'RPA': ('MPA',),
    'PV': ('SMV', 'SV'), 'SMV': ('PV',), 'SV': ('PV',),
    'LTS': ('SSS', 'StrS'), 'RTS': ('SSS', 'StrS'),
    'SSS': ('LTS', 'RTS', 'StrS'), 'StrS': ('LTS', 'RTS', 'SSS'),
    'LMCA': ('LICA',), 'RMCA': ('RICA',),
    'LPCA': ('BA', 'LICA'), 'RPCA': ('BA', 'RICA'), 'ACA': ('LICA', 'RICA'),
    **{name: ('AAO', 'ARCH', 'DAO') for name in ('RBCT', 'CCA', 'LBCT', 'LRA', 'RRA', 'HA', 'SMA')},
    'ARCH': ('AAO', 'DAO'),
}


def _key(value):
    return repr(value)


class LabelTrunks:
    def __init__(self, context):
        self.context = context
        self.cache = {}
        self.audit = []

    def network(self, value):
        if value in self.cache:
            return self.cache[value]
        ctx = self.context
        graph, arms = nx.Graph(), {}
        for (index, label), spans in sorted(ctx.spans.items()):
            if label != value:
                continue
            for span in spans:
                points = np.asarray(ctx.paths[index][span['sample_start']:span['sample_end']+1])
                if len(points) < 2:
                    continue
                endpoints = []
                for side, point in zip(('start', 'end'), (points[0], points[-1])):
                    node = ctx.edges[index][side == 'end'] if span.get('touches_path_'+side) else (
                        str(span.get('group_name', '')), 'label_end', index, span['segment_index'], side)
                    graph.add_node(node, point=np.asarray(point))
                    endpoints.append(node)
                radii = ctx.trunk_radii(index, points)
                dl = np.linalg.norm(np.diff(points, axis=0), axis=1)
                support = float(np.sum(dl * (.5*(radii[:-1]+radii[1:])+.5*ctx.trunk_spacing.min())**2))
                identity = (index, span['segment_index'])
                arm = dict(path_index=index, span=dict(span, segmentation_label=value), reverse=False)
                if graph.has_edge(*endpoints) or endpoints[0] == endpoints[1]:
                    graph.graph.setdefault('ambiguous_cycle_nodes', set()).update(endpoints)
                graph.add_edge(*endpoints, arm=arm, support=max(support, 1e-9), length=float(dl.sum()))
                arms[identity] = tuple(endpoints)
        self.cache[value] = graph, arms
        return graph, arms

    def route(self, graph, nodes, *, orient_flow):
        ctx = self.context
        chain, score, signed, evidence, length = [], 0., 0., 0., 0.
        for u, v in zip(nodes, nodes[1:]):
            edge = graph.edges[u, v]
            arm = dict(edge['arm'])
            original = self.network(int(arm['span']['segmentation_label']))[1][(arm['path_index'], arm['span']['segment_index'])]
            arm['reverse'] = original[0] != u
            arm['trunk_edge_support_mm3'] = edge['support']
            chain.append(arm)
            score += edge['support']; length += edge['length']
            hierarchy = ctx.infos[arm['path_index']].get('hierarchy', {}) if arm['path_index'] < len(ctx.infos) else {}
            sign = hierarchy.get('flow_sign')
            if sign in (-1, 1):
                weight = edge['length'] * max(float(hierarchy.get('direction_confidence') or 1.), .05)
                signed += (-1 if arm['reverse'] else 1) * sign * weight
                evidence += weight
        if orient_flow and signed < 0:
            chain = [dict(arm, reverse=not arm['reverse']) for arm in reversed(chain)]
        return {'chain': chain, 'anchor_nodes': [], 'boundary_path_indices': [],
                'route_path_indices': [arm['path_index'] for arm in chain],
                'trunk_support_mm3': score, 'trunk_length_mm': length,
                'trunk_direction_reliable': bool(evidence and abs(signed)/evidence >= .05),
                'selection_scope': 'label', 'trunk_route_schema': 'autoflow.continuous_label_trunk.v1'}

    @staticmethod
    def _routes(graph, root=None):
        """Weighted tree diameter plus terminal alternatives, without T squared enumeration."""
        leaves = sorted((n for n, d in graph.degree() if d == 1), key=_key)
        if len(leaves) < 2:
            return []
        if root is None:
            distances = nx.single_source_dijkstra_path_length(graph, leaves[0], weight='support')
            first = min(leaves, key=lambda n: (-distances[n], _key(n)))
            distances = nx.single_source_dijkstra_path_length(graph, first, weight='support')
            second = min(leaves, key=lambda n: (-distances[n], _key(n)))
            starts = [first, second]
        else:
            starts = [root]
        rows, seen = [], set()
        for start in starts:
            distances, paths = nx.single_source_dijkstra(graph, start, weight='support')
            for end in leaves:
                if end == start:
                    continue
                route = paths[end]
                identity = tuple(sorted((_key(route[0]), _key(route[-1]))))
                if identity in seen:
                    continue
                seen.add(identity); rows.append((distances[end], route))
        return [route for _, route in sorted(rows, key=lambda r: (-r[0], tuple(map(_key, r[1]))))]

    def targets(self, route, spec, *, with_normals=False):
        points = []
        for arm in route['chain']:
            span = arm['span']
            p = self.context.paths[arm['path_index']][span['sample_start']:span['sample_end']+1]
            if arm['reverse']:
                p = p[::-1]
            points.extend(p)
        points = np.asarray(points, dtype=float)
        arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
        length = arc[-1]
        if spec['position'] == 'fractions':
            distances = [length*(1-f if spec.get('fraction_reference') == 'flow_end' else f) for f in spec['fractions']]
        elif spec['position'] == 'before_end':
            distances = [length-spec['offset_mm']]
        else:
            distances = [spec.get('offset_mm', 5.)]
        if any(d < 0 or d > length for d in distances):
            return None
        locations = np.array([[np.interp(d, arc, points[:, a]) for a in range(3)] for d in distances])
        if not with_normals:
            return locations
        normals = []
        for d in distances:
            lo, hi = max(0., d-.5), min(length, d+.5)
            normal = np.array([np.interp(hi, arc, points[:, a])-np.interp(lo, arc, points[:, a]) for a in range(3)])
            normals.append(normal/max(np.linalg.norm(normal),1e-12))
        return locations, np.asarray(normals)

    def choose(self, routes, spec):
        if not routes:
            return [], 'no_label_trunk'
        routes.sort(key=lambda r: (-r['trunk_support_mm3'], tuple(r['route_path_indices'])))
        best = routes[0]; geometry = self.targets(best, spec, with_normals=True)
        target = None if geometry is None else geometry[0]
        record = {'candidate_count':len(routes), 'candidates':[
            {key: route[key] for key in ('route_path_indices','trunk_support_mm3','trunk_length_mm')}
            for route in routes[:12]]}
        self.audit.append(record)
        fractions = np.asarray(spec.get('fractions', [.5]), dtype=float)
        needs_direction = (spec['position'] in {'before_end','after_start'} or bool(spec.get('fraction_reference')) or
            (spec['anchor']=='label_trunk' and not np.allclose(np.sort(fractions), np.sort(1-fractions),atol=.015,rtol=0)))
        if needs_direction and not best['trunk_direction_reliable']:
            record['reason']='flow_direction_unavailable'
            return [], 'flow_direction_unavailable'
        alternatives, max_shift = 0, 0.
        best_edges = {(a['path_index'],a['span']['segment_index']):a['trunk_edge_support_mm3'] for a in best['chain']}
        for other in routes[1:]:
            if other['trunk_support_mm3'] < COMPETING_SUPPORT_RATIO*best['trunk_support_mm3']:
                break
            shared = sum(best_edges.get((a['path_index'],a['span']['segment_index']),0.) for a in other['chain'])
            best_tail, other_tail = best['trunk_support_mm3']-shared, other['trunk_support_mm3']-shared
            if best_tail > 1e-9 and other_tail < COMPETING_SUPPORT_RATIO*best_tail:
                continue
            alternatives += 1
            other_geometry = self.targets(other, spec, with_normals=True)
            test = None if other_geometry is None else other_geometry[0]
            if target is None or test is None:
                if target is not None or test is not None:
                    record['reason']='ambiguous_label_trunk'
                    return [], 'ambiguous_label_trunk'
                continue
            shift = np.linalg.norm(target-test, axis=1)
            max_shift = max(max_shift, float(shift.max(initial=0.)))
            # Close-score routes are interchangeable only at the actual requested positions.
            angles = np.degrees(np.arccos(np.clip(np.abs(np.sum(geometry[1]*other_geometry[1],axis=1)),0.,1.)))
            for p, distance, angle in zip(target, shift, angles):
                radius = self.context.trunk_radii(best['chain'][0]['path_index'], np.asarray([p]))[0]
                if distance > max(2.*float(self.context.trunk_spacing.max()), float(radius)) or angle > MAX_COMPETING_NORMAL_ANGLE_DEG:
                    record.update(reason='ambiguous_label_trunk', target_shift_mm=float(distance), normal_angle_deg=float(angle))
                    return [], 'ambiguous_label_trunk'
        best.update(trunk_competing_routes=alternatives, trunk_candidate_target_shift_mm=max_shift,
                    trunk_support_margin_threshold=.1)
        return [best], ''

    def volume_references(self, graph, value, expected):
        """Observe face contact at real label endpoints, including across groups.

        This supplies anchor evidence only: it creates no edge, joins no path,
        and does not search across empty voxels for the nearest parent.
        """
        ctx = self.context
        if ctx.trunk_segmentation is None:
            return {}, 0
        volume = np.asarray(ctx.trunk_segmentation)
        shape = np.asarray(volume.shape)
        spacing = ctx.trunk_spacing
        threshold = MIN_REFERENCE_EXTENT_VOXELS * float(spacing.min())
        steps = np.vstack((np.eye(3, dtype=int), -np.eye(3, dtype=int)))
        roots, weak = {}, 0
        for node, degree in graph.degree():
            if degree != 1:
                continue
            point = graph.nodes[node]['point']
            voxel = np.rint((point-np.asarray(ctx.trunk_origin))/spacing).astype(int)
            if np.any(voxel < 0) or np.any(voxel >= shape) or volume[tuple(voxel)] != value:
                continue
            for step in steps:
                neighbor = voxel + step
                if np.any(neighbor < 0) or np.any(neighbor >= shape):
                    continue
                label = int(volume[tuple(neighbor)])
                if label not in expected:
                    continue
                # Stop once a physical extent lower bound proves this is more
                # than a one-voxel label island. Small islands are fully read.
                seen = {tuple(neighbor)}
                queue = deque(seen)
                lo = neighbor.copy(); hi = neighbor.copy(); extent = 0.
                while queue and extent < threshold:
                    current = np.asarray(queue.popleft())
                    for direction in steps:
                        test = current + direction
                        key = tuple(test)
                        if (key in seen or np.any(test < 0) or np.any(test >= shape)
                                or volume[key] != label):
                            continue
                        seen.add(key); queue.append(key)
                        lo = np.minimum(lo, test); hi = np.maximum(hi, test)
                        extent = float(np.max((hi-lo)*spacing))
                        if extent >= threshold:
                            break
                if extent < threshold:
                    weak += 1
                    continue
                roots.setdefault(node, dict(event_id='volume_interface:'+repr(node),
                    labels=sorted((value, label)), kind='label_volume_interface',
                    group_name=str(node[0]), boundary_ports=[],
                    reference_component_extent_mm=extent))
        return roots, weak

    def resolve(self, name, spec):
        ctx = self.context
        self.audit = []
        value = int(ctx.labels.get(name, int(name) if str(name).isdigit() else 0))
        graph, arms = self.network(value)
        if not graph:
            return [], 'no_label_trunk'
        components = list(nx.connected_components(graph))
        candidates, failures = [], []
        reference = spec['anchor'] == 'label_reference'
        independent = spec.get('path_selection') == 'all' or (not reference and spec.get('scope','component') == 'component')
        roots = {}
        weak_references = 0
        if reference:
            expected = {int(ctx.labels[token]) for token in REFERENCE_LABELS.get(name, ()) if token in ctx.labels}
            for event in ctx.observed_connections()[0]:
                if not expected.intersection(event['labels']):
                    continue
                extent = 0.
                for label in expected.intersection(event['labels']):
                    foreign, foreign_arms = self.network(label)
                    for span, _ in event['selected'].get(label, []):
                        ends = foreign_arms.get((span['path_index'],span['segment_index']))
                        if ends:
                            nodes = nx.node_connected_component(foreign,ends[0])
                            coordinates = np.asarray([foreign.nodes[n]['point'] for n in nodes])
                            extent = max(extent,float(np.ptp(coordinates,axis=0).max(initial=0.)))
                if extent < MIN_REFERENCE_EXTENT_VOXELS*float(ctx.trunk_spacing.min()):
                    weak_references += 1
                    self.audit.append({'reason':'label_reference_too_short','connection_id':event['event_id'],
                        'reference_component_extent_mm':extent})
                    continue
                event = dict(event, reference_component_extent_mm=extent)
                for span, reverse in event['selected'].get(value, []):
                    identity = (span['path_index'], span['segment_index'])
                    if identity in arms:
                        node = arms[identity][bool(reverse)]
                        roots.setdefault(node, event)
            volume_roots, volume_weak = self.volume_references(graph, value, expected)
            for node, event in volume_roots.items():
                roots.setdefault(node, event)
            weak_references += volume_weak
            if not roots:
                return [], 'label_reference_too_short' if weak_references else 'label_reference_not_found'
        if spec.get('path_selection') == 'unique' and (len(roots) if reference else len(components)) > 1:
            return [], 'ambiguous_label_reference' if reference else 'ambiguous_label_trunk'
        for nodes in components:
            check_cancelled()
            local = graph.subgraph(nodes)
            sources = sorted((n for n in roots if n in nodes), key=_key) if reference else [None]
            if not sources:
                continue
            if nodes.intersection(graph.graph.get('ambiguous_cycle_nodes', set())) or not nx.is_tree(local):
                failures.append('ambiguous_label_trunk_cycle')
                continue
            for source in sources:
                routes = [self.route(local, r, orient_flow=not reference) for r in self._routes(local, source)]
                if source is not None:
                    event = roots[source]
                    for route in routes:
                        route.update(connection_id=event['event_id'], connected_labels=event['labels'],
                            connection_kind=event['kind'], reference_labels=list(REFERENCE_LABELS.get(name, ())),
                            reference_component_extent_mm=event['reference_component_extent_mm'],
                            anchor_nodes=[dict(point_mm=graph.nodes[source]['point'].tolist(), group_name=event['group_name'])],
                            boundary_path_indices=sorted({p['path_index'] for p in event['boundary_ports']}))
                if independent:
                    selected, reason = self.choose(routes, spec)
                    candidates.extend(selected)
                    if reason:
                        failures.append(reason)
                else:
                    candidates.extend(routes)
        if not independent:
            if failures:
                self.audit.append({'reason':failures[0], 'failed_components':len(failures)})
                return [], failures[0]
            candidates, reason = self.choose(candidates, spec)
            if reason:
                failures.append(reason)
        for route in candidates:
            route['selection_source'] = 'continuous_reference_trunk' if reference else 'continuous_label_trunk'
            route['path_selection'] = spec.get('path_selection','longest')
            route['connection_fallback'] = False
        if failures:
            self.audit.append({'reason':failures[0], 'failed_components':len(failures)})
        return candidates, failures[0] if failures else ''
