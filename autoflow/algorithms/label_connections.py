"""Observed cross-label interfaces on the physical path multigraph.

Detection uses labels, real path endpoints and contiguous samples only. It does
not depend on anatomical names or invent edges across background gaps.
"""

import networkx as nx
import numpy as np

from .planes import _build_plane


def cumulative(points):
    return np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]


def candidate_key(span):
    return int(span['segmentation_label']), int(span['path_index']), int(span['segment_index'])

def build_observed_connections(paths, candidates, infos=None):
    infos = list(infos or [])
    candidates = {int(value): [dict(row) for row in spans] for value, spans in candidates.items()}
    graph, by_path, ends = nx.MultiGraph(), {}, {}
    for value, spans in candidates.items():
        for span in spans:
            span['segmentation_label'] = int(value)
            by_path.setdefault(span['path_index'], []).append(span)
    distances = [cumulative(path) for path in paths]
    incident = {}
    for i, path in enumerate(paths):
        if len(path) < 2:
            continue
        info = infos[i] if i < len(infos) else {}
        group = str(info.get('group_name', ''))
        def endpoint(side, point):
            node = int(info.get(side + '_node', -1))
            return (group, 'node', node) if node >= 0 else (group, 'point', *np.round(point, 6))
        u, v = endpoint('start', path[0]), endpoint('end', path[-1])
        graph.add_edge(u, v, key=i)
        ends[i] = (u, v)
        incident.setdefault(u, []).append((i, False))
        incident.setdefault(v, []).append((i, True))
    component_of = {node: i for i, nodes in enumerate(nx.connected_components(graph)) for node in nodes}
    for spans in candidates.values():
        for span in spans:
            span['component_index'] = component_of[ends[span['path_index']][0]]
    ports, gaps, junction_ports = [], [], {}
    for i, spans in by_path.items():
        spans.sort(key=lambda s: s['sample_start'])
        for a, b in zip(spans, spans[1:]):
            if a['segmentation_label'] == b['segmentation_label']:
                continue
            if b['sample_start'] != a['sample_end'] + 1:
                gaps.append({'path_index': i, 'labels': [a['segmentation_label'], b['segmentation_label']], 'reason': 'intervening_background'})
                continue
            distance = .5 * (a['end_distance_mm'] + b['start_distance_mm'])
            event = {'kind': 'label_boundary', 'path_index': i, 'point_mm': _build_plane(paths[i], distances[i], distance, i).center.tolist(),
                     'distance_mm': distance, 'arms': [(a, True), (b, False)], 'group_name': ends[i][0][0],
                     'component_index': a['component_index'], 'node_id': None, 'boundary_ports': []}
            event['boundary_ports'] = [{'path_index': i, 'distance_mm': distance, 'labels': [a['segmentation_label'], b['segmentation_label']], 'point_mm': event['point_mm']}]
            options = []
            if a['sample_start'] == 0 and len(incident[ends[i][0]]) >= 3:
                options.append((distance, ends[i][0], a['segmentation_label']))
            if b['sample_end'] == len(paths[i])-1 and len(incident[ends[i][1]]) >= 3:
                options.append((distances[i][-1]-distance, ends[i][1], b['segmentation_label']))
            options.sort(key=lambda x: x[0])
            if options and (len(options) == 1 or abs(options[0][0]-options[1][0]) > 1e-6):
                _, node, node_label = options[0]
                junction_ports.setdefault((node, node_label), []).append(event)
            ports.append(event)
    events, consumed = [], set()
    for (node, node_label), local_ports in sorted(junction_ports.items()):
        labels = {s['segmentation_label'] for e in local_ports for s, _ in e['arms']}
        if len({e['path_index'] for e in local_ports}) < 2 or len(labels) < 2:
            continue
        arms = []
        boundary_paths = {event['path_index'] for event in local_ports}
        for event in local_ports:
            consumed.add(id(event))
            arms.extend((s, reverse) for s, reverse in event['arms'] if s['segmentation_label'] != node_label)
        junction_label_arms = []
        for i, reverse in incident[node]:
            spans = sorted(by_path.get(i, []), key=lambda s: s['sample_start'])
            if not spans:
                continue
            span = spans[-1] if reverse else spans[0]
            touches = span['sample_end'] == len(paths[i])-1 if reverse else span['sample_start'] == 0
            if touches and span['segmentation_label'] == node_label:
                junction_label_arms.append((span, reverse))
        # Prefixes leading from this node to a foreign-label boundary are
        # part of the shared junction region, not extra parent-vessel arms.
        # Keep all actual continuing parent arms, and all foreign arms above.
        continuing = [row for row in junction_label_arms if row[0]['path_index'] not in boundary_paths]
        if continuing:
            arms.extend(continuing)
        elif junction_label_arms:
            arms.append(min(junction_label_arms, key=lambda row: (-row[0]['length_mm'], row[0]['path_index'])))
        i, reverse = incident[node][0]
        events.append({'kind': 'junction_region', 'point_mm': paths[i][-1 if reverse else 0].tolist(),
            'arms': arms, 'group_name': node[0], 'node_id': int(node[2]) if node[1]=='node' else None, 'junction_label': node_label, 'component_index': component_of[node],
            'boundary_ports': [p for e in local_ports for p in e['boundary_ports']]})
    events.extend(event for event in ports if id(event) not in consumed)
    serialized = []
    for index, event in enumerate(events):
        # Boundary orientation is part of an arm's identity. It is never
        # silently replaced by a remote physical junction endpoint.
        unique = {(candidate_key(s), bool(reverse)): (s, reverse) for s, reverse in event['arms']}
        labels = sorted({s['segmentation_label'] for s, _ in unique.values()})
        selected = {value: sorted((row for row in unique.values() if row[0]['segmentation_label'] == value),
            key=lambda row: (row[0]['path_index'], row[0]['segment_index'], row[1])) for value in labels}
        event.update(event_id=f'C{index:03d}', labels=labels, cross_label=len(labels) >= 2, selected=selected)
        record = {k: v for k, v in event.items() if k not in ('arms', 'selected')}
        record['selected_arms'] = [{'label': value, 'path_index': s['path_index'], 'segment_index': s['segment_index'],
                                   'length_mm': s['length_mm'], 'near_end': reverse}
                                  for value, rows in selected.items() for s, reverse in rows]
        serialized.append(record)
    return events, serialized, gaps

