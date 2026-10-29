"""Open only graph cycles with a filled-disk certificate inside the lumen."""

import networkx as nx
import numpy as np
from scipy.ndimage import distance_transform_edt

from ..core.models import GraphData
from ..task_control import check_cancelled
from .graph import graph_to_networkx


def open_redundant_cycles(graph, support_mask, spacing=(1, 1, 1), *, origin=(0, 0, 0)):
    """Remove one existing edge of each certified redundant basis cycle.

    A fan spanning the cycle must fit entirely inside the voxel-solid lumen.
    EDT balls certify triangle containment; an unresolved triangle keeps the
    cycle. This sufficient condition deliberately leaves uncertain cycles.
    It is not a classifier of anatomical versus segmentation-induced rings.
    Points, endpoints, physical components and all other edges are preserved.
    Remaining cycles receive ordinary downstream branch processing.
    """
    points = np.asarray(graph.points, float).reshape(-1, 3)
    step = np.asarray(spacing, float).reshape(3)
    origin = np.asarray(origin, float).reshape(3)
    if not np.all(np.isfinite(step)) or np.any(step <= 0):
        raise ValueError('Cycle-opening spacing must be positive and finite')
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(origin)):
        raise ValueError('Cycle-opening geometry must be finite')
    source = graph_to_networkx(graph)
    key = lambda n: tuple(points[n])
    net = nx.Graph()
    net.add_nodes_from(sorted(source, key=key))
    net.add_edges_from(sorted((tuple(sorted(e, key=key)) for e in source.edges()),
                             key=lambda e: (key(e[0]), key(e[1]))))
    rank = lambda g: g.number_of_edges()-g.number_of_nodes()+nx.number_connected_components(g)
    report = dict(schema='autoflow.lumen_cycle_opening.v1', status='unchanged',
        cycle_rank_before=rank(net), cycle_rank_after=rank(net), removed_edge_count=0,
        decisions=[], component_connectivity_preserved=True, points_unchanged=True)
    if report['cycle_rank_before'] == 0:
        return graph, report
    if support_mask is None:
        report.update(status='skipped', reason='support_mask_unavailable')
        return graph, report
    mask = np.asarray(support_mask, bool)
    if mask.ndim != 3:
        raise ValueError('Cycle opening requires a 3-D support mask')
    # A padded, cropped EDT treats the outside as background. Subtracting half
    # a voxel diagonal bounds distance to the background voxel solid, rather
    # than only to its centers. The bound remains conservative off-grid.
    voxels = np.rint((points-origin)/step).astype(int)
    lo = np.maximum(0, voxels.min(axis=0)-2)
    hi = np.minimum(mask.shape, voxels.max(axis=0)+3)
    if np.any(hi <= lo):
        report.update(status='skipped', reason='graph_outside_support_mask')
        return graph, report
    local = np.pad(mask[tuple(slice(a,b) for a,b in zip(lo,hi))], 1)
    radii = distance_transform_edt(local, sampling=step)
    local_origin = origin+(lo-1)*step
    half_diagonal = .5*float(np.linalg.norm(step))

    def clearance(point):
        index = np.rint((point-local_origin)/step).astype(int)
        if np.any(index < 0) or np.any(index >= local.shape) or not local[tuple(index)]:
            return -1.
        return float(radii[tuple(index)]-half_diagonal-
                     np.linalg.norm(point-(local_origin+index*step)))

    def certificate(curve):
        center = curve.mean(axis=0)
        if clearance(center) <= 0:
            return False
        pending = [(np.array([center,a,b]), 0) for a,b in zip(curve,np.roll(curve,-1,axis=0))]
        checked = 0
        while pending:
            check_cancelled()
            triangle, depth = pending.pop()
            midpoint = triangle.mean(axis=0)
            radius = clearance(midpoint)
            if radius >= np.linalg.norm(triangle-midpoint,axis=1).max()+1e-9:
                continue
            checked += 1
            if radius <= 0 or depth >= 18 or checked >= 10000:
                return False
            pairs = [(0,1),(1,2),(2,0)]
            a,b = max(pairs,key=lambda pair: np.linalg.norm(triangle[pair[0]]-triangle[pair[1]]))
            c = 3-a-b
            middle = .5*(triangle[a]+triangle[b])
            pending.extend([(np.array([triangle[a],middle,triangle[c]]),depth+1),
                            (np.array([middle,triangle[b],triangle[c]]),depth+1)])
        return True

    def canonical(cycle):
        first = min(range(len(cycle)),key=lambda i:key(cycle[i]))
        cycle = cycle[first:]+cycle[:first]
        other = cycle[:1]+cycle[:0:-1]
        return min((cycle,other),key=lambda row:tuple(map(key,row)))

    cycles = [canonical(c) for c in nx.cycle_basis(net)]
    cycles.sort(key=lambda c:(float(np.linalg.norm(np.diff(points[c+[c[0]]],axis=0),axis=1).sum()),
                             tuple(map(key,c))))
    for cycle in cycles:
        edges = list(zip(cycle,cycle[1:]+cycle[:1]))
        if not all(net.has_edge(a,b) for a,b in edges):
            continue
        curve = points[cycle]
        record = dict(points_mm=curve.tolist(), decision='retained', reason='filled_disk_not_certified')
        report['decisions'].append(record)
        if not certificate(curve):
            continue
        # Prefer the least central existing edge; geometry breaks exact ties.
        a,b = min(edges,key=lambda e:(clearance(.5*(points[e[0]]+points[e[1]])),
                                     tuple(sorted((key(e[0]),key(e[1]))))))
        net.remove_edge(a,b)
        record.update(decision='opened', reason='filled_disk_inside_lumen',
                      removed_edge_points_mm=points[[a,b]].tolist())
        report['removed_edge_count'] += 1
    report['cycle_rank_after'] = rank(net)
    if not report['removed_edge_count']:
        return graph, report
    if (nx.number_connected_components(net) != nx.number_connected_components(source)
            or set(net) != set(source)):
        raise RuntimeError('Cycle opening changed physical components or graph nodes')
    report['status'] = 'applied'
    return GraphData(points=points.copy(), edges=np.asarray(sorted(tuple(sorted(e)) for e in net.edges()),
                                                       int).reshape(-1,2)), report
