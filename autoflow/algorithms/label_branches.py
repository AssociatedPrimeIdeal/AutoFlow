"""Select one label-bounded graph arm; never join arms through a fork."""

from collections import defaultdict, deque

import networkx as nx
import numpy as np

from ..task_control import check_cancelled
from .label_trunks import REFERENCE_LABELS


BRANCH_ANCHORS = {'label_branch', 'label_reference_branch', 'flow_inlet_branch', 'portal_branch'}


class LabelBranches:
    def __init__(self, context):
        self.context = context
        self.cache = {}
        self.audit = []

    def rows(self, value):
        if value in self.cache:
            return self.cache[value]
        ctx, rows = self.context, []
        for (index, label), spans in sorted(ctx.spans.items()):
            if label != value:
                continue
            info = ctx.infos[index] if index < len(ctx.infos) else {}
            hierarchy = info.get('hierarchy', {})
            for span in spans:
                points = np.asarray(ctx.paths[index][span['sample_start']:span['sample_end']+1])
                if len(points) < 2:
                    continue
                ends = tuple(ctx.edges[index][side == 'end'] if span.get('touches_path_'+side) else
                    (str(info.get('group_name','')), 'label_end', index, span['segment_index'], side)
                    for side in ('start','end'))
                radii = ctx.trunk_radii(index, points)
                dl = np.linalg.norm(np.diff(points,axis=0),axis=1)
                support = float(np.sum(dl*(.5*(radii[:-1]+radii[1:])+.5*ctx.trunk_spacing.min())**2))
                rows.append(dict(path_index=index, span=span, points=points, nodes=ends,
                    closed=ends[0]==ends[1], length_mm=float(dl.sum()), support_mm3=support,
                    flow_sign=hierarchy.get('flow_sign'), direction_status=hierarchy.get('direction_status'),
                    group_name=str(info.get('group_name',''))))
        self.cache[value] = rows
        return rows

    @staticmethod
    def identity(row):
        return row['path_index'], row['span']['segment_index']

    def extent(self, value, spans):
        rows = self.rows(value)
        network = nx.Graph()
        by_id = {self.identity(row):row for row in rows}
        for row in rows:
            network.add_edge(*row['nodes'])
        nodes = set()
        for span in spans:
            row = by_id.get((span['path_index'],span['segment_index']))
            if row:
                nodes.update(nx.node_connected_component(network,row['nodes'][0]))
        points = [row['points'] for row in rows if set(row['nodes']).intersection(nodes)]
        return float(np.ptp(np.vstack(points),axis=0).max(initial=0.)) if points else 0.

    def selection(self, row, *, reverse=False, event=None):
        span = dict(row['span'])
        arm = dict(path_index=row['path_index'], span=span, reverse=bool(reverse))
        points = row['points'][::-1] if reverse else row['points']
        result = dict(chain=[arm], anchor_nodes=[], boundary_path_indices=[],
            route_path_indices=[row['path_index']], branch_route_schema='autoflow.label_branch.v1',
            selection_scope='label',
            branch_path_index=row['path_index'], branch_segment_index=span['segment_index'],
            branch_support_mm3=row['support_mm3'], branch_length_mm=row['length_mm'],
            branch_start_point_mm=points[0].tolist(), branch_end_point_mm=points[-1].tolist(),
            branch_flow_sign=row['flow_sign'], branch_closed=bool(row['closed']))
        if event:
            result.update(connection_id=event['event_id'], connection_kind=event['kind'],
                connected_labels=event['labels'],
                anchor_nodes=[dict(group_name=event['group_name'],node_id=event.get('node_id'),
                    point_mm=event['point_mm'])],
                boundary_path_indices=sorted({p['path_index'] for p in event.get('boundary_ports',[])}),
                branch_reference_component_extent_mm=event.get('reference_component_extent_mm'))
        return result

    def targets(self, selection, spec):
        arm = selection['chain'][0]
        points = self.context.paths[arm['path_index']][arm['span']['sample_start']:arm['span']['sample_end']+1]
        points = points[::-1] if arm['reverse'] else points
        arc = np.r_[0.,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
        length = arc[-1]
        distances = [length*f for f in spec['fractions']] if spec['position']=='fractions' else [spec.get('offset_mm',5.)]
        if spec['position']=='before_end':
            distances = [length-spec.get('offset_mm',5.)]
        if spec.get('fraction_reference')=='flow_end':
            distances = [length-d for d in distances]
        if any(d<0 or d>length for d in distances):
            return None
        locations, normals = [], []
        for distance in distances:
            locations.append([np.interp(distance,arc,points[:,axis]) for axis in range(3)])
            lo,hi = max(0.,distance-.5),min(length,distance+.5)
            normal = np.array([np.interp(hi,arc,points[:,axis])-np.interp(lo,arc,points[:,axis]) for axis in range(3)])
            normals.append(normal/max(np.linalg.norm(normal),1e-12))
        return np.asarray(locations),np.asarray(normals)

    def choose(self, choices, spec, reason='no_label_branch'):
        if not choices:
            return [],reason
        policy = spec.get('path_selection','unique')
        if policy=='all':
            return choices,''
        if policy=='unique':
            return (choices,'') if len(choices)==1 else ([], 'ambiguous_label_branch')
        choices.sort(key=lambda row:(-row['branch_support_mm3'],-row['branch_length_mm'],
            tuple(row['branch_start_point_mm']),tuple(row['branch_end_point_mm'])))
        best = choices[0]
        first = self.targets(best,spec)
        for other in choices[1:]:
            if other['branch_support_mm3'] < .9*best['branch_support_mm3']:
                break
            second = self.targets(other,spec)
            if first is None or second is None:
                return [],'ambiguous_label_branch'
            shifts = np.linalg.norm(first[0]-second[0],axis=1)
            angles = np.degrees(np.arccos(np.clip(np.abs(np.sum(first[1]*second[1],axis=1)),0.,1.)))
            radii = self.context.trunk_radii(best['branch_path_index'],first[0])
            if np.any(shifts>np.maximum(2*self.context.trunk_spacing.max(),radii)) or np.any(angles>30.):
                self.audit.append(dict(reason='ambiguous_label_branch',target_shift_mm=float(shifts.max()),
                    normal_angle_deg=float(angles.max())))
                return [],'ambiguous_label_branch'
        return [best],''

    def volume_connections(self, rows, value, expected):
        ctx = self.context
        if ctx.trunk_segmentation is None:
            return [],0
        volume = np.asarray(ctx.trunk_segmentation)
        shape = np.asarray(volume.shape)
        steps = np.vstack((np.eye(3,dtype=int),-np.eye(3,dtype=int)))
        threshold = 2*float(ctx.trunk_spacing.min())
        choices, weak = [],0
        for row in rows:
            for reverse,point in ((False,row['points'][0]),(True,row['points'][-1])):
                voxel = np.rint((point-np.asarray(ctx.trunk_origin))/ctx.trunk_spacing).astype(int)
                if np.any(voxel<0) or np.any(voxel>=shape) or volume[tuple(voxel)]!=value:
                    continue
                for step in steps:
                    neighbor = voxel+step
                    if np.any(neighbor<0) or np.any(neighbor>=shape):
                        continue
                    label = int(volume[tuple(neighbor)])
                    if label not in expected:
                        continue
                    seen = {tuple(neighbor)};queue = deque(seen)
                    lo = neighbor.copy();hi = neighbor.copy();extent = 0.
                    while queue and extent<threshold:
                        check_cancelled()
                        current = np.asarray(queue.popleft())
                        for direction in steps:
                            test = current+direction;key = tuple(test)
                            if key in seen or np.any(test<0) or np.any(test>=shape) or volume[key]!=label:
                                continue
                            seen.add(key);queue.append(key)
                            lo = np.minimum(lo,test);hi = np.maximum(hi,test)
                            extent = float(np.max((hi-lo)*ctx.trunk_spacing))
                            if extent>=threshold:
                                break
                    if extent<threshold:
                        weak += 1
                        continue
                    event = dict(event_id='volume_interface:'+repr(row['nodes'][int(reverse)]),
                        kind='label_volume_interface',labels=sorted((value,label)),point_mm=point.tolist(),
                        group_name=row['group_name'],boundary_ports=[],reference_component_extent_mm=extent)
                    choices.append((row,reverse,event))
        return choices,weak

    def references(self, name, value, spec):
        ctx, rows = self.context,self.rows(value)
        expected = {int(ctx.labels[token]) for token in REFERENCE_LABELS.get(name,()) if token in ctx.labels}
        by_id = {self.identity(row):row for row in rows}
        candidates, weak = [],0
        for source in ctx.observed_connections()[0]:
            foreign = expected.intersection(source['labels'])
            if not foreign or value not in source['selected']:
                continue
            extent = max((self.extent(label,[s for s,_ in source['selected'].get(label,[])]) for label in foreign),default=0.)
            if extent < 2*float(ctx.trunk_spacing.min()):
                weak += 1
                continue
            event = dict(source,reference_component_extent_mm=extent)
            for span,reverse in source['selected'][value]:
                row = by_id.get((span['path_index'],span['segment_index']))
                if row:
                    candidates.append((row,reverse,event))
        volume, volume_weak = self.volume_connections(rows,value,expected)
        candidates.extend(volume);weak += volume_weak
        unique = {}
        for row,reverse,event in candidates:
            key = self.identity(row)+(bool(reverse),)
            priority = (event['kind']=='junction_region',event['kind']!='label_volume_interface',len(event['labels']))
            if key not in unique or priority>unique[key][0]:
                unique[key] = priority,self.selection(row,reverse=reverse,event=event)
        choices = [item[1] for item in unique.values()]
        for selection in choices:
            selection['reference_labels'] = list(REFERENCE_LABELS.get(name,()))
        if spec.get('path_selection')=='all':
            return choices, '' if choices else 'label_reference_too_short' if weak else 'label_reference_not_found'
        return self.choose(choices,spec,'label_reference_too_short' if weak else 'label_reference_not_found')

    def portal(self, spec):
        matches,reason = self.context.match('portal_confluence','PV')
        reason = {'landmark_not_found':'portal_confluence_not_found',
                  'ambiguous_landmark':'ambiguous_portal_confluence'}.get(reason,reason)
        value = int(self.context.labels.get('PV',0))
        by_id = {self.identity(row):row for row in self.rows(value)}
        choices = []
        for match in matches:
            if any(self.extent(int(self.context.labels[name]),[match['arms'][name]['span']]) <
                   2*float(self.context.trunk_spacing.min()) for name in ('SMV','SV')):
                reason = 'label_reference_too_short'
                continue
            arm = match['arms']['PV']
            row = by_id[(arm['path_index'],arm['span']['segment_index'])]
            point = self.context.paths[arm['path_index']][-1 if arm['reverse'] else 0]
            event = dict(event_id='portal_confluence:'+repr(match['nodes']),kind='portal_confluence_branch',
                labels=sorted(int(self.context.labels[name]) for name in ('SMV','SV','PV')),
                group_name=row['group_name'],point_mm=np.asarray(point).tolist(),boundary_ports=[])
            selection = self.selection(row,reverse=arm['reverse'],event=event)
            selection['reference_labels'] = ['SMV','SV']
            selection['matched_arms'] = {name:arm['path_index'] for name,arm in match['arms'].items()}
            choices.append(selection)
        return self.choose(choices,spec,reason or 'portal_confluence_not_found')

    def resolve(self, name, spec):
        self.audit = []
        value = int(self.context.labels.get(name,int(name) if str(name).isdigit() else 0))
        rows = self.rows(value)
        self.audit.append(dict(candidates=[dict(path_index=row['path_index'],segment_index=row['span']['segment_index'],
            length_mm=row['length_mm'],support_mm3=row['support_mm3'],closed=row['closed'],
            flow_sign=row['flow_sign']) for row in rows]))
        anchor = spec['anchor']
        if anchor=='portal_branch':
            choices,reason = self.portal(spec)
        elif anchor=='label_reference_branch':
            choices,reason = self.references(name,value,spec)
        else:
            fractions = np.asarray(spec.get('fractions',[.5]))
            directed = anchor=='flow_inlet_branch' or spec['position'] in {'before_end','after_start'} or bool(spec.get('fraction_reference')) or (
                not np.allclose(np.sort(fractions),np.sort(1-fractions),atol=.015,rtol=0))
            incoming,uncertain = defaultdict(set),defaultdict(set)
            for index,row in enumerate(rows):
                if row['flow_sign'] in (-1,1):
                    incoming[row['nodes'][row['flow_sign']==1]].add(index)
                else:
                    for node in row['nodes']:
                        uncertain[node].add(index)
            choices = []
            for index,row in enumerate(rows):
                if directed and row['flow_sign'] not in (-1,1):
                    continue
                reverse = row['flow_sign']==-1
                upstream = row['nodes'][int(reverse)]
                if anchor=='flow_inlet_branch' and (incoming[upstream] or uncertain[upstream]-{index}):
                    continue
                choices.append(self.selection(row,reverse=reverse))
            missing = ('flow_direction_unavailable' if directed and not any(row['flow_sign'] in (-1,1) for row in rows) else
                'flow_inlet_branch_not_found' if anchor=='flow_inlet_branch' else 'no_label_branch')
            if spec.get('scope','label')=='component':
                by_component = defaultdict(list)
                for selection in choices:
                    index = selection['branch_path_index']
                    by_component[self.context.components[self.context.edges[index][0]]].append(selection)
                chosen,reasons = [],[]
                for options in by_component.values():
                    kept,why = self.choose(options,spec,missing)
                    chosen.extend(kept)
                    if why:
                        reasons.append(why)
                choices,reason = chosen,reasons[0] if reasons else '' if chosen else missing
            else:
                choices,reason = self.choose(choices,spec,missing)
        for selection in choices:
            selection.update(selection_source='single_label_branch',path_selection=spec.get('path_selection','unique'),
                branch_selection_anchor=anchor,selection_scope=spec.get('scope','label'),connection_fallback=False)
        self.audit.append(dict(reason=reason,selected_count=len(choices)))
        return choices,reason
