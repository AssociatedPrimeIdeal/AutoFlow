"""Read-only QC for plane-rule fallback, relocation and pulmonary trunk coverage."""
from .algorithms.placement_warnings import placement_warning_codes


EXPLANATIONS = {
    'plane_mode_fallback': 'The requested by-label plan could not use a label map or numeric label rules; global per-path fixed-step placement was used.',
    'connection_anchor_fallback': 'No observed connection was usable in this component; the configured alternate anchor was used.',
    'short_interval_midpoint': 'The interval is shorter than twice the requested offset; placement moved to its midpoint.',
    'placement_search_shift': 'The nominal position failed geometry/support checks; the plane moved within the same label interval.',
    'conditional_level_fallback': 'The requested neighbouring labels were not connected; the configured alternate label level was used.',
    'multiple_connection_anchors': 'One label rule expanded onto several observed connections; confirm that all locations are intended.',
}


def build_plane_placement_checks(workspace, make_check):
    raw = dict(getattr(getattr(workspace, 'derived', None), 'plane_qc', {}) or {})
    layout = dict(raw.get('plane_layout', raw) or {})
    planes = list(getattr(workspace, 'planes', []) or [])
    events, seen = [], set()

    def append(code, label, path, placement, plane_index=None):
        event = dict(code=code, explanation=EXPLANATIONS.get(code, code), label_name=label,
            path_index=path, plane_index=plane_index, requested_anchor=placement.get('requested_anchor', placement.get('anchor')),
            effective_anchor=placement.get('anchor'), requested_offset_mm=placement.get('requested_offset_mm'),
            effective_distance_in_interval_mm=placement.get('distance_in_interval_mm'),
            nominal_distance_in_interval_mm=placement.get('nominal_distance_in_interval_mm'),
            search_shift_mm=placement.get('search_shift_mm'), label_interval_mm=placement.get('label_interval_mm'),
            label_level=placement.get('label_level'), requested_label_level=placement.get('requested_label_level'),
            missing_connected_labels=placement.get('missing_connected_labels', []),
            connection_id=placement.get('connection_id'), scope='plane' if plane_index is not None else 'label_rule')
        signature = (code, label, path, plane_index, placement.get('placement_index'), placement.get('connection_id'))
        if signature not in seen:
            seen.add(signature);events.append(event)

    for code in layout.get('layout_warnings', []):
        append(code, 'global', None, dict(requested_anchor=layout.get('requested_plane_mode'), anchor=layout.get('plane_mode')))
    label_names = {int(value): str(name) for name, value in workspace.skeleton_params.label_map.items()}
    for index, plane in enumerate(planes):
        placement = dict(getattr(plane, 'placement', {}) or {})
        for code in placement_warning_codes(placement):
            append(code, label_names.get(int(plane.segmentation_label), str(plane.segmentation_label)),
                   int(plane.path_index), placement, index)
    for label in layout.get('labels', []):
        name = str(label.get('label_name', 'unknown'))
        for item in label.get('placements', []):
            if name not in {'LRA','RRA'} and item.get('spec', {}).get('anchor') == 'label_connections' and int(item.get('resolved_count', 0)) > 1:
                append('multiple_connection_anchors', name, None, dict(item.get('spec', {}),
                    placement_index=item.get('placement_index')))
                events[-1]['resolved_anchor_count'] = int(item['resolved_count'])
        # Snapshot/legacy layout provenance is usable even without PlaneData.
        if not planes:
            for selected in label.get('selected_main_paths', []):
                placement = dict(selected.get('placement', {}) or {})
                for code in placement_warning_codes(placement):
                    append(code, name, selected.get('path_index'), placement)
        present_codes = {event['code'] for event in events if event['label_name'] == name}
        for code in label.get('placement_warnings', []):
            if code not in present_codes:
                append(code, name, None, {})
        for warning in label.get('selection_warnings', []):
            append(str(warning), name, None, {})
    errors = list(layout.get('errors', []))
    status = 'fail' if errors else 'warn' if events else 'pass' if layout or planes else 'not_run'
    check = make_check('planes.placement_rules', 'Centerline & Planes', status,
        'Plane placement',
        f'{len(events)} placement warning(s); requested and effective anchors/positions remain inspectable.',
        value=dict(events=events, event_count=len(events)),
        threshold='No silent alternate anchor, midpoint substitution, relocation, conditional level fallback or multiple-anchor expansion.',
        action='Review each affected plane and its requested versus effective rule.' if events else '')
    check['details'] = [make_check(f'planes.placement_rules:{index}', 'Centerline & Planes', 'warn',
        f"{event['label_name']} · {event['code']}",
        f"{event['explanation']} Path={event['path_index']}; plane={event['plane_index']}; "
        f"anchor={event['requested_anchor']} → {event['effective_anchor']}; "
        f"offset={event['requested_offset_mm']} mm; effective distance={event['effective_distance_in_interval_mm']} mm; "
        f"shift={event['search_shift_mm']} mm.", value=event, action=check['action']) for index, event in enumerate(events)]
    if errors:
        check['summary'] = 'Unavailable plane positions: ' + '; '.join(e.get('label_name','global') + ': ' + ', '.join(e.get('reasons',{})) for e in errors)
        check['value'] = dict(errors=errors, historical_events=events)
        check['action'] = 'Inspect plane_qc.json and correct the missing candidate, orientation, interval length or image support. No replacement position was used.'
        error_details = []
        for error in errors:
            label = str(error.get('label_name', 'global'))
            reasons = dict(error.get('reasons', {}))
            detail = make_check('planes.placement_rules:error:' + label, 'Centerline & Planes', 'fail',
                label + ' · Unavailable plane positions',
                '; '.join(str(reason) + ' (' + str(count) + ')' for reason, count in reasons.items()),
                value=error, action=check['action'])
            error_details.append(detail)
        check['details'] = error_details + check['details']

    checks = [check]

    mpa = next((row for row in layout.get('labels', []) if row.get('label_name') == 'MPA'), None)
    if not mpa or mpa.get('status') in ('disabled', 'not_configured', 'label_absent'):
        return checks
    specs = mpa.get('rule', {}).get('placements', [])
    if specs and all(spec.get('anchor') != 'label_connections' for spec in specs):
        return checks
    mpa_value = int(mpa.get('segmentation_label') or workspace.skeleton_params.label_map.get('MPA', 0))
    hierarchy = (getattr(workspace, 'graph_hierarchy', {}) or {}).get('paths', [])
    pure = [row for row in hierarchy if row.get('group_name') == 'pulmonary_arteries' and
            set(row.get('label_names', [])) == {'MPA'}]
    inlet = [row for row in pure if row.get('is_inlet')]
    pool = inlet if len(inlet) == 1 else pure
    pool = sorted(pool, key=lambda row: (-float(row.get('length_mm', 0.)), int(row['path_index'])))
    candidates = pool[:1] if len(pool) < 2 or float(pool[0]['length_mm']) > float(pool[1]['length_mm']) + 1e-6 else pool[:2]
    candidate_ids = [int(row['path_index']) for row in candidates]
    if planes:
        actual = [int(plane.path_index) for plane in planes if int(plane.segmentation_label) == mpa_value]
    else:
        actual = [int(row['path_index']) for row in mpa.get('selected_main_paths', []) if row.get('actual_plane_count')]
    covered = len(candidate_ids) == 1 and candidate_ids[0] in actual
    check = make_check('planes.pulmonary_trunk', 'Centerline & Planes', 'pass' if covered else 'warn',
        'MPA trunk plane coverage',
        f'MPA trunk candidate paths={candidate_ids}; actual MPA plane paths={actual}; '
        + ('the trunk candidate has a plane.' if covered else 'the trunk candidate is unmeasured or ambiguous; inspect the connection-based rule.'),
        value=dict(candidate_path_indices=candidate_ids, actual_mpa_plane_path_indices=actual,
            candidate_lengths_mm=[row.get('length_mm') for row in candidates],
            candidate_method='unique pure-MPA inlet, otherwise unique longest pure-MPA path',
            interpretation='A review heuristic, not an anatomical ground truth or an automatic repositioning rule.'),
        threshold='The identified MPA trunk candidate has an MPA plane; ambiguous candidates require review.',
        action='Inspect the MPA trunk and bifurcation; an observed label boundary can lie on a daughter arm instead of the trunk.' if not covered else '')
    checks.append(check)
    return checks
