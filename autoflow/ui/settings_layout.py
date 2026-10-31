"""Basic/Advanced presentation of existing action fields; no value conversion."""

from ..core.models import StepId


def compact_action_sections(action, sections):
    basic_fields = {
        StepId.GENERATE_SKELETON: {'special_handling', 'remove_small_cc'},
        StepId.GENERATE_GRAPH: {'cycle_flow_threshold', 'cycle_preserve_connectivity'},
        StepId.GENERATE_PLANES: {'positions_file', 'import_mode', 'plane_mode', 'label_rules_file', 'label_rules', 'plane_count',
                               'anchor', 'direction', 'spacing_mode', 'spacing_ratio',
                               'cross_section_distance', 'anchor_offset_mm', 'start_distance', 'end_distance'},
        StepId.COMPUTE_DERIVED_METRICS: {'pressure_method'},
        StepId.COMPUTE_PWV: {'groups', 'plane_interval_mm', 'start_distance', 'end_distance',
                            'waveform_key', 'transit_time_method', 'foot_method'},
        StepId.PLANE_STREAMLINES: {'pathline_seed_region', 'pathline_grid_spacing_mm',
                                  'pathline_max_seeds', 'pathline_seed_ratio', 'pathline_seed_info'},
    }
    keys = basic_fields.get(action)
    if keys is None:
        return sections
    basic, advanced = [], []
    for title, fields in sections:
        basic.extend(field for field in fields if field.key.rsplit('.', 1)[-1] in keys)
        remaining = [field for field in fields if field.key.rsplit('.', 1)[-1] not in keys]
        if remaining:
            advanced.append((title, remaining))
    if action in {StepId.COMPUTE_DERIVED_METRICS, StepId.PLANE_STREAMLINES}:
        return [('Basic', basic)] + [('Advanced::' + title, fields) for title, fields in advanced]
    return [('Basic', basic), ('Advanced', [field for _, fields in advanced for field in fields])]
