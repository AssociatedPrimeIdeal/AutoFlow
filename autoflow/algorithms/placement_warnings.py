"""Describe deviations from requested plane placement without moving a plane."""


def placement_warning_codes(placement):
    placement = dict(placement or {})
    warnings = []
    if placement.get('connection_fallback'):
        warnings.append('connection_anchor_fallback')
    if placement.get('short_interval_midpoint'):
        warnings.append('short_interval_midpoint')
    if abs(float(placement.get('search_shift_mm') or 0.)) > 1e-8:
        warnings.append('placement_search_shift')
    if placement.get('conditional_level_fallback'):
        warnings.append('conditional_level_fallback')
    return warnings
