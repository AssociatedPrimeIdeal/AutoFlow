"""Plane measurements and explicit reasons for unavailable samples."""

import numpy as np
from ..core.plane_state import metric_family, metric_validity

_PLANE_CURVE_SERIES_OPTIONS = [
    ("flowrate_mL_s", "Flowrate (mL/s)"),
    ("flowrate_forward_mL_s", "Forward Flowrate (mL/s)"),
    ("flowrate_reverse_mL_s", "Reverse Flowrate (mL/s)"),
    ("flowrate_signed_mL_s", "Signed Flowrate (mL/s)"),
    ("area_mm2", "Area (mm^2)"),
    ("meanv_cm_s_t", "Mean Velocity (cm/s)"),
    ("meanv_forward_cm_s_t", "Forward Mean Velocity (cm/s)"),
    ("meanv_reverse_cm_s_t", "Reverse Mean Velocity (cm/s)"),
    ("meanv_signed_cm_s_t", "Signed Mean Velocity (cm/s)"),
    ("peakv_cm_s_t", "Peak Velocity (cm/s)"),
    ("peakv_forward_cm_s_t", "Forward Peak Velocity (cm/s)"),
    ("peakv_reverse_cm_s_t", "Reverse Peak Velocity (cm/s)"),
    ("tke_mean_J_m3_t", "TKE Mean (J/m^3)"),
    ("tke_peak_J_m3_t", "TKE Peak (J/m^3)"),
    ("tke_p95_J_m3_t", "TKE P95 (J/m^3)"),
    ("pressure_gradient_mag_mean_Pa_m_t", "Pressure Gradient Mean (Pa/m)"),
    ("pressure_gradient_mag_peak_Pa_m_t", "Pressure Gradient Peak (Pa/m)"),
    ("pressure_gradient_mag_p95_Pa_m_t", "Pressure Gradient P95 (Pa/m)"),
    ("pressure_gradient_normal_mean_Pa_m_t", "Pressure Gradient Normal Mean (Pa/m)"),
    ("pressure_gradient_normal_peak_Pa_m_t", "Pressure Gradient Normal Peak (Pa/m)"),
    ("pressure_gradient_normal_p95_Pa_m_t", "Pressure Gradient Normal P95 (Pa/m)"),
    ("relative_pressure_mean_Pa_t", "Relative Pressure Mean (Pa)"),
    ("relative_pressure_peak_Pa_t", "Relative Pressure Peak (Pa)"),
    ("relative_pressure_p95_Pa_t", "Relative Pressure P95 (Pa)"),
    ("wss_wall_mean_Pa_t", "WSS Mean (Pa)"),
    ("wss_wall_peak_Pa_t", "WSS Peak (Pa)"),
    ("wss_wall_p95_Pa_t", "WSS P95 (Pa)"),
]


PLANE_CYCLE_OPTIONS = [
    ('netflow_mL_beat', 'Net Flow Magnitude (mL/beat)'),
    ('netflow_forward_mL_beat', 'Forward Volume (mL/beat)'),
    ('netflow_reverse_mL_beat', 'Reverse Volume (mL/beat)'),
    ('net_netflow_signed_mL_beat', 'Signed Net Volume (mL/beat)'),
    ('reflux_fraction', 'Reflux Fraction'),
]


def measurement_options(metric):
    options = list(_PLANE_CURVE_SERIES_OPTIONS)
    options.extend((key, label) for key, label in PLANE_CYCLE_OPTIONS if key in (metric or {}))
    return options


def measurement_unavailable_reason(workspace, plane_index, metric, key):
    if not 0 <= int(plane_index) < len(workspace.planes):
        return 'Select a plane in the Browser or 3D View.'
    plane = workspace.planes[int(plane_index)]
    original = plane.metrics or metric or {}
    if not metric or (original and metric_validity(plane, original) != 'current'):
        return 'Apply the plane/ROI draft, then calculate plane metrics.'
    family = metric_family(key)
    stale = workspace.input_state.metadata.get('stale_metric_families', [])
    if family and (family in stale or original.get('field_validity', {}).get(family) == 'stale'):
        return f'{family.upper()} results are stale. Run Hemodynamics → Advanced Metrics, then update plane metrics.'
    value = metric.get(key)
    if value is None:
        fields = {'pressure': workspace.derived.pressure_gradient_array if key.startswith('pressure_gradient_')
                  else workspace.derived.relative_pressure_array,
                  'wss': workspace.derived.wss_surfaces, 'tke': workspace.derived.tke_array}
        field = fields.get(family)
        exists = bool(len(field)) if isinstance(field, (list, tuple)) else field is not None
        if family and not exists:
            return f'{family.upper()} is unavailable. Run Hemodynamics → Advanced Metrics for this family if supported by the input.'
        if family:
            return f'Volume {family.upper()} exists; Apply ROI or run plane metrics to sample it on this plane.'
        return 'Calculate plane metrics for this plane.'
    values = np.asarray(value, dtype=float).reshape(-1)
    if values.size and np.any(np.isfinite(values)):
        return ''
    if family == 'pressure':
        counts = metric.get('relative_pressure_valid_cell_count_t' if key.startswith('relative_pressure_')
                            else 'pressure_gradient_valid_cell_count_t', [])
        if len(counts) and not np.any(np.asarray(counts) > 0):
            return 'The plane ROI has no valid pressure support. Check the contour and the pressure support region.'
    return 'This plane has no finite samples for this measurement. Check the ROI in each phase.'
