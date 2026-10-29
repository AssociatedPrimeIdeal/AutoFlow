"""Cycle summaries pinned beside a plane's selected measurement curve."""

import numpy as np

from ..quality import _peak_flow_rate


DEFAULT_PINNED_METRICS = ("mean_velocity", "peak_velocity", "peak_flow", "net_flow")

# Stable preference key, display label, unit, existing metric field.
PINNED_METRICS = (
    ("mean_velocity", "Mean velocity", "cm/s", "meanv_cm_s"),
    ("peak_velocity", "Peak velocity", "cm/s", "peakv_cm_s"),
    ("peak_flow", "Peak flow", "mL/s", None),
    ("net_flow", "Net flow", "mL/beat", "netflow_mL_beat"),
    ("mean_area", "Mean area", "mm²", "area_mm2"),
    ("forward_flow", "Forward flow", "mL/beat", "netflow_forward_mL_beat"),
    ("reverse_flow", "Reverse flow", "mL/beat", "netflow_reverse_mL_beat"),
    ("signed_net_flow", "Signed net flow", "mL/beat", "net_netflow_signed_mL_beat"),
    ("reflux_fraction", "Reflux fraction", "%", "reflux_fraction"),
    ("mean_wss", "Mean WSS", "Pa", "wss_wall_mean_Pa"),
    ("peak_wss", "Peak WSS", "Pa", "wss_wall_peak_Pa_t"),
    ("mean_tke", "Mean TKE", "J/m³", "tke_mean_J_m3"),
    ("peak_tke", "Peak TKE", "J/m³", "tke_peak_J_m3_t"),
    ("mean_pg", "Mean pressure gradient", "Pa/m", "pressure_gradient_mag_mean_Pa_m"),
    ("peak_pg", "Peak pressure gradient", "Pa/m", "pressure_gradient_mag_peak_Pa_m_t"),
    ("mean_relative_pressure", "Mean relative pressure", "Pa", "relative_pressure_mean_Pa"),
    ("peak_relative_pressure", "Peak |relative pressure|", "Pa", "relative_pressure_peak_Pa_t"),
)


def pinned_metric_keys(workspace):
    cfg = (getattr(workspace, "render_settings", {}) or {}).get("measurements", {})
    value = cfg.get("pinned_metrics", DEFAULT_PINNED_METRICS) if isinstance(cfg, dict) else DEFAULT_PINNED_METRICS
    if not isinstance(value, (list, tuple)):
        value = DEFAULT_PINNED_METRICS
    allowed = {item[0] for item in PINNED_METRICS}
    return list(dict.fromkeys(key for key in value if isinstance(key, str) and key in allowed))


def pinned_metric_rows(metric, keys):
    metric = metric or {}
    valid = metric.get("sample_valid_t")
    complete = metric.get("sampling_status") not in {"partial", "not_assessable"}
    if valid is not None:
        complete = complete and bool(np.all(np.asarray(valid, dtype=bool)))
    catalog = {item[0]: item[1:] for item in PINNED_METRICS}
    rows = []
    for key in keys:
        if key not in catalog:
            continue
        name, unit, field = catalog[key]
        value = None
        if complete:
            if key == "peak_flow":
                value = _peak_flow_rate(metric)
            elif field in metric and metric[field] is not None:
                try:
                    values = np.asarray(metric[field], dtype=float).reshape(-1)
                    if values.size and np.all(np.isfinite(values)):
                        value = float(np.max(np.abs(values))) if key.startswith("peak_") else float(np.mean(values))
                        if key == "reflux_fraction":
                            value *= 100.0
                except (TypeError, ValueError):
                    pass
        text = f"{float(value):.4g}" if value is not None and np.isfinite(value) else "—"
        rows.append((name, text, unit))
    return rows
