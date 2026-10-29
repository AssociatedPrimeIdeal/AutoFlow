"""Read-only, self-contained HTML reports from one result snapshot."""

import base64
import copy
from datetime import datetime, timezone
import hashlib
import html
import io
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from .core.plane_state import metric_validity, selected_metric, workspace_signature
from .quality import _json_value, build_quality_report
from .qc_locations import iter_issues


def _report_json_value(value):
    """JSON-safe report values, including NumPy bools in mixed Python lists."""
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return _report_json_value(value.tolist())
    if isinstance(value, dict):
        return {str(key): _report_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_report_json_value(item) for item in value]
    return _json_value(value)


def build_report_snapshot(workspace, *, images=(), selected_plane_ids=None, quality_report=None,
                          include_spatial=True, spatial_limits=None):
    """Copy bounded evidence from one revision without recomputing analysis."""
    from .report_spatial import build_spatial_context, unavailable_spatial_context
    from .task_control import TaskCancelled, check_cancelled

    check_cancelled()
    signature = workspace_signature(workspace)
    if quality_report is None:
        quality = build_quality_report(workspace)
    else:
        if not isinstance(quality_report, dict):
            raise TypeError("quality_report must be a signed quality-report mapping")
        quality = quality_report
    if quality.get("dependency_signature") != signature:
        raise ValueError("Quality report belongs to a different result revision")
    quality = copy.deepcopy(quality)
    selected = set(selected_plane_ids) if selected_plane_ids is not None else None
    planes = []
    for index, plane in enumerate(workspace.planes):
        check_cancelled()
        if selected is not None and plane.plane_id not in selected:
            continue
        state = metric_validity(plane, selected_metric(workspace, index, current_only=False))
        metric = selected_metric(workspace, index) if state == "current" else None
        if metric:
            metric = copy.deepcopy(metric)
            raw_valid = metric.get("sample_valid_t")
            valid = np.asarray(raw_valid, dtype=bool).reshape(-1) if raw_valid is not None else np.array([], dtype=bool)
            if valid.size and not np.all(valid):
                # Invalid samples must not be displayed as measured zero.
                for key, value in list(metric.items()):
                    if key != "sample_valid_t" and isinstance(value, (list, np.ndarray)) and np.ndim(value) > 0 and len(value) == len(valid) and key.endswith(("_t", "_s", "_mm2")):
                        metric[key] = [item if ok else None for item, ok in zip(value, valid)]
                for key in ("netflow_mL_beat", "net_netflow_signed_mL_beat", "peakv_cm_s", "meanv_cm_s", "reflux_fraction"):
                    metric[key] = None
        planes.append({"index": index, **plane.identity_dict(), "center_mm": np.asarray(plane.center).tolist(),
                       "normal": np.asarray(plane.normal).tolist(), "path_index": plane.path_index,
                       "segmentation_label": int(getattr(plane, "segmentation_label", 0) or 0),
                       "group": plane.group_name, "validity": state, "metrics": metric,
                       "roi_operations": copy.deepcopy(plane.roi_edit_operations)})
    payload = {"schema": "autoflow.report.v1", "template_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(), "dependency_signature": signature,
        "analysis_revision": workspace.analysis_revision, "source": str(workspace.paths.flow_path),
        "source_metadata": copy.deepcopy(workspace.input_state.metadata),
        "segmentation_source": workspace.segmentation.active_source,
        "segmentation_provenance": workspace.get_active_segmentation_provenance(),
        "parameters": {key: value.to_dict() for key, value in (
            ("unwrap", workspace.phase_unwrap_params), ("planes", workspace.plane_gen_params),
            ("derived", workspace.derived_params), ("pwv", workspace.pwv_params))},
        "spacing_mm": np.asarray(workspace.resolution).tolist(), "origin_mm": np.asarray(workspace.origin).tolist(),
        "rr_ms": float(workspace.rr), "phase_count": workspace.time_count(),
        "artifact_signatures": dict(workspace.derived.artifact_signatures),
        "planes": planes, "quality": quality, "pwv": copy.deepcopy(workspace.derived.pwv_results),
        "images": copy.deepcopy(list(images))}
    from .report_media import metric_family_status, report_video_records
    payload["metric_output_status"] = metric_family_status(workspace)
    payload["videos"] = report_video_records(workspace)
    if include_spatial:
        try:
            payload["spatial_context"] = build_spatial_context(
                workspace, quality, selected_plane_ids=selected, limits=spatial_limits)
        except TaskCancelled:
            raise
        except Exception as exc:
            payload["spatial_context"] = unavailable_spatial_context(
                "Spatial evidence could not be prepared: " + str(exc), error_type=type(exc).__name__)
    payload["parameters_sha256"] = hashlib.sha256(json.dumps(payload["parameters"], sort_keys=True).encode()).hexdigest()
    check_cancelled()
    if signature != workspace_signature(workspace):
        raise RuntimeError("Workspace changed while creating report; export again.")
    for image in payload["images"]:
        if image.get("dependency_signature") != signature:
            raise ValueError("Report image belongs to a different result revision")
        raw = base64.b64decode(image.get("data_base64", ""), validate=True)
        if image.get("mime") != "image/png" or not raw.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("Report evidence must be an embedded PNG")
    payload = _report_json_value(payload)
    payload["snapshot_id"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return payload



def _curve_svg(metric, rr):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_svg import FigureCanvasSVG
    values = metric.get("flowrate_signed_mL_s", metric.get("flowrate_mL_s", []))
    if not values:
        return "<p>Flow curve —</p>"
    values = np.asarray([np.nan if value is None else value for value in values], dtype=float)
    valid = np.asarray(metric.get("sample_valid_t", np.ones(len(values), bool)), dtype=bool)
    values[~valid] = np.nan
    figure = Figure(figsize=(7, 2.5), layout="constrained")
    canvas = FigureCanvasSVG(figure)
    axes = figure.add_subplot(111)
    axes.plot(np.arange(len(values))*float(rr)/max(1, len(values)), values, marker="o", markersize=3)
    axes.axhline(0., color="gray", linewidth=.5)
    axes.set(xlabel="Time (ms)", ylabel="Signed flow (mL/s)")
    axes.grid(alpha=.2)
    buffer = io.StringIO()
    canvas.print_svg(buffer)
    svg = buffer.getvalue()
    return svg[svg.index("<svg"):]


def render_report_html(snapshot, *, sections=None, video_urls=None):
    """Compatibility entry point for the standalone interactive renderer."""
    from .report_html import render_report_html as render
    return render(snapshot, sections=sections, video_urls=video_urls)


def save_report_html(snapshot, path, *, sections=None):
    """Single-writer atomic output, including on Windows with no open handles."""
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=".autoflow-report-", suffix=".html", dir=target.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            from .report_media import prepare_report_media
            videos = prepare_report_media(snapshot, path)
            stream.write(render_report_html(snapshot, sections=sections, video_urls=videos))
        from .task_control import check_cancelled
        check_cancelled()
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return str(target)


