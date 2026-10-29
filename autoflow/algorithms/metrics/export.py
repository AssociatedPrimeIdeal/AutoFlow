"""Atomic plane pixelwise H5 export and metric-table loading."""

import os
import h5py
import numpy as np
from ...task_control import check_cancelled, report_progress


def save_plane_pixelwise_h5(path, plane_payloads, rr_ms=None, source_format="", *, input_signature=""):
    """Publish a complete file, preserving the previous export on cancellation."""
    import uuid
    import stat
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    temporary = os.path.join(directory, f".autoflow_plane_{uuid.uuid4().hex}.h5")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    os.close(fd)
    try:
        _write_plane_pixelwise_h5(temporary, plane_payloads, rr_ms, source_format, input_signature=input_signature)
        check_cancelled()
        if os.path.isfile(path):
            os.chmod(temporary, stat.S_IMODE(os.stat(path).st_mode))
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def _write_plane_pixelwise_h5(path, plane_payloads, rr_ms=None, source_format="", *, input_signature=""):
    with h5py.File(path, "w") as h5:
        meta = h5.create_group("meta")
        meta.attrs["validity"] = "current"
        meta.attrs["input_signature"] = input_signature
        meta.create_dataset("version", data=np.bytes_("2.0"))
        meta.create_dataset("source_format", data=np.bytes_(str(source_format or "")))
        if rr_ms is not None:
            meta.create_dataset("rr_ms", data=float(rr_ms))
        planes_group = h5.create_group("planes")
        for export_index, payload in enumerate(plane_payloads):
            report_progress({"stage": "plane_pixelwise_export", "current": export_index,
                             "total": len(plane_payloads) if hasattr(plane_payloads, "__len__") else 0,
                             "message": f"Writing pixelwise plane {export_index + 1}"})
            plane_idx = int(payload.get("plane_index", len(planes_group)))
            grp = planes_group.create_group(f"{plane_idx:03d}")
            for key in ("plane_id", "geometry_revision", "roi_revision", "dependency_signature", "validity"):
                if payload.get(key) is not None:
                    grp.attrs[key] = payload[key]
            grp.create_dataset("center_xyz", data=np.asarray(payload.get("center", [0.0, 0.0, 0.0]), dtype=np.float32))
            grp.create_dataset("normal_xyz", data=np.asarray(payload.get("normal", [1.0, 0.0, 0.0]), dtype=np.float32))
            grp.attrs["label"] = int(payload.get("label", 0))
            grp.attrs["path_index"] = int(payload.get("path_index", -1))
            timepoints = payload.get("timepoints", [])
            times_grp = grp.create_group("timepoints")
            for entry in timepoints:
                tidx = int(entry.get("time_index", len(times_grp)))
                tgrp = times_grp.create_group(f"{tidx:03d}")
                for key, value in entry.items():
                    if key == "time_index" or value is None:
                        continue
                    arr = np.asarray(value)
                    if arr.dtype.kind in {"U", "O"}:
                        continue
                    tgrp.create_dataset(key, data=arr, compression="gzip")


def load_metrics_as_table(metrics_json_path, qc_json_path=None):
    import json as _json
    with open(metrics_json_path, "r", encoding="utf-8") as f:
        metrics = _json.load(f)
    scalar_keys = [
        "plane_id", "geometry_revision", "roi_revision", "validity", "stale_reason", "sampling_status",
        "label", "segmentation_label", "path_index", "distance", "target_branch_label",
        "peakv_cm_s", "netflow_mL_beat", "meanv_cm_s", "path_ic", "segmentation_label_ic",
        "path_direction",
        "peakv_forward_cm_s", "peakv_reverse_cm_s",
        "meanv_forward_cm_s", "meanv_reverse_cm_s",
        "netflow_forward_mL_beat", "netflow_reverse_mL_beat",
        "net_netflow_signed_mL_beat", "reflux_fraction",
        "meanv_signed_cm_s",
        "forward_sign", "forward_sign_source",
        "local_path_direction", "normal_tangent_cos",
        "tke_mean_J_m3", "tke_peak_J_m3", "tke_p95_J_m3",
        "pressure_gradient_mag_mean_Pa_m", "pressure_gradient_mag_peak_Pa_m", "pressure_gradient_mag_p95_Pa_m",
        "pressure_gradient_normal_mean_Pa_m", "pressure_gradient_normal_peak_Pa_m", "pressure_gradient_normal_p95_Pa_m",
        "relative_pressure_mean_Pa", "relative_pressure_peak_Pa", "relative_pressure_p95_Pa",
        "wss_wall_mean_Pa", "wss_wall_peak_Pa", "wss_wall_p95_Pa",
    ]
    table_rows = []
    for i, m in enumerate(metrics):
        row = {"plane_index": int(m.get("plane_index", i))}
        for k in scalar_keys:
            if k in m:
                row[k] = m[k]
        row["center_x"] = m["center"][0] if "center" in m else None
        row["center_y"] = m["center"][1] if "center" in m else None
        row["center_z"] = m["center"][2] if "center" in m else None
        Nt = len(m.get("flowrate_mL_s", []))
        for t in range(Nt):
            row[f"flowrate_t{t}"] = m["flowrate_mL_s"][t]
        for t in range(len(m.get("area_mm2", []))):
            row[f"area_t{t}"] = m["area_mm2"][t]
        for key in ("roi_source_frame_t", "roi_carried_forward_t", "roi_interpolated_t",
                    "roi_source_frames_t", "roi_interpolation_weight_t", "roi_resolution_t"):
            for t, value in enumerate(m.get(key, [])):
                row[f"{key}{t}"] = value
        valid = m.get("sample_valid_t", [])
        for t, ok in enumerate(valid):
            row[f"sample_valid_t{t}"] = bool(ok)
            if not ok:
                for key in list(row):
                    if key.endswith(f"_t{t}") and not key.startswith(("roi_", "sample_")):
                        row[key] = None
        for t in range(len(m.get("meanv_cm_s_t", []))):
            row[f"meanv_t{t}"] = m["meanv_cm_s_t"][t]
        for t in range(len(m.get("flowrate_forward_mL_s", []))):
            row[f"flowrate_fwd_t{t}"] = m["flowrate_forward_mL_s"][t]
        for t in range(len(m.get("flowrate_reverse_mL_s", []))):
            row[f"flowrate_rev_t{t}"] = m["flowrate_reverse_mL_s"][t]
        for t in range(len(m.get("meanv_forward_cm_s_t", []))):
            row[f"meanv_fwd_t{t}"] = m["meanv_forward_cm_s_t"][t]
        for t in range(len(m.get("meanv_reverse_cm_s_t", []))):
            row[f"meanv_rev_t{t}"] = m["meanv_reverse_cm_s_t"][t]
        for t in range(len(m.get("flowrate_signed_mL_s", []))):
            row[f"flowrate_signed_t{t}"] = m["flowrate_signed_mL_s"][t]
        for t in range(len(m.get("meanv_signed_cm_s_t", []))):
            row[f"meanv_signed_t{t}"] = m["meanv_signed_cm_s_t"][t]
        fork_ic = m.get("fork_ic", [])
        for fi, fic in enumerate(fork_ic):
            row[f"fork{fi}_id"] = fic.get("fork_id", -1)
            row[f"fork{fi}_role"] = fic.get("role", "")
            row[f"fork{fi}_ic"] = fic.get("ic", 1.0)
        unavailable = m.get("validity") == "stale" or ("sample_valid_t" in m and (not valid or not all(valid)))
        if unavailable:
            geometry = {"label", "segmentation_label", "path_index", "distance", "target_branch_label", "plane_id",
                        "geometry_revision", "roi_revision", "validity", "stale_reason", "sampling_status"}
            for key in scalar_keys:
                if key not in geometry and isinstance(row.get(key), (int, float, np.number)):
                    row[key] = None
        for t, ok in enumerate(valid):
            if not ok or m.get("validity") == "stale":
                for key in list(row):
                    if key.endswith(f"_t{t}") and not key.startswith(("roi_", "sample_")):
                        row[key] = None
        from ...core.plane_state import metric_family
        flags = m.get("field_validity", {})
        row["field_validity"] = _json.dumps(flags, sort_keys=True)
        for key in list(row):
            if flags.get(metric_family(key)) == "stale":
                row[key] = None
        table_rows.append(row)
    qc_data = None
    if qc_json_path is not None:
        with open(qc_json_path, "r", encoding="utf-8") as f:
            qc_data = _json.load(f)
    return table_rows, metrics, qc_data


def refresh_pixelwise_validity(path, workspace):
    """Keep a previous H5 explicit after mutation; revalidate matching undo."""
    from ...core.plane_state import workspace_signature
    if not os.path.isfile(path):
        return False
    with h5py.File(path, "r+") as handle:
        meta = handle.require_group("meta")
        groups = handle.get("planes")
        current = meta.attrs.get("input_signature", "") == workspace_signature(workspace, include_analysis=False)
        current = current and groups is not None and len(groups) == len(workspace.planes)
        if current:
            for index, plane in enumerate(workspace.planes):
                group = groups.get(f"{index:03d}")
                if group is None or any(group.attrs.get(key) != value for key, value in plane.identity_dict().items()):
                    current = False
                    break
        meta.attrs["validity"] = "current" if current else "stale"
        meta.attrs["checked_analysis_revision"] = int(workspace.analysis_revision)
    return bool(current)
