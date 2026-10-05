"""Background-correction cache validation/publication and loader progress."""

import numpy as np
from ..phase_correction import _emit_progress, background_phase_correction_cache_metadata, coerce_background_phase_correction_config

from .h5_metadata import (
    _canonical_h5_key,
    _find_h5_dataset,
    _h5_group_source_name,
    _h5_member_name_map,
)


def _progress_prefix(progress_callback, prefix):
    if progress_callback is None:
        return None

    def _wrapped(payload):
        data = dict(payload or {})
        stage = str(data.get("stage", ""))
        data["stage"] = f"{prefix}{stage}" if prefix else stage
        if prefix in {"h5_dual_lv_", "h5_dual_hv_"}:
            label = "Low VENC" if prefix == "h5_dual_lv_" else "High VENC"
            data["message"] = f"{label}: {data.get('message', '')}"
        progress_callback(data)

    return _wrapped


def _loader_correction_config(correction_config):
    if correction_config is None:
        return coerce_background_phase_correction_config({"enabled": False})
    return coerce_background_phase_correction_config(correction_config)

def _background_phase_corr_attr_scalar(attrs, name, default=None):
    if attrs is None:
        return default
    value = None
    try:
        value = attrs.get(name)
    except Exception:
        value = None
    if value is None:
        return default
    arr = np.asarray(value).reshape(-1)
    if arr.size == 0:
        return default
    item = arr[0]
    if isinstance(item, (bytes, np.bytes_)):
        return item.decode("utf-8", errors="ignore")
    if isinstance(default, (bool, np.bool_)):
        return bool(item)
    if isinstance(default, (int, np.integer)) and not isinstance(default, bool):
        return int(item)
    if isinstance(default, (float, np.floating)):
        return float(item)
    return item


def _read_background_phase_corr_cache(scope, cache_name, expected_shape, cfg, expected_source_group=None, allow_untagged_root=True, match_config=True):
    report = {
        "cache_hit": False,
        "cache_name": str(cache_name),
        "cache_reason": "missing",
    }
    ds = _find_h5_dataset(scope, cache_name)
    if ds is None:
        return None, report

    corr = np.asarray(ds[:], dtype=np.float32)
    if corr.ndim == len(expected_shape) - 1 and corr.shape[-1] == 3 and len(expected_shape) == corr.ndim + 1:
        corr = corr[..., np.newaxis, :]
    corr_shape = tuple(corr.shape)
    expected_shape = tuple(expected_shape)
    singleton_time_match = (
        len(corr_shape) == 5
        and len(expected_shape) == 5
        and corr_shape[:3] == expected_shape[:3]
        and corr_shape[3] == 1
        and corr_shape[4] == expected_shape[4]
    )
    if corr_shape != expected_shape and not singleton_time_match:
        report["cache_reason"] = f"shape_mismatch:{tuple(corr.shape)}"
        return None, report

    stored_group = _background_phase_corr_attr_scalar(ds.attrs, "corr_source_group", None)
    if expected_source_group is not None and stored_group is not None:
        if str(stored_group).strip("/") != str(expected_source_group).strip("/"):
            report["cache_reason"] = f"group_mismatch:{stored_group}"
            return None, report
    if expected_source_group is not None and stored_group is None:
        scope_group = _h5_group_source_name(scope)
        if scope_group is None and not allow_untagged_root:
            report["cache_reason"] = "missing_group_tag"
            return None, report

    if not match_config:
        if expected_source_group is None and stored_group is not None and str(stored_group).strip("/"):
            report["cache_reason"] = f"group_mismatch:{stored_group}"
            return None, report
        # Loading a saved result checks geometry/ownership, not the settings
        # selected for the next manual computation.
        if not np.all(np.isfinite(corr)):
            report["cache_reason"] = "nonfinite_correction"
            return None, report
        for name in ds.attrs:
            if str(name).startswith("corr_"):
                value = _background_phase_corr_attr_scalar(ds.attrs, name)
                if isinstance(value, (str, int, float, bool, np.generic)):
                    report[str(name)] = value.item() if isinstance(value, np.generic) else value
        report.update(cache_hit=True, cache_reason="hit", corr_components=int(corr.shape[-1]))
        report.setdefault("corr_algorithm", "msac")
        report.setdefault("corr_version", 1)
        if "corr_stationary_voxels" in report:
            report["stationary_voxels"] = int(report["corr_stationary_voxels"])
        return corr, report

    expected_metadata = background_phase_correction_cache_metadata(cfg)
    expected_algorithm = str(expected_metadata["corr_algorithm"])
    algorithm = _background_phase_corr_attr_scalar(ds.attrs, "corr_algorithm", "")
    if algorithm in ("", None):
        if expected_algorithm != "msac":
            report["cache_reason"] = "algorithm_mismatch:untagged"
            return None, report
        algorithm = "msac"
    elif str(algorithm).lower() != expected_algorithm:
        report["cache_reason"] = f"algorithm_mismatch:{algorithm}"
        return None, report

    version = _background_phase_corr_attr_scalar(ds.attrs, "corr_version", 1)
    if version is not None and int(version) != int(expected_metadata["corr_version"]):
        report["cache_reason"] = f"version_mismatch:{version}"
        return None, report

    fit_order = _background_phase_corr_attr_scalar(ds.attrs, "corr_fit_order", None)
    if fit_order is not None and int(fit_order) != int(cfg.corr_fit_order):
        report["cache_reason"] = f"fit_order_mismatch:{fit_order}"
        return None, report

    algorithm_metadata = {
        key: value
        for key, value in expected_metadata.items()
        if key not in {"corr_algorithm", "corr_version", "corr_fit_order"}
    }
    stored_algorithm_metadata = {}
    for key, expected_value in algorithm_metadata.items():
        stored_value = _background_phase_corr_attr_scalar(ds.attrs, key, None)
        if stored_value is None:
            if expected_algorithm == "msac" and key == "corr_threshold":
                stored_value = expected_value
            else:
                report["cache_reason"] = f"parameter_missing:{key}"
                return None, report
        if isinstance(expected_value, (float, np.floating)):
            matches = np.isclose(float(stored_value), float(expected_value))
        elif isinstance(expected_value, (int, np.integer)):
            matches = int(stored_value) == int(expected_value)
        else:
            matches = str(stored_value) == str(expected_value)
        if not bool(matches):
            report["cache_reason"] = f"parameter_mismatch:{key}={stored_value}"
            return None, report
        stored_algorithm_metadata[key] = expected_value

    report.update({
        "cache_hit": True,
        "cache_reason": "hit",
        "corr_algorithm": expected_algorithm,
        "corr_version": int(version) if version is not None else int(expected_metadata["corr_version"]),
        "corr_fit_order": int(fit_order) if fit_order is not None else int(cfg.corr_fit_order),
        "corr_components": int(corr.shape[-1]),
    })
    report.update(stored_algorithm_metadata)
    if stored_group is not None:
        report["corr_source_group"] = str(stored_group)
    source_mode = _background_phase_corr_attr_scalar(ds.attrs, "corr_source_mode", None)
    if source_mode is not None:
        report["corr_source_mode"] = str(source_mode)
    stationary_voxels = _background_phase_corr_attr_scalar(ds.attrs, "corr_stationary_voxels", None)
    if stationary_voxels is not None:
        report["stationary_voxels"] = int(stationary_voxels)
    return corr, report


def _read_background_phase_corr_cache_from_scopes(scopes, cache_name, expected_shape, cfg, expected_source_group=None, allow_untagged_root=True, match_config=True):
    if match_config and bool(getattr(cfg, "force_recompute", False)):
        return None, {"cache_hit": False, "cache_reason": "force_recompute", "cache_name": str(cache_name)}
    last_report = None
    for scope in scopes:
        corr, report = _read_background_phase_corr_cache(
            scope,
            cache_name,
            expected_shape,
            cfg,
            expected_source_group=expected_source_group,
            allow_untagged_root=allow_untagged_root,
            match_config=match_config,
        )
        if (
            last_report is not None
            and report.get("cache_reason") == "missing"
            and last_report.get("cache_reason") not in (None, "missing")
        ):
            report = dict(report)
            report["cache_reason"] = last_report.get("cache_reason")
        last_report = report
        if bool(report.get("cache_hit", False)):
            return corr, report
    return None, last_report or {"cache_hit": False, "cache_reason": "missing", "cache_name": str(cache_name)}


def _prepare_background_phase_corr_cache(scopes, cache_name, expected_shape, cfg,
                                       expected_source_group=None, allow_untagged_root=True,
                                       reuse_existing_corr=False, progress_callback=None):
    """Prepare a normal correction or a read-only application of saved results."""
    if not cfg.enabled and not reuse_existing_corr:
        return cfg, None, {"cache_hit": False, "cache_reason": "hit", "cache_name": cache_name}
    if reuse_existing_corr:
        _emit_progress(progress_callback, "background_phase_cache_read",
                       message=f"Loading saved background correction: {cache_name}")
    corr, report = _read_background_phase_corr_cache_from_scopes(
        scopes, cache_name, expected_shape, cfg,
        expected_source_group=expected_source_group,
        allow_untagged_root=allow_untagged_root,
        match_config=not reuse_existing_corr,
    )
    if reuse_existing_corr:
        cfg = coerce_background_phase_correction_config(cfg)
        cfg.enabled = corr is not None
        cfg.force_recompute = False
        cfg.write_cache = False
        if corr is None:
            _emit_progress(progress_callback, "background_phase_cache_skip",
                           message=f"Saved background correction unavailable: {cache_name} ({report['cache_reason']})")
    return cfg, corr, report


def _merge_background_phase_cache_report(report, cache_report, reuse_existing_corr=False):
    if reuse_existing_corr:
        report["cache_only"] = True
        if report.get("cache_hit"):
            # Do not attribute saved corrections to the current UI parameters.
            for key in list(report):
                if key.startswith("corr_") and key not in {"corr_source", "corr_components"}:
                    report.pop(key)
            report.pop("threshold", None)
            report.update(cache_report)
    if not report.get("cache_hit"):
        report["cache_reason"] = cache_report.get("cache_reason", "missing")
    if "stationary_voxels" in cache_report and report.get("cache_hit"):
        report["stationary_voxels"] = int(cache_report["stationary_voxels"])


def _write_background_phase_corr_cache(
    scope,
    cache_name,
    report,
    expected_source_group=None,
    progress_callback=None,
):
    corr = report.get("corr") if isinstance(report, dict) else None
    if corr is None:
        return False
    corr_arr = np.asarray(corr, dtype=np.float32)
    if corr_arr.ndim == 4 and corr_arr.shape[-1] == 3:
        corr_arr = corr_arr[..., np.newaxis, :]
    if corr_arr.ndim != 5 or corr_arr.shape[-1] != 3:
        return False
    try:
        if progress_callback is not None:
            progress_callback({
                "stage": "background_phase_cache_write",
                "message": f"Writing background phase cache: {cache_name}",
            })
        existing_name = _h5_member_name_map(scope).get(_canonical_h5_key(cache_name))
        if existing_name is not None:
            del scope[existing_name]
        ds = scope.create_dataset(str(cache_name), data=corr_arr, compression="gzip")
        ds.attrs["corr_algorithm"] = str(report.get("corr_algorithm", "msac"))
        ds.attrs["corr_version"] = int(report.get("corr_version", 1))
        ds.attrs["corr_fit_order"] = int(report.get("corr_fit_order", 3))
        for key, value in report.items():
            if not str(key).startswith("corr_") or key in {
                "corr_algorithm",
                "corr_version",
                "corr_fit_order",
                "corr_components",
                "corr_source",
            }:
                continue
            if isinstance(value, (str, int, float, bool, np.integer, np.floating, np.bool_)):
                ds.attrs[str(key)] = value
        ds.attrs["corr_components"] = int(report.get("corr_components", corr_arr.shape[-1]))
        ds.attrs["corr_source_mode"] = str(report.get("source_mode", ""))
        ds.attrs["corr_cache_hit"] = int(bool(report.get("cache_hit", False)))
        if expected_source_group is None:
            expected_source_group = _h5_group_source_name(scope)
        if expected_source_group is not None:
            ds.attrs["corr_source_group"] = str(expected_source_group)
        if report.get("stationary_voxels") is not None:
            ds.attrs["corr_stationary_voxels"] = int(report.get("stationary_voxels"))
    except Exception:
        return False
    if progress_callback is not None:
        progress_callback({
            "stage": "background_phase_cache_done",
            "current": 1,
            "total": 1,
            "message": f"Background phase cache saved: {cache_name}",
        })
    return True
