import copy
import json
import os
import hashlib
import numpy as np

from .models import StepId, ObjectKind, GraphData, DerivedResults
from ..rendering.style import wss_scalar_range, configured_metric_style, render_style_settings
from ..algorithms import (
    load_input_data,
    filter_segmask_labels, binarize_segmask, merge_segmask_to_3d,
    preprocess_mask_for_skeleton,
    separate_special_label_contacts,
    generate_skeleton_from_mask3d, generate_three_pass_special_skeleton, build_graph_from_points,
    remove_short_terminal_branches,
    segment_vessels_from_graph_and_mask,
    generate_planes_from_paths,
    compute_plane_metrics, compute_derived_metrics,
    compute_plane_metrics_multithread,
    filter_planes_by_branch_support,
    augment_plane_metrics_with_derived, save_plane_pixelwise_h5,
    generate_seed_points,
    compute_pwv_groups,
    save_pwv_results,
    segmentation_timestamp,
)
from ..algorithms.phase_unwrapping import unwrap_phase, resolve_mask_source
from ..algorithms.noise_removal import pcmra_render_mask
from ..algorithms.flow_seeding import volume_seed_domain
from ..algorithms.flow_display import pathline_color_style
from ..algorithms.geometry_grid import measurement_segmentation, QUANTITATIVE_POLICY
from ..case_types import InputCase
from ..task_control import TaskCancelled, check_cancelled, report_progress
from ..plane_io import (build_plane_records, save_planes_h5, save_planes_json, save_pwv_h5,
                        load_plane_position_payload, project_planes_to_workspace, save_plane_positions)
from ..algorithms.label_planes import generate_label_planes
from ..algorithms.hierarchy import build_workspace_hierarchy
from ..algorithms.graph_cycles import remove_intralabel_cycles
from ..algorithms.graph_junctions import simplify_junction_clusters


_DERIVED_ALGORITHM_VERSIONS = {
    "wss": "wss-native-segmentation-native-velocity-v7",
    "tke": "tke-masked-v2",
    "pressure": "pressure-profile-support-v3",
    "vortex": "vortex-finite-stencil-v2",
}


def _array_fingerprint(value):
    if value is None:
        return None
    arr = np.asarray(value)
    digest = hashlib.blake2b(digest_size=16)
    digest.update(str(arr.dtype).encode("ascii"))
    digest.update(json.dumps(list(arr.shape)).encode("ascii"))
    if arr.size:
        contiguous = np.ascontiguousarray(arr)
        digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def _payload_fingerprint(payload):
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.blake2b(encoded, digest_size=16).hexdigest()


class StepResult:
    def __init__(self, step, success=True, skipped=False, message="", outputs=None):
        self.step = step
        self.success = success
        self.skipped = skipped
        self.message = message
        self.outputs = outputs or []


class PipelineEngine:
    def _repeat_mask_to_time(self, mask_3d, time_count):
        mask = np.asarray(mask_3d, dtype=bool)
        nt = max(1, int(time_count))
        return np.repeat(mask[..., np.newaxis], nt, axis=3).astype(bool)

    def _group_data_key(self, prefix, group_name, index=None):
        if index is None:
            return f"{prefix}_{group_name}"
        return f"{prefix}_{group_name}_{int(index)}"

    def _group_object_name(self, prefix, group_name, index=None):
        return self._group_data_key(prefix, group_name, index=index)

    def _special_label_values(self, ws, group_params, group_label_values):
        """Resolve configured special-label names to values present in a group."""
        label_map = dict(getattr(getattr(ws, "skeleton_params", None), "label_map", {}) or {})
        if not label_map:
            label_map = dict(getattr(getattr(ws, "label_params", None), "label_map", {}) or {})
        group_values = {int(value) for value in (group_label_values or [])}
        values = []
        for token in list(getattr(group_params, "special_contact_labels", []) or []):
            value = label_map.get(str(token).strip())
            if value is None:
                try:
                    value = int(token)
                except (TypeError, ValueError):
                    continue
            value = int(value)
            if value in group_values and value not in values:
                values.append(value)
        return values


    def _indexed_object_name(self, prefix, index):
        label_map = {
            "plane": "plane",
            "smooth_path": "path",
            "pathline": "pathline",
        }
        label = str(label_map.get(prefix, prefix)).replace("_", " ")
        return f"{label} {int(index)}"

    def _plane_group_name(self, ws, plane_idx):
        if not (0 <= int(plane_idx) < len(ws.planes)):
            return ""
        return str(getattr(ws.planes[int(plane_idx)], "group_name", "") or "")

    def _plane_data_key(self, prefix, ws, plane_idx):
        group_name = self._plane_group_name(ws, plane_idx)
        if group_name:
            return self._group_data_key(prefix, group_name, plane_idx)
        return f"{prefix}_{int(plane_idx)}"

    def _plane_object_name(self, prefix, ws, plane_idx):
        group_name = self._plane_group_name(ws, plane_idx)
        if group_name:
            return self._group_object_name(prefix, group_name, plane_idx)
        return f"{prefix}_{int(plane_idx)}"

    def _clear_pwv_state(self, ws):
        ws.derived.pwv_results = []
        ws.derived.pwv_planes = []
        ws.derived.pwv_file = ""
        ws.derived.pwv_json_file = ""
        ws.derived.pwv_h5_file = ""
        ws.remove_object_by_data_key("pwv_planes")


    def _clear_plane_state(self, ws, *, preserve_paths=False):
        """Discard measurements and scene objects tied to the old plane layout."""
        self._clear_pwv_state(ws)
        ws.planes = []
        if not preserve_paths:
            ws.centerline_paths_smooth = []
        ws.clear_pathlines()
        ws.pathline_colors = {}
        ws.derived.plane_metrics = []
        ws.derived.plane_qc = {}
        ws.derived.plane_pixelwise_file = ""
        prefixes = ("plane_",) if preserve_paths else ("plane_", "smooth_path_", "path_arrow_")
        for prefix in prefixes:
            ws.remove_objects_by_prefix(prefix)
        for group_state in ws.multilabel_groups.values():
            group_state["planes"] = []
            if not preserve_paths:
                group_state["centerline_paths_smooth"] = []
            group_state["plane_index_offset"] = 0
        for step in (StepId.GENERATE_PLANES, StepId.EDIT_PLANES,
                     StepId.COMPUTE_PLANE_METRICS, StepId.COMPUTE_PWV,
                     StepId.COMPUTE_DERIVED_METRICS, StepId.PLANE_STREAMLINES):
            ws.pipeline.completed.pop(step.value, None)
            ws.pipeline.skipped.pop(step.value, None)

    def _clear_graph_state(self, ws):
        """Discard the topology and plane layout derived from the old skeleton."""
        self._clear_plane_state(ws)
        ws.graph = GraphData()
        ws.graph_hierarchy = {}
        ws.remove_objects_by_prefix("hierarchy_path_")
        ws.branch_labels = None
        ws.centerline_paths = []
        ws.centerline_node_paths = []
        ws.path_info = []
        ws.forks = []
        ws.selected_path_index = -1
        ws.derived.centerline_pressure_profiles = []
        for key in ("graph_lines", "fork_markers", "branch_surface", "willis_ring_graph"):
            ws.remove_object_by_data_key(key)
        for prefix in ("graph_", "forks_", "path_"):
            ws.remove_objects_by_prefix(prefix)
        for group_state in ws.multilabel_groups.values():
            group_state["graph"] = GraphData()
            group_state["branch_labels"] = None
            for key in ("centerline_paths", "centerline_node_paths", "path_info", "forks"):
                group_state[key] = []
            group_state["path_index_offset"] = 0
        for step in (StepId.GENERATE_GRAPH, StepId.EDIT_GRAPH):
            ws.pipeline.completed.pop(step.value, None)
            ws.pipeline.skipped.pop(step.value, None)

    def _register_pwv_scene_object(self, ws):
        ws.remove_object_by_data_key("pwv_planes")
        if not list(ws.derived.pwv_planes or []):
            return
        ws.add_object(
            name="PWV planes",
            kind=ObjectKind.AUX,
            data_key="pwv_planes",
            group_name="PWV",
            browser_color=str(ws.pwv_params.scene_color or "#ffd43b"),
            visible=bool(ws.pwv_params.scene_visible),
            opacity=0.45,
            color=str(ws.pwv_params.scene_color or "#ffd43b"),
            line_width=2,
        )

    def _compute_pwv_internal(self, ws, save=True):
        self._clear_pwv_state(ws)
        if not bool(getattr(ws.pwv_params, "enabled", False)):
            return [], "PWV skipped: disabled"
        if ws.segmask_raw is None:
            return [], self._missing_segmentation_message(ws, "PWV")
        if not ws.has_flow():
            return [], "PWV skipped: no flow"
        groups = list(getattr(ws.pwv_params, "groups", []) or [])
        if not groups:
            return [], "PWV skipped: no groups configured"
        out_dir = self._output_dir(ws) if save else ""
        results, scene_planes = compute_pwv_groups(
            ws.flow_raw,
            ws.segmask_raw,
            ws.resolution,
            ws.origin,
            ws.rr,
            ws.skeleton_params,
            ws.pwv_params,
            out_dir=out_dir,
        )
        ws.derived.pwv_results = list(results or [])
        ws.derived.pwv_planes = list(scene_planes or [])
        if save:
            source_path = str(ws.paths.flow_path or ws.paths.segmask_path or "")
            ws.derived.pwv_json_file = save_pwv_results(ws.derived.pwv_results, os.path.join(out_dir, "pwv.json"))
            ws.derived.pwv_h5_file = save_pwv_h5(
                ws.derived.pwv_results,
                os.path.join(out_dir, "pwv.h5"),
                source_path=source_path,
                source_format=ws.input_state.source_format,
                source_group=ws.input_state.source_group,
            )
            ws.derived.pwv_file = ws.derived.pwv_json_file
        self._register_pwv_scene_object(ws)
        ok_count = sum(1 for item in ws.derived.pwv_results if str(item.get("status", "")) == "ok")
        total = len(ws.derived.pwv_results)
        msg = f"PWV: {ok_count}/{total} groups"
        saved_parts = []
        if ws.derived.pwv_json_file:
            saved_parts.append(f"json={ws.derived.pwv_json_file}")
        if ws.derived.pwv_h5_file:
            saved_parts.append(f"h5={ws.derived.pwv_h5_file}")
        if saved_parts:
            msg += " saved=" + " ".join(saved_parts)
        return ws.derived.pwv_results, msg

    def _label_name_for_value(self, ws, label_value):
        for name, value in dict(ws.skeleton_params.label_map or {}).items():
            try:
                if int(value) == int(label_value):
                    return str(name)
            except Exception:
                continue
        return f"label_{int(label_value)}"

    def _build_group_specs(self, ws, labels_3d):
        positive = sorted(int(x) for x in np.unique(labels_3d) if int(x) != 0)
        if not positive:
            return []
        if len(positive) == 1:
            group_name = str(ws.skeleton_params.single_label_group_name or "single_label")
            return [{
                "name": group_name,
                "labels": [int(positive[0])],
                "browser_color": ws.skeleton_params.browser_color_for_group(group_name),
                "scene_color": ws.skeleton_params.scene_color_for_group(group_name),
            }]
        specs = []
        used = set()
        configured = dict(ws.skeleton_params.label_groups or {})
        for group_name, cfg in configured.items():
            labels = [int(x) for x in list(cfg.get("labels", [])) if int(x) in positive]
            if not labels:
                continue
            specs.append({
                "name": str(group_name),
                "labels": sorted(set(labels)),
            })
            used.update(labels)
        leftovers = [int(x) for x in positive if int(x) not in used]
        for label_value in leftovers:
            name = self._label_name_for_value(ws, label_value)
            if any(spec["name"] == name for spec in specs):
                name = f"{name}_{int(label_value)}"
            specs.append({"name": name, "labels": [int(label_value)]})
        for spec in specs:
            spec["browser_color"] = ws.skeleton_params.browser_color_for_group(spec["name"])
            spec["scene_color"] = ws.skeleton_params.scene_color_for_group(spec["name"])
        return specs

    def _register_derived_scene_objects(self, ws):
        for dk in [
            "wss_surface_live",
            "tke_volume",
            "pressure_gradient_volume",
            "relative_pressure_volume",
            "vorticity_magnitude_volume",
            "q_criterion_volume",
            "swirling_strength_volume",
        ]:
            ws.remove_object_by_data_key(dk)
        from ..rendering.style import rounded_display_clim
        render_cfg = dict(getattr(ws, "render_settings", {}) or {})
        styles = {key: configured_metric_style(ws, key) for key in (
            "wss_surface_live", "tke_volume", "pressure_gradient_volume", "relative_pressure_volume")}
        tke_max = float(np.nanmax(ws.derived.tke_array)) if ws.derived.tke_array is not None and np.size(ws.derived.tke_array) else 0.0
        wss_clim = render_cfg.get("wss_clim") or wss_scalar_range(ws.derived.wss_surfaces)
        tke_clim = render_cfg.get("tke_clim") or (0.0, tke_max if tke_max > 0 else 1.0)
        pressure_gradient_clim = render_cfg.get("pressure_gradient_clim")
        if pressure_gradient_clim is None:
            pressure_gradient_clim = rounded_display_clim(tuple(ws.derived.pressure_gradient_display_clim) if ws.derived.pressure_gradient_display_clim is not None else (0.0, 1.0))
        relative_pressure_clim = render_cfg.get("relative_pressure_clim")
        if relative_pressure_clim is None:
            relative_pressure_clim = rounded_display_clim(tuple(ws.derived.relative_pressure_display_clim) if ws.derived.relative_pressure_display_clim is not None else (-1.0, 1.0))
        ws.add_object(name="wss_surface", kind=ObjectKind.METRIC,
                      data_key="wss_surface_live", visible=False, opacity=styles["wss_surface_live"]["opacity"],
                      scalars="wss", cmap=styles["wss_surface_live"]["cmap"], clim=tuple(wss_clim), dynamic=True,
                      show_scalar_bar=bool(render_cfg.get("wss_show_scalar_bar", True)), scalar_bar_title="WSS (Pa)",
                      scalar_bar_cfg=dict(render_cfg.get("wss_bar_cfg", {}) or {}))
        has_tke_local = ws.derived.tke_array is not None or ws.derived.tke_volume is not None
        if has_tke_local:
            ws.add_object(name="tke_volume", kind=ObjectKind.METRIC,
                          data_key="tke_volume", visible=False, opacity=styles["tke_volume"]["opacity"],
                          scalars="TKE", cmap=styles["tke_volume"]["cmap"], clim=tuple(tke_clim), dynamic=True,
                          show_scalar_bar=bool(render_cfg.get("tke_show_scalar_bar", True)), scalar_bar_title="TKE (J/m³)",
                          scalar_bar_cfg=dict(render_cfg.get("tke_bar_cfg", {}) or {}))
        if ws.derived.pressure_gradient_magnitude is not None:
            ws.add_object(name="pressure_gradient_volume", kind=ObjectKind.METRIC,
                          data_key="pressure_gradient_volume", visible=False, opacity=float(np.clip(ws.derived_params.pressure_gradient_layer_opacity, 0.0, 1.0)),
                          scalars="PressureGradient", cmap=styles["pressure_gradient_volume"]["cmap"], clim=pressure_gradient_clim, dynamic=True,
                          show_scalar_bar=bool(render_cfg.get("pressure_gradient_show_scalar_bar", True)), scalar_bar_title="|Pressure Grad| (Pa/m)",
                          scalar_bar_cfg=dict(render_cfg.get("pressure_gradient_bar_cfg", {}) or {}))
        if ws.derived.relative_pressure_array is not None:
            ws.add_object(name="relative_pressure_volume", kind=ObjectKind.METRIC,
                          data_key="relative_pressure_volume", visible=False, opacity=float(np.clip(ws.derived_params.relative_pressure_layer_opacity, 0.0, 1.0)),
                          scalars="RelativePressure", cmap=styles["relative_pressure_volume"]["cmap"], clim=relative_pressure_clim, dynamic=True,
                          show_scalar_bar=bool(render_cfg.get("relative_pressure_show_scalar_bar", render_cfg.get("pressure_gradient_show_scalar_bar", True))), scalar_bar_title="Relative Pressure (Pa)",
                          scalar_bar_cfg=dict(render_cfg.get("relative_pressure_bar_cfg", render_cfg.get("pressure_gradient_bar_cfg", {}) or {}) or {}))

        vortex_specs = (
            ("vorticity_magnitude_volume", "Vorticity Magnitude", "vorticity_magnitude", "Vorticity Magnitude (s⁻¹)", "turbo", False),
            ("q_criterion_volume", "Q-Criterion", "q_criterion_array", "Q-Criterion (s⁻²)", "RdBu_r", True),
            ("swirling_strength_volume", "Swirling Strength", "swirling_strength_array", "Swirling Strength λci (s⁻¹)", "turbo", False),
        )
        vortex_support = ws.derived.vortex_support_mask
        for data_key, name, field_name, scalar_bar_title, cmap, signed in vortex_specs:
            values = getattr(ws.derived, field_name, None)
            if values is None or vortex_support is None or np.size(values) == 0:
                continue
            values = np.asarray(values, dtype=float)
            support = np.asarray(vortex_support, dtype=bool)
            finite = values[support & np.isfinite(values)] if support.shape == values.shape else values[np.isfinite(values)]
            upper = float(np.percentile(np.abs(finite), 99.0)) if finite.size else 1.0
            upper = max(upper, 1e-6)
            style = configured_metric_style(ws, data_key)
            clim = style.get("clim") or rounded_display_clim((-upper, upper) if signed else (0.0, upper))
            ws.add_object(
                name=name,
                kind=ObjectKind.METRIC,
                data_key=data_key,
                visible=False,
                opacity=style["opacity"],
                scalars=name,
                cmap=style["cmap"],
                clim=clim,
                dynamic=True,
                show_scalar_bar=bool(style.get("show_scalar_bar", True)),
                scalar_bar_title=scalar_bar_title,
                scalar_bar_cfg=dict(render_cfg.get("shared_colorbar_bar_cfg", {}) or {}),
            )


    def _missing_segmentation_message(self, ws, action):
        if not ws.input_state.capabilities.has_segmentation:
            return f"{action} skipped: input has no segmentation"
        return f"{action} skipped: no segmentation available"

    def _output_dir(self, ws):
        out_dir = getattr(ws.paths, "output_dir", "") or ""
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            return out_dir
        base = ws.paths.segmask_path or ws.paths.flow_path or "."
        out_dir = os.path.dirname(base) or "."
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def _json_safe(self, obj):
        if isinstance(obj, np.floating):
            val = float(obj)
            return val if np.isfinite(val) else None
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, float):
            return obj if np.isfinite(obj) else None
        if isinstance(obj, dict):
            return {k: self._json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._json_safe(v) for v in obj]
        return obj

    def load_data(self, ws, log, input_source=None, progress_callback=None, reuse_existing_corr=False,
                  apply_background_correction=False, parameter_overrides=None):
        path = ws.paths.segmask_path or ws.paths.flow_path
        if input_source is None and not path:
            raise ValueError("data path is empty")
        load_target = path if input_source is None else input_source
        load_kwargs = {
            "correction_config": dict(ws.loader_params.background_phase_correction.to_dict(), enabled=bool(apply_background_correction)),
            "reuse_existing_corr": bool(reuse_existing_corr),
            "force_recompute_seg": bool(getattr(ws.segmentation, "force_recompute_auto_cache", False)),
            "ignore_embedded_segmentation": bool(
                getattr(ws.loader_params, "ignore_embedded_segmentation", False)
            ),
        }
        if apply_background_correction:
            load_kwargs["preserve_source_sigma"] = True
        if parameter_overrides:
            load_kwargs["parameter_overrides"] = copy.deepcopy(parameter_overrides)
        load_kwargs["dicom_backend"] = getattr(ws.loader_params, "dicom_backend", "dicom2h5")
        load_kwargs["dicom_h5_dir"] = getattr(ws.loader_params, "dicom_h5_dir", "")
        source_metadata = getattr(load_target, "metadata", {}) if not isinstance(load_target, (str, bytes, os.PathLike)) else {}
        if isinstance(source_metadata, dict) and source_metadata.get("dual_venc_mode"):
            load_kwargs["dual_venc_mode"] = str(source_metadata["dual_venc_mode"])
        if progress_callback is not None:
            load_kwargs["progress_callback"] = progress_callback
        try:
            data = load_input_data(load_target, **load_kwargs)
        except TypeError as exc:
            if parameter_overrides or "unexpected keyword argument" not in str(exc):
                raise
            data = load_input_data(load_target)
        # Reruns and saved workspaces must reread the preserved H5, rather than
        # trying to convert the original directory into an existing destination.
        converted_h5 = data.metadata.get("converted_h5")
        if converted_h5:
            load_target = InputCase(
                str(converted_h5), "h5", source_group=data.source_group,
                metadata=dict(data.metadata),
            )
            ws.paths.flow_path = ws.paths.segmask_path = str(converted_h5)
        if apply_background_correction:
            report = dict(data.metadata.get("background_phase_correction", {}) or {})
            parts = [report.get("dual_venc_low", {}), report.get("dual_venc_high", {})] if "dual_venc_low" in report else [report]
            if not all(part.get("applied", False) for part in parts):
                reason = "; ".join(str(part.get("skipped_reason", "not applied")) for part in parts)
                raise RuntimeError(f"Background correction not applied: {reason}")
        ws._loaded_input_source = copy.deepcopy(load_target)
        ws._loaded_input_kwargs = dict(load_kwargs)
        ws.pcmra_render_mask = None
        ws.pcmra_array = None
        ws.noise_removal_result = {}
        for step in (StepId.BACKGROUND_CORRECTION, StepId.REMOVE_NOISE, StepId.UNWRAP_PHASE, StepId.GENERATE_PCMRA):
            ws.pipeline.completed.pop(step.value, None)
            ws.pipeline.skipped.pop(step.value, None)
        flow = np.asarray(data.flow, dtype=np.float32)
        mag = np.asarray(data.mag, dtype=np.float32)
        seg = None if data.segmentation is None else np.asarray(data.segmentation, dtype=np.int16)

        # Loading a new case clears the active source, but must retain the
        # segmentation configuration supplied by the caller.  In particular,
        # read-only benchmarks rely on ``write_auto_cache=False`` and the
        # force/ignore flags must survive this reset as well.
        segmentation_settings = {
            name: getattr(ws.segmentation, name)
            for name in (
                "visible",
                "opacity",
                "active_label",
                "label_names",
                "label_colors",
                "editing_enabled",
                "tool",
                "brush_radius",
                "edit_all_timepoints",
                "mode",
                "input_source",
                "import_path",
                "threshold_scalar",
                "threshold_value",
                "threshold_keep_largest_cc",
                "threshold_min_component_volume_mm3",
                "threshold_closing",
                "threshold_opening",
                "force_recompute_auto_cache",
                "write_auto_cache",
                "auto_backend",
                "auto_model",
                "auto_checkpoint",
                "auto_folds",
                "auto_device",
                "auto_label_map",
                "cleanup_4d_components",
                "cleanup_4d_mode",
                "cleanup_4d_min_volume_mm3",
            )
            if hasattr(ws.segmentation, name)
        }
        ws.segmentation = ws.segmentation.__class__()
        for name, value in segmentation_settings.items():
            setattr(ws.segmentation, name, value)
        ws.segmask_raw = None
        if seg is not None:
            ws.set_segmentation_source(
                "original",
                seg,
                provenance={
                    "source": "original",
                    "source_format": str(data.source_format or ""),
                    "source_group": data.source_group,
                    "created_at": segmentation_timestamp(),
                    "metadata": dict(data.metadata or {}),
                },
            )
            ws.activate_segmentation_source("original")
        ws.resolution = np.asarray(data.resolution, dtype=float).reshape(3)
        ws.origin = np.asarray(data.origin, dtype=float).reshape(3)
        ws.spatial_order = [str(x) for x in dict(data.metadata or {}).get("spatial_order_raw", ["LR", "AP", "FH"])]
        ws.venc_order = [str(x) for x in dict(data.metadata or {}).get("venc_order_raw", ["LR", "AP", "FH"])]
        ws.venc = np.asarray(data.venc, dtype=float).reshape(-1)
        ws.rr = float(data.rr)
        ws.current_t = 0
        ws.flow_raw = flow
        ws.flow_input = np.array(flow, copy=True)
        phase_loaded = getattr(data, "phase_wrapped", None)
        phase_reconstructed = False
        if phase_loaded is None and str(data.source_format or "").lower() != "legacy_h5_dual_venc":
            # Normalized H5 and direct DICOM loaders expose velocity rather
            # than raw phase.  The encoded velocity is periodic at ±VENC, so
            # reconstruct the wrapped phase representation needed by the
            # optional unwrap stage.  Keep the original velocity untouched.
            venc_arr = np.asarray(data.venc, dtype=np.float32).reshape(1, 1, 1, 1, -1)
            from ..algorithms.pointwise import wrapped_phase_from_flow
            phase_loaded = wrapped_phase_from_flow(flow, venc_arr)
            phase_reconstructed = True
        ws.phase_wrapped = None if phase_loaded is None else np.asarray(phase_loaded, dtype=np.float32)
        ws.phase_wrapped_high = None if getattr(data, "phase_wrapped_high", None) is None else np.asarray(data.phase_wrapped_high, dtype=np.float32)
        ws.phase_unwrap_result = {}
        ws.mag_raw = mag
        ws.input_state.source_format = str(data.source_format or "")
        ws.input_state.source_group = data.source_group
        ws.input_state.metadata = dict(data.metadata or {})
        if phase_reconstructed:
            ws.input_state.metadata["phase_wrapped_reconstructed_from_velocity"] = True
        ws.input_state.capabilities = data.capabilities
        if ws.phase_wrapped is not None:
            ws.input_state.capabilities.has_wrapped_phase = True
            ws.input_state.capabilities.supports_phase_unwrap = True
        ws.source_sigma = None if data.sigma is None else np.asarray(data.sigma, dtype=np.float32)
        ws.correction_raw = None if getattr(data, "correction", None) is None else np.asarray(data.correction, dtype=np.float32)
        ws.correction_high_raw = None if getattr(data, "correction_high", None) is None else np.asarray(data.correction_high, dtype=np.float32)
        ws.source_tke_array = None if data.tke_array is None else np.asarray(data.tke_array, dtype=np.float32)
        ws.derived.tke_array = None
        ws.derived.tke_volume = None
        ws.derived.pressure_gradient_array = None
        ws.derived.pressure_gradient_magnitude = None
        ws.derived.pressure_gradient_peak = None
        ws.derived.pressure_gradient_dt_s = None
        ws.derived.pressure_gradient_temporal_scheme = ""
        ws.derived.pressure_gradient_support_mask = None
        ws.derived.pressure_gradient_display_clim = None
        ws.derived.relative_pressure_array = None
        ws.derived.relative_pressure_peak = None
        ws.derived.relative_pressure_display_clim = None
        ws.derived.centerline_pressure_profiles = []
        ws.derived.vorticity_array = None
        ws.derived.vorticity_magnitude = None
        ws.derived.vorticity_magnitude_peak = None
        ws.derived.q_criterion_array = None
        ws.derived.q_criterion_peak = None
        ws.derived.swirling_strength_array = None
        ws.derived.swirling_strength_peak = None
        ws.derived.vortex_support_mask = None
        ws.derived.wss_surfaces = []
        ws.derived.wss_volume = None
        ws.derived.pixelwise_export = {}
        ws.derived.plane_pixelwise_file = ""
        ws.derived.pwv_results = []
        ws.derived.pwv_planes = []
        ws.derived.pwv_file = ""
        ws.derived.pwv_json_file = ""
        ws.derived.pwv_h5_file = ""
        ws.derived.artifact_signatures = {}
        ws.data_loaded = True
        if apply_background_correction:
            ws.loader_params.background_phase_correction.enabled = True
            ws.pipeline.mark_done(StepId.BACKGROUND_CORRECTION)
        ws._preprocess_signature = None

        for data_key in [
            "segmask_raw_surface", "segmask_pre_surface", "wss_surface_live", "tke_volume",
            "pressure_gradient_volume", "relative_pressure_volume", "vorticity_magnitude_volume",
            "q_criterion_volume", "swirling_strength_volume", "phase_wrap_mask", "phase_wrap_count",
            "pcmra_volume", "noise_region",
        ]:
            ws.remove_object_by_data_key(data_key)
        ws.remove_object_by_data_key("pwv_planes")
        if ws.segmask_raw is not None:
            ws.add_object(name="segmask_raw", kind=ObjectKind.SEGMENTATION,
                          data_key="segmask_raw_surface", visible=True, opacity=0.3,
                          scalars="label", cmap="tab10", dynamic=True,
                          show_scalar_bar=True, scalar_bar_title="Label")

        seg_desc = "none"
        if ws.segmask_raw is not None:
            seg_desc = f"{ws.segmask_raw.shape} labels={ws.unique_labels()}"
        msg = f"Loaded: segmask={seg_desc} rr={ws.rr}"
        msg += f" flow={ws.flow_raw.shape} mag={ws.mag_raw.shape}"
        msg += f" origin={ws.origin.tolist()}"
        msg += f" resolution={np.asarray(ws.resolution, dtype=float).reshape(-1)[:3].tolist()}"
        msg += f" venc={np.asarray(ws.venc, dtype=float).reshape(-1)[:3].tolist()}"
        msg += f" spatial_order={list(ws.spatial_order[:3])}"
        msg += f" venc_order={list(ws.venc_order[:3])}"
        corr_meta = ws.input_state.metadata.get("background_phase_correction", {})
        if isinstance(corr_meta, dict):
            if corr_meta.get("applied"):
                msg += " bpc=applied"
            elif corr_meta.get("enabled"):
                reason = str(corr_meta.get("skipped_reason", "") or "skipped")
                msg += f" bpc={reason}"
        if bool(ws.input_state.metadata.get("phase_wrapped_reconstructed_from_velocity", False)):
            msg += " wrapped_phase=reconstructed_from_velocity"
        msg += f" caps={ws.input_state.capabilities.to_dict()}"
        log(msg)
        return msg

    def preprocess(self, ws):
        if ws.segmask_raw is None:
            raise ValueError("segmask_raw is None")
        from ..algorithms.preprocess import majority_vote_labels_3d, filter_connected_components, filter_4d_labeled_components
        from ..algorithms.preprocess_qc import build_segmentation_preprocess_qc
        from ..algorithms.geometry_grid import prepare_segmentation_geometry, segmentation_geometry_data, labels_on_flow_grid
        geometry_labels, geometry_spacing, geometry_info = prepare_segmentation_geometry(ws)
        geometry_temporal = segmentation_geometry_data(ws)
        ws.skeleton_params.geometry_spacing_mm = 1.

        signature = (
            (id(ws.segmask_raw), int(getattr(getattr(ws, "segmentation", None), "revision", 0))),
            tuple(np.asarray(ws.segmask_raw).shape),
            tuple(float(x) for x in np.asarray(ws.resolution, dtype=float).reshape(-1)[:3]),
            int(ws.time_count()),
            json.dumps(ws.skeleton_params.to_dict(), sort_keys=True, default=str),
            bool(getattr(getattr(ws, "segmentation", None), "cleanup_4d_components", False)),
            str(getattr(getattr(ws, "segmentation", None), "cleanup_4d_mode", "absolute")),
            float(getattr(getattr(ws, "segmentation", None), "cleanup_4d_min_volume_mm3", 50.0)),
            QUANTITATIVE_POLICY,
        )
        if (
            getattr(ws, "_preprocess_signature", None) == signature
            and ws.segmask_binary is not None
            and ws.segmask_3d is not None
        ):
            return False

        ws._video_union_surface_cache = {}
        ws._pressure_support_surface_cache = {}
        previous_groups = dict(ws.multilabel_groups or {})
        geometry_filtered = filter_segmask_labels(geometry_temporal)
        if bool(getattr(getattr(ws, "segmentation", None), "cleanup_4d_components", False)) and np.asarray(geometry_filtered).ndim == 4:
            geometry_filtered = filter_4d_labeled_components(
                geometry_filtered,
                geometry_spacing,
                mode=str(getattr(getattr(ws, "segmentation", None), "cleanup_4d_mode", "absolute")),
                min_volume_mm3=float(getattr(getattr(ws, "segmentation", None), "cleanup_4d_min_volume_mm3", 50.0)),
            )
        voted_labels_3d = (majority_vote_labels_3d(geometry_filtered)
                           if ws.segmentation.cleanup_4d_components else geometry_labels)
        ws._geometry_vote_cache = (signature[0], voted_labels_3d)
        # Retain components after merging labels into their anatomical group.
        # Small labels attached to the retained group must survive unchanged.
        retained_labels_3d = np.zeros_like(voted_labels_3d, dtype=np.int16)
        specs = self._build_group_specs(ws, voted_labels_3d)
        ws.group_order = [str(spec["name"]) for spec in specs]
        ws.multilabel_groups = {}
        time_count = max(1, int(ws.time_count()))
        binary_shape = voted_labels_3d.shape + (time_count,)
        global_binary = np.zeros(binary_shape, dtype=bool)
        global_mask_3d = np.zeros(voted_labels_3d.shape, dtype=bool)
        morphology_groups = {}
        for spec_index, spec in enumerate(specs):
            report_progress({"stage": "segmentation_preprocess", "current": spec_index, "total": len(specs),
                             "message": f"Preparing segmentation group {spec_index + 1}/{len(specs)}"})
            group_name = str(spec["name"])
            labels = [int(x) for x in spec["labels"]]
            group_mask_3d = np.isin(voted_labels_3d, labels)
            group_params = ws.skeleton_params.params_for_group(group_name)
            protected_values = self._special_label_values(ws, group_params, labels)
            special_handling = str(getattr(group_params, "special_handling", "three_pass_merge") or "three_pass_merge").strip().lower().replace("-", "_")
            contact_cut = None
            if special_handling == "contact_surface" and getattr(group_params, "separate_special_label_contacts", True) and len(protected_values) >= 2:
                contact_cut = lambda mask: separate_special_label_contacts(
                    mask, voted_labels_3d, protected_values, resolution=geometry_spacing)
            processed_mask_3d, preprocess_details = preprocess_mask_for_skeleton(
                group_mask_3d, group_params, resolution=geometry_spacing, return_details=True,
                post_morphology=contact_cut)
            morphology_groups[group_name] = preprocess_details["morphology_mask"]
            group_mask_3d = preprocess_details["retained_source_mask"]
            retained_labels_3d[group_mask_3d] = voted_labels_3d[group_mask_3d]
            # Keep the temporal prediction for phase-dependent metrics.  The
            # voted 3D mask above remains the stable topology input for
            # skeleton/graph generation, while WSS, pressure and plane flow
            # use the actual per-frame lumen where a 4D segmentation exists.
            if np.asarray(geometry_filtered).ndim == 4:
                group_binary = np.isin(geometry_filtered, labels).astype(bool)
                if group_params.remove_small_cc:
                    # Filter merged labels per frame to preserve lumen motion.
                    for frame in range(group_binary.shape[3]):
                        group_binary[..., frame] = filter_connected_components(
                            group_binary[..., frame], geometry_spacing,
                            mode=group_params.cc_filter_mode,
                            min_volume_mm3=group_params.min_cc_volume_mm3,
                            rel_min_ratio=group_params.cc_rel_min_ratio,
                        )
            else:
                group_binary = self._repeat_mask_to_time(group_mask_3d, time_count)
            previous_state = dict(previous_groups.get(group_name, {})) if isinstance(previous_groups.get(group_name, {}), dict) else {}
            global_binary |= np.asarray(group_binary, dtype=bool)
            global_mask_3d |= np.asarray(processed_mask_3d, dtype=bool)
            ws.multilabel_groups[group_name] = {
                "labels": labels,
                "browser_color": str(spec["browser_color"]),
                "scene_color": str(spec["scene_color"]),
                "geometry_labels_3d": retained_labels_3d,
                "geometry_binary": np.asarray(group_binary, dtype=bool),
                "geometry_mask_3d": np.asarray(processed_mask_3d, dtype=bool),
                "geometry_clean_mask_3d": np.asarray(group_mask_3d, dtype=bool),
                "segmask_binary": labels_on_flow_grid(np.asarray(group_binary, dtype=bool), ws),
                "segmask_3d": labels_on_flow_grid(np.asarray(processed_mask_3d, dtype=bool), ws),
                "clean_mask_3d": labels_on_flow_grid(np.asarray(group_mask_3d, dtype=bool), ws),
                "graph": previous_state.get("graph", GraphData()),
                "branch_labels": previous_state.get("branch_labels"),
                "geometry_branch_labels": previous_state.get("geometry_branch_labels"),
                "centerline_paths": list(previous_state.get("centerline_paths", [])),
                "centerline_node_paths": list(previous_state.get("centerline_node_paths", [])),
                "centerline_paths_smooth": list(previous_state.get("centerline_paths_smooth", [])),
                "path_info": list(previous_state.get("path_info", [])),
                "forks": list(previous_state.get("forks", [])),
                "planes": list(previous_state.get("planes", [])),
                "path_index_offset": int(previous_state.get("path_index_offset", 0)),
                "plane_index_offset": int(previous_state.get("plane_index_offset", 0)),
                "skeleton_points": previous_state.get("skeleton_points"),
                "skeleton_mask": previous_state.get("skeleton_mask"),
            }
        ws.segmentation_analysis_labels = geometry_filtered
        ws.segmentation_analysis_labels_3d = retained_labels_3d
        ws.segmentation_analysis_binary = np.asarray(global_binary, dtype=bool)
        ws.segmentation_analysis_mask = np.asarray(global_mask_3d, dtype=bool)
        ws.segmentation_preprocess_qc = build_segmentation_preprocess_qc(
            geometry_temporal, geometry_filtered, voted_labels_3d, retained_labels_3d, global_mask_3d,
            resolution=geometry_spacing, label_map=ws.skeleton_params.label_map,
            label_groups=ws.skeleton_params.label_groups, group_specs=specs,
            processed_groups={name: state["geometry_mask_3d"] for name, state in ws.multilabel_groups.items()},
            temporal_group_mask=global_binary,
            morphology_groups=morphology_groups,
        )
        ws.segmentation_preprocess_qc["preprocessing_order"] = [
            "merge_group_labels", "morphology_and_gaussian", "optional_special_contact_separation",
            "connected_component_filter", "skeleton", "graph"]
        ws.segmentation_preprocess_qc["cleanup_4d"] = {
            "enabled": bool(getattr(getattr(ws, "segmentation", None), "cleanup_4d_components", False)),
            "mode": str(getattr(getattr(ws, "segmentation", None), "cleanup_4d_mode", "absolute")),
            "min_volume_mm3": float(getattr(getattr(ws, "segmentation", None), "cleanup_4d_min_volume_mm3", 50.0)),
        }
        ws.segmentation_preprocess_qc["group_filters"] = [{
            "group_name": name, "labels": list(ws.multilabel_groups[name]["labels"]),
            "parameters": ws.skeleton_params.params_for_group(name).to_dict(),
        } for name in ws.group_order]
        # Quantitative masks always start from the active native source.
        # A direct model cache is a topology/display copy, never a replacement
        # for the model's exported native prediction.
        from ..algorithms.geometry_grid import retain_native_measurement_masks
        native_vote = majority_vote_labels_3d(ws.segmask_raw)
        native_specs = self._build_group_specs(ws, native_vote)
        retain_native_measurement_masks(ws, native_specs, voted_labels=native_vote)
        geometry_info['quantitative_source'] = measurement_segmentation(ws).source
        geometry_info['quantitative_policy'] = QUANTITATIVE_POLICY
        ws.segmentation_geometry.update(geometry_info)
        ws.segmentation_preprocess_qc["segmentation_geometry"] = dict(geometry_info)
        ws.segmentation_preprocess_qc["counts_grid"] = "shared_1mm_segmentation"
        ws.set_object_visible_by_data_key("segmask_raw_surface", False)
        ws.remove_object_by_data_key("segmask_pre_surface")
        ws.remove_objects_by_prefix("segmask_group_")
        for group_name in ws.group_order:
            group_state = ws.multilabel_groups.get(group_name, {})
            ws.add_object(
                name=self._group_object_name("segmask", group_name),
                kind=ObjectKind.SEGMENTATION,
                data_key=self._group_data_key("segmask_group", group_name),
                group_name=group_name,
                browser_color=str(group_state.get("browser_color", "") or ""),
                visible=True,
                opacity=render_style_settings(ws)["context"]["opacity"],
                color=ws.skeleton_params.scene_color_for_group(group_name, "scene"),
            )
        check_cancelled()
        ws._preprocess_signature = signature
        return True

    def refresh_segmentation_dependents(self, ws):
        """Keep geometry only when its processed topology and configuration match."""
        previous_signature = ws._preprocess_signature
        previous_labels, previous_binary = ws.segmentation_analysis_labels, ws.segmentation_analysis_binary
        previous_native_labels, previous_native_binary = ws.segmask_labels, ws.segmask_binary
        previous_labels_3d, previous_mask = ws.segmentation_analysis_labels_3d, ws.segmentation_analysis_mask
        previous_order, previous_groups = list(ws.group_order), dict(ws.multilabel_groups)
        if previous_signature is None or previous_labels_3d is None:
            ws.reset_segmentation_results()
            return "reset"
        self.preprocess(ws)
        check_cancelled()
        geometry_same = True
        previous_geometry_labels = getattr(ws, "skeleton_labels", None)
        if previous_geometry_labels is not None and float(getattr(ws.skeleton_params, "geometry_spacing_mm", 1.0)) > 0:
            from ..algorithms.geometry_grid import prepare_skeleton_geometry
            current_geometry_labels, _, _ = prepare_skeleton_geometry(ws)
            geometry_same = np.array_equal(previous_geometry_labels, current_geometry_labels)
        topology_same = (
            geometry_same and previous_signature[1:] == ws._preprocess_signature[1:]
            and previous_order == ws.group_order
            and np.array_equal(previous_labels_3d, ws.segmentation_analysis_labels_3d)
            and np.array_equal(previous_mask, ws.segmentation_analysis_mask)
            and all(
                previous_groups[name].get("labels") == ws.multilabel_groups[name].get("labels")
                and np.array_equal(previous_groups[name].get("geometry_clean_mask_3d"), ws.multilabel_groups[name].get("geometry_clean_mask_3d"))
                and np.array_equal(previous_groups[name].get("geometry_mask_3d"), ws.multilabel_groups[name].get("geometry_mask_3d"))
                for name in previous_order
            )
        )
        if not topology_same:
            ws.reset_segmentation_results()
            return "reset"
        if (np.array_equal(previous_labels, ws.segmentation_analysis_labels)
                and np.array_equal(previous_binary, ws.segmentation_analysis_binary)
                and np.array_equal(previous_native_labels, ws.segmask_labels)
                and np.array_equal(previous_native_binary, ws.segmask_binary)):
            return "unchanged"
        # Temporal changes affect numerical support and trajectories even when
        # the voted 3-D centerline topology is stable. Recompute all metrics.
        ws.clear_streamlines()
        ws.clear_pathlines()
        ws.derived = DerivedResults()
        for plane in ws.planes:
            plane.metrics = {}
        for step in (StepId.COMPUTE_PLANE_METRICS, StepId.COMPUTE_DERIVED_METRICS,
                     StepId.COMPUTE_PWV, StepId.GENERATE_STREAMLINES, StepId.PLANE_STREAMLINES):
            ws.pipeline.completed.pop(step.value, None)
            ws.pipeline.skipped.pop(step.value, None)
        for key, obj in list(ws.scene_objects.items()):
            if obj.kind == ObjectKind.METRIC or obj.data_key in {"pwv_planes", "derived_streamlines_live"}:
                ws.scene_objects.pop(key)
        return "geometry"

    def run_step(self, ws, step, log, progress_callback=None, force_recompute_corr=False):
        dispatch = {
            StepId.BACKGROUND_CORRECTION: lambda workspace: self._step_background_correction(
                workspace, progress_callback=progress_callback, force_recompute=force_recompute_corr),
            StepId.REMOVE_NOISE: self._step_remove_noise,
            StepId.GENERATE_PCMRA: self._step_generate_pcmra,
            StepId.UNWRAP_PHASE: self._step_unwrap_phase,
            StepId.GENERATE_SKELETON: self._step_generate_skeleton,
            StepId.EDIT_SKELETON: self._step_edit_skeleton,
            StepId.GENERATE_GRAPH: self._step_generate_graph,
            StepId.EDIT_GRAPH: self._step_edit_graph,
            StepId.GENERATE_PLANES: self._step_generate_planes,
            StepId.EDIT_PLANES: self._step_edit_planes,
            StepId.COMPUTE_PWV: self._step_compute_pwv,
            StepId.GENERATE_STREAMLINES: self._step_generate_streamlines,
            StepId.PLANE_STREAMLINES: self._step_plane_streamlines,
            StepId.COMPUTE_PLANE_METRICS: lambda workspace: self._step_compute_plane_metrics(workspace, progress_callback=progress_callback),
            StepId.COMPUTE_DERIVED_METRICS: self._step_compute_derived_metrics,
        }
        if not hasattr(ws.pipeline, "errors"):
            ws.pipeline.errors = {}
        try:
            result = dispatch[step](ws)
        except TaskCancelled:
            raise
        except Exception as exc:
            ws.pipeline.errors[step.value] = {"step": step.value, "message": str(exc),
                "error_type": type(exc).__name__, "output_status": "failed"}
            raise
        if result.success:
            ws.pipeline.errors.pop(step.value, None)
        else:
            ws.pipeline.errors[step.value] = {"step": step.value, "message": str(result.message),
                "error_type": "StepFailed", "output_status": "skipped" if result.skipped else "failed"}
        return result

    def _step_background_correction(self, ws, progress_callback=None, force_recompute=False):
        """Correct source encoding, retaining all manually generated downstream artifacts."""
        if not ws.data_loaded or ws.flow_raw is None:
            return StepResult(StepId.BACKGROUND_CORRECTION, False, False, "Load data before background correction")
        source = getattr(ws, "_loaded_input_source", None)
        if source is None:
            path = ws.paths.flow_path or ws.paths.segmask_path
            if not path:
                return StepResult(StepId.BACKGROUND_CORRECTION, False, False, "Source input is unavailable")
            source = InputCase(path, "dicom" if "dicom" in ws.input_state.source_format else "h5",
                               source_group=ws.input_state.source_group, metadata=dict(ws.input_state.metadata))
        cfg = copy.deepcopy(ws.loader_params.background_phase_correction)
        cfg.enabled = True
        cfg.force_recompute = bool(force_recompute or cfg.force_recompute)
        kwargs = dict(getattr(ws, "_loaded_input_kwargs", {}) or {})
        kwargs.pop("progress_callback", None)
        kwargs.pop("dicom_read_workers", None)
        kwargs.pop("reuse_existing_corr", None)
        if progress_callback is not None:
            kwargs["progress_callback"] = progress_callback
        kwargs.update(correction_config=cfg, ignore_embedded_segmentation=True, force_recompute_seg=False)
        if not getattr(ws, "_loaded_input_kwargs", None):
            if ws.input_state.metadata.get("input_parameter_overrides"):
                kwargs["parameter_overrides"] = copy.deepcopy(ws.input_state.metadata["input_parameter_overrides"])
            kwargs["dicom_backend"] = ws.loader_params.dicom_backend
            kwargs["dicom_h5_dir"] = ws.loader_params.dicom_h5_dir
            dual_info = ws.input_state.metadata.get("dual_venc", {})
            if isinstance(dual_info, dict) and dual_info.get("selected_mode"):
                kwargs["dual_venc_mode"] = dual_info["selected_mode"]
        try:
            data = load_input_data(source, **kwargs)
            flow = np.asarray(data.flow, dtype=np.float32)
            if flow.shape != ws.flow_raw.shape:
                raise ValueError("corrected source dimensions differ from the loaded input")
            for field in ("resolution", "origin", "venc"):
                if not np.allclose(np.asarray(getattr(data, field)), np.asarray(getattr(ws, field))):
                    raise ValueError(f"source {field} differs from the loaded input; reload before correction")
        except TaskCancelled:
            raise
        except Exception as exc:
            return StepResult(StepId.BACKGROUND_CORRECTION, False, False, f"Background correction failed: {exc}")
        report = dict(data.metadata.get("background_phase_correction", {}) or {})
        parts = [report.get("dual_venc_low", {}), report.get("dual_venc_high", {})] if "dual_venc_low" in report else [report]
        if not all(part.get("applied", False) for part in parts):
            reason = "; ".join(str(part.get("skipped_reason", "not applied")) for part in parts)
            return StepResult(StepId.BACKGROUND_CORRECTION, False, False, f"Background correction not applied: {reason}")
        check_cancelled()
        ws.flow_raw = flow
        ws.flow_input = flow.copy()
        phase = getattr(data, "phase_wrapped", None)
        if phase is None:
            venc = np.asarray(ws.venc, dtype=np.float32).reshape(1, 1, 1, 1, 3)
            phase = np.angle(np.exp(1j * flow * np.pi / np.maximum(venc, 1e-6)))
        ws.phase_wrapped = np.asarray(phase, dtype=np.float32)
        ws.phase_wrapped_high = None if data.phase_wrapped_high is None else np.asarray(data.phase_wrapped_high, dtype=np.float32)
        ws.correction_raw = None if data.correction is None else np.asarray(data.correction, dtype=np.float32)
        ws.correction_high_raw = None if data.correction_high is None else np.asarray(data.correction_high, dtype=np.float32)
        for key in ("background_phase_correction", "dual_venc"):
            if key in data.metadata:
                ws.input_state.metadata[key] = copy.deepcopy(data.metadata[key])
        ws.loader_params.background_phase_correction.enabled = True
        ws.phase_unwrap_result = {}
        ws.remove_object_by_data_key("phase_wrap_mask")
        ws.remove_object_by_data_key("phase_wrap_count")
        ws.pipeline.completed.pop(StepId.UNWRAP_PHASE.value, None)
        ws.pipeline.skipped.pop(StepId.UNWRAP_PHASE.value, None)
        ws.pipeline.mark_done(StepId.BACKGROUND_CORRECTION)
        return StepResult(StepId.BACKGROUND_CORRECTION, True, False,
                          "Background correction applied to working flow; existing downstream results retained. Rerun them manually.")

    def _step_remove_noise(self, ws):
        if not ws.data_loaded or ws.mag_raw is None or ws.flow_raw is None:
            return StepResult(StepId.REMOVE_NOISE, False, False, "Load magnitude and velocity before noise removal")
        cfg = ws.noise_removal_params
        try:
            mask, report = pcmra_render_mask(ws.mag_raw, ws.flow_raw, ws.venc, method=cfg.method,
                                              magnitude_fraction=cfg.magnitude_fraction, velocity_std_max=cfg.velocity_std_max)
        except Exception as exc:
            return StepResult(StepId.REMOVE_NOISE, False, False, f"Noise removal failed: {exc}")
        ws.pcmra_render_mask = mask
        ws.noise_removal_result = report
        ws.remove_object_by_data_key("noise_region")
        ws.add_object(name="Noise Region", kind=ObjectKind.AUX, data_key="noise_region",
                      group_name="Global", visible=False, opacity=0.3, point_size=2.0, color="#ff0000", dynamic=False)
        ws.pipeline.mark_done(StepId.REMOVE_NOISE)
        return StepResult(StepId.REMOVE_NOISE, True, False,
                          f"PC-MRA rendering region retained {report['retained_fraction']:.1%}; magnitude and flow unchanged")

    def _step_generate_pcmra(self, ws):
        """Materialize PC-MRA only when requested, after all correction actions."""
        if not ws.data_loaded or ws.mag_raw is None or ws.flow_raw is None:
            return StepResult(StepId.GENERATE_PCMRA, False, False, "Load magnitude and velocity before generating PC-MRA")
        magnitude = np.asarray(ws.mag_raw)
        flow = np.asarray(ws.flow_raw)
        if magnitude.ndim == 3:
            magnitude = magnitude[..., None]
        if (flow.ndim != 5 or flow.shape[-1] != 3 or magnitude.ndim != 4
                or magnitude.shape[:3] != flow.shape[:3] or magnitude.shape[3] not in (1, flow.shape[3])):
            return StepResult(StepId.GENERATE_PCMRA, False, False, "Magnitude/velocity dimensions do not match for PC-MRA")
        pcmra = np.empty(flow.shape[:4], dtype=np.float32)
        for tidx in range(flow.shape[3]):
            report_progress({"stage": "pcmra", "current": tidx, "total": flow.shape[3],
                             "message": f"Generating PC-MRA phase {tidx + 1}/{flow.shape[3]}"})
            mag_t = np.asarray(magnitude[..., min(tidx, magnitude.shape[3] - 1)], dtype=np.float32)
            speed = np.linalg.norm(np.asarray(flow[..., tidx, :], dtype=np.float32), axis=-1)
            pcmra[..., tidx] = np.nan_to_num(mag_t * speed, nan=0.0, posinf=0.0, neginf=0.0)
        check_cancelled()
        ws.pcmra_array = pcmra
        obj = next((obj for obj in ws.scene_objects.values() if obj.data_key == "pcmra_volume"), None)
        if obj is None:
            ws.add_object(name="PC-MRA (4D)", kind=ObjectKind.AUX, data_key="pcmra_volume", group_name="Global",
                          visible=True, opacity=0.85, scalars="PC-MRA", cmap="gray", dynamic=True, show_scalar_bar=False)
        else:
            obj.visible = True
            obj.dynamic = True
        ws.pipeline.mark_done(StepId.GENERATE_PCMRA)
        return StepResult(StepId.GENERATE_PCMRA, True, False,
                          "Generated PC-MRA from current working velocity; noise mask applies to 3D rendering only")

    def _step_unwrap_phase(self, ws):
        """Optional traditional or learned phase unwrapping with diagnostics."""
        cfg = getattr(ws, "phase_unwrap_params", None)
        method = str(getattr(cfg, "method", "") or "").strip()
        # Method selection is the single enable switch.  ``enabled`` and the
        # old ``none`` method are accepted only for backwards-compatible
        # workspace/config loading and no longer control the GUI.
        if cfg is None or method.lower() in {"", "none", "disabled"}:
            ws.phase_unwrap_result = {"skipped": True, "reason": "disabled"}
            ws.pipeline.mark_done(StepId.UNWRAP_PHASE, skipped=True)
            return StepResult(StepId.UNWRAP_PHASE, True, True, "Phase unwrapping skipped: disabled")
        metadata = dict(ws.input_state.metadata or {})
        dual_info = metadata.get("dual_venc", {})
        if not isinstance(dual_info, dict):
            dual_info = {}
        dual_mode = str(dual_info.get("selected_mode", dual_info.get("mode", "dv")) or "dv").lower()
        dual = (
            (bool(dual_info.get("enabled", False)) or str(ws.input_state.source_format).lower() == "legacy_h5_dual_venc")
            and dual_mode == "dv"
        )
        if dual:
            ws.phase_unwrap_result = {"skipped": True, "reason": "dual_venc"}
            ws.pipeline.mark_done(StepId.UNWRAP_PHASE, skipped=True)
            return StepResult(StepId.UNWRAP_PHASE, True, True, "Phase unwrapping skipped: dual-VENC input")
        if ws.phase_wrapped is None:
            ws.phase_unwrap_result = {"skipped": True, "reason": "wrapped_phase_unavailable"}
            ws.pipeline.mark_done(StepId.UNWRAP_PHASE, skipped=True)
            return StepResult(StepId.UNWRAP_PHASE, True, True, "Phase unwrapping skipped: wrapped phase unavailable")
        try:
            mask_source = resolve_mask_source(method, getattr(cfg, "mask_source", "auto"))
        except ValueError as exc:
            return StepResult(StepId.UNWRAP_PHASE, False, False, str(exc))
        phase = np.asarray(ws.phase_wrapped, dtype=np.float32)
        phase_shape = phase.shape[:4]
        # A segmentation is used only when selected, never implicitly by a PCMRA source.
        if mask_source == "segmask":
            if ws.segmask_raw is None:
                return StepResult(StepId.UNWRAP_PHASE, False, False, "segmask requires an available segmentation")
            mask = np.asarray(ws.segmask_raw) > 0
            if mask.ndim == 3:
                mask = np.broadcast_to(mask[..., None], phase_shape)
            if tuple(mask.shape) != tuple(phase_shape):
                return StepResult(StepId.UNWRAP_PHASE, False, False, "segmask dimensions do not match wrapped phase")
            backend_weightmask = mask.astype(np.float32)
            backend_center_confidence = np.any(mask, axis=3).astype(np.float32)
        elif mask_source == "none":
            mask = np.ones(phase_shape, dtype=bool)
            backend_weightmask = np.ones(phase_shape, dtype=np.float32)
            backend_center_confidence = np.ones(phase_shape[:3], dtype=np.float32)
        else:
            magnitude = np.asarray(ws.mag_raw, dtype=np.float32)
            if magnitude.ndim == 3:
                magnitude = magnitude[..., None]
            pcmra = magnitude * np.linalg.norm(phase, axis=-1)
            pcmra = np.nan_to_num(pcmra, nan=0.0, posinf=0.0, neginf=0.0)
            scale = float(np.max(pcmra))
            if scale > 0:
                pcmra /= scale
            confidence = (np.std(pcmra, axis=3) if mask_source == "pcmra_std"
                          else np.mean(pcmra, axis=3)).astype(np.float32)
            scale = float(np.max(confidence))
            if scale > 0:
                confidence /= scale
            backend_center_confidence = confidence
            backend_weightmask = np.asarray(pcmra * confidence[..., None], dtype=np.float32)
            mask = np.broadcast_to(confidence[..., None] > 0, phase_shape).copy()
        if not np.any(mask):
            ws.phase_unwrap_result = {"skipped": True, "reason": "empty_mask"}
            ws.pipeline.mark_done(StepId.UNWRAP_PHASE, skipped=True)
            return StepResult(StepId.UNWRAP_PHASE, True, True, "Phase unwrapping skipped: mask is empty")
        params = {
            "tfc": bool(getattr(cfg, "tfc", True)),
            "lap4d_ts": float(getattr(cfg, "lap4d_ts", 2.0)),
            "nprs_upsampling_factor": int(getattr(cfg, "nprs_upsampling_factor", 2)),
            "nprs_pi_unwrap": bool(getattr(cfg, "nprs_pi_unwrap", True)),
            "nprs_auto_crop": bool(getattr(cfg, "nprs_auto_crop", True)),
            "backend_params": copy.deepcopy(getattr(cfg, "backend_params", {}) or {}),
        }
        # Both learned backends use physical voxel spacing.  Keep explicit
        # backend overrides intact while supplying the loaded case geometry by
        # default.
        backend_params = params["backend_params"]
        if isinstance(backend_params.get("pudip"), dict):
            backend_params["pudip"].setdefault(
                "spacing", tuple(float(v) for v in np.asarray(ws.resolution).reshape(-1)[:3])
            )
        if isinstance(backend_params.get("gust"), dict):
            backend_params["gust"].setdefault(
                "voxel_spacing", tuple(float(v) for v in np.asarray(ws.resolution).reshape(-1)[:3])
            )
        try:
            result = unwrap_phase(
                ws.phase_wrapped, mask, ws.venc, str(cfg.method),
                params=params, device=str(getattr(cfg, "device", "auto")),
                backend_weightmask=backend_weightmask,
                backend_center_confidence=backend_center_confidence,
            )
            result["mask_source"] = mask_source
        except Exception as exc:
            ws.phase_unwrap_result = {"skipped": False, "error": str(exc)}
            ws.pipeline.completed.pop(StepId.UNWRAP_PHASE.value, None)
            ws.pipeline.skipped.pop(StepId.UNWRAP_PHASE.value, None)
            return StepResult(StepId.UNWRAP_PHASE, False, False, f"Phase unwrapping failed: {exc}")
        if ws.flow_input is None:
            ws.flow_input = np.array(ws.flow_raw, copy=True) if ws.flow_raw is not None else None
        result["flow_unwrapped"] = np.where(mask[..., None], result["flow_unwrapped"], ws.flow_raw).astype(np.float32)
        venc = np.asarray(ws.venc, dtype=np.float32).reshape(1, 1, 1, 1, 3)
        result["phase_unwrapped"] = result["flow_unwrapped"] * np.pi / venc
        result["diagnostic_scope"] = "latest_masked_run"
        ws.flow_raw = result["flow_unwrapped"]
        ws.phase_unwrap_result = result
        # Existing segmentation, geometry, metrics and trajectories remain until manually rerun.
        ws.remove_object_by_data_key("phase_wrap_mask")
        ws.remove_object_by_data_key("phase_wrap_count")
        ws.add_object(name="Estimated Wrap Locations", kind=ObjectKind.AUX, data_key="phase_wrap_mask", visible=False,
                      opacity=0.65, scalars="wrap_mask", cmap="Reds", clim=(0, 1), dynamic=True)
        ws.add_object(name="Phase Wrap Count", kind=ObjectKind.AUX, data_key="phase_wrap_count", visible=False,
                      opacity=0.7, scalars="wrap_count", cmap="coolwarm", clim=(-2, 2), dynamic=True)
        ws.pipeline.mark_done(StepId.UNWRAP_PHASE)
        stats = result.get("statistics", {})
        msg = (f"Phase unwrapping: method={result.get('method')} device={result.get('device')} "
               f"wrapped_voxels={stats.get('wrapped_voxels_any', 0)} max|k|={stats.get('max_abs_k', 0)} "
               f"elapsed={result.get('elapsed_sec', 0.0):.2f}s; rerun downstream results manually")
        return StepResult(StepId.UNWRAP_PHASE, True, False, msg)

    def revert_phase_unwrap(self, ws):
        """Restore loaded flow without discarding wrap diagnostics for review."""
        if ws.flow_input is None or "flow_unwrapped" not in (ws.phase_unwrap_result or {}):
            return StepResult(StepId.UNWRAP_PHASE, True, True, "Revert skipped: no active phase-unwrapping result")
        ws.flow_raw = np.asarray(ws.flow_input, dtype=np.float32).copy()
        ws.pipeline.completed.pop(StepId.UNWRAP_PHASE.value, None)
        ws.pipeline.skipped.pop(StepId.UNWRAP_PHASE.value, None)
        return StepResult(StepId.UNWRAP_PHASE, True, False, "Restored pre-unwrapping flow; diagnostics and downstream results retained")

    def _step_generate_skeleton(self, ws):
        if ws.segmask_raw is None:
            return StepResult(
                StepId.GENERATE_SKELETON, True, True,
                self._missing_segmentation_message(ws, "Skeleton"),
            )
        self._clear_pwv_state(ws)
        self.preprocess(ws)
        from ..algorithms.geometry_grid import prepare_skeleton_geometry
        from ..algorithms.preprocess import preprocess_mask_for_skeleton
        geometry_labels, geometry_spacing, geometry_info = prepare_skeleton_geometry(ws)
        ws.skeleton_labels = geometry_labels
        ws.skeleton_spacing = geometry_spacing.copy()
        ws.skeleton_geometry = geometry_info
        ws.segmentation_preprocess_qc["skeleton_geometry"] = dict(geometry_info,
            spacing_mm=geometry_spacing.tolist(), native_spacing_mm=np.asarray(ws.resolution).tolist(),
            native_counts_scope="shared 1-mm segmentation preprocessing; velocity-grid masks are sampling adapters")
        for spec in self._build_group_specs(ws, geometry_labels):
            name = spec["name"]
            if name not in ws.multilabel_groups:
                ws.group_order.append(name)
                ws.multilabel_groups[name] = dict(labels=[], browser_color=spec["browser_color"],
                    scene_color=spec["scene_color"], segmask_3d=np.zeros_like(ws.segmask_3d),
                    segmask_binary=np.zeros_like(ws.segmask_binary), graph=GraphData())
            state = ws.multilabel_groups[name]
            state["labels"] = sorted(set(state["labels"]) | set(spec["labels"]))
        points_all = []
        skeleton_mask = np.zeros(geometry_labels.shape, dtype=bool)
        ws.remove_object_by_data_key("willis_ring_graph")
        ws.remove_object_by_data_key("skeleton_points")
        ws.remove_object_by_data_key("skeleton_mask_surface")
        ws.remove_object_by_data_key("segmask_3d_surface")
        ws.remove_objects_by_prefix("skeleton_")
        for group_index, group_name in enumerate(ws.group_order):
            report_progress({"stage": "skeleton_group", "current": group_index, "total": len(ws.group_order),
                             "message": f"Processing {group_name} ({group_index + 1}/{len(ws.group_order)})"})
            group_state = ws.multilabel_groups.get(group_name, {})
            group_params = ws.skeleton_params.params_for_group(group_name)
            native_grid = float(getattr(ws.skeleton_params, "geometry_spacing_mm", 1.0)) == 0
            protected_geometry = self._special_label_values(ws, group_params, group_state.get("labels", []))
            cut_geometry = None
            if group_params.special_handling == "contact_surface" and group_params.separate_special_label_contacts and len(protected_geometry) >= 2:
                cut_geometry = lambda m: separate_special_label_contacts(m, geometry_labels, protected_geometry, resolution=geometry_spacing)
            processed = group_state.get("geometry_mask_3d")
            if processed is None:
                processed = preprocess_mask_for_skeleton(
                    np.isin(geometry_labels, group_state.get("labels", [])), group_params, resolution=geometry_spacing,
                    post_morphology=cut_geometry)
            group_state["skeleton_geometry_mask"] = processed
            group_state["skeleton_geometry_labels"] = geometry_labels
            special_handling = str(getattr(group_params, "special_handling", "three_pass_merge") or "three_pass_merge").strip().lower().replace("-", "_")
            if special_handling == "three_pass_merge":
                protected_values = self._special_label_values(ws, group_params, group_state.get("labels", []))
                if len(protected_values) >= 2 and set(group_state.get("labels", [])) - set(protected_values):
                    pts, mask = generate_three_pass_special_skeleton(
                        geometry_labels, group_state.get("labels", []),
                        protected_values, group_params, geometry_spacing, exact_union=not native_grid)
                else:
                    pts, mask = generate_skeleton_from_mask3d(processed, geometry_spacing)
            else:
                pts, mask = generate_skeleton_from_mask3d(processed, geometry_spacing)
            group_state["skeleton_points"] = np.asarray(pts, dtype=float)
            group_state["skeleton_mask"] = np.asarray(mask, dtype=bool)
            ws.multilabel_groups[group_name] = group_state
            if len(pts):
                points_all.append(np.asarray(pts, dtype=float))
            skeleton_mask |= np.asarray(mask, dtype=bool)
            ws.add_object(
                name=self._group_object_name("skeleton", group_name),
                kind=ObjectKind.SKELETON,
                data_key=self._group_data_key("skeleton", group_name),
                group_name=group_name,
                browser_color=str(group_state.get("browser_color", "") or ""),
                visible=True,
                opacity=1.0,
                color=ws.skeleton_params.scene_color_for_group(group_name, "skeleton"),
                point_size=8,
            )
        ws.skeleton_points = np.vstack(points_all) if points_all else np.empty((0, 3), dtype=float)
        ws.skeleton_mask = np.asarray(skeleton_mask, dtype=bool)
        check_cancelled()
        self._clear_graph_state(ws)
        ws.pipeline.mark_done(StepId.GENERATE_SKELETON)
        return StepResult(StepId.GENERATE_SKELETON, True, False, f"Skeleton: {len(ws.skeleton_points)} points groups={len(ws.group_order)}")

    def _step_edit_skeleton(self, ws):
        ws.pipeline.mark_done(StepId.EDIT_SKELETON, skipped=True)
        return StepResult(StepId.EDIT_SKELETON, True, True, "Skeleton edit")

    def _step_generate_graph(self, ws, *, rebuild_groups=None, write_output=True):
        if ws.segmask_raw is None:
            return StepResult(
                StepId.GENERATE_GRAPH, True, True,
                self._missing_segmentation_message(ws, "Graph"),
            )
        if ws.segmentation_analysis_mask is None:
            self.preprocess(ws)
        self._clear_pwv_state(ws)
        requested_spacing = float(getattr(ws.skeleton_params, "geometry_spacing_mm", 1.0))
        expected_spacing = np.ones(3)
        ws.skeleton_params.geometry_spacing_mm = 1.
        actual_spacing = getattr(ws, "skeleton_spacing", None)
        grid_changed = actual_spacing is not None and not np.allclose(actual_spacing, expected_spacing)
        if (rebuild_groups is None and (ws.skeleton_points is None or len(ws.skeleton_points) == 0)) or grid_changed:
            skel_result = self._step_generate_skeleton(ws)
            if skel_result.skipped or not skel_result.success:
                return StepResult(StepId.GENERATE_GRAPH, skel_result.success, True, skel_result.message)
        self._clear_plane_state(ws)
        ws.selected_path_index = -1
        ws.derived.centerline_pressure_profiles = []
        ws.graph_hierarchy = {}
        ws.remove_objects_by_prefix("hierarchy_path_")
        geometry_spacing = np.asarray(getattr(ws, "skeleton_spacing", None) if getattr(ws, "skeleton_spacing", None) is not None else ws.resolution)
        graph_points = []
        graph_edges = []
        from ..algorithms.geometry_grid import labels_on_flow_grid
        branch_labels = np.zeros(ws.segmask_3d.shape, dtype=np.int16)
        geometry_branch_labels = np.zeros(ws.segmentation_analysis_mask.shape, dtype=np.int16)
        centerline_paths = []
        centerline_node_paths = []
        path_info = []
        forks = []
        graph_node_offset = 0
        path_offset = 0
        ws.remove_object_by_data_key("graph_lines")
        ws.remove_object_by_data_key("fork_markers")
        ws.remove_objects_by_prefix("graph_")
        ws.remove_objects_by_prefix("forks_")
        ws.remove_objects_by_prefix("path_")
        ws.remove_objects_by_prefix("smooth_path_")
        ws.remove_objects_by_prefix("path_arrow_")
        for group_index, group_name in enumerate(ws.group_order):
            report_progress({"stage": "graph_group", "current": group_index, "total": len(ws.group_order),
                             "message": f"Processing {group_name} ({group_index + 1}/{len(ws.group_order)})"})
            group_state = ws.multilabel_groups.get(group_name, {})
            group_params = ws.skeleton_params.params_for_group(group_name)
            local_points = np.asarray(group_state.get("skeleton_points"), dtype=float).reshape(-1, 3) if group_state.get("skeleton_points") is not None else np.empty((0, 3), dtype=float)
            protect_contacts = (group_params.special_handling == "three_pass_merge" or
                                group_params.separate_special_label_contacts)
            separated_values = self._special_label_values(ws, group_params, group_state.get("labels", [])) if protect_contacts else []
            geometry_labels = getattr(ws, "skeleton_labels", None)
            if geometry_labels is None:
                geometry_labels = ws.segmask_labels_3d
            rebuild = rebuild_groups is None or group_name in rebuild_groups
            if rebuild:
                local_graph = build_graph_from_points(local_points, geometry_spacing, connectivity=group_params.graph_connectivity,
                    segmentation_labels=geometry_labels, separated_label_values=separated_values)
                local_graph, junction_cleanup = simplify_junction_clusters(local_graph, geometry_labels,
                    geometry_spacing, enabled=group_params.junction_cluster_enabled,
                    max_extent_voxels=group_params.junction_cluster_max_extent_voxels,
                    min_cycle_rank=group_params.junction_cluster_min_cycle_rank,
                    flow=ws.flow_raw, flow_spacing=ws.resolution)
                group_state["junction_cluster_cleanup"] = junction_cleanup
                local_graph, cycle_cleanup = remove_intralabel_cycles(local_graph, geometry_labels,
                    geometry_spacing, flow_spacing=ws.resolution, support_mask=group_state.get("skeleton_geometry_mask"), flow=ws.flow_raw,
                    temporal_mask=group_state.get("segmask_binary"), flow_threshold=group_params.cycle_flow_threshold,
                    preserve_connectivity=group_params.cycle_preserve_connectivity)
                group_state["cycle_cleanup"] = cycle_cleanup
                min_edge_points = max(0, int(getattr(group_params, "min_edge_points", 9)))
                local_graph = remove_short_terminal_branches(local_graph, min_edge_points=min_edge_points)
                from ..algorithms.cycle_opening import open_redundant_cycles
                local_graph, opening = open_redundant_cycles(local_graph,
                    group_state.get("skeleton_geometry_mask", group_state.get("geometry_mask_3d")), geometry_spacing)
                group_state["cycle_cleanup"]["lumen_cycle_opening"] = opening
            else:
                local_graph = group_state.get("graph", GraphData())
            group_state["graph"] = local_graph
            local_binary = np.asarray(group_state.get("geometry_binary"), dtype=bool)
            local_labels, local_paths, local_node_paths, local_path_info, local_forks = segment_vessels_from_graph_and_mask(
                np.asarray(group_state.get("geometry_mask_3d"), dtype=bool),
                local_graph,
                geometry_spacing,
                flow_xyzt3=ws.flow_raw if local_binary.size else None,
                segmask_binary_4d=local_binary,
                origin=ws.origin,
                segmentation_labels=ws.segmentation_analysis_labels_3d, flow_spacing=ws.resolution,
            )
            native_local_labels = labels_on_flow_grid(local_labels, ws)
            if np.any(local_labels > 0):
                geometry_branch_labels[local_labels > 0] = local_labels[local_labels > 0] + int(path_offset)
                branch_labels[native_local_labels > 0] = native_local_labels[native_local_labels > 0] + int(path_offset)
            adjusted_paths = [np.asarray(p, dtype=float) for p in local_paths]
            adjusted_node_paths = [[int(node) + int(graph_node_offset) for node in path_nodes] for path_nodes in local_node_paths]
            adjusted_path_info = []
            for item in local_path_info:
                payload = dict(item)
                payload["path_index"] = int(payload.get("path_index", 0)) + int(path_offset)
                for node_key in ("start_node", "end_node"):
                    local_node = int(payload.get(node_key, -1))
                    payload[node_key] = local_node + int(graph_node_offset) if local_node >= 0 else -1
                payload["fork_ids"] = [int(x) + len(forks) for x in payload.get("fork_ids", [])]
                adjusted_roles = []
                for role in payload.get("fork_roles", []):
                    adjusted_role = dict(role)
                    local_fork_id = int(adjusted_role.get("fork_id", -1))
                    adjusted_role["fork_id"] = local_fork_id + len(forks) if local_fork_id >= 0 else -1
                    adjusted_roles.append(adjusted_role)
                payload["fork_roles"] = adjusted_roles
                payload["incoming_path_ids"] = [int(x) + int(path_offset) for x in payload.get("incoming_path_ids", [])]
                payload["outgoing_path_ids"] = [int(x) + int(path_offset) for x in payload.get("outgoing_path_ids", [])]
                payload["group_name"] = str(group_name)
                adjusted_path_info.append(payload)
            adjusted_forks = []
            fork_offset = len(forks)
            for fork in local_forks:
                payload = dict(fork)
                payload["left"] = [int(x) + int(path_offset) for x in payload.get("left", [])]
                payload["right"] = [int(x) + int(path_offset) for x in payload.get("right", [])]
                local_node = int(payload.get("node", -1))
                payload["node"] = local_node + int(graph_node_offset) if local_node >= 0 else -1
                payload["group_name"] = str(group_name)
                adjusted_forks.append(payload)
            if len(local_graph.points) > 0:
                graph_points.append(np.asarray(local_graph.points, dtype=float))
            if len(local_graph.edges) > 0:
                graph_edges.append(np.asarray(local_graph.edges, dtype=int) + int(graph_node_offset))
            centerline_paths.extend(adjusted_paths)
            centerline_node_paths.extend(adjusted_node_paths)
            path_info.extend(adjusted_path_info)
            forks.extend(adjusted_forks)
            group_state["branch_labels"] = np.asarray(native_local_labels, dtype=np.int16)
            group_state["geometry_branch_labels"] = np.asarray(local_labels, dtype=np.int16)
            group_state["centerline_paths"] = adjusted_paths
            group_state["centerline_node_paths"] = adjusted_node_paths
            group_state["path_info"] = adjusted_path_info
            group_state["forks"] = adjusted_forks
            group_state["path_index_offset"] = int(path_offset)
            ws.multilabel_groups[group_name] = group_state
            ws.add_object(
                name=self._group_object_name("graph", group_name),
                kind=ObjectKind.GRAPH,
                data_key=self._group_data_key("graph", group_name),
                group_name=group_name,
                browser_color=str(group_state.get("browser_color", "") or ""),
                visible=True,
                opacity=1.0,
                color=ws.skeleton_params.scene_color_for_group(group_name, "graph"),
                line_width=2,
            )
            if len(adjusted_forks) > 0:
                ws.add_object(
                    name=self._group_object_name("forks", group_name),
                    kind=ObjectKind.AUX,
                    data_key=self._group_data_key("forks", group_name),
                    group_name=group_name,
                    browser_color=str(group_state.get("browser_color", "") or ""),
                    visible=True,
                    opacity=1.0,
                    color="magenta",
                    point_size=12,
                )
            graph_node_offset += len(local_graph.points)
            path_offset += len(adjusted_paths)
        ws.graph = GraphData(
            points=np.vstack(graph_points) if graph_points else np.empty((0, 3), dtype=float),
            edges=np.vstack(graph_edges) if graph_edges else np.empty((0, 2), dtype=int),
        )
        ws.branch_labels = branch_labels
        ws.geometry_branch_labels = geometry_branch_labels
        ws.centerline_paths = centerline_paths
        ws.centerline_node_paths = centerline_node_paths
        ws.path_info = path_info
        ws.forks = forks
        ws.selected_path_index = -1
        ws.graph_hierarchy = build_workspace_hierarchy(ws)
        for info in ws.path_info:
            group_name, index = info["group_name"], info["path_index"]
            ws.add_object(name=f"path {index}", kind=ObjectKind.BRANCH,
                data_key=self._group_data_key("smooth_path", group_name, index), group_name=group_name,
                browser_color=ws.skeleton_params.browser_color_for_group(group_name), visible=True,
                color=ws.skeleton_params.scene_color_for_group(group_name, "path"), line_width=3)
        if write_output:
            with open(os.path.join(self._output_dir(ws), "graph_hierarchy.json"), "w", encoding="utf-8") as handle:
                json.dump(ws.graph_hierarchy, handle, indent=2)

        check_cancelled()
        ws.pipeline.mark_done(StepId.GENERATE_GRAPH)
        return StepResult(StepId.GENERATE_GRAPH, True, False,
                          f"Graph: {len(ws.graph.points)} nodes, {len(ws.graph.edges)} edges | "
                          f"paths={len(ws.centerline_paths)} forks={len(ws.forks)}")

    def _step_edit_graph(self, ws):
        ws.pipeline.mark_done(StepId.EDIT_GRAPH, skipped=True)
        return StepResult(StepId.EDIT_GRAPH, True, True, "Graph edit")

    def _derived_artifact_signatures(self, ws):
        dp = ws.derived_params
        geometry = measurement_segmentation(ws)
        paths = ws.centerline_paths_smooth if len(ws.centerline_paths_smooth) > 0 else ws.centerline_paths
        common = {
            "flow": _array_fingerprint(ws.flow_raw),
            "mask": _array_fingerprint(ws.segmask_binary),
            "spacing": np.asarray(ws.resolution, dtype=float).reshape(3).tolist(),
            "origin": np.asarray(ws.origin, dtype=float).reshape(3).tolist(),
        }
        return {
            "wss": _payload_fingerprint({
                **common,
                "algorithm": _DERIVED_ALGORITHM_VERSIONS["wss"],
                "geometry_mask": _array_fingerprint(geometry.binary),
                "geometry_spacing": geometry.spacing.tolist(),
                "geometry_source": geometry.source,
                "params": {
                    "smoothing_iteration": int(dp.wss_smoothing_iteration),
                    "viscosity": float(dp.wss_viscosity),
                    "inward_distance": None if dp.wss_inward_distance is None else float(dp.wss_inward_distance),
                    "parabolic_fitting": bool(dp.wss_parabolic_fitting),
                    "no_slip_condition": bool(dp.wss_no_slip_condition),
                },
            }),
            "tke": _payload_fingerprint({
                "mask": common["mask"],
                "spacing": common["spacing"],
                "origin": common["origin"],
                "source_tke": _array_fingerprint(ws.source_tke_array),
                "source_sigma": _array_fingerprint(ws.source_sigma),
                "algorithm": _DERIVED_ALGORITHM_VERSIONS["tke"],
                "rho": float(dp.tke_rho),
            }),
            "pressure": _payload_fingerprint({
                **common,
                "algorithm": _DERIVED_ALGORITHM_VERSIONS["pressure"],
                "rr_ms": float(ws.rr),
                "centerline_paths": [_array_fingerprint(path) for path in paths],
                "params": {
                    "rho": float(dp.pressure_gradient_rho),
                    "viscosity": float(dp.pressure_gradient_viscosity),
                    "smoothing_sigma": float(dp.pressure_gradient_smoothing_sigma),
                    "support_erosion_iters": int(dp.pressure_gradient_support_erosion_iters),
                    "use_convective_acceleration": bool(dp.pressure_gradient_use_convective_acceleration),
                    "method": str(dp.pressure_method),
                },
            }),
            "vortex": _payload_fingerprint({
                **common,
                "algorithm": _DERIVED_ALGORITHM_VERSIONS["vortex"],
                "params": {
                    "smoothing_sigma": float(dp.vortex_smoothing_sigma),
                    "support_erosion_iters": int(dp.vortex_support_erosion_iters),
                },
            }),
        }

    @staticmethod
    def _drop_stale_pixelwise_family(pixelwise, family):
        keys = {
            "wss": {"wss"},
            "tke": {"tke", "tke_time"},
            "pressure": {
                "pressure_gradient",
                "pressure_gradient_mag",
                "pressure_gradient_peak",
                "pressure_gradient_support_mask",
                "relative_pressure",
                "relative_pressure_peak",
            },
            "vortex": {
                "vorticity",
                "vorticity_magnitude",
                "vorticity_magnitude_peak",
                "q_criterion",
                "q_criterion_peak",
                "swirling_strength",
                "swirling_strength_peak",
                "vortex_support_mask",
            },
        }[family]
        for key in keys:
            pixelwise.pop(key, None)

    @staticmethod
    def _derived_artifact_validity(ws, signatures):
        stored = dict(ws.derived.artifact_signatures or {})
        return {
            "wss": (
                ws.derived.wss_volume is not None
                and np.size(ws.derived.wss_volume) > 0
                and stored.get("wss") == signatures["wss"]
            ),
            "tke": (
                (ws.derived.tke_array is not None or ws.derived.tke_volume is not None)
                and stored.get("tke") == signatures["tke"]
            ),
            "pressure": (
                ws.derived.pressure_gradient_array is not None
                and np.size(ws.derived.pressure_gradient_array) > 0
                and stored.get("pressure") == signatures["pressure"]
            ),
            "vortex": (
                all(
                    getattr(ws.derived, name, None) is not None
                    and np.size(getattr(ws.derived, name)) > 0
                    for name in (
                        "vorticity_array",
                        "vorticity_magnitude",
                        "vorticity_magnitude_peak",
                        "q_criterion_array",
                        "q_criterion_peak",
                        "swirling_strength_array",
                        "swirling_strength_peak",
                        "vortex_support_mask",
                    )
                )
                and stored.get("vortex") == signatures["vortex"]
            ),
        }

    def _ensure_derived_metrics(
        self,
        ws,
        save_pixelwise=False,
        refresh_scene_objects=False,
        compute_wss=True,
        compute_tke=True,
        compute_pressure_gradient=True,
        compute_vortex=False,
    ):
        self.preprocess(ws)
        signatures = self._derived_artifact_signatures(ws)
        validity = self._derived_artifact_validity(ws, signatures)
        has_wss = validity["wss"]
        has_pg = validity["pressure"]
        has_tke = validity["tke"]
        has_vortex = validity["vortex"]
        source_tke = ws.source_tke_array
        source_sigma = ws.source_sigma if ws.input_state.capabilities.has_complex_source else None
        request_tke = bool(compute_tke and (source_tke is not None or source_sigma is not None))
        pixelwise = dict(ws.derived.pixelwise_export or {})
        if compute_wss and not has_wss:
            self._drop_stale_pixelwise_family(pixelwise, "wss")
        if compute_pressure_gradient and not has_pg:
            self._drop_stale_pixelwise_family(pixelwise, "pressure")
        if request_tke and not has_tke:
            self._drop_stale_pixelwise_family(pixelwise, "tke")
        if compute_vortex and not has_vortex:
            self._drop_stale_pixelwise_family(pixelwise, "vortex")
        need_wss = bool(compute_wss and (not has_wss or (save_pixelwise and "wss" not in pixelwise)))
        need_pg = bool(
            compute_pressure_gradient
            and (not has_pg or (save_pixelwise and "pressure_gradient" not in pixelwise))
        )
        need_tke = bool(request_tke and (not has_tke or (save_pixelwise and "tke_time" not in pixelwise)))
        need_vortex = bool(compute_vortex and (not has_vortex or (save_pixelwise and "q_criterion" not in pixelwise)))
        if not (need_wss or need_pg or need_tke or need_vortex):
            if refresh_scene_objects:
                self._register_derived_scene_objects(ws)
            return ws.derived
        dp = ws.derived_params
        geometry = measurement_segmentation(ws)
        result = compute_derived_metrics(
            flow=ws.flow_raw,
            mask4d=ws.segmask_binary,
            spacing=ws.resolution,
            origin=ws.origin,
            smoothing_iteration=dp.smoothing_iteration,
            viscosity=dp.viscosity,
            inward_distance=dp.inward_distance,
            parabolic_fitting=dp.parabolic_fitting,
            no_slip_condition=dp.no_slip_condition,
            step_size=dp.step_size,
            tube_radius=dp.tube_radius,
            rho=dp.rho,
            # These payloads are views of the computed arrays, not file writes.
            # Cache them even during plane preparation so a later export or
            # addition of another family does not recompute valid WSS/pressure.
            save_pixelwise=True,
            tke_array=source_tke,
            sigma=source_sigma,
            rr=ws.rr,
            pressure_gradient_smoothing_sigma=dp.pressure_gradient_smoothing_sigma,
            pressure_gradient_support_erosion_iters=dp.pressure_gradient_support_erosion_iters,
            pressure_gradient_use_convective_acceleration=dp.pressure_gradient_use_convective_acceleration,
            pressure_method=dp.pressure_method,
            centerline_paths=ws.centerline_paths_smooth if len(ws.centerline_paths_smooth) > 0 else ws.centerline_paths,
            compute_wss=need_wss,
            wss_mask4d=geometry.binary, wss_spacing=geometry.spacing,
            compute_tke=need_tke,
            compute_pressure_gradient=need_pg,
            compute_vortex=need_vortex,
            wss_smoothing_iteration=dp.wss_smoothing_iteration,
            wss_viscosity=dp.wss_viscosity,
            wss_inward_distance=dp.wss_inward_distance,
            wss_parabolic_fitting=dp.wss_parabolic_fitting,
            wss_no_slip_condition=dp.wss_no_slip_condition,
            tke_rho=dp.tke_rho,
            pressure_gradient_rho=dp.pressure_gradient_rho,
            pressure_gradient_viscosity=dp.pressure_gradient_viscosity,
            vortex_smoothing_sigma=dp.vortex_smoothing_sigma,
            vortex_support_erosion_iters=dp.vortex_support_erosion_iters,
        )
        if need_wss:
            ws.derived.wss_surfaces = result["wss_surfaces"]
            ws.derived.wss_volume = result.get("wss_volume")
            ws.derived.artifact_signatures["wss"] = signatures["wss"]
            self._clear_global_family_stale(ws, "wss")
        if need_tke:
            ws.derived.tke_volume = result["tke_volume"]
            ws.derived.tke_array = result.get("tke_array")
            ws.derived.artifact_signatures["tke"] = signatures["tke"]
            self._clear_global_family_stale(ws, "tke")
        if need_pg:
            ws.derived.pressure_gradient_array = result.get("pressure_gradient_array")
            ws.derived.pressure_gradient_magnitude = result.get("pressure_gradient_magnitude")
            ws.derived.pressure_gradient_peak = result.get("pressure_gradient_peak")
            ws.derived.pressure_gradient_dt_s = result.get("pressure_gradient_dt_s")
            ws.derived.pressure_gradient_temporal_scheme = str(
                result.get("pressure_gradient_temporal_scheme", "") or ""
            )
            ws.derived.pressure_gradient_support_mask = result.get("pressure_gradient_support_mask")
            ws.derived.pressure_gradient_display_clim = result.get("pressure_gradient_display_clim")
            ws.derived.relative_pressure_array = result.get("relative_pressure_array")
            ws.derived.relative_pressure_peak = result.get("relative_pressure_peak")
            ws.derived.relative_pressure_display_clim = result.get("relative_pressure_display_clim")
            ws.derived.centerline_pressure_profiles = list(result.get("centerline_pressure_profiles", []) or [])
            ws.derived.artifact_signatures["pressure"] = signatures["pressure"]
            self._clear_global_family_stale(ws, "pressure")
        if need_vortex:
            ws.derived.vorticity_array = result.get("vorticity_array")
            ws.derived.vorticity_magnitude = result.get("vorticity_magnitude")
            ws.derived.vorticity_magnitude_peak = result.get("vorticity_magnitude_peak")
            ws.derived.q_criterion_array = result.get("q_criterion_array")
            ws.derived.q_criterion_peak = result.get("q_criterion_peak")
            ws.derived.swirling_strength_array = result.get("swirling_strength_array")
            ws.derived.swirling_strength_peak = result.get("swirling_strength_peak")
            ws.derived.vortex_support_mask = result.get("vortex_support_mask")
            ws.derived.artifact_signatures["vortex"] = signatures["vortex"]
            self._clear_global_family_stale(ws, "vortex")
        ws.derived.streamlines = []
        result_pixelwise = dict(result.get("pixelwise_export", {}) or {})
        if result_pixelwise:
            pixelwise.update(result_pixelwise)
        ws.derived.pixelwise_export = pixelwise
        if refresh_scene_objects:
            self._register_derived_scene_objects(ws)
        return ws.derived

    def _compute_plane_metrics_internal(
        self,
        ws,
        save=True,
        use_multithread=False,
        include_derived=True,
        ensure_derived=True,
        compute_wss=True,
        compute_tke=True,
        compute_pressure_gradient=True,
        progress_callback=None,
    ):
        if not ws.has_flow():
            return [], {}, "Plane metrics skipped: no flow"
        if ws.segmask_raw is None:
            return [], {}, self._missing_segmentation_message(ws, "Plane metrics")
        previous_policy = ws.segmentation_geometry.get('quantitative_policy')
        if (ws.segmentation_analysis_binary is None
                or (previous_policy and previous_policy != QUANTITATIVE_POLICY)):
            self.preprocess(ws)
        geometry = measurement_segmentation(ws)
        if include_derived and ensure_derived:
            self._ensure_derived_metrics(
                ws,
                save_pixelwise=False,
                refresh_scene_objects=False,
                compute_wss=compute_wss,
                compute_tke=compute_tke,
                compute_pressure_gradient=compute_pressure_gradient,
            )
        derived_validity = {"wss": False, "tke": False, "pressure": False}
        if include_derived:
            signatures = self._derived_artifact_signatures(ws)
            derived_validity = self._derived_artifact_validity(ws, signatures)
            include_derived = any(derived_validity.values())
        # Prefer the smoothed centerlines (better local tangents) but fall back
        # to the raw ordered ones if the smoothing step hasn't been run yet.
        paths_for_tangent = ws.centerline_paths_smooth if len(ws.centerline_paths_smooth) > 0 else ws.centerline_paths
        from ..algorithms.metrics.sampling import PlaneSamplingContext
        from .plane_state import stamp_metric, workspace_signature
        sampling_context = None
        # Preserve isolated-process dispatch for large jobs. Small jobs already
        # run serially; their two passes can now share one geometry context.
        if ws.planes and (not use_multithread or len(ws.planes) * ws.time_count() < 512):
            sampling_context = PlaneSamplingContext(
                geometry.binary, geometry.spacing, ws.origin, ws.planes,
                branch_labels_3d=geometry.branches, segmentation_labels_3d=geometry.labels_3d,
                segmentation_labels_4d=geometry.labels, voxel_centered=True)
        if use_multithread and sampling_context is None:
            metrics, qc = compute_plane_metrics_multithread(
                ws.flow_raw, geometry.binary, geometry.spacing, ws.origin, ws.planes,
                RR=ws.rr, branch_labels_3d=geometry.branches,
                path_info=ws.path_info, forks=ws.forks, paths=paths_for_tangent,
                return_qc=True, segmentation_labels_3d=geometry.labels_3d,
                segmentation_labels_4d=geometry.labels,
                progress_callback=progress_callback, flow_spacing=ws.resolution)
        else:
            metrics, qc = compute_plane_metrics(
                ws.flow_raw, geometry.binary, geometry.spacing, ws.origin, ws.planes,
                RR=ws.rr, branch_labels_3d=geometry.branches,
                path_info=ws.path_info, forks=ws.forks, paths=paths_for_tangent,
                return_qc=True, segmentation_labels_3d=geometry.labels_3d,
                segmentation_labels_4d=geometry.labels,
                progress_callback=progress_callback, sampling_context=sampling_context, flow_spacing=ws.resolution)
        if include_derived:
            metrics, plane_pixelwise = augment_plane_metrics_with_derived(
                metrics, ws.planes, geometry.binary, geometry.spacing, ws.origin,
                branch_labels_3d=geometry.branches,
                segmentation_labels_3d=geometry.labels_3d,
                segmentation_labels_4d=geometry.labels,
                tke_array=ws.derived.tke_array if derived_validity["tke"] else None,
                pressure_gradient_array=(
                    ws.derived.pressure_gradient_array if derived_validity["pressure"] else None
                ),
                relative_pressure_array=(
                    ws.derived.relative_pressure_array if derived_validity["pressure"] else None
                ),
                pressure_gradient_support_mask=(
                    ws.derived.pressure_gradient_support_mask if derived_validity["pressure"] else None
                ),
                wss_surfaces=ws.derived.wss_surfaces if derived_validity["wss"] else None,
                use_multithread=use_multithread, progress_callback=progress_callback,
                sampling_context=sampling_context, field_spacing=ws.resolution,
            )
        else:
            metrics = [dict(metric) for metric in metrics]
            plane_pixelwise = []
            for idx, plane in enumerate(ws.planes):
                plane_pixelwise.append(
                    {
                        "plane_index": int(idx),
                        **plane.identity_dict(),
                        "center": np.asarray(plane.center, dtype=float).reshape(3).tolist(),
                        "normal": np.asarray(plane.normal, dtype=float).reshape(3).tolist(),
                        "label": int(getattr(plane, "label", 0) or 0),
                        "path_index": int(getattr(plane, "path_index", -1)),
                        "timepoints": [],
                    }
                )
        metrics = [dict(metric, roi_geometry_source=geometry.source,
                        roi_spacing_mm=geometry.spacing.tolist(), roi_geometry_policy=QUANTITATIVE_POLICY)
                   for metric in metrics]
        metrics = [stamp_metric(plane, metric, i) for i, (plane, metric) in enumerate(zip(ws.planes, metrics))]
        ws.derived.plane_metrics = metrics
        ws.derived.artifact_signatures["plane_metrics"] = workspace_signature(ws)
        generated_qc = dict(ws.derived.plane_qc or {})
        generated_layout = generated_qc.get("plane_layout", generated_qc)
        if isinstance(generated_layout, dict) and any(
                key in generated_layout for key in ("paths", "failure_policy", "errors")):
            qc = dict(qc or {})
            plane_layout = dict(generated_layout)
            plane_layout["actual_plane_count"] = int(len(ws.planes))
            qc["plane_layout"] = plane_layout
        ws.derived.plane_qc = qc
        for i, metric in enumerate(metrics):
            if i < len(ws.planes):
                ws.planes[i].metrics = dict(metric)
        msg = f"Plane metrics: {len(metrics)} paths={len(qc.get('path_ic', {}))} forks={len(qc.get('forks', []))}"
        if save:
            out_dir = self._output_dir(ws)
            plane_metric_path = os.path.join(out_dir, "plane_metrics.json")
            qc_path = os.path.join(out_dir, "plane_qc.json")
            plane_pixelwise_path = os.path.join(out_dir, "plane_metrics_pixelwise.h5")
            from .output import atomic_json_write
            atomic_json_write(plane_metric_path, metrics)
            atomic_json_write(qc_path, qc)
            save_plane_pixelwise_h5(plane_pixelwise_path, plane_pixelwise, rr_ms=ws.rr, source_format=ws.input_state.source_format,
                                    input_signature=workspace_signature(ws, include_analysis=False))
            ws.derived.plane_pixelwise_file = plane_pixelwise_path
            msg += f" saved={plane_metric_path} qc={qc_path} pixelwise={plane_pixelwise_path}"
        if sampling_context is not None:
            sampling_context.close()
        return metrics, qc, msg

    @staticmethod
    def _clear_global_family_stale(ws, family):
        metadata = ws.input_state.metadata
        metadata["stale_metric_families"] = [key for key in metadata.get("stale_metric_families", []) if key != family]

    def _save_planes_json(self, ws):
        out_dir = self._output_dir(ws)
        source_path = str(ws.paths.flow_path or ws.paths.segmask_path or "")
        out_path = os.path.join(out_dir, "planes.json")
        payload = build_plane_records(ws)
        save_planes_json(ws, out_path, source_path=source_path, payload=payload)
        save_planes_h5(ws, os.path.join(out_dir, "planes.h5"), source_path=source_path, payload=payload)
        return out_path

    def _step_import_plane_positions(self, ws, *, write_output=True):
        if not ws.data_loaded:
            raise ValueError("Load the target data before placing planes from a file")
        source = str(ws.plane_gen_params.positions_file).strip()
        payload = load_plane_position_payload(source)
        imported, report = project_planes_to_workspace(
            payload, ws, mapping_mode=ws.plane_gen_params.import_mode, return_report=True)
        if not imported:
            raise ValueError("The plane positions file did not contain any importable planes")
        check_cancelled()
        # Validate before replacing the current layout; preserve centerline geometry.
        from uuid import uuid4
        for plane in imported:
            plane.placement["source_plane_id"] = plane.plane_id
            plane.plane_id = uuid4().hex
        self._clear_plane_state(ws, preserve_paths=True)
        ws.planes = imported
        ws.analysis_revision += 1
        for index, plane in enumerate(imported):
            group = ws.multilabel_groups.get(plane.group_name, {})
            if group:
                if not group["planes"]:
                    group["plane_index_offset"] = index
                group["planes"].append(plane)
            ws.add_object(name=self._indexed_object_name("plane", index), kind=ObjectKind.PLANE,
                data_key=self._group_data_key("plane", plane.group_name, index), group_name=plane.group_name,
                browser_color=str(group.get("browser_color", "") or ""), visible=True, opacity=.6,
                color=ws.skeleton_params.scene_color_for_group(plane.group_name, "plane"), line_width=2)
        ws.derived.plane_qc = {**report, "plane_mode": "imported", "source_file": source,
                               "actual_plane_count": len(imported)}
        if write_output:
            self._save_planes_json(ws)
            out_dir = self._output_dir(ws)
            save_plane_positions(ws, os.path.join(out_dir, "plane_positions.json"), source_path=str(ws.paths.flow_path or ""))
            from .output import atomic_json_write
            atomic_json_write(os.path.join(out_dir, "plane_qc.json"), ws.derived.plane_qc)
        ws.pipeline.mark_done(StepId.GENERATE_PLANES)
        message = f"Planes: {len(imported)} from {source} | coordinates={ws.plane_gen_params.import_mode}"
        if report["skipped_count"]:
            message += f" | skipped={report['skipped_count']} (see plane_qc.json)"
        return StepResult(StepId.GENERATE_PLANES, True, False, message)

    def _step_generate_planes(self, ws, *, write_output=True, generate_missing_graph=True, use_position_file=True):
        if use_position_file and str(ws.plane_gen_params.positions_file or "").strip():
            return self._step_import_plane_positions(ws, write_output=write_output)
        if ws.segmask_raw is None:
            return StepResult(
                StepId.GENERATE_PLANES, True, True,
                self._missing_segmentation_message(ws, "Planes"),
            )
        previous_policy = ws.segmentation_geometry.get('quantitative_policy')
        if (ws.segmentation_analysis_mask is None
                or (previous_policy and previous_policy != QUANTITATIVE_POLICY)):
            self.preprocess(ws)
        self._clear_pwv_state(ws)
        if generate_missing_graph and (ws.graph is None or len(ws.graph.points) == 0):
            graph_result = self._step_generate_graph(ws)
            if graph_result.skipped or not graph_result.success:
                return StepResult(StepId.GENERATE_PLANES, graph_result.success, True, graph_result.message)

        self._clear_plane_state(ws)
        measurement = measurement_segmentation(ws)
        from ..algorithms.path_smoothing import PathSmoothingContext
        from ..plane_rules import smoothing_options
        pgp = ws.plane_gen_params
        smoothing = smoothing_options(vars(pgp))
        smoothing_contexts = [None]*len(ws.centerline_paths)
        if pgp.smoothing_method == 'radius_adaptive':
            for group_name in ws.group_order:
                group = ws.multilabel_groups.get(group_name,{})
                mask = group.get('skeleton_geometry_mask',group.get('geometry_mask_3d'))
                if mask is None:
                    mask = ws.segmentation_analysis_mask
                context = PathSmoothingContext(mask, ws.segmentation_spacing)
                offset = int(group.get('path_index_offset',0))
                for index in range(offset,offset+len(group.get('centerline_paths',[]))):
                    smoothing_contexts[index] = context
        planes = []
        smooth_paths = []
        plane_layout_qc = {
            "paths": [],
            "planes": [],
            "requested_plane_mode": str(pgp.plane_mode),
            "path_smoothing": smoothing,
            "requested_plane_count": int(pgp.plane_count),
            "fov_boundary_filter": bool(pgp.fov_boundary_filter),
            "fov_boundary_max_fraction": float(pgp.fov_boundary_max_fraction),
        }
        by_label = str(pgp.plane_mode) == "by_label" and (
            bool(ws.skeleton_params.label_map) or any(str(name).isdigit() for name in pgp.label_rules)
        )
        label_plan = None
        if str(pgp.plane_mode) == "by_label" and not by_label:
            plane_layout_qc.update(plane_mode="by_label", actual_plane_count=0, failure_policy="qc_error",
                errors=[{"label_name":"global", "reasons":{"label_map_unavailable":1},
                         "message":"By-label placement requires a label map or numeric label rules; no global fallback was used."}])
            ws.derived.plane_qc = plane_layout_qc
            path = self._save_planes_json(ws)
            with open(os.path.join(os.path.dirname(path), "plane_qc.json"), "w", encoding="utf-8") as handle:
                json.dump(plane_layout_qc, handle, indent=2)
            ws.pipeline.mark_done(StepId.GENERATE_PLANES)
            return StepResult(StepId.GENERATE_PLANES, True, False,
                "Planes: 0 | QC error: label_map_unavailable; see plane_qc.json")
        if by_label:
            label_plan = generate_label_planes(
                ws.centerline_paths, ws.segmentation_analysis_labels_3d, ws.skeleton_params.label_map, pgp.label_rules,
                spacing=ws.segmentation_spacing, path_info=ws.path_info,
                branch_labels_3d=ws.geometry_branch_labels,
                filter_defaults={"segmentation_filter": pgp.segmentation_filter, "voxel_centered": True,
                                 "fov_boundary_filter": pgp.fov_boundary_filter,
                                 "fov_boundary_max_fraction": pgp.fov_boundary_max_fraction},
                flow_xyzt3=ws.flow_raw, segmask_binary_4d=ws.segmentation_analysis_binary, flow_spacing=ws.resolution,
                **smoothing, smoothing_contexts=smoothing_contexts,
                smoothing_window=pgp.smoothing_window * pgp.inter_time,
                smoothing_polyorder=pgp.smoothing_polyorder, inter_time=pgp.inter_time,
                fork_points=[fork["crosspoint"] for fork in ws.forks if fork.get("crosspoint") is not None],
            )
            plane_layout_qc["labels"] = label_plan[3]
            plane_layout_qc["requested_plane_count"] = sum(
                q["requested_plane_count"] for q in label_plan[3]
                if q["status"] not in {"label_absent", "disabled"}
            )
        # Unmapped workspaces retain global per-path placement compatibility.
        plane_layout_qc["plane_mode"] = "by_label" if by_label else ("fixed_step" if pgp.plane_mode == "by_label" else pgp.plane_mode)
        plane_layout_qc["layout_warnings"] = ["plane_mode_fallback"] if str(pgp.plane_mode) == "by_label" and not by_label else []
        branch_support_cache = {}
        candidate_plane_offset = 0
        for group_index, group_name in enumerate(ws.group_order):
            report_progress({"stage": "planes_group", "current": group_index, "total": len(ws.group_order),
                             "message": f"Processing {group_name} ({group_index + 1}/{len(ws.group_order)})"})
            group_state = ws.multilabel_groups.get(group_name, {})
            local_paths = list(group_state.get("centerline_paths", []))
            fork_points = [
                fork.get("crosspoint") for fork in list(group_state.get("forks", []))
                if isinstance(fork, dict) and fork.get("crosspoint") is not None
            ]
            path_offset = int(group_state.get("path_index_offset", 0))
            if by_label:
                path_end = path_offset + len(local_paths)
                local_planes = [copy.deepcopy(plane) for plane in label_plan[0]
                                if path_offset <= plane.path_index < path_end]
                for plane in local_planes:
                    plane.path_index -= path_offset
                local_smooth_paths = label_plan[1][path_offset:path_end]
                local_filter_qc = label_plan[2][path_offset:path_end]
            else:
                local_planes, local_smooth_paths, local_filter_qc = generate_planes_from_paths(
                    local_paths,
                    cross_section_distance=pgp.cross_section_distance,
                    start_distance=pgp.start_distance,
                    end_distance=pgp.end_distance,
                    **smoothing, smoothing_context=(smoothing_contexts[path_offset] if local_paths else None),
                    smoothing_window=pgp.smoothing_window * pgp.inter_time,
                    smoothing_polyorder=pgp.smoothing_polyorder,
                    inter_time=pgp.inter_time,
                    plane_mode="fixed_step" if pgp.plane_mode == "by_label" else pgp.plane_mode,
                    plane_count=pgp.plane_count,
                    anchor=pgp.anchor,
                    anchor_offset_mm=pgp.anchor_offset_mm,
                    direction=getattr(pgp, "direction", "both"),
                    spacing_mode=getattr(pgp, "spacing_mode", "fraction"),
                    spacing_ratio=getattr(pgp, "spacing_ratio", 0.25),
                    segmentation_labels=ws.segmentation_analysis_labels_3d,
                    segmentation_filter=bool(getattr(pgp, "segmentation_filter", True)),
                    path_info=group_state.get("path_info", []),
                    segmentation_spacing=ws.segmentation_spacing,
                    segmentation_origin=np.zeros(3),
                    return_qc=True,
                    fork_points=fork_points,
                    forks=group_state.get("forks", []),
                )
            # The branch volume uses globally offset path IDs, while the
            # freshly generated local planes still carry local path indices.
            # Remove only planes with no branch-supported cross-section; keep
            # the graph/path topology and all paths themselves intact.
            if (
                (by_label or bool(getattr(pgp, "segmentation_filter", True)) or pgp.fov_boundary_filter)
                and ws.segmentation_analysis_binary is not None
            ):
                batches = {}
                for candidate_index, plane in enumerate(local_planes):
                    rule = getattr(plane, "_label_plane_rule", {}) if by_label else {}
                    check_support = rule.get("segmentation_filter")
                    check_support = bool(pgp.segmentation_filter) if check_support is None else check_support
                    fov_enabled = rule.get("fov_boundary_filter")
                    fov_enabled = pgp.fov_boundary_filter if fov_enabled is None else fov_enabled
                    fraction = rule.get("fov_boundary_max_fraction")
                    fraction = pgp.fov_boundary_max_fraction if fraction is None else fraction
                    batches.setdefault((check_support, float(fraction) if fov_enabled else None), []).append(candidate_index)
                original_planes, kept_indices, support_records = local_planes, [], {}
                for (check_support, fraction), indices in batches.items():
                    _, records = filter_planes_by_branch_support(
                        [original_planes[i] for i in indices], measurement.binary, measurement.spacing, ws.origin,
                        branch_labels_3d=measurement.branches, segmentation_labels_3d=measurement.labels_3d,
                        branch_label_offset=path_offset, support_cache=branch_support_cache,
                        check_branch_support=check_support, fov_boundary_max_fraction=fraction, voxel_centered=True)
                    for record in records:
                        original_index = indices[record["plane_index"]]
                        record["plane_index"] = original_index
                        if by_label:
                            source_plane = original_planes[original_index]
                            record.update(main_path_index=source_plane._main_path_index,
                                          main_path_fraction=source_plane._main_path_fraction)
                        support_records[original_index] = record
                        if record["valid"]:
                            kept_indices.append(original_index)
                local_planes = [original_planes[i] for i in sorted(kept_indices)]
                local_support_qc = [support_records[i] for i in sorted(support_records)]
            else:
                support_reason = (
                    "segmentation_filter_disabled"
                    if not bool(getattr(pgp, "segmentation_filter", True))
                    else "branch_support_unavailable"
                )
                local_support_qc = [
                    {
                        "plane_index": int(i),
                        "path_index": int(getattr(plane, "path_index", -1)),
                        "branch_label": int(getattr(plane, "path_index", -1)) + 1 + int(path_offset),
                        "segmentation_label": int(getattr(plane, "segmentation_label", 0) or 0),
                        "valid": True,
                        "reason": support_reason,
                        "supported_phase_count": 0,
                        "phase_count": 0,
                        "max_cell_count": 0,
                    }
                    for i, plane in enumerate(local_planes)
                ]
            plane_offset = len(planes)
            for local_path_idx, raw_q in enumerate(local_filter_qc):
                q = dict(raw_q or {})
                q["path_index"] = int(path_offset + local_path_idx)
                q["requested_plane_count"] = int(q.get("requested_plane_count", pgp.plane_count))
                q["actual_plane_count"] = int(sum(1 for p in local_planes if int(p.path_index) == int(local_path_idx)))
                q["dropped_plane_count"] = int(sum(
                    1 for item in local_support_qc
                    if int(item.get("path_index", -1)) == int(local_path_idx)
                    and not bool(item.get("valid", False))
                ))
                q["path_status"] = (
                    "not_selected" if by_label and not q["requested_plane_count"] else
                    "no_valid_planes" if int(q["actual_plane_count"]) == 0 else "ok"
                )
                plane_layout_qc["paths"].append(q)
            for item in local_support_qc:
                support = dict(item)
                support["path_index"] = int(path_offset + int(support.get("path_index", -1)))
                support["plane_index_local"] = int(support.get("plane_index", -1))
                source_plane_index = int(candidate_plane_offset + int(item.get("plane_index", -1)))
                output_plane_index = int(plane_offset + sum(
                    1 for prior in local_support_qc[:int(item.get("plane_index", 0))]
                    if bool(prior.get("valid", False))
                )) if bool(item.get("valid", False)) else None
                support["plane_index_source"] = source_plane_index
                support["plane_index"] = output_plane_index
                support["group_name"] = str(group_name)
                plane_layout_qc["planes"].append(support)
            candidate_plane_offset += len(local_support_qc)
            adjusted_smooth_paths = [np.asarray(path, dtype=float) for path in local_smooth_paths]
            adjusted_planes = []
            for local_idx, plane in enumerate(local_planes):
                plane.path_index = int(plane.path_index) + int(path_offset)
                plane.label = int(plane.path_index) + 1
                plane.group_name = str(group_name)
                adjusted_planes.append(plane)
                global_plane_idx = plane_offset + local_idx
                ws.add_object(
                    name=self._indexed_object_name("plane", global_plane_idx),
                    kind=ObjectKind.PLANE,
                    data_key=self._group_data_key("plane", group_name, global_plane_idx),
                    group_name=group_name,
                    browser_color=str(group_state.get("browser_color", "") or ""),
                    visible=True,
                    opacity=0.6,
                    color=ws.skeleton_params.scene_color_for_group(group_name, "plane"),
                    line_width=2,
                )
            for local_idx, _path in enumerate(adjusted_smooth_paths):
                global_path_idx = path_offset + local_idx
                ws.add_object(
                    name=self._indexed_object_name("smooth_path", global_path_idx),
                    kind=ObjectKind.BRANCH,
                    data_key=self._group_data_key("smooth_path", group_name, global_path_idx),
                    group_name=group_name,
                    browser_color=str(group_state.get("browser_color", "") or ""),
                    visible=True,
                    opacity=1.0,
                    color=ws.skeleton_params.scene_color_for_group(group_name, "path"),
                    line_width=3,
                )
            group_state["centerline_paths_smooth"] = adjusted_smooth_paths
            group_state["planes"] = adjusted_planes
            group_state["plane_index_offset"] = plane_offset
            ws.multilabel_groups[group_name] = group_state
            smooth_paths.extend(adjusted_smooth_paths)
            planes.extend(adjusted_planes)
        ws.planes = planes
        ws.centerline_paths_smooth = smooth_paths
        plane_layout_qc["actual_plane_count"] = int(len(ws.planes))
        plane_layout_qc["segmentation_filter"] = by_label or bool(getattr(pgp, "segmentation_filter", True))
        preprocess_labels = {int(item["label"]): item for item in ws.segmentation_preprocess_qc.get("labels", [])}
        for label_qc in plane_layout_qc.get("labels", []):
            actual = sum(int(plane.segmentation_label) == int(label_qc["segmentation_label"]) for plane in ws.planes)
            label_qc["actual_plane_count"] = int(actual)
            for main_path in label_qc.get("selected_main_paths", []):
                main_path["actual_plane_count"] = sum(
                    int(plane.segmentation_label) == int(label_qc["segmentation_label"])
                    and int(plane.path_index) == int(main_path["path_index"])
                    and getattr(plane, "_main_path_index", -1) == main_path["main_path_index"]
                    for plane in ws.planes)
            if label_qc["status"] == "selected":
                label_qc["status"] = "ok" if actual == label_qc["requested_plane_count"] else "incomplete"
            if label_qc["status"] == "label_absent":
                provenance = preprocess_labels.get(int(label_qc["segmentation_label"]))
                label_qc["reason"] = provenance["status"] if provenance else "absent_from_input"
            omitted = list(label_qc.get("omitted_positions", []))
            omitted.extend(item for item in plane_layout_qc["planes"]
                           if int(item.get("segmentation_label", 0)) == int(label_qc["segmentation_label"])
                           and not item.get("valid", True))
            reasons = {}
            for item in omitted:
                reason = str(item.get("reason", "unknown"))
                reasons[reason] = reasons.get(reason, 0) + 1
            label_qc["excluded_plane_counts"] = reasons
        errors = []
        if by_label:
            for q in plane_layout_qc.get("labels", []):
                if q.get("status") in {"disabled", "not_configured"}:
                    continue
                if q.get("status") == "label_absent":
                    provenance = preprocess_labels.get(int(q["segmentation_label"]), {})
                    if int(provenance.get("input_union_voxels", 0)) <= 0:
                        continue
                elif q.get("status") == "ok":
                    continue
                reasons = dict(q.get("excluded_plane_counts", {}))
                if not reasons:
                    reasons = {str(reason):1 for reason in q.get("selection_warnings", [])}
                if not reasons:
                    reasons = {str(q.get("reason") or q.get("status") or "placement_unavailable"):1}
                q["status"] = "error"
                errors.append({"label_name":q["label_name"], "segmentation_label":q["segmentation_label"],
                    "requested":int(q.get("requested_plane_count", 0)), "actual":int(q.get("actual_plane_count", 0)),
                    "reasons":reasons, "placements":q.get("rule", {}).get("placements", []),
                    "available_levels":sorted({c["label_level"] for c in q.get("candidates", []) if c.get("label_level") is not None})})
        plane_layout_qc.update(failure_policy="qc_error", errors=errors)
        ws.derived.plane_qc = plane_layout_qc

        planes_path = "pending"
        if write_output:
            planes_path = self._save_planes_json(ws)
            with open(os.path.join(os.path.dirname(planes_path), "plane_qc.json"), "w", encoding="utf-8") as handle:
                json.dump(plane_layout_qc, handle, indent=2)
        check_cancelled()
        dropped_planes = sum(
            1 for item in plane_layout_qc.get("planes", [])
            if not bool(item.get("valid", True))
        )
        msg = (
            f"Planes: {len(ws.planes)} paths={len(ws.centerline_paths_smooth)} "
            f"forks={len(ws.forks)} dropped={dropped_planes} saved={planes_path}"
        )
        fallback = [q["label_name"] for q in plane_layout_qc.get("labels", [])
                    if q.get("selection_warnings") or q.get("placement_warnings")]
        if fallback:
            msg += " | Placement warning (see QC): " + ", ".join(fallback)
        if errors:
            summary = "; ".join(e["label_name"] + ": " + ", ".join(e["reasons"]) for e in errors)
            msg += " | QC error: " + summary + " | see plane_qc.json"
        ws.pipeline.mark_done(StepId.GENERATE_PLANES)
        return StepResult(StepId.GENERATE_PLANES, True, False, msg)

    def _step_edit_planes(self, ws):
        ws.pipeline.mark_done(StepId.EDIT_PLANES, skipped=True)
        return StepResult(StepId.EDIT_PLANES, True, True, "Plane edit")

    def _step_compute_pwv(self, ws):
        if not ws.has_flow():
            return StepResult(StepId.COMPUTE_PWV, True, True, "PWV skipped: no flow")
        if ws.segmask_raw is None:
            return StepResult(
                StepId.COMPUTE_PWV, True, True,
                self._missing_segmentation_message(ws, "PWV"),
            )
        results, msg = self._compute_pwv_internal(ws, save=True)
        skipped = str(msg).startswith("PWV skipped:")
        ws.pipeline.mark_done(StepId.COMPUTE_PWV, skipped=skipped)
        return StepResult(StepId.COMPUTE_PWV, True, skipped, msg, outputs=list(results or []))

    def _step_generate_streamlines(self, ws):
        if ws.streamline_params.seed_region == "segmentation" and ws.segmask_raw is None:
            return StepResult(
                StepId.GENERATE_STREAMLINES, True, True,
                self._missing_segmentation_message(ws, "Streamlines"),
            )
        if ws.flow_raw is None:
            return StepResult(StepId.GENERATE_STREAMLINES, True, True, "Streamlines skipped: no flow")
        if ws.streamline_params.seed_region == "segmentation":
            self.preprocess(ws)
        try:
            volume_seed_domain(ws.flow_raw, ws.streamline_params.seed_region,
                segmentation=ws.segmask_binary, noise_keep=ws.pcmra_render_mask, phase=int(ws.current_t))
        except ValueError as exc:
            return StepResult(StepId.GENERATE_STREAMLINES, False, False, str(exc))
        # Frame domains/grids are prepared by the bounded background worker.
        ws.streamline_seeds = None
        ws.streamline_cache.clear()
        ws.streamline_active = True
        style = configured_metric_style(ws, "streamlines_live")
        ws.remove_object_by_data_key("streamlines_live")
        render_cfg = dict(getattr(ws, "render_settings", {}) or {})
        ws.add_object(
            name="streamlines", kind=ObjectKind.FLOW,
            data_key="streamlines_live", visible=True, opacity=style["opacity"], line_width=ws.streamline_params.line_width,
            scalars="Velocity", cmap=style["cmap"], clim=render_cfg.get("streamline_clim"), dynamic=True,
            show_scalar_bar=bool(render_cfg.get("streamline_show_scalar_bar", True)), scalar_bar_title="Velocity (m/s)",
            scalar_bar_cfg=dict(render_cfg.get("streamline_bar_cfg", {}) or {}),
            tube_radius=ws.streamline_params.tube_radius)
        ws.pipeline.mark_done(StepId.GENERATE_STREAMLINES)
        p = ws.streamline_params
        param_msg = (f"Streamlines enabled: seed_ratio={p.seed_ratio} max_steps={p.max_steps} "
                     f"min_seeds={p.min_seeds} terminal_speed={p.terminal_speed} rng_seed={p.rng_seed}")
        return StepResult(StepId.GENERATE_STREAMLINES, True, False, param_msg)

    def _step_plane_streamlines(self, ws):
        if ws.streamline_params.pathline_seed_region == "segmentation" and ws.segmask_raw is None:
            return StepResult(
                StepId.PLANE_STREAMLINES, True, True,
                self._missing_segmentation_message(ws, "Pathlines"),
            )
        if ws.flow_raw is None:
            return StepResult(StepId.PLANE_STREAMLINES, True, True, "Pathlines skipped: no flow")
        if len(ws.planes) == 0:
            return StepResult(StepId.PLANE_STREAMLINES, True, True, "Pathlines skipped: no planes")
        if ws.segmask_raw is not None:
            self.preprocess(ws)
        active_indices = list(range(len(ws.planes)))
        ws.active_pathline_plane_indices = sorted({
            *(
                int(idx)
                for idx in ws.active_pathline_plane_indices
                if 0 <= int(idx) < len(ws.planes)
            ),
            *active_indices,
        })
        for plane_idx in active_indices:
            group_name = self._plane_group_name(ws, plane_idx)
            data_key = self._plane_data_key("pathline", ws, plane_idx)
            if any(obj.data_key == data_key for obj in ws.scene_objects.values()):
                continue
            ws.add_object(
                name=self._indexed_object_name("pathline", plane_idx), kind=ObjectKind.FLOW,
                data_key=data_key, group_name=group_name,
                browser_color=ws.skeleton_params.browser_color_for_group(group_name) if group_name else "",
                visible=True, opacity=1.0,
                dynamic=True, **pathline_color_style(ws, plane_idx),
                tube_radius=ws.streamline_params.pathline_tube_radius)
        ws.pipeline.mark_done(StepId.PLANE_STREAMLINES)
        p = ws.streamline_params
        return StepResult(
            StepId.PLANE_STREAMLINES,
            True,
            False,
            f"Pathlines enabled for {len(active_indices)} planes: region={p.pathline_seed_region} distribution={p.pathline_seed_distribution} grid_spacing_mm={p.pathline_grid_spacing_mm} release_phase={int(ws.current_t) if p.pathline_seed_region == 'contour' else 0} max_steps={p.pathline_max_steps} terminal_speed={p.pathline_terminal_speed}",
        )

    def _step_compute_plane_metrics(self, ws, progress_callback=None):
        if not ws.has_flow():
            return StepResult(StepId.COMPUTE_PLANE_METRICS, True, True, "Plane metrics skipped: no flow")
        if ws.segmask_raw is None:
            return StepResult(
                StepId.COMPUTE_PLANE_METRICS, True, True,
                self._missing_segmentation_message(ws, "Plane metrics"),
            )
        if len(ws.planes) == 0:
            plane_result = self._step_generate_planes(ws)
            if plane_result.skipped or not plane_result.success:
                return StepResult(StepId.COMPUTE_PLANE_METRICS, plane_result.success, True, plane_result.message)
        use_mt = getattr(ws.derived_params, "use_multithread", False)
        include_derived = bool(
            (ws.derived.wss_surfaces and ws.derived.wss_volume is not None)
            or ws.derived.tke_array is not None
            or ws.derived.pressure_gradient_array is not None
            or ws.derived.relative_pressure_array is not None
        )
        _, _, msg = self._compute_plane_metrics_internal(
            ws,
            save=True,
            use_multithread=use_mt,
            include_derived=include_derived,
            ensure_derived=False,
            progress_callback=progress_callback,
        )
        msg += " derived=reused" if include_derived else " derived=not_requested"
        self._save_planes_json(ws)
        ws.pipeline.mark_done(StepId.COMPUTE_PLANE_METRICS)
        return StepResult(StepId.COMPUTE_PLANE_METRICS, True, False, msg)

    def _step_compute_derived_metrics(self, ws):
        if not ws.has_flow():
            return StepResult(StepId.COMPUTE_DERIVED_METRICS, True, True, "Derived metrics skipped: no flow")
        if ws.segmask_raw is None:
            return StepResult(
                StepId.COMPUTE_DERIVED_METRICS, True, True,
                self._missing_segmentation_message(ws, "Derived metrics"),
            )
        self._ensure_derived_metrics(
            ws,
            save_pixelwise=False,
            refresh_scene_objects=True,
            compute_vortex=True,
        )
        augmented_planes = 0
        geometry = measurement_segmentation(ws)
        if len(ws.planes) > 0 and len(ws.derived.plane_metrics) == len(ws.planes):
            metrics, plane_pixelwise = augment_plane_metrics_with_derived(
                ws.derived.plane_metrics,
                ws.planes,
                geometry.binary,
                geometry.spacing,
                ws.origin,
                branch_labels_3d=geometry.branches,
                segmentation_labels_3d=geometry.labels_3d,
                segmentation_labels_4d=geometry.labels,
                tke_array=ws.derived.tke_array,
                pressure_gradient_array=ws.derived.pressure_gradient_array,
                relative_pressure_array=ws.derived.relative_pressure_array,
                pressure_gradient_support_mask=ws.derived.pressure_gradient_support_mask,
                wss_surfaces=ws.derived.wss_surfaces,

                field_spacing=ws.resolution,
            )
            ws.derived.plane_metrics = metrics
            for index, metric in enumerate(metrics):
                ws.planes[index].metrics = dict(metric)
            out_dir = self._output_dir(ws)
            with open(os.path.join(out_dir, "plane_metrics.json"), "w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)
            if ws.derived.plane_qc:
                with open(os.path.join(out_dir, "plane_qc.json"), "w", encoding="utf-8") as f:
                    json.dump(ws.derived.plane_qc, f, ensure_ascii=False, indent=2)
            plane_pixelwise_path = os.path.join(out_dir, "plane_metrics_pixelwise.h5")
            save_plane_pixelwise_h5(
                plane_pixelwise_path,
                plane_pixelwise,
                rr_ms=ws.rr,
                source_format=ws.input_state.source_format,
            )
            ws.derived.plane_pixelwise_file = plane_pixelwise_path
            self._save_planes_json(ws)
            augmented_planes = len(metrics)
        has_tke = ws.derived.tke_array is not None or ws.derived.tke_volume is not None
        msg = f"Derived: Nt={len(ws.derived.wss_surfaces)}"
        if not has_tke:
            msg += " tke=unavailable"
        if ws.derived.q_criterion_array is not None:
            msg += " vortex=ready"
        if augmented_planes:
            msg += f" plane_summaries={augmented_planes}"
        ws.pipeline.mark_done(StepId.COMPUTE_DERIVED_METRICS)
        return StepResult(StepId.COMPUTE_DERIVED_METRICS, True, False, msg)
