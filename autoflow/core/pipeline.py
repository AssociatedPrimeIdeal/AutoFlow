import json
import os
import numpy as np

from .models import StepId, ObjectKind, GraphData
from ..algorithms import (
    load_input_data,
    filter_segmask_labels, binarize_segmask, merge_segmask_to_3d,
    preprocess_mask_for_skeleton,
    generate_skeleton_from_mask3d, build_graph_from_points,
    segment_vessels_from_graph_and_mask,
    generate_planes_from_paths,
    compute_plane_metrics, compute_derived_metrics,
    compute_plane_metrics_multithread,
    augment_plane_metrics_with_derived, save_plane_pixelwise_h5,
    generate_seed_points,
    largest_connected_component,
    compute_pwv_groups,
    save_pwv_results,
    segmentation_timestamp,
)


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
        ws.remove_object_by_data_key("pwv_planes")

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
            ws.derived.pwv_file = save_pwv_results(ws.derived.pwv_results, os.path.join(out_dir, "pwv.json"))
        self._register_pwv_scene_object(ws)
        ok_count = sum(1 for item in ws.derived.pwv_results if str(item.get("status", "")) == "ok")
        total = len(ws.derived.pwv_results)
        msg = f"PWV: {ok_count}/{total} groups"
        if ws.derived.pwv_file:
            msg += f" saved={ws.derived.pwv_file}"
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
        for dk in ["wss_surface_live", "tke_volume", "pressure_gradient_volume", "relative_pressure_volume"]:
            ws.remove_object_by_data_key(dk)
        render_cfg = dict(getattr(ws, "render_settings", {}) or {})
        wss_max = float(np.nanmax(ws.derived.wss_volume)) if ws.derived.wss_volume is not None and np.size(ws.derived.wss_volume) else 0.0
        tke_max = float(np.nanmax(ws.derived.tke_array)) if ws.derived.tke_array is not None and np.size(ws.derived.tke_array) else 0.0
        wss_clim = render_cfg.get("wss_clim") or (0.0, wss_max if wss_max > 0 else 1.0)
        tke_clim = render_cfg.get("tke_clim") or (0.0, tke_max if tke_max > 0 else 1.0)
        pressure_gradient_clim = render_cfg.get("pressure_gradient_clim")
        if pressure_gradient_clim is None:
            pressure_gradient_clim = tuple(ws.derived.pressure_gradient_display_clim) if ws.derived.pressure_gradient_display_clim is not None else (0.0, 1.0)
        relative_pressure_clim = render_cfg.get("relative_pressure_clim")
        if relative_pressure_clim is None:
            relative_pressure_clim = tuple(ws.derived.relative_pressure_display_clim) if ws.derived.relative_pressure_display_clim is not None else (-1.0, 1.0)
        ws.add_object(name="wss_surface", kind=ObjectKind.METRIC,
                      data_key="wss_surface_live", visible=False, opacity=1.0,
                      scalars="wss", cmap="jet", clim=tuple(wss_clim), dynamic=True,
                      show_scalar_bar=bool(render_cfg.get("wss_show_scalar_bar", True)), scalar_bar_title="WSS (Pa)",
                      scalar_bar_cfg=dict(render_cfg.get("wss_bar_cfg", {}) or {}))
        has_tke_local = ws.derived.tke_array is not None or ws.derived.tke_volume is not None
        if has_tke_local:
            ws.add_object(name="tke_volume", kind=ObjectKind.METRIC,
                          data_key="tke_volume", visible=False, opacity=0.5,
                          scalars="TKE", cmap="hot", clim=tuple(tke_clim), dynamic=True,
                          show_scalar_bar=bool(render_cfg.get("tke_show_scalar_bar", True)), scalar_bar_title="TKE (J/m³)",
                          scalar_bar_cfg=dict(render_cfg.get("tke_bar_cfg", {}) or {}))
        if ws.derived.pressure_gradient_magnitude is not None:
            ws.add_object(name="pressure_gradient_volume", kind=ObjectKind.METRIC,
                          data_key="pressure_gradient_volume", visible=False, opacity=float(np.clip(ws.derived_params.pressure_gradient_layer_opacity, 0.0, 1.0)),
                          scalars="PressureGradient", cmap="magma", clim=pressure_gradient_clim, dynamic=True,
                          show_scalar_bar=bool(render_cfg.get("pressure_gradient_show_scalar_bar", True)), scalar_bar_title="|Pressure Grad| (Pa/m)",
                          scalar_bar_cfg=dict(render_cfg.get("pressure_gradient_bar_cfg", {}) or {}))
        if ws.derived.relative_pressure_array is not None:
            ws.add_object(name="relative_pressure_volume", kind=ObjectKind.METRIC,
                          data_key="relative_pressure_volume", visible=False, opacity=float(np.clip(ws.derived_params.relative_pressure_layer_opacity, 0.0, 1.0)),
                          scalars="RelativePressure", cmap="RdBu_r", clim=relative_pressure_clim, dynamic=True,
                          show_scalar_bar=bool(render_cfg.get("relative_pressure_show_scalar_bar", render_cfg.get("pressure_gradient_show_scalar_bar", True))), scalar_bar_title="Relative Pressure (Pa)",
                          scalar_bar_cfg=dict(render_cfg.get("relative_pressure_bar_cfg", render_cfg.get("pressure_gradient_bar_cfg", {}) or {}) or {}))


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

    def load_data(self, ws, log, input_source=None, progress_callback=None):
        path = ws.paths.segmask_path or ws.paths.flow_path
        if input_source is None and not path:
            raise ValueError("data path is empty")
        load_target = path if input_source is None else input_source
        load_kwargs = {
            "correction_config": ws.loader_params.background_phase_correction,
        }
        dicom_overrides = ws.loader_params.dicom_parameter_overrides.to_loader_kwargs()
        if dicom_overrides:
            load_kwargs["parameter_overrides"] = dicom_overrides
        load_kwargs["dicom_read_workers"] = int(getattr(ws.loader_params, "dicom_read_workers", 1) or 1)
        if progress_callback is not None:
            load_kwargs["progress_callback"] = progress_callback
        try:
            data = load_input_data(load_target, **load_kwargs)
        except TypeError as exc:
            if "unexpected keyword argument" not in str(exc):
                raise
            data = load_input_data(load_target)
        flow = np.asarray(data.flow, dtype=np.float32)
        mag = np.asarray(data.mag, dtype=np.float32)
        seg = None if data.segmentation is None else np.asarray(data.segmentation, dtype=np.int16)

        ws.segmentation = ws.segmentation.__class__()
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
        ws.mag_raw = mag
        ws.input_state.source_format = str(data.source_format or "")
        ws.input_state.source_group = data.source_group
        ws.input_state.metadata = dict(data.metadata or {})
        ws.input_state.capabilities = data.capabilities
        ws.source_sigma = None if data.sigma is None else np.asarray(data.sigma, dtype=np.float32)
        ws.source_tke_array = None if data.tke_array is None else np.asarray(data.tke_array, dtype=np.float32)
        ws.derived.tke_array = None
        ws.derived.tke_volume = None
        ws.derived.pressure_gradient_array = None
        ws.derived.pressure_gradient_magnitude = None
        ws.derived.pressure_gradient_peak = None
        ws.derived.pressure_gradient_support_mask = None
        ws.derived.pressure_gradient_display_clim = None
        ws.derived.relative_pressure_array = None
        ws.derived.relative_pressure_peak = None
        ws.derived.relative_pressure_display_clim = None
        ws.derived.centerline_pressure_profiles = []
        ws.derived.wss_surfaces = []
        ws.derived.wss_volume = None
        ws.derived.pixelwise_export = {}
        ws.derived.plane_pixelwise_file = ""
        ws.derived.pwv_results = []
        ws.derived.pwv_planes = []
        ws.derived.pwv_file = ""
        ws.data_loaded = True

        for data_key in ["segmask_raw_surface", "segmask_pre_surface", "wss_surface_live", "tke_volume", "pressure_gradient_volume", "relative_pressure_volume"]:
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
        msg += f" caps={ws.input_state.capabilities.to_dict()}"
        log(msg)
        return msg

    def preprocess(self, ws):
        if ws.segmask_raw is None:
            raise ValueError("segmask_raw is None")
        from ..algorithms.preprocess import majority_vote_labels_3d, remove_small_cc_from_labeled_mask

        previous_groups = dict(ws.multilabel_groups or {})
        ws.segmask_labels = filter_segmask_labels(ws.segmask_raw)
        voted_labels_3d = majority_vote_labels_3d(ws.segmask_labels)
        if ws.skeleton_params.remove_small_cc:
            voted_labels_3d = remove_small_cc_from_labeled_mask(
                voted_labels_3d,
                ws.resolution,
                ws.skeleton_params.min_cc_volume_mm3,
            )
        specs = self._build_group_specs(ws, voted_labels_3d)
        ws.group_order = [str(spec["name"]) for spec in specs]
        ws.multilabel_groups = {}
        time_count = max(1, int(ws.time_count()))
        binary_shape = voted_labels_3d.shape + (time_count,)
        global_binary = np.zeros(binary_shape, dtype=bool)
        global_mask_3d = np.zeros(voted_labels_3d.shape, dtype=bool)
        for spec in specs:
            group_name = str(spec["name"])
            labels = [int(x) for x in spec["labels"]]
            group_mask_3d = np.isin(voted_labels_3d, labels)
            group_mask_3d = largest_connected_component(group_mask_3d)
            group_binary = self._repeat_mask_to_time(group_mask_3d, time_count)
            group_params = ws.skeleton_params.params_for_group(group_name)
            processed_mask_3d = preprocess_mask_for_skeleton(group_mask_3d, group_params, resolution=ws.resolution)
            processed_mask_3d = largest_connected_component(processed_mask_3d)
            previous_state = dict(previous_groups.get(group_name, {})) if isinstance(previous_groups.get(group_name, {}), dict) else {}
            global_binary |= np.asarray(group_binary, dtype=bool)
            global_mask_3d |= np.asarray(processed_mask_3d, dtype=bool)
            ws.multilabel_groups[group_name] = {
                "labels": labels,
                "browser_color": str(spec["browser_color"]),
                "scene_color": str(spec["scene_color"]),
                "segmask_binary": np.asarray(group_binary, dtype=bool),
                "segmask_3d": np.asarray(processed_mask_3d, dtype=bool),
                "clean_mask_3d": np.asarray(group_mask_3d, dtype=bool),
                "graph": previous_state.get("graph", GraphData()),
                "branch_labels": previous_state.get("branch_labels"),
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
        ws.segmask_binary = np.asarray(global_binary, dtype=bool)
        ws.segmask_3d = np.asarray(global_mask_3d, dtype=bool)
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
                opacity=0.15,
                color=ws.skeleton_params.scene_color_for_group(group_name, "scene"),
            )

    def run_step(self, ws, step, log):
        dispatch = {
            StepId.GENERATE_SKELETON: self._step_generate_skeleton,
            StepId.EDIT_SKELETON: self._step_edit_skeleton,
            StepId.GENERATE_GRAPH: self._step_generate_graph,
            StepId.EDIT_GRAPH: self._step_edit_graph,
            StepId.GENERATE_PLANES: self._step_generate_planes,
            StepId.EDIT_PLANES: self._step_edit_planes,
            StepId.COMPUTE_PWV: self._step_compute_pwv,
            StepId.GENERATE_STREAMLINES: self._step_generate_streamlines,
            StepId.PLANE_STREAMLINES: self._step_plane_streamlines,
            StepId.COMPUTE_PLANE_METRICS: self._step_compute_plane_metrics,
            StepId.COMPUTE_DERIVED_METRICS: self._step_compute_derived_metrics,
        }
        return dispatch[step](ws)

    def _step_generate_skeleton(self, ws):
        if ws.segmask_raw is None:
            return StepResult(
                StepId.GENERATE_SKELETON, True, True,
                self._missing_segmentation_message(ws, "Skeleton"),
            )
        self._clear_pwv_state(ws)
        self.preprocess(ws)
        points_all = []
        skeleton_mask = np.zeros_like(ws.segmask_3d, dtype=bool)
        ws.remove_object_by_data_key("skeleton_points")
        ws.remove_object_by_data_key("skeleton_mask_surface")
        ws.remove_object_by_data_key("segmask_3d_surface")
        ws.remove_objects_by_prefix("skeleton_")
        for group_name in ws.group_order:
            group_state = ws.multilabel_groups.get(group_name, {})
            processed = np.asarray(group_state.get("segmask_3d"), dtype=bool)
            pts, mask = generate_skeleton_from_mask3d(processed, ws.resolution)
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
        ws.pipeline.mark_done(StepId.GENERATE_SKELETON)
        return StepResult(StepId.GENERATE_SKELETON, True, False, f"Skeleton: {len(ws.skeleton_points)} points groups={len(ws.group_order)}")

    def _step_edit_skeleton(self, ws):
        ws.pipeline.mark_done(StepId.EDIT_SKELETON, skipped=True)
        return StepResult(StepId.EDIT_SKELETON, True, True, "Skeleton edit")

    def _step_generate_graph(self, ws):
        if ws.segmask_raw is None:
            return StepResult(
                StepId.GENERATE_GRAPH, True, True,
                self._missing_segmentation_message(ws, "Graph"),
            )
        self._clear_pwv_state(ws)
        if ws.skeleton_points is None or len(ws.skeleton_points) == 0:
            skel_result = self._step_generate_skeleton(ws)
            if skel_result.skipped or not skel_result.success:
                return StepResult(StepId.GENERATE_GRAPH, skel_result.success, True, skel_result.message)
        graph_points = []
        graph_edges = []
        branch_labels = np.zeros(ws.segmask_3d.shape, dtype=np.int16)
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
        for group_name in ws.group_order:
            group_state = ws.multilabel_groups.get(group_name, {})
            local_points = np.asarray(group_state.get("skeleton_points"), dtype=float).reshape(-1, 3) if group_state.get("skeleton_points") is not None else np.empty((0, 3), dtype=float)
            local_graph = build_graph_from_points(local_points, ws.resolution)
            group_state["graph"] = local_graph
            flow_for_orientation = None
            local_binary = np.asarray(group_state.get("segmask_binary"), dtype=bool)
            if ws.flow_raw is not None and local_binary.size:
                flow_for_orientation = ws.flow_raw * local_binary[..., None]
            local_labels, local_paths, local_node_paths, local_path_info, local_forks = segment_vessels_from_graph_and_mask(
                np.asarray(group_state.get("segmask_3d"), dtype=bool),
                local_graph,
                ws.resolution,
                flow_xyzt3=flow_for_orientation,
                segmask_binary_4d=local_binary,
                origin=ws.origin,
            )
            if np.any(local_labels > 0):
                branch_labels[local_labels > 0] = local_labels[local_labels > 0] + int(path_offset)
            adjusted_paths = [np.asarray(p, dtype=float) for p in local_paths]
            adjusted_node_paths = [[int(node) + int(graph_node_offset) for node in path_nodes] for path_nodes in local_node_paths]
            adjusted_path_info = []
            for item in local_path_info:
                payload = dict(item)
                payload["path_index"] = int(payload.get("path_index", 0)) + int(path_offset)
                payload["fork_ids"] = [int(x) + len(forks) for x in payload.get("fork_ids", [])]
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
            group_state["branch_labels"] = np.asarray(local_labels, dtype=np.int16)
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
        ws.centerline_paths = centerline_paths
        ws.centerline_node_paths = centerline_node_paths
        ws.path_info = path_info
        ws.forks = forks
        ws.selected_path_index = -1

        ws.pipeline.mark_done(StepId.GENERATE_GRAPH)
        return StepResult(StepId.GENERATE_GRAPH, True, False,
                          f"Graph: {len(ws.graph.points)} nodes, {len(ws.graph.edges)} edges | "
                          f"paths={len(ws.centerline_paths)} forks={len(ws.forks)}")

    def _step_edit_graph(self, ws):
        ws.pipeline.mark_done(StepId.EDIT_GRAPH, skipped=True)
        return StepResult(StepId.EDIT_GRAPH, True, True, "Graph edit")

    
    def _ensure_derived_metrics(
        self,
        ws,
        save_pixelwise=False,
        refresh_scene_objects=False,
        compute_wss=True,
        compute_tke=True,
        compute_pressure_gradient=True,
    ):
        has_wss = ws.derived.wss_volume is not None and np.size(ws.derived.wss_volume) > 0
        has_pg = ws.derived.pressure_gradient_array is not None and np.size(ws.derived.pressure_gradient_array) > 0
        has_tke = ws.derived.tke_array is not None or ws.derived.tke_volume is not None
        source_tke = ws.source_tke_array
        source_sigma = ws.source_sigma if ws.input_state.capabilities.has_complex_source else None
        need_tke = bool(compute_tke and (source_tke is not None or source_sigma is not None))
        has_requested = True
        if compute_wss:
            has_requested = has_requested and has_wss
        if compute_pressure_gradient:
            has_requested = has_requested and has_pg
        if need_tke:
            has_requested = has_requested and has_tke
        if save_pixelwise and not ws.derived.pixelwise_export:
            has_requested = False
        if has_requested:
            if refresh_scene_objects:
                self._register_derived_scene_objects(ws)
            return ws.derived
        self.preprocess(ws)
        dp = ws.derived_params
        result = compute_derived_metrics(
            flow=ws.flow_raw * ws.segmask_binary[..., None],
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
            save_pixelwise=save_pixelwise,
            tke_array=source_tke,
            sigma=source_sigma,
            rr=ws.rr,
            pressure_gradient_smoothing_sigma=dp.pressure_gradient_smoothing_sigma,
            pressure_gradient_support_erosion_iters=dp.pressure_gradient_support_erosion_iters,
            pressure_gradient_use_convective_acceleration=dp.pressure_gradient_use_convective_acceleration,
            pressure_method=dp.pressure_method,
            centerline_paths=ws.centerline_paths_smooth if len(ws.centerline_paths_smooth) > 0 else ws.centerline_paths,
            compute_wss=bool(compute_wss),
            compute_tke=bool(compute_tke),
            compute_pressure_gradient=bool(compute_pressure_gradient),
            wss_smoothing_iteration=dp.wss_smoothing_iteration,
            wss_viscosity=dp.wss_viscosity,
            wss_inward_distance=dp.wss_inward_distance,
            wss_parabolic_fitting=dp.wss_parabolic_fitting,
            wss_no_slip_condition=dp.wss_no_slip_condition,
            tke_rho=dp.tke_rho,
            pressure_gradient_rho=dp.pressure_gradient_rho,
            pressure_gradient_viscosity=dp.pressure_gradient_viscosity,
        )
        if compute_wss:
            ws.derived.wss_surfaces = result["wss_surfaces"]
            ws.derived.wss_volume = result.get("wss_volume")
        if compute_tke:
            ws.derived.tke_volume = result["tke_volume"]
            ws.derived.tke_array = result.get("tke_array")
        if compute_pressure_gradient:
            ws.derived.pressure_gradient_array = result.get("pressure_gradient_array")
            ws.derived.pressure_gradient_magnitude = result.get("pressure_gradient_magnitude")
            ws.derived.pressure_gradient_peak = result.get("pressure_gradient_peak")
            ws.derived.pressure_gradient_support_mask = result.get("pressure_gradient_support_mask")
            ws.derived.pressure_gradient_display_clim = result.get("pressure_gradient_display_clim")
            ws.derived.relative_pressure_array = result.get("relative_pressure_array")
            ws.derived.relative_pressure_peak = result.get("relative_pressure_peak")
            ws.derived.relative_pressure_display_clim = result.get("relative_pressure_display_clim")
            ws.derived.centerline_pressure_profiles = list(result.get("centerline_pressure_profiles", []) or [])
        ws.derived.streamlines = []
        ws.derived.pixelwise_export = result.get("pixelwise_export", {})
        if refresh_scene_objects:
            self._register_derived_scene_objects(ws)
        return ws.derived

    def _compute_plane_metrics_internal(self, ws, save=True, use_multithread=False, include_derived=True):
        if not ws.has_flow():
            return [], {}, "Plane metrics skipped: no flow"
        if ws.segmask_raw is None:
            return [], {}, self._missing_segmentation_message(ws, "Plane metrics")
        if ws.segmask_binary is None:
            self.preprocess(ws)
        if include_derived:
            self._ensure_derived_metrics(ws, save_pixelwise=False, refresh_scene_objects=False)
        # Prefer the smoothed centerlines (better local tangents) but fall back
        # to the raw ordered ones if the smoothing step hasn't been run yet.
        paths_for_tangent = ws.centerline_paths_smooth if len(ws.centerline_paths_smooth) > 0 else ws.centerline_paths
        if use_multithread:
            metrics, qc = compute_plane_metrics_multithread(
                ws.flow_raw, ws.segmask_binary, ws.resolution, ws.origin, ws.planes,
                RR=ws.rr, branch_labels_3d=ws.branch_labels,
                path_info=ws.path_info, forks=ws.forks, paths=paths_for_tangent,
                return_qc=True)
        else:
            metrics, qc = compute_plane_metrics(
                ws.flow_raw, ws.segmask_binary, ws.resolution, ws.origin, ws.planes,
                RR=ws.rr, branch_labels_3d=ws.branch_labels,
                path_info=ws.path_info, forks=ws.forks, paths=paths_for_tangent,
                return_qc=True)
        if include_derived:
            metrics, plane_pixelwise = augment_plane_metrics_with_derived(
                metrics, ws.planes, ws.segmask_binary, ws.resolution, ws.origin,
                branch_labels_3d=ws.branch_labels,
                tke_array=ws.derived.tke_array,
                pressure_gradient_array=ws.derived.pressure_gradient_array,
                relative_pressure_array=ws.derived.relative_pressure_array,
                wss_surfaces=ws.derived.wss_surfaces,
            )
        else:
            metrics = [dict(metric) for metric in metrics]
            plane_pixelwise = []
            for idx, plane in enumerate(ws.planes):
                plane_pixelwise.append(
                    {
                        "plane_index": int(idx),
                        "center": np.asarray(plane.center, dtype=float).reshape(3).tolist(),
                        "normal": np.asarray(plane.normal, dtype=float).reshape(3).tolist(),
                        "label": int(getattr(plane, "label", 0) or 0),
                        "path_index": int(getattr(plane, "path_index", -1)),
                        "timepoints": [],
                    }
                )
        ws.derived.plane_metrics = metrics
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
            with open(plane_metric_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)
            with open(qc_path, "w", encoding="utf-8") as f:
                json.dump(qc, f, ensure_ascii=False, indent=2)
            save_plane_pixelwise_h5(plane_pixelwise_path, plane_pixelwise, rr_ms=ws.rr, source_format=ws.input_state.source_format)
            ws.derived.plane_pixelwise_file = plane_pixelwise_path
            msg += f" saved={plane_metric_path} qc={qc_path} pixelwise={plane_pixelwise_path}"
        return metrics, qc, msg

    def _save_planes_json(self, ws):
        out_dir = self._output_dir(ws)
        out_path = os.path.join(out_dir, "planes.json")
        payload = []
        origin = np.asarray(ws.origin, dtype=float).reshape(3)
        for i, p in enumerate(ws.planes):
            center_local = np.asarray(p.center, dtype=float).reshape(3)
            item = {
                "plane_index": int(i),
                "center": center_local.tolist(),
                "center_world": (center_local + origin).tolist(),
                "normal": np.asarray(p.normal).tolist(),
                "label": int(p.label),
                "path_index": int(p.path_index),
                "distance": float(p.distance),
            }
            if p.metrics:
                item.update(json.loads(json.dumps(p.metrics, ensure_ascii=False)))
            if 0 <= int(p.path_index) < len(ws.path_info):
                item["path_info"] = ws.path_info[int(p.path_index)]
            payload.append(item)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return out_path

    def _step_generate_planes(self, ws):
        if ws.segmask_raw is None:
            return StepResult(
                StepId.GENERATE_PLANES, True, True,
                self._missing_segmentation_message(ws, "Planes"),
            )
        self._clear_pwv_state(ws)
        if ws.graph is None or len(ws.graph.points) == 0:
            graph_result = self._step_generate_graph(ws)
            if graph_result.skipped or not graph_result.success:
                return StepResult(StepId.GENERATE_PLANES, graph_result.success, True, graph_result.message)

        ws.remove_objects_by_prefix("smooth_path_")
        ws.remove_objects_by_prefix("path_arrow_")
        pgp = ws.plane_gen_params
        ws.remove_objects_by_prefix("plane_")
        planes = []
        smooth_paths = []
        ws.clear_pathlines()
        ws.pathline_colors = {}
        for group_name in ws.group_order:
            group_state = ws.multilabel_groups.get(group_name, {})
            local_paths = list(group_state.get("centerline_paths", []))
            local_planes, local_smooth_paths = generate_planes_from_paths(
                local_paths,
                cross_section_distance=pgp.cross_section_distance,
                start_distance=pgp.start_distance,
                end_distance=pgp.end_distance,
                smoothing_window=pgp.smoothing_window * pgp.inter_time,
                smoothing_polyorder=pgp.smoothing_polyorder,
                inter_time=pgp.inter_time,
                use_center_plane=pgp.use_center_plane,
            )
            path_offset = int(group_state.get("path_index_offset", 0))
            plane_offset = len(planes)
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

        planes_path = self._save_planes_json(ws)
        ws.pipeline.mark_done(StepId.GENERATE_PLANES)
        msg = f"Planes: {len(ws.planes)} paths={len(ws.centerline_paths_smooth)} forks={len(ws.forks)} saved={planes_path}"
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
        if ws.segmask_raw is None:
            return StepResult(
                StepId.GENERATE_STREAMLINES, True, True,
                self._missing_segmentation_message(ws, "Streamlines"),
            )
        if ws.flow_raw is None or ws.segmask_3d is None:
            return StepResult(StepId.GENERATE_STREAMLINES, True, True, "Streamlines skipped: no flow or mask")
        self.preprocess(ws)
        ws.streamline_seeds = generate_seed_points(
            ws.segmask_3d,
            ws.resolution,
            ws.origin,
            ratio=ws.streamline_params.seed_ratio,
            rng_seed=ws.streamline_params.rng_seed,
            min_seeds=ws.streamline_params.min_seeds,
        )
        ws.streamline_cache.clear()
        ws.streamline_active = True
        ws.remove_object_by_data_key("streamlines_live")
        render_cfg = dict(getattr(ws, "render_settings", {}) or {})
        ws.add_object(
            name="streamlines", kind=ObjectKind.FLOW,
            data_key="streamlines_live", visible=True, opacity=1.0,
            scalars="Velocity", cmap="turbo", clim=render_cfg.get("streamline_clim"), dynamic=True,
            show_scalar_bar=bool(render_cfg.get("streamline_show_scalar_bar", True)), scalar_bar_title="Velocity (m/s)",
            scalar_bar_cfg=dict(render_cfg.get("streamline_bar_cfg", {}) or {}),
            tube_radius=ws.streamline_params.tube_radius)
        ws.pipeline.mark_done(StepId.GENERATE_STREAMLINES)
        p = ws.streamline_params
        param_msg = (f"Streamlines enabled: seed_ratio={p.seed_ratio} max_steps={p.max_steps} "
                     f"min_seeds={p.min_seeds} terminal_speed={p.terminal_speed} rng_seed={p.rng_seed}")
        return StepResult(StepId.GENERATE_STREAMLINES, True, False, param_msg)

    def _step_plane_streamlines(self, ws):
        if ws.segmask_raw is None:
            return StepResult(
                StepId.PLANE_STREAMLINES, True, True,
                self._missing_segmentation_message(ws, "Pathlines"),
            )
        if ws.flow_raw is None or ws.segmask_3d is None:
            return StepResult(StepId.PLANE_STREAMLINES, True, True, "Pathlines skipped: no flow or mask")
        if len(ws.planes) == 0:
            return StepResult(StepId.PLANE_STREAMLINES, True, True, "Pathlines skipped: no planes")
        self.preprocess(ws)
        active_indices = list(range(len(ws.planes)))
        ws.clear_pathlines()
        ws.active_pathline_plane_indices = active_indices
        for plane_idx in active_indices:
            group_name = self._plane_group_name(ws, plane_idx)
            ws.add_object(
                name=self._indexed_object_name("pathline", plane_idx), kind=ObjectKind.FLOW,
                data_key=self._plane_data_key("pathline", ws, plane_idx), group_name=group_name,
                browser_color=ws.skeleton_params.browser_color_for_group(group_name) if group_name else "",
                visible=True, opacity=1.0,
                color=ws.pathline_color_for_plane(plane_idx), dynamic=True,
                show_scalar_bar=False, tube_radius=ws.streamline_params.tube_radius)
        ws.pipeline.mark_done(StepId.PLANE_STREAMLINES)
        p = ws.streamline_params
        return StepResult(
            StepId.PLANE_STREAMLINES,
            True,
            False,
            f"Pathlines enabled for {len(active_indices)} planes: seed_ratio={p.seed_ratio} min_seeds={p.min_seeds} max_steps={p.max_steps} terminal_speed={p.terminal_speed} rng_seed={p.rng_seed} color={p.pathline_color}",
        )

    def _step_compute_plane_metrics(self, ws):
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
        _, _, msg = self._compute_plane_metrics_internal(ws, save=True, use_multithread=use_mt, include_derived=True)
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
        self._ensure_derived_metrics(ws, save_pixelwise=False, refresh_scene_objects=True)
        has_tke = ws.derived.tke_array is not None or ws.derived.tke_volume is not None
        msg = f"Derived: Nt={len(ws.derived.wss_surfaces)}"
        if not has_tke:
            msg += " tke=unavailable"
        ws.pipeline.mark_done(StepId.COMPUTE_DERIVED_METRICS)
        return StepResult(StepId.COMPUTE_DERIVED_METRICS, True, False, msg)
