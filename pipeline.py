import json
import os
import numpy as np

from models import StepId, ObjectKind
from algorithms import (
    load_h5_data,
    filter_segmask_labels, merge_segmask_to_3d,
    preprocess_mask_for_skeleton,
    generate_skeleton_from_mask3d, build_graph_from_points,
    segment_vessels_from_graph_and_mask,
    generate_planes_from_paths,
    compute_plane_metrics, compute_derived_metrics,
    compute_plane_metrics_multithread,
    generate_seed_points,
)


class StepResult:
    def __init__(self, step, success=True, skipped=False, message="", outputs=None):
        self.step = step
        self.success = success
        self.skipped = skipped
        self.message = message
        self.outputs = outputs or []


class PipelineEngine:
    def _output_dir(self, ws):
        out_dir = getattr(ws.paths, "output_dir", "") or ""
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            return out_dir
        base = ws.paths.segmask_path or ws.paths.flow_path or "."
        out_dir = os.path.dirname(base) or "."
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def load_data(self, ws, log):
        path = ws.paths.segmask_path or ws.paths.flow_path
        if not path:
            raise ValueError("data path is empty")
        data = load_h5_data(path)
        flow = np.asarray(data["flow"], dtype=np.float32)
        mag = np.asarray(data["mag"], dtype=np.float32)
        seg = np.asarray(data["segmask"], dtype=np.int16)
        if flow.ndim == 4 and flow.shape[-1] == 3:
            flow = flow[..., np.newaxis, :]
        if mag.ndim == 3:
            mag = mag[..., np.newaxis]
        if seg.ndim == 3:
            seg = np.repeat(seg[..., np.newaxis], flow.shape[3], axis=3)
        elif seg.ndim == 4 and seg.shape[3] == 1 and flow.shape[3] > 1:
            seg = np.repeat(seg, flow.shape[3], axis=3)
        if seg.shape[3] != flow.shape[3]:
            raise ValueError(f"segmask time dimension {seg.shape[3]} != flow {flow.shape[3]}")

        ws.segmask_raw = seg
        ws.resolution = np.asarray(data["resolution"], dtype=float).reshape(3)
        ws.origin = np.asarray(data.get("origin", [0.0, 0.0, 0.0]), dtype=float).reshape(3)
        ws.venc = np.asarray(data["venc"], dtype=float).reshape(-1)
        ws.rr = float(data.get("rr", 1000.0))
        ws.current_t = 0
        ws.flow_raw = flow
        ws.mag_raw = mag
        ws.data_loaded = True

        ws.remove_object_by_data_key("segmask_raw_surface")
        ws.add_object(name="segmask_raw", kind=ObjectKind.SEGMENTATION,
                      data_key="segmask_raw_surface", visible=True, opacity=0.3,
                      scalars="label", cmap="tab10", dynamic=True,
                      show_scalar_bar=True, scalar_bar_title="Label")

        ulabels = ws.unique_labels()
        msg = f"Loaded: segmask={ws.segmask_raw.shape} labels={ulabels} rr={ws.rr}"
        msg += f" flow={ws.flow_raw.shape} mag={ws.mag_raw.shape}"
        msg += f" origin={ws.origin.tolist()}"
        log(msg)
        return msg

    def preprocess(self, ws):
        if ws.segmask_raw is None:
            raise ValueError("segmask_raw is None")
        ws.segmask_labels = filter_segmask_labels(ws.segmask_raw)
        ws.segmask_binary = (ws.segmask_labels > 0).astype(bool)
        ws.segmask_3d = merge_segmask_to_3d(ws.segmask_binary)
        ws.set_object_visible_by_data_key("segmask_raw_surface", False)
        ws.remove_object_by_data_key("segmask_pre_surface")
        ws.add_object(name="segmask_preprocessed", kind=ObjectKind.SEGMENTATION,
                      data_key="segmask_pre_surface", visible=True, opacity=0.25,
                      scalars="label", cmap="tab10", dynamic=True,
                      show_scalar_bar=True, scalar_bar_title="Label")

    def run_step(self, ws, step, log):
        dispatch = {
            StepId.GENERATE_SKELETON: self._step_generate_skeleton,
            StepId.EDIT_SKELETON: self._step_edit_skeleton,
            StepId.GENERATE_GRAPH: self._step_generate_graph,
            StepId.EDIT_GRAPH: self._step_edit_graph,
            StepId.GENERATE_PLANES: self._step_generate_planes,
            StepId.EDIT_PLANES: self._step_edit_planes,
            StepId.GENERATE_STREAMLINES: self._step_generate_streamlines,
            StepId.PLANE_STREAMLINES: self._step_plane_streamlines,
            StepId.COMPUTE_PLANE_METRICS: self._step_compute_plane_metrics,
            StepId.COMPUTE_DERIVED_METRICS: self._step_compute_derived_metrics,
        }
        return dispatch[step](ws)

    def _step_generate_skeleton(self, ws):
        self.preprocess(ws)
        if ws.skeleton_params.remove_small_cc:
            from algorithms import remove_small_cc_from_binary_mask
            ws.segmask_binary = remove_small_cc_from_binary_mask(
                ws.segmask_binary, ws.resolution, ws.skeleton_params.min_cc_volume_mm3)
            ws.segmask_3d = merge_segmask_to_3d(ws.segmask_binary)
        processed = preprocess_mask_for_skeleton(ws.segmask_3d, ws.skeleton_params, resolution=ws.resolution)
        pts, mask = generate_skeleton_from_mask3d(processed, ws.resolution)
        ws.skeleton_points = pts
        ws.skeleton_mask = mask
        ws.remove_object_by_data_key("skeleton_points")
        ws.remove_object_by_data_key("skeleton_mask_surface")
        ws.remove_object_by_data_key("segmask_3d_surface")
        ws.add_object(name="skeleton_points", kind=ObjectKind.SKELETON,
                      data_key="skeleton_points", visible=True, opacity=1.0,
                      color="red", point_size=8)
        # ws.add_object(name="skeleton_mask", kind=ObjectKind.SKELETON,
                    #   data_key="skeleton_mask_surface", visible=False, opacity=0.15, color="yellow")
        ws.add_object(name="segmask_mesh", kind=ObjectKind.SEGMENTATION,
                      data_key="segmask_3d_surface", visible=True, opacity=0.15,
                      color="gray")
        ws.pipeline.mark_done(StepId.GENERATE_SKELETON)
        return StepResult(StepId.GENERATE_SKELETON, True, False, f"Skeleton: {len(pts)} points")

    def _step_edit_skeleton(self, ws):
        ws.pipeline.mark_done(StepId.EDIT_SKELETON, skipped=True)
        return StepResult(StepId.EDIT_SKELETON, True, True, "Skeleton edit")

    def _step_generate_graph(self, ws):
        if ws.skeleton_points is None or len(ws.skeleton_points) == 0:
            self._step_generate_skeleton(ws)
        graph = build_graph_from_points(ws.skeleton_points, ws.resolution)
        ws.graph = graph

        flow_for_orientation = None
        if ws.flow_raw is not None and ws.segmask_binary is not None:
            flow_for_orientation = ws.flow_raw * ws.segmask_binary[..., None]
        labels, paths, node_paths, path_info, forks = segment_vessels_from_graph_and_mask(
            ws.segmask_3d, ws.graph, ws.resolution,
            flow_xyzt3=flow_for_orientation,
            segmask_binary_4d=ws.segmask_binary,
            origin=ws.origin,
        )
        ws.branch_labels = labels
        ws.centerline_paths = [np.asarray(p, dtype=float) for p in paths]
        ws.centerline_node_paths = [list(map(int, p)) for p in node_paths]
        ws.path_info = path_info
        ws.forks = forks
        ws.selected_path_index = -1

        ws.remove_object_by_data_key("graph_lines")
        ws.add_object(name="graph_lines", kind=ObjectKind.GRAPH,
                      data_key="graph_lines", visible=True, opacity=1.0,
                      color="blue", line_width=2)

        ws.remove_objects_by_prefix("path_")
        ws.remove_objects_by_prefix("smooth_path_")
        ws.remove_objects_by_prefix("path_arrow_")
        ws.remove_object_by_data_key("fork_markers")

        if len(ws.forks) > 0:
            ws.add_object(name="Forks", kind=ObjectKind.AUX,
                          data_key="fork_markers", visible=True, opacity=1.0,
                          color="magenta", point_size=12)

        ws.pipeline.mark_done(StepId.GENERATE_GRAPH)
        return StepResult(StepId.GENERATE_GRAPH, True, False,
                          f"Graph: {len(graph.points)} nodes, {len(graph.edges)} edges | "
                          f"paths={len(ws.centerline_paths)} forks={len(ws.forks)}")

    def _step_edit_graph(self, ws):
        ws.pipeline.mark_done(StepId.EDIT_GRAPH, skipped=True)
        return StepResult(StepId.EDIT_GRAPH, True, True, "Graph edit")

    def _compute_plane_metrics_internal(self, ws, save=True, use_multithread=False):
        if not ws.has_flow():
            return [], {}, "Plane metrics skipped: no flow"
        if ws.segmask_binary is None:
            self.preprocess(ws)
        if use_multithread:
            metrics, qc = compute_plane_metrics_multithread(
                ws.flow_raw, ws.segmask_binary, ws.resolution, ws.origin, ws.planes,
                RR=ws.rr, branch_labels_3d=ws.branch_labels,
                path_info=ws.path_info, forks=ws.forks, return_qc=True)
        else:
            metrics, qc = compute_plane_metrics(
                ws.flow_raw, ws.segmask_binary, ws.resolution, ws.origin, ws.planes,
                RR=ws.rr, branch_labels_3d=ws.branch_labels,
                path_info=ws.path_info, forks=ws.forks, return_qc=True)
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
            with open(plane_metric_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)
            with open(qc_path, "w", encoding="utf-8") as f:
                json.dump(qc, f, ensure_ascii=False, indent=2)
            msg += f" saved={plane_metric_path} qc={qc_path}"
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
        if ws.graph is None or len(ws.graph.points) == 0:
            self._step_generate_graph(ws)

        if len(ws.centerline_paths) == 0:
            flow_for_orientation = None
            if ws.flow_raw is not None and ws.segmask_binary is not None:
                flow_for_orientation = ws.flow_raw * ws.segmask_binary[..., None]
            labels, paths, node_paths, path_info, forks = segment_vessels_from_graph_and_mask(
                ws.segmask_3d, ws.graph, ws.resolution,
                flow_xyzt3=flow_for_orientation,
                segmask_binary_4d=ws.segmask_binary,
                origin=ws.origin,
            )
            ws.branch_labels = labels
            ws.centerline_paths = [np.asarray(p, dtype=float) for p in paths]
            ws.centerline_node_paths = [list(map(int, p)) for p in node_paths]
            ws.path_info = path_info
            ws.forks = forks
            ws.selected_path_index = -1

        ws.remove_objects_by_prefix("smooth_path_")
        ws.remove_objects_by_prefix("path_arrow_")
        ws.remove_object_by_data_key("fork_markers")

        pgp = ws.plane_gen_params
        planes, smooth_paths = generate_planes_from_paths(
            ws.centerline_paths,
            cross_section_distance=pgp.cross_section_distance,
            start_distance=pgp.start_distance,
            end_distance=pgp.end_distance,
            smoothing_window=pgp.smoothing_window * pgp.inter_time,
            smoothing_polyorder=pgp.smoothing_polyorder,
            inter_time=pgp.inter_time,
            use_center_plane=pgp.use_center_plane,
        )
        ws.planes = planes
        ws.centerline_paths_smooth = smooth_paths
        for i in range(len(ws.centerline_paths_smooth)):
            direction_text = ""
            if i < len(ws.path_info):
                direction_text = ws.path_info[i].get("direction_text", "")
            name = f"Path {i}" if not direction_text else f"Path {i} [{direction_text}]"
            ws.add_object(name=name, kind=ObjectKind.BRANCH,
                          data_key=f"smooth_path_{i}", visible=True, opacity=1.0,
                          color="red", line_width=3)
            # ws.add_object(name=f"Path {i} Arrow", kind=ObjectKind.AUX,
            #               data_key=f"path_arrow_{i}", visible=True, opacity=1.0,
            #               color="lime", line_width=2)
        if len(ws.forks) > 0:
            ws.remove_object_by_data_key("fork_markers")
            ws.add_object(name="Forks", kind=ObjectKind.AUX,
                          data_key="fork_markers", visible=True, opacity=1.0,
                          color="magenta", point_size=12)

        ws.remove_objects_by_prefix("plane_")
        for i in range(len(ws.planes)):
            ws.add_object(name=f"Plane {i}", kind=ObjectKind.PLANE,
                          data_key=f"plane_{i}", visible=True, opacity=0.6,
                          color="yellow", line_width=2)

        planes_path = self._save_planes_json(ws)

        metric_msg = ""
        if ws.has_flow() and len(ws.planes) > 0:
            _, _, metric_msg = self._compute_plane_metrics_internal(ws, save=False)
            ws.pipeline.mark_done(StepId.COMPUTE_PLANE_METRICS)
        ws.pipeline.mark_done(StepId.GENERATE_PLANES)
        msg = f"Planes: {len(ws.planes)} paths={len(ws.centerline_paths_smooth)} forks={len(ws.forks)} saved={planes_path}"
        if metric_msg:
            msg += f" | {metric_msg}"
        return StepResult(StepId.GENERATE_PLANES, True, False, msg)

    def _step_edit_planes(self, ws):
        ws.pipeline.mark_done(StepId.EDIT_PLANES, skipped=True)
        return StepResult(StepId.EDIT_PLANES, True, True, "Plane edit")

    def _step_generate_streamlines(self, ws):
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
        ws.add_object(
            name="streamlines", kind=ObjectKind.FLOW,
            data_key="streamlines_live", visible=True, opacity=1.0,
            scalars="Velocity", cmap="turbo", dynamic=True,
            show_scalar_bar=True, scalar_bar_title="Velocity (m/s)")
        ws.pipeline.mark_done(StepId.GENERATE_STREAMLINES)
        p = ws.streamline_params
        param_msg = (f"Streamlines enabled: seed_ratio={p.seed_ratio} max_steps={p.max_steps} "
                     f"min_seeds={p.min_seeds} terminal_speed={p.terminal_speed} rng_seed={p.rng_seed}")
        return StepResult(StepId.GENERATE_STREAMLINES, True, False, param_msg)

    def _step_plane_streamlines(self, ws):
        if ws.flow_raw is None or ws.segmask_3d is None:
            return StepResult(StepId.PLANE_STREAMLINES, True, True, "Plane streamlines skipped: no flow or mask")
        if len(ws.planes) == 0:
            return StepResult(StepId.PLANE_STREAMLINES, True, True, "Plane streamlines skipped: no planes")
        self.preprocess(ws)
        ws.plane_streamline_cache.clear()
        ws.plane_streamline_active = True
        ws.remove_object_by_data_key("plane_streamlines_live")
        ws.add_object(
            name="plane_streamlines", kind=ObjectKind.FLOW,
            data_key="plane_streamlines_live", visible=True, opacity=1.0,
            scalars="Velocity", cmap="turbo", dynamic=True,
            show_scalar_bar=True, scalar_bar_title="Velocity (m/s)")
        ws.pipeline.mark_done(StepId.PLANE_STREAMLINES)
        pidx = ws.plane_streamline_plane_idx
        return StepResult(StepId.PLANE_STREAMLINES, True, False,
                          f"Plane streamlines enabled from plane {pidx}")

    def _step_compute_plane_metrics(self, ws):
        if not ws.has_flow():
            return StepResult(StepId.COMPUTE_PLANE_METRICS, True, True, "Plane metrics skipped: no flow")
        if len(ws.planes) == 0:
            self._step_generate_planes(ws)
        use_mt = getattr(ws.derived_params, "use_multithread", False)
        _, _, msg = self._compute_plane_metrics_internal(ws, save=True, use_multithread=use_mt)
        self._save_planes_json(ws)
        ws.pipeline.mark_done(StepId.COMPUTE_PLANE_METRICS)
        return StepResult(StepId.COMPUTE_PLANE_METRICS, True, False, msg)

    def _step_compute_derived_metrics(self, ws):
        if not ws.has_flow():
            return StepResult(StepId.COMPUTE_DERIVED_METRICS, True, True, "Derived metrics skipped: no flow")
        self.preprocess(ws)
        dp = ws.derived_params
        result = compute_derived_metrics(
            flow=ws.flow_raw * ws.segmask_binary[..., None], mask4d=ws.segmask_binary, spacing=ws.resolution,
            origin=ws.origin,
            smoothing_iteration=dp.smoothing_iteration,
            viscosity=dp.viscosity,
            inward_distance=dp.inward_distance,
            parabolic_fitting=dp.parabolic_fitting,
            no_slip_condition=dp.no_slip_condition,
            step_size=dp.step_size,
            tube_radius=dp.tube_radius,
            rho=dp.rho,
            save_pixelwise=False)
        ws.derived.wss_surfaces = result["wss_surfaces"]
        ws.derived.wss_volume = result.get("wss_volume")
        ws.derived.tke_volume = result["tke_volume"]
        ws.derived.tke_array = result.get("tke_array")
        ws.derived.streamlines = []
        ws.derived.pixelwise_export = result.get("pixelwise_export", {})
        for dk in ["wss_surface_live", "tke_volume"]:
            ws.remove_object_by_data_key(dk)
        wss_max = float(np.nanmax(ws.derived.wss_volume)) if ws.derived.wss_volume is not None and np.size(ws.derived.wss_volume) else 0.0
        tke_max = float(np.nanmax(ws.derived.tke_array)) if ws.derived.tke_array is not None and np.size(ws.derived.tke_array) else 0.0
        ws.add_object(name="wss_surface", kind=ObjectKind.METRIC,
                      data_key="wss_surface_live", visible=False, opacity=1.0,
                      scalars="wss", cmap="jet", clim=(0.0, wss_max if wss_max > 0 else 1.0), dynamic=True,
                      show_scalar_bar=True, scalar_bar_title="WSS (Pa)")
        ws.add_object(name="tke_volume", kind=ObjectKind.METRIC,
                      data_key="tke_volume", visible=False, opacity=0.5,
                      scalars="TKE", cmap="hot", clim=(0.0, tke_max if tke_max > 0 else 1.0),
                      show_scalar_bar=True, scalar_bar_title="TKE (J/m\u00b3)")

        msg = f"Derived: Nt={len(ws.derived.wss_surfaces)}"
        ws.pipeline.mark_done(StepId.COMPUTE_DERIVED_METRICS)
        return StepResult(StepId.COMPUTE_DERIVED_METRICS, True, False, msg)