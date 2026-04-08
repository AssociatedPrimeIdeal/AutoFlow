import numpy as np
import pyvista as pv

from models import ObjectKind
from algorithms import (
    build_multilabel_surface_t,
    build_surface_from_mask3d,
    graph_to_polydata,
    generate_seed_points,
    generate_streamlines_at_t,
    generate_streamlines_from_plane_at_t,
)


def _parse_indexed_data_key(data_key, prefix):
    token = f"{prefix}_"
    if not isinstance(data_key, str) or not data_key.startswith(token):
        return None
    suffix = data_key[len(token):]
    if suffix.isdigit():
        return int(suffix)
    return None


def _path_polydata(path, origin):
    pts = np.asarray(path, dtype=float)
    if len(pts) == 0:
        return None
    poly = pv.PolyData(pts + np.asarray(origin, dtype=float).reshape(1, 3))
    if len(pts) >= 2:
        cells = np.empty((len(pts) - 1, 3), dtype=np.int64)
        cells[:, 0] = 2
        cells[:, 1] = np.arange(len(pts) - 1)
        cells[:, 2] = np.arange(1, len(pts))
        poly.lines = cells.ravel()
    return poly


class SceneController:
    def __init__(self, plotter, workspace, logger):
        self.plotter = plotter
        self.workspace = workspace
        self.logger = logger
        self._axes_shown = True
        self._mesh_cache = {}
        self._tracked_actors = {}
        self._saved_camera = None
        self._playback_active = False
        self._highlight_plane_uid = None
        self._highlight_plane_actor = None
        self._highlight_path_uid = None
        self._highlight_path_actor = None
        self._context_path_actors = []
        self._highlight_fork_actor = None
        self._plane_pick_obs_id = None
        self._path_pick_obs_id = None
        self._plane_pick_callback = None
        self._path_pick_callback = None
        self._shared_pick_obs_id = None

    def initialize(self):
        self.plotter.set_background("white")
        self.plotter.add_axes(line_width=2)
        self.plotter.reset_camera()

    def reset_scene(self):
        try:
            self.plotter.clear()
        except Exception:
            try:
                self.plotter.renderer.RemoveAllViewProps()
            except Exception:
                pass
        for obj in self.workspace.scene_objects.values():
            obj.actor = None
            obj.label_actor = None
        self._tracked_actors.clear()
        self._mesh_cache.clear()
        self._remove_plane_highlight()
        self._remove_path_highlight()
        self.initialize()

    def invalidate_cache(self, prefix=None):
        if prefix is None:
            self._mesh_cache.clear()
        else:
            self._mesh_cache = {k: v for k, v in self._mesh_cache.items() if not k[0].startswith(prefix)}

    def set_background(self, color):
        self.plotter.set_background(color)
        self.render_all()

    def toggle_axes(self):
        self._axes_shown = not self._axes_shown
        self.reset_scene()
        if not self._axes_shown:
            try:
                self.plotter.hide_axes()
            except Exception:
                pass
        self.render_all()

    def reset_camera(self):
        try:
            self.plotter.reset_camera()
            self.plotter.render()
        except Exception:
            pass

    def save_camera(self):
        try:
            self._saved_camera = self.plotter.camera_position
        except Exception:
            self._saved_camera = None

    def restore_camera(self):
        if self._saved_camera is not None:
            try:
                self.plotter.camera_position = self._saved_camera
            except Exception:
                pass

    def set_playback_active(self, active):
        self._playback_active = active
        if active:
            self.save_camera()

    def sync_from_workspace(self):
        current_uids = set(self.workspace.scene_objects.keys())
        stale = set(self._tracked_actors.keys()) - current_uids
        for uid in stale:
            actor = self._tracked_actors.pop(uid, None)
            if actor is not None:
                try:
                    self.plotter.remove_actor(actor)
                except Exception:
                    try:
                        self.plotter.renderer.RemoveActor(actor)
                    except Exception:
                        pass
        self.render_all()

    def remove_object(self, uid):
        obj = self.workspace.scene_objects.get(uid)
        if obj is not None:
            self._remove_actor(obj)
            del self.workspace.scene_objects[uid]
        actor = self._tracked_actors.pop(uid, None)
        if actor is not None:
            try:
                self.plotter.remove_actor(actor)
            except Exception:
                pass
        if self._highlight_plane_uid == uid:
            self._remove_plane_highlight()
        if self._highlight_path_uid == uid:
            self._remove_path_highlight()

    def render_all(self):
        for obj in self.workspace.scene_objects.values():
            self._render_object(obj)
        try:
            self.plotter.render()
        except Exception:
            pass

    def update_time(self, t):
        self.workspace.current_t = int(t)
        cam_before = None
        if self._playback_active:
            try:
                cam_before = self.plotter.camera_position
            except Exception:
                cam_before = None
        for obj in self.workspace.scene_objects.values():
            if obj.dynamic:
                self.readd_object(obj)
        if self._playback_active and cam_before is not None:
            try:
                self.plotter.camera_position = cam_before
            except Exception:
                pass
        try:
            self.plotter.render()
        except Exception:
            pass

    def rebuild_dynamic(self):
        for obj in self.workspace.scene_objects.values():
            if obj.dynamic:
                self.readd_object(obj)

    def readd_object(self, obj):
        self._remove_actor(obj)
        self._render_object(obj)

    def apply_object_properties(self, obj):
        if obj.actor is None:
            self._render_object(obj)
            return
        try:
            obj.actor.SetVisibility(1 if obj.visible else 0)
        except Exception:
            pass
        try:
            prop = obj.actor.GetProperty()
            prop.SetOpacity(float(obj.opacity))
            prop.SetLineWidth(float(obj.line_width))
            prop.SetPointSize(float(obj.point_size))
        except Exception:
            pass
        if obj.visible:
            self.readd_object(obj)
            return
        try:
            self.plotter.render()
        except Exception:
            pass

    def highlight_plane(self, uid):
        self._remove_plane_highlight()
        self._highlight_plane_uid = uid
        if uid is None:
            try:
                self.plotter.render()
            except Exception:
                pass
            return
        obj = self.workspace.scene_objects.get(uid)
        if obj is None or obj.kind != ObjectKind.PLANE:
            self._highlight_plane_uid = None
            return
        data = self._build_dataset(obj.data_key)
        if data is None:
            return
        try:
            self._highlight_plane_actor = self.plotter.add_mesh(
                data, color="magenta", opacity=0.9, line_width=4,
                style="wireframe", name="__plane_highlight__")
            self._promote_overlay_actor(self._highlight_plane_actor)
        except Exception:
            self._highlight_plane_actor = None
        try:
            self.plotter.render()
        except Exception:
            pass

    def highlight_path(self, uid):
        self._remove_path_highlight()
        self._highlight_path_uid = uid
        if uid is None:
            try:
                self.plotter.render()
            except Exception:
                pass
            return
        obj = self.workspace.scene_objects.get(uid)
        if obj is None or obj.kind != ObjectKind.BRANCH:
            self._highlight_path_uid = None
            return
        data = self._build_dataset(obj.data_key)
        if data is None:
            return
        try:
            self._highlight_path_actor = self.plotter.add_mesh(
                data, color="magenta", opacity=1.0, line_width=8,
                render_lines_as_tubes=True,
                name="__path_highlight__")
            self._promote_overlay_actor(self._highlight_path_actor)
        except Exception:
            self._highlight_path_actor = None
        try:
            self.plotter.render()
        except Exception:
            pass

    def show_forks_for_path(self, path_idx):
        self._clear_fork_and_context_actors()
        if int(path_idx) < 0:
            try:
                self.plotter.render()
            except Exception:
                pass
            return
        org = np.asarray(self.workspace.origin, dtype=float).reshape(3)
        pts = []
        incoming_ids = set()
        outgoing_ids = set()
        if 0 <= int(path_idx) < len(self.workspace.path_info):
            info = self.workspace.path_info[int(path_idx)]
            incoming_ids.update(int(x) for x in info.get("incoming_path_ids", []))
            outgoing_ids.update(int(x) for x in info.get("outgoing_path_ids", []))
        for fork in self.workspace.forks:
            if int(path_idx) in fork.get("left", []) or int(path_idx) in fork.get("right", []):
                pts.append(np.asarray(fork.get("crosspoint", [0.0, 0.0, 0.0]), dtype=float) + org)
                incoming_ids.update(int(x) for x in fork.get("left", []) if int(x) != int(path_idx))
                outgoing_ids.update(int(x) for x in fork.get("right", []) if int(x) != int(path_idx))
        incoming_ids.discard(int(path_idx))
        outgoing_ids.discard(int(path_idx))
        for pid, color in [(sorted(incoming_ids), "deepskyblue"), (sorted(outgoing_ids), "orange")]:
            for idx in pid:
                if not (0 <= int(idx) < len(self.workspace.centerline_paths_smooth)):
                    continue
                poly = _path_polydata(self.workspace.centerline_paths_smooth[int(idx)], org)
                if poly is None:
                    continue
                try:
                    actor = self.plotter.add_mesh(
                        poly, color=color, opacity=1.0, line_width=8,
                        render_lines_as_tubes=True,
                        name=f"__path_context_{color}_{int(idx)}__")
                    self._promote_overlay_actor(actor)
                    self._context_path_actors.append(actor)
                except Exception:
                    pass
        if pts:
            try:
                poly = pv.PolyData(np.asarray(pts, dtype=float).reshape(-1, 3))
                self._highlight_fork_actor = self.plotter.add_mesh(
                    poly, color="magenta", point_size=22, render_points_as_spheres=True,
                    name="__fork_highlight__")
                self._promote_overlay_actor(self._highlight_fork_actor)
            except Exception:
                self._highlight_fork_actor = None
        try:
            self.plotter.render()
        except Exception:
            pass

    def _remove_plane_highlight(self):
        if self._highlight_plane_actor is not None:
            try:
                self.plotter.remove_actor(self._highlight_plane_actor)
            except Exception:
                try:
                    self.plotter.renderer.RemoveActor(self._highlight_plane_actor)
                except Exception:
                    pass
        self._highlight_plane_actor = None
        self._highlight_plane_uid = None

    def _remove_path_highlight(self):
        if self._highlight_path_actor is not None:
            try:
                self.plotter.remove_actor(self._highlight_path_actor)
            except Exception:
                try:
                    self.plotter.renderer.RemoveActor(self._highlight_path_actor)
                except Exception:
                    pass
        self._highlight_path_actor = None
        self._highlight_path_uid = None
        self._clear_fork_and_context_actors()

    def _clear_fork_and_context_actors(self):
        if self._highlight_fork_actor is not None:
            try:
                self.plotter.remove_actor(self._highlight_fork_actor)
            except Exception:
                try:
                    self.plotter.renderer.RemoveActor(self._highlight_fork_actor)
                except Exception:
                    pass
        self._highlight_fork_actor = None
        for actor in list(self._context_path_actors):
            if actor is not None:
                try:
                    self.plotter.remove_actor(actor)
                except Exception:
                    try:
                        self.plotter.renderer.RemoveActor(actor)
                    except Exception:
                        pass
        self._context_path_actors = []
        try:
            self.plotter.remove_actor("__fork_highlight__")
        except Exception:
            pass
        try:
            renderer = self.plotter.renderer
            actors_to_remove = []
            it = renderer.GetActors()
            it.InitTraversal()
            for _ in range(it.GetNumberOfItems()):
                a = it.GetNextItem()
                if a is not None:
                    try:
                        name = a.GetObjectName() if hasattr(a, "GetObjectName") else ""
                        if name and ("__path_context_" in name or "__fork_highlight__" in name):
                            actors_to_remove.append(a)
                    except Exception:
                        pass
            for a in actors_to_remove:
                try:
                    renderer.RemoveActor(a)
                except Exception:
                    pass
        except Exception:
            pass


    def _promote_overlay_actor(self, actor):
        if actor is None:
            return
        try:
            actor.PickableOff()
        except Exception:
            pass
        try:
            prop = actor.GetProperty()
            prop.SetLighting(False)
        except Exception:
            pass

    def refresh_plane_labels(self):
        pass

    def remove_all_plane_labels(self):
        pass

    def _remove_actor(self, obj):
        if obj.actor is not None:
            try:
                self.plotter.remove_actor(obj.actor)
            except Exception:
                try:
                    self.plotter.renderer.RemoveActor(obj.actor)
                except Exception:
                    pass
        if getattr(obj, "label_actor", None) is not None:
            try:
                self.plotter.remove_actor(obj.label_actor)
            except Exception:
                try:
                    self.plotter.renderer.RemoveActor(obj.label_actor)
                except Exception:
                    pass
        self._tracked_actors.pop(obj.uid, None)
        obj.actor = None
        obj.label_actor = None

    def _render_object(self, obj):
        if not obj.visible:
            if obj.actor is not None:
                try:
                    obj.actor.SetVisibility(0)
                except Exception:
                    pass
            return
        data = self._build_dataset(obj.data_key)
        if data is None:
            self._remove_actor(obj)
            return
        if obj.actor is not None:
            self._remove_actor(obj)
        kwargs = self._mesh_kwargs(obj, data)
        try:
            if obj.tube_radius > 0 and hasattr(data, "tube") and obj.kind.value in ("Graph", "Branch", "Flow", "Metric", "Skeleton"):
                data_show = data.tube(radius=float(obj.tube_radius))
            else:
                data_show = data
            obj.actor = self.plotter.add_mesh(data_show, name=obj.uid, **kwargs)
            self._tracked_actors[obj.uid] = obj.actor
            self._apply_basic_properties_only(obj)
        except Exception as e:
            self.logger(f"Render failed: {obj.name}: {type(e).__name__}: {e}")

    def _apply_basic_properties_only(self, obj):
        try:
            obj.actor.SetVisibility(1 if obj.visible else 0)
        except Exception:
            pass
        try:
            prop = obj.actor.GetProperty()
            prop.SetOpacity(float(obj.opacity))
            prop.SetLineWidth(float(obj.line_width))
            prop.SetPointSize(float(obj.point_size))
        except Exception:
            pass

    def _mesh_kwargs(self, obj, data):
        kw = {"opacity": float(obj.opacity), "show_scalar_bar": bool(obj.show_scalar_bar)}
        use_scalars = False
        if obj.scalars:
            if hasattr(data, "point_data") and obj.scalars in data.point_data:
                use_scalars = True
            if hasattr(data, "cell_data") and obj.scalars in data.cell_data:
                use_scalars = True
        if use_scalars:
            kw["scalars"] = obj.scalars
            kw["cmap"] = obj.cmap
            if obj.clim:
                kw["clim"] = obj.clim
            if obj.scalar_bar_title:
                kw["scalar_bar_args"] = {
                    "title": obj.scalar_bar_title,
                    "vertical": True,
                    "title_font_size": 14,
                    "label_font_size": 12,
                    "n_labels": 5,
                    "fmt": "%.3g",
                }
        else:
            kw["color"] = obj.color
        if obj.kind.value in ("Skeleton", "Aux"):
            kw["render_points_as_spheres"] = True
            kw["point_size"] = obj.point_size
        if obj.kind.value in ("Graph", "Branch", "Flow", "Aux"):
            kw["line_width"] = obj.line_width
            kw["render_lines_as_tubes"] = True
        if obj.kind == ObjectKind.PLANE:
            kw["show_edges"] = True
            kw["edge_color"] = "black"
            kw["line_width"] = max(float(obj.line_width), 2.0)
        return kw

    def _build_dataset(self, data_key):
        ws = self.workspace
        t = ws.current_t
        sp = ws.resolution
        org = ws.origin

        if data_key == "segmask_raw_surface":
            if ws.segmask_raw is None:
                return None
            return self._cached(data_key, t, lambda: build_multilabel_surface_t(ws.segmask_raw, t, sp, org))

        if data_key == "segmask_pre_surface":
            if ws.segmask_labels is None:
                return None
            return self._cached(data_key, t, lambda: build_multilabel_surface_t(ws.segmask_labels, t, sp, org))

        if data_key == "segmask_3d_surface":
            if ws.segmask_3d is None:
                return None
            return self._cached(data_key, 0, lambda: build_surface_from_mask3d(ws.segmask_3d, sp, org, smooth_iter=1000))

        if data_key == "skeleton_points":
            if ws.skeleton_points is None or len(ws.skeleton_points) == 0:
                return None
            return pv.PolyData(np.asarray(ws.skeleton_points, dtype=float) + np.asarray(org, dtype=float).reshape(1, 3))

        if data_key == "skeleton_mask_surface":
            if ws.skeleton_mask is None:
                return None
            return self._cached(data_key, 0, lambda: build_surface_from_mask3d(ws.skeleton_mask, sp, org, smooth_iter=1000))

        if data_key == "graph_lines":
            if ws.graph is None or len(ws.graph.points) == 0:
                return None
            return graph_to_polydata(np.asarray(ws.graph.points) + np.asarray(org).reshape(1, 3), ws.graph.edges)

        if data_key == "streamlines_live":
            return self._get_streamline_mesh(t)

        if data_key == "plane_streamlines_live":
            return self._get_plane_streamline_mesh(t)

        if data_key == "wss_surface_live":
            if not ws.derived.wss_surfaces:
                return None
            return ws.derived.wss_surfaces[min(max(0, t), len(ws.derived.wss_surfaces) - 1)]

        if data_key == "tke_volume":
            return ws.derived.tke_volume

        if data_key == "derived_streamlines_live":
            if not ws.derived.streamlines:
                return None
            return ws.derived.streamlines[min(max(0, t), len(ws.derived.streamlines) - 1)]

        idx = _parse_indexed_data_key(data_key, "smooth_path")
        if idx is not None:
            if idx >= len(ws.centerline_paths_smooth):
                return None
            path = np.asarray(ws.centerline_paths_smooth[idx], dtype=float)
            if len(path) == 0:
                return None
            return _path_polydata(path, org)

        idx = _parse_indexed_data_key(data_key, "path_arrow")
        if idx is not None:
            if idx >= len(ws.centerline_paths_smooth):
                return None
            path = np.asarray(ws.centerline_paths_smooth[idx], dtype=float)
            if len(path) < 2:
                return None
            org_r = np.asarray(org, dtype=float).reshape(3)
            seglens = np.linalg.norm(np.diff(path, axis=0), axis=1)
            total = float(np.sum(seglens))
            if total < 1e-6:
                return None
            overall = path[-1] - path[0]
            n = np.linalg.norm(overall)
            if n < 1e-12:
                return None
            overall = overall / n
            mid = 0.5 * (path[0] + path[-1])
            arrow_len = max(2.0, total * 0.45)
            shaft_r = max(0.25, arrow_len * 0.05)
            tip_r = max(0.6, arrow_len * 0.12)
            tip_l = max(2.0, arrow_len * 0.25)
            start = mid + org_r - overall * (arrow_len * 0.5)
            return pv.Arrow(
                start=start,
                direction=overall * arrow_len,
                shaft_radius=shaft_r,
                tip_radius=tip_r,
                tip_length=tip_l,
            )

        idx = _parse_indexed_data_key(data_key, "path")
        if idx is not None:
            if idx >= len(ws.centerline_paths):
                return None
            path = np.asarray(ws.centerline_paths[idx], dtype=float)
            if len(path) == 0:
                return None
            return _path_polydata(path, org)

        if data_key == "fork_markers":
            pts = [np.asarray(f.get("crosspoint", [0.0, 0.0, 0.0]), dtype=float) + np.asarray(org, dtype=float).reshape(3) for f in ws.forks]
            if not pts:
                return None
            return pv.PolyData(np.asarray(pts, dtype=float).reshape(-1, 3))

        idx = _parse_indexed_data_key(data_key, "plane")
        if idx is not None:
            if idx >= len(ws.planes):
                return None
            p = ws.planes[idx]
            return pv.Plane(center=np.asarray(p.center) + np.asarray(org), direction=np.asarray(p.normal), i_size=25, j_size=25)

        return None

    def _cached(self, data_key, t, builder):
        key = (data_key, t)
        if key in self._mesh_cache:
            return self._mesh_cache[key]
        mesh = builder()
        if mesh is not None:
            self._mesh_cache[key] = mesh
        return mesh

    def _get_streamline_mesh(self, t):
        ws = self.workspace
        if not ws.streamline_active:
            return None
        if t in ws.streamline_cache:
            return ws.streamline_cache[t]
        if ws.flow_raw is None or ws.segmask_binary is None:
            return None
        p = ws.streamline_params
        mask_t = ws.segmask_binary[..., min(max(0, int(t)), ws.segmask_binary.shape[3] - 1)]
        sl = generate_streamlines_at_t(
            ws.flow_raw, t, ws.streamline_seeds, ws.resolution, ws.origin,
            mask_3d=mask_t,
            max_steps=p.max_steps,
            terminal_speed=p.terminal_speed,
            seed_ratio=p.seed_ratio,
            min_seeds=p.min_seeds,
            rng_seed=p.rng_seed,
        )
        ws.streamline_cache[t] = sl
        return sl

    def _get_plane_streamline_mesh(self, t):
        ws = self.workspace
        if not ws.plane_streamline_active:
            return None
        if t in ws.plane_streamline_cache:
            return ws.plane_streamline_cache[t]
        if ws.flow_raw is None or ws.segmask_binary is None:
            return None
        pidx = ws.plane_streamline_plane_idx
        if pidx < 0 or pidx >= len(ws.planes):
            return None
        plane = ws.planes[pidx]
        p = ws.streamline_params
        mask_t = ws.segmask_binary[..., min(max(0, int(t)), ws.segmask_binary.shape[3] - 1)]
        sl = generate_streamlines_from_plane_at_t(
            ws.flow_raw, t, plane, ws.resolution, ws.origin,
            mask_3d=mask_t,
            max_steps=p.max_steps,
            terminal_speed=p.terminal_speed,
            seed_ratio=p.seed_ratio,
            min_seeds=p.min_seeds,
            rng_seed=p.rng_seed,
            branch_labels_3d=ws.branch_labels,
        )
        ws.plane_streamline_cache[t] = sl
        return sl

    def trigger_streamlines(self):
        ws = self.workspace
        if ws.flow_raw is None or ws.segmask_3d is None:
            self.logger("Cannot generate streamlines: need flow + segmask_3d")
            return
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
        p = ws.streamline_params
        self.logger(f"Streamlines enabled: seed_ratio={p.seed_ratio} max_steps={p.max_steps} min_seeds={p.min_seeds} terminal_speed={p.terminal_speed} rng_seed={p.rng_seed}")
        ws.remove_object_by_data_key("streamlines_live")
        ws.add_object(name="streamlines", kind=ObjectKind.FLOW,
                      data_key="streamlines_live", visible=True, opacity=1.0,
                      scalars="Velocity", cmap="turbo", dynamic=True,
                      show_scalar_bar=True, scalar_bar_title="Velocity (m/s)")
        self.sync_from_workspace()

    def trigger_plane_streamlines(self, plane_idx):
        ws = self.workspace
        if ws.flow_raw is None or ws.segmask_3d is None:
            self.logger("Cannot generate plane streamlines: need flow + segmask_3d")
            return
        if plane_idx < 0 or plane_idx >= len(ws.planes):
            self.logger(f"Invalid plane index: {plane_idx}")
            return
        ws.plane_streamline_cache.clear()
        ws.plane_streamline_active = True
        ws.plane_streamline_plane_idx = plane_idx
        p = ws.streamline_params
        self.logger(f"Plane streamlines enabled from plane {plane_idx}: seed_ratio={p.seed_ratio} min_seeds={p.min_seeds} max_steps={p.max_steps} terminal_speed={p.terminal_speed} rng_seed={p.rng_seed}")
        ws.remove_object_by_data_key("plane_streamlines_live")
        ws.add_object(name="plane_streamlines", kind=ObjectKind.FLOW,
                      data_key="plane_streamlines_live", visible=True, opacity=1.0,
                      scalars="Velocity", cmap="turbo", dynamic=True,
                      show_scalar_bar=True, scalar_bar_title="Velocity (m/s)")
        self.sync_from_workspace()

    def clear_streamlines(self):
        self.workspace.clear_streamlines()
        self.invalidate_cache("streamlines")
        self.sync_from_workspace()
        self.logger("Streamlines cleared")

    def clear_plane_streamlines(self):
        self.workspace.clear_plane_streamlines()
        self.invalidate_cache("plane_streamlines")
        self.sync_from_workspace()
        self.logger("Plane streamlines cleared")

    def find_plane_uid_at_position(self, picked_point):
        ws = self.workspace
        if picked_point is None:
            return None, None
        picked = np.asarray(picked_point, dtype=float).reshape(3)
        best_uid, best_idx, best_dist = None, None, float("inf")
        org = np.asarray(ws.origin, dtype=float).reshape(3)
        for uid, obj in ws.scene_objects.items():
            if obj.kind != ObjectKind.PLANE:
                continue
            pidx = _parse_indexed_data_key(obj.data_key, "plane")
            if pidx is None:
                continue
            if pidx >= len(ws.planes):
                continue
            center = np.asarray(ws.planes[pidx].center, dtype=float) + org
            d = float(np.linalg.norm(picked - center))
            if d < best_dist:
                best_uid, best_idx, best_dist = uid, pidx, d
        return (best_uid, best_idx) if best_dist <= 30.0 else (None, None)

    def find_path_uid_at_position(self, picked_point):
        ws = self.workspace
        if picked_point is None:
            return None, None
        picked = np.asarray(picked_point, dtype=float).reshape(3)
        best_uid, best_idx, best_dist = None, None, float("inf")
        org = np.asarray(ws.origin, dtype=float).reshape(3)
        for uid, obj in ws.scene_objects.items():
            if obj.kind != ObjectKind.BRANCH:
                continue
            if not obj.data_key.startswith("smooth_path_"):
                continue
            try:
                pidx = int(obj.data_key.split("_")[2])
            except Exception:
                continue
            if pidx >= len(ws.centerline_paths_smooth):
                continue
            path = np.asarray(ws.centerline_paths_smooth[pidx], dtype=float) + org.reshape(1, 3)
            if len(path) == 0:
                continue
            d = float(np.min(np.linalg.norm(path - picked.reshape(1, 3), axis=1)))
            if d < best_dist:
                best_uid, best_idx, best_dist = uid, pidx, d
        return (best_uid, best_idx) if best_dist <= 15.0 else (None, None)

    def _ensure_shared_right_click_picking(self):
        if self._shared_pick_obs_id is not None:
            return
        try:
            iren = self.plotter.iren.interactor
        except Exception:
            return
        picker = pv._vtk.vtkCellPicker()
        picker.SetTolerance(0.005)

        def _on_right_click(obj, ev):
            try:
                x, y = iren.GetEventPosition()
            except Exception:
                return
            ren = self.plotter.renderer
            ok = picker.Pick(float(x), float(y), 0.0, ren)
            pos = picker.GetPickPosition() if ok else None
            plane_uid, plane_idx = self.find_plane_uid_at_position(pos) if pos is not None else (None, None)
            if plane_uid is not None and plane_idx is not None:
                if self._plane_pick_callback is not None:
                    self._plane_pick_callback(plane_uid, plane_idx)
                return
            path_uid, path_idx = self.find_path_uid_at_position(pos) if pos is not None else (None, None)
            if path_uid is not None and path_idx is not None:
                if self._path_pick_callback is not None:
                    self._path_pick_callback(path_uid, path_idx)
                return
            if self._plane_pick_callback is not None:
                self._plane_pick_callback(None, None)
            if self._path_pick_callback is not None:
                self._path_pick_callback(None, None)

        self._shared_pick_obs_id = iren.AddObserver("RightButtonPressEvent", _on_right_click)
        self._plane_pick_obs_id = self._shared_pick_obs_id
        self._path_pick_obs_id = self._shared_pick_obs_id

    def enable_plane_picking(self, callback):
        self._plane_pick_callback = callback
        self._ensure_shared_right_click_picking()

    def enable_path_picking(self, callback):
        self._path_pick_callback = callback
        self._ensure_shared_right_click_picking()
