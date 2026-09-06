import numpy as np
import pyvista as pv

from ..core.models import ObjectKind
from ..algorithms import (
    build_multilabel_surface_t,
    build_surface_from_mask3d,
    build_cell_mask_surface,
    graph_to_polydata,
    generate_seed_points,
    generate_streamlines_at_t,
    generate_streamlines_from_plane_at_t,
    generate_pathlines_from_plane_at_t,
    create_uniform_grid,
    sample_volume_on_existing_surface,
)
from ..algorithms.streamlines import (
    _plane_seeds,
    automatic_streamline_clim,
    pathline_prefix_at_phase,
)


_DISPLAY_AXIS_DEFAULTS = ("LR", "AP", "FH")
_DISPLAY_AXIS_CHOICES = (("LR", "RL"), ("AP", "PA"), ("FH", "HF"))


def _parse_indexed_data_key(data_key, prefix):
    token = f"{prefix}_"
    if not isinstance(data_key, str) or not data_key.startswith(token):
        return None
    suffix = data_key[len(token):]
    tail = suffix.rsplit("_", 1)[-1]
    if tail.isdigit():
        return int(tail)
    return None


def _parse_group_name_from_data_key(data_key, prefix):
    token = f"{prefix}_"
    if not isinstance(data_key, str) or not data_key.startswith(token):
        return ""
    suffix = data_key[len(token):]
    parts = suffix.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return str(parts[0])
    if suffix and not suffix.isdigit():
        return str(suffix)
    return ""


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


def _pwv_planes_polydata(planes, origin, plane_size=25.0):
    rows = list(planes or [])
    if not rows:
        return None
    org = np.asarray(origin, dtype=float).reshape(3)
    meshes = []
    for row in rows:
        center = np.asarray(row.get("center", [0.0, 0.0, 0.0]), dtype=float).reshape(3) + org
        normal = np.asarray(row.get("normal", [1.0, 0.0, 0.0]), dtype=float).reshape(3)
        if np.linalg.norm(normal) <= 1e-12:
            normal = np.array([1.0, 0.0, 0.0], dtype=float)
        meshes.append(pv.Plane(center=center, direction=normal, i_size=float(plane_size), j_size=float(plane_size)))
    merged = meshes[0].copy(deep=True)
    for mesh in meshes[1:]:
        merged = merged.merge(mesh)
    return merged


class SceneController:
    def __init__(self, plotter, workspace, logger):
        self.plotter = plotter
        self.workspace = workspace
        self.logger = logger
        self._axes_shown = True
        self._mesh_cache = {}
        self._display_mesh_cache = {}
        self._phase_lookup_cache = {}
        self._automatic_clim_cache = {}
        self._tracked_actors = {}
        self._saved_camera = None
        self._background_color = "#000000"
        self._display_axis_directions = list(_DISPLAY_AXIS_DEFAULTS)
        self._display_axis_signs = np.ones(3, dtype=float)
        self._display_center = None
        self._display_transform_cache = {}
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
        self._active_scalar_bar_uid = None
        self._volume_range_cache = {}
        self._volume_wl_observer_ids = []
        self._volume_wl_active = False
        self._volume_wl_pending_event = None
        self._volume_wl_uid = None
        self._volume_wl_start = None
        self._volume_wl_base = None
        self._interaction_last_position = None
        self._volume_wl_user_callback = None
        self._qt_mouse_filter = None

    @staticmethod
    def _is_volume_object(obj):
        """Return whether *obj* should be drawn with VTK volume rendering."""
        return str(getattr(obj, "data_key", "")) == "pcmra_volume"

    def _volume_scalar_range(self, obj, data, respect_clim=True):
        """Resolve the PC-MRA window from the current frame foreground."""
        if respect_clim and getattr(obj, "clim", None) is not None:
            lo, hi = (float(x) for x in obj.clim)
        else:
            if self._is_volume_object(obj) and self.workspace.mag_raw is not None and self.workspace.flow_raw is not None:
                key = (id(self.workspace.mag_raw), id(self.workspace.flow_raw), int(self.workspace.current_t))
                cached = self._volume_range_cache.get(key)
                if cached is not None:
                    return cached
                try:
                    mag_arr = np.asarray(self.workspace.mag_raw, dtype=np.float32)
                    flow_arr = np.asarray(self.workspace.flow_raw, dtype=np.float32)
                    mag_arr = mag_arr[..., None] if mag_arr.ndim == 3 else mag_arr
                    nt = min(int(mag_arr.shape[3]), int(flow_arr.shape[3]))
                    frame = min(max(0, int(self.workspace.current_t)), max(0, nt - 1))
                    values_t = mag_arr[..., frame] * np.linalg.norm(flow_arr[..., frame, :], axis=-1)
                    finite_t = np.asarray(values_t, dtype=float)
                    finite_t = finite_t[np.isfinite(finite_t) & (finite_t > 0.0)]
                except Exception:
                    finite_t = np.empty(0)
                if finite_t.size:
                    lo = float(np.percentile(finite_t, 5.0))
                    hi = float(np.percentile(finite_t, 99.0))
                else:
                    lo, hi = 0.0, 1.0
                result = (lo, float(hi if np.isfinite(hi) and hi > lo else lo + 1.0))
                self._volume_range_cache[key] = result
                return result
            values = None
            name = str(getattr(obj, "scalars", "") or "PC-MRA")
            try:
                if name in data.cell_data:
                    values = np.asarray(data.cell_data[name], dtype=float)
                elif name in data.point_data:
                    values = np.asarray(data.point_data[name], dtype=float)
            except Exception:
                values = None
            finite = values[np.isfinite(values)] if values is not None else np.empty(0)
            lo = 0.0
            hi = float(np.percentile(finite, 99.5)) if finite.size else 1.0
        if not np.isfinite(lo):
            lo = 0.0
        if not np.isfinite(hi) or hi <= lo:
            hi = lo + 1.0
        return float(lo), float(hi)

    def _set_volume_opacity(self, obj, data=None):
        """Apply the browser opacity to a volume's scalar opacity transfer function."""
        actor = getattr(obj, "actor", None)
        if actor is None or not self._is_volume_object(obj):
            return
        try:
            prop = actor.GetProperty()
            transfer = prop.GetScalarOpacity(0)
            if transfer is None:
                return
            if data is None:
                data = self._build_dataset(obj.data_key)
            if data is None:
                return
            lo, hi = self._volume_scalar_range(obj, data)
            mapper = actor.GetMapper()
            try:
                mapper.scalar_range = (lo, hi)
            except Exception:
                pass
            try:
                prop.SetInterpolationTypeToNearest()
                prop.SetScalarOpacityUnitDistance(max(float(np.mean(np.asarray(self.workspace.resolution, dtype=float))), 0.1))
            except Exception:
                pass
        except Exception:
            pass

    def initialize(self):
        self.plotter.set_background(self._background_color)
        self._add_orientation_axes()
        self._ensure_volume_window_level_interaction()
        self._ensure_native_qt_mouse_bridge()
        self.plotter.reset_camera()

    def _visible_volume_object(self):
        for obj in reversed(list(self.workspace.scene_objects.values())):
            if self._is_volume_object(obj) and bool(getattr(obj, "visible", False)) and getattr(obj, "actor", None) is not None:
                return obj
        return None

    def _ensure_native_qt_mouse_bridge(self):
        """Intercept only Shift+left window/level gestures.

        All other native Qt mouse events are left to QVTK's normal event
        handler so ``vtkInteractorStyleTrackballCamera`` owns the camera
        gestures.
        """
        if self._qt_mouse_filter is not None or hasattr(self.plotter, "_plotter"):
            return
        widget = self.plotter
        if not callable(getattr(widget, "installEventFilter", None)):
            return
        try:
            from PySide6 import QtCore
        except Exception:
            return
        try:
            interactor = widget.iren
        except Exception:
            return

        controller = self

        class _MouseBridge(QtCore.QObject):
            def eventFilter(self, watched, event):  # noqa: N802
                event_type = event.type()
                mouse_move = event_type == QtCore.QEvent.Type.MouseMove
                button_press = event_type == QtCore.QEvent.Type.MouseButtonPress
                button_release = event_type == QtCore.QEvent.Type.MouseButtonRelease
                if not (mouse_move or button_press or button_release):
                    return False
                try:
                    position = event.position()
                    x, y = int(round(position.x())), int(round(position.y()))
                except Exception:
                    try:
                        x, y = int(event.x()), int(event.y())
                    except Exception:
                        return False
                try:
                    modifiers = event.modifiers()
                    ctrl = int(bool(modifiers & QtCore.Qt.KeyboardModifier.ControlModifier))
                    shift = int(bool(modifiers & QtCore.Qt.KeyboardModifier.ShiftModifier))
                    widget._setEventInformation(x, y, ctrl, shift, chr(0), 0, None)
                    interactor.SetAltKey(
                        int(bool(modifiers & QtCore.Qt.KeyboardModifier.AltModifier))
                    )
                except Exception:
                    return False
                if mouse_move:
                    shift_left = controller._volume_wl_active
                elif button_press:
                    shift_left = bool(
                        event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier
                    ) and event.button() == QtCore.Qt.MouseButton.LeftButton and (
                        controller._visible_volume_object() is not None
                    )
                else:
                    shift_left = (
                        controller._volume_wl_active
                        and event.button() == QtCore.Qt.MouseButton.LeftButton
                    )
                if not shift_left:
                    return False
                if mouse_move:
                    controller._dispatch_volume_window_level_event("move")
                elif button_press:
                    controller._dispatch_volume_window_level_event("press")
                elif button_release:
                    controller._dispatch_volume_window_level_event("release")
                return True

        try:
            self._qt_mouse_filter = _MouseBridge(widget)
            widget.installEventFilter(self._qt_mouse_filter)
        except Exception:
            self._qt_mouse_filter = None

    @staticmethod
    def _event_position(interactor):
        try:
            x, y = interactor.GetEventPosition()
            return float(x), float(y)
        except Exception:
            return None

    def _ensure_volume_window_level_interaction(self):
        """Add Shift+left window/level without changing VTK camera gestures."""
        if self._volume_wl_observer_ids:
            return
        try:
            interactor = self.plotter.iren
        except Exception:
            return
        try:
            style = interactor.GetInteractorStyle()
            if style is not None:
                style.SetEnabled(1)
        except Exception:
            pass

        def _on_press():
            pos = self._event_position(interactor)
            if pos is None:
                return
            volume = self._visible_volume_object()
            if volume is not None:
                data = self._build_dataset(volume.data_key)
                if data is None:
                    return
                self._volume_wl_active = True
                try:
                    self.plotter._autoflow_window_level_active = True
                except Exception:
                    pass
                self._volume_wl_uid = volume.uid
                self._volume_wl_start = pos
                self._volume_wl_base = self._volume_scalar_range(volume, data)
                self._interaction_last_position = None
                return

        def _on_move(_obj, _event):
            pos = self._event_position(interactor)
            if pos is None:
                return
            # Native Qt and the bridge can both report the same move.  Avoid
            # applying a camera/WL step twice for one screen position.
            if self._interaction_last_position == pos:
                return
            self._interaction_last_position = pos
            if self._volume_wl_active:
                volume = self.workspace.scene_objects.get(self._volume_wl_uid)
                if volume is None or not self._is_volume_object(volume):
                    return
                if self._volume_wl_start is None or self._volume_wl_base is None:
                    return
                dx = float(pos[0] - self._volume_wl_start[0])
                dy = float(pos[1] - self._volume_wl_start[1])
                base_low, base_high = self._volume_wl_base
                base_width = max(float(base_high - base_low), 1e-6)
                base_level = (float(base_low) + float(base_high)) * 0.5
                width = max(base_width * 0.01, base_width * float(np.exp(dx * 0.012)))
                level = base_level - dy * width * 0.004
                self._apply_volume_window_level(
                    volume, (level - width * 0.5, level + width * 0.5), render=True
                )
                return
            return

        def _on_release():
            if self._volume_wl_active:
                self._volume_wl_active = False
                self._volume_wl_uid = None
                self._volume_wl_start = None
                self._volume_wl_base = None
                try:
                    self.plotter._autoflow_window_level_active = False
                except Exception:
                    pass
            return

        def _on_native_left_press(_obj, _event):
            if interactor.GetShiftKey() and self._visible_volume_object() is not None:
                _on_press()

        def _on_native_left_release(_obj, _event):
            if self._volume_wl_active:
                _on_release()

        def _on_user_event(_obj, _event):
            event_name = self._volume_wl_pending_event
            self._volume_wl_pending_event = None
            if event_name == "press":
                _on_press()
            elif event_name == "move":
                _on_move(_obj, _event)
            elif event_name == "release":
                _on_release()

        try:
            self._volume_wl_user_callback = _on_user_event
            self._volume_wl_observer_ids = [
                interactor.AddObserver("UserEvent", _on_user_event, 1.0),
                interactor.AddObserver("MouseMoveEvent", _on_move, 1.0),
                interactor.AddObserver("LeftButtonPressEvent", _on_native_left_press, 1.0),
                interactor.AddObserver("LeftButtonReleaseEvent", _on_native_left_release, 1.0),
            ]
            self.plotter._autoflow_window_level_dispatch = (
                self._dispatch_volume_window_level_event
            )
        except Exception:
            self._volume_wl_observer_ids = []

    def _dispatch_volume_window_level_event(self, event_name):
        self._volume_wl_pending_event = str(event_name)
        try:
            callback = self._volume_wl_user_callback
            if callback is not None:
                callback(self.plotter.iren, "UserEvent")
            else:
                self.plotter.iren.InvokeEvent("UserEvent")
        except Exception:
            self._volume_wl_pending_event = None

    def _apply_volume_window_level(self, obj, clim, *, render=True):
        if not self._is_volume_object(obj):
            return
        low, high = (float(value) for value in clim)
        if not np.isfinite(low) or not np.isfinite(high):
            return
        if high <= low:
            high = low + 1e-6
        obj.clim = (low, high)
        actor = getattr(obj, "actor", None)
        if actor is not None:
            try:
                transfer = actor.GetProperty().GetRGBTransferFunction(0)
                transfer.RemoveAllPoints()
                for fraction in (0.0, 0.18, 0.42, 0.68, 1.0):
                    value = low + (high - low) * fraction
                    transfer.AddRGBPoint(value, fraction, fraction, fraction)
                self._set_volume_opacity(obj)
                actor.GetProperty().Modified()
            except Exception:
                pass
        if render:
            try:
                self.plotter.render()
            except Exception:
                pass

    def reset_volume_window_level(self):
        """Restore the automatic current-frame PC-MRA window and level."""
        obj = self._visible_volume_object()
        if obj is None:
            return False
        obj.clim = None
        self.readd_object(obj, refresh_scalar_bar=False)
        try:
            self.plotter.render()
        except Exception:
            pass
        return True

    def _add_orientation_axes(self):
        try:
            self.plotter.add_axes(
                xlabel=self._display_axis_directions[0],
                ylabel=self._display_axis_directions[1],
                zlabel=self._display_axis_directions[2],
                line_width=2,
            )
            if not self._axes_shown:
                self.plotter.hide_axes()
        except Exception:
            pass

    def set_display_axis_directions(self, directions):
        """Set display-only positive axis directions and rebuild rendered actors."""
        values = [str(value or "").strip().upper() for value in (directions or ())]
        if len(values) < 3:
            return False
        normalized = []
        for value, choices in zip(values[:3], _DISPLAY_AXIS_CHOICES):
            normalized.append(value if value in choices else choices[0])
        signs = np.asarray([1.0 if value == default else -1.0 for value, default in zip(normalized, _DISPLAY_AXIS_DEFAULTS)], dtype=float)
        if normalized == self._display_axis_directions:
            return True
        if self._display_center is None:
            self._display_center = self._current_scene_center()
        self._display_axis_directions = normalized
        self._display_axis_signs = signs
        self._display_transform_cache.clear()
        highlighted_plane = self._highlight_plane_uid
        highlighted_path = self._highlight_path_uid
        highlighted_path_idx = None
        if highlighted_path is not None:
            highlighted_obj = self.workspace.scene_objects.get(highlighted_path)
            if highlighted_obj is not None:
                highlighted_path_idx = _parse_indexed_data_key(highlighted_obj.data_key, "smooth_path")
        self._remove_plane_highlight()
        self._remove_path_highlight()
        self._clear_fork_and_context_actors()
        try:
            remover = getattr(self.plotter, "_remove_axes_widget", None)
            if callable(remover):
                remover()
            else:
                self.plotter.hide_axes()
        except Exception:
            pass
        self._add_orientation_axes()
        self.sync_from_workspace()
        if highlighted_plane is not None:
            self.highlight_plane(highlighted_plane)
        if highlighted_path is not None:
            self.highlight_path(highlighted_path)
            if highlighted_path_idx is not None:
                self.show_forks_for_path(highlighted_path_idx)
        return True

    def display_axis_directions(self):
        return tuple(self._display_axis_directions)

    def _current_scene_center(self):
        try:
            bounds = np.asarray(tuple(float(value) for value in self.plotter.bounds), dtype=float)
            if bounds.size == 6 and np.all(np.isfinite(bounds)):
                extent = bounds[[1, 3, 5]] - bounds[[0, 2, 4]]
                if np.any(extent > 1e-9):
                    return bounds.reshape(3, 2).mean(axis=1)
        except Exception:
            pass
        return None

    def _ensure_display_center(self, data=None):
        if self._display_center is not None:
            return np.asarray(self._display_center, dtype=float).reshape(3)
        center = self._current_scene_center()
        if center is None and data is not None:
            try:
                bounds = np.asarray(data.bounds, dtype=float).reshape(3, 2)
                center = bounds.mean(axis=1)
            except Exception:
                center = np.zeros(3, dtype=float)
        self._display_center = np.asarray(center if center is not None else np.zeros(3), dtype=float).reshape(3)
        return self._display_center

    def _workspace_scene_center(self):
        bounds_rows = []
        for obj in self.workspace.scene_objects.values():
            if not getattr(obj, "visible", True):
                continue
            try:
                data = self._build_dataset(obj.data_key)
                bounds = np.asarray(data.bounds, dtype=float).reshape(3, 2) if data is not None else None
                if bounds is not None and np.all(np.isfinite(bounds)):
                    bounds_rows.append(bounds)
            except Exception:
                continue
        if not bounds_rows:
            return None
        bounds = np.stack(bounds_rows, axis=0)
        return np.stack([np.min(bounds[:, axis, 0]) for axis in range(3)] + [
            np.max(bounds[:, axis, 1]) for axis in range(3)
        ]).reshape(2, 3).mean(axis=0)

    def world_to_display_points(self, points):
        arr = np.asarray(points, dtype=float)
        original_shape = arr.shape
        flat = arr.reshape(-1, 3)
        if np.all(self._display_axis_signs == 1.0):
            return arr.copy()
        center = self._ensure_display_center()
        return (center + (flat - center) * self._display_axis_signs.reshape(1, 3)).reshape(original_shape)

    def display_to_world_points(self, points):
        return self.world_to_display_points(points)

    def world_to_display_point(self, point):
        return self.world_to_display_points(np.asarray(point, dtype=float).reshape(1, 3))[0]

    def display_to_world_point(self, point):
        return self.display_to_world_points(np.asarray(point, dtype=float).reshape(1, 3))[0]

    def world_to_display_vector(self, vector):
        value = np.asarray(vector, dtype=float).reshape(3)
        return value * self._display_axis_signs

    def display_to_world_vector(self, vector):
        return self.world_to_display_vector(vector)

    def _transform_display_dataset(self, data):
        if data is None or np.all(self._display_axis_signs == 1.0):
            return data
        center = tuple(float(value) for value in self._ensure_display_center(data))
        key = (id(data), tuple(float(value) for value in self._display_axis_signs), center)
        cached = self._display_transform_cache.get(key)
        if cached is not None:
            return cached
        try:
            matrix = np.eye(4, dtype=float)
            matrix[:3, :3] = np.diag(self._display_axis_signs)
            matrix[:3, 3] = np.asarray(center) - self._display_axis_signs * np.asarray(center)
            transformed = data.copy(deep=True)
            transformed.transform(matrix, transform_all_input_vectors=True, inplace=True)
        except Exception:
            try:
                transformed = data.copy(deep=True)
                transformed.points = self.world_to_display_points(np.asarray(data.points, dtype=float))
            except Exception:
                transformed = data
        self._display_transform_cache[key] = transformed
        return transformed

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
        self._display_mesh_cache.clear()
        self._phase_lookup_cache.clear()
        self._automatic_clim_cache.clear()
        self._volume_range_cache.clear()
        self._display_transform_cache.clear()
        self._active_scalar_bar_uid = None
        self._remove_plane_highlight()
        self._remove_path_highlight()
        self.initialize()

    def reset_display_reference(self):
        self._display_center = None
        self._display_transform_cache.clear()

    def invalidate_cache(self, prefix=None):
        self._phase_lookup_cache.clear()
        self._automatic_clim_cache.clear()
        self._volume_range_cache.clear()
        self._display_transform_cache.clear()
        if prefix is None:
            self._mesh_cache.clear()
            self._display_mesh_cache.clear()
        else:
            self._mesh_cache = {k: v for k, v in self._mesh_cache.items() if not k[0].startswith(prefix)}
            self._display_mesh_cache = {
                key: value for key, value in self._display_mesh_cache.items()
                if not key[0].startswith(prefix)
            }

    def set_background(self, color):
        self._background_color = str(color or "#000000")
        self.plotter.set_background(self._background_color)
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

    def sync_from_workspace(self, rebuild_prefixes=None):
        had_rendered_scene = bool(self._tracked_actors)
        if self._display_center is None and not np.all(self._display_axis_signs == 1.0):
            self._display_center = self._workspace_scene_center()
        prefixes = None if rebuild_prefixes is None else tuple(str(x) for x in rebuild_prefixes)
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
        for obj in self.workspace.scene_objects.values():
            rebuild = prefixes is None or any(obj.data_key.startswith(prefix) for prefix in prefixes)
            if obj.actor is None or rebuild:
                self._render_object(obj, refresh_scalar_bar=False)
            else:
                self._apply_basic_properties_only(obj)
        self._refresh_shared_scalar_bar(render=False)
        if not had_rendered_scene and self._tracked_actors:
            self.plotter.reset_camera()
        try:
            self.plotter.render()
        except Exception:
            pass

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
        self._refresh_shared_scalar_bar(render=False)
        try:
            self.plotter.render()
        except Exception:
            pass

    def render_all(self):
        had_rendered_scene = bool(self._tracked_actors)
        for obj in self.workspace.scene_objects.values():
            if obj.actor is None:
                self._render_object(obj, refresh_scalar_bar=False)
            else:
                self._apply_basic_properties_only(obj)
        self._refresh_shared_scalar_bar(render=False)
        if not had_rendered_scene and self._tracked_actors:
            self.plotter.reset_camera()
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
            # Hidden layers can be costly to rebuild (for example TKE,
            # pressure, streamlines, and 4D segmentation surfaces). Keep the
            # actor at its last phase and bring it current only when shown.
            if obj.dynamic and obj.visible:
                self._update_dynamic_object(obj)
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
                self._update_dynamic_object(obj)

    def _update_dynamic_object(self, obj):
        data = self._build_dataset(obj.data_key)
        if data is None:
            self._remove_actor(obj)
            return
        if obj.actor is None:
            self._render_object(obj, refresh_scalar_bar=False)
            return
        if self._is_volume_object(obj):
            self.readd_object(obj, refresh_scalar_bar=False)
            return
        try:
            self._segmentation_category_metadata(obj, data)
            data_show = self._display_dataset(obj, data)
            data_show = self._transform_display_dataset(data_show)
            mapper = obj.actor.GetMapper()
            # Preserve PyVista's active-scalar pipeline when swapping phases.
            # Raw VTK SetInputData leaves the mapper's scalar texture connected
            # to the previous dataset and renders much of the new mesh black.
            mapper.dataset = data_show
            mapper.Update()
            if self._is_volume_object(obj):
                self._set_volume_opacity(obj, data_show)
        except Exception:
            self.readd_object(obj, refresh_scalar_bar=False)

    def readd_object(self, obj, refresh_scalar_bar=True):
        self._remove_actor(obj)
        self._render_object(obj, refresh_scalar_bar=refresh_scalar_bar)

    def update_plane_geometry(self, uid, *, render=True):
        obj = self.workspace.scene_objects.get(uid)
        if obj is None or obj.kind != ObjectKind.PLANE:
            return False
        data = self._build_dataset(obj.data_key)
        if data is None:
            return False
        data = self._transform_display_dataset(data)
        actors = [obj.actor]
        if self._highlight_plane_uid == uid:
            actors.append(self._highlight_plane_actor)
        updated = False
        for actor in actors:
            if actor is None:
                continue
            try:
                mapper = actor.GetMapper()
                mapper.dataset = data
                mapper.Update()
                updated = True
            except Exception:
                continue
        if not updated and obj.actor is not None:
            self.readd_object(obj, refresh_scalar_bar=False)
            updated = True
        if render:
            try:
                self.plotter.render()
            except Exception:
                pass
        return updated

    def apply_object_properties(self, obj, *, render=True, refresh_scalar_bar=True):
        if obj.actor is None:
            self._render_object(obj, refresh_scalar_bar=True)
            return
        was_visible = False
        try:
            was_visible = bool(obj.actor.GetVisibility())
        except Exception:
            pass
        if obj.dynamic and obj.visible and not was_visible:
            self._update_dynamic_object(obj)
        try:
            obj.actor.SetVisibility(1 if obj.visible else 0)
        except Exception:
            pass
        if self._is_volume_object(obj):
            self._set_volume_opacity(obj)
            if refresh_scalar_bar:
                self._refresh_shared_scalar_bar(render=False)
            if render:
                try:
                    self.plotter.render()
                except Exception:
                    pass
            return
        try:
            prop = obj.actor.GetProperty()
            prop.SetOpacity(float(obj.opacity))
            prop.SetLineWidth(float(obj.line_width))
            prop.SetPointSize(float(obj.point_size))
            if not obj.scalars and obj.color:
                prop.SetColor(*pv.Color(obj.color).float_rgb)
        except Exception:
            pass
        if refresh_scalar_bar:
            self._refresh_shared_scalar_bar(render=False)
        if render:
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
        data = self._transform_display_dataset(data)
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
        data = self._transform_display_dataset(data)
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
                poly = self._transform_display_dataset(poly)
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
                poly = self._transform_display_dataset(poly)
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

    def _render_object(self, obj, refresh_scalar_bar=True):
        if not obj.visible:
            if obj.actor is not None:
                try:
                    obj.actor.SetVisibility(0)
                except Exception:
                    pass
            if refresh_scalar_bar:
                self._refresh_shared_scalar_bar(render=False)
            return
        data = self._build_dataset(obj.data_key)
        if data is None:
            self._remove_actor(obj)
            if refresh_scalar_bar:
                self._refresh_shared_scalar_bar(render=False)
            return
        if obj.actor is not None:
            self._remove_actor(obj)
        kwargs = self._mesh_kwargs(obj, data)
        try:
            data_show = self._display_dataset(obj, data)
            data_show = self._transform_display_dataset(data_show)
            if self._is_volume_object(obj):
                obj.actor = self.plotter.add_volume(data_show, name=obj.uid, **kwargs)
                try:
                    obj.actor.PickableOff()
                except Exception:
                    pass
                self._set_volume_opacity(obj, data_show)
            else:
                obj.actor = self.plotter.add_mesh(data_show, name=obj.uid, **kwargs)
            self._tracked_actors[obj.uid] = obj.actor
            self._apply_basic_properties_only(obj)
        except Exception as e:
            self.logger(f"Render failed: {obj.name}: {type(e).__name__}: {e}")
        if refresh_scalar_bar:
            self._refresh_shared_scalar_bar(preferred_uid=obj.uid if obj.visible else None, render=False)

    def _apply_basic_properties_only(self, obj):
        try:
            obj.actor.SetVisibility(1 if obj.visible else 0)
        except Exception:
            pass
        if self._is_volume_object(obj):
            self._set_volume_opacity(obj)
            return
        try:
            prop = obj.actor.GetProperty()
            prop.SetOpacity(float(obj.opacity))
            prop.SetLineWidth(float(obj.line_width))
            prop.SetPointSize(float(obj.point_size))
        except Exception:
            pass

    def _mesh_kwargs(self, obj, data):
        if self._is_volume_object(obj):
            return {
                "scalars": str(obj.scalars or "PC-MRA"),
                "cmap": str(obj.cmap or "gray"),
                "clim": self._volume_scalar_range(obj, data),
                "opacity": "linear",
                "ambient": 0.35,
                "diffuse": 0.65,
                "specular": 0.05,
                "specular_power": 8.0,
                "shade": False,
                "blending": "composite",
                "mapper": "fixed_point",
                "opacity_unit_distance": max(float(np.mean(np.asarray(self.workspace.resolution, dtype=float))), 0.1),
                "show_scalar_bar": False,
            }
        kw = {"opacity": float(obj.opacity), "show_scalar_bar": False}
        use_scalars = False
        if obj.scalars:
            if hasattr(data, "point_data") and obj.scalars in data.point_data:
                use_scalars = True
            if hasattr(data, "cell_data") and obj.scalars in data.cell_data:
                use_scalars = True
        if use_scalars:
            category_metadata = self._segmentation_category_metadata(obj, data)
            if category_metadata is None:
                kw["scalars"] = obj.scalars
                kw["cmap"] = obj.cmap
            else:
                display_name, labels = category_metadata
                kw["scalars"] = display_name
                label_names = getattr(self.workspace.segmentation, "label_names", {})
                label_colors = getattr(self.workspace.segmentation, "label_colors", {})
                kw["categories"] = True
                kw["n_colors"] = max(2, len(labels))
                kw["annotations"] = {
                    float(index): str(label_names.get(str(label), f"Label {label}"))
                    for index, label in enumerate(labels)
                }
                colors = [label_colors.get(str(label), "") for label in labels]
                kw["cmap"] = colors if colors and all(colors) else obj.cmap
            clim = self._resolved_object_clim(obj)
            if clim is not None and category_metadata is None:
                kw["clim"] = clim
        else:
            kw["color"] = obj.color
        if obj.data_key == "pwv_planes":
            kw["show_edges"] = True
            kw["edge_color"] = "black"
            kw["line_width"] = max(float(obj.line_width), 2.0)
            return kw
        if obj.kind.value in ("Skeleton", "Aux"):
            kw["render_points_as_spheres"] = True
            kw["point_size"] = obj.point_size
        if obj.kind.value in ("Graph", "Branch", "Flow", "Aux"):
            kw["line_width"] = obj.line_width
            kw["render_lines_as_tubes"] = True
        if obj.data_key.startswith("pathline_") and getattr(data, "n_lines", 0) == 0:
            kw["render_points_as_spheres"] = True
            kw["point_size"] = max(float(obj.point_size), 6.0)
        if obj.kind == ObjectKind.FLOW and use_scalars:
            # Quantitative streamline colors must match the scalar bar instead
            # of being darkened by the tube surface orientation.
            kw["lighting"] = False
        if obj.kind == ObjectKind.PLANE:
            kw["show_edges"] = True
            kw["edge_color"] = "black"
            kw["line_width"] = max(float(obj.line_width), 2.0)
        return kw

    def _resolved_object_clim(self, obj):
        if obj.clim is not None:
            return tuple(obj.clim)
        if obj.data_key != "streamlines_live" or obj.scalars != "Velocity":
            return None
        ws = self.workspace
        flow = ws.flow_raw
        mask = ws.segmask_binary if ws.segmask_binary is not None else ws.segmask_3d
        key = (id(flow), id(mask))
        clim = self._automatic_clim_cache.get(key)
        if clim is None:
            clim = automatic_streamline_clim(flow, mask)
            self._automatic_clim_cache[key] = clim
        return clim

    def _display_dataset(self, obj, data):
        if not (
            obj.tube_radius > 0
            and hasattr(data, "tube")
            and obj.kind.value in ("Graph", "Branch", "Flow", "Metric", "Skeleton")
        ):
            return data
        key = (str(obj.data_key), id(data), float(obj.tube_radius))
        cached = self._display_mesh_cache.get(key)
        if cached is None:
            cached = data.tube(radius=float(obj.tube_radius))
            self._display_mesh_cache[key] = cached
        return cached

    def _segmentation_category_metadata(self, obj, data):
        if obj.kind != ObjectKind.SEGMENTATION or not obj.scalars:
            return None
        association = None
        if hasattr(data, "point_data") and obj.scalars in data.point_data:
            association = data.point_data
        elif hasattr(data, "cell_data") and obj.scalars in data.cell_data:
            association = data.cell_data
        if association is None:
            return None
        scalar_values = np.asarray(association[obj.scalars])
        labels = sorted(int(value) for value in np.unique(scalar_values) if np.isfinite(value))
        if not labels:
            return None
        display_name = f"__autoflow_category_{obj.scalars}"
        association[display_name] = np.searchsorted(
            np.asarray(labels, dtype=np.int64), scalar_values.astype(np.int64)
        ).astype(np.int16, copy=False)
        return display_name, labels

    def _scalar_bar_args_for_object(self, obj):
        scalar_bar_args = {
            "title": str(obj.scalar_bar_title or ""),
            "vertical": True,
            "title_font_size": 14,
            "label_font_size": 12,
            "n_labels": 5,
            "fmt": "%.3g",
        }
        shared_cfg = dict(getattr(self.workspace, "render_settings", {}).get("shared_colorbar_bar_cfg", {}) or {})
        if shared_cfg:
            scalar_bar_args.update({k: v for k, v in shared_cfg.items() if k != "stack_gap"})
        width = min(max(float(scalar_bar_args.get("width", 0.08)), 0.02), 0.4)
        height = min(max(float(scalar_bar_args.get("height", 0.6)), 0.15), 0.95)
        scalar_bar_args["width"] = width
        scalar_bar_args["height"] = height
        scalar_bar_args["position_x"] = min(
            max(float(scalar_bar_args.get("position_x", 0.87)), 0.0),
            max(0.0, 0.98 - width),
        )
        scalar_bar_args["position_y"] = min(
            max(float(scalar_bar_args.get("position_y", 0.15)), 0.0),
            max(0.0, 0.98 - height),
        )
        try:
            background = np.asarray(pv.Color(self.plotter.background_color).float_rgb, dtype=float)
            luminance = float(np.dot(background, [0.2126, 0.7152, 0.0722]))
            scalar_bar_args.setdefault("color", "black" if luminance >= 0.5 else "white")
        except Exception:
            pass
        if obj.kind == ObjectKind.SEGMENTATION:
            scalar_bar_args["n_labels"] = 0
        return scalar_bar_args

    def _object_can_drive_scalar_bar(self, obj):
        render_settings = getattr(self.workspace, "render_settings", {}) or {}
        if not bool(render_settings.get("shared_colorbar_show", True)):
            return False
        if obj is None or not bool(obj.visible) or not bool(obj.show_scalar_bar) or not obj.scalars:
            return False
        actor = getattr(obj, "actor", None)
        if actor is None:
            return False
        try:
            return actor.GetMapper() is not None
        except Exception:
            return False

    def _clear_shared_scalar_bar(self, render=False):
        try:
            scalar_bars = getattr(self.plotter, "scalar_bars", None)
            titles = list(scalar_bars.keys()) if scalar_bars is not None else []
        except Exception:
            titles = []
        removed = False
        for title in titles:
            try:
                self.plotter.remove_scalar_bar(title=title, render=False)
                removed = True
            except Exception:
                pass
        if not titles:
            try:
                self.plotter.remove_scalar_bar(render=False)
                removed = True
            except Exception:
                pass
        if removed:
            self._active_scalar_bar_uid = None
        if render:
            try:
                self.plotter.render()
            except Exception:
                pass

    def _select_scalar_bar_object(self, preferred_uid=None):
        if preferred_uid is not None:
            preferred = self.workspace.scene_objects.get(preferred_uid)
            if self._object_can_drive_scalar_bar(preferred):
                return preferred
        if self._active_scalar_bar_uid is not None:
            active = self.workspace.scene_objects.get(self._active_scalar_bar_uid)
            if self._object_can_drive_scalar_bar(active):
                return active
        candidates = [obj for obj in self.workspace.scene_objects.values() if self._object_can_drive_scalar_bar(obj)]
        if not candidates:
            return None
        return candidates[-1]

    def _refresh_shared_scalar_bar(self, preferred_uid=None, render=False):
        obj = self._select_scalar_bar_object(preferred_uid=preferred_uid)
        if obj is None:
            self._clear_shared_scalar_bar(render=render)
            return
        try:
            mapper = obj.actor.GetMapper()
        except Exception:
            mapper = None
        if mapper is None:
            self._clear_shared_scalar_bar(render=render)
            return
        self._clear_shared_scalar_bar(render=False)
        try:
            self.plotter.add_scalar_bar(mapper=mapper, render=False, **self._scalar_bar_args_for_object(obj))
            self._active_scalar_bar_uid = obj.uid
        except Exception as e:
            self._active_scalar_bar_uid = None
            self.logger(f"Scalar bar refresh failed: {obj.name}: {type(e).__name__}: {e}")
        if render:
            try:
                self.plotter.render()
            except Exception:
                pass

    def _build_dataset(self, data_key):
        ws = self.workspace
        t = ws.current_t
        sp = ws.resolution
        org = ws.origin

        if data_key == "pcmra_volume":
            if ws.mag_raw is None or ws.flow_raw is None:
                return None
            mag = np.asarray(ws.mag_raw, dtype=np.float32)
            flow = np.asarray(ws.flow_raw, dtype=np.float32)
            if flow.ndim != 5 or mag.ndim not in (3, 4):
                return None
            cache_key = f"pcmra_volume_{id(ws.mag_raw)}_{id(ws.flow_raw)}"
            def _build_pcmra():
                mag_t = mag[..., None] if mag.ndim == 3 else mag
                if mag_t.shape[:3] != flow.shape[:3]:
                    return None
                nt = min(int(mag_t.shape[3]), int(flow.shape[3]))
                if nt <= 0:
                    return None
                tidx = min(max(0, int(t)), nt - 1)
                values = mag_t[..., tidx] * np.linalg.norm(flow[..., tidx, :], axis=-1)
                values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
                grid = pv.ImageData(
                    dimensions=tuple((np.asarray(values.shape, dtype=int) + 1).tolist()),
                    spacing=tuple(np.asarray(sp, dtype=float).reshape(-1)[:3].tolist()),
                    origin=tuple(np.asarray(org, dtype=float).reshape(-1)[:3].tolist()),
                )
                grid.cell_data["PC-MRA"] = values.flatten(order="F")
                # Volume mappers consume point scalars.  Convert once here so
                # timeline updates can swap mapper input directly instead of
                # recreating the VTK volume actor for every cardiac frame.
                return grid.cell_data_to_point_data(pass_cell_data=False)
            return self._cached(cache_key, int(t), _build_pcmra)

        if data_key == "segmask_raw_surface":
            seg_display = ws.segmentation_display_4d()
            if seg_display is None:
                return None
            rep_t = self._representative_phase(seg_display, t)
            return self._cached(data_key, rep_t, lambda: build_multilabel_surface_t(seg_display, rep_t, sp, org))

        if data_key == "segmask_pre_surface":
            if ws.segmask_labels is None:
                return None
            return self._cached(data_key, t, lambda: build_multilabel_surface_t(ws.segmask_labels, t, sp, org))

        if data_key == "segmask_3d_surface":
            if ws.segmask_3d is None:
                return None
            return self._cached(data_key, 0, lambda: build_surface_from_mask3d(ws.segmask_3d, sp, org, smooth_iter=1000))

        if data_key in {"phase_wrap_mask", "phase_wrap_count"}:
            result = getattr(ws, "phase_unwrap_result", {}) or {}
            field = result.get("wrap_mask" if data_key == "phase_wrap_mask" else "wrap_count")
            if field is None:
                return None
            arr = np.asarray(field)
            if arr.ndim == 5:
                arr = arr[:, :, :, min(max(0, int(t)), arr.shape[3] - 1), :]
            if arr.ndim == 4:
                arr = np.any(arr, axis=-1) if data_key == "phase_wrap_mask" else arr[..., 0]
            mesh = create_uniform_grid(np.asarray(arr), sp, org, name="wrap_mask" if data_key == "phase_wrap_mask" else "wrap_count")
            return mesh

        if isinstance(data_key, str) and data_key.startswith("segmask_group_"):
            group_name = str(data_key[len("segmask_group_"):])
            group_state = ws.multilabel_groups.get(group_name, {})
            mask = group_state.get("segmask_3d")
            if mask is None:
                return None
            return self._cached(
                data_key,
                0,
                lambda: build_surface_from_mask3d(
                    np.asarray(mask, dtype=bool), sp, org, smooth_iter=1000
                ),
            )

        if data_key == "skeleton_points":
            if ws.skeleton_points is None or len(ws.skeleton_points) == 0:
                return None
            return pv.PolyData(
                np.asarray(ws.skeleton_points, dtype=float)
                + np.asarray(org, dtype=float).reshape(1, 3)
            )

        if data_key == "skeleton_mask_surface":
            if ws.skeleton_mask is None:
                return None
            return self._cached(
                data_key,
                0,
                lambda: build_surface_from_mask3d(
                    ws.skeleton_mask, sp, org, smooth_iter=1000
                ),
            )

        if isinstance(data_key, str) and data_key.startswith("skeleton_"):
            group_name = str(data_key[len("skeleton_"):])
            if group_name and group_name != "points":
                group_state = ws.multilabel_groups.get(group_name, {})
                points = group_state.get("skeleton_points")
                if points is None or len(points) == 0:
                    return None
                return pv.PolyData(
                    np.asarray(points, dtype=float)
                    + np.asarray(org, dtype=float).reshape(1, 3)
                )

        if data_key == "graph_lines":
            if ws.graph is None or len(ws.graph.points) == 0:
                return None
            return graph_to_polydata(
                np.asarray(ws.graph.points, dtype=float)
                + np.asarray(org, dtype=float).reshape(1, 3),
                ws.graph.edges,
            )

        if isinstance(data_key, str) and data_key.startswith("graph_"):
            group_name = str(data_key[len("graph_"):])
            if group_name and group_name != "lines":
                group_state = ws.multilabel_groups.get(group_name, {})
                graph = group_state.get("graph")
                if graph is None or len(getattr(graph, "points", [])) == 0:
                    return None
                return graph_to_polydata(
                    np.asarray(graph.points, dtype=float)
                    + np.asarray(org, dtype=float).reshape(1, 3),
                    graph.edges,
                )

        if isinstance(data_key, str) and data_key.startswith("forks_"):
            group_name = str(data_key[len("forks_"):])
            group_state = ws.multilabel_groups.get(group_name, {})
            forks = list(group_state.get("forks", []))
            points = [
                np.asarray(fork.get("crosspoint", [0.0, 0.0, 0.0]), dtype=float)
                + np.asarray(org, dtype=float).reshape(3)
                for fork in forks
            ]
            if not points:
                return None
            return pv.PolyData(np.asarray(points, dtype=float).reshape(-1, 3))

        if data_key == "streamlines_live":
            return self._get_streamline_mesh(t)

        idx = _parse_indexed_data_key(data_key, "pathline")
        if idx is not None:
            return self._get_pathline_mesh(idx, t)

        if data_key == "wss_surface_live":
            if not ws.derived.wss_surfaces:
                return None
            return ws.derived.wss_surfaces[min(max(0, t), len(ws.derived.wss_surfaces) - 1)]

        if data_key == "tke_volume":
            if ws.derived.tke_array is not None:
                def _build_tke_t():
                    arr = np.asarray(ws.derived.tke_array, dtype=np.float32)
                    if arr.ndim == 4:
                        vol_t = arr[..., min(max(0, int(t)), arr.shape[3] - 1)]
                    else:
                        vol_t = arr
                    if ws.segmask_binary is not None:
                        if ws.segmask_binary.ndim == 4:
                            mask_t = ws.segmask_binary[..., min(max(0, int(t)), ws.segmask_binary.shape[3] - 1)]
                        else:
                            mask_t = ws.segmask_binary
                    elif ws.segmask_3d is not None:
                        mask_t = ws.segmask_3d
                    else:
                        mask_t = np.ones(vol_t.shape, dtype=bool)
                    vol_t = vol_t * np.asarray(mask_t, dtype=np.float32)

                    tke_grid = create_uniform_grid(vol_t, sp, origin=org, name="TKE")
                    mask_grid = create_uniform_grid(np.asarray(mask_t, dtype=np.float32), sp, origin=org, name="mask")
                    mask_mesh = mask_grid.threshold(0.1, scalars="mask")
                    if mask_mesh is None or mask_mesh.n_cells == 0:
                        return None
                    return mask_mesh.sample(tke_grid)
                return self._cached(data_key, t, _build_tke_t)
            return ws.derived.tke_volume

        vortex_fields = {
            "vorticity_magnitude_volume": ("vorticity_magnitude", "Vorticity Magnitude"),
            "q_criterion_volume": ("q_criterion_array", "Q-Criterion"),
            "swirling_strength_volume": ("swirling_strength_array", "Swirling Strength"),
        }
        if data_key in vortex_fields:
            field_name, scalar_name = vortex_fields[data_key]
            source = getattr(ws.derived, field_name, None)
            support_source = ws.derived.vortex_support_mask
            if source is None or support_source is None:
                return None

            def _build_vortex_t():
                arr = np.asarray(source, dtype=np.float32)
                tidx = min(max(0, int(t)), arr.shape[3] - 1) if arr.ndim == 4 else 0
                vol_t = arr[..., tidx] if arr.ndim == 4 else arr
                support = np.asarray(support_source, dtype=bool)
                if support.ndim == 4:
                    support_t = support[..., min(max(0, int(t)), support.shape[3] - 1)]
                else:
                    support_t = support
                vol_t = np.where(support_t, vol_t, np.float32(0.0)).astype(np.float32, copy=False)
                return self._sample_supported_surface(
                    vol_t,
                    support_t,
                    0,
                    sp,
                    org,
                    name=scalar_name,
                    cache_key=data_key,
                )

            return self._cached(data_key, int(t), _build_vortex_t)

        if data_key == "pressure_gradient_volume":
            if ws.derived.pressure_gradient_magnitude is None:
                return None
            def _build_pressure_gradient_t():
                arr = np.asarray(ws.derived.pressure_gradient_magnitude, dtype=np.float32)
                support = ws.derived.pressure_gradient_support_mask
                support_source = support
                if arr.ndim == 4:
                    tidx = min(max(0, int(t)), arr.shape[3] - 1)
                    vol_t = arr[..., tidx]
                    support_t = np.asarray(support[..., tidx], dtype=bool) if support is not None and np.asarray(support).ndim == 4 else None
                else:
                    vol_t = arr
                    support_t = np.asarray(support, dtype=bool) if support is not None else None
                if support_t is None:
                    if ws.segmask_binary is not None:
                        support_source = ws.segmask_binary
                        if ws.segmask_binary.ndim == 4:
                            support_t = np.asarray(ws.segmask_binary[..., min(max(0, int(t)), ws.segmask_binary.shape[3] - 1)], dtype=bool)
                        else:
                            support_t = np.asarray(ws.segmask_binary, dtype=bool)
                    elif ws.segmask_3d is not None:
                        support_source = ws.segmask_3d
                        support_t = np.asarray(ws.segmask_3d, dtype=bool)
                    else:
                        support_t = np.ones(vol_t.shape, dtype=bool)
                        support_source = support_t
                vol_t = np.where(support_t, vol_t, 0.0)
                return self._sample_supported_surface(
                    vol_t, support_source, t, sp, org, name="PressureGradient"
                )
            return self._cached(data_key, t, _build_pressure_gradient_t)

        if data_key == "relative_pressure_volume":
            if ws.derived.relative_pressure_array is None:
                return None
            def _build_relative_pressure_t():
                arr = np.asarray(ws.derived.relative_pressure_array, dtype=np.float32)
                support = ws.derived.pressure_gradient_support_mask
                support_source = support
                if arr.ndim == 4:
                    tidx = min(max(0, int(t)), arr.shape[3] - 1)
                    vol_t = arr[..., tidx]
                    support_t = np.asarray(support[..., tidx], dtype=bool) if support is not None and np.asarray(support).ndim == 4 else None
                else:
                    vol_t = arr
                    support_t = np.asarray(support, dtype=bool) if support is not None else None
                if support_t is None:
                    if ws.segmask_binary is not None:
                        support_source = ws.segmask_binary
                        if ws.segmask_binary.ndim == 4:
                            support_t = np.asarray(ws.segmask_binary[..., min(max(0, int(t)), ws.segmask_binary.shape[3] - 1)], dtype=bool)
                        else:
                            support_t = np.asarray(ws.segmask_binary, dtype=bool)
                    elif ws.segmask_3d is not None:
                        support_source = ws.segmask_3d
                        support_t = np.asarray(ws.segmask_3d, dtype=bool)
                    else:
                        support_t = np.ones(vol_t.shape, dtype=bool)
                        support_source = support_t
                vol_t = np.where(support_t, vol_t, 0.0)
                return self._sample_supported_surface(
                    vol_t, support_source, t, sp, org, name="RelativePressure"
                )
            return self._cached(data_key, t, _build_relative_pressure_t)

        if data_key == "derived_streamlines_live":
            if not ws.derived.streamlines:
                return None
            return ws.derived.streamlines[min(max(0, t), len(ws.derived.streamlines) - 1)]

        if data_key == "pwv_planes":
            if not ws.derived.pwv_planes:
                return None
            plane_size = max(10.0, float(np.mean(np.asarray(sp, dtype=float).reshape(3))) * 12.0)
            return self._cached(data_key, 0, lambda: _pwv_planes_polydata(ws.derived.pwv_planes, org, plane_size=plane_size))

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

    def _representative_phase(self, array, t):
        arr = np.asarray(array)
        if arr.ndim < 4 or arr.shape[3] <= 1:
            return 0
        key = (id(array), tuple(int(x) for x in arr.shape), arr.dtype.str)
        lookup = self._phase_lookup_cache.get(key)
        if lookup is None:
            representatives = {}
            lookup = []
            for tidx in range(int(arr.shape[3])):
                token = np.ascontiguousarray(arr[..., tidx]).tobytes()
                rep_t = representatives.setdefault(token, int(tidx))
                lookup.append(rep_t)
            self._phase_lookup_cache[key] = lookup
        return int(lookup[min(max(0, int(t)), len(lookup) - 1)])

    def _sample_supported_surface(self, volume_t, support_4d_or_3d, t, spacing, origin, *, name, cache_key=None):
        support = np.asarray(support_4d_or_3d, dtype=bool)
        if support.ndim == 4:
            rep_t = self._representative_phase(support_4d_or_3d, t)
            support_t = support[..., rep_t]
        else:
            rep_t = 0
            support_t = support
        surface = self._cached(
            f"{str(cache_key or name)}_support_surface_{id(support_4d_or_3d)}",
            rep_t,
            lambda: build_cell_mask_surface(
                support_t, spacing, origin=origin, smooth_iter=80
            ),
        )
        return sample_volume_on_existing_surface(
            volume_t, surface, spacing, origin=origin, name=name
        )

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

    def _get_pathline_mesh(self, plane_idx, t):
        ws = self.workspace
        if int(plane_idx) not in ws.active_pathline_plane_indices:
            return None
        # A pathline is the one-cycle trajectory of particles released at
        # phase zero. Playback only reveals its cached prefix.
        launch_time = 0
        plane_cache = ws.pathline_cache.setdefault(int(plane_idx), {})
        full_pathline = plane_cache.get(launch_time)
        if full_pathline is None:
            if ws.flow_raw is None or ws.segmask_binary is None:
                return None
            if plane_idx < 0 or plane_idx >= len(ws.planes):
                return None
            plane = ws.planes[int(plane_idx)]
            p = ws.streamline_params
            mask_t = ws.segmask_binary[..., launch_time]
            seed_cache = getattr(ws, "pathline_seed_cache", {})
            seeds = seed_cache.get(int(plane_idx))
            if seeds is None:
                seeds = _plane_seeds(
                    mask_t,
                    plane,
                    ws.resolution,
                    ws.origin,
                    seed_ratio=p.pathline_seed_ratio,
                    min_seeds=p.pathline_min_seeds,
                    max_seeds=getattr(p, "pathline_max_seeds", 250),
                    rng_seed=p.pathline_rng_seed,
                    branch_labels_3d=ws.branch_labels,
                    t=launch_time,
                    seed_mode=getattr(p, "pathline_seed_mode", "fixed"),
                )
                if seeds is not None:
                    seed_cache[int(plane_idx)] = np.asarray(seeds, dtype=float)
            full_pathline = generate_pathlines_from_plane_at_t(
                ws.flow_raw, launch_time, plane, ws.resolution, ws.origin,
                mask_4d=ws.segmask_binary,
                mask_3d=mask_t,
                max_steps=p.pathline_max_steps,
                terminal_speed=p.pathline_terminal_speed,
                seed_ratio=p.pathline_seed_ratio,
                min_seeds=p.pathline_min_seeds,
                rng_seed=p.pathline_rng_seed,
                rr=ws.rr,
                branch_labels_3d=ws.branch_labels,
                max_seeds=getattr(p, "pathline_max_seeds", 250),
                seeds=seeds,
                seed_mode=getattr(p, "pathline_seed_mode", "fixed"),
            )
            plane_cache[launch_time] = full_pathline
        time_count = max(1, int(ws.time_count()))
        progress = float(np.clip(int(t), 0, time_count - 1)) / float(max(time_count - 1, 1))
        return self._cached(
            f"pathline_display_{int(plane_idx)}",
            int(t),
            lambda: pathline_prefix_at_phase(full_pathline, progress),
        )

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
        render_cfg = dict(getattr(ws, "render_settings", {}) or {})
        self.logger(f"Streamlines enabled: seed_ratio={p.seed_ratio} max_steps={p.max_steps} min_seeds={p.min_seeds} terminal_speed={p.terminal_speed} rng_seed={p.rng_seed}")
        ws.remove_object_by_data_key("streamlines_live")
        ws.add_object(name="streamlines", kind=ObjectKind.FLOW,
                      data_key="streamlines_live", visible=True, opacity=1.0,
                      scalars="Velocity", cmap="turbo", clim=render_cfg.get("streamline_clim"), dynamic=True,
                      show_scalar_bar=bool(render_cfg.get("streamline_show_scalar_bar", True)), scalar_bar_title="Velocity (m/s)",
                      scalar_bar_cfg=dict(render_cfg.get("streamline_bar_cfg", {}) or {}),
                      tube_radius=ws.streamline_params.tube_radius)
        self.sync_from_workspace()

    def trigger_pathlines(self, plane_indices=None, *, precomputed=None):
        ws = self.workspace
        if ws.flow_raw is None or ws.segmask_3d is None:
            self.logger("Cannot generate Pathlines: need flow + segmask_3d")
            return
        if plane_indices is None:
            plane_indices = list(range(len(ws.planes)))
        valid = sorted({int(idx) for idx in plane_indices if 0 <= int(idx) < len(ws.planes)})
        if not valid:
            self.logger("Cannot generate Pathlines: no valid planes")
            return

        # Pathlines are accumulated per launch plane.  In particular, a
        # single-plane request after an all-plane request must not discard the
        # already integrated planes or their t=0 trajectory/seed caches.
        active = {
            int(idx)
            for idx in ws.active_pathline_plane_indices
            if 0 <= int(idx) < len(ws.planes)
        }
        active.update(valid)
        ws.active_pathline_plane_indices = sorted(active)
        for plane_idx, time_meshes in dict(precomputed or {}).items():
            plane_idx = int(plane_idx)
            if plane_idx not in valid:
                continue
            plane_cache = ws.pathline_cache.setdefault(plane_idx, {})
            plane_cache.update({
                int(time_idx): mesh for time_idx, mesh in dict(time_meshes or {}).items()
            })
        p = ws.streamline_params
        self.logger(f"Pathlines enabled for planes {valid} from t=0: seed_mode={getattr(p, 'pathline_seed_mode', 'fixed')} seed_ratio={p.pathline_seed_ratio} min_seeds={p.pathline_min_seeds} seed_count_or_limit={getattr(p, 'pathline_max_seeds', 250)} max_steps={p.pathline_max_steps} terminal_speed={p.pathline_terminal_speed} rng_seed={p.pathline_rng_seed} color_mode={getattr(p, 'pathline_color_mode', 'per_plane')}")
        for plane_idx in valid:
            group_name = str(getattr(ws.planes[int(plane_idx)], "group_name", "") or "")
            if group_name:
                name = f"pathline {int(plane_idx)}"
                data_key = f"pathline_{group_name}_{int(plane_idx)}"
            else:
                name = f"pathline {int(plane_idx)}"
                data_key = f"pathline_{int(plane_idx)}"
            if any(obj.data_key == data_key for obj in ws.scene_objects.values()):
                continue
            ws.add_object(
                name=name,
                kind=ObjectKind.FLOW,
                data_key=data_key,
                group_name=group_name,
                browser_color=ws.skeleton_params.browser_color_for_group(group_name) if group_name else "",
                visible=True,
                opacity=1.0,
                color=ws.pathline_color_for_plane(plane_idx),
                dynamic=True,
                show_scalar_bar=False,
                tube_radius=ws.streamline_params.pathline_tube_radius,
            )
        self.invalidate_cache("pathline_")
        self.sync_from_workspace()

    def trigger_plane_streamlines(self, plane_idx):
        self.trigger_pathlines([int(plane_idx)])

    def clear_streamlines(self):
        self.workspace.clear_streamlines()
        self.invalidate_cache("streamlines")
        self.sync_from_workspace()
        self.logger("Streamlines cleared")

    def clear_pathlines(self):
        self.workspace.clear_pathlines()
        self.invalidate_cache("pathline_")
        self.sync_from_workspace()
        self.logger("Pathlines cleared")

    def clear_plane_streamlines(self):
        self.clear_pathlines()

    def find_plane_uid_at_position(self, picked_point):
        ws = self.workspace
        if picked_point is None:
            return None, None
        picked = self.display_to_world_point(picked_point)
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
        picked = self.display_to_world_point(picked_point)
        best_uid, best_idx, best_dist = None, None, float("inf")
        org = np.asarray(ws.origin, dtype=float).reshape(3)
        for uid, obj in ws.scene_objects.items():
            if obj.kind != ObjectKind.BRANCH:
                continue
            pidx = _parse_indexed_data_key(obj.data_key, "smooth_path")
            if pidx is None:
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
            iren = self.plotter.iren
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
