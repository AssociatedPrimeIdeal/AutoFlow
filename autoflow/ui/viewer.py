import numpy as np
from ..config import METRIC_DATA_KEYS
from ..rendering.style import configured_metric_style, metric_colormap
from ..rendering.datasets import metric_display_mesh
import hashlib
import weakref
from contextlib import contextmanager
import pyvista as pv
from pyvista import _vtk

from ..core.models import ObjectKind
from ..core.cache import ByteLRUCache
from ..algorithms.flow_display import clip_flow_lines, pathline_display, pathline_color_style
from ..algorithms.flow_seeding import plane_seed_points, volume_seed_domain, phase_mask
from ..config import BACKGROUND_COLOR
from ..rendering.style import (
    configure_plotter, scalar_bar_args, style_scalar_bar, surface_style, plane_display_size,
    wss_display_surface, metric_volume_kwargs, apply_metric_volume_opacity,
)
from ..rendering.datasets import (tke_display_mesh, display_support_surface,
                                  sample_display_field, active_wrap_mesh, noise_display_points)
from .content_rendering import ContentRenderingMixin
from .interaction_rendering import InteractionRenderingMixin
from .scene_layers import SceneLayersMixin
from .orientation_marker import AnatomicalOrientationMarker
from ..algorithms import (
    build_multilabel_surface_t,
    build_surface_from_mask3d,
    graph_to_polydata,
    generate_seed_points,
    generate_streamlines_at_t,
    generate_streamlines_from_plane_at_t,
    generate_pathlines_from_plane_at_t,
    create_uniform_grid,
)
from ..algorithms.surfaces import build_cell_mask_surface
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


class SceneController(ContentRenderingMixin, InteractionRenderingMixin, SceneLayersMixin):
    def __init__(self, plotter, workspace, logger):
        self.plotter = plotter
        self.workspace = workspace
        self.logger = logger
        self._axes_shown = True
        self._orientation_marker = None
        self._mesh_cache = ByteLRUCache()
        self._display_mesh_cache = ByteLRUCache(512 * 1024 ** 2)
        self._phase_lookup_cache = ByteLRUCache(4 * 1024 ** 2)
        self._render_depth = 0
        self._render_pending = False
        self._render_timer = None
        self._automatic_clim_cache = {}
        self._tracked_actors = {}
        self._saved_camera = None
        self._background_color = BACKGROUND_COLOR
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
        self._plane_selection_locked = False
        self._active_scalar_bar_uid = None
        self._volume_range_cache = {}
        self._volume_scalar_range_cache = ByteLRUCache(128 * 1024)
        self._volume_wl_observer_ids = []
        self._volume_wl_active = False
        self._volume_wl_pending_event = None
        self._volume_wl_uid = None
        self._volume_wl_start = None
        self._volume_wl_base = None
        self._interaction_last_position = None
        self._volume_wl_user_callback = None
        self._qt_mouse_filter = None
        self._plane_overlay_renderer = None
        self._initialize_scene_layers()
        self._initialize_content_rendering()
        self._initialize_interaction_rendering()

    def request_render(self, *, defer=True):
        self._render_pending = True
        if self._render_depth:
            return
        if defer:
            from PySide6 import QtCore, QtWidgets
            if QtWidgets.QApplication.instance() is not None:
                if self._render_timer is None:
                    self._render_timer = QtCore.QTimer()
                    self._render_timer.setSingleShot(True)
                    self._render_timer.timeout.connect(self.flush_render)
                if not self._render_timer.isActive():
                    self._render_timer.start(0)
                return
        self.flush_render()

    def flush_render(self):
        if self._render_timer is not None:
            self._render_timer.stop()
        if self._render_pending and not self._render_depth:
            self._render_pending = False
            self.plotter.render()

    def cancel_pending_render(self):
        if self._render_timer is not None:
            self._render_timer.stop()
        self._render_pending = False

    @contextmanager
    def render_batch(self):
        self._render_depth += 1
        try:
            yield
        finally:
            self._render_depth -= 1
            if not self._render_depth:
                self.flush_render()

    def _is_volume_object(self, obj):
        if self.is_content_render(obj):
            return self.content_render_settings(obj.data_key)["mode"] != "slices"
        return obj.data_key == "tke_volume" and configured_metric_style(self.workspace, "tke_volume")["mode"] == "volume"

    @staticmethod
    def _foreground_volume_range(values):
        finite = values[np.isfinite(values) & (values > 0.0)]
        if not finite.size:
            return 0.0, 1.0
        lo, hi = (float(value) for value in np.percentile(finite, (5.0, 99.0)))
        return lo, hi if hi > lo else lo + 1.0

    def _volume_mapper_name(self):
        try:
            if self.plotter.render_window.GetClassName() == "vtkOSOpenGLRenderWindow":
                return "fixed_point"
        except AttributeError:
            pass
        return "smart"

    def _volume_scalar_range(self, obj, data, respect_clim=True):
        """Resolve the PC-MRA window from the current frame foreground."""
        if obj.data_key == 'tke_volume' and obj.clim is None:
            values = self.workspace.derived.tke_array
            values = np.asarray(data['TKE'] if values is None else values)
            finite = values[np.isfinite(values)]
            return 0.0, max(float(np.max(finite)) if finite.size else 1.0, 1e-6)
        range_cache_key = None
        range_cache_stamp = None
        if respect_clim and getattr(obj, "clim", None) is not None:
            lo, hi = (float(x) for x in obj.clim)
        else:
            if self.is_content_render(obj):
                if data is not None and "Content display range" in data.field_data:
                    lo, hi = (float(value) for value in data.field_data["Content display range"])
                    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                        return lo, hi
            if obj.data_key == "pcmra_volume" and self.workspace.pcmra_array is not None:
                key = (id(self.workspace.pcmra_array), id(self.workspace.pcmra_render_mask), int(self.workspace.current_t))
                cached = self._volume_range_cache.get(key)
                if cached is not None:
                    return cached
                volume = self._build_dataset(obj.data_key)
                cached = self._volume_range_cache.get(key)
                if cached is None and volume is not None and "PC-MRA display range" in volume.field_data:
                    cached = tuple(float(value) for value in volume.field_data["PC-MRA display range"])
                    self._volume_range_cache[key] = cached
                if cached is not None:
                    return cached
            values = None
            name = str(getattr(obj, "scalars", "") or "PC-MRA")
            try:
                if name in data.cell_data:
                    attributes, association = data.cell_data, "cell"
                    array = data.GetCellData().GetArray(name)
                elif name in data.point_data:
                    attributes, association = data.point_data, "point"
                    array = data.GetPointData().GetArray(name)
                else:
                    attributes, array = None, None
                if array is not None:
                    range_cache_key = (id(data), association, name)
                    range_cache_stamp = int(array.GetMTime())
                    cached = self._volume_scalar_range_cache.get(range_cache_key)
                    if cached is not None and cached[0]() is data and cached[1] == range_cache_stamp:
                        return cached[2]
                if attributes is not None:
                    values = np.asarray(attributes[name], dtype=float)
            except Exception:
                values = None
            finite = values[np.isfinite(values)] if values is not None else np.empty(0)
            lo = 0.0
            hi = float(np.percentile(finite, 99.5)) if finite.size else 1.0
        if not np.isfinite(lo):
            lo = 0.0
        if not np.isfinite(hi) or hi <= lo:
            hi = lo + 1.0
        result = (float(lo), float(hi))
        if range_cache_key is not None:
            self._volume_scalar_range_cache[range_cache_key] = (weakref.ref(data), range_cache_stamp, result)
        return result

    def _set_volume_opacity(self, obj, data=None):
        """Update window/level and opacity without recreating the volume actor."""
        actor = getattr(obj, "actor", None)
        if actor is None or not self._is_volume_object(obj):
            return
        try:
            prop = actor.GetProperty()
            if data is None:
                data = self._build_dataset(obj.data_key)
            if data is None:
                return
            lo, hi = self._volume_scalar_range(obj, data)
            if obj.data_key == 'tke_volume':
                actor.GetMapper().scalar_range = (lo, hi)
                apply_metric_volume_opacity(actor, (lo, hi), obj.opacity, self.workspace.resolution,
                                            workspace=self.workspace, data_key=obj.data_key)
                return
            mapper = actor.GetMapper()
            # AutoFlow builds both transfer functions below. Avoid PyVista
            # rebuilding its 256-entry functions first on every range change.
            # Older PyVista versions without this observer hook keep the
            # original (correct, but slower) path.
            try:
                lookup = getattr(prop, "_lookup_table", None)
                observer = getattr(prop, "_lookup_table_observer_id", None)
                if observer is not None and lookup is mapper.lookup_table:
                    lookup.RemoveObserver(observer)
                    prop._lookup_table_observer_id = None
            except (AttributeError, TypeError):
                pass
            try:
                if tuple(mapper.scalar_range) != (lo, hi):
                    mapper.scalar_range = (lo, hi)
            except Exception:
                pass
            try:
                # Changing the LUT range makes PyVista replace both transfer
                # functions. Fetch them afterwards to avoid modifying an old
                # opacity function that is no longer attached to the actor.
                transfer = prop.GetScalarOpacity(0)
                if transfer is None:
                    return
                colors = prop.GetRGBTransferFunction(0)
                colors.RemoveAllPoints()
                anatomy = (self.content_render_settings(obj.data_key) if self.is_content_render(obj) else None)
                if anatomy:
                    lut = pv.LookupTable(cmap=obj.cmap, n_values=32)
                    for fraction, rgba in zip(np.linspace(0, 1, 32), lut.values):
                        colors.AddRGBPoint(lo + (hi - lo) * fraction, *(np.asarray(rgba[:3]) / 255.0))
                    prop.SetInterpolationTypeToLinear()
                    prop.SetShade(bool(anatomy["shade"]))
                else:
                    for fraction in (0.0, 0.18, 0.42, 0.68, 1.0):
                        colors.AddRGBPoint(lo + (hi - lo) * fraction, fraction, fraction, fraction)
                    prop.SetInterpolationTypeToNearest()
                prop.SetScalarOpacityUnitDistance(max(float(np.mean(np.asarray(self.workspace.resolution, dtype=float))), 0.1))
                transfer.RemoveAllPoints()
                width = max(float(hi - lo), 1e-6)
                opacity_scale = float(np.clip(getattr(obj, "opacity", 1.0), 0.0, 1.0))
                for fraction, alpha in (
                    (0.00, 0.00),
                    (0.05, 0.00),
                    (0.18, 0.015),
                    (0.38, 0.05),
                    (0.62, 0.14),
                    (0.82, 0.28),
                    (1.00, 0.42),
                ):
                    value = lo + width * fraction
                    transfer.AddPoint(value, alpha * opacity_scale if value > 0.0 else 0.0)
                transfer.AddPoint(0.0, 0.0)
                if anatomy:
                    cutoff = lo + width * float(anatomy["threshold"])
                    transfer.RemoveAllPoints()
                    transfer.AddPoint(min(0.0, lo), 0.0)
                    transfer.AddPoint(cutoff, 0.0)
                    for fraction, alpha in ((0.18, 0.015), (0.38, 0.05), (0.62, 0.14), (0.82, 0.28), (1.0, 0.42)):
                        value = cutoff + (hi - cutoff) * fraction
                        transfer.AddPoint(value, alpha * opacity_scale)
                    if lo < 0:
                        extent = max(abs(lo), abs(hi))
                        cutoff = extent * float(anatomy["threshold"])
                        transfer.RemoveAllPoints()
                        for value in np.linspace(lo, hi, 129):
                            alpha = max(0.0, (abs(value) - cutoff) / max(extent - cutoff, 1e-12))
                            transfer.AddPoint(float(value), 0.42 * opacity_scale * alpha)
                        transfer.AddPoint(0.0, 0.0)
            except Exception:
                pass
        except Exception:
            pass

    def _set_noise_point_opacity(self, obj):
        data = obj.actor.GetMapper().dataset
        weights = np.asarray(data.point_data['Noise mask'], dtype=float)
        rgba = np.zeros((data.n_points, 4), dtype=np.uint8)
        rgba[:, :3] = pv.Color(obj.color or '#ff0000').int_rgb
        rgba[:, 3] = np.rint(255 * np.clip(weights * obj.opacity, 0.0, 1.0)).astype(np.uint8)
        data.point_data['Noise RGBA'] = rgba
        obj.actor.GetMapper().Update()
        obj.actor.GetProperty().SetOpacity(1.0)
        obj.actor.GetProperty().SetPointSize(float(obj.point_size))

    def initialize(self):
        configure_plotter(self.plotter, self._background_color, workspace=self.workspace)
        self._ensure_interaction_rendering()
        self._ensure_plane_overlay_renderer()
        self._ensure_scene_layers()
        self._add_orientation_axes()
        self._ensure_volume_window_level_interaction()
        self._ensure_native_qt_mouse_bridge()
        # Start from the patient's anterior side, matching the marker's face.
        camera = self.plotter.camera
        center = np.asarray(camera.GetFocalPoint())
        distance = max(np.linalg.norm(np.asarray(camera.GetPosition()) - center), 1.0)
        camera.SetPosition(*(center + np.array([0., -1., 0.]) * self._display_axis_signs * distance))
        camera.SetViewUp(*(np.array([0., 0., 1.]) * self._display_axis_signs))
        self.reset_camera(render=False)

    def _ensure_plane_overlay_renderer(self):
        render_window = getattr(self.plotter, "render_window", None)
        main_renderer = getattr(self.plotter, "renderer", None)
        if render_window is None or main_renderer is None:
            return None
        try:
            render_window.SetNumberOfLayers(max(2, render_window.GetNumberOfLayers()))
        except Exception:
            return None
        if self._plane_overlay_renderer is None:
            try:
                renderer = _vtk.vtkRenderer()
                renderer.SetLayer(1)
                renderer.SetInteractive(0)
                # Clear this layer's depth on every frame while retaining the
                # main scene's colours. EraseOff also disables depth clearing.
                renderer.SetErase(True)
                renderer.SetPreserveColorBuffer(True)
                renderer.SetPreserveDepthBuffer(False)
                renderer.SetBackgroundAlpha(0.0)
                renderer.SetActiveCamera(main_renderer.GetActiveCamera())
                render_window.AddRenderer(renderer)
                self._plane_overlay_renderer = renderer
            except Exception:
                self._plane_overlay_renderer = None
        else:
            try:
                self._plane_overlay_renderer.SetActiveCamera(main_renderer.GetActiveCamera())
            except Exception:
                pass
        return self._plane_overlay_renderer

    def _move_actor_to_plane_overlay(self, actor):
        overlay = self._ensure_plane_overlay_renderer()
        if actor is None or overlay is None:
            return
        try:
            self.plotter.renderer.RemoveActor(actor)
        except Exception:
            pass
        try:
            overlay.AddActor(actor)
        except Exception:
            pass

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
            interactor = getattr(widget.iren, "interactor", widget.iren)
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
                marker = controller._orientation_marker
                if marker is not None and (mouse_move or event.button() == QtCore.Qt.MouseButton.LeftButton):
                    kind = "move" if mouse_move else ("press" if button_press else "release")
                    if marker.dispatch(kind):
                        return True
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
                self.set_interaction_preview(True)
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
                self.set_interaction_preview(False)
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
                self._set_volume_opacity(obj)
                actor.GetProperty().Modified()
            except Exception:
                pass
        if render:
            try:
                self.request_render()
            except Exception:
                pass

    def reset_volume_window_level(self):
        """Restore the automatic current-frame PC-MRA window and level."""
        obj = self._visible_volume_object()
        if obj is None:
            return False
        obj.clim = None
        self._update_dynamic_object(obj)
        try:
            self.request_render()
        except Exception:
            pass
        return True

    def _add_orientation_axes(self):
        if not callable(getattr(self.plotter, "add_orientation_widget", None)):
            return
        if self._orientation_marker is not None:
            self._orientation_marker.close()
        self._orientation_marker = AnatomicalOrientationMarker(self)
        self.plotter._autoflow_orientation_dispatch = self._orientation_marker.dispatch
        if not self._axes_shown:
            self.plotter.hide_axes()

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
        key = (id(data), int(data.GetMTime()), tuple(float(value) for value in self._display_axis_signs), center)
        cached = self._display_transform_cache.get(key)
        if cached is not None:
            return cached[1]
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
        # Retain the source with its transform: temporary cut/axis meshes can
        # otherwise reuse an object's Python id and retrieve unrelated scalars.
        if len(self._display_transform_cache) >= 64:
            self._display_transform_cache.pop(next(iter(self._display_transform_cache)))
        self._display_transform_cache[key] = (data, transformed)
        return transformed

    def reset_scene(self):
        self.set_interaction_preview(False, render=False)
        self._clear_scene_layers()
        self._clear_content_widgets()
        self._content_selected_uid = None
        self._ensure_content_source()
        if self._plane_overlay_renderer is not None:
            try:
                self._plane_overlay_renderer.RemoveAllViewProps()
            except Exception:
                pass
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
        self._volume_scalar_range_cache.clear()
        self._display_transform_cache.clear()
        self._active_scalar_bar_uid = None
        self._remove_plane_highlight()
        self._remove_path_highlight()
        self.initialize()

    def reset_display_reference(self):
        self._display_center = None
        self._display_transform_cache.clear()

    def invalidate_cache(self, prefix=None):
        manager = getattr(self, "_streamline_requests", None)
        if manager is not None and (prefix is None or str(prefix).startswith("streamlines")):
            manager.invalidate()
        self._content_source._cache.clear()
        self._phase_lookup_cache.clear()
        self._automatic_clim_cache.clear()
        self._volume_range_cache.clear()
        self._volume_scalar_range_cache.clear()
        self._display_transform_cache.clear()
        if prefix is None:
            self._mesh_cache.clear()
            self._display_mesh_cache.clear()
        else:
            for cache in (self._mesh_cache, self._display_mesh_cache):
                for key in list(cache):
                    if str(key[0]).startswith(prefix):
                        del cache[key]

    def set_background(self, color):
        self._background_color = str(color or BACKGROUND_COLOR)
        self.workspace.render_settings["render_background_color"] = self._background_color
        self.plotter.set_background(self._background_color)
        self.render_all()

    def toggle_axes(self):
        self._axes_shown = not self._axes_shown
        try:
            if self._axes_shown:
                self.plotter.show_axes()
            else:
                self.plotter.hide_axes()
            self.request_render()
        except Exception:
            pass

    def reset_camera(self, render=True):
        try:
            self.plotter.reset_camera(bounds=self._visible_scene_bounds(), render=False)
            if render:
                self.request_render()
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
        if self._highlight_plane_uid is not None and self._highlight_plane_uid not in current_uids:
            self._remove_plane_highlight()
        if self._highlight_path_uid is not None and self._highlight_path_uid not in current_uids:
            self._remove_path_highlight()
            self._clear_fork_and_context_actors()
        stale = set(self._tracked_actors.keys()) - current_uids
        for uid in stale:
            actor = self._tracked_actors.pop(uid, None)
            self._detach_actor(actor)
        for obj in self.workspace.scene_objects.values():
            rebuild = prefixes is None or any(obj.data_key.startswith(prefix) for prefix in prefixes)
            if obj.actor is None or rebuild:
                self._render_object(obj, refresh_scalar_bar=False)
            else:
                self._apply_basic_properties_only(obj)
        self._refresh_shared_scalar_bar(render=False)
        if not had_rendered_scene and self._tracked_actors:
            self.reset_camera(render=False)
        try:
            self.request_render()
        except Exception:
            pass

    def remove_object(self, uid):
        if self._content_selected_uid == uid:
            self.select_content_render(None)
        obj = self.workspace.scene_objects.get(uid)
        if obj is not None:
            self._remove_actor(obj)
            del self.workspace.scene_objects[uid]
        actor = self._tracked_actors.pop(uid, None)
        self._detach_actor(actor)
        if self._highlight_plane_uid == uid:
            self._remove_plane_highlight()
        if self._highlight_path_uid == uid:
            self._remove_path_highlight()
        self._refresh_shared_scalar_bar(render=False)
        try:
            self.request_render()
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
            self.reset_camera(render=False)
        try:
            self.request_render()
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
            self.request_render()
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
        try:
            self._segmentation_category_metadata(obj, data)
            data_show = self._display_dataset(obj, data)
            data_show = self._transform_display_dataset(data_show)
            mapper = obj.actor.GetMapper()
            if self._is_volume_object(obj):
                # Select the scalar explicitly on each cached volume. Meshes
                # keep the PyVista mapper pipeline below for LUT colors.
                data_show.set_active_scalars(str(obj.scalars or "PC-MRA"), preference="point")
            # Preserve PyVista's active-scalar pipeline when swapping phases.
            # Raw VTK SetInputData leaves the mapper's scalar texture connected
            # to the previous dataset and renders much of the new mesh black.
            mapper.dataset = data_show
            mapper.Update()
            if self._is_volume_object(obj):
                self._set_volume_opacity(obj, data_show)
                self._apply_interaction_mapper_quality(obj)
            elif (self.content_render_settings(obj.data_key) if self.is_content_render(obj) else None):
                mapper.scalar_range = self._volume_scalar_range(obj, data_show)
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
                self.request_render()
            except Exception:
                pass
        return updated

    def apply_object_properties(self, obj, *, render=True, refresh_scalar_bar=True):
        if self.is_content_render(obj):
            self._sync_content_widgets()
        self._sync_plane_highlight_visibility(obj)
        if obj.actor is None:
            self._render_object(obj, refresh_scalar_bar=refresh_scalar_bar)
            if render:
                self.request_render()
            return
        was_visible = False
        try:
            was_visible = bool(obj.actor.GetVisibility())
        except Exception:
            pass
        if obj.dynamic and obj.visible and not was_visible:
            self._update_dynamic_object(obj)
        try:
            obj.actor.SetVisibility(1 if self._object_visible_in_view(obj) else 0)
        except Exception:
            pass
        if self._is_volume_object(obj):
            self._set_volume_opacity(obj)
            if refresh_scalar_bar:
                self._refresh_shared_scalar_bar(render=False)
            if render:
                try:
                    self.request_render()
                except Exception:
                    pass
            return
        if obj.data_key == "noise_region":
            self._set_noise_point_opacity(obj)
            if render:
                self.request_render()
            return
        try:
            prop = obj.actor.GetProperty()
            prop.SetOpacity(float(obj.opacity))
            prop.SetLineWidth(max(float(obj.line_width), 4.0) if obj.kind == ObjectKind.PLANE else float(obj.line_width))
            prop.SetPointSize(float(obj.point_size))
            if obj.kind == ObjectKind.PLANE:
                prop.SetRepresentationToWireframe()
                prop.SetLighting(False)
                try:
                    obj.actor.ForceOpaqueOn()
                except Exception:
                    pass
            if not obj.scalars and obj.color:
                prop.SetColor(*pv.Color(obj.color).float_rgb)
        except Exception:
            pass
        if refresh_scalar_bar:
            self._refresh_shared_scalar_bar(render=False)
        if render:
            try:
                self.request_render()
            except Exception:
                pass


    def plane_display_size(self):
        return plane_display_size(self.workspace)

    def _object_visible_in_view(self, obj):
        if not bool(obj.visible):
            return False
        if obj.kind == ObjectKind.PLANE:
            cfg = self.workspace.render_settings.get("plane_render_cfg", {}) or {}
            if cfg.get("selected_only", False):
                return obj.uid == self._highlight_plane_uid
        return True

    def _sync_plane_highlight_visibility(self, obj):
        if obj.uid == self._highlight_plane_uid and self._highlight_plane_actor is not None:
            self._highlight_plane_actor.SetVisibility(1 if self._object_visible_in_view(obj) else 0)

    def _refresh_plane_visibility(self):
        for obj in self.workspace.scene_objects.values():
            if obj.kind != ObjectKind.PLANE:
                continue
            if obj.actor is None and self._object_visible_in_view(obj):
                self._render_object(obj, refresh_scalar_bar=False)
            elif obj.actor is not None:
                self._apply_basic_properties_only(obj)

    def set_plane_display_settings(self, *, size_mm=None, selected_only=None, render=True):
        cfg = self.workspace.render_settings.setdefault("plane_render_cfg", {})
        if size_mm is not None:
            value = float(size_mm)
            if not np.isfinite(value) or not 1.0 <= value <= 200.0:
                raise ValueError("plane display size must be between 1 and 200 mm")
            cfg["plane_size_mm"] = value
            for obj in self.workspace.scene_objects.values():
                if obj.kind == ObjectKind.PLANE:
                    self.update_plane_geometry(obj.uid, render=False)
        if selected_only is not None:
            cfg["selected_only"] = bool(selected_only)
        self._refresh_plane_visibility()
        if render:
            self.request_render()

    def highlight_plane(self, uid):
        self._remove_plane_highlight()
        self._highlight_plane_uid = uid
        self._refresh_plane_visibility()
        if uid is None:
            try:
                self.request_render()
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
            self._sync_plane_highlight_visibility(obj)
        except Exception:
            self._highlight_plane_actor = None
        try:
            self.request_render()
        except Exception:
            pass

    def highlight_path(self, uid):
        self._remove_path_highlight()
        self._highlight_path_uid = uid
        if uid is None:
            try:
                self.request_render()
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
            self.request_render()
        except Exception:
            pass

    def show_forks_for_path(self, path_idx):
        self._clear_fork_and_context_actors()
        if int(path_idx) < 0:
            try:
                self.request_render()
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
            self.request_render()
        except Exception:
            pass

    def _remove_plane_highlight(self):
        if self._highlight_plane_actor is not None:
            self._forget_front_prop(self._highlight_plane_actor)
            if self._plane_overlay_renderer is not None:
                try:
                    self._plane_overlay_renderer.RemoveActor(self._highlight_plane_actor)
                except Exception:
                    pass
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
            self._forget_front_prop(self._highlight_path_actor)
            if self._plane_overlay_renderer is not None:
                try:
                    self._plane_overlay_renderer.RemoveActor(self._highlight_path_actor)
                except Exception:
                    pass
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
            if self._plane_overlay_renderer is not None:
                try:
                    self._plane_overlay_renderer.RemoveActor(self._highlight_fork_actor)
                except Exception:
                    pass
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
                if self._plane_overlay_renderer is not None:
                    try:
                        self._plane_overlay_renderer.RemoveActor(actor)
                    except Exception:
                        pass
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
        self._move_actor_to_plane_overlay(actor)
        self._place_selected_highlights()

    def refresh_plane_labels(self):
        pass

    def remove_all_plane_labels(self):
        pass

    def _detach_actor(self, actor):
        """Remove an actor from every renderer that can own a scene object."""
        if actor is not None:
            self._forget_front_prop(actor)
            if self._plane_overlay_renderer is not None:
                try:
                    self._plane_overlay_renderer.RemoveActor(actor)
                except Exception:
                    pass
            try:
                self.plotter.remove_actor(actor, render=False)
            except Exception:
                try:
                    self.plotter.renderer.RemoveActor(actor)
                except Exception:
                    pass

    def _remove_actor(self, obj):
        if self._content_widget_uid == obj.uid:
            self._clear_content_widgets()
        self._detach_actor(obj.actor)
        self._detach_actor(getattr(obj, "label_actor", None))
        self._tracked_actors.pop(obj.uid, None)
        obj.actor = None
        obj.label_actor = None

    def _render_object(self, obj, refresh_scalar_bar=True):
        if not self._object_visible_in_view(obj):
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
                if obj.kind == ObjectKind.PLANE:
                    self._move_actor_to_plane_overlay(obj.actor)
            self._tracked_actors[obj.uid] = obj.actor
            self._apply_interaction_mapper_quality(obj)
            self._apply_basic_properties_only(obj)
            self._place_selected_object(obj)
        except Exception as e:
            self.logger(f"Render failed: {obj.name}: {type(e).__name__}: {e}")
        if refresh_scalar_bar:
            self._refresh_shared_scalar_bar(preferred_uid=obj.uid if obj.visible else None, render=False)

    def _apply_basic_properties_only(self, obj):
        if self.is_content_render(obj):
            self._sync_content_widgets()
        self._sync_plane_highlight_visibility(obj)
        try:
            obj.actor.SetVisibility(1 if self._object_visible_in_view(obj) else 0)
        except Exception:
            pass
        if self._is_volume_object(obj):
            self._set_volume_opacity(obj)
            return
        if obj.data_key == "noise_region":
            self._set_noise_point_opacity(obj)
            return
        try:
            prop = obj.actor.GetProperty()
            prop.SetOpacity(float(obj.opacity))
            prop.SetLineWidth(max(float(obj.line_width), 4.0) if obj.kind == ObjectKind.PLANE else float(obj.line_width))
            prop.SetPointSize(float(obj.point_size))
            if obj.kind == ObjectKind.PLANE:
                prop.SetRepresentationToWireframe()
                prop.SetLighting(False)
                obj.actor.ForceOpaqueOn()
        except Exception:
            pass

    def _mesh_kwargs(self, obj, data):
        anatomy = (self.content_render_settings(obj.data_key) if self.is_content_render(obj) else None)
        if anatomy and anatomy["mode"] == "slices":
            # An explicit LUT uses byte alpha, not normalized 0–1 values.
            lut = np.full(256, 255, dtype=np.uint8)
            lo, hi = self._volume_scalar_range(obj, data)
            levels = np.linspace(lo, hi, 256)
            if lo < 0:
                lut[np.abs(levels) <= anatomy["threshold"] * max(abs(lo), abs(hi))] = 0
            else:
                lut[levels <= lo + (hi - lo) * anatomy["threshold"]] = 0
            return dict(scalars=obj.scalars, cmap=obj.cmap, clim=self._volume_scalar_range(obj, data),
                        opacity=lut, lighting=False, show_scalar_bar=False, render=False, reset_camera=False)
        if obj.data_key == 'tke_volume' and self._is_volume_object(obj):
            kwargs = metric_volume_kwargs(self.workspace, obj.data_key, self._volume_scalar_range(obj, data))
            return dict(kwargs, scalars='TKE', mapper=self._volume_mapper_name(),
                        show_scalar_bar=False, render=False, reset_camera=False)
        if obj.data_key == 'noise_region':
            weights = np.asarray(data['Noise mask'], dtype=float)
            rgba = np.zeros((data.n_points, 4), dtype=np.uint8)
            rgba[:, :3] = pv.Color(obj.color or '#ff0000').int_rgb
            rgba[:, 3] = np.rint(255 * np.clip(weights * obj.opacity, 0.0, 1.0)).astype(np.uint8)
            data.point_data['Noise RGBA'] = rgba
            return dict(scalars='Noise RGBA', rgba=True, opacity=1.0,
                        point_size=float(obj.point_size), render_points_as_spheres=False,
                        lighting=False, show_scalar_bar=False, render=False, reset_camera=False)
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
                "blending": anatomy["mode"] if anatomy else "composite",
                "mapper": self._volume_mapper_name(),
                "opacity_unit_distance": max(float(np.mean(np.asarray(self.workspace.resolution, dtype=float))), 0.1),
                "show_scalar_bar": False,
                "render": False,
                "reset_camera": False,
            }
        kw = {"opacity": float(obj.opacity), "show_scalar_bar": False, "render": False, "reset_camera": False}
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
                kw["cmap"] = (metric_colormap(self.workspace, obj.data_key, obj.cmap)
                              if obj.data_key in METRIC_DATA_KEYS else obj.cmap)
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
        kw.update(surface_style(quantitative=use_scalars and obj.kind != ObjectKind.SEGMENTATION, workspace=self.workspace))
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
        if obj.data_key.startswith("pathline_") and (getattr(data, "n_lines", 0) == 0 or self.workspace.streamline_params.pathline_display_style in {"particles", "trails"}):
            kw["render_points_as_spheres"] = True
            kw["point_size"] = max(float(obj.point_size), 6.0)
        if obj.kind == ObjectKind.FLOW and use_scalars:
            # Quantitative streamline colors must match the scalar bar instead
            # of being darkened by the tube surface orientation.
            kw["lighting"] = False
        if obj.kind == ObjectKind.PLANE:
            kw["style"] = "wireframe"
            kw["lighting"] = False
            kw["line_width"] = max(float(obj.line_width), 4.0)
        return kw

    def _resolved_object_clim(self, obj):
        if obj.clim is not None:
            return tuple(obj.clim)
        if obj.data_key != "streamlines_live" or obj.scalars != "Velocity":
            return None
        ws = self.workspace
        flow = ws.flow_raw
        region = ws.streamline_params.seed_region
        mask = (ws.segmask_binary if ws.segmask_binary is not None else ws.segmask_3d) if region == "segmentation" else (ws.pcmra_render_mask if region == "noise_free" else None)
        key = (id(flow), id(mask))
        clim = self._automatic_clim_cache.get(key)
        if clim is None:
            clim = automatic_streamline_clim(flow, mask)
            self._automatic_clim_cache[key] = clim
        return clim

    def _display_dataset(self, obj, data):
        if obj.data_key == "wss_surface_live":
            key = (obj.data_key, id(data), "wss_display")
            if key not in self._display_mesh_cache:
                self._display_mesh_cache[key] = wss_display_surface(data)
            return self._display_mesh_cache[key]
        if not (
            obj.tube_radius > 0
            and hasattr(data, "tube")
            and obj.kind.value in ("Graph", "Branch", "Flow", "Metric", "Skeleton")
        ):
            return data
        # Keep live streamlines as lightweight polylines.  ``PolyData.tube``
        # expands every trajectory into a large triangle mesh before VTK can
        # render it; camera interaction then has to redraw that mesh on every
        # mouse move.  ``_mesh_kwargs`` already enables VTK line tubes for the
        # flow actor, which preserves the rounded appearance without the
        # heavy CPU-side geometry expansion.
        if str(getattr(obj, "data_key", "")) == "streamlines_live":
            if self.workspace.streamline_params.display_style == "lines":
                return data
            if self._streamline_interaction_preview(obj, data):
                self._interaction_flow_uids.add(obj.uid)
                return data
        if str(getattr(obj, "data_key", "")).startswith("pathline_") and self.workspace.streamline_params.pathline_display_style in {"particles", "trails"}:
            return data
        key = (str(obj.data_key), id(data), float(obj.tube_radius), str(obj.scalars or ""))
        cached = self._display_mesh_cache.get(key)
        if cached is None:
            source = data
            if obj.data_key == "streamlines_live":
                # Auxiliary integration/vector arrays are retained in the
                # scientific polyline cache, not expanded onto every tube
                # vertex. Keep only arrays needed for this actor's colors.
                source = data.copy(deep=False)
                for attributes in (source.point_data, source.cell_data):
                    for name in list(attributes):
                        if name != obj.scalars:
                            del attributes[name]
            cached = source.tube(radius=float(obj.tube_radius))
            cached._flow_display_source = data
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
        settings = getattr(self.workspace, "render_settings", {}) or {}
        args = scalar_bar_args(
            obj.scalar_bar_title, obj.scalar_bar_cfg,
            settings.get("shared_colorbar_bar_cfg"), self.plotter.background_color, workspace=self.workspace,
        )
        if obj.kind == ObjectKind.SEGMENTATION:
            args["n_labels"] = 0
        return args

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
        self._clear_hud_scalar_bars()
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
                self.request_render()
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
            bar = self.plotter.add_scalar_bar(mapper=mapper, render=False, **self._scalar_bar_args_for_object(obj))
            style_scalar_bar(bar)
            self._place_hud_scalar_bar(bar)
            self._active_scalar_bar_uid = obj.uid
        except Exception as e:
            self._active_scalar_bar_uid = None
            self.logger(f"Scalar bar refresh failed: {obj.name}: {type(e).__name__}: {e}")
        if render:
            try:
                self.request_render()
            except Exception:
                pass

    def _voxel_centered_grid(self, values, name):
        spacing = np.asarray(self.workspace.resolution, dtype=float).reshape(-1)[:3]
        offset = 1.0 if self.workspace.segmentation_labels_1mm is not None else 0.5
        origin = np.asarray(self.workspace.origin, dtype=float).reshape(-1)[:3] - offset * spacing
        grid = pv.ImageData(
            dimensions=tuple((np.asarray(values.shape, dtype=int) + 2).tolist()),
            spacing=tuple(spacing.tolist()),
            origin=tuple(origin.tolist()),
        )
        grid.point_data[name] = np.pad(values, 1).ravel(order="F")
        return grid

    def _build_dataset(self, data_key):
        ws = self.workspace
        t = ws.current_t
        sp = ws.resolution
        seg_sp = ws.segmentation_spacing
        org = ws.origin

        if data_key.startswith("content_volume_"):
            return self._build_content_volume(data_key)

        if data_key == "magnitude_volume":
            if ws.mag_raw is None:
                return None
            magnitude = np.asarray(ws.mag_raw)
            if magnitude.ndim not in {3, 4} or not magnitude.size:
                return None
            phase = min(max(int(t), 0), magnitude.shape[3] - 1) if magnitude.ndim == 4 else 0
            def build_magnitude():
                values = magnitude[..., phase] if magnitude.ndim == 4 else magnitude
                values = np.where(np.isfinite(values) & (values > 0), values, 0).astype(np.float32)
                grid = self._voxel_centered_grid(values, "Magnitude")
                grid.field_data["Magnitude display range"] = self._foreground_volume_range(values)
                return grid
            grid = self._cached(f"magnitude_volume_{id(ws.mag_raw)}", phase, build_magnitude)
            return self._content_slice_dataset(data_key, grid)

        if data_key == 'noise_region':
            return self._cached(f'noise_region_{id(ws.pcmra_render_mask)}_{id(ws.mag_raw)}', 0,
                                lambda: noise_display_points(ws))

        if data_key == "pcmra_volume":
            if ws.pcmra_array is None:
                return None
            pcmra = np.asarray(ws.pcmra_array)
            if pcmra.ndim != 4 or pcmra.shape[3] == 0:
                return None
            cache_key = f"pcmra_volume_{id(ws.pcmra_array)}_{id(ws.pcmra_render_mask)}"
            def _build_pcmra():
                tidx = min(max(0, int(t)), pcmra.shape[3] - 1)
                values = np.asarray(pcmra[..., tidx], dtype=np.float32)
                if ws.pcmra_render_mask is not None:
                    region = np.asarray(ws.pcmra_render_mask)
                    if region.shape == pcmra.shape[:3]:
                        values = np.where(region, values, 0.0)
                    elif region.shape == pcmra.shape:
                        values = np.where(region[..., tidx], values, 0.0)
                range_key = (id(ws.pcmra_array), id(ws.pcmra_render_mask), int(t))
                self._volume_range_cache[range_key] = self._foreground_volume_range(values)
                grid = self._voxel_centered_grid(values, "PC-MRA")
                grid.field_data["PC-MRA display range"] = self._volume_range_cache[range_key]
                return grid
            return self._content_slice_dataset(data_key, self._cached(cache_key, t, _build_pcmra))

        if data_key == "segmask_raw_surface":
            seg_display = ws.segmentation_display_4d()
            if seg_display is None:
                return None
            rep_t = self._representative_phase(seg_display, t)
            return self._cached(data_key, rep_t, lambda: build_multilabel_surface_t(seg_display, rep_t, seg_sp, np.asarray(org)-0.5*seg_sp))

        if data_key == "segmask_pre_surface":
            labels = ws.segmentation_analysis_labels
            if labels is None:
                return None
            if labels.ndim == 3:
                labels = labels[..., None]
            phase = min(int(t), labels.shape[3]-1)
            return self._cached(data_key, phase, lambda: build_multilabel_surface_t(labels, phase, seg_sp, np.asarray(org)-0.5*seg_sp))

        if data_key == "segmask_3d_surface":
            if ws.segmentation_analysis_mask is None:
                return None
            return self._cached(data_key, 0, lambda: build_cell_mask_surface(ws.segmentation_analysis_mask, seg_sp, np.asarray(org)-0.5*seg_sp, smooth_iter=1000))

        if data_key in {"phase_wrap_mask", "phase_wrap_count"}:
            result = getattr(ws, "phase_unwrap_result", {}) or {}
            return self._cached(f'{data_key}_{id(result)}', int(t), lambda: active_wrap_mesh(
                result, t, sp, org, counts=data_key == 'phase_wrap_count'))

        if isinstance(data_key, str) and data_key.startswith("segmask_group_"):
            group_name = str(data_key[len("segmask_group_"):])
            group_state = ws.multilabel_groups.get(group_name, {})
            mask = group_state.get("geometry_mask_3d")
            if mask is None:
                return None
            return self._cached(
                data_key,
                0,
                lambda: build_cell_mask_surface(
                    np.asarray(mask, dtype=bool), seg_sp, np.asarray(org)-0.5*seg_sp, smooth_iter=1000
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
                    ws.skeleton_mask, getattr(ws, "skeleton_spacing", None) if getattr(ws, "skeleton_spacing", None) is not None else sp, org, smooth_iter=1000
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
            mesh = self._get_streamline_mesh(t)
            minimum = float(ws.streamline_params.display_min_speed_cm_s)
            if mesh is None or minimum == 0:
                return mesh
            visible = self._cached("streamline_display_speed", (id(mesh), minimum),
                                   lambda: clip_flow_lines(mesh, minimum))
            return visible if visible.n_points else None

        idx = _parse_indexed_data_key(data_key, "pathline")
        if idx is not None:
            return self._get_pathline_mesh(idx, t)

        if data_key in METRIC_DATA_KEYS and data_key != "streamlines_live":
            if data_key == "tke_volume" and configured_metric_style(ws, data_key)["mode"] == "volume":
                return self._cached(data_key, t, lambda: tke_display_mesh(ws, t))
            return self._cached(data_key, t, lambda: metric_display_mesh(ws, data_key, t))

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
            paths = ws.centerline_paths_smooth if ws.centerline_paths_smooth else ws.centerline_paths
            if idx >= len(paths):
                return None
            path = np.asarray(paths[idx], dtype=float)
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
            return pv.Plane(
                center=np.asarray(p.center) + np.asarray(org),
                direction=np.asarray(p.normal),
                i_size=self.plane_display_size(),
                j_size=self.plane_display_size(),
                i_resolution=1,
                j_resolution=1,
            )

        return None

    def _cached(self, data_key, t, builder):
        ws = self.workspace
        key = (data_key, t, int(ws.segmentation.revision), id(ws.flow_raw), id(ws.pcmra_array),
               id(ws.derived.wss_volume), id(ws.derived.pressure_gradient_array),
               id(ws.derived.relative_pressure_array))
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
        key = (id(array), tuple(int(x) for x in arr.shape), arr.dtype.str,
               int(self.workspace.segmentation.revision))
        lookup = self._phase_lookup_cache.get(key)
        if lookup is None:
            representatives = {}
            lookup = []
            for tidx in range(int(arr.shape[3])):
                value = np.ascontiguousarray(arr[..., tidx])
                token = hashlib.blake2b(value.view(np.uint8), digest_size=16).digest()
                candidates = representatives.setdefault(token, [])
                rep_t = next((index for index in candidates if np.array_equal(value, arr[..., index])), None)
                if rep_t is None:
                    rep_t = int(tidx)
                    candidates.append(rep_t)
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
            lambda: display_support_surface(
                support_t, spacing, origin=origin, smooth_iter=80
            ),
        )
        return sample_display_field(
            volume_t, support_t, surface, spacing, origin=origin, name=name
        )

    def _get_streamline_mesh(self, t):
        from .streamline_jobs import streamline_requests
        ws = self.workspace
        manager = streamline_requests(ws)
        previous = getattr(self, "_streamline_requests", None)
        if previous is not manager:
            if previous is not None:
                try:
                    previous.ready.disconnect(self._streamline_mesh_ready)
                    previous.failed.disconnect(self.logger)
                except (RuntimeError, TypeError):
                    pass
            manager.ready.connect(self._streamline_mesh_ready)
            manager.failed.connect(self.logger)
            self._streamline_requests = manager
        return manager.request(t)

    def _streamline_mesh_ready(self, phase):
        if (self.workspace.streamline_active and int(self.workspace.current_t) == phase
                and self.workspace.streamline_cache.get(phase) is not None):
            self.sync_from_workspace(rebuild_prefixes=("streamlines_live",))

    def _get_pathline_mesh(self, plane_idx, t):
        ws = self.workspace
        if int(plane_idx) not in ws.active_pathline_plane_indices:
            return None
        if not 0 <= int(plane_idx) < len(ws.planes) or ws.flow_raw is None:
            return None
        p = ws.streamline_params
        plane_cache = ws.pathline_cache.setdefault(int(plane_idx), {})
        launch = int(next(iter(plane_cache))) if plane_cache else (int(ws.current_t) if p.pathline_seed_region == "contour" else 0)
        if launch not in plane_cache:
            segmentation = ws.segmask_binary if ws.segmask_binary is not None else ws.segmask_3d
            if p.pathline_seed_region == "segmentation" and segmentation is None:
                return None
            plane = ws.planes[int(plane_idx)]
            seeds, roi_source = plane_seed_points(ws.flow_raw, plane, ws.resolution, ws.origin,
                phase=launch, region=p.pathline_seed_region, segmentation=segmentation,
                segmentation_labels=ws.segmask_labels, branch_labels=ws.branch_labels,
                distribution=p.pathline_seed_distribution, grid_spacing_mm=p.pathline_grid_spacing_mm,
                mode=p.pathline_seed_mode, count=p.pathline_max_seeds, ratio=p.pathline_seed_ratio,
                min_seeds=p.pathline_min_seeds, rng_seed=p.pathline_rng_seed)
            ws.pathline_seed_cache[int(plane_idx)] = seeds
            domain = segmentation if p.pathline_seed_region == "segmentation" else np.ones(ws.flow_raw.shape[:3], bool)
            mask = phase_mask(domain, ws.flow_raw.shape[:3], launch)
            mesh = generate_pathlines_from_plane_at_t(ws.flow_raw, launch, plane, ws.resolution, ws.origin,
                mask_4d=domain, mask_3d=mask, max_steps=p.pathline_max_steps,
                terminal_speed=p.pathline_terminal_speed, rng_seed=p.pathline_rng_seed, rr=ws.rr,
                seeds=seeds, max_seeds=None if p.pathline_seed_distribution == "uniform" else p.pathline_max_seeds,
                integration_step_mm=p.pathline_integration_step_mm,
                temporal_cache_mb=p.pathline_temporal_cache_mb)
            if mesh is not None:
                mesh.field_data["ROISourcePhase"] = np.asarray([-1 if roi_source is None else roi_source], np.int64)
            plane_cache[launch] = mesh
            ws.input_state.metadata.setdefault("flow_seed_counts", {}).setdefault("pathlines", {})[plane.plane_id] = len(seeds)
        full = plane_cache[launch]
        if full is None:
            return None
        if "PathlinePhase" not in getattr(full, "point_data", {}):
            return full
        time_count = max(1, int(ws.time_count()))
        progress = float((int(t) - launch) % time_count) / float(time_count)
        key = (id(full), int(t), launch, p.pathline_display_style, p.pathline_trail_duration_ms,
               p.pathline_display_min_speed_cm_s, float(ws.rr))
        result = self._cached(f"pathline_display_{int(plane_idx)}", key,
            lambda: pathline_display(full, progress, style=p.pathline_display_style,
                trail_duration_ms=p.pathline_trail_duration_ms, rr_ms=ws.rr,
                minimum_cm_s=p.pathline_display_min_speed_cm_s))
        return result if result is not None and result.n_points else None

    def trigger_streamlines(self):
        ws = self.workspace
        if ws.flow_raw is None:
            self.logger("Cannot generate streamlines: need velocity data")
            return False
        try:
            volume_seed_domain(ws.flow_raw, ws.streamline_params.seed_region,
                segmentation=ws.segmask_binary, noise_keep=ws.pcmra_render_mask, phase=int(ws.current_t))
        except ValueError as exc:
            self.logger(str(exc)); return False
        ws.streamline_seeds = None
        ws.streamline_cache.clear()
        ws.streamline_active = True
        p = ws.streamline_params
        render_cfg = dict(getattr(ws, "render_settings", {}) or {})
        self.logger(f"Streamlines enabled: seed_ratio={p.seed_ratio} max_steps={p.max_steps} min_seeds={p.min_seeds} terminal_speed={p.terminal_speed} rng_seed={p.rng_seed}")
        ws.remove_object_by_data_key("streamlines_live")
        ws.add_object(name="streamlines", kind=ObjectKind.FLOW,
                      data_key="streamlines_live", visible=True, opacity=1.0, line_width=p.line_width,
                      scalars="Velocity", cmap="turbo", clim=render_cfg.get("streamline_clim"), dynamic=True,
                      show_scalar_bar=bool(render_cfg.get("streamline_show_scalar_bar", True)), scalar_bar_title="Velocity (m/s)",
                      scalar_bar_cfg=dict(render_cfg.get("streamline_bar_cfg", {}) or {}),
                      tube_radius=ws.streamline_params.tube_radius)
        self.sync_from_workspace()
        return True

    def trigger_pathlines(self, plane_indices=None, *, precomputed=None):
        ws = self.workspace
        if ws.flow_raw is None or (ws.streamline_params.pathline_seed_region == "segmentation" and ws.segmask_binary is None and ws.segmask_3d is None):
            self.logger("Cannot generate Pathlines: need velocity and the selected seed region")
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
        self.logger(f"Pathlines enabled for planes {valid}: region={p.pathline_seed_region} distribution={p.pathline_seed_distribution} grid_spacing_mm={p.pathline_grid_spacing_mm} color_mode={p.pathline_color_mode}")
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
                **pathline_color_style(ws, plane_idx),
                dynamic=True,
                tube_radius=ws.streamline_params.pathline_tube_radius,
            )
        self.invalidate_cache("pathline_")
        self.sync_from_workspace()

    def apply_flow_display_settings(self):
        """Refresh presentation while retaining the integrated trajectory caches."""
        p = self.workspace.streamline_params
        for obj in self.workspace.scene_objects.values():
            if obj.data_key == "streamlines_live":
                obj.line_width, obj.tube_radius = p.line_width, p.tube_radius
                self.readd_object(obj, refresh_scalar_bar=False)
            else:
                index = _parse_indexed_data_key(obj.data_key, "pathline")
                if index is None:
                    continue
                obj.tube_radius = p.pathline_tube_radius
                for key, value in pathline_color_style(self.workspace, index).items():
                    setattr(obj, key, value)
                self.readd_object(obj, refresh_scalar_bar=False)
        self._refresh_shared_scalar_bar(render=False)
        self.request_render()

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
            if obj.kind != ObjectKind.PLANE or not self._object_visible_in_view(obj):
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
        return (best_uid, best_idx) if best_dist <= self.plane_display_size() else (None, None)

    def _plane_uid_at_display_position(self, x, y):
        renderer = self.plotter.renderer
        target = np.asarray([float(x), float(y)], dtype=float)
        candidates = []
        for uid, obj in self.workspace.scene_objects.items():
            if obj.kind != ObjectKind.PLANE or obj.actor is None:
                continue
            try:
                if not obj.actor.GetVisibility():
                    continue
            except Exception:
                continue
            plane_idx = _parse_indexed_data_key(obj.data_key, "plane")
            if plane_idx is None or not (0 <= plane_idx < len(self.workspace.planes)):
                continue
            plane = self.workspace.planes[plane_idx]
            center = np.asarray(plane.center, dtype=float).reshape(3) + np.asarray(self.workspace.origin, dtype=float).reshape(3)
            normal = np.asarray(plane.normal, dtype=float).reshape(3).copy()
            norm = float(np.linalg.norm(normal))
            if norm <= 1e-12:
                continue
            normal /= norm
            reference = np.eye(3, dtype=float)[int(np.argmin(np.abs(normal)))]
            axis_u = np.cross(reference, normal)
            axis_u /= np.linalg.norm(axis_u) + 1e-12
            axis_v = np.cross(normal, axis_u)
            axis_v /= np.linalg.norm(axis_v) + 1e-12
            half_size = self.plane_display_size() * 0.5
            corners = [
                center - half_size * axis_u - half_size * axis_v,
                center + half_size * axis_u - half_size * axis_v,
                center + half_size * axis_u + half_size * axis_v,
                center - half_size * axis_u + half_size * axis_v,
            ]
            projected = []
            for corner in corners:
                display_corner = self.world_to_display_point(corner)
                renderer.SetWorldPoint(*display_corner.tolist(), 1.0)
                renderer.WorldToDisplay()
                projected.append(np.asarray(renderer.GetDisplayPoint(), dtype=float))
            polygon = np.asarray(projected, dtype=float)
            if not np.all(np.isfinite(polygon)) or np.any((polygon[:, 2] < 0) | (polygon[:, 2] > 1)):
                continue
            edge = np.roll(polygon[:, :2], -1, axis=0) - polygon[:, :2]
            offset = target.reshape(1, 2) - polygon[:, :2]
            cross = edge[:, 0] * offset[:, 1] - edge[:, 1] * offset[:, 0]
            area = abs(np.sum(polygon[:, 0] * np.roll(polygon[:, 1], -1)
                              - polygon[:, 1] * np.roll(polygon[:, 0], -1))) * 0.5
            inside = area > 1.0 and (np.all(cross >= -1e-6) or np.all(cross <= 1e-6))
            fraction = np.clip(np.sum(offset * edge, axis=1) / np.maximum(np.sum(edge * edge, axis=1), 1e-12), 0, 1)
            near_edge = np.min(np.linalg.norm(offset - fraction[:, None] * edge, axis=1)) <= 6.0
            if not (inside or near_edge):
                continue
            center_display = self.world_to_display_point(center)
            renderer.SetWorldPoint(*center_display.tolist(), 1.0)
            renderer.WorldToDisplay()
            center_depth = float(renderer.GetDisplayPoint()[2])
            candidates.append((center_depth, uid, plane_idx))
        if not candidates:
            return None, None
        _depth, uid, plane_idx = min(candidates, key=lambda item: item[0])
        return uid, plane_idx

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

        def _on_right_click(obj=None, ev=None):
            if self._plane_selection_locked:
                return
            try:
                x, y = iren.GetEventPosition()
            except Exception:
                return
            ren = self.plotter.renderer
            pos = None
            # Test the full displayed plane before path actors. Thin outlines
            # should not make an underlying path steal a plane-face click.
            screen_plane_uid, screen_plane_idx = self._plane_uid_at_display_position(x, y)
            if screen_plane_uid is not None and screen_plane_idx is not None:
                if self._plane_pick_callback is not None:
                    self._plane_pick_callback(screen_plane_uid, screen_plane_idx)
                return
            for overlay in (self._selection_renderer, self._plane_overlay_renderer):
                if overlay is None:
                    continue
                overlay_ok = picker.Pick(float(x), float(y), 0.0, overlay)
                picked_actor = picker.GetActor() if overlay_ok else None
                picked_plane_uid = None
                picked_plane_idx = None
                for uid, plane in self.workspace.scene_objects.items():
                    if plane.kind != ObjectKind.PLANE or plane.actor is None:
                        continue
                    if picked_actor is plane.actor or picked_actor == plane.actor:
                        picked_plane_uid = uid
                        picked_plane_idx = _parse_indexed_data_key(plane.data_key, "plane")
                        break
                if picked_plane_uid is not None and picked_plane_idx is not None:
                    if self._plane_pick_callback is not None:
                        self._plane_pick_callback(picked_plane_uid, picked_plane_idx)
                    return
                if overlay_ok and overlay is self._selection_renderer:
                    pos = picker.GetPickPosition()
                    path_uid, path_idx = self.find_path_uid_at_position(pos)
                    if path_uid is not None and path_idx is not None and self._path_pick_callback is not None:
                        self._path_pick_callback(path_uid, path_idx)
                        return
            if pos is None:
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

        add_observer = getattr(iren, "AddObserver", None)
        if add_observer is None:
            add_observer = iren.add_observer
        self._shared_pick_obs_id = add_observer("RightButtonPressEvent", _on_right_click)
        self._plane_pick_obs_id = self._shared_pick_obs_id
        self._path_pick_obs_id = self._shared_pick_obs_id
        try:
            self.plotter._autoflow_right_click_dispatch = lambda: _on_right_click(
                iren, "RightButtonPressEvent"
            )
        except Exception:
            pass

    def enable_plane_picking(self, callback):
        self._plane_pick_callback = callback
        self._ensure_shared_right_click_picking()

    def enable_path_picking(self, callback):
        self._path_pick_callback = callback
        self._ensure_shared_right_click_picking()
