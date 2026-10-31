"""Compact in-place Skeleton/Graph editing in the existing 3-D workspace."""
import copy
import numpy as np
import pyvista as pv
from PySide6 import QtCore, QtGui, QtWidgets
from vtkmodules.vtkCommonCore import vtkFloatArray
from vtkmodules.util.numpy_support import vtk_to_numpy

from .geometry_editor import GeometryDraft
from .theme import theme_colors, standard_icon
from ..core.geometry_edit import (geometry_snapshot, restore_geometry_snapshot,
    prepare_geometry_edit, publish_geometry_edit, persist_geometry_edit)
from ..task_control import TaskCancelled


class _GeometryCommand(QtGui.QUndoCommand):
    def __init__(self, owner, before, after, text):
        super().__init__(text)
        self.owner, self.before, self.after, self.initial = owner, before, after, True

    def undo(self):
        self.owner._restore_geometry_commit(self.before)

    def redo(self):
        if self.initial:
            self.initial = False
        else:
            self.owner._restore_geometry_commit(self.after)


class _GeometryEvents(QtCore.QObject):
    def __init__(self, owner):
        super().__init__(owner)
        self.owner = owner
        self.press = None
        self.box = None
        self.drag_anchor = None
        self.drag_depth = None
        self.rubber = QtWidgets.QRubberBand(QtWidgets.QRubberBand.Rectangle, owner.plotter)

    def cancel(self):
        self.press = self.box = self.drag_anchor = self.drag_depth = None
        self.rubber.hide()

    def eventFilter(self, watched, event):
        owner = self.owner
        kind = event.type()
        editing = getattr(owner, '_edit_mode', None) in {'skeleton', 'graph'}
        if getattr(owner, '_geometry_busy', False):
            return False
        if not editing:
            if watched is owner.plotter and kind in (QtCore.QEvent.ShortcutOverride, QtCore.QEvent.KeyPress):
                control = bool(event.modifiers() & (QtCore.Qt.ControlModifier | QtCore.Qt.MetaModifier))
                if control and event.key() in (QtCore.Qt.Key_Z, QtCore.Qt.Key_Y):
                    if kind == QtCore.QEvent.ShortcutOverride:
                        event.accept()
                    else:
                        owner._edit_history_action(event.key() == QtCore.Qt.Key_Y or bool(event.modifiers() & QtCore.Qt.ShiftModifier))
                    return True
            return False
        if kind == QtCore.QEvent.Resize and watched is owner._edit_info_label:
            owner._edit_info_label.setText(owner._edit_info_label.fontMetrics().elidedText(
                owner._geometry_title, QtCore.Qt.ElideRight, max(40, owner._edit_info_label.width())))
            return False
        if kind in (QtCore.QEvent.ShortcutOverride, QtCore.QEvent.KeyPress):
            focus = QtWidgets.QApplication.focusWidget()
            if isinstance(focus, (QtWidgets.QLineEdit, QtWidgets.QTextEdit,
                                  QtWidgets.QPlainTextEdit, QtWidgets.QAbstractSpinBox)):
                return False
            modal = QtWidgets.QApplication.activeModalWidget()
            if modal is not None:
                return False
            key = event.key()
            modifier = event.modifiers()
            control = bool(modifier & (QtCore.Qt.ControlModifier | QtCore.Qt.MetaModifier))
            handled = key in (QtCore.Qt.Key_Escape, QtCore.Qt.Key_Delete, QtCore.Qt.Key_Backspace)
            handled |= control and key in (QtCore.Qt.Key_Z, QtCore.Qt.Key_Y, QtCore.Qt.Key_A)
            if not handled:
                return False
            if kind == QtCore.QEvent.ShortcutOverride:
                event.accept()
                return True
            if key == QtCore.Qt.Key_Escape:
                owner._cancel_geometry_action()
            elif key in (QtCore.Qt.Key_Delete, QtCore.Qt.Key_Backspace):
                owner._delete_geometry_selection()
            elif control and key == QtCore.Qt.Key_A:
                owner._geometry_draft.nodes = set(range(len(owner._geometry_draft.points)))
                owner._geometry_draft.edge_ids.clear()
                owner._render_geometry_selection()
            elif control:
                owner._geometry_history(redo=key == QtCore.Qt.Key_Y or bool(modifier & QtCore.Qt.ShiftModifier))
            return True
        if watched is not owner.plotter:
            return False
        if kind == QtCore.QEvent.Leave:
            if self.press is None:
                owner._geometry_hover(None)
            return False
        if kind not in (QtCore.QEvent.MouseButtonPress, QtCore.QEvent.MouseButtonRelease,
                        QtCore.QEvent.MouseButtonDblClick, QtCore.QEvent.MouseMove):
            return False
        pos = event.position().toPoint()
        draft = owner._geometry_draft
        if kind in (QtCore.QEvent.MouseButtonPress, QtCore.QEvent.MouseButtonRelease) and event.button() == QtCore.Qt.RightButton:
            if kind == QtCore.QEvent.MouseButtonRelease:
                owner._geometry_context_menu(pos)
            return True
        if kind == QtCore.QEvent.MouseMove:
            if self.box is not None:
                self.rubber.setGeometry(QtCore.QRect(self.box, pos).normalized())
                self.rubber.show()
                return True
            if self.drag_anchor is not None:
                raw = owner._geometry_unproject(pos, self.drag_depth)
                draft.move_selection(raw-self.drag_anchor)
                owner._render_geometry_draft(update_cursor=False)
                return True
            hit = owner._geometry_hit(pos)
            owner._geometry_hover(hit)
            if owner._geometry_source is not None:
                owner._geometry_connection_preview(pos, hit)
            return self.press is not None
        if kind == QtCore.QEvent.MouseButtonRelease and event.button() == QtCore.Qt.LeftButton:
            if self.box is not None:
                rect = QtCore.QRect(self.box, pos).normalized()
                owner._geometry_box_select(rect)
                self.cancel()
                return True
            if self.drag_anchor is not None:
                draft.finish_gesture()
                self.cancel()
                owner._render_geometry_draft()
                return True
            consumed = self.press is not None
            self.cancel()
            return consumed
        if event.button() != QtCore.Qt.LeftButton:
            return False
        hit = owner._geometry_hit(pos, prefer_edges=kind == QtCore.QEvent.MouseButtonDblClick)
        additive = bool(event.modifiers() & QtCore.Qt.ShiftModifier)
        if kind == QtCore.QEvent.MouseButtonDblClick and hit is not None and hit[0] == 'edge':
            draft.select_branch(hit[1], additive)
            owner._render_geometry_selection()
            self.press = pos
            return True
        if kind != QtCore.QEvent.MouseButtonPress:
            return False
        if additive and hit is None:
            self.press = self.box = pos
            self.rubber.setGeometry(QtCore.QRect(pos, pos))
            return True
        if hit is None:
            if not additive:
                draft.clear_selection()
                owner._render_geometry_selection()
            return False  # Empty-space dragging retains ordinary camera navigation.
        self.press = pos
        owner.plotter.setFocus(QtCore.Qt.MouseFocusReason)
        if owner._edit_mode == 'graph' and owner._edit_btn_edge.isChecked():
            if hit[0] == 'node':
                owner._connect_geometry_node(hit[1])
            return True
        already_selected = hit[0] == 'node' and hit[1] in draft.nodes
        if not (already_selected and not additive):
            draft.select(*hit, additive=additive)
        owner._render_geometry_selection()
        if already_selected and not additive:
            screen = owner._geometry_screen_points()[hit[1]]
            self.drag_depth = float(screen[2])
            self.drag_anchor = owner._geometry_unproject(pos, self.drag_depth)
            draft.begin_gesture()
        return True


class GeometryEditingMixin:
    def _build_geometry_edit_menu(self, menu_bar):
        menu = menu_bar.addMenu('Edit')
        self.edit_menu = menu
        self._edit_undo_action = QtGui.QAction('Undo', self)
        self._edit_undo_action.setShortcut(QtGui.QKeySequence.Undo)
        self._edit_undo_action.triggered.connect(lambda: self._edit_history_action(False))
        self._edit_redo_action = QtGui.QAction('Redo', self)
        self._edit_redo_action.setShortcuts([QtGui.QKeySequence('Ctrl+Shift+Z'), QtGui.QKeySequence('Ctrl+Y')])
        self._edit_redo_action.triggered.connect(lambda: self._edit_history_action(True))
        menu.addAction(self._edit_undo_action)
        menu.addAction(self._edit_redo_action)
        menu.aboutToShow.connect(self._update_geometry_edit_actions)
        self._plane_undo_stack.indexChanged.connect(self._update_geometry_edit_actions)
        self._update_geometry_edit_actions()

    def _edit_history_action(self, redo):
        if getattr(self, '_geometry_busy', False):
            return
        if self._edit_mode in {'skeleton', 'graph'}:
            self._geometry_history(redo)
        elif getattr(self, '_seg_edit_active', False):
            self._redo_segmentation_edit() if redo else self._undo_segmentation_edit()
        elif not self._pipeline_running_guard('editing history'):
            self._redo_plane_change() if redo else self._undo_plane_change()
        self._update_geometry_edit_actions()

    def _update_geometry_edit_actions(self, *_):
        if not hasattr(self, '_edit_undo_action'):
            return
        draft = getattr(self, '_geometry_draft', None)
        if draft is not None:
            undo = draft.undo_stack[-1][0] if draft.undo_stack else ''
            redo = draft.redo_stack[-1][0] if draft.redo_stack else ''
        else:
            undo, redo = self._plane_undo_stack.undoText(), self._plane_undo_stack.redoText()
        self._edit_undo_action.setText('Undo'+(' '+undo if undo else ''))
        self._edit_redo_action.setText('Redo'+(' '+redo if redo else ''))
        self._edit_undo_action.setEnabled(bool(undo))
        self._edit_redo_action.setEnabled(bool(redo))

    def _build_geometry_toolbar(self, layout):
        self._geometry_draft = None
        self._geometry_source = None
        self._geometry_actors = {}
        self._geometry_hover_key = None
        self._geometry_busy = False
        self._geometry_title = ''
        self._geometry_locked_controls = []
        self._edit_panel = QtWidgets.QFrame()
        self._edit_panel.setObjectName('geometryEditToolbar')
        row = QtWidgets.QHBoxLayout(self._edit_panel)
        row.setContentsMargins(8, 5, 8, 5)
        row.setSpacing(7)
        self._edit_info_label = QtWidgets.QLabel()
        self._edit_info_label.setMinimumWidth(80)
        self._edit_info_label.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        row.addWidget(self._edit_info_label, 1)
        self._edit_btn_edge = QtWidgets.QToolButton()
        self._edit_btn_edge.setText('Connect')
        self._edit_btn_edge.setCheckable(True)
        self._edit_btn_edge.setToolTip('Connect two nodes. Select an edge and press Delete to disconnect it.')
        self._edit_btn_edge.toggled.connect(self._geometry_connect_toggled)
        row.addWidget(self._edit_btn_edge)
        self._geometry_undo = QtWidgets.QToolButton()
        self._geometry_undo.setIcon(QtGui.QIcon.fromTheme('edit-undo', standard_icon(self, 'SP_ArrowBack')))
        self._geometry_undo.setAccessibleName('Undo')
        self._geometry_undo.clicked.connect(lambda: self._geometry_history(False))
        row.addWidget(self._geometry_undo)
        self._geometry_redo = QtWidgets.QToolButton()
        self._geometry_redo.setIcon(QtGui.QIcon.fromTheme('edit-redo', standard_icon(self, 'SP_ArrowForward')))
        self._geometry_redo.setAccessibleName('Redo')
        self._geometry_redo.clicked.connect(lambda: self._geometry_history(True))
        row.addWidget(self._geometry_redo)
        self._edit_btn_cancel = QtWidgets.QPushButton('Cancel')
        self._edit_btn_cancel.setFlat(True)
        self._edit_btn_cancel.setToolTip('Discard this editing session')
        self._edit_btn_cancel.clicked.connect(lambda: self._exit_interactive_edit(False))
        row.addWidget(self._edit_btn_cancel)
        self._edit_btn_save = QtWidgets.QPushButton('Done')
        self._edit_btn_save.setProperty('role', 'primary')
        self._edit_btn_save.setToolTip('Apply changes and update centerlines')
        self._edit_btn_save.clicked.connect(lambda: self._exit_interactive_edit(True))
        row.addWidget(self._edit_btn_save)
        layout.addWidget(self._edit_panel)
        self._edit_panel.hide()
        self._geometry_events = _GeometryEvents(self)
        # Application-wide Python filters can wrap Chromium internal objects
        # recursively. Only editor widgets receive this filter.
        for widget in (self.plotter, self._edit_panel, self._edit_info_label, self._edit_btn_edge,
                       self._geometry_undo, self._geometry_redo,
                       self._edit_btn_cancel, self._edit_btn_save):
            widget.installEventFilter(self._geometry_events)
        self.plotter.setMouseTracking(True)

    def _geometry_connect_toggled(self, enabled):
        self._geometry_source = None
        self._remove_geometry_actor('preview')
        if self._geometry_draft is not None:
            self._geometry_draft.clear_selection()
            self._render_geometry_selection()

    def _geometry_history(self, redo=False):
        if self._geometry_draft is None:
            return
        self._geometry_events.cancel()
        self._geometry_source = None
        self._remove_geometry_actor('preview')
        self._geometry_draft.cancel_gesture()
        self._geometry_draft.redo() if redo else self._geometry_draft.undo()
        self._render_geometry_draft()

    def _cancel_geometry_action(self):
        if self._geometry_draft is None:
            return
        self._geometry_draft.cancel_gesture()
        self._geometry_events.cancel()
        self._geometry_source = None
        self._remove_geometry_actor('preview')
        self._geometry_draft.clear_selection()
        self._edit_btn_edge.setChecked(False)
        self._render_geometry_draft()

    def _remove_geometry_actor(self, key):
        actor = self._geometry_actors.pop(key, None)
        if actor is not None:
            self.plotter.remove_actor(actor, render=False)

    def _geometry_mesh(self, key, points, edges=None, *, color=None, width=3, size=6):
        self._remove_geometry_actor(key)
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        if not len(points):
            return
        display = self._display_edit_points(points)
        poly = pv.PolyData(display)
        if edges is not None:
            edges = np.asarray(edges, dtype=int).reshape(-1, 2)
            if not len(edges):
                return
            poly.lines = np.column_stack([np.full(len(edges), 2), edges]).ravel()
        actor = self.plotter.add_mesh(poly, name='geometry_edit_'+key,
            color=color or theme_colors(self)['accent'], line_width=width,
            point_size=size, render_points_as_spheres=False,
            pickable=False, render=False)
        self._geometry_actors[key] = actor

    def _render_geometry_draft(self, update_cursor=True):
        draft = self._geometry_draft
        if draft is None:
            return
        self._edit_points, self._edit_edges = draft.points, draft.edges
        group = self._edit_group_name
        color = self.workspace.skeleton_params.scene_color_for_group(group, self._edit_mode)
        if self._edit_mode == 'graph':
            self._geometry_mesh('base', draft.points, draft.edges, color=color, width=2)
            isolated = np.setdiff1d(np.arange(len(draft.points)), draft.edges.ravel())
            self._geometry_mesh('isolated', draft.points[isolated], color=color, size=4)
        else:
            self._geometry_mesh('base', draft.points, color=color, size=4)
        self._geometry_hover_key = None
        self._remove_geometry_actor('hover')
        self._render_geometry_selection(update_cursor=update_cursor)

    def _render_geometry_selection(self, update_cursor=True):
        draft = self._geometry_draft
        if draft is None:
            return
        self._geometry_mesh('nodes', draft.points[sorted(draft.nodes)] if draft.nodes else [], size=9)
        self._geometry_mesh('edges', draft.points, draft.edges[sorted(draft.edge_ids)] if draft.edge_ids else [], width=5)
        self._geometry_undo.setEnabled(bool(draft.undo_stack))
        self._geometry_redo.setEnabled(bool(draft.redo_stack))
        self._geometry_undo.setToolTip(('Undo '+draft.undo_stack[-1][0] if draft.undo_stack else 'Undo')+' (Ctrl+Z)')
        self._geometry_redo.setToolTip(('Redo '+draft.redo_stack[-1][0] if draft.redo_stack else 'Redo')+' (Ctrl+Shift+Z)')
        self._update_geometry_edit_actions()
        if self._geometry_source is not None:
            message = 'Choose the second node. Esc cancels the connection.'
        elif draft.nodes:
            message = f'{len(draft.nodes)} '+('point' if self._edit_mode == 'skeleton' else 'node')+('s' if len(draft.nodes) != 1 else '')+' selected'
        elif draft.edge_ids:
            lengths = [np.linalg.norm(draft.points[b]-draft.points[a]) for a, b in draft.edges[sorted(draft.edge_ids)]]
            message = f'{len(draft.edge_ids)} '+('edge' if len(draft.edge_ids) == 1 else 'edges')+f' selected · {sum(lengths):.2f} mm'
        else:
            message = 'Choose the first node.' if self._edit_btn_edge.isChecked() else 'Click to select. Shift-drag to select visible points.'
        self.statusBar().showMessage(message)
        if len(draft.nodes) == 1:
            point = draft.points[next(iter(draft.nodes))]
            self.statusBar().setToolTip('Local position (mm): '+', '.join(f'{axis} {value:.3f}' for axis,value in zip('XYZ',point)))
        else:
            self.statusBar().setToolTip('')
        if update_cursor and len(draft.nodes) == 1:
            point = draft.points[next(iter(draft.nodes))]
            voxel = np.rint(point/np.asarray(self.workspace.resolution)).astype(int)
            if self.workspace.mag_raw is not None:
                shape = self.workspace.mag_raw.shape[:3]
                self.ortho_viewer._set_cursor(*np.clip(voxel, 0, np.asarray(shape)-1))
        self.plotter.render()

    def _geometry_screen_points(self):
        points = self._display_edit_points(self._geometry_draft.points)
        renderer = self.plotter.renderer
        matrix = renderer.GetActiveCamera().GetCompositeProjectionTransformMatrix(renderer.GetTiledAspectRatio(), 0., 1.)
        matrix = np.array([[matrix.GetElement(i, j) for j in range(4)] for i in range(4)])
        clip = np.column_stack([points, np.ones(len(points))]) @ matrix.T
        ndc = np.divide(clip[:, :3], clip[:, 3:], out=np.full_like(clip[:, :3], np.nan), where=abs(clip[:, 3:]) > 1e-12)
        return np.column_stack([(ndc[:, 0]+1)*self.plotter.width()/2,
                                (1-ndc[:, 1])*self.plotter.height()/2, ndc[:, 2]])

    def _geometry_visible(self, screen):
        window = self.plotter.renderer.GetRenderWindow()
        width, height = window.GetSize()
        visible = np.all(np.isfinite(screen), axis=1) & (screen[:, 2] >= 0) & (screen[:, 2] <= 1)
        coords = np.nan_to_num(screen[:, :2], nan=-1e6)
        xs = np.rint(coords[:, 0]*width/max(1, self.plotter.width())).astype(int)
        ys = np.rint((self.plotter.height()-coords[:, 1])*height/max(1, self.plotter.height())).astype(int)
        visible &= (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
        if np.count_nonzero(visible) > 32:
            # One GPU readback for box selection, instead of one per vertex.
            indices = np.flatnonzero(visible)
            x0, x1 = int(xs[indices].min()), int(xs[indices].max())
            y0, y1 = int(ys[indices].min()), int(ys[indices].max())
            buffer = vtkFloatArray()
            window.GetZbufferData(x0, y0, x1, y1, buffer)
            depth = vtk_to_numpy(buffer).reshape(y1-y0+1, x1-x0+1)
            visible[indices] = screen[indices, 2] <= depth[ys[indices]-y0, xs[indices]-x0]+.002
            return visible
        for i in np.flatnonzero(visible):
            z = self.plotter.renderer.GetZ(int(xs[i]), int(ys[i]))
            visible[i] = screen[i, 2] <= z+.002
        return visible

    def _geometry_hit(self, pos, prefer_edges=False):
        draft = self._geometry_draft
        if draft is None or not len(draft.points):
            return None
        screen = self._geometry_screen_points()
        cursor = np.array([pos.x(), pos.y()])
        distance = np.linalg.norm(screen[:, :2]-cursor, axis=1)
        candidates = np.flatnonzero(distance <= 8)
        node = None
        node_distance = np.inf
        if len(candidates):
            candidates = candidates[self._geometry_visible(screen[candidates])]
            if len(candidates):
                node = int(min(candidates, key=lambda i: (distance[i], screen[i, 2])))
                node_distance = distance[node]
        if self._edit_mode != 'graph' or not len(draft.edges):
            return ('node', node) if node is not None else None
        if self._edit_btn_edge.isChecked() and node is not None and not prefer_edges:
            return 'node', node
        a, b = screen[draft.edges[:, 0]], screen[draft.edges[:, 1]]
        delta = b[:, :2]-a[:, :2]
        denominator = np.sum(delta*delta, axis=1)
        t = np.clip(np.divide(np.sum((cursor-a[:, :2])*delta, axis=1), denominator,
                             out=np.zeros(len(delta)), where=denominator > 1e-12), 0, 1)
        projected = a+(b-a)*t[:, None]
        distance = np.linalg.norm(projected[:, :2]-cursor, axis=1)
        candidates = np.flatnonzero(distance <= 6)
        if len(candidates):
            candidates = candidates[self._geometry_visible(projected[candidates])]
            if len(candidates):
                edge = int(min(candidates, key=lambda i: (distance[i], projected[i, 2])))
                if node is None or prefer_edges or (node_distance > .75 and distance[edge] < node_distance*.65):
                    return 'edge', edge
        return ('node', node) if node is not None else None

    def _geometry_unproject(self, pos, depth):
        renderer = self.plotter.renderer
        width, height = renderer.GetRenderWindow().GetSize()
        renderer.SetDisplayPoint(pos.x()*width/max(1, self.plotter.width()),
                                 (self.plotter.height()-pos.y())*height/max(1, self.plotter.height()), depth)
        renderer.DisplayToWorld()
        world = np.asarray(renderer.GetWorldPoint())
        return self._display_to_edit_point(world[:3]/world[3])

    def _geometry_hover(self, hit):
        if hit == self._geometry_hover_key:
            return
        self._geometry_hover_key = hit
        self._remove_geometry_actor('hover')
        if hit is not None:
            kind, index = hit
            draft = self._geometry_draft
            if kind == 'node':
                self._geometry_mesh('hover', draft.points[[index]], size=7)
            else:
                self._geometry_mesh('hover', draft.points, draft.edges[[index]], width=4)
        self.plotter.render()

    def _geometry_box_select(self, rectangle):
        screen = self._geometry_screen_points()
        inside = ((screen[:, 0] >= rectangle.left()) & (screen[:, 0] <= rectangle.right()) &
                  (screen[:, 1] >= rectangle.top()) & (screen[:, 1] <= rectangle.bottom()))
        candidates = np.flatnonzero(inside)
        if len(candidates):
            candidates = candidates[self._geometry_visible(screen[candidates])]
            self._geometry_draft.nodes.update(map(int, candidates))
        self._render_geometry_selection()

    def _connect_geometry_node(self, index):
        if self._geometry_source is None:
            self._geometry_source = int(index)
            self._geometry_draft.select('node', index)
            self._render_geometry_selection()
            return
        start, self._geometry_source = self._geometry_source, None
        changed = self._geometry_draft.connect(start, index)
        self._remove_geometry_actor('preview')
        self._render_geometry_draft()
        if not changed:
            self.statusBar().showMessage('Choose two different, unconnected nodes.', 4000)

    def _geometry_connection_preview(self, pos, hit):
        draft, start = self._geometry_draft, self._geometry_source
        if start is None:
            return
        if hit is not None and hit[0] == 'node':
            end = draft.points[hit[1]]
        else:
            end = self._geometry_unproject(pos, self._geometry_screen_points()[start, 2])
        self._geometry_mesh('preview', [draft.points[start], end], [[0, 1]], width=2)
        self.plotter.render()

    def _delete_geometry_selection(self):
        if self._geometry_draft is None:
            return
        self._geometry_events.cancel()
        self._geometry_source = None
        self._remove_geometry_actor('preview')
        self._geometry_draft.cancel_gesture()
        self._geometry_draft.delete_selection()
        self._render_geometry_draft()

    def _geometry_context_menu(self, pos):
        hit, draft = self._geometry_hit(pos), self._geometry_draft
        if hit is not None:
            selected = draft.nodes if hit[0] == 'node' else draft.edge_ids
            if hit[1] not in selected:
                draft.select(*hit)
                self._render_geometry_selection()
        menu = QtWidgets.QMenu(self)
        if hit is not None and hit[0] == 'edge':
            menu.addAction('Select branch', lambda: self._select_geometry_branch(hit[1]))
        if draft.nodes or draft.edge_ids:
            title = 'Disconnect' if not draft.nodes and len(draft.edge_ids) == 1 else 'Delete selection'
            menu.addAction(title, self._delete_geometry_selection)
        if len(draft.nodes) == 1:
            menu.addAction('Move to slice cursor', self._move_geometry_to_cursor)
        if draft.undo_stack:
            menu.addAction('Undo '+draft.undo_stack[-1][0], lambda: self._geometry_history(False))
        if menu.actions():
            menu.exec(self.plotter.mapToGlobal(pos))

    def _select_geometry_branch(self, edge):
        self._geometry_draft.select_branch(edge)
        self._render_geometry_selection()

    def _move_geometry_to_cursor(self):
        draft = self._geometry_draft
        if len(draft.nodes) != 1:
            return
        point = np.asarray(self.workspace.ortho_cursor)*np.asarray(self.workspace.resolution)
        delta = point-draft.points[next(iter(draft.nodes))]
        draft.begin_gesture()
        draft.move_selection(delta)
        draft.finish_gesture()
        self._render_geometry_draft()

    def _enter_interactive_edit(self, mode, points, edges=None, *, group_name=''):
        if self._edit_mode is not None or self._plane_edit_session is not None:
            self.statusBar().showMessage('Finish the current edit first.', 4000)
            return
        if group_name not in self.workspace.multilabel_groups:
            self.statusBar().showMessage('Select a vessel group before editing.', 4000)
            return
        self._snapshot_edit_visibility()
        self._edit_mode, self._edit_group_name = mode, str(group_name)
        self._geometry_draft = GeometryDraft(points, edges)
        self._geometry_source = None
        self._geometry_hover_key = None
        self._edit_btn_edge.setChecked(False)
        self._edit_btn_edge.setVisible(mode == 'graph')
        title = mode.title()+' · '+group_name.replace('_', ' ')
        self._geometry_title = title
        self._edit_info_label.setText(title)
        self._edit_info_label.setToolTip(title)
        self._edit_panel.show()
        self.ortho_viewer.set_selected_plane(None)
        self.steps_group.hide()
        self._geometry_splitter_sizes = self.mid_splitter.sizes()
        self._geometry_bottom_visible = self.mid_splitter.widget(1).isVisible()
        self.mid_splitter.widget(1).hide()
        self.workflow_tabs.setEnabled(False)
        self.btn_delete_obj.setEnabled(False)
        controls = [getattr(self, name, None) for name in ('btn_add_plane', 'btn_edit_plane', 'check_plane_unlocked')]
        controls += getattr(self, '_input_case_actions', [])
        self._geometry_locked_controls = [(control, control.isEnabled()) for control in controls if control is not None]
        for control, _ in self._geometry_locked_controls:
            control.setEnabled(False)
        # Keep segmentation and visibility controls available; draft geometry
        # replaces only the edited object in the scene.
        for obj in self.workspace.scene_objects.values():
            if (obj.kind.value.lower() in {'skeleton', 'graph', 'branch', 'plane'}
                    or obj.data_key.startswith(('forks_', 'hierarchy_path_', 'path_arrow_'))):
                obj.visible = False
        self.scene.highlight_plane(None)
        self.scene.highlight_path(None)
        self.scene.invalidate_cache()
        self.scene.sync_from_workspace()
        self._refresh_browser()
        self._render_geometry_draft()
        self.plotter.setFocus(QtCore.Qt.OtherFocusReason)

    def _close_geometry_edit(self):
        self._geometry_events.cancel()
        for key in list(self._geometry_actors):
            self._remove_geometry_actor(key)
        self._edit_panel.hide()
        self.steps_group.show()
        self.mid_splitter.widget(1).setVisible(self._geometry_bottom_visible)
        self.mid_splitter.setSizes(self._geometry_splitter_sizes)
        self.workflow_tabs.setEnabled(True)
        self.btn_delete_obj.setEnabled(True)
        for control, enabled in self._geometry_locked_controls:
            control.setEnabled(enabled)
        self._edit_mode = None
        self._edit_group_name = ''
        self._edit_points = self._edit_edges = None
        self._geometry_draft = None
        self._geometry_source = None
        self._selected_plane_index = -1
        self._update_geometry_edit_actions()
        self._restore_edit_visibility()
        self.scene.invalidate_cache()
        self.scene.sync_from_workspace()
        self._refresh_all()
        self.statusBar().clearMessage()
        self.statusBar().setToolTip('')

    def _exit_interactive_edit(self, apply_changes):
        draft = self._geometry_draft
        if draft is None or self._geometry_busy:
            return
        if not apply_changes or not draft.changed:
            self._close_geometry_edit()
            return
        self._geometry_events.cancel()
        mode, group = self._edit_mode, self._edit_group_name
        before = geometry_snapshot(self.workspace)
        visibility = self._edit_visibility_snapshot or {}
        for uid, obj in before['values']['scene_objects'].items():
            if uid in visibility:
                obj.visible = visibility[uid][2]
        self._geometry_busy = True
        try:
            values = self._run_modal_task('Update centerlines', 'Applying geometry changes…',
                lambda _progress, _log: prepare_geometry_edit(self.workspace, self.pipeline, mode, group, draft.points, draft.edges))
            publish_geometry_edit(self.workspace, values)
            self._close_geometry_edit()
            after = geometry_snapshot(self.workspace)
            self._plane_undo_stack.push(_GeometryCommand(self, before, after, 'Edit '+mode))
            if self._persist_geometry_commit():
                self.statusBar().showMessage('Centerlines updated. Affected measurements need recalculation.', 7000)
        except TaskCancelled:
            self.statusBar().showMessage('Update cancelled. Your draft is still available.', 5000)
        except Exception as error:
            self.log('Geometry update failed: '+str(error))
            self.statusBar().showMessage('Could not update centerlines. Your draft is still available.', 7000)
        finally:
            self._geometry_busy = False

    def _persist_geometry_commit(self):
        try:
            persist_geometry_edit(self.workspace, self.pipeline)
            return True
        except Exception as error:
            self.log('Geometry applied, but output files could not be updated: '+str(error))
            self.statusBar().showMessage('Changes are applied; output files could not be saved.', 7000)
            return False

    def _restore_geometry_commit(self, snapshot):
        restore_geometry_snapshot(self.workspace, snapshot)
        self._selected_plane_index = -1
        self.scene.invalidate_cache()
        self.scene.sync_from_workspace()
        self._refresh_all()
        self._persist_geometry_commit()

    def _force_exit_edit(self):
        if self._edit_mode in {'skeleton', 'graph'}:
            self._cancel_geometry_action()
        elif self.ortho_viewer._roi_transaction is not None:
            self.ortho_viewer._cancel_plane_roi()
        elif self._plane_edit_session is not None:
            self._cancel_plane_edit()

