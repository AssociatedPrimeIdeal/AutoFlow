"""Camera-linked orientation markers, portable VTP assets and standard views."""

import numpy as np
import pyvista as pv
from pathlib import Path
from vtkmodules.vtkRenderingCore import vtkPropAssembly


ASSET_DIRECTORY = Path(__file__).with_name('assets') / 'orientation'
BUILTIN_MARKERS = {'penguin': 'Penguin', 'rabbit': 'Rabbit'}


def read_marker_asset(path):
    """Read a finite polygonal VTP model, with optional uint8 RGB(A) colours."""
    if Path(path).suffix.lower() != '.vtp':
        raise ValueError('Choose a VTK PolyData (.vtp) model.')
    mesh = pv.read(path)
    if not isinstance(mesh, pv.PolyData) or not mesh.n_points or not (mesh.faces.size or mesh.n_strips):
        raise ValueError('The VTP model must contain a polygonal surface.')
    if not np.isfinite(mesh.points).all() or np.ptp(mesh.points, axis=0).max() <= 0:
        raise ValueError('The VTP model has invalid coordinates or zero size.')
    return mesh


def _asset_parts(mesh):
    sizes = mesh.field_data.get('part_sizes')
    normals_present = mesh.field_data.get('part_has_normals')
    if sizes is None or normals_present is None:
        return [mesh]
    if (sizes.ndim != 2 or sizes.shape[1] != 5 or not np.issubdtype(sizes.dtype, np.integer)
            or len(sizes) != len(normals_present)
            or np.any(sizes < 0) or not np.array_equal(sizes.sum(axis=0),
                [mesh.n_points, mesh.verts.size, mesh.lines.size, mesh.faces.size, mesh.strips.size])):
        raise ValueError('Invalid VTP part metadata.')
    offsets = np.zeros(5, dtype=int)
    result = []
    for size, has_normals in zip(sizes, normals_present):
        start, stop = int(offsets[0]), int(offsets[0]+size[0])
        cells = {}
        for index, name in enumerate(('verts', 'lines', 'faces', 'strips'), 1):
            values = getattr(mesh, name)[offsets[index]:offsets[index]+size[index]].copy()
            cursor = 0
            while cursor < len(values):
                count = int(values[cursor])
                if count <= 0 or cursor+1+count > len(values):
                    raise ValueError('Invalid VTP part connectivity.')
                values[cursor+1:cursor+1+count] -= start
                if np.any((values[cursor+1:cursor+1+count] < 0)
                          | (values[cursor+1:cursor+1+count] >= size[0])):
                    raise ValueError('Invalid VTP part point indices.')
                cursor += count+1
            cells[name] = values
        surface = pv.PolyData(mesh.points[start:stop].copy(), **cells)
        for name, values in mesh.point_data.items():
            if name != 'Normals' or has_normals:
                surface.point_data[name] = values[start:stop].copy()
        result.append(surface)
        offsets += size
    return result


def anatomical_marker(signs, asset_path=None):
    """Build the marker with the scene's real LR/AP/FH display-axis signs."""
    mesh = read_marker_asset(asset_path or ASSET_DIRECTORY / 'penguin.vtp')
    assembly = vtkPropAssembly()
    # Bundled models retain each original actor's normals and draw order. This
    # avoids changing the penguin's smooth ellipsoids or faceted heart.
    for surface in _asset_parts(mesh):
        if 'Normals' in surface.point_data:
            surface.point_data.active_normals_name = 'Normals'
        mapper = pv.DataSetMapper(dataset=surface)
        solid_color = None
        for association, attributes in (('point', surface.point_data), ('cell', surface.cell_data)):
            color = next((name for name in ('RGB', 'RGBA', 'Colors') if name in attributes
                          and attributes[name].dtype == np.uint8 and attributes[name].ndim == 2
                          and attributes[name].shape[1] in (3, 4)), None)
            if color:
                colors = attributes[color]
                if len(colors) and np.all(colors == colors[0]):
                    solid_color = colors[0]
                    mapper.scalar_visibility = False
                else:
                    mapper.scalar_map_mode = association
                    mapper.color_mode = 'direct'
                    mapper.array_name = color
                    mapper.scalar_visibility = True
                break
        else:
            mapper.scalar_visibility = False
        actor = pv.Actor(mapper=mapper)
        actor.GetProperty().SetColor(pv.Color('#f4f3eb').float_rgb if solid_color is None
                                     else solid_color[:3].astype(float)/255)
        if solid_color is not None and len(solid_color) == 4:
            actor.GetProperty().SetOpacity(float(solid_color[3])/255)
        actor.GetProperty().SetAmbient(.35)
        actor.GetProperty().SetDiffuse(.65)
        actor.GetProperty().SetSpecular(.04)
        actor.GetProperty().SetInterpolationToPhong()
        actor.SetScale(*map(float, signs))
        assembly.AddPart(actor)
    return assembly


class AnatomicalOrientationMarker:
    VIEWS = (
        ("Anterior", (0, -1, 0), (0, 0, 1)),
        ("Posterior", (0, 1, 0), (0, 0, 1)),
        ("Left", (-1, 0, 0), (0, 0, 1)),
        ("Right", (1, 0, 0), (0, 0, 1)),
        ("Superior", (0, 0, 1), (0, -1, 0)),
        ("Inferior", (0, 0, -1), (0, 1, 0)),
    )

    def __init__(self, scene):
        self.scene = scene
        self.plotter = scene.plotter
        self._pressed = None
        self._dragged = False
        self._next_view = 0
        self._last_direction = None
        self._right_pressed = False
        self._menu = None
        from PySide6 import QtCore
        self._settings = QtCore.QSettings('AutoFlow', 'AutoFlow')
        try:
            self.size_percent = int(np.clip(int(self._settings.value('viewer/orientation_marker_size', 100)), 50, 200))
        except (TypeError, ValueError):
            self.size_percent = 100
        self.selection = str(self._settings.value('viewer/orientation_marker', 'penguin'))
        if self.selection == 'milk_frog':
            self.selection = 'rabbit'
            self._settings.setValue('viewer/orientation_marker', self.selection)
        self.asset_path = self._asset_path(self.selection)
        try:
            self.actor = anatomical_marker(scene._display_axis_signs, self.asset_path)
        except Exception as error:
            scene.logger(f'Orientation marker unavailable; using Penguin: {error}')
            self.selection = 'penguin'
            self.asset_path = self._asset_path(self.selection)
            self.actor = anatomical_marker(scene._display_axis_signs, self.asset_path)
        self.widget = self.plotter.add_orientation_widget(
            self.actor, interactive=False, viewport=(.01, .01, .28, .32))
        self.widget.SetShouldConstrainSize(False)
        self._apply_size()
        self.widget.GetRenderer().SetLayer(3)
        if hasattr(self.plotter, "setToolTip"):
            self.plotter.setToolTip("Orientation marker: built-in heart = patient left. Click for standard views. "
                                   "Right-click to change the icon or the 3D scene's LR/AP/FH directions.")

    @staticmethod
    def _custom_path():
        from PySide6 import QtCore
        return Path(QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.AppDataLocation)) / 'orientation-markers/custom.vtp'

    def _asset_path(self, selection):
        if selection == 'custom':
            return self._custom_path()
        return ASSET_DIRECTORY / f'{selection if selection in BUILTIN_MARKERS else "penguin"}.vtp'

    def set_asset(self, selection):
        path = self._asset_path(selection)
        actor = anatomical_marker(self.scene._display_axis_signs, path)
        self.widget.SetOrientationMarker(actor)
        self.actor, self.selection, self.asset_path = actor, selection, path
        self._settings.setValue('viewer/orientation_marker', selection)
        self.scene.request_render()

    def _apply_size(self):
        factor = self.size_percent / 100
        self.widget.SetViewport(.01, .01, .01+.27*factor, .01+.31*factor)

    def set_size(self, percent):
        self.size_percent = int(np.clip(int(percent), 50, 200))
        self._apply_size()
        self._settings.setValue('viewer/orientation_marker_size', self.size_percent)
        self.scene.request_render()

    def import_asset(self, path):
        """Keep a portable local copy so moving the source does not break reloads."""
        import os
        import tempfile
        mesh = read_marker_asset(path)
        # Bundled assets already use the original model scale and origin.
        # External models are centered and fitted without changing proportions.
        if 'autoflow_orientation_asset' not in mesh.field_data:
            span = np.ptp(mesh.points, axis=0).max()
            mesh.points = (mesh.points-np.asarray(mesh.center))*(2.2/span)
        target = self._custom_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        # Validate reconstruction before replacing a previously imported icon.
        descriptor, temporary = tempfile.mkstemp(suffix='.vtp', dir=target.parent)
        os.close(descriptor)
        try:
            mesh.save(temporary)
            anatomical_marker(self.scene._display_axis_signs, temporary)
            os.replace(temporary, target)
        finally:
            Path(temporary).unlink(missing_ok=True)
        self.set_asset('custom')

    def _choose_asset(self):
        from PySide6 import QtWidgets
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self.plotter, 'Import orientation icon', '', 'VTK PolyData (*.vtp)')
        if path:
            try:
                self.import_asset(path)
            except Exception as error:
                QtWidgets.QMessageBox.warning(self.plotter, 'Cannot load icon', str(error))

    def _export_asset(self):
        from PySide6 import QtWidgets
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self.plotter, 'Export orientation icon',
            f'{self.selection}.vtp', 'VTK PolyData (*.vtp)')
        if path:
            try:
                read_marker_asset(self.asset_path).save(Path(path).with_suffix('.vtp'))
            except Exception as error:
                QtWidgets.QMessageBox.warning(self.plotter, 'Cannot export icon', str(error))

    def set_axis_direction(self, axis, direction):
        directions = list(self.scene.display_axis_directions())
        directions[axis] = direction
        if self.scene.set_display_axis_directions(directions):
            self.scene.logger(f'3D display orientation: {", ".join(directions)}')

    def context_menu(self):
        from PySide6 import QtCore, QtGui, QtWidgets
        menu = QtWidgets.QMenu(self.plotter)
        icons = QtWidgets.QMenu('Icon', menu)
        menu.addMenu(icons)
        menu._submenus = [icons]
        group = QtGui.QActionGroup(icons)
        menu._action_groups = [group]
        for key, name in BUILTIN_MARKERS.items():
            action = icons.addAction(name)
            action.setCheckable(True)
            action.setChecked(self.selection == key)
            group.addAction(action)
            action.triggered.connect(lambda checked=False, key=key: self.set_asset(key))
        if self._custom_path().is_file():
            action = icons.addAction('Custom')
            action.setCheckable(True)
            action.setChecked(self.selection == 'custom')
            group.addAction(action)
            action.triggered.connect(lambda: self.set_asset('custom'))
        icons.addSeparator()
        icons.addAction('Import VTP...', self._choose_asset)
        icons.addAction('Export VTP...', self._export_asset)
        size_menu = QtWidgets.QMenu('Size', menu)
        menu.addMenu(size_menu)
        menu._submenus.append(size_menu)
        controls = QtWidgets.QWidget(size_menu)
        layout = QtWidgets.QHBoxLayout(controls)
        slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, controls)
        slider.setRange(50, 200)
        slider.setMinimumWidth(140)
        slider.setValue(self.size_percent)
        percent = QtWidgets.QSpinBox(controls)
        percent.setRange(50, 200)
        percent.setSuffix('%')
        percent.setValue(self.size_percent)
        slider.valueChanged.connect(percent.setValue)
        percent.valueChanged.connect(slider.setValue)
        slider.valueChanged.connect(self.set_size)
        layout.addWidget(slider)
        layout.addWidget(percent)
        size_action = QtWidgets.QWidgetAction(size_menu)
        size_action.setDefaultWidget(controls)
        size_menu.addAction(size_action)
        size_menu.addAction('Reset size', lambda: percent.setValue(100))
        menu.addSeparator()
        current = self.scene.display_axis_directions()
        for axis, (name, choices) in enumerate((('Left / right', ('LR', 'RL')),
                                               ('Front / back', ('AP', 'PA')),
                                               ('Feet / head', ('FH', 'HF')))):
            submenu = QtWidgets.QMenu(name, menu)
            menu.addMenu(submenu)
            menu._submenus.append(submenu)
            group = QtGui.QActionGroup(submenu)
            menu._action_groups.append(group)
            for direction in choices:
                action = submenu.addAction(direction)
                action.setCheckable(True)
                action.setChecked(current[axis] == direction)
                group.addAction(action)
                action.triggered.connect(lambda checked=False, axis=axis, direction=direction:
                                         self.set_axis_direction(axis, direction))
        menu.addAction('Reset directions', lambda: self.scene.set_display_axis_directions(('LR', 'AP', 'FH')))
        return menu

    def _show_context_menu(self):
        from PySide6 import QtGui
        # Nonblocking popup: release events must finish before any scene rebuild.
        if self._menu is not None:
            self._menu.close()
            self._menu.deleteLater()
        self._menu = self.context_menu()
        self._menu.popup(QtGui.QCursor.pos())

    def _camera_direction(self):
        camera = self.plotter.camera
        direction = np.asarray(camera.GetPosition())-camera.GetFocalPoint()
        direction /= max(np.linalg.norm(direction), 1e-12)
        up = np.asarray(camera.GetViewUp())
        up /= max(np.linalg.norm(up), 1e-12)
        return np.r_[direction, up]

    def snap_next(self):
        if self._last_direction is None or not np.allclose(
                self._camera_direction(), self._last_direction, atol=1e-5):
            self._next_view = 0
        name, direction, up = self.VIEWS[self._next_view]
        camera = self.plotter.camera
        center = np.asarray(camera.GetFocalPoint())
        distance = max(np.linalg.norm(np.asarray(camera.GetPosition())-center), 1.)
        signs = self.scene._display_axis_signs
        camera.SetPosition(*(center+np.asarray(direction)*signs*distance))
        camera.SetViewUp(*(np.asarray(up)*signs))
        camera.OrthogonalizeViewUp()
        self.plotter.renderer.ResetCameraClippingRange()
        self._last_direction = self._camera_direction()
        self._next_view = (self._next_view+1) % len(self.VIEWS)
        self.scene.request_render()
        return name

    def dispatch(self, event):
        iren = getattr(self.plotter.iren, "interactor", self.plotter.iren)
        point = np.asarray(iren.GetEventPosition(), dtype=float)
        if event in ("press", "right_press"):
            if not self.scene._axes_shown or not self.widget.GetEnabled():
                return False
            size = np.asarray(self.plotter.render_window.GetSize())
            viewport = np.asarray(self.widget.GetViewport()).reshape(2, 2)
            if not np.all((point >= viewport[0]*size) & (point <= viewport[1]*size)):
                return False
            if event == 'right_press':
                self._right_pressed = True
            else:
                self._pressed, self._dragged = point, False
            return True
        if self._right_pressed:
            if event == 'right_release':
                self._right_pressed = False
                self._show_context_menu()
            return True
        if self._pressed is None:
            return False
        self._dragged |= np.linalg.norm(point-self._pressed) > 6
        if event == "release":
            self._pressed = None
            if not self._dragged:
                self.snap_next()
        return True

    def close(self):
        self._pressed = None
        self._right_pressed = False
        if self._menu is not None:
            self._menu.close()
            self._menu.deleteLater()
            self._menu = None
        self.widget.SetEnabled(False)
        self.widget.SetInteractor(None)
