"""Contextual workspace panels and selected-layer appearance controls."""

from PySide6 import QtCore, QtGui, QtWidgets

from ..core.models import ObjectKind
from .parameter_help import bind_help
from .browser import BrowserInlineMixin


class WorkspaceLayoutMixin(BrowserInlineMixin):
    def _build_layer_appearance(self, layout):
        self.appearance_scroll = QtWidgets.QScrollArea()
        self.appearance_scroll.setObjectName('selectedLayerAppearance')
        self.appearance_scroll.setWidgetResizable(True)
        self.appearance_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.appearance_scroll.setMaximumHeight(420)
        self.appearance_panel = QtWidgets.QGroupBox('Appearance')
        self.appearance_layout = QtWidgets.QVBoxLayout(self.appearance_panel)
        self.appearance_layout.setContentsMargins(6, 6, 6, 6)
        self.appearance_layout.setAlignment(QtCore.Qt.AlignTop)
        self.appearance_form = QtWidgets.QFormLayout()
        self.layer_cmap = QtWidgets.QComboBox()
        self.layer_cmap.addItems(['jet', 'turbo', 'viridis', 'plasma', 'magma', 'gray', 'inferno', 'RdBu_r', 'tab10'])
        self.layer_cmap.currentTextChanged.connect(self._set_layer_cmap)
        self.appearance_form.addRow('Colormap', self.layer_cmap)
        bind_help(self.appearance_form, self.layer_cmap, 'Colormap for the selected scalar layer(s). Source data and numerical results are unchanged.')
        self.layer_color = QtWidgets.QPushButton('Choose color…')
        self.layer_color.clicked.connect(self._choose_layer_color)
        self.appearance_form.addRow('Color', self.layer_color)
        bind_help(self.appearance_form, self.layer_color, 'Uniform color for the selected layer(s) without scalar coloring. Segmentation label colors remain in the segmentation workspace.')
        self.layer_scalar_bar = QtWidgets.QCheckBox('Show colorbar')
        self.layer_scalar_bar.toggled.connect(self._set_layer_scalar_bar)
        self.layer_colorbar_row = QtWidgets.QWidget()
        bar_row = QtWidgets.QHBoxLayout(self.layer_colorbar_row)
        bar_row.setContentsMargins(0, 0, 0, 0)
        bar_row.addWidget(self.layer_scalar_bar)
        self.layer_colorbar_layout = QtWidgets.QPushButton('Layout…')
        self.layer_colorbar_layout.setObjectName('colorbarLayoutButton')
        self.layer_colorbar_layout.clicked.connect(lambda: self._open_action_settings('render'))
        bar_row.addWidget(self.layer_colorbar_layout)
        self.appearance_form.addRow(self.layer_colorbar_row)
        self.appearance_layout.addLayout(self.appearance_form)
        self.appearance_scroll.setWidget(self.appearance_panel)
        self.appearance_scroll.hide()
        layout.addWidget(self.appearance_scroll)

    def _finish_compact_workspace(self, log_widget):
        self._build_inline_inspector()
        self.log_dock = QtWidgets.QDockWidget('Log', self)
        self.log_dock.setObjectName('LogDock')
        self.log_dock.setWidget(log_widget)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.log_dock)
        self.log_dock.hide()
        self.pwv_dock.hide()
        self.qc_page = QtWidgets.QStackedWidget()
        self.qc_page.setObjectName('qcPage')
        self.qc_empty = QtWidgets.QLabel('Load a case to view its QC report.')
        self.qc_empty.setAlignment(QtCore.Qt.AlignCenter)
        self.qc_empty.setStyleSheet('font-size: 16px;')
        self.qc_page.addWidget(self.qc_empty)
        self.workspace_pages.addWidget(self.qc_page)
        self._document_dock_state = None
        self.workspace_pages.currentChanged.connect(self._sync_workspace_docks)
        view_menu = self.view_menu
        view_menu.addSeparator()
        view_menu.addAction(self.pwv_dock.toggleViewAction())
        view_menu.addAction(self.log_dock.toggleViewAction())
        self.log_toggle = QtWidgets.QToolButton()
        self.log_toggle.setDefaultAction(self.log_dock.toggleViewAction())
        self.statusBar().addPermanentWidget(self.log_toggle)
        self._measurement_plane = None
        self._sync_context_panels()

    def _select_workflow_page(self, key):
        for index in range(self.workflow_tabs.count()):
            if self._workflow_stage_key(index) == key:
                blocker = QtCore.QSignalBlocker(self.workflow_tabs)
                self.workflow_tabs.setCurrentIndex(index)
                del blocker
                self._on_workflow_stage_changed(index, open_report=False)
                return

    def _reopen_workflow_page(self, index):
        if index == self.workflow_tabs.currentIndex() and self._workflow_stage_key(index) == 'review':
            if self.workspace_pages.currentWidget() is self.workspace_page:
                self._on_workflow_stage_changed(index)

    def _inspect_qc_in_3d(self):
        self.workspace_pages.setCurrentWidget(self.workspace_page)
        self.review_panel.show()
        self.mid_splitter.setSizes([330, 360])
        self.scene.request_render()

    def _sync_workspace_docks(self, index):
        docks = {'segmentation': self.segmentation_dock, 'measurements': self.pwv_dock}
        if self.workspace_pages.currentWidget() is not self.workspace_page:
            if self._document_dock_state is None:
                self._document_dock_state = {key: dock.isVisible() for key, dock in docks.items()}
            for dock in docks.values():
                dock.hide()
        elif self._document_dock_state is not None:
            stage = self._workflow_stage_key()
            for key, dock in docks.items():
                allowed = stage == 'segmentation' if key == 'segmentation' else stage in {'hemodynamics', 'review'}
                dock.setVisible(self._document_dock_state.get(key, False) and allowed)
            self._document_dock_state = None

    def _build_export_page(self):
        page = self._export_page = QtWidgets.QScrollArea(self.workspace_pages)
        page.setObjectName('exportPage')
        page.setWidgetResizable(True)
        page.setFrameShape(QtWidgets.QFrame.NoFrame)
        body = QtWidgets.QWidget()
        outer = QtWidgets.QHBoxLayout(body)
        outer.setContentsMargins(24, 24, 24, 24)
        panel = QtWidgets.QWidget()
        panel.setMaximumWidth(800)
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(18)
        title = QtWidgets.QLabel('Export')
        title.setStyleSheet('font-size: 24px; font-weight: 600;')
        layout.addWidget(title)
        self.export_output_label = QtWidgets.QLabel()
        self.export_output_label.setWordWrap(True)
        self.export_output_label.setTextFormat(QtCore.Qt.PlainText)
        self.export_output_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.export_output_label.setStyleSheet('font-size: 14px;')
        layout.addWidget(self.export_output_label)
        self._export_case_buttons = []

        def group(title, buttons):
            box = QtWidgets.QGroupBox(title)
            grid = QtWidgets.QGridLayout(box)
            grid.setSpacing(10)
            for index, button in enumerate(buttons):
                button.setMinimumHeight(38)
                grid.addWidget(button, index // 2, index % 2)
                self._export_case_buttons.append(button)
            grid.setColumnStretch(0, 1)
            grid.setColumnStretch(1, 1)
            layout.addWidget(box)

        group('Reports', [self.btn_export_analysis_report, self.btn_export_quality])
        group('Plane coordinates', [self.btn_export_planes])
        videos = []
        for label, callback, name in [
                ('Export videos…', self._on_export_videos, 'exportVideos'),
                ('Current 3D view…', self._on_export_current_view_video, 'exportCurrentViewVideo'),
                ('Rotating 3D view…', lambda checked=False: self._on_export_current_view_video(rotate=True), 'exportRotatingViewVideo'),
                ('Generate QC videos…', self._on_generate_qc_videos, 'generateQCVideos')]:
            button = QtWidgets.QPushButton(label)
            button.setObjectName(name)
            button.clicked.connect(callback)
            videos.append(button)
        group('Videos', videos)
        layout.addStretch(1)
        outer.addWidget(panel, 1)
        outer.addStretch(1)
        page.setWidget(body)
        self.workspace_pages.addWidget(page)

    def _on_open_export_page(self):
        self._select_workflow_page('export')

    def _refresh_export_page(self):
        self.export_output_label.setText(
            str(self.pipeline._output_dir(self.workspace)) if self.workspace.data_loaded
            else 'Load a case to export results.')
        for button in self._export_case_buttons:
            button.setEnabled(bool(self.workspace.data_loaded))

    def _build_browser_tools(self, layout):
        row = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel('Browser')
        title.setObjectName('browserTitle')
        row.addWidget(title, 1)
        self.btn_browser_add_plane = QtWidgets.QPushButton('+ Add plane')
        menu = QtWidgets.QMenu(self.btn_browser_add_plane)
        for label, index in [('At cursor', 0), ('On selected path', 1)]:
            action = menu.addAction(label)
            action.triggered.connect(lambda checked=False, source=index: self._add_plane_from_browser(source))
        self.btn_browser_add_plane.setMenu(menu)
        row.addWidget(self.btn_browser_add_plane)
        self.browser_options_button = QtWidgets.QToolButton()
        self.browser_options_button.setText('…')
        self.browser_options_button.setToolTip('Browser options')
        self.browser_options_button.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        row.addWidget(self.browser_options_button)
        self.btn_measurements = QtWidgets.QPushButton('Measurements', self._browser_parking)
        self.btn_measurements.clicked.connect(lambda: self._show_measurements())
        layout.addLayout(row)

    def _add_plane_from_browser(self, source):
        self.combo_plane_add_source.setCurrentIndex(source)
        self._add_manual_plane()
        self._sync_context_panels()

    def _show_measurements(self, mode=None):
        if mode is not None:
            index = self.combo_analysis_mode.findData(mode)
            if index >= 0:
                self.combo_analysis_mode.setCurrentIndex(index)
        if self.workspace_pages.currentWidget() is self.workspace_page:
            self.pwv_dock.show()
            self.pwv_dock.raise_()
        elif self._document_dock_state is not None:
            self._document_dock_state['measurements'] = True
        self._refresh_analysis_panel()

    def _on_browser_select(self):
        self._apply_browser_selection()
        self._sync_context_panels()

    def _sync_context_panels(self):
        if not hasattr(self, '_inline_inspector'):
            return
        self.btn_browser_add_plane.setEnabled(bool(self.workspace.data_loaded) and not self._plane_edit_enabled)
        items = self.tree_objects.selectedItems()
        self.btn_delete_obj.setEnabled(bool(items) and not self._plane_edit_enabled and self._edit_mode is None)
        current = self.tree_objects.currentItem()
        item = current if current in items else (items[0] if items else None)
        if item is None:
            self._detach_inline_inspector()
            self._inline_selection_key = None
            self.selection_tabs.hide()
            return
        plane = int(self._selected_plane_index)
        node = item.data(0, int(QtCore.Qt.UserRole)+1)
        single = len(items) == 1
        has_plane = single and 0 <= plane < len(self.workspace.planes)
        has_path = single and (self.workspace.selected_path_index >= 0 or node in {'path', 'path_reference', 'joint_path'})
        if has_plane:
            self.selection_tabs.setCurrentIndex(0)
            information = self.text_plane_info
        elif has_path:
            self.selection_tabs.setCurrentIndex(1)
            information = self.text_path_info
        else:
            self.selection_tabs.setCurrentIndex(2)
            information = self.text_layer_info
            objects = self._browser_selected_objects()
            title = item.text(0) if single else f'{len(objects)} layers selected'
            if not single:
                subtitle = 'Shared appearance controls'
            elif item.data(0, QtCore.Qt.UserRole) is None:
                subtitle = f'{len(objects)} layers'
            elif objects and objects[0].kind == ObjectKind.SEGMENTATION:
                subtitle = 'Segmentation'
            elif objects and objects[0].kind == ObjectKind.METRIC:
                subtitle = 'Metric layer'
            else:
                subtitle = '3D layer'
            information.set_summary(title, subtitle, [], '', plain_text=title)
        information.header_layout.addWidget(self.btn_measurements)
        self.btn_measurements.setVisible(has_plane or has_path)
        self.selection_tabs.show()
        key = tuple(self._browser_view_key(selected) for selected in items)
        changed = key != getattr(self, '_inline_selection_key', None)
        if changed:
            self._inline_selection_key = key
            self.appearance_toggle.setChecked(not (has_plane or has_path))
        self._refresh_layer_appearance(self._browser_selected_objects())
        self._attach_inline_inspector(item, expand=changed or item.isExpanded())
        self.btn_browser_add_plane.setEnabled(bool(self.workspace.data_loaded) and not self._plane_edit_enabled)
        identity = (id(self.workspace.planes[plane]), plane) if has_plane else None
        if identity is not None and identity != getattr(self, '_measurement_plane', None):
            self._show_measurements('plane_curve')
        self._measurement_plane = identity
        self._queue_inline_inspector_size()

    def _refresh_layer_appearance(self, objects):
        if not hasattr(self, 'appearance_scroll'):
            return
        self.appearance_toggle.setVisible(bool(objects))
        self.appearance_scroll.setVisible(bool(objects) and self.appearance_toggle.isChecked())
        if not objects:
            return
        obj = objects[0]
        self.appearance_panel.setTitle('')
        scalar = all(bool(item.scalars) or item.data_key == 'pcmra_volume' or self.scene.is_content_render(item) for item in objects)
        categorical = any(item.kind == ObjectKind.SEGMENTATION for item in objects)
        solid = all(not item.scalars and item.data_key != 'pcmra_volume' and not self.scene.is_content_render(item) for item in objects)
        for control, visible in ((self.layer_cmap, scalar and not categorical),
                                 (self.layer_color, solid), (self.layer_colorbar_row, scalar)):
            control.setVisible(visible)
            label = self.appearance_form.labelForField(control)
            if label is not None:
                label.setVisible(visible)
        blocker = QtCore.QSignalBlocker(self.layer_cmap)
        if self.layer_cmap.findText(str(obj.cmap)) < 0:
            self.layer_cmap.addItem(str(obj.cmap))
        self.layer_cmap.setCurrentText(str(obj.cmap))
        del blocker
        blocker = QtCore.QSignalBlocker(self.layer_scalar_bar)
        self.layer_scalar_bar.setChecked(all(item.show_scalar_bar for item in objects))
        del blocker
        color = QtGui.QColor(str(obj.color or '#ffffff'))
        swatch = QtGui.QPixmap(12, 12)
        swatch.fill(color if color.isValid() else QtCore.Qt.transparent)
        self.layer_color.setIcon(QtGui.QIcon(swatch))
        self.plane_display_controls.setVisible(any(item.kind == ObjectKind.PLANE for item in objects))
        self._size_layer_appearance()

    def _size_layer_appearance(self):
        self.appearance_layout.activate()
        self.appearance_scroll.setFixedHeight(min(420, max(70, self.appearance_panel.sizeHint().height() + 4)))
        self._queue_inline_inspector_size()

    def _set_layer_cmap(self, value):
        if not self.scene:
            return
        for obj in self._browser_selected_objects():
            if self.scene.is_content_render(obj):
                self.scene.update_content_render(obj.uid, cmap=value)
            elif obj.scalars or obj.data_key == 'pcmra_volume':
                obj.cmap = value
                self.scene.readd_object(obj, refresh_scalar_bar=False)
        self.scene.render_all()
        self._refresh_content_render_controls()

    def _set_layer_scalar_bar(self, checked):
        if not self.scene:
            return
        for obj in self._browser_selected_objects():
            obj.show_scalar_bar = bool(checked)
            self.scene.apply_object_properties(obj, render=False)
        self.scene.render_all()

    def _choose_layer_color(self):
        objects = [obj for obj in self._browser_selected_objects() if not obj.scalars]
        if not objects:
            return
        color = QtWidgets.QColorDialog.getColor(QtGui.QColor(str(objects[0].color)), self, 'Layer color')
        if not color.isValid():
            return
        for obj in objects:
            if obj.data_key.startswith('pathline_') and obj.data_key.rsplit('_', 1)[-1].isdigit():
                self._set_pathline_color(int(obj.data_key.rsplit('_', 1)[-1]), color.name())
            else:
                obj.color = color.name()
                self.scene.apply_object_properties(obj, render=False)
        self.scene.render_all()
        self._refresh_layer_appearance(objects)
