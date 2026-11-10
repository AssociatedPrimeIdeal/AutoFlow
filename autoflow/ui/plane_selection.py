"""Shared Browser/3D plane selection, context actions and pair measurements."""

import copy
import json

import numpy as np
from PySide6 import QtCore, QtWidgets

from ..core.models import ObjectKind
from ..core.plane_state import selected_metric, workspace_signature
from ..task_control import TaskCancelled
from .analysis_plotting import analysis_palette, style_analysis_figure


class PlaneSelectionMixin:
    def _sync_plane_selection_order(self):
        items = self.tree_objects.selectedItems()
        uids = [item.data(0, QtCore.Qt.UserRole) for item in items
                if item.data(0, QtCore.Qt.UserRole) is not None]
        old = getattr(self, '_plane_selection_order', [])
        order = [uid for uid in old if uid in uids]
        added = [uid for uid in uids if uid not in order]
        current = self.tree_objects.currentItem()
        current_uid = current.data(0, QtCore.Qt.UserRole) if current is not None else None
        if current_uid in added:
            added.remove(current_uid)
            added.append(current_uid)
        order.extend(added)
        self._plane_selection_order = order
        if order != old:
            self._plane_pressure_result = None
            if self._analysis_mode() == 'plane_pressure_difference':
                self.combo_analysis_mode.setCurrentIndex(self.combo_analysis_mode.findData('plane_curve'))
        if current in items and current_uid is None:
            return current
        if order:
            return self._find_browser_item_by_uid(order[-1])
        return current if current in items else (items[-1] if items else None)

    def _pressure_pair_indices(self):
        items = self.tree_objects.selectedItems()
        order = getattr(self, '_plane_selection_order', [])
        if len(items) != 2 or len(order) != 2:
            return []
        indices = []
        from .viewer import _parse_indexed_data_key
        for uid in order:
            obj = self.workspace.scene_objects.get(uid)
            if obj is None or obj.kind != ObjectKind.PLANE:
                return []
            index = _parse_indexed_data_key(obj.data_key, 'plane')
            if index is None or not 0 <= index < len(self.workspace.planes):
                return []
            indices.append(index)
        return indices

    def _select_view_plane(self, uid, additive=False):
        if self._edit_mode is not None or self._plane_edit_enabled:
            return
        tree = self.tree_objects
        item = self._find_browser_item_by_uid(uid) if uid else None
        if uid and item is None and uid in self.workspace.scene_objects:
            self._refresh_browser()
            item = self._find_browser_item_by_uid(uid)
        previous = tree.blockSignals(True)
        try:
            if not additive:
                tree.clearSelection()
            if item is not None:
                item.setSelected(not item.isSelected() if additive else True)
                if item.isSelected():
                    tree.setCurrentItem(item, 0, QtCore.QItemSelectionModel.NoUpdate)
                    parent = item.parent()
                    while parent is not None:
                        parent.setExpanded(True)
                        parent = parent.parent()
        finally:
            tree.blockSignals(previous)
        self._on_browser_select()

    def _show_plane_context_menu(self, uid, position):
        obj = self.workspace.scene_objects.get(uid)
        if obj is None or obj.kind != ObjectKind.PLANE:
            return
        previous = getattr(self, '_plane_context_menu', None)
        if previous is not None:
            previous.close()
            previous.deleteLater()
        menu = QtWidgets.QMenu(self)
        self._plane_context_menu = menu
        locked = self._plane_edit_enabled or self._edit_mode is not None or self._pipeline_thread is not None
        edit = menu.addAction('Edit Plane', lambda: self._edit_context_plane(uid))
        edit.setEnabled(not locked)
        pressure = menu.addAction('Calculate Pressure Difference', self._calculate_plane_pressure_difference)
        pressure.setEnabled(len(self._pressure_pair_indices()) == 2 and not locked)
        pressure.setToolTip('Select exactly two planes. Direction: first selected minus last selected.')
        menu.addSeparator()
        from .viewer import _parse_indexed_data_key
        index = _parse_indexed_data_key(obj.data_key, 'plane')
        menu.addAction('Generate Pathlines for This Plane',
                       lambda: self._trigger_pathlines(plane_indices=[index], selected_plane_idx=index))
        selected = self._selected_plane_indices()
        if len(selected) > 1:
            menu.addAction(f'Generate Pathlines for Selected Planes ({len(selected)})',
                           lambda: self._trigger_pathlines(plane_indices=selected, selected_plane_idx=index))
        menu.addSeparator()
        item = self._find_browser_item_by_uid(uid)
        menu.addAction('Toggle Visibility', lambda: self._toggle_context_plane_visibility(uid))
        menu.addAction('Set Opacity…', lambda: self._set_context_plane_opacity(uid))
        menu.addAction('Reset Opacity (100%)', lambda: self._set_browser_opacity_for_items([item], 1.0))
        menu.addAction('Delete', self._on_delete_object)
        menu.popup(position)
        return menu

    def _edit_context_plane(self, uid):
        self._select_view_plane(uid)
        self._toggle_selected_plane_edit()

    def _toggle_context_plane_visibility(self, uid):
        item = self._find_browser_item_by_uid(uid)
        obj = self.workspace.scene_objects.get(uid)
        if item is not None and obj is not None:
            self._set_browser_item_visibility(item, not obj.visible)
            self._sync_browser_parent_states(item)
            self._refresh_scene()

    def _set_context_plane_opacity(self, uid):
        obj = self.workspace.scene_objects.get(uid)
        if obj is None:
            return
        value, accepted = QtWidgets.QInputDialog.getDouble(self, 'Set Opacity',
            f'Opacity for {obj.name} (0–100%):', float(obj.opacity) * 100, 0, 100, 1)
        if accepted:
            self._set_browser_opacity_for_items([self._find_browser_item_by_uid(uid)], value / 100)

    def _pressure_comparison_signature(self):
        ws = self.workspace
        indices = self._pressure_pair_indices()
        return (workspace_signature(ws), tuple(indices), id(ws.derived.relative_pressure_array),
                id(ws.derived.pressure_gradient_support_mask),
                tuple(json.dumps(selected_metric(ws, i), sort_keys=True, default=str) for i in indices),
                tuple(ws.input_state.metadata.get('stale_metric_families', [])))

    def _calculate_plane_pressure_difference(self):
        if self._plane_edit_enabled or self._pipeline_running_guard('calculating pressure difference'):
            return
        ws = self.workspace
        indices = self._pressure_pair_indices()
        if len(indices) != 2:
            return
        metrics = [selected_metric(ws, i) for i in indices]
        if (ws.derived.relative_pressure_array is None or ws.derived.pressure_gradient_support_mask is None
                or 'pressure' in ws.input_state.metadata.get('stale_metric_families', [])
                or any(not m or m.get('relative_pressure_mean_Pa_t') is None for m in metrics)):
            QtWidgets.QMessageBox.information(self, 'Pressure Difference',
                'Compute current relative pressure and plane metrics first (Hemodynamics → Advanced Metrics).')
            return
        from ..algorithms.geometry_grid import measurement_segmentation
        from ..algorithms.metrics.plane_pressure import compute_plane_pressure_difference
        geometry = measurement_segmentation(ws, manual_planes=[ws.planes[i] for i in indices])
        if geometry.binary is None:
            QtWidgets.QMessageBox.information(self, 'Pressure Difference', 'Segmentation is required for pressure sampling.')
            return
        signature = self._pressure_comparison_signature()
        planes = [copy.deepcopy(ws.planes[i]) for i in indices]
        try:
            result = self._run_modal_task('Pressure Difference', 'Checking pressure support…',
                lambda _progress, _log: compute_plane_pressure_difference(planes, metrics,
                    geometry.binary, geometry.spacing, ws.origin, ws.derived.relative_pressure_array,
                    ws.derived.pressure_gradient_support_mask, branch_labels=geometry.branches,
                    labels3=geometry.labels_3d, labels4=geometry.labels, field_spacing=ws.resolution))
        except TaskCancelled:
            return
        except Exception as exc:
            self.log(f'Pressure difference failed: {exc}')
            QtWidgets.QMessageBox.information(self, 'Pressure Difference',
                'Pressure difference could not be calculated. See Log for details.')
            return
        if signature != self._pressure_comparison_signature():
            self.log('Selection or pressure inputs changed; discarded pressure difference.')
            return
        self._plane_pressure_result = dict(result, indices=indices, signature=signature)
        self._show_measurements('plane_pressure_difference')

    def _plot_plane_pressure_difference(self):
        result = getattr(self, '_plane_pressure_result', None)
        if result is None or result['signature'] != self._pressure_comparison_signature():
            message = ('Select two planes, then right-click in Browser → Calculate Pressure Difference.' if result is None
                       else 'Pressure inputs or plane geometry changed. Recalculate pressure difference.')
            self.label_pwv_status.setText(message)
            self._clear_pwv_axes(message, title='Plane Pressure Difference')
            return
        values = np.asarray(result['pressure_difference_Pa_t'], dtype=float)
        current = int(np.clip(self.workspace.current_t, 0, max(0, values.size - 1)))
        first, last = result['indices']
        title = f'Plane {first} − Plane {last}'
        if np.any(np.isfinite(values)):
            self.fig_pwv.clear()
            axes = self.fig_pwv.add_subplot(111)
            axes.plot(np.arange(values.size), values, color=analysis_palette(self)['curve'], marker='o', markersize=3)
            axes.axhline(0, color=analysis_palette(self)['spine'], linewidth=.7)
            axes.axvline(current, color=analysis_palette(self)['cursor'], linestyle='--', linewidth=.8)
            if values.size and np.isfinite(values[current]):
                axes.scatter([current], [values[current]], color=analysis_palette(self)['curve'], s=28)
            axes.set(title=title, xlabel='Cardiac Phase', ylabel='Pressure Difference (Pa)')
            style_analysis_figure(self)
            self.canvas_pwv.draw_idle()
        else:
            reasons = list(dict.fromkeys(reason for reason in result['reason_t'] if reason))
            self._clear_pwv_axes('\n'.join(reasons) or 'No valid pressure samples.', title=title)
        def number(value):
            return '—' if value is None or not np.isfinite(value) else f'{float(value):.4g}'
        valid_count = int(np.count_nonzero(result['valid_t']))
        reason = result['reason_t'][current] if values.size else ''
        notes = f'{valid_count}/{values.size} phases available'
        if reason:
            notes += '\n' + reason
        self.label_pwv_status.set_measurement(title, f'Phase {current}', number(values[current]), 'Pa',
            [('Cycle mean', number(result['mean_Pa'])), ('Peak |Δp|', number(result['peak_abs_Pa']))],
            notes=notes, plain_text=f'{title}\nPhase {current}: {number(values[current])} Pa\n{notes}')
