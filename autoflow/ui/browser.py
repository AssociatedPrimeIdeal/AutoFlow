"""Browser row presentation and one reusable inline inspector."""

from PySide6 import QtCore, QtGui, QtWidgets


INLINE_ROLE = int(QtCore.Qt.UserRole) + 20
VIEW_KEY_ROLE = int(QtCore.Qt.UserRole) + 21


class ContextStack(QtWidgets.QStackedWidget):
    """Size the inspector from the active page, rather than its longest page."""

    def sizeHint(self):
        page = self.currentWidget()
        return page.sizeHint() if page is not None else QtCore.QSize(0, 0)

    def minimumSizeHint(self):
        return QtCore.QSize(0, 0)


class BrowserDelegate(QtWidgets.QStyledItemDelegate):
    def __init__(self, parent):
        super().__init__(parent)
        self._icons = {}

    def _icon(self, kind, color):
        key = (kind, color.rgba())
        if key in self._icons:
            return self._icons[key]
        pixmap = QtGui.QPixmap(32, 32)
        pixmap.fill(QtCore.Qt.transparent)
        painter = QtGui.QPainter(pixmap)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.scale(2, 2)
        painter.setPen(QtGui.QPen(color, 1.2, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
        if kind == 'plane':
            painter.drawRoundedRect(QtCore.QRectF(3, 3, 10, 10), 1, 1)
        elif kind == 'path':
            path = QtGui.QPainterPath(QtCore.QPointF(3, 12))
            path.cubicTo(15, 13, 1, 4, 13, 4)
            painter.drawPath(path)
            painter.drawEllipse(QtCore.QPointF(3, 12), 1.5, 1.5)
            painter.drawEllipse(QtCore.QPointF(13, 4), 1.5, 1.5)
        elif kind == 'flow':
            for y, end in ((4, 12), (8, 14), (12, 10)):
                painter.drawLine(QtCore.QPointF(2, y), QtCore.QPointF(end, y))
        elif kind == 'volume':
            painter.drawRect(QtCore.QRectF(3, 6, 8, 8))
            painter.drawPolyline(QtGui.QPolygonF([QtCore.QPointF(3, 6), QtCore.QPointF(6, 2),
                                  QtCore.QPointF(14, 2), QtCore.QPointF(14, 10), QtCore.QPointF(11, 14)]))
            painter.drawLine(QtCore.QPointF(11, 6), QtCore.QPointF(14, 2))
        else:
            for y in (4, 8, 12):
                painter.drawPolyline(QtGui.QPolygonF([QtCore.QPointF(2, y), QtCore.QPointF(8, y+2), QtCore.QPointF(14, y)]))
        painter.end()
        pixmap.setDevicePixelRatio(2)
        result = QtGui.QIcon(pixmap)
        self._icons[key] = result
        return result

    def paint(self, painter, option, index):
        if index.data(INLINE_ROLE):
            return
        if index.column() == 1:
            background = QtWidgets.QStyleOptionViewItem(option)
            self.initStyleOption(background, index)
            background.text = ''
            background.features &= ~(QtWidgets.QStyleOptionViewItem.HasCheckIndicator | QtWidgets.QStyleOptionViewItem.HasDisplay)
            widget = background.widget
            style = widget.style() if widget is not None else QtWidgets.QApplication.style()
            style.drawControl(QtWidgets.QStyle.CE_ItemViewItem, background, painter, widget)
            state = index.siblingAtColumn(0).data(QtCore.Qt.CheckStateRole)
            if state is None:
                return
            state = state.value if hasattr(state, 'value') else int(state)
            checked = state != QtCore.Qt.Unchecked.value
            color = option.palette.color(QtGui.QPalette.Text) if checked else option.palette.color(QtGui.QPalette.Disabled, QtGui.QPalette.Text)
            painter.save()
            painter.setRenderHint(QtGui.QPainter.Antialiasing)
            painter.translate(option.rect.center().x()-8, option.rect.center().y()-8)
            painter.setPen(QtGui.QPen(color, 1.15, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap))
            if state == QtCore.Qt.PartiallyChecked.value:
                painter.setOpacity(.65)
            eye = QtGui.QPainterPath(QtCore.QPointF(1, 8))
            eye.cubicTo(5, 2, 11, 2, 15, 8)
            eye.cubicTo(11, 14, 5, 14, 1, 8)
            painter.drawPath(eye)
            painter.drawEllipse(QtCore.QPointF(8, 8), 2, 2)
            if not checked:
                painter.drawLine(QtCore.QPointF(2, 2), QtCore.QPointF(14, 14))
            painter.restore()
            return
        styled = QtWidgets.QStyleOptionViewItem(option)
        self.initStyleOption(styled, index)
        styled.features &= ~QtWidgets.QStyleOptionViewItem.HasCheckIndicator
        text = str(index.siblingAtColumn(1).data() or '').lower()
        if text in {'path', 'branch', 'skeleton', 'graph', 'analysis branch', 'connection'}:
            kind = 'path'
        elif text == 'plane':
            kind = 'plane'
        elif text == 'flow':
            kind = 'flow'
        elif text in {'aux', 'metric'}:
            kind = 'volume'
        else:
            kind = 'layers'
        styled.icon = self._icon(kind, option.palette.color(QtGui.QPalette.Text))
        styled.decorationSize = QtCore.QSize(16, 16)
        styled.features |= QtWidgets.QStyleOptionViewItem.HasDecoration
        widget = styled.widget
        style = widget.style() if widget is not None else QtWidgets.QApplication.style()
        style.drawControl(QtWidgets.QStyle.CE_ItemViewItem, styled, painter, widget)

    def sizeHint(self, option, index):
        explicit = index.data(QtCore.Qt.SizeHintRole)
        if index.data(INLINE_ROLE) and explicit is not None:
            return explicit
        hint = super().sizeHint(option, index)
        return QtCore.QSize(hint.width(), max(30, hint.height()))

    def editorEvent(self, event, model, option, index):
        if event.type() in {QtCore.QEvent.MouseButtonPress, QtCore.QEvent.MouseButtonRelease,
                            QtCore.QEvent.MouseButtonDblClick}:
            return False
        return super().editorEvent(event, model, option, index)


class BrowserTree(QtWidgets.QTreeWidget):
    inspectorResized = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.inline_row = None
        self.inline_host = None
        self.setObjectName('browserObjects')
        self.setColumnCount(2)
        self.setHeaderLabels(['Name', 'Visibility'])
        self.setHeaderHidden(True)
        self.header().setStretchLastSection(False)
        self.header().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.header().setSectionResizeMode(1, QtWidgets.QHeaderView.Fixed)
        self.setColumnWidth(1, 32)
        self.setIndentation(14)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setUniformRowHeights(False)
        self.setItemDelegate(BrowserDelegate(self))
        font = self.font(); font.setPixelSize(14); self.setFont(font)

    def clear(self):
        callback = getattr(self, 'before_clear', None)
        if callback is not None:
            callback()
        super().clear()

    def mousePressEvent(self, event):
        index = self.indexAt(event.position().toPoint())
        if event.button() == QtCore.Qt.LeftButton and index.isValid() and index.column() == 1:
            item = self.itemFromIndex(index)
            if item.flags() & QtCore.Qt.ItemIsUserCheckable:
                item.setCheckState(0, QtCore.Qt.Unchecked if item.checkState(0) == QtCore.Qt.Checked else QtCore.Qt.Checked)
                self._visibility_press_active = True
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton and getattr(self, '_visibility_press_active', False):
            self._visibility_press_active = False
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        index = self.indexAt(event.position().toPoint())
        if event.button() == QtCore.Qt.LeftButton and index.isValid() and index.column() == 1:
            self.mousePressEvent(event)
            return
        super().mouseDoubleClickEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.inspectorResized.emit()

    def updateEditorGeometries(self):
        super().updateEditorGeometries()
        self._position_inspector()

    def scrollContentsBy(self, dx, dy):
        super().scrollContentsBy(dx, dy)
        self._position_inspector()

    def _position_inspector(self):
        if self.inline_row is None or self.inline_host is None:
            return
        rectangle = self.visualItemRect(self.inline_row)
        self.inline_host.setVisible(not rectangle.isEmpty())
        if not rectangle.isEmpty():
            self.inline_host.setGeometry(8, rectangle.y(), max(1, self.viewport().width()-16), rectangle.height())


class BrowserInlineMixin:
    def _build_inline_inspector(self):
        self._inline_size_timer = QtCore.QTimer(self)
        self._inline_size_timer.setSingleShot(True)
        self._inline_size_timer.timeout.connect(self._layout_inline_inspector)
        self._inline_selection_key = None
        self._inline_inspector = QtWidgets.QFrame(self._browser_parking)
        self._inline_inspector.setObjectName('browserInlineInspector')
        font = self._inline_inspector.font(); font.setPixelSize(14); self._inline_inspector.setFont(font)
        self._inline_inspector.hide()
        layout = QtWidgets.QVBoxLayout(self._inline_inspector)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)
        layout.addWidget(self.selection_tabs)
        self.appearance_toggle = QtWidgets.QToolButton()
        self.appearance_toggle.setObjectName('browserAppearanceToggle')
        self.appearance_toggle.setText('Appearance')
        self.appearance_toggle.setCheckable(True)
        self.appearance_toggle.setArrowType(QtCore.Qt.RightArrow)
        self.appearance_toggle.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.appearance_toggle.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.appearance_toggle.toggled.connect(self._toggle_inline_appearance)
        layout.addWidget(self.appearance_toggle)
        layout.addWidget(self.appearance_scroll)
        self.appearance_panel.setTitle('')
        self.appearance_panel.setFlat(True)
        self.tree_objects.before_clear = self._detach_inline_inspector
        self.tree_objects.inspectorResized.connect(self._queue_inline_inspector_size)
        self.tree_objects.itemExpanded.connect(self._queue_inline_inspector_size)
        self.tree_objects.itemCollapsed.connect(self._queue_inline_inspector_size)
        self.selection_tabs.currentChanged.connect(self._queue_inline_inspector_size)

    def _toggle_inline_appearance(self, checked):
        self.appearance_toggle.setArrowType(QtCore.Qt.DownArrow if checked else QtCore.Qt.RightArrow)
        self.appearance_scroll.setVisible(bool(checked) and bool(self._browser_selected_objects()))
        self._queue_inline_inspector_size()

    def _queue_inline_inspector_size(self, *_):
        timer = getattr(self, '_inline_size_timer', None)
        if timer is not None and not timer.isActive():
            timer.start(0)

    def _detach_inline_inspector(self):
        tree = self.tree_objects
        row, host = tree.inline_row, tree.inline_host
        tree.inline_row = tree.inline_host = None
        if row is None:
            return
        blocker = QtCore.QSignalBlocker(tree)
        # Only the throw-away host belongs to QTreeWidget. Reparent the shared
        # controls before removing an index widget, whose deletion is deferred.
        self._inline_inspector.hide()
        if host is not None:
            host.layout().removeWidget(self._inline_inspector)
        self._inline_inspector.setParent(self._browser_parking)
        tree.removeItemWidget(row, 0)
        parent = row.parent()
        if parent is not None:
            parent.takeChild(parent.indexOfChild(row))
        del blocker

    def _attach_inline_inspector(self, parent, *, expand=True):
        tree = self.tree_objects
        if tree.inline_row is not None and tree.inline_row.parent() is parent:
            self._queue_inline_inspector_size()
            return
        self._detach_inline_inspector()
        blocker = QtCore.QSignalBlocker(tree)
        row = QtWidgets.QTreeWidgetItem()
        row.setData(0, INLINE_ROLE, True)
        row.setFlags(QtCore.Qt.NoItemFlags)
        parent.insertChild(0, row)
        row.setFirstColumnSpanned(True)
        row.setSizeHint(0, QtCore.QSize(0, 200))
        host = QtWidgets.QWidget(tree)
        layout = QtWidgets.QVBoxLayout(host)
        layout.setContentsMargins(0, 3, 0, 5)
        layout.addWidget(self._inline_inspector)
        tree.inline_row, tree.inline_host = row, host
        self._inline_pending_scroll = True
        tree.setItemWidget(row, 0, host)
        if expand:
            parent.setExpanded(True)
        self._inline_inspector.show()
        del blocker
        self._queue_inline_inspector_size()

    def _layout_inline_inspector(self):
        tree = self.tree_objects
        if tree.inline_row is None:
            return
        width = max(180, tree.viewport().width()-16)
        self._inline_inspector.setFixedWidth(width)
        inner_width = max(1, width-22)
        page = self.selection_tabs.currentWidget()
        page.setFixedWidth(inner_width)
        self.selection_tabs.setFixedWidth(inner_width)
        if self.selection_tabs.currentIndex() == 0:
            controls = self.plane_controls_scroll.widget()
            for spins in (self.plane_center_controls, self.plane_normal_controls):
                row = spins[0].parentWidget().layout()
                narrow = inner_width < 290
                row.setDirection(QtWidgets.QBoxLayout.TopToBottom if narrow else QtWidgets.QBoxLayout.LeftToRight)
                for axis, spin in zip(('LR', 'AP', 'FH'), spins):
                    spin.setPrefix(axis+': ' if narrow else '')
            controls.layout().activate()
            show_controls = self._plane_edit_enabled or not self.btn_undo_plane.isHidden() or not self.btn_redo_plane.isHidden() or self._plane_jobs.is_running
            self.plane_controls_scroll.setVisible(show_controls)
            self.plane_controls_scroll.setFixedHeight(min(320, max(1, controls.sizeHint().height()+4)))
        page.layout().activate()
        height = page.layout().totalHeightForWidth(inner_width)
        if height < 0:
            height = page.sizeHint().height()
        self.selection_tabs.setFixedHeight(max(1, height))
        self._inline_inspector.layout().activate()
        height = self._inline_inspector.sizeHint().height()+8
        if tree.inline_row.sizeHint(0).height() != height:
            blocker = QtCore.QSignalBlocker(tree)
            tree.inline_row.setSizeHint(0, QtCore.QSize(0, height))
            del blocker
            tree.doItemsLayout()
        tree._position_inspector()
        if getattr(self, '_inline_pending_scroll', False):
            self._inline_pending_scroll = False
            rectangle = tree.visualItemRect(tree.inline_row)
            if not rectangle.isEmpty() and rectangle.y()+min(180, height) > tree.viewport().height():
                tree.scrollToItem(tree.inline_row.parent(), QtWidgets.QAbstractItemView.PositionAtTop)

    def _browser_view_key(self, item):
        if item is None or item.data(0, INLINE_ROLE):
            return None
        stored = item.data(0, VIEW_KEY_ROLE)
        if stored is not None:
            return stored
        uid = item.data(0, QtCore.Qt.UserRole)
        obj = self.workspace.scene_objects.get(uid) if uid else None
        if obj is not None:
            return ('object', str(obj.uid))
        parent = item.parent()
        return ('node', item.text(0), self._browser_view_key(parent))

    def _capture_browser_view(self):
        tree = self.tree_objects
        scope = (id(self.workspace), str(self.workspace.paths.flow_path))
        if getattr(self, '_browser_case_scope', None) != scope:
            return None
        if not tree.topLevelItemCount():
            return None
        selected, expanded, known = set(), set(), set()
        iterator = QtWidgets.QTreeWidgetItemIterator(tree)
        while iterator.value():
            item = iterator.value()
            key = self._browser_view_key(item)
            if key is not None:
                known.add(key)
                if item.isSelected(): selected.add(key)
                if item.isExpanded(): expanded.add(key)
            iterator += 1
        return selected, expanded, self._browser_view_key(tree.currentItem()), known

    def _restore_browser_view(self, state):
        self._browser_case_scope = (id(self.workspace), str(self.workspace.paths.flow_path))
        if state is None:
            return
        selected, expanded, current, known = state
        iterator = QtWidgets.QTreeWidgetItemIterator(self.tree_objects)
        current_item = None
        while iterator.value():
            item = iterator.value()
            key = self._browser_view_key(item)
            if key in known:
                item.setExpanded(key in expanded)
            item.setSelected(key in selected)
            if key == current:
                current_item = item
            iterator += 1
        if current_item is not None:
            self.tree_objects.setCurrentItem(current_item, 0, QtCore.QItemSelectionModel.NoUpdate)

    def _compact_browser_types(self):
        # A lone layer does not need a second row repeating its type/name.
        iterator = QtWidgets.QTreeWidgetItemIterator(self.tree_objects)
        candidates = []
        while iterator.value():
            item = iterator.value()
            if item.text(1) == 'Type' and item.text(0) != 'Paths' and item.childCount() == 1:
                child = item.child(0)
                if child.data(0, QtCore.Qt.UserRole) is not None:
                    candidates.append(item)
            iterator += 1
        for item in candidates:
            parent = item.parent()
            if parent is not None:
                index = parent.indexOfChild(item)
                child = item.takeChild(0)
                parent.takeChild(index)
                parent.insertChild(index, child)
        iterator = QtWidgets.QTreeWidgetItemIterator(self.tree_objects)
        while iterator.value():
            item = iterator.value()
            uid = item.data(0, QtCore.Qt.UserRole)
            obj = self.workspace.scene_objects.get(uid) if uid else None
            if obj is not None:
                key = 'object:'+str(obj.uid)
                if str(obj.data_key).startswith(('plane_', 'pathline_')):
                    suffix = str(obj.data_key).rsplit('_', 1)[-1]
                    index = int(suffix) if suffix.isdigit() else -1
                    if 0 <= index < len(self.workspace.planes):
                        plane_id = getattr(self.workspace.planes[index], 'plane_id', None)
                        if plane_id:
                            key = ('plane:' if str(obj.data_key).startswith('plane_') else 'pathline:')+str(plane_id)
                item.setData(0, VIEW_KEY_ROLE, key)
            elif item.data(0, int(QtCore.Qt.UserRole)+1) == 'path':
                for child_index in range(item.childCount()):
                    child = item.child(child_index)
                    child_obj = self.workspace.scene_objects.get(child.data(0, QtCore.Qt.UserRole))
                    if child_obj is not None and str(child_obj.data_key).startswith('smooth_path_'):
                        item.setData(0, VIEW_KEY_ROLE, 'path-node:'+str(child_obj.uid))
                        break
            if item.flags() & QtCore.Qt.ItemIsUserCheckable:
                item.setToolTip(1, 'Show or hide '+item.text(0))
            iterator += 1
