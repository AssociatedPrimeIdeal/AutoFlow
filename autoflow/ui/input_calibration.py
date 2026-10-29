"""Label physical calibration separately from stored axes and velocity channels."""

import html

from PySide6 import QtCore, QtGui, QtWidgets

from ..algorithms.data.input_parameters import normalize_input_parameters


_PHYSICAL_AXES = ("LR", "AP", "FH")
_AXIS_NAMES = ("LR · left / right", "AP · front / back", "FH · feet / head")
_CANONICAL = {"LR": "LR", "RL": "LR", "AP": "AP", "PA": "AP", "FH": "FH", "HF": "FH"}
_DIRECTIONS = (("LR", "LR"), ("RL · reverse LR", "RL"), ("AP", "AP"),
               ("PA · reverse AP", "PA"), ("FH", "FH"), ("HF · reverse FH", "HF"))


class InputTripletEditor(QtWidgets.QWidget):
    """Three named fields; keep the settings dialog's text-value contract."""

    changed = QtCore.Signal()

    def __init__(self, key, parent=None):
        super().__init__(parent)
        self.key = key
        self.direction_order = key in {"spatial_order", "venc_order"}
        self.entries = []
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        row = QtWidgets.QGridLayout()
        for index in range(3):
            if key == "spatial_order":
                caption = f"Array axis {index} · dimension {index+1}"
            elif key == "venc_order":
                caption = f"Velocity channel {index}"
            else:
                caption = _AXIS_NAMES[index]
            label = QtWidgets.QLabel(caption)
            if self.direction_order:
                entry = QtWidgets.QComboBox()
                for text, value in _DIRECTIONS:
                    entry.addItem(text, value)
                entry.currentIndexChanged.connect(self.changed)
            else:
                entry = QtWidgets.QLineEdit()
                validator = QtGui.QDoubleValidator(0., 1e12, 15, entry)
                validator.setLocale(QtCore.QLocale.c())
                entry.setValidator(validator)
                entry.textChanged.connect(self.changed)
                entry.setPlaceholderText("mm" if key == "resolution" else "cm/s")
            entry.setObjectName(f"input.{key}.{index}")
            entry.setAccessibleName(f"{key}: {caption}")
            label.setBuddy(entry)
            row.addWidget(label, 0, index)
            row.addWidget(entry, 1, index)
            row.setColumnStretch(index, 1)
            self.entries.append(entry)
        layout.addLayout(row)
        hints = {
            "resolution": "Voxel spacing in millimetres. The fields always mean LR, AP, FH, even when the stored spatial axes use another order.",
            "venc": "Velocity encoding limit in cm/s, entered by physical direction: LR, AP, FH. These three numbers are not entered in channel order.",
            "spatial_order": "Source array dimensions 0/1/2 before reorientation; independent from the order of the slice windows.",
            "venc_order": "Stored velocity channels before reorientation. VENC values stay in physical LR/AP/FH order.",
        }
        hint = QtWidgets.QLabel(hints[key])
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #52636b")
        layout.addWidget(hint)
        self.setToolTip(hints[key])

    def set_values(self, values):
        if isinstance(values, str):
            values = [x.strip() for x in values.replace(';', ',').split(',')]
        elif not isinstance(values, (list, tuple)):
            values = [values]
        if len(values) == 1 and not self.direction_order:
            values = values*3
        if len(values) != 3:
            raise ValueError(f'{self.key} requires three entries')
        blocker = QtCore.QSignalBlocker(self)
        for entry, value in zip(self.entries, values):
            if self.direction_order:
                index = entry.findData(str(value).strip().upper())
                if index < 0:
                    raise ValueError(f'Unknown direction: {value}')
                entry.setCurrentIndex(index)
            else:
                entry.setText(str(value))
        del blocker
        self.changed.emit()

    def setText(self, text):
        self.set_values(text)

    def text(self):
        return ', '.join(entry.currentData() if self.direction_order else entry.text().strip()
                         for entry in self.entries)


def input_mapping_html(values):
    """Preview the existing loader's permutation, sign and calibration contract."""
    values = normalize_input_parameters(values)
    rows = []
    for index, direction in enumerate(values['spatial_order']):
        axis = _CANONICAL[direction]
        position = _PHYSICAL_AXES.index(axis)
        operation = 'reversed' if direction != axis else 'same direction'
        rows.append((f'Array axis {index}', direction, f'Output axis {position} · {axis}',
                     f"{values['resolution'][position]:g} mm · {operation}"))
    for index, direction in enumerate(values['venc_order']):
        axis = _CANONICAL[direction]
        position = _PHYSICAL_AXES.index(axis)
        sign = '−' if direction != axis else '+'
        rows.append((f'Velocity channel {index}', direction, f'Flow {axis} · sign {sign}',
                     f"VENC {axis} = {values['venc'][position]:g} cm/s"))
    body = ''.join('<tr>'+''.join('<td style="padding:3px 9px; border-bottom:1px solid #dce3e6">'+html.escape(cell)+'</td>' for cell in row)+'</tr>'
                   for row in rows)
    return ('<b>Mapping after reload · preview</b><table cellspacing="0" width="100%">'
            '<tr><th align="left">Stored position (zero-based)</th><th align="left">Direction</th>'
            '<th align="left">Normalized destination</th><th align="left">Calibration</th></tr>'+body+'</table>')


def source_channel_help(workspace):
    metadata = workspace.input_state.metadata
    layout = str(metadata.get('h5_layout', ''))
    dual = metadata.get('dual_venc', {})
    if layout == 'complex_img':
        if isinstance(dual, dict) and dual.get('enabled'):
            return ('For this dual-VENC complex input: channel 0/1/2 are the three velocity channels within each encoding triplet '
                    '(stored image slots 1/2/3 and 4/5/6). Image slot 0 is the reference; both triplets share VENCOrder.')
        return 'Complex input: velocity channel 0/1/2 = image slots 1/2/3; reference slot 0 is excluded.'
    if layout == 'combined_img_real':
        return ('For this combined image input: velocity channel 0/1/2 corresponds to stored image slots 1/2/3. '
                'Image slot 0 is magnitude and is not included in VENCOrder.')
    return ('For a mag + flow input: velocity channel 0/1/2 is flow[..., 0/1/2] in the source file. '
            'For a complex reference + 3-channel input, the reference is excluded and velocity channels use image slots 1/2/3.')
