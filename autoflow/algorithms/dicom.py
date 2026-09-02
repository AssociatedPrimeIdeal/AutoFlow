import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..case_types import InputCase, LoadedCase, LoaderCapabilities
from .data import (
    _axis_pair,
    discover_h5_input_cases,
    load_h5_data,
    normalize_loaded_case,
    reorient,
)
from .phase_correction import (
    apply_background_phase_correction_to_mag_flow,
    background_phase_report_for_metadata,
    coerce_background_phase_correction_config,
)

_H5_SUFFIXES = (".h5", ".hdf5")
_DIR_TOKEN_RE = re.compile(r"(?<![A-Z])(RL|LR|AP|PA|HF|FH|SI|IS|RO|PE|SS|IN|TH)(?![A-Z])")
_RR_RE = re.compile(r"RR\s+(\d+)", re.IGNORECASE)


def _import_pydicom():
    try:
        import pydicom  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pydicom is required for direct DICOM loading. Install the project with pydicom available."
        ) from exc
    return pydicom


def _safe_float(value, default=None):
    if value in (None, ""):
        return default
    try:
        out = float(value)
    except Exception:
        return default
    if not np.isfinite(out):
        return default
    return out


def _normalize_vector(vec):
    arr = np.asarray(vec, dtype=float).reshape(-1)
    if arr.size < 3:
        return None
    arr = arr[:3]
    nrm = float(np.linalg.norm(arr))
    if nrm <= 1e-12:
        return None
    return arr / nrm


def _axis_label_from_patient_vector(vec):
    arr = _normalize_vector(vec)
    if arr is None:
        return None
    idx = int(np.argmax(np.abs(arr)))
    sign = float(arr[idx])
    if idx == 0:
        return "RL" if sign >= 0 else "LR"
    if idx == 1:
        return "AP" if sign >= 0 else "PA"
    return "FH" if sign >= 0 else "HF"


def _normalize_direction_token(token):
    token = str(token or "").upper()
    if token in {"RL", "LR", "AP", "PA", "HF", "FH"}:
        return token
    if token == "SI":
        return "HF"
    if token == "IS":
        return "FH"
    return None


def _sanitize_case_name(text):
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text or "").strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "dicom_case"


def _short_uid(uid):
    uid = str(uid or "").strip()
    return uid[-8:] if uid else "case"


def _is_h5_path(path):
    return os.path.isfile(path) and path.lower().endswith(_H5_SUFFIXES)


def _image_type_tokens(ds):
    values = getattr(ds, "ImageType", [])
    if isinstance(values, str):
        values = [values]
    return [str(value).upper() for value in values]


def _header_text(ds):
    fields = [
        getattr(ds, "Manufacturer", ""),
        getattr(ds, "ProtocolName", ""),
        getattr(ds, "SeriesDescription", ""),
        getattr(ds, "SequenceName", ""),
        getattr(ds, "PulseSequenceName", ""),
        getattr(ds, "ComplexImageComponent", ""),
        getattr(ds, "InPlanePhaseEncodingDirection", ""),
    ]
    fields.extend(_image_type_tokens(ds))
    return " ".join(str(field) for field in fields if field).upper()


def _extract_first_number(text):
    match = re.search(r"(\d+(?:\.\d+)?)", str(text or ""))
    if not match:
        return None
    return _safe_float(match.group(1), default=None)


def _extract_venc_from_text(text):
    matches = re.findall(r"(?i)v(\d+(?:\.\d+)?)", str(text or ""))
    if matches:
        return _safe_float(matches[-1], default=None)
    matches = re.findall(r"(\d+(?:\.\d+)?)", str(text or ""))
    if not matches:
        return None
    return _safe_float(matches[-1], default=None)


def _extract_rr_ms(ds):
    for attr in ("HeartRate", "CardiacRate"):
        rate = _safe_float(getattr(ds, attr, None), default=None)
        if rate and rate > 0:
            return float(60000.0 / rate)
    rr = _safe_float(getattr(ds, "CardiacRRIntervalSpecified", None), default=None)
    if rr is not None:
        return rr
    comments = str(getattr(ds, "ImageComments", "") or "")
    match = _RR_RE.search(comments)
    if match:
        return _safe_float(match.group(1), default=None)
    return None


def _extract_iop(ds):
    iop = getattr(ds, "ImageOrientationPatient", None)
    if iop is None:
        shared = getattr(ds, "SharedFunctionalGroupsSequence", None)
        if shared:
            plane = getattr(shared[0], "PlaneOrientationSequence", None)
            if plane and hasattr(plane[0], "ImageOrientationPatient"):
                iop = plane[0].ImageOrientationPatient
    if iop is None:
        return None
    arr = np.asarray(iop, dtype=float).reshape(-1)
    if arr.size < 6:
        return None
    return arr[:6]


def _extract_image_position(ds):
    pos = getattr(ds, "ImagePositionPatient", None)
    if pos is None:
        shared = getattr(ds, "SharedFunctionalGroupsSequence", None)
        if shared:
            plane = getattr(shared[0], "PlanePositionSequence", None)
            if plane and hasattr(plane[0], "ImagePositionPatient"):
                pos = plane[0].ImagePositionPatient
    if pos is None:
        return None
    arr = np.asarray(pos, dtype=float).reshape(-1)
    if arr.size < 3:
        return None
    return arr[:3]


def _extract_pixel_measures(ds):
    spacing = getattr(ds, "PixelSpacing", None)
    thickness = getattr(ds, "SliceThickness", None)
    if spacing is not None and len(spacing) >= 2:
        return (
            _safe_float(spacing[0], default=None),
            _safe_float(spacing[1], default=None),
            _safe_float(thickness, default=None),
        )
    for seq_name in ("SharedFunctionalGroupsSequence", "PerFrameFunctionalGroupsSequence"):
        seq = getattr(ds, seq_name, None)
        if not seq:
            continue
        for item in (seq[0], seq[-1]):
            measures = getattr(item, "PixelMeasuresSequence", None)
            if not measures:
                continue
            spacing = getattr(measures[0], "PixelSpacing", None)
            thickness = getattr(measures[0], "SliceThickness", None)
            if spacing is not None and len(spacing) >= 2:
                return (
                    _safe_float(spacing[0], default=None),
                    _safe_float(spacing[1], default=None),
                    _safe_float(thickness, default=None),
                )
    return None, None, None


def _extract_row_col_slice_dirs(ds):
    iop = _extract_iop(ds)
    if iop is None:
        return None, None, None
    axis1 = _normalize_vector(iop[:3])
    axis0 = _normalize_vector(iop[3:6])
    if axis0 is None or axis1 is None:
        return None, None, None
    axis2 = _normalize_vector(np.cross(axis1, axis0))
    return axis0, axis1, axis2


def _iter_functional_groups(ds):
    shared = getattr(ds, "SharedFunctionalGroupsSequence", None)
    if shared:
        yield shared[0]
    per_frame = getattr(ds, "PerFrameFunctionalGroupsSequence", None)
    if per_frame:
        yield per_frame[0]
        if len(per_frame) > 1:
            yield per_frame[-1]


def _extract_velocity_dir_label(ds):
    for item in _iter_functional_groups(ds):
        vel_seq = getattr(item, "MRVelocityEncodingSequence", None)
        if vel_seq and hasattr(vel_seq[0], "VelocityEncodingDirection"):
            label = _axis_label_from_patient_vector(vel_seq[0].VelocityEncodingDirection)
            if label is not None:
                return label
    return None


def _extract_multiframe_venc(ds):
    for item in _iter_functional_groups(ds):
        vel_seq = getattr(item, "MRVelocityEncodingSequence", None)
        if vel_seq and hasattr(vel_seq[0], "VelocityEncodingMaximumValue"):
            value = _safe_float(vel_seq[0].VelocityEncodingMaximumValue, default=None)
            if value is not None:
                return abs(value)
    return None


def _coerce_dicom_read_workers(dicom_read_workers, total):
    total = int(total or 0)
    if total <= 1:
        return 1
    try:
        requested = 1 if dicom_read_workers is None else int(dicom_read_workers)
    except Exception:
        requested = 1
    if requested < 0:
        requested = 1
    if requested == 0:
        if total < 8:
            return 1
        cpu_count = max(1, int(os.cpu_count() or 1))
        requested = min(total, cpu_count, 8)
    return max(1, min(total, requested))


def _emit_dicom_progress(progress_callback, stage, current, total, path, message_prefix):
    if progress_callback is None:
        return
    try:
        progress_callback(
            {
                "stage": stage,
                "current": int(current),
                "total": int(total),
                "message": f"{message_prefix}: {os.path.basename(path)}",
            }
        )
    except Exception:
        pass


def _map_dicom_entries(
    entries,
    worker,
    progress_callback=None,
    stage="dicom_load_file",
    message_prefix="Loading DICOM pixels",
    dicom_read_workers=1,
):
    total = len(entries)
    actual_workers = _coerce_dicom_read_workers(dicom_read_workers, total)
    results = []
    if actual_workers <= 1:
        for index, entry in enumerate(entries, start=1):
            _emit_dicom_progress(progress_callback, stage, index, total, entry["path"], message_prefix)
            results.append(worker(entry))
        return results, actual_workers

    with ThreadPoolExecutor(max_workers=actual_workers) as pool:
        for index, result in enumerate(pool.map(worker, entries), start=1):
            entry = entries[index - 1]
            _emit_dicom_progress(progress_callback, stage, index, total, entry["path"], message_prefix)
            results.append(result)
    return results, actual_workers


def _label_from_axis_role(ds, role):
    role = str(role or "").upper()
    axis0, axis1, axis2 = _extract_row_col_slice_dirs(ds)
    if axis2 is None:
        return None
    phase_dir = str(getattr(ds, "InPlanePhaseEncodingDirection", "") or "").upper()
    if role == "SS":
        return _axis_label_from_patient_vector(axis2)
    if phase_dir == "ROW":
        pe_axis = axis0
        ro_axis = axis1
    elif phase_dir == "COL":
        pe_axis = axis1
        ro_axis = axis0
    else:
        return None
    if role == "PE":
        return _axis_label_from_patient_vector(pe_axis)
    if role == "RO":
        return _axis_label_from_patient_vector(ro_axis)
    return None


def _extract_component_label_from_text(ds):
    for field_name in ("SequenceName", "ProtocolName", "SeriesDescription", "PulseSequenceName", "ImageComments"):
        text = str(getattr(ds, field_name, "") or "").upper()
        for token in _DIR_TOKEN_RE.findall(text):
            label = _normalize_direction_token(token)
            if label is not None:
                return label
            axis_role = "SS" if token in {"IN", "TH"} else token
            label = _label_from_axis_role(ds, axis_role)
            if label is not None:
                return label
    return None


def _looks_like_flow_related(text, image_type_tokens, component_label):
    if component_label is not None:
        return True
    hints = (
        "4DFLOW",
        "4D FLOW",
        "FLOW",
        "VENC",
        "DELREC",
        "PHASE CONTRAST",
        "PCMRI",
        "WIP",
        "FQ",
    )
    if any(hint in text for hint in hints):
        return True
    if any(token in {"VELOCITY", "FLOW_ENCODED", "PHASE"} for token in image_type_tokens):
        return True
    return False


def _find_referenced_uid(obj, depth=0):
    if obj is None or depth > 4:
        return None
    for attr in ("ReferencedSOPInstanceUID", "ReferencedSeriesInstanceUID", "ReferencedFrameOfReferenceUID"):
        value = getattr(obj, attr, None)
        if value:
            return str(value)
    if depth > 0:
        series_uid = getattr(obj, "SeriesInstanceUID", None)
        if series_uid:
            return str(series_uid)
    for seq_name in (
        "ReferencedImageEvidenceSequence",
        "ReferencedSeriesSequence",
        "ReferencedSOPSequence",
        "ReferencedImageSequence",
    ):
        seq = getattr(obj, seq_name, None)
        if not seq:
            continue
        for item in seq:
            value = _find_referenced_uid(item, depth=depth + 1)
            if value:
                return value
    return None


def _case_uid_from_ds(ds, path):
    uid = _find_referenced_uid(ds)
    if uid:
        return uid
    for attr in ("FrameOfReferenceUID", "StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID"):
        value = getattr(ds, attr, None)
        if value:
            return str(value)
    return os.path.abspath(path)


def _classify_dicom_header(path, ds):
    modality = str(getattr(ds, "Modality", "") or "").upper()
    if modality and modality != "MR":
        return None

    manufacturer = str(getattr(ds, "Manufacturer", "") or "").strip()
    manufacturer_key = manufacturer.lower()
    image_type = _image_type_tokens(ds)
    text = _header_text(ds)
    component_label = _extract_velocity_dir_label(ds) or _extract_component_label_from_text(ds)
    flow_hint = _looks_like_flow_related(text, image_type, component_label)
    number_of_frames = int(_safe_float(getattr(ds, "NumberOfFrames", 1), default=1) or 1)
    is_multiframe = number_of_frames > 1
    group_kind = 2 if is_multiframe and "philips" in manufacturer_key else 1 if is_multiframe else 0
    token = image_type[2] if len(image_type) > 2 else (image_type[-1] if image_type else "")
    is_magnitude = False

    if "siemens" in manufacturer_key:
        if is_multiframe:
            if token == "VELOCITY":
                if component_label is None:
                    return None
            elif token in {"T1", "M", "MAGNITUDE"}:
                is_magnitude = True
            elif flow_hint:
                is_magnitude = True
            else:
                return None
        else:
            if token == "P":
                if component_label is None:
                    return None
            elif token in {"M", "T1"} and flow_hint:
                is_magnitude = True
            else:
                return None
    elif "philips" in manufacturer_key:
        tail = image_type[-2] if len(image_type) >= 2 else token
        if is_multiframe:
            if component_label is not None:
                pass
            elif tail in {"P", "PHASE"} or any(v in image_type for v in ("FLOW_ENCODED", "VELOCITY")):
                component_label = _extract_component_label_from_text(ds)
                if component_label is None:
                    return None
            elif flow_hint:
                is_magnitude = True
            else:
                return None
        else:
            if tail == "P":
                if component_label is None:
                    return None
            elif tail == "M" and "M_PCA" not in text and flow_hint:
                is_magnitude = True
            else:
                return None
    elif "ge" in manufacturer_key:
        desc = str(getattr(ds, "SeriesDescription", "") or "").upper()
        if "ANATOMY" in desc:
            is_magnitude = True
        else:
            if component_label is None:
                if "SI" in desc:
                    component_label = "FH"
                elif "IS" in desc:
                    component_label = "HF"
                elif "AP" in desc:
                    component_label = "AP"
                elif "PA" in desc:
                    component_label = "PA"
                elif "LR" in desc:
                    component_label = "LR"
                elif "RL" in desc:
                    component_label = "RL"
            if component_label is None:
                return None
    elif "uih" in manufacturer_key:
        sequence_name = str(getattr(ds, "SequenceName", "") or "").upper()
        if component_label is None:
            if "RO" in text or "PE" in text or "SS" in text:
                component_label = _extract_component_label_from_text(ds)
        if component_label is None:
            if "FQ" not in sequence_name and not flow_hint:
                return None
            is_magnitude = True
        elif "FQ" not in sequence_name and not flow_hint:
            return None
    else:
        if component_label is None and not flow_hint:
            return None
        if component_label is None:
            is_magnitude = True

    return {
        "path": os.path.abspath(path),
        "case_id": _case_uid_from_ds(ds, path),
        "manufacturer": manufacturer,
        "group_kind": int(group_kind),
        "component_label": component_label,
        "is_magnitude": bool(is_magnitude),
        "series_description": str(getattr(ds, "SeriesDescription", "") or "").strip(),
        "protocol_name": str(getattr(ds, "ProtocolName", "") or "").strip(),
    }


def _build_dicom_display_name(root, case_id, items):
    desc = next((item["series_description"] for item in items if item.get("series_description")), "")
    protocol = next((item["protocol_name"] for item in items if item.get("protocol_name")), "")
    title = desc or protocol or os.path.basename(root.rstrip(os.sep)) or "DICOM Case"
    manufacturer = next((item["manufacturer"] for item in items if item.get("manufacturer")), "Unknown")
    pairs = {}
    for item in items:
        label = item.get("component_label")
        if label:
            pairs[_axis_pair(label)] = label
    components = ",".join(pairs.get(pair, pair) for pair in ("LR", "AP", "HF") if pair in pairs)
    detail = components if components else "mag+flow"
    return f"{title} | {manufacturer} | {detail} | {len(items)} files | { _short_uid(case_id)}"


def _build_dicom_output_name(root, case_id, items):
    desc = next((item["series_description"] for item in items if item.get("series_description")), "")
    protocol = next((item["protocol_name"] for item in items if item.get("protocol_name")), "")
    stem = os.path.basename(root.rstrip(os.sep)) or "dicom"
    title = desc or protocol or "case"
    return f"{_sanitize_case_name(stem)}__{_sanitize_case_name(title)}__{_short_uid(case_id)}"


def _iter_candidate_dicom_files(root):
    for dirpath, _, filenames in os.walk(root):
        for filename in sorted(filenames):
            if filename == "DICOMDIR":
                continue
            lower = filename.lower()
            if lower.endswith(".dcm") or lower.endswith(".ima") or "." not in filename:
                yield os.path.join(dirpath, filename)


def _collect_h5_files_from_dir(root):
    matches = []
    for filename in sorted(os.listdir(root)):
        path = os.path.join(root, filename)
        if not os.path.isfile(path):
            continue
        lower = filename.lower()
        if lower.endswith(_H5_SUFFIXES):
            try:
                import h5py

                if h5py.is_hdf5(path):
                    matches.append(path)
            except Exception:
                continue
    return sorted(matches)


def scan_dicom_cases(root, progress_callback=None):
    root = os.path.abspath(root)
    if not os.path.isdir(root):
        raise ValueError(f"not a directory: {root}")
    pydicom = _import_pydicom()
    grouped = {}
    paths = list(_iter_candidate_dicom_files(root))
    total = len(paths)

    for index, path in enumerate(paths, start=1):
        if progress_callback is not None:
            try:
                progress_callback(
                    {
                        "stage": "dicom_scan_file",
                        "current": int(index),
                        "total": int(total),
                        "message": f"Scanning DICOM headers: {os.path.basename(path)}",
                    }
                )
            except Exception:
                pass
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        except Exception:
            continue
        try:
            item = _classify_dicom_header(path, ds)
        except Exception:
            item = None
        if item is None:
            continue
        grouped.setdefault(item["case_id"], []).append(item)

    cases = []
    for case_id, items in grouped.items():
        if not any(item["is_magnitude"] for item in items):
            continue
        pairs = {}
        for item in items:
            label = item.get("component_label")
            if label:
                pairs[_axis_pair(label)] = label
        if len(pairs) < 3:
            continue
        group_kind = int(round(float(np.median([item["group_kind"] for item in items]))))
        display_name = _build_dicom_display_name(root, case_id, items)
        output_name = _build_dicom_output_name(root, case_id, items)
        cases.append(
            InputCase(
                input_path=root,
                input_kind="dicom",
                display_name=display_name,
                output_name=output_name,
                source_group=str(case_id),
                metadata={
                    "manufacturer": next((item["manufacturer"] for item in items if item.get("manufacturer")), ""),
                    "group_kind": group_kind,
                    "series_description": next((item["series_description"] for item in items if item.get("series_description")), ""),
                    "protocol_name": next((item["protocol_name"] for item in items if item.get("protocol_name")), ""),
                    "file_count": len(items),
                    "components": [pairs[pair] for pair in ("LR", "AP", "HF") if pair in pairs],
                    "dicom_entries": items,
                },
            )
        )

    if progress_callback is not None:
        try:
            progress_callback(
                {
                    "stage": "dicom_scan_done",
                    "current": int(total),
                    "total": int(total),
                    "message": f"Found {len(cases)} DICOM case(s)",
                }
            )
        except Exception:
            pass

    return sorted(cases, key=lambda case: case.display_name.lower())


def resolve_input_case(input_source):
    if isinstance(input_source, InputCase):
        return input_source

    path = os.path.abspath(str(input_source))
    if _is_h5_path(path):
        cases = discover_h5_input_cases(path)
        if not cases:
            raise ValueError(f"no supported H5 cases found: {path}")
        if len(cases) > 1:
            labels = ", ".join(case.display_name for case in cases[:3])
            suffix = "" if len(cases) <= 3 else f", ... ({len(cases)} total)"
            raise ValueError(f"multiple H5 cases found in {path}: {labels}{suffix}")
        return cases[0]

    root = path if os.path.isdir(path) else os.path.dirname(path)
    if not root:
        raise ValueError(f"unsupported input path: {path}")

    cases = scan_dicom_cases(root)
    if os.path.isfile(path):
        cases = [
            case
            for case in cases
            if any(entry["path"] == path for entry in case.metadata.get("dicom_entries", []))
        ]
    if not cases:
        raise ValueError(f"no supported DICOM cases found: {path}")
    if len(cases) > 1:
        labels = ", ".join(case.display_name for case in cases[:3])
        suffix = "" if len(cases) <= 3 else f", ... ({len(cases)} total)"
        raise ValueError(f"multiple DICOM cases found in {path}: {labels}{suffix}")
    return cases[0]


def collect_input_cases(inputs):
    discovered = []
    seen = set()

    def _append(case):
        key = (case.input_kind, os.path.abspath(case.input_path), str(case.source_group or ""))
        if key not in seen:
            seen.add(key)
            discovered.append(case)

    for item in inputs:
        path = os.path.abspath(str(item))
        if _is_h5_path(path):
            for case in discover_h5_input_cases(path):
                _append(case)
            continue
        if os.path.isdir(path):
            for h5_path in _collect_h5_files_from_dir(path):
                for case in discover_h5_input_cases(h5_path):
                    _append(case)
            for case in scan_dicom_cases(path):
                _append(case)
            continue
        if os.path.isfile(path):
            root = os.path.dirname(path)
            cases = scan_dicom_cases(root)
            for case in cases:
                if any(entry["path"] == path for entry in case.metadata.get("dicom_entries", [])):
                    _append(case)

    return discovered


def _component_label_order(labels):
    by_pair = {}
    for label in labels:
        by_pair[_axis_pair(label)] = label
    ordered = []
    for pair in ("LR", "AP", "HF"):
        if pair not in by_pair:
            raise ValueError(f"missing velocity component for axis pair {pair}")
        ordered.append(by_pair[pair])
    return ordered


def _extract_time_key(ds):
    for attr in ("TriggerTime", "TemporalPositionIdentifier"):
        value = _safe_float(getattr(ds, attr, None), default=None)
        if value is not None:
            return value
    acq_time = str(getattr(ds, "AcquisitionTime", "") or "").strip()
    if acq_time:
        digits = re.sub(r"[^0-9.]", "", acq_time)
        value = _safe_float(digits, default=None)
        if value is not None:
            return value
    value = _safe_float(getattr(ds, "InstanceNumber", None), default=None)
    return 0.0 if value is None else value


def _extract_slice_key(ds, axis2_dir):
    pos = _extract_image_position(ds)
    if pos is not None and axis2_dir is not None:
        return float(np.dot(np.asarray(pos, dtype=float).reshape(3), np.asarray(axis2_dir, dtype=float).reshape(3)))
    for attr in ("SliceLocation", "InStackPositionNumber", "InstanceNumber"):
        value = _safe_float(getattr(ds, attr, None), default=None)
        if value is not None:
            return value
    return 0.0


def _slice_spacing_from_keys(keys, fallback):
    uniq = sorted({float(key) for key in keys})
    if len(uniq) >= 2:
        diffs = np.diff(np.asarray(uniq, dtype=float))
        diffs = np.abs(diffs[diffs != 0])
        if diffs.size > 0:
            return float(np.median(diffs))
    if fallback is not None:
        return float(fallback)
    return 1.0


def _reorder_row_col_slice_to_xyz(arr, axis_dirs, axis_spacings):
    dominant = [int(np.argmax(np.abs(np.asarray(vec, dtype=float).reshape(3)))) for vec in axis_dirs]
    if len(set(dominant)) != 3:
        raise ValueError("oblique DICOM orientation is not supported by the direct loader")
    perm = [dominant.index(0), dominant.index(1), dominant.index(2)]
    arr_xyz = np.transpose(arr, perm + list(range(3, arr.ndim)))
    spatial_order = [_axis_label_from_patient_vector(axis_dirs[idx]) for idx in perm]
    if any(label is None for label in spatial_order):
        raise ValueError("failed to determine DICOM spatial axis labels")
    resolution = np.asarray([axis_spacings[idx] for idx in perm], dtype=float)
    return arr_xyz, spatial_order, resolution


def _default_venc(value):
    value = _safe_float(value, default=None)
    return 150.0 if value is None else abs(value)


def _coerce_float_triplet(value, default):
    arr = np.asarray(default, dtype=float).reshape(-1)
    if value is None:
        return np.asarray(arr[:3], dtype=float)
    try:
        raw = np.asarray(value, dtype=float).reshape(-1)
    except Exception:
        return np.asarray(arr[:3], dtype=float)
    if raw.size == 1:
        raw = np.repeat(raw, 3)
    if raw.size < 3:
        return np.asarray(arr[:3], dtype=float)
    return np.asarray(raw[:3], dtype=float)


def _coerce_label_triplet(value, default):
    labels = [str(x).upper() for x in list(default)[:3]]
    if value is None:
        return labels
    raw = [str(x).strip().upper() for x in value]
    if len(raw) < 3:
        return labels
    return raw[:3]


def _resolve_dicom_entries(case):
    if not isinstance(case, InputCase):
        case = resolve_input_case(case)
    if case.input_kind != "dicom":
        raise ValueError(f"not a DICOM case: {case.input_kind}")
    entries = list(case.metadata.get("dicom_entries", []))
    if not entries:
        resolved = resolve_input_case(case.input_path)
        entries = list(resolved.metadata.get("dicom_entries", []))
        if case.source_group:
            entries = [entry for entry in entries if entry.get("case_id") == case.source_group]
    if not entries:
        raise ValueError(f"empty DICOM case: {case.input_path}")
    return case, entries


def _estimate_entry_venc(ds, entry, group_kind):
    label = entry.get("component_label")
    if entry.get("is_magnitude", False) or not label:
        return None
    manufacturer = str(entry.get("manufacturer", "") or "").lower()
    if group_kind == 0:
        if "siemens" in manufacturer:
            return _extract_venc_from_text(getattr(ds, "SequenceName", ""))
        if "philips" in manufacturer:
            intercept = _safe_float(getattr(ds, "RescaleIntercept", None), default=0.0)
            return abs(intercept) if intercept is not None and abs(intercept) > 1e-12 else _extract_first_number(getattr(ds, "ProtocolName", ""))
        if "ge" in manufacturer:
            try:
                venc = _safe_float(ds[0x0019, 0x10CC].value, default=None)
                return None if venc is None else venc / 10.0
            except Exception:
                return None
        if "uih" in manufacturer:
            return _extract_venc_from_text(getattr(ds, "SeriesDescription", ""))
        return _extract_first_number(_header_text(ds))
    return _extract_multiframe_venc(ds)


def _preview_matrix_size_xyz(axis_dirs, row_count, col_count, slice_count, time_count):
    raw_shape = (
        max(1, int(row_count or 1)),
        max(1, int(col_count or 1)),
        max(1, int(slice_count or 1)),
    )
    spatial_shape_xyz = _reorder_row_col_slice_to_xyz(
        np.zeros(raw_shape, dtype=np.uint8),
        axis_dirs,
        (1.0, 1.0, 1.0),
    )[0].shape[:3]
    return [int(x) for x in spatial_shape_xyz] + [max(1, int(time_count or 1))]


def inspect_dicom_case(case, progress_callback=None):
    case, entries = _resolve_dicom_entries(case)
    pydicom = _import_pydicom()
    group_kind = int(case.metadata.get("group_kind", 0))
    rr_values = []
    venc_map = {}
    slice_keys = []
    time_keys = []
    axis0 = axis1 = axis2 = None
    row_spacing = col_spacing = thickness = None
    row_count = col_count = None
    frame_time_count = None
    multiframe_slice_count = None
    total = len(entries)

    for index, entry in enumerate(entries, start=1):
        if progress_callback is not None:
            try:
                progress_callback(
                    {
                        "stage": "dicom_inspect_file",
                        "current": int(index),
                        "total": int(total),
                        "message": f"Inspecting DICOM parameters: {os.path.basename(entry['path'])}",
                    }
                )
            except Exception:
                pass
        ds = pydicom.dcmread(entry["path"], stop_before_pixels=True, force=True)
        rr = _extract_rr_ms(ds)
        if rr is not None:
            rr_values.append(rr)
        venc = _estimate_entry_venc(ds, entry, group_kind)
        label = entry.get("component_label")
        if label and venc is not None:
            venc_map[label] = venc
        a0, a1, a2 = _extract_row_col_slice_dirs(ds)
        if a2 is not None:
            axis0, axis1, axis2 = a0, a1, a2
        sp0, sp1, thick = _extract_pixel_measures(ds)
        row_spacing = sp0 if sp0 is not None else row_spacing
        col_spacing = sp1 if sp1 is not None else col_spacing
        thickness = thick if thick is not None else thickness
        rows = int(_safe_float(getattr(ds, "Rows", None), default=0) or 0)
        cols = int(_safe_float(getattr(ds, "Columns", None), default=0) or 0)
        row_count = rows if rows > 0 else row_count
        col_count = cols if cols > 0 else col_count
        if axis2 is not None:
            if group_kind == 0:
                slice_keys.append(_extract_slice_key(ds, axis2))
                time_keys.append(_extract_time_key(ds))
            elif group_kind == 1:
                frame_count = int(_safe_float(getattr(ds, "NumberOfFrames", 1), default=1) or 1)
                frame_time_count = frame_count if frame_time_count is None else max(int(frame_time_count), frame_count)
                try:
                    last_fg = ds.PerFrameFunctionalGroupsSequence[-1]
                    fc = getattr(last_fg, "FrameContentSequence", None)
                    if fc and hasattr(fc[0], "InStackPositionNumber"):
                        slice_keys.append(float(_safe_float(fc[0].InStackPositionNumber, default=0.0) or 0.0))
                except Exception:
                    slice_keys.append(_extract_slice_key(ds, axis2))
            elif group_kind == 2:
                frame_count = int(_safe_float(getattr(ds, "NumberOfFrames", 0), default=0) or 0)
                slice_count = None
                try:
                    last_fg = ds.PerFrameFunctionalGroupsSequence[-1]
                    fc = getattr(last_fg, "FrameContentSequence", None)
                    if fc and hasattr(fc[0], "InStackPositionNumber"):
                        slice_count = int(_safe_float(fc[0].InStackPositionNumber, default=0) or 0)
                except Exception:
                    slice_count = None
                if slice_count and slice_count > 0:
                    multiframe_slice_count = (
                        slice_count
                        if multiframe_slice_count is None
                        else max(int(multiframe_slice_count), int(slice_count))
                    )
                    factor = 2 if bool(entry.get("is_magnitude", False)) else 3
                    if frame_count >= factor * slice_count:
                        time_candidate = max(1, frame_count // (factor * slice_count))
                        frame_time_count = (
                            time_candidate
                            if frame_time_count is None
                            else max(int(frame_time_count), int(time_candidate))
                        )

    if axis2 is None:
        raise ValueError("ImageOrientationPatient is required for direct DICOM loading")

    component_labels = _component_label_order([entry["component_label"] for entry in entries if entry.get("component_label")])
    slice_spacing = _slice_spacing_from_keys(slice_keys, thickness)
    _dummy, spatial_order, resolution = _reorder_row_col_slice_to_xyz(
        np.zeros((1, 1, 1), dtype=np.float32),
        (axis0, axis1, axis2),
        (
            1.0 if row_spacing is None else float(row_spacing),
            1.0 if col_spacing is None else float(col_spacing),
            float(slice_spacing) if group_kind in (0, 1) else (1.0 if thickness is None else float(thickness)),
        ),
    )
    if group_kind == 0:
        matrix_time_count = len(set(time_keys))
        matrix_slice_count = len(set(slice_keys))
    elif group_kind == 1:
        matrix_time_count = frame_time_count
        matrix_slice_count = len(set(slice_keys))
    else:
        matrix_time_count = frame_time_count
        matrix_slice_count = multiframe_slice_count
    raw_venc = np.asarray([_default_venc(venc_map.get(label)) for label in component_labels], dtype=float)
    preview = {
        "resolution": np.asarray(resolution, dtype=float).reshape(3).tolist(),
        "venc": raw_venc.reshape(3).tolist(),
        "spatial_order": list(spatial_order),
        "venc_order": list(component_labels),
        "matrix_size": _preview_matrix_size_xyz(
            (axis0, axis1, axis2),
            row_count,
            col_count,
            matrix_slice_count,
            matrix_time_count,
        ),
        "rr": float(rr_values[0]) if rr_values else 1000.0,
        "display_name": case.display_name,
        "file_count": int(len(entries)),
        "group_kind": group_kind,
    }
    return preview


def _finalize_loaded_dicom_case(
    case,
    entries,
    mag_rczt,
    flow_rczt3,
    component_labels,
    axis_dirs,
    axis_spacings,
    venc_map,
    rr,
    correction_config=None,
    progress_callback=None,
    parameter_overrides=None,
    dicom_read_workers=1,
):
    mag_xyz, spatial_order, resolution = _reorder_row_col_slice_to_xyz(mag_rczt, axis_dirs, axis_spacings)
    flow_xyz, _, _ = _reorder_row_col_slice_to_xyz(flow_rczt3, axis_dirs, axis_spacings)
    overrides = dict(parameter_overrides or {})
    spatial_order = _coerce_label_triplet(overrides.get("spatial_order"), spatial_order)
    resolution = _coerce_float_triplet(overrides.get("resolution"), resolution)
    component_labels = _coerce_label_triplet(overrides.get("venc_order"), component_labels)
    zero_seg = np.zeros(mag_xyz.shape, dtype=np.int16)
    raw_venc = _coerce_float_triplet(
        overrides.get("venc"),
        [_default_venc(venc_map.get(label)) for label in component_labels],
    )
    rr_value = _safe_float(overrides.get("rr"), default=rr)
    flow_out, mag_out, _, venc_out, resolution_out = reorient(
        mag_xyz,
        flow_xyz,
        zero_seg,
        venc=raw_venc,
        resolution=resolution,
        spatial_order=spatial_order,
        venc_order=component_labels,
        target_spatial_order=("LR", "AP", "FH"),
        target_venc_order=("LR", "AP", "FH"),
        return_velocity=False,
    )
    if correction_config is None:
        cfg = coerce_background_phase_correction_config({"enabled": False})
    else:
        cfg = coerce_background_phase_correction_config(correction_config)
    flow_corr, _stationary_mask, corr_report = apply_background_phase_correction_to_mag_flow(
        mag_out,
        flow_out,
        venc_out,
        config=cfg,
        progress_callback=progress_callback,
    )
    meta = {
        "manufacturer": case.metadata.get("manufacturer", ""),
        "series_description": case.metadata.get("series_description", ""),
        "protocol_name": case.metadata.get("protocol_name", ""),
        "dicom_root": case.input_path,
        "group_kind": case.metadata.get("group_kind"),
        "file_count": len(entries),
        "spatial_order_raw": list(spatial_order),
        "venc_order_raw": list(component_labels),
        "dicom_read_workers": int(dicom_read_workers),
        "background_phase_correction": background_phase_report_for_metadata(corr_report),
        "dicom_parameter_overrides": dict(overrides),
    }
    return normalize_loaded_case(
        flow=flow_corr,
        mag=mag_out,
        segmentation=None,
        resolution=resolution_out,
        origin=np.zeros(3, dtype=float),
        venc=venc_out,
        rr=1000.0 if rr_value is None else float(rr_value),
        metadata=meta,
        source_format="dicom",
        source_group=case.source_group,
        capabilities=LoaderCapabilities(
            has_segmentation=False,
            has_tke=False,
            has_complex_source=False,
            supports_wss=True,
            supports_plane_metrics=True,
        ),
    )


def _extract_group_rescale(ds):
    for item in _iter_functional_groups(ds):
        pv = getattr(item, "PixelValueTransformationSequence", None)
        if pv:
            slope = _safe_float(getattr(pv[0], "RescaleSlope", None), default=1.0)
            intercept = _safe_float(getattr(pv[0], "RescaleIntercept", None), default=0.0)
            return slope, intercept
    slope = _safe_float(getattr(ds, "RescaleSlope", None), default=1.0)
    intercept = _safe_float(getattr(ds, "RescaleIntercept", None), default=0.0)
    return slope, intercept


def _extract_siemens_phase_scale(ds, data=None, intercept=None):
    intercept_value = _safe_float(intercept, default=None)
    if intercept_value is not None and abs(intercept_value) > 1e-12:
        return abs(intercept_value)

    largest = _safe_float(getattr(ds, "LargestImagePixelValue", None), default=None)
    smallest = _safe_float(getattr(ds, "SmallestImagePixelValue", None), default=None)
    if largest is not None or smallest is not None:
        candidates = [abs(value) for value in (largest, smallest) if value is not None]
        if candidates:
            return max(candidates)

    bits_stored = _safe_float(getattr(ds, "BitsStored", None), default=None)
    pixel_representation = int(_safe_float(getattr(ds, "PixelRepresentation", None), default=1) or 1)
    if bits_stored is not None and bits_stored > 0:
        bits_stored = int(bits_stored)
        if pixel_representation == 1:
            return float(2 ** max(bits_stored - 1, 0))
        return float(max((2 ** bits_stored) - 1, 1))

    if data is not None:
        max_abs = float(np.nanmax(np.abs(np.asarray(data, dtype=float)))) if np.size(data) else 0.0
        if np.isfinite(max_abs) and max_abs > 1e-12:
            return max_abs

    return None


def _scale_siemens_velocity_frame(pixel, ds, venc, slope=1.0, intercept=0.0):
    data = np.asarray(pixel, dtype=np.float32) * float(slope) + float(intercept)
    if venc is None:
        return np.asarray(data, dtype=np.float32)
    phase_scale = _extract_siemens_phase_scale(ds, data=data, intercept=intercept)
    if phase_scale is None or phase_scale <= 1e-12:
        return np.asarray(data, dtype=np.float32)
    return np.asarray((data / float(phase_scale)) * float(venc), dtype=np.float32)


def _convert_group0_frame(ds, entry, pixel=None):
    pixel = np.asarray(ds.pixel_array if pixel is None else pixel, dtype=np.float32)
    slope = _safe_float(getattr(ds, "RescaleSlope", None), default=1.0)
    intercept = _safe_float(getattr(ds, "RescaleIntercept", None), default=0.0)
    manufacturer = str(entry.get("manufacturer", "") or "").lower()
    label = entry.get("component_label")

    if entry.get("is_magnitude", False):
        return pixel * slope + intercept, None

    if "siemens" in manufacturer:
        venc = _extract_venc_from_text(getattr(ds, "SequenceName", ""))
        data = _scale_siemens_velocity_frame(pixel, ds, venc, slope=slope, intercept=intercept)
        return data.astype(np.float32), venc

    if "philips" in manufacturer:
        data = pixel * slope + intercept
        venc = abs(intercept) if abs(intercept) > 1e-12 else _extract_first_number(getattr(ds, "ProtocolName", ""))
        return data.astype(np.float32), venc

    if "ge" in manufacturer:
        data = pixel / 10.0
        venc = None
        try:
            venc = _safe_float(ds[0x0019, 0x10CC].value, default=None)
            if venc is not None:
                venc /= 10.0
        except Exception:
            venc = None
        return data.astype(np.float32), venc

    if "uih" in manufacturer:
        data = pixel * slope + intercept
        venc = _extract_venc_from_text(getattr(ds, "SeriesDescription", ""))
        return data.astype(np.float32), venc

    data = pixel * slope + intercept
    venc = _extract_first_number(_header_text(ds))
    return data.astype(np.float32), venc


def _load_group0_case(
    case,
    entries,
    correction_config=None,
    progress_callback=None,
    parameter_overrides=None,
    dicom_read_workers=1,
):
    pydicom = _import_pydicom()
    mag_records = []
    vel_records = []
    rr_values = []
    axis0 = axis1 = axis2 = None
    row_spacing = col_spacing = thickness = None

    def _read_entry(entry):
        ds = pydicom.dcmread(entry["path"], force=True)
        pixel = np.asarray(ds.pixel_array, dtype=np.float32)
        if pixel.ndim != 2:
            return None
        data, venc = _convert_group0_frame(ds, entry, pixel=pixel)
        rr = _extract_rr_ms(ds)
        a0, a1, a2 = _extract_row_col_slice_dirs(ds)
        sp0, sp1, thick = _extract_pixel_measures(ds)
        return {
            "rr": rr,
            "axis_dirs": (a0, a1, a2),
            "spacing": (sp0, sp1, thick),
            "slice_key": _extract_slice_key(ds, a2),
            "time_key": _extract_time_key(ds),
            "data": np.asarray(data, dtype=np.float32),
            "label": entry.get("component_label"),
            "venc": venc,
            "is_magnitude": bool(entry.get("is_magnitude", False)),
        }

    loaded_entries, actual_workers = _map_dicom_entries(
        entries,
        _read_entry,
        progress_callback=progress_callback,
        dicom_read_workers=dicom_read_workers,
    )
    for record in loaded_entries:
        if record is None:
            continue
        if record["rr"] is not None:
            rr_values.append(record["rr"])
        a0, a1, a2 = record["axis_dirs"]
        if a2 is not None:
            axis0, axis1, axis2 = a0, a1, a2
        sp0, sp1, thick = record["spacing"]
        row_spacing = sp0 if sp0 is not None else row_spacing
        col_spacing = sp1 if sp1 is not None else col_spacing
        thickness = thick if thick is not None else thickness
        if record["is_magnitude"]:
            mag_records.append(record)
        else:
            vel_records.append(record)

    if not mag_records or not vel_records:
        raise ValueError("incomplete classic DICOM case")
    if axis2 is None:
        raise ValueError("ImageOrientationPatient is required for direct DICOM loading")

    component_labels = _component_label_order([record["label"] for record in vel_records if record.get("label")])
    slice_keys = sorted({record["slice_key"] for record in mag_records + vel_records})
    time_keys = sorted({record["time_key"] for record in mag_records + vel_records})
    row_count, col_count = mag_records[0]["data"].shape
    mag = np.zeros((row_count, col_count, len(slice_keys), len(time_keys)), dtype=np.float32)
    flow_map = {label: np.zeros_like(mag) for label in component_labels}
    slice_index = {value: idx for idx, value in enumerate(slice_keys)}
    time_index = {value: idx for idx, value in enumerate(time_keys)}
    venc_map = {}

    for record in mag_records:
        mag[:, :, slice_index[record["slice_key"]], time_index[record["time_key"]]] = record["data"]
    for record in vel_records:
        label = record["label"]
        if label not in flow_map:
            continue
        flow_map[label][:, :, slice_index[record["slice_key"]], time_index[record["time_key"]]] = record["data"]
        if record["venc"] is not None:
            venc_map[label] = record["venc"]

    flow = np.stack([flow_map[label] for label in component_labels], axis=-1)
    slice_spacing = _slice_spacing_from_keys(slice_keys, thickness)
    return _finalize_loaded_dicom_case(
        case,
        entries,
        mag,
        flow,
        component_labels,
        (axis0, axis1, axis2),
        (
            1.0 if row_spacing is None else float(row_spacing),
            1.0 if col_spacing is None else float(col_spacing),
            float(slice_spacing),
        ),
        venc_map,
        rr_values[0] if rr_values else None,
        correction_config=correction_config,
        progress_callback=progress_callback,
        parameter_overrides=parameter_overrides,
        dicom_read_workers=actual_workers,
    )


def _convert_group1_entry(ds, entry):
    pixel = np.asarray(ds.pixel_array, dtype=np.float32)
    if pixel.ndim != 3:
        raise ValueError("expected Siemens enhanced 3D pixel array [T, Y, X]")
    slope, intercept = _extract_group_rescale(ds)
    label = entry.get("component_label")
    venc = _extract_multiframe_venc(ds) if label else None
    data = _scale_siemens_velocity_frame(pixel, ds, venc, slope=slope, intercept=intercept) if label else pixel * slope + intercept
    slice_key = None
    for item in _iter_functional_groups(ds):
        frame_content = getattr(item, "FrameContentSequence", None)
        if frame_content and hasattr(frame_content[0], "InStackPositionNumber"):
            slice_key = _safe_float(frame_content[0].InStackPositionNumber, default=None)
    if slice_key is None:
        _, _, axis2 = _extract_row_col_slice_dirs(ds)
        slice_key = _extract_slice_key(ds, axis2)
    return np.asarray(data, dtype=np.float32), venc, float(slice_key)


def _load_group1_case(
    case,
    entries,
    correction_config=None,
    progress_callback=None,
    parameter_overrides=None,
    dicom_read_workers=1,
):
    pydicom = _import_pydicom()
    mag_records = []
    vel_records = []
    rr_values = []
    axis0 = axis1 = axis2 = None
    row_spacing = col_spacing = thickness = None

    def _read_entry(entry):
        ds = pydicom.dcmread(entry["path"], force=True)
        data_tyx, venc, slice_key = _convert_group1_entry(ds, entry)
        rr = _extract_rr_ms(ds)
        a0, a1, a2 = _extract_row_col_slice_dirs(ds)
        sp0, sp1, thick = _extract_pixel_measures(ds)
        return {
            "rr": rr,
            "axis_dirs": (a0, a1, a2),
            "spacing": (sp0, sp1, thick),
            "slice_key": float(slice_key),
            "data": data_tyx,
            "label": entry.get("component_label"),
            "venc": venc,
            "is_magnitude": bool(entry.get("is_magnitude", False)),
        }

    loaded_entries, actual_workers = _map_dicom_entries(
        entries,
        _read_entry,
        progress_callback=progress_callback,
        dicom_read_workers=dicom_read_workers,
    )
    for record in loaded_entries:
        if record["rr"] is not None:
            rr_values.append(record["rr"])
        a0, a1, a2 = record["axis_dirs"]
        if a2 is not None:
            axis0, axis1, axis2 = a0, a1, a2
        sp0, sp1, thick = record["spacing"]
        row_spacing = sp0 if sp0 is not None else row_spacing
        col_spacing = sp1 if sp1 is not None else col_spacing
        thickness = thick if thick is not None else thickness
        if record["is_magnitude"]:
            mag_records.append(record)
        else:
            vel_records.append(record)

    if not mag_records or not vel_records:
        raise ValueError("incomplete Siemens enhanced DICOM case")
    if axis2 is None:
        raise ValueError("ImageOrientationPatient is required for direct DICOM loading")

    component_labels = _component_label_order([record["label"] for record in vel_records if record.get("label")])
    slice_keys = sorted({record["slice_key"] for record in mag_records + vel_records})
    row_count, col_count = mag_records[0]["data"].shape[1:]
    time_count = mag_records[0]["data"].shape[0]
    mag = np.zeros((row_count, col_count, len(slice_keys), time_count), dtype=np.float32)
    flow_map = {label: np.zeros_like(mag) for label in component_labels}
    slice_index = {value: idx for idx, value in enumerate(slice_keys)}
    venc_map = {}

    for record in mag_records:
        mag[:, :, slice_index[record["slice_key"]], :] = np.transpose(record["data"], (1, 2, 0))
    for record in vel_records:
        label = record["label"]
        if label not in flow_map:
            continue
        flow_map[label][:, :, slice_index[record["slice_key"]], :] = np.transpose(record["data"], (1, 2, 0))
        if record["venc"] is not None:
            venc_map[label] = record["venc"]

    flow = np.stack([flow_map[label] for label in component_labels], axis=-1)
    slice_spacing = _slice_spacing_from_keys(slice_keys, thickness)
    return _finalize_loaded_dicom_case(
        case,
        entries,
        mag,
        flow,
        component_labels,
        (axis0, axis1, axis2),
        (
            1.0 if row_spacing is None else float(row_spacing),
            1.0 if col_spacing is None else float(col_spacing),
            float(slice_spacing),
        ),
        venc_map,
        rr_values[0] if rr_values else None,
        correction_config=correction_config,
        progress_callback=progress_callback,
        parameter_overrides=parameter_overrides,
        dicom_read_workers=actual_workers,
    )


def _convert_group2_entry(ds, entry):
    pixel = np.asarray(ds.pixel_array, dtype=np.float32)
    if pixel.ndim != 3:
        raise ValueError("expected Philips enhanced 3D pixel array [F, Y, X]")
    first_fg = None
    last_fg = None
    per_frame = getattr(ds, "PerFrameFunctionalGroupsSequence", None)
    if per_frame:
        first_fg = per_frame[0]
        last_fg = per_frame[-1]
    if first_fg is None or last_fg is None:
        raise ValueError("Philips enhanced DICOM is missing PerFrameFunctionalGroupsSequence")

    frame_content = getattr(last_fg, "FrameContentSequence", None)
    if not frame_content or not hasattr(frame_content[0], "InStackPositionNumber"):
        raise ValueError("Philips enhanced DICOM is missing InStackPositionNumber")
    slice_count = int(_safe_float(frame_content[0].InStackPositionNumber, default=0) or 0)
    if slice_count <= 0:
        raise ValueError("invalid Philips enhanced slice count")

    rows = pixel.shape[1]
    cols = pixel.shape[2]
    label = entry.get("component_label")

    if entry.get("is_magnitude", False):
        pv = getattr(first_fg, "PixelValueTransformationSequence", None)
        slope = _safe_float(getattr(pv[0], "RescaleSlope", None), default=1.0) if pv else 1.0
        intercept = _safe_float(getattr(pv[0], "RescaleIntercept", None), default=0.0) if pv else 0.0
        data = pixel.reshape(2, slice_count, -1, rows, cols)[0]
        data = data * slope + intercept
        return np.asarray(data, dtype=np.float32), None

    pv = getattr(last_fg, "PixelValueTransformationSequence", None)
    slope = _safe_float(getattr(pv[0], "RescaleSlope", None), default=1.0) if pv else 1.0
    intercept = _safe_float(getattr(pv[0], "RescaleIntercept", None), default=0.0) if pv else 0.0
    venc = _extract_multiframe_venc(ds)
    data = pixel.reshape(3, slice_count, -1, rows, cols)[-1]
    data = data * slope + intercept
    return np.asarray(data, dtype=np.float32), venc


def _load_group2_case(
    case,
    entries,
    correction_config=None,
    progress_callback=None,
    parameter_overrides=None,
    dicom_read_workers=1,
):
    pydicom = _import_pydicom()
    mag_data = None
    flow_map = {}
    venc_map = {}
    rr_values = []
    axis0 = axis1 = axis2 = None
    row_spacing = col_spacing = thickness = None

    def _read_entry(entry):
        ds = pydicom.dcmread(entry["path"], force=True)
        data_sytx, venc = _convert_group2_entry(ds, entry)
        rr = _extract_rr_ms(ds)
        a0, a1, a2 = _extract_row_col_slice_dirs(ds)
        sp0, sp1, thick = _extract_pixel_measures(ds)
        return {
            "rr": rr,
            "axis_dirs": (a0, a1, a2),
            "spacing": (sp0, sp1, thick),
            "is_magnitude": bool(entry.get("is_magnitude", False)),
            "label": entry.get("component_label"),
            "data": np.transpose(data_sytx, (2, 3, 0, 1)),
            "venc": venc,
        }

    loaded_entries, actual_workers = _map_dicom_entries(
        entries,
        _read_entry,
        progress_callback=progress_callback,
        dicom_read_workers=dicom_read_workers,
    )
    for record in loaded_entries:
        if record["rr"] is not None:
            rr_values.append(record["rr"])
        a0, a1, a2 = record["axis_dirs"]
        if a2 is not None:
            axis0, axis1, axis2 = a0, a1, a2
        sp0, sp1, thick = record["spacing"]
        row_spacing = sp0 if sp0 is not None else row_spacing
        col_spacing = sp1 if sp1 is not None else col_spacing
        thickness = thick if thick is not None else thickness
        if record["is_magnitude"]:
            mag_data = record["data"]
            continue
        label = record["label"]
        if label:
            flow_map[label] = record["data"]
            if record["venc"] is not None:
                venc_map[label] = record["venc"]

    if mag_data is None or len(flow_map) < 3:
        raise ValueError("incomplete Philips enhanced DICOM case")
    if axis2 is None:
        raise ValueError("ImageOrientationPatient is required for direct DICOM loading")

    component_labels = _component_label_order(list(flow_map.keys()))
    flow = np.stack([flow_map[label] for label in component_labels], axis=-1)
    return _finalize_loaded_dicom_case(
        case,
        entries,
        mag_data,
        flow,
        component_labels,
        (axis0, axis1, axis2),
        (
            1.0 if row_spacing is None else float(row_spacing),
            1.0 if col_spacing is None else float(col_spacing),
            1.0 if thickness is None else float(thickness),
        ),
        venc_map,
        rr_values[0] if rr_values else None,
        correction_config=correction_config,
        progress_callback=progress_callback,
        parameter_overrides=parameter_overrides,
        dicom_read_workers=actual_workers,
    )


def load_dicom_case(
    case,
    correction_config=None,
    progress_callback=None,
    parameter_overrides=None,
    dicom_read_workers=1,
):
    if not isinstance(case, InputCase):
        case = resolve_input_case(case)
    if case.input_kind != "dicom":
        raise ValueError(f"not a DICOM case: {case.input_kind}")

    entries = list(case.metadata.get("dicom_entries", []))
    if not entries:
        resolved = resolve_input_case(case.input_path)
        entries = list(resolved.metadata.get("dicom_entries", []))
        if case.source_group:
            entries = [entry for entry in entries if entry.get("case_id") == case.source_group]
    if not entries:
        raise ValueError(f"empty DICOM case: {case.input_path}")

    group_kind = int(case.metadata.get("group_kind", 0))
    if group_kind == 0:
        return _load_group0_case(
            case,
            entries,
            correction_config=correction_config,
            progress_callback=progress_callback,
            parameter_overrides=parameter_overrides,
            dicom_read_workers=dicom_read_workers,
        )
    if group_kind == 1:
        return _load_group1_case(
            case,
            entries,
            correction_config=correction_config,
            progress_callback=progress_callback,
            parameter_overrides=parameter_overrides,
            dicom_read_workers=dicom_read_workers,
        )
    if group_kind == 2:
        return _load_group2_case(
            case,
            entries,
            correction_config=correction_config,
            progress_callback=progress_callback,
            parameter_overrides=parameter_overrides,
            dicom_read_workers=dicom_read_workers,
        )
    raise ValueError(f"unsupported DICOM group kind: {group_kind}")


def load_input_data(
    input_source,
    correction_config=None,
    progress_callback=None,
    parameter_overrides=None,
    dicom_read_workers=1,
    force_recompute_seg=False,
    ignore_embedded_segmentation=False,
):
    case = resolve_input_case(input_source)
    if case.input_kind == "h5":
        return load_h5_data(
            case.input_path,
            correction_config=correction_config,
            progress_callback=progress_callback,
            source_group=case.source_group,
            force_recompute_seg=force_recompute_seg,
            ignore_embedded_segmentation=ignore_embedded_segmentation,
        )
    loaded = load_dicom_case(
        case,
        correction_config=correction_config,
        progress_callback=progress_callback,
        parameter_overrides=parameter_overrides,
        dicom_read_workers=dicom_read_workers,
    )
    if ignore_embedded_segmentation and loaded.segmentation is not None:
        # DICOM loaders may expose an imported/embedded mask in future. Keep
        # the bypass semantics consistent with H5 without touching the source.
        loaded.segmentation = None
        loaded.capabilities.has_segmentation = False
        loaded.metadata = dict(loaded.metadata or {})
        loaded.metadata["ignore_embedded_segmentation"] = True
    return loaded
