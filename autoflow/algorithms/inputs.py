"""Resolve H5 inputs and convert DICOM directories through Dicom2H5."""

import os
from pathlib import Path

from ..case_types import InputCase, normalize_dicom_backend
from .data import discover_h5_input_cases, load_h5_data
from .dicom_conversion import converted_input_cases


_H5_SUFFIXES = {".h5", ".hdf5"}


def _is_h5_path(path):
    return Path(path).suffix.lower() in _H5_SUFFIXES


def _directory_input_cases(path, dicom_h5_dir):
    """Treat top-level H5 files as a batch; otherwise convert the directory."""
    h5_paths = sorted(
        entry for entry in Path(path).iterdir()
        if entry.is_file() and _is_h5_path(entry)
    )
    if h5_paths:
        return [case for entry in h5_paths for case in discover_h5_input_cases(entry)]
    return converted_input_cases(path, dicom_h5_dir or "./results/_dicom_h5")


def resolve_input_case(input_source, dicom_backend="dicom2h5", dicom_h5_dir=""):
    """Resolve one H5 case, preserving converted files when group selection fails."""
    normalize_dicom_backend(dicom_backend)
    if isinstance(input_source, InputCase):
        if input_source.input_kind == "h5":
            return input_source
        if input_source.input_kind != "dicom":
            raise ValueError(f"unsupported input kind: {input_source.input_kind}")
        input_source = input_source.input_path

    path = os.path.abspath(os.fspath(input_source))
    if _is_h5_path(path):
        cases = discover_h5_input_cases(path)
    elif os.path.isdir(path):
        cases = _directory_input_cases(path, dicom_h5_dir)
    else:
        raise ValueError(f"input must be an H5 file or a DICOM/H5 directory: {path}")
    if not cases:
        raise ValueError(f"no supported H5 cases found: {path}")
    if len(cases) != 1:
        labels = ", ".join(case.display_name for case in cases[:3])
        raise ValueError(
            f"multiple H5 cases found in {path}: {labels}. "
            "Select a saved H5 group with InputCase, open the H5 in the GUI, or use run_batch()."
        )
    return cases[0]


def collect_input_cases(inputs, dicom_backend="dicom2h5", dicom_h5_dir=""):
    """Discover H5 groups or convert each distinct DICOM directory once."""
    normalize_dicom_backend(dicom_backend)
    discovered = []
    seen_cases = set()
    seen_paths = set()

    def append(case):
        key = (os.path.realpath(case.input_path), str(case.source_group or ""))
        if key not in seen_cases:
            seen_cases.add(key)
            discovered.append(case)

    for item in inputs:
        if isinstance(item, InputCase) and item.input_kind == "h5":
            append(item)
            continue
        if isinstance(item, InputCase):
            if item.input_kind != "dicom":
                raise ValueError(f"unsupported input kind: {item.input_kind}")
            item = item.input_path
        path = os.path.abspath(os.fspath(item))
        path_key = os.path.realpath(path)
        if path_key in seen_paths:
            continue
        seen_paths.add(path_key)
        if _is_h5_path(path):
            cases = discover_h5_input_cases(path)
        elif os.path.isdir(path):
            cases = _directory_input_cases(path, dicom_h5_dir)
        else:
            raise ValueError(f"input must be an H5 file or a DICOM/H5 directory: {path}")
        for case in cases:
            append(case)
    return discovered


def load_input_data(
    input_source,
    correction_config=None,
    progress_callback=None,
    force_recompute_seg=False,
    ignore_embedded_segmentation=False,
    dual_venc_mode="dv",
    dicom_backend="dicom2h5",
    dicom_h5_dir="",
):
    """Load normalized H5 data; DICOM decoding belongs to Dicom2H5."""
    case = resolve_input_case(input_source, dicom_backend, dicom_h5_dir)
    loaded = load_h5_data(
        case.input_path,
        correction_config=correction_config,
        progress_callback=progress_callback,
        source_group=case.source_group,
        force_recompute_seg=force_recompute_seg,
        ignore_embedded_segmentation=ignore_embedded_segmentation,
        dual_venc_mode=dual_venc_mode,
    )
    if case.metadata.get("dicom_backend"):
        loaded.metadata.update({
            key: case.metadata[key]
            for key in ("dicom_backend", "dicom_source_directory", "converted_h5")
            if key in case.metadata
        })
    return loaded
