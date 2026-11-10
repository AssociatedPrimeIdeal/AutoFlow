"""Prepare selected persistent Labeler images and preserve edited working masks."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import time
import os
import tempfile

import numpy as np

from ..task_control import check_cancelled, current_cancellation_token, task_scope
from ..algorithms.segmentation import (
    compute_reference_scalar, load_segmentation_file, save_segmentation_file,
    segmentation_timestamp,
)


LABELER_FEATURES = ("mag", "flow_x", "flow_y", "flow_z", "pcmra")


def labeler_features(additional=()):
    """PC-MRA is always available; only explicitly requested extras are cached."""
    if not isinstance(additional, (list, tuple)) or any(key not in LABELER_FEATURES for key in additional):
        raise ValueError("Labeler images must be mag, flow_x, flow_y, flow_z or pcmra")
    return [name for name in LABELER_FEATURES if name == "pcmra" or name in additional]


def labeler_segmentation_grid(workspace):
    """Use the acquired mask grid; retain an explicitly edited 1-mm source.

    A topology/model preview never justifies upsampling an existing native
    mask for manual review. Already committed fine-grid manual edits and live
    working labels keep their original geometry rather than being downsampled.
    """
    state = workspace.segmentation
    working = (state.working_labels_4d if state.working_labels_4d is not None
               else state.working_labels_3d)
    provenance = workspace.get_active_segmentation_provenance()
    fine = working is not None or provenance.get("geometry_source") == "imported_or_edited_1mm"
    if working is not None:
        labels = working
    elif fine:
        labels = workspace.segmentation_geometry_data(display=True)
    else:
        labels = workspace.get_active_segmentation()
    if labels is None:
        return None, np.asarray(workspace.resolution, dtype=float), "native"
    labels = np.asarray(labels, dtype=np.int16)
    if labels.ndim == 3:
        labels = np.broadcast_to(labels[..., None], labels.shape+(max(1, workspace.time_count()),))
    return labels, (np.ones(3) if fine else np.asarray(workspace.resolution, dtype=float)), ("shared_1mm" if fine else "native")


def _array_digest(array):
    values = np.asarray(array)
    digest = hashlib.sha256()
    digest.update(str((values.shape, values.dtype.str)).encode("ascii"))
    for slab in values:
        check_cancelled()
        digest.update(np.ascontiguousarray(slab).view(np.uint8))
    return digest.hexdigest()


def _save_nrrd_sequence(path, volume, resolution, origin):
    """Write float32 XYZT, raw/little-endian, without copying a full 4D field."""
    spacing = np.asarray(resolution, dtype=float).reshape(3)
    translation = np.asarray(origin, dtype=float).reshape(3)
    vector = lambda values: "(" + ",".join(f"{value:.17g}" for value in values) + ")"
    directions = " ".join(vector(row) for row in np.diag(spacing)) + " none"
    header = "\n".join((
        "NRRD0005", "type: float", "dimension: 4", "space: right-anterior-superior",
        "sizes: " + " ".join(str(value) for value in volume.shape),
        "space directions: " + directions, "kinds: domain domain domain list",
        "endian: little", "encoding: raw", "space origin: " + vector(translation), "", "",
    ))
    with open(path, "wb") as stream:
        stream.write(header.encode("ascii"))
        for frame in range(volume.shape[3]):
            check_cancelled()
            stream.write(np.asarray(volume[..., frame], dtype="<f4").tobytes(order="F"))


def export_labeler_exchange(directory, metadata, mag, flow, segmentation, resolution, origin,
                           progress_callback=None, *, features=None):
    """Cache PC-MRA plus selected extras; reuse each unchanged image separately.

    The active segmentation is always included. Saved Labeler work survives
    changes to the image checklist and is backed up before a seed replacement.
    The caller runs this function and its progress callback outside the GUI thread.
    """
    directory = Path(directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    names = labeler_features(features if features is not None else ())
    feature_paths = [directory / f"{name}.seq.nrrd" for name in names]
    label_path = directory / "segmentation.nii"
    manifest_path = directory / "exchange.json"
    metadata = dict(metadata)
    metadata.pop("features", None)
    metadata.update(schema_version=4,
                    resolution=[float(value) for value in resolution],
                    origin=[float(value) for value in origin],
                    mag_shape=list(mag.shape), flow_shape=list(flow.shape),
                    segmentation_shape=list(segmentation.shape))
    # PC-MRA depends on both inputs; component images only on their own channel.
    digests = {"pcmra": [_array_digest(mag), _array_digest(flow)]}
    for name in names:
        if name != "pcmra":
            array = mag if name == "mag" else flow[..., ("flow_x", "flow_y", "flow_z").index(name)]
            digests[name] = _array_digest(array)
    seed_digest = _array_digest(segmentation)
    try:
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        saved = {}
    if not isinstance(saved, dict):
        saved = {}
    same_image_context = all(saved.get(key) == value for key, value in metadata.items())
    saved_digests = saved.get("image_digests", {})
    if not isinstance(saved_digests, dict):
        saved_digests = {}
    same_label_context = all(saved.get(key) == metadata.get(key)
                             for key in ("source_id", "resolution", "origin", "segmentation_shape"))
    reuse_labels = same_label_context and label_path.is_file() and saved.get("seed_digest") == seed_digest
    if same_label_context and label_path.is_file() and not reuse_labels:
        try:
            current, _ = load_segmentation_file(
                label_path, spatial_shape=segmentation.shape[:3], time_count=segmentation.shape[3])
            reuse_labels = np.array_equal(current, segmentation)
        except (OSError, ValueError):
            reuse_labels = False

    jobs = []
    for name, path in zip(names, feature_paths):
        if same_image_context and saved_digests.get(name) == digests[name] and path.is_file():
            continue
        if name == "pcmra":
            reference = compute_reference_scalar(mag, flow, "pcmra")
            volume = np.broadcast_to(reference[..., None], segmentation.shape)
        else:
            volume = mag if name == "mag" else flow[..., ("flow_x", "flow_y", "flow_z").index(name)]
        jobs.append((path, volume, False))
    reuse_features = not jobs
    if not reuse_labels:
        jobs.append((label_path, segmentation, True))
    previous_label_path = None
    if not reuse_labels and label_path.is_file():
        previous_label_path = directory / f"segmentation.previous.{time.time_ns()}.nii"
        shutil.copy2(label_path, previous_label_path)

    token = current_cancellation_token()
    def write(job):
        path, volume, is_label = job
        with task_scope(token), tempfile.TemporaryDirectory(prefix=".autoflow_labeler_", dir=directory) as temp_dir:
            temporary = Path(temp_dir) / path.name
            if is_label:
                save_segmentation_file(temporary, volume, resolution=resolution, origin=origin,
                                       provenance={"source": "autoflow_spatiotemporal_labeler_exchange",
                                                   "created_at": segmentation_timestamp()})
            else:
                _save_nrrd_sequence(temporary, volume, resolution, origin)
            check_cancelled()
            os.replace(temporary, path)
        return path.name

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="autoflow-labeler-export") as pool:
        for future in as_completed([pool.submit(write, job) for job in jobs]):
            name = future.result()
            if progress_callback is not None:
                progress_callback(name)
    check_cancelled()
    manifest_path.write_text(json.dumps(dict(metadata, features=names, image_digests=digests,
                                             seed_digest=seed_digest), indent=2, sort_keys=True), encoding="utf-8")
    # Only managed, regenerable image caches are removed. Saved masks and their
    # Labeler metadata/history are never removed by a checklist change.
    for name in LABELER_FEATURES:
        for suffix in (".nii", ".seq.nrrd"):
            obsolete = directory / f"{name}{suffix}"
            if obsolete not in feature_paths:
                obsolete.unlink(missing_ok=True)
    return {"feature_paths": [str(path) for path in feature_paths], "label_path": str(label_path),
            "reused_features": reuse_features, "reused_labels": reuse_labels,
            "written_files": len(jobs),
            "previous_label_path": str(previous_label_path) if previous_label_path else None}
