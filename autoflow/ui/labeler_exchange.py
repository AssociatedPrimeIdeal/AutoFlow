"""Prepare the persistent Labeler files without changing their voxel values."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import time
import os
import tempfile
from ..task_control import check_cancelled, current_cancellation_token, task_scope

import numpy as np

from ..algorithms.segmentation import (
    compute_reference_scalar,
    load_segmentation_file,
    save_nifti_volume,
    save_segmentation_file,
    segmentation_timestamp,
)


def _array_digest(array):
    values = np.asarray(array)
    digest = hashlib.sha256()
    digest.update(str((values.shape, values.dtype.str)).encode("ascii"))
    # Bound extra memory for noncontiguous flow components and large cases.
    for slab in values:
        check_cancelled()
        digest.update(np.ascontiguousarray(slab).view(np.uint8))
    return digest.hexdigest()


def export_labeler_exchange(directory, metadata, mag, flow, segmentation, resolution, origin,
                           progress_callback=None):
    """Reuse unchanged images and retain saved edits until the active seed changes.

    The caller runs this in a worker thread; the callback receives completed
    filenames and must not directly access Qt widgets.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    feature_names = ("mag", "flow_x", "flow_y", "flow_z", "pcmra")
    feature_paths = [directory / f"{name}.nii" for name in feature_names]
    label_path = directory / "segmentation.nii"
    manifest_path = directory / "exchange.json"
    metadata = dict(
        metadata, schema_version=3,
        resolution=[float(value) for value in resolution],
        origin=[float(value) for value in origin],
        mag_shape=list(mag.shape), flow_shape=list(flow.shape),
        segmentation_shape=list(segmentation.shape),
    )
    metadata["image_digest"] = [_array_digest(mag), _array_digest(flow)]
    seed_digest = _array_digest(segmentation)
    try:
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        saved = {}
    saved_seed = saved.pop("seed_digest", None) if isinstance(saved, dict) else None
    reuse_features = saved == metadata and all(path.is_file() for path in feature_paths)
    same_label_context = isinstance(saved, dict) and all(
        saved.get(key) == metadata.get(key)
        for key in ("source_id", "resolution", "origin", "segmentation_shape")
    )
    reuse_labels = same_label_context and label_path.is_file() and saved_seed == seed_digest
    if same_label_context and label_path.is_file() and not reuse_labels:
        # Applying a Labeler edit changes the active seed. Keep its label
        # definitions and file when it already contains that exact new seed.
        try:
            current, _ = load_segmentation_file(
                label_path, spatial_shape=segmentation.shape[:3], time_count=segmentation.shape[3]
            )
            reuse_labels = np.array_equal(current, segmentation)
        except (OSError, ValueError):
            reuse_labels = False
    jobs = []
    if not reuse_features:
        pcmra = np.repeat(
            compute_reference_scalar(mag, flow, "pcmra")[..., None], segmentation.shape[3], axis=3
        )
        features = (mag, flow[..., 0], flow[..., 1], flow[..., 2], pcmra)
        for path, volume in zip(feature_paths, features):
            jobs.append((path, volume, False))
    if not reuse_labels:
        jobs.append((label_path, segmentation, True))
    previous_label_path = None
    if not reuse_labels and label_path.is_file():
        # Keep earlier manual work recoverable during a new-seed export or
        # migration from a manifest that did not record the original seed.
        previous_label_path = directory / f"segmentation.previous.{time.time_ns()}.nii"
        shutil.copy2(label_path, previous_label_path)

    token = current_cancellation_token()
    def write(job):
        path, volume, is_label = job
        with task_scope(token), tempfile.TemporaryDirectory(prefix=".autoflow_labeler_", dir=directory) as temp_dir:
            temporary = Path(temp_dir) / path.name
            if is_label:
                save_segmentation_file(
                    temporary, volume, resolution=resolution, origin=origin,
                    provenance={"source": "autoflow_spatiotemporal_labeler_exchange",
                                "created_at": segmentation_timestamp()},
                )
            else:
                save_nifti_volume(temporary, volume, resolution=resolution, origin=origin)
            check_cancelled()
            os.replace(temporary, path)
        return path.name

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="autoflow-labeler-nifti") as pool:
        for future in as_completed([pool.submit(write, job) for job in jobs]):
            name = future.result()
            if progress_callback is not None:
                progress_callback(name)
    check_cancelled()
    manifest_path.write_text(
        json.dumps(dict(metadata, seed_digest=seed_digest), indent=2, sort_keys=True), encoding="utf-8"
    )
    return {
        "feature_paths": [str(path) for path in feature_paths],
        "label_path": str(label_path),
        "reused_features": reuse_features,
        "reused_labels": reuse_labels,
        "written_files": len(jobs),
        "previous_label_path": str(previous_label_path) if previous_label_path else None,
    }
