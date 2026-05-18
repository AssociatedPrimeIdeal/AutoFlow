import numpy as np
import pytest

import autoflow.core.pipeline as pipeline_module
from autoflow.algorithms import (
    broadcast_segmentation_to_time,
    generate_threshold_segmentation,
    normalize_loaded_case,
)
from autoflow.core.models import LoaderCapabilities, Workspace
from autoflow.core.pipeline import PipelineEngine


def test_pipeline_load_data_initializes_original_segmentation_source(monkeypatch):
    case = normalize_loaded_case(
        flow=np.zeros((2, 2, 2, 2, 3), dtype=np.float32),
        mag=np.ones((2, 2, 2, 2), dtype=np.float32),
        segmentation=np.ones((2, 2, 2), dtype=np.int16),
        resolution=[1.0, 1.0, 1.0],
        origin=[0.0, 0.0, 0.0],
        venc=[120.0, 120.0, 120.0],
        rr=1000.0,
        source_format="normalized_h5",
        capabilities=LoaderCapabilities(
            has_segmentation=True,
            has_tke=False,
            has_complex_source=False,
            supports_wss=True,
            supports_plane_metrics=True,
        ),
    )
    monkeypatch.setattr(pipeline_module, "load_input_data", lambda path: case)

    ws = Workspace()
    ws.paths.flow_path = "dummy.h5"
    PipelineEngine().load_data(ws, lambda msg: None)

    assert ws.segmentation.active_source == "original"
    assert ws.get_segmentation_source("original").shape == (2, 2, 2, 2)
    assert np.array_equal(ws.segmask_raw, ws.get_segmentation_source("original"))


def test_workspace_source_switch_preserves_original_segmentation():
    ws = Workspace()
    original = np.ones((2, 2, 2, 3), dtype=np.int16)
    imported = np.full((2, 2, 2, 3), 2, dtype=np.int16)

    ws.set_segmentation_source("original", original, provenance={"source": "original"})
    ws.activate_segmentation_source("original")
    ws.set_segmentation_source("imported", imported, provenance={"source": "import"})
    ws.activate_segmentation_source("imported")

    assert ws.segmentation.active_source == "imported"
    assert np.array_equal(ws.get_segmentation_source("original"), original)
    assert np.array_equal(ws.segmask_raw, imported)


def test_broadcast_segmentation_to_time_repeats_working_labels_over_time():
    labels_3d = np.array(
        [
            [[0, 1], [2, 0]],
            [[3, 0], [0, 4]],
        ],
        dtype=np.int16,
    )
    seg4d = broadcast_segmentation_to_time(labels_3d, 3)

    assert seg4d.shape == (2, 2, 2, 3)
    assert np.array_equal(seg4d[..., 0], labels_3d)
    assert np.array_equal(seg4d[..., 1], labels_3d)
    assert np.array_equal(seg4d[..., 2], labels_3d)


def test_workspace_working_labels_4d_do_not_broadcast_across_time():
    ws = Workspace()
    active = np.zeros((2, 2, 2, 3), dtype=np.int16)
    ws.set_segmentation_source("original", active, provenance={"source": "original"})
    ws.activate_segmentation_source("original")

    working = active.copy()
    working[0, 0, 0, 1] = 5
    ws.segmentation.working_labels_4d = working

    display = ws.segmentation_display_4d()

    assert display.shape == (2, 2, 2, 3)
    assert int(display[0, 0, 0, 0]) == 0
    assert int(display[0, 0, 0, 1]) == 5
    assert int(display[0, 0, 0, 2]) == 0


@pytest.mark.parametrize(
    ("scalar_name", "threshold_value"),
    [("mag", 0.5), ("pcmra", 0.0), ("pcmra_std", -1.0)],
)
def test_generate_threshold_segmentation_reports_shape_and_provenance(scalar_name, threshold_value):
    mag = np.ones((2, 2, 2, 4), dtype=np.float32)
    flow = np.ones((2, 2, 2, 4, 3), dtype=np.float32)

    seg, provenance, scalar, _resolved_threshold = generate_threshold_segmentation(
        mag=mag,
        flow=flow,
        resolution=(1.0, 1.0, 1.0),
        time_count=4,
        scalar_name=scalar_name,
        threshold=threshold_value,
        keep_largest_cc=False,
        min_component_volume_mm3=0.0,
        closing=False,
        opening=False,
    )

    assert scalar.shape == (2, 2, 2)
    assert seg.shape == (2, 2, 2, 4)
    assert provenance["source"] == "threshold"
    assert provenance["scalar"] == scalar_name
