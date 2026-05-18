import numpy as np
import pytest

import autoflow.core.pipeline as pipeline_module
from autoflow.algorithms import (
    broadcast_segmentation_to_time,
    generate_threshold_segmentation,
    normalize_loaded_case,
)
from autoflow.core.models import LoaderCapabilities, SegmentationState, Workspace
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

    seg, provenance, scalar, threshold_info = generate_threshold_segmentation(
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
    assert provenance["threshold_mode"] == "manual_absolute"
    assert provenance["threshold_value_min"] == pytest.approx(float(threshold_value))
    assert threshold_info["mode"] == "manual_absolute"
    assert threshold_info["min_value"] == pytest.approx(float(threshold_value))


def test_generate_threshold_segmentation_supports_manual_percent_range():
    mag = np.array(
        [
            [[[0.0], [0.25]], [[0.5], [0.75]]],
            [[[1.0], [0.0]], [[0.0], [0.0]]],
        ],
        dtype=np.float32,
    )
    flow = np.zeros((2, 2, 2, 1, 3), dtype=np.float32)

    seg, provenance, scalar, threshold_info = generate_threshold_segmentation(
        mag=mag,
        flow=flow,
        resolution=(1.0, 1.0, 1.0),
        time_count=1,
        scalar_name="mag",
        threshold={"mode": "manual", "min_percent": 25.0, "max_percent": 75.0},
        keep_largest_cc=False,
        min_component_volume_mm3=0.0,
        closing=False,
        opening=False,
    )

    expected = np.array(
        [
            [[0, 1], [1, 1]],
            [[0, 0], [0, 0]],
        ],
        dtype=np.int16,
    )

    assert scalar.shape == (2, 2, 2)
    assert seg.shape == (2, 2, 2, 1)
    assert np.array_equal(seg[..., 0], expected)
    assert provenance["threshold_mode"] == "manual_percent"
    assert provenance["threshold_percent_min"] == pytest.approx(25.0)
    assert provenance["threshold_percent_max"] == pytest.approx(75.0)
    assert provenance["threshold_value_min"] == pytest.approx(0.25)
    assert provenance["threshold_value_max"] == pytest.approx(0.75)
    assert threshold_info["mode"] == "manual_percent"
    assert threshold_info["min_value"] == pytest.approx(0.25)
    assert threshold_info["max_value"] == pytest.approx(0.75)


def test_segmentation_state_defaults_to_manual_threshold_percent_range_and_discards_fill_tool():
    seg = SegmentationState.from_dict({"tool": "fill"})

    assert seg.tool == "brush"
    assert seg.editing_enabled is False
    assert seg.threshold_value == {"mode": "manual", "min_percent": 10.0, "max_percent": 100.0}


def test_segmentation_state_round_trips_editing_enabled():
    seg = SegmentationState.from_dict({"editing_enabled": True})

    payload = seg.to_dict()

    assert payload["editing_enabled"] is True
    assert SegmentationState.from_dict(payload).editing_enabled is True
