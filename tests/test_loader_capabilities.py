from pathlib import Path
import threading
import time

import h5py
import numpy as np
import pytest

import autoflow.algorithms.dicom as dicom_module
import autoflow.algorithms.metrics as metrics_module
import autoflow.core.pipeline as pipeline_module
from autoflow.algorithms import (
    apply_background_phase_correction_to_mag_flow,
    load_h5_data,
    normalize_loaded_case,
)
from autoflow.case_types import InputCase
from autoflow.core.models import LoaderCapabilities, StepId, Workspace
from autoflow.core.pipeline import PipelineEngine


def _write_legacy_complex_h5(path: Path):
    img_complex = np.ones((2, 2, 2, 1, 4), dtype=np.complex64)
    img_complex[..., 0] = 1.0 + 0.0j
    for idx, phase in enumerate((0.1, 0.2, 0.3), start=1):
        img_complex[..., idx] = 0.5 * np.exp(1j * phase)
    segmask = np.ones((2, 2, 2), dtype=np.int16)

    with h5py.File(path, "w") as f:
        f["img_complex"] = img_complex
        f["segmask"] = segmask
        f["VENC"] = np.array([150.0, 160.0, 170.0], dtype=np.float32)
        f["Resolution"] = np.array([1.1, 1.2, 1.3], dtype=np.float32)
        f["RR"] = 920.0


def _write_normalized_h5(path: Path):
    flow = np.zeros((2, 2, 2, 3), dtype=np.float32)
    flow[..., 0] = 15.0
    mag = np.ones((2, 2, 2), dtype=np.float32)

    with h5py.File(path, "w") as f:
        f["flow"] = flow
        f["mag"] = mag
        f["VENC"] = np.array([120.0, 130.0, 140.0], dtype=np.float32)
        f["Resolution"] = np.array([1.0, 1.5, 2.0], dtype=np.float32)
        f["RR"] = 1000.0


def test_load_h5_data_legacy_complex_is_normalized_but_tke_is_lazy(tmp_path):
    path = tmp_path / "legacy_complex.h5"
    _write_legacy_complex_h5(path)

    case = load_h5_data(str(path))

    assert case.source_format == "legacy_h5"
    assert case.flow.shape == (2, 2, 2, 1, 3)
    assert case.mag.shape == (2, 2, 2, 1)
    assert case.segmentation.shape == (2, 2, 2, 1)
    assert case.sigma.shape == (2, 2, 2, 1, 3)
    assert case.tke_array is None
    assert case.metadata["background_phase_correction"]["enabled"] is False
    assert case.metadata["background_phase_correction"]["applied"] is False
    assert "stationary_voxels" not in case.metadata["background_phase_correction"]
    assert case.capabilities.to_dict() == {
        "has_segmentation": True,
        "has_tke": True,
        "has_complex_source": True,
        "supports_wss": True,
        "supports_plane_metrics": True,
    }


def test_load_h5_data_normalized_flow_mag_only_keeps_optional_fields_empty(tmp_path):
    path = tmp_path / "normalized_flow_only.h5"
    _write_normalized_h5(path)

    case = load_h5_data(str(path))

    assert case.source_format == "normalized_h5"
    assert case.flow.shape == (2, 2, 2, 1, 3)
    assert case.mag.shape == (2, 2, 2, 1)
    assert case.segmentation is None
    assert case.sigma is None
    assert case.tke_array is None
    assert case.metadata["background_phase_correction"]["enabled"] is False
    assert case.metadata["background_phase_correction"]["applied"] is False
    assert "stationary_voxels" not in case.metadata["background_phase_correction"]
    assert case.capabilities.to_dict() == {
        "has_segmentation": False,
        "has_tke": False,
        "has_complex_source": False,
        "supports_wss": True,
        "supports_plane_metrics": True,
    }


def test_load_h5_data_legacy_complex_does_not_crop_to_segmentation_bbox(tmp_path):
    path = tmp_path / "legacy_complex_uncropped.h5"
    img_complex = np.ones((6, 7, 8, 1, 4), dtype=np.complex64)
    img_complex[..., 0] = 1.0 + 0.0j
    for idx, phase in enumerate((0.1, 0.2, 0.3), start=1):
        img_complex[..., idx] = 0.5 * np.exp(1j * phase)
    segmask = np.zeros((6, 7, 8), dtype=np.int16)
    segmask[2:4, 3:5, 1:3] = 1

    with h5py.File(path, "w") as f:
        f["img_complex"] = img_complex
        f["segmask"] = segmask
        f["VENC"] = np.array([150.0, 150.0, 150.0], dtype=np.float32)
        f["Resolution"] = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        f["SpatialOrder"] = np.array(["LR", "AP", "FH"], dtype="S2")
        f["VENCOrder"] = np.array(["LR", "AP", "FH"], dtype="S2")
        f["RR"] = 1000.0

    case = load_h5_data(str(path))

    assert case.flow.shape == (6, 7, 8, 1, 3)
    assert case.mag.shape == (6, 7, 8, 1)
    assert case.segmentation.shape == (6, 7, 8, 1)


def test_background_phase_correction_on_mag_flow_recovers_static_background():
    shape = (8, 8, 8, 2)
    mag = np.zeros(shape, dtype=np.float32)
    mag[1:7, 1:7, 1:7, :] = 1.0
    flow = np.zeros(shape + (3,), dtype=np.float32)
    flow[3:5, 3:5, 3:5, :, 0] = 30.0
    flow[3:5, 3:5, 3:5, :, 1] = -20.0
    flow[3:5, 3:5, 3:5, :, 2] = 10.0
    venc = np.array([150.0, 150.0, 150.0], dtype=np.float32)
    offset = np.array([0.2, -0.15, 0.1], dtype=np.float32)
    flow_with_offset = flow + (offset / np.pi) * venc.reshape((1, 1, 1, 1, 3))

    corrected_flow, stationary_mask, report = apply_background_phase_correction_to_mag_flow(
        mag,
        flow_with_offset,
        venc,
    )

    static_region = stationary_mask & (mag[..., 0] > 0.5)
    assert report["applied"] is True
    assert stationary_mask is not None
    assert int(np.sum(static_region)) > 0
    assert float(np.max(np.abs(corrected_flow[static_region, :]))) < 3.0
    assert np.allclose(corrected_flow[4, 4, 4, 0, :], flow[4, 4, 4, 0, :], atol=8.0)


def test_pipeline_load_data_keeps_tke_sources_out_of_derived_state(monkeypatch):
    sigma = np.ones((2, 2, 2, 1, 3), dtype=np.float32)
    case = normalize_loaded_case(
        flow=np.zeros((2, 2, 2, 3), dtype=np.float32),
        mag=np.ones((2, 2, 2), dtype=np.float32),
        segmentation=np.ones((2, 2, 2), dtype=np.int16),
        resolution=[1.0, 1.0, 1.0],
        origin=[0.0, 0.0, 0.0],
        venc=[150.0, 150.0, 150.0],
        rr=1000.0,
        sigma=sigma,
        tke_array=None,
        source_format="legacy_h5",
        capabilities=LoaderCapabilities(
            has_segmentation=True,
            has_tke=True,
            has_complex_source=True,
            supports_wss=True,
            supports_plane_metrics=True,
        ),
    )
    monkeypatch.setattr(pipeline_module, "load_input_data", lambda path: case)

    ws = Workspace()
    ws.paths.flow_path = "dummy.h5"
    PipelineEngine().load_data(ws, lambda msg: None)

    assert ws.input_state.capabilities.has_tke is True
    assert ws.source_sigma is not None
    assert ws.source_tke_array is None
    assert ws.derived.tke_array is None
    assert ws.derived.tke_volume is None
    assert all(obj.data_key != "tke_volume" for obj in ws.scene_objects.values())


def test_pipeline_load_data_passes_dicom_parameter_overrides(monkeypatch):
    captured = {}
    loaded = normalize_loaded_case(
        flow=np.zeros((2, 2, 2, 1, 3), dtype=np.float32),
        mag=np.ones((2, 2, 2, 1), dtype=np.float32),
        segmentation=None,
        resolution=[1.0, 1.0, 1.0],
        origin=[0.0, 0.0, 0.0],
        venc=[150.0, 150.0, 150.0],
        rr=950.0,
        source_format="dicom",
        capabilities=LoaderCapabilities(
            has_segmentation=False,
            has_tke=False,
            has_complex_source=False,
            supports_wss=True,
            supports_plane_metrics=True,
        ),
    )

    def fake_load_input_data(target, **kwargs):
        captured["target"] = target
        captured["kwargs"] = dict(kwargs)
        return loaded

    monkeypatch.setattr(pipeline_module, "load_input_data", fake_load_input_data)

    case = InputCase(
        input_path="/tmp/dicom_case",
        input_kind="dicom",
        display_name="dicom_case",
        output_name="dicom_case",
        source_group="case-1",
        metadata={},
    )
    ws = Workspace()
    ws.loader_params.dicom_parameter_overrides.resolution = [1.2, 1.3, 1.4]
    ws.loader_params.dicom_parameter_overrides.venc = [210.0, 220.0, 230.0]
    ws.loader_params.dicom_parameter_overrides.spatial_order = ["LR", "AP", "FH"]
    ws.loader_params.dicom_parameter_overrides.venc_order = ["LR", "AP", "FH"]
    ws.loader_params.dicom_parameter_overrides.rr = 875.0
    ws.loader_params.dicom_read_workers = 6

    PipelineEngine().load_data(ws, lambda _msg: None, input_source=case)

    assert captured["target"] is case
    assert captured["kwargs"]["parameter_overrides"] == {
        "resolution": [1.2, 1.3, 1.4],
        "venc": [210.0, 220.0, 230.0],
        "spatial_order": ["LR", "AP", "FH"],
        "venc_order": ["LR", "AP", "FH"],
        "rr": 875.0,
    }
    assert captured["kwargs"]["dicom_read_workers"] == 6


def test_pipeline_skips_segmentation_steps_when_input_has_no_segmentation(monkeypatch):
    case = normalize_loaded_case(
        flow=np.zeros((2, 2, 2, 3), dtype=np.float32),
        mag=np.ones((2, 2, 2), dtype=np.float32),
        segmentation=None,
        resolution=[1.0, 1.0, 1.0],
        origin=[0.0, 0.0, 0.0],
        venc=[150.0, 150.0, 150.0],
        rr=1000.0,
        source_format="normalized_h5",
        capabilities=LoaderCapabilities(
            has_segmentation=False,
            has_tke=False,
            has_complex_source=False,
            supports_wss=True,
            supports_plane_metrics=True,
        ),
    )
    monkeypatch.setattr(pipeline_module, "load_input_data", lambda path: case)

    ws = Workspace()
    ws.paths.flow_path = "dummy.h5"
    engine = PipelineEngine()
    engine.load_data(ws, lambda msg: None)

    result = engine.run_step(ws, StepId.GENERATE_SKELETON, lambda msg: None)

    assert result.skipped is True
    assert "no segmentation" in result.message


def test_pipeline_derived_metrics_does_not_use_sigma_without_complex_source(monkeypatch):
    captured = {}

    def fake_compute_derived_metrics(**kwargs):
        captured["sigma"] = kwargs.get("sigma")
        captured["tke_array"] = kwargs.get("tke_array")
        return {
            "wss_surfaces": [object()],
            "wss_volume": np.ones((2, 2, 2, 1), dtype=np.float32),
            "tke_volume": None,
            "tke_array": None,
            "tke_peak": None,
            "streamlines": [],
            "tube_radius": kwargs["tube_radius"],
            "pixelwise_export": {},
        }

    monkeypatch.setattr(pipeline_module, "compute_derived_metrics", fake_compute_derived_metrics)

    ws = Workspace()
    ws.flow_raw = np.ones((2, 2, 2, 1, 3), dtype=np.float32)
    ws.mag_raw = np.ones((2, 2, 2, 1), dtype=np.float32)
    ws.segmask_raw = np.ones((2, 2, 2, 1), dtype=np.int16)
    ws.resolution = np.array([1.0, 1.0, 1.0], dtype=float)
    ws.origin = np.array([0.0, 0.0, 0.0], dtype=float)
    ws.source_sigma = np.ones((2, 2, 2, 1, 3), dtype=np.float32)
    ws.source_tke_array = None
    ws.input_state.capabilities = LoaderCapabilities(
        has_segmentation=True,
        has_tke=False,
        has_complex_source=False,
        supports_wss=True,
        supports_plane_metrics=True,
    )

    result = PipelineEngine()._step_compute_derived_metrics(ws)

    assert result.success is True
    assert result.skipped is False
    assert captured["sigma"] is None
    assert captured["tke_array"] is None
    assert ws.derived.tke_array is None
    assert any(obj.data_key == "wss_surface_live" for obj in ws.scene_objects.values())
    assert all(obj.data_key != "tke_volume" for obj in ws.scene_objects.values())
    assert "tke=unavailable" in result.message


def test_pipeline_load_message_hides_stationary_mask_details(monkeypatch):
    case = normalize_loaded_case(
        flow=np.zeros((2, 2, 2, 1, 3), dtype=np.float32),
        mag=np.ones((2, 2, 2, 1), dtype=np.float32),
        segmentation=np.ones((2, 2, 2, 1), dtype=np.int16),
        resolution=[1.0, 1.0, 1.0],
        origin=[0.0, 0.0, 0.0],
        venc=[150.0, 150.0, 150.0],
        rr=1000.0,
        metadata={"background_phase_correction": {"enabled": True, "applied": True, "stationary_voxels": 42}},
        source_format="legacy_h5",
        capabilities=LoaderCapabilities(
            has_segmentation=True,
            has_tke=False,
            has_complex_source=False,
            supports_wss=True,
            supports_plane_metrics=True,
        ),
    )
    monkeypatch.setattr(pipeline_module, "load_input_data", lambda path: case)

    messages = []
    ws = Workspace()
    ws.paths.flow_path = "dummy.h5"
    PipelineEngine().load_data(ws, messages.append)

    assert messages
    assert any("bpc=applied" in message for message in messages)
    assert all("stationary" not in message and "stat=" not in message for message in messages)
    assert 99 not in ws.unique_labels()


def test_compute_derived_metrics_allows_wss_only_when_tke_is_absent(monkeypatch):
    def fake_compute_wss_metrics(*args, **kwargs):
        return {
            "wss_surfaces": [object()],
            "wss_volume": np.ones((2, 2, 2, 1), dtype=np.float32),
        }

    def fail_compute_tke_metrics(*args, **kwargs):
        raise AssertionError("compute_tke_metrics should not be called without a TKE source")

    monkeypatch.setattr(metrics_module, "compute_wss_metrics", fake_compute_wss_metrics)
    monkeypatch.setattr(metrics_module, "compute_tke_metrics", fail_compute_tke_metrics)

    result = metrics_module.compute_derived_metrics(
        mask4d=np.ones((2, 2, 2, 1), dtype=bool),
        flow=np.zeros((2, 2, 2, 1, 3), dtype=np.float32),
        spacing=(1.0, 1.0, 1.0),
        save_pixelwise=True,
        tke_array=None,
        sigma=None,
    )

    assert result["tke_array"] is None
    assert result["tke_volume"] is None
    assert result["tke_peak"] is None
    assert set(result["pixelwise_export"]) == {"wss", "spacing", "origin"}


def test_compute_plane_metrics_multithread_handles_empty_planes():
    metrics, qc = metrics_module.compute_plane_metrics_multithread(
        flow_xyzt3=np.zeros((2, 2, 2, 1, 3), dtype=np.float32),
        segmask_binary_4d=np.ones((2, 2, 2, 1), dtype=bool),
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        planes=[],
        return_qc=True,
    )

    assert metrics == []
    assert qc == {"path_ic": {}, "fork_ic": {}, "forks": []}


def test_finalize_loaded_dicom_case_applies_parameter_overrides(monkeypatch):
    captured = {}

    def fake_apply_background_phase_correction_to_mag_flow(mag, flow, venc, config=None, progress_callback=None):
        captured["enabled"] = bool(getattr(config, "enabled", None))
        return np.asarray(flow, dtype=np.float32), np.zeros(flow.shape[:3], dtype=bool), {
            "enabled": True,
            "applied": False,
            "stationary_voxels": 0,
            "skipped_reason": "disabled",
        }

    monkeypatch.setattr(
        dicom_module,
        "apply_background_phase_correction_to_mag_flow",
        fake_apply_background_phase_correction_to_mag_flow,
    )

    case = InputCase(
        input_path="/tmp/dicom_case",
        input_kind="dicom",
        display_name="dicom_case",
        output_name="dicom_case",
        source_group="series-1",
        metadata={
            "manufacturer": "TestVendor",
            "group_kind": 0,
            "series_description": "Flow",
            "protocol_name": "4D Flow",
        },
    )
    loaded = dicom_module._finalize_loaded_dicom_case(
        case=case,
        entries=[{"path": "/tmp/a.dcm"}],
        mag_rczt=np.ones((2, 2, 2, 1), dtype=np.float32),
        flow_rczt3=np.zeros((2, 2, 2, 1, 3), dtype=np.float32),
        component_labels=["RL", "PA", "HF"],
        axis_dirs=(
            np.array([1.0, 0.0, 0.0], dtype=float),
            np.array([0.0, 1.0, 0.0], dtype=float),
            np.array([0.0, 0.0, 1.0], dtype=float),
        ),
        axis_spacings=(2.0, 2.5, 3.0),
        venc_map={"RL": 150.0, "PA": 160.0, "HF": 170.0},
        rr=990.0,
        parameter_overrides={
            "resolution": [1.1, 1.2, 1.3],
            "venc": [210.0, 220.0, 230.0],
            "spatial_order": ["LR", "AP", "FH"],
            "venc_order": ["LR", "AP", "FH"],
            "rr": 875.0,
        },
    )

    assert np.allclose(loaded.resolution, [1.1, 1.2, 1.3])
    assert np.allclose(loaded.venc, [210.0, 220.0, 230.0])
    assert loaded.rr == pytest.approx(875.0)
    assert captured["enabled"] is False
    assert loaded.metadata["spatial_order_raw"] == ["LR", "AP", "FH"]
    assert loaded.metadata["venc_order_raw"] == ["LR", "AP", "FH"]
    assert loaded.metadata["dicom_parameter_overrides"] == {
        "resolution": [1.1, 1.2, 1.3],
        "venc": [210.0, 220.0, 230.0],
        "spatial_order": ["LR", "AP", "FH"],
        "venc_order": ["LR", "AP", "FH"],
        "rr": 875.0,
    }
    assert "stationary_voxels" not in loaded.metadata["background_phase_correction"]


def test_inspect_dicom_case_reports_matrix_size(monkeypatch):
    class FakeDataset:
        def __init__(self, **attrs):
            for key, value in attrs.items():
                setattr(self, key, value)

    datasets = {}
    entries = []
    labels = [None, "RL", "AP", "FH"]
    venc_lookup = {"RL": 150.0, "AP": 160.0, "FH": 170.0}

    for slice_idx in range(2):
        for time_idx in range(2):
            for label in labels:
                is_magnitude = label is None
                tag = "mag" if is_magnitude else label.lower()
                path = f"/tmp/{tag}_z{slice_idx}_t{time_idx}.dcm"
                sequence_name = "Magnitude"
                if label is not None:
                    sequence_name = f"4D FLOW {int(venc_lookup[label])} {label}"
                datasets[path] = FakeDataset(
                    Manufacturer="Siemens",
                    ProtocolName="4D Flow",
                    SeriesDescription=sequence_name,
                    SequenceName=sequence_name,
                    ImageOrientationPatient=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                    ImagePositionPatient=[0.0, 0.0, float(slice_idx)],
                    PixelSpacing=[1.0, 2.0],
                    SliceThickness=3.0,
                    TriggerTime=float(100 * time_idx),
                    CardiacRRIntervalSpecified=900.0,
                    Rows=2,
                    Columns=3,
                )
                entries.append(
                    {
                        "path": path,
                        "case_id": "case-1",
                        "manufacturer": "Siemens",
                        "group_kind": 0,
                        "component_label": label,
                        "is_magnitude": is_magnitude,
                        "series_description": "4D Flow",
                        "protocol_name": "4D Flow",
                    }
                )

    class FakePydicom:
        def dcmread(self, path, force=True, stop_before_pixels=False):
            return datasets[path]

    monkeypatch.setattr(dicom_module, "_import_pydicom", lambda: FakePydicom())

    case = InputCase(
        input_path="/tmp/dicom_case",
        input_kind="dicom",
        display_name="dicom_case",
        output_name="dicom_case",
        source_group="case-1",
        metadata={
            "manufacturer": "Siemens",
            "group_kind": 0,
            "series_description": "4D Flow",
            "protocol_name": "4D Flow",
            "dicom_entries": entries,
        },
    )

    preview = dicom_module.inspect_dicom_case(case)

    assert preview["matrix_size"] == [3, 2, 2, 2]
    assert preview["resolution"] == pytest.approx([2.0, 1.0, 1.0])
    assert preview["venc"] == pytest.approx([150.0, 160.0, 170.0])
    assert preview["rr"] == pytest.approx(900.0)


def test_load_dicom_case_multithread_matches_single_thread(monkeypatch):
    class FakeDataset:
        def __init__(self, pixel_array, **attrs):
            self.pixel_array = pixel_array
            for key, value in attrs.items():
                setattr(self, key, value)

    datasets = {}
    entries = []
    labels = [None, "RL", "AP", "FH"]
    venc_lookup = {"RL": 150.0, "AP": 160.0, "FH": 170.0}
    for slice_idx in range(2):
        for time_idx in range(2):
            for label in labels:
                is_magnitude = label is None
                tag = "mag" if is_magnitude else label.lower()
                path = f"/tmp/{tag}_z{slice_idx}_t{time_idx}.dcm"
                base_value = 10.0 * (slice_idx + 1) + float(time_idx)
                if label == "RL":
                    base_value += 100.0
                elif label == "AP":
                    base_value += 200.0
                elif label == "FH":
                    base_value += 300.0
                series_description = "Magnitude"
                if label is not None:
                    series_description = f"Velocity {int(venc_lookup[label])} {label}"
                datasets[path] = FakeDataset(
                    pixel_array=np.full((2, 3), base_value, dtype=np.float32),
                    Manufacturer="TestVendor",
                    ProtocolName="4D Flow",
                    SeriesDescription=series_description,
                    SequenceName=series_description,
                    ImageOrientationPatient=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                    ImagePositionPatient=[0.0, 0.0, float(slice_idx)],
                    PixelSpacing=[1.0, 2.0],
                    SliceThickness=3.0,
                    TriggerTime=float(100 * time_idx),
                    CardiacRRIntervalSpecified=900.0,
                )
                entries.append(
                    {
                        "path": path,
                        "case_id": "case-1",
                        "manufacturer": "TestVendor",
                        "group_kind": 0,
                        "component_label": label,
                        "is_magnitude": is_magnitude,
                        "series_description": "4D Flow",
                        "protocol_name": "4D Flow",
                    }
                )

    thread_ids = {"single": set(), "multi": set()}
    mode = {"name": "single"}

    class FakePydicom:
        def dcmread(self, path, force=True, stop_before_pixels=False):
            thread_ids[mode["name"]].add(threading.get_ident())
            time.sleep(0.01)
            return datasets[path]

    def fake_bpc(mag, flow, venc, config=None, progress_callback=None):
        return np.asarray(flow, dtype=np.float32), np.zeros(flow.shape[:3], dtype=bool), {
            "enabled": True,
            "applied": False,
            "stationary_voxels": 0,
            "skipped_reason": "disabled",
        }

    monkeypatch.setattr(dicom_module, "_import_pydicom", lambda: FakePydicom())
    monkeypatch.setattr(dicom_module, "apply_background_phase_correction_to_mag_flow", fake_bpc)

    case = InputCase(
        input_path="/tmp/dicom_case",
        input_kind="dicom",
        display_name="dicom_case",
        output_name="dicom_case",
        source_group="case-1",
        metadata={
            "manufacturer": "TestVendor",
            "group_kind": 0,
            "series_description": "4D Flow",
            "protocol_name": "4D Flow",
            "dicom_entries": entries,
        },
    )

    single = dicom_module.load_dicom_case(case, dicom_read_workers=1)
    mode["name"] = "multi"
    multi = dicom_module.load_dicom_case(case, dicom_read_workers=4)

    assert single.metadata["dicom_read_workers"] == 1
    assert multi.metadata["dicom_read_workers"] == 4
    assert len(thread_ids["single"]) == 1
    assert len(thread_ids["multi"]) > 1
    assert single.rr == pytest.approx(multi.rr)
    assert np.allclose(single.mag, multi.mag)
    assert np.allclose(single.flow, multi.flow)
    assert np.allclose(single.resolution, multi.resolution)
    assert np.allclose(single.venc, multi.venc)
