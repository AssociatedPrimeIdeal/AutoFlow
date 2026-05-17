from pathlib import Path

import h5py
import numpy as np
import pytest

import autoflow.algorithms.metrics as metrics_module
import autoflow.core.pipeline as pipeline_module
from autoflow.algorithms import load_h5_data, normalize_loaded_case
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
    assert case.capabilities.to_dict() == {
        "has_segmentation": False,
        "has_tke": False,
        "has_complex_source": False,
        "supports_wss": True,
        "supports_plane_metrics": True,
    }


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
    monkeypatch.setattr(pipeline_module, "load_h5_data", lambda path: case)

    ws = Workspace()
    ws.paths.flow_path = "dummy.h5"
    PipelineEngine().load_data(ws, lambda msg: None)

    assert ws.input_state.capabilities.has_tke is True
    assert ws.source_sigma is not None
    assert ws.source_tke_array is None
    assert ws.derived.tke_array is None
    assert ws.derived.tke_volume is None
    assert all(obj.data_key != "tke_volume" for obj in ws.scene_objects.values())


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
    monkeypatch.setattr(pipeline_module, "load_h5_data", lambda path: case)

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
