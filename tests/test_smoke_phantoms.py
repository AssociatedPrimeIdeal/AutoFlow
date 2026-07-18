import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import h5py
import numpy as np

from autoflow import AutoFlowConfig, run_batch, run_case
from autoflow.algorithms.data import load_h5_data
from autoflow.algorithms.dicom import collect_input_cases
from autoflow.algorithms.segmentation import default_nnunet_model_folder
from autoflow.algorithms.pwv import compute_cross_correlation_delay_ms, detect_waveform_foot_time_ms
from autoflow.algorithms.preprocess import filter_connected_components
from autoflow.algorithms.segmentation import generate_nnunet_auto_segmentation, save_segmentation_to_source_h5
from autoflow.config import bundle_to_autoflow_kwargs
from autoflow.core.pipeline import PipelineEngine
from autoflow.core.models import SkeletonParams
from autoflow.plane_io import build_plane_records
from autoflow.plane_io import save_pwv_h5
from autoflow.rendering.videos import _build_union_surface, _path_color, _path_group_name, _write_video, render_plane_rotation_video


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
PHANTOM_CASES = ("phantom_S", "phantom_U", "phantom_Y")
REL_TOL = 0.05
PLANE_SPACING_MM = 15.0


def _run_case(input_name: str, *, skip_derived: bool = False):
    output_root = Path(tempfile.mkdtemp(prefix=f"autoflow_{input_name}_", dir="/tmp"))
    cfg = AutoFlowConfig(
        inputs=[str(DATA_DIR / f"{input_name}.h5")],
        output_dir=str(output_root),
        skip_derived=skip_derived,
        skip_plane_metrics=False,
        use_multithread=False,
        plane_mode="distance",
        cross_section_dist=PLANE_SPACING_MM,
        make_plane_video=False,
        make_wss_video=False,
        make_streamlines_video=False,
        make_tke_video=False,
    )
    results, case_out = run_batch(cfg)
    assert len(results) == 1
    assert results[0]["status"] == "ok"
    case_dir = Path(case_out)
    metrics = json.loads((case_dir / "plane_metrics.json").read_text(encoding="utf-8"))
    return case_dir, metrics


def _relative_error(measured: float, truth: float) -> float:
    denom = max(abs(float(truth)), 1e-12)
    return abs(float(measured) - float(truth)) / denom


def _mean_relative_error(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=float))) if values else 0.0


def _truth_mean_velocity_cm_s(flow_rate_ml_s: np.ndarray, area_mm2: float) -> float:
    area = max(float(area_mm2), 1e-12)
    flow_rate = np.asarray(flow_rate_ml_s, dtype=float)
    return float(np.mean(flow_rate) * 100.0 / area)


def test_repo_phantom_h5_files_embed_truth_group():
    expected = {
        "phantom_S": (1,),
        "phantom_U": (1,),
        "phantom_Y": (3,),
        "phantom_P": None,
    }
    for case_name, path_scale_shape in expected.items():
        path = DATA_DIR / f"{case_name}.h5"
        assert path.is_file()
        with h5py.File(path, "r") as handle:
            assert "truth" in handle
            truth = handle["truth"]
            assert truth["flow_xyzt3_cm_s"].shape[-1] == 3
            if case_name != "phantom_P":
                assert truth["segmentation_xyzt"].shape == handle["segmask"].shape
                assert truth["mag_xyzt"].shape == handle["img_complex"].shape[:-1]
                assert truth["path_scale_values"].shape == path_scale_shape
            else:
                assert truth["segmentation_xyzt"].shape == handle["segmentation"].shape
                assert truth["mag_xyzt"].shape == handle["mag"].shape
                assert truth["pressure_gradient_xyzt3_pa_m"].shape[-1] == 3


def test_load_h5_data_supports_nested_group_real_img_layout_case_insensitive_keys(tmp_path):
    path = tmp_path / "nested_real_img.h5"
    mag = np.arange(24, dtype=np.float32).reshape(2, 3, 4) + 1.0
    flow = np.stack(
        [
            100.0 + mag,
            200.0 + mag,
            300.0 + mag,
        ],
        axis=-1,
    )
    seg = (mag > 12).astype(np.int16)
    with h5py.File(path, "w") as handle:
        handle.attrs["origin"] = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        case = handle.create_group("Case001")
        case.create_dataset("IMG", data=np.concatenate([mag[..., None], flow], axis=-1))
        case.create_dataset("resolution", data=np.array([1.2, 1.3, 1.4], dtype=np.float32))
        case.create_dataset("rr", data=np.array(812.5, dtype=np.float32))
        case.create_dataset("Venc", data=np.array([50.0, 60.0, 70.0], dtype=np.float32))
        case.create_dataset("Spatial_Order", data=np.asarray(["FH", "RL", "PA"], dtype="S4"))
        case.create_dataset("venc_order", data=np.asarray(["HF", "RL", "PA"], dtype="S4"))
        case.create_dataset("Segmentation", data=seg)

    loaded = load_h5_data(str(path))
    expected_mag = np.flip(np.flip(np.transpose(mag, (1, 2, 0)), axis=0), axis=1)
    flow_spatial = np.flip(np.flip(np.transpose(flow, (1, 2, 0, 3)), axis=0), axis=1)
    expected_flow = np.stack(
        [
            -flow_spatial[..., 1],
            -flow_spatial[..., 2],
            -flow_spatial[..., 0],
        ],
        axis=-1,
    )
    expected_seg = np.flip(np.flip(np.transpose(seg, (1, 2, 0)), axis=0), axis=1)

    assert loaded.source_format == "normalized_h5"
    assert loaded.source_group == "Case001"
    assert loaded.metadata["h5_layout"] == "combined_img_real"
    assert loaded.mag.shape == (3, 4, 2, 1)
    assert loaded.flow.shape == (3, 4, 2, 1, 3)
    assert loaded.segmentation.shape == (3, 4, 2, 1)
    assert np.allclose(loaded.mag[..., 0], expected_mag)
    assert np.allclose(loaded.flow[..., 0, :], expected_flow)
    assert np.array_equal(loaded.segmentation[..., 0], expected_seg)
    assert np.allclose(loaded.resolution, [1.3, 1.4, 1.2])
    assert np.allclose(loaded.origin, [1.0, 2.0, 3.0])
    assert np.allclose(loaded.venc, [60.0, 70.0, 50.0])
    assert loaded.rr == 812.5


def test_load_h5_data_orders_dual_venc_channel_groups_by_venc(tmp_path):
    path = tmp_path / "dual_venc_high_first.h5"
    velocity = np.zeros((2, 2, 2, 1, 3), dtype=np.float32)
    velocity[..., 0] = 20.0
    velocity[..., 1] = -10.0
    velocity[..., 2] = 5.0
    high_venc = np.array([150.0, 150.0, 150.0], dtype=np.float32)
    low_venc = np.array([50.0, 50.0, 50.0], dtype=np.float32)
    high_phase = np.pi * velocity / high_venc.reshape((1, 1, 1, 1, 3))
    low_phase = np.pi * velocity / low_venc.reshape((1, 1, 1, 1, 3))
    img = np.ones((2, 2, 2, 1, 7), dtype=np.complex64)
    img[..., 1:4] = np.exp(1j * high_phase)
    img[..., 4:7] = np.exp(1j * low_phase)

    with h5py.File(path, "w") as handle:
        handle.create_dataset("img", data=img)
        handle.create_dataset("Resolution", data=np.ones(3, dtype=np.float32))
        handle.create_dataset("VENC", data=np.concatenate([high_venc, low_venc]))
        handle.create_dataset("SpatialOrder", data=np.asarray(["LR", "AP", "FH"], dtype="S4"))
        handle.create_dataset("VENCOrder", data=np.asarray(["LR", "AP", "FH"], dtype="S4"))

    loaded = load_h5_data(str(path))

    assert loaded.source_format == "legacy_h5_dual_venc"
    assert np.allclose(loaded.flow, velocity, atol=1e-5)
    assert np.allclose(loaded.venc, high_venc)
    assert loaded.metadata["dual_venc"]["lv_channel_indices"] == [4, 5, 6]
    assert loaded.metadata["dual_venc"]["hv_channel_indices"] == [1, 2, 3]


def test_load_h5_data_reuses_dual_venc_singleton_time_corr_cache(tmp_path, monkeypatch):
    path = tmp_path / "dual_venc_corr_cache.h5"
    img = np.ones((2, 2, 2, 2, 7), dtype=np.complex64)
    corr = np.zeros((2, 2, 2, 1, 3), dtype=np.float32)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("img", data=img)
        handle.create_dataset("Resolution", data=np.ones(3, dtype=np.float32))
        handle.create_dataset(
            "VENC",
            data=np.array([150.0, 150.0, 150.0, 50.0, 50.0, 50.0], dtype=np.float32),
        )
        handle.create_dataset("SpatialOrder", data=np.asarray(["LR", "AP", "FH"], dtype="S4"))
        handle.create_dataset("VENCOrder", data=np.asarray(["LR", "AP", "FH"], dtype="S4"))
        for name in ("corr_low", "corr_high"):
            ds = handle.create_dataset(name, data=corr)
            ds.attrs["corr_algorithm"] = "msac"
            ds.attrs["corr_version"] = 1
            ds.attrs["corr_fit_order"] = 3
            ds.attrs["corr_threshold"] = 0.1

    def fail_execute_msac(*_args, **_kwargs):
        raise AssertionError("singleton-time dual-venc corr cache should avoid MSAC")

    monkeypatch.setattr("autoflow.algorithms.phase_correction.execute_msac", fail_execute_msac)
    loaded = load_h5_data(
        str(path),
        correction_config={"enabled": True, "corr_fit_order": 3, "threshold": 0.1},
    )

    reports = loaded.metadata["background_phase_correction"]
    assert reports["dual_venc_low"]["cache_hit"] is True
    assert reports["dual_venc_high"]["cache_hit"] is True
    assert loaded.flow.shape == (2, 2, 2, 2, 3)


def test_run_case_segmentation_only_stops_before_skeleton(tmp_path, monkeypatch):
    path = tmp_path / "segmentation_only.h5"
    mag = np.ones((2, 2, 2, 1), dtype=np.float32)
    flow = np.zeros((2, 2, 2, 1, 3), dtype=np.float32)
    segmentation = np.ones((2, 2, 2, 1), dtype=np.int16)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("mag", data=mag)
        handle.create_dataset("flow", data=flow)
        handle.create_dataset("segmentation", data=segmentation)
        handle.create_dataset("Resolution", data=np.ones(3, dtype=np.float32))
        handle.create_dataset("VENC", data=np.full(3, 100.0, dtype=np.float32))
        handle.create_dataset("SpatialOrder", data=np.asarray(["LR", "AP", "FH"], dtype="S4"))
        handle.create_dataset("VENCOrder", data=np.asarray(["LR", "AP", "FH"], dtype="S4"))

    def fail_run_step(*_args, **_kwargs):
        raise AssertionError("segmentation-only must stop before skeleton")

    monkeypatch.setattr(PipelineEngine, "run_step", fail_run_step)
    out_dir = tmp_path / "out"
    summary = run_case(
        str(path),
        output_dir=str(out_dir),
        config=AutoFlowConfig(segmentation_only=True),
    )

    assert summary["segmentation_only"] is True
    assert summary["segmentation_source"] == "original"
    assert summary["stage_times_sec"].keys() == {"load"}
    assert (out_dir / "summary.json").is_file()


def test_load_h5_data_flattens_singleton_row_and_column_metadata_vectors(tmp_path):
    path = tmp_path / "real_img_vector_metadata.h5"
    mag = np.ones((2, 2, 2, 1), dtype=np.float32) * 5.0
    flow = np.zeros((2, 2, 2, 1, 3), dtype=np.float32)
    flow[..., 0] = 10.0
    flow[..., 1] = 20.0
    flow[..., 2] = 30.0
    img = np.concatenate([mag[..., None], flow], axis=-1)

    with h5py.File(path, "w") as handle:
        handle.create_dataset("img", data=img)
        handle.create_dataset("Resolution", data=np.array([[1.2, 1.3, 1.4]], dtype=np.float32))
        handle.create_dataset("Origin", data=np.array([[5.0, 6.0, 7.0]], dtype=np.float32))
        handle.create_dataset("RR", data=np.array([[812.5]], dtype=np.float32))
        handle.create_dataset("VENC", data=np.array([[50.0], [60.0], [70.0]], dtype=np.float32))
        handle.create_dataset("SpatialOrder", data=np.asarray(["LR", "AP", "FH"], dtype="S4"))
        handle.create_dataset("VENCOrder", data=np.asarray(["LR", "AP", "FH"], dtype="S4"))

    loaded = load_h5_data(str(path))

    assert np.allclose(loaded.mag, mag)
    assert np.allclose(loaded.flow, flow)
    assert np.allclose(loaded.resolution, [1.2, 1.3, 1.4])
    assert np.allclose(loaded.origin, [5.0, 6.0, 7.0])
    assert np.allclose(loaded.venc, [50.0, 60.0, 70.0])
    assert loaded.rr == 812.5


def test_load_h5_data_accepts_comma_separated_spatial_and_venc_order_strings(tmp_path):
    path = tmp_path / "real_img_csv_orders.h5"
    mag = np.arange(24, dtype=np.float32).reshape(2, 3, 4) + 1.0
    flow = np.stack(
        [
            100.0 + mag,
            200.0 + mag,
            300.0 + mag,
        ],
        axis=-1,
    )
    with h5py.File(path, "w") as handle:
        handle.create_dataset("img", data=np.concatenate([mag[..., None], flow], axis=-1))
        handle.create_dataset("Resolution", data=np.array([1.2, 1.3, 1.4], dtype=np.float32))
        handle.create_dataset("RR", data=np.array(812.5, dtype=np.float32))
        handle.create_dataset("VENC", data=np.array([50.0, 60.0, 70.0], dtype=np.float32))
        handle.create_dataset("SpatialOrder", data=np.bytes_("FH,RL,PA"))
        handle.create_dataset("VENCOrder", data=np.bytes_("HF,RL,PA"))

    loaded = load_h5_data(str(path))
    expected_mag = np.flip(np.flip(np.transpose(mag, (1, 2, 0)), axis=0), axis=1)
    flow_spatial = np.flip(np.flip(np.transpose(flow, (1, 2, 0, 3)), axis=0), axis=1)
    expected_flow = np.stack(
        [
            -flow_spatial[..., 1],
            -flow_spatial[..., 2],
            -flow_spatial[..., 0],
        ],
        axis=-1,
    )

    assert loaded.metadata["spatial_order_raw"] == ["FH", "RL", "PA"]
    assert loaded.metadata["venc_order_raw"] == ["HF", "RL", "PA"]
    assert np.allclose(loaded.mag[..., 0], expected_mag)
    assert np.allclose(loaded.flow[..., 0, :], expected_flow)


def test_load_h5_data_rescales_real_img_phase_radians_to_venc(tmp_path):
    path = tmp_path / "real_img_phase_units.h5"
    mag = np.full((2, 2, 2, 1), 5.0, dtype=np.float32)
    flow_velocity = np.zeros((2, 2, 2, 1, 3), dtype=np.float32)
    flow_velocity[..., 0] = 45.0
    flow_velocity[..., 1] = -54.0
    flow_velocity[..., 2] = 63.0
    venc = np.array([50.0, 60.0, 70.0], dtype=np.float32)
    flow_phase = np.pi * flow_velocity / venc.reshape((1, 1, 1, 1, 3))
    img = np.concatenate([mag[..., None], flow_phase], axis=-1)

    with h5py.File(path, "w") as handle:
        handle.create_dataset("img", data=img)
        handle.create_dataset("Resolution", data=np.array([1.0, 1.1, 1.2], dtype=np.float32))
        handle.create_dataset("RR", data=np.array(700.0, dtype=np.float32))
        handle.create_dataset("VENC", data=venc)
        handle.create_dataset("SpatialOrder", data=np.asarray(["LR", "AP", "FH"], dtype="S4"))
        handle.create_dataset("VENCOrder", data=np.asarray(["LR", "AP", "FH"], dtype="S4"))

    loaded = load_h5_data(str(path))

    assert loaded.metadata["h5_layout"] == "combined_img_real"
    assert loaded.metadata["flow_value_unit_raw"] == "phase_radians"
    assert loaded.metadata["flow_rescaled_from_pi_to_venc"] is True
    assert np.allclose(loaded.flow, flow_velocity, atol=1e-5)
    assert np.allclose(loaded.mag, mag)


def test_load_h5_data_transposes_channel_first_real_img_layout(tmp_path):
    path = tmp_path / "real_img_channel_first.h5"
    mag = (np.arange(48, dtype=np.float32).reshape(2, 3, 4, 2) + 1.0) / 10.0
    flow = np.stack(
        [
            10.0 + mag,
            20.0 + mag,
            30.0 + mag,
        ],
        axis=-1,
    )
    img = np.concatenate([mag[..., None], flow], axis=-1)
    img_channel_first = np.transpose(img, (4, 3, 2, 1, 0))

    with h5py.File(path, "w") as handle:
        handle.create_dataset("img", data=img_channel_first)
        handle.create_dataset("Resolution", data=np.array([1.0, 1.1, 1.2], dtype=np.float32))
        handle.create_dataset("RR", data=np.array(720.0, dtype=np.float32))
        handle.create_dataset("VENC", data=np.array([50.0, 60.0, 70.0], dtype=np.float32))
        handle.create_dataset("SpatialOrder", data=np.asarray(["LR", "AP", "FH"], dtype="S4"))
        handle.create_dataset("VENCOrder", data=np.asarray(["LR", "AP", "FH"], dtype="S4"))

    loaded = load_h5_data(str(path))

    assert loaded.metadata["h5_layout"] == "combined_img_real"
    assert loaded.metadata["real_img_channel_axis_raw"] == "first"
    assert loaded.mag.shape == mag.shape
    assert loaded.flow.shape == flow.shape
    assert np.allclose(loaded.mag, mag)
    assert np.allclose(loaded.flow, flow)


def test_load_h5_data_reorients_normalized_h5_real_flow_mag_and_optional_fields(tmp_path):
    path = tmp_path / "normalized_real_fields.h5"
    mag = (np.arange(48, dtype=np.float32).reshape(2, 3, 4, 2) + 1.0) / 10.0
    flow = np.stack(
        [
            10.0 + mag,
            20.0 + mag,
            30.0 + mag,
        ],
        axis=-1,
    )
    seg = (mag > 2.0).astype(np.int16)
    sigma = np.stack(
        [
            1.0 + mag,
            2.0 + mag,
            3.0 + mag,
        ],
        axis=-1,
    )
    tke = 100.0 + mag

    with h5py.File(path, "w") as handle:
        case = handle.create_group("Case002")
        case.create_dataset("mag", data=mag)
        case.create_dataset("flow", data=flow)
        case.create_dataset("segmask", data=seg)
        case.create_dataset("sigma", data=sigma)
        case.create_dataset("tke_array", data=tke)
        case.create_dataset("Resolution", data=np.array([1.1, 1.2, 1.3], dtype=np.float32))
        case.create_dataset("RR", data=np.array(640.0, dtype=np.float32))
        case.create_dataset("Venc", data=np.array([45.0, 55.0, 65.0], dtype=np.float32))
        case.create_dataset("SpatialOrder", data=np.asarray(["FH", "RL", "PA"], dtype="S4"))
        case.create_dataset("VENCOrder", data=np.asarray(["HF", "RL", "PA"], dtype="S4"))

    loaded = load_h5_data(str(path))
    expected_mag = np.flip(np.flip(np.transpose(mag, (1, 2, 0, 3)), axis=0), axis=1)
    flow_spatial = np.flip(np.flip(np.transpose(flow, (1, 2, 0, 3, 4)), axis=0), axis=1)
    expected_flow = np.stack(
        [
            -flow_spatial[..., 1],
            -flow_spatial[..., 2],
            -flow_spatial[..., 0],
        ],
        axis=-1,
    )
    expected_seg = np.flip(np.flip(np.transpose(seg, (1, 2, 0, 3)), axis=0), axis=1)
    sigma_spatial = np.flip(np.flip(np.transpose(sigma, (1, 2, 0, 3, 4)), axis=0), axis=1)
    expected_sigma = np.stack(
        [
            sigma_spatial[..., 1],
            sigma_spatial[..., 2],
            sigma_spatial[..., 0],
        ],
        axis=-1,
    )
    expected_tke = np.flip(np.flip(np.transpose(tke, (1, 2, 0, 3)), axis=0), axis=1)

    assert loaded.source_format == "normalized_h5"
    assert loaded.source_group == "Case002"
    assert loaded.metadata["h5_layout"] == "normalized_h5"
    assert loaded.mag.shape == (3, 4, 2, 2)
    assert loaded.flow.shape == (3, 4, 2, 2, 3)
    assert loaded.segmentation.shape == (3, 4, 2, 2)
    assert loaded.sigma is not None
    assert loaded.tke_array is not None
    assert np.allclose(loaded.mag, expected_mag)
    assert np.allclose(loaded.flow, expected_flow)
    assert np.array_equal(loaded.segmentation, expected_seg)
    assert np.allclose(loaded.sigma, expected_sigma)
    assert np.allclose(loaded.tke_array, expected_tke)
    assert np.allclose(loaded.resolution, [1.2, 1.3, 1.1])
    assert np.allclose(loaded.venc, [55.0, 65.0, 45.0])
    assert loaded.rr == 640.0


def test_load_h5_data_writes_and_reuses_background_phase_corr_cache(monkeypatch, tmp_path):
    path = tmp_path / "bgc_cache.h5"
    mag = np.ones((2, 2, 2, 1), dtype=np.float32)
    flow = np.zeros((2, 2, 2, 1, 3), dtype=np.float32)
    corr_xyzt3 = np.zeros((2, 2, 2, 1, 3), dtype=np.float32)
    corr_xyzt3[..., 0] = 0.1
    corr_xyzt3[..., 1] = -0.2
    corr_xyzt3[..., 2] = 0.3
    with h5py.File(path, "w") as handle:
        handle.create_dataset("mag", data=mag)
        handle.create_dataset("flow", data=flow)
        handle.create_dataset("Resolution", data=np.array([1.0, 1.0, 1.0], dtype=np.float32))
        handle.create_dataset("RR", data=np.array(1000.0, dtype=np.float32))
        handle.create_dataset("VENC", data=np.array([100.0, 100.0, 100.0], dtype=np.float32))
        handle.create_dataset("SpatialOrder", data=np.asarray(["LR", "AP", "FH"], dtype="S4"))
        handle.create_dataset("VENCOrder", data=np.asarray(["LR", "AP", "FH"], dtype="S4"))

    calls = {"count": 0}

    def fake_execute_msac(im, corr_fit_order=3, th=0.1):
        calls["count"] += 1
        corr_nvtzyx = np.transpose(corr_xyzt3, (4, 3, 2, 1, 0)).astype(np.float32)
        stationary = np.ones(corr_xyzt3.shape[:3], dtype=bool)
        return corr_nvtzyx, stationary, {
            "applied": True,
            "corr_fit_order": int(corr_fit_order),
            "threshold": float(th),
            "stationary_voxels": int(np.sum(stationary)),
            "skipped_reason": "",
        }

    monkeypatch.setattr("autoflow.algorithms.phase_correction.execute_msac", fake_execute_msac)
    cfg = {"enabled": True, "corr_fit_order": 2, "threshold": 0.25}
    loaded = load_h5_data(str(path), correction_config=cfg)

    assert calls["count"] == 1
    first_meta = loaded.metadata["background_phase_correction"]
    assert first_meta["applied"] is True
    assert first_meta["cache_hit"] is False
    assert first_meta["cache_written"] is True
    with h5py.File(path, "r") as handle:
        assert "corr" in handle
        assert handle["corr"].shape == corr_xyzt3.shape
        assert np.allclose(handle["corr"][:], corr_xyzt3)
        assert int(handle["corr"].attrs["corr_fit_order"]) == 2
        assert np.isclose(float(handle["corr"].attrs["corr_threshold"]), 0.25)

    def fail_execute_msac(*_args, **_kwargs):
        raise AssertionError("cached corr should avoid MSAC")

    monkeypatch.setattr("autoflow.algorithms.phase_correction.execute_msac", fail_execute_msac)
    loaded_cached = load_h5_data(str(path), correction_config=cfg)

    cached_meta = loaded_cached.metadata["background_phase_correction"]
    assert cached_meta["applied"] is True
    assert cached_meta["cache_hit"] is True
    assert cached_meta["cache_name"] == "corr"
    assert np.allclose(loaded_cached.flow, loaded.flow)


def test_load_h5_data_grouped_h5_does_not_reuse_untagged_root_corr(monkeypatch, tmp_path):
    path = tmp_path / "grouped_bgc_cache.h5"
    mag = np.ones((2, 2, 2, 1), dtype=np.float32)
    flow = np.zeros((2, 2, 2, 1, 3), dtype=np.float32)
    stale_corr = np.full((2, 2, 2, 1, 3), 0.5, dtype=np.float32)
    fresh_corr = np.zeros((2, 2, 2, 1, 3), dtype=np.float32)
    fresh_corr[..., 0] = 0.1
    fresh_corr[..., 1] = -0.2
    fresh_corr[..., 2] = 0.3
    with h5py.File(path, "w") as handle:
        handle.create_dataset("corr", data=stale_corr)
        for case_name in ("CaseA", "CaseB"):
            case = handle.create_group(case_name)
            case.create_dataset("mag", data=mag)
            case.create_dataset("flow", data=flow)
            case.create_dataset("Resolution", data=np.array([1.0, 1.0, 1.0], dtype=np.float32))
            case.create_dataset("RR", data=np.array(1000.0, dtype=np.float32))
            case.create_dataset("VENC", data=np.array([100.0, 100.0, 100.0], dtype=np.float32))
            case.create_dataset("SpatialOrder", data=np.asarray(["LR", "AP", "FH"], dtype="S4"))
            case.create_dataset("VENCOrder", data=np.asarray(["LR", "AP", "FH"], dtype="S4"))

    calls = {"count": 0}

    def fake_execute_msac(im, corr_fit_order=3, th=0.1):
        calls["count"] += 1
        corr_nvtzyx = np.transpose(fresh_corr, (4, 3, 2, 1, 0)).astype(np.float32)
        stationary = np.ones(fresh_corr.shape[:3], dtype=bool)
        return corr_nvtzyx, stationary, {
            "applied": True,
            "corr_fit_order": int(corr_fit_order),
            "threshold": float(th),
            "stationary_voxels": int(np.sum(stationary)),
            "skipped_reason": "",
        }

    monkeypatch.setattr("autoflow.algorithms.phase_correction.execute_msac", fake_execute_msac)
    cfg = {"enabled": True, "corr_fit_order": 2, "threshold": 0.25}
    loaded = load_h5_data(str(path), correction_config=cfg, source_group="CaseA")

    assert calls["count"] == 1
    first_meta = loaded.metadata["background_phase_correction"]
    assert first_meta["cache_hit"] is False
    assert first_meta["cache_reason"] == "missing_group_tag"
    with h5py.File(path, "r") as handle:
        assert np.allclose(handle["corr"][:], stale_corr)
        assert np.allclose(handle["CaseA"]["corr"][:], fresh_corr)
        assert handle["CaseA"]["corr"].attrs["corr_source_group"] == "CaseA"

    def fail_execute_msac(*_args, **_kwargs):
        raise AssertionError("group-specific cached corr should avoid MSAC")

    monkeypatch.setattr("autoflow.algorithms.phase_correction.execute_msac", fail_execute_msac)
    loaded_cached = load_h5_data(str(path), correction_config=cfg, source_group="CaseA")

    cached_meta = loaded_cached.metadata["background_phase_correction"]
    assert cached_meta["cache_hit"] is True
    assert cached_meta["cache_name"] == "corr"
    assert np.allclose(loaded_cached.flow, loaded.flow)


def test_save_segmentation_to_source_h5_writes_reusable_embedded_segmentation(tmp_path):
    path = tmp_path / "embedded_autoseg_cache.h5"
    mag = np.arange(48, dtype=np.float32).reshape(2, 3, 4, 2) + 1.0
    flow = np.zeros((2, 3, 4, 2, 3), dtype=np.float32)
    seg_internal = np.zeros((3, 4, 2, 2), dtype=np.int16)
    seg_internal[1:, 1:, :, :] = 2

    with h5py.File(path, "w") as handle:
        case = handle.create_group("CaseAuto")
        case.create_dataset("mag", data=mag)
        case.create_dataset("flow", data=flow)
        case.create_dataset("Resolution", data=np.array([1.1, 1.2, 1.3], dtype=np.float32))
        case.create_dataset("RR", data=np.array(700.0, dtype=np.float32))
        case.create_dataset("Venc", data=np.array([45.0, 55.0, 65.0], dtype=np.float32))
        case.create_dataset("SpatialOrder", data=np.asarray(["FH", "RL", "PA"], dtype="S4"))
        case.create_dataset("VENCOrder", data=np.asarray(["HF", "RL", "PA"], dtype="S4"))

    save_segmentation_to_source_h5(
        str(path),
        seg_internal,
        resolution=np.array([1.2, 1.3, 1.1], dtype=np.float32),
        origin=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        provenance={"source": "auto", "created_at": "2026-01-01T00:00:00Z"},
        source_spatial_order=("FH", "RL", "PA"),
    )

    with h5py.File(path, "r") as handle:
        ds = handle["CaseAuto"]["segmask"]
        assert ds.attrs["autoflow_source"] == "auto_segmentation"
        assert tuple(x.decode() if isinstance(x, bytes) else str(x) for x in ds.attrs["SpatialOrder"]) == ("FH", "RL", "PA")

    loaded = load_h5_data(str(path))
    assert loaded.segmentation is not None
    assert np.array_equal(loaded.segmentation, seg_internal)


def test_save_segmentation_to_source_h5_targets_selected_group(tmp_path):
    path = tmp_path / "multi_case_embedded_autoseg_cache.h5"
    mag = np.arange(48, dtype=np.float32).reshape(2, 3, 4, 2) + 1.0
    flow = np.zeros((2, 3, 4, 2, 3), dtype=np.float32)
    seg_internal = np.full((3, 4, 2, 2), 3, dtype=np.int16)

    with h5py.File(path, "w") as handle:
        for case_name in ("CaseA", "CaseB"):
            case = handle.create_group(case_name)
            case.create_dataset("mag", data=mag)
            case.create_dataset("flow", data=flow)
            case.create_dataset("Resolution", data=np.array([1.1, 1.2, 1.3], dtype=np.float32))
            case.create_dataset("RR", data=np.array(700.0, dtype=np.float32))
            case.create_dataset("Venc", data=np.array([45.0, 55.0, 65.0], dtype=np.float32))
            case.create_dataset("SpatialOrder", data=np.asarray(["FH", "RL", "PA"], dtype="S4"))
            case.create_dataset("VENCOrder", data=np.asarray(["HF", "RL", "PA"], dtype="S4"))

    save_segmentation_to_source_h5(
        str(path),
        seg_internal,
        resolution=np.array([1.2, 1.3, 1.1], dtype=np.float32),
        origin=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        provenance={"source": "auto", "created_at": "2026-01-01T00:00:00Z"},
        source_spatial_order=("FH", "RL", "PA"),
        source_group="CaseB",
    )

    with h5py.File(path, "r") as handle:
        assert "segmask" not in handle["CaseA"]
        ds = handle["CaseB"]["segmask"]
        assert ds.attrs["autoflow_source"] == "auto_segmentation"
        assert ds.attrs["autoflow_source_group"] == "CaseB"

    loaded_a = load_h5_data(str(path), source_group="CaseA")
    loaded_b = load_h5_data(str(path), source_group="CaseB")
    assert loaded_a.segmentation is None
    assert loaded_b.segmentation is not None
    assert np.array_equal(loaded_b.segmentation, seg_internal)


def test_collect_input_cases_expands_multiple_h5_data_groups(tmp_path):
    path = tmp_path / "multi_group_same_leaf.h5"
    mag = np.ones((2, 2, 2, 1), dtype=np.float32)
    flow = np.zeros((2, 2, 2, 1, 3), dtype=np.float32)
    with h5py.File(path, "w") as handle:
        for group_name in ("A/Flow", "B/Flow"):
            case = handle.create_group(group_name)
            case.create_dataset("mag", data=mag)
            case.create_dataset("flow", data=flow)

    cases = collect_input_cases([str(path)])

    assert [case.source_group for case in cases] == ["A/Flow", "B/Flow"]
    assert [case.output_name for case in cases] == [
        "multi_group_same_leaf__A_Flow",
        "multi_group_same_leaf__B_Flow",
    ]


def test_collect_input_cases_only_scans_top_level_h5_files(tmp_path):
    root = tmp_path / "batch_root"
    root.mkdir()
    top_h5 = root / "ST0.h5"
    nested_dir = root / "autoflow_out" / "ST0"
    nested_dir.mkdir(parents=True)
    nested_h5 = nested_dir / "derived.h5"
    with h5py.File(top_h5, "w") as handle:
        case = handle.create_group("CaseTop")
        case.create_dataset("mag", data=np.ones((2, 2, 2, 1), dtype=np.float32))
        case.create_dataset("flow", data=np.zeros((2, 2, 2, 1, 3), dtype=np.float32))
    nested_h5.write_bytes(b"nested")

    cases = collect_input_cases([str(root)])

    assert [Path(case.input_path) for case in cases if case.input_kind == "h5"] == [top_h5]


def test_load_h5_data_supports_nested_group_complex_img_layout(tmp_path):
    path = tmp_path / "nested_complex_img.h5"
    mag = np.full((2, 2, 2, 1), 3.0, dtype=np.float32)
    flow = np.zeros((2, 2, 2, 1, 3), dtype=np.float32)
    flow[..., 0] = 12.0
    flow[..., 1] = -7.5
    flow[..., 2] = 3.25
    venc = np.array([50.0, 60.0, 70.0], dtype=np.float32)
    phase = np.pi * flow / venc.reshape((1, 1, 1, 1, 3))
    img = np.zeros((2, 2, 2, 1, 4), dtype=np.complex64)
    img[..., 0] = mag.astype(np.complex64)
    img[..., 1:4] = mag[..., None] * np.exp(1j * phase)
    with h5py.File(path, "w") as handle:
        root = handle.create_group("StudyRoot")
        case = root.create_group("SeriesA")
        case.create_dataset("Img", data=img)
        case.create_dataset("Resolution", data=np.array([1.0, 1.1, 1.2], dtype=np.float32))
        case.create_dataset("RR", data=np.array(900.0, dtype=np.float32))
        case.create_dataset("vEnC", data=venc)
        case.create_dataset("SpatialOrder", data=np.asarray(["LR", "AP", "FH"], dtype="S4"))
        case.create_dataset("VencOrder", data=np.asarray(["LR", "AP", "FH"], dtype="S4"))

    loaded = load_h5_data(str(path))

    assert loaded.source_format == "legacy_h5"
    assert loaded.source_group == "StudyRoot/SeriesA"
    assert loaded.metadata["h5_layout"] == "complex_img"
    assert loaded.mag.shape == (2, 2, 2, 1)
    assert loaded.flow.shape == (2, 2, 2, 1, 3)
    assert np.allclose(loaded.flow, flow, atol=1e-5)
    assert loaded.sigma is not None
    assert loaded.capabilities.has_complex_source is True


def test_s_u_y_plane_metrics_match_truth_within_two_percent_mean_error():
    for case_name in PHANTOM_CASES:
        case_dir, metrics = _run_case(case_name)
        assert (case_dir / "plane_metrics.json").is_file()
        with h5py.File(DATA_DIR / f"{case_name}.h5", "r") as handle:
            truth = handle["truth"]
            truth_flow_per_beat = float(truth["flow_ml_per_beat"][()])
            truth_peak = float(truth["peak_centerline_cm_s"][()])
            path_scales = {idx: float(val) for idx, val in enumerate(truth["path_scale_values"][()].tolist())}
            area_mm2 = float(np.pi * float(truth["tube_radius_mm"][()]) ** 2)
            path_flow_rate_ml_s = np.asarray(truth["path_flow_rate_ml_s"][()], dtype=float)

        assert len(metrics) >= len(path_scales)
        seen_paths = set()
        flow_errors = []
        peak_errors = []
        meanv_errors = []
        for plane_index, metric in enumerate(metrics):
            assert int(metric["plane_index"]) == plane_index
            path_index = int(metric["path_index"])
            seen_paths.add(path_index)
            scale = path_scales[path_index]
            expected_flow = truth_flow_per_beat * scale
            expected_peak = truth_peak * scale
            expected_meanv = _truth_mean_velocity_cm_s(path_flow_rate_ml_s[path_index], area_mm2)
            flow_errors.append(_relative_error(metric["netflow_mL_beat"], expected_flow))
            peak_errors.append(_relative_error(metric["peakv_cm_s"], expected_peak))
            meanv_errors.append(_relative_error(metric["meanv_cm_s"], expected_meanv))

        assert seen_paths == set(path_scales)
        assert _mean_relative_error(flow_errors) < REL_TOL
        assert _mean_relative_error(peak_errors) < REL_TOL
        assert _mean_relative_error(meanv_errors) < REL_TOL


def test_write_video_retries_mp4_without_gif_fallback(monkeypatch, tmp_path):
    attempts = []

    class DummyWriter:
        def __init__(self, out_path):
            self.out_path = Path(out_path)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            if exc_type is None:
                self.out_path.write_bytes(b"mp4")
            return False

        def append_data(self, frame):
            assert np.asarray(frame).shape == (2, 2, 3)

    formats = []

    def fake_get_writer(out_path, fps=24, **kwargs):
        attempts.append(kwargs.get("codec", "default"))
        formats.append(kwargs.get("format"))
        if kwargs.get("codec") == "libx264":
            raise RuntimeError("missing codec")
        return DummyWriter(out_path)

    monkeypatch.setattr("autoflow.rendering.videos.imageio.get_writer", fake_get_writer)
    frames = [np.zeros((2, 2, 3), dtype=np.uint8)]
    out_path = _write_video(frames, str(tmp_path / "planes_rotate.mp4"), fps=12)

    assert out_path.endswith(".mp4")
    assert Path(out_path).is_file()
    assert attempts[0] == "libx264"
    assert formats and all(fmt == "ffmpeg" for fmt in formats)
    assert not (tmp_path / "planes_rotate.gif").exists()


def test_nnunet_autoseg_progress_callback_reports_stage_updates(monkeypatch, tmp_path):
    model_dir = tmp_path / "model"
    fold_dir = model_dir / "fold_all"
    fold_dir.mkdir(parents=True)
    (model_dir / "dataset.json").write_text(
        json.dumps({
            "channel_names": {"0": "mag", "1": "flow_x_mean_xyz"},
            "labels": {"background": 0, "vessel": 1},
            "file_ending": ".nii.gz",
        }),
        encoding="utf-8",
    )
    (model_dir / "plans.json").write_text(json.dumps({"plans": "ok"}), encoding="utf-8")

    def fake_run_subprocess(command, *, env=None, cwd=None, runner=None):
        out_dir = Path(command[command.index("-o") + 1])
        pred = out_dir / "autoflow_case.nii.gz"
        pred.write_bytes(b"fake")
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr('autoflow.algorithms.segmentation._run_subprocess', fake_run_subprocess)
    monkeypatch.setattr('autoflow.algorithms.segmentation._read_nifti_segmentation', lambda path: np.ones((2, 2, 2), dtype=np.int16))

    events = []
    mag = np.ones((2, 2, 2, 3), dtype=np.float32)
    flow = np.zeros((2, 2, 2, 3, 3), dtype=np.float32)
    artifact_prefix = tmp_path / "artifacts" / "autoflow_case"
    seg, provenance = generate_nnunet_auto_segmentation(
        mag=mag,
        flow=flow,
        resolution=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        model_folder=str(model_dir),
        device="cpu",
        artifact_prefix=str(artifact_prefix),
        progress_callback=events.append,
    )

    assert seg.shape == (2, 2, 2, 3)
    assert provenance["device"] == "cpu"
    assert len(provenance["feature_files"]) == 2
    assert all(Path(path).is_file() for path in provenance["feature_files"])
    assert Path(provenance["segmentation_nifti"]).is_file()
    assert provenance["prediction_file"] == provenance["segmentation_nifti"]
    stages = [event["stage"] for event in events]
    assert stages[0] == "autoseg_start"
    assert "autoseg_model_ready" in stages
    assert "autoseg_prepare_inputs" in stages
    assert "autoseg_run_inference" in stages
    assert "autoseg_read_prediction" in stages
    assert stages[-1] == "autoseg_finalize"
    assert all("elapsed_sec" in event for event in events)

def test_pwv_timing_methods_on_synthetic_waveforms():
    rr_ms = 1000.0
    x = np.linspace(0.0, 1.0, 20, endpoint=False)
    proximal = np.exp(-0.5 * ((x - 0.25) / 0.08) ** 2)
    distal = np.roll(proximal, 2)

    foot_prox, meta_prox = detect_waveform_foot_time_ms(
        proximal, rr_ms, method="tangent", allow_cycle_wrap=True
    )
    foot_dist, meta_dist = detect_waveform_foot_time_ms(
        distal, rr_ms, method="threshold", threshold_percent=10.0, allow_cycle_wrap=True
    )
    assert foot_prox is not None
    assert foot_dist is not None
    assert meta_prox.get("method") == "tangent"
    assert meta_dist.get("method") == "threshold"

    delay_ms, cc_meta = compute_cross_correlation_delay_ms(
        proximal, distal, rr_ms, window="full", allow_cycle_wrap=True
    )
    assert delay_ms is not None
    assert abs(float(cc_meta.get("lag_samples")) - 2.0) < 0.15
    assert abs(float(delay_ms) - 100.0) < 10.0
    assert int(cc_meta.get("interp_factor")) == 10

    delay_upstroke_ms, up_meta = compute_cross_correlation_delay_ms(
        proximal, distal, rr_ms, window="upstroke", allow_cycle_wrap=True
    )
    assert delay_upstroke_ms is not None
    assert abs(float(delay_upstroke_ms) - 100.0) < 10.0
    assert abs(float(up_meta.get("offset_lag_samples")) - 2.0) < 0.15


def test_pwv_wrapped_tangent_and_subframe_xcorr():
    rr_ms = 1000.0
    waveform = np.array([
        -15.166193483117201,
        418.84413044447297,
        435.4283063098984,
        354.5884270552724,
        288.8676802814978,
        137.02311573973498,
        60.88961149981558,
        16.682423130255852,
        2.4873332685670873,
        25.087174459539828,
        24.03118575795991,
        14.845060197501105,
        24.57764249657329,
        14.24500673573191,
        2.2664627477484784,
        -0.5951218100742958,
        3.0340050909445813,
        16.08187168750316,
        8.482758977647443,
        -8.544392118097038,
    ], dtype=float)
    foot_ms, foot_meta = detect_waveform_foot_time_ms(
        waveform, rr_ms, method="tangent", allow_cycle_wrap=True
    )
    assert foot_ms is not None
    assert foot_meta.get("cycle_wrapped") is True
    assert int(foot_meta.get("slope_index_wrapped")) >= waveform.size
    assert int(foot_meta.get("baseline_index")) == waveform.size - 1
    assert 900.0 < float(foot_ms) < 1000.0

    x = np.linspace(0.0, 1.0, 20, endpoint=False)
    proximal = np.exp(-0.5 * ((x - 0.25) / 0.08) ** 2)
    shifted = np.exp(-0.5 * ((((x - 0.25 - 0.075) + 0.5) % 1.0) - 0.5) ** 2 / (0.08 ** 2))
    delay_ms, cc_meta = compute_cross_correlation_delay_ms(
        proximal, shifted, rr_ms, window="full", allow_cycle_wrap=True, interp_factor=20
    )
    assert delay_ms is not None
    assert abs(float(delay_ms) - 75.0) < 15.0
    assert abs(float(cc_meta.get("lag_samples")) - 1.5) < 0.25
    assert int(cc_meta.get("interp_factor")) == 20


def test_plane_video_path_color_uses_group_scene_color():
    skeleton_params = SkeletonParams(
        label_groups={
            "aorta": {"path_color": "#f76707"},
            "pulmonary": {"path_color": "#4dabf7"},
        }
    )

    class _Ws:
        pass

    ws = _Ws()
    ws.skeleton_params = skeleton_params
    ws.path_info = [
        {"group_name": "aorta"},
        {"group_name": "pulmonary"},
        {},
    ]
    ws.group_order = ["aorta", "pulmonary"]
    ws.multilabel_groups = {
        "aorta": {"path_index_offset": 0, "centerline_paths_smooth": [np.zeros((2, 3))]},
        "pulmonary": {"path_index_offset": 1, "centerline_paths_smooth": [np.zeros((2, 3))]},
    }

    assert _path_group_name(ws, 0) == "aorta"
    assert _path_group_name(ws, 1) == "pulmonary"
    assert _path_group_name(ws, 2) == ""
    assert _path_color(ws, 0) == "#f76707"
    assert _path_color(ws, 1) == "#4dabf7"
    assert _path_color(ws, 2) == "deepskyblue"


def test_config_dir_reads_feature_render_settings_from_metric_jsons(tmp_path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    video_exporting_payload = {
        "window_size": [1111, 777],
        "rotate_dynamic_video": False,
        "plane_video": {
            "default": {
                "plane_color": "#654321",
                "plane_opacity": 0.4,
            },
            "label": {
                "prefix": "plane=",
                "font_size": 17,
                "text_color": "white",
            },
        },
    }
    planes_payload = {
        "render": {
            "default": {
                "plane_color": "#123456",
                "plane_opacity": 0.25,
            }
        }
    }
    wss_payload = {
        "render": {
            "clim": [1.0, 9.0],
            "show_scalar_bar": False,
            "bar_cfg": {"width": 0.11},
        }
    }
    tke_payload = {
        "render": {
            "clim": [2.0, 22.0],
        }
    }
    pressure_gradient_payload = {
        "render": {
            "clim": [3.0, 33.0],
            "show_scalar_bar": False,
        }
    }
    streamlines_payload = {
        "render": {
            "clim": [4.0, 44.0],
            "show_scalar_bar": False,
            "bar_cfg": {"position_x": 0.66},
        }
    }
    (config_dir / "video_exporting.json").write_text(json.dumps(video_exporting_payload), encoding="utf-8")
    (config_dir / "planes.json").write_text(json.dumps(planes_payload), encoding="utf-8")
    (config_dir / "wss.json").write_text(json.dumps(wss_payload), encoding="utf-8")
    (config_dir / "tke.json").write_text(json.dumps(tke_payload), encoding="utf-8")
    (config_dir / "pressure_gradient.json").write_text(json.dumps(pressure_gradient_payload), encoding="utf-8")
    (config_dir / "streamlines.json").write_text(json.dumps(streamlines_payload), encoding="utf-8")

    cfg = AutoFlowConfig.from_config_dir(str(config_dir))
    resolved = bundle_to_autoflow_kwargs({
        "video_exporting": video_exporting_payload,
        "planes": planes_payload,
        "wss": wss_payload,
        "tke": tke_payload,
        "pressure_gradient": pressure_gradient_payload,
        "streamlines": streamlines_payload,
    })

    assert cfg.window_size == (1111, 777)
    assert cfg.rotate_dynamic_video is False
    assert cfg.plane_video_cfg["default"]["plane_color"] == "#654321"
    assert cfg.plane_video_cfg["default"]["plane_opacity"] == 0.4
    assert cfg.plane_video_cfg["label"]["prefix"] == "plane="
    assert cfg.plane_video_cfg["label"]["font_size"] == 17
    assert cfg.plane_video_cfg["label"]["text_color"] == "white"
    assert cfg.wss_clim == (1.0, 9.0)
    assert cfg.wss_show_scalar_bar is False
    assert cfg.wss_bar_cfg["width"] == 0.11
    assert cfg.tke_clim == (2.0, 22.0)
    assert cfg.pressure_gradient_clim == (3.0, 33.0)
    assert cfg.pressure_gradient_show_scalar_bar is False
    assert cfg.streamline_clim == (4.0, 44.0)
    assert cfg.streamline_show_scalar_bar is False
    assert cfg.streamline_bar_cfg["position_x"] == 0.66
    assert resolved["wss_clim"] == (1.0, 9.0)
    assert resolved["streamline_clim"] == (4.0, 44.0)
    assert resolved["plane_video_cfg"]["label"]["prefix"] == "plane="


def test_hybrid_component_filter_keeps_multiple_group_components():
    mask = np.zeros((16, 16, 16), dtype=bool)
    mask[1:5, 1:5, 1:5] = True
    mask[8:12, 8:12, 8:11] = True
    mask[14:15, 14:15, 14:15] = True

    filtered = filter_connected_components(
        mask,
        np.ones(3, dtype=float),
        mode="hybrid",
        min_volume_mm3=10.0,
        rel_min_ratio=0.01,
    )

    assert int(np.sum(filtered)) == (4 * 4 * 4) + (4 * 4 * 3)
    assert bool(filtered[2, 2, 2]) is True
    assert bool(filtered[9, 9, 9]) is True
    assert bool(filtered[14, 14, 14]) is False


def test_pipeline_group_preprocess_keeps_multiple_components_under_hybrid_filter():
    ws = SimpleNamespace(
        segmask_raw=np.zeros((16, 16, 16), dtype=np.int16),
        multilabel_groups={},
        resolution=np.ones(3, dtype=float),
        skeleton_params=SkeletonParams(
            remove_small_cc=True,
            min_cc_volume_mm3=10.0,
            cc_filter_mode="hybrid",
            cc_rel_min_ratio=0.01,
            label_groups={"vessel": {"labels": [1]}},
            single_label_group_name="vessel",
        ),
        time_count=lambda: 1,
        set_object_visible_by_data_key=lambda *args, **kwargs: None,
        remove_object_by_data_key=lambda *args, **kwargs: None,
        remove_objects_by_prefix=lambda *args, **kwargs: None,
        add_object=lambda *args, **kwargs: None,
    )
    ws.segmask_raw[1:5, 1:5, 1:5] = 1
    ws.segmask_raw[8:12, 8:12, 8:11] = 1
    ws.segmask_raw[14:15, 14:15, 14:15] = 1

    PipelineEngine().preprocess(ws)

    group_state = ws.multilabel_groups["vessel"]
    assert ws.group_order == ["vessel"]
    assert int(np.sum(group_state["clean_mask_3d"])) == (4 * 4 * 4) + (4 * 4 * 3)
    assert bool(group_state["clean_mask_3d"][2, 2, 2]) is True
    assert bool(group_state["clean_mask_3d"][9, 9, 9]) is True
    assert bool(group_state["clean_mask_3d"][14, 14, 14]) is False

def test_plane_video_label_style_uses_plane_render_config(monkeypatch, tmp_path):
    class DummyPlotter:
        def __init__(self):
            self.label_calls = []
            self.camera_position = None

        def add_mesh(self, *args, **kwargs):
            return None

        def add_point_labels(self, points, labels, **kwargs):
            self.label_calls.append((np.asarray(points), list(labels), dict(kwargs)))
            return None

        def add_text(self, *args, **kwargs):
            return None

        def render(self):
            return None

        def screenshot(self, return_img=True):
            assert return_img is True
            return np.zeros((4, 4, 3), dtype=np.uint8)

        def remove_actor(self, name):
            return None

        def close(self):
            return None

    plotter = DummyPlotter()

    class DummyPoly:
        n_points = 1
        bounds = (0.0, 1.0, 0.0, 1.0, 0.0, 1.0)

    ws = SimpleNamespace(
        origin=np.zeros(3, dtype=float),
        centerline_paths_smooth=[np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)],
        planes=[SimpleNamespace(center=np.array([0.5, 0.0, 0.0], dtype=float), normal=np.array([1.0, 0.0, 0.0], dtype=float), path_index=0, group_name="")],
        group_order=[],
        multilabel_groups={},
        skeleton_points=np.empty((0, 3), dtype=float),
        skeleton_params=SkeletonParams(),
    )

    monkeypatch.setattr("autoflow.rendering.videos._build_union_surface", lambda ws, smoothing_iteration=200: (None, DummyPoly()))
    monkeypatch.setattr("autoflow.rendering.videos._make_plotter", lambda window_size=None: plotter)
    monkeypatch.setattr("autoflow.rendering.videos._plane_mesh", lambda center_world, normal, size: SimpleNamespace(n_points=4))
    monkeypatch.setattr("autoflow.rendering.videos._path_polydata", lambda path_world: SimpleNamespace(n_points=len(path_world)))
    monkeypatch.setattr("autoflow.rendering.videos._orbit_camera", lambda poly, azimuth_deg, elevation_deg=0.0, distance_scale=1.0: [tuple([0.0, 0.0, 0.0]), tuple([0.0, 0.0, 0.0]), (0.0, 0.0, 1.0)])
    monkeypatch.setattr("autoflow.rendering.videos._write_video", lambda frames, out_path, fps=24: str(out_path))

    out = render_plane_rotation_video(
        ws,
        str(tmp_path),
        fps=12,
        n_frames=1,
        add_plane_idx=True,
        plane_video_cfg={
            "label": {
                "prefix": "P-",
                "font_size": 19,
                "text_color": "white",
                "shape_color": "#123456",
                "shape_opacity": 0.4,
            }
        },
    )

    assert out.endswith("planes_rotate.mp4")
    assert len(plotter.label_calls) == 1
    _points, labels, kwargs = plotter.label_calls[0]
    assert labels == ["P-0"]
    assert kwargs["font_size"] == 19
    assert kwargs["text_color"] == "white"
    assert kwargs["shape_color"] == "#123456"
    assert kwargs["shape_opacity"] == 0.4


def test_build_union_surface_falls_back_when_extract_surface_algorithm_kwarg_is_unsupported(monkeypatch):
    class DummySurface:
        n_points = 1

        def smooth(self, n_iter=0):
            return self

    class DummyMesh:
        n_cells = 1

        def __init__(self):
            self.calls = []

        def threshold(self, value):
            assert value == 0.1
            return self

        def extract_surface(self, **kwargs):
            self.calls.append(dict(kwargs))
            if "algorithm" in kwargs:
                raise TypeError("extract_surface() got an unexpected keyword argument algorithm")
            return DummySurface()

    dummy_mesh = DummyMesh()
    monkeypatch.setattr("autoflow.rendering.videos.create_uniform_grid", lambda mask3d, resolution, origin=None: dummy_mesh)

    ws = SimpleNamespace(
        segmask_binary=None,
        segmask_3d=np.ones((2, 2, 2), dtype=bool),
        resolution=np.ones(3, dtype=float),
        origin=np.zeros(3, dtype=float),
    )

    mesh, surf = _build_union_surface(ws, smoothing_iteration=0)

    assert mesh is dummy_mesh
    assert surf is not None
    assert dummy_mesh.calls == [{"algorithm": "dataset_surface"}, {}]


def test_build_plane_records_include_label_name_from_label_map():
    ws = SimpleNamespace(
        origin=np.zeros(3, dtype=float),
        resolution=np.ones(3, dtype=float),
        segmask_raw=np.array(
            [
                [[1, 2], [1, 2]],
                [[1, 2], [1, 2]],
            ],
            dtype=np.int16,
        ),
        planes=[
            SimpleNamespace(
                center=np.array([0.0, 0.0, 0.0], dtype=float),
                normal=np.array([1.0, 0.0, 0.0], dtype=float),
                label=1,
                path_index=0,
                distance=0.0,
                group_name="aorta_systemic_branches",
                metrics={},
            ),
            SimpleNamespace(
                center=np.array([0.0, 0.0, 1.0], dtype=float),
                normal=np.array([1.0, 0.0, 0.0], dtype=float),
                label=2,
                path_index=1,
                distance=5.0,
                group_name="pulmonary_arteries",
                metrics={},
            ),
        ],
        path_info=[],
        multilabel_groups={
            "aorta_systemic_branches": {"labels": [1]},
            "pulmonary_arteries": {"labels": [2, 8, 9]},
        },
        segmentation=SimpleNamespace(label_names={"1": "Ascending Aorta", "2": "Main Pulmonary Artery"}),
        label_params=SimpleNamespace(label_map={"AAO": 1, "MPA": 2}),
        skeleton_params=SimpleNamespace(label_map={"AAO": 1, "MPA": 2}),
    )

    payload = build_plane_records(ws)

    assert payload[0]["label_name"] == "AAO"
    assert payload[1]["label_name"] == "MPA"


def test_default_nnunet_model_folder_prefers_partbalanced():
    model_dir = default_nnunet_model_folder()
    assert model_dir.name == "nnUNetTrainerPartBalanced__nnUNetPlans__3d_fullres_iso1mm"
    assert model_dir.is_dir()


def test_save_pwv_h5_exports_group_payloads(tmp_path):
    out = tmp_path / "pwv.h5"
    results = [{
        "name": "aorta",
        "status": "ok",
        "pwv_m_s": 4.2,
        "position_mm": [0.0, 10.0],
        "arrival_time_ms": [0.0, 2.0],
        "planes": [{"distance_mm": 0.0, "waveform": [1.0, 2.0], "flowrate_mL_s": [3.0, 4.0]}],
    }]
    save_pwv_h5(results, str(out), source_path="case.h5", source_format="normalized_h5", source_group="Case001")
    assert out.is_file()
    with h5py.File(out, "r") as handle:
        assert handle.attrs["schema"] == "autoflow.pwv.v1"
        assert int(handle.attrs["group_count"]) == 1
        assert "payload_json" in handle
        grp = handle["group_0000"]
        assert grp.attrs["name"] == "aorta"
        assert "payload_json" in grp
        assert np.allclose(np.asarray(grp["position_mm"][()], dtype=float), [0.0, 10.0])
        assert np.allclose(np.asarray(grp["arrival_time_ms"][()], dtype=float), [0.0, 2.0])
        assert "planes" in grp
