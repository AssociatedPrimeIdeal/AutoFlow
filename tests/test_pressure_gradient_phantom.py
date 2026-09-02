import json
from pathlib import Path
import tempfile

import h5py
import numpy as np
import pytest

from autoflow import AutoFlowConfig, run_batch
from autoflow.algorithms.metrics import _periodic_central_difference, compute_centerline_pressure_profiles


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
REL_TOL = 0.05
PLANE_SPACING_MM = 15.0


def test_pressure_temporal_derivative_wraps_first_and_last_phases():
    phase = np.arange(8, dtype=np.float32)
    waveform = np.sin(2.0 * np.pi * phase / 8.0)
    values = waveform.reshape(1, 1, 1, 8, 1)
    derivative = _periodic_central_difference(values, 0.1)
    expected = (np.roll(waveform, -1) - np.roll(waveform, 1)) / 0.2

    assert derivative.reshape(-1) == pytest.approx(expected)
    assert abs(float(derivative[0, 0, 0, 0, 0])) > 0.0
    assert abs(float(derivative[0, 0, 0, -1, 0])) > 0.0


def test_centerline_pressure_sampling_uses_local_coordinates_with_nonzero_origin():
    pressure = np.zeros((5, 3, 3, 2), dtype=np.float32)
    pressure[:] = np.arange(5, dtype=np.float32).reshape(5, 1, 1, 1)
    path_local = np.array([[0.0, 1.0, 1.0], [4.0, 1.0, 1.0]], dtype=float)

    at_zero = compute_centerline_pressure_profiles(
        pressure, [path_local], (1.0, 1.0, 1.0), (0.0, 0.0, 0.0)
    )
    shifted = compute_centerline_pressure_profiles(
        pressure, [path_local], (1.0, 1.0, 1.0), (100.0, 200.0, 300.0)
    )

    assert shifted == at_zero
    assert shifted[0]["relative_pressure_Pa_t"][0] == pytest.approx([0.0, 4.0])


def _relative_error(measured: float, truth: float) -> float:
    denom = max(abs(float(truth)), 1e-12)
    return abs(float(measured) - float(truth)) / denom


def _mean_relative_error(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=float))) if values else 0.0


def _truth_plane_stats(flow_z_xyzt: np.ndarray, seg_xyzt: np.ndarray, z_idx: int, cell_area_mm2: float, rr_ms: float):
    flowrate_t = []
    meanv_t = []
    for tidx in range(seg_xyzt.shape[3]):
        mask_t = seg_xyzt[:, :, z_idx, tidx]
        vals = np.asarray(flow_z_xyzt[:, :, z_idx, tidx][mask_t], dtype=float)
        if vals.size == 0:
            flowrate_t.append(0.0)
            meanv_t.append(0.0)
            continue
        flowrate_t.append(float(np.sum(vals) * cell_area_mm2 / 100.0))
        meanv_t.append(float(np.mean(vals)))
    truth_netflow = abs(float(np.mean(flowrate_t)) * rr_ms / 1000.0)
    truth_meanv = float(np.mean(meanv_t)) if meanv_t else 0.0
    return truth_netflow, truth_meanv


def _run_phantom_p_case():
    output_root = Path(tempfile.mkdtemp(prefix="autoflow_phantom_P_", dir="/tmp"))
    cfg = AutoFlowConfig(
        inputs=[str(DATA_DIR / "phantom_P.h5")],
        output_dir=str(output_root),
        skip_derived=False,
        skip_plane_metrics=False,
        use_multithread=False,
        plane_mode="distance",
        cross_section_dist=PLANE_SPACING_MM,
        requested_metrics=["pg", "vortex"],
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


def test_phantom_p_plane_metrics_include_mean_velocity_error_and_valid_distance_planes():
    case_dir, metrics = _run_phantom_p_case()
    plane_pixelwise_path = case_dir / "plane_metrics_pixelwise.h5"
    plane_positions_path = case_dir / "plane_positions.json"
    planes_h5_path = case_dir / "planes.h5"
    summary_path = case_dir / "summary.json"
    pixelwise_npz_path = case_dir / "derived_metrics_pixelwise.npz"
    assert plane_pixelwise_path.is_file()
    assert plane_positions_path.is_file()
    assert planes_h5_path.is_file()
    assert summary_path.is_file()
    assert pixelwise_npz_path.is_file()

    plane_positions = json.loads(plane_positions_path.read_text(encoding="utf-8"))["planes"]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    with h5py.File(planes_h5_path, "r") as planes_h5:
        plane_keys = sorted(planes_h5.keys())
        assert plane_keys == [f"plane_{idx:04d}" for idx in range(len(plane_positions))]
        first_plane = planes_h5[plane_keys[0]]
        assert int(first_plane["plane_index"][()]) == 0
        assert np.allclose(np.asarray(first_plane["center_world"][()], dtype=float), np.asarray(plane_positions[0]["center_world"], dtype=float))
        assert int(first_plane["path_index"][()]) == int(plane_positions[0]["path_index"])
        assert "label_name" in first_plane
        assert first_plane["label_name"][()].decode("utf-8")
        assert "payload_json" in first_plane
    with np.load(pixelwise_npz_path) as pixelwise:
        assert pixelwise["vorticity"].shape[-1] == 3
        assert pixelwise["vorticity_magnitude"].ndim == 4
        assert pixelwise["q_criterion"].shape == pixelwise["swirling_strength"].shape
        assert pixelwise["vortex_support_mask"].dtype == np.uint8

    planes_json = json.loads((case_dir / "planes.json").read_text(encoding="utf-8"))
    assert len(planes_json) == len(plane_positions)
    assert all(str(item.get("label_name", "")).strip() for item in planes_json)
    assert len(plane_positions) == len(metrics)
    assert len(metrics) >= 1
    assert summary["pressure_method"] == "least_squares"
    assert summary["pressure_gradient_temporal_scheme"] == "periodic_central_difference"
    assert "pwv_h5_file" in summary
    assert "pwv_json_file" in summary


    centerline_profiles = list(summary.get("centerline_pressure_profiles", []) or [])
    assert centerline_profiles
    assert len(centerline_profiles) == int(summary.get("n_paths", len(centerline_profiles)))
    assert any(float(item.get("pressure_drop_peak_Pa", 0.0)) > 0.0 for item in centerline_profiles)
    assert any(len(item.get("pressure_drop_Pa_t", [])) > 1 for item in centerline_profiles)

    with np.load(pixelwise_npz_path) as pixelwise_npz:
        assert "relative_pressure" in pixelwise_npz.files
        assert "relative_pressure_peak" in pixelwise_npz.files
        assert "pressure_gradient" in pixelwise_npz.files
        assert "pressure_gradient_mag" in pixelwise_npz.files
        rel_pressure = np.asarray(pixelwise_npz["relative_pressure"], dtype=float)
        rel_peak = np.asarray(pixelwise_npz["relative_pressure_peak"], dtype=float)
        grad = np.asarray(pixelwise_npz["pressure_gradient"], dtype=float)
        assert rel_pressure.ndim == 4
        assert rel_peak.ndim == 3
        assert grad.ndim == 5 and grad.shape[-1] == 3
        assert np.isfinite(rel_pressure).any()
        assert float(np.nanmax(np.abs(rel_peak))) > 0.0

    with h5py.File(DATA_DIR / "phantom_P.h5", "r") as handle:
        truth = handle["truth"]
        flow_z = np.asarray(truth["flow_xyzt3_cm_s"][..., 2], dtype=float)
        seg = np.asarray(truth["segmentation_xyzt"][()], dtype=bool)
        origin = np.asarray(truth["origin_mm"][()], dtype=float)
        spacing = np.asarray(truth["resolution_mm"][()], dtype=float)
        cell_area_mm2 = float(spacing[0]) * float(spacing[1])
        rr_ms = float(handle["RR"][()])

        valid_metrics = []
        flow_errors = []
        meanv_errors = []
        with h5py.File(plane_pixelwise_path, "r") as plane_h5:
            plane_keys = sorted(plane_h5["planes"].keys())
            assert len(plane_keys) == len(metrics)
            for plane_key, metric, plane_entry in zip(plane_keys, metrics, plane_positions):
                times_grp = plane_h5["planes"][plane_key]["timepoints"]
                if len(times_grp) == 0:
                    continue
                first_key = sorted(times_grp.keys())[0]
                if "pressure_gradient_vec_Pa_m" not in times_grp[first_key]:
                    continue
                z_world = float(plane_entry["center_world"][2])
                z_idx = int(round((z_world - float(origin[2])) / float(spacing[2])))
                z_idx = int(np.clip(z_idx, 0, flow_z.shape[2] - 1))
                truth_netflow, truth_meanv = _truth_plane_stats(flow_z, seg, z_idx, cell_area_mm2, rr_ms)
                flow_errors.append(_relative_error(metric["netflow_mL_beat"], truth_netflow))
                meanv_errors.append(_relative_error(metric["meanv_cm_s"], truth_meanv))
                valid_metrics.append(metric)

    assert valid_metrics
    assert len(valid_metrics) == len(metrics)
    assert _mean_relative_error(flow_errors) < REL_TOL
    assert _mean_relative_error(meanv_errors) < REL_TOL
    for metric in valid_metrics:
        assert metric["distance"] >= 0.0
        assert metric["netflow_mL_beat"] > 0.0
        assert metric["meanv_cm_s"] > 0.0
        assert metric["peakv_cm_s"] > 0.0
        assert metric["peakv_cm_s"] >= metric["meanv_cm_s"]
        assert "relative_pressure_mean_Pa" in metric
        assert metric["relative_pressure_peak_Pa"] >= metric["relative_pressure_mean_Pa"]
        assert "pressure_gradient_mag_mean_Pa_m" in metric
        assert metric["pressure_gradient_mag_mean_Pa_m"] > 0.0
        assert metric["pressure_gradient_mag_peak_Pa_m"] >= metric["pressure_gradient_mag_mean_Pa_m"]
