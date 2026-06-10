import json
from pathlib import Path
import tempfile

import h5py
import numpy as np

from autoflow import AutoFlowConfig, run_batch


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
REL_TOL = 0.05
PLANE_SPACING_MM = 15.0


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
        use_center_plane=False,
        cross_section_dist=PLANE_SPACING_MM,
        requested_metrics=["pg"],
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
    assert plane_pixelwise_path.is_file()
    assert plane_positions_path.is_file()

    plane_positions = json.loads(plane_positions_path.read_text(encoding="utf-8"))["planes"]
    assert len(plane_positions) == len(metrics)
    assert len(metrics) >= 1

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
    assert len(valid_metrics) < len(metrics)
    assert _mean_relative_error(flow_errors) < REL_TOL
    assert _mean_relative_error(meanv_errors) < REL_TOL
    for metric in valid_metrics:
        assert metric["distance"] >= 0.0
        assert metric["netflow_mL_beat"] > 0.0
        assert metric["meanv_cm_s"] > 0.0
        assert metric["peakv_cm_s"] > 0.0
        assert metric["peakv_cm_s"] >= metric["meanv_cm_s"]
        assert "pressure_gradient_mag_mean_Pa_m" in metric
        assert metric["pressure_gradient_mag_mean_Pa_m"] > 0.0
        assert metric["pressure_gradient_mag_peak_Pa_m"] >= metric["pressure_gradient_mag_mean_Pa_m"]
