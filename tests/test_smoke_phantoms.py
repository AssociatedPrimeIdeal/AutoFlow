import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import h5py
import numpy as np

from autoflow import AutoFlowConfig, run_batch
from autoflow.algorithms.segmentation import generate_nnunet_auto_segmentation


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
        use_center_plane=False,
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
        for metric in metrics:
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
    seg, provenance = generate_nnunet_auto_segmentation(
        mag=mag,
        flow=flow,
        resolution=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        model_folder=str(model_dir),
        device="cpu",
        progress_callback=events.append,
    )

    assert seg.shape == (2, 2, 2, 3)
    assert provenance["device"] == "cpu"
    stages = [event["stage"] for event in events]
    assert stages[0] == "autoseg_start"
    assert "autoseg_model_ready" in stages
    assert "autoseg_prepare_inputs" in stages
    assert "autoseg_run_inference" in stages
    assert "autoseg_read_prediction" in stages
    assert stages[-1] == "autoseg_finalize"
    assert all("elapsed_sec" in event for event in events)
