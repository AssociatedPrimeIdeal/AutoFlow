import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from autoflow import AutoFlowConfig, run_batch


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
PHANTOM_CASES = ("phantom_S", "phantom_U", "phantom_Y")
EXPECTED_PIXELWISE_KEYS = {"origin", "spacing", "tke", "tke_time", "wss"}
EXPECTED_PHANTOM_DISTANCE_BASELINE = {
    "phantom_S": {
        "n_paths": 1,
        "n_planes": 5,
        "n_forks": 0,
        "path_ic": {"0": 1.0},
        "fork_ic": {},
        "netflow_mL_beat": [
            16.04088137052839,
            16.04088137052839,
            16.04088137052839,
            16.04088137052839,
            16.04088137052839,
        ],
        "peakv_cm_s": [
            79.99999999992,
            79.99999999992,
            79.99999999992,
            79.99999999992,
            79.99999999992,
        ],
        "meanv_cm_s": [
            22.003952497295465,
            22.003952497295465,
            22.003952497295465,
            22.003952497295465,
            22.003952497295465,
        ],
        "reflux_fraction": [
            0.0022392402587455857,
            0.0022392402587455857,
            0.0022392402587455857,
            0.0022392402587455857,
            0.0022392402587455857,
        ],
    },
    "phantom_U": {
        "n_paths": 1,
        "n_planes": 8,
        "n_forks": 0,
        "path_ic": {"0": 0.9954934356203274},
        "fork_ic": {},
        "netflow_mL_beat": [
            16.04088137052839,
            16.04088137052839,
            16.040881381705866,
            16.18966765452808,
            16.216262234630666,
            16.181039532153374,
            16.04088137052839,
            16.04088137052839,
        ],
        "peakv_cm_s": [
            79.99999999992,
            79.99999999992,
            79.23234740849787,
            79.9332685476675,
            79.38889387140485,
            79.82523981007421,
            79.99999999992,
            79.99999999992,
        ],
        "meanv_cm_s": [
            22.003952497295465,
            22.003952497295465,
            21.79281011182328,
            23.304847079073053,
            23.706846418754395,
            23.655460631153808,
            22.003952497295465,
            22.003952497295465,
        ],
        "reflux_fraction": [
            0.0022392402587455857,
            0.0022392402587455857,
            0.002239240488770073,
            0.0,
            0.0,
            0.0,
            0.0022392402587455857,
            0.0022392402587455857,
        ],
    },
    "phantom_Y": {
        "n_paths": 3,
        "n_planes": 7,
        "n_forks": 1,
        "path_ic": {"0": 1.0, "1": 0.9989900747351891, "2": 0.9986394360071724},
        "fork_ic": {"0": 0.9888506466012691},
        "netflow_mL_beat": [
            16.04088137052839,
            16.04088137052839,
            16.04088137052839,
            8.116032203094532,
            8.132441947654,
            8.107508141067772,
            8.085476549161838,
        ],
        "peakv_cm_s": [
            79.99999999992,
            79.99999999992,
            79.99999999992,
            39.92927948437385,
            40.00000268713704,
            39.929279484373836,
            40.00000268713704,
        ],
        "meanv_cm_s": [
            22.003952497295465,
            22.003952497295465,
            22.003952497295465,
            12.057293966466908,
            11.307682761652485,
            11.470991211132334,
            12.101338858921398,
        ],
        "reflux_fraction": [
            0.0022392402587455857,
            0.0022392402587455857,
            0.0022392402587455857,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
    },
}


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    repo_root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = repo_root if not existing else os.pathsep.join((repo_root, existing))
    return subprocess.run(
        [sys.executable, "-m", "autoflow.cli", *args],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _assert_case_matches_distance_baseline(output_dir: Path, case_name: str) -> None:
    expected = EXPECTED_PHANTOM_DISTANCE_BASELINE[case_name]
    case_dir = output_dir / case_name
    summary_path = case_dir / "summary.json"
    plane_metrics_path = case_dir / "plane_metrics.json"
    plane_qc_path = case_dir / "plane_qc.json"
    pixelwise_path = case_dir / "derived_metrics_pixelwise.npz"

    assert summary_path.is_file()
    assert plane_metrics_path.is_file()
    assert plane_qc_path.is_file()
    assert pixelwise_path.is_file()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metrics = json.loads(plane_metrics_path.read_text(encoding="utf-8"))
    plane_qc = json.loads(plane_qc_path.read_text(encoding="utf-8"))

    assert summary["input"].endswith(f"{case_name}.h5")
    assert summary["output_dir"] == str(case_dir)
    assert summary["n_paths"] == expected["n_paths"]
    assert summary["n_planes"] == expected["n_planes"]
    assert summary["n_forks"] == expected["n_forks"]
    assert len(metrics) == expected["n_planes"]
    assert set(summary["pixelwise_export"]) == EXPECTED_PIXELWISE_KEYS
    assert plane_qc["path_ic"] == pytest.approx(expected["path_ic"])
    assert plane_qc["fork_ic"] == pytest.approx(expected["fork_ic"])
    assert [m["netflow_mL_beat"] for m in metrics] == pytest.approx(expected["netflow_mL_beat"])
    assert [m["peakv_cm_s"] for m in metrics] == pytest.approx(expected["peakv_cm_s"])
    assert [m["meanv_cm_s"] for m in metrics] == pytest.approx(expected["meanv_cm_s"])
    assert [m["reflux_fraction"] for m in metrics] == pytest.approx(expected["reflux_fraction"])


def test_run_batch_smoke_on_phantom_data(tmp_path):
    inputs = [str(DATA_DIR / f"{case}.h5") for case in PHANTOM_CASES]
    output_dir = tmp_path / "smoke_outputs"

    config = AutoFlowConfig(
        inputs=inputs,
        output_dir=str(output_dir),
        skip_derived=True,
        skip_plane_metrics=True,
        use_multithread=False,
        use_center_plane=True,
        make_plane_video=False,
        make_wss_video=False,
        make_streamlines_video=False,
        make_tke_video=False,
    )

    results, last_case_out = run_batch(config)

    assert len(results) == len(PHANTOM_CASES)
    assert last_case_out == str(output_dir / PHANTOM_CASES[-1])

    batch_report = output_dir / "batch_report.json"
    time_summary = output_dir / "time_summary.txt"
    assert batch_report.is_file()
    assert time_summary.is_file()

    report_items = json.loads(batch_report.read_text(encoding="utf-8"))
    assert len(report_items) == len(PHANTOM_CASES)

    for result, case_name in zip(results, PHANTOM_CASES):
        assert result["status"] == "ok"
        assert Path(result["file"]).name == f"{case_name}.h5"

        case_dir = output_dir / case_name
        summary_path = case_dir / "summary.json"
        plane_positions_path = case_dir / "plane_positions.json"
        planes_path = case_dir / "planes.json"

        assert summary_path.is_file()
        assert plane_positions_path.is_file()
        assert planes_path.is_file()

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        assert summary["input"].endswith(f"{case_name}.h5")
        assert summary["output_dir"] == str(case_dir)
        assert summary["n_paths"] >= 1
        assert summary["n_planes"] == summary["n_paths"]
        assert summary["plane_metrics"] == []
        assert summary["plane_qc"] == {}
        assert summary["videos"] == {}
        assert summary["pixelwise_export"] == {}

    assert [item["status"] for item in report_items] == ["ok", "ok", "ok"]


def test_cli_phantom_distance_mode_matches_regression_baseline(tmp_path):
    output_dir = tmp_path / "phantom_cli_regression"

    result = _run_cli(
        str(DATA_DIR / "phantom_S.h5"),
        str(DATA_DIR / "phantom_U.h5"),
        str(DATA_DIR / "phantom_Y.h5"),
        "--output-dir",
        str(output_dir),
        "--plane-by-distance",
        "--cross-section-dist",
        "15",
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert "Found 3 file(s) to process." in result.stdout
    assert "Done: 3/3 succeeded." in result.stdout

    batch_report = output_dir / "batch_report.json"
    time_summary = output_dir / "time_summary.txt"
    assert batch_report.is_file()
    assert time_summary.is_file()

    report_items = json.loads(batch_report.read_text(encoding="utf-8"))
    assert [item["status"] for item in report_items] == ["ok", "ok", "ok"]
    assert [Path(item["file"]).name for item in report_items] == ["phantom_S.h5", "phantom_U.h5", "phantom_Y.h5"]

    for case_name in PHANTOM_CASES:
        _assert_case_matches_distance_baseline(output_dir, case_name)
