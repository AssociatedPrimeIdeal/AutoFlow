import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from autoflow import AutoFlowConfig, run_batch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_INPUT = REPO_ROOT / "data" / "demo_data.h5"
EXPECTED_PIXELWISE_KEYS = {"origin", "spacing", "tke", "tke_time", "wss"}


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(REPO_ROOT) if not existing else os.pathsep.join((str(REPO_ROOT), existing))
    return subprocess.run(
        [sys.executable, "-m", "autoflow.cli", *args],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _assert_demo_outputs(output_root: Path, expected_video_keys: set[str]) -> dict:
    case_dir = output_root / "demo_data"
    summary_path = case_dir / "summary.json"
    plane_positions_path = case_dir / "plane_positions.json"
    planes_path = case_dir / "planes.json"
    plane_metrics_path = case_dir / "plane_metrics.json"
    plane_qc_path = case_dir / "plane_qc.json"
    pixelwise_path = case_dir / "derived_metrics_pixelwise.npz"
    batch_report_path = output_root / "batch_report.json"
    time_summary_path = output_root / "time_summary.txt"

    for path in (
        summary_path,
        plane_positions_path,
        planes_path,
        plane_metrics_path,
        plane_qc_path,
        pixelwise_path,
        batch_report_path,
        time_summary_path,
    ):
        assert path.is_file(), path

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["input"].endswith("data/demo_data.h5")
    assert summary["output_dir"] == str(case_dir)
    assert summary["n_planes"] >= 1
    assert summary["n_paths"] >= 1
    assert summary["n_forks"] >= 1
    assert len(summary["plane_metrics"]) == summary["n_planes"]
    assert set(summary["pixelwise_export"]) == EXPECTED_PIXELWISE_KEYS
    assert Path(summary["plane_positions_file"]).is_file()
    assert summary["reused_planes_file"] == ""

    batch_report = json.loads(batch_report_path.read_text(encoding="utf-8"))
    assert len(batch_report) == 1
    assert batch_report[0]["status"] == "ok"
    assert batch_report[0]["file"].endswith("data/demo_data.h5")

    assert set(summary["videos"]) == expected_video_keys
    for key in expected_video_keys:
        video_path = Path(summary["videos"][key])
        assert video_path.is_file(), video_path
        assert video_path.suffix in {".gif", ".mp4"}

    pixelwise = np.load(pixelwise_path)
    assert set(pixelwise.files) == EXPECTED_PIXELWISE_KEYS

    return summary


def test_cli_demo_command_processes_demo_data(tmp_path):
    output_root = tmp_path / "cli_demo"

    result = _run_cli(str(DEMO_INPUT), "--output-dir", str(output_root))

    assert result.returncode == 0, result.stderr or result.stdout
    assert "Found 1 file(s) to process." in result.stdout
    assert "Done: 1/1 succeeded." in result.stdout

    summary = _assert_demo_outputs(output_root, expected_video_keys=set())
    assert summary["videos"] == {}


def test_library_demo_notebook_workflow_processes_demo_data(tmp_path):
    output_root = tmp_path / "library_demo"
    config = AutoFlowConfig(
        inputs=[str(DEMO_INPUT)],
        output_dir=str(output_root),
        skip_derived=False,
        use_multithread=True,
        reuse_planes="",
        use_center_plane=True,
        cross_section_dist=15.0,
        start_dist=5.0,
        end_dist=0.0,
        remove_small_cc=True,
        min_cc_volume=50.0,
        make_plane_video=False,
        make_wss_video=False,
        make_streamlines_video=False,
        make_tke_video=False,
        camera_view="right",
        camera_distance_scale=1.5,
        rotate_dynamic_video=True,
        dynamic_rotation_frames=180,
        dynamic_rotation_elevation_deg=10.0,
        dynamic_time_repeat=3,
        add_path_idx=False,
    )

    results, case_out = run_batch(config)

    assert len(results) == 1
    assert results[0]["status"] == "ok"
    assert case_out == str(output_root / "demo_data")

    summary = _assert_demo_outputs(output_root, expected_video_keys=set())
    assert summary["videos"] == {}
