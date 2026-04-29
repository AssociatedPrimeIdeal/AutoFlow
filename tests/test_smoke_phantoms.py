import json
from pathlib import Path

from autoflow import AutoFlowConfig, run_batch


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
PHANTOM_CASES = ("phantom_S", "phantom_U", "phantom_Y")


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
