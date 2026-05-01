import os
from pathlib import Path
import subprocess
import sys
import textwrap


REPO_ROOT = Path(__file__).resolve().parents[1]
GUI_MODULES = ("PyQt5", "pyvistaqt", "matplotlib")


def _run_python(code: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(REPO_ROOT) if not existing else os.pathsep.join((str(REPO_ROOT), existing))
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _block_optional_gui_imports(body: str) -> str:
    return textwrap.dedent(
        f"""
        import builtins

        _blocked = {GUI_MODULES!r}
        _real_import = builtins.__import__

        def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            base = name.split(".", 1)[0]
            if base in _blocked:
                raise ModuleNotFoundError(f"No module named '{{base}}'", name=base)
            return _real_import(name, globals, locals, fromlist, level)

        builtins.__import__ = _guarded_import

        {body}
        """
    )


def test_import_autoflow_module():
    import autoflow

    assert autoflow.__all__ == [
        "AutoFlowConfig",
        "build_workspace",
        "run_batch",
        "run_case",
        "launch_gui",
    ]


def test_import_public_api_symbols():
    from autoflow import AutoFlowConfig, build_workspace, run_batch, run_case

    assert AutoFlowConfig.__name__ == "AutoFlowConfig"
    assert build_workspace.__name__ == "build_workspace"
    assert run_case.__name__ == "run_case"
    assert run_batch.__name__ == "run_batch"


def test_build_workspace_applies_config():
    from autoflow import AutoFlowConfig, build_workspace

    config = AutoFlowConfig(
        use_center_plane=False,
        cross_section_dist=12.5,
        start_dist=1.5,
        end_dist=2.5,
        remove_small_cc=True,
        min_cc_volume=42.0,
        seed_ratio=0.03,
        tube_radius=0.12,
    )

    ws = build_workspace(config)

    assert ws.plane_gen_params.use_center_plane is False
    assert ws.plane_gen_params.cross_section_distance == 12.5
    assert ws.plane_gen_params.start_distance == 1.5
    assert ws.plane_gen_params.end_distance == 2.5
    assert ws.skeleton_params.remove_small_cc is True
    assert ws.skeleton_params.min_cc_volume_mm3 == 42.0
    assert ws.streamline_params.seed_ratio == 0.03
    assert ws.streamline_params.tube_radius == 0.12


def test_base_imports_do_not_require_optional_gui_dependencies():
    code = _block_optional_gui_imports(
        """
        import autoflow
        from autoflow import AutoFlowConfig, build_workspace, run_case, run_batch

        print(autoflow.__all__)
        print(AutoFlowConfig.__name__, build_workspace.__name__, run_case.__name__, run_batch.__name__)
        """
    )
    result = _run_python(code)
    assert result.returncode == 0, result.stderr or result.stdout


def test_refactor_modules_and_utils_compatibility_import():
    import autoflow.plane_io as plane_io
    import autoflow.processing as processing
    import autoflow.rendering as rendering
    import autoflow.reporting as reporting
    import autoflow.utils as utils

    assert utils.process_single is processing.process_single
    assert utils.resolve_reuse_plane_file is plane_io.resolve_reuse_plane_file
    assert utils.print_qc_summary is reporting.print_qc_summary
    assert utils.render_tke_video is rendering.render_tke_video


def test_core_and_ui_packages_reexport_legacy_entry_points():
    import autoflow.app as app
    import autoflow.core as core
    import autoflow.core.models as core_models
    import autoflow.core.pipeline as core_pipeline
    import autoflow.editors as editors
    import autoflow.gui as gui
    import autoflow.models as models
    import autoflow.ortho_viewer as ortho_viewer
    import autoflow.pipeline as pipeline
    import autoflow.ui as ui
    import autoflow.viewer as viewer

    assert models.Workspace is core_models.Workspace
    assert pipeline.PipelineEngine is core_pipeline.PipelineEngine
    assert viewer.SceneController is ui.SceneController
    assert ortho_viewer.OrthoViewer is ui.OrthoViewer
    assert editors.PlaneEditor is ui.PlaneEditor
    assert app.main is ui.main
    assert gui.launch_gui is ui.launch_gui
    assert core.PipelineEngine is core_pipeline.PipelineEngine


def test_launch_gui_reports_missing_optional_dependencies():
    code = _block_optional_gui_imports(
        """
        from autoflow.gui import launch_gui

        try:
            launch_gui()
        except SystemExit as exc:
            message = str(exc)
            print(message)
            if "GUI dependencies are not installed" not in message:
                raise SystemExit(message)
        else:
            raise SystemExit("launch_gui unexpectedly succeeded")
        """
    )
    result = _run_python(code)
    assert result.returncode == 0, result.stderr or result.stdout
    assert "GUI dependencies are not installed" in result.stdout
