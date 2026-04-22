__all__ = [
    "AutoFlowConfig",
    "build_workspace",
    "run_batch",
    "run_case",
    "launch_gui",
]

__version__ = "0.1.0"


def __getattr__(name):
    if name in {"AutoFlowConfig", "build_workspace", "run_batch", "run_case"}:
        from .api import AutoFlowConfig, build_workspace, run_batch, run_case

        return {
            "AutoFlowConfig": AutoFlowConfig,
            "build_workspace": build_workspace,
            "run_batch": run_batch,
            "run_case": run_case,
        }[name]
    if name == "launch_gui":
        from .gui import launch_gui

        return launch_gui
    raise AttributeError(f"module 'autoflow' has no attribute {name!r}")
