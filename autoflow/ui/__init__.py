__all__ = [
    "GraphEditor",
    "MainWindow",
    "OrthoViewer",
    "PlaneEditor",
    "SceneController",
    "SkeletonEditor",
    "launch_gui",
    "main",
]


def __getattr__(name):
    if name in {"MainWindow", "main"}:
        from .app import MainWindow, main

        return {
            "MainWindow": MainWindow,
            "main": main,
        }[name]
    if name in {"GraphEditor", "PlaneEditor", "SkeletonEditor"}:
        from .editors import GraphEditor, PlaneEditor, SkeletonEditor

        return {
            "GraphEditor": GraphEditor,
            "PlaneEditor": PlaneEditor,
            "SkeletonEditor": SkeletonEditor,
        }[name]
    if name == "launch_gui":
        from .launcher import launch_gui

        return launch_gui
    if name == "OrthoViewer":
        from .ortho_viewer import OrthoViewer

        return OrthoViewer
    if name == "SceneController":
        from .viewer import SceneController

        return SceneController
    raise AttributeError(f"module 'autoflow.ui' has no attribute {name!r}")
