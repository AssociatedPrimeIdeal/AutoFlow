"""Compatibility re-exports for UI viewer components."""

__all__ = ["SceneController"]


def __getattr__(name):
    if name == "SceneController":
        from .ui.viewer import SceneController

        return SceneController
    raise AttributeError(f"module 'autoflow.viewer' has no attribute {name!r}")
