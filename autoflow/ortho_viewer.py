"""Compatibility re-exports for the ortho viewer."""

__all__ = ["OrthoViewer"]


def __getattr__(name):
    if name == "OrthoViewer":
        from .ui.ortho_viewer import OrthoViewer

        return OrthoViewer
    raise AttributeError(f"module 'autoflow.ortho_viewer' has no attribute {name!r}")
