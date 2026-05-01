"""Compatibility re-exports for the main GUI application."""

__all__ = ["MainWindow", "main"]


def __getattr__(name):
    if name in {"MainWindow", "main"}:
        from .ui.app import MainWindow, main

        return {
            "MainWindow": MainWindow,
            "main": main,
        }[name]
    raise AttributeError(f"module 'autoflow.app' has no attribute {name!r}")
