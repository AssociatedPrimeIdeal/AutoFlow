import sys


_GUI_IMPORTS = {"PyQt5", "pyvistaqt", "matplotlib", "pyvista", "vtk"}


def launch_gui() -> None:
    try:
        from app import main as app_main
    except ModuleNotFoundError as exc:
        module_name = exc.name or ""
        base_name = module_name.split(".", 1)[0]
        if base_name in _GUI_IMPORTS:
            raise SystemExit(
                "GUI dependencies are not installed. Run `pip install \".[gui]\"` in the repo root first."
            ) from exc
        raise
    app_main()


def main() -> None:
    try:
        launch_gui()
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
