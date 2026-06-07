import argparse
import sys


_GUI_IMPORTS = {"PyQt5", "pyvistaqt", "matplotlib", "pyvista", "vtk"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the AutoFlow GUI.")
    parser.add_argument("--config-dir", default=None, help="Directory containing per-module JSON configs for GUI defaults.")
    return parser


def launch_gui(config_dir=None) -> None:
    try:
        from .app import main as app_main
    except ModuleNotFoundError as exc:
        module_name = exc.name or ""
        base_name = module_name.split(".", 1)[0]
        if base_name in _GUI_IMPORTS:
            raise SystemExit(
                "GUI dependencies are not installed. Run `pip install \".[gui]\"` in the repo root first."
            ) from exc
        raise
    app_main(config_dir=config_dir)


def main() -> None:
    args = build_parser().parse_args()
    try:
        launch_gui(config_dir=args.config_dir)
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
