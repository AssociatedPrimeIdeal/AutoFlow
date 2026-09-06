import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


_GUI_IMPORTS = {"PySide6", "pyqtgraph", "pyvistaqt", "matplotlib", "pyvista", "vtk"}
_INTERNAL_NNUNET_FLAG = "--autoflow-internal-nnunet-predict"


def _is_forwarded_x11(display: str) -> bool:
    host = str(display or "").strip().split(":", 1)[0].lower()
    return host in {"localhost", "127.0.0.1", "::1"}


def _display_is_reachable(display: str) -> bool:
    checker = shutil.which("xdpyinfo")
    if checker is None:
        return True
    environment = os.environ.copy()
    environment["DISPLAY"] = str(display)
    try:
        result = subprocess.run(
            [checker],
            env=environment,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=2.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def _reachable_local_displays() -> list[str]:
    socket_dir = Path("/tmp/.X11-unix")
    if not socket_dir.is_dir():
        return []
    candidates = []
    for socket_path in sorted(socket_dir.glob("X*")):
        suffix = socket_path.name[1:]
        if suffix.isdigit():
            display = f":{int(suffix)}"
            if _display_is_reachable(display):
                candidates.append(display)
    return candidates


def validate_display() -> None:
    if not sys.platform.startswith("linux"):
        return
    platform = str(os.environ.get("QT_QPA_PLATFORM", "")).strip().lower()
    if platform in {"offscreen", "minimal", "minimalegl", "vnc"}:
        return
    if os.environ.get("WAYLAND_DISPLAY") and platform != "xcb":
        return
    display = str(os.environ.get("DISPLAY", "")).strip()
    forwarded = bool(display) and (
        _is_forwarded_x11(display) or bool(os.environ.get("SSH_CONNECTION"))
    )
    if forwarded:
        os.environ["AUTOFLOW_SSH_RENDERING"] = "1"
        os.environ["VTK_DEFAULT_OPENGL_WINDOW"] = "vtkOSOpenGLRenderWindow"
        os.environ["QT_X11_NO_MITSHM"] = "1"
        return
    if display and _display_is_reachable(display):
        return
    local_displays = _reachable_local_displays()
    current = display or "<unset>"
    hint = ""
    if local_displays:
        hint = f" Available local display(s): {', '.join(local_displays)}; for example, run `DISPLAY={local_displays[0]} autoflow-gui`."
    raise SystemExit(
        f"Cannot connect to DISPLAY={current}. Qt would abort before opening the GUI.{hint} "
        "For SSH forwarding, reconnect with `ssh -Y <host>` and verify `xdpyinfo` succeeds."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the AutoFlow GUI.")
    parser.add_argument("--config-dir", default=None, help="Directory containing per-module JSON configs for GUI defaults.")
    return parser


def _run_internal_nnunet_predictor() -> None:
    sys.argv.pop(1)
    from nnunetv2.inference.predict_from_raw_data import predict_entry_point_modelfolder

    predict_entry_point_modelfolder()


def launch_gui(config_dir=None) -> None:
    validate_display()
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
    if len(sys.argv) > 1 and sys.argv[1] == _INTERNAL_NNUNET_FLAG:
        _run_internal_nnunet_predictor()
        return
    args = build_parser().parse_args()
    try:
        launch_gui(config_dir=args.config_dir)
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
