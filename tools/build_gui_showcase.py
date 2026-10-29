"""Capture the real Qt GUI and compose the README's animated feature cards.

Run on a GPU server in autoflow311; see docs/en/developer/gui-showcase.md.
The bundled demo H5 is read only. All intermediate files stay in --work-dir.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.update(QT_QPA_PLATFORM="offscreen", AUTOFLOW_SSH_RENDERING="1",
                  PYVISTA_OFF_SCREEN="true", VTK_DEFAULT_OPENGL_WINDOW="vtkEGLRenderWindow",
                  OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4",
                  PYTHONDONTWRITEBYTECODE="1")


def prepare(work):
    import json
    import joblib
    import numpy as np
    from autoflow.core.models import Workspace, StepId
    from autoflow.core.pipeline import PipelineEngine
    from autoflow.config import apply_config_bundle_to_workspace, load_config_bundle

    ws = Workspace()
    apply_config_bundle_to_workspace(ws, load_config_bundle(ROOT / "configs"))
    ws.paths.flow_path = ws.paths.segmask_path = str(ROOT / "data/demo_data.h5")
    ws.paths.output_dir = str(work / "analysis")
    ws.segmentation.write_auto_cache = False
    ws.loader_params.background_phase_correction.cache_write = False
    engine = PipelineEngine()
    engine.load_data(ws, print, reuse_existing_corr=False, apply_background_correction=False)
    engine.preprocess(ws)
    for step in (StepId.GENERATE_PCMRA, StepId.GENERATE_SKELETON,
                 StepId.GENERATE_GRAPH, StepId.GENERATE_PLANES,
                 StepId.COMPUTE_PLANE_METRICS, StepId.GENERATE_STREAMLINES,
                 StepId.COMPUTE_DERIVED_METRICS):
        result = engine.run_step(ws, step, print)
        print(step.name, result, flush=True)
        if not result.success:
            raise RuntimeError(result.message)
    for obj in ws.scene_objects.values():
        obj.actor = obj.label_actor = None
    joblib.dump(ws, work / "workspace.joblib", compress=1)
    info = {"shape": list(ws.flow_raw.shape), "planes": len(ws.planes),
            "labels": np.unique(ws.segmask_3d).tolist(),
            "objects": [{"key": obj.data_key, "kind": obj.kind.value,
                         "visible": obj.visible} for obj in ws.scene_objects.values()]}
    (work / "workspace.json").write_text(json.dumps(info, indent=2))
    print(info, flush=True)


def capture(work):
    import hashlib
    import json
    import math
    import joblib
    import numpy as np
    from PySide6 import QtCore, QtWidgets, QtTest
    from autoflow.ui.app import MainWindow, configure_high_dpi, apply_application_theme
    from autoflow.ui.action_settings import ActionSettingsDialog
    from autoflow.core.models import Workspace, StepId
    from autoflow.config import apply_config_bundle_to_workspace

    configure_high_dpi()
    # Keep demo capture preferences separate from the user's real GUI session.
    QtCore.QSettings.setDefaultFormat(QtCore.QSettings.IniFormat)
    QtCore.QSettings.setPath(QtCore.QSettings.IniFormat, QtCore.QSettings.UserScope,
                            str(work / "preferences"))
    app = QtWidgets.QApplication([])
    apply_application_theme(app)
    window = MainWindow(str(ROOT / "configs"))
    window.resize(1800, 1040)
    window.show()
    ws = joblib.load(work / "workspace.joblib")
    for obj in ws.scene_objects.values():
        obj.actor = obj.label_actor = None
        obj.visible = obj.data_key.startswith(("segmask_group", "smooth_path", "plane_"))
        if obj.data_key.startswith("segmask_group"):
            obj.opacity = .35
        if obj.data_key.startswith("smooth_path"):
            obj.color, obj.line_width = "#78dce8", 3.0
    window.workspace = window.scene.workspace = window.ortho_viewer.workspace = ws
    window.ortho_viewer.reset_state()
    window.ortho_viewer.update_slider_ranges()
    window.scene.reset_display_reference()
    window.scene.reset_scene()
    window.workflow_tabs.setCurrentIndex(4)
    window._refresh_all()
    window.scene.reset_camera()
    window.scene.flush_render()
    app.processEvents()
    # A frontal oblique camera keeps the aortic arch legible in all card sizes.
    window.plotter.view_xz()
    window.plotter.camera.Azimuth(18)
    window.plotter.camera.Elevation(8)
    window.plotter.reset_camera()
    base_camera = [tuple(v) for v in window.plotter.camera_position]
    frames = work / "clips"
    frames.mkdir(exist_ok=True)
    manifest = {"source": "Current AutoFlow MainWindow, actual Qt widget captures",
                "input": "data/demo_data.h5", "phases": ws.time_count(),
                "data_sha256": hashlib.sha256((ROOT / "data/demo_data.h5").read_bytes()).hexdigest(),
                "note": "Completed demo workspace; loops demonstrate interaction, not processing time.",
                "fps": 8, "frames_per_clip": 48, "clips": {},
                "source_hashes": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                    for name in ("autoflow/ui/app.py", "autoflow/ui/ortho_viewer.py",
                                 "autoflow/ui/viewer.py", "autoflow/ui/plane_editing.py")}}

    def settle():
        app.processEvents()
        window.scene.flush_render()
        window.plotter._capture_frame()
        app.processEvents()

    def record(name, widget, update, count=48):
        folder = frames / name
        folder.mkdir(exist_ok=True)
        for i in range(count):
            update(i)
            settle()
            shot = widget().grab()
            if not shot.save(str(folder / f"{i:03d}.png")):
                raise RuntimeError("Unable to save " + name)
        manifest["clips"][name] = {"frames": count, "size": [shot.width(), shot.height()]}
        (work / "capture.json").write_text(json.dumps(manifest, indent=2))
        print("CAPTURED", name, flush=True)

    def phase(i):
        ws.current_t = i % ws.time_count()
        window._refresh_time_views()
        window._refresh_timeline()

    def orbit(i):
        phase(i)
        window.plotter.camera_position = base_camera
        window.plotter.camera.Azimuth(22 * math.sin(2 * math.pi * i / 48))

    record("workspace", lambda: window, orbit)
    window.grab().save(str(work / "workspace.png"))
    # Actual source slice browsing, including the image controls and segmentation overlay.
    window.workflow_tabs.setCurrentIndex(2)
    window._refresh_all()
    window.segmentation_dock.hide()
    shape = ws.flow_raw.shape[:3]
    center = np.array(np.argwhere(ws.segmask_3d).mean(axis=0), dtype=int)
    def slices(i):
        phase(i)
        window.ortho_viewer.slider_x.setValue(int(center[0]))
        window.ortho_viewer.slider_y.setValue(int(center[1]))
        window.ortho_viewer.slider_z.setValue(int(center[2] + 14 * math.sin(2 * math.pi * i / 48)))
        window.ortho_viewer._apply_pending_cursor()
    record("slices", lambda: window.ortho_viewer, slices)

    window.workflow_tabs.setCurrentIndex(3)
    window._refresh_all()
    window.segmentation_dock.hide()
    def layers(i):
        phase(i)
        for obj in ws.scene_objects.values():
            if obj.data_key.startswith("segmask_group"):
                obj.opacity = .12 + .55 * (.5 + .5 * math.sin(2 * math.pi * i / 48))
                window.scene._apply_basic_properties_only(obj)
        window.plotter.camera_position = base_camera
        window.plotter.camera.Azimuth(20 * math.sin(2 * math.pi * i / 48))
    record("geometry", lambda: window.plotter, layers)

    def select(index):
        window._selected_plane_index = index
        window.ortho_viewer.set_selected_plane(index)
        window._refresh_selection_info()
        window._sync_context_panels()

    window.workflow_tabs.setCurrentIndex(4)
    select(0)
    def plane_sequence(i):
        index = (i // 16) % len(ws.planes)
        if window._selected_plane_index != index:
            select(index)
        phase(i)
    record("planes", lambda: window.ortho_viewer, plane_sequence)
    select(0)
    window._show_measurements("plane_curve")
    window.pwv_dock.setFloating(True)
    window.pwv_dock.resize(520, 550)
    def curve(i):
        phase(i)
        window._refresh_analysis_panel()
        window.fig_pwv.tight_layout(pad=1.3)
        window.canvas_pwv.draw()
    record("curve", lambda: window.pwv_dock.widget(), curve)
    window.pwv_dock.setFloating(False)

    # Use the actual transactional controls; cancel the draft after capturing.
    window.check_plane_unlocked.setChecked(True)
    app.processEvents()
    edit_position = window.ortho_viewer.mapTo(window, QtCore.QPoint(0, 0))
    manifest["edit_view_rect"] = [edit_position.x(), edit_position.y(),
                                  edit_position.x() + window.ortho_viewer.width(),
                                  edit_position.y() + window.ortho_viewer.height()]
    origin = np.array(ws.planes[0].center, dtype=float)
    def edit(i):
        center_mm = origin.copy()
        center_mm[2] += 4 * math.sin(2 * math.pi * i / 48)
        for control, value in zip(window.plane_center_controls, center_mm):
            control.setValue(float(value))
        window._apply_plane_numeric_controls()
        window._preview_plane_geometry()
    record("edit", lambda: window, edit)
    window._cancel_plane_edit()
    window.pwv_dock.hide()
    select(0)
    # Capture actual settings tabs rather than drawing lookalike controls.
    defaults = Workspace()
    apply_config_bundle_to_workspace(defaults, window._config_bundle)
    dialog = ActionSettingsDialog("Generate Planes", window._settings_sections(StepId.GENERATE_PLANES),
                                  ws, defaults, window)
    dialog.resize(700, 650)
    dialog.show()
    def settings(i):
        dialog.tabs.setCurrentIndex((i // 24) % dialog.tabs.count())
    record("settings", lambda: dialog, settings)
    dialog.reject()

    window._selected_plane_index = -1
    window.ortho_viewer.set_selected_plane(None)
    window.pwv_dock.hide()
    window.selection_tabs.hide()
    window._refresh_all()
    for name, key in (("streamlines", "streamlines_live"), ("wss", "wss_surface_live"),
                       ("pressure", "pressure_gradient_volume")):
        for obj in ws.scene_objects.values():
            obj.visible = obj.data_key == key or obj.data_key.startswith("segmask_group")
            if obj.data_key.startswith("segmask_group"):
                obj.opacity = .09 if name != "wss" else .02
        window.scene.sync_from_workspace()
        def flow_frame(i):
            orbit(i)
            if name == "streamlines":
                import time
                from autoflow.ui.streamline_jobs import streamline_requests
                manager = streamline_requests(ws)
                manager.request(ws.current_t)
                deadline = time.monotonic() + 30
                while ws.current_t not in ws.streamline_cache:
                    if time.monotonic() > deadline:
                        raise RuntimeError("Streamline frame did not complete")
                    QtTest.QTest.qWait(10)
                mesh = ws.streamline_cache[ws.current_t]
                if mesh is None or mesh.n_points == 0:
                    raise RuntimeError("Empty streamline frame")
                window.scene.sync_from_workspace(rebuild_prefixes=("streamlines_live",))
        record(name, lambda: window.plotter, flow_frame)
    window.close()
    app.processEvents()
    print("CAPTURE COMPLETE", flush=True)


def compose(work):
    import json
    import subprocess
    import imageio_ffmpeg
    from PIL import Image, ImageDraw, ImageFont, ImageOps

    output = ROOT / "docs/assets/gui"
    output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((work / "capture.json").read_text())
    font_root = Path("/usr/share/fonts/truetype/dejavu")
    title_font = ImageFont.truetype(str(font_root / "DejaVuSans-Bold.ttf"), 23)
    number_font = ImageFont.truetype(str(font_root / "DejaVuSans-Bold.ttf"), 19)
    navy, teal, amber = "#213e50", "#007f78", "#cb941e"
    frames_dir = work / "composed"
    frames_dir.mkdir(exist_ok=True)

    def card(canvas, box, title, source, number=None, crop=None, accent=amber):
        x, y, width, height = box
        draw = ImageDraw.Draw(canvas)
        draw.rounded_rectangle((x + 1, y + 4, x + width + 1, y + height + 4), 15,
                               fill="#e4eaed")
        draw.rounded_rectangle((x, y, x + width, y + height), 15,
                               fill="white", outline="#ccdbe3", width=1)
        draw.rounded_rectangle((x + 1, y, x + width - 1, y + 8), 14, fill=accent)
        draw.rectangle((x + 1, y + 6, x + width - 1, y + 11), fill="white")
        text_x = x + 15
        if number is not None:
            draw.ellipse((text_x, y + 15, text_x + 29, y + 44), fill=accent)
            draw.text((text_x + 14, y + 29), str(number), anchor="mm", font=number_font, fill="white")
            text_x += 39
        draw.text((text_x, y + 17), title, font=title_font, fill=navy)
        with Image.open(source) as original:
            image = original.convert("RGB")
        if crop:
            image = image.crop(crop)
        area = (width - 24, height - 67)
        image = ImageOps.contain(image, area, Image.Resampling.LANCZOS)
        left, top = x + 12 + (area[0] - image.width) // 2, y + 55 + (area[1] - image.height) // 2
        canvas.paste(image, (left, top))
        draw.rounded_rectangle((left - 1, top - 1, left + image.width, top + image.height),
                               3, outline="#d6e1e6", width=1)

    def source(clip, frame):
        return work / "clips" / clip / f"{frame:03d}.png"

    def workspace(frame):
        result = Image.new("RGB", (1280, 1000), "#f2f5f7")
        card(result, (16, 16, 800, 532), "AutoFlow GUI", source("workspace", frame), accent=navy)
        card(result, (832, 16, 432, 532), "Browse anatomy", source("slices", frame), 1)
        card(result, (16, 564, 400, 420), "Vessel geometry", source("geometry", frame), 2)
        # A close-up of the real selected-plane viewport keeps the ROI readable.
        plane = Image.open(source("planes", frame))
        card(result, (432, 564, 400, 420), "Inspect planes", source("planes", frame), 3,
             crop=(0, 0, plane.width, min(446, plane.height)))
        plane.close()
        card(result, (848, 564, 416, 420), "Measure flow", source("curve", frame), 4, accent=teal)
        return result

    def editing(frame):
        result = Image.new("RGB", (1280, 720), "#f2f5f7")
        card(result, (16, 16, 616, 688), "Adjust plane position", source("edit", frame), 5,
             crop=metadata["edit_view_rect"])
        card(result, (648, 16, 616, 688), "Configure measurements", source("settings", frame), 6, accent=teal)
        return result

    def hemodynamics(frame):
        result = Image.new("RGB", (1280, 504), "#f2f5f7")
        for i, (clip, title) in enumerate((("streamlines", "Streamlines"), ("wss", "Wall shear stress"),
                                            ("pressure", "Pressure gradient"))):
            card(result, (16 + i * 421, 16, 406, 472), title, source(clip, frame), i + 1,
                 accent=teal if i == 1 else amber)
        return result

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    assets = []
    for name, builder in (("gui-workspace", workspace), ("gui-editing", editing),
                           ("gui-hemodynamics", hemodynamics)):
        folder = frames_dir / name
        folder.mkdir(exist_ok=True)
        for frame in range(48):
            builder(frame).save(folder / f"{frame:03d}.png")
        pattern = str(folder / "%03d.png")
        palette = folder / "palette.png"
        subprocess.run([ffmpeg, "-v", "error", "-y", "-framerate", "8", "-i", pattern,
                        "-vf", "palettegen=stats_mode=diff", "-frames:v", "1", str(palette)], check=True)
        subprocess.run([ffmpeg, "-v", "error", "-y", "-framerate", "8", "-i", pattern,
                        "-i", str(palette), "-lavfi", "paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle",
                        "-loop", "0", str(output / (name + ".gif"))], check=True)
        subprocess.run([ffmpeg, "-v", "error", "-y", "-framerate", "8", "-i", pattern,
                        "-an", "-c:v", "libx264", "-preset", "slow", "-crf", "18",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output / (name + ".mp4"))], check=True)
        builder(0).save(output / (name + ".png"))
        assets.append({"name": name, "size": list(builder(0).size), "frames": 48, "fps": 8,
                       "gif_bytes": (output / (name + ".gif")).stat().st_size})
        print("COMPOSED", name, flush=True)
    (work / "assets.json").write_text(json.dumps(assets, indent=2))


def verify(work):
    import hashlib
    import json
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image, ImageChops

    output = ROOT / "docs/assets/gui"
    assets = json.loads((work / "assets.json").read_text())
    checks = []
    for asset in assets:
        name = asset["name"]
        path = output / (name + ".gif")
        with Image.open(path) as gif:
            assert gif.size == tuple(asset["size"])
            assert gif.n_frames == 48, (name, gif.n_frames)
            assert gif.info.get("loop") == 0
            frames = []
            duration = 0
            for i in range(gif.n_frames):
                gif.seek(i)
                frames.append(gif.convert("RGB"))
                duration += gif.info["duration"]
            assert all(ImageChops.difference(frames[0], frame).getbbox() for frame in frames[1:])
        reader = imageio.get_reader(output / (name + ".mp4"))
        count = sum(1 for frame in reader if np.ptp(frame) > 0)
        reader.close()
        assert count == 48
        checks.append({"name": name, "gif_frames": 48, "duration_ms": duration,
                       "mp4_frames": count, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (work / "verification.json").write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "capture", "compose", "verify"))
    parser.add_argument("--work-dir", type=Path, default=ROOT / "validation/gui-showcase-2026-10-28")
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    globals()[args.stage](args.work_dir)


if __name__ == "__main__":
    main()
