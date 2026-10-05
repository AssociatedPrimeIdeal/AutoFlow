"""Isolated GUI video export; VTK contexts stay in the rendering process."""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import stat
import tempfile
import time

import joblib

from ..task_control import check_cancelled, stop_process, task_scope


def export_videos_in_process(workspace, out_dir, config, requested, progress_callback=None):
    """Render privately, then publish completed MP4s after successful export."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot = copy.copy(workspace)
    snapshot.scene_objects = {key: copy.copy(value) for key, value in workspace.scene_objects.items()}
    for value in snapshot.scene_objects.values():
        value.actor = value.label_actor = None
    # GUI display caches can contain live VTK/Qt objects. The child rebuilds
    # its own surfaces and needs only numerical state and rendering styles.
    for key in list(vars(snapshot)):
        if key.endswith("_surface_cache"):
            delattr(snapshot, key)
    with tempfile.TemporaryDirectory(prefix=".autoflow_export_", dir=out_dir) as directory:
        directory = Path(directory)
        request = directory / "request.joblib"
        progress_path = directory / "progress.jsonl"
        result_path = directory / "result.json"
        joblib.dump({"workspace": snapshot, "config": config, "requested": requested,
                     "directory": str(directory), "progress": str(progress_path),
                     "result": str(result_path)}, request)
        check_cancelled()
        env = dict(os.environ)
        env["PYVISTA_OFF_SCREEN"] = "true"
        package_root = str(Path(__file__).resolve().parents[2])
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [package_root, env.get("PYTHONPATH", "")]))
        with (directory / "render.log").open("w+") as output:
            process = subprocess.Popen([sys.executable, "-m", "autoflow.rendering.jobs", str(request)],
                                       env=env, stdout=output, stderr=subprocess.STDOUT,
                                       start_new_session=os.name != "nt",
                                       creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0)
            offset = 0
            try:
                while True:
                    check_cancelled()
                    if progress_path.exists():
                        with progress_path.open() as events:
                            events.seek(offset)
                            for line in events:
                                try:
                                    payload = json.loads(line)
                                except json.JSONDecodeError:
                                    break
                                if progress_callback is not None:
                                    progress_callback(payload)
                                offset += len(line.encode("utf-8"))
                    if process.poll() is not None:
                        break
                    time.sleep(0.1)
                check_cancelled()
                if process.returncode != 0:
                    output.seek(0)
                    raise RuntimeError(output.read()[-12000:])
            finally:
                if process.poll() is None:
                    stop_process(process)
            result = json.loads(result_path.read_text())
            for name, staged in result["outputs"].items():
                if staged:
                    check_cancelled()
                    target = out_dir / Path(staged).name
                    if target.is_file():
                        os.chmod(staged, stat.S_IMODE(target.stat().st_mode))
                    os.replace(staged, target)
                    result["outputs"][name] = str(target)
            return result


def _render_request(request_path):
    from . import videos
    request = joblib.load(request_path, mmap_mode="r")
    ws, config, requested = request["workspace"], request["config"], request["requested"]
    directory = request["directory"]
    ws.render_settings.update({key: config[key] for key in
                              ("render_background_color", "shared_colorbar_show", "shared_colorbar_bar_cfg", "render_style_cfg", "plane_render_cfg")
                              if key in config})
    progress_path = Path(request["progress"])
    common = dict(fps=config["fps"], smoothing_iteration=ws.derived_params.smoothing_iteration,
                  view=config["camera_view"], distance_scale=config["camera_distance_scale"],
                  rotate=config["rotate_dynamic_video"], rotation_frames=config["dynamic_rotation_frames"],
                  elevation_deg=config["dynamic_rotation_elevation_deg"], time_repeat=config["dynamic_time_repeat"],
                  window_size=config["window_size"])
    jobs = [
        ("plane", bool(ws.planes), videos.render_plane_rotation_video,
         dict(fps=config["fps"], n_frames=config["plane_rotation_frames"],
              smoothing_iteration=ws.derived_params.smoothing_iteration,
              distance_scale=config["camera_distance_scale"], add_plane_idx=config["add_plane_idx"],
              add_path_idx=config["add_path_idx"], plane_video_cfg=config["plane_video_cfg"], window_size=config["window_size"])),
        ("wss", bool(ws.derived.wss_surfaces), videos.render_wss_video,
         dict(common, wss_clim=config["wss_clim"], show_scalar_bar=config["wss_show_scalar_bar"], wss_bar_cfg=config["wss_bar_cfg"])),
        ("tke", ws.derived.tke_array is not None or ws.derived.tke_volume is not None, videos.render_tke_video,
         dict(common, tke_clim=config["tke_clim"], show_scalar_bar=config["tke_show_scalar_bar"], tke_bar_cfg=config["tke_bar_cfg"])),
        ("pressure_gradient", ws.derived.pressure_gradient_magnitude is not None, videos.render_pressure_gradient_video,
         dict(common, pressure_gradient_clim=config["pressure_gradient_clim"], show_scalar_bar=config["pressure_gradient_show_scalar_bar"], pressure_gradient_bar_cfg=config["pressure_gradient_bar_cfg"])),
        ("relative_pressure", ws.derived.relative_pressure_array is not None, videos.render_relative_pressure_video,
         dict(common, relative_pressure_clim=config["relative_pressure_clim"], show_scalar_bar=config["relative_pressure_show_scalar_bar"], relative_pressure_bar_cfg=config["relative_pressure_bar_cfg"])),
        ("streamlines", ws.flow_raw is not None and ws.segmask_binary is not None and ws.segmask_3d is not None,
         videos.render_streamlines_video, dict(common, streamline_clim=config["streamline_clim"], show_scalar_bar=config["streamline_show_scalar_bar"], streamline_bar_cfg=config["streamline_bar_cfg"])),
    ]
    jobs = [job for job in jobs if requested.get("pg" if job[0] in {"pressure_gradient", "relative_pressure"} else job[0])]
    outputs, timings = {}, {}
    def emit(payload):
        with progress_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(payload) + "\n")
    for index, (name, available, renderer, kwargs) in enumerate(jobs):
        emit({"stage": "video", "current": index, "total": len(jobs), "message": f"Rendering {name}…"})
        if not available:
            outputs[name] = ""
            continue
        def frame_progress(payload):
            emit({"stage": "video", "current": index, "total": len(jobs), "message": f"Rendering {name}…",
                  "detail_current": payload.get("current", 0), "detail_total": payload.get("total", 0),
                  "detail_message": payload.get("message", "")})
        started = time.perf_counter()
        with task_scope(progress=frame_progress):
            outputs[name] = renderer(ws, directory, **kwargs) or ""
        timings[name] = time.perf_counter() - started
    emit({"stage": "video", "current": len(jobs), "total": len(jobs), "message": "Video rendering complete"})
    Path(request["result"]).write_text(json.dumps({"outputs": outputs, "times": timings}))


if __name__ == "__main__":
    _render_request(sys.argv[1])
