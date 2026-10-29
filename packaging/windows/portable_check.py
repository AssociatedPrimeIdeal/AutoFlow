"""Manual distribution diagnostics; no patient data or model training is run."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import importlib
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback


def _worker(value):
    import numpy as np

    return int(np.sum(np.arange(value)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gui", action="store_true", help="Also open and close the GUI and render a VTK frame")
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()
    report = {"python": sys.executable, "checks": {}, "errors": {}}

    def check(name, operation):
        try:
            report["checks"][name] = operation()
            print(f"PASS {name}", flush=True)
        except Exception:
            report["errors"][name] = traceback.format_exc()
            print(f"FAIL {name}: {report['errors'][name]}", flush=True)

    def documentation():
        from autoflow.runtime_paths import documentation_index

        index = documentation_index()
        root = Path(os.environ["AUTOFLOW_PORTABLE_ROOT"]).resolve()
        if index is None or not index.resolve().is_relative_to(root):
            raise RuntimeError("Portable documentation home page is missing")
        manifest = json.loads((index.parent / "offline-manifest.json").read_text(encoding="utf-8"))
        if manifest.get("offline") is not True or not (index.parent / "search/search_index.js").is_file():
            raise RuntimeError("Portable offline documentation/search is incomplete")
        return {"index": str(index), "offline": True, "pages": manifest["pages"]}

    def imports():
        modules = ("numpy", "scipy", "pyamg", "skimage", "networkx", "pyvista", "vtk",
                   "h5py", "imageio_ffmpeg", "maxflow", "PySide6", "pyqtgraph", "pyvistaqt",
                   "matplotlib", "nnunetv2", "nibabel", "nrrd", "pydicom", "torch", "cupy",
                   "dicom2h5.converter", "pudipflow", "gustflow", "spatiotemporal_labeler.app",
                   "autoflow.api", "autoflow.ui.app", "autoflow.rendering.jobs")
        return {name: str(importlib.import_module(name).__file__) for name in modules}

    def models():
        import torch
        import numpy as np
        from autoflow.algorithms.segmentation.models import default_nnunet_model_folder, resolve_nnunet_4d_model_folder
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        from nnunetv2.preprocessing.resampling.gpu_resampling import resample_torch_cuda_fornnunet

        result = {}
        for name, folder in (("3d", default_nnunet_model_folder()), ("4d", Path(resolve_nnunet_4d_model_folder()))):
            predictor = nnUNetPredictor(device=torch.device("cpu"), perform_everything_on_device=False)
            checkpoint = "checkpoint_best.pth" if name == "4d" else "checkpoint_final.pth"
            predictor.initialize_from_trained_model_folder(str(folder), use_folds=("all",), checkpoint_name=checkpoint)
            # Resolve all named resamplers, including the deployed helper, now.
            _ = predictor.configuration_manager.resampling_fn_data
            _ = predictor.configuration_manager.resampling_fn_seg
            _ = predictor.configuration_manager.resampling_fn_probabilities
            result[name] = {"folder": str(folder), "checkpoint": checkpoint,
                            "channels": len(predictor.dataset_json["channel_names"])}
            strides = getattr(getattr(predictor.network, "encoder", None), "strides", None)
            if strides is None:
                raise RuntimeError("Cannot determine a valid small nnUNet forward shape")
            strides = np.asarray([([s]*3 if isinstance(s, int) else s) for s in strides], dtype=int)
            shape = tuple((2*np.prod(strides, axis=0)).tolist())
            old_threads = torch.get_num_threads()
            try:
                torch.set_num_threads(min(4, os.cpu_count() or 1))
                predictor.network.eval()
                with torch.inference_mode():
                    output = predictor.network(torch.zeros((1, result[name]["channels"])+shape))
                output = output[0] if isinstance(output, (tuple, list)) else output
                if not torch.isfinite(output).all():
                    raise ValueError("Non-finite nnUNet forward output")
                result[name]["cpu_forward_shape"] = list(output.shape)
            finally:
                torch.set_num_threads(old_threads)
            del predictor
        previous = os.environ.get("AUTOFLOW_NNUNET_DEVICE")
        os.environ["AUTOFLOW_NNUNET_DEVICE"] = "cpu"
        try:
            sample = np.arange(64, dtype=np.float32).reshape(1, 4, 4, 4)
            resampled = resample_torch_cuda_fornnunet(sample, (6, 6, 6), (1, 1, 1), (2 / 3,) * 3)
            if resampled.shape != (1, 6, 6, 6) or not np.isfinite(resampled).all():
                raise ValueError("CPU model resampling failed")
            result["explicit_cpu_resampling"] = True
        finally:
            if previous is None:
                os.environ.pop("AUTOFLOW_NNUNET_DEVICE", None)
            else:
                os.environ["AUTOFLOW_NNUNET_DEVICE"] = previous
        return result

    def msvc_runtime():
        if os.name != "nt":
            return "not applicable"
        import ctypes
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
        kernel.GetModuleHandleW.restype = wintypes.HMODULE
        kernel.GetModuleFileNameW.argtypes = [wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD]
        kernel.GetModuleFileNameW.restype = wintypes.DWORD
        buffer = ctypes.create_unicode_buffer(32768)
        kernel.GetModuleFileNameW(kernel.GetModuleHandleW("msvcp140.dll"), buffer, len(buffer))
        root = Path(os.environ["AUTOFLOW_PORTABLE_ROOT"]).resolve()
        if not buffer.value or not Path(buffer.value).resolve().is_relative_to(root):
            raise RuntimeError(f"C++ runtime was not loaded from the portable package: {buffer.value}")
        return buffer.value

    def workers():
        with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as pool:
            result = pool.submit(_worker, 10).result(timeout=90)
        if result != 45:
            raise ValueError(f"Invalid spawned worker result: {result}")
        return result

    def commands():
        result = {}
        for module in ("autoflow.ui.launcher", "spatiotemporal_labeler"):
            completed = subprocess.run([sys.executable, "-m", module, "--help"], capture_output=True,
                                       text=True, timeout=90, check=True,
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            if "usage:" not in completed.stdout.lower():
                raise ValueError(f"No help output from {module}")
            result[module] = completed.returncode
        return result

    def encoder():
        import imageio.v2 as imageio
        import numpy as np

        with tempfile.TemporaryDirectory(prefix="autoflow_encoder_") as temp:
            target = Path(temp) / "test.mp4"
            with imageio.get_writer(target, format="ffmpeg", fps=2) as writer:
                for value in (0, 128):
                    writer.append_data(np.full((32, 32, 3), value, dtype=np.uint8))
            with imageio.get_reader(target, format="ffmpeg") as reader:
                if reader.get_data(0).shape != (32, 32, 3):
                    raise ValueError("Video decoder returned an invalid frame")
            return target.stat().st_size

    def gpu():
        import torch

        if not torch.cuda.is_available():
            if args.require_gpu:
                raise RuntimeError("A CUDA GPU is required for this check")
            return {"available": False, "kernel_execution": "not tested on this host"}
        import cupy as cp
        from gustflow.cuda_backend import _kernels
        from gustflow.local_renderer import kernels as local_kernels
        from gustflow import GUSTFlow

        if int(cp.sum(cp.arange(10)).get()) != 45:
            raise ValueError("CuPy CUDA execution failed")
        _kernels()
        local_kernels()
        tensor = torch.arange(10, device="cuda", dtype=torch.float32)
        if float(cp.from_dlpack(tensor).sum().get()) != 45:
            raise ValueError("Torch/CuPy DLPack exchange failed")
        wrapped = torch.zeros((3, 2, 8, 8, 8), device="cuda")
        result = GUSTFlow(num_primitives=8, num_iter=1).fit(
            wrapped, torch.ones_like(wrapped), center_confidence=torch.ones((8, 8, 8), device="cuda")
        )
        if result.recovered.shape != wrapped.shape or not torch.isfinite(result.recovered).all():
            raise ValueError("GUST forward/backward iteration failed")
        return {"available": True, "device": torch.cuda.get_device_name(), "torch_cuda": torch.version.cuda,
                "cupy_runtime": cp.cuda.runtime.runtimeGetVersion(), "gust_kernels_compiled": True,
                "gust_iteration_completed": True}

    def gui():
        from PySide6 import QtWidgets, QtTest
        import pyvista as pv
        from autoflow.ui.app import MainWindow, configure_high_dpi, apply_application_theme

        configure_high_dpi()
        application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        apply_application_theme(application)
        window = MainWindow()
        import numpy as np
        import h5py
        from autoflow.core.models import PlaneData
        workspace = window.workspace
        temporary_case = tempfile.TemporaryDirectory(prefix="autoflow_gui_case_")
        input_path = Path(temporary_case.name)/"case space 测试.h5"
        with h5py.File(input_path, "w") as handle:
            handle["mag"] = np.ones((16, 16, 16, 2), dtype=np.float32)
            velocity = np.zeros((16, 16, 16, 2, 3), dtype=np.float32)
            velocity[..., 1, 2] = 20.
            handle["flow"] = velocity
            handle["segmask"] = np.ones((16, 16, 16, 2), dtype=np.int16)
            handle["Resolution"] = np.ones(3)
            handle["Origin"] = np.array([10., -20., 30.])
            handle["VENC"] = np.full(3, 150.)
            handle["RR"] = 1000.
        workspace.paths.output_dir = str(Path(temporary_case.name)/"outputs")
        workspace.paths.flow_path = workspace.paths.segmask_path = str(input_path)
        window.pipeline.load_data(workspace, lambda _: None, reuse_existing_corr=False, apply_background_correction=False)
        workspace.planes = [PlaneData(center=np.array([8., 8., 8.]), normal=np.array([0., 0., 1.]), path_index=-1)]
        window._rebuild_plane_objects()
        window._refresh_all()
        window._selected_plane_index = 0
        window.ortho_viewer.set_selected_plane(0)
        window.show()
        application.processEvents()
        window.scene.reset_camera()
        window.scene.flush_render()
        QtTest.QTest.qWait(50)
        case_pixels = window.plotter.screenshot(return_img=True)
        if case_pixels is None or np.ptp(case_pixels[..., :3]) == 0:
            raise ValueError("MainWindow did not render the phantom plane")
        if args.output:
            from PIL import Image
            Image.fromarray(case_pixels).save(str(args.output.with_suffix(".case.png")))
        if args.output:
            window.grab().save(str(args.output.with_suffix(".gui.png")))
        # Verify that the shipped Chromium helper/resources can load the QC page.
        import time
        def document_loaded(pane, token):
            answer = []
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                pane.view.page().runJavaScript("document.body && document.body.innerText.includes(" + json.dumps(token) + ")", lambda value: answer.append(value))
                QtTest.QTest.qWait(100)
                if any(answer):
                    return True
            raise ValueError("Bundled HTML tab did not load: " + token)
        window._on_open_case_qc()
        document_loaded(window._html_document_panes["qc"], "Measurement")
        window.close()
        temporary_case.cleanup()
        application.processEvents()
        plotter = pv.Plotter(off_screen=True, window_size=(64, 64))
        plotter.add_mesh(pv.Sphere())
        pixels = plotter.screenshot(return_img=True)
        plotter.close()
        if pixels is None or pixels.shape[:2] != (64, 64):
            raise ValueError("No VTK screenshot produced")
        return {"window_created": True, "phantom_case_loaded": True, "qc_tab_loaded": True, "plane_count": 1, "render_shape": list(pixels.shape)}

    for name, operation in (("documentation", documentation), ("imports", imports), ("msvc_runtime", msvc_runtime), ("models", models), ("spawn_workers", workers),
                            ("subprocess_entries", commands), ("video_encoder", encoder), ("cuda", gpu)):
        check(name, operation)
    if args.gui:
        check("gui_render", gui)
    report["ok"] = not report["errors"]
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ok": report["ok"], "failed": list(report["errors"])}), flush=True)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
