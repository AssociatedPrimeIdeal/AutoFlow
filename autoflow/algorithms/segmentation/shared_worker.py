"""Opt-in persistent inference child. JSON load/run/close; cancellation reaps it."""

import atexit
import contextlib
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np
from ...task_control import check_cancelled, stop_process


class SharedInferenceWorker:
    def __init__(self, python=None):
        self.python = python or sys.executable
        self.process = None
        self.responses = queue.Queue()
        self.lock = threading.RLock()
        self.log = None
        self.request_id = 0
        self.reader = None

    def _start(self):
        if self.process is not None and self.process.poll() is None:
            return
        self.responses = queue.Queue()
        self.log = tempfile.TemporaryFile(mode="w+b")
        env = dict(os.environ, AUTOFLOW_SHARED_WORKER_CHILD="1")
        root = str(Path(__file__).resolve().parents[3])
        env["PYTHONPATH"] = root+os.pathsep+env.get("PYTHONPATH", "")
        options = {"start_new_session": True} if os.name != "nt" else {
            "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW}
        code = "import sys; sys.path.insert(0, sys.argv.pop(1)); from autoflow.algorithms.segmentation.shared_worker import main; main()"
        self.process = subprocess.Popen([self.python, "-c", code, root], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=self.log, text=True, encoding="utf-8", env=env, **options)
        process, responses = self.process, self.responses
        def read():
            for line in process.stdout:
                try:
                    responses.put(json.loads(line))
                except ValueError:
                    continue
            responses.put({"kind": "eof"})
        self.reader = threading.Thread(target=read, name="autoflow-inference-protocol", daemon=True)
        self.reader.start()

    def request(self, kind, payload=None, progress=None):
        with self.lock:
            check_cancelled()
            self._start()
            self.request_id += 1
            request_id = self.request_id
            self.process.stdin.write(json.dumps({"kind": kind, "request_id": request_id, "payload": payload or {}})+"\n")
            self.process.stdin.flush()
            try:
                while True:
                    check_cancelled()
                    try:
                        response = self.responses.get(timeout=.1)
                    except queue.Empty:
                        continue
                    if response.get("kind") == "eof":
                        self.log.seek(0)
                        raise RuntimeError("Shared inference worker exited: "+self.log.read().decode("utf-8", errors="replace")[-8000:])
                    if response.get("request_id") != request_id:
                        continue
                    if response["kind"] == "progress":
                        if progress:
                            progress(response["payload"])
                    elif response["kind"] == "error":
                        raise RuntimeError(response["message"])
                    else:
                        return response.get("payload", {})
            except BaseException:
                self.close()
                raise

    def load(self, model_path, folds, checkpoint, device):
        return self.request("load", {"model_path": str(model_path), "folds": list(folds), "checkpoint": checkpoint, "device": device})

    def cancel(self):
        # Callable from another thread; never wait on the request lock.
        process = self.process
        if process is not None:
            stop_process(process)

    def close(self):
        process, self.process = self.process, None
        if process is not None:
            try:
                if process.poll() is None:
                    process.stdin.write('{"kind":"close"}\n')
                    process.stdin.flush()
                    process.wait(timeout=2)
            except (OSError, subprocess.TimeoutExpired):
                stop_process(process)
            finally:
                if process.poll() is None:
                    stop_process(process)
                process.stdin.close()
                process.stdout.close()
        if self.reader is not None:
            self.reader.join(timeout=2)
            self.reader = None
        if self.log is not None:
            self.log.close()
            self.log = None


_client = SharedInferenceWorker()
atexit.register(_client.close)


def close_shared_inference_worker():
    _client.close()


def run_shared_inference(mag_nnunet, flow_nnunet, resolution_nnunet, model_path, dataset_json,
        selected_folds, checkpoint_name, resolved_device, label_map, safe_case_id, affine,
        pipeline_script, num_processes_segmentation_export, artifact_prefix, progress_callback, t_total_start, origin=None):
    from .models import _python_from_4d_pipeline_script
    python = _python_from_4d_pipeline_script(pipeline_script) or sys.executable
    with _client.lock:
        if str(python) != str(_client.python):
            _client.close()
            _client.python = str(python)
        with tempfile.TemporaryDirectory(prefix="autoflow_shared_request_") as directory:
            directory = Path(directory)
            check_cancelled()
            np.save(directory/"mag.npy", mag_nnunet, allow_pickle=False)
            check_cancelled()
            np.save(directory/"flow.npy", flow_nnunet, allow_pickle=False)
            payload = {"directory": str(directory), "resolution_nnunet": np.asarray(resolution_nnunet).tolist(),
                "model_path": str(model_path), "dataset_json": dataset_json, "selected_folds": list(selected_folds),
                "checkpoint_name": checkpoint_name, "resolved_device": resolved_device, "label_map": label_map,
                "safe_case_id": safe_case_id, "affine": np.asarray(affine).tolist(), "pipeline_script": str(pipeline_script),
                "num_processes_segmentation_export": num_processes_segmentation_export, "artifact_prefix": str(artifact_prefix), "origin": np.asarray(origin if origin is not None else (0, 0, 0)).tolist()}
            provenance = _client.request("run", payload, progress_callback)
            check_cancelled()
            segmentation = np.load(directory/"result.npy", allow_pickle=False)
            provenance["elapsed_sec"] = time.perf_counter()-t_total_start
            return segmentation, provenance


def main():
    from .nnunet_grouped import _generate_nnunet_4d_grouped, _get_predictor
    def reply(kind, request_id, payload=None, **extra):
        sys.__stdout__.write(json.dumps({"kind": kind, "request_id": request_id, "payload": payload, **extra})+"\n")
        sys.__stdout__.flush()
    requests = 0
    for line in sys.stdin:
        message = json.loads(line)
        if message.get("kind") == "close":
            break
        request_id = message["request_id"]
        payload = message["payload"]
        try:
            with contextlib.redirect_stdout(sys.stderr):
                if message["kind"] == "ping":
                    reply("ready", request_id, {"pid": os.getpid(), "requests": requests})
                    continue
                if message["kind"] == "load":
                    import torch
                    _, reused = _get_predictor(Path(payload["model_path"]), payload["folds"], payload["checkpoint"], torch.device(payload["device"]))
                    reply("loaded", request_id, {"model_reused": reused})
                    continue
                if message["kind"] != "run":
                    raise ValueError("Unknown inference command")
                directory = Path(payload.pop("directory"))
                mag = np.load(directory/"mag.npy", mmap_mode="r", allow_pickle=False)
                flow = np.load(directory/"flow.npy", mmap_mode="r", allow_pickle=False)
                payload["model_path"] = Path(payload["model_path"])
                payload["resolution_nnunet"] = np.asarray(payload["resolution_nnunet"])
                payload["affine"] = np.asarray(payload["affine"])
                payload["label_map"] = {int(key): int(value) for key, value in payload["label_map"].items()}
                segmentation, provenance = _generate_nnunet_4d_grouped(mag, flow, **payload,
                    progress_callback=lambda record: reply("progress", request_id, record), t_total_start=time.perf_counter())
                np.save(directory/"result.npy", segmentation, allow_pickle=False)
                del segmentation, mag, flow
                requests += 1
                provenance.update(worker_pid=os.getpid(), worker_request_count=requests, experimental=True)
                reply("result", request_id, provenance)
        except Exception as exc:
            import traceback
            traceback.print_exc(file=sys.stderr)
            reply("error", request_id, message=f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
