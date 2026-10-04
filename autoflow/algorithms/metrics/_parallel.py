"""Cancellation and progress transport for isolated plane workers."""

import os
from ...task_control import TaskCancelled, check_cancelled, current_cancellation_token


class _PlaneProcessToken:
    def __init__(self, progress_path):
        self.path = f"{progress_path}.cancel" if progress_path else ""

    def check(self):
        if self.path and os.path.exists(self.path):
            raise TaskCancelled("Cancelled by user")


def _mark_plane_process_progress(progress_path):
    check_cancelled()
    if progress_path:
        fd = os.open(progress_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, b"1\n")
        finally:
            os.close(fd)


def _wait_plane_processes(calculate, progress_path, total, progress_callback, *, stage="plane_metric"):
    if progress_callback is None and current_cancellation_token() is None:
        return calculate()
    import threading
    state = {"result": None, "error": None}
    def run():
        try:
            state["result"] = calculate()
        except BaseException as exc:
            state["error"] = exc
    runner = threading.Thread(target=run, name="autoflow-plane-processes")
    runner.start()
    offset = completed = 0
    def update():
        nonlocal offset, completed
        with open(progress_path, "rb") as handle:
            handle.seek(offset)
            events = handle.readlines()
            offset = handle.tell()
        # Preserve one callback per completed plane even when several worker
        # markers arrive before the parent next polls the shared progress file.
        for _event in events:
            completed += 1
            if progress_callback is not None:
                progress_callback({"stage": stage, "current": completed, "total": total,
                                   "message": f"{'Sampled derived' if stage == 'plane_derived' else 'Calculated'} plane metrics ({completed}/{total})"})
    try:
        while runner.is_alive():
            check_cancelled()
            update()
            runner.join(timeout=0.05)
        check_cancelled()
        update()
        if state["error"] is not None:
            raise state["error"]
        return state["result"]
    except BaseException:
        # Workers observe this marker at phase/plane boundaries. Join before
        # deleting shared files or unlocking the workspace.
        open(f"{progress_path}.cancel", "ab").close()
        runner.join()
        raise
