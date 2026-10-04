"""Cooperative cancellation and progress for numerical/background tasks."""

from contextlib import contextmanager
import os
import signal
import subprocess
import threading


class TaskCancelled(RuntimeError):
    """A user requested cancellation at a safe task boundary."""


class CancellationToken:
    def __init__(self):
        self._event = threading.Event()

    def cancel(self):
        self._event.set()

    @property
    def cancelled(self):
        return self._event.is_set()

    def check(self):
        if self.cancelled:
            raise TaskCancelled("Cancelled by user")


_state = threading.local()


def check_cancelled():
    token = getattr(_state, "token", None)
    if token is not None:
        token.check()


def report_progress(payload):
    check_cancelled()
    callback = getattr(_state, "progress", None)
    if callback is not None:
        callback(payload)
    check_cancelled()


def current_cancellation_token():
    return getattr(_state, "token", None)


@contextmanager
def task_scope(token=None, progress=None):
    previous = (getattr(_state, "token", None), getattr(_state, "progress", None))
    _state.token = token if token is not None else previous[0]
    _state.progress = progress if progress is not None else previous[1]
    try:
        check_cancelled()
        yield
    finally:
        _state.token, _state.progress = previous


def stop_process(process):
    """Stop only a process group created for this task, and reap its parent."""
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            process.kill()
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.wait()
