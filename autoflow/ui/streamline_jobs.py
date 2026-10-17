"""Shared, bounded background computation for live 3D streamline meshes."""

import weakref
import numpy as np
from PySide6 import QtCore

from ..algorithms.streamlines import generate_streamlines_at_t


_requests = {}


def _context(workspace):
    parameters = workspace.streamline_params
    return (id(workspace.flow_raw), id(workspace.segmask_binary), id(workspace.streamline_seeds),
            tuple(workspace.resolution), tuple(workspace.origin), parameters.max_steps,
            parameters.terminal_speed, parameters.seed_ratio, parameters.min_seeds, parameters.rng_seed)


class _MeshTask(QtCore.QRunnable):
    def __init__(self, owner, request):
        super().__init__()
        self.owner, self.request = owner, request

    def run(self):
        context, phase, args, kwargs = self.request
        try:
            mesh, error = generate_streamlines_at_t(*args, **kwargs), None
        except Exception as exc:
            mesh, error = None, f"{type(exc).__name__}: {exc}"
        try:
            self.owner.completed.emit((context, phase, mesh, error))
        except RuntimeError:
            # The application may have closed while a numerical call finished.
            pass


class StreamlineRequests(QtCore.QObject):
    ready = QtCore.Signal(int)
    failed = QtCore.Signal(str)
    completed = QtCore.Signal(object)

    def __init__(self, workspace):
        super().__init__(QtCore.QCoreApplication.instance())
        self.workspace = weakref.ref(workspace)
        self.context = None
        self.generation = 0
        self.clear_on_request = False
        self.busy = False
        self.active = None
        self.queued = None
        self.errors = set()
        self.completed.connect(self._finish, QtCore.Qt.ConnectionType.QueuedConnection)

    def invalidate(self):
        self.context = None
        self.generation += 1
        self.clear_on_request = True
        self.queued = None
        self.errors.clear()

    def request(self, phase):
        workspace = self.workspace()
        if (workspace is None or not workspace.streamline_active
                or workspace.flow_raw is None):
            return None
        context = (_context(workspace), self.generation)
        if context != self.context:
            if self.context is not None or self.clear_on_request:
                workspace.streamline_cache.clear()
            self.context = context
            self.clear_on_request = False
            self.errors.clear()
            self.queued = None
        phase = min(max(0, int(phase)), workspace.flow_raw.shape[3] - 1)
        if phase in workspace.streamline_cache:
            self.queued = None
            return workspace.streamline_cache[phase]
        if (context, phase) in self.errors:
            return None
        if self.active is not None and self.active[:2] == (context, phase):
            # Returning to the currently running phase supersedes a queued one.
            self.queued = None
            return None
        if self.queued is not None and self.queued[:2] == (context, phase):
            return None
        if workspace.segmask_binary is None:
            return None
        parameters = workspace.streamline_params
        binary = np.asarray(workspace.segmask_binary)
        mask = binary[..., min(phase, binary.shape[3] - 1)] if binary.ndim == 4 else binary
        seeds = workspace.streamline_seeds
        request = (context, phase,
                   (workspace.flow_raw, phase, None if seeds is None else np.array(seeds, copy=True),
                    np.array(workspace.resolution, copy=True), np.array(workspace.origin, copy=True)),
                   dict(mask_3d=np.array(mask, dtype=bool, copy=True), max_steps=parameters.max_steps,
                        terminal_speed=parameters.terminal_speed, seed_ratio=parameters.seed_ratio,
                        min_seeds=parameters.min_seeds, rng_seed=parameters.rng_seed))
        if QtCore.QCoreApplication.instance() is None:
            workspace.streamline_cache[phase] = generate_streamlines_at_t(*request[2], **request[3])
            return workspace.streamline_cache[phase]
        if self.busy:
            # Playback may request many phases; retain only the newest pending
            # request instead of growing an unbounded queue of stale frames.
            self.queued = request
        else:
            self._start(request)
        return None

    def _start(self, request):
        self.busy = True
        self.active = request[:2]
        QtCore.QThreadPool.globalInstance().start(_MeshTask(self, request))

    @QtCore.Slot(object)
    def _finish(self, result):
        context, phase, mesh, error = result
        self.busy = False
        self.active = None
        workspace = self.workspace()
        current = (workspace is not None and workspace.streamline_active
                   and context == self.context and context == (_context(workspace), self.generation))
        if current:
            if error is None:
                workspace.streamline_cache[phase] = mesh
            else:
                self.errors.add((context, phase))
                self.failed.emit(f"Streamline generation failed: {error}")
        queued, self.queued = self.queued, None
        if (queued is not None and workspace is not None and workspace.streamline_active
                and queued[0] == self.context
                and queued[0] == (_context(workspace), self.generation)):
            self._start(queued)
        if current and error is None:
            self.ready.emit(phase)


def streamline_requests(workspace):
    """Keep Qt workers outside Workspace so serialization/Labeler stay numerical."""
    key = id(workspace)
    entry = _requests.get(key)
    if entry is not None and entry[0]() is workspace:
        return entry[1]
    def release(unused):
        entry = _requests.pop(key, None)
        if entry is not None:
            entry[1].invalidate()
            try:
                entry[1].deleteLater()
            except RuntimeError:
                pass
    reference = weakref.ref(workspace, release)
    manager = StreamlineRequests(workspace)
    _requests[key] = (reference, manager)
    return manager
