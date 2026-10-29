"""One numerical plane job at a time, with latest requests keyed by plane ID."""

from collections import OrderedDict
import copy

from PySide6 import QtCore

from ..task_control import CancellationToken, TaskCancelled, task_scope


def calculate_plane(payload):
    from ..algorithms.metrics.planes import compute_plane_metrics
    from ..algorithms.metrics.plane_derived import augment_plane_metrics_with_derived
    from ..algorithms.metrics.sampling import PlaneSamplingContext

    plane = payload["plane"]
    data = payload["data"]
    context = PlaneSamplingContext(data["mask"], data["spacing"], data["origin"], [plane],
        branch_labels_3d=data["branches"], segmentation_labels_3d=data["labels3"],
        segmentation_labels_4d=data["labels4"])
    try:
        metrics = compute_plane_metrics(data["flow"], data["mask"], data["spacing"], data["origin"],
            [plane], RR=data["rr"], branch_labels_3d=data["branches"],
            segmentation_labels_3d=data["labels3"], segmentation_labels_4d=data["labels4"],
            path_info=data["path_info"], forks=data["forks"], paths=data["paths"],
            sampling_context=context)
        if any(data.get(key) is not None for key in ("tke", "pg", "pressure")) or data.get("wss"):
            metrics, _ = augment_plane_metrics_with_derived(metrics, [plane], data["mask"],
                data["spacing"], data["origin"], branch_labels_3d=data["branches"],
                segmentation_labels_3d=data["labels3"], segmentation_labels_4d=data["labels4"],
                tke_array=data["tke"], pressure_gradient_array=data["pg"],
                relative_pressure_array=data["pressure"], wss_surfaces=data["wss"],
                pressure_gradient_support_mask=data.get("pressure_support"), sampling_context=context)
        return {"request": payload, "metric": metrics[0] if metrics else {}}
    finally:
        context.close()


class _PlaneWorker(QtCore.QObject):
    ready = QtCore.Signal(object)
    failed = QtCore.Signal(str)
    done = QtCore.Signal()

    def __init__(self, payload, token):
        super().__init__()
        self.payload, self.token = payload, token

    @QtCore.Slot()
    def run(self):
        try:
            with task_scope(self.token):
                result = calculate_plane(self.payload)
                self.token.check()
                self.ready.emit(result)
        except TaskCancelled:
            pass
        except Exception as exc:
            self.failed.emit(f"Plane calculation failed: {type(exc).__name__}: {exc}")
        finally:
            self.done.emit()


class PlaneCalculationQueue(QtCore.QObject):
    ready = QtCore.Signal(object)
    failed = QtCore.Signal(str)
    idle = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pending = OrderedDict()
        self.thread = self.worker = self.token = None
        self.active_id = None

    @property
    def is_running(self):
        return self.thread is not None or bool(self.pending)

    def submit(self, payload):
        key = payload["plane"].plane_id
        self.pending[key] = payload
        if key == self.active_id and self.token is not None:
            self.token.cancel()
        self._start()

    def _start(self):
        if self.thread is not None or not self.pending:
            return
        self.active_id, payload = self.pending.popitem(last=False)
        # Own VTK surfaces in this worker; GUI mappers keep their originals.
        payload = dict(payload, data=dict(payload["data"]))
        payload["data"]["wss"] = [surface.copy(deep=True) if hasattr(surface, "GetMTime") else copy.deepcopy(surface)
                                   for surface in (payload["data"].get("wss") or [])]
        self.token = CancellationToken()
        self.thread = QtCore.QThread(self)
        self.worker = _PlaneWorker(payload, self.token)
        self.worker.moveToThread(self.thread)
        self.worker.ready.connect(self._deliver)
        self.worker.failed.connect(self._failed)
        self.worker.done.connect(self.thread.quit)
        self.worker.done.connect(self.worker.deleteLater)
        self.thread.finished.connect(self._finished)
        self.thread.started.connect(self.worker.run)
        self.thread.start()

    @QtCore.Slot(object)
    def _deliver(self, result):
        if self.token is not None and not self.token.cancelled:
            self.ready.emit(result)

    @QtCore.Slot(str)
    def _failed(self, message):
        if self.token is not None and not self.token.cancelled:
            self.failed.emit(message)

    @QtCore.Slot()
    def _finished(self):
        thread = self.thread
        self.thread = self.worker = self.token = None
        self.active_id = None
        if thread is not None:
            thread.deleteLater()
        if self.pending:
            self._start()
        else:
            self.idle.emit()

    def cancel(self):
        self.pending.clear()
        if self.token is not None:
            self.token.cancel()

    def close(self, timeout_ms=10000):
        self.cancel()
        if self.thread is not None:
            # Completion is cooperatively checked between planes/phases.
            self.thread.quit()
            return self.thread.wait(timeout_ms)
        return True
