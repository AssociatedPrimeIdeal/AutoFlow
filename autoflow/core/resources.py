"""Conservative optional RAM limits; existing process caps remain upper bounds."""
import os


def memory_budget_bytes(fraction=.5):
    configured = os.environ.get("AUTOFLOW_MEMORY_BUDGET_MB")
    if configured:
        return max(1, int(float(configured)*1024**2))
    try:
        import psutil
        return max(1, int(psutil.virtual_memory().available*fraction))
    except ImportError:
        return None


def bounded_workers(requested, estimated_worker_bytes):
    budget = memory_budget_bytes()
    if budget is None:
        return max(1, int(requested))
    return max(1, min(int(requested), budget // max(1, int(estimated_worker_bytes))))
