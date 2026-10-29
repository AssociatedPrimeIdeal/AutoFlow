"""Batch completion states and durable per-case progress (no analysis changes)."""
from collections import Counter
from pathlib import Path

from .core.output import atomic_json_write


def case_status(summary):
    """Distinguish completed requests from missing outputs, not clinical QC grades."""
    if (summary.get("video_errors") or summary.get("analysis_report_status") == "failed"
            or summary.get("output_status") in {"incomplete", "partial"}):
        return "partial"
    return "ok"


def checkpoint_batch(results, output_dir, total_cases):
    """Persist the report first; it is authoritative if the progress write fails."""
    output = Path(output_dir)
    atomic_json_write(output / "batch_report.json", results)
    html_error = None
    try:
        from .batch_report_html import save_batch_report_html
        save_batch_report_html(results, output / "batch_report.html")
    except Exception as exc:
        from .task_control import TaskCancelled
        if isinstance(exc, TaskCancelled):
            raise
        html_error = f"{type(exc).__name__}: {exc}"
        print(f"[REPORT ERROR] Batch HTML export failed; JSON results preserved: {html_error}")
    counts = Counter(str(item.get("status", "error")) for item in results)
    atomic_json_write(output / "batch_progress.json", {
        "schema": "autoflow.batch_progress.v1",
        "total_cases": int(total_cases),
        "completed_cases": len(results),
        "remaining_cases": max(0, int(total_cases) - len(results)),
        "status_counts": dict(counts),
        "all_requests_complete": len(results) == int(total_cases) and counts.get("ok", 0) == len(results),
        "source_of_truth": "batch_report.json",
        "batch_html_status": "failed" if html_error else "complete",
        "batch_html_error": html_error,
    })


def record_batch_result(results, result, output_dir, total_cases):
    """Checkpoint each terminal case; callers must not classify I/O failures as cases."""
    pending = [*results, result]
    checkpoint_batch(pending, output_dir, total_cases)
    results.append(result)
