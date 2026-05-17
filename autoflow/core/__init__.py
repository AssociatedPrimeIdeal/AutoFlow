from .models import *  # noqa: F401,F403

__all__ = [name for name in globals() if not name.startswith("_")] + ["PipelineEngine", "StepResult"]


def __getattr__(name):
    if name in {"PipelineEngine", "StepResult"}:
        from .pipeline import PipelineEngine, StepResult

        return {
            "PipelineEngine": PipelineEngine,
            "StepResult": StepResult,
        }[name]
    raise AttributeError(f"module 'autoflow.core' has no attribute {name!r}")
