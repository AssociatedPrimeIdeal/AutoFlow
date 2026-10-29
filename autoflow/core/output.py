"""Atomic small outputs shared by pipeline and GUI writers."""
import json
import os
import stat
import tempfile
from pathlib import Path
from ..task_control import check_cancelled


def publish_video(temporary, target):
    """Publish mkstemp movies with readable modes for Linux/SMB playback."""
    target = Path(target)
    if os.name != "nt":
        mode = stat.S_IMODE(target.stat().st_mode) if target.exists() else 0o644
        os.chmod(temporary, mode | stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    os.replace(temporary, target)


def atomic_json_write(path, payload):
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".autoflow-", suffix=".json", dir=target.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
        check_cancelled()
        if target.exists():
            os.chmod(temporary, stat.S_IMODE(target.stat().st_mode))
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return str(target)
