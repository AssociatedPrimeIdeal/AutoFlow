"""GUI entry point for the folder distribution."""

import datetime
import multiprocessing
import os
from pathlib import Path
import sys
import traceback


def main():
    multiprocessing.freeze_support()
    from autoflow.ui.launcher import main as run

    run()
    return 0


if __name__ == "__main__":
    if sys.stdout is None or sys.stderr is None:
        directory = Path(os.environ.get("LOCALAPPDATA", str(Path(__file__).resolve().parents[1]))) / "AutoFlow" / "logs"
        directory.mkdir(parents=True, exist_ok=True)
        log = directory / ("startup-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}.log")
        stream = log.open("a", encoding="utf-8", buffering=1)
        sys.stdout = sys.stderr = stream
        print(f"AutoFlow startup: {sys.executable}", flush=True)
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
