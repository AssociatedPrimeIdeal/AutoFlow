import json
from models import Workspace


def save_workspace_file(path, workspace):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(workspace.snapshot_dict(), f, ensure_ascii=False, indent=2)


def load_workspace_file(path):
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    ws = Workspace()
    ws.restore_dict(d)
    return ws