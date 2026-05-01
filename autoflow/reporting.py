import json
import os

import numpy as np

from .algorithms import load_metrics_as_table


def load_metrics_from_output(out_dir):
    metrics_path = os.path.join(out_dir, "plane_metrics.json")
    qc_path = os.path.join(out_dir, "plane_qc.json")
    if not os.path.isfile(metrics_path):
        return None, None, None
    qc_p = qc_path if os.path.isfile(qc_path) else None
    table_rows, raw_metrics, qc_data = load_metrics_as_table(metrics_path, qc_p)
    return table_rows, raw_metrics, qc_data


def print_metrics_summary(table_rows):
    if not table_rows:
        print("  No metrics to summarize.")
        return
    print(
        f"  {'Plane':>6} {'Path':>5} {'Net Flow(mL/beat)':>18} "
        f"{'Peak Velocity(cm/s)':>20} {'Mean Velocity(cm/s)':>20} "
        f"{'Reflux':>7} {'IC':>6}"
    )
    print(f"  {'-'*6} {'-'*5}  {'-'*18} {'-'*20} {'-'*20} " f"{'-'*7} {'-'*6}")
    for row in table_rows:
        pidx = row.get("plane_index", "?")
        path = row.get("path_index", "?")
        nf = row.get("netflow_mL_beat", 0.0)
        pv_ = row.get("peakv_cm_s", 0.0)
        if "meanv_signed_cm_s" in row:
            mv = row["meanv_signed_cm_s"]
        else:
            mv = row.get("meanv_cm_s", 0.0)
        refl = row.get("reflux_fraction", 0.0)
        ic = row.get("path_ic", 1.0)
        print(f"  {pidx:>6} {path:>5} {nf:>18.4f} " f"{pv_:>20.3f} {mv:>20.3f} {refl:>7.3f} {ic:>6.3f}")


def _format_path_group(v):
    if v is None:
        return "?"
    if isinstance(v, dict):
        if "paths" in v:
            v = v["paths"]
        elif "path_indices" in v:
            v = v["path_indices"]
    if isinstance(v, (int, np.integer)):
        return f"Branch{int(v)}"
    if isinstance(v, str):
        return v
    try:
        vals = list(v)
    except Exception:
        return str(v)
    if not vals:
        return "-"
    out = []
    for x in vals:
        if isinstance(x, (int, np.integer)):
            out.append(f"Branch{int(x)}")
        else:
            out.append(str(x))
    return "+".join(out)


def _fork_side_text(fork):
    if not isinstance(fork, dict):
        return "left=?", "right=?"

    left = (
        fork.get("left")
        or fork.get("left_paths")
        or fork.get("left_path_indices")
        or fork.get("in_paths")
        or fork.get("in_path_indices")
    )
    right = (
        fork.get("right")
        or fork.get("right_paths")
        or fork.get("right_path_indices")
        or fork.get("out_paths")
        or fork.get("out_path_indices")
    )

    return f"{_format_path_group(left)}", f"{_format_path_group(right)}"


def print_qc_summary(qc_data, forks=None):
    if not qc_data:
        print("  No QC results to summarize.")
        return

    items = []
    if isinstance(qc_data, list):
        items = qc_data
    elif isinstance(qc_data, dict):
        if isinstance(qc_data.get("forks"), list):
            items = qc_data["forks"]
        elif isinstance(qc_data.get("fork_qc"), list):
            items = qc_data["fork_qc"]
        else:
            for k, v in qc_data.items():
                if isinstance(v, dict):
                    row = dict(v)
                    row.setdefault("fork_index", k)
                    items.append(row)

    rows = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        fork_idx = item.get("fork_index", item.get("fork_id", item.get("fork", i)))
        ic = item.get("internal_consistency", item.get("ic", item.get("path_ic", np.nan)))

        fork_obj = None
        if isinstance(forks, list):
            try:
                if isinstance(fork_idx, (int, np.integer)) and 0 <= int(fork_idx) < len(forks):
                    fork_obj = forks[int(fork_idx)]
                elif i < len(forks):
                    fork_obj = forks[i]
            except Exception:
                pass

        left_txt, right_txt = _fork_side_text(fork_obj)
        rows.append((fork_idx, ic, left_txt, right_txt))

    if not rows:
        print("  No fork-level QC results found.")
        print(json.dumps(qc_data, ensure_ascii=False, indent=2))
        return

    print(f"  {'Fork':>6} {'Internal Consistency':>24} {'Left':>24} {'Right':>24}")
    print(f"  {'-'*6} {'-'*24} {'-'*24} {'-'*24}")
    for fork_idx, ic, left_txt, right_txt in rows:
        ic_str = f"{float(ic):.6f}" if np.isfinite(ic) else "nan"
        print(f"  {str(fork_idx):>6} {ic_str:>24} {left_txt:>24} {right_txt:>24}")
