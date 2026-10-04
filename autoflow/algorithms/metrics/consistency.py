"""Internal-consistency summaries for paths, labels and forks."""

import numpy as np


def summarize_internal_consistency(plane_metrics, path_info=None, forks=None):
    by_path = {}
    for metric in plane_metrics:
        pidx = int(metric.get("path_index", -1))
        by_path.setdefault(pidx, []).append(abs(float(metric.get("netflow_mL_beat", 0.0))))
    path_ic = {}
    by_path_mean = {}
    for pidx, values in by_path.items():
        arr = np.abs(np.asarray(values, dtype=float))
        mu = float(np.mean(arr)) if len(arr) else 0.0
        by_path_mean[pidx] = mu
        if len(arr) <= 1:
            # A path with zero or one usable plane has no consistency
            # comparison to make.  Report it as undefined instead of
            # presenting a vacuous perfect score.
            ic = None
        elif mu <= 1e-12:
            ic = 1.0 if float(np.max(arr)) <= 1e-12 else 0.0
        else:
            ic = 1.0 - float(np.mean(np.abs(arr - mu)) / mu)
        path_ic[str(int(pidx))] = None if ic is None else float(np.clip(ic, 0.0, 1.0))
    # Keep topology paths with no usable plane metrics visible in QC as
    # undefined.  They must not silently disappear or be interpreted as zero
    # flow by fork consistency calculations.
    if path_info is not None:
        for pidx in range(len(path_info)):
            path_ic.setdefault(str(int(pidx)), None)
    # When segmentation filtering is active, expose an additional consistency
    # view keyed by the numeric segmentation label.  Path consistency remains
    # available for topology QC and backwards compatibility.
    by_seg = {}
    for metric in plane_metrics:
        label = int(metric.get("segmentation_label", 0) or 0)
        if label > 0:
            by_seg.setdefault(label, []).append(abs(float(metric.get("netflow_mL_beat", 0.0))))
    segmentation_ic = {}
    for label, values in by_seg.items():
        arr = np.asarray(values, dtype=float)
        mu = float(np.mean(arr)) if len(arr) else 0.0
        if len(arr) <= 1 or mu <= 1e-12:
            ic = 1.0
        else:
            ic = 1.0 - float(np.mean(np.abs(arr - mu)) / mu)
        segmentation_ic[str(int(label))] = float(np.clip(ic, 0.0, 1.0))
    fork_items = []
    fork_ic = {}
    for fork_id, fork in enumerate(forks or []):
        left = [int(x) for x in fork.get("left", [])]
        right = [int(x) for x in fork.get("right", [])]
        sum_left = float(np.sum([abs(by_path_mean.get(x, 0.0)) for x in left]))
        sum_right = float(np.sum([abs(by_path_mean.get(x, 0.0)) for x in right]))
        # A topology fork with no incoming or no outgoing path has no
        # meaningful conservation comparison.  This can happen when local
        # flow orientation is ambiguous; report it as undefined instead of
        # manufacturing an IC of zero (or one for an empty fork).
        missing_left = [x for x in left if x not in by_path_mean]
        missing_right = [x for x in right if x not in by_path_mean]
        if not left or not right:
            ic = None
            status = "one_sided_topology"
        elif missing_left or missing_right:
            # A missing path means no plane supplied a measurable flow for
            # that side.  Treating it as numerical zero would bias the fork
            # conservation score, so report the comparison as incomplete.
            ic = None
            status = "missing_path_metrics"
        else:
            denom = sum_left + sum_right
            if denom <= 1e-12:
                ic = 1.0
            else:
                ic = 1.0 - 2.0 * abs(sum_left - sum_right) / denom
            ic = float(np.clip(ic, 0.0, 1.0))
            status = "ok"
        fork_ic[str(int(fork_id))] = ic
        item = {
            "fork_id": int(fork_id),
            "left": left,
            "right": right,
            "crosspoint": fork.get("crosspoint", [0.0, 0.0, 0.0]),
            "node": int(fork.get("node", -1)),
            "ic": ic,
            "status": status,
        }
        if path_info is not None:
            item["left_dirs"] = [path_info[x].get("direction_text", "") for x in left if 0 <= x < len(path_info)]
            item["right_dirs"] = [path_info[x].get("direction_text", "") for x in right if 0 <= x < len(path_info)]
        fork_items.append(item)
    return {"path_ic": path_ic, "segmentation_label_ic": segmentation_ic,
            "fork_ic": fork_ic, "forks": fork_items}


def apply_internal_consistency_to_metrics(plane_metrics, path_info=None, forks=None):
    metrics = [dict(metric) for metric in plane_metrics]
    qc = summarize_internal_consistency(metrics, path_info=path_info, forks=forks)
    for metric in metrics:
        pidx = str(int(metric.get("path_index", -1)))
        path_ic = qc["path_ic"].get(pidx, None)
        metric["path_ic"] = None if path_ic is None else float(path_ic)
        seg_label = str(int(metric.get("segmentation_label", 0) or 0))
        metric["segmentation_label_ic"] = float(qc.get("segmentation_label_ic", {}).get(seg_label, 1.0))
        rel = []
        for fork in qc.get("forks", []):
            pid = int(metric.get("path_index", -1))
            if pid in fork.get("left", []) or pid in fork.get("right", []):
                role = "incoming" if pid in fork.get("left", []) else "outgoing"
                fork_ic_value = fork.get("ic", 1.0)
                rel.append({
                    "fork_id": int(fork.get("fork_id", -1)),
                    "role": role,
                    "ic": None if fork_ic_value is None else float(fork_ic_value),
                })
        metric["fork_ic"] = rel
    return metrics, qc
