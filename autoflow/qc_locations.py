"""Typed QC scopes and physical anchors; display text is never parsed."""

import hashlib
import json
import numpy as np

from .core.plane_state import metric_validity, selected_metric, workspace_signature


def qc_status(check):
    """Quality threshold failures need review; only execution failures are errors."""
    status = str(check.get("status", "unknown")).lower()
    if status == "fail" and check.get("failure_kind") != "execution":
        return "warn"
    return status


def _issue(check, ws, kind="global", plane_ids=(), path_ids=(), *, observed=None, unit=None, validity=None):
    scope = {"scope_kind": kind, "plane_ids": list(plane_ids), "path_ids": list(path_ids),
             "label_ids": [], "phase_scope": "whole_cycle", "location_method": {
                 "plane": "plane_center", "path": "path_representative", "fork": "fork_node", "global": "none"}[kind]}
    signature = workspace_signature(ws)
    identity = (check["id"], scope, signature)
    check.update(scope, rule_id=check["id"].split(":")[0], rule_version=2,
                 issue_id=hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20],
                 dependency_signature=signature, observed=observed if observed is not None else check.get("value"),
                 unit=unit, evidence_ref="checks/"+check["id"], validity=validity or check.get("validity", "current"),
                 review={"status": "unreviewed"})
    return check


def annotate_checks(ws, checks):
    """Add actionable children while retaining existing summary check IDs."""
    for check in checks:
        _issue(check, ws)
        details = []
        if check["id"] in {"planes.geometry", "hemodynamics.plane_metrics"}:
            paths = ws.centerline_paths_smooth or ws.centerline_paths
            for index, plane in enumerate(ws.planes):
                state = metric_validity(plane, selected_metric(ws, index, current_only=False))
                status, text, value, unit = "pass", "", None, None
                if check["id"] == "planes.geometry":
                    n, c = np.asarray(plane.normal), np.asarray(plane.center)
                    if not np.all(np.isfinite(c)) or not np.all(np.isfinite(n)) or np.linalg.norm(n) < 1e-12:
                        status, text = "fail", "Non-finite center or invalid plane normal."
                    elif 0 <= plane.path_index < len(paths):
                        path = np.asarray(paths[plane.path_index]).reshape(-1, 3)
                        if len(path) > 1:
                            nearest = int(np.argmin(np.linalg.norm(path-c, axis=1)))
                            tangent = path[min(nearest+1, len(path)-1)]-path[max(0, nearest-1)]
                            length = np.linalg.norm(tangent)
                            if length > 1e-12:
                                angle = float(np.degrees(np.arccos(np.clip(abs(np.dot(n, tangent)/np.linalg.norm(n)/length), 0, 1))))
                                offset = float(np.linalg.norm(path[nearest]-c))
                                status = "fail" if angle > 45 else ("warn" if angle > 20 or offset > max(2., 2*np.mean(ws.resolution)) else "pass")
                                text, value, unit = f"Normal angle {angle:.2f} deg; centerline offset {offset:.2f} mm.", {"angle_deg": angle, "offset_mm": offset}, "deg / mm"
                else:
                    metric = selected_metric(ws, index)
                    if state != "current":
                        status, text = "not_run", "Previous result / needs recalculation" if state == "stale" else "Not calculated"
                    elif metric:
                        areas = np.asarray(metric.get("area_mm2", []), dtype=float)
                        valid = np.asarray(metric.get("sample_valid_t", np.isfinite(areas) & (areas > 0)), dtype=bool)
                        cv = float(np.std(areas)/np.mean(areas)) if areas.size and np.mean(areas) > 0 else None
                        status = "fail" if not valid.size or not np.all(valid) else ("warn" if cv and cv > .2 else "pass")
                        value = {"invalid_phases": np.flatnonzero(~valid).tolist(), "area_cv": cv}
                        text = f"Invalid phases: {value['invalid_phases']}; area CV: {cv if cv is not None else 'unavailable'}."
                if status != "pass":
                    child = {"id": check["id"]+":"+plane.plane_id, "stage": check["stage"], "title": f"Plane {index}",
                             "status": status, "summary": text, "threshold": check.get("threshold"),
                             "action": check.get("action") or "Recalculate this plane after reviewing its ROI.", "required": True}
                    details.append(_issue(child, ws, "plane", [plane.plane_id],
                                          [plane.path_index] if plane.path_index >= 0 else [], observed=value, unit=unit,
                                          validity="stale" if state == "stale" else "current"))
        elif check["id"] == "centerline.cycles":
            for child in check.get("details", []):
                cycle = child.get("value", {})
                paths = list(cycle.get("path_indices", []))
                _issue(child, ws, "path", [p.plane_id for p in ws.planes if p.path_index in paths],
                       paths, observed=cycle, unit="mm")
                child.update(label_ids=list(cycle.get("label_ids", [])), phase_scope="static_geometry",
                             location_method="graph_cycle_center", cycle_id=cycle.get("cycle_id"),
                             center_local_mm=cycle.get("center_local_mm"),
                             cycle_points_local_mm=cycle.get("points_local_mm", []))
                details.append(child)
        elif check["id"] == "hemodynamics.flow_consistency":
            for path_id, value in (check.get("value", {}).get("path_ic", {}) or {}).items():
                if value is None or not np.isfinite(value) or float(value) >= .8:
                    continue
                path_id = int(path_id)
                child = dict(check, id=f"{check['id']}:path:{path_id}", title=f"Path {path_id} · whole cycle",
                             summary=f"Path-level internal consistency {float(value):.3f}.",
                             status="fail" if value < .6 else "warn")
                details.append(_issue(child, ws, "path", [p.plane_id for p in ws.planes if p.path_index == path_id],
                                      [path_id], observed=float(value), unit="ratio"))
            for fork_id, value in (check.get("value", {}).get("fork_ic", {}) or {}).items():
                if value is None or not np.isfinite(value) or float(value) >= .8:
                    continue
                fork_id = int(fork_id)
                if not 0 <= fork_id < len(ws.forks):
                    continue
                fork = ws.forks[fork_id]
                paths = [int(i) for i in fork.get("left", [])+fork.get("right", [])]
                child = dict(check, id=f"{check['id']}:fork:{fork_id}", title=f"Fork {fork_id} · whole cycle",
                             summary=f"Fork-level conservation consistency {float(value):.3f}.", status="fail" if value < .6 else "warn")
                child = _issue(child, ws, "fork", [p.plane_id for p in ws.planes if p.path_index in paths], paths,
                               observed=float(value), unit="ratio")
                child.update(fork_id=fork_id, node_id=int(fork.get("node", -1)))
                details.append(child)
        if details:
            check["details"] = details
        def legacy_details(parent):
            for index, detail in enumerate(parent.get("details", [])):
                if "issue_id" not in detail:
                    detail.setdefault("id", f"{parent['id']}:detail:{index}")
                    detail.setdefault("stage", parent.get("stage", ""))
                    detail.setdefault("title", detail["id"])
                    detail.setdefault("summary", "")
                    detail.setdefault("status", "not_run")
                    _issue(detail, ws)
                legacy_details(detail)
        legacy_details(check)
        if check["id"] == "hemodynamics.plane_metrics":
            states = [(p.plane_id, metric_validity(p, selected_metric(ws, i, current_only=False))) for i, p in enumerate(ws.planes)]
            stale = [uid for uid, state in states if state == "stale"]
            missing = [uid for uid, state in states if state == "missing"]
            if stale or missing:
                check.update(status="not_run", summary=f"{len(stale)} stale and {len(missing)} uncalculated planes; current results only.",
                             validity="stale" if stale else "missing")
    return checks


def iter_issues(report):
    def visit(check):
        yield check
        for child in check.get("details", []):
            yield from visit(child)
    for item in report.get("checks", []):
        yield from visit(item)


def issue_anchors(ws, report):
    """Anchors in local physical mm. Paths explicitly use representative points."""
    planes = {p.plane_id: p for p in ws.planes}
    paths = ws.centerline_paths_smooth or ws.centerline_paths
    anchors = []
    for issue in iter_issues(report):
        stale = issue.get("validity") == "stale"
        if issue.get("status") not in {"warn", "fail"} and not stale:
            continue
        points = []
        if issue.get("location_method") == "graph_cycle_center":
            point = issue.get("center_local_mm")
            if point is not None and np.asarray(point).shape == (3,):
                points = [point]
        elif issue.get("scope_kind") == "plane":
            points = [planes[uid].center for uid in issue.get("plane_ids", []) if uid in planes]
        elif issue.get("scope_kind") == "path":
            for index in issue.get("path_ids", []):
                if 0 <= index < len(paths) and len(paths[index]):
                    path = np.asarray(paths[index], dtype=float)
                    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
                    points.append([np.interp(arc[-1]/2, arc, path[:, axis]) for axis in range(3)])
        elif issue.get("scope_kind") == "fork":
            node = int(issue.get("node_id", -1))
            if 0 <= node < len(ws.graph.points):
                points = [ws.graph.points[node]]
        for point in points:
            if np.all(np.isfinite(point)):
                anchors.append({"issue_id": issue["issue_id"], "point_mm": np.asarray(point).tolist(),
                                "status": "stale" if stale else issue["status"], "scope_kind": issue["scope_kind"]})
    return anchors
