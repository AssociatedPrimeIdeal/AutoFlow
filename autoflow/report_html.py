"""Self-contained, interactive English HTML for an immutable report snapshot.

Only the Python standard library is used here.  Geometry and measurements are
copied from the snapshot; the renderer does not run analysis or infer anatomy.
"""

from __future__ import annotations

import base64
import hashlib
import html
import json
import math
from collections import OrderedDict

from .report_html_assets import REPORT_CSS, REPORT_JS


_DEFAULT_SECTIONS = {"metrics", "curves", "images", "qc", "methods", "videos"}
_RANK = {"error": 4, "warning": 3, "unavailable": 2, "pass": 1}
_LABELS = {"error": "Error", "warning": "Warning", "unavailable": "Unavailable", "pass": "Pass"}


def _dict(value):
    return value if isinstance(value, dict) else {}


def _list(value):
    return value if isinstance(value, (list, tuple)) else []


def _text(value):
    return "" if value is None else str(value)


def _number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def _first_number(*values):
    return next((value for value in values if _number(value) is not None), None)


def _unique(values):
    return list(dict.fromkeys(values))


def _ids(values):
    result = []
    for value in _list(values):
        if isinstance(value, bool) or value is None:
            continue
        try:
            integer = int(value)
        except (ValueError, TypeError, OverflowError):
            continue
        if integer >= 0 and integer == value:
            result.append(integer)
    return _unique(result)


def _point(value):
    values = _list(value)
    return list(values) if len(values) == 3 and all(_number(v) is not None for v in values) else None


def _safe(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    return str(value)


def _encoded(value):
    return base64.b64encode(json.dumps(_safe(value), ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")).decode("ascii")


def _digest(value):
    return hashlib.sha256(json.dumps(_safe(value), sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:20]


def _esc(value):
    return html.escape(_text(value), quote=True)


def _fmt(value, digits=3):
    number = _number(value)
    return "—" if number is None else f"{number:,.{digits}f}".rstrip("0").rstrip(".") if digits else f"{number:,.0f}"


def _status(raw, signature=""):
    validity = _text(raw.get("validity")).lower()
    own_signature = _text(raw.get("dependency_signature"))
    if validity in {"stale", "missing", "partial", "unsupported", "not_assessable", "unavailable"}:
        return "unavailable"
    if own_signature and signature and own_signature != signature:
        return "unavailable"
    from .qc_locations import qc_status
    return {"fail": "error", "error": "error", "failed": "error", "warn": "warning", "warning": "warning", "pass": "pass"}.get(qc_status(raw), "unavailable")


def _worst(states):
    return max(states, key=lambda state: _RANK.get(state, 2), default="unavailable")


def _state_label(state, validity=""):
    if validity == "stale":
        return "Previous result / needs recalculation"
    return {"current": "Current measurements", "unverified": "Validity not recorded", "missing": "Missing / not calculated", "partial": "Incomplete acquired cycle", "not_run": "Not assessed", "unsupported": "Unsupported input", "not_assessable": "Not assessable"}.get(validity, _LABELS.get(state, "Unavailable"))


def _label_definitions(spatial):
    result = {}
    for raw in _list(spatial.get("label_definitions")):
        raw = _dict(raw)
        label = _first_number(raw.get("label_id"), raw.get("global_id"), raw.get("id"))
        if label is None:
            continue
        name = _text(raw.get("name") or raw.get("source_token"))
        result[label] = {"id": label, "name": name, "nameSource": _text(raw.get("name_source") or raw.get("source")), "aliases": _safe(raw.get("aliases", [])), "verified": raw.get("verified") is True}
    return result


def _label_title(label, definitions):
    record = definitions.get(label, {})
    name = record.get("name")
    return f"{name} [ID {label}]" if name else f"Label {label}"


def _normalise_plane(raw, index, phase_count, definitions):
    raw = _dict(raw)
    metric = _dict(raw.get("metrics"))
    validity = _text(raw.get("validity") or ("unverified" if metric else "missing"))
    plane_id = _text(raw.get("plane_id") or f"legacy-plane-{index}")
    plane_index = _first_number(raw.get("index"), raw.get("plane_index"), index)
    path_index = _first_number(raw.get("path_index"), metric.get("path_index"), -1)
    label = _first_number(raw.get("segmentation_label"), raw.get("label"), metric.get("segmentation_label"), metric.get("label"))
    supplied_name = _text(raw.get("label_name") or metric.get("label_name"))
    label_name = _label_title(label, definitions) if label is not None else supplied_name or _text(raw.get("group")) or "Unlabelled plane"
    if label is not None and label not in definitions and supplied_name:
        label_name = f"{supplied_name} [ID {label}]"
    q_key = "flowrate_signed_mL_s" if "flowrate_signed_mL_s" in metric else "flowrate_mL_s"
    sources = {"q": _list(metric.get(q_key)), "area": _list(metric.get("area_mm2")), "meanVelocity": _list(metric.get("meanv_signed_cm_s_t", metric.get("meanv_cm_s_t"))), "peakVelocity": _list(metric.get("peakv_cm_s_t"))}
    count = max([phase_count, *[len(v) for v in sources.values()], len(_list(metric.get("sample_valid_t")))])
    validity_mask = _list(metric.get("sample_valid_t"))
    current = validity in {"current", "partial", "unverified"} and bool(metric)
    sample_valid = [not validity_mask or (t < len(validity_mask) and validity_mask[t] is True) for t in range(count)]
    series = {key: [(_number(values[t]) if t < len(values) and sample_valid[t] and current else None) for t in range(count)] for key, values in sources.items()}
    # Area is not invented when an older snapshot only supplies flow.  An
    # explicitly non-positive recorded area, however, cannot be a valid ROI.
    for t in range(count):
        if t < len(sources["area"]) and (_number(sources["area"][t]) is None or sources["area"][t] <= 0):
            sample_valid[t] = False
            for values in series.values():
                values[t] = None
    full_cycle = bool(count) and all(sample_valid) and len(sources["q"]) == count and all(value is not None for value in series["q"])
    if current and count and not full_cycle:
        validity = "partial"
    if not metric and validity != "stale":
        validity = "missing"
    aggregate = current and full_cycle and validity != "partial"
    volume_key = "net_netflow_signed_mL_beat" if "net_netflow_signed_mL_beat" in metric else "netflow_mL_beat"
    scalar = lambda key: _number(metric.get(key)) if aggregate else None
    areas = [v for v in series["area"] if v is not None]
    summary = {"meanArea": sum(areas)/len(areas) if aggregate and len(areas) == count else None, "cycleVolume": scalar(volume_key), "cycleVolumeSigned": volume_key == "net_netflow_signed_mL_beat", "peakNormalVelocity": scalar("peakv_cm_s"), "meanNormalVelocity": scalar("meanv_signed_cm_s" if "meanv_signed_cm_s" in metric else "meanv_cm_s"), "refluxFraction": scalar("reflux_fraction")}
    extra_keys = [key for key in metric if any(token in key.lower() for token in ("wss", "pressure", "tke", "vortic", "swirl", "q_criterion"))]
    extras = {key: _safe(metric[key]) for key in extra_keys}
    if not current:
        extras = {key: None for key in extra_keys}
    return {"id": plane_id, "index": plane_index, "pathId": path_index, "labelId": label, "labelName": label_name, "labelDefinition": definitions.get(label), "group": _text(raw.get("group") or raw.get("group_name")), "center": _point(raw.get("center_mm") or raw.get("center")), "normal": _point(raw.get("normal")), "validity": validity, "stateLabel": _state_label("pass" if validity == "current" else "unavailable", validity), "geometryRevision": raw.get("geometry_revision"), "roiRevision": raw.get("roi_revision"), "phaseCount": count, "series": series, "sampleValid": sample_valid, "invalidReasons": _list(metric.get("sample_invalid_reason_t")), "summary": summary, "qConvention": "Signed flow" if q_key == "flowrate_signed_mL_s" else "Plane-normal flow", "extras": extras, "fieldValidity": _dict(metric.get("field_validity")), "staleReason": _text(raw.get("stale_reason") or metric.get("stale_reason")), "issueIds": []}


def _normalise_check(raw, parents, signature, planes, definitions):
    raw = _dict(raw)
    identity = _text(raw.get("issue_id") or raw.get("id") or "check-"+_digest(raw))
    status = _status(raw, signature)
    validity = _text(raw.get("validity") or "current")
    if raw.get("dependency_signature") and signature and raw.get("dependency_signature") != signature:
        validity = "stale"
    plane_ids = _unique([_text(v) for v in _list(raw.get("plane_ids"))])
    path_ids = _ids(raw.get("path_ids"))
    path_ids = _unique(path_ids + [p["pathId"] for p in planes if p["id"] in plane_ids and p["pathId"] >= 0])
    label_ids = _ids(raw.get("label_ids"))
    association_source = "Typed source scope"
    chain = [raw, *reversed(parents)]
    if not label_ids:
        labelled = next((r for r in chain if _ids([_dict(r.get("observed", r.get("value"))).get("label")])), None)
        if labelled:
            label_ids = _ids([_dict(labelled.get("observed", labelled.get("value"))).get("label")])
            association_source = "Structured observed.label" if labelled is raw else "Structured parent observed.label"
    phase_indices = _ids(raw.get("phase_indices"))
    if not phase_indices:
        for node in chain:
            phase = _first_number(node.get("phase_index"), _dict(node.get("observed", node.get("value"))).get("phase_index"))
            if phase is not None and phase >= 0 and int(phase) == phase:
                phase_indices = [int(phase)]
                break
    source_scope = _text(raw.get("scope_kind") or "global")
    scope = "label" if label_ids and not path_ids and not plane_ids else source_scope
    rule = _text(raw.get("rule_id") or _text(raw.get("id")).split(":")[0])
    title = _text(raw.get("title") or raw.get("id") or "QC check")
    source_phase_scope = _text(raw.get("phase_scope") or "whole_cycle")
    review = _dict(raw.get("review"))
    review_current = review.get("status") == "reviewed" and validity == "current" and bool(review.get("dependency_signature")) and review.get("dependency_signature") == raw.get("dependency_signature", signature)
    return {"id": identity, "ruleId": rule, "ruleTitle": _text(parents[0].get("title")) if parents else title, "title": title, "status": status, "sourceStatus": _text(raw.get("status") or "not_run"), "validity": validity, "statusLabel": _state_label(status, validity if validity != "current" else _text(raw.get("status")) if status == "unavailable" else ""), "summary": _text(raw.get("summary") or raw.get("message")), "threshold": _safe(raw.get("threshold")), "observed": _safe(raw.get("observed", raw.get("value"))), "unit": _text(raw.get("unit")), "action": _text(raw.get("action") or "Review the supplied evidence before changing this result."), "stage": _text(raw.get("stage")), "scope": scope, "sourceScope": source_scope, "phaseScope": source_phase_scope, "phaseIndices": phase_indices, "planeIds": plane_ids, "pathIds": path_ids, "labelIds": label_ids, "labelNames": [_label_title(label, definitions) for label in label_ids], "associationSource": association_source, "locationMethod": _text(raw.get("location_method") or "none"), "signature": _text(raw.get("dependency_signature") or signature), "review": {"status": _text(review.get("status") or "unreviewed"), "current": review_current, "note": _text(review.get("note")), "by": _text(review.get("by") or review.get("reviewer")), "at": _text(review.get("at"))}, "sourceIssueIds": [identity], "records": [], "required": raw.get("required") is not False}


def _collect_checks(snapshot, planes, definitions):
    signature = _text(snapshot.get("dependency_signature"))
    all_checks = OrderedDict()
    selected = []

    def visit(raw, parents):
        check = _normalise_check(raw, parents, signature, planes, definitions)
        all_checks.setdefault(check["id"], check)
        children = []
        for child in _list(raw.get("details")):
            if isinstance(child, dict):
                children.extend(visit(child, [*parents, raw]))
        if check["status"] == "pass":
            return children
        if children:
            if _RANK[check["status"]] > max(_RANK[c["status"]] for c in children):
                return [check, *children]
            return children
        return [check]

    for raw in _list(_dict(snapshot.get("quality")).get("checks")):
        if isinstance(raw, dict):
            selected.extend(visit(raw, []))
    spatial = _dict(snapshot.get("spatial_context"))
    for raw in _list(spatial.get("unlocated_checks")):
        if isinstance(raw, dict):
            check = _normalise_check(raw, [], signature, planes, definitions)
            is_new = check["id"] not in all_checks
            all_checks.setdefault(check["id"], check)
            if is_new and check["status"] != "pass":
                selected.append(all_checks[check["id"]])
    for marker in _list(spatial.get("markers")):
        marker = _dict(marker)
        for raw in _list(marker.get("checks")):
            if not isinstance(raw, dict):
                continue
            raw = {**{key: marker[key] for key in ("scope_kind", "plane_ids", "path_ids", "label_ids", "phase_scope", "phase_indices", "location_method") if key in marker}, **raw}
            check = _normalise_check(raw, [], signature, planes, definitions)
            is_new = check["id"] not in all_checks
            all_checks.setdefault(check["id"], check)
            if is_new and check["status"] != "pass":
                selected.append(all_checks[check["id"]])
    selected = list(OrderedDict((c["id"], c) for c in selected).values())
    buckets = OrderedDict()
    for check in selected:
        key = _digest({k: check[k] for k in ("ruleId", "planeIds", "pathIds", "labelIds", "scope", "validity", "signature", "threshold")}) if check["phaseIndices"] and (check["planeIds"] or check["pathIds"] or check["labelIds"]) else check["id"]
        buckets.setdefault(key, []).append(check)
    groups = []
    for key, records in buckets.items():
        if len(records) == 1:
            groups.append(records[0])
            continue
        worst = max(records, key=lambda c: _RANK[c["status"]])
        phases = sorted(_unique([t for c in records for t in c["phaseIndices"]]))
        location = " / ".join(worst["labelNames"] or [f"Path {v}" for v in worst["pathIds"]]) or "Linked planes"
        groups.append({**worst, "id": "group-"+key, "title": f"{location} · {worst['ruleTitle']} · {len(phases)} phases", "summary": f"{len(records)} original records for this rule and location. Expand the individual phase evidence below.", "sourceIssueIds": [c["id"] for c in records], "phaseIndices": phases, "records": sorted(records, key=lambda c: min(c["phaseIndices"] or [-1]))})
    groups.sort(key=lambda c: (-_RANK[c["status"]], c["title"]))
    return list(all_checks.values()), groups


def _normalise_spatial(raw, checks, groups):
    raw = _dict(raw)
    check_map = {c["id"]: c for c in checks}
    group_by_source = {source_id: group["id"] for group in groups for source_id in group["sourceIssueIds"]}
    paths = []
    for index, source in enumerate(_list(raw.get("paths"))):
        source = _dict(source)
        points = [_point(v) for v in _list(source.get("points_mm"))]
        points = [v for v in points if v is not None]
        if points:
            paths.append({"id": _first_number(source.get("path_id"), index), "points": points, "labelIds": _ids(source.get("label_ids")), "group": _text(source.get("group_name")), "sampled": bool(source.get("display_sampled")), "source": _text(source.get("source"))})
    segmentation = _dict(raw.get("segmentation"))
    points, labels = [], []
    raw_labels = _list(segmentation.get("label_ids"))
    for index, point in enumerate(_list(segmentation.get("points_mm"))):
        point = _point(point)
        if point is not None:
            points.append(point)
            labels.append(raw_labels[index] if index < len(raw_labels) and _number(raw_labels[index]) is not None else None)
    skeleton = [_point(p) for p in _list(_dict(raw.get("skeleton")).get("points_mm"))]
    skeleton = [p for p in skeleton if p is not None]
    markers = []
    for index, source in enumerate(_list(raw.get("markers"))):
        source = _dict(source)
        point = _point(source.get("point_mm")) or _point(source.get("position_mm"))
        if point is None:
            continue
        issue_ids = _unique([_text(v) for v in _list(source.get("issue_ids"))] + [_text(v.get("issue_id") or v.get("id")) for v in _list(source.get("checks")) if isinstance(v, dict)])
        resolved = [check_map[uid] for uid in issue_ids if uid in check_map]
        state = _worst([c["status"] for c in resolved])
        if _text(source.get("status")) in {"stale", "missing", "not_run", "unavailable"}:
            state = "unavailable"
        # A green point requires an explicit pass check, not the absence of a
        # failure or a guessed state from valid-looking geometry.
        if not resolved:
            state = "unavailable"
        markers.append({"id": _text(source.get("marker_id") or f"marker-{index}"), "point": point, "status": state, "sourceStatus": _text(source.get("status")), "title": _text(source.get("title") or (resolved[0]["title"] if resolved else "Unassessed location")), "summary": _text(source.get("summary") or (resolved[0]["summary"] if resolved else "No explicit check evidence was supplied for this location.")), "issueIds": issue_ids, "groupIds": _unique([group_by_source[uid] for uid in issue_ids if uid in group_by_source]), "planeIds": _unique([_text(v) for v in _list(source.get("plane_ids"))] + [uid for c in resolved for uid in c["planeIds"]]), "pathIds": _unique(_ids(source.get("path_ids")) + [uid for c in resolved for uid in c["pathIds"]]), "labelIds": _unique(_ids(source.get("label_ids")) + [uid for c in resolved for uid in c["labelIds"]]), "scope": _text(source.get("scope_kind") or "unknown"), "phaseScope": _text(source.get("phase_scope") or "whole_cycle"), "phaseIndices": _unique(_ids(source.get("phase_indices")) + [phase for c in resolved for phase in c["phaseIndices"]]), "locationMethod": _text(source.get("location_method") or "source coordinate")})
    surfaces = []
    for source in _list(raw.get("surfaces")):
        source = _dict(source)
        vertices = [_point(v) for v in _list(source.get("vertices_mm"))]
        if not vertices or any(v is None for v in vertices):
            continue
        triangles = [list(tri) for tri in _list(source.get("triangles")) if len(_list(tri)) == 3 and all(isinstance(i, int) and not isinstance(i, bool) and 0 <= i < len(vertices) for i in tri)]
        if triangles:
            surfaces.append({"vertices": vertices, "triangles": triangles, "labelId": source.get("label_id")})
    bounds = _list(raw.get("bounds_mm"))
    bounds = [_point(bounds[0]), _point(bounds[1])] if len(bounds) == 2 else []
    if len(bounds) != 2 or any(v is None for v in bounds):
        all_points = [*points, *skeleton, *[p for path in paths for p in path["points"]], *[m["point"] for m in markers]]
        bounds = [[min(p[a] for p in all_points) for a in range(3)], [max(p[a] for p in all_points) for a in range(3)]] if all_points else [[-1, -1, -1], [1, 1, 1]]
    return {"available": bool(paths or points or skeleton or surfaces or markers), "coordinateSystem": _text(raw.get("coordinate_system") or "local_mm"), "originMm": _point(raw.get("origin_mm")), "bounds": bounds, "paths": paths, "skeleton": skeleton, "segmentation": {"points": points, "labels": labels, "representation": _text(segmentation.get("representation") or "unavailable"), "source": _text(segmentation.get("source")), "phaseScope": _text(segmentation.get("phase_scope") or "not supplied"), "sampled": bool(segmentation.get("display_sampled")), "originalCount": segmentation.get("original_count")}, "surfaces": surfaces, "markers": markers, "notes": [_text(v) for v in _list(raw.get("notes"))], "globalIssueIds": [_text(v) for v in _list(raw.get("global_issue_ids"))], "unlocatedIssueIds": [_text(v) for v in _list(raw.get("unlocated_issue_ids"))]}


def _build_model(snapshot, sections):
    spatial_raw = _dict(snapshot.get("spatial_context"))
    definitions = _label_definitions(spatial_raw)
    phase_count = int(_first_number(snapshot.get("phase_count"), 0))
    planes = [_normalise_plane(raw, index, phase_count, definitions) for index, raw in enumerate(_list(snapshot.get("planes")))]
    phase_count = max([phase_count, *[p["phaseCount"] for p in planes]])
    checks, groups = _collect_checks(snapshot, planes, definitions)
    spatial = _normalise_spatial(spatial_raw, checks, groups)
    located_ids = {uid for marker in spatial["markers"] for uid in marker["issueIds"]}
    for group in groups:
        group["located"] = any(uid in located_ids for uid in group["sourceIssueIds"])
        group["global"] = not group["planeIds"] and not group["pathIds"] and not group["labelIds"]
    vessel_map = OrderedDict()

    def vessel(key, path_id=None, label_id=None):
        if key in vessel_map:
            return vessel_map[key]
        linked = [p for p in planes if p["pathId"] == path_id] if path_id is not None else [p for p in planes if p["labelId"] == label_id]
        names = _unique([p["labelName"] for p in linked])
        if not names and label_id is not None:
            names = [_label_title(label_id, definitions)]
        if not names and path_id is not None:
            source_path = next((p for p in spatial["paths"] if p["id"] == path_id), {})
            names = [_label_title(label, definitions) for label in source_path.get("labelIds", [])]
        row = {"key": key, "title": " / ".join(names) or "Unlabelled path", "pathId": path_id, "labelId": label_id, "planeIds": [p["id"] for p in linked], "groupIds": [], "markerIds": [], "status": "unavailable", "location": f"Path {path_id}" if path_id is not None else f"Label {label_id}"}
        vessel_map[key] = row
        return row

    for plane in planes:
        if plane["pathId"] >= 0:
            vessel(f"path:{plane['pathId']}", path_id=plane["pathId"])
        elif plane["labelId"] is not None:
            vessel(f"label:{plane['labelId']}", label_id=plane["labelId"])
    for path in spatial["paths"]:
        vessel(f"path:{path['id']}", path_id=path["id"])
    for group in groups:
        keys = []
        for path_id in group["pathIds"]:
            keys.append(vessel(f"path:{path_id}", path_id=path_id)["key"])
        if not keys:
            for label_id in group["labelIds"]:
                keys.append(vessel(f"label:{label_id}", label_id=label_id)["key"])
        group["vesselKeys"] = keys
        for key in keys:
            vessel_map[key]["groupIds"].append(group["id"])
    for row in vessel_map.values():
        linked_markers = [m for m in spatial["markers"] if (row["pathId"] is not None and row["pathId"] in m["pathIds"]) or (row["labelId"] is not None and row["labelId"] in m["labelIds"]) or any(p in row["planeIds"] for p in m["planeIds"])]
        row["markerIds"] = [m["id"] for m in linked_markers]
        row["status"] = _worst([g["status"] for g in groups if g["id"] in row["groupIds"]] + [m["status"] for m in linked_markers])
        row["errors"] = sum(g["status"] == "error" for g in groups if g["id"] in row["groupIds"])
        row["warnings"] = sum(g["status"] == "warning" for g in groups if g["id"] in row["groupIds"])
    vessels = sorted(vessel_map.values(), key=lambda v: (-_RANK[v["status"]], v["title"], v["pathId"] if v["pathId"] is not None else -1))
    for plane in planes:
        plane["issueIds"] = [g["id"] for g in groups if plane["id"] in g["planeIds"] or plane["pathId"] in g["pathIds"] or plane["labelId"] is not None and plane["labelId"] in g["labelIds"]]
    counts = {"errors": sum(g["status"] == "error" for g in groups), "warnings": sum(g["status"] == "warning" for g in groups), "unavailable": sum(g["status"] == "unavailable" for g in groups), "rawRecords": sum(len(g["sourceIssueIds"]) for g in groups), "planes": len(planes), "currentPlanes": sum(p["validity"] == "current" for p in planes), "stalePlanes": sum(p["validity"] == "stale" for p in planes), "missingPlanes": sum(p["validity"] in {"missing", "partial", "unverified"} for p in planes), "markers": {state: sum(m["status"] == state for m in spatial["markers"]) for state in _RANK}}
    quality = _dict(snapshot.get("quality"))
    family_states = _dict(snapshot.get("metric_output_status"))
    families = []
    for key, title, aliases in (("wss", "Wall shear stress", ("wss",)), ("pressure", "Pressure / pressure gradient", ("pg", "pressure")), ("vortex", "Vortex measurements", ("vortex",)), ("tke", "Turbulent kinetic energy", ("tke",)), ("pwv", "Pulse-wave velocity", ("pwv",))):
        state = next((_text(family_states[a]) for a in aliases if a in family_states), "not_recorded")
        if key == "pwv" and state == "not_recorded" and snapshot.get("pwv"):
            state = "complete" if all(p.get("status") == "ok" for p in _list(snapshot.get("pwv"))) else "not_assessable"
        families.append({"key": key, "title": title, "state": state})
    images = []
    for source in _list(snapshot.get("images")):
        source = _dict(source)
        if source.get("mime") != "image/png" or source.get("dependency_signature") != snapshot.get("dependency_signature"):
            continue
        try:
            data = base64.b64decode(source.get("data_base64", ""), validate=True)
        except (ValueError, TypeError):
            continue
        if len(data) < 33 or not data.startswith(b"\x89PNG\r\n\x1a\n") or data[12:16] != b"IHDR":
            continue
        images.append({"data": base64.b64encode(data).decode("ascii"), "caption": _text(source.get("caption")), "planeId": source.get("plane_id"), "phase": _number(source.get("phase"))})
    source = _text(snapshot.get("source"))
    source_name = source.replace("\\", "/").rsplit("/", 1)[-1] or "Untitled case"
    return {"schema": "autoflow.html-view.v1", "title": source_name, "source": source, "createdAt": _text(snapshot.get("created_at")), "revision": snapshot.get("analysis_revision"), "snapshotId": _text(snapshot.get("snapshot_id")), "signature": _text(snapshot.get("dependency_signature")), "overallStatus": _text(quality.get("overall_status") or "not_recorded"), "interpretation": _text(quality.get("interpretation") or "Automated checks support review; they do not identify a clinical diagnosis."), "sections": sorted(sections), "phaseCount": phase_count, "rrMs": _number(snapshot.get("rr_ms")), "planes": planes, "checks": checks, "groups": groups, "vessels": vessels, "spatial": spatial, "counts": counts, "families": families, "images": images, "labelDefinitions": list(definitions.values()), "pwv": _safe(_list(snapshot.get("pwv")))}


def _badge(state, label=None):
    return f'<span class="badge {_esc(state)}">{_esc(label or _LABELS.get(state, "Unavailable"))}</span>'


def _evidence_markup(check):
    observed = json.dumps(check["observed"], ensure_ascii=False, indent=2) if isinstance(check["observed"], (dict, list)) else _text(check["observed"])
    threshold = json.dumps(check["threshold"], ensure_ascii=False) if isinstance(check["threshold"], (dict, list)) else _text(check["threshold"])
    location = " · ".join([*([f"Path {v}" for v in check["pathIds"]]), *check["labelNames"]]) or ("Global / no spatial anchor" if check.get("global") else check["scope"])
    rows = [("Message", check["summary"] or "No additional message supplied."), ("Scope", location), ("Phases", ", ".join(map(str, check["phaseIndices"])) if check["phaseIndices"] else check["phaseScope"].replace("_", " ")), ("Threshold", threshold or "Not supplied"), ("Next action", check["action"]), ("Source issue ID", ", ".join(check["sourceIssueIds"]))]
    output = '<dl class="evidence-grid">'+"".join(f"<dt>{_esc(label)}</dt><dd>{_esc(value)}</dd>" for label, value in rows)+"</dl>"
    if observed:
        output += '<details class="source-values" data-print-open><summary>Recorded evidence</summary><pre>'+_esc(observed)+"</pre></details>"
    if check["records"]:
        output += '<details class="phase-records" data-print-open><summary>'+str(len(check["records"]))+' original phase records</summary><div class="table-wrap"><table><thead><tr><th>Phase</th><th>Result</th><th>Message and recorded values</th><th>Source ID</th></tr></thead><tbody>'
        for record in check["records"]:
            output += "<tr><td>"+_esc(", ".join(map(str, record["phaseIndices"])))+"</td><td>"+_badge(record["status"])+"</td><td>"+_esc(record["summary"])+"<pre>"+_esc(json.dumps(record["observed"], ensure_ascii=False, indent=2))+"</pre></td><td class=mono>"+_esc(record["id"])+"</td></tr>"
        output += "</tbody></table></div></details>"
    return output


def _required_markup(model):
    groups = model["groups"]
    rows = []
    for index, check in enumerate(groups):
        location = "Global" if check["global"] else "Located" if check["located"] else "No spatial anchor"
        rows.append(f'<details class="finding {check["status"]}" id="finding-{index}" data-issue-id="{_esc(check["id"])}" data-print-open><summary>{_badge(check["status"], check["statusLabel"])}<span><strong>{_esc(check["title"])}</strong><small>{_esc(check["summary"])} · {_esc(location)}</small></span></summary><div class="finding-body"><button type="button" data-select-issue="{_esc(check["id"])}" class="no-print">Inspect linked evidence</button>'+_evidence_markup(check)+"</div></details>")
    if not rows:
        rows.append('<p class="empty">No required findings were supplied. Missing check coverage is not treated as a pass.</p>')
    return '<section class="panel" id="required"><div class="panel-heading"><div><h2>Required findings and missing results</h2><p>Review groups retain every original check. Global findings and unavailable locations remain separate from vessel markers.</p></div><span class="count">'+str(len(groups))+' groups</span></div><div class="panel-body">'+"".join(rows)+"</div></section>"


def _measurement_markup(model):
    rows = []
    for plane in model["planes"]:
        summary = plane["summary"]
        row_state = "pass" if plane["validity"] == "current" else "unavailable"
        rows.append('<tr data-measurement-plane="'+_esc(plane["id"])+'"><td><button class="link" type="button" data-select-plane="'+_esc(plane["id"])+'">'+_esc(plane["labelName"])+"</button><small>"+_esc(f"Path {plane['pathId']}" if plane["pathId"] >= 0 else "No path association")+"</small></td><td>"+_esc(plane["index"])+"</td><td>"+_badge(row_state, plane["stateLabel"])+"<small>"+str(sum(plane["sampleValid"]))+" / "+str(plane["phaseCount"])+" samples</small></td>"+"".join('<td class="num">'+_fmt(value)+"</td>" for value in (summary["meanArea"], summary["cycleVolume"], summary["peakNormalVelocity"], summary["meanNormalVelocity"], summary["refluxFraction"]*100 if summary["refluxFraction"] is not None else None))+"</tr>")
    table = '<div class="table-wrap"><table class="metrics-table"><thead><tr><th>Vessel / path</th><th>Plane</th><th>Result</th><th>Mean area<br>mm²</th><th>Net flow volume<br>mL/cycle</th><th>Peak |normal v|<br>cm/s</th><th>Mean normal v<br>cm/s</th><th>Reflux<br>%</th></tr></thead><tbody>'+"".join(rows)+"</tbody></table></div>" if rows else '<p class="empty">No plane measurements were supplied.</p>'
    return '<section class="panel" id="measurements"><div class="panel-heading"><div><h2>Core plane measurements</h2><p>Signed cycle volume is preferred. Legacy unsigned values are identified in the selected-plane details. Incomplete cycles have no complete-cycle summary.</p></div></div>'+table+"</section>"


def _video_markup(model):
    rows = model.get("videos", [])
    result = ['<section class="panel" id="videos"><div class="panel-heading"><div><h2>Rotation and playback videos</h2><p>Each video belongs to this case and result version. Camera rotation and cardiac-phase playback are recorded separately. Missing or stale videos are never shown as current evidence.</p></div></div><div class="panel-body">']
    if not rows:
        result.append('<p>No report videos are available. In the GUI, choose Review &amp; Export &gt; Generate QC Videos to render the available computed families.</p>')
    for row in rows:
        state = _text(row.get('validity', 'missing'))
        result.append('<details><summary>'+_esc(row.get('title') or row.get('family'))+' · '+_esc(state)+'</summary>')
        url = _text(row.get('url'))
        # These paths are produced by the local asset publisher, never by a
        # check description or another free-text field.
        safe_url = bool(url) and not url.startswith(('/', '\\')) and ':' not in url.split('/')[0]
        if state == 'current' and safe_url:
            result.append('<video controls loop playsinline preload="metadata" style="width:100%;max-width:1100px"><source src="'+_esc(url)+'" type="'+_esc(row.get('mime', 'video/mp4'))+'">Video playback is unavailable. Open the exported video in a compatible player.</video>')
            result.append('<p><a href="'+_esc(url)+'">Open video file</a></p>')
        else:
            result.append('<p>Needs generation or refresh. A completed metric does not imply that its video was generated.</p>')
        if row.get('browser_error'):
            result.append('<p>Browser playback preparation failed: '+_esc(row['browser_error'])+'</p>')
        result.append('<p>Camera: '+_esc(row.get('camera_behavior', 'not recorded'))+' · Phases: '+_esc(row.get('phase_behavior', 'not recorded'))+'</p></details>')
    return ''.join(result)+'</div></section>'


def render_report_html(snapshot, *, sections=None, video_urls=None):
    """Render v1 snapshots without Node, a server, external assets or analysis."""
    snapshot = _safe(_dict(snapshot))
    sections = _DEFAULT_SECTIONS.copy() if sections is None else set(sections) & _DEFAULT_SECTIONS
    model = _build_model(snapshot, sections)
    model["videos"] = _safe(list(video_urls if video_urls is not None else snapshot.get("videos", [])))
    counts = model["counts"]
    headline = "Action required" if counts["errors"] else "Review required" if counts["warnings"] else "Results incomplete" if counts["unavailable"] or counts["stalePlanes"] or counts["missingPlanes"] or not model["checks"] else "No required findings"
    state = "error" if counts["errors"] else "warning" if counts["warnings"] else "unavailable" if headline == "Results incomplete" else "pass"
    body = ['<header class="topbar"><div><b>AutoFlow</b><span>Measurement &amp; QC report</span></div><button type="button" id="print-report">Print / Save PDF</button></header><main>', f'<div class="title-row"><div><p class="eyebrow">Flow MRI · immutable result snapshot</p><h1>{_esc(model["title"])}</h1><p class="source-line">Source: {_esc(model["source"])}</p></div><div class="run-meta">{_esc(model["createdAt"])}<br>Revision {_esc(model["revision"])} · {model["phaseCount"]} acquired phases<br>{counts["currentPlanes"]} / {counts["planes"]} plane results current</div></div>', f'<section class="headline {state}"><div><strong>{headline}</strong><p>{counts["errors"]} error groups · {counts["warnings"]} warning groups · {counts["unavailable"]} unavailable groups</p></div><div class="headline-side">Source automatic status: <b>{_esc(model["overallStatus"].replace("_", " "))}</b><small>{counts["rawRecords"]} underlying records; counts are not numbers of diseased vessels.</small></div></section>', '<p class="interpretation">'+_esc(model["interpretation"])+"</p>"]
    if counts["stalePlanes"] or counts["missingPlanes"]:
        body.append(f'<p class="notice">{counts["stalePlanes"]} stale planes: <b>Previous result / needs recalculation</b>. {counts["missingPlanes"]} missing, incomplete or unverified plane results. Unavailable values are not replaced by zero.</p>')
    body.append('<nav class="jump-links no-print"><a href="#review">Spatial review</a><a href="#required">Required findings</a>')
    if "videos" in sections:
        body.append('<a href="#videos">Videos</a>')
    if "metrics" in sections:
        body.append('<a href="#measurements">Measurements</a>')
    if "curves" in sections:
        body.append('<a href="#curves">Curves and phases</a>')
    if "methods" in sections:
        body.append('<a href="#methods">Methods and provenance</a>')
    body.append('</nav><section class="hero-grid" id="review"><div class="panel scene-panel"><div class="panel-heading"><div><h2>Segmentation, skeleton &amp; QC locations</h2><p>Rotate and zoom the actual source coordinates. Select a marker to inspect its checks and linked plane.</p></div></div>')
    if "images" in sections:
        body.append('<div class="scene-toolbar no-print"><div class="marker-filters" id="marker-filters">')
        for marker_state, label in (("pass", "Pass"), ("warning", "Warning"), ("error", "Error"), ("unavailable", "Unavailable")):
            body.append(f'<label><input type="checkbox" data-marker-filter="{marker_state}" checked><span class="dot {marker_state}"></span>{label} <span class="filter-count">{counts["markers"][marker_state]}</span></label>')
        body.append('</div><div class="scene-buttons"><button type="button" id="reset-view">Reset view</button><button type="button" id="zoom-out" aria-label="Zoom out">−</button><button type="button" id="zoom-in" aria-label="Zoom in">+</button></div></div><div class="canvas-wrap"><canvas id="spatial-canvas" tabindex="0" aria-label="Interactive source geometry. Drag to rotate, scroll to zoom. Use arrow keys to rotate. Select a marker or use the marker list."></canvas><div id="scene-empty" class="scene-empty" hidden>Spatial context unavailable. No coordinates or surface were supplied; this report does not invent geometry.</div><div id="scene-tooltip" class="scene-tooltip" hidden></div><span class="scene-instruction no-print">Drag to rotate · Shift-drag to pan · Wheel to zoom</span></div><img id="print-scene" class="print-only" alt="Current source geometry view"><div class="scene-footer"><span id="scene-description"></span><label class="no-print"><input type="checkbox" id="show-segmentation" checked> Segmentation context</label><label class="no-print"><input type="checkbox" id="show-skeleton" checked> Skeleton / paths</label></div>')
    else:
        body.append('<p class="empty">Spatial context and images were excluded by the report options. Required findings remain available below.</p>')
    body.append('<div class="selection-strip"><label class="field">Selected measurement plane<select id="plane-select" aria-label="Selected measurement plane"></select></label><span id="selected-plane-state" class="selected-state"></span></div>')
    if "metrics" in sections:
        body.append('<div class="readouts" id="core-readouts">'+"".join(f'<div><span>{label}</span><strong id="{key}">—</strong><small>{unit}</small></div>' for key, label, unit in (("value-q", "Flow at selected phase", "mL/s"), ("value-area", "Area at selected phase", "mm²"), ("value-peak", "Peak |normal velocity|", "cm/s"), ("value-volume", "Net flow volume", "mL/cycle")))+'</div><p id="selected-plane-note" class="selection-note"></p>')
    body.append('<div class="phase-controls"><button type="button" id="phase-back" aria-label="Previous acquired phase">‹</button><label for="phase-slider">Acquired phase</label><input type="range" id="phase-slider" min="0" max="'+str(max(0, model["phaseCount"]-1))+'" value="0" step="1"><button type="button" id="phase-forward" aria-label="Next acquired phase">›</button><output id="phase-label">0</output></div></div><aside class="review-side"><section class="panel"><div class="panel-heading"><div><h2>Vessels / paths to review</h2><p>Errors and warnings first. Green means explicit checks passed; gray means unavailable or unassessed.</p></div></div><div id="vessel-triage" class="triage-list"></div></section><section class="panel inspector-panel" id="inspector"><div class="panel-heading"><h2>Selected check</h2><span id="selection-badge"></span></div><div class="panel-body" id="marker-inspector"><p>Select a marker or a vessel / path row to inspect its source evidence.</p></div></section></aside></section>')
    body.append('<section class="panel global-panel"><div class="panel-heading"><div><h2>Global and unlocated findings</h2><p>These checks do not provide a usable point location. They are never attached to an arbitrary vessel.</p></div></div><div class="global-list">')
    unlocated = [g for g in model["groups"] if g["global"] or not g["located"]]
    for group in unlocated:
        body.append(f'<button class="global-card {group["status"]}" type="button" data-select-issue="{_esc(group["id"])}">{_badge(group["status"])}<strong>{_esc(group["title"])}</strong><span>{_esc(group["summary"])}</span><small>{"Global analysis / configuration" if group["global"] else "No surviving spatial anchor"}</small></button>')
    if not unlocated:
        body.append('<p class="empty">No required global or unlocated findings were supplied.</p>')
    body.append('</div></section>')
    if "images" in sections:
        body.append('<details class="panel" id="marker-directory"><summary>Accessible marker list · '+str(len(model["spatial"]["markers"]))+' source locations</summary><div class="panel-body" id="marker-list"></div></details>')
    if "curves" in sections:
        body.append('<section class="panel" id="curves"><div class="panel-heading"><div><h2>Recorded curves and acquired phases</h2><p>Every supplied sample is retained. Missing samples make gaps; a measured zero remains zero. Selecting a phase changes the readout, not the analysis.</p></div><label class="field no-print">Measurement<select id="curve-metric"><option value="q">Flow (mL/s)</option><option value="area">Area (mm²)</option><option value="meanVelocity">Mean normal velocity (cm/s)</option><option value="peakVelocity">Peak |normal velocity| (cm/s)</option></select></label></div><div class="panel-body"><div id="main-curve" class="curve-host"></div><p id="curve-note" class="small muted"></p></div></section><section class="panel print-only" id="all-print-curves"><h2>All recorded flow curves</h2><div class="print-curves" id="print-curves"></div></section>')
    if "videos" in sections:
        body.append(_video_markup(model))
    body.append(_required_markup(model))
    if "metrics" in sections:
        body.append(_measurement_markup(model))
        if model["pwv"]:
            body.append('<section class="panel"><div class="panel-heading"><h2>PWV analysis groups</h2></div><div class="table-wrap"><table><thead><tr><th>Requested group</th><th>Result</th><th>PWV (m/s)</th><th>R²</th><th>Acquired interval (ms)</th><th>Effective span (mm)</th><th>Message</th></tr></thead><tbody>')
            for result in model["pwv"]:
                current = result.get("status") == "ok" and result.get("validity") != "stale"
                body.append("<tr>"+"".join("<td>"+_esc(value)+"</td>" for value in (result.get("name"), "Current" if current else "Previous result / needs recalculation" if result.get("validity") == "stale" else "Not assessable", _fmt(result.get("pwv_m_s")) if current else "—", _fmt(result.get("fit_r2")) if current else "—", _fmt(result.get("original_sample_interval_ms")), _fmt(result.get("effective_span_mm")), result.get("message")))+"</tr>")
            body.append('</tbody></table></div><p class="panel-note">A requested analysis group is not a detected anatomy label. Interpolation does not increase acquired temporal resolution.</p></section>')
        body.append('<details class="panel" id="advanced-measurements"><summary>Advanced measurements</summary><div class="panel-body"><p class="muted">Source keys retain their recorded units and validity. Missing and stale fields are not replaced by zero.</p><div id="advanced-values"></div></div></details>')
    if "qc" in sections:
        body.append('<details class="panel" id="all-checks" data-print-open><summary>All source checks, including explicit passes</summary><div class="table-wrap"><table><thead><tr><th>Rule / source ID</th><th>Automatic status</th><th>Message</th><th>Scope</th></tr></thead><tbody>')
        for check in model["checks"]:
            body.append('<tr><td>'+_esc(check["title"])+'<small class="mono">'+_esc(check["id"])+"</small></td><td>"+_badge(check["status"], check["statusLabel"])+"</td><td>"+_esc(check["summary"])+"</td><td>"+_esc(check["scope"])+"</td></tr>")
        body.append('</tbody></table></div></details>')
    if "images" in sections and model["images"]:
        body.append('<section class="panel"><div class="panel-heading"><h2>Additional source image evidence</h2></div><div class="panel-body image-grid">')
        for image in model["images"]:
            caption = image["caption"]+(f" · source phase {image['phase']}" if image["phase"] is not None else "")+(" · plane "+_text(image["planeId"]) if image["planeId"] is not None else "")
            body.append(f'<figure><img loading="lazy" src="data:image/png;base64,{image["data"]}" alt="{_esc(caption)}"><figcaption>{_esc(caption)}</figcaption></figure>')
        body.append('</div></section>')
    if "methods" in sections:
        methods = {key: snapshot.get(key) for key in ("parameters", "parameters_sha256", "source_metadata", "segmentation_source", "segmentation_provenance", "spacing_mm", "origin_mm", "rr_ms", "artifact_signatures")}
        body.append('<section class="panel" id="methods"><div class="panel-heading"><h2>Methods and provenance</h2></div><div class="panel-body"><ul class="method-notes"><li>This report reads an immutable snapshot. It performs no inference, image sampling, correction or measurement recalculation.</li><li>Plane and acquired-phase indices retain source zero-based numbering. Flow sign and measurement units follow the exported fields.</li><li>Cycle volume uses signed net flow when supplied. Peak velocity is the maximum absolute normal component over the recorded ROI and cycle.</li><li>The geometry view uses actual local millimeter coordinates; world coordinates equal local coordinates plus the recorded origin. Surface or boundary simplification, if supplied, affects display only.</li><li>A static segmentation context is not a phase-resolved image. Acquired-phase selection changes measurements and phase-specific check emphasis only.</li><li>Label names and aliases come from the supplied label definitions. Their name_source is retained; configured aliases are not independent anatomical validation.</li><li>Human review is read-only and does not alter automatic checks. An earlier review does not approve a changed dependency signature.</li></ul><p class="mono">Snapshot: '+_esc(model["snapshotId"])+'<br>Dependency: '+_esc(model["signature"])+'</p><details><summary>Parameters, input provenance and label definitions</summary><pre>'+_esc(json.dumps({**methods, "label_definitions": model["labelDefinitions"], "spatial_notes": model["spatial"]["notes"]}, ensure_ascii=False, indent=2))+"</pre></details></div></section>")
    body.append('<details class="panel machine-snapshot"><summary>Machine-readable snapshot (base64 JSON)</summary><pre id="snapshot">'+_encoded(snapshot)+'</pre></details><footer>AutoFlow · self-contained English measurement and QC report · filters change the view only.</footer><noscript><p class="notice">JavaScript is disabled. Source measurements and required findings remain readable; interactive geometry and phase selection require JavaScript.</p></noscript></main>')
    return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+_esc(model["title"])+" · AutoFlow report</title><style>"+REPORT_CSS+'</style></head><body>'+"".join(body)+'<script id="report-data" type="application/octet-stream">'+_encoded(model)+'</script><script>'+REPORT_JS+"</script></body></html>"
