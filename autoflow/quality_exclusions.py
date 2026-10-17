"""Readable QC provenance for labels and plane candidates omitted by processing."""


def _percent(value):
    return "N/A" if value is None else f"{float(value):.1%}"


def _detail(title, summary, *, status="pass", stage="", value=None, details=()):
    return {"title": str(title), "summary": str(summary), "status": status,
            "stage": stage, "value": value, "details": list(details)}


def _label_summary(label):
    return {key: value for key, value in label.items() if key != "phases"}


def build_exclusion_checks(workspace, make_check):
    preprocessing = dict(getattr(workspace, "segmentation_preprocess_qc", {}) or {})
    plane_qc = dict(getattr(getattr(workspace, "derived", None), "plane_qc", {}) or {})
    layout = dict(plane_qc.get("plane_layout", plane_qc if "paths" in plane_qc else {}) or {})
    payload = {"segmentation_preprocess": preprocessing, "plane_layout": layout}
    checks = []
    labels = list(preprocessing.get("labels", []))
    if preprocessing:
        voting_details, component_details, cleanup_details, morphology_details, temporal_details = [], [], [], [], []
        for label in labels:
            title = f"{label['label_name']} ({label['label']})"
            stage = ", ".join(label.get("group_names", [])) or "Segmentation"
            phases, cleanup_phases, temporal_phases = [], [], []
            for phase in label.get("phases", []):
                lost = phase["vote_removed_voxels"]
                count = phase["vote_input_voxels"]
                destinations = ", ".join(f"{item['label_name']}={item['voxel_count']}" for item in phase.get("vote_destinations", []))
                summary = (f"input={count}; overlap={phase['vote_overlap_voxels']}; "
                           f"lost={lost}/{count} ({_percent(phase['vote_removed_fraction'])}); "
                           f"added={phase['vote_added_voxels']}")
                if destinations:
                    summary += f"; lost to {destinations}"
                phases.append(_detail(f"Phase {phase['phase_index']}", summary,
                                      status="warn" if count and lost == count else "pass", stage=stage, value=phase))
                cleanup_phases.append(_detail(f"Phase {phase['phase_index']}",
                    f"before cleanup={phase['input_voxels']}; removed={phase['cleanup_removed_voxels']} "
                    f"({_percent(phase['cleanup_removed_fraction'])}); vote input={count}",
                    status="warn" if phase["input_voxels"] and not count else "pass", stage=stage, value=phase))
                if phase.get("temporal_group_removed_voxels") is not None:
                    removed = int(phase["temporal_group_removed_voxels"])
                    temporal_phases.append(_detail(f"Phase {phase['phase_index']}",
                        f"phase input={count}; removed={removed}/{count} ({_percent(phase['temporal_group_removed_fraction'])}); "
                        f"retained={phase['temporal_group_retained_voxels']}",
                        status="warn" if count and removed == count else "pass", stage=stage,
                        value={key: phase[key] for key in ("phase_index", "vote_input_voxels", "temporal_group_removed_voxels",
                                                         "temporal_group_removed_fraction", "temporal_group_retained_voxels")}))
            vote_status = "warn" if any(phase["status"] == "warn" for phase in phases) else "pass"
            voting_details.append(_detail(title,
                f"union={label['vote_input_union_voxels']}; voted={label['voted_voxels']}; "
                f"union loss={label['vote_removed_union_voxels']} ({_percent(label['vote_removed_union_fraction'])}); "
                f"status={label['status']}",
                status=vote_status, stage=stage, value=_label_summary(label), details=phases))
            if label["voted_voxels"]:
                component_details.append(_detail(title,
                    f"voted={label['voted_voxels']}; removed={label['component_removed_voxels']} "
                    f"({_percent(label['component_removed_fraction'])}); retained={label['retained_group_voxels']}",
                    status="warn" if not label["retained_group_voxels"] else "pass", stage=stage, value=_label_summary(label)))
            if any(phase["cleanup_removed_voxels"] for phase in label.get("phases", [])):
                cleanup_details.append(_detail(title, "Per-phase cleanup losses before voting",
                    status="warn" if not label["vote_input_union_voxels"] else "pass", stage=stage,
                    value=_label_summary(label), details=cleanup_phases))
            if label["morphology_removed_voxels"]:
                morphology_details.append(_detail(title,
                    f"group-retained={label['retained_group_voxels']}; removed={label['morphology_removed_voxels']} "
                    f"({_percent(label['morphology_removed_fraction'])}); skeleton input={label['skeleton_input_voxels']}",
                    status="warn" if not label["skeleton_input_voxels"] else "pass", stage=stage, value=_label_summary(label)))
            if any(phase.get("temporal_group_removed_voxels") for phase in label.get("phases", [])):
                reason = "label unavailable after voting" if not label["voted_voxels"] else "per-phase merged-group component filtering"
                temporal_details.append(_detail(title, reason,
                    status="warn" if any(item["status"] == "warn" for item in temporal_phases) else "pass",
                    stage=stage, value=_label_summary(label), details=temporal_phases))
        fully_lost = [item["label_name"] for item in labels if item["status"] == "removed_by_majority_vote"]
        partial = sum(bool(item["voted_voxels"] and item["vote_removed_union_voxels"]) for item in labels)
        check = make_check("segmentation.majority_vote", "Segmentation",
            "warn" if any(item["status"] == "warn" for item in voting_details) else "pass",
            "Temporal label voting",
            f"labels={len(labels)}; fully lost={', '.join(fully_lost) or 'none'}; partial union loss={partial}",
            value={"provenance_path": "exclusions.segmentation_preprocess", "label_count": len(labels),
                   "fully_lost_labels": fully_lost, "partial_union_loss_label_count": partial},
            threshold="phase loss / phase label voxels; union loss / union label voxels",
            action="Expand labels and phases to review lost support and winning labels.")
        check["details"] = voting_details
        checks.append(check)
        for check_id, title, details in (
            ("segmentation.group_components", "Merged-group component filtering", component_details),
            ("segmentation.temporal_group_components", "Per-phase group-mask exclusions", temporal_details),
            ("segmentation.cleanup_4d", "4D component cleanup before voting", cleanup_details),
            ("segmentation.skeleton_preprocessing", "Skeleton-mask preprocessing", morphology_details),
        ):
            if details:
                check = make_check(check_id, "Segmentation",
                    "warn" if any(item["status"] == "warn" for item in details) else "pass", title,
                    f"labels reported={len(details)}; labels requiring review={sum(item['status'] == 'warn' for item in details)}",
                    value={"provenance_path": "exclusions.segmentation_preprocess", "labels_reported": len(details)},
                    action="Review label counts separately from temporal voting losses.")
                check["details"] = details
                checks.append(check)
    if layout:
        details, records = [], list(layout.get("planes", []))
        labels_by_id = {int(item["segmentation_label"]): item for item in layout.get("labels", [])}
        for item in records:
            value = int(item.get("segmentation_label", 0))
            if value not in labels_by_id:
                labels_by_id[value] = {"label_name": f"label_{value}" if value else "Unlabelled planes",
                                       "segmentation_label": value, "status": "generated"}
        for value, label in labels_by_id.items():
            plane_details = []
            for item in records:
                if int(item.get("segmentation_label", 0)) != value:
                    continue
                valid = bool(item.get("valid", True))
                fov = dict(item.get("fov_boundary", {}) or {})
                summary = "Kept" if valid else f"Excluded: {item.get('reason', 'unknown')}"
                if fov.get("contact_fraction") is not None:
                    faces = ",".join(face["face"] for face in fov.get("faces", [])) or "none"
                    summary += (f"; FOV={fov['contact_length_mm']:.3f}/{fov['perimeter_mm']:.3f} mm "
                                f"({_percent(fov['contact_fraction'])}); threshold={_percent(fov['max_fraction'])}; faces={faces}")
                elif fov.get("enabled"):
                    summary += f"; FOV check={fov.get('status', 'unavailable')}"
                status = "warn" if not valid or (fov.get("enabled") and fov.get("status") in {"contour_unavailable", "no_roi"}) else "pass"
                plane_details.append(_detail(f"Candidate {item.get('plane_index_source', item.get('plane_index'))}",
                    summary, status=status, stage=f"{item.get('group_name', '')} / path {item.get('path_index', -1)}", value=item))
            for item in label.get("omitted_positions", []):
                title = f"Anchor {item['anchor']}" if item.get("anchor") else f"Position {_percent(item.get('fraction'))}"
                plane_details.append(_detail(title,
                    f"Excluded: {item.get('reason', 'unknown')}; center={item.get('center_local_mm')}",
                    status="warn", stage=f"path {item.get('path_index', -1)}", value=item))
            label_status = str(label.get("status", ""))
            reason = str(label.get("reason", ""))
            for main in label.get("selected_main_paths", []):
                if label.get("main_path") == "landmark":
                    placement = main.get("placement", {})
                    plane_details.append(_detail(f"Placement {main['main_path_index']}",
                        f"anchor={placement.get('anchor', main.get('selection_reason'))}; "
                        f"interval={placement.get('label_interval_mm')}; shift={placement.get('search_shift_mm')} mm; "
                        f"planes={main['actual_plane_count']}/{main['requested_plane_count']}",
                        stage=f"path {main['path_index']}", value=main))
                    continue
                confidence = main.get("flow_confidence")
                confidence_text = "unavailable" if confidence is None else f"{confidence:.3f}"
                plane_details.append(_detail(f"Main path {main['main_path_index']}",
                    f"{label.get('main_path')}; {main.get('selection_reason')}; "
                    f"segment={main['segment_index']}; length={main['length_mm']:.2f} mm; "
                    f"direction={main.get('flow_direction_status')}; confidence={confidence_text}; "
                    f"planes={main['actual_plane_count']}/{main['requested_plane_count']}",
                    status="warn" if main.get("selection_reason") == "geometry_fallback_longest" else "pass",
                    stage=f"path {main['path_index']}", value=main))
            for candidate in label.get("topology_candidates", []):
                if candidate.get("selected_main_path"):
                    continue
                plane_details.append(_detail(f"Unselected segment {candidate['segment_index']}",
                    f"{candidate.get('selection_reason')}; length={candidate['length_mm']:.2f} mm; "
                    f"direction={candidate.get('flow_direction_status')}; "
                    f"upstream={candidate.get('upstream_segments', [])}; "
                    f"downstream={candidate.get('downstream_segments', [])}",
                    stage=f"path {candidate['path_index']}", value=candidate))
            warning = (label_status in {"needs_selection", "no_label_path", "incomplete"}
                       or bool(label.get("selection_warnings"))
                       or reason.startswith("removed_by_") or any(item["status"] == "warn" for item in plane_details))
            summary = f"{label_status}; reason={reason or 'none'}"
            if "requested_plane_count" in label:
                summary += f"; requested={label['requested_plane_count']}; retained={label['actual_plane_count']}"
            if label.get("excluded_plane_counts"):
                summary += f"; exclusions={label['excluded_plane_counts']}"
            if "main_path_count" in label:
                summary += (f"; selected placement intervals={label['main_path_count']}" if label.get("main_path") == "landmark"
                            else f"; {label.get('main_path')} main paths={label['main_path_count']}")
            if label.get("selection_warnings"):
                summary += f"; warnings={','.join(label['selection_warnings'])}"
            details.append(_detail(f"{label['label_name']} ({value})", summary,
                                   status="warn" if warning else "pass", value=label, details=plane_details))
        excluded = sum(not bool(item.get("valid", True)) for item in records)
        excluded += sum(len(item.get("omitted_positions", [])) for item in layout.get("labels", []))
        check = make_check("planes.exclusions", "Centerline & Planes",
            "warn" if any(item["status"] == "warn" for item in details) else "pass", "Plane selection and exclusions",
            f"retained={layout.get('actual_plane_count', len(getattr(workspace, 'planes', [])))}; "
            f"planned={layout.get('requested_plane_count', 0)}; excluded candidates={excluded}",
            value={"provenance_path": "exclusions.plane_layout", "excluded_candidate_count": excluded,
                   "fov_boundary_max_fraction": layout.get("fov_boundary_max_fraction")},
            threshold="FOV contact length / outer ROI perimeter >= configured fraction",
            action="Expand labels and candidates for source ids, geometry, exclusion reasons, and FOV measurements.")
        check["details"] = details
        checks.append(check)
    return checks, payload
