"""Label-preserving provenance for temporal voting and skeleton preprocessing."""

import numpy as np


def _fraction(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def build_segmentation_preprocess_qc(raw, vote_input, voted, retained, processed,
                                    *, resolution, label_map=None, label_groups=None,
                                    group_specs=(), processed_groups=None, temporal_group_mask=None,
                                    morphology_groups=None):
    """Compare spatial support, keeping phase and union denominators explicit.

    Voting is the existing per-voxel mode, including background and breaking
    ties by the smallest numeric label. Source phase labels remain unchanged;
    downstream temporal masks additionally depend on group availability and
    per-phase component filtering.
    """
    raw = np.asarray(raw)
    source = np.asarray(vote_input)
    if raw.ndim == 3:
        raw = raw[..., None]
    if source.ndim == 3:
        source = source[..., None]
    voted, retained = np.asarray(voted), np.asarray(retained)
    processed = np.asarray(processed, dtype=bool)
    names = {int(value): str(name) for name, value in dict(label_map or {}).items()}
    groups = {}
    for name, config in dict(label_groups or {}).items():
        for value in config.get("labels", []):
            groups.setdefault(int(value), []).append(str(name))
    for spec in group_specs:
        for value in spec["labels"]:
            groups[int(value)] = [str(spec["name"])]

    def destinations(winners, denominator):
        values, counts = np.unique(winners, return_counts=True)
        return [{"label": int(value), "label_name": names.get(int(value), "background" if int(value) == 0 else f"label_{value}"),
                 "voxel_count": int(count), "fraction": _fraction(int(count), denominator)}
                for value, count in zip(values, counts)]

    records = []
    values = sorted(set(int(value) for value in np.unique(raw) if int(value) > 0)
                    | set(int(value) for value in np.unique(source) if int(value) > 0))
    for value in values:
        raw_presence, presence = raw == value, source == value
        raw_counts = np.count_nonzero(raw_presence, axis=(0, 1, 2))
        counts = np.count_nonzero(presence, axis=(0, 1, 2))
        union = np.any(presence, axis=3)
        raw_union_count = int(np.count_nonzero(np.any(raw_presence, axis=3)))
        union_count = int(np.count_nonzero(union))
        winner, kept = voted == value, retained == value
        processed_for_label = processed
        if processed_groups is not None and groups.get(value):
            processed_for_label = np.zeros(voted.shape, dtype=bool)
            for group in groups[value]:
                if group in processed_groups:
                    processed_for_label |= np.asarray(processed_groups[group], dtype=bool)
        voted_count, kept_count = int(np.count_nonzero(winner)), int(np.count_nonzero(kept))
        union_lost = union & ~winner
        lost_count = int(np.count_nonzero(union_lost))
        component_removed = int(np.count_nonzero(winner & ~kept))
        morphology_removed = int(np.count_nonzero(kept & ~processed_for_label))
        morphology_for_label = None
        if morphology_groups is not None:
            morphology_for_label = np.zeros(voted.shape, dtype=bool)
            for group in groups.get(value, []):
                if group in morphology_groups:
                    morphology_for_label |= np.asarray(morphology_groups[group], dtype=bool)
            component_removed = int(np.count_nonzero(winner & morphology_for_label & ~processed_for_label))
            morphology_removed = int(np.count_nonzero(winner & ~morphology_for_label))
        component_input = voted_count if morphology_for_label is None else int(np.count_nonzero(winner & morphology_for_label))
        morphology_input = kept_count if morphology_for_label is None else voted_count
        phases = []
        for phase_index in range(source.shape[3]):
            current = presence[..., phase_index]
            original_count, count = int(raw_counts[phase_index]), int(counts[phase_index])
            overlap = int(np.count_nonzero(current & winner))
            removed = count - overlap
            cleanup_removed = original_count - int(np.count_nonzero(raw_presence[..., phase_index] & current))
            loss = current & ~winner
            background = int(np.count_nonzero(loss & (voted == 0)))
            phase_component_removed = int(np.count_nonzero(current & winner & ~kept))
            phase_component_input = overlap
            if morphology_for_label is not None:
                phase_component_input = int(np.count_nonzero(current & winner & morphology_for_label))
                phase_component_removed = int(np.count_nonzero(current & winner & morphology_for_label & ~processed_for_label))
            temporal_kept = None if temporal_group_mask is None else int(np.count_nonzero(
                current & np.asarray(temporal_group_mask)[..., min(phase_index, temporal_group_mask.shape[3] - 1)]
            ))
            phases.append({
                "phase_index": int(phase_index),
                "input_voxels": original_count, "vote_input_voxels": count,
                "cleanup_removed_voxels": cleanup_removed,
                "cleanup_removed_fraction": _fraction(cleanup_removed, original_count),
                "vote_overlap_voxels": overlap, "vote_removed_voxels": removed,
                "vote_removed_fraction": _fraction(removed, count),
                "vote_added_voxels": voted_count - overlap,
                "vote_removed_to_background_voxels": background,
                "vote_removed_to_other_labels_voxels": removed - background,
                "vote_destinations": destinations(voted[loss], count),
                "component_removed_voxels": phase_component_removed,
                "component_removed_fraction": _fraction(phase_component_removed, phase_component_input),
                "retained_overlap_voxels": int(np.count_nonzero(current & kept)),
                "temporal_group_retained_voxels": temporal_kept,
                "temporal_group_removed_voxels": None if temporal_kept is None else count - temporal_kept,
                "temporal_group_removed_fraction": None if temporal_kept is None else _fraction(count - temporal_kept, count),
            })
        if not union_count:
            status = "removed_by_4d_cleanup"
        elif not voted_count:
            status = "removed_by_majority_vote"
        elif morphology_for_label is not None and not component_input:
            status = "removed_by_skeleton_preprocessing"
        elif not kept_count:
            status = "removed_by_group_component_filter"
        elif not np.any(kept & processed_for_label):
            status = "removed_by_skeleton_preprocessing"
        elif lost_count or component_removed or morphology_removed or np.any(raw_counts != counts):
            status = "partial_loss"
        else:
            status = "retained"
        records.append({
            "label": value, "label_name": names.get(value, f"label_{value}"),
            "group_names": groups.get(value, []), "status": status,
            "input_union_voxels": raw_union_count, "vote_input_union_voxels": union_count,
            "voted_voxels": voted_count, "retained_group_voxels": kept_count,
            "skeleton_input_voxels": int(np.count_nonzero(kept & processed_for_label)),
            "vote_removed_union_voxels": lost_count,
            "vote_removed_union_fraction": _fraction(lost_count, union_count),
            "vote_destinations": destinations(voted[union_lost], union_count),
            "component_removed_voxels": component_removed,
            "component_removed_fraction": _fraction(component_removed, component_input),
            "morphology_removed_voxels": morphology_removed,
            "morphology_removed_fraction": _fraction(morphology_removed, morphology_input),
            "phases": phases,
        })
    return {
        "schema": "autoflow.segmentation_preprocess_qc.v1",
        "spatial_shape": list(voted.shape), "phase_count": int(source.shape[3]),
        "voxel_volume_mm3": float(np.prod(np.asarray(resolution, dtype=float))),
        "voting": {"method": "per_voxel_most_frequent_label", "background_included": True,
                   "tie_break": "lowest_numeric_label", "affects": "3d_topology_and_group_label_availability",
                   "source_4d_labels_unchanged": True},
        "denominators": {
            "vote_removed_union_fraction": "spatial union of this label in the voting input",
            "phase_vote_removed_fraction": "this label in this phase of the voting input",
            "phase_cleanup_removed_fraction": "this label in this phase before 4D cleanup",
            "component_removed_fraction": "this label after morphology" if morphology_groups is not None else "this label in the voted 3D volume",
            "phase_component_removed_fraction": "this phase's label voxels that survived voting and morphology" if morphology_groups is not None else "this phase's label voxels that survived voting",
            "morphology_removed_fraction": "this label in the voted 3D volume" if morphology_groups is not None else "this label after merged-group component filtering",
            "phase_temporal_group_removed_fraction": "this phase's label in the voting input, compared with the downstream temporal group mask",
        },
        "labels": records,
    }
