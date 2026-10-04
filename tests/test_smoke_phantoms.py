import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from autoflow import AutoFlowConfig, build_workspace, run_batch, run_case
from autoflow.algorithms.data import discover_h5_input_cases, inspect_h5_input_case, load_h5_data
from autoflow.algorithms.inputs import collect_input_cases
from autoflow.algorithms.segmentation import default_nnunet_model_folder
from autoflow.algorithms.pwv import compute_cross_correlation_delay_ms, detect_waveform_foot_time_ms
from autoflow.algorithms.preprocess import filter_connected_components, separate_longitudinal_label_contacts
from autoflow.algorithms.skeleton import generate_three_pass_special_skeleton
from autoflow.algorithms.intracranial import detect_willis_ring
from autoflow.algorithms.graph import remove_short_terminal_branches
from autoflow.algorithms.metrics import (
    compute_plane_metrics,
    compute_plane_metrics_multithread,
    compute_vortex_metrics,
    compute_wss_metrics,
    cal_wss_from_surf,
    calculate_gradient,
    filter_planes_by_branch_support,
)
from autoflow.algorithms.planes import filter_paths_by_segmentation, generate_planes_from_paths
from autoflow.algorithms.segmentation import generate_nnunet_auto_segmentation, save_segmentation_to_source_h5
from autoflow.algorithms.streamlines import (
    _plane_seeds,
    automatic_streamline_clim,
    create_pathline_temporal_source,
    generate_pathlines_from_plane_at_t,
    generate_streamlines_at_t,
    pathline_prefix_at_phase,
)
from autoflow.config import bundle_to_autoflow_kwargs
from autoflow.core.pipeline import PipelineEngine
from autoflow.core.models import GraphData, PlaneData, SkeletonParams, StepId, Workspace
from autoflow.case_types import LoaderCapabilities
from autoflow.plane_io import (
    PLANE_POSITION_SCHEMA,
    build_plane_records,
    load_plane_position_payload,
    project_planes_to_workspace,
    save_plane_positions,
)
from autoflow.plane_io import save_pwv_h5
from autoflow.quality import QUALITY_REPORT_SCHEMA, build_quality_report, save_quality_report
from autoflow.rendering.videos import _build_union_surface, _path_color, _path_group_name, _write_video, render_plane_rotation_video
from autoflow.ui.contour_edit import new_contour_from_stroke, polygon_area, replace_contour_segment


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
PHANTOM_CASES = ("phantom_S", "phantom_U", "phantom_Y")
REL_TOL = 0.05
PLANE_SPACING_MM = 15.0


def test_cvi_style_contour_edit_creates_and_replaces_one_local_arc():
    theta = np.linspace(0.0, 2.0 * np.pi, 160, endpoint=False)
    original = np.column_stack((10.0 * np.cos(theta), 8.0 * np.sin(theta)))
    created, message = new_contour_from_stroke(np.vstack((original, original[0])), 0.35)
    assert created is not None, message
    assert abs(polygon_area(created)) > 150.0

    outward_stroke = np.asarray([
        [8.0, -4.0], [13.0, -2.0], [14.0, 0.0], [13.0, 2.0], [8.0, 4.0]
    ])
    edited, message = replace_contour_segment(original, outward_stroke, 0.35, 1.5)
    assert edited is not None, message
    assert abs(polygon_area(edited)) > abs(polygon_area(original))

    inward_stroke = np.asarray([
        [8.0, -4.0], [5.0, -2.0], [4.0, 0.0], [5.0, 2.0], [8.0, 4.0]
    ])
    edited, message = replace_contour_segment(original, inward_stroke, 0.35, 1.5)
    assert edited is not None, message
    assert abs(polygon_area(edited)) < abs(polygon_area(original))

    # The final point may stop just inside/outside the vessel. It is projected
    # to the nearest unambiguous boundary point so the user does not need to
    # land exactly on the original contour.
    open_stroke = np.asarray([
        [8.0, -4.0], [13.0, -2.0], [14.0, 0.0], [13.0, 2.0], [7.0, 4.0]
    ])
    edited, message = replace_contour_segment(original, open_stroke, 0.35, 1.5)
    assert edited is not None, message
    assert abs(polygon_area(edited)) > abs(polygon_area(original))


def test_willis_ring_detection_requires_a_mixed_arterial_cycle():
    size = 41
    labels = np.zeros((size, size, 3), dtype=np.int16)
    yy, xx = np.indices((size, size))
    radius = np.sqrt((xx - 20) ** 2 + (yy - 20) ** 2)
    ring = (radius > 8.0) & (radius < 11.0)
    labels[:, :, 1][ring & (xx < 20)] = 17  # LICA/anterior side
    labels[:, :, 1][ring & (xx >= 20)] = 26  # PCA/posterior side
    detected = detect_willis_ring(labels, spacing=(1.0, 1.0, 1.0),
                                  anterior_label_values=[17],
                                  posterior_label_values=[26])
    assert detected["status"] == "detected"
    assert detected["cycle_rank"] >= 1
    assert set(detected["member_labels"]) == {17, 26}

    disconnected = np.zeros((24, 24, 3), dtype=np.int16)
    disconnected[3:19, 5, 1] = 17
    disconnected[3:19, 17, 1] = 26
    absent = detect_willis_ring(disconnected, spacing=(1.0, 1.0, 1.0),
                                anterior_label_values=[17],
                                posterior_label_values=[26])
    assert absent["status"] == "not_detected"


def test_three_pass_special_skeleton_merges_configured_branch_passes():
    labels = np.zeros((15, 15, 15), dtype=np.int16)
    labels[7, 2:12, 7] = 1  # A: shared trunk
    labels[7, 11:14, 7] = 2  # B
    labels[7, 8, 8:12] = 3  # C
    labels[8:12, 8, 7] = 4  # D
    params = SkeletonParams(
        remove_small_cc=False,
        do_closing=False,
        do_opening=False,
        gaussian_enabled=False,
        special_merge_radius_mm=0.5,
    )

    points, skeleton_mask = generate_three_pass_special_skeleton(
        labels,
        group_label_values=[1, 2, 3, 4],
        special_label_values=[2, 3, 4],
        params=params,
        resolution=np.ones(3, dtype=float),
    )

    assert points.ndim == 2 and points.shape[1] == 3
    assert len(points) > 0
    assert skeleton_mask.shape == labels.shape
    assert bool(np.any(skeleton_mask))
    assert SkeletonParams.from_dict({}).special_handling == "three_pass_merge"
    assert SkeletonParams.from_dict({"special_handling": "contact_surface"}).special_handling == "contact_surface"


def test_short_terminal_graph_branches_use_minimum_edge_points():
    points = np.arange(7 * 3, dtype=float).reshape(7, 3)
    graph = GraphData(
        points=points,
        edges=np.asarray([[0, 1], [1, 2], [2, 3], [0, 4], [0, 5], [5, 6]], dtype=int),
    )

    filtered = remove_short_terminal_branches(graph, min_edge_points=3)
    assert {tuple(edge) for edge in filtered.edges.tolist()} == {(0, 1), (1, 2), (2, 3)}
    assert SkeletonParams.from_dict({}).min_edge_points == 3
    assert SkeletonParams.from_dict({"min_edge_points": 0}).min_edge_points == 0
    assert SkeletonParams.from_dict({"min_edge_count": 4}).min_edge_points == 4


def test_separate_longitudinal_label_contacts_preserves_end_to_end_transition():
    from scipy.ndimage import label as ndi_label

    labels = np.zeros((40, 20, 20), dtype=np.int16)
    labels[2:20, 5:8, 5:8] = 1
    labels[20:38, 5:8, 5:8] = 2
    sequential = separate_longitudinal_label_contacts(labels > 0, labels, [1, 2], [1.0, 1.0, 1.0])
    assert np.array_equal(sequential, labels > 0)
    forced = separate_longitudinal_label_contacts(
        labels > 0, labels, [1, 2], [1.0, 1.0, 1.0], force_pairs=[(1, 2)]
    )
    assert ndi_label(forced, structure=np.ones((3, 3, 3), dtype=bool))[1] == 2

    labels[:, :, :] = 0
    labels[2:38, 5:8, 5:8] = 1
    labels[2:38, 8:11, 5:8] = 2
    side_by_side = separate_longitudinal_label_contacts(labels > 0, labels, [1, 2], [1.0, 1.0, 1.0])
    assert np.count_nonzero(side_by_side) < np.count_nonzero(labels)


def test_topology_fork_detection_is_independent_of_path_orientation():
    from autoflow.algorithms.branch import find_path_forks

    points = np.asarray([
        [2.0, 2.0, 2.0],
        [1.0, 2.0, 2.0],
        [3.0, 2.0, 2.0],
        [2.0, 3.0, 2.0],
    ])
    graph = GraphData(
        points=points,
        edges=np.asarray([[0, 1], [0, 2], [0, 3]], dtype=int),
    )
    # All three paths happen to start at the branch node.  Endpoint matching
    # alone would miss this junction; graph degree must still expose it.
    forks = find_path_forks([[0, 1], [0, 2], [0, 3]], points, graph=graph)
    assert len(forks) == 1
    assert forks[0]["node"] == 0
    assert forks[0]["degree"] == 3
    assert forks[0]["left"] == []
    assert forks[0]["right"] == [0, 1, 2]


def test_flow_orientation_uses_segmentation_filtered_path_geometry():
    from autoflow.algorithms.branch import _orient_node_paths_by_flow

    flow = np.zeros((11, 1, 1, 1, 3), dtype=np.float32)
    flow[..., 0] = 1.0
    mask = np.ones((11, 1, 1, 1), dtype=bool)
    graph_points = np.asarray([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    raw_paths = [[0, 1]]
    filtered_paths = [np.asarray([[10.0, 0.0, 0.0], [0.0, 0.0, 0.0]])]

    oriented = _orient_node_paths_by_flow(
        raw_paths,
        graph_points,
        flow_xyzt3=flow,
        segmask_binary_4d=mask,
        sample_paths=filtered_paths,
    )
    assert oriented == [[1, 0]]


def test_fork_internal_consistency_is_undefined_for_one_sided_topology_fork():
    from autoflow.algorithms.metrics import apply_internal_consistency_to_metrics

    metrics, qc = apply_internal_consistency_to_metrics(
        [
            {"path_index": 0, "netflow_mL_beat": 2.0},
            {"path_index": 1, "netflow_mL_beat": 1.0},
        ],
        path_info=[{}, {}],
        forks=[{"node": 10, "left": [], "right": [0, 1]}],
    )
    assert qc["fork_ic"] == {"0": None}
    assert qc["forks"][0]["ic"] is None
    assert metrics[0]["fork_ic"] == [{"fork_id": 0, "role": "outgoing", "ic": None}]


def test_plane_branch_support_drops_only_unmatched_planes():
    mask = np.ones((5, 5, 5, 1), dtype=bool)
    branch_labels = np.ones((5, 5, 5), dtype=np.int16)
    segmentation_labels = np.ones((5, 5, 5), dtype=np.int16)
    planes = [
        PlaneData(
            center=np.asarray([2.0, 2.0, 2.0]),
            normal=np.asarray([1.0, 0.0, 0.0]),
            label=1,
            path_index=0,
            segmentation_label=1,
        ),
        PlaneData(
            center=np.asarray([2.0, 2.0, 2.0]),
            normal=np.asarray([1.0, 0.0, 0.0]),
            label=2,
            path_index=1,
            segmentation_label=1,
        ),
    ]
    kept, qc = filter_planes_by_branch_support(
        planes,
        mask,
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        branch_labels_3d=branch_labels,
        segmentation_labels_3d=segmentation_labels,
    )
    assert kept == [planes[0]]
    assert qc[0]["valid"] is True
    assert qc[1]["valid"] is False
    assert qc[1]["reason"] == "no_branch_support"


def test_internal_consistency_marks_missing_path_metrics_undefined():
    from autoflow.algorithms.metrics import apply_internal_consistency_to_metrics

    metrics, qc = apply_internal_consistency_to_metrics(
        [{"path_index": 0, "netflow_mL_beat": 2.0}],
        path_info=[{}, {}],
        forks=[{"node": 10, "left": [0], "right": [1]}],
    )
    assert qc["path_ic"]["0"] is None
    assert qc["path_ic"]["1"] is None
    assert qc["fork_ic"] == {"0": None}
    assert qc["forks"][0]["status"] == "missing_path_metrics"
    assert metrics[0]["path_ic"] is None


@pytest.mark.parametrize("sparse_mask", [False, True])
def test_vortex_kinematics_matches_rigid_rotation_and_rejects_pure_shear(sparse_mask):
    shape = (11, 11, 11, 2)
    mask = np.ones(shape, dtype=bool)
    if sparse_mask:
        mask[:] = False
        mask[2:9, 2:9, 2:9, :] = True
    x, y, _z = np.meshgrid(
        np.arange(shape[0], dtype=np.float32),
        np.arange(shape[1], dtype=np.float32),
        np.arange(shape[2], dtype=np.float32),
        indexing="ij",
    )
    omega = 20.0
    rotation = np.zeros(shape + (3,), dtype=np.float32)
    rotation[..., 0] = (-omega * y[..., None] * 1e-3) * 100.0
    rotation[..., 1] = (omega * x[..., None] * 1e-3) * 100.0

    result = compute_vortex_metrics(mask, rotation, (1.0, 1.0, 1.0), support_erosion_iters=1)
    center = (5, 5, 5, 0)
    assert result["vortex_support_mask"][center] == 1
    assert result["vorticity_array"][center] == pytest.approx([0.0, 0.0, 2.0 * omega])
    assert float(result["q_criterion_array"][center]) == pytest.approx(omega ** 2)
    assert float(result["swirling_strength_array"][center]) == pytest.approx(omega)
    assert not np.any(result["vortex_support_mask"][0])
    outside_support = result["vortex_support_mask"] == 0
    for name in ("vorticity_array", "vorticity_magnitude", "q_criterion_array", "swirling_strength_array"):
        assert not np.any(result[name][outside_support])

    shear_rate = 30.0
    shear = np.zeros_like(rotation)
    shear[..., 0] = (shear_rate * y[..., None] * 1e-3) * 100.0
    shear_result = compute_vortex_metrics(mask, shear, (1.0, 1.0, 1.0), support_erosion_iters=1)
    assert float(shear_result["vorticity_magnitude"][center]) == pytest.approx(shear_rate)
    assert float(shear_result["q_criterion_array"][center]) == pytest.approx(0.0, abs=1e-5)
    assert float(shear_result["swirling_strength_array"][center]) == pytest.approx(0.0, abs=1e-5)

    no_erosion = compute_vortex_metrics(mask, rotation, (1.0, 1.0, 1.0), support_erosion_iters=0)
    assert not np.any(no_erosion["vortex_support_mask"][0])
    assert not np.any(no_erosion["vortex_support_mask"][:, 0])


def test_nonzero_origin_preserves_plane_metrics_and_plane_seeds():
    mask = np.zeros((12, 12, 12, 4), dtype=bool)
    mask[2:10, 2:10, 2:10, :] = True
    flow = np.zeros(mask.shape + (3,), dtype=np.float32)
    flow[..., 0] = 10.0
    plane = PlaneData(
        center=np.array([6.0, 6.0, 6.0], dtype=float),
        normal=np.array([1.0, 0.0, 0.0], dtype=float),
    )

    at_zero = compute_plane_metrics(flow, mask, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0), [plane])[0]
    shifted = compute_plane_metrics(flow, mask, (1.0, 1.0, 1.0), (100.0, 200.0, 300.0), [plane])[0]
    assert shifted["area_mm2"] == pytest.approx(at_zero["area_mm2"])
    assert shifted["flowrate_mL_s"] == pytest.approx(at_zero["flowrate_mL_s"])
    assert all(value > 0.0 for value in shifted["area_mm2"])

    progress = []
    compute_plane_metrics(
        flow, mask, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0), [plane, plane],
        progress_callback=progress.append,
    )
    assert [(event["current"], event["total"]) for event in progress] == [(1, 2), (2, 2)]

    parallel_progress = []
    parallel = compute_plane_metrics_multithread(
        flow, mask, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0), [plane, plane],
        max_workers=2,
        progress_callback=parallel_progress.append,
    )
    assert np.asarray([metric["flowrate_mL_s"] for metric in parallel]) == pytest.approx(
        np.asarray([metric["flowrate_mL_s"] for metric in compute_plane_metrics(
            flow, mask, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0), [plane, plane]
        )])
    )
    assert [(event["current"], event["total"]) for event in parallel_progress] == [(1, 2), (2, 2)]


def test_oblique_cylinder_plane_metrics_match_analytic_area_and_flow():
    size = 96
    spacing = 1.0
    radius = 16.0
    speed = 80.0
    center = np.array([48.0, 48.0, 48.0])
    x = (np.arange(size, dtype=float) + 0.5)[:, None, None] * spacing
    y = (np.arange(size, dtype=float) + 0.5)[None, :, None] * spacing
    z = (np.arange(size, dtype=float) + 0.5)[None, None, :] * spacing
    mask = (
        ((x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius ** 2)
        & (np.abs(z - center[2]) < 30.0)
    )[..., None]
    flow = np.zeros((size, size, size, 1, 3), dtype=np.float32)
    flow[..., 0, 2] = speed
    normal = np.array([0.45, 0.25, 1.0], dtype=float)
    normal /= np.linalg.norm(normal)
    plane = PlaneData(center=center, normal=normal, label=1, path_index=0)

    metric = compute_plane_metrics(
        flow,
        mask,
        spacing=(spacing, spacing, spacing),
        origin=(0.0, 0.0, 0.0),
        planes=[plane],
        RR=1000.0,
    )[0]

    expected_area = np.pi * radius ** 2 / normal[2]
    expected_flow = speed * normal[2] * expected_area / 100.0
    assert metric["area_mm2"][0] == pytest.approx(expected_area, rel=0.02)
    assert metric["flowrate_mL_s"][0] == pytest.approx(expected_flow, rel=0.02)
    assert metric["meanv_cm_s"] == pytest.approx(speed * normal[2], rel=1e-6)

    seeds = _plane_seeds(
        mask[..., 0], plane, (1.0, 1.0, 1.0), (100.0, 200.0, 300.0),
        seed_ratio=1.0, min_seeds=1, rng_seed=0,
    )
    assert seeds is not None and len(seeds) > 0
    assert np.all(np.min(seeds, axis=0) >= np.array([100.0, 200.0, 300.0]))


def test_pathline_seed_modes_support_fixed_count_and_ratio_limits():
    mask = np.ones((24, 24, 24), dtype=bool)
    plane = PlaneData(
        center=np.array([12.0, 12.0, 12.0], dtype=float),
        normal=np.array([1.0, 0.0, 0.0], dtype=float),
    )
    fixed = _plane_seeds(
        mask, plane, np.ones(3), np.zeros(3),
        seed_ratio=0.02, min_seeds=1, max_seeds=250, rng_seed=0,
        seed_mode="fixed",
    )
    ratio = _plane_seeds(
        mask, plane, np.ones(3), np.zeros(3),
        seed_ratio=0.02, min_seeds=1, max_seeds=250, rng_seed=0,
        seed_mode="ratio",
    )

    assert fixed is not None and len(fixed) == 250
    assert ratio is not None and 1 <= len(ratio) < len(fixed)


def test_pathline_temporal_source_reuses_cached_frames_across_planes():
    flow = np.zeros((20, 4, 4, 3, 3), dtype=np.float32)
    flow[..., 0] = 0.2
    mask = np.ones((20, 4, 4, 3), dtype=bool)
    source = create_pathline_temporal_source(
        flow, mask, np.ones(3), np.zeros(3), 1, 1000.0,
        temporal_cache_mb=16.0,
    )

    meshes = [
        generate_pathlines_from_plane_at_t(
            flow, 1, SimpleNamespace(), np.ones(3), np.zeros(3),
            mask_4d=mask, mask_3d=mask[..., 1], terminal_speed=0.001,
            rr=1000.0, seeds=np.asarray([[float(x), 1.5, 1.5]]),
            temporal_source=source,
        )
        for x in (2.0, 3.0)
    ]

    assert source.cache_all_phases is True
    assert all(mesh is not None and mesh.n_lines == 1 for mesh in meshes)
    assert len(source._datasets) == 3


def test_pathline_color_modes_are_stable_and_preserve_manual_overrides():
    workspace = Workspace()
    workspace.group_order = ["aorta", "pulmonary"]
    workspace.planes = [
        PlaneData(center=np.zeros(3), normal=np.array([1.0, 0.0, 0.0]), group_name="aorta"),
        PlaneData(center=np.ones(3), normal=np.array([1.0, 0.0, 0.0]), group_name="aorta"),
        PlaneData(center=np.full(3, 2.0), normal=np.array([1.0, 0.0, 0.0]), group_name="pulmonary"),
    ]

    workspace.streamline_params.pathline_color_mode = "per_plane"
    assert workspace.pathline_color_for_plane(0) != workspace.pathline_color_for_plane(1)
    workspace.planes.extend([
        PlaneData(center=np.full(3, float(index)), normal=np.array([1.0, 0.0, 0.0]))
        for index in range(3, 22)
    ])
    assert workspace.pathline_color_for_plane(0) != workspace.pathline_color_for_plane(20)

    workspace.streamline_params.pathline_color_mode = "per_group"
    assert workspace.pathline_color_for_plane(0) == workspace.pathline_color_for_plane(1)
    assert workspace.pathline_color_for_plane(0) != workspace.pathline_color_for_plane(2)

    workspace.streamline_params.pathline_color_mode = "uniform"
    workspace.streamline_params.pathline_color = "deepskyblue"
    workspace.set_pathline_color_for_plane(1, "#123456")
    assert workspace.pathline_color_for_plane(0) == "deepskyblue"
    assert workspace.pathline_color_for_plane(1) == "#123456"


def test_workspace_snapshot_restores_nonzero_origin():
    workspace = Workspace()
    workspace.origin = np.array([10.0, 20.0, 30.0], dtype=float)
    workspace.correction_raw = np.ones((2, 2, 2, 1, 3), dtype=np.float32)
    workspace.correction_high_raw = np.full((2, 2, 2, 1, 3), 2.0, dtype=np.float32)
    restored = Workspace()
    restored.restore_dict(workspace.snapshot_dict())
    assert np.allclose(restored.origin, workspace.origin)
    assert np.array_equal(restored.correction_raw, workspace.correction_raw)
    assert np.array_equal(restored.correction_high_raw, workspace.correction_high_raw)


def test_loading_case_preserves_segmentation_cache_settings(monkeypatch):
    import autoflow.core.pipeline as pipeline_module

    loaded = SimpleNamespace(
        flow=np.zeros((2, 2, 2, 1, 3), dtype=np.float32),
        mag=np.ones((2, 2, 2, 1), dtype=np.float32),
        segmentation=None,
        resolution=np.ones(3, dtype=float),
        origin=np.zeros(3, dtype=float),
        venc=np.full(3, 100.0, dtype=float),
        rr=1000.0,
        source_format="normalized_h5",
        source_group=None,
        metadata={},
        capabilities=LoaderCapabilities(
            has_segmentation=False,
            has_tke=False,
            has_complex_source=False,
            supports_wss=True,
            supports_plane_metrics=True,
        ),
        sigma=None,
        correction=None,
        correction_high=None,
        tke_array=None,
    )
    monkeypatch.setattr(pipeline_module, "load_input_data", lambda *_args, **_kwargs: loaded)

    workspace = Workspace()
    workspace.paths.flow_path = "case.h5"
    workspace.paths.segmask_path = "case.h5"
    workspace.segmentation.force_recompute_auto_cache = True
    workspace.segmentation.write_auto_cache = False
    workspace.segmentation.auto_backend = "nnUNet4D"
    workspace.segmentation.auto_folds = "all"
    workspace.segmentation.cleanup_4d_components = True
    workspace.segmentation.cleanup_4d_mode = "relative"

    PipelineEngine().load_data(workspace, lambda _message: None)

    assert workspace.segmentation.force_recompute_auto_cache is True
    assert workspace.segmentation.write_auto_cache is False
    assert workspace.segmentation.auto_backend == "nnUNet4D"
    assert workspace.segmentation.auto_folds == "all"
    assert workspace.segmentation.cleanup_4d_components is True
    assert workspace.segmentation.cleanup_4d_mode == "relative"


def test_plane_modes_support_junction_spacing_count_and_all_count():
    path = np.column_stack((np.arange(0.0, 31.0, 1.0), np.zeros(31), np.zeros(31)))

    anchored, _ = generate_planes_from_paths(
        [path],
        plane_mode="anchored_offset",
        plane_count=3,
        cross_section_distance=5.0,
        start_distance=0.0,
        end_distance=0.0,
        anchor="end",
        anchor_offset_mm=5.0,
        fork_points=[path[0]],
        smoothing_window=3,
        inter_time=1,
    )
    assert [plane.distance for plane in anchored] == pytest.approx([5.0, 10.0, 15.0])

    limited, _ = generate_planes_from_paths(
        [path],
        plane_mode="distance",
        plane_count=2,
        cross_section_distance=10.0,
        start_distance=0.0,
        end_distance=0.0,
        smoothing_window=3,
        inter_time=1,
    )
    assert [plane.distance for plane in limited] == pytest.approx([0.0, 10.0])

    all_fit, _ = generate_planes_from_paths(
        [path],
        plane_mode="distance",
        plane_count=-1,
        cross_section_distance=10.0,
        start_distance=0.0,
        end_distance=0.0,
        smoothing_window=3,
        inter_time=1,
    )
    assert [plane.distance for plane in all_fit] == pytest.approx([0.0, 10.0, 20.0, 30.0])

    centered, _ = generate_planes_from_paths(
        [path], plane_mode="fixed_step", plane_count=3, anchor="center",
        direction="both", spacing_mode="fraction", spacing_ratio=0.25,
        start_distance=0.0, end_distance=0.0, smoothing_window=3, inter_time=1,
    )
    assert [plane.distance for plane in centered] == pytest.approx([7.5, 15.0, 22.5])

    even_symmetric, _ = generate_planes_from_paths(
        [path], plane_mode="fixed_step", plane_count=4, anchor="center",
        direction="both", spacing_mode="fraction", spacing_ratio=0.25,
        start_distance=0.0, end_distance=0.0, smoothing_window=3, inter_time=1,
    )
    assert [plane.distance for plane in even_symmetric] == pytest.approx([3.75, 11.25, 18.75, 26.25])


def test_segmentation_plane_filter_uses_free_endpoint_run_and_clips_path():
    labels = np.zeros((30, 3, 3), dtype=np.int16)
    labels[:8, :, :] = 3       # junction-parent label
    labels[8:, :, :] = 5       # short branch label
    path = np.column_stack((np.arange(0.0, 20.0, 1.0), np.ones(20), np.ones(20)))
    filtered, qc = filter_paths_by_segmentation(
        [path], labels, path_info=[{"fork_roles": [{"role": "outgoing"}]}], inter_time=1,
    )
    assert qc[0]["owner_label"] == 5
    assert qc[0]["owner_label_source"] == "free_end_of_outgoing"
    assert float(filtered[0][0, 0]) >= 7.5  # first retained sample rounds to label 5


@pytest.mark.parametrize("a,b,c,d", [(0.4, -0.04, 0.2, 0.03), (-1.0, 1.0, 0.0, 0.0)])
def test_wss_vector_matches_analytic_quadratic_wall_shear(a, b, c, d):
    import pyvista as pv

    grid = pv.ImageData(dimensions=(61, 9, 9), spacing=(0.1, 0.5, 0.5), origin=(-3, -2, -2))
    s = np.abs(grid.points[:, 0])
    grid.point_data["u"] = np.zeros(len(s))
    grid.point_data["v"] = a * s + b * s**2
    grid.point_data["w"] = c * s + d * s**2
    wall = pv.Plane(center=(0, 0, 0), direction=(1, 0, 0), i_size=1, j_size=1,
                    i_resolution=2, j_resolution=2)
    result = cal_wss_from_surf(wall, grid, inward_distance=1.0, viscosity=4.0)
    np.testing.assert_allclose(result["wss_vectors"], np.tile([0, 4*a, 4*c], (wall.n_points, 1)), atol=1e-10)
    np.testing.assert_allclose(result["wss"], 4 * np.hypot(a, c), atol=1e-10)
    assert np.all(result["wss_valid"])


def test_wss_interpolates_cell_velocity_continuously_and_retains_signed_components():
    import pyvista as pv
    from autoflow.algorithms.surfaces import create_uniform_vector

    v = 0.1 * (np.arange(8, dtype=float) + 0.5)[:, None, None] * np.ones((1, 8, 8))
    grid = create_uniform_vector(np.zeros_like(v), v, np.zeros_like(v), (1, 1, 1))
    wall = pv.Plane(center=(3, 3, 3), direction=(1, 0, 0), i_size=1, j_size=1,
                    i_resolution=2, j_resolution=2)
    result = cal_wss_from_surf(wall, grid, inward_distance=0.3, viscosity=4.0, no_slip_condition=False)
    np.testing.assert_allclose(result["wss"], 0.4, atol=1e-10)
    # An open plane has no geometric inside: its supplied mesh winding defines
    # the normal. The analytic directional derivative is 0.1 * normal_x.
    np.testing.assert_allclose(result["wss_vectors"][:, 1], 0.4 * result.point_normals[:, 0], atol=1e-10)


def test_wss_linear_mode_and_invalid_wall_normal_samples():
    np.testing.assert_allclose(calculate_gradient([0.0], [0.3], [0.4], 1.0, use_parabolic=False), [0.3])
    mask = np.zeros((8, 8, 8, 2), dtype=bool)
    mask[2:6, 2:6, 2:6] = True
    flow = np.zeros(mask.shape + (3,), dtype=np.float32)
    result = compute_wss_metrics(mask, flow, (1, 1, 1), smoothing_iteration=0, inward_distance=10)
    for wall in result["wss_surfaces"]:
        assert not np.any(wall["wss_valid"])
        assert np.isnan(wall["wss"]).all()
        assert np.isnan(wall["wss_vectors"]).all()
    assert np.isnan(result["wss_volume"]).any()


def test_static_masks_reuse_plane_and_wss_geometry(monkeypatch):
    from autoflow.algorithms.metrics import sampling, wss

    mask = np.zeros((8, 8, 8, 3), dtype=bool)
    mask[1:7, 1:7, 1:7, :] = True
    flow = np.zeros(mask.shape + (3,), dtype=np.float32)
    flow[..., 0] = 10.0
    planes = [
        PlaneData(center=np.array([x, 4.0, 4.0]), normal=np.array([1.0, 0.0, 0.0]))
        for x in (2.0, 3.0, 4.0, 5.0)
    ]

    support_calls = 0
    original_support = sampling._build_plane_support_mesh

    def counting_support(*args, **kwargs):
        nonlocal support_calls
        support_calls += 1
        return original_support(*args, **kwargs)

    monkeypatch.setattr(sampling, "_build_plane_support_mesh", counting_support)
    compute_plane_metrics(flow, mask, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0), planes)
    assert support_calls == 1

    surface_calls = 0
    original_extract_surface = wss._extract_surface

    def counting_surface(*args, **kwargs):
        nonlocal surface_calls
        surface_calls += 1
        return original_extract_surface(*args, **kwargs)

    monkeypatch.setattr(wss, "_extract_surface", counting_surface)
    result = compute_wss_metrics(
        mask, flow, (1.0, 1.0, 1.0), smoothing_iteration=0,
    )
    assert surface_calls == 1
    assert len(result["wss_surfaces"]) == mask.shape[3]
    for wall in result["wss_surfaces"]:
        assert np.all(wall["wss_valid"])
        assert np.all(np.sum((wall.points - np.array([4, 4, 4])) * wall.point_normals, axis=1) < 0)


def test_derived_cache_signature_invalidates_changed_wss_parameters(monkeypatch):
    import autoflow.core.pipeline as pipeline_module

    workspace = Workspace()
    workspace.flow_raw = np.zeros((3, 3, 3, 2, 3), dtype=np.float32)
    workspace.segmask_raw = np.ones((3, 3, 3, 2), dtype=np.int16)
    workspace.segmask_binary = np.ones((3, 3, 3, 2), dtype=bool)
    engine = PipelineEngine()
    monkeypatch.setattr(engine, "preprocess", lambda _workspace: False)
    calls = []

    def fake_compute_derived_metrics(**kwargs):
        calls.append(float(kwargs["wss_viscosity"]))
        volume = np.full(workspace.segmask_binary.shape, calls[-1], dtype=np.float32)
        return {
            "wss_surfaces": [object(), object()],
            "wss_volume": volume,
            "pixelwise_export": {"wss": volume} if kwargs["save_pixelwise"] else {},
        }

    monkeypatch.setattr(pipeline_module, "compute_derived_metrics", fake_compute_derived_metrics)
    engine._ensure_derived_metrics(
        workspace, compute_wss=True, compute_tke=False, compute_pressure_gradient=False,
    )
    engine._ensure_derived_metrics(
        workspace, compute_wss=True, compute_tke=False, compute_pressure_gradient=False,
    )
    engine._ensure_derived_metrics(
        workspace, save_pixelwise=True, compute_wss=True, compute_tke=False, compute_pressure_gradient=False,
    )
    assert calls == [4.0]
    assert workspace.derived.pixelwise_export["wss"] is workspace.derived.wss_volume
    workspace.derived_params.wss_viscosity = 5.0
    engine._ensure_derived_metrics(
        workspace, compute_wss=True, compute_tke=False, compute_pressure_gradient=False,
    )

    assert calls == [4.0, 5.0]
    assert float(workspace.derived.wss_volume[0, 0, 0, 0]) == pytest.approx(5.0)
    assert workspace.derived.pixelwise_export["wss"] is workspace.derived.wss_volume


def test_streamline_auto_clim_uses_segmented_velocity_across_time():
    from autoflow.core.models import ObjectKind, SceneObject, Workspace
    from autoflow.ui.viewer import SceneController

    flow = np.zeros((2, 2, 2, 2, 3), dtype=np.float32)
    flow[0, 0, 0, 0] = [30.0, 40.0, 0.0]
    flow[0, 0, 0, 1] = [30.0, 40.0, 0.0]
    flow[1, 1, 1, 1] = [300.0, 400.0, 0.0]
    mask = np.zeros(flow.shape[:4], dtype=bool)
    mask[0, 0, 0, 0] = True

    assert automatic_streamline_clim(flow, mask) == pytest.approx((0.0, 0.5))
    assert automatic_streamline_clim(flow, np.any(mask, axis=3)) == pytest.approx((0.0, 0.5))

    workspace = Workspace()
    workspace.flow_raw = flow
    workspace.segmask_binary = mask
    obj = SceneObject(
        uid="streamlines",
        name="streamlines",
        kind=ObjectKind.FLOW,
        data_key="streamlines_live",
        scalars="Velocity",
        clim=None,
    )
    dataset = SimpleNamespace(point_data={"Velocity": np.array([0.25])}, cell_data={})
    controller = SceneController(SimpleNamespace(), workspace, lambda _message: None)
    mesh_kwargs = controller._mesh_kwargs(obj, dataset)

    assert mesh_kwargs["clim"] == pytest.approx((0.0, 0.5))
    assert mesh_kwargs["lighting"] is False

    robust_flow = np.zeros((10, 10, 10, 1, 3), dtype=np.float32)
    robust_flow[..., 0] = 100.0
    robust_flow[0, 0, 0, 0, 0] = 10000.0
    assert automatic_streamline_clim(
        robust_flow, np.ones(robust_flow.shape[:4], dtype=bool)
    ) == pytest.approx((0.0, 1.0))


def test_streamline_velocity_scalar_matches_interpolated_vector_magnitude():
    flow = np.zeros((6, 6, 6, 1, 3), dtype=np.float32)
    flow[..., 0, 0] = 100.0
    flow[:, :3, :, 0, 1] = 100.0
    flow[:, 3:, :, 0, 1] = -100.0
    mask = np.ones((6, 6, 6), dtype=bool)
    seeds = np.array(
        [[1.0, 2.5, 2.5], [2.0, 2.5, 3.0], [1.0, 3.0, 2.0]],
        dtype=float,
    )

    streamlines = generate_streamlines_at_t(
        flow,
        0,
        seeds,
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        mask_3d=mask,
        max_steps=30,
        terminal_speed=0.01,
    )

    assert streamlines is not None
    expected = np.linalg.norm(np.asarray(streamlines.point_data["vector"]), axis=1)
    assert np.asarray(streamlines.point_data["Velocity"]) == pytest.approx(expected)


def _run_case(input_name: str, *, skip_derived: bool = False):
    output_root = Path(tempfile.mkdtemp(prefix=f"autoflow_{input_name}_", dir="/tmp"))
    cfg = AutoFlowConfig(
        inputs=[str(DATA_DIR / f"{input_name}.h5")],
        output_dir=str(output_root),
        skip_derived=skip_derived,
        skip_plane_metrics=False,
        use_multithread=False,
        plane_mode="distance",
        cross_section_dist=PLANE_SPACING_MM,
        make_plane_video=False,
        make_wss_video=False,
        make_streamlines_video=False,
        make_tke_video=False,
    )
    results, case_out = run_batch(cfg)
    assert len(results) == 1
    assert results[0]["status"] == "ok"
    case_dir = Path(case_out)
    metrics = json.loads((case_dir / "plane_metrics.json").read_text(encoding="utf-8"))
    return case_dir, metrics


def _relative_error(measured: float, truth: float) -> float:
    denom = max(abs(float(truth)), 1e-12)
    return abs(float(measured) - float(truth)) / denom


def _mean_relative_error(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=float))) if values else 0.0


def _truth_mean_velocity_cm_s(flow_rate_ml_s: np.ndarray, area_mm2: float) -> float:
    area = max(float(area_mm2), 1e-12)
    flow_rate = np.asarray(flow_rate_ml_s, dtype=float)
    return float(np.mean(flow_rate) * 100.0 / area)


def test_repo_phantom_h5_files_embed_truth_group():
    expected = {
        "phantom_S": (1,),
        "phantom_U": (1,),
        "phantom_Y": (3,),
        "phantom_P": None,
    }
    for case_name, path_scale_shape in expected.items():
        path = DATA_DIR / f"{case_name}.h5"
        assert path.is_file()
        with h5py.File(path, "r") as handle:
            assert "truth" in handle
            truth = handle["truth"]
            assert truth["flow_xyzt3_cm_s"].shape[-1] == 3
            if case_name != "phantom_P":
                assert truth["segmentation_xyzt"].shape == handle["segmask"].shape
                assert truth["mag_xyzt"].shape == handle["img_complex"].shape[:-1]
                assert truth["path_scale_values"].shape == path_scale_shape
            else:
                assert truth["segmentation_xyzt"].shape == handle["segmentation"].shape
                assert truth["mag_xyzt"].shape == handle["mag"].shape
                assert truth["pressure_gradient_xyzt3_pa_m"].shape[-1] == 3


def test_load_h5_data_supports_nested_group_real_img_layout_case_insensitive_keys(tmp_path):
    path = tmp_path / "nested_real_img.h5"
    mag = np.arange(24, dtype=np.float32).reshape(2, 3, 4) + 1.0
    flow = np.stack(
        [
            100.0 + mag,
            200.0 + mag,
            300.0 + mag,
        ],
        axis=-1,
    )
    seg = (mag > 12).astype(np.int16)
    with h5py.File(path, "w") as handle:
        handle.attrs["origin"] = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        case = handle.create_group("Case001")
        case.create_dataset("IMG", data=np.concatenate([mag[..., None, None], flow[..., None, :]], axis=-1))
        case.create_dataset("resolution", data=np.array([1.2, 1.3, 1.4], dtype=np.float32))
        case.create_dataset("rr", data=np.array(812.5, dtype=np.float32))
        case.create_dataset("Venc", data=np.array([50.0, 60.0, 70.0], dtype=np.float32))
        case.create_dataset("Spatial_Order", data=np.asarray(["FH", "RL", "PA"], dtype="S4"))
        case.create_dataset("venc_order", data=np.asarray(["HF", "RL", "PA"], dtype="S4"))
        case.create_dataset("Segmentation", data=seg)

    loaded = load_h5_data(str(path))
    expected_mag = np.flip(np.flip(np.transpose(mag, (1, 2, 0)), axis=0), axis=1)
    flow_spatial = np.flip(np.flip(np.transpose(flow, (1, 2, 0, 3)), axis=0), axis=1)
    expected_flow = np.stack(
        [
            -flow_spatial[..., 1],
            -flow_spatial[..., 2],
            -flow_spatial[..., 0],
        ],
        axis=-1,
    )
    expected_seg = np.flip(np.flip(np.transpose(seg, (1, 2, 0)), axis=0), axis=1)

    assert loaded.source_format == "normalized_h5"
    assert loaded.source_group == "Case001"
    assert loaded.metadata["h5_layout"] == "combined_img_real"
    assert loaded.mag.shape == (3, 4, 2, 1)
    assert loaded.flow.shape == (3, 4, 2, 1, 3)
    assert loaded.segmentation.shape == (3, 4, 2, 1)
    assert np.allclose(loaded.mag[..., 0], expected_mag)
    assert np.allclose(loaded.flow[..., 0, :], expected_flow)
    assert np.array_equal(loaded.segmentation[..., 0], expected_seg)
    assert np.allclose(loaded.resolution, [1.3, 1.4, 1.2])
    assert np.allclose(loaded.origin, [1.0, 2.0, 3.0])
    assert np.allclose(loaded.venc, [60.0, 70.0, 50.0])
    assert loaded.rr == 812.5


def test_group_h5_corr_and_seg_are_discovered_and_loaded(tmp_path):
    path = tmp_path / "group_features.h5"
    img = np.zeros((2, 2, 2, 1, 4), dtype=np.float32)
    with h5py.File(path, "w") as handle:
        complete = handle.create_group("Complete")
        complete.create_dataset("img", data=img)
        corr = complete.create_dataset("corr", data=np.zeros((2, 2, 2, 1, 3), dtype=np.float32))
        corr.attrs["corr_algorithm"] = "wrls_arto"
        complete.create_dataset("seg", data=np.ones((2, 2, 2, 1), dtype=np.int16))
        incomplete = handle.create_group("Incomplete")
        incomplete.create_dataset("img", data=img)

    cases = discover_h5_input_cases(str(path))
    by_group = {case.source_group: case for case in cases}

    assert by_group["Complete"].metadata["has_background_correction_cache"] is True
    assert by_group["Complete"].metadata["background_correction_method"] == "wrls_arto"
    assert by_group["Complete"].metadata["has_embedded_segmentation"] is True
    assert by_group["Incomplete"].metadata["has_background_correction_cache"] is False
    assert by_group["Incomplete"].metadata["has_embedded_segmentation"] is False
    inspected = inspect_h5_input_case(by_group["Complete"])
    assert inspected["source_group"] == "Complete"
    assert inspected["background_correction_method"] == "wrls_arto"
    loaded = load_h5_data(str(path), source_group="Complete")
    assert loaded.segmentation is not None
    assert loaded.capabilities.has_segmentation is True


def test_load_h5_data_orders_dual_venc_channel_groups_by_venc(tmp_path):
    path = tmp_path / "dual_venc_high_first.h5"
    velocity = np.zeros((2, 2, 2, 1, 3), dtype=np.float32)
    velocity[..., 0] = 20.0
    velocity[..., 1] = -10.0
    velocity[..., 2] = 5.0
    high_venc = np.array([150.0, 150.0, 150.0], dtype=np.float32)
    low_venc = np.array([50.0, 50.0, 50.0], dtype=np.float32)
    high_phase = np.pi * velocity / high_venc.reshape((1, 1, 1, 1, 3))
    low_phase = np.pi * velocity / low_venc.reshape((1, 1, 1, 1, 3))
    img = np.ones((2, 2, 2, 1, 7), dtype=np.complex64)
    img[..., 1:4] = np.exp(1j * high_phase)
    img[..., 4:7] = np.exp(1j * low_phase)

    with h5py.File(path, "w") as handle:
        handle.create_dataset("img", data=img)
        handle.create_dataset("Resolution", data=np.ones(3, dtype=np.float32))
        handle.create_dataset("VENC", data=np.concatenate([high_venc, low_venc]))
        handle.create_dataset("SpatialOrder", data=np.asarray(["LR", "AP", "FH"], dtype="S4"))
        handle.create_dataset("VENCOrder", data=np.asarray(["LR", "AP", "FH"], dtype="S4"))

    loaded = load_h5_data(str(path))

    assert loaded.source_format == "legacy_h5_dual_venc"
    assert np.allclose(loaded.flow, velocity, atol=1e-5)
    assert np.allclose(loaded.venc, high_venc)
    assert loaded.metadata["dual_venc"]["lv_channel_indices"] == [4, 5, 6]
    assert loaded.metadata["dual_venc"]["hv_channel_indices"] == [1, 2, 3]


@pytest.mark.parametrize(
    ("mode", "expected_venc", "expected_velocity"),
    [
        ("lv", 50.0, [20.0, -10.0, 5.0]),
        ("hv", 150.0, [20.0, -10.0, 5.0]),
        ("dv", 150.0, [20.0, -10.0, 5.0]),
    ],
)
def test_load_h5_data_selects_dual_venc_source(tmp_path, mode, expected_venc, expected_velocity):
    path = tmp_path / f"dual_venc_{mode}.h5"
    velocity = np.asarray(expected_velocity, dtype=np.float32).reshape(1, 1, 1, 1, 3)
    high_venc = np.full(3, 150.0, dtype=np.float32)
    low_venc = np.full(3, 50.0, dtype=np.float32)
    img = np.ones((1, 1, 1, 1, 7), dtype=np.complex64)
    img[..., 1:4] = np.exp(1j * np.pi * velocity / high_venc.reshape((1, 1, 1, 1, 3)))
    img[..., 4:7] = np.exp(1j * np.pi * velocity / low_venc.reshape((1, 1, 1, 1, 3)))
    with h5py.File(path, "w") as handle:
        handle.create_dataset("img", data=img)
        handle.create_dataset("Resolution", data=np.ones(3, dtype=np.float32))
        handle.create_dataset("VENC", data=np.concatenate([high_venc, low_venc]))
        handle.create_dataset("SpatialOrder", data=np.asarray(["LR", "AP", "FH"], dtype="S4"))
        handle.create_dataset("VENCOrder", data=np.asarray(["LR", "AP", "FH"], dtype="S4"))

    case = discover_h5_input_cases(str(path))[0]
    assert case.metadata["is_dual_venc"] is True
    loaded = load_h5_data(str(path), dual_venc_mode=mode)

    assert np.allclose(loaded.venc, expected_venc)
    assert np.allclose(loaded.flow.reshape(-1, 3)[0], expected_velocity, atol=1e-4)
    assert loaded.metadata["dual_venc"]["selected_mode"] == mode


def test_load_h5_data_reuses_dual_venc_singleton_time_corr_cache(tmp_path, monkeypatch):
    path = tmp_path / "dual_venc_corr_cache.h5"
    img = np.ones((2, 2, 2, 2, 7), dtype=np.complex64)
    corr = np.zeros((2, 2, 2, 1, 3), dtype=np.float32)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("img", data=img)
        handle.create_dataset("Resolution", data=np.ones(3, dtype=np.float32))
        handle.create_dataset(
            "VENC",
            data=np.array([150.0, 150.0, 150.0, 50.0, 50.0, 50.0], dtype=np.float32),
        )
        handle.create_dataset("SpatialOrder", data=np.asarray(["LR", "AP", "FH"], dtype="S4"))
        handle.create_dataset("VENCOrder", data=np.asarray(["LR", "AP", "FH"], dtype="S4"))
        for name in ("corr_low", "corr_high"):
            ds = handle.create_dataset(name, data=corr)
            ds.attrs["corr_algorithm"] = "msac"
            ds.attrs["corr_version"] = 1
            ds.attrs["corr_fit_order"] = 3
            ds.attrs["corr_threshold"] = 0.1

    def fail_execute_msac(*_args, **_kwargs):
        raise AssertionError("singleton-time dual-venc corr cache should avoid MSAC")

    monkeypatch.setattr("autoflow.algorithms.phase_correction.execute_msac", fail_execute_msac)
    loaded = load_h5_data(
        str(path),
        correction_config={"enabled": True, "corr_fit_order": 3, "threshold": 0.1},
    )

    reports = loaded.metadata["background_phase_correction"]
    assert reports["dual_venc_low"]["cache_hit"] is True
    assert reports["dual_venc_high"]["cache_hit"] is True
    assert loaded.flow.shape == (2, 2, 2, 2, 3)


def test_load_h5_data_runs_dual_venc_corrections_concurrently(tmp_path, monkeypatch):
    import threading

    path = tmp_path / "dual_venc_parallel_corr.h5"
    img = np.ones((2, 2, 2, 2, 7), dtype=np.complex64)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("img", data=img)
        handle.create_dataset("Resolution", data=np.ones(3, dtype=np.float32))
        handle.create_dataset(
            "VENC",
            data=np.array([50.0, 50.0, 50.0, 150.0, 150.0, 150.0], dtype=np.float32),
        )
        handle.create_dataset("SpatialOrder", data=np.asarray(["LR", "AP", "FH"], dtype="S4"))
        handle.create_dataset("VENCOrder", data=np.asarray(["LR", "AP", "FH"], dtype="S4"))

    barrier = threading.Barrier(2)
    worker_names = set()
    worker_names_lock = threading.Lock()

    def fake_apply(values, config=None, progress_callback=None, source_mode="", cached_corr=None):
        with worker_names_lock:
            worker_names.add(threading.current_thread().name)
        barrier.wait(timeout=2.0)
        if progress_callback:
            progress_callback({"stage":"background_phase_done","message":source_mode})
        return values, None, {
            "enabled": True,
            "applied": False,
            "source_mode": source_mode,
            "cache_hit": False,
            "cache_reason": "missing",
            "skipped_reason": "test",
        }

    monkeypatch.setattr("autoflow.algorithms.data.dual_venc.apply_background_phase_correction_to_complex", fake_apply)
    callback_threads = []
    caller_thread = threading.get_ident()
    loaded = load_h5_data(str(path), correction_config={"enabled": True},
                          progress_callback=lambda payload: callback_threads.append(threading.get_ident()))
    assert callback_threads
    assert set(callback_threads) == {caller_thread}

    assert len(worker_names) == 2
    assert all(name.startswith("autoflow-bgc") for name in worker_names)
    assert loaded.flow.shape == (2, 2, 2, 2, 3)


def test_run_case_segmentation_only_stops_before_skeleton(tmp_path, monkeypatch):
    path = tmp_path / "segmentation_only.h5"
    mag = np.ones((2, 2, 2, 1), dtype=np.float32)
    flow = np.zeros((2, 2, 2, 1, 3), dtype=np.float32)
    segmentation = np.ones((2, 2, 2, 1), dtype=np.int16)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("mag", data=mag)
        handle.create_dataset("flow", data=flow)
        handle.create_dataset("segmentation", data=segmentation)
        handle.create_dataset("Resolution", data=np.ones(3, dtype=np.float32))
        handle.create_dataset("VENC", data=np.full(3, 100.0, dtype=np.float32))
        handle.create_dataset("SpatialOrder", data=np.asarray(["LR", "AP", "FH"], dtype="S4"))
        handle.create_dataset("VENCOrder", data=np.asarray(["LR", "AP", "FH"], dtype="S4"))

    def fail_run_step(*_args, **_kwargs):
        raise AssertionError("segmentation-only must stop before skeleton")

    monkeypatch.setattr(PipelineEngine, "run_step", fail_run_step)
    out_dir = tmp_path / "out"
    summary = run_case(
        str(path),
        output_dir=str(out_dir),
        config=AutoFlowConfig(segmentation_only=True),
    )

    assert summary["segmentation_only"] is True
    assert summary["segmentation_source"] == "original"
    assert summary["stage_times_sec"].keys() == {"load"}
    assert summary["quality_report"]["overall_status"] == "incomplete"
    assert summary["quality_report_file"] == str(out_dir / "quality_report.json")
    assert (out_dir / "summary.json").is_file()
    assert (out_dir / "quality_report.json").is_file()


def test_load_h5_data_flattens_singleton_row_and_column_metadata_vectors(tmp_path):
    path = tmp_path / "real_img_vector_metadata.h5"
    mag = np.ones((2, 2, 2, 1), dtype=np.float32) * 5.0
    flow = np.zeros((2, 2, 2, 1, 3), dtype=np.float32)
    flow[..., 0] = 10.0
    flow[..., 1] = 20.0
    flow[..., 2] = 30.0
    img = np.concatenate([mag[..., None], flow], axis=-1)

    with h5py.File(path, "w") as handle:
        handle.create_dataset("img", data=img)
        handle.create_dataset("Resolution", data=np.array([[1.2, 1.3, 1.4]], dtype=np.float32))
        handle.create_dataset("Origin", data=np.array([[5.0, 6.0, 7.0]], dtype=np.float32))
        handle.create_dataset("RR", data=np.array([[812.5]], dtype=np.float32))
        handle.create_dataset("VENC", data=np.array([[50.0], [60.0], [70.0]], dtype=np.float32))
        handle.create_dataset("SpatialOrder", data=np.asarray(["LR", "AP", "FH"], dtype="S4"))
        handle.create_dataset("VENCOrder", data=np.asarray(["LR", "AP", "FH"], dtype="S4"))

    loaded = load_h5_data(str(path))

    assert np.allclose(loaded.mag, mag)
    assert np.allclose(loaded.flow, flow)
    assert np.allclose(loaded.resolution, [1.2, 1.3, 1.4])
    assert np.allclose(loaded.origin, [5.0, 6.0, 7.0])
    assert np.allclose(loaded.venc, [50.0, 60.0, 70.0])
    assert loaded.rr == 812.5


def test_load_h5_data_accepts_comma_separated_spatial_and_venc_order_strings(tmp_path):
    path = tmp_path / "real_img_csv_orders.h5"
    mag = np.arange(24, dtype=np.float32).reshape(2, 3, 4) + 1.0
    flow = np.stack(
        [
            100.0 + mag,
            200.0 + mag,
            300.0 + mag,
        ],
        axis=-1,
    )
    with h5py.File(path, "w") as handle:
        handle.create_dataset("img", data=np.concatenate([mag[..., None, None], flow[..., None, :]], axis=-1))
        handle.create_dataset("Resolution", data=np.array([1.2, 1.3, 1.4], dtype=np.float32))
        handle.create_dataset("RR", data=np.array(812.5, dtype=np.float32))
        handle.create_dataset("VENC", data=np.array([50.0, 60.0, 70.0], dtype=np.float32))
        handle.create_dataset("SpatialOrder", data=np.bytes_("FH,RL,PA"))
        handle.create_dataset("VENCOrder", data=np.bytes_("HF,RL,PA"))

    loaded = load_h5_data(str(path))
    expected_mag = np.flip(np.flip(np.transpose(mag, (1, 2, 0)), axis=0), axis=1)
    flow_spatial = np.flip(np.flip(np.transpose(flow, (1, 2, 0, 3)), axis=0), axis=1)
    expected_flow = np.stack(
        [
            -flow_spatial[..., 1],
            -flow_spatial[..., 2],
            -flow_spatial[..., 0],
        ],
        axis=-1,
    )

    assert loaded.metadata["spatial_order_raw"] == ["FH", "RL", "PA"]
    assert loaded.metadata["venc_order_raw"] == ["HF", "RL", "PA"]
    assert np.allclose(loaded.mag[..., 0], expected_mag)
    assert np.allclose(loaded.flow[..., 0, :], expected_flow)


def test_load_h5_data_rescales_real_img_phase_radians_to_venc(tmp_path):
    path = tmp_path / "real_img_phase_units.h5"
    mag = np.full((2, 2, 2, 1), 5.0, dtype=np.float32)
    flow_velocity = np.zeros((2, 2, 2, 1, 3), dtype=np.float32)
    flow_velocity[..., 0] = 45.0
    flow_velocity[..., 1] = -54.0
    flow_velocity[..., 2] = 63.0
    venc = np.array([50.0, 60.0, 70.0], dtype=np.float32)
    flow_phase = np.pi * flow_velocity / venc.reshape((1, 1, 1, 1, 3))
    img = np.concatenate([mag[..., None], flow_phase], axis=-1)

    with h5py.File(path, "w") as handle:
        handle.create_dataset("img", data=img)
        handle.create_dataset("Resolution", data=np.array([1.0, 1.1, 1.2], dtype=np.float32))
        handle.create_dataset("RR", data=np.array(700.0, dtype=np.float32))
        handle.create_dataset("VENC", data=venc)
        handle.create_dataset("SpatialOrder", data=np.asarray(["LR", "AP", "FH"], dtype="S4"))
        handle.create_dataset("VENCOrder", data=np.asarray(["LR", "AP", "FH"], dtype="S4"))

    loaded = load_h5_data(str(path))

    assert loaded.metadata["h5_layout"] == "combined_img_real"
    assert loaded.metadata["flow_value_unit_raw"] == "phase_radians"
    assert loaded.metadata["flow_rescaled_from_pi_to_venc"] is True
    assert np.allclose(loaded.flow, flow_velocity, atol=1e-5)
    assert np.allclose(loaded.mag, mag)


def test_load_h5_data_rejects_non_xyztv_real_img_layouts(tmp_path):
    path = tmp_path / "real_img_channel_first.h5"
    mag = (np.arange(48, dtype=np.float32).reshape(2, 3, 4, 2) + 1.0) / 10.0
    flow = np.stack(
        [
            10.0 + mag,
            20.0 + mag,
            30.0 + mag,
        ],
        axis=-1,
    )
    img = np.concatenate([mag[..., None], flow], axis=-1)
    img_channel_first = np.transpose(img, (4, 3, 2, 1, 0))

    with h5py.File(path, "w") as handle:
        handle.create_dataset("img", data=img_channel_first)
        handle.create_dataset("Resolution", data=np.array([1.0, 1.1, 1.2], dtype=np.float32))
        handle.create_dataset("RR", data=np.array(720.0, dtype=np.float32))
        handle.create_dataset("VENC", data=np.array([50.0, 60.0, 70.0], dtype=np.float32))
        handle.create_dataset("SpatialOrder", data=np.asarray(["LR", "AP", "FH"], dtype="S4"))
        handle.create_dataset("VENCOrder", data=np.asarray(["LR", "AP", "FH"], dtype="S4"))

    with pytest.raises(ValueError, match=r"XYZTV layout with V=4.*\(4, 2, 4, 3, 2\)"):
        load_h5_data(str(path))

    no_time_path = tmp_path / "real_img_without_time_axis.h5"
    with h5py.File(no_time_path, "w") as handle:
        handle.create_dataset("img", data=img[..., 0, :])
        handle.create_dataset("Resolution", data=np.array([1.0, 1.1, 1.2], dtype=np.float32))
        handle.create_dataset("RR", data=np.array(720.0, dtype=np.float32))
        handle.create_dataset("VENC", data=np.array([50.0, 60.0, 70.0], dtype=np.float32))
        handle.create_dataset("SpatialOrder", data=np.asarray(["LR", "AP", "FH"], dtype="S4"))
        handle.create_dataset("VENCOrder", data=np.asarray(["LR", "AP", "FH"], dtype="S4"))

    with pytest.raises(ValueError, match=r"XYZTV layout with V=4.*\(2, 3, 4, 4\)"):
        load_h5_data(str(no_time_path))


def test_load_h5_data_reorients_normalized_h5_real_flow_mag_and_optional_fields(tmp_path):
    path = tmp_path / "normalized_real_fields.h5"
    mag = (np.arange(48, dtype=np.float32).reshape(2, 3, 4, 2) + 1.0) / 10.0
    flow = np.stack(
        [
            10.0 + mag,
            20.0 + mag,
            30.0 + mag,
        ],
        axis=-1,
    )
    seg = (mag > 2.0).astype(np.int16)
    sigma = np.stack(
        [
            1.0 + mag,
            2.0 + mag,
            3.0 + mag,
        ],
        axis=-1,
    )
    tke = 100.0 + mag

    with h5py.File(path, "w") as handle:
        case = handle.create_group("Case002")
        case.create_dataset("mag", data=mag)
        case.create_dataset("flow", data=flow)
        case.create_dataset("segmask", data=seg)
        case.create_dataset("sigma", data=sigma)
        case.create_dataset("tke_array", data=tke)
        case.create_dataset("Resolution", data=np.array([1.1, 1.2, 1.3], dtype=np.float32))
        case.create_dataset("RR", data=np.array(640.0, dtype=np.float32))
        case.create_dataset("Venc", data=np.array([45.0, 55.0, 65.0], dtype=np.float32))
        case.create_dataset("SpatialOrder", data=np.asarray(["FH", "RL", "PA"], dtype="S4"))
        case.create_dataset("VENCOrder", data=np.asarray(["HF", "RL", "PA"], dtype="S4"))

    loaded = load_h5_data(str(path))
    expected_mag = np.flip(np.flip(np.transpose(mag, (1, 2, 0, 3)), axis=0), axis=1)
    flow_spatial = np.flip(np.flip(np.transpose(flow, (1, 2, 0, 3, 4)), axis=0), axis=1)
    expected_flow = np.stack(
        [
            -flow_spatial[..., 1],
            -flow_spatial[..., 2],
            -flow_spatial[..., 0],
        ],
        axis=-1,
    )
    expected_seg = np.flip(np.flip(np.transpose(seg, (1, 2, 0, 3)), axis=0), axis=1)
    sigma_spatial = np.flip(np.flip(np.transpose(sigma, (1, 2, 0, 3, 4)), axis=0), axis=1)
    expected_sigma = np.stack(
        [
            sigma_spatial[..., 1],
            sigma_spatial[..., 2],
            sigma_spatial[..., 0],
        ],
        axis=-1,
    )
    expected_tke = np.flip(np.flip(np.transpose(tke, (1, 2, 0, 3)), axis=0), axis=1)

    assert loaded.source_format == "normalized_h5"
    assert loaded.source_group == "Case002"
    assert loaded.metadata["h5_layout"] == "normalized_h5"
    assert loaded.mag.shape == (3, 4, 2, 2)
    assert loaded.flow.shape == (3, 4, 2, 2, 3)
    assert loaded.segmentation.shape == (3, 4, 2, 2)
    assert loaded.sigma is not None
    assert loaded.tke_array is not None
    assert np.allclose(loaded.mag, expected_mag)
    assert np.allclose(loaded.flow, expected_flow)
    assert np.array_equal(loaded.segmentation, expected_seg)
    assert np.allclose(loaded.sigma, expected_sigma)
    assert np.allclose(loaded.tke_array, expected_tke)
    assert np.allclose(loaded.resolution, [1.2, 1.3, 1.1])
    assert np.allclose(loaded.venc, [55.0, 65.0, 45.0])
    assert loaded.rr == 640.0


def test_load_h5_data_writes_and_reuses_background_phase_corr_cache(monkeypatch, tmp_path):
    path = tmp_path / "bgc_cache.h5"
    mag = np.ones((2, 2, 2, 1), dtype=np.float32)
    flow = np.zeros((2, 2, 2, 1, 3), dtype=np.float32)
    corr_xyzt3 = np.zeros((2, 2, 2, 1, 3), dtype=np.float32)
    corr_xyzt3[..., 0] = 0.1
    corr_xyzt3[..., 1] = -0.2
    corr_xyzt3[..., 2] = 0.3
    with h5py.File(path, "w") as handle:
        handle.create_dataset("mag", data=mag)
        handle.create_dataset("flow", data=flow)
        handle.create_dataset("Resolution", data=np.array([1.0, 1.0, 1.0], dtype=np.float32))
        handle.create_dataset("RR", data=np.array(1000.0, dtype=np.float32))
        handle.create_dataset("VENC", data=np.array([100.0, 100.0, 100.0], dtype=np.float32))
        handle.create_dataset("SpatialOrder", data=np.asarray(["LR", "AP", "FH"], dtype="S4"))
        handle.create_dataset("VENCOrder", data=np.asarray(["LR", "AP", "FH"], dtype="S4"))

    calls = {"count": 0}

    def fake_execute_msac(im, corr_fit_order=3, th=0.1, progress_callback=None):
        calls["count"] += 1
        if progress_callback is not None:
            progress_callback({
                "stage": "background_phase_fit",
                "current": 1,
                "total": 1,
                "message": "fit",
            })
        corr_nvtzyx = np.transpose(corr_xyzt3, (4, 3, 2, 1, 0)).astype(np.float32)
        stationary = np.ones(corr_xyzt3.shape[:3], dtype=bool)
        return corr_nvtzyx, stationary, {
            "applied": True,
            "corr_fit_order": int(corr_fit_order),
            "threshold": float(th),
            "stationary_voxels": int(np.sum(stationary)),
            "skipped_reason": "",
        }

    monkeypatch.setattr("autoflow.algorithms.phase_correction.execute_msac", fake_execute_msac)
    cfg = {"enabled": True, "corr_fit_order": 2, "threshold": 0.25}
    progress_events = []
    loaded = load_h5_data(str(path), correction_config=cfg, progress_callback=progress_events.append)

    assert calls["count"] == 1
    first_meta = loaded.metadata["background_phase_correction"]
    assert first_meta["applied"] is True
    assert first_meta["cache_hit"] is False
    assert first_meta["cache_written"] is True
    assert any(event["stage"] == "h5_background_phase_fit" for event in progress_events)
    assert any(event["stage"] == "h5_background_phase_cache_write" for event in progress_events)
    assert any(event["stage"] == "h5_background_phase_cache_done" for event in progress_events)
    with h5py.File(path, "r") as handle:
        assert "corr" in handle
        assert handle["corr"].shape == corr_xyzt3.shape
        assert np.allclose(handle["corr"][:], corr_xyzt3)
        assert int(handle["corr"].attrs["corr_fit_order"]) == 2
        assert np.isclose(float(handle["corr"].attrs["corr_threshold"]), 0.25)

    def fail_execute_msac(*_args, **_kwargs):
        raise AssertionError("cached corr should avoid MSAC")

    monkeypatch.setattr("autoflow.algorithms.phase_correction.execute_msac", fail_execute_msac)
    loaded_cached = load_h5_data(str(path), correction_config=cfg)

    cached_meta = loaded_cached.metadata["background_phase_correction"]
    assert cached_meta["applied"] is True
    assert cached_meta["cache_hit"] is True
    assert cached_meta["cache_name"] == "corr"
    assert np.allclose(loaded_cached.flow, loaded.flow)


def test_disabled_background_correction_does_not_read_compressed_cache(monkeypatch, tmp_path):
    path = tmp_path / "disabled_corr_cache.h5"
    mag = np.ones((3, 3, 2, 1), dtype=np.float32)
    flow = np.zeros((3, 3, 2, 1, 3), dtype=np.float32)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("mag", data=mag)
        handle.create_dataset("flow", data=flow)
        handle.create_dataset("corr", data=np.ones((3, 3, 2, 1, 3), dtype=np.float32), compression="gzip")
        handle.create_dataset("Resolution", data=np.ones(3, dtype=np.float32))
        handle.create_dataset("VENC", data=np.full(3, 100.0, dtype=np.float32))
        handle.create_dataset("SpatialOrder", data=np.asarray(["LR", "AP", "FH"], dtype="S4"))
        handle.create_dataset("VENCOrder", data=np.asarray(["LR", "AP", "FH"], dtype="S4"))

    def fail_cache_read(*_args, **_kwargs):
        raise AssertionError("disabled correction must not read the corr dataset")

    monkeypatch.setattr("autoflow.algorithms.data._read_background_phase_corr_cache_from_scopes", fail_cache_read)
    loaded = load_h5_data(str(path), correction_config={"enabled": False})
    assert np.array_equal(loaded.flow, flow)
    assert loaded.metadata["background_phase_correction"]["skipped_reason"] == "disabled"


def test_msac_reports_trial_progress():
    from autoflow.algorithms.phase_correction import msac

    points = np.zeros((6, 3), dtype=np.float32)
    parameters = {
        "samples": 2,
        "msac_thresh": 0.1,
        "trials": 4,
        "n_enc": 1,
    }
    functions = {
        "msac_fit": lambda _sample: np.zeros((1, 1), dtype=np.float32),
        "msac_dist": lambda _coeffs, values: np.zeros((values.shape[0], 1), dtype=np.float32),
    }
    progress = []

    msac(points, parameters, functions, progress_callback=lambda current, total: progress.append((current, total)))

    assert progress == [(1, 4), (2, 4), (3, 4), (4, 4)]


def test_wrls_arto_recovers_synthetic_polynomial_background():
    from autoflow.algorithms.phase_correction import execute_wrls_arto

    nt, nz, ny, nx = 8, 6, 7, 8
    x = np.arange(nx, dtype=np.float32) - nx // 2
    y = np.arange(ny, dtype=np.float32) - ny // 2
    z = np.arange(nz, dtype=np.float32) - nz // 2
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    expected_fields = [
        0.10 + 0.005 * xx - 0.003 * yy,
        -0.05 + 0.002 * zz,
        0.03 + 0.001 * xx * yy,
    ]
    rng = np.random.default_rng(17)
    image = np.ones((4, nt, nz, ny, nx), dtype=np.complex64)
    for direction, expected in enumerate(expected_fields):
        phase_tzyx = np.transpose(expected, (2, 1, 0))[np.newaxis, ...]
        phase_tzyx = phase_tzyx + rng.normal(0.0, 0.003, size=(nt, nz, ny, nx))
        image[direction + 1] = np.exp(1j * phase_tzyx).astype(np.complex64)

    correction, stationary, report = execute_wrls_arto(
        image,
        corr_fit_order=3,
        fista_iterations=200,
        gmm_iterations=30,
    )

    assert report["applied"] is True
    assert correction.shape == (3, 1, nz, ny, nx)
    assert stationary.shape == (nz, ny, nx)
    assert np.any(stationary)
    for direction, expected in enumerate(expected_fields):
        actual = np.transpose(correction[direction, 0], (2, 1, 0))
        assert np.sqrt(np.mean((actual - expected) ** 2)) < 8e-4


def test_wrls_arto_cache_is_separate_from_msac(monkeypatch, tmp_path):
    path = tmp_path / "wrls_arto_cache.h5"
    mag = np.ones((3, 3, 3, 2), dtype=np.float32)
    flow = np.zeros((3, 3, 3, 2, 3), dtype=np.float32)
    corr_xyzt3 = np.zeros((3, 3, 3, 1, 3), dtype=np.float32)
    corr_xyzt3[..., 0] = 0.02
    with h5py.File(path, "w") as handle:
        handle.create_dataset("mag", data=mag)
        handle.create_dataset("flow", data=flow)
        handle.create_dataset("Resolution", data=np.ones(3, dtype=np.float32))
        handle.create_dataset("VENC", data=np.full(3, 100.0, dtype=np.float32))
        handle.create_dataset("SpatialOrder", data=np.asarray(["LR", "AP", "FH"], dtype="S4"))
        handle.create_dataset("VENCOrder", data=np.asarray(["LR", "AP", "FH"], dtype="S4"))
        stale = handle.create_dataset("corr", data=np.zeros_like(corr_xyzt3))
        stale.attrs["corr_algorithm"] = "msac"
        stale.attrs["corr_version"] = 1
        stale.attrs["corr_fit_order"] = 3
        stale.attrs["corr_threshold"] = 0.1

    calls = {"count": 0}

    def fake_execute_wrls_arto(im, **kwargs):
        calls["count"] += 1
        stationary = np.ones(im.shape[2:], dtype=bool)
        return np.transpose(corr_xyzt3, (4, 3, 2, 1, 0)), stationary, {
            "applied": True,
            "corr_fit_order": int(kwargs["corr_fit_order"]),
            "stationary_voxels": int(np.sum(stationary)),
            "skipped_reason": "",
        }

    monkeypatch.setattr(
        "autoflow.algorithms.phase_correction.execute_wrls_arto",
        fake_execute_wrls_arto,
    )
    config = {
        "enabled": True,
        "method": "wrls_arto",
        "corr_fit_order": 3,
        "wrls_fista_iterations": 200,
        "wrls_gmm_iterations": 30,
    }
    first = load_h5_data(str(path), correction_config=config)

    assert calls["count"] == 1
    assert first.metadata["background_phase_correction"]["cache_reason"].startswith("algorithm_mismatch")
    with h5py.File(path, "r") as handle:
        attrs = handle["corr"].attrs
        assert attrs["corr_algorithm"] == "wrls_arto"
        assert int(attrs["corr_wrls_fista_iterations"]) == 200
        assert int(attrs["corr_wrls_gmm_iterations"]) == 30

    def fail_execute_wrls_arto(*_args, **_kwargs):
        raise AssertionError("compatible WRLS+ARTO cache should be reused")

    monkeypatch.setattr(
        "autoflow.algorithms.phase_correction.execute_wrls_arto",
        fail_execute_wrls_arto,
    )
    cached = load_h5_data(str(path), correction_config=config)
    assert cached.metadata["background_phase_correction"]["cache_hit"] is True
    assert np.allclose(cached.flow, first.flow)


def test_load_h5_data_grouped_h5_does_not_reuse_untagged_root_corr(monkeypatch, tmp_path):
    path = tmp_path / "grouped_bgc_cache.h5"
    mag = np.ones((2, 2, 2, 1), dtype=np.float32)
    flow = np.zeros((2, 2, 2, 1, 3), dtype=np.float32)
    stale_corr = np.full((2, 2, 2, 1, 3), 0.5, dtype=np.float32)
    fresh_corr = np.zeros((2, 2, 2, 1, 3), dtype=np.float32)
    fresh_corr[..., 0] = 0.1
    fresh_corr[..., 1] = -0.2
    fresh_corr[..., 2] = 0.3
    with h5py.File(path, "w") as handle:
        handle.create_dataset("corr", data=stale_corr)
        for case_name in ("CaseA", "CaseB"):
            case = handle.create_group(case_name)
            case.create_dataset("mag", data=mag)
            case.create_dataset("flow", data=flow)
            case.create_dataset("Resolution", data=np.array([1.0, 1.0, 1.0], dtype=np.float32))
            case.create_dataset("RR", data=np.array(1000.0, dtype=np.float32))
            case.create_dataset("VENC", data=np.array([100.0, 100.0, 100.0], dtype=np.float32))
            case.create_dataset("SpatialOrder", data=np.asarray(["LR", "AP", "FH"], dtype="S4"))
            case.create_dataset("VENCOrder", data=np.asarray(["LR", "AP", "FH"], dtype="S4"))

    calls = {"count": 0}

    def fake_execute_msac(im, corr_fit_order=3, th=0.1, progress_callback=None):
        calls["count"] += 1
        corr_nvtzyx = np.transpose(fresh_corr, (4, 3, 2, 1, 0)).astype(np.float32)
        stationary = np.ones(fresh_corr.shape[:3], dtype=bool)
        return corr_nvtzyx, stationary, {
            "applied": True,
            "corr_fit_order": int(corr_fit_order),
            "threshold": float(th),
            "stationary_voxels": int(np.sum(stationary)),
            "skipped_reason": "",
        }

    monkeypatch.setattr("autoflow.algorithms.phase_correction.execute_msac", fake_execute_msac)
    cfg = {"enabled": True, "corr_fit_order": 2, "threshold": 0.25}
    loaded = load_h5_data(str(path), correction_config=cfg, source_group="CaseA")

    assert calls["count"] == 1
    first_meta = loaded.metadata["background_phase_correction"]
    assert first_meta["cache_hit"] is False
    assert first_meta["cache_reason"] == "missing_group_tag"
    with h5py.File(path, "r") as handle:
        assert np.allclose(handle["corr"][:], stale_corr)
        assert np.allclose(handle["CaseA"]["corr"][:], fresh_corr)
        assert handle["CaseA"]["corr"].attrs["corr_source_group"] == "CaseA"

    def fail_execute_msac(*_args, **_kwargs):
        raise AssertionError("group-specific cached corr should avoid MSAC")

    monkeypatch.setattr("autoflow.algorithms.phase_correction.execute_msac", fail_execute_msac)
    loaded_cached = load_h5_data(str(path), correction_config=cfg, source_group="CaseA")

    cached_meta = loaded_cached.metadata["background_phase_correction"]
    assert cached_meta["cache_hit"] is True
    assert cached_meta["cache_name"] == "corr"
    assert np.allclose(loaded_cached.flow, loaded.flow)


def test_save_segmentation_to_source_h5_writes_reusable_embedded_segmentation(tmp_path):
    path = tmp_path / "embedded_autoseg_cache.h5"
    mag = np.arange(48, dtype=np.float32).reshape(2, 3, 4, 2) + 1.0
    flow = np.zeros((2, 3, 4, 2, 3), dtype=np.float32)
    seg_internal = np.zeros((3, 4, 2, 2), dtype=np.int16)
    seg_internal[1:, 1:, :, :] = 2

    with h5py.File(path, "w") as handle:
        case = handle.create_group("CaseAuto")
        case.create_dataset("mag", data=mag)
        case.create_dataset("flow", data=flow)
        case.create_dataset("Resolution", data=np.array([1.1, 1.2, 1.3], dtype=np.float32))
        case.create_dataset("RR", data=np.array(700.0, dtype=np.float32))
        case.create_dataset("Venc", data=np.array([45.0, 55.0, 65.0], dtype=np.float32))
        case.create_dataset("SpatialOrder", data=np.asarray(["FH", "RL", "PA"], dtype="S4"))
        case.create_dataset("VENCOrder", data=np.asarray(["HF", "RL", "PA"], dtype="S4"))

    save_segmentation_to_source_h5(
        str(path),
        seg_internal,
        resolution=np.array([1.2, 1.3, 1.1], dtype=np.float32),
        origin=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        provenance={"source": "auto", "created_at": "2026-01-01T00:00:00Z"},
        source_spatial_order=("FH", "RL", "PA"),
    )

    with h5py.File(path, "r") as handle:
        ds = handle["CaseAuto"]["segmask"]
        assert ds.attrs["autoflow_source"] == "auto_segmentation"
        assert tuple(x.decode() if isinstance(x, bytes) else str(x) for x in ds.attrs["SpatialOrder"]) == ("FH", "RL", "PA")

    loaded = load_h5_data(str(path))
    assert loaded.segmentation is not None
    assert np.array_equal(loaded.segmentation, seg_internal)


def test_save_segmentation_to_source_h5_targets_selected_group(tmp_path):
    path = tmp_path / "multi_case_embedded_autoseg_cache.h5"
    mag = np.arange(48, dtype=np.float32).reshape(2, 3, 4, 2) + 1.0
    flow = np.zeros((2, 3, 4, 2, 3), dtype=np.float32)
    seg_internal = np.full((3, 4, 2, 2), 3, dtype=np.int16)

    with h5py.File(path, "w") as handle:
        for case_name in ("CaseA", "CaseB"):
            case = handle.create_group(case_name)
            case.create_dataset("mag", data=mag)
            case.create_dataset("flow", data=flow)
            case.create_dataset("Resolution", data=np.array([1.1, 1.2, 1.3], dtype=np.float32))
            case.create_dataset("RR", data=np.array(700.0, dtype=np.float32))
            case.create_dataset("Venc", data=np.array([45.0, 55.0, 65.0], dtype=np.float32))
            case.create_dataset("SpatialOrder", data=np.asarray(["FH", "RL", "PA"], dtype="S4"))
            case.create_dataset("VENCOrder", data=np.asarray(["HF", "RL", "PA"], dtype="S4"))

    save_segmentation_to_source_h5(
        str(path),
        seg_internal,
        resolution=np.array([1.2, 1.3, 1.1], dtype=np.float32),
        origin=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        provenance={"source": "auto", "created_at": "2026-01-01T00:00:00Z"},
        source_spatial_order=("FH", "RL", "PA"),
        source_group="CaseB",
    )

    with h5py.File(path, "r") as handle:
        assert "segmask" not in handle["CaseA"]
        ds = handle["CaseB"]["segmask"]
        assert ds.attrs["autoflow_source"] == "auto_segmentation"
        assert ds.attrs["autoflow_source_group"] == "CaseB"

    loaded_a = load_h5_data(str(path), source_group="CaseA")
    loaded_b = load_h5_data(str(path), source_group="CaseB")
    assert loaded_a.segmentation is None
    assert loaded_b.segmentation is not None
    assert np.array_equal(loaded_b.segmentation, seg_internal)


def test_collect_input_cases_expands_multiple_h5_data_groups(tmp_path):
    path = tmp_path / "multi_group_same_leaf.h5"
    mag = np.ones((2, 2, 2, 1), dtype=np.float32)
    flow = np.zeros((2, 2, 2, 1, 3), dtype=np.float32)
    with h5py.File(path, "w") as handle:
        for group_name in ("A/Flow", "B/Flow"):
            case = handle.create_group(group_name)
            case.create_dataset("mag", data=mag)
            case.create_dataset("flow", data=flow)

    cases = collect_input_cases([str(path)])

    assert [case.source_group for case in cases] == ["A/Flow", "B/Flow"]
    assert [case.output_name for case in cases] == [
        "multi_group_same_leaf__A_Flow",
        "multi_group_same_leaf__B_Flow",
    ]


def test_collect_input_cases_only_scans_top_level_h5_files(tmp_path):
    root = tmp_path / "batch_root"
    root.mkdir()
    top_h5 = root / "ST0.h5"
    nested_dir = root / "autoflow_out" / "ST0"
    nested_dir.mkdir(parents=True)
    nested_h5 = nested_dir / "derived.h5"
    with h5py.File(top_h5, "w") as handle:
        case = handle.create_group("CaseTop")
        case.create_dataset("mag", data=np.ones((2, 2, 2, 1), dtype=np.float32))
        case.create_dataset("flow", data=np.zeros((2, 2, 2, 1, 3), dtype=np.float32))
    nested_h5.write_bytes(b"nested")

    cases = collect_input_cases([str(root)])

    assert [Path(case.input_path) for case in cases if case.input_kind == "h5"] == [top_h5]


def test_load_h5_data_supports_nested_group_complex_img_layout(tmp_path):
    path = tmp_path / "nested_complex_img.h5"
    mag = np.full((2, 2, 2, 1), 3.0, dtype=np.float32)
    flow = np.zeros((2, 2, 2, 1, 3), dtype=np.float32)
    flow[..., 0] = 12.0
    flow[..., 1] = -7.5
    flow[..., 2] = 3.25
    venc = np.array([50.0, 60.0, 70.0], dtype=np.float32)
    phase = np.pi * flow / venc.reshape((1, 1, 1, 1, 3))
    img = np.zeros((2, 2, 2, 1, 4), dtype=np.complex64)
    img[..., 0] = mag.astype(np.complex64)
    img[..., 1:4] = mag[..., None] * np.exp(1j * phase)
    with h5py.File(path, "w") as handle:
        root = handle.create_group("StudyRoot")
        case = root.create_group("SeriesA")
        case.create_dataset("Img", data=img)
        case.create_dataset("Resolution", data=np.array([1.0, 1.1, 1.2], dtype=np.float32))
        case.create_dataset("RR", data=np.array(900.0, dtype=np.float32))
        case.create_dataset("vEnC", data=venc)
        case.create_dataset("SpatialOrder", data=np.asarray(["LR", "AP", "FH"], dtype="S4"))
        case.create_dataset("VencOrder", data=np.asarray(["LR", "AP", "FH"], dtype="S4"))

    loaded = load_h5_data(str(path))

    assert loaded.source_format == "legacy_h5"
    assert loaded.source_group == "StudyRoot/SeriesA"
    assert loaded.metadata["h5_layout"] == "complex_img"
    assert loaded.mag.shape == (2, 2, 2, 1)
    assert loaded.flow.shape == (2, 2, 2, 1, 3)
    assert np.allclose(loaded.flow, flow, atol=1e-5)
    assert loaded.sigma is not None
    assert loaded.capabilities.has_complex_source is True


def test_load_h5_data_rejects_channel_first_complex_img_layout(tmp_path):
    path = tmp_path / "complex_img_channel_first.h5"
    mag = np.full((2, 3, 4, 2), 3.0, dtype=np.float32)
    flow = np.zeros((2, 3, 4, 2, 3), dtype=np.float32)
    flow[..., 0] = 12.0
    flow[..., 1] = -7.5
    flow[..., 2] = 3.25
    venc = np.array([50.0, 60.0, 70.0], dtype=np.float32)
    phase = np.pi * flow / venc.reshape((1, 1, 1, 1, 3))
    img = np.zeros((2, 3, 4, 2, 4), dtype=np.complex64)
    img[..., 0] = mag.astype(np.complex64)
    img[..., 1:4] = mag[..., None] * np.exp(1j * phase)
    img_channel_first = np.transpose(img, (4, 3, 2, 1, 0))
    with h5py.File(path, "w") as handle:
        handle.create_dataset("img", data=img_channel_first)
        handle.create_dataset("Resolution", data=np.array([1.0, 1.1, 1.2], dtype=np.float32))
        handle.create_dataset("RR", data=np.array(900.0, dtype=np.float32))
        handle.create_dataset("VENC", data=venc)
        handle.create_dataset("SpatialOrder", data=np.asarray(["LR", "AP", "FH"], dtype="S4"))
        handle.create_dataset("VENCOrder", data=np.asarray(["LR", "AP", "FH"], dtype="S4"))

    with pytest.raises(ValueError, match=r"XYZT4 or XYZT7.*\(4, 2, 4, 3, 2\)"):
        load_h5_data(str(path))


def test_s_u_y_plane_metrics_match_truth_within_two_percent_mean_error():
    for case_name in PHANTOM_CASES:
        case_dir, metrics = _run_case(case_name)
        assert (case_dir / "plane_metrics.json").is_file()
        with h5py.File(DATA_DIR / f"{case_name}.h5", "r") as handle:
            truth = handle["truth"]
            truth_flow_per_beat = float(truth["flow_ml_per_beat"][()])
            truth_peak = float(truth["peak_centerline_cm_s"][()])
            path_scales = {idx: float(val) for idx, val in enumerate(truth["path_scale_values"][()].tolist())}
            area_mm2 = float(np.pi * float(truth["tube_radius_mm"][()]) ** 2)
            path_flow_rate_ml_s = np.asarray(truth["path_flow_rate_ml_s"][()], dtype=float)

        assert len(metrics) >= len(path_scales)
        seen_paths = set()
        flow_errors = []
        peak_errors = []
        meanv_errors = []
        for plane_index, metric in enumerate(metrics):
            assert int(metric["plane_index"]) == plane_index
            path_index = int(metric["path_index"])
            seen_paths.add(path_index)
            scale = path_scales[path_index]
            expected_flow = truth_flow_per_beat * scale
            expected_peak = truth_peak * scale
            expected_meanv = _truth_mean_velocity_cm_s(path_flow_rate_ml_s[path_index], area_mm2)
            flow_errors.append(_relative_error(metric["netflow_mL_beat"], expected_flow))
            peak_errors.append(_relative_error(metric["peakv_cm_s"], expected_peak))
            meanv_errors.append(_relative_error(metric["meanv_cm_s"], expected_meanv))

        assert seen_paths == set(path_scales)
        assert _mean_relative_error(flow_errors) < REL_TOL
        assert _mean_relative_error(peak_errors) < REL_TOL
        assert _mean_relative_error(meanv_errors) < REL_TOL


def test_write_video_retries_mp4_without_gif_fallback(monkeypatch, tmp_path):
    attempts = []

    class DummyWriter:
        def __init__(self, out_path):
            self.out_path = Path(out_path)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            if exc_type is None:
                self.out_path.write_bytes(b"mp4")
            return False

        def append_data(self, frame):
            assert np.asarray(frame).shape == (2, 2, 3)

    formats = []

    def fake_get_writer(out_path, fps=24, **kwargs):
        attempts.append(kwargs.get("codec", "default"))
        formats.append(kwargs.get("format"))
        if kwargs.get("codec") == "libx264":
            raise RuntimeError("missing codec")
        return DummyWriter(out_path)

    monkeypatch.setattr("autoflow.rendering.videos.imageio.get_writer", fake_get_writer)
    frames = [np.zeros((2, 2, 3), dtype=np.uint8)]
    out_path = _write_video(frames, str(tmp_path / "planes_rotate.mp4"), fps=12)

    assert out_path.endswith(".mp4")
    assert Path(out_path).is_file()
    assert attempts[0] == "libx264"
    assert formats and all(fmt == "ffmpeg" for fmt in formats)
    assert not (tmp_path / "planes_rotate.gif").exists()


def test_nnunet_autoseg_progress_callback_reports_stage_updates(monkeypatch, tmp_path):
    model_dir = tmp_path / "model"
    fold_dir = model_dir / "fold_all"
    fold_dir.mkdir(parents=True)
    (model_dir / "dataset.json").write_text(
        json.dumps({
            "channel_names": {"0": "mag", "1": "flow_x_mean_xyz"},
            "labels": {"background": 0, "vessel": 1},
            "file_ending": ".nii.gz",
        }),
        encoding="utf-8",
    )
    (model_dir / "plans.json").write_text(json.dumps({"plans": "ok"}), encoding="utf-8")

    def fake_run_subprocess(command, *, env=None, cwd=None, runner=None,
                            progress_callback=None, output_dir=None,
                            expected_predictions=1, progress_total=5):
        assert callable(progress_callback)
        assert str(output_dir) == command[command.index("-o") + 1]
        assert expected_predictions == 1
        out_dir = Path(command[command.index("-o") + 1])
        pred = out_dir / "autoflow_case.nii.gz"
        pred.write_bytes(b"fake")
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr('autoflow.algorithms.segmentation.nnunet_static._run_subprocess', fake_run_subprocess)
    monkeypatch.setattr('autoflow.algorithms.segmentation.nnunet_static._read_nifti_segmentation', lambda path: np.ones((2, 2, 2), dtype=np.int16))

    events = []
    mag = np.ones((2, 2, 2, 3), dtype=np.float32)
    flow = np.zeros((2, 2, 2, 3, 3), dtype=np.float32)
    artifact_prefix = tmp_path / "artifacts" / "autoflow_case"
    seg, provenance = generate_nnunet_auto_segmentation(
        mag=mag,
        flow=flow,
        resolution=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        model_folder=str(model_dir),
        device="cpu",
        artifact_prefix=str(artifact_prefix),
        progress_callback=events.append,
    )

    assert seg.shape == (2, 2, 2, 3)
    assert np.array_equal(seg[..., 0], seg[..., 1])
    assert np.array_equal(seg[..., 1], seg[..., 2])
    assert provenance["device"] == "cpu"
    assert len(provenance["feature_files"]) == 2
    assert all(Path(path).is_file() for path in provenance["feature_files"])
    assert Path(provenance["segmentation_nifti"]).is_file()
    assert provenance["prediction_file"] == provenance["segmentation_nifti"]
    stages = [event["stage"] for event in events]
    assert stages[0] == "autoseg_start"
    assert "autoseg_model_ready" in stages
    assert "autoseg_prepare_inputs" in stages
    assert "autoseg_run_inference" in stages
    assert "autoseg_read_prediction" in stages
    assert stages[-1] == "autoseg_finalize"
    assert all("elapsed_sec" in event for event in events)


@pytest.mark.parametrize("hard_links", [True, False])
def test_nnunet4d_auto_profile_uses_best_checkpoint_and_keeps_frames(monkeypatch, tmp_path, hard_links):
    if not hard_links:
        def unavailable_link(*args, **kwargs):
            raise OSError("hard links unavailable")
        monkeypatch.setattr("autoflow.algorithms.segmentation.io.os.link", unavailable_link)
    model_dir = tmp_path / "temporal_model"
    fold_dir = model_dir / "fold_all"
    fold_dir.mkdir(parents=True)
    (fold_dir / "checkpoint_best.pth").write_bytes(b"checkpoint")
    (model_dir / "dataset.json").write_text(
        json.dumps({
            "channel_names": {"0": "mag_mean_xyz", "1": "tm1_mag", "2": "tp0_mag", "3": "tp1_mag"},
            "labels": {"background": 0, "vessel": 1},
            "file_ending": ".nii.gz",
        }),
        encoding="utf-8",
    )
    (model_dir / "plans.json").write_text(json.dumps({"plans": "ok"}), encoding="utf-8")

    def fake_runner(command, **_kwargs):
        import nibabel as nib
        assert "configure_exact_inference_runtime" in command[command.index("-c") + 1]
        input_dir = Path(command[command.index("-i") + 1])
        inputs = sorted(input_dir.glob("*.nii.gz"))
        assert len(inputs) == 12
        for path in inputs:
            frame = int(path.name.split("_t", 1)[1].split("_", 1)[0])
            channel = int(path.name.split("_", 3)[-1].split(".", 1)[0])
            expected = 2.0 if channel == 0 else float(((frame + channel - 2) % 3) + 1)
            assert np.all(np.asarray(nib.load(path).dataobj) == expected)
        output_dir = Path(command[command.index("-o") + 1])
        for frame in range(3):
            (output_dir / f"autoflow_case_t{frame:03d}.nii.gz").write_bytes(b"fake")
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    def fake_read(path):
        frame = int(Path(path).name.split("_t", 1)[1].split(".", 1)[0])
        return np.full((2, 2, 2), frame + 1, dtype=np.int16)

    monkeypatch.setattr("autoflow.algorithms.segmentation.nnunet_temporal._read_nifti_segmentation", fake_read)
    seg, provenance = generate_nnunet_auto_segmentation(
        mag=np.broadcast_to(np.arange(1, 4, dtype=np.float32), (2, 2, 2, 3)),
        flow=np.zeros((2, 2, 2, 3, 3), dtype=np.float32),
        resolution=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        model_folder=str(model_dir),
        backend="nnUNet4D",
        checkpoint_name="auto",
        folds="single",
        device="cpu",
        runner=fake_runner,
    )

    assert seg.shape == (2, 2, 2, 3)
    assert [int(seg[..., frame].flat[0]) for frame in range(3)] == [1, 2, 3]
    assert provenance["backend"] == "nnUNet4D"
    assert provenance["checkpoint"] == "checkpoint_best.pth"


def test_labeler_exchange_preserves_edits_and_refreshes_changed_seed(tmp_path):
    from autoflow.algorithms.segmentation import load_segmentation_file, save_segmentation_file
    from autoflow.ui.labeler_exchange import export_labeler_exchange

    mag = np.arange(48, dtype=np.float32).reshape(2, 3, 4, 2)
    flow = np.zeros((*mag.shape, 3), dtype=np.float32)
    seed = np.ones(mag.shape, dtype=np.int16)
    metadata = {"source_id": "phantom", "resolution": [1.0, 2.0, 3.0]}
    spacing, origin = (1.0, 2.0, 3.0), (4.0, 5.0, 6.0)
    first = export_labeler_exchange(tmp_path, metadata, mag, flow, seed, spacing, origin)
    assert first["written_files"] == 6
    feature_stats = [Path(path).stat().st_mtime_ns for path in first["feature_paths"]]
    edited = seed.copy()
    edited[0, 0, 0, 0] = 2
    save_segmentation_file(first["label_path"], edited, resolution=spacing, origin=origin)
    # Reopening without applying an edit retains the saved Labeler working mask.
    reopened = export_labeler_exchange(tmp_path, metadata, mag, flow, seed, spacing, origin)
    assert reopened["written_files"] == 0
    # Applying it also retains the file, including Labeler's label metadata.
    accepted = export_labeler_exchange(tmp_path, metadata, mag, flow, edited, spacing, origin)
    assert accepted["written_files"] == 0
    # A different automatic/manual seed refreshes only the mask.
    next_seed = seed.copy()
    next_seed[1, 1, 1, 1] = 3
    refreshed = export_labeler_exchange(tmp_path, metadata, mag, flow, next_seed, spacing, origin)
    assert refreshed["written_files"] == 1
    previous, _ = load_segmentation_file(refreshed["previous_label_path"], spatial_shape=seed.shape[:3], time_count=2)
    assert np.array_equal(previous, edited)
    actual, _ = load_segmentation_file(first["label_path"], spatial_shape=seed.shape[:3], time_count=2)
    assert np.array_equal(actual, next_seed)
    assert feature_stats == [Path(path).stat().st_mtime_ns for path in first["feature_paths"]]
    # A same-shape input with changed values must invalidate the image cache.
    changed_mag = mag.copy()
    changed_mag[0, 0, 0, 0] += 1
    changed = export_labeler_exchange(tmp_path, metadata, changed_mag, flow, next_seed, spacing, origin)
    assert changed["written_files"] == 5
    # Older manifests cannot identify their seed: preserve edited work before migration.
    manifest = tmp_path / "exchange.json"
    legacy_metadata = json.loads(manifest.read_text(encoding="utf-8"))
    legacy_metadata["schema_version"] = 2
    legacy_metadata.pop("image_digest")
    legacy_metadata.pop("seed_digest")
    manifest.write_text(json.dumps(legacy_metadata), encoding="utf-8")
    legacy_edit = next_seed.copy()
    legacy_edit[0, 0, 0, 0] = 4
    save_segmentation_file(first["label_path"], legacy_edit, resolution=spacing, origin=origin)
    migrated = export_labeler_exchange(tmp_path, metadata, changed_mag, flow, next_seed, spacing, origin)
    previous, _ = load_segmentation_file(migrated["previous_label_path"], spatial_shape=seed.shape[:3], time_count=2)
    assert np.array_equal(previous, legacy_edit)
    assert migrated["written_files"] == 6


def test_nnunet_runtime_keeps_cpu_and_allocation_failure_fallbacks():
    from autoflow.nnunet_runtime import _gpu_first_predict

    class AllocationError(RuntimeError):
        pass

    class Input:
        device = SimpleNamespace(type="cpu")
        shape = (3, 4, 5, 6)
        def numel(self): return 360
        def element_size(self): return 4
        def to(self, device): raise AllocationError("allocation failed")

    image = Input()
    seen = []
    cleared = []
    def original(predictor, value):
        seen.append(value)
        return "original-result"
    cuda = SimpleNamespace(mem_get_info=lambda device: (0, 0),
                           OutOfMemoryError=AllocationError, empty_cache=lambda: cleared.append(True))
    torch = SimpleNamespace(cuda=cuda)
    predictor = SimpleNamespace(device=SimpleNamespace(type="cpu"), perform_everything_on_device=True,
                                configuration_manager=SimpleNamespace(patch_size=[8, 8, 8]))
    assert _gpu_first_predict(original, predictor, image, torch) == "original-result"
    predictor.device = SimpleNamespace(type="cuda")
    assert _gpu_first_predict(original, predictor, image, torch) == "original-result"
    cuda.mem_get_info = lambda device: (100 * 1024**3, 100 * 1024**3)
    assert _gpu_first_predict(original, predictor, image, torch) == "original-result"
    assert seen == [image, image, image]
    assert cleared == [True]


def test_nnunet_autoseg_prefers_gpu_resampling_and_falls_back_to_cpu(monkeypatch, tmp_path):
    model_dir = tmp_path / "model"
    fold_dir = model_dir / "fold_all"
    fold_dir.mkdir(parents=True)
    (fold_dir / "checkpoint_final.pth").write_bytes(b"checkpoint")
    (model_dir / "dataset.json").write_text(
        json.dumps({
            "channel_names": {"0": "mag"},
            "labels": {"background": 0, "vessel": 1},
            "file_ending": ".nii.gz",
        }),
        encoding="utf-8",
    )
    (model_dir / "plans.json").write_text(
        json.dumps({
            "configurations": {
                "3d_fullres": {
                    "resampling_fn_data": "resample_data_or_seg_to_shape",
                    "resampling_fn_data_kwargs": {
                        "is_seg": False,
                        "order": 3,
                        "force_separate_z": None,
                    },
                },
            },
        }),
        encoding="utf-8",
    )

    calls = []

    def fake_runner(command, **_kwargs):
        calls.append(list(command))
        model_path = Path(command[command.index("-m") + 1])
        output_path = Path(command[command.index("-o") + 1])
        if model_path != model_dir:
            gpu_plans = json.loads((model_path / "plans.json").read_text(encoding="utf-8"))
            gpu_config = gpu_plans["configurations"]["3d_fullres"]
            assert gpu_config["resampling_fn_data"] == "resample_torch_fornnunet"
            assert gpu_config["resampling_fn_data_kwargs"]["device"] == "cuda"
            return SimpleNamespace(returncode=1, stdout="", stderr="CUDA out of memory")
        (output_path / "autoflow_case.nii.gz").write_bytes(b"fake")
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(
        "autoflow.algorithms.segmentation.nnunet_static._read_nifti_segmentation",
        lambda _path: np.ones((2, 2, 2), dtype=np.int16),
    )
    events = []
    seg, provenance = generate_nnunet_auto_segmentation(
        mag=np.ones((2, 2, 2, 2), dtype=np.float32),
        flow=np.zeros((2, 2, 2, 2, 3), dtype=np.float32),
        resolution=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        model_folder=str(model_dir),
        device="cuda",
        runner=fake_runner,
        progress_callback=events.append,
    )

    assert seg.shape == (2, 2, 2, 2)
    assert len(calls) == 2
    assert calls[0][calls[0].index("-m") + 1] != str(model_dir)
    assert calls[1][calls[1].index("-m") + 1] == str(model_dir)
    assert calls[1][calls[1].index("-nps") + 1] == "1"
    assert provenance["preprocessing_device_requested"] == "cuda"
    assert provenance["preprocessing_device"] == "cpu"
    assert provenance["gpu_preprocessing_fallback"] is True
    assert "CUDA out of memory" in provenance["gpu_preprocessing_fallback_reason"]
    assert "autoseg_gpu_preprocessing_fallback" in [event["stage"] for event in events]

def test_pwv_timing_methods_on_synthetic_waveforms():
    rr_ms = 1000.0
    x = np.linspace(0.0, 1.0, 20, endpoint=False)
    proximal = np.exp(-0.5 * ((x - 0.25) / 0.08) ** 2)
    distal = np.roll(proximal, 2)

    foot_prox, meta_prox = detect_waveform_foot_time_ms(
        proximal, rr_ms, method="tangent", allow_cycle_wrap=True
    )
    foot_dist, meta_dist = detect_waveform_foot_time_ms(
        distal, rr_ms, method="threshold", threshold_percent=10.0, allow_cycle_wrap=True
    )
    assert foot_prox is not None
    assert foot_dist is not None
    assert meta_prox.get("method") == "tangent"
    assert meta_dist.get("method") == "threshold"

    delay_ms, cc_meta = compute_cross_correlation_delay_ms(
        proximal, distal, rr_ms, window="full", allow_cycle_wrap=True
    )
    assert delay_ms is not None
    assert abs(float(cc_meta.get("lag_samples")) - 2.0) < 0.15
    assert abs(float(delay_ms) - 100.0) < 10.0
    assert int(cc_meta.get("interp_factor")) == 10

    delay_upstroke_ms, up_meta = compute_cross_correlation_delay_ms(
        proximal, distal, rr_ms, window="upstroke", allow_cycle_wrap=True
    )
    assert delay_upstroke_ms is not None
    assert abs(float(delay_upstroke_ms) - 100.0) < 10.0
    assert abs(float(up_meta.get("offset_lag_samples")) - 2.0) < 0.15


def test_pwv_wrapped_tangent_and_subframe_xcorr():
    rr_ms = 1000.0
    waveform = np.array([
        -15.166193483117201,
        418.84413044447297,
        435.4283063098984,
        354.5884270552724,
        288.8676802814978,
        137.02311573973498,
        60.88961149981558,
        16.682423130255852,
        2.4873332685670873,
        25.087174459539828,
        24.03118575795991,
        14.845060197501105,
        24.57764249657329,
        14.24500673573191,
        2.2664627477484784,
        -0.5951218100742958,
        3.0340050909445813,
        16.08187168750316,
        8.482758977647443,
        -8.544392118097038,
    ], dtype=float)
    foot_ms, foot_meta = detect_waveform_foot_time_ms(
        waveform, rr_ms, method="tangent", allow_cycle_wrap=True
    )
    assert foot_ms is not None
    assert foot_meta.get("cycle_wrapped") is True
    assert int(foot_meta.get("slope_index_wrapped")) >= waveform.size
    assert int(foot_meta.get("baseline_index")) == waveform.size - 1
    assert 900.0 < float(foot_ms) < 1000.0

    x = np.linspace(0.0, 1.0, 20, endpoint=False)
    proximal = np.exp(-0.5 * ((x - 0.25) / 0.08) ** 2)
    shifted = np.exp(-0.5 * ((((x - 0.25 - 0.075) + 0.5) % 1.0) - 0.5) ** 2 / (0.08 ** 2))
    delay_ms, cc_meta = compute_cross_correlation_delay_ms(
        proximal, shifted, rr_ms, window="full", allow_cycle_wrap=True, interp_factor=20
    )
    assert delay_ms is not None
    assert abs(float(delay_ms) - 75.0) < 15.0
    assert abs(float(cc_meta.get("lag_samples")) - 1.5) < 0.25
    assert int(cc_meta.get("interp_factor")) == 20


def test_plane_video_path_color_uses_group_scene_color():
    skeleton_params = SkeletonParams(
        label_groups={
            "aorta": {"path_color": "#f76707"},
            "pulmonary": {"path_color": "#4dabf7"},
        }
    )

    class _Ws:
        pass

    ws = _Ws()
    ws.skeleton_params = skeleton_params
    ws.path_info = [
        {"group_name": "aorta"},
        {"group_name": "pulmonary"},
        {},
    ]
    ws.group_order = ["aorta", "pulmonary"]
    ws.multilabel_groups = {
        "aorta": {"path_index_offset": 0, "centerline_paths_smooth": [np.zeros((2, 3))]},
        "pulmonary": {"path_index_offset": 1, "centerline_paths_smooth": [np.zeros((2, 3))]},
    }

    assert _path_group_name(ws, 0) == "aorta"
    assert _path_group_name(ws, 1) == "pulmonary"
    assert _path_group_name(ws, 2) == ""
    assert _path_color(ws, 0) == "#f76707"
    assert _path_color(ws, 1) == "#4dabf7"
    assert _path_color(ws, 2) == "deepskyblue"


def test_config_dir_reads_feature_render_settings_from_metric_jsons(tmp_path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    video_exporting_payload = {
        "window_size": [1111, 777],
        "rotate_dynamic_video": False,
        "plane_video": {
            "default": {
                "plane_color": "#654321",
                "plane_opacity": 0.4,
            },
            "label": {
                "prefix": "plane=",
                "font_size": 17,
                "text_color": "white",
            },
        },
    }
    planes_payload = {
        "render": {
            "default": {
                "plane_color": "#123456",
                "plane_opacity": 0.25,
            }
        }
    }
    wss_payload = {
        "render": {
            "clim": [1.0, 9.0],
            "show_scalar_bar": False,
            "bar_cfg": {"width": 0.11},
        }
    }
    tke_payload = {
        "render": {
            "clim": [2.0, 22.0],
        }
    }
    pressure_gradient_payload = {
        "render": {
            "clim": [3.0, 33.0],
            "show_scalar_bar": False,
        }
    }
    streamlines_payload = {
        "seed_ratio": 0.03,
        "max_steps": 333,
        "min_seeds": 12,
        "terminal_speed": 0.25,
        "rng_seed": 7,
        "tube_radius": 0.4,
        "render": {
            "clim": [4.0, 44.0],
            "show_scalar_bar": False,
            "bar_cfg": {"position_x": 0.66},
        }
    }
    pathlines_payload = {
        "seed_ratio": 0.12,
        "max_steps": 123,
        "min_seeds": 11,
        "seed_mode": "ratio",
        "seed_count": 123,
        "terminal_speed": 0.002,
        "rng_seed": 17,
        "tube_radius": 0.6,
        "color_mode": "per_group",
        "temporal_cache_mb": 321.0,
    }
    (config_dir / "video_exporting.json").write_text(json.dumps(video_exporting_payload), encoding="utf-8")
    (config_dir / "planes.json").write_text(json.dumps(planes_payload), encoding="utf-8")
    (config_dir / "wss.json").write_text(json.dumps(wss_payload), encoding="utf-8")
    (config_dir / "tke.json").write_text(json.dumps(tke_payload), encoding="utf-8")
    (config_dir / "pressure_gradient.json").write_text(json.dumps(pressure_gradient_payload), encoding="utf-8")
    (config_dir / "streamlines.json").write_text(json.dumps(streamlines_payload), encoding="utf-8")
    (config_dir / "pathlines.json").write_text(json.dumps(pathlines_payload), encoding="utf-8")

    cfg = AutoFlowConfig.from_config_dir(str(config_dir))
    resolved = bundle_to_autoflow_kwargs({
        "video_exporting": video_exporting_payload,
        "planes": planes_payload,
        "wss": wss_payload,
        "tke": tke_payload,
        "pressure_gradient": pressure_gradient_payload,
        "streamlines": streamlines_payload,
        "pathlines": pathlines_payload,
    })

    assert cfg.window_size == (1111, 777)
    assert cfg.rotate_dynamic_video is False
    assert cfg.plane_video_cfg["default"]["plane_color"] == "#654321"
    assert cfg.plane_video_cfg["default"]["plane_opacity"] == 0.4
    assert cfg.plane_video_cfg["label"]["prefix"] == "plane="
    assert cfg.plane_video_cfg["label"]["font_size"] == 17
    assert cfg.plane_video_cfg["label"]["text_color"] == "white"
    assert cfg.wss_clim == (1.0, 9.0)
    assert cfg.wss_show_scalar_bar is False
    assert cfg.wss_bar_cfg["width"] == 0.11
    assert cfg.tke_clim == (2.0, 22.0)
    assert cfg.pressure_gradient_clim == (3.0, 33.0)
    assert cfg.pressure_gradient_show_scalar_bar is False
    assert cfg.streamline_clim == (4.0, 44.0)
    assert cfg.streamline_show_scalar_bar is False
    assert cfg.streamline_bar_cfg["position_x"] == 0.66
    assert resolved["wss_clim"] == (1.0, 9.0)
    assert resolved["streamline_clim"] == (4.0, 44.0)
    assert cfg.tube_radius == pytest.approx(0.4)
    assert cfg.seed_ratio == pytest.approx(0.03)
    assert cfg.pathline_seed_mode == "ratio"
    assert cfg.pathline_max_seeds == 123
    assert cfg.pathline_seed_ratio == pytest.approx(0.12)
    assert cfg.pathline_max_steps == 123
    assert cfg.pathline_min_seeds == 11
    assert cfg.pathline_terminal_speed == pytest.approx(0.002)
    assert cfg.pathline_rng_seed == 17
    assert cfg.pathline_tube_radius == pytest.approx(0.6)
    assert cfg.pathline_color_mode == "per_group"
    assert cfg.pathline_temporal_cache_mb == pytest.approx(321.0)
    assert resolved["tube_radius"] == pytest.approx(0.4)
    assert resolved["pathline_seed_mode"] == "ratio"
    assert resolved["pathline_max_seeds"] == 123
    assert resolved["pathline_seed_ratio"] == pytest.approx(0.12)
    assert resolved["pathline_max_steps"] == 123
    assert resolved["pathline_min_seeds"] == 11
    assert resolved["pathline_terminal_speed"] == pytest.approx(0.002)
    assert resolved["pathline_rng_seed"] == 17
    assert resolved["pathline_tube_radius"] == pytest.approx(0.6)
    assert resolved["pathline_color_mode"] == "per_group"
    assert resolved["pathline_temporal_cache_mb"] == pytest.approx(321.0)
    assert resolved["plane_video_cfg"]["label"]["prefix"] == "plane="

    workspace = build_workspace(cfg)
    assert workspace.streamline_params.seed_ratio == pytest.approx(0.03)
    assert workspace.streamline_params.max_steps == 333
    assert workspace.streamline_params.pathline_seed_ratio == pytest.approx(0.12)
    assert workspace.streamline_params.pathline_max_steps == 123
    assert workspace.streamline_params.pathline_min_seeds == 11
    assert workspace.streamline_params.pathline_terminal_speed == pytest.approx(0.002)
    assert workspace.streamline_params.pathline_rng_seed == 17
    assert workspace.streamline_params.pathline_tube_radius == pytest.approx(0.6)

    auto_resolved = bundle_to_autoflow_kwargs({"streamlines": {"render": {"clim": None}}})
    assert auto_resolved["streamline_clim"] is None
    assert AutoFlowConfig().streamline_clim is None
    assert AutoFlowConfig().seed_ratio == pytest.approx(0.1)
    assert AutoFlowConfig().max_steps == 200
    assert AutoFlowConfig().tube_radius == pytest.approx(0.05)


def test_hybrid_component_filter_keeps_multiple_group_components():
    mask = np.zeros((16, 16, 16), dtype=bool)
    mask[1:5, 1:5, 1:5] = True
    mask[8:12, 8:12, 8:11] = True
    mask[14:15, 14:15, 14:15] = True

    filtered = filter_connected_components(
        mask,
        np.ones(3, dtype=float),
        mode="hybrid",
        min_volume_mm3=10.0,
        rel_min_ratio=0.01,
    )

    assert int(np.sum(filtered)) == (4 * 4 * 4) + (4 * 4 * 3)
    assert bool(filtered[2, 2, 2]) is True
    assert bool(filtered[9, 9, 9]) is True
    assert bool(filtered[14, 14, 14]) is False


def test_pipeline_group_preprocess_keeps_multiple_components_under_hybrid_filter():
    ws = SimpleNamespace(
        segmask_raw=np.zeros((16, 16, 16), dtype=np.int16),
        multilabel_groups={},
        resolution=np.ones(3, dtype=float),
        skeleton_params=SkeletonParams(
            remove_small_cc=True,
            min_cc_volume_mm3=10.0,
            cc_filter_mode="hybrid",
            cc_rel_min_ratio=0.01,
            label_groups={"vessel": {"labels": [1]}},
            single_label_group_name="vessel",
        ),
        time_count=lambda: 1,
        set_object_visible_by_data_key=lambda *args, **kwargs: None,
        remove_object_by_data_key=lambda *args, **kwargs: None,
        remove_objects_by_prefix=lambda *args, **kwargs: None,
        add_object=lambda *args, **kwargs: None,
    )
    ws.segmask_raw[1:5, 1:5, 1:5] = 1
    ws.segmask_raw[8:12, 8:12, 8:11] = 1
    ws.segmask_raw[14:15, 14:15, 14:15] = 1

    engine = PipelineEngine()
    assert engine.preprocess(ws) is True
    first_binary = ws.segmask_binary
    assert engine.preprocess(ws) is False
    assert ws.segmask_binary is first_binary

    group_state = ws.multilabel_groups["vessel"]
    assert ws.group_order == ["vessel"]
    assert int(np.sum(group_state["clean_mask_3d"])) == (4 * 4 * 4) + (4 * 4 * 3)
    assert bool(group_state["clean_mask_3d"][2, 2, 2]) is True
    assert bool(group_state["clean_mask_3d"][9, 9, 9]) is True
    assert bool(group_state["clean_mask_3d"][14, 14, 14]) is False


def test_plane_metric_step_does_not_eagerly_compute_derived_metrics(monkeypatch, tmp_path):
    ws = Workspace()
    ws.flow_raw = np.zeros((2, 2, 2, 1, 3), dtype=np.float32)
    ws.segmask_raw = np.ones((2, 2, 2, 1), dtype=np.int16)
    ws.planes = [
        PlaneData(
            center=np.zeros(3, dtype=float),
            normal=np.array([1.0, 0.0, 0.0], dtype=float),
        )
    ]
    ws.paths.output_dir = str(tmp_path)
    engine = PipelineEngine()
    captured = {}

    def fake_compute(_workspace, **kwargs):
        captured.update(kwargs)
        return [], {}, "Plane metrics: 0"

    monkeypatch.setattr(engine, "_compute_plane_metrics_internal", fake_compute)
    monkeypatch.setattr(engine, "_save_planes_json", lambda _workspace: str(tmp_path / "planes.json"))

    result = engine.run_step(ws, StepId.COMPUTE_PLANE_METRICS, lambda _message: None)

    assert result.success is True
    assert captured["include_derived"] is False
    assert captured["ensure_derived"] is False


def test_plane_video_label_style_uses_plane_render_config(monkeypatch, tmp_path):
    class DummyPlotter:
        def __init__(self):
            self.label_calls = []
            self.camera_position = None

        def add_mesh(self, *args, **kwargs):
            return None

        def add_point_labels(self, points, labels, **kwargs):
            self.label_calls.append((np.asarray(points), list(labels), dict(kwargs)))
            return None

        def add_text(self, *args, **kwargs):
            return None

        def render(self):
            return None

        def screenshot(self, return_img=True):
            assert return_img is True
            return np.zeros((4, 4, 3), dtype=np.uint8)

        def remove_actor(self, name):
            return None

        def close(self):
            return None

    plotter = DummyPlotter()

    class DummyPoly:
        n_points = 1
        bounds = (0.0, 1.0, 0.0, 1.0, 0.0, 1.0)

    ws = SimpleNamespace(
        origin=np.zeros(3, dtype=float),
        centerline_paths_smooth=[np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)],
        planes=[SimpleNamespace(center=np.array([0.5, 0.0, 0.0], dtype=float), normal=np.array([1.0, 0.0, 0.0], dtype=float), path_index=0, group_name="")],
        group_order=[],
        multilabel_groups={},
        skeleton_points=np.empty((0, 3), dtype=float),
        skeleton_params=SkeletonParams(),
    )

    monkeypatch.setattr("autoflow.rendering.videos._build_union_surface", lambda ws, smoothing_iteration=200: (None, DummyPoly()))
    monkeypatch.setattr("autoflow.rendering.videos._make_plotter", lambda window_size=None: plotter)
    monkeypatch.setattr("autoflow.rendering.videos._plane_mesh", lambda center_world, normal, size: SimpleNamespace(n_points=4))
    monkeypatch.setattr("autoflow.rendering.videos._path_polydata", lambda path_world: SimpleNamespace(n_points=len(path_world)))
    monkeypatch.setattr("autoflow.rendering.videos._orbit_camera", lambda poly, azimuth_deg, elevation_deg=0.0, distance_scale=1.0: [tuple([0.0, 0.0, 0.0]), tuple([0.0, 0.0, 0.0]), (0.0, 0.0, 1.0)])
    def consume_video(frames, out_path, fps=24):
        assert len(list(frames() if callable(frames) else frames)) == 1
        return str(out_path)
    monkeypatch.setattr("autoflow.rendering.videos._write_video", consume_video)

    out = render_plane_rotation_video(
        ws,
        str(tmp_path),
        fps=12,
        n_frames=1,
        add_plane_idx=True,
        plane_video_cfg={
            "label": {
                "prefix": "P-",
                "font_size": 19,
                "text_color": "white",
                "shape_color": "#123456",
                "shape_opacity": 0.4,
            }
        },
    )

    assert out.endswith("planes_rotate.mp4")
    assert len(plotter.label_calls) == 1
    _points, labels, kwargs = plotter.label_calls[0]
    assert labels == ["P-0"]
    assert kwargs["font_size"] == 19
    assert kwargs["text_color"] == "white"
    assert kwargs["shape_color"] == "#123456"
    assert kwargs["shape_opacity"] == 0.4


def test_build_union_surface_falls_back_when_extract_surface_algorithm_kwarg_is_unsupported(monkeypatch):
    class DummySurface:
        n_points = 1

        def smooth(self, n_iter=0):
            return self

    class DummyMesh:
        n_cells = 1

        def __init__(self):
            self.calls = []

        def threshold(self, value):
            assert value == 0.1
            return self

        def extract_surface(self, **kwargs):
            self.calls.append(dict(kwargs))
            if "algorithm" in kwargs:
                raise TypeError("extract_surface() got an unexpected keyword argument algorithm")
            return DummySurface()

    dummy_mesh = DummyMesh()
    monkeypatch.setattr("autoflow.rendering.videos.create_uniform_grid", lambda mask3d, resolution, origin=None: dummy_mesh)

    ws = SimpleNamespace(
        segmask_binary=None,
        segmask_3d=np.ones((2, 2, 2), dtype=bool),
        resolution=np.ones(3, dtype=float),
        origin=np.zeros(3, dtype=float),
    )

    mesh, surf = _build_union_surface(ws, smoothing_iteration=0)

    assert mesh is dummy_mesh
    assert surf is not None
    assert dummy_mesh.calls == [{"algorithm": "dataset_surface"}, {}]


def test_build_plane_records_include_label_name_from_label_map():
    ws = SimpleNamespace(
        origin=np.zeros(3, dtype=float),
        resolution=np.ones(3, dtype=float),
        segmask_raw=np.array(
            [
                [[1, 2], [1, 2]],
                [[1, 2], [1, 2]],
            ],
            dtype=np.int16,
        ),
        planes=[
            SimpleNamespace(
                center=np.array([0.0, 0.0, 0.0], dtype=float),
                normal=np.array([1.0, 0.0, 0.0], dtype=float),
                label=1,
                path_index=0,
                distance=0.0,
                group_name="aorta_systemic_branches",
                metrics={},
            ),
            SimpleNamespace(
                center=np.array([0.0, 0.0, 1.0], dtype=float),
                normal=np.array([1.0, 0.0, 0.0], dtype=float),
                label=2,
                path_index=1,
                distance=5.0,
                group_name="pulmonary_arteries",
                metrics={},
            ),
        ],
        path_info=[],
        multilabel_groups={
            "aorta_systemic_branches": {"labels": [1]},
            "pulmonary_arteries": {"labels": [2, 8, 9]},
        },
        segmentation=SimpleNamespace(label_names={"1": "Ascending Aorta", "2": "Main Pulmonary Artery"}),
        label_params=SimpleNamespace(label_map={"AAO": 1, "MPA": 2}),
        skeleton_params=SimpleNamespace(label_map={"AAO": 1, "MPA": 2}),
    )

    payload = build_plane_records(ws)

    assert payload[0]["label_name"] == "AAO"
    assert payload[1]["label_name"] == "MPA"


def test_manual_plane_records_reload_without_path_projection():
    from autoflow.core.models import Workspace

    workspace = Workspace()
    workspace.origin = np.array([10.0, 20.0, 30.0], dtype=float)
    workspace.centerline_paths_smooth = [
        np.array([[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]], dtype=float)
    ]
    workspace.planes = [
        PlaneData(
            center=np.array([4.0, 7.0, 9.0], dtype=float),
            normal=np.array([0.0, 1.0, 0.0], dtype=float),
            label=0,
            path_index=-1,
            group_name="manual",
        )
    ]

    records = build_plane_records(workspace)
    restored = project_planes_to_workspace(records, workspace)

    assert records[0]["placement_mode"] == "manual"
    assert len(restored) == 1
    assert restored[0].path_index == -1
    assert restored[0].group_name == "manual"
    assert np.allclose(restored[0].center, workspace.planes[0].center)
    assert np.allclose(restored[0].normal, workspace.planes[0].normal)


def test_plane_coordinate_file_supports_world_local_and_relative_path_mapping(tmp_path):
    source = Workspace()
    source.origin = np.array([10.0, 20.0, 30.0], dtype=float)
    source.resolution = np.array([1.0, 1.0, 1.0], dtype=float)
    source.centerline_paths_smooth = [
        np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=float)
    ]
    source.path_info = [{"group_name": "aorta"}]
    source.multilabel_groups = {
        "aorta": {
            "path_index_offset": 0,
            "centerline_paths_smooth": source.centerline_paths_smooth,
        }
    }
    source.planes = [
        PlaneData(
            center=np.array([5.0, 0.0, 0.0], dtype=float),
            normal=np.array([1.0, 0.0, 0.0], dtype=float),
            path_index=0,
            distance=5.0,
            group_name="aorta",
        )
    ]
    coordinate_path = tmp_path / "planes.json"
    save_plane_positions(source, coordinate_path, source_path="source.h5")
    payload = load_plane_position_payload(coordinate_path)

    assert payload["schema"] == PLANE_POSITION_SCHEMA
    assert payload["coordinate_system"] == "autoflow_canonical_world_mm"
    assert payload["planes"][0]["path_fraction"] == pytest.approx(0.5)

    target = Workspace()
    target.origin = np.array([100.0, 200.0, 300.0], dtype=float)
    target.centerline_paths_smooth = [
        np.array([[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]], dtype=float)
    ]
    target.path_info = [{"group_name": "aorta"}]
    target.multilabel_groups = {
        "aorta": {
            "path_index_offset": 0,
            "centerline_paths_smooth": target.centerline_paths_smooth,
        }
    }

    world_planes = project_planes_to_workspace(payload, target, mapping_mode="world")
    local_planes = project_planes_to_workspace(payload, target, mapping_mode="local")
    relative_planes, report = project_planes_to_workspace(
        payload,
        target,
        mapping_mode="path_relative",
        return_report=True,
    )

    assert np.allclose(world_planes[0].center, [-85.0, -180.0, -270.0])
    assert np.allclose(local_planes[0].center, [5.0, 0.0, 0.0])
    assert np.allclose(relative_planes[0].center, [10.0, 0.0, 0.0])
    assert relative_planes[0].distance == pytest.approx(10.0)
    assert report["mapping_mode"] == "path_relative"
    assert report["planes"][0]["mapping_source"] == "group_path_index"


def test_quality_report_collects_actionable_pipeline_checks(tmp_path):
    workspace = Workspace()
    workspace.data_loaded = True
    workspace.paths.flow_path = "case.h5"
    workspace.resolution = np.ones(3, dtype=float)
    workspace.venc = np.full(3, 150.0, dtype=float)
    workspace.flow_raw = np.ones((8, 8, 8, 2, 3), dtype=np.float32)
    workspace.mag_raw = np.ones((8, 8, 8, 2), dtype=np.float32)
    workspace.segmask_raw = np.ones((8, 8, 8, 2), dtype=np.int16)
    workspace.segmask_labels = workspace.segmask_raw.copy()
    workspace.segmask_labels_3d = workspace.segmask_raw[..., 0].copy()
    workspace.segmask_binary = workspace.segmask_raw.astype(bool)
    workspace.segmask_3d = workspace.segmask_labels_3d.astype(bool)
    workspace.graph.points = np.array([[1.0, 4.0, 4.0], [6.0, 4.0, 4.0]], dtype=float)
    workspace.graph.edges = np.array([[0, 1]], dtype=int)
    workspace.centerline_paths_smooth = [workspace.graph.points.copy()]
    workspace.planes = [
        PlaneData(
            center=np.array([3.0, 4.0, 4.0], dtype=float),
            normal=np.array([1.0, 0.0, 0.0], dtype=float),
            path_index=0,
            distance=2.0,
        )
    ]
    workspace.derived.plane_metrics = [{"area_mm2": [16.0, 16.0], "path_index": 0}]
    workspace.derived.plane_qc = {"path_ic": {"0": 1.0}, "fork_ic": {}, "forks": []}

    report = build_quality_report(workspace)
    report_path = tmp_path / "quality_report.json"
    save_quality_report(workspace, report_path, report=report)

    assert report["schema"] == QUALITY_REPORT_SCHEMA
    assert report["overall_status"] == "ready"
    assert report["status_counts"]["fail"] == 0
    assert {item["id"] for item in report["checks"]} >= {
        "input.venc_saturation",
        "segmentation.availability",
        "centerline.topology",
        "planes.geometry",
        "hemodynamics.flow_consistency",
    }
    assert json.loads(report_path.read_text(encoding="utf-8"))["overall_status"] == "ready"


def test_quality_report_names_flow_hierarchy_and_label_equations():
    from autoflow.algorithms.metrics import apply_internal_consistency_to_metrics

    workspace = Workspace()
    workspace.skeleton_params.label_map = {"PV": 14, "SMV": 15, "SV": 16}
    workspace.label_params.label_map = dict(workspace.skeleton_params.label_map)
    workspace.path_info = [
        {"path_index": path_index, "group_name": "portal_splenic_venous"}
        for path_index in range(5)
    ]
    workspace.forks = [
        {"left": [0, 1], "right": [2]},
        {"left": [2], "right": [3, 4]},
    ]
    labels = [15, 16, 14, 14, 14]
    flows = [6.0, 4.0, 10.0, 6.0, 4.0]
    metrics = []
    for path_index, (label_value, flow_value) in enumerate(zip(labels, flows)):
        for distance, scale in ((0.0, 0.9), (5.0, 1.1)):
            metrics.append({
                "plane_index": len(metrics),
                "path_index": path_index,
                "segmentation_label": label_value,
                "distance": distance,
                "netflow_mL_beat": flow_value * scale,
                "flowrate_mL_s": [flow_value * scale, flow_value * scale * 2.0],
                "meanv_cm_s": flow_value,
                "peakv_cm_s": flow_value * 2.0,
                "area_mm2": [10.0, 11.0],
            })
    metrics, qc = apply_internal_consistency_to_metrics(
        metrics,
        path_info=workspace.path_info,
        forks=workspace.forks,
    )
    workspace.derived.plane_metrics = metrics
    workspace.derived.plane_qc = qc

    hierarchy = build_quality_report(workspace)["flow_hierarchy"]
    roots = {item["name"]: item for item in hierarchy["roots"]}

    assert hierarchy["schema"] == "autoflow.flow_hierarchy.v1"
    assert set(roots) == {"PV", "SMV", "SV"}
    assert [item["name"] for item in roots["PV"]["children"]] == ["PV1", "PV2"]
    assert [item["sequence"] for item in roots["PV"]["planes"]] == [1, 2]
    assert roots["PV"]["statistics"]["net_flow_mL_beat"]["mean"] == pytest.approx(10.0)
    assert roots["PV"]["statistics"]["net_flow_mL_beat"]["std"] == pytest.approx(1.0)
    assert {item["equation"] for item in hierarchy["junctions"]} == {
        "SMV + SV = PV",
        "PV = PV1 + PV2",
    }


def test_pathline_step_uses_per_plane_colors():
    ws = Workspace()
    ws.segmask_raw = np.ones((16, 16, 16, 2), dtype=np.int16)
    ws.flow_raw = np.zeros((16, 16, 16, 2, 3), dtype=np.float32)
    ws.flow_raw[..., 0] = 10.0
    ws.planes = [
        PlaneData(center=np.array([4.0, 8.0, 8.0], dtype=float), normal=np.array([1.0, 0.0, 0.0], dtype=float)),
        PlaneData(center=np.array([10.0, 8.0, 8.0], dtype=float), normal=np.array([1.0, 0.0, 0.0], dtype=float)),
    ]
    ws.streamline_params.pathline_color = "deepskyblue"
    ws.pathline_colors = {0: "lime", 1: "#ff8800"}

    engine = PipelineEngine()
    engine.preprocess(ws)
    result = engine._step_plane_streamlines(ws)

    assert result.success and not result.skipped
    colors = {
        obj.data_key: obj.color
        for obj in ws.scene_objects.values()
        if obj.data_key.startswith("pathline_")
    }
    assert colors == {
        "pathline_0": "lime",
        "pathline_1": "#ff8800",
    }


def test_vtk_particle_tracer_pathline_returns_polyline():
    flow = np.zeros((20, 4, 4, 3, 3), dtype=np.float32)
    flow[..., 0] = 0.2
    mask = np.ones((20, 4, 4, 3), dtype=bool)
    mesh = generate_pathlines_from_plane_at_t(
        flow,
        1,
        SimpleNamespace(),
        np.ones(3),
        np.zeros(3),
        mask_4d=mask,
        mask_3d=mask[..., 1],
        terminal_speed=0.001,
        rr=1000.0,
        seeds=np.asarray([[2.0, 1.5, 1.5]], dtype=float),
    )
    assert mesh is not None
    assert mesh.n_lines == 1
    assert mesh.n_points >= 2
    assert "Velocity" in mesh.point_data
    assert "PathlinePhase" in mesh.point_data
    prefix = pathline_prefix_at_phase(mesh, 0.5)
    assert prefix is not None
    assert 0 < prefix.n_points <= mesh.n_points
    assert prefix.n_lines == 1


def test_pathline_rejects_invalid_plane_before_vtk_execution():
    flow = np.zeros((4, 4, 4, 2, 3), dtype=np.float32)
    mask = np.ones((4, 4, 4, 2), dtype=bool)
    invalid_plane = SimpleNamespace(
        center=np.array([2.0, 2.0, 2.0]),
        normal=np.zeros(3),
    )

    with pytest.raises(ValueError, match="non-zero normal"):
        generate_pathlines_from_plane_at_t(
            flow,
            0,
            invalid_plane,
            np.ones(3),
            np.zeros(3),
            mask_4d=mask,
            mask_3d=mask[..., 0],
            rr=1000.0,
        )


def test_gui_run_all_scopes_steps_to_the_active_workflow_panel():
    from autoflow.ui.app import _workflow_run_all_steps

    assert _workflow_run_all_steps("correction") == [
        StepId.BACKGROUND_CORRECTION, StepId.REMOVE_NOISE, StepId.UNWRAP_PHASE, StepId.GENERATE_PCMRA,
    ]
    assert _workflow_run_all_steps("centerline") == [
        StepId.GENERATE_SKELETON,
        StepId.GENERATE_GRAPH,
        StepId.GENERATE_PLANES,
    ]
    assert _workflow_run_all_steps("hemodynamics") == [
        StepId.COMPUTE_PLANE_METRICS,
        StepId.COMPUTE_DERIVED_METRICS,
    ]
    assert _workflow_run_all_steps("review") == []


def test_scene_pathlines_accumulate_requested_plane_indices(monkeypatch):
    from autoflow.ui.viewer import SceneController

    workspace = Workspace()
    workspace.flow_raw = np.zeros((4, 4, 4, 2, 3), dtype=np.float32)
    workspace.segmask_3d = np.ones((4, 4, 4), dtype=bool)
    workspace.planes = [
        PlaneData(center=np.array([float(index), 1.0, 1.0]), normal=np.array([1.0, 0.0, 0.0]))
        for index in range(3)
    ]
    controller = SceneController(SimpleNamespace(), workspace, lambda _message: None)
    monkeypatch.setattr(controller, "sync_from_workspace", lambda: None)
    monkeypatch.setattr(controller, "invalidate_cache", lambda _prefix=None: None)

    controller.trigger_pathlines([0, 2])

    assert workspace.active_pathline_plane_indices == [0, 2]
    assert sorted(
        obj.data_key
        for obj in workspace.scene_objects.values()
        if obj.data_key.startswith("pathline_")
    ) == ["pathline_0", "pathline_2"]
    assert all(
        obj.dynamic
        for obj in workspace.scene_objects.values()
        if obj.data_key.startswith("pathline_")
    )

    cached = SimpleNamespace(name="already-calculated")
    workspace.pathline_cache[0] = {0: cached}
    workspace.pathline_seed_cache[0] = np.array([[0.0, 1.0, 1.0]])
    controller.trigger_pathlines([1])
    controller.trigger_pathlines([0])

    assert workspace.active_pathline_plane_indices == [0, 1, 2]
    assert workspace.pathline_cache[0][0] is cached
    assert controller._get_pathline_mesh(0, 1) is cached
    assert np.array_equal(workspace.pathline_seed_cache[0], [[0.0, 1.0, 1.0]])
    assert sorted(
        obj.data_key
        for obj in workspace.scene_objects.values()
        if obj.data_key.startswith("pathline_")
    ) == ["pathline_0", "pathline_1", "pathline_2"]


def test_default_nnunet_model_folder_prefers_dataset7010_when_available():
    model_dir = default_nnunet_model_folder()
    preferred = Path(
        "/nas-data/ryy_rawdata/aorta_seg/nnres/Dataset7010_All_Mean/"
        "nnUNetTrainerPartBalancedTversky__nnUNetPlans__3d_fullres"
    )
    if preferred.is_dir():
        assert model_dir == preferred
    else:
        assert model_dir.name == "nnUNetTrainerPartBalanced__nnUNetPlans__3d_fullres_iso1mm"
    assert model_dir.is_dir()


def test_nnunet_4d_fold_modes_keep_single_and_ensemble_branches(tmp_path):
    from autoflow.algorithms.segmentation import _resolve_nnunet_folds

    model_dir = tmp_path / "model"
    (model_dir / "fold_all").mkdir(parents=True)
    for fold in range(5):
        (model_dir / f"fold_{fold}").mkdir()

    assert _resolve_nnunet_folds(model_dir, "single") == ["all"]
    assert _resolve_nnunet_folds(model_dir, "all") == ["0", "1", "2", "3", "4"]
    assert _resolve_nnunet_folds(model_dir, "0,2,4") == ["0", "2", "4"]


def test_auto_nnunet_models_use_backend_specific_absolute_paths(monkeypatch, tmp_path):
    import autoflow.algorithms.segmentation as segmentation_module
    from autoflow.algorithms.segmentation import (
        resolve_nnunet_model_folder,
    )
    from autoflow.config import load_config_bundle
    from autoflow.core.models import SegmentationState

    fake_cwd_model = (
        tmp_path
        / "autoflow"
        / "segmodel"
        / "nnUNetTrainerPartBalanced__nnUNetPlans__3d_fullres_iso1mm"
    )
    fake_cwd_model.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    resolved = Path(resolve_nnunet_model_folder())

    assert SegmentationState().auto_model == "auto"
    assert SegmentationState().auto_checkpoint == "auto"
    assert load_config_bundle()["segmentation"]["auto_model"] == "auto"
    assert load_config_bundle()["segmentation"]["auto_checkpoint"] == "auto"
    assert resolved == default_nnunet_model_folder()
    assert resolved != fake_cwd_model
    assert resolved.is_absolute()
    assert resolved.is_dir()
    assert segmentation_module._NNUNET_4D_MODEL_DEFAULT == Path(
        "/nas-data/ryy_rawdata/aorta_seg/nnres_noCC/Dataset7020_Aorta_4DTemporalFT/"
        "nnUNetTrainerPartBalancedTversky__nnUNetPlansIso1mm__3d_fullres"
    )


def test_bundled_nnunet_relative_path_resolves_outside_repo_cwd(monkeypatch, tmp_path):
    from autoflow.algorithms.segmentation import resolve_nnunet_model_folder

    monkeypatch.chdir(tmp_path)
    resolved = Path(resolve_nnunet_model_folder(
        "autoflow/segmodel/nnUNetTrainerPartBalanced__nnUNetPlans__3d_fullres_iso1mm"
    ))
    assert resolved.is_absolute()
    assert resolved.name == "nnUNetTrainerPartBalanced__nnUNetPlans__3d_fullres_iso1mm"
    assert resolved.is_dir()


def test_frozen_nnunet_predict_command_reuses_main_executable(monkeypatch):
    from autoflow.algorithms.segmentation import runtime as segmentation_module

    monkeypatch.setattr(segmentation_module.sys, "frozen", True, raising=False)
    monkeypatch.setattr(segmentation_module.sys, "executable", "AutoFlow-GUI.exe")
    assert segmentation_module._nnunet_predict_command() == [
        "AutoFlow-GUI.exe",
        "--autoflow-internal-nnunet-predict",
    ]


def test_nnunet_predict_command_stays_in_current_environment(monkeypatch):
    from autoflow.algorithms.segmentation import runtime as segmentation_module

    monkeypatch.setattr(segmentation_module.sys, "frozen", False, raising=False)
    monkeypatch.setattr(segmentation_module.sys, "executable", "/current/env/bin/python")
    monkeypatch.setattr(segmentation_module.shutil, "which", lambda _name: None)

    command = segmentation_module._nnunet_predict_command()

    assert command[0] == "/current/env/bin/python"
    assert command[1] == "-c"


def test_save_pwv_h5_exports_group_payloads(tmp_path):
    out = tmp_path / "pwv.h5"
    results = [{
        "name": "aorta",
        "status": "ok",
        "pwv_m_s": 4.2,
        "position_mm": [0.0, 10.0],
        "arrival_time_ms": [0.0, 2.0],
        "planes": [{"distance_mm": 0.0, "waveform": [1.0, 2.0], "flowrate_mL_s": [3.0, 4.0]}],
    }]
    save_pwv_h5(results, str(out), source_path="case.h5", source_format="normalized_h5", source_group="Case001")
    assert out.is_file()
    with h5py.File(out, "r") as handle:
        assert handle.attrs["schema"] == "autoflow.pwv.v1"
        assert int(handle.attrs["group_count"]) == 1
        assert "payload_json" in handle
        grp = handle["group_0000"]
        assert grp.attrs["name"] == "aorta"
        assert "payload_json" in grp
        assert np.allclose(np.asarray(grp["position_mm"][()], dtype=float), [0.0, 10.0])
        assert np.allclose(np.asarray(grp["arrival_time_ms"][()], dtype=float), [0.0, 2.0])
        assert "planes" in grp


def test_remote_plotter_packs_rgba_screenshot_for_qimage():
    QtGui = pytest.importorskip("PySide6.QtGui")
    from autoflow.ui.remote_plotter import _prepare_rgb_frame

    rgba = np.zeros((4, 5, 4), dtype=np.uint8)
    rgba[1, 2] = [12, 34, 56, 78]
    rgb_view = rgba[:, :, :3]
    assert not rgb_view.flags.c_contiguous

    rgb = _prepare_rgb_frame(rgba)
    assert rgb.flags.c_contiguous
    assert rgb.strides[0] == rgb.shape[1] * 3

    qimage = QtGui.QImage(
        rgb.data,
        rgb.shape[1],
        rgb.shape[0],
        int(rgb.strides[0]),
        QtGui.QImage.Format_RGB888,
    ).copy()
    assert not qimage.isNull()
    assert qimage.pixelColor(2, 1).getRgb()[:3] == (12, 34, 56)


def test_scene_controller_builds_grouped_skeleton_graph_and_forks():
    from autoflow.core.models import GraphData, Workspace
    from autoflow.ui.viewer import SceneController

    workspace = Workspace()
    workspace.origin = np.array([10.0, 20.0, 30.0], dtype=float)
    skeleton_points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=float,
    )
    graph = GraphData(
        points=skeleton_points.copy(),
        edges=np.array([[0, 1], [1, 2]], dtype=int),
    )
    workspace.multilabel_groups = {
        "single_label": {
            "skeleton_points": skeleton_points,
            "graph": graph,
            "forks": [{"crosspoint": [1.0, 0.0, 0.0]}],
        }
    }
    controller = SceneController(SimpleNamespace(), workspace, lambda _message: None)

    skeleton_mesh = controller._build_dataset("skeleton_single_label")
    graph_mesh = controller._build_dataset("graph_single_label")
    fork_mesh = controller._build_dataset("forks_single_label")

    expected_points = skeleton_points + workspace.origin.reshape(1, 3)
    assert skeleton_mesh.n_points == 3
    assert np.allclose(skeleton_mesh.points, expected_points)
    assert graph_mesh.n_points == 3
    assert graph_mesh.n_lines == 2
    assert np.allclose(graph_mesh.points, expected_points)
    assert fork_mesh.n_points == 1
    assert np.allclose(fork_mesh.points[0], [11.0, 20.0, 30.0])

    workspace.skeleton_points = skeleton_points
    workspace.graph = graph
    assert controller._build_dataset("skeleton_points").n_points == 3
    assert controller._build_dataset("graph_lines").n_lines == 2


@pytest.mark.parametrize("static_magnitude", [False, True])
def test_pcmra_phantom_phase_updates_preserve_window_opacity_and_actor(static_magnitude):
    from contextlib import closing
    import pyvista as pv
    from autoflow.core.models import ObjectKind, SceneObject
    from autoflow.ui.viewer import SceneController

    magnitude = np.arange(1, 126, dtype=np.float64).reshape(5, 5, 5)
    flow = np.zeros((5, 5, 5, 3, 3), dtype=np.float64)
    flow[..., 0, 0], flow[..., 1, 0], flow[..., 2, 0] = 1.0, 2.0, 4.0
    workspace = Workspace()
    workspace.mag_raw = magnitude if static_magnitude else np.repeat(magnitude[..., None], 3, axis=3)
    workspace.flow_raw = flow
    workspace.data_loaded = True
    assert PipelineEngine().run_step(workspace, StepId.GENERATE_PCMRA, lambda _: None).success
    original_mag, original_flow = workspace.mag_raw.copy(), flow.copy()
    obj = SceneObject("pcmra", "PC-MRA", ObjectKind.AUX, "pcmra_volume",
                      scalars="PC-MRA", cmap="gray", opacity=0.8, dynamic=True)
    workspace.scene_objects[obj.uid] = obj
    errors = []
    with closing(pv.Plotter(off_screen=True)) as plotter:
        controller = SceneController(plotter, workspace, errors.append)
        controller.render_all()
        actor = obj.actor
        assert actor is not None
        for phase, speed in enumerate((1.0, 2.0, 4.0)):
            controller.update_time(phase)
            assert obj.actor is actor
            dataset = actor.GetMapper().dataset
            assert dataset.active_scalars_name == "PC-MRA"
            assert np.isclose(dataset.get_data_range("PC-MRA")[1], magnitude.max() * speed)
            expected = np.percentile(magnitude * speed, (5.0, 99.0))
            assert np.allclose(controller._volume_scalar_range(obj, dataset), expected)
            prop = actor.GetProperty()
            assert np.allclose(prop.GetRGBTransferFunction(0).GetRange(), expected)
            # A LUT update must not restore PyVista's default linear opacity.
            assert np.isclose(prop.GetScalarOpacity(0).GetValue(expected[1]), 0.42 * 0.8)
        controller._apply_volume_window_level(obj, (20.0, 80.0), render=False)
        controller.update_time(1)
        assert np.allclose(actor.GetProperty().GetRGBTransferFunction(0).GetRange(), (20.0, 80.0))
        controller.invalidate_cache("streamlines")
        assert controller.reset_volume_window_level()
        assert obj.actor is actor
        assert np.allclose(actor.GetProperty().GetRGBTransferFunction(0).GetRange(),
                           np.percentile(magnitude * 2.0, (5.0, 99.0)))
        controller._apply_volume_window_level(obj, (-50.0, 80.0), render=False)
        assert actor.GetProperty().GetScalarOpacity(0).GetValue(0.0) == 0.0
        obj.opacity = 0.0
        controller.apply_object_properties(obj, render=False)
        assert actor.GetProperty().GetScalarOpacity(0).GetValue(80.0) == 0.0
    assert not errors
    assert np.array_equal(workspace.mag_raw, original_mag)
    assert np.array_equal(workspace.flow_raw, original_flow)


def test_scene_display_axis_orientation_mirrors_points_without_mutating_world_data():
    from autoflow.ui.viewer import SceneController

    class DummyPlotter:
        bounds = (0.0, 10.0, 0.0, 20.0, 0.0, 30.0)

        def add_axes(self, **_kwargs):
            return None

        def hide_axes(self):
            return None

        def render(self):
            return None

    workspace = Workspace()
    controller = SceneController(DummyPlotter(), workspace, lambda _message: None)
    world = np.asarray([[0.0, 10.0, 15.0], [10.0, 10.0, 15.0]], dtype=float)
    assert controller.set_display_axis_directions(["RL", "AP", "FH"])
    displayed = controller.world_to_display_points(world)
    assert np.allclose(displayed, [[10.0, 10.0, 15.0], [0.0, 10.0, 15.0]])
    assert np.allclose(controller.display_to_world_points(displayed), world)


@pytest.mark.parametrize("dicom_backend", [None, "dicom2h5", "native"])
def test_dicom2h5_groups_route_through_h5_loader(tmp_path, monkeypatch, dicom_backend):
    from autoflow.algorithms import dicom_conversion as conversion
    from autoflow.algorithms.inputs import load_input_data
    root = tmp_path / "dicom"
    root.mkdir()
    original = root / "source.dcm"
    original.write_bytes(b"read-only-source")
    def convert(source, destination):
        assert Path(source) == root
        with h5py.File(destination, "w") as h5:
            for name in ("sequence_a", "sequence_b"):
                group = h5.create_group(name)
                group["mag"] = np.ones((4, 5, 6, 2), dtype=np.float32)
                group["flow"] = np.full((4, 5, 6, 2, 3), 12.0, dtype=np.float32)
                group["segmask"] = np.ones((4, 5, 6), dtype=np.int16)
                group["Resolution"] = [1.0, 2.0, 3.0]
                group["Origin"] = [0.0, 0.0, 0.0]
                group["VENC"] = [150.0, 150.0, 150.0]
                group["RR"] = 900.0
                group["SpatialOrder"] = np.asarray(["LR", "AP", "FH"], dtype=h5py.string_dtype())
                group["VENCOrder"] = np.asarray(["LR", "AP", "FH"], dtype=h5py.string_dtype())
    backend = SimpleNamespace(convert_dicom_to_h5=convert, validate_native_h5=lambda _: {"valid": True, "groups": ["sequence_a", "sequence_b"]})
    monkeypatch.setattr(conversion, "_converter_module", lambda: backend)
    backend_kwargs = {} if dicom_backend is None else {"dicom_backend": dicom_backend}
    cases = collect_input_cases([str(root), str(root)], dicom_h5_dir=str(tmp_path / "converted"), **backend_kwargs)
    assert len(cases) == 2
    assert {case.source_group for case in cases} == {"sequence_a", "sequence_b"}
    assert cases[0].input_kind == "h5"
    assert ".dicom2h5-" not in cases[0].output_name
    loaded = load_input_data(cases[0])
    assert loaded.flow.shape == (4, 5, 6, 2, 3)
    np.testing.assert_allclose(loaded.flow, 12.0)
    assert loaded.metadata["dicom_backend"] == "dicom2h5"
    assert loaded.tke_array is None and not loaded.capabilities.has_tke
    result = run_case(cases[0], output_dir=str(tmp_path / "results"), config=AutoFlowConfig(segmentation_only=True, background_phase_write_cache=False))
    assert result["segmentation_only"] and result["source_group"] == "sequence_a"
    assert original.read_bytes() == b"read-only-source"
    assert not list((tmp_path / "converted").glob(".dicom2h5-*"))
    with pytest.raises(ValueError, match="multiple H5 cases"):
        load_input_data(root, dicom_h5_dir=str(tmp_path / "ambiguous"))
    assert len(list((tmp_path / "ambiguous").glob("*.h5"))) == 1


def test_dicom2h5_failed_conversion_never_publishes_or_replaces(tmp_path, monkeypatch):
    from autoflow.algorithms import dicom_conversion as conversion
    root = tmp_path / "dicom"
    root.mkdir()
    target = tmp_path / "result.h5"
    target.write_bytes(b"existing-result")
    monkeypatch.setattr(conversion, "_converter_module", lambda: (_ for _ in ()).throw(AssertionError("existing output must fail before converter starts")))
    with pytest.raises(FileExistsError):
        conversion.convert_dicom_input(root, target)
    assert target.read_bytes() == b"existing-result"
    target.unlink()
    def convert(_source, destination):
        with h5py.File(destination, "w"):
            pass
    monkeypatch.setattr(conversion, "_converter_module", lambda: SimpleNamespace(convert_dicom_to_h5=convert, validate_native_h5=lambda _: {"valid": False, "groups": [], "errors": ["no flow"]}))
    with pytest.raises(ValueError, match="no valid flow cases"):
        conversion.convert_dicom_input(root, target)
    assert not target.exists()
    assert not list(tmp_path.glob(".dicom2h5-*"))
    def invalid_calibration(_source, destination):
        with h5py.File(destination, "w") as h5:
            h5["RR"] = np.nan
            h5["Resolution"] = [1.0, 1.0, 1.0]
            h5["VENC"] = [150.0, 150.0, 150.0]
            h5["Origin"] = [0.0, 0.0, 0.0]
    monkeypatch.setattr(conversion, "_converter_module", lambda: SimpleNamespace(convert_dicom_to_h5=invalid_calibration, validate_native_h5=lambda _: {"valid": True, "groups": [""]}))
    with pytest.raises(ValueError, match="invalid RR"):
        conversion.convert_dicom_input(root, target)
    assert not target.exists()
    assert not list(tmp_path.glob(".dicom2h5-*"))


def test_dicom2h5_loader_config_and_cli_selection():
    from autoflow.cli import build_parser
    from autoflow.core.models import LoaderParams
    assert AutoFlowConfig().dicom_backend == "dicom2h5"
    assert LoaderParams.from_dict({"dicom_backend": "native", "dicom_read_workers": 8}).to_dict()["dicom_backend"] == "dicom2h5"
    assert "dicom_read_workers" not in LoaderParams().to_dict()
    config = AutoFlowConfig(dicom_backend="dicom2h5", dicom_h5_dir="converted")
    workspace = build_workspace(config)
    restored = LoaderParams.from_dict(workspace.loader_params.to_dict())
    assert restored.dicom_backend == "dicom2h5" and restored.dicom_h5_dir == "converted"
    args = build_parser().parse_args(["dicom-root", "--dicom-backend", "dicom2h5", "--dicom-h5-dir", "converted"])
    assert args.dicom_backend == "dicom2h5" and args.dicom_h5_dir == "converted"
    with pytest.raises(ValueError, match="Unknown DICOM backend"):
        collect_input_cases([], dicom_backend="unsupported")


@pytest.mark.parametrize("method,default", [
    ("lap4D", "none"), ("gc3D", "none"), ("nprs", "none"),
    ("pudip", "pcmra_std"), ("gust", "pcmra_std"),
])
def test_correction_method_mask_defaults(method, default):
    from autoflow.algorithms.phase_unwrapping import mask_sources_for_method, resolve_mask_source
    assert resolve_mask_source(method) == default
    assert "segmask" in mask_sources_for_method(method)
    if default == "none":
        with pytest.raises(ValueError):
            resolve_mask_source(method, "pcmra_std")
    else:
        assert resolve_mask_source(method, "pcmramean") == "pcmra_mean"


def test_noise_removal_only_masks_pcmra_and_keeps_steady_flow():
    from autoflow.algorithms.noise_removal import pcmra_render_mask
    magnitude = np.ones((4, 3, 2, 4), dtype=np.float32)
    velocity = np.full(magnitude.shape + (3,), 30.0, dtype=np.float32)
    magnitude[0] = 0.001
    velocity[1, 0, 0, :, 0] = [0, 149, 0, 149]
    original_m, original_v = magnitude.copy(), velocity.copy()
    mask, report = pcmra_render_mask(magnitude, velocity, [150, 150, 150], magnitude_fraction=0.04)
    assert not mask[0].any()
    assert not mask[1, 0, 0]
    assert mask[2:].all()  # Steady flow is not discarded as static tissue.
    assert report["scope"] == "pcmra_rendering_only"
    np.testing.assert_array_equal(magnitude, original_m)
    np.testing.assert_array_equal(velocity, original_v)
    whole, report = pcmra_render_mask(np.ones_like(magnitude), np.ones_like(velocity), 150)
    assert whole.all() and report["magnitude_threshold_mode"] == "fraction_of_max"
    assert report["magnitude_threshold"] == pytest.approx(0.05)
    assert report["temporal_std_threshold"] == 0.0
    empty, _ = pcmra_render_mask(np.zeros_like(magnitude), velocity, 150)
    assert not empty.any()


def test_noise_removal_defaults_and_manual_configuration():
    import inspect
    from autoflow.case_types import NoiseRemovalConfig
    from autoflow.cli import build_parser
    from autoflow.config import load_config_bundle
    from autoflow.processing import process_single

    defaults = NoiseRemovalConfig()
    assert defaults.magnitude_fraction == 0.05
    assert defaults.velocity_std_max == 0.80
    bundle = load_config_bundle()
    for resolved in (bundle_to_autoflow_kwargs(bundle), bundle_to_autoflow_kwargs({})):
        assert resolved["noise_magnitude_fraction"] == 0.05
        assert resolved["noise_velocity_std_max"] == 0.80
    api_config = AutoFlowConfig()
    assert api_config.noise_magnitude_fraction == 0.05
    assert api_config.noise_velocity_std_max == 0.80
    signature = inspect.signature(process_single)
    assert signature.parameters["noise_magnitude_fraction"].default == 0.05
    assert signature.parameters["noise_velocity_std_max"].default == 0.80
    assert Workspace().noise_removal_params == defaults
    assert build_workspace(api_config).noise_removal_params == defaults
    args = build_parser().parse_args(["case.h5", "--noise-magnitude-fraction", "0.06",
                                     "--noise-velocity-std-max", "0.24"])
    custom = AutoFlowConfig(noise_magnitude_fraction=args.noise_magnitude_fraction,
                            noise_velocity_std_max=args.noise_velocity_std_max)
    workspace = build_workspace(custom)
    assert workspace.noise_removal_params.magnitude_fraction == 0.06
    assert workspace.noise_removal_params.velocity_std_max == 0.24
    restored = Workspace()
    restored.restore_dict(workspace.snapshot_dict())
    assert restored.noise_removal_params == workspace.noise_removal_params
    for name in ("magnitude_fraction", "velocity_std_max"):
        for invalid in (-0.01, 1.01, np.nan, np.inf):
            with pytest.raises(ValueError):
                NoiseRemovalConfig(**{name: invalid})


def test_noise_removal_phantom_gui_controls_have_no_explanation(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets
    from autoflow.ui.app import MainWindow

    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    panel = QtWidgets.QWidget()
    owner = SimpleNamespace(params_layout=QtWidgets.QVBoxLayout(panel), _reset_noise_removal=lambda: None)
    try:
        MainWindow._build_noise_removal_params(owner)
        assert owner.spin_noise_magnitude.value() == 0.05
        assert owner.spin_noise_std_max.value() == 0.80
        assert [label.text() for label in panel.findChildren(QtWidgets.QLabel)] == [
            "Method", "Magnitude fraction of maximum", "Temporal SD fraction of maximum"]
        owner.spin_noise_magnitude.setValue(0.03)
        owner.spin_noise_std_max.setValue(0.90)
        assert owner.spin_noise_magnitude.value() == 0.03
        assert owner.spin_noise_std_max.value() == 0.90
    finally:
        panel.close()
        panel.deleteLater()
        application.processEvents()


def test_noise_removal_phantom_uses_maximum_magnitude_and_speed_sd():
    from autoflow.algorithms.noise_removal import pcmra_render_mask

    magnitude = np.ones((102, 1, 1, 4), dtype=np.float32)
    magnitude[0] = 0.4
    magnitude[2] = 0.01
    magnitude[-1] = [4, 8, 12, 16]
    velocity = np.zeros(magnitude.shape + (3,), dtype=np.float32)
    velocity[..., 0] = 30
    velocity[1, 0, 0, :, 0] = [10, 90, 10, 90]
    velocity[2, 0, 0, :, 0] = [0, 100, 0, 100]
    velocity[3, 0, 0, :, 0] = [0, 81, 0, 81]
    velocity[4, 0, 0, :, 0] = [-30, 30, -30, 30]
    velocity[5] = np.inf
    velocity[6] = np.nan
    magnitude[7] = np.nan
    magnitude[8] = np.inf
    mask, report = pcmra_render_mask(magnitude, velocity, 150)
    assert not mask[0] and mask[1] and not mask[2] and not mask[3]
    assert mask[4] and not mask[5:9].any() and mask[9:].all()
    assert report["magnitude_statistic"] == "temporal_mean"
    assert report["magnitude_reference_max"] == 10.0
    assert report["magnitude_threshold"] == pytest.approx(0.5)
    assert report["temporal_std_statistic"] == "speed"
    assert report["temporal_std_reference_max"] == 50.0
    assert report["temporal_std_threshold"] == 40.0
    different_venc, _ = pcmra_render_mask(magnitude, velocity, [50, 100, 200])
    np.testing.assert_array_equal(mask, different_venc)
    relaxed, _ = pcmra_render_mask(magnitude, velocity, 150,
                                   magnitude_fraction=0.03, velocity_std_max=0.90)
    assert relaxed[0] and relaxed[3]
    magnitude_only, report = pcmra_render_mask(magnitude, velocity, 150, method="magnitude")
    assert magnitude_only[3] and not report["temporal_screening_applied"]
    assert report["temporal_std_threshold"] is None
    positive_only, report = pcmra_render_mask(magnitude, velocity, 150,
                                             magnitude_fraction=0, velocity_std_max=0)
    assert positive_only[:5].all() and not positive_only[5:9].any()
    assert not report["temporal_screening_applied"]
    magnitude[0] = 0
    positive_only, _ = pcmra_render_mask(magnitude, velocity, 150,
                                        magnitude_fraction=0, velocity_std_max=0)
    assert not positive_only[0]


@pytest.mark.parametrize("frames", [1, 3, 15])
def test_noise_removal_phantom_shared_magnitude_and_constant_speed(frames):
    from autoflow.algorithms.noise_removal import pcmra_render_mask

    magnitude = np.ones((2, 2, 2), dtype=np.float32)
    velocity = np.full(magnitude.shape + (frames, 3), 30, dtype=np.float32)
    mask, report = pcmra_render_mask(magnitude, velocity, 150)
    assert mask.all()
    assert report["temporal_screening_applied"] == (frames > 1)
    assert report["temporal_std_threshold"] == (0.0 if frames > 1 else None)
    velocity[:] = np.nan
    mask, _ = pcmra_render_mask(magnitude, velocity, 150)
    assert not mask.any()


def test_correction_updates_working_flow_and_preserves_downstream(monkeypatch, tmp_path):
    from autoflow.core import pipeline as module
    from autoflow.case_types import LoadedCase, PhaseUnwrappingConfig
    shape = (4, 4, 4, 3)
    flow = np.full(shape + (3,), 10.0, dtype=np.float32)
    seg = np.ones(shape, dtype=np.int16)
    calls = []
    def load(_source, **kwargs):
        cfg = kwargs["correction_config"]
        enabled = cfg.get("enabled", False) if isinstance(cfg, dict) else cfg.enabled
        calls.append(enabled)
        return LoadedCase(mag=np.ones(shape, dtype=np.float32), flow=flow + (5 if enabled else 0),
            resolution=np.ones(3), origin=np.zeros(3), venc=np.full(3, 150.0), rr=1000,
            segmentation=seg, metadata={"background_phase_correction": {"applied": enabled}})
    monkeypatch.setattr(module, "load_input_data", load)
    ws = Workspace()
    ws.paths.flow_path = str(tmp_path / "source.h5")
    ws.loader_params.background_phase_correction.enabled = True
    engine = PipelineEngine()
    engine.load_data(ws, lambda _: None)
    assert calls == [False]  # Correction is an explicit action after loading.
    assert ws.pcmra_array is None and not any(obj.data_key == "pcmra_volume" for obj in ws.scene_objects.values())
    planes = [PlaneData(center=np.zeros(3), normal=np.ones(3), label=1)]
    metrics = [{"sentinel": 1}]
    ws.planes = planes
    ws.derived.plane_metrics = metrics
    ws.pathline_cache = {"sentinel": object()}
    old_trajectories = ws.pathline_cache
    old_seg = ws.segmask_raw
    ws.pipeline.mark_done(StepId.COMPUTE_PLANE_METRICS)
    assert engine.run_step(ws, StepId.BACKGROUND_CORRECTION, lambda _: None).success
    np.testing.assert_array_equal(ws.flow_raw, flow + 5)
    np.testing.assert_array_equal(ws.flow_input, flow + 5)
    assert ws.segmask_raw is old_seg and ws.planes is planes and ws.derived.plane_metrics is metrics
    assert ws.pathline_cache is old_trajectories
    assert ws.pipeline.is_done(StepId.COMPUTE_PLANE_METRICS)
    before = ws.flow_raw.copy()
    assert engine.run_step(ws, StepId.REMOVE_NOISE, lambda _: None).success
    np.testing.assert_array_equal(ws.flow_raw, before)
    assert ws.planes is planes and ws.derived.plane_metrics is metrics
    ws.phase_unwrap_params = PhaseUnwrappingConfig(method="lap4D", mask_source="segmask")
    old_seg[2:] = 0
    def unwrap(phase, mask, venc, method, **kwargs):
        assert method == "lap4D"
        output = phase * np.asarray(venc).reshape(1, 1, 1, 1, 3) / np.pi + 300
        return {"flow_unwrapped": output, "phase_unwrapped": phase + 2*np.pi,
                "mask_used": mask, "statistics": {}}
    monkeypatch.setattr(module, "unwrap_phase", unwrap)
    ws.flow_raw[2:] = 50  # Preserve an earlier whole-volume result outside the new mask.
    assert engine.run_step(ws, StepId.UNWRAP_PHASE, lambda _: None).success
    np.testing.assert_allclose(ws.flow_raw[:2], 315, atol=1e-4)
    np.testing.assert_allclose(ws.flow_raw[2:], 50, atol=1e-4)
    np.testing.assert_allclose(ws.phase_unwrap_result["phase_unwrapped"] * 150 / np.pi, ws.flow_raw, atol=1e-4)
    assert ws.segmask_raw is old_seg and ws.planes is planes and ws.derived.plane_metrics is metrics
    assert ws.pathline_cache is old_trajectories and ws.pipeline.is_done(StepId.COMPUTE_PLANE_METRICS)
    assert engine.revert_phase_unwrap(ws).success
    np.testing.assert_array_equal(ws.flow_raw, before)
    assert ws.planes is planes and ws.derived.plane_metrics is metrics


def test_phase_segmask_requires_an_available_segmentation():
    from autoflow.case_types import PhaseUnwrappingConfig
    ws = Workspace()
    ws.phase_wrapped = np.zeros((4, 4, 4, 3, 3), dtype=np.float32)
    ws.phase_unwrap_params = PhaseUnwrappingConfig(method="lap4D", mask_source="segmask")
    result = PipelineEngine().run_step(ws, StepId.UNWRAP_PHASE, lambda _: None)
    assert not result.success and "segmask requires" in result.message


def test_correction_cli_api_order_and_default_mask(monkeypatch, tmp_path):
    from autoflow.core import pipeline as module
    from autoflow import processing
    from autoflow.case_types import LoadedCase, InputCase
    from autoflow.cli import build_parser
    events = []
    shape = (4, 4, 4, 3)
    velocity = np.ones(shape + (3,), dtype=np.float32)
    def load(_source, **kwargs):
        cfg = kwargs["correction_config"]
        enabled = cfg.get("enabled", False) if isinstance(cfg, dict) else cfg.enabled
        events.append("background" if enabled else "load")
        return LoadedCase(mag=np.ones(shape, dtype=np.float32), flow=velocity * (2 if enabled else 1),
            resolution=np.ones(3), origin=np.zeros(3), venc=np.full(3, 150.0), rr=1000,
            metadata={"background_phase_correction": {"applied": enabled}})
    monkeypatch.setattr(module, "load_input_data", load)
    original_noise = module.pcmra_render_mask
    def noise(*args, **kwargs):
        events.append("noise")
        return original_noise(*args, **kwargs)
    monkeypatch.setattr(module, "pcmra_render_mask", noise)
    original_generate = PipelineEngine._step_generate_pcmra
    def generate(engine, ws):
        events.append("pcmra")
        return original_generate(engine, ws)
    monkeypatch.setattr(PipelineEngine, "_step_generate_pcmra", generate)
    def unwrap(phase, mask, venc, method, **kwargs):
        events.append("unwrap")
        assert method == "lap4D" and mask.all()
        return {"flow_unwrapped": np.full_like(phase, 20), "phase_unwrapped": phase,
            "wrap_count": np.zeros_like(phase, dtype=np.int16), "wrap_mask": np.zeros_like(phase, dtype=bool),
            "mask_used": mask, "method": method, "statistics": {}}
    monkeypatch.setattr(module, "unwrap_phase", unwrap)
    def segment(ws, _out_dir, **kwargs):
        events.append("segmentation")
        np.testing.assert_allclose(ws.flow_raw, 20)
        ws.set_segmentation_source("auto", np.ones(shape, dtype=np.int16))
        ws.activate_segmentation_source("auto")
        return ""
    monkeypatch.setattr(processing, "_run_cli_auto_segmentation", segment)
    args = build_parser().parse_args(["case.h5", "--correction", "--noise-removal"])
    assert args.correction and args.noise_removal
    summary = run_case(InputCase(str(tmp_path / "case.h5"), "h5"), output_dir=str(tmp_path / "results"),
        config=AutoFlowConfig(correction_all=True, autoseg=True, segmentation_only=True,
                              background_phase_write_cache=False))
    assert events == ["load", "background", "noise", "unwrap", "pcmra", "segmentation"]
    assert summary["phase_unwrap"]["mask_source"] == "none"
    with np.load(summary["pcmra_file"]) as saved:
        np.testing.assert_allclose(saved["pcmra"], 20 * np.sqrt(3), rtol=1e-6)
    mask_file = Path(summary["noise_removal_file"])
    assert mask_file.is_file()
    with np.load(mask_file) as saved:
        assert saved["mask"].shape == shape[:3]
        np.testing.assert_allclose(saved["resolution"], 1)



def test_generated_pcmra_is_explicit_and_noise_region_is_renderable():
    import pyvista as pv
    from contextlib import closing
    from autoflow.ui.viewer import SceneController
    ws = Workspace()
    ws.data_loaded = True
    ws.mag_raw = np.ones((5, 5, 5, 3), dtype=np.float32)
    ws.mag_raw[:2] = 0.001
    ws.flow_raw = np.full(ws.mag_raw.shape + (3,), 20, dtype=np.float32)
    ws.venc = np.full(3, 150.0)
    ws.resolution = np.array([1.2, 2.3, 3.4])
    ws.origin = np.array([-2.0, 5.0, 12.0])
    engine = PipelineEngine()
    with closing(pv.Plotter(off_screen=True)) as plotter:
        controller = SceneController(plotter, ws, lambda _: None)
        assert controller._build_dataset("pcmra_volume") is None
        ws.noise_removal_params.magnitude_fraction = 0.04
        assert engine.run_step(ws, StepId.REMOVE_NOISE, lambda _: None).success
        assert ws.pcmra_array is None
        noise = next(obj for obj in ws.scene_objects.values() if obj.data_key == "noise_region")
        assert not noise.visible
        assert noise.color == "#ff0000"
        mesh = controller._build_dataset("noise_region")
        assert isinstance(mesh, pv.ImageData)
        rejected = mesh.point_data["Noise mask"] > 0
        assert np.count_nonzero(rejected) == np.count_nonzero(~ws.pcmra_render_mask)
        noise_positions = mesh.points[rejected]
        indices = np.rint((noise_positions - ws.origin) / ws.resolution - 0.5).astype(int)
        assert not ws.pcmra_render_mask[tuple(indices.T)].any()
        assert engine.run_step(ws, StepId.GENERATE_PCMRA, lambda _: None).success
        stored = ws.pcmra_array.copy()
        ws.flow_raw *= 2
        controller.invalidate_cache("pcmra_volume")
        dataset = controller._build_dataset("pcmra_volume")
        display = dataset.point_data["PC-MRA"].reshape(tuple(np.array(ws.pcmra_render_mask.shape) + 2), order="F")
        np.testing.assert_allclose(display[1:-1, 1:-1, 1:-1], np.where(ws.pcmra_render_mask, stored[..., 0], 0.0))
        assert not display[0].any() and not display[-1].any()
        assert not display[:, 0].any() and not display[:, -1].any()
        assert not display[:, :, 0].any() and not display[:, :, -1].any()
        sampled = pv.PolyData(noise_positions).sample(dataset)
        np.testing.assert_array_equal(sampled["vtkValidPointMask"], 1)
        np.testing.assert_allclose(sampled["PC-MRA"], 0.0, atol=1e-6)
        # Display refreshes do not regenerate PC-MRA after a velocity change.
        assert np.isclose(dataset.get_data_range("PC-MRA")[1], 20 * np.sqrt(3))
        np.testing.assert_array_equal(ws.pcmra_array, stored)
        assert engine.run_step(ws, StepId.GENERATE_PCMRA, lambda _: None).success
        np.testing.assert_allclose(ws.pcmra_array, stored * 2, rtol=1e-6)
        restored = Workspace()
        restored.restore_dict(ws.snapshot_dict())
        np.testing.assert_allclose(restored.pcmra_array, ws.pcmra_array)
        np.testing.assert_array_equal(restored.pcmra_render_mask, ws.pcmra_render_mask)


@pytest.mark.parametrize("selected_plane", [False, True])
def test_noise_overlay_phantom_paints_over_segmentation_and_refreshes(monkeypatch, selected_plane):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtCore, QtWidgets
    from autoflow.ui.ortho_viewer import OrthoViewer

    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    ws = Workspace()
    ws.mag_raw = np.full((12, 12, 12, 2), 0.2, dtype=np.float32)
    ws.pcmra_render_mask = np.ones(ws.mag_raw.shape[:3], dtype=bool)
    ws.pcmra_render_mask[:6, :6, :6] = False
    ws.segmask_raw = np.ones(ws.mag_raw.shape, dtype=np.int16)
    ws.segmentation.opacity = 1.0
    ws.segmentation.label_colors = {"1": "#00ff00"}
    ws.planes = [PlaneData(center=np.array([5.0, 5.0, 5.0]), normal=np.array([1.0, 1.0, 1.0]), label=1)]
    viewer = OrthoViewer(ws)
    try:
        viewer.resize(1100, 850)
        viewer.show()
        viewer.update_slider_ranges()
        for slider in (viewer.slider_x, viewer.slider_y, viewer.slider_z):
            slider.setValue(4)
        viewer._apply_pending_cursor()
        if selected_plane:
            viewer.set_selected_plane(0)
        assert viewer.btn_noise_overlay.isEnabled()
        assert not viewer.btn_noise_overlay.isChecked()
        assert isinstance(viewer.btn_noise_overlay, QtWidgets.QCheckBox)
        assert viewer.slider_noise_overlay_opacity.value() == 0
        viewer.slider_noise_overlay_opacity.setValue(100)
        assert viewer.btn_noise_overlay.isChecked()
        application.processEvents()
        for view in viewer.slice_views.values():
            view.reset_view(zoom=1.0)
        application.processEvents()

        def painted_color(view, rejected):
            image = view.grab().toImage()
            noise = view.noise_overlay_item.image[..., 3] > 0
            labels = view.overlay_item.image[..., 3] > 0
            positions = np.argwhere(labels & (noise if rejected else ~noise))
            for row, column in positions[len(positions) // 3:]:
                horizontal = view._sample_origin[0] + column * view._spacing[0]
                vertical = view._sample_origin[1] + row * view._spacing[1]
                if abs(horizontal - view.vertical_line.value()) < view._spacing[0]:
                    continue
                if abs(vertical - view.horizontal_line.value()) < view._spacing[1]:
                    continue
                position = view.view_box.mapViewToScene(QtCore.QPointF(horizontal, vertical))
                viewport_position = view.plot.mapFromScene(position)
                widget_position = view.plot.viewport().mapTo(view, viewport_position)
                if image.rect().contains(widget_position):
                    return image.pixelColor(widget_position).getRgb()[:3]
            pytest.fail("No unobscured overlay pixel available")

        for view in viewer.slice_views.values():
            assert view.noise_overlay_item.isVisible()
            assert view.noise_overlay_item.zValue() > view.overlay_item.zValue()
            assert painted_color(view, True) == (255, 0, 0)
            assert painted_color(view, False) == (0, 255, 0)
        viewer.slider_noise_overlay_opacity.setValue(50)
        application.processEvents()
        for view in viewer.slice_views.values():
            assert view.noise_overlay_item.opacity() == 0.5
            red, green, blue = painted_color(view, True)
            assert 120 <= red <= 135 and 120 <= green <= 135 and blue == 0
        viewer.set_playback_active(True)
        ws.current_t = 1
        viewer.refresh(update_plane=False)
        for view in viewer.slice_views.values():
            assert view.noise_overlay_item.isVisible()
            assert view.noise_overlay_item.opacity() == 0.5
        viewer.btn_noise_overlay.setChecked(False)
        assert viewer.slider_noise_overlay_opacity.value() == 0
        assert all(not view.noise_overlay_item.isVisible() for view in viewer.slice_views.values())
        viewer.btn_noise_overlay.setChecked(True)
        assert viewer.slider_noise_overlay_opacity.value() == 50
        viewer.slider_noise_overlay_opacity.setValue(0)
        assert not viewer.btn_noise_overlay.isChecked()
        viewer.btn_noise_overlay.setChecked(True)
        assert viewer.slider_noise_overlay_opacity.value() == 35
        ws.pcmra_render_mask = np.ones(ws.mag_raw.shape[:3], dtype=bool)
        viewer.refresh(update_plane=False)
        assert all(not view.noise_overlay_item.image[..., 3].any() for view in viewer.slice_views.values())
        ws.pcmra_render_mask = None
        viewer.refresh(update_plane=False)
        assert not viewer.btn_noise_overlay.isEnabled() and not viewer.btn_noise_overlay.isChecked()
        assert all(not view.noise_overlay_item.isVisible() for view in viewer.slice_views.values())
    finally:
        viewer.close()
        viewer.deleteLater()
        application.processEvents()


@pytest.mark.parametrize("mapper", ["smart", "fixed_point"])
def test_noise_region_phantom_opacity_does_not_accumulate_with_depth(mapper):
    from contextlib import closing
    import pyvista as pv
    from autoflow.core.models import ObjectKind
    from autoflow.ui.viewer import SceneController

    pixels = []
    for depth in (2, 60):
        ws = Workspace()
        ws.pcmra_render_mask = np.zeros((5, depth, 5), dtype=bool)
        uid = ws.add_object("Noise Region", ObjectKind.AUX, "noise_region", visible=True, color="#ff0000", opacity=0.15)
        errors = []
        with closing(pv.Plotter(off_screen=True, window_size=(128, 128))) as plotter:
            plotter.set_background("black")
            controller = SceneController(plotter, ws, errors.append)
            controller._volume_mapper_name = lambda: mapper
            controller.render_all()
            obj = ws.scene_objects[uid]
            assert obj.actor is not None
            assert obj.actor.GetMapper().GetBlendMode() == 1
            assert obj.actor.GetProperty().GetScalarOpacity(0).GetValue(0.0) == 0.0
            assert obj.actor.GetProperty().GetScalarOpacity(0).GetValue(1.0) == 0.15
            plotter.camera_position = [(2.5, -150.0, 2.5), (2.5, 2.5, 2.5), (0.0, 0.0, 1.0)]
            plotter.enable_parallel_projection()
            plotter.camera.parallel_scale = 4.0
            image = plotter.screenshot()
            pixels.append(image[64, 64])
            assert 30 <= int(image[64, 64, 0]) <= 45
            assert not image[64, 64, 1:].any()
            obj.opacity = 0.05
            controller.apply_object_properties(obj)
            assert obj.actor.GetProperty().GetScalarOpacity(0).GetValue(1.0) == 0.05
            image = plotter.screenshot()
            assert 8 <= int(image[64, 64, 0]) <= 18
            ws.pcmra_render_mask = np.ones_like(ws.pcmra_render_mask)
            controller.readd_object(obj)
            assert obj.actor is None
        assert not errors
    np.testing.assert_allclose(pixels[0], pixels[1], atol=1)


def test_correction_phantom_gui_refresh_enables_noise_overlay_and_pcmra_controls(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets
    from unittest.mock import Mock
    from autoflow.ui.app import MainWindow
    from autoflow.ui.ortho_viewer import OrthoViewer

    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    ws = Workspace()
    ws.mag_raw = np.ones((6, 6, 6, 2), dtype=np.float32)
    ws.pcmra_render_mask = np.ones(ws.mag_raw.shape[:3], dtype=bool)
    ws.pcmra_render_mask[:2] = False
    viewer = OrthoViewer(ws)
    owner = Mock(workspace=ws, ortho_viewer=viewer, _pipeline_progress_dialog=None)
    try:
        MainWindow._finish_pipeline_scene_refresh(owner, [StepId.REMOVE_NOISE, StepId.GENERATE_PCMRA])
        owner._refresh_render_range_control.assert_called_once()
        assert viewer.btn_noise_overlay.isChecked()
        assert viewer.slider_noise_overlay_opacity.value() == 35
        assert all(view.noise_overlay_item.isVisible() for view in viewer.slice_views.values())
        volume = SimpleNamespace(data_key="pcmra_volume", scalars="PC-MRA")
        segmentation = SimpleNamespace(data_key="segmask_raw_surface", scalars="label")
        noise = SimpleNamespace(data_key="noise_region", scalars="")
        owner.scene._visible_volume_object.return_value = volume
        owner._browser_selected_objects.return_value = []
        assert MainWindow._selected_render_object(owner) is volume
        owner._browser_selected_objects.return_value = [segmentation, noise]
        assert MainWindow._selected_render_object(owner) is volume
        quantitative = SimpleNamespace(data_key="streamlines_live", scalars="Velocity")
        owner._browser_selected_objects.return_value = [quantitative]
        assert MainWindow._selected_render_object(owner) is quantitative
    finally:
        viewer.close()
        viewer.deleteLater()
        application.processEvents()


def test_content_menu_only_exposes_available_results_and_keeps_selection(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets
    from autoflow.ui.ortho_viewer import OrthoViewer
    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    ws = Workspace()
    viewer = OrthoViewer(ws)
    assert viewer._noise_region_overlay(np.ones((2, 2), dtype=bool)) is None
    viewer._noise_overlay_visible = True
    overlay = viewer._noise_region_overlay(np.array([[True, False], [False, True]], dtype=bool))
    np.testing.assert_array_equal(overlay[..., 3], np.array([[255, 0], [0, 255]], dtype=np.uint8))
    np.testing.assert_array_equal(overlay[..., :3][overlay[..., 3] > 0], np.array([[255, 0, 0], [255, 0, 0]], dtype=np.uint8))
    def keys():
        return [viewer.combo_content.itemData(i) for i in range(viewer.combo_content.count())]
    assert keys() == []
    ws.mag_raw = np.ones((4, 4, 4, 3), dtype=np.float32)
    ws.flow_raw = np.ones(ws.mag_raw.shape + (3,), dtype=np.float32)
    viewer._refresh_content_choices()
    assert keys() == [0, 1, 2, 3, 5]  # No PC-MRA or uncomputed derived metrics.
    viewer.combo_content.setCurrentIndex(viewer.combo_content.findData(1))
    ws.pcmra_array = np.full_like(ws.mag_raw, 4.0)
    ws.derived.pressure_gradient_array = np.ones_like(ws.flow_raw)
    viewer._refresh_content_choices()
    assert viewer.combo_content.currentData() == 1
    assert keys() == [0, 1, 2, 3, 4, 5, 8, 9, 10, 11]
    viewer.combo_content.setCurrentIndex(viewer.combo_content.findData(11))
    volume, title, _ = viewer._get_scalar_slice(0)
    assert title == "|Pressure Grad| (Pa/m)"
    np.testing.assert_allclose(volume, np.sqrt(3), rtol=1e-6)
    ws.derived.pressure_gradient_array = None
    viewer._refresh_content_choices()
    assert viewer.combo_content.currentData() == 3
    assert all(key not in keys() for key in (6, 7, 8, 9, 10, 11, 12, 13, 14, 15))
    ws.planes = [PlaneData(center=np.zeros(3), normal=np.array([1., 0, 0]), label=1)]
    viewer.set_selected_plane(0)
    assert 27 in keys()
    viewer.set_selected_plane(None)
    assert 27 not in keys()
    viewer.close()
    application.processEvents()


def test_derived_plane_cube_reuses_geometry_and_applies_frame_roi(monkeypatch):
    from autoflow.algorithms.metrics import sampling, summarize_plane_derived_metrics
    shape = (12, 12, 12)
    mask = np.ones(shape + (2,), dtype=bool)
    pressure = np.repeat(np.indices(shape)[1][..., None], 2, axis=3).astype(np.float32)
    plane = PlaneData(center=np.array([5.5, 5.5, 5.5]), normal=np.array([1., 0., 0.]))
    original = sampling._build_plane_slice_spec
    frames = []
    def counted(*args, **kwargs):
        frames.append(kwargs.get("frame_index"))
        return original(*args, **kwargs)
    monkeypatch.setattr(sampling, "_build_plane_slice_spec", counted)
    _, payload = summarize_plane_derived_metrics(plane, mask, (1, 1, 1), (0, 0, 0),
                                                relative_pressure_array=pressure)
    assert frames == [0]
    assert len(payload["timepoints"][1]["relative_pressure_Pa"]) == 144
    plane.roi_edit_operations = {"1": [{"mode": "replace", "polygon": [[0, 0], [3, 0], [3, 3], [0, 3]]}]}
    frames.clear()
    _, payload = summarize_plane_derived_metrics(plane, mask, (1, 1, 1), (0, 0, 0),
                                                relative_pressure_array=pressure)
    assert frames == [0, 1]
    values = payload["timepoints"][1]["relative_pressure_Pa"]
    assert len(values) == 16
    assert np.mean(values) == pytest.approx(6.5)


@pytest.mark.parametrize("change, expected", [
    ("identical", "unchanged"), ("temporal", "geometry"),
    ("topology", "reset"), ("config", "reset"),
])
def test_segmentation_cube_retains_only_valid_geometry(change, expected):
    ws = Workspace()
    ws.flow_raw = np.zeros((14, 14, 14, 3, 3), dtype=np.float32)
    ws.segmask_raw = np.zeros((14, 14, 14, 3), dtype=np.int16)
    ws.segmask_raw[3:11, 3:11, 3:11, :] = 1
    ws.skeleton_params = SkeletonParams(label_groups={"vessel": {"labels": [1]}},
        remove_small_cc=False, do_closing=False, do_opening=False, gaussian_enabled=False,
        dilation_iters=0, erosion_iters=0)
    engine = PipelineEngine()
    engine.preprocess(ws)
    plane = PlaneData(center=np.array([7., 7., 7.]), normal=np.array([1., 0., 0.]),
                      metrics={"flow": [1., 2., 3.]})
    ws.planes = [plane]
    ws.derived.plane_metrics = [dict(plane.metrics)]
    ws.pipeline.mark_done(StepId.GENERATE_PLANES)
    ws.pipeline.mark_done(StepId.COMPUTE_PLANE_METRICS)
    ws.pathline_cache = {0: {0: "old trajectory"}}
    changed = ws.segmask_raw.copy()
    if change == "temporal":
        changed[3, 3:11, 3:11, 0] = 0
    elif change == "topology":
        changed[3, 3:11, 3:11, :] = 0
    elif change == "config":
        ws.skeleton_params.erosion_iters = 1
    ws.segmask_raw = changed
    assert engine.refresh_segmentation_dependents(ws) == expected
    if expected == "reset":
        assert ws.planes == []
        assert not ws.pipeline.is_done(StepId.GENERATE_PLANES)
    else:
        assert ws.planes[0] is plane
        assert ws.pipeline.is_done(StepId.GENERATE_PLANES)
    if expected == "geometry":
        assert ws.derived.plane_metrics == []
        assert ws.pathline_cache == {}
        assert plane.metrics == {}
        assert not ws.pipeline.is_done(StepId.COMPUTE_PLANE_METRICS)
    elif expected == "unchanged":
        assert ws.derived.plane_metrics == [{"flow": [1., 2., 3.]}]


def test_derived_plane_processes_preserve_cube_roi_and_pixelwise_order():
    from autoflow.algorithms.metrics import augment_plane_metrics_with_derived
    shape = (12, 12, 12)
    mask = np.ones(shape + (2,), dtype=bool)
    pressure = np.repeat(np.indices(shape)[1][..., None], 2, axis=3).astype(np.float32)
    planes = [
        PlaneData(center=np.array([5.5, 5.5, 5.5]), normal=np.array([1., 0., 0.])),
        PlaneData(center=np.array([5.5, 5.5, 5.5]), normal=np.array([0., 0., 1.])),
    ]
    planes[0].roi_edit_operations = {"1": [{"mode": "replace", "polygon": [[0, 0], [3, 0], [3, 3], [0, 3]]}]}
    args = ([{"plane_index": 0}, {"plane_index": 1}], planes, mask, (1, 1, 1), (0, 0, 0))
    serial = augment_plane_metrics_with_derived(*args, relative_pressure_array=pressure)
    events = []
    process = augment_plane_metrics_with_derived(*args, relative_pressure_array=pressure,
        use_multithread=True, max_workers=2, progress_callback=events.append)
    def equivalent(left, right):
        if isinstance(left, dict):
            assert left.keys() == right.keys()
            for key in left:
                equivalent(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            assert len(left) == len(right)
            for a, b in zip(left, right):
                equivalent(a, b)
        elif isinstance(left, (np.ndarray, np.number, int, float)):
            np.testing.assert_allclose(left, right, rtol=1e-6, atol=1e-7, equal_nan=True)
        else:
            assert left == right
    equivalent(serial, process)
    assert [p["plane_index"] for p in process[1]] == [0, 1]
    assert events[-1]["current"] == 2


def test_streaming_video_cancellation_preserves_existing_mp4(monkeypatch, tmp_path):
    from autoflow.task_control import CancellationToken, TaskCancelled, task_scope
    target = tmp_path / "existing.mp4"
    target.write_bytes(b"previous complete export")
    token = CancellationToken()
    closed = []
    class Writer:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            closed.append(True)
        def append_data(self, frame):
            assert frame.shape == (8, 8, 3)
    def frames():
        yield np.zeros((8, 8, 3), dtype=np.uint8)
        token.cancel()
        yield np.ones((8, 8, 3), dtype=np.uint8)
    monkeypatch.setattr("autoflow.rendering.videos.imageio.get_writer", lambda *args, **kwargs: Writer())
    with task_scope(token), pytest.raises(TaskCancelled):
        _write_video(frames, target)
    assert target.read_bytes() == b"previous complete export"
    assert closed == [True]
    assert not list(tmp_path.glob(".autoflow_video_*"))


_qt_smoke_application = None


def _smoke_qt_application(monkeypatch):
    global _qt_smoke_application
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6 import QtWidgets
    _qt_smoke_application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    return _qt_smoke_application


def test_progress_dialog_blocks_other_windows_and_keeps_activity_until_cancel_stops(monkeypatch):
    app = _smoke_qt_application(monkeypatch)
    from PySide6 import QtCore, QtGui, QtTest, QtWidgets
    from autoflow.ui.progress import TaskProgressDialog
    main = QtWidgets.QWidget()
    button = QtWidgets.QPushButton("Other operation", main)
    layout = QtWidgets.QVBoxLayout(main)
    layout.addWidget(button)
    operations = []
    button.clicked.connect(lambda: operations.append("clicked"))
    shortcut = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+R"), main)
    shortcut.setContext(QtCore.Qt.ApplicationShortcut)
    shortcut.activated.connect(lambda: operations.append("shortcut"))
    main.show()
    dialog = TaskProgressDialog("Processing", "Computing planes", main)
    dialog.show()
    try:
        dialog.update_progress({"current": 1, "total": 4, "detail_current": 3, "detail_total": 20})
        phase = dialog.activity.phase
        QtTest.QTest.qWait(240)
        assert dialog.activity.phase > phase
        assert dialog.bar.value() == 1 and dialog.detail_bar.value() == 3
        QtTest.QTest.mouseClick(button, QtCore.Qt.LeftButton)
        QtTest.QTest.keyClick(main, QtCore.Qt.Key_R, QtCore.Qt.ControlModifier)
        main.close()
        assert operations == [] and main.isVisible()
        QtTest.QTest.keyClick(dialog, QtCore.Qt.Key_Escape)
        assert dialog.isVisible() and not dialog.cancel_token.cancelled
        dialog.close()
        assert dialog.cancel_token.cancelled and dialog.isVisible()
        QtTest.QTest.mouseClick(button, QtCore.Qt.LeftButton)
        assert operations == []
        dialog.finish()
        app.processEvents()
        QtTest.QTest.mouseClick(button, QtCore.Qt.LeftButton)
        assert operations == ["clicked"]
    finally:
        dialog.finish()
        main.close()


def test_pipeline_cancel_retains_completed_steps_without_partial_workspace(monkeypatch):
    _smoke_qt_application(monkeypatch)
    monkeypatch.setenv("AUTOFLOW_SSH_RENDERING", "1")
    from autoflow.ui.app import _PipelineTaskWorker
    from autoflow.core.pipeline import StepResult
    ws = Workspace()
    ws.flow_raw = np.zeros((2, 2, 2, 1, 3), dtype=np.float32)
    worker = None
    class Engine:
        def run_step(self, workspace, step, log, progress_callback=None):
            workspace.flow_raw = np.full_like(workspace.flow_raw, 1 if step == StepId.GENERATE_PCMRA else 2)
            workspace.pipeline.mark_done(step)
            if step == StepId.REMOVE_NOISE:
                worker.cancel_token.cancel()
            return StepResult(step)
    worker = _PipelineTaskWorker(Engine(), ws, [StepId.GENERATE_PCMRA, StepId.REMOVE_NOISE])
    errors = []
    worker.failed.connect(errors.append)
    worker.run()
    assert errors == ["Cancelled"]
    assert worker.completed_steps == [StepId.GENERATE_PCMRA]
    assert np.all(ws.flow_raw == 1)
    assert ws.pipeline.is_done(StepId.GENERATE_PCMRA)
    assert not ws.pipeline.is_done(StepId.REMOVE_NOISE)


def test_cancelled_plane_h5_export_preserves_previous_complete_file(tmp_path):
    from autoflow.algorithms.metrics import save_plane_pixelwise_h5
    from autoflow.task_control import CancellationToken, TaskCancelled, task_scope
    path = tmp_path / "plane_metrics_pixelwise.h5"
    with h5py.File(path, "w") as saved:
        saved["completed"] = [123]
    token = CancellationToken()
    def payloads():
        yield {"plane_index": 0, "timepoints": [{"time_index": 0, "relative_pressure_Pa": [1., 2.]}]}
        token.cancel()
        yield {"plane_index": 1}
    with task_scope(token), pytest.raises(TaskCancelled):
        save_plane_pixelwise_h5(path, payloads())
    with h5py.File(path) as saved:
        assert saved["completed"][0] == 123
        assert "planes" not in saved
    assert not list(tmp_path.glob(".autoflow_plane_*"))
