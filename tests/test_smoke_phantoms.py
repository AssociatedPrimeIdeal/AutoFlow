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


def _verify_hierarchy_browser_phantom(ws, report, monkeypatch):
    from autoflow.core.models import ObjectKind
    from types import SimpleNamespace, MethodType
    from PySide6 import QtCore, QtWidgets
    import autoflow.ui.app as ui_app
    _smoke_qt_application(monkeypatch)
    browser = SimpleNamespace(workspace=ws)
    for name in ("_browser_path_label", "_browser_make_checkable", "_browser_add_path_node", "_browser_add_leaf",
                 "_browser_subgroup_nodes", "_browser_apply_path_hierarchy", "_browser_add_joint_hierarchies",
                 "_browser_sort_path_nodes", "_browser_sort_path_children",
                 "_iter_browser_leaf_items", "_browser_check_state_for_item"):
        setattr(browser, name, MethodType(getattr(ui_app.MainWindow, name), browser))
    top = QtWidgets.QTreeWidgetItem(["phantom", "Group"])
    group = {"name": "phantom", "item": top, "types": {}, "paths": {}}
    ws.path_info = [{"hierarchy": record} for record in report["paths"]]
    for record in report["paths"]:
        index = record["path_index"]
        uid = ws.add_object(f"path {index}", ObjectKind.BRANCH, f"smooth_path_phantom_{index}", group_name="phantom")
        node = browser._browser_add_path_node(group, index)
        browser._browser_add_leaf(node, ws.scene_objects[uid])
    browser._browser_apply_path_hierarchy(group)
    browser._browser_sort_path_nodes(group["types"]["Paths"])
    leaves = list(browser._iter_browser_leaf_items(top))
    assert len(leaves) == len(report["paths"])  # References never duplicate objects/visibility controls.
    for record in report["paths"]:
        node = group["paths"]["phantom", record["path_index"]]
        if record["loop_id"]:
            assert node.parent().data(0, ui_app._BROWSER_NODE_ROLE) == "loop"
        elif record["display_parent_path_index"] is not None:
            parent = record["display_parent_path_index"]
            expected = group["paths"]["phantom", parent]
            if report["paths"][parent]["loop_id"]:
                expected = expected.parent()
            assert node.parent() is expected
        else:
            assert node.parent().data(0, ui_app._BROWSER_NODE_ROLE) == "subgroup"
        ancestor = node.parent()
        while ancestor is not None and ancestor.data(0, ui_app._BROWSER_NODE_ROLE) != "subgroup":
            ancestor = ancestor.parent()
        assert ancestor is not None
        assert ancestor.text(0).startswith(f"Subgroup {record['subgroup_index'] + 1} ·")
        for child in [node.child(i) for i in range(node.childCount())]:
            if child.data(0, ui_app._BROWSER_NODE_ROLE) == "path_reference":
                assert not child.flags() & QtCore.Qt.ItemIsUserCheckable
    browser.tree_objects = QtWidgets.QTreeWidget()
    ws.graph_hierarchy = {"joint_systems": [report]}
    for record in report["paths"]:
        ws.add_object("joint", ObjectKind.BRANCH,
                      f"hierarchy_path_{report['group_name']}_{record['path_index']}", visible=False)
    browser._browser_add_joint_hierarchies()
    joint = browser.tree_objects.topLevelItem(0)
    assert joint.childCount() == len(report["subgroups"])
    assert len(list(browser._iter_browser_leaf_items(joint))) == len(report["paths"])
    for index in range(joint.childCount()):
        assert joint.child(index).data(0, ui_app._BROWSER_NODE_ROLE) == "subgroup"


def test_graph_hierarchy_phantom_multiple_inlets_merge_and_label_boundaries(monkeypatch):
    from autoflow.algorithms.hierarchy import analyze_group_graph, graph_edge_chains
    graph = GraphData(points=np.array([[2, 8, 3], [12, 2, 3], [7, 8, 3],
                                      [10, 14, 3], [17, 8, 3], [25, 8, 3]], dtype=float),
                      edges=np.array([[0, 2], [2, 3], [2, 4], [1, 4], [4, 5]]))
    flow = np.zeros((30, 20, 7, 2, 3), dtype=np.float32)
    flow[..., 0] = 20
    labels = np.ones(flow.shape[:3], dtype=np.int16) * 14
    labels[:8] = 15
    labels[:, :4] = 16
    chains = graph_edge_chains(graph)
    original = graph.edges.copy()
    report = analyze_group_graph(graph, chains, flow=flow, segmentation_labels=labels,
                                label_map={"PV": 14, "SMV": 15, "SV": 16}, group_name="portal")
    records = {(r["source_node"], r["target_node"]): r for r in report["paths"]}
    assert records[0, 2]["group_level"] == records[1, 4]["group_level"] == 1
    assert records[2, 4]["group_level"] == 2
    merged = records[4, 5]
    assert merged["group_level"] == 3 and len(merged["parent_path_indices"]) == 2
    assert len(merged["root_path_indices"]) == 2
    assert {r["kind"] for r in report["junctions"]} >= {"inlet", "outlet", "split", "merge"}
    assert records[2, 4]["label_local_levels"]["14"] == 0
    assert merged["label_local_levels"]["14"] == 1
    np.testing.assert_array_equal(graph.edges, original)
    _verify_hierarchy_browser_phantom(Workspace(), report, monkeypatch)


@pytest.mark.parametrize("with_terminals", [True, False])
def test_graph_hierarchy_phantom_directed_cycle_keeps_edges_and_closed_rings(with_terminals, monkeypatch):
    from autoflow.algorithms.hierarchy import analyze_group_graph, graph_edge_chains
    points = [[8, 8, 3], [18, 8, 3], [18, 18, 3], [8, 18, 3]]
    edges = [[0, 1], [1, 2], [2, 3], [3, 0]]
    if with_terminals:
        points += [[3, 8, 3], [23, 18, 3]]
        edges += [[4, 0], [2, 5]]
    graph = GraphData(points=np.asarray(points, dtype=float), edges=np.asarray(edges))
    x, y, _ = np.indices((28, 28, 7))
    flow = np.zeros((28, 28, 7, 1, 3))
    flow[..., 0, 0] = -(y - 13)
    flow[..., 0, 1] = x - 13
    if with_terminals:
        flow[x > 18, 0, 0] = 20
    chains = graph_edge_chains(graph)
    covered = [tuple(sorted((a, b))) for chain in chains for a, b in zip(chain, chain[1:])]
    assert sorted(covered) == sorted(tuple(sorted(edge)) for edge in edges)
    report = analyze_group_graph(graph, chains, flow=flow, group_name="ring")
    assert report["undirected_cycle_rank"] == 1 and len(report["loops"]) == 1
    if with_terminals:
        assert [r["group_level"] for r in report["paths"] if r["is_inlet"]] == [1]
        assert all(r["group_level"] == 2 for r in report["paths"] if r["loop_id"])
        assert [r["group_level"] for r in report["paths"] if r["is_outlet"]] == [3]
        _verify_hierarchy_browser_phantom(Workspace(), report, monkeypatch)
    else:
        assert len(chains) == 1 and chains[0][0] == chains[0][-1]
        assert report["paths"][0]["group_level"] is None
        assert report["paths"][0]["level_status"] == "unrooted_loop"


def test_graph_hierarchy_phantom_missing_direction_does_not_invent_roots():
    from autoflow.algorithms.hierarchy import analyze_group_graph, graph_edge_chains
    graph = GraphData(points=np.array([[2., 2, 2], [8, 2, 2], [12, 5, 2], [12, 1, 2]]),
                      edges=np.array([[0, 1], [1, 2], [1, 3]]))
    report = analyze_group_graph(graph, graph_edge_chains(graph), flow=np.zeros((15, 8, 5, 1, 3)))
    assert all(r["group_level"] is None and r["level_status"] == "unoriented" for r in report["paths"])
    assert not report["inlet_path_indices"] and not report["outlet_path_indices"]


def test_graph_hierarchy_phantom_connected_subgroups_grade_independently(monkeypatch):
    from autoflow.algorithms.hierarchy import analyze_group_graph, graph_edge_chains
    graph = GraphData(points=np.array([
        [2, 8, 3], [12, 2, 3], [7, 8, 3], [10, 14, 3], [17, 8, 3], [25, 8, 3],
        [2, 4, 6], [8, 4, 6], [15, 2, 6], [15, 6, 6], [3, 22, 8], [10, 22, 8]], dtype=float),
        edges=np.array([[0, 2], [2, 3], [2, 4], [1, 4], [4, 5], [6, 7], [7, 8], [7, 9], [10, 11]]))
    flow = np.zeros((30, 26, 10, 2, 3), dtype=np.float32)
    flow[:, :, :7, :, 0] = 20
    report = analyze_group_graph(graph, graph_edge_chains(graph), flow=flow, group_name="phantom")
    assert report["grading_scope"] == "connected_subgroup"
    first, second, uncertain = report["subgroups"]
    assert len(first["inlet_path_indices"]) == 2 and first["maximum_level"] == 3
    assert len(second["inlet_path_indices"]) == 1 and second["maximum_level"] == 2
    assert uncertain["maximum_level"] is None and uncertain["unresolved_path_count"] == 1
    assert [s["subgroup_id"] for s in report["subgroups"]] == [f"phantom:subgroup_{i}" for i in (1, 2, 3)]
    by_index = {r["path_index"]: r for r in report["paths"]}
    for subgroup in report["subgroups"]:
        for index in subgroup["path_indices"]:
            record = by_index[index]
            assert record["subgroup_id"] == subgroup["subgroup_id"]
            assert record["subgroup_level"] == record["group_level"]
            assert set(record["parent_path_indices"]) <= set(subgroup["path_indices"])
            if record["is_inlet"]:
                assert record["subgroup_level"] == 1
    ws = Workspace()
    _verify_hierarchy_browser_phantom(ws, report, monkeypatch)
    ws.centerline_paths = [graph.points[chain] for chain in graph_edge_chains(graph)]
    ws.graph_hierarchy.update(paths=report["paths"], groups=[report])
    flow_report = build_quality_report(ws)["flow_hierarchy"]
    assert len(flow_report["subgroups"]) == 3
    assert {root["subgroup_id"] for root in flow_report["roots"]} == {s["subgroup_id"] for s in report["subgroups"]}


def test_graph_hierarchy_phantom_joint_analysis_and_snapshot(tmp_path):
    from autoflow.algorithms.hierarchy import build_workspace_hierarchy
    ws = build_workspace(AutoFlowConfig.from_config_dir())
    # Thinning produces diagonal corner steps; this ring phantom explicitly
    # explicitly tests the 26-neighbor topology.
    ws.skeleton_params.graph_connectivity = 26
    ws.paths.output_dir = str(tmp_path)
    labels = np.zeros((32, 32, 7), dtype=np.int16)
    # A closed voxel ring with two source groups; each separate group is open.
    labels[8:24, 8, 3] = 17
    labels[23, 8:24, 3] = 17
    labels[8:24, 23, 3] = 21
    labels[8, 8:24, 3] = 21
    ws.segmask_raw = labels
    ws.flow_raw = np.zeros(labels.shape + (1, 3))
    x, y, _ = np.indices(labels.shape)
    ws.flow_raw[..., 0, 0] = -(y - 15.5)
    ws.flow_raw[..., 0, 1] = x - 15.5
    ws.skeleton_params.remove_small_cc = False
    ws.skeleton_params.gaussian_enabled = False
    ws.skeleton_params.min_edge_points = 0
    engine = PipelineEngine()
    assert engine._step_generate_skeleton(ws).success
    assert engine._step_generate_graph(ws).success
    hierarchy = ws.graph_hierarchy
    assert hierarchy["schema"] == "autoflow.graph_hierarchy.v1"
    assert "ascending level" in hierarchy["path_numbering"]
    joint = hierarchy["joint_systems"][0]
    assert joint["view_only"] and len(joint["source_group_names"]) == 2
    assert joint["undirected_cycle_rank"] >= 1
    assert joint["paths"] and all(item["group_level"] is None for item in joint["paths"])
    for report in [*hierarchy["groups"], *hierarchy["joint_systems"]]:
        by_index = {record["path_index"]: record for record in report["paths"]}
        for subgroup in report["subgroups"]:
            keys = [(by_index[index]["subgroup_level"] is None, by_index[index]["subgroup_level"] or 0)
                    for index in subgroup["path_indices"]]
            assert keys == sorted(keys)
    assert json.loads((tmp_path / "graph_hierarchy.json").read_text())["schema"] == hierarchy["schema"]
    assert all("hierarchy" in info for info in ws.path_info)
    assert any(obj.data_key.startswith("hierarchy_path_") for obj in ws.scene_objects.values())
    restored = Workspace()
    restored.restore_dict(ws.snapshot_dict())
    assert restored.graph_hierarchy == hierarchy
    assert build_quality_report(ws)["graph_hierarchy"] == hierarchy
    import pyvista as pv
    from autoflow.ui.viewer import SceneController
    plotter = pv.Plotter(off_screen=True)
    try:
        scene = SceneController(plotter, ws, lambda *_: None)
        native = scene._build_dataset("smooth_path_intracranial_anterior_arteries_0")
        joint_curve = scene._build_dataset("hierarchy_path_cerebral_arterial_system_0")
        assert native.n_points > 0 and joint_curve.n_points > 0
    finally:
        plotter.close()
    engine._clear_graph_state(restored)
    assert not restored.graph_hierarchy
    assert not any(obj.data_key.startswith("hierarchy_path_") for obj in restored.scene_objects.values())


def test_graph_hierarchy_phantom_path_numbers_follow_depth_and_branch_references(monkeypatch):
    import autoflow.algorithms.branch as branch_module
    from autoflow.algorithms.branch import segment_vessels_from_graph_and_mask
    graph = GraphData(points=np.array([[2, 8, 3], [12, 2, 3], [7, 8, 3],
                                      [10, 14, 3], [17, 8, 3], [25, 8, 3]], dtype=float),
                      edges=np.array([[0, 2], [2, 3], [2, 4], [1, 4], [4, 5]]))
    mask = np.ones((30, 20, 7), dtype=bool)
    flow = np.zeros(mask.shape + (2, 3), dtype=np.float32)
    flow[..., 0] = 20
    branch_labels, paths, chains, infos, forks = segment_vessels_from_graph_and_mask(
        mask, graph, (1, 1, 1), flow_xyzt3=flow, segmask_binary_4d=mask[..., None])
    assert [info["path_index"] for info in infos] == list(range(5))
    assert [info["hierarchy"]["subgroup_level"] for info in infos] == [1, 1, 2, 2, 3]
    for index, (path, chain, info) in enumerate(zip(paths, chains, infos)):
        np.testing.assert_array_equal(path, graph.points[chain])
        assert info["start_node"] == chain[0] and info["end_node"] == chain[-1]
        for node in chain:
            if np.count_nonzero(graph.edges == node) == 1:
                voxel = np.rint(graph.points[node]).astype(int)
                assert branch_labels[tuple(voxel)] == index + 1
        for parent in info["hierarchy"]["parent_path_indices"]:
            assert parent < index
    merged = infos[-1]["hierarchy"]
    assert len(merged["parent_path_indices"]) == 2
    assert any(4 in fork["right"] and len(fork["left"]) == 2 for fork in forks)
    # Number changes must preserve voxel ownership, including nearest-node ties.
    monkeypatch.setattr(branch_module, "hierarchy_path_order",
                        lambda report: list(range(len(report["paths"]))))
    old_labels, _, old_chains, _, _ = segment_vessels_from_graph_and_mask(
        mask, graph, (1, 1, 1), flow_xyzt3=flow, segmask_binary_4d=mask[..., None])
    remap = np.zeros(len(old_chains) + 1, dtype=np.int16)
    for old, chain in enumerate(old_chains):
        remap[old + 1] = next(index + 1 for index, candidate in enumerate(chains) if candidate == chain)
    np.testing.assert_array_equal(branch_labels, remap[old_labels])


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
                                      posterior_label_values=[26], connectivity=26)
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


@pytest.mark.parametrize("morphology", ["closing", "dilation"])
def test_three_pass_special_skeleton_excludes_neighbour_labels_after_morphology(monkeypatch, morphology):
    import autoflow.algorithms.skeleton as skeleton_module
    from autoflow.algorithms.preprocess import preprocess_mask_for_skeleton

    # A thick trunk and three close, parallel branches. Closing or dilation
    # can refill the excluded neighbour at the inside corner of a junction.
    labels = np.zeros((18, 20, 24), dtype=np.int16)
    labels[3:12, 3:16, 3:9] = 1
    labels[6:9, 4:7, 9:20] = 5
    labels[6:9, 8:11, 9:20] = 6
    labels[6:9, 11:14, 9:20] = 7
    original_labels = labels.copy()
    special_values = [5, 6, 7]
    params = SkeletonParams(
        remove_small_cc=False,
        do_closing=morphology == "closing",
        dilation_iters=1 if morphology == "dilation" else 0,
        gaussian_enabled=False,
    )
    spacing = np.full(3, 2.4)
    # Ensure the phantom actually exercises growth into another label.
    leaked_voxels = 0
    for value in special_values:
        excluded = np.isin(labels, [other for other in special_values if other != value])
        processed = preprocess_mask_for_skeleton(np.isin(labels, [1, value]), params, spacing)
        leaked_voxels += int(np.count_nonzero(processed & excluded))
    assert leaked_voxels > 0

    pass_masks = []
    original_skeletonizer = skeleton_module.generate_skeleton_from_mask3d

    def record_skeleton_input(mask, resolution):
        pass_masks.append(mask.copy())
        return original_skeletonizer(mask, resolution)

    monkeypatch.setattr(skeleton_module, "generate_skeleton_from_mask3d", record_skeleton_input)
    points, skeleton_mask = generate_three_pass_special_skeleton(
        labels, [1, 5, 6, 7], special_values, params, spacing,
    )

    assert len(pass_masks) == len(special_values)
    for value, mask in zip(special_values, pass_masks):
        excluded = np.isin(labels, [other for other in special_values if other != value])
        assert not np.any(mask & excluded)
        assert np.any(mask & (labels == value))
    assert len(points) > 0 and np.any(skeleton_mask)
    assert np.array_equal(labels, original_labels)


@pytest.mark.parametrize("offset", [(1, 0, 0), (1, 1, 0), (1, 1, 1)])
def test_skeleton_graph_connectivity_phantom_respects_faces_edges_and_corners(offset):
    from autoflow.algorithms.graph import build_graph_from_points
    spacing = np.array([1.2, 1.2, 1.1])
    points = np.asarray([[0, 0, 0], offset], dtype=float) * spacing
    six = build_graph_from_points(points, spacing, connectivity=6)
    twenty_six = build_graph_from_points(points, spacing, connectivity=26)
    assert len(six.edges) == (1 if offset == (1, 0, 0) else 0)
    assert len(twenty_six.edges) == 1
    assert len(build_graph_from_points(points, spacing).edges) == 1


def test_skeleton_graph_connectivity_phantom_config_group_gui_and_snapshot():
    from PySide6 import QtWidgets
    from autoflow.ui.action_settings import ActionSettingsDialog, action_sections
    assert AutoFlowConfig().graph_connectivity == AutoFlowConfig.from_config_dir().graph_connectivity == 26
    assert SkeletonParams.from_dict({}).graph_connectivity == 26
    ws = build_workspace(AutoFlowConfig.from_config_dir(graph_connectivity=26))
    assert ws.skeleton_params.graph_connectivity == 26
    ws.skeleton_params.label_groups['custom'] = {'labels': [17], 'preprocess': {'graph_connectivity': 6}}
    assert ws.skeleton_params.params_for_group('custom').graph_connectivity == 6
    restored = Workspace()
    restored.restore_dict(ws.snapshot_dict())
    assert restored.skeleton_params.graph_connectivity == 26
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    dialog = ActionSettingsDialog('Graph', action_sections(StepId.GENERATE_GRAPH, ws), ws, ws)
    try:
        control = dialog.controls['skeleton_params.graph_connectivity']
        assert control.currentData() == 26
        control.setCurrentIndex(control.findData(6))
        assert dialog.values()['skeleton_params.graph_connectivity'] == 6
    finally:
        dialog.close()
        dialog.deleteLater()
        app.processEvents()


@pytest.mark.parametrize("connectivity", [18, True, 6.5])
def test_skeleton_graph_connectivity_phantom_rejects_invalid_values(connectivity):
    from autoflow.algorithms.graph import build_graph_from_points
    with pytest.raises(ValueError, match='graph_connectivity'):
        build_graph_from_points(np.zeros((0, 3)), [1, 1, 1], connectivity=connectivity)
    with pytest.raises(ValueError, match='graph_connectivity'):
        build_workspace(AutoFlowConfig(graph_connectivity=connectivity))


def _cycle_cleanup_phantom():
    points = np.array([[4, 4, 2], [14, 4, 2], [14, 14, 2], [4, 14, 2], [6, 5, 2], [5, 7, 2]], dtype=float)
    graph = GraphData(points=points, edges=np.array([[0, 1], [1, 2], [2, 3], [3, 0], [0, 4], [4, 5], [5, 0]]))
    labels = np.ones((20, 20, 5), dtype=np.int16) * 17
    labels[13:16, 3:6] = 30
    labels[13:16, 13:16] = 20
    labels[3:6, 13:16] = 26
    return graph, labels


def _uniform_cycle_flow(labels, vector=(1., 0., 0.)):
    return np.broadcast_to(np.asarray(vector), labels.shape + (1, 3)).copy()


@pytest.mark.parametrize("permuted", [False, True])
def test_intralabel_cycle_phantom_preserves_cross_label_ring_and_interfaces(permuted):
    import networkx as nx
    from autoflow.algorithms.graph import graph_to_networkx
    from autoflow.algorithms.graph_cycles import remove_intralabel_cycles
    graph, labels = _cycle_cleanup_phantom()
    if permuted:
        order = np.array([5, 2, 0, 4, 1, 3])
        graph = GraphData(points=graph.points[order], edges=np.argsort(order)[graph.edges[::-1, ::-1]])
    original_points, original_edges = graph.points.copy(), graph.edges.copy()
    cleaned, qc = remove_intralabel_cycles(graph, labels, flow=_uniform_cycle_flow(labels))
    assert len(nx.cycle_basis(graph_to_networkx(graph))) == 2
    assert len(nx.cycle_basis(graph_to_networkx(cleaned))) == 1
    assert qc['cycle_rank_before'] == 2 and qc['cycle_rank_after'] == 1
    assert qc['protected_edge_count'] == 4 and qc['stage'] == 'before_terminal_pruning'
    assert all(row['cycle_rank_after'] == 0 for row in qc['labels'])
    assert qc['interface_ports_preserved'] and qc['label_component_connectivity_preserved']
    remaining = {tuple(point) for point in cleaned.points}
    assert {(4., 4., 2.), (14., 4., 2.), (14., 14., 2.), (4., 14., 2.)} <= remaining
    for left, right in cleaned.edges:
        assert not np.array_equal(cleaned.points[left], cleaned.points[right])
    np.testing.assert_array_equal(graph.points, original_points)
    np.testing.assert_array_equal(graph.edges, original_edges)


@pytest.mark.parametrize('unknown', [False, True])
def test_intralabel_cycle_phantom_isolated_ring_and_unknown_label_policy(unknown):
    import networkx as nx
    from autoflow.algorithms.graph import graph_to_networkx
    from autoflow.algorithms.graph_cycles import remove_intralabel_cycles
    graph = GraphData(points=np.array([[3, 3, 2], [8, 3, 2], [8, 8, 2], [3, 8, 2]], dtype=float),
        edges=np.array([[0, 1], [1, 2], [2, 3], [3, 0]]))
    labels = np.full((12, 12, 5), 0 if unknown else 31, dtype=np.int16)
    cleaned, qc = remove_intralabel_cycles(graph, labels, flow=_uniform_cycle_flow(labels))
    if unknown:
        assert cleaned is graph and qc['protected_edge_count'] == 4
        assert qc['unknown_node_count'] == 4 and not qc['removed_paths']
    else:
        # Removing its only complete path would erase the whole component.
        assert cleaned is graph and len(cleaned.points) == 4
        assert qc['cycle_rank_after'] == 1 and qc['removed_edge_count'] == 0
        assert qc['rejected_paths'][0]['reason'] == 'preserve_label_component_or_interface'


def test_intralabel_cycle_phantom_does_not_cut_edge_crossing_another_label():
    from autoflow.algorithms.graph_cycles import remove_intralabel_cycles
    graph = GraphData(points=np.array([[2, 2, 2], [8, 2, 2], [5, 8, 2]], dtype=float),
        edges=np.array([[0, 1], [1, 2], [2, 0]]))
    labels = np.ones((11, 11, 5), dtype=np.int16)
    labels[5, 2, 2] = 2  # Endpoints alone would miss this interface.
    cleaned, qc = remove_intralabel_cycles(graph, labels, flow=_uniform_cycle_flow(labels))
    assert cleaned is graph and qc['protected_edge_count'] == 1
    assert qc['cycle_rank_after'] == 1


def test_intralabel_cycle_phantom_retains_component_of_two_rings_sharing_one_junction():
    import networkx as nx
    from autoflow.algorithms.graph import graph_to_networkx
    from autoflow.algorithms.graph_cycles import remove_intralabel_cycles
    graph = GraphData(points=np.array([[5, 5, 2], [7, 5, 2], [6, 7, 2], [3, 5, 2], [4, 7, 2]], dtype=float),
        edges=np.array([[0, 1], [1, 2], [2, 0], [0, 3], [3, 4], [4, 0]]))
    labels = np.ones((12, 12, 5), dtype=np.int16)
    cleaned, qc = remove_intralabel_cycles(graph, labels, flow=_uniform_cycle_flow(labels))
    assert nx.is_connected(graph_to_networkx(cleaned))
    assert qc['cycle_rank_before'] == 2 and qc['cycle_rank_after'] == 1
    assert len(qc['removed_paths']) == len(qc['rejected_paths']) == 1


def test_intralabel_cycle_phantom_respects_spacing_origin_and_input_permutation():
    from autoflow.algorithms.graph_cycles import remove_intralabel_cycles
    graph, labels = _cycle_cleanup_phantom()
    spacing, origin = np.array([1.2, 2., 3.]), np.array([40., -20., 10.])
    graph.points = graph.points * spacing + origin
    first, qc = remove_intralabel_cycles(graph, labels, spacing, origin=origin, flow=_uniform_cycle_flow(labels))
    order = np.array([5, 2, 0, 4, 1, 3])
    shuffled = GraphData(points=graph.points[order], edges=np.argsort(order)[graph.edges[::-1, ::-1]])
    second, other = remove_intralabel_cycles(shuffled, labels, spacing, origin=origin, flow=_uniform_cycle_flow(labels))
    def physical_edges(g):
        return {tuple(sorted((tuple(g.points[a]), tuple(g.points[b])))) for a, b in g.edges}
    assert physical_edges(first) == physical_edges(second)
    assert qc['cycle_rank_after'] == other['cycle_rank_after'] == 1


def test_intralabel_cycle_phantom_keeps_diagonal_edge_ownership_across_background_samples():
    import networkx as nx
    from autoflow.algorithms.graph import graph_to_networkx
    from autoflow.algorithms.graph_cycles import remove_intralabel_cycles
    graph = GraphData(points=np.array([[2.1, 2.4, 2], [3, 3, 2], [1, 2, 2], [2, 1, 2]], dtype=float),
        edges=np.array([[0, 1], [1, 2], [2, 0], [0, 3]]))
    labels = np.zeros((6, 6, 5), dtype=np.int16)
    for point in np.rint(graph.points).astype(int):
        labels[tuple(point)] = 17
    # A 26-neighbor edge can traverse background in its interior. It remains
    # an existing same-label edge; introducing another positive label protects it.
    cleaned, qc = remove_intralabel_cycles(graph, labels, flow=_uniform_cycle_flow(labels))
    assert nx.is_tree(graph_to_networkx(cleaned)) and qc['cycle_rank_after'] == 0
    assert qc['protected_edge_count'] == 0


def test_triangle_junction_phantom_contracts_across_label_interfaces():
    from autoflow.algorithms.graph import remove_triangle_cycles
    graph = GraphData(points=np.array([[3, 3, 2], [4, 3, 2], [3, 4, 2],
        [2, 2, 2], [5, 2, 2], [2, 5, 2]], dtype=float),
        edges=np.array([[0, 1], [0, 2], [1, 2], [0, 3], [1, 4], [2, 5]]))
    labels = np.ones((8, 8, 5), dtype=np.int16)
    labels[4, 3, 2] = 2
    contracted = remove_triangle_cycles(graph, segmentation_labels=labels)
    assert len(contracted.points) == len(graph.points) - 2
    labels[4, 3, 2] = 1
    contracted = remove_triangle_cycles(graph, segmentation_labels=labels)
    assert len(contracted.points) == len(graph.points) - 2


def test_intralabel_cycle_phantom_pipeline_cleans_before_short_pruning_and_persists(tmp_path, monkeypatch):
    import networkx as nx
    import autoflow.core.pipeline as pipeline_module
    from autoflow.algorithms.graph import graph_to_networkx
    points = np.array([[x, 3, 2] for x in range(1, 9)] + [[4, 15, 2], [5, 15, 2]], dtype=float)
    graph = GraphData(points=points, edges=np.array([[i, i + 1] for i in range(7)] + [[2, 8], [8, 9], [9, 5]]))
    ws = Workspace()
    ws.paths.output_dir = str(tmp_path)
    ws.skeleton_params.label_map = {'vessel': 1}
    ws.skeleton_params.label_groups = {'phantom': {'labels': [1]}}
    ws.skeleton_params.min_edge_points = 5
    ws.segmask_labels_3d = np.ones((12, 20, 5), dtype=np.int16)
    ws.segmask_raw = ws.segmask_labels_3d.copy()
    ws.segmask_3d = ws.segmask_labels_3d > 0
    ws.flow_raw = np.zeros(ws.segmask_3d.shape + (1, 3))
    ws.flow_raw[..., 0] = 20
    ws.group_order = ['phantom']
    ws.skeleton_points = points.copy()
    ws.multilabel_groups = {'phantom': {'labels': [1], 'skeleton_points': points.copy(),
        'segmask_3d': ws.segmask_3d.copy(), 'segmask_binary': ws.segmask_3d[..., None]}}
    monkeypatch.setattr(pipeline_module, 'build_graph_from_points', lambda *a, **k: graph)
    called = []
    def check_order(cleaned, **kwargs):
        assert nx.is_forest(graph_to_networkx(cleaned))
        called.append(True)
        return remove_short_terminal_branches(cleaned, **kwargs)
    monkeypatch.setattr(pipeline_module, 'remove_short_terminal_branches', check_order)
    assert PipelineEngine()._step_generate_graph(ws).success
    assert called == [True]
    assert {(1., 3., 2.), (8., 3., 2.)} <= {tuple(p) for p in ws.graph.points}
    qc = ws.graph_hierarchy['groups'][0]['cycle_cleanup']
    assert qc['stage'] == 'before_terminal_pruning' and qc['cycle_rank_after'] == 0
    restored = Workspace()
    restored.restore_dict(ws.snapshot_dict())
    assert restored.multilabel_groups['phantom']['cycle_cleanup'] == qc
    assert restored.graph_hierarchy['groups'][0]['cycle_cleanup'] == qc
    exported = json.loads((tmp_path / 'graph_hierarchy.json').read_text())
    assert exported['groups'][0]['cycle_cleanup'] == qc


def _flow_detour_phantom():
    points = np.array([[2, 4, 2], [4, 4, 2], [12, 4, 2], [14, 4, 2],
                       [4, 16, 2], [12, 16, 2]], dtype=float)
    graph = GraphData(points=points, edges=np.array([[0, 1], [1, 2], [2, 3], [1, 4], [4, 5], [5, 2]]))
    return graph, np.ones((18, 25, 5), dtype=np.int16)


@pytest.mark.parametrize("sign", [1, -1])
@pytest.mark.parametrize("threshold,expected_cycles", [(.2, 1), (.25, 1), (.3, 0)])
def test_flow_cycle_phantom_threshold_is_strict_and_reverse_alignment_is_kept(sign, threshold, expected_cycles):
    from autoflow.algorithms.graph_cycles import remove_intralabel_cycles
    graph, labels = _flow_detour_phantom()
    cleaned, qc = remove_intralabel_cycles(graph, labels, flow=_uniform_cycle_flow(labels, (sign, 0., 0.)),
                                           flow_threshold=threshold)
    assert qc['cycle_rank_after'] == expected_cycles
    assert qc['algorithm'] == 'flow_alignment_filter' and qc['flow_threshold'] == threshold
    assert qc['preserve_connectivity'] and qc['original_endpoints_preserved']
    assert {(2., 4., 2.), (4., 4., 2.), (12., 4., 2.), (14., 4., 2.)} <= {tuple(p) for p in cleaned.points}
    if expected_cycles == 0:
        assert len(qc['removed_paths']) == 1
        # Uniform resampling approximates the two corners of the detour.
        assert .25 < qc['removed_paths'][0]['direction_confidence'] < .27
        assert qc['removed_paths'][0]['length_mm'] == pytest.approx(32.)
        assert 'mean_radius_mm' not in qc['removed_paths'][0]


@pytest.mark.parametrize('threshold,vector,deleted', [(.6,(3.,4.,0.),False),(.60001,(3.,4.,0.),True),(0.,(0.,1.,0.),False)])
def test_flow_cycle_phantom_exact_alignment_threshold_and_zero_disable(threshold, vector, deleted):
    from autoflow.algorithms.graph_cycles import remove_intralabel_cycles
    graph = GraphData(points=np.array([[3.,3.,2.],[8.,3.,2.]]),edges=np.array([[0,1]]))
    labels = np.ones((12,8,5),dtype=np.int16)
    cleaned,qc=remove_intralabel_cycles(graph,labels,flow=_uniform_cycle_flow(labels,vector),
        flow_threshold=threshold,preserve_connectivity=False)
    assert bool(qc['removed_paths']) == deleted
    assert len(cleaned.edges) == (0 if deleted else 1)


@pytest.mark.parametrize("kind", ["missing", "zero", "invalid_mask", "nan"])
def test_flow_cycle_phantom_skips_unavailable_valid_signal(kind):
    from autoflow.algorithms.graph_cycles import remove_intralabel_cycles
    graph, labels = _flow_detour_phantom()
    flow = None if kind == 'missing' else _uniform_cycle_flow(labels, (0., 0., 0.) if kind == 'zero' else (1., 0., 0.))
    mask = np.zeros(labels.shape + (1,), dtype=bool) if kind == 'invalid_mask' else None
    if kind == 'nan':
        flow[:] = np.nan
    cleaned, qc = remove_intralabel_cycles(graph, labels, flow=flow, temporal_mask=mask)
    assert cleaned is graph and qc['cycle_rank_after'] == 1
    assert qc['status'] == 'skipped' and qc['reason'] == 'flow_signal_unavailable'
    assert qc['flow_unavailable_path_count'] == qc['evaluated_path_count']


@pytest.mark.parametrize("preserve", [True, False])
def test_flow_cycle_phantom_connectivity_toggle_controls_terminal_path_removal(preserve):
    from autoflow.algorithms.graph_cycles import remove_intralabel_cycles
    graph, labels = _flow_detour_phantom()
    graph.points = np.vstack([graph.points, [4, 1, 2]])
    graph.edges = np.vstack([graph.edges, [1, 6]])
    cleaned, qc = remove_intralabel_cycles(graph, labels, flow=_uniform_cycle_flow(labels), preserve_connectivity=preserve)
    remaining = {tuple(p) for p in cleaned.points}
    assert ((4., 1., 2.) in remaining) == preserve
    assert qc['cycle_rank_after'] == 0
    assert qc['original_endpoints_preserved'] == preserve
    assert bool(qc['rejected_paths']) == preserve
    assert all(r['direction_confidence'] < .3 for r in qc['removed_paths'])


def test_flow_cycle_phantom_config_group_gui_cli_and_snapshot_roundtrip():
    from PySide6 import QtWidgets
    from autoflow.cli import build_parser
    from autoflow.ui.action_settings import ActionSettingsDialog, action_sections
    default = AutoFlowConfig.from_config_dir()
    assert default.cycle_flow_threshold == .3 and default.cycle_preserve_connectivity
    ws = build_workspace(AutoFlowConfig.from_config_dir(cycle_flow_threshold=.2, cycle_preserve_connectivity=False))
    ws.skeleton_params.label_groups['custom'] = {'labels': [1], 'preprocess': {'cycle_flow_threshold': .7, 'cycle_preserve_connectivity': True}}
    inherited = ws.skeleton_params.params_for_group('custom')
    assert inherited.cycle_flow_threshold == .7 and inherited.cycle_preserve_connectivity
    restored = Workspace();restored.restore_dict(ws.snapshot_dict())
    assert restored.skeleton_params.cycle_flow_threshold == .2 and not restored.skeleton_params.cycle_preserve_connectivity
    args = build_parser().parse_args(['case.h5', '--cycle-flow-threshold', '.4', '--no-cycle-preserve-connectivity'])
    assert args.cycle_flow_threshold == .4 and not args.cycle_preserve_connectivity
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    dialog = ActionSettingsDialog('Graph', action_sections(StepId.GENERATE_GRAPH, ws), ws, ws)
    try:
        dialog.controls['skeleton_params.cycle_flow_threshold'].setValue(.45)
        dialog.controls['skeleton_params.cycle_preserve_connectivity'].setChecked(True)
        values = dialog.values()
        assert values['skeleton_params.cycle_flow_threshold'] == .45
        assert values['skeleton_params.cycle_preserve_connectivity'] is True
    finally:
        dialog.close();dialog.deleteLater();app.processEvents()


@pytest.mark.parametrize('threshold', [-.1, 1.1, np.nan, np.inf, True])
def test_flow_cycle_phantom_rejects_invalid_thresholds(threshold):
    from autoflow.algorithms.graph_cycles import remove_intralabel_cycles
    graph, labels = _flow_detour_phantom()
    with pytest.raises(ValueError, match='cycle_flow_threshold'):
        remove_intralabel_cycles(graph, labels, flow_threshold=threshold)
    with pytest.raises(ValueError, match='cycle_flow_threshold'):
        build_workspace(AutoFlowConfig(cycle_flow_threshold=threshold))


def test_flow_cycle_phantom_requires_boolean_connectivity_parameter():
    with pytest.raises(ValueError, match='cycle_preserve_connectivity'):
        SkeletonParams.from_dict({'cycle_preserve_connectivity': 'false'})


def test_special_branch_phantom_protects_abdominal_contacts_and_retains_trunk_interfaces():
    from autoflow.algorithms.graph import build_graph_from_points, _sample_graph_labels
    expected = {'RBCT', 'CCA', 'LBCT', 'HA', 'SMA', 'LRA', 'RRA'}
    cfg = AutoFlowConfig.from_config_dir()
    ws = build_workspace(cfg)
    assert set(cfg.special_contact_labels) == set(SkeletonParams.from_dict({}).special_contact_labels) == expected
    names = ['DAO', 'HA', 'SMA', 'LRA', 'RRA']
    values = [ws.skeleton_params.label_map[n] for n in names]
    voxels = np.array([[4, 4, 2], [5, 4, 2], [5, 5, 2], [4, 5, 2], [4, 4, 3]])
    labels = np.zeros((9, 9, 6), dtype=np.int16)
    for p, value in zip(voxels, values):labels[tuple(p)] = value
    spacing = np.full(3, 2.4)
    raw = build_graph_from_points(voxels * spacing, spacing, segmentation_labels=labels)
    protected = build_graph_from_points(voxels * spacing, spacing, segmentation_labels=labels, separated_label_values=values[1:])
    assert len(raw.edges) > len(protected.edges) == 4
    node_values = _sample_graph_labels(protected.points, labels, spacing)
    assert all(values[0] in node_values[[a,b]] for a,b in protected.edges)
    assert {int(v) for edge in protected.edges for v in node_values[edge]} == set(values)


def test_triangle_junction_phantom_uses_centroid_and_existing_external_edges():
    from autoflow.algorithms.graph import build_graph_from_points, graph_to_networkx

    spacing = np.array([1.2, 1.2, 1.1])
    voxels = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0],
                       [-1, -1, 0], [2, -1, 0], [-1, 2, 0]], dtype=float)
    points = voxels * spacing
    original = points.copy()
    graph = build_graph_from_points(points, spacing)
    assert graph.points.shape == (4, 3)
    np.testing.assert_allclose(graph.points[0], original[:3].mean(axis=0))
    np.testing.assert_array_equal(graph.points[1:], original[3:])
    assert {tuple(edge) for edge in graph.edges} == {(0, 1), (0, 2), (0, 3)}
    assert dict(graph_to_networkx(graph).degree()) == {0: 3, 1: 1, 2: 1, 3: 1}
    # Rebuilding adjacency after moving the junction would drop this branch.
    assert np.linalg.norm((graph.points[0] - graph.points[1]) / spacing) > np.sqrt(3)
    np.testing.assert_array_equal(points, original)


@pytest.mark.parametrize("permuted", [False, True])
def test_triangle_junction_phantom_preserves_external_ring_and_components(permuted):
    import networkx as nx
    from autoflow.algorithms.graph import graph_to_networkx, remove_triangle_cycles
    from autoflow.algorithms.hierarchy import graph_edge_chains

    points = np.array([[0, 1, 0], [1, 0, 1], [1, 1, 1],
                       [1, 1, -1], [1, -1, 2], [2, 1, 2],
                       [1, -1, 3], [2, 1, 3], [10, 0, 0],
                       [11, 0, 0], [20, 0, 0]], dtype=float)
    edges = np.array([[0, 1], [0, 2], [1, 2], [0, 3], [1, 4], [2, 5],
                      [4, 6], [6, 7], [7, 5], [8, 9]], dtype=int)
    if permuted:
        order = np.array([5, 2, 10, 1, 8, 4, 0, 9, 3, 7, 6])
        edges = np.argsort(order)[edges[::-1, ::-1]]
        points = points[order]
    graph = GraphData(points=points.copy(), edges=edges.copy())
    cleaned = remove_triangle_cycles(graph)
    before, after = graph_to_networkx(graph), graph_to_networkx(cleaned)
    assert len(cleaned.points) == len(points) - 2
    assert len(cleaned.edges) == len(edges) - 3
    assert nx.number_connected_components(before) == nx.number_connected_components(after) == 3
    assert len(nx.cycle_basis(before)) == 2 and len(nx.cycle_basis(after)) == 1
    junction = np.flatnonzero(np.all(np.isclose(cleaned.points, [2 / 3, 2 / 3, 2 / 3]), axis=1))
    assert len(junction) == 1 and after.degree(int(junction[0])) == 3
    neighbors = {tuple(cleaned.points[node]) for node in after[int(junction[0])]}
    assert neighbors == {(1., 1., -1.), (1., -1., 2.), (2., 1., 2.)}
    chains = graph_edge_chains(cleaned)
    assert sum(len(chain) - 1 for chain in chains) == len(cleaned.edges)
    assert any(chain[0] == chain[-1] for chain in chains)
    np.testing.assert_array_equal(graph.points, points)
    np.testing.assert_array_equal(graph.edges, edges)


@pytest.mark.parametrize("kind", ["closed_ring", "overlap", "parallel_connections"])
def test_triangle_junction_phantom_keeps_ambiguous_regions(kind):
    from autoflow.algorithms.graph import remove_triangle_cycles

    if kind == "closed_ring":
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        edges = [[0, 1], [0, 2], [1, 2]]
    elif kind == "overlap":
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
                           [-1, 0, 0], [2, 0, 0], [2, 1, 0]], dtype=float)
        edges = [[0, 1], [0, 2], [1, 2], [1, 3], [2, 3], [0, 4], [1, 5], [3, 6]]
    else:
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0],
                           [2, 0, 0], [3, 0, 0], [2, 1, 0],
                           [0, 2, 0], [2, 2, 0]], dtype=float)
        edges = [[0, 1], [0, 2], [1, 2], [3, 4], [3, 5], [4, 5],
                 [0, 3], [1, 4], [2, 6], [5, 7]]
    graph = GraphData(points=points, edges=np.asarray(edges, dtype=int))
    cleaned = remove_triangle_cycles(graph)
    np.testing.assert_array_equal(cleaned.points, graph.points)
    np.testing.assert_array_equal(cleaned.edges, graph.edges)


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

    # The default retains the three-edge arm and removes the one- and
    # two-edge terminal arms. This is a length threshold, not a deletion quota.
    default_filtered = remove_short_terminal_branches(graph)
    assert {tuple(edge) for edge in default_filtered.edges.tolist()} == {(0, 1), (1, 2), (2, 3)}
    np.testing.assert_array_equal(default_filtered.points, graph.points[:4])
    assert AutoFlowConfig().min_edge_points == 3
    assert AutoFlowConfig.from_config_dir().min_edge_points == 3



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


@pytest.mark.parametrize("face", ["min", "max"])
@pytest.mark.parametrize("hole", [False, True])
def test_fov_contact_phantom_uses_outer_physical_perimeter(face, hole):
    mask = np.zeros((20, 20, 8, 1), dtype=bool)
    xs = slice(0, 8) if face == "min" else slice(12, 20)
    mask[xs, 4:12, 2:6, 0] = True
    if hole:
        inner = slice(2, 4) if face == "min" else slice(14, 16)
        mask[inner, 7:9, 2:6, 0] = False
    plane = PlaneData(center=np.array([4.5 if face == "min" else 16.5, 8.5, 3.5]),
                      normal=np.array([0., 0., 1.]), path_index=0, label=1)
    origin = np.array([100., -200., 300.])
    kept, qc = filter_planes_by_branch_support(
        [plane], mask, np.ones(3), origin, branch_labels_3d=mask[..., 0].astype(np.int16),
        fov_boundary_max_fraction=0.25,
    )
    assert not kept
    assert qc[0]["reason"] == "fov_boundary_contact"
    fov = qc[0]["fov_boundary"]
    assert fov["perimeter_mm"] == pytest.approx(32.)
    assert fov["contact_length_mm"] == pytest.approx(8.)
    assert fov["contact_fraction"] == pytest.approx(0.25)
    assert fov["faces"] == [{"face": f"x_{face}", "contact_length_mm": pytest.approx(8.)}]
    kept, qc = filter_planes_by_branch_support(
        [plane], mask, np.ones(3), origin, branch_labels_3d=mask[..., 0].astype(np.int16),
        fov_boundary_max_fraction=0.26,
    )
    assert kept == [plane] and qc[0]["fov_boundary"]["status"] == "below_threshold"


def test_fov_contact_phantom_is_independent_of_branch_filter_and_respects_spacing():
    mask = np.zeros((20, 20, 8, 1), dtype=bool)
    mask[0:8, 4:12, 2:6, 0] = True
    plane = PlaneData(center=np.array([9., 8.5, 10.5]), normal=np.array([0., 0., 1.]))
    kept, qc = filter_planes_by_branch_support(
        [plane], mask, np.array([2., 1., 3.]), np.zeros(3),
        check_branch_support=False, fov_boundary_max_fraction=0.25,
    )
    assert kept == [plane]
    assert not qc[0]["branch_support_checked"]
    assert qc[0]["fov_boundary"]["perimeter_mm"] == pytest.approx(48.)
    assert qc[0]["fov_boundary"]["contact_fraction"] == pytest.approx(1 / 6)
    kept, qc = filter_planes_by_branch_support(
        [plane], mask, np.array([2., 1., 3.]), np.zeros(3),
        check_branch_support=False, fov_boundary_max_fraction=0.15,
    )
    assert not kept and qc[0]["reason"] == "fov_boundary_contact"


def test_vote_loss_phantom_reports_each_phase_and_winning_labels(tmp_path):
    ws = Workspace()
    raw = np.zeros((7, 7, 7, 5), dtype=np.int16)
    raw[1, 1, 1] = [1, 1, 1, 1, 1]
    raw[1, 1, 2] = [1, 0, 0, 0, 0]
    raw[1, 1, 3] = [1, 2, 2, 2, 2]
    raw[1, 1, 4] = [0, 1, 1, 1, 1]
    raw[2, 2, 2] = [3, 0, 0, 0, 0]
    raw[2, 2, 3] = [4, 4, 0, 0, 0]
    raw[2, 2, 4] = [5, 5, 6, 6, 0]
    ws.segmask_raw = raw.copy()
    ws.flow_raw = np.zeros(raw.shape + (3,), dtype=np.float32)
    ws.skeleton_params.label_map = {"AAO": 1, "MPA": 2, "RPA": 3, "LPA": 4, "BA": 5, "LPCA": 6}
    ws.skeleton_params.remove_small_cc = False
    ws.skeleton_params.do_closing = False
    ws.skeleton_params.do_opening = False
    ws.skeleton_params.gaussian_enabled = False
    engine = PipelineEngine()
    assert engine.preprocess(ws)
    records = {item["label"]: item for item in ws.segmentation_preprocess_qc["labels"]}
    aao = records[1]
    assert aao["vote_input_union_voxels"] == 4 and aao["voted_voxels"] == 2
    assert aao["vote_removed_union_fraction"] == pytest.approx(0.5)
    first = aao["phases"][0]
    assert first["vote_input_voxels"] == 3 and first["vote_overlap_voxels"] == 1
    assert first["vote_removed_fraction"] == pytest.approx(2 / 3)
    assert first["vote_added_voxels"] == 1
    assert first["vote_removed_to_background_voxels"] == first["vote_removed_to_other_labels_voxels"] == 1
    assert {item["label"]: item["voxel_count"] for item in first["vote_destinations"]} == {0: 1, 2: 1}
    assert aao["phases"][1]["vote_removed_fraction"] == 0
    assert records[2]["phases"][0]["vote_removed_fraction"] is None
    assert records[3]["status"] == records[6]["status"] == "removed_by_majority_vote"
    assert records[3]["phases"][0]["vote_removed_fraction"] == 1
    assert records[3]["phases"][0]["temporal_group_removed_fraction"] == 1
    assert ws.segmask_labels_3d[2, 2, 4] == 5  # plurality tie, not an absolute majority
    assert np.array_equal(ws.segmask_raw, raw)
    report = build_quality_report(ws)
    vote_check = next(item for item in report["checks"] if item["id"] == "segmentation.majority_vote")
    assert vote_check["status"] == "warn"
    assert len(vote_check["details"]) == 6
    assert "lost=2/3 (66.7%)" in vote_check["details"][0]["details"][0]["summary"]
    phase_group_check = next(item for item in report["checks"] if item["id"] == "segmentation.temporal_group_components")
    assert phase_group_check["status"] == "warn"
    saved = tmp_path / "quality_report.json"
    save_quality_report(ws, str(saved), report=report)
    assert json.loads(saved.read_text())["exclusions"]["segmentation_preprocess"] == ws.segmentation_preprocess_qc


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
    workspace.segmask_labels_3d = np.ones((2, 2, 2), dtype=np.int16)
    workspace.segmentation_preprocess_qc = {"schema": "autoflow.segmentation_preprocess_qc.v1", "labels": []}
    restored = Workspace()
    restored.restore_dict(workspace.snapshot_dict())
    assert np.allclose(restored.origin, workspace.origin)
    assert np.array_equal(restored.correction_raw, workspace.correction_raw)
    assert np.array_equal(restored.correction_high_raw, workspace.correction_high_raw)
    assert np.array_equal(restored.segmask_labels_3d, workspace.segmask_labels_3d)
    assert restored.segmentation_preprocess_qc == workspace.segmentation_preprocess_qc


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


def test_label_plane_phantom_default_connections_and_explicit_fallbacks():
    from autoflow.algorithms.label_planes import generate_label_planes
    from autoflow.plane_rules import default_label_plane_rules

    rules = default_label_plane_rules()
    counts = {name: rule["plane_count"] for name, rule in rules.items()}
    counts.update(dict.fromkeys(["LICA", "RICA", "LVA", "RVA"], 2))
    counts.update(dict.fromkeys(["LRA","RRA"],1))
    label_map = {name: i + 1 for i, name in enumerate(counts)}
    labels = np.zeros((35, len(counts) * 4 + 5, 5), dtype=np.int16)
    paths = []
    for row, (name, value) in enumerate(label_map.items()):
        y = row * 4 + 2
        labels[1:32, y - 1:y + 2, 1:4] = value
        paths.append(np.column_stack((np.arange(1., 32.), np.full(31, y), np.full(31, 2.))))
    infos = [{"start_node": 2 * i, "end_node": 2 * i + 1,
              "hierarchy": {"label_local_levels": {str(label_map[name]): 0}, "flow_sign": 1}}
             for i, name in enumerate(label_map)]
    planes, smooth, _, qc = generate_label_planes(
        paths, labels, label_map, rules, path_info=infos, smoothing_window=3, inter_time=1)
    # Each isolated physical component uses its declared no-connection plan.
    assert len(planes) == sum(counts.values()) == 41
    for name, count in counts.items():
        selected = [p for p in planes if p.segmentation_label == label_map[name]]
        expected = ([] if not count else [.2, .5, .8] if name in {"AAO", "PV"}
                    else [.1, .3, .5] if name == "DAO" else [.2] if name == "MPA"
                    else [1 / 3, 2 / 3] if name in {"LIJV", "RIJV"}
                    else [.33, .66] if name in {"LICA", "RICA", "LVA", "RVA"} else [.5])
        assert len(selected) == len(expected) == count
        assert [p.distance / 30 for p in selected] == pytest.approx(expected)
        assert all(np.allclose(p.normal, [1, 0, 0]) for p in selected)
    assert rules["LPCA"]["plane_count"] == rules["RPCA"]["plane_count"] == 1
    assert rules["LICA"]["plane_count"] == rules["RICA"]["plane_count"] == 1
    assert rules["ACA"]["plane_count"] == 1
    assert all(rules[name]["plane_count"] == 0 for name in ("ARCH", "HA", "SMA"))
    assert len(smooth) == len(paths)
    assert all(q["status"] == ("disabled" if rules[q["label_name"]]["plane_count"] == 0 else "selected") for q in qc)
    assert all("main_path" not in rule and rule["plane_mode"] == "landmark" for rule in rules.values())
    config = AutoFlowConfig.from_config_dir()
    assert config.plane_mode == "by_label" and config.cc_filter_mode == "hybrid"
    assert config.cc_rel_min_ratio == .10 and config.min_cc_volume == 50.
    assert build_workspace(config).plane_gen_params.label_rules == default_label_plane_rules()
    from autoflow.plane_rules import load_plane_rules_file
    plan_dir = DATA_DIR.parent / "configs" / "plane_plans"
    for name, total in [("cardiopulmonary", 12), ("portal", 5), ("renal", 6), ("brain", 14), ("neck", 8)]:
        source = plan_dir / f"{name}.json"
        case_config = AutoFlowConfig.from_config_dir(plane_config=str(source))
        regional_rules = build_workspace(case_config).plane_gen_params.label_rules
        assert set(regional_rules) == set(load_plane_rules_file(source))
        if name == "brain":
            assert {"LICA", "RICA", "ACA"} <= set(regional_rules)
        if name == "neck":
            assert regional_rules["LICA"]["plane_count"] == regional_rules["RICA"]["plane_count"] == 1
            assert regional_rules["LICA"]["placements"][0]["fallback"]["fractions"] == [.33, .66]
        assert sum(rule["plane_count"] for rule in regional_rules.values()) == total, name
        assert all(rule["placements"] and "main_path" not in rule for rule in regional_rules.values())
    from autoflow.core.models import PlaneGenerationParams
    assert PlaneGenerationParams.from_dict({"use_center_plane": False}).plane_mode == "distance"


def _landmark_phantom(paths, labels, endpoints, label_map, hierarchy=None):
    """Small physical-path phantom, including label transitions inside edges."""
    volume = np.zeros((65, 65, 21), dtype=np.int16)
    for points, values in zip(paths, labels):
        for point, value in zip(points, values):
            x, y, z = np.rint(point).astype(int)
            volume[x, y, z] = value
    infos = [{"start_node": u, "end_node": v, "group_name": "phantom",
              "hierarchy": (hierarchy or [{}] * len(paths))[index]}
             for index, (u, v) in enumerate(endpoints)]
    return volume, infos


def _legacy_landmark_rule(anchor, *, fractions=None, offset=5.):
    """Explicit saved named-anchor plans remain supported after default changes."""
    spec = {"anchor": anchor, "position": "fractions" if fractions else "near"}
    spec.update({"fractions": fractions} if fractions else {"offset_mm": offset})
    return {"plane_mode": "landmark", "plane_count": len(fractions) if fractions else 1,
            "placements": [spec]}


def test_connection_default_phantom_retains_two_same_label_renal_arms():
    from autoflow.algorithms.label_planes import generate_label_planes
    from autoflow.plane_rules import default_label_plane_rules
    paths = [np.column_stack((np.arange(5., 21.), np.full(16, 20.), np.full(16, 10.))),
             np.column_stack((np.arange(20., 51.), np.full(31, 20.), np.full(31, 10.))),
             np.column_stack((np.full(31, 20.), np.arange(20., 51.), np.full(31, 10.)))]
    mapping = {"DAO": 4, "LRA": 12}
    labels = [np.full(16, 4), np.where(np.arange(31) < 3, 4, 12),
              np.where(np.arange(31) < 3, 4, 12)]
    volume, infos = _landmark_phantom(paths, labels, [(0, 1), (1, 2), (1, 3)], mapping)
    rules = {name: default_label_plane_rules()[name] for name in mapping}
    planes, _, _, qc = generate_label_planes(paths, volume, mapping, rules,
        path_info=infos, smoothing_window=1, inter_time=1)
    renal = [plane for plane in planes if plane.segmentation_label == 12]
    aorta = [plane for plane in planes if plane.segmentation_label == 4]
    assert {plane.path_index for plane in renal} == {1, 2} and len(renal) == 6
    for index in (1,2):
        assert sorted(p.placement["fraction"] for p in renal if p.path_index==index)==[.25,.5,.75]
    assert len(aorta) == 3 and {plane.path_index for plane in aorta} == {0}
    assert sorted(plane.placement["fraction"] for plane in aorta) == [.1, .3, .5]
    assert all(not plane.placement["connection_fallback"] for plane in renal)
    assert len({plane.placement["connection_id"] for plane in renal}) == 1
    assert all(p.placement["anchor"]=="label_longest" for p in aorta)
    assert {row["selection_source"] for row in qc} == {"observed_label_connections","label_path_properties"}


@pytest.mark.parametrize("name", ["PV", "SMV", "SV", "RBCT", "CCA", "LBCT", "RPA", "LPA"])
@pytest.mark.parametrize("length", [12, 40])
def test_connection_default_phantom_positions_stay_inside_short_label(name, length):
    from autoflow.algorithms.label_planes import generate_label_planes
    from autoflow.plane_rules import default_label_plane_rules
    path = np.column_stack((np.arange(5., 56.), np.full(51, 20.), np.full(51, 10.)))
    # A two-label boundary is sufficient; no named anatomical triple is needed.
    values = np.where(np.arange(51) <= length, 101, 102)
    volume, infos = _landmark_phantom([path], [values], [(0, 1)], {name: 101, "other": 102})
    planes, _, _, _ = generate_label_planes([path], volume, {name: 101, "other": 102},
        {name: default_label_plane_rules()[name]}, path_info=infos, smoothing_window=1, inter_time=1)
    if name == "PV":
        assert len(planes) == 3
        assert [p.placement["fraction"] for p in planes] == pytest.approx([.2, .5, .8])
        assert [p.center[0] for p in planes] == pytest.approx([5 + length * (1 - f) for f in (.2, .5, .8)])
        assert all(p.placement["anchor"] == "label_connections" and p.placement["near_end"] for p in planes)
        assert all(not p.placement["connection_fallback"] and not p.placement["short_interval_midpoint"] for p in planes)
        return
    assert len(planes) == 1
    placement = planes[0].placement
    assert placement["anchor"] == "label_connections" and placement["near_end"]
    assert placement["nominal_distance_in_interval_mm"] == pytest.approx(min(15., length / 2))
    assert placement["short_interval_midpoint"] == (length < 30)
    assert planes[0].center[0] == pytest.approx(5 + length - min(15., length / 2))


@pytest.mark.parametrize("name", ["StrS", "LTS", "RTS", "SSS"])
@pytest.mark.parametrize("length,flow_sign", [(12, 1), (40, 1), (40, -1)])
def test_connection_explicit_sinus_phantom_saved_l1_offset(name, length, flow_sign):
    from autoflow.algorithms.label_planes import generate_label_planes
    from autoflow.plane_rules import default_label_plane_rules
    # A longer L2 branch must not replace the requested L1 interval.
    x = 5. + length
    paths = [np.column_stack((np.arange(5., x + 1), np.full(length + 1, 20.), np.full(length + 1, 10.))),
             np.column_stack((np.full(45, x), np.arange(20., 65.), np.full(45, 10.))),
             np.column_stack((np.full(16, x), np.arange(20., 4., -1), np.full(16, 10.)))]
    hierarchy = [{"label_local_levels": {"23": level}, "flow_sign": flow_sign} for level in (0, 1, 1)]
    volume, infos = _landmark_phantom(paths, [np.full(len(p), 23) for p in paths],
                                    [(0, 1), (1, 2), (1, 3)], {name: 23}, hierarchy)
    rule = {"plane_mode":"landmark","plane_count":1,"placements":[{"anchor":"label_connections",
        "position":"near","offset_mm":15.,"fallback":{"anchor":"label_level","level":1,
        "path_selection":"longest","position":"near","offset_mm":15.}}]}
    planes, _, _, qc = generate_label_planes(paths, volume, {name: 23}, {name: rule},
        path_info=infos, smoothing_window=1, inter_time=1)
    assert len(planes) == 1 and planes[0].path_index == 0
    placement = planes[0].placement
    assert placement["connection_fallback"] and placement["anchor"] == "label_level"
    assert placement["label_level"] == 1 and placement["path_selection"] == "longest"
    distance = min(15., length / 2)
    assert placement["nominal_distance_in_interval_mm"] == pytest.approx(distance)
    assert placement["short_interval_midpoint"] == (length < 30)
    assert planes[0].center[0] == pytest.approx(5. + (distance if flow_sign == 1 else length - distance))
    assert planes[0].normal == pytest.approx([flow_sign, 0., 0.])
    assert not qc[0]["selection_warnings"]
    # Without L1, report the unavailable level rather than choosing L2.
    infos[0]["hierarchy"]["label_local_levels"]["23"] = 1
    omitted, _, _, qc = generate_label_planes(paths, volume, {name: 23}, {name: rule},
        path_info=infos, smoothing_window=1, inter_time=1)
    assert not omitted and qc[0]["selection_warnings"] == ["label_level_unavailable"]


@pytest.mark.parametrize("name", ["StrS","LTS","RTS","SSS","PV","SMV","SV"])
@pytest.mark.parametrize("length,sign", [(24,1),(60,1),(60,-1)])
def test_label_offset_phantom_uses_requested_level_and_flow_reference(name,length,sign):
    from autoflow.algorithms.label_planes import generate_label_planes
    from autoflow.plane_rules import default_label_plane_rules
    paths=[np.column_stack((np.arange(2.,63.),np.full(61,2.),np.full(61,10.))),
           np.column_stack((np.full(length+1,62.),np.arange(2.,length+3.),np.full(length+1,10.))),
           np.column_stack((np.arange(62.,41.,-1),np.full(21,2.),np.full(21,10.)))]
    # Avoid overlapping the root with the shorter daughter in the sampled volume.
    paths[2][:,1]=np.linspace(2,15,21)
    hierarchy=[{"label_local_levels":{"23":level},"flow_sign":sign} for level in (0,2,2)]
    volume,infos=_landmark_phantom(paths,[np.full(len(p),23) for p in paths],[(0,1),(1,2),(1,3)],{name:23},hierarchy)
    # A connection is absent; PV/SMV/SV must use only their declared fallback.
    planes,_,_,qc=generate_label_planes(paths,volume,{name:23},{name:default_label_plane_rules()[name]},
        path_info=infos,smoothing_window=1,inter_time=1)
    chosen=0 if name=="SSS" else 1
    assert len(planes)==(3 if name=="PV" else 1) and all(p.path_index==chosen for p in planes)
    if name=="PV":
        assert [p.placement["fraction"] for p in planes]==pytest.approx([.2,.5,.8])
        assert [p.center[1] for p in planes]==pytest.approx([2+length*(f if sign==1 else 1-f) for f in (.2,.5,.8)])
        assert all(p.placement["connection_fallback"] and p.placement["label_level"]==3 for p in planes)
        return
    placement=planes[0].placement
    selected_length=60 if chosen==0 else length
    from_start=name in {"StrS","LTS","RTS"}
    distance=selected_length/2 if selected_length<40 else 20 if from_start else selected_length-20
    assert placement["position"]==("after_start" if from_start else "before_end")
    assert placement["offset_reference"]==("flow_start" if from_start else "flow_end")
    assert placement["requested_offset_mm"]==20 and placement["label_level"]==(1 if name=="SSS" else 3)
    assert placement["nominal_distance_in_interval_mm"]==pytest.approx(distance)
    axis=0 if chosen==0 else 1
    assert planes[0].center[axis]==pytest.approx(2+(distance if sign==1 else selected_length-distance))
    infos[chosen]["hierarchy"]["flow_sign"]=None
    omitted,_,_,qc=generate_label_planes(paths,volume,{name:23},{name:default_label_plane_rules()[name]},
        path_info=infos,smoothing_window=1,inter_time=1)
    assert not omitted and qc[0]["omitted_positions"][0]["reason"]=="flow_direction_unavailable"


@pytest.mark.parametrize("policy,expected",[("unique",[]),("all",[0,1]),("longest",[1])])
def test_label_scope_phantom_honors_selection_policy_across_components(policy,expected):
    from autoflow.algorithms.label_planes import generate_label_planes
    paths=[np.column_stack((np.arange(5.,26.),np.full(21,10.),np.full(21,10.))),
           np.column_stack((np.arange(5.,56.),np.full(51,30.),np.full(51,10.)))]
    hierarchy=[{"label_local_levels":{"4":0},"flow_sign":1} for _ in paths]
    volume,infos=_landmark_phantom(paths,[np.full(len(p),4) for p in paths],[(0,1),(2,3)],{"DAO":4},hierarchy)
    rule={"plane_mode":"landmark","plane_count":1,"placements":[{"anchor":"label_level","level":1,
          "scope":"label","path_selection":policy,"position":"fractions","fractions":[.5]}]}
    planes,_,_,_=generate_label_planes(paths,volume,{"DAO":4},{"DAO":rule},path_info=infos,smoothing_window=1,inter_time=1)
    assert sorted(p.path_index for p in planes)==expected


def test_dao_phantom_longest_interval_is_global_without_connection_requirement():
    from autoflow.algorithms.label_planes import generate_label_planes
    from autoflow.plane_rules import default_label_plane_rules
    paths=[np.column_stack((np.arange(5.,26.),np.full(21,10.),np.full(21,10.))),
           np.column_stack((np.arange(5.,56.),np.full(51,30.),np.full(51,10.)))]
    volume,infos=_landmark_phantom(paths,[np.full(len(p),4) for p in paths],[(0,1),(2,3)],{"DAO":4})
    planes,_,_,_=generate_label_planes(paths,volume,{"DAO":4},{"DAO":default_label_plane_rules()["DAO"]},
        path_info=infos,smoothing_window=1,inter_time=1)
    assert len(planes)==3 and {p.path_index for p in planes}=={1}
    assert [p.center[0] for p in planes]==pytest.approx([10,20,30])
    assert all(p.placement["selection_scope"]=="label" for p in planes)


@pytest.mark.parametrize("connected", [True, False])
def test_mpa_phantom_default_twenty_percent_for_connection_and_fallback(connected):
    from autoflow.algorithms.label_planes import generate_label_planes
    from autoflow.plane_rules import default_label_plane_rules
    path = np.column_stack((np.arange(5., 56.), np.full(51, 20.), np.full(51, 10.)))
    values = np.where(np.arange(51) < 41, 2, 9) if connected else np.full(51, 2)
    hierarchy = [{"label_local_levels": {"2": 0}, "flow_sign": 1}]
    volume, infos = _landmark_phantom([path], [values], [(0, 1)], {"MPA": 2, "LPA": 9}, hierarchy)
    planes, _, _, _ = generate_label_planes([path], volume, {"MPA": 2, "LPA": 9},
        {"MPA": default_label_plane_rules()["MPA"]}, path_info=infos, smoothing_window=1, inter_time=1)
    assert len(planes) == 1
    plane = planes[0]
    placement = plane.placement
    lo, hi = placement["label_interval_mm"]
    assert placement["fraction"] == pytest.approx(.2)
    assert placement["nominal_distance_in_interval_mm"] == pytest.approx((hi - lo) * .2)
    assert bool(placement.get("connection_fallback")) == (not connected)
    assert plane.distance == pytest.approx(hi - (hi - lo) * .2 if connected else lo + (hi - lo) * .2)


@pytest.mark.parametrize("name",["LIJV","RIJV"])
def test_jugular_phantom_highest_level_wins_over_longer_low_level(name):
    from autoflow.algorithms.label_planes import generate_label_planes
    from autoflow.plane_rules import default_label_plane_rules
    paths=[np.column_stack((np.arange(5.,56.),np.full(51,10.),np.full(51,10.))),
           np.column_stack((np.arange(5.,21.),np.full(16,30.),np.full(16,10.)))]
    hierarchy=[{"label_local_levels":{"32":level},"flow_sign":1} for level in (0,3)]
    volume,infos=_landmark_phantom(paths,[np.full(len(p),32) for p in paths],[(0,1),(2,3)],{name:32},hierarchy)
    planes,_,_,_=generate_label_planes(paths,volume,{name:32},{name:default_label_plane_rules()[name]},
        path_info=infos,smoothing_window=1,inter_time=1)
    assert len(planes)==2 and {p.path_index for p in planes}=={1}
    assert all(p.placement["label_level"]==4 for p in planes)
    assert [p.center[0] for p in planes]==pytest.approx([10,15])


def test_connection_phantom_background_gap_uses_only_declared_fallback():
    from autoflow.algorithms.label_planes import generate_label_planes
    from autoflow.plane_rules import default_label_plane_rules
    path = np.column_stack((np.arange(5., 56.), np.full(51, 20.), np.full(51, 10.)))
    values = np.where(np.arange(51) < 20, 12, np.where(np.arange(51) == 20, 0, 4))
    volume, infos = _landmark_phantom([path], [values], [(0, 1)], {"LRA": 12, "DAO": 4})
    rule = default_label_plane_rules()["LRA"]
    planes, _, _, _ = generate_label_planes([path], volume, {"LRA": 12, "DAO": 4}, {"LRA": rule},
        path_info=infos, smoothing_window=1, inter_time=1)
    assert len(planes) == 1 and planes[0].placement["connection_fallback"]
    assert planes[0].placement["anchor"] == "label_longest"
    rule["placements"][0].pop("fallback")
    omitted, _, _, qc = generate_label_planes([path], volume, {"LRA": 12, "DAO": 4}, {"LRA": rule},
        path_info=infos, smoothing_window=1, inter_time=1)
    assert not omitted and qc[0]["selection_warnings"] == ["label_connection_not_found"]


@pytest.mark.parametrize("both_connected", [True, False])
def test_connection_ba_phantom_level_depends_on_actual_connections(both_connected):
    from autoflow.algorithms.label_planes import generate_label_planes
    from autoflow.plane_rules import default_label_plane_rules
    paths = [np.column_stack((np.arange(5., 21.), np.full(16, 20.), np.full(16, 10.))),
             np.column_stack((np.arange(20., 51.), np.full(31, 20.), np.full(31, 10.))),
             np.column_stack((np.full(16, 20.), np.arange(5., 21.), np.full(16, 10.))),
             np.column_stack((np.full(21, 20.), np.arange(20., 41.), np.full(21, 10.)))]
    mapping = {"BA": 21, "LVA": 18, "RVA": 19}
    hierarchy = [{"label_local_levels": {"21": level}, "flow_sign": 1} for level in (0, 1, 0, 0)]
    if not both_connected:
        paths[3][:, 0] = 55.
    volume, infos = _landmark_phantom(paths, [np.full(len(p), value) for p, value in zip(paths, (21, 21, 18, 19))],
        [(0, 1), (1, 2), (3, 1), (1, 4)], mapping, hierarchy)
    volume[20, 19:23, 10] = 21
    if not both_connected:
        infos[3]["group_name"] = "independent-RVA"
    planes, _, _, _ = generate_label_planes(paths, volume, mapping,
        {"BA": default_label_plane_rules()["BA"]}, path_info=infos, smoothing_window=1, inter_time=1)
    assert len(planes) == 1
    assert planes[0].path_index == (1 if both_connected else 0)
    assert planes[0].placement["label_level"] == (2 if both_connected else 1)
    assert planes[0].placement["fraction"] == .5


def test_connection_default_gui_phantom_preserves_fallback_and_conditional_levels(tmp_path):
    from PySide6 import QtWidgets
    from autoflow.plane_rules import default_label_plane_rules, save_plane_rules_file, load_plane_rules_file
    from autoflow.ui.landmark_editor import LandmarkPlanDialog
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    rules = default_label_plane_rules()
    edited = {}
    for name in ("LICA", "BA", "RIJV", "StrS", "LTS", "RTS", "SSS"):
        dialog = LandmarkPlanDialog(name, rules[name])
        try:
            dialog.accept()
            assert dialog.result() == QtWidgets.QDialog.Accepted
            assert dialog.rule == rules[name]
            edited[name] = dialog.rule
        finally:
            dialog.close()
            dialog.deleteLater()
            app.processEvents()
    source = tmp_path / "connections.json"
    save_plane_rules_file(source, edited)
    assert load_plane_rules_file(source) == edited


def test_landmark_pulmonary_phantom_uses_label_boundary_and_never_flow_roots(monkeypatch):
    import autoflow.algorithms.label_planes as lp
    from autoflow.plane_rules import default_label_plane_rules
    paths = [np.column_stack((np.arange(5., 21.), np.full(16, 20.), np.full(16, 10.))),
             np.column_stack((np.arange(20., 51.), np.full(31, 20.), np.full(31, 10.))),
             np.column_stack((np.full(31, 20.), np.arange(20., 51.), np.full(31, 10.)))]
    mapping = {"MPA": 2, "LPA": 9, "RPA": 8}
    labels = [np.full(16, 2), np.where(np.arange(31) < 10, 2, 9), np.where(np.arange(31) < 7, 2, 8)]
    volume, infos = _landmark_phantom(paths, labels, [(0, 1), (1, 2), (1, 3)], mapping)
    def forbidden(*args, **kwargs):
        raise AssertionError("landmark plans must not call the old flow-root selector")
    monkeypatch.setattr(lp, "_select_main_paths", forbidden)
    rules = {name: _legacy_landmark_rule("pulmonary_split") for name in mapping}
    planes, _, _, qc = lp.generate_label_planes(paths, volume, mapping, rules,
        path_info=infos, smoothing_window=1, inter_time=10, flow_xyzt3=np.zeros((*volume.shape, 2, 3)))
    assert len(planes) == 3
    by_label = {plane.segmentation_label: plane for plane in planes}
    assert by_label[2].path_index == 0 and by_label[2].center[0] == pytest.approx(15.)
    assert by_label[9].path_index == 1 and by_label[9].center[0] >= 34.
    assert by_label[8].path_index == 2 and by_label[8].center[1] >= 31.
    assert all(p.placement["anchor"] == "pulmonary_split" for p in planes)
    assert all(p.placement["matched_arms"] == {"MPA": 0, "LPA": 1, "RPA": 2} for p in planes)
    assert all(row["selection_source"] == "anatomical_landmarks" for row in qc)
    # Removing one daughter cannot silently select the remaining label root.
    no_rpa = volume.copy()
    no_rpa[no_rpa == 8] = 0
    omitted, _, _, qc = lp.generate_label_planes(paths, no_rpa, mapping, {"MPA": rules["MPA"]},
        path_info=infos, smoothing_window=1, inter_time=10)
    assert not omitted and qc[0]["requested_plane_count"] == 1
    assert qc[0]["omitted_positions"][0]["reason"] == "landmark_not_found"


@pytest.mark.parametrize("system", ["portal", "posterior"])
def test_landmark_confluence_split_phantom_selects_common_trunk(system):
    from autoflow.algorithms.label_planes import generate_label_planes
    from autoflow.plane_rules import default_label_plane_rules
    paths = [np.column_stack((np.arange(5., 21.), np.full(16, 20.), np.full(16, 10.))),
             np.column_stack((np.full(16, 20.), np.arange(5., 21.), np.full(16, 10.))),
             np.column_stack((np.arange(20., 41.), np.full(21, 20.), np.full(21, 10.))),
             np.column_stack((np.full(16, 40.), np.arange(20., 36.), np.full(16, 10.))),
             np.column_stack((np.full(16, 40.), np.arange(20., 4., -1), np.full(16, 10.)))]
    names = ["SV", "SMV", "PV", "PV", "PV"] if system == "portal" else ["LVA", "RVA", "BA", "LPCA", "RPCA"]
    mapping = {name: i + 1 for i, name in enumerate(dict.fromkeys(names))}
    labels = [np.full(len(path), mapping[name]) for path, name in zip(paths, names)]
    volume, infos = _landmark_phantom(paths, labels, [(0, 2), (1, 2), (2, 3), (3, 4), (3, 5)], mapping)
    name = "PV" if system == "portal" else "BA"
    # Deliberately disagree with L1: the anatomical common trunk remains P2.
    infos[0]["hierarchy"] = {"label_local_levels": {str(mapping[name]): 0}}
    infos[2]["hierarchy"] = {"label_local_levels": {str(mapping[name]): 2}, "flow_sign": 1}
    rule = (_legacy_landmark_rule("portal_trunk", fractions=[.25, .5, .75]) if system == "portal"
            else {"plane_mode": "landmark", "plane_count": 2, "placements": [
                {"anchor": anchor, "position": "near", "offset_mm": 5.}
                for anchor in ("vertebral_confluence", "pca_split")]})
    planes, _, _, qc = generate_label_planes(paths, volume, mapping,
        {name: rule}, path_info=infos, smoothing_window=1, inter_time=10)
    expected = [25., 30., 35.] if system == "portal" else [25., 20. + planes[-1].placement["label_interval_mm"][1] - 5.]
    assert [p.center[0] for p in planes] == pytest.approx(expected)
    assert all(p.path_index == 2 for p in planes)
    assert qc[0]["actual_plane_count"] == len(expected)
    if system == "portal":
        assert planes[1].placement["route_path_indices"] == [2]
    else:
        assert [p.placement["anchor"] for p in planes] == ["vertebral_confluence", "pca_split"]


def test_landmark_phantom_groups_do_not_join_by_proximity_and_ties_omit():
    from autoflow.algorithms.label_planes import generate_label_planes
    from autoflow.plane_rules import default_label_plane_rules
    paths = [np.column_stack((np.arange(5., 21.), np.full(16, 20.), np.full(16, 10.))),
             np.column_stack((np.arange(20., 41.), np.full(21, 20.), np.full(21, 10.))),
             np.column_stack((np.full(21, 20.), np.arange(20., 41.), np.full(21, 10.)))]
    mapping = {"MPA": 2, "LPA": 9, "RPA": 8}
    volume, infos = _landmark_phantom(paths, [np.full(len(p), label) for p, label in zip(paths, (2, 9, 8))],
        [(0, 1), (1, 2), (1, 3)], mapping)
    infos[2]["group_name"] = "disconnected"
    planes, _, _, qc = generate_label_planes(paths, volume, mapping,
        {"MPA": _legacy_landmark_rule("pulmonary_split")}, path_info=infos, smoothing_window=1, inter_time=10)
    assert not planes and qc[0]["selection_warnings"] == ["landmark_not_found"]
    # Two equally supported L1 paths in the SAME physical component are
    # ambiguous; choosing the longest would reintroduce the rejected policy.
    infos[2]["group_name"] = "phantom"
    for row in infos:
        row["hierarchy"] = {"label_local_levels": {"2": 0}}
    volume[volume > 0] = 2
    rule = {"plane_mode": "landmark", "plane_count": 1,
            "placements": [{"anchor": "label_level", "level": 1, "position": "fractions", "fractions": [.5]}]}
    planes, _, _, qc = generate_label_planes(paths, volume, mapping, {"MPA": rule},
        path_info=infos, smoothing_window=1, inter_time=10)
    assert not planes and qc[0]["omitted_positions"][0]["reason"] == "ambiguous_label_level"


@pytest.mark.parametrize("policy,expected", [("unique", []), ("longest", [1]), ("all", [0, 1, 2])])
def test_landmark_same_level_path_selection_phantom_is_explicit(policy, expected):
    from autoflow.algorithms.label_planes import generate_label_planes
    paths = [np.column_stack((np.arange(5., 21.), np.full(16, 20.), np.full(16, 10.))),
             np.column_stack((np.arange(20., 51.), np.full(31, 20.), np.full(31, 10.))),
             np.column_stack((np.full(21, 20.), np.arange(20., 41.), np.full(21, 10.)))]
    mapping = {"LICA": 17}
    hierarchy = [{"label_local_levels": {"17": 0}, "flow_sign": 1}] * 3
    volume, infos = _landmark_phantom(paths, [np.full(len(p), 17) for p in paths],
                                    [(0, 1), (1, 2), (1, 3)], mapping, hierarchy)
    rules = {"LICA": {"plane_mode": "landmark", "plane_count": 1, "placements": [
        {"anchor": "label_level", "level": 1, "path_selection": policy, "position": "fractions", "fractions": [.5]}]}}
    planes, _, _, qc = generate_label_planes(paths, volume, mapping, rules, path_info=infos, smoothing_window=1, inter_time=10)
    assert [p.path_index for p in planes] == expected
    assert all(p.placement["path_selection"] == policy for p in planes)
    assert all(p.placement["paths_at_level"] == [0, 1, 2] for p in planes)
    if not expected:
        assert qc[0]["selection_warnings"] == ["ambiguous_label_level"]


def test_landmark_structured_gui_phantom_edits_anchor_positions_and_count():
    from PySide6 import QtWidgets
    from autoflow.ui.landmark_editor import LandmarkPlanDialog
    from autoflow.plane_rules import default_label_plane_rules
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    original = default_label_plane_rules()["MPA"]
    dialog = LandmarkPlanDialog("MPA", original)
    try:
        table = dialog.table
        assert table.rowCount() == 1
        anchor = table.cellWidget(0, 0)
        assert anchor.currentData() == "label_connections"
        anchor.setCurrentIndex(anchor.findData("label_level"))
        table.cellWidget(0, 1).setValue(3)
        paths = table.cellWidget(0, 2)
        paths.setCurrentIndex(paths.findData("all"))
        position = table.cellWidget(0, 3)
        position.setCurrentIndex(position.findData("fractions"))
        table.cellWidget(0, 5).setText("20, 50, 80")
        dialog.accept()
        assert dialog.result() == QtWidgets.QDialog.Accepted
        assert dialog.rule["plane_count"] == 3
        assert dialog.rule["placements"] == [{"anchor": "label_level", "level": 3, "path_selection": "all",
                                                "position": "fractions", "fractions": [.2, .5, .8]}]
        assert original["placements"][0]["anchor"] == "label_connections"
        assert not dialog.findChildren(QtWidgets.QPlainTextEdit)
    finally:
        dialog.close()
        dialog.deleteLater()
        app.processEvents()


def test_landmark_neck_preset_gui_phantom_is_visible_and_roundtrips(tmp_path):
    from PySide6 import QtWidgets
    from autoflow.ui.action_settings import ActionSettingsDialog, action_sections
    from autoflow.plane_rules import save_plane_rules_file, load_plane_rules_file
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    ws = Workspace()
    dialog = ActionSettingsDialog("Planes", action_sections(StepId.GENERATE_PLANES, ws), ws, ws)
    try:
        preset = dialog.findChild(QtWidgets.QComboBox, "plane_plan_preset")
        index = next(i for i in range(preset.count()) if preset.itemText(i).startswith("Neck"))
        dialog._load_plane_preset(preset.itemData(index))
        table = dialog.controls["plane_gen_params.label_rules"]
        assert table.isColumnHidden(2) and table.isColumnHidden(3)
        row = next(i for i in range(table.rowCount()) if table.item(i, 0).text() == "LICA")
        assert "L1" in table.item(row, 6).text() and "Longest" in table.item(row, 6).text()
        rules = dialog.values()["plane_gen_params.label_rules"]
        assert rules["LICA"]["plane_count"] == 1
        assert rules["LICA"]["placements"][0]["anchor"] == "label_connections"
        assert rules["LICA"]["placements"][0]["fallback"]["path_selection"] == "longest"
        filename = tmp_path / "neck.json"
        save_plane_rules_file(filename, rules)
        assert load_plane_rules_file(filename) == rules
    finally:
        dialog.close()
        dialog.deleteLater()
        app.processEvents()


def test_landmark_phantom_gui_and_provenance_roundtrip(tmp_path):
    from PySide6 import QtWidgets
    from autoflow.plane_rules import default_label_plane_rules, save_plane_rules_file, load_plane_rules_file
    from autoflow.ui.action_settings import ActionSettingsDialog, action_sections
    from autoflow.plane_io import build_plane_records, save_plane_positions, load_plane_positions, project_planes_to_workspace
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    ws = Workspace()
    ws.skeleton_params.label_map = {"MPA": 2}
    rules = {"MPA": default_label_plane_rules()["MPA"]}
    ws.plane_gen_params.label_rules = rules
    dialog = ActionSettingsDialog("Planes", action_sections(StepId.GENERATE_PLANES, ws), ws, ws)
    try:
        table = dialog.controls["plane_gen_params.label_rules"]
        assert table.cellWidget(0, 4).currentData() == "landmark"
        assert not table.cellWidget(0, 3).isEnabled()
        normalized = dialog.values()["plane_gen_params.label_rules"]
        assert normalized == rules and "main_path" not in normalized["MPA"]
        filename = tmp_path / "landmarks.json"
        save_plane_rules_file(filename, normalized)
        assert load_plane_rules_file(filename) == rules
    finally:
        dialog.close()
        dialog.deleteLater()
        app.processEvents()
    ws.planes = [PlaneData(center=np.array([1., 2., 3.]), normal=np.array([1., 0., 0.]),
                           path_index=-1, segmentation_label=2, placement={"anchor": "pulmonary_split", "anchor_nodes": [{"node_id": 4}]})]
    assert build_plane_records(ws)[0]["placement"] == ws.planes[0].placement
    restored = Workspace()
    restored.restore_dict(ws.snapshot_dict())
    assert restored.planes[0].placement == ws.planes[0].placement
    positions = tmp_path / "positions.json"
    save_plane_positions(ws, str(positions))
    imported = project_planes_to_workspace(load_plane_positions(str(positions)), restored)
    assert imported[0].placement == ws.planes[0].placement


def test_landmark_sinus_phantom_matches_extended_two_fork_region():
    from autoflow.algorithms.label_planes import generate_label_planes
    from autoflow.plane_rules import default_label_plane_rules
    paths = [np.column_stack((np.arange(5., 21.), np.full(16, 20.), np.full(16, 10.))),
             np.column_stack((np.full(16, 20.), np.arange(5., 21.), np.full(16, 10.))),
             np.column_stack((np.arange(20., 31.), np.full(11, 20.), np.full(11, 10.))),
             np.column_stack((np.arange(30., 51.), np.full(21, 20.), np.full(21, 10.))),
             np.column_stack((np.full(21, 30.), np.arange(20., 41.), np.full(21, 10.)))]
    mapping = {"SSS": 23, "StrS": 25, "LTS": 22, "RTS": 24}
    labels = [np.full(len(path), value) for path, value in zip(paths, (23, 25, 23, 22, 24))]
    volume, infos = _landmark_phantom(paths, labels, [(0, 2), (1, 2), (2, 3), (3, 4), (3, 5)], mapping)
    planes, _, _, qc = generate_label_planes(paths, volume, mapping,
        {name: _legacy_landmark_rule("sinus_confluence") for name in mapping}, path_info=infos,
        smoothing_window=1, inter_time=10)
    assert {plane.path_index for plane in planes} == {0, 1, 3, 4}
    assert all(len(plane.placement["anchor_nodes"]) == 2 for plane in planes)
    assert all(plane.placement["boundary_path_indices"] == [0, 1, 3, 4] for plane in planes)
    assert all(row["actual_plane_count"] == 1 for row in qc)


def test_landmark_search_phantom_stays_on_selected_label_arm(monkeypatch):
    from autoflow.algorithms.label_planes import generate_label_planes
    import autoflow.algorithms.metrics.sampling as sampling
    from autoflow.plane_rules import default_label_plane_rules
    path = np.column_stack((np.arange(5., 36.), np.full(31, 20.), np.full(31, 10.)))
    mapping = {"AAO": 1}
    volume, infos = _landmark_phantom([path], [np.ones(31, dtype=int)], [(0, 1)], mapping,
        [{"label_local_levels": {"1": 0}, "flow_sign": 1}])
    inspected = []
    def support(planes, *args, **kwargs):
        plane = planes[0]
        inspected.append(plane.center.copy())
        valid = plane.center[0] >= 12. - 1e-8
        assert plane.path_index == 0 and plane.segmentation_label == 1
        assert kwargs["fov_boundary_max_fraction"] == .2
        return ([plane] if valid else []), [{"valid": valid, "reason": "fov_boundary_contact"}]
    monkeypatch.setattr(sampling, "filter_planes_by_branch_support", support)
    rule = default_label_plane_rules()["AAO"]
    rule["plane_count"] = 1
    rule["placements"] = [{"anchor": "label_level", "level": 1, "position": "fractions", "fractions": [.2]}]
    planes, _, _, qc = generate_label_planes([path], volume, mapping, {"AAO": rule}, path_info=infos,
        smoothing_window=1, inter_time=10, segmask_binary_4d=(volume > 0)[..., None],
        filter_defaults={"fov_boundary_max_fraction": .2})
    assert len(planes) == 1 and planes[0].center[0] == pytest.approx(12.)
    assert planes[0].placement["search_shift_mm"] == pytest.approx(1.)
    assert len(inspected) >= 3 and qc[0]["requested_plane_count"] == 1


@pytest.mark.parametrize("placement", [
    {"anchor": "missing", "position": "near"},
    {"anchor": "pulmonary_split", "position": "fractions", "fractions": [0.]},
    {"anchor": "pulmonary_split", "position": "near", "offset_mm": -1},
    {"anchor": "label_level", "position": "near", "level": 0},
])
def test_landmark_config_phantom_rejects_invalid_positions(placement):
    from autoflow.plane_rules import normalize_label_plane_rules
    with pytest.raises(ValueError):
        normalize_label_plane_rules({"MPA": {"plane_mode": "landmark", "plane_count": 1, "placements": [placement]}})


def test_label_plane_phantom_cross_label_path_and_disconnected_roots():
    from autoflow.algorithms.label_planes import generate_label_planes

    labels = np.zeros((70, 8, 5), dtype=np.int16)
    labels[1:21, 1:4, 1:4] = 1
    labels[21:41, 1:4, 1:4] = 4
    path = np.column_stack((np.arange(1., 41.), np.full(40, 2.), np.full(40, 2.)))
    planes, smooth, _, qc = generate_label_planes(
        [path], labels, {"AAO": 1, "DAO": 4},
        {"AAO": {"plane_count": 3}, "DAO": {"plane_count": 3}},
        smoothing_window=3, inter_time=1)
    assert len(planes) == 6 and len(smooth) == 1
    assert all(p.path_index == 0 for p in planes)
    assert [p.distance for p in planes[:3]] == pytest.approx([3.8, 9.5, 15.2])
    assert [p.distance for p in planes[3:]] == pytest.approx([23.8, 29.5, 35.2])
    assert all(labels[tuple(np.rint(p.center).astype(int))] == p.segmentation_label for p in planes)

    # Separate components of a label each retain their own root, automatically.
    labels[41:51, 1:4, 1:4] = 0
    labels[51:66, 1:4, 1:4] = 1
    longer = np.column_stack((np.arange(1., 66.), np.full(65, 2.), np.full(65, 2.)))
    planes, _, _, qc = generate_label_planes(
        [longer], labels, {"AAO": 1}, {"AAO": {"plane_count": 3, "path_index": 0}},
        smoothing_window=3, inter_time=1)
    assert len(planes) == 6 and qc[0]["status"] == "selected"
    assert qc[0]["main_path_count"] == 2 and qc[0]["requested_plane_count"] == 6
    assert len(qc[0]["candidates"]) == 2
    planes, _, _, qc = generate_label_planes(
        [longer], labels, {"AAO": 1},
        {"AAO": {"plane_count": 1, "path_index": 0, "segment_index": 1}},
        smoothing_window=3, inter_time=1)
    assert len(planes) == 2
    assert [p.center[0] for p in planes] == pytest.approx([10.5, 58.0])
    assert not {"path_index", "segment_index"} & set(qc[0]["rule"])


def test_label_plane_phantom_sv_outlets_do_not_require_pv():
    from autoflow.algorithms.label_planes import generate_label_planes

    labels = np.zeros((65, 12, 5), dtype=np.int16)
    labels[1:21, 1:4, 1:4] = 16
    labels[21:61, 1:4, 1:4] = 14
    labels[1:21, 7:10, 1:4] = 16
    paths = [np.column_stack((np.arange(1., 61.), np.full(60, 2.), np.full(60, 2.))),
             np.column_stack((np.arange(1., 21.), np.full(20, 8.), np.full(20, 2.)))]
    rule = {"SV": {"plane_count": 1, "require_pv_connection": True}}
    planes, _, _, qc = generate_label_planes(
        paths, labels, {"SV": 16, "PV": 14}, rule, smoothing_window=3, inter_time=1)
    assert len(planes) == 2
    assert qc[0]["main_path"] == "outflow" and qc[0]["main_path_count"] == 2
    assert "require_pv_connection" not in qc[0]["rule"]

    # Neither an intervening label nor an absent PV disables SV measurements.
    labels[21, 1:4, 1:4] = 10
    planes, _, _, qc = generate_label_planes(
        paths, labels, {"SV": 16, "PV": 14}, rule, smoothing_window=3, inter_time=1)
    assert len(planes) == 2

    labels[labels == 14] = 0
    planes, _, _, qc = generate_label_planes(
        paths, labels, {"SV": 16, "PV": 14}, rule, smoothing_window=3, inter_time=1)
    assert len(planes) == 2
    rule["SV"]["path_index"] = 1
    planes, _, _, qc = generate_label_planes(
        paths, labels, {"SV": 16, "PV": 14}, rule, smoothing_window=3, inter_time=1)
    assert len(planes) == 2
    assert "path_index" not in qc[0]["rule"]


@pytest.mark.parametrize("name,side,sign,expected", [
    ("MPA", "inflow", 1, [0]), ("MPA", "outflow", 1, [1, 2]),
    ("SV", "outflow", -1, [0]), ("SMV", "outflow", -1, [0]),
    ("PV", "inflow", 1, [0]), ("PV", "inflow", -1, [1, 2]),
    ("ACA", "inflow", -1, [1, 2]),
])
def test_label_plane_branch_phantom_flow_roots_and_isolated_pv(name, side, sign, expected):
    from scipy.ndimage import binary_dilation
    from autoflow.algorithms.label_planes import generate_label_planes

    trunk = np.column_stack((np.arange(2., 11.), np.full(9, 15.), np.full(9, 3.)))
    arms = [np.column_stack((np.linspace(10, 18, 17), np.linspace(15, y, 17), np.full(17, 3.)))
            for y in (7., 23.)]
    paths = [trunk[::-1], *arms]  # Graph traversal order must not dictate flow direction.
    labels = np.zeros((25, 31, 7), dtype=np.int16)
    for path in paths:
        labels[tuple(np.rint(path).astype(int).T)] = 1
    labels = binary_dilation(labels, iterations=1).astype(np.int16)
    flow = np.zeros(labels.shape + (3, 3), dtype=np.float32)
    flow[..., :, 0] = np.asarray([20., -5., 20.]) * sign
    flow[labels == 0] = 0
    # Deliberately omit all other vessel labels, including SMV/SV for PV.
    planes, _, _, qc = generate_label_planes(paths, labels, {name: 1},
        {name: {"plane_count": 2, "main_path": side}}, flow_xyzt3=flow,
        segmask_binary_4d=labels > 0, smoothing_window=3, inter_time=2)
    selected = qc[0]["selected_main_paths"]
    assert [p["path_index"] for p in selected] == expected
    assert len(planes) == 2 * len(expected) == qc[0]["requested_plane_count"]
    assert qc[0]["main_path_count"] == len(expected)
    assert not qc[0]["selection_warnings"]
    assert all(p.normal[0] * sign > 0 for p in planes)
    assert all(c["flow_direction_status"] == "reliable" for c in qc[0]["topology_candidates"])
    assert all(labels[tuple(np.rint(p.center).astype(int))] == 1 for p in planes)


@pytest.mark.parametrize("flow_available", [False, True])
def test_label_plane_branch_phantom_unreliable_flow_uses_visible_fallback(flow_available):
    from scipy.ndimage import binary_dilation
    from autoflow.algorithms.label_planes import generate_label_planes

    paths = [np.array([[3., 10., 3.], [10., 10., 3.]]),
             np.array([[10., 10., 3.], [20., 10., 3.]]),
             np.array([[10., 10., 3.], [10., 15., 3.]])]
    labels = np.zeros((25, 20, 7), dtype=np.int16)
    for path in paths:
        for point in np.linspace(*path, 30):
            labels[tuple(np.rint(point).astype(int))] = 1
    labels = binary_dilation(labels).astype(np.int16)
    flow = np.zeros(labels.shape + (2, 3)) if flow_available else None
    planes, _, _, qc = generate_label_planes(paths, labels, {"MPA": 1}, {"MPA": 1},
        flow_xyzt3=flow, smoothing_window=3, inter_time=10)
    assert len(planes) == 1 and planes[0].path_index == 1
    assert qc[0]["selection_source"] == "automatic_with_fallback"
    assert qc[0]["selection_warnings"] == ["flow_direction_unreliable" if flow_available else "flow_unavailable"]
    assert qc[0]["selected_main_paths"][0]["selection_reason"] == "geometry_fallback_longest"


def test_label_plane_cycle_phantom_uses_visible_fallback():
    from scipy.ndimage import binary_dilation
    from autoflow.algorithms.label_planes import generate_label_planes

    vertices = np.array([[5., 5., 3.], [19., 5., 3.], [19., 19., 3.], [5., 19., 3.]])
    paths = [np.linspace(vertices[i], vertices[(i + 1) % 4], 15) for i in range(4)]
    labels = np.zeros((25, 25, 7), dtype=np.int16)
    for path in paths:
        labels[tuple(path.astype(int).T)] = 1
    labels = binary_dilation(labels).astype(np.int16)
    x, y, _ = np.indices(labels.shape)
    flow = np.zeros(labels.shape + (1, 3))
    flow[..., 0, 0] = -(y - 12)
    flow[..., 0, 1] = x - 12
    planes, _, _, qc = generate_label_planes(paths, labels, {"ACA": 1}, {"ACA": 1},
        flow_xyzt3=flow, smoothing_window=3, inter_time=1)
    assert len(planes) == 1 and planes[0].path_index == 0
    assert qc[0]["selection_warnings"] == ["directed_cycle_no_root"]


def test_label_plane_curved_phantom_smoothing_cannot_create_extra_roots():
    from scipy.ndimage import binary_dilation
    from autoflow.algorithms.label_planes import generate_label_planes
    from autoflow.algorithms.paths import inter_points, smooth_path_savgol
    from autoflow.algorithms.planes import _sample_path_labels

    x = np.linspace(4., 35., 40)
    path = np.column_stack((x, 12 + 7 * np.sin((x - 4) * .32), np.full(40, 3.)))
    labels = np.zeros((40, 25, 7), dtype=np.int16)
    dense = inter_points(path, time=5)
    labels[tuple(np.rint(dense).astype(int).T)] = 1
    labels = binary_dilation(labels).astype(np.int16)
    unconstrained = smooth_path_savgol(dense, window=151, polyorder=2)
    assert np.any(_sample_path_labels(unconstrained, labels, np.ones(3), np.zeros(3)) == 0)
    flow = np.zeros(labels.shape + (2, 3))
    flow[..., 0] = 20.
    source = path.copy(), labels.copy()
    planes, smooth, _, qc = generate_label_planes([path], labels, {"RRA": 1}, {"RRA": 3},
        flow_xyzt3=flow, smoothing_window=151, inter_time=5)
    assert qc[0]["main_path_count"] == 1 and len(planes) == 3
    assert qc[0]["selected_main_paths"][0]["smoothing_label_restored_samples"] > 0
    assert np.all(_sample_path_labels(smooth[0], labels, np.ones(3), np.zeros(3)) == 1)
    assert all(labels[tuple(np.rint(p.center).astype(int))] == 1 for p in planes)
    np.testing.assert_array_equal(path, source[0])
    np.testing.assert_array_equal(labels, source[1])


def test_label_plane_phantom_pipeline_offsets_origin_and_qc(tmp_path):
    ws = Workspace()
    ws.paths.output_dir = str(tmp_path)
    ws.origin = np.array([100., 200., 300.])
    labels = np.zeros((35, 14, 7), dtype=np.int16)
    labels[1:32, 1:6, 1:6] = 1
    labels[1:32, 8:13, 1:6] = 21
    ws.segmask_raw = np.repeat(labels[..., None], 2, axis=3)
    ws.segmask_labels = ws.segmask_raw.copy()
    ws.segmask_labels_3d = labels
    ws.segmask_binary = ws.segmask_raw > 0
    ws.flow_raw = np.zeros(labels.shape + (2, 3), dtype=np.float32)
    ws.flow_raw[..., 0] = ws.segmask_binary * 20.
    ws.skeleton_params.label_map = {"AAO": 1, "BA": 21}
    ws.branch_labels = np.where(labels == 1, 1, np.where(labels == 21, 2, 0)).astype(np.int16)
    ws.plane_gen_params.label_rules = {"AAO": {"plane_count": 3}, "BA": {"plane_count": 2}}
    ws.plane_gen_params.smoothing_window = 3
    ws.plane_gen_params.inter_time = 1
    ws.centerline_paths = [np.column_stack((np.arange(1., 32.), np.full(31, y), np.full(31, 3.)))
                           for y in (3., 10.)]
    ws.graph = GraphData(points=np.vstack(ws.centerline_paths))
    ws.group_order = ["aorta", "brain"]
    ws.path_info = [{"group_name": name} for name in ws.group_order]
    ws.multilabel_groups = {
        name: {"centerline_paths": [ws.centerline_paths[i]], "path_index_offset": i}
        for i, name in enumerate(ws.group_order)
    }
    engine = PipelineEngine()
    result = engine._step_generate_planes(ws)
    assert result.success and len(ws.planes) == 5, result.message
    assert [p.path_index for p in ws.planes] == [0, 0, 0, 1, 1]
    assert [p.segmentation_label for p in ws.planes] == [1, 1, 1, 21, 21]
    layout = json.loads((tmp_path / "plane_qc.json").read_text())
    assert layout["requested_plane_count"] == layout["actual_plane_count"] == 5
    assert all(q["status"] == "ok" for q in layout["labels"])
    records = json.loads((tmp_path / "planes.json").read_text())
    assert len(records) == 5
    metrics, qc, _ = engine._compute_plane_metrics_internal(ws, save=True, include_derived=False)
    assert len(metrics) == 5
    assert qc["plane_layout"]["plane_mode"] == "by_label"
    assert qc["plane_layout"]["labels"] == layout["labels"]
    assert qc["plane_layout"]["requested_plane_count"] == 5


def test_label_plane_phantom_pipeline_counts_each_aca_root(tmp_path):
    ws = Workspace()
    ws.paths.output_dir = str(tmp_path)
    labels = np.zeros((35, 16, 7), dtype=np.int16)
    for y in (4, 11):
        labels[1:32, y - 1:y + 2, 1:6] = 30
    ws.segmask_raw = labels[..., None]
    ws.segmask_labels_3d = labels
    ws.segmask_binary = ws.segmask_raw > 0
    ws.flow_raw = np.zeros(labels.shape + (1, 3), dtype=np.float32)
    ws.flow_raw[..., 0] = 20
    ws.skeleton_params.label_map = {"ACA": 30}
    ws.plane_gen_params.label_rules = {"ACA": {"plane_count": 2, "main_path": "inflow"}}
    ws.plane_gen_params.smoothing_window = 3
    ws.plane_gen_params.inter_time = 1
    ws.centerline_paths = [np.column_stack((np.arange(1., 32.), np.full(31, y), np.full(31, 3.)))
                           for y in (4., 11.)]
    ws.graph = GraphData(points=np.vstack(ws.centerline_paths))
    ws.branch_labels = np.where(labels > 0, np.where(np.indices(labels.shape)[1] < 8, 1, 2), 0).astype(np.int16)
    ws.group_order = ["brain"]
    ws.path_info = [{"group_name": "brain"}, {"group_name": "brain"}]
    ws.multilabel_groups = {"brain": {"centerline_paths": ws.centerline_paths, "path_index_offset": 0}}
    result = PipelineEngine()._step_generate_planes(ws)
    assert result.success and len(ws.planes) == 4
    layout = json.loads((tmp_path / "plane_qc.json").read_text())
    assert layout["requested_plane_count"] == layout["actual_plane_count"] == 4
    row = layout["labels"][0]
    assert row["main_path_count"] == 2 and row["plane_count_per_main_path"] == 2
    assert [root["actual_plane_count"] for root in row["selected_main_paths"]] == [2, 2]
    assert row["status"] == "ok"


def test_fov_plane_phantom_pipeline_preserves_excluded_candidate_ids_and_qc(tmp_path):
    ws = Workspace()
    ws.paths.output_dir = str(tmp_path)
    labels = np.zeros((20, 20, 20), dtype=np.int16)
    labels[0:8, 4:12, 1:18] = 1
    labels[10:18, 4:12, 1:18] = 21
    ws.segmask_raw = labels[..., None]
    ws.segmask_labels = ws.segmask_raw.copy()
    ws.segmask_labels_3d = labels.copy()
    ws.segmask_binary = ws.segmask_raw > 0
    ws.flow_raw = np.zeros(labels.shape + (1, 3), dtype=np.float32)
    ws.skeleton_params.label_map = {"AAO": 1, "BA": 21}
    ws.branch_labels = np.where(labels == 1, 1, np.where(labels == 21, 2, 0)).astype(np.int16)
    ws.plane_gen_params.label_rules = {"AAO": 1, "BA": 1}
    ws.plane_gen_params.smoothing_window = 3
    ws.plane_gen_params.inter_time = 1
    ws.centerline_paths = [np.column_stack((np.full(17, x), np.full(17, 8.), np.arange(1., 18.)))
                           for x in (4., 14.)]
    ws.graph = GraphData(points=np.vstack(ws.centerline_paths))
    ws.group_order = ["aorta", "brain"]
    ws.path_info = [{"group_name": name} for name in ws.group_order]
    ws.multilabel_groups = {name: {"centerline_paths": [ws.centerline_paths[i]], "path_index_offset": i}
                           for i, name in enumerate(ws.group_order)}
    engine = PipelineEngine()
    result = engine._step_generate_planes(ws)
    assert result.success and len(ws.planes) == 1, result.message
    layout = json.loads((tmp_path / "plane_qc.json").read_text())
    assert layout["fov_boundary_max_fraction"] == 0.25
    assert layout["planes"][0]["plane_index_source"] == 0
    assert layout["planes"][0]["plane_index"] is None
    assert layout["planes"][0]["reason"] == "fov_boundary_contact"
    assert layout["planes"][1]["plane_index_source"] == 1
    assert layout["planes"][1]["plane_index"] == 0
    assert layout["labels"][0]["excluded_plane_counts"] == {"fov_boundary_contact": 1}
    assert layout["labels"][0]["selected_main_paths"][0]["actual_plane_count"] == 0
    _, qc, _ = engine._compute_plane_metrics_internal(ws, save=True, include_derived=False)
    assert qc["plane_layout"]["fov_boundary_max_fraction"] == 0.25
    assert qc["plane_layout"]["planes"] == layout["planes"]
    report = build_quality_report(ws)
    check = next(item for item in report["checks"] if item["id"] == "planes.exclusions")
    assert check["status"] == "warn"
    assert "FOV=8.000/32.000 mm (25.0%)" in check["details"][0]["details"][0]["summary"]
    ws.plane_gen_params.label_rules = {"AAO": {"plane_count": 1, "fov_boundary_filter": False}, "BA": 1}
    assert engine._step_generate_planes(ws).success and len(ws.planes) == 2
    records = ws.derived.plane_qc["planes"]
    assert [record["plane_index_source"] for record in records] == [0, 1]
    assert records[0]["main_path_fraction"] == .5
    assert records[0]["fov_boundary"]["status"] == "disabled"


def test_fov_plane_phantom_point_contact_does_not_discard_roi():
    import pyvista as pv
    from autoflow.algorithms.metrics.fov import measure_fov_boundary_contact
    region = pv.PolyData(np.array([[0., 3., 4.], [2., 1., 4.], [4., 3., 4.], [2., 5., 4.]]),
                         np.array([4, 0, 1, 2, 3]))
    plane = PlaneData(center=np.array([2., 3., 4.]), normal=np.array([0., 0., 1.]))
    record = measure_fov_boundary_contact(region, plane, (10, 10, 10), np.ones(3), np.zeros(3), 0.25)
    assert record["status"] == "clear"
    assert record["contact_length_mm"] == record["contact_fraction"] == 0


def test_merged_group_largest_phantom_preserves_small_attached_labels_and_motion():
    ws = Workspace()
    ws.skeleton_params.cc_filter_mode = "largest"  # Explicit legacy mode remains supported.
    first = np.zeros((30, 30, 30), dtype=np.int16)
    first[2:8, 2:8, 2:8] = 1
    first[8:10, 3:5, 3:5] = 2   # attached small label must survive
    first[20:23, 20:23, 20:23] = 2  # larger island of label 2 must be removed
    first[2:4, 20:22, 2:4] = 3   # another group survives independently
    first[10, 24, 10] = 3
    second = np.zeros_like(first)
    second[3:9, 2:8, 2:8] = 1
    second[9:11, 3:5, 3:5] = 2
    second[20:23, 20:23, 20:23] = 2
    second[first == 3] = 3
    ws.segmask_raw = np.stack([first, second], axis=3)
    ws.flow_raw = np.zeros(first.shape + (2, 3), dtype=np.float32)
    ws.skeleton_params.label_groups = {"mixed": {"labels": [1, 2]}, "other": {"labels": [3]}}
    ws.skeleton_params.do_closing = False
    ws.skeleton_params.gaussian_enabled = False
    assert ws.skeleton_params.remove_small_cc and ws.skeleton_params.cc_filter_mode == "largest"
    engine = PipelineEngine()
    assert engine.preprocess(ws)
    assert np.any(ws.segmask_labels_3d == 2)
    assert not ws.multilabel_groups["mixed"]["clean_mask_3d"][21, 21, 21]
    assert ws.multilabel_groups["other"]["clean_mask_3d"].sum() == 8
    assert not ws.multilabel_groups["other"]["clean_mask_3d"][10, 24, 10]
    assert ws.multilabel_groups["mixed"]["segmask_binary"][10, 3, 3, 1]
    assert not ws.multilabel_groups["mixed"]["segmask_binary"][10, 3, 3, 0]
    assert not ws.multilabel_groups["mixed"]["segmask_binary"][21, 21, 21].any()
    assert not np.any(ws.segmask_raw != np.stack([first, second], axis=3))
    qc = {item["label"]: item for item in ws.segmentation_preprocess_qc["labels"]}
    assert qc[2]["voted_voxels"] > qc[2]["retained_group_voxels"] > 0
    assert qc[2]["component_removed_voxels"] == 27
    assert qc[2]["component_removed_fraction"] == pytest.approx(27 / qc[2]["voted_voxels"])
    assert qc[2]["phases"][0]["temporal_group_removed_fraction"] == pytest.approx(27 / 35)


@pytest.mark.parametrize("rule", [
    {"plane_count": -2}, {"plane_count": True}, {"spacing_ratio": float("nan")},
    {"plane_count": 3, "spacing_ratio": 1.7}, {"main_path": "both"},
    {"main_path": []}, {"unknown": True},
    {"plane_mode": "by_label"}, {"anchor": "unknown"}, {"inter_time": 0},
    {"fov_boundary_max_fraction": 1.1}, {"position_shift_mm": 1},
])
def test_label_plane_phantom_rejects_invalid_rules(rule):
    from autoflow.plane_rules import normalize_label_plane_rules
    with pytest.raises(ValueError):
        normalize_label_plane_rules({"AAO": rule})


@pytest.mark.parametrize("mode,extra,expected", [
    ("uniform", {"plane_count": 4, "start_distance": 3, "end_distance": 3}, [3, 11, 19, 27]),
    ("count", {"plane_count": 1, "start_distance": 4, "end_distance": 2}, [16]),
    ("distance", {"plane_count": -1, "cross_section_distance": 10}, [0, 10, 20, 30]),
    ("fixed_step", {"plane_count": 3, "anchor": "start", "direction": "toward_end",
                    "spacing_mode": "distance", "cross_section_distance": 7}, [0, 7, 14]),
    ("anchored_offset", {"plane_count": 3, "anchor_offset_mm": 4, "cross_section_distance": 6}, [4, 10, 16]),
])
@pytest.mark.parametrize("sign", [1, -1])
def test_label_plane_layout_phantom_matches_global_modes_in_flow_coordinates(mode, extra, expected, sign):
    from autoflow.algorithms.label_planes import generate_label_planes
    labels = np.ones((35, 5, 5), dtype=np.int16)
    path = np.column_stack((np.arange(1., 32.), np.full(31, 2.), np.full(31, 2.)))
    flow = np.zeros(labels.shape + (1, 3))
    flow[..., 0] = 20 * sign
    fork = path[[0 if sign == 1 else -1]]
    planes, _, _, qc = generate_label_planes([path], labels, {"AAO": 1},
        {"AAO": {"plane_mode": mode, **extra}}, flow_xyzt3=flow,
        fork_points=fork.tolist(), smoothing_window=3, inter_time=1)
    actual = [(plane.center[0] - 1) if sign == 1 else (31 - plane.center[0]) for plane in planes]
    assert actual == pytest.approx(expected, abs=1e-8)
    assert all(plane.normal[0] * sign > 0 for plane in planes)
    assert qc[0]["requested_plane_count"] == len(expected)


def test_group_preprocessing_phantom_filters_after_morphology_and_keeps_bilateral_lumens():
    from autoflow.algorithms.preprocess import preprocess_mask_for_skeleton
    from autoflow.core.models import SkeletonParams
    source = np.zeros((35, 35, 35), dtype=bool)
    source[3:8, 3:8, 3:8] = True
    source[15:31, 20, 20] = True  # Below 50 mm3 before dilation, above it after.
    params = SkeletonParams(gaussian_enabled=False, dilation_iters=1)
    result, details = preprocess_mask_for_skeleton(source, params, return_details=True)
    assert np.all(result[source])
    assert np.array_equal(details["retained_source_mask"], source)

    ws = build_workspace(AutoFlowConfig.from_config_dir())
    assert not ws.skeleton_params.do_closing
    assert all(ws.skeleton_params.params_for_group(name).closing_iters == 0
               for name in ws.skeleton_params.label_groups)
    assert ws.skeleton_params.cc_filter_mode == "hybrid"
    assert ws.skeleton_params.cc_rel_min_ratio == .1
    labels = np.zeros((35, 35, 35), dtype=np.int16)
    labels[2:10, 2:10, 2:10] = 17
    labels[20:26, 20:26, 20:26] = 20
    labels[12:15, 12:15, 12:15] = 17
    ws.segmask_raw = labels
    ws.skeleton_params.gaussian_enabled = False
    PipelineEngine().preprocess(ws)
    assert np.count_nonzero(ws.segmask_labels_3d == 17) == 512
    assert np.count_nonzero(ws.segmask_labels_3d == 20) == 216
    assert not ws.segmask_3d[13, 13, 13]
    qc = ws.segmentation_preprocess_qc
    assert qc["preprocessing_order"].index("connected_component_filter") > qc["preprocessing_order"].index("morphology_and_gaussian")


def test_label_plane_phantom_short_renal_count_needs_no_note():
    from autoflow.plane_rules import normalize_label_plane_rules
    rule = normalize_label_plane_rules({"LRA": 2})["LRA"]
    assert rule["spacing_ratio"] == pytest.approx(1 / 3)
    assert "reason" not in rule
    assert normalize_label_plane_rules({"LICA": 0})["LICA"]["plane_count"] == 0
    assert "reason" not in normalize_label_plane_rules({"LRA": {"plane_count": 2, "reason": "old note"}})["LRA"]


def test_label_plane_phantom_gui_configures_root_side(tmp_path, monkeypatch):
    from PySide6 import QtWidgets
    from autoflow.ui.action_settings import ActionSettingsDialog, action_sections

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    ws = Workspace()
    ws.skeleton_params.label_map = {"AAO": 1}
    ws.segmask_labels_3d = np.zeros((35, 14, 5), dtype=np.int16)
    ws.centerline_paths = []
    for y in (3., 10.):
        ws.segmask_labels_3d[1:32, int(y) - 1:int(y) + 2, 1:4] = 1
        ws.centerline_paths.append(np.column_stack((np.arange(1., 32.), np.full(31, y), np.full(31, 2.))))
    ws.plane_gen_params.label_rules = {"AAO": {"plane_count": 3}, "LICA": {"plane_count": 0}, "LPCA": {"plane_count": 1}}
    dialog = ActionSettingsDialog("Planes", action_sections(StepId.GENERATE_PLANES, ws), ws, ws)
    try:
        table = dialog.controls["plane_gen_params.label_rules"]
        assert table.rowCount() == 3 and table.columnCount() == 7
        combo = table.cellWidget(0, 3)
        assert combo.count() == 3
        assert combo.currentData() == "inflow"
        combo.setCurrentIndex(combo.findData("outflow"))
        table.cellWidget(0, 1).setValue(2)
        ratio = table.cellWidget(0, 2)
        ratio.interpretText()
        assert abs(ratio.value() - 1 / 3) < 1e-12
        values = dialog.values()
        rule = values["plane_gen_params.label_rules"]["AAO"]
        assert rule["main_path"] == "outflow"
        assert not {"path_index", "segment_index", "require_pv_connection"} & set(rule)
        assert rule["plane_count"] == 2 and rule["spacing_ratio"] == pytest.approx(1 / 3)
        assert dialog.rows["plane_gen_params.plane_count"].isHidden()
        assert dialog.values()["plane_gen_params.label_rules"]["LICA"]["plane_count"] == 0
        from autoflow.plane_rules import load_plane_rules_file, save_plane_rules_file
        shared = tmp_path / "shared.json"
        monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", lambda *args: (str(shared), "JSON"))
        dialog._save_plane_config()
        shared_rules = load_plane_rules_file(shared)
        assert shared_rules["AAO"]["main_path"] == "outflow"
        assert shared_rules["AAO"]["plane_count"] == 2
        assert not any("reason" in value for value in shared_rules.values())
        assert build_workspace(AutoFlowConfig(plane_config=str(shared))).plane_gen_params.label_rules == shared_rules
        other = tmp_path / "only_aao.json"
        save_plane_rules_file(other, {"AAO": 1})
        monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName", lambda *args: (str(other), "JSON"))
        dialog._load_plane_config()
        assert table.rowCount() == 1
        assert set(dialog.values()["plane_gen_params.label_rules"]) == {"AAO"}
        assert dialog.values()["plane_gen_params.label_rules_file"] == str(other)
        def edit_parameters(child):
            mode = child.controls["plane_gen_params.plane_mode"]
            assert {mode.itemData(i) for i in range(mode.count())} == {"fixed_step", "uniform", "anchored_offset", "count", "distance"}
            mode.setCurrentIndex(mode.findData("fixed_step"))
            anchor = child.controls["plane_gen_params.anchor"]
            anchor.setCurrentIndex(anchor.findData("start"))
            spacing = child.controls["plane_gen_params.spacing_mode"]
            spacing.setCurrentIndex(spacing.findData("distance"))
            child.controls["plane_gen_params.cross_section_distance"].setValue(7)
            child.controls["plane_gen_params.fov_boundary_filter"].setChecked(False)
            child.controls["plane_gen_params.smoothing_window"].setValue(5)
            child.accept()
            return child.result()
        monkeypatch.setattr(ActionSettingsDialog, "exec", edit_parameters)
        table.cellWidget(0, 4).setCurrentIndex(table.cellWidget(0, 4).findData("uniform"))
        dialog._edit_label_rule(table, 0)
        rule = dialog.values()["plane_gen_params.label_rules"]["AAO"]
        assert rule["plane_mode"] == "fixed_step" and rule["anchor"] == "start"
        assert rule["spacing_mode"] == "distance" and rule["cross_section_distance"] == 7
        assert rule["fov_boundary_filter"] is False and rule["smoothing_window"] == 5
        ws.skeleton_params.label_map["DAO"] = 4
        dialog._populate_plane_rules(table, {"AAO": rule})
        assert table.rowCount() == 2
        assert dialog.values()["plane_gen_params.label_rules"]["DAO"]["plane_count"] == 0
    finally:
        dialog.close()
        dialog.deleteLater()
        app.processEvents()


def test_label_plane_config_phantom_cli_and_inline_replace_defaults(tmp_path, monkeypatch):
    import autoflow.cli as cli
    from autoflow.plane_rules import load_plane_rules_file, save_plane_rules_file

    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    (bundle_dir / "planes.json").write_text(json.dumps({"label_rules": {"AAO": 1}}))
    inline = AutoFlowConfig.from_config_dir(str(bundle_dir))
    assert set(build_workspace(inline).plane_gen_params.label_rules) == {"AAO"}
    assert build_workspace(inline).plane_gen_params.fov_boundary_max_fraction == 0.25

    shared = tmp_path / "shared.json"
    save_plane_rules_file(shared, {"BA": 2})
    assert set(load_plane_rules_file(shared)) == {"BA"}
    received = []
    def capture(config):
        received.append(build_workspace(config).plane_gen_params)
        return [{"status": "ok"}], str(tmp_path)
    monkeypatch.setattr(cli, "run_batch", capture)
    import sys
    monkeypatch.setattr(sys, "argv", ["autoflow-run", "case.h5", "--config-dir", str(bundle_dir), "--plane-config", str(shared),
                                     "--fov-boundary-max-fraction", "0.3", "--no-fov-boundary-filter"])
    assert cli.main() == 0
    assert set(received[0].label_rules) == {"BA"}  # no AAO or other defaults are merged in
    assert received[0].label_rules_file == str(shared)
    assert received[0].fov_boundary_max_fraction == 0.3 and not received[0].fov_boundary_filter
    assert not list(tmp_path.glob(".plane_rules_*.tmp"))

    # Loading a replacement config never generates planes for omitted labels.
    from autoflow.algorithms.label_planes import generate_label_planes
    labels = np.zeros((40, 8, 5), dtype=np.int16)
    labels[1:31, 1:4, 1:4] = 1
    labels[1:31, 5:8, 1:4] = 4
    paths = [np.column_stack((np.arange(1., 31.), np.full(30, y), np.full(30, 2.))) for y in (2., 6.)]
    planes, _, _, qc = generate_label_planes(paths, labels, {"AAO": 1, "DAO": 4},
                                             {"AAO": 1}, smoothing_window=3, inter_time=1)
    assert len(planes) == 1 and planes[0].segmentation_label == 1
    assert [entry["label_name"] for entry in qc] == ["AAO", "DAO"]
    assert qc[1]["status"] == "not_configured" and qc[1]["reason"] == "not_in_plane_config"


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


@pytest.mark.parametrize("support_kind", ["image", "unstructured"])
def test_wss_resolves_supported_normal_direction(support_kind):
    import pyvista as pv
    from autoflow.algorithms.surfaces import create_uniform_grid

    grid = pv.ImageData(dimensions=(41, 9, 9), spacing=(0.1, 0.5, 0.5), origin=(-2, -2, -2))
    s = -grid.points[:, 0]
    grid.point_data["u"] = np.zeros(len(s))
    grid.point_data["v"] = 0.4 * s + 0.04 * s**2
    grid.point_data["w"] = np.zeros(len(s))
    mask = np.zeros((4, 4, 4), dtype=np.uint8)
    mask[:2] = 1
    support = create_uniform_grid(mask, (1, 1, 1), origin=(-2, -2, -2), name="wss_lumen")
    if support_kind == "unstructured":
        support = support.cast_to_unstructured_grid()
    # The supplied point normal points outward; the lumen resolves its sign.
    wall = pv.Plane(center=(0, 0, 0), direction=(1, 0, 0), i_size=1, j_size=1,
                    i_resolution=2, j_resolution=2)
    result = cal_wss_from_surf(wall, grid, inward_distance=0.5, viscosity=4.0,
                               support_grid=support)
    assert np.all(result["wss_valid"])
    assert np.all(result["wss_normal_flipped"])
    np.testing.assert_allclose(result.point_normals[:, 0], -1.0)
    np.testing.assert_allclose(result["wss_vectors"], np.tile([0, 1.6, 0], (wall.n_points, 1)), atol=1e-10)
    np.testing.assert_allclose(result.points[:, 0], 0.0, atol=1e-7)


def test_wss_linear_mode_and_invalid_wall_normal_samples():
    import pyvista as pv
    from autoflow.rendering.style import wss_display_surface, wss_scalar_range

    np.testing.assert_allclose(calculate_gradient([0.0], [0.3], [0.4], 1.0, use_parabolic=False), [0.3])
    mask = np.zeros((8, 8, 8, 2), dtype=bool)
    mask[2:6, 2:6, 2:6] = True
    flow = np.zeros(mask.shape + (3,), dtype=np.float32)
    result = compute_wss_metrics(mask, flow, (1, 1, 1), smoothing_iteration=0, inward_distance=10)
    for wall in result["wss_surfaces"]:
        assert not np.any(wall["wss_valid"])
        assert np.isnan(wall["wss"]).all()
        assert np.isnan(wall["wss_vectors"]).all()
        assert np.isnan(wss_display_surface(wall)["wss"]).all()
    assert np.isnan(result["wss_volume"]).any()
    assert wss_scalar_range(result["wss_surfaces"]) == (0.0, 1.0)

    # An invalid finite probe and infinity must remain unavailable in the
    # view; a measured zero is still valid. Display preparation cannot alter
    # the scientific arrays used for metrics and exports.
    wall = pv.Plane(i_resolution=1, j_resolution=1)
    wall["wss"] = [0.0, 1.0, np.inf, 2.0]
    wall["wss_valid"] = [1, 0, 1, 1]
    before = wall["wss"].copy()
    display = wss_display_surface(wall)
    np.testing.assert_array_equal(wall["wss"], before)
    np.testing.assert_array_equal(wall["wss_valid"], [1, 0, 1, 1])
    np.testing.assert_array_equal(display["wss"][[0, 3]], [0.0, 2.0])
    assert np.isnan(display["wss"][[1, 2]]).all()
    assert wss_scalar_range([wall]) == (0.0, 2.0)


def test_metric_display_preserves_constant_values_at_supported_wall():
    from autoflow.rendering.datasets import display_support_surface, sample_display_field, active_wrap_mesh

    mask = np.zeros((9, 10, 11), dtype=bool)
    mask[2:7, 2:8, 2:9] = True
    spacing, origin = np.array([1.0, 2.0, 3.0]), np.array([-4.0, 7.0, 2.0])
    field = np.where(mask, 12.0, np.nan).astype(np.float32)
    wall = display_support_surface(mask, spacing, origin)
    assert wall.n_open_edges == 0
    display = sample_display_field(field, mask, wall, spacing, origin, 'metric')
    np.testing.assert_allclose(display['metric'], 12.0, atol=1e-5)
    assert np.isnan(field[~mask]).all()

    counts = np.zeros(mask.shape + (2, 3), dtype=np.int16)
    counts[3, 4, 5, 0, 1] = -2
    counts[4, 4, 5, 0, 2] = 1
    result = {'wrap_count': counts, 'wrap_mask': counts != 0}
    wrap = active_wrap_mesh(result, 0, spacing, origin, counts=True)
    assert set(np.unique(wrap['wrap_count'])) == {-2, 1}
    assert not np.any(wrap['wrap_count'] == 0)
    assert active_wrap_mesh(result, 1, spacing, origin, counts=True) is None


def test_vortex_display_tracks_phase_specific_support_geometry():
    from contextlib import closing
    import pyvista as pv
    from autoflow.ui.viewer import SceneController

    ws = Workspace()
    support = np.zeros((14, 8, 8, 3), dtype=np.uint8)
    support[2:5, 2:6, 2:6, 0] = 1
    support[7:10, 2:6, 2:6, 1] = 1
    support[..., 2] = support[..., 0]
    ws.derived.vortex_support_mask = support
    ws.derived.vorticity_magnitude = support.astype(np.float32) * 20.0
    with closing(pv.Plotter(off_screen=True)) as plotter:
        scene = SceneController(plotter, ws, lambda _: None)
        centers = []
        for phase in range(3):
            ws.current_t = phase
            mesh = scene._build_dataset('vorticity_magnitude_volume')
            centers.append(mesh.center)
            np.testing.assert_allclose(mesh['Vorticity Magnitude'], 20.0, atol=1e-5)
        np.testing.assert_allclose(np.subtract(centers[1], centers[0]), [5., 0., 0.], atol=1e-5)
        np.testing.assert_allclose(centers[2], centers[0], atol=1e-5)


def test_tke_volume_preserves_energy_and_keeps_colourbar_opaque():
    from contextlib import closing
    import pyvista as pv
    from autoflow.core.models import ObjectKind
    from autoflow.ui.viewer import SceneController

    ws = Workspace()
    mask = np.zeros((8, 8, 8, 2), dtype=bool)
    mask[2:6, 2:6, 2:6] = True
    ws.segmask_binary = mask
    energy = np.full(mask.shape, np.nan, dtype=np.float32)
    energy[mask] = 12.0
    energy[..., 1][mask[..., 1]] = 0.0
    ws.derived.tke_array = energy
    uid = ws.add_object('TKE', ObjectKind.METRIC, 'tke_volume', visible=True,
                        scalars='TKE', cmap='inferno', clim=(0, 20), opacity=1.0, dynamic=True)
    errors = []
    with closing(pv.Plotter(off_screen=True)) as plotter:
        scene = SceneController(plotter, ws, errors.append)
        obj = ws.scene_objects[uid]
        scene._render_object(obj)
        assert obj.actor is not None
        volume = obj.actor.mapper.dataset
        assert isinstance(volume, pv.ImageData)
        values = volume['TKE'].reshape((10, 10, 10), order='F')[1:-1, 1:-1, 1:-1]
        np.testing.assert_allclose(values[mask[..., 0]], 12.0)
        assert not np.any(values[~mask[..., 0]])
        assert np.all(obj.actor.mapper.lookup_table.values[:, 3] == 255)
        assert obj.actor.GetProperty().GetScalarOpacity(0).GetValue(0.0) == 0.0
        obj.opacity = 0.0
        scene.apply_object_properties(obj)
        assert obj.actor.GetProperty().GetScalarOpacity(0).GetValue(12.0) == 0.0
        scene.update_time(1)
        assert not np.any(obj.actor.mapper.dataset['TKE'])
    assert np.isnan(energy[~mask]).all()
    assert not errors


def test_tke_sigma_units_and_masked_invalid_background():
    from autoflow.algorithms.metrics.tke import compute_tke_metrics, compute_tke_array_from_sigma

    sigma = np.full((5, 5, 5, 2, 3), 10.0)
    np.testing.assert_allclose(compute_tke_array_from_sigma(sigma), 15.9, rtol=1e-6)
    mask = np.zeros(sigma.shape[:4], dtype=bool)
    mask[1:4, 1:4, 1:4] = True
    sigma[~mask] = np.nan
    result = compute_tke_metrics(mask, (1, 1, 1), sigma=sigma)
    assert not np.any(result['tke_array'][~mask])
    np.testing.assert_allclose(result['tke_array'][mask], 15.9, rtol=1e-6)
    with pytest.raises(ValueError, match='density'):
        compute_tke_array_from_sigma(sigma, rho=-1)


def test_vortex_excludes_nonfinite_stencil_even_without_extra_erosion():
    from autoflow.algorithms.metrics.vortex import compute_vortex_metrics

    mask = np.ones((9, 9, 9, 1), dtype=bool)
    xyz = np.indices(mask.shape[:3], dtype=float)
    flow = np.zeros(mask.shape + (3,), dtype=float)
    flow[..., 0] = (-xyz[1])[..., None]
    flow[..., 1] = xyz[0][..., None]
    flow[4, 4, 4, 0, 0] = np.nan
    result = compute_vortex_metrics(mask, flow, (1, 1, 1), support_erosion_iters=0)
    support = result['vortex_support_mask'].astype(bool)
    assert not support[4, 4, 4, 0]
    assert not support[3, 4, 4, 0]
    assert support[2, 2, 2, 0]
    np.testing.assert_allclose(result['vorticity_magnitude'][support], 20.0)
    assert np.isfinite(result['swirling_strength_array']).all()


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
    original_extract_surface = wss._prepare_wss_wall

    def counting_surface(*args, **kwargs):
        nonlocal surface_calls
        surface_calls += 1
        return original_extract_surface(*args, **kwargs)

    monkeypatch.setattr(wss, "_prepare_wss_wall", counting_surface)
    result = compute_wss_metrics(
        mask, flow, (1.0, 1.0, 1.0), smoothing_iteration=0,
    )
    assert surface_calls == 1
    assert len(result["wss_surfaces"]) == mask.shape[3]
    for wall in result["wss_surfaces"]:
        assert np.all(wall["wss_valid"])
        assert np.all(np.sum((wall.points - np.array([4, 4, 4])) * wall.point_normals, axis=1) < 0)


def test_wss_closed_wall_preserves_cavity_and_separate_lumens():
    from autoflow.algorithms.metrics.wss import _prepare_wss_wall

    spacing = np.array([1.0, 1.0, 2.0])
    origin = np.array([-5.0, 3.0, 9.0])
    mask = np.zeros((24, 16, 16), dtype=bool)
    mask[2:12, 2:14, 2:14] = True
    mask[5:9, 5:11, 5:11] = False
    mask[17:22, 4:12, 4:12] = True
    wall, support = _prepare_wss_wall(mask, spacing, origin, 200)
    assert wall.n_open_edges == 0
    indices = np.array([[3, 7, 7], [6, 7, 7], [14, 7, 7], [19, 7, 7], [0, 0, 0]])
    points = origin + (indices + 0.5) * spacing
    np.testing.assert_array_equal(support.contains(points), [True, False, False, True, False])


@pytest.mark.parametrize("spacing,direction", [((2.5, 2.5, 2.5), (0, 0, 1)),
                                              ((1.0, 1.0, 2.5), (1, 1, 2))])
def test_wss_poiseuille_curved_wall_phantom(spacing, direction):
    spacing = np.asarray(spacing, dtype=float)
    axis = np.asarray(direction, dtype=float)
    axis /= np.linalg.norm(axis)
    radius, length = 10.0, 40.0
    extent = radius + np.abs(axis) * length / 2 + 4 * np.max(spacing)
    shape = np.ceil(2 * extent / spacing).astype(int)
    origin = -shape * spacing / 2
    xyz = np.stack(np.meshgrid(*[origin[k] + (np.arange(size) + 0.5) * spacing[k]
                                for k, size in enumerate(shape)], indexing="ij"), axis=-1)
    axial = xyz @ axis
    r2 = np.sum(xyz * xyz, axis=-1) - axial * axial
    mask = (r2 < radius ** 2) & (np.abs(axial) < length / 2)
    flow = 100 * np.maximum(0.5 * (1 - r2 / radius ** 2), 0)[..., None] * axis
    wall = compute_wss_metrics(mask[..., None], flow[..., None, :], spacing,
                               origin=origin)["wss_surfaces"][0]
    side = np.abs(np.asarray(wall.points) @ axis) < length / 4
    assert wall.n_open_edges == 0
    assert np.all(wall["wss_valid"][side])
    true_wss = 0.4
    # This end-to-end bound includes reconstructed-wall and interpolation
    # error, unlike a test of the derivative formula on exact point fields.
    assert np.mean(np.abs(wall["wss"][side] - true_wss)) / true_wss < 0.15
    vectors = np.asarray(wall["wss_vectors"])[side]
    assert np.all(vectors @ axis > 0)


def test_wss_thin_lumen_does_not_gain_unsupported_samples():
    mask = np.zeros((12, 12, 12, 1), dtype=bool)
    # Touching the acquisition boundary must still yield a closed wall, and
    # its broad faces cannot support a two-voxel path across the thin axis.
    mask[0:10, 5:6, 3:9] = True
    flow = np.ones(mask.shape + (3,), dtype=float) * 10
    wall = compute_wss_metrics(mask, flow, (1, 1, 1))["wss_surfaces"][0]
    assert wall.n_open_edges == 0
    thin_faces = np.abs(wall.point_normals[:, 1]) > 0.9
    assert np.any(thin_faces)
    assert not np.any(wall["wss_valid"][thin_faces])
    assert np.isnan(wall["wss"][thin_faces]).all()


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


def test_planar_streamline_phantom_respects_anisotropic_spacing_and_mask_gaps():
    from autoflow.algorithms.slice_streamlines import trace_planar_streamlines, limit_polyline_length

    vectors = np.zeros((32, 18, 2))
    vectors[..., 0] = 20.0
    mask = np.zeros(vectors.shape[:2], dtype=bool)
    mask[2:14, 3:15] = True
    mask[18:30, 3:15] = True
    spacing, offset = np.array([2.0, 0.75]), np.array([-10.0, 4.0])
    curves = trace_planar_streamlines(vectors, spacing, mask=mask, offset=offset)
    assert curves
    for curve in curves:
        assert np.all(np.isfinite(curve))
        assert np.all(np.diff(curve[:, 0]) > 0.0)
        np.testing.assert_allclose(curve[:, 1], curve[0, 1])
        indices = np.rint((curve - offset) / spacing).astype(int)
        assert np.all(mask[indices[:, 0], indices[:, 1]])
        assert np.all(indices[:, 0] < 14) or np.all(indices[:, 0] >= 18)
        trimmed = limit_polyline_length(curve, 0.1)
        assert np.sum(np.linalg.norm(np.diff(trimmed, axis=0), axis=1)) == pytest.approx(0.1)
    assert trace_planar_streamlines(np.zeros_like(vectors), spacing) == []
    assert trace_planar_streamlines(vectors, spacing, mask=np.zeros_like(mask)) == []
    assert trace_planar_streamlines(vectors * np.nan, spacing) == []
    assert trace_planar_streamlines(vectors, spacing, terminal_speed=30.0) == []


def test_planar_streamline_vortex_phantom_follows_circular_in_plane_velocity():
    from autoflow.algorithms.slice_streamlines import trace_planar_streamlines

    x, y = np.meshgrid(np.arange(41) - 20.0, np.arange(41) - 20.0, indexing="ij")
    radius = np.hypot(x, y)
    curves = trace_planar_streamlines(
        np.stack((-y, x), axis=-1), (1.0, 1.0),
        mask=(radius > 8.0) & (radius < 16.0), offset=(-20.0, -20.0), max_steps=120,
    )
    assert curves
    for curve in curves:
        if len(curve) < 15:
            continue
        radii = np.linalg.norm(curve, axis=1)
        assert np.ptp(radii) < 0.15
        tangent = np.diff(curve, axis=0)
        assert np.all(curve[:-1, 0] * tangent[:, 1] - curve[:-1, 1] * tangent[:, 0] > 0)


def test_streamline_slab_phantom_clips_crossings_without_joining_exits_or_lines():
    import pyvista as pv
    from autoflow.algorithms.slice_streamlines import project_streamline_slab

    origin = np.array([11.0, -23.0, 31.0])
    u = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
    v = np.array([0.0, 0.0, 1.0])
    n = np.cross(u, v)
    # Each crossing has both endpoints outside the slab. The return trip
    # creates a separate visible piece; the parallel outside line disappears.
    local = np.array([[-3, 0, -3], [3, 0, 3], [9, 0, -3],
                      [0, 4, 0], [2, 4, 0], [0, 8, 4], [2, 8, 4]], dtype=float)
    mesh = pv.PolyData(origin + local @ np.stack((u, v, n)))
    mesh.lines = np.array([3, 0, 1, 2, 2, 3, 4, 2, 5, 6])
    curves = project_streamline_slab(mesh, origin, u, v, thickness=2.0)
    assert len(curves) == 3
    np.testing.assert_allclose(curves[0], [[-1, 0], [1, 0]], atol=1e-10)
    np.testing.assert_allclose(curves[1], [[5, 0], [7, 0]], atol=1e-10)
    np.testing.assert_allclose(curves[2], [[0, 4], [2, 4]], atol=1e-10)
    # Cached VTK backward integrations have the opposite point ordering.
    # Arrow direction must still agree with the vector used by the tracer.
    mesh.point_data["vector"] = np.tile(-u, (mesh.n_points, 1))
    oriented = project_streamline_slab(mesh, origin, u, v, thickness=2.0)
    for curve in oriented:
        assert np.all(np.diff(curve[:, 0]) < 0.0)
    assert project_streamline_slab(None, origin, u, v, 2.0) == []


def test_streamline_slab_phantom_generated_backward_traces_follow_velocity():
    from autoflow.algorithms.slice_streamlines import project_streamline_slab

    flow = np.zeros((18, 18, 18, 1, 3), dtype=np.float32)
    flow[..., 2] = 30.0
    spacing, origin = np.array([1.0, 1.5, 2.0]), np.array([10.0, -20.0, 30.0])
    seed = origin + np.array([8.5, 8.5, 8.5]) * spacing
    mesh = generate_streamlines_at_t(
        flow, 0, seed[None, :], spacing, origin,
        mask_3d=np.ones(flow.shape[:3], dtype=bool), max_steps=40,
    )
    assert mesh is not None
    curves = project_streamline_slab(mesh, seed, [1, 0, 0], [0, 0, 1], 2.0)
    assert len(curves) >= 2
    for curve in curves:
        assert np.all(np.diff(curve[:, 1]) > 0.0)


@pytest.mark.parametrize("selected_plane", [False, True])
def test_slice_streamline_phantom_modes_time_geometry_and_reset(monkeypatch, selected_plane):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    import pyvista as pv
    from PySide6 import QtWidgets
    from autoflow.ui.ortho_viewer import OrthoViewer

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    ws = Workspace()
    ws.mag_raw = np.ones((18, 18, 18, 2), dtype=np.float32)
    ws.flow_raw = np.zeros(ws.mag_raw.shape + (3,), dtype=np.float32)
    ws.flow_raw[..., 0, :] = [20.0, 15.0, 10.0]
    ws.segmask_raw = np.ones(ws.mag_raw.shape, dtype=np.int16)
    ws.segmask_raw[..., 1] = 0
    ws.resolution = np.array([1.0, 1.5, 2.0])
    ws.origin = np.array([10.0, -20.0, 30.0])
    center = np.array([8.0, 12.0, 16.0])
    ws.planes = [PlaneData(center=center, normal=np.array([0.2, 0.3, 1.0]), label=1)]
    mesh = pv.PolyData(ws.origin + np.array([[2, 3, 4], [14, 21, 28]], dtype=float))
    mesh.lines = np.array([2, 0, 1])
    ws.streamline_active = True
    ws.streamline_cache = {0: mesh, 1: None}
    viewer = OrthoViewer(ws)
    try:
        viewer.update_slider_ranges()
        for slider in (viewer.slider_x, viewer.slider_y, viewer.slider_z):
            slider.setValue(8)
        viewer._apply_pending_cursor()
        if selected_plane:
            viewer.set_selected_plane(0)
        assert not viewer.check_planar_streamlines.isChecked()
        assert not viewer.check_projected_streamlines.isChecked()
        viewer.check_planar_streamlines.setChecked(True)
        viewer.check_projected_streamlines.setChecked(True)
        for view in viewer.slice_views.values():
            for items in view.streamline_items.values():
                assert items[0].isVisible()
                x, y = items[0].getData()
                assert len(x) >= 2 and len(x) == len(y)
                assert np.all(np.isfinite(x)) and np.all(np.isfinite(y))
                assert items[0].zValue() > view.noise_overlay_item.zValue()
                assert items[0].zValue() < view.vertical_line.zValue()
        if selected_plane:
            # Nonzero world origin must not move local plane sampling outside
            # an otherwise uniform image/segmentation volume.
            view = viewer.slice_views["axial"]
            mid = tuple(np.array(view.image_item.image.shape) // 2)
            assert view.image_item.image[mid] == pytest.approx(1.0)
            assert view.overlay_item.image[mid][3] > 0
        current = viewer.slice_views["axial"]
        persistent_items = current.streamline_items["planar"]
        geometry = current._streamline_geometry["planar"]
        viewer.refresh(update_plane=False)
        assert current._streamline_geometry["planar"] is geometry
        viewer.slider_streamline_opacity.setValue(40)
        assert persistent_items[0].opacity() == pytest.approx(0.4)
        viewer.spin_streamline_slab.setValue(6.0)
        viewer.set_playback_active(True)
        ws.current_t = 1
        viewer.refresh(update_plane=False)
        assert all(not items[0].isVisible() for view in viewer.slice_views.values()
                   for items in view.streamline_items.values())
        assert current.streamline_items["planar"] == persistent_items
        ws.current_t = 0
        viewer.refresh(update_plane=False)
        viewer.check_planar_streamlines.setChecked(False)
        assert not current.streamline_items["planar"][0].isVisible()
        assert current.streamline_items["projection"][0].isVisible()
        ws.streamline_active = False
        viewer.refresh(update_plane=False)
        assert "Generate Streamlines" in viewer.label_streamline_status.text()
        assert not current.streamline_items["projection"][0].isVisible()
        viewer.reset_state()
        assert not viewer.check_projected_streamlines.isChecked()
        assert not viewer._streamline_cache
        assert not current.streamline_items["projection"][0].isVisible()
    finally:
        viewer.close()
        viewer.deleteLater()
        app.processEvents()


def test_viewer_display_phantom_speed_colors_layer_styles_and_workspace_roundtrip(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    import copy
    import pyvista as pv
    from PySide6 import QtWidgets
    from autoflow.ui.ortho_viewer import OrthoViewer

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    ws = Workspace()
    ws.mag_raw = np.ones((14, 14, 14, 2), dtype=np.float32)
    ws.flow_raw = np.zeros(ws.mag_raw.shape + (3,), dtype=np.float32)
    ws.flow_raw[..., 0, 0] = np.linspace(10, 50, 14)[:, None, None]
    ws.flow_raw[..., 1, 0] = ws.flow_raw[..., 0, 0] * 2
    ws.segmask_raw = np.ones(ws.mag_raw.shape, dtype=np.int16)
    ws.segmask_raw[7:, ..., :] = 2
    ws.origin = np.array([10.0, -20.0, 30.0])
    points = np.column_stack((np.arange(2, 11), np.full(9, 2.5), np.full(9, 2.5)))
    mesh = pv.PolyData(points + ws.origin)
    mesh.lines = np.r_[9, np.arange(9)]
    mesh.point_data["vector"] = np.column_stack((np.linspace(0.1, 0.5, 9), np.zeros((9, 2))))
    ws.streamline_active = True
    ws.streamline_cache = {0: mesh, 1: mesh.copy()}
    viewer = OrthoViewer(ws)
    try:
        viewer.update_slider_ranges()
        viewer.slider_z.setValue(2)
        viewer._apply_pending_cursor()
        state = copy.deepcopy(ws.render_settings["viewer_display"])
        state["planar"].update(enabled=True, range_mode="manual", low=0, high=100, arrows=False)
        state["projection"].update(enabled=True, range_mode="manual", low=0, high=100, cmap="viridis", opacity=0.4)
        state["images"]["3"] = dict(cmap="gray", range_mode="manual", low=0, high=3)
        state["segmentation"].update(labels="1", colors={"1": "#00ff00"}, opacity=0.6)
        state["views"]["coronal"] = dict(guides=dict(crosshairs=False, orientation=False))
        viewer.apply_display_settings(state)
        view = viewer.slice_views["axial"]
        assert len(np.unique(view.streamline_items["planar"][0].segment_colors, axis=0)) > 1
        assert len(np.unique(view.streamline_items["projection"][0].segment_colors, axis=0)) > 1
        assert not view.streamline_items["planar"][1].isVisible()
        assert view.streamline_items["projection"][0].opacity() == pytest.approx(0.4)
        assert len(viewer._velocity_bars) == 2
        assert not viewer.slice_views["coronal"].vertical_line.isVisible()
        assert not viewer.slice_views["coronal"].orientation_items["h0"].isVisible()
        np.testing.assert_allclose(view.image_item.levels, [0, 3])
        assert not np.any(view.overlay_item.image[:, 7:, 3])
        original = ws.mag_raw.copy(), ws.flow_raw.copy(), ws.segmask_raw.copy()
        ws.current_t = 1
        viewer.refresh(update_plane=False)
        assert view._flow_clims["planar"] == (0, 100)
        state["planar"].update(color_mode="fixed", color="#ff00ff")
        state["segmentation"]["mode"] = "outline"
        viewer.apply_display_settings(state)
        assert len(np.unique(view.streamline_items["planar"][0].segment_colors, axis=0)) == 1
        assert not view.overlay_item.isVisible() and view.segmentation_boundaries
        assert len(viewer._velocity_bars) == 1
        for array, before in zip((ws.mag_raw, ws.flow_raw, ws.segmask_raw), original):
            np.testing.assert_array_equal(array, before)
        restored = Workspace()
        restored.restore_dict(ws.snapshot_dict())
        assert restored.render_settings["viewer_display"] == ws.render_settings["viewer_display"]
        viewer.open_display_settings()
        assert [viewer._display_panel.tabs.tabText(i) for i in range(4)] == ["Image", "Overlays", "Flow", "View"]
        assert viewer._display_panel.tabs.count() == 4
        viewer._display_panel._controls[("planar", "density")][0].setValue(0.5)
        assert ws.render_settings["viewer_display"]["planar"]["density"] == 0.5
        panel = viewer._display_panel
        panel.target.setCurrentIndex(panel.target.findData("coronal"))
        panel._controls[("planar", "density")][0].setValue(0.75)
        assert ws.render_settings["viewer_display"]["views"]["coronal"]["planar"]["density"] == 0.75
        assert ws.render_settings["viewer_display"]["planar"]["density"] == 0.5
        panel._controls[("contours", "snap_distance")][0].setValue(1.25)
        assert ws.render_settings["viewer_display"]["contours"]["snap_distance"] == 1.25
        assert viewer._contour_snap_distance_mm == 1.25
    finally:
        if viewer._display_dock:
            viewer._display_dock.close()
        viewer.close()
        viewer.deleteLater()
        app.processEvents()


@pytest.mark.parametrize("content", [0, 3, 4, 5])
@pytest.mark.parametrize("mode", ["composite", "maximum", "slices"])
def test_content_render_phantom_sources_modes_geometry_phase_and_axis_drag(content, mode):
    import pyvista as pv
    from autoflow.ui.viewer import SceneController

    ws = Workspace()
    values = np.arange(8 * 10 * 12, dtype=np.float32).reshape(8, 10, 12) + 1
    ws.mag_raw = np.stack((values, values * 2), axis=3)
    ws.pcmra_array = ws.mag_raw * 0.5
    ws.flow_raw = np.zeros(ws.mag_raw.shape + (3,), dtype=np.float32)
    ws.flow_raw[..., 0] = ws.mag_raw - values.max() / 2
    ws.resolution = np.array([1.0, 1.5, 2.0])
    ws.origin = np.array([10.0, -20.0, 30.0])
    original_mag, original_pcmra, original_flow = ws.mag_raw.copy(), ws.pcmra_array.copy(), ws.flow_raw.copy()
    errors = []
    plotter = pv.Plotter(off_screen=True)
    scene = SceneController(plotter, ws, errors.append)
    try:
        obj = scene.add_content_render(content)
        assert obj is not None and obj.actor is not None
        scene.update_content_render(obj.uid, mode=mode, positions=[0.5, 0.25, 0.5], planes=[False, True, False])
        data = scene._build_dataset(obj.data_key)
        assert obj.actor is not None and obj.show_scalar_bar is False
        if mode == "slices":
            expected = ws.origin[1] + (0.5 + 0.25 * 9) * ws.resolution[1]
            np.testing.assert_allclose(data.points[:, 1], expected)
            assert not scene._is_volume_object(obj)
            assert np.max(obj.actor.mapper.lookup_table.values[:, 3]) == 255
            from scipy.ndimage import map_coordinates
            raw, _, _ = scene._content_source.scalar_volume(0, content)
            coordinates = ((data.points - (ws.origin - 0.5 * ws.resolution)) / ws.resolution).T
            expected_values = map_coordinates(np.pad(raw, 1), coordinates, order=1, mode="constant", cval=0)
            np.testing.assert_allclose(data.point_data[obj.scalars], expected_values, rtol=1e-5, atol=1e-5)
            scene.update_content_render(obj.uid, planes=[True, True, True])
            assert len(scene._content_widgets) == 3
            # Exercise the real VTK interaction callback with a constrained axis handle.
            widget, rep, axis = scene._content_widgets[1]
            endpoint = scene.display_to_world_point(rep.GetWorldPosition())
            endpoint[1] += ws.resolution[1] * 9 * 0.25
            rep.SetWorldPosition(scene.world_to_display_point(endpoint))
            widget.InvokeEvent("InteractionEvent")
            np.testing.assert_allclose(scene.content_render_settings(obj.data_key)["positions"], [0.5, 0.5, 0.5])
            assert len(scene._content_widgets) == 3
            scene.set_display_axis_directions(["RL", "AP", "HF"])
            widget, rep, axis = scene._content_widgets[0]
            endpoint = scene.display_to_world_point(rep.GetWorldPosition())
            endpoint[0] -= ws.resolution[0] * 7 * 0.25
            rep.SetWorldPosition(scene.world_to_display_point(endpoint))
            widget.InvokeEvent("InteractionEvent")
            assert scene.content_render_settings(obj.data_key)["positions"][0] == pytest.approx(0.25)
        else:
            assert scene._is_volume_object(obj)
            assert tuple(data.dimensions) == (10, 12, 14)
            assert obj.actor.GetProperty().GetScalarOpacity().GetValue(0) == 0
            if content == 0:
                assert obj.actor.GetProperty().GetScalarOpacity().GetValue(-300) > 0
        actor = obj.actor
        scene.update_time(1)
        assert obj.actor is actor
        obj.visible = False
        scene.apply_object_properties(obj)
        assert not obj.actor.GetVisibility() and not scene._content_widgets
        assert scene.add_content_render(content) is obj
        assert len(ws.scene_objects) == 1
        scene.update_content_render(obj.uid, mode=mode)
        restored = Workspace()
        restored.restore_dict(ws.snapshot_dict())
        assert restored.render_settings["content_renders"] == ws.render_settings["content_renders"]
        scene.remove_object(obj.uid)
        assert not scene._content_widgets and not ws.scene_objects
        assert not errors
        np.testing.assert_array_equal(ws.mag_raw, original_mag)
        np.testing.assert_array_equal(ws.pcmra_array, original_pcmra)
        np.testing.assert_array_equal(ws.flow_raw, original_flow)
    finally:
        scene._clear_content_widgets()
        plotter.close()


def test_content_render_phantom_static_magnitude_and_missing_pcmra():
    import pyvista as pv
    from autoflow.ui.viewer import SceneController

    ws = Workspace()
    ws.mag_raw = np.ones((5, 6, 7), dtype=np.float32)
    plotter = pv.Plotter(off_screen=True)
    scene = SceneController(plotter, ws, lambda _: None)
    try:
        obj = scene.add_content_render(3)
        assert obj is not None and obj.actor is not None
        assert scene.add_content_render(4) is None
        assert ws.pcmra_array is None
        assert not any(o.data_key == "pcmra_volume" for o in ws.scene_objects.values())
    finally:
        scene._clear_content_widgets()
        plotter.close()


def test_content_render_phantom_every_available_scalar_matches_slice_source(monkeypatch):
    _smoke_qt_application(monkeypatch)
    import pyvista as pv
    from autoflow.ui.viewer import SceneController
    from autoflow.ui.ortho_viewer import OrthoViewer

    ws = Workspace()
    values = np.arange(5 * 6 * 7 * 2, dtype=np.float32).reshape(5, 6, 7, 2) / 100 + 1
    ws.mag_raw = values
    ws.flow_raw = np.stack((values - 2, values, -values), axis=-1)
    ws.pcmra_array = values * 3
    ws.segmask_binary = np.ones(values.shape, dtype=bool)
    ws.derived.wss_volume = values
    ws.derived.tke_array = values * 2
    ws.derived.pressure_gradient_array = ws.flow_raw.copy()
    ws.derived.pressure_gradient_magnitude = np.linalg.norm(ws.flow_raw, axis=-1)
    ws.derived.pressure_gradient_support_mask = ws.segmask_binary
    ws.derived.relative_pressure_array = values - 2
    ws.derived.vorticity_magnitude = values
    ws.derived.q_criterion_array = values - 2
    ws.derived.swirling_strength_array = values
    ws.derived.vortex_support_mask = ws.segmask_binary
    ws.correction_raw = ws.flow_raw * 0.1
    ws.correction_high_raw = ws.flow_raw * 0.2
    ws.phase_unwrap_result = dict(wrap_mask=np.ones(ws.flow_raw.shape, dtype=bool),
                                 wrap_count=np.ones(ws.flow_raw.shape, dtype=np.int8),
                                 flow_wrapped=ws.flow_raw, flow_unwrapped=ws.flow_raw * 1.1)
    ws.planes = [PlaneData(center=np.array([2., 3., 4.]), normal=np.array([1., 0., 0.]))]
    viewer = OrthoViewer(ws)
    viewer.set_selected_plane(0)
    plotter = pv.Plotter(off_screen=True)
    errors = []
    scene = SceneController(plotter, ws, errors.append)
    try:
        assert viewer._available_content_keys() == list(range(28))
        for content in range(28):
            viewer.combo_content.setCurrentIndex(viewer.combo_content.findData(content))
            expected, title, _ = viewer._get_scalar_slice(0)
            obj = scene.add_content_render(content, plane_index=0)
            assert obj is not None and obj.actor is not None, (content, errors)
            grid = scene._build_dataset(obj.data_key)
            actual = np.asarray(grid.point_data[obj.scalars]).reshape(grid.dimensions, order="F")[1:-1, 1:-1, 1:-1]
            np.testing.assert_allclose(actual, expected, err_msg=title)
            scene.remove_object(obj.uid)
        assert not errors
    finally:
        scene._clear_content_widgets()
        plotter.close()
        viewer.close()


@pytest.mark.parametrize("layout,mode", [("normalized", "dv"), ("complex", "dv"),
                                         ("dual", "lv"), ("dual", "hv"), ("dual", "dv")])
def test_gui_input_reload_phantom_applies_edited_calibration(monkeypatch, tmp_path, layout, mode):
    _smoke_qt_application(monkeypatch)
    monkeypatch.setenv("AUTOFLOW_SSH_RENDERING", "1")
    from PySide6 import QtCore, QtTest
    from autoflow.case_types import InputCase
    import autoflow.ui.app as ui_app
    from autoflow.ui.remote_plotter import RemotePlotter
    monkeypatch.setattr(ui_app, "RemotePlotter", RemotePlotter, raising=False)
    shape = (3, 4, 5, 2)
    phase = np.array([0.2, 0.3, 0.4], dtype=np.float32)
    path = tmp_path / f"reload_{layout}_{mode}.h5"
    with h5py.File(path, "w") as handle:
        if layout == "normalized":
            handle["mag"] = np.ones(shape, dtype=np.float32)
            handle["flow"] = np.broadcast_to(phase * 100 / np.pi, shape + (3,))
        else:
            img = np.ones(shape + (7 if layout == "dual" else 4,), dtype=np.complex64)
            img[..., 1:4] = np.exp(1j * phase)
            if layout == "dual":
                img[..., 4:7] = np.exp(1j * phase / 2)
            handle["img"] = img
        handle["VENC"] = [100., 100., 100., 200., 200., 200.] if layout == "dual" else [100., 100., 100.]
        handle["Resolution"] = [1., 1.5, 2.]
        handle["SpatialOrder"] = np.array(["LR", "AP", "FH"], dtype="S4")
        handle["VENCOrder"] = np.array(["LR", "AP", "FH"], dtype="S4")
        labels = np.zeros(shape, dtype=np.int16)
        labels[0, 1, 2, :] = 1
        handle["seg"] = labels
    source_bytes = path.read_bytes()
    window = ui_app.MainWindow()
    try:
        assert window._load_selected_input_case(InputCase(str(path), "h5", metadata={"dual_venc_mode": mode}))
        previous = window.workspace
        old_flow = previous.flow_raw.copy()
        previous.planes = [PlaneData(center=np.zeros(3), normal=np.array([1., 0., 0.]))]
        window._publish_action_settings({
            "input.resolution": "2, 3, 4", "input.venc": "150, 180, 210",
            "input.spatial_order": "FH, AP, LR", "input.venc_order": "RL, AP, FH",
        })
        # Other processing actions must not silently change loaded calibration.
        window._sync_params_to_ws()
        np.testing.assert_allclose(previous.resolution, [1., 1.5, 2.])
        np.testing.assert_array_equal(previous.flow_raw, old_flow)
        assert previous.spatial_order == ["LR", "AP", "FH"]
        QtTest.QTest.mouseClick(window.btn_reload_input, QtCore.Qt.LeftButton)
        ws = window.workspace
        assert ws is not previous, window.console.toPlainText()
        np.testing.assert_allclose(ws.resolution, [2., 3., 4.])
        np.testing.assert_allclose(ws.venc, [150., 180., 210.])
        assert ws.spatial_order == ["FH", "AP", "LR"]
        assert ws.venc_order == ["RL", "AP", "FH"]
        assert ws.flow_raw.shape == (5, 4, 3, 2, 3)
        np.testing.assert_array_equal(ws.segmask_raw, labels.transpose(2, 1, 0, 3))
        if layout == "normalized":
            expected = phase * np.array([-1., 1., 1.]) * 100 / np.pi
        else:
            expected = phase * np.array([-150., 180., 210.]) / np.pi
            if layout == "dual" and mode in {"dv", "hv"}:
                expected /= 2
        np.testing.assert_allclose(ws.flow_raw, np.broadcast_to(expected, ws.flow_raw.shape), atol=3e-5)
        assert window._input_parameters_from_ui() == {
            "resolution": [2., 3., 4.], "venc": [150., 180., 210.],
            "spatial_order": ["FH", "AP", "LR"], "venc_order": ["RL", "AP", "FH"],
        }
        assert not ws.planes
        # A numeric formatting change does not reload or invalidate anything.
        window._publish_action_settings({"input.resolution": "2.000, 3.0, 4", **{
            "input." + key: value for key, value in window._input_parameters_from_ui().items() if key != "resolution"}})
        window._reload_input_case()
        assert window.workspace is ws
        # Invalid input and cancelled/failed loads leave the loaded case intact.
        with pytest.raises(ValueError):
            window._publish_action_settings({"input.resolution": "2, 0, 4"})
        assert window.workspace is ws
        np.testing.assert_allclose(ws.resolution, [2., 3., 4.])
        window._publish_action_settings({"input." + key: ("5, 6, 7" if key == "resolution" else value)
                                         for key, value in window._input_parameters_from_ui().items()})
        def cancel(*_args, **_kwargs):
            raise ui_app.TaskCancelled()
        with monkeypatch.context() as patch:
            patch.setattr(window, "_run_modal_task", cancel)
            window._reload_input_case()
        assert window.workspace is ws
        np.testing.assert_allclose(ws.resolution, [2., 3., 4.])
        window._publish_action_settings({"input." + key: ([2., 3., 4.] if key == "resolution" else value)
                                         for key, value in window._input_parameters_from_ui().items()})
        # Correction must reopen the same source with the session calibration.
        def fit(values, **_kwargs):
            corr = np.zeros((3, 1) + values.shape[2:], dtype=np.float32)
            return corr, np.ones(values.shape[2:], dtype=bool), {"applied": True}
        monkeypatch.setattr("autoflow.algorithms.phase_correction.execute_wrls_arto", fit)
        result = window.pipeline._step_background_correction(ws, force_recompute=True)
        assert result.success, result.message
        np.testing.assert_allclose(ws.flow_raw, np.broadcast_to(expected, ws.flow_raw.shape), atol=3e-5)
        assert path.read_bytes() == source_bytes
        # Opening a different case uses that file's metadata, not the edits.
        assert window._load_selected_input_case(InputCase(str(path), "h5"))
        np.testing.assert_allclose(window.workspace.resolution, [1., 1.5, 2.])
        np.testing.assert_allclose(window.workspace.venc, [200., 200., 200.] if layout == "dual" else [100., 100., 100.])
    finally:
        window.close()


def test_content_render_phantom_gui_open_input_then_render_and_switch_case(monkeypatch, tmp_path):
    app = _smoke_qt_application(monkeypatch)
    monkeypatch.setenv("AUTOFLOW_SSH_RENDERING", "1")
    from PySide6 import QtCore, QtTest
    from autoflow.case_types import InputCase
    import autoflow.ui.app as ui_app
    from autoflow.ui.remote_plotter import RemotePlotter
    monkeypatch.setattr(ui_app, "RemotePlotter", RemotePlotter, raising=False)
    window = ui_app.MainWindow()
    window.show()
    app.processEvents()
    try:
        startup_workspace = window.workspace
        for case_index, shape in enumerate(((7, 8, 9, 2), (9, 6, 8, 2))):
            path = tmp_path / f"content_case_{case_index}.h5"
            signal = float(case_index + 1)
            with h5py.File(path, "w") as handle:
                handle.create_dataset("mag", data=np.full(shape, signal, dtype=np.float32))
                handle.create_dataset("flow", data=np.full(shape + (3,), signal * 10, dtype=np.float32))
                handle.create_dataset("Resolution", data=np.array([1., 1.5, 2.], dtype=np.float32))
                handle.create_dataset("RR", data=1000.)
                handle.create_dataset("VENC", data=np.full(3, 100., dtype=np.float32))
                handle.create_dataset("SpatialOrder", data=np.asarray(["LR", "AP", "FH"], dtype="S4"))
                handle.create_dataset("VENCOrder", data=np.asarray(["LR", "AP", "FH"], dtype="S4"))
            window._load_selected_input_case(InputCase(str(path), "h5"))
            ws = window.workspace
            assert ws is not startup_workspace and ws.data_loaded, window.console.toPlainText()
            window.ortho_viewer.combo_content.setCurrentIndex(window.ortho_viewer.combo_content.findData(3))
            QtTest.QTest.mouseClick(window.ortho_viewer.btn_render_content, QtCore.Qt.LeftButton)
            renders = [obj for obj in ws.scene_objects.values() if obj.data_key == "magnitude_volume"]
            assert len(renders) == 1, window.console.toPlainText()
            obj = renders[0]
            assert obj.visible and obj.actor is not None
            assert window._find_browser_item_by_uid(obj.uid) is not None
            assert "3D volume shown:" in window.statusBar().currentMessage()
            grid = window.scene._build_dataset(obj.data_key)
            assert np.max(grid.point_data["Magnitude"]) == signal
            window.content_render_controls.mode.setCurrentIndex(window.content_render_controls.mode.findData("slices"))
            assert len(window.scene._content_widgets) == 3
            assert window.scene._content_source.workspace is ws
    finally:
        window.scene._clear_content_widgets()
        window.close()
        window.deleteLater()
        app.processEvents()


def test_content_render_phantom_gui_pcmra_handoff_and_browser_properties(monkeypatch, tmp_path):
    app = _smoke_qt_application(monkeypatch)
    monkeypatch.setenv("AUTOFLOW_SSH_RENDERING", "1")
    from PySide6 import QtCore, QtTest
    import autoflow.ui.app as ui_app
    from autoflow.ui.remote_plotter import RemotePlotter
    monkeypatch.setattr(ui_app, "RemotePlotter", RemotePlotter, raising=False)
    window = ui_app.MainWindow()
    try:
        ws = window.workspace
        ws.paths.output_dir = str(tmp_path)
        ws.data_loaded = True
        ws.mag_raw = np.ones((7, 8, 9, 2), dtype=np.float32)
        ws.flow_raw = np.ones(ws.mag_raw.shape + (3,), dtype=np.float32) * 10
        window.ortho_viewer.update_slider_ranges()
        window.ortho_viewer.combo_content.setCurrentIndex(window.ortho_viewer.combo_content.findData(3))
        window.ortho_viewer.btn_render_content.click()
        mag = next(o for o in ws.scene_objects.values() if o.data_key == "magnitude_volume")
        assert mag.actor is not None and mag.visible
        result = PipelineEngine()._step_generate_pcmra(ws)
        assert result.success
        window._finish_pipeline_scene_refresh([StepId.GENERATE_PCMRA])
        assert window.ortho_viewer.combo_content.currentData() == 4
        pcmra = next(o for o in ws.scene_objects.values() if o.data_key == "pcmra_volume")
        assert pcmra.actor is not None and pcmra.visible and mag.visible
        window.ortho_viewer.btn_render_content.click()
        assert len([o for o in ws.scene_objects.values() if o.data_key == "pcmra_volume"]) == 1
        controls = window.content_render_controls
        assert not controls.isHidden()
        assert window._selected_uid() == pcmra.uid
        assert window.chk_selection_on_top.isChecked()
        assert window.scene._selection_renderer.HasViewProp(pcmra.actor)
        window.chk_selection_on_top.setChecked(False)
        assert ws.render_settings["selected_layer_on_top"] is False
        assert window.plotter.renderer.HasViewProp(pcmra.actor)
        window.chk_selection_on_top.setChecked(True)
        assert window.scene._selection_renderer.HasViewProp(pcmra.actor)
        assert controls.mode.currentData() == "composite"
        assert all(check.isEnabled() for check in controls.planes)
        assert all(slider.isEnabled() for slider in controls.positions)
        QtTest.QTest.mouseClick(controls.planes[0], QtCore.Qt.LeftButton)
        assert controls.mode.currentData() == "slices"
        assert window.scene.content_render_settings(pcmra.data_key)["planes"] == [True, False, False]
        assert len(window.scene._content_widgets) == 1
        controls.mode.setCurrentIndex(controls.mode.findData("composite"))
        controls.positions[2].setValue(650)
        assert controls.mode.currentData() == "slices"
        assert window.scene.content_render_settings(pcmra.data_key)["positions"][2] == 0.65
        assert window.scene.content_render_settings(pcmra.data_key)["planes"] == [False, False, True]
        controls.show_planes.click()
        assert len(window.scene._content_widgets) == 3
        controls.positions[1].setValue(750)
        assert window.scene.content_render_settings(pcmra.data_key)["positions"][1] == 0.75
        window.show()
        app.processEvents()
        window.plotter.view_isometric()
        window.plotter.reset_camera()
        window.plotter.render()
        app.processEvents()
        widget, rep, axis = window.scene._content_widgets[0]
        endpoint = np.asarray(rep.GetWorldPosition())
        target = endpoint + np.array([0.9, 0, 0])
        renderer = window.plotter.renderer

        def screen_point(point):
            renderer.SetWorldPoint(*point, 1)
            renderer.WorldToDisplay()
            x, y, _ = renderer.GetDisplayPoint()
            rw, rh = window.plotter.render_window.GetSize()
            return QtCore.QPoint(round(x * window.plotter.width() / rw),
                                 round((rh - 1 - y) * window.plotter.height() / rh))

        start, end = screen_point(endpoint), screen_point(target)
        camera = np.asarray(tuple(window.plotter.camera_position), dtype=float)
        QtTest.QTest.mouseMove(window.plotter, start)
        QtTest.QTest.mousePress(window.plotter, QtCore.Qt.LeftButton, pos=start)
        assert window.scene._content_widget_dragging
        QtTest.QTest.mouseMove(window.plotter, end)
        QtTest.QTest.mouseRelease(window.plotter, QtCore.Qt.LeftButton, pos=end)
        assert not window.scene._content_widget_dragging
        assert window.scene.content_render_settings(pcmra.data_key)["positions"][0] > 0.55
        np.testing.assert_allclose(tuple(window.plotter.camera_position), camera)
        pcmra.opacity = 0.42
        pcmra.clim = (2, 30)
        assert PipelineEngine()._step_generate_pcmra(ws).success
        window._finish_pipeline_scene_refresh([StepId.GENERATE_PCMRA])
        assert ws.scene_objects[pcmra.uid] is pcmra
        assert pcmra.opacity == 0.42 and pcmra.clim == (2, 30)
        assert controls.mode.currentData() == "slices" and len(window.scene._content_widgets) == 3
        assert window.ortho_viewer.combo_content.currentData() == 4
        item = window._find_browser_item_by_uid(pcmra.uid)
        item.setCheckState(0, QtCore.Qt.Unchecked)
        assert not pcmra.actor.GetVisibility() and not window.scene._content_widgets
        item.setCheckState(0, QtCore.Qt.Checked)
        assert pcmra.actor.GetVisibility() and len(window.scene._content_widgets) == 3
        window.scene.remove_object(pcmra.uid)
        window._refresh_browser()
        assert not window.scene._content_widgets
    finally:
        window.scene._clear_content_widgets()
        window.close()
        window.deleteLater()
        app.processEvents()


@pytest.mark.parametrize("mapper_name", ["smart", "fixed_point"])
def test_interaction_render_phantom_mouse_preview_restores_quality_and_coordinates(monkeypatch, mapper_name):
    app = _smoke_qt_application(monkeypatch)
    monkeypatch.setenv("AUTOFLOW_SSH_RENDERING", "1")
    from PySide6 import QtCore, QtTest
    from autoflow.ui.remote_plotter import RemotePlotter
    from autoflow.ui.viewer import SceneController

    ws = Workspace()
    ws.mag_raw = np.arange(8 * 9 * 10, dtype=np.float32).reshape(8, 9, 10, 1) + 1
    original = ws.mag_raw.copy()
    plotter = RemotePlotter()
    plotter.resize(1400, 900)
    plotter.show()
    app.processEvents()
    scene = SceneController(plotter, ws, lambda _: None)
    scene._volume_mapper_name = lambda: mapper_name
    scene.initialize()
    try:
        obj = scene.add_content_render(3)
        plotter.reset_camera()
        plotter.render()
        mapper = obj.actor.GetMapper()
        names = [name for name in ("SampleDistance", "ImageSampleDistance", "AutoAdjustSampleDistances",
                                  "InteractiveAdjustSampleDistances") if hasattr(mapper, "Get" + name)]
        before = {name: getattr(mapper, "Get" + name)() for name in names}
        dataset = mapper.dataset
        render_pass = plotter.renderer.GetPass()
        size = tuple(plotter.window_size)
        QtTest.QTest.mousePress(plotter, QtCore.Qt.LeftButton, pos=QtCore.QPoint(400, 300))
        assert scene._interaction_preview_active
        assert max(plotter.window_size) <= 960
        assert mapper.GetSampleDistance() >= 2.0
        assert plotter.renderer.GetPass() is None
        assert mapper.dataset is dataset
        x, y = plotter.iren.GetEventPosition()
        width, height = plotter.window_size
        assert x == round(400 * width / plotter.width())
        assert y == height - 1 - round(300 * height / plotter.height())
        QtTest.QTest.mouseMove(plotter, QtCore.QPoint(460, 330))
        QtTest.QTest.mouseRelease(plotter, QtCore.Qt.LeftButton, pos=QtCore.QPoint(460, 330))
        assert not scene._interaction_preview_active
        assert tuple(plotter.window_size) == size
        assert plotter.renderer.GetPass() is render_pass
        assert {name: getattr(mapper, "Get" + name)() for name in names} == before
        assert mapper.dataset is dataset
        scene.update_content_render(obj.uid, mode="slices")
        plotter.render()
        plotter._capture_frame()
        widget, rep, _ = scene._content_widgets[0]
        endpoint = np.asarray(rep.GetWorldPosition())
        renderer = plotter.renderer

        def screen_point(point):
            renderer.SetWorldPoint(*point, 1)
            renderer.WorldToDisplay()
            x, y, _ = renderer.GetDisplayPoint()
            width, height = plotter.window_size
            return QtCore.QPoint(round(x * plotter.width() / width),
                                 round((height - 1 - y) * plotter.height() / height))

        start, end = screen_point(endpoint), screen_point(endpoint + [0.8, 0, 0])
        QtTest.QTest.mouseMove(plotter, start)
        QtTest.QTest.mousePress(plotter, QtCore.Qt.LeftButton, pos=start)
        assert scene._content_widget_dragging and scene._interaction_preview_active
        QtTest.QTest.mouseMove(plotter, end)
        QtTest.QTest.mouseRelease(plotter, QtCore.Qt.LeftButton, pos=end)
        assert not scene._content_widget_dragging and not scene._interaction_preview_active
        assert tuple(plotter.window_size) == size
        assert scene.content_render_settings(obj.data_key)["positions"][0] > 0.55
        np.testing.assert_array_equal(ws.mag_raw, original)
    finally:
        scene._close_interaction_rendering()
        plotter.close()
        plotter.deleteLater()
        app.processEvents()


def test_browser_selection_phantom_draws_flow_above_opaque_cut_and_cleans_up():
    import pyvista as pv
    from autoflow.core.models import ObjectKind
    from autoflow.ui.viewer import SceneController

    ws = Workspace()
    line = pv.Line((-3, 0, 0), (3, 0, 0))
    line["Speed"] = np.ones(line.n_points)
    meshes = {"phantom_cut": pv.Plane(center=(0, 0, 1), i_size=8, j_size=8),
              "phantom_flow": line}
    cut = ws.scene_objects[ws.add_object("Cut", ObjectKind.AUX, "phantom_cut", color="white")]
    flow = ws.scene_objects[ws.add_object("Flow", ObjectKind.FLOW, "phantom_flow", color="red", line_width=12)]
    errors = []
    plotter = pv.Plotter(off_screen=True, window_size=(320, 320))
    scene = SceneController(plotter, ws, errors.append)
    scene._build_dataset = meshes.get
    scene.initialize()
    try:
        scene.render_all()
        plotter.camera_position = [(0, 0, 12), (0, 0, 0), (0, 1, 0)]
        scene.reset_camera()

        def red_pixels():
            pixels = plotter.screenshot(return_img=True)[120:200, 80:240, :3].astype(int)
            return np.count_nonzero((pixels[..., 0] > 100) & (pixels[..., 0] > 2 * pixels[..., 1])
                                    & (pixels[..., 0] > 2 * pixels[..., 2]))

        assert red_pixels() == 0
        actor, mapper = flow.actor, flow.actor.mapper
        scene.set_selected_layers([flow.uid])
        assert red_pixels() > 300
        assert scene._selection_renderer.HasViewProp(actor)
        assert not plotter.renderer.HasViewProp(actor)
        scene.update_time(0)
        assert flow.actor is actor and actor.mapper is mapper
        scene.set_selected_layers([flow.uid], on_top=False)
        assert red_pixels() == 0
        assert plotter.renderer.HasViewProp(actor)
        restored = Workspace()
        restored.restore_dict(ws.snapshot_dict())
        assert restored.render_settings["selected_layer_on_top"] is False
        flow.scalars, flow.cmap, flow.clim, flow.show_scalar_bar = "Speed", "Reds", (0, 2), True
        scene.readd_object(flow)
        scene.set_selected_layers([flow.uid], on_top=True)
        assert len(scene._hud_props) == 1
        assert scene._hud_renderer.HasViewProp(scene._hud_props[0])
        scene.set_selected_layers([cut.uid, flow.uid])
        scene.reset_camera()
        assert np.all(np.isfinite(np.asarray(tuple(plotter.camera_position), dtype=float)))
        scene.remove_object(flow.uid)
        assert scene._selection_renderer.GetViewProps().GetNumberOfItems() == 1
        scene.reset_scene()
        assert not scene._front_props and not scene._hud_props
        assert scene._selection_renderer.GetViewProps().GetNumberOfItems() == 0
        assert not errors
    finally:
        plotter.close()


def test_remote_drag_phantom_coalesces_moves_and_applies_final_position(monkeypatch):
    app = _smoke_qt_application(monkeypatch)
    from PySide6 import QtCore, QtGui
    from autoflow.ui.remote_plotter import RemotePlotter

    plotter = RemotePlotter()
    plotter.resize(640, 480)
    moves = []
    plotter._autoflow_content_widget_dispatch = lambda kind: moves.append((kind, plotter.iren.GetEventPosition())) or True

    def mouse(kind, x, button, buttons):
        event = QtGui.QMouseEvent(kind, QtCore.QPointF(x, 200), QtCore.QPointF(x, 200),
                                 button, buttons, QtCore.Qt.NoModifier)
        QtCore.QCoreApplication.sendEvent(plotter, event)

    try:
        mouse(QtCore.QEvent.MouseButtonPress, 100, QtCore.Qt.LeftButton, QtCore.Qt.LeftButton)
        # Keep the event burst within one preview interval, independent of CPU speed.
        import time
        plotter._last_motion_at = time.monotonic() + 3600
        for x in range(101, 401):
            mouse(QtCore.QEvent.MouseMove, x, QtCore.Qt.NoButton, QtCore.Qt.LeftButton)
        assert not [kind for kind, _ in moves if kind == "move"]
        mouse(QtCore.QEvent.MouseButtonRelease, 450, QtCore.Qt.LeftButton, QtCore.Qt.NoButton)
        assert [kind for kind, _ in moves] == ["press", "move", "release"]
        assert moves[-2][1] == (450, 279)
        assert not plotter._motion_timer.isActive() and plotter._pending_mouse_payload is None
        assert not plotter._mouse_buttons_active
    finally:
        plotter.close()
        plotter.deleteLater()
        app.processEvents()


def _run_case(input_name: str, *, skip_derived: bool = False):
    output_root = Path(tempfile.mkdtemp(prefix=f"autoflow_{input_name}_", dir="/tmp"))
    cfg = AutoFlowConfig(
        graph_connectivity=26,  # analytical phantom truth assumes a connected curved centerline
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
    from autoflow import processing
    original_save = processing.save_quality_report
    def slow_save(*args, **kwargs):
        import time
        time.sleep(0.03)
        return original_save(*args, **kwargs)
    monkeypatch.setattr(processing, "save_quality_report", slow_save)
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
    assert summary["total_time_sec"] >= summary["compute_time_sec"] + 0.03
    assert summary["output_time_sec"] == pytest.approx(summary["total_time_sec"] - summary["compute_time_sec"])
    assert json.loads((out_dir / "summary.json").read_text())["total_time_sec"] == summary["total_time_sec"]


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


@pytest.mark.parametrize("layout", ["normalized", "complex", "dual", "real_phase"])
def test_gui_load_saved_corr_and_recompute_from_source(monkeypatch, tmp_path, layout):
    _smoke_qt_application(monkeypatch)
    from autoflow.ui.app import _PipelineTaskWorker
    import autoflow.algorithms.phase_correction as correction
    shape = (3, 2, 4, 2)
    phase = np.array([0.4, -0.3, 0.2], dtype=np.float32)
    old_corr = np.full(shape[:3] + (1, 3), 0.05, dtype=np.float32)
    path = tmp_path / f"saved_corr_{layout}.h5"
    venc = np.full(6 if layout == "dual" else 3, 100.0, dtype=np.float32)
    if layout == "dual":
        venc[3:] = 200.0
    with h5py.File(path, "w") as handle:
        if layout == "normalized":
            handle["mag"] = np.ones(shape, dtype=np.float32)
            handle["flow"] = np.broadcast_to(phase * 100 / np.pi, shape + (3,))
        elif layout == "real_phase":
            img = np.ones(shape + (4,), dtype=np.float32)
            # The loader recognizes combined real input near the full phase range.
            phase = np.array([3.0, -3.0, 0.2], dtype=np.float32)
            img[..., 1:4] = phase
            handle["img"] = img
        else:
            img = np.ones(shape + (7 if layout == "dual" else 4,), dtype=np.complex64)
            img[..., 1:4] = np.exp(1j * phase)
            if layout == "dual":
                img[..., 4:7] = np.exp(1j * phase / 2)
            handle["img"] = img
        handle["VENC"] = venc
        handle["SpatialOrder"] = np.array(["LR", "AP", "FH"], dtype="S4")
        handle["VENCOrder"] = np.array(["LR", "AP", "FH"], dtype="S4")
        handle["seg"] = np.ones(shape, dtype=np.int16)
        cache_names = ("corr_low", "corr_high") if layout == "dual" else ("corr",)
        for name in cache_names:
            # Untagged legacy MSAC caches must load even when the GUI selects WRLS.
            handle[name] = old_corr
    original = path.read_bytes()
    calls = []
    def fit(values, **kwargs):
        calls.append(values.copy())
        corr = np.full((3, 1) + values.shape[2:], 0.1, dtype=np.float32)
        return corr, np.ones(values.shape[2:], dtype=bool), {"applied": True}
    monkeypatch.setattr(correction, "execute_wrls_arto", fit)
    monkeypatch.setattr(correction, "execute_msac", fit)
    ws = Workspace()
    ws.paths.flow_path = str(path)
    ws.loader_params.background_phase_correction.method = "wrls_arto"
    # Force-recompute settings apply to manual runs, not opening saved results.
    ws.loader_params.background_phase_correction.force_recompute = True
    events = []
    engine = PipelineEngine()
    engine.load_data(ws, lambda _: None, progress_callback=events.append, reuse_existing_corr=True)
    assert calls == []
    assert path.read_bytes() == original  # Opening applies a cache without writing H5.
    expected = (phase - 0.05) * 100 / np.pi
    np.testing.assert_allclose(ws.flow_raw, np.broadcast_to(expected, shape + (3,)), atol=2e-5)
    assert ws.correction_raw is not None
    if layout == "dual":
        assert ws.correction_high_raw is not None
    else:
        meta = ws.input_state.metadata["background_phase_correction"]
        assert meta["corr_algorithm"] == "msac" and meta["cache_only"]
    messages = [e.get("message", "") for e in events]
    assert any("Loading saved background correction" in text for text in messages)
    assert any("Applying saved corr" in text for text in messages)
    if layout == "dual":
        assert any("Low VENC" in text and "Applying saved corr" in text for text in messages)
        assert any("High VENC" in text and "Applying saved corr" in text for text in messages)
    if layout != "dual":
        np.testing.assert_allclose(ws.phase_wrapped, np.broadcast_to(phase - 0.05, shape + (3,)), atol=2e-6)
    # The GUI worker must force recomputation even when force_recompute is off.
    ws.loader_params.background_phase_correction.force_recompute = False
    ws.loader_params.background_phase_correction.write_cache = True
    expected_calls = 2 if layout == "dual" else 1
    source_key = "flow" if layout == "normalized" else "img"
    with h5py.File(path, "r") as handle:
        source = handle[source_key][:]
    for _ in range(2):
        worker = _PipelineTaskWorker(engine, ws, [StepId.BACKGROUND_CORRECTION])
        failures = []
        worker.failed.connect(failures.append)
        worker.run()
        assert not failures
        assert len(calls) == expected_calls
        source_phases = [np.angle(values[1:4, 0, 0, 0, 0] * np.conj(values[0, 0, 0, 0, 0])) for values in calls]
        source_phases.sort(key=lambda value: float(value[0]))
        expected_phases = [phase / 2, phase] if layout == "dual" else [phase]
        expected_phases.sort(key=lambda value: float(value[0]))
        np.testing.assert_allclose(source_phases, expected_phases, atol=2e-6)
        calls.clear()
        expected = (phase - 0.1) * 100 / np.pi
        np.testing.assert_allclose(ws.flow_raw, np.broadcast_to(expected, shape + (3,)), atol=2e-5)
        with h5py.File(path, "r") as handle:
            np.testing.assert_array_equal(handle[source_key][:], source)
            for name in cache_names:
                np.testing.assert_allclose(handle[name][:], 0.1)
    reloaded = load_h5_data(str(path), reuse_existing_corr=True)
    np.testing.assert_allclose(reloaded.flow, ws.flow_raw, atol=2e-5)


@pytest.mark.parametrize("cache", ["missing", "shape", "nonfinite", "wrong_group"])
def test_gui_load_invalid_corr_keeps_source_without_fitting(monkeypatch, tmp_path, cache):
    shape = (3, 2, 4, 2)
    path = tmp_path / "invalid_corr.h5"
    with h5py.File(path, "w") as handle:
        handle["mag"] = np.ones(shape, dtype=np.float32)
        handle["flow"] = np.full(shape + (3,), 10.0, dtype=np.float32)
        handle["SpatialOrder"] = np.array(["LR", "AP", "FH"], dtype="S4")
        handle["VENCOrder"] = np.array(["LR", "AP", "FH"], dtype="S4")
        if cache != "missing":
            corr = np.full((1, 1, 1, 1, 3) if cache == "shape" else shape[:3] + (1, 3), 0.05)
            if cache == "nonfinite":
                corr[..., 0] = np.nan
            ds = handle.create_dataset("corr", data=corr)
            if cache == "wrong_group":
                ds.attrs["corr_source_group"] = "AnotherCase"
    calls = []
    def fit(*args, **kwargs):
        calls.append(True)
        raise AssertionError("opening a case must never fit a missing or invalid corr")
    monkeypatch.setattr("autoflow.algorithms.phase_correction.execute_msac", fit)
    monkeypatch.setattr("autoflow.algorithms.phase_correction.execute_wrls_arto", fit)
    loaded = load_h5_data(str(path), reuse_existing_corr=True)
    assert calls == []
    np.testing.assert_array_equal(loaded.flow, 10.0)
    assert loaded.correction is None
    assert not loaded.metadata["background_phase_correction"]["applied"]


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
    monkeypatch.setattr("autoflow.rendering.videos._make_plotter", lambda window_size=None, background=None, workspace=None: plotter)
    monkeypatch.setattr("autoflow.rendering.videos.VideoOverlay",
                        lambda *args: SimpleNamespace(update=lambda **kwargs: None))
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


def test_dicom2h5_gui_phantom_destination_feedback_and_worker_progress(monkeypatch, tmp_path):
    app = _smoke_qt_application(monkeypatch)
    monkeypatch.setenv("AUTOFLOW_SSH_RENDERING", "1")
    from PySide6 import QtCore, QtTest, QtWidgets
    import autoflow.ui.app as ui_app
    from autoflow.ui.remote_plotter import RemotePlotter
    monkeypatch.setattr(ui_app, "RemotePlotter", RemotePlotter, raising=False)
    window = ui_app.MainWindow()
    root, target = tmp_path / "dicom", tmp_path / "converted.h5"
    root.mkdir()
    warnings, progress, applied, calls = [], [], [], []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *args: warnings.append(args[2]))

    def choose_root(*args, **kwargs):
        assert kwargs["options"] & QtWidgets.QFileDialog.Option.DontUseNativeDialog
        return str(root)

    monkeypatch.setattr(QtWidgets.QFileDialog, "getExistingDirectory", choose_root)
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", lambda *a, **k: ("", ""))
    original_bind = window._bind_task_progress

    class ProgressProbe(QtCore.QObject):
        @QtCore.Slot(dict)
        def record(self, payload):
            progress.append(payload["message"])

    probe = ProgressProbe()

    def record_signals(dialog, worker):
        original_bind(dialog, worker)
        worker.progress.connect(probe.record, QtCore.Qt.QueuedConnection)

    monkeypatch.setattr(window, "_bind_task_progress", record_signals)
    monkeypatch.setattr(window, "_apply_dicom_conversion_cases", applied.append)

    def convert(source, destination, progress_callback=None):
        calls.append(QtCore.QThread.currentThread() is app.thread())
        assert source == str(root) and destination == str(target)
        progress_callback({"message": "Converting", "current": 0, "total": 0})
        progress_callback({"message": "Validating", "current": 0, "total": 0})
        return [SimpleNamespace(input_path=str(target))]

    monkeypatch.setattr(ui_app, "convert_dicom_input", convert)
    try:
        window._on_load_dicom()
        assert "no H5 destination" in window.statusBar().currentMessage()
        assert not calls
        target.write_bytes(b"preserved")
        monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", lambda *a, **k: (str(target), ""))
        window._on_load_dicom()
        assert warnings and "Destination exists" in warnings[-1]
        assert target.read_bytes() == b"preserved" and not calls
        target.unlink()
        window._on_load_dicom()
        assert window._dicom_conversion_progress.isVisible()
        for _ in range(500):
            app.processEvents()
            if applied:
                break
            QtTest.QTest.qWait(10)
        assert applied and progress == ["Converting", "Validating"]
        assert calls == [False]
        assert window._dicom_conversion_thread is None
        assert window._dicom_conversion_progress is None

        def fail_conversion(*args, **kwargs):
            raise ValueError("phantom conversion failure")

        monkeypatch.setattr(ui_app, "convert_dicom_input", fail_conversion)
        window._on_load_dicom()
        for _ in range(500):
            app.processEvents()
            if len(warnings) > 1:
                break
            QtTest.QTest.qWait(10)
        assert "phantom conversion failure" in warnings[-1]
        assert "failed" in window.statusBar().currentMessage()
        assert window._dicom_conversion_thread is None
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()


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


def test_noise_removal_phantom_settings_hide_inactive_method_and_cancel(monkeypatch):
    application = _smoke_qt_application(monkeypatch)
    from autoflow.ui.action_settings import ActionSettingsDialog, action_sections
    from autoflow.config import apply_config_bundle_to_workspace, load_config_bundle

    ws = Workspace()
    apply_config_bundle_to_workspace(ws, load_config_bundle("configs"))
    panel = ActionSettingsDialog("Noise removal", action_sections(StepId.REMOVE_NOISE, ws), ws, ws)
    try:
        key = "noise_removal_params."
        assert panel.controls[key + "magnitude_fraction"].value() == .05
        assert panel.controls[key + "velocity_std_max"].value() == .80
        panel.controls[key + "magnitude_fraction"].setValue(.03)
        panel.controls[key + "velocity_std_max"].setValue(.90)
        method = panel.controls[key + "method"]
        method.setCurrentIndex(method.findData("magnitude"))
        assert panel.rows[key + "velocity_std_max"].isHidden()
        assert panel.forms[key + "velocity_std_max"].labelForField(panel.rows[key + "velocity_std_max"]).isHidden()
        assert panel.values()[key + "velocity_std_max"] == .80
        panel.reject()
        assert ws.noise_removal_params.magnitude_fraction == .05
        assert ws.noise_removal_params.velocity_std_max == .80
    finally:
        panel.close()
        panel.deleteLater()
        application.processEvents()


def test_action_settings_phantom_context_save_run_and_conditional_rows(monkeypatch):
    application = _smoke_qt_application(monkeypatch)
    monkeypatch.setenv("AUTOFLOW_SSH_RENDERING", "1")
    from PySide6 import QtCore, QtGui, QtWidgets, QtTest
    import autoflow.ui.app as ui_app
    from autoflow.ui.remote_plotter import RemotePlotter
    from autoflow.ui.action_settings import ActionSettingsDialog, action_sections

    monkeypatch.setattr(ui_app, "RemotePlotter", RemotePlotter, raising=False)
    window = ui_app.MainWindow()
    try:
        ws = window.workspace
        ws.data_loaded = True
        ws.flow_raw = np.zeros((3, 3, 3, 2, 3), dtype=np.float32)
        ws.mag_raw = np.ones((3, 3, 3, 2), dtype=np.float32)
        window.show()
        window.workflow_tabs.setCurrentIndex(1)
        application.processEvents()
        assert not hasattr(window, "btn_show_advanced")
        assert window.review_panel.isHidden()
        button = window.step_buttons[StepId.REMOVE_NOISE]
        key = "noise_removal_params.magnitude_fraction"

        def context_edit(cancel):
            def edit():
                dialog = application.activeModalWidget()
                assert isinstance(dialog, ActionSettingsDialog)
                dialog.controls[key].setValue(.07)
                dialog.reject() if cancel else dialog.accept()
            QtCore.QTimer.singleShot(0, edit)
            application.sendEvent(button, QtGui.QContextMenuEvent(
                QtGui.QContextMenuEvent.Mouse, QtCore.QPoint(4, 4), button.mapToGlobal(QtCore.QPoint(4, 4))))

        context_edit(True)
        assert ws.noise_removal_params.magnitude_fraction == .05
        ws.pipeline.mark_done(StepId.REMOVE_NOISE)
        context_edit(False)
        assert ws.noise_removal_params.magnitude_fraction == .07
        assert not ws.pipeline.is_done(StepId.REMOVE_NOISE)
        assert "Settings changed" in button.toolTip()

        ran = []
        def save_and_run():
            application.activeModalWidget()._save_and_run()
        QtCore.QTimer.singleShot(0, save_and_run)
        window._open_action_settings(StepId.REMOVE_NOISE, lambda: ran.append(ws.noise_removal_params.magnitude_fraction))
        assert ran == [.07]

        keyboard_dialogs = []
        def close_keyboard_dialog():
            dialog = application.activeModalWidget()
            if isinstance(dialog, ActionSettingsDialog):
                keyboard_dialogs.append(dialog.windowTitle())
                dialog.reject()
        window.activateWindow()
        button.setFocus()
        application.processEvents()
        QtCore.QTimer.singleShot(0, close_keyboard_dialog)
        QtTest.QTest.keyClick(button, QtCore.Qt.Key_F10, QtCore.Qt.ShiftModifier)
        assert len(keyboard_dialogs) == 1

        # A changed pathline setting must replace existing trajectories rather
        # than taking the unchanged-request cache shortcut.
        ws.segmask_3d = np.ones((3, 3, 3), dtype=bool)
        ws.segmask_raw = ws.segmask_3d
        ws.segmask_binary = ws.segmask_3d
        ws.planes = [PlaneData(center=np.ones(3), normal=np.array([0., 0., 1.]))]
        ws.active_pathline_plane_indices = [0]
        window._mark_action_settings_pending((StepId.PLANE_STREAMLINES,))
        requested = []
        monkeypatch.setattr(window, "_find_uid_by_indexed_data_key", lambda *args: "existing")
        monkeypatch.setattr(window, "_start_pathline_task", lambda indices, **kwargs: requested.append(indices))
        window._trigger_pathlines([0], prepare_segmentation=False)
        assert requested == [[0]]
        window._close_progress_dialog(window._pathline_progress_dialog)
        window._pathline_progress_dialog = None

        dialogs = []
        for action in (StepId.BACKGROUND_CORRECTION, StepId.UNWRAP_PHASE, StepId.GENERATE_SKELETON,
                       StepId.GENERATE_PLANES, StepId.PLANE_STREAMLINES, StepId.COMPUTE_PWV):
            dialog = ActionSettingsDialog(action.label, action_sections(action, ws), ws, ws)
            dialogs.append(dialog)
        bpc, unwrap, skeleton, planes, pathlines, pwv = dialogs
        def choose(dialog, key, value):
            control = dialog.controls[key]
            control.setCurrentIndex(control.findData(value))
        choose(bpc, "loader_params.background_phase_correction.method", "wrls_arto")
        assert bpc.rows["loader_params.background_phase_correction.threshold"].isHidden()
        choose(bpc, "loader_params.background_phase_correction.method", "msac")
        assert not bpc.rows["loader_params.background_phase_correction.threshold"].isHidden()
        choose(unwrap, "phase_unwrap_params.method", "nprs")
        assert unwrap.rows["phase_unwrap_params.lap4d_ts"].isHidden()
        assert not unwrap.rows["phase_unwrap_params.nprs_upsampling_factor"].isHidden()
        choose(unwrap, "phase_unwrap_params.method", "gc3D")
        assert unwrap.rows["phase_unwrap_params.nprs_upsampling_factor"].isHidden()
        skeleton.controls["skeleton_params.remove_small_cc"].setChecked(True)
        choose(skeleton, "skeleton_params.cc_filter_mode", "largest")
        assert skeleton.rows["skeleton_params.min_cc_volume_mm3"].isHidden()
        assert skeleton.rows["skeleton_params.cc_rel_min_ratio"].isHidden()
        choose(planes, "plane_gen_params.plane_mode", "uniform")
        assert planes.rows["plane_gen_params.anchor"].isHidden()
        assert planes.rows["plane_gen_params.cross_section_distance"].isHidden()
        choose(planes, "plane_gen_params.plane_mode", "fixed_step")
        choose(planes, "plane_gen_params.spacing_mode", "fraction")
        assert not planes.rows["plane_gen_params.spacing_ratio"].isHidden()
        assert planes.rows["plane_gen_params.cross_section_distance"].isHidden()
        choose(pathlines, "streamline_params.pathline_seed_mode", "fixed")
        assert pathlines.rows["streamline_params.pathline_seed_ratio"].isHidden()
        choose(pwv, "pwv_params.transit_time_method", "cross_correlation")
        assert pwv.rows["pwv_params.foot_method"].isHidden()
        assert not pwv.rows["pwv_params.xcorr_window"].isHidden()
        for dialog in dialogs:
            dialog.close()
            dialog.deleteLater()
    finally:
        window.close()
        window.deleteLater()
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
    assert events == ["background", "noise", "unwrap", "pcmra", "segmentation"]
    assert "load_and_background_correction" in summary["stage_times_sec"]
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
        assert isinstance(mesh, pv.PolyData)
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


@pytest.mark.parametrize('depth', [2, 60])
def test_noise_region_phantom_points_exclude_retained_voxels_and_zero_padding(depth):
    from contextlib import closing
    import pyvista as pv
    from autoflow.core.models import ObjectKind
    from autoflow.ui.viewer import SceneController

    ws = Workspace()
    ws.pcmra_render_mask = np.ones((5, depth, 5), dtype=bool)
    ws.pcmra_render_mask[2, :, 2] = False
    ws.pcmra_render_mask[0] = False
    ws.mag_raw = np.ones((5, depth, 5, 1), dtype=np.float32)
    ws.mag_raw[0] = 0.0
    uid = ws.add_object('Noise Region', ObjectKind.AUX, 'noise_region', visible=True,
                        color='#ff0000', opacity=0.15, point_size=2)
    errors = []
    with closing(pv.Plotter(off_screen=True, window_size=(128, 128))) as plotter:
        controller = SceneController(plotter, ws, errors.append)
        controller.render_all()
        obj = ws.scene_objects[uid]
        assert obj.actor is not None
        cloud = obj.actor.mapper.dataset
        assert cloud.n_points == depth
        voxels = np.rint(cloud.points / ws.resolution - 0.5).astype(int)
        assert not ws.pcmra_render_mask[tuple(voxels.T)].any()
        assert np.all(ws.mag_raw[tuple(voxels.T) + (0,)] > 0)
        np.testing.assert_array_equal(cloud['Noise RGBA'][:, :3], [[255, 0, 0]] * depth)
        np.testing.assert_allclose(cloud['Noise RGBA'][:, 3] / 255, .15, atol=1/255)
        obj.opacity = 0.05
        controller.apply_object_properties(obj)
        np.testing.assert_allclose(cloud['Noise RGBA'][:, 3] / 255, .05, atol=1/255)
        ws.pcmra_render_mask = np.ones_like(ws.pcmra_render_mask)
        controller.readd_object(obj)
        assert obj.actor is None
    assert not errors


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


@pytest.mark.parametrize("mirrored", [False, True])
def test_manual_plane_phantom_handles_selection_lock_and_deferred_metrics(monkeypatch, tmp_path, mirrored):
    app = _smoke_qt_application(monkeypatch)
    monkeypatch.setenv("AUTOFLOW_SSH_RENDERING", "1")
    from PySide6 import QtCore, QtTest
    import autoflow.ui.app as ui_app
    from autoflow.ui.remote_plotter import RemotePlotter
    monkeypatch.setattr(ui_app, "RemotePlotter", RemotePlotter, raising=False)
    window = ui_app.MainWindow()
    try:
        ws = window.workspace
        ws.paths.output_dir = str(tmp_path)
        ws.data_loaded = True
        ws.mag_raw = np.ones((20, 20, 20, 2), dtype=np.float32)
        ws.flow_raw = np.zeros((20, 20, 20, 2, 3), dtype=np.float32)
        ws.segmask_binary = np.ones((20, 20, 20, 2), dtype=bool)
        ws.resolution = np.ones(3)
        ws.origin = np.array([11., -7., 23.])
        ws.ortho_cursor = np.array([10, 10, 10])
        ws.planes = [PlaneData(center=np.array([8., 8., 8.]), normal=np.array([0., 0., 1.]))]
        window._rebuild_plane_objects()
        first_uid = window._find_uid_by_indexed_data_key("plane", 0)
        first_actor = ws.scene_objects[first_uid].actor
        ws.pathline_cache[0] = "keep existing pathline"
        if mirrored:
            window.scene._display_axis_signs = np.array([-1., 1., 1.])
            window.scene.reset_display_reference()
        window.show()
        app.processEvents()
        window._add_manual_plane()
        assert ws.scene_objects[first_uid].actor is first_actor
        assert ws.pathline_cache[0] == "keep existing pathline"
        assert window._plane_edit_enabled and window.scene._plane_selection_locked
        handles = window._plane_handles
        assert [widget.GetEnabled() for widget, _ in handles.handles] == [1, 1, 1]
        assert "left-drag" in handles.help.GetInput()
        writes, recomputes = [], []
        monkeypatch.setattr(window, "_persist_plane_outputs", lambda **kw: writes.append(kw))
        monkeypatch.setattr(window, "_recompute_dragged_plane_metrics", lambda **kw: recomputes.append(kw))
        window._on_3d_plane_picked(None, None)
        window._on_3d_path_picked("accidental path", 0)
        assert window._selected_plane_index == 1 and window._plane_handles is handles
        app.processEvents()
        window.plotter.view_isometric()
        window.scene.reset_camera()
        window.plotter.render()
        camera = np.asarray(tuple(window.plotter.camera_position), dtype=float)

        def screen_point(point):
            renderer = window.plotter.renderer
            renderer.SetWorldPoint(*point, 1)
            renderer.WorldToDisplay()
            x, y, _ = renderer.GetDisplayPoint()
            rw, rh = window.plotter.render_window.GetSize()
            return QtCore.QPoint(round(x * window.plotter.width() / rw),
                                 round((rh - 1 - y) * window.plotter.height() / rh))

        def drag(index, target):
            start = screen_point(handles.handles[index][1].GetWorldPosition())
            end = screen_point(target)
            QtTest.QTest.mouseMove(window.plotter, start)
            QtTest.QTest.mousePress(window.plotter, QtCore.Qt.LeftButton, pos=start)
            assert handles.dragging
            QtTest.QTest.mouseMove(window.plotter, end)
            QtTest.QTest.mouseRelease(window.plotter, QtCore.Qt.LeftButton, pos=end)
            assert not handles.dragging and not window.scene._interaction_preview_active
            np.testing.assert_allclose(tuple(window.plotter.camera_position), camera)
            assert not writes and not recomputes

        before = ws.planes[1].center.copy()
        drag(0, np.asarray(handles.handles[0][1].GetWorldPosition()) + [1., 0., 1.])
        assert np.linalg.norm(ws.planes[1].center - before) > 0.2
        for index in (1, 2):
            plane = ws.planes[1]
            before_normal = plane.normal.copy()
            axis = window._plane_axis_u if index == 1 else window._plane_axis_v
            target = plane.center + ws.origin + window._plane_widget_distance() * (0.866 * axis + 0.5 * plane.normal)
            drag(index, window.scene.world_to_display_point(target))
            assert np.linalg.norm(plane.normal - before_normal) > 0.01
            np.testing.assert_allclose(np.dot(window._plane_axis_u, window._plane_axis_v), 0., atol=1e-10)
            np.testing.assert_allclose(np.cross(window._plane_axis_u, window._plane_axis_v), plane.normal, atol=1e-10)
        assert "pending" in window.ortho_viewer.label_plane_metric.text()
        window._toggle_selected_plane_edit()
        assert len(writes) == 1 and not recomputes  # A new plane has no existing metrics to refresh.
        assert not window.scene._plane_selection_locked and window._plane_handles is None
        assert not window.scene._hud_renderer.HasViewProp(handles.help)

        # Existing measurements refresh once on Finish, never at each release.
        ws.planes[1].metrics = {"area_mm2": [1., 1.]}
        window._activate_plane_drag_widgets(1)
        window._update_plane_from_drag(1, center=ws.planes[1].center + [0.5, 0., 0.])
        assert not ws.planes[1].metrics
        window._toggle_selected_plane_edit()
        assert recomputes == [{"persist": True}]
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()


def _smoke_qt_application(monkeypatch):
    global _qt_smoke_application
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6 import QtWidgets
    _qt_smoke_application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    return _qt_smoke_application


def test_browser_plane_phantom_display_controls_and_visibility_clicks(monkeypatch, tmp_path):
    app = _smoke_qt_application(monkeypatch)
    monkeypatch.setenv("AUTOFLOW_SSH_RENDERING", "1")
    from PySide6 import QtCore, QtTest
    import pyvista as pv
    import autoflow.ui.app as ui_app
    from autoflow.ui.remote_plotter import RemotePlotter
    from autoflow.core.models import ObjectKind
    monkeypatch.setattr(ui_app, "RemotePlotter", RemotePlotter, raising=False)
    window = ui_app.MainWindow()
    try:
        window.show()
        app.processEvents()
        assert not window.chk_all_planes_visible.isEnabled()
        assert not window.chk_all_pathlines_visible.isEnabled()
        browser = window.tree_objects.parentWidget()
        assert window.slider_plane_size.parentWidget() is browser
        assert window.spin_plane_size.parentWidget() is browser
        assert window.chk_plane_selected_only.parentWidget() is browser
        ws = window.workspace
        ws.paths.output_dir = str(tmp_path)
        ws.data_loaded = True
        ws.flow_raw = np.zeros((7, 7, 7, 2, 3), dtype=np.float32)
        ws.mag_raw = np.ones((7, 7, 7, 2), dtype=np.float32)
        ws.segmask_3d = np.ones((7, 7, 7), dtype=bool)
        ws.planes = [PlaneData(center=np.array([3., 3., float(z)]), normal=np.array([0., 0., 1.]))
                     for z in (2, 4)]
        planes = [ws.scene_objects[ws.add_object(f"Plane {i}", ObjectKind.PLANE, f"plane_{i}")]
                  for i in range(2)]
        ws.active_pathline_plane_indices = [0, 1]
        for i in range(2):
            ws.pathline_cache[i] = {0: pv.Line((2, 3, i + 2), (4, 3, i + 2))}
        pathlines = [ws.scene_objects[ws.add_object(f"Pathline {i}", ObjectKind.FLOW, f"pathline_{i}")]
                     for i in range(2)]
        window.scene.sync_from_workspace()
        window._refresh_browser()
        item = window._find_browser_item_by_uid(planes[0].uid)
        window.tree_objects.setCurrentItem(item)
        window.chk_plane_selected_only.setChecked(True)
        app.processEvents()
        assert [obj.actor.GetVisibility() for obj in planes] == [1, 0]
        highlight = window.scene._highlight_plane_actor
        assert highlight.GetVisibility()

        def click(checkbox):
            QtTest.QTest.mouseClick(checkbox, QtCore.Qt.LeftButton,
                                   pos=QtCore.QPoint(8, checkbox.height() // 2))
            app.processEvents()

        # Exercise Qt's actual signal type and mouse-click state transitions.
        click(window.chk_all_planes_visible)
        assert window.chk_all_planes_visible.checkState() == QtCore.Qt.Unchecked
        assert not any(obj.visible or obj.actor.GetVisibility() for obj in planes)
        assert not highlight.GetVisibility()
        assert window.tree_objects.currentItem() is item and item.isSelected()
        click(window.chk_all_planes_visible)
        assert all(obj.visible for obj in planes)
        assert [obj.actor.GetVisibility() for obj in planes] == [1, 0]
        assert highlight.GetVisibility()
        assert window.chk_plane_selected_only.isChecked()
        window._find_browser_item_by_uid(planes[1].uid).setCheckState(0, QtCore.Qt.Unchecked)
        assert window.chk_all_planes_visible.checkState() == QtCore.Qt.PartiallyChecked
        click(window.chk_all_planes_visible)
        assert all(obj.visible for obj in planes)
        assert window.chk_all_planes_visible.checkState() == QtCore.Qt.Checked
        assert window.chk_all_pathlines_visible.checkState() == QtCore.Qt.PartiallyChecked
        click(window.chk_all_pathlines_visible)
        assert all(obj.visible and obj.actor.GetVisibility() for obj in pathlines)
        click(window.chk_all_pathlines_visible)
        assert not any(obj.visible or obj.actor.GetVisibility() for obj in pathlines)
        click(window.chk_all_pathlines_visible)
        assert all(obj.visible and obj.actor.GetVisibility() for obj in pathlines)
        window.spin_plane_size.setValue(6.0)
        assert window.slider_plane_size.value() == 60
        assert planes[0].actor.mapper.dataset.length == pytest.approx(6 * np.sqrt(2))
        window.chk_plane_selected_only.setChecked(False)
        assert all(obj.actor.GetVisibility() for obj in planes)
    finally:
        window.plotter.close()
        window.hide()


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


def test_plane_overlay_phantom_keeps_edges_visible_and_removes_stale_actors():
    from contextlib import closing
    import pyvista as pv
    from autoflow.core.models import ObjectKind
    from autoflow.ui.viewer import SceneController

    ws = Workspace()
    ws.planes = [PlaneData(center=np.zeros(3), normal=np.array([0., 0., 1.]))]
    ws.render_settings["plane_render_cfg"] = {"plane_size_mm": 25.0}
    uid = ws.add_object("Plane", ObjectKind.PLANE, "plane_0", color="yellow",
                        visible=True, opacity=0.75, line_width=4)
    errors = []
    with closing(pv.Plotter(off_screen=True, window_size=(256, 256))) as plotter:
        scene = SceneController(plotter, ws, errors.append)
        scene.initialize()
        scene._render_object(ws.scene_objects[uid])
        overlay = scene._plane_overlay_renderer
        # Do not let a spare orientation renderer accidentally clear depth
        # before the plane layer and conceal this regression.
        for renderer in plotter.render_window.GetRenderers():
            if renderer.GetLayer() > 0 and renderer is not overlay:
                renderer.SetDraw(False)
        plotter.disable_anti_aliasing()
        shell = plotter.add_mesh(pv.Sphere(radius=15), color="red", render=False,
                                 reset_camera=False, opacity=1.0)
        plotter.camera_position = [(0., 0., 40.), (0., 0., 0.), (0., 1., 0.)]
        plotter.camera.parallel_projection = True
        plotter.camera.parallel_scale = 18.0
        plotter.show(auto_close=False, interactive=False)

        def yellow_pixels():
            frame = plotter.screenshot()
            return ((frame[..., 0] > 150) & (frame[..., 1] > 150)
                    & (frame[..., 2] < 80))

        actor = ws.scene_objects[uid].actor
        foreground = yellow_pixels()
        shell.SetVisibility(False)
        plotter.render()
        np.testing.assert_array_equal(yellow_pixels(), foreground)
        assert np.count_nonzero(foreground) > 500
        shell.SetVisibility(True)
        for _ in range(6):
            ws.scene_objects[uid].visible = False
            scene.apply_object_properties(ws.scene_objects[uid])
            ws.scene_objects[uid].visible = True
            scene.apply_object_properties(ws.scene_objects[uid])
        np.testing.assert_array_equal(yellow_pixels(), foreground)
        camera = list(plotter.camera_position)
        scene.toggle_axes()
        scene.toggle_axes()
        assert ws.scene_objects[uid].actor is actor
        np.testing.assert_allclose(list(plotter.camera_position), camera)
        scene.highlight_plane(uid)
        highlight = scene._highlight_plane_actor
        assert highlight is not None
        del ws.scene_objects[uid]
        scene.sync_from_workspace(rebuild_prefixes=("pcmra_volume", "noise_region"))
        assert not overlay.HasViewProp(actor)
        assert not overlay.HasViewProp(highlight)
        assert scene._highlight_plane_uid is None
        assert overlay.GetViewProps().GetNumberOfItems() == 0
    assert not errors


def test_geometry_phantom_reruns_replace_old_planes_and_clear_measurements(tmp_path):
    from contextlib import closing
    import pyvista as pv
    from autoflow.core.models import ObjectKind
    from autoflow.ui.viewer import SceneController

    x, y, z = np.indices((25, 25, 41))
    mask = ((x - 12) ** 2 + (y - 12) ** 2 < 25) & (z >= 3) & (z <= 37)
    ws = Workspace()
    ws.data_loaded = True
    ws.segmask_raw = np.repeat(mask[..., None].astype(np.int16), 2, axis=3)
    ws.flow_raw = np.zeros(mask.shape + (2, 3), dtype=np.float32)
    ws.flow_raw[..., 2] = ws.segmask_raw * 20.0
    ws.paths.output_dir = str(tmp_path)
    ws.plane_gen_params.plane_mode = "evenly_spaced"
    ws.plane_gen_params.plane_count = 3
    ws.plane_gen_params.start_distance = ws.plane_gen_params.end_distance = 0.0
    ws.plane_gen_params.segmentation_filter = False
    engine = PipelineEngine()
    for step in (StepId.GENERATE_SKELETON, StepId.GENERATE_GRAPH, StepId.GENERATE_PLANES):
        result = engine.run_step(ws, step, lambda _: None)
        assert result.success and not result.skipped, result.message
    assert len(ws.planes) == 3
    errors = []
    with closing(pv.Plotter(off_screen=True, window_size=(160, 160))) as plotter:
        scene = SceneController(plotter, ws, errors.append)
        scene.initialize()
        scene.render_all()
        plane_objects = [obj for obj in ws.scene_objects.values() if obj.kind == ObjectKind.PLANE]
        old_actors = [obj.actor for obj in plane_objects]
        measurements, _, _ = engine._compute_plane_metrics_internal(ws, save=False, include_derived=False)
        centers = [plane.center.copy() for plane in ws.planes]
        normals = [plane.normal.copy() for plane in ws.planes]
        scene.set_plane_display_settings(size_mm=6.0)
        for obj, old_actor in zip(plane_objects, old_actors):
            assert obj.actor is old_actor
            assert np.isclose(obj.actor.mapper.dataset.length, 6.0 * np.sqrt(2))
        scene.highlight_plane(plane_objects[0].uid)
        scene.set_plane_display_settings(selected_only=True)
        assert [obj.actor.GetVisibility() for obj in plane_objects] == [1, 0, 0]
        assert all(obj.visible for obj in plane_objects)
        scene.highlight_plane(plane_objects[1].uid)
        assert [obj.actor.GetVisibility() for obj in plane_objects] == [0, 1, 0]
        scene.set_plane_display_settings(selected_only=False)
        scene.highlight_plane(None)
        assert all(obj.actor.GetVisibility() for obj in plane_objects)
        recomputed, _, _ = engine._compute_plane_metrics_internal(ws, save=False, include_derived=False)
        assert recomputed == measurements
        for plane, center, normal in zip(ws.planes, centers, normals):
            np.testing.assert_array_equal(plane.center, center)
            np.testing.assert_array_equal(plane.normal, normal)
        ws.derived.plane_metrics = [{"old_measurement": 1.0}] * len(ws.planes)
        ws.derived.plane_pixelwise_file = "previous-plane-metrics.h5"
        ws.pathline_cache[0] = "old-pathline"
        ws.pathline_seed_cache[0] = "old-seed"
        ws.active_pathline_plane_indices = [0]
        ws.add_object("Old pathline", ObjectKind.FLOW, "pathline_0", visible=False)
        ws.pipeline.mark_done(StepId.COMPUTE_PLANE_METRICS)
        ws.plane_gen_params.plane_count = 1
        assert engine.run_step(ws, StepId.GENERATE_PLANES, lambda _: None).success
        scene.sync_from_workspace(rebuild_prefixes=("plane_", "smooth_path_", "pathline_"))
        assert len(ws.planes) == 1
        assert not any(scene._plane_overlay_renderer.HasViewProp(actor) for actor in old_actors)
        assert scene._plane_overlay_renderer.GetViewProps().GetNumberOfItems() == 1
        assert ws.derived.plane_metrics == []
        assert not ws.derived.plane_pixelwise_file
        assert not ws.pathline_cache and not ws.pathline_seed_cache
        assert not ws.active_pathline_plane_indices
        assert not ws.pipeline.is_done(StepId.COMPUTE_PLANE_METRICS)

        assert engine.run_step(ws, StepId.GENERATE_GRAPH, lambda _: None).success
        scene.sync_from_workspace(rebuild_prefixes=("graph_", "path_"))
        assert not ws.planes
        assert scene._plane_overlay_renderer.GetViewProps().GetNumberOfItems() == 0
        assert not ws.pipeline.is_done(StepId.GENERATE_PLANES)
        assert engine.run_step(ws, StepId.GENERATE_PLANES, lambda _: None).success
        scene.sync_from_workspace()
        assert len(ws.planes) == 1
        assert engine.run_step(ws, StepId.GENERATE_SKELETON, lambda _: None).success
        scene.sync_from_workspace(rebuild_prefixes=("skeleton_",))
        assert not ws.planes and not ws.centerline_paths
        assert ws.graph.points.size == 0 and ws.branch_labels is None
        assert scene._plane_overlay_renderer.GetViewProps().GetNumberOfItems() == 0
        assert not ws.pipeline.is_done(StepId.GENERATE_GRAPH)
        assert not ws.pipeline.is_done(StepId.GENERATE_PLANES)
        assert not any(obj.kind == ObjectKind.PLANE for obj in ws.scene_objects.values())
    assert not errors


def test_video_phase_schedule_repeats_every_phase_and_cycles_for_rotation():
    from autoflow.rendering.videos import _time_index, _time_and_azimuth

    expected = [0, 0, 1, 1, 2, 2, 3, 3]
    assert [_time_index(i, 4, 2) for i in range(8)] == expected
    rotating = [_time_and_azimuth(i, 12, 4, 2) for i in range(12)]
    assert [phase for phase, _ in rotating] == expected + [0, 0, 1, 1]
    assert [angle for _, angle in rotating] == list(range(0, 360, 30))
    assert _time_index(0, 0, 0) == 0


def test_plane_support_matches_all_fields_and_prepares_tke_once(monkeypatch):
    from autoflow.algorithms.metrics import plane_derived
    from autoflow.algorithms.branch import segment_vessels_from_graph_and_mask
    from autoflow.algorithms.metrics.planes import compute_plane_metrics
    from autoflow.algorithms.surfaces import create_uniform_field_grid
    labels = np.zeros((20, 20, 20, 2), dtype=np.int16)
    labels[4:8, 8:12, 4:16, :] = 1
    labels[12:16, 8:12, 4:16, :] = 2
    labels[8:12, 8:12, 14:16, :] = 2
    # The binary vessel stays fixed while its selected label changes by phase.
    labels[6:8, 8:12, 4:14, 1] = 2
    mask = labels > 0
    flow = np.zeros(labels.shape + (3,), dtype=np.float32)
    flow[..., 2] = mask * 10
    points = np.array([(6, 10, z) for z in range(4, 16)] + [(x, 10, 15) for x in range(7, 15)] + [(14, 10, z) for z in range(14, 3, -1)], dtype=float)
    graph = GraphData(points=points, edges=np.column_stack((np.arange(len(points)-1), np.arange(1, len(points)))))
    branches, paths, _, path_info, forks = segment_vessels_from_graph_and_mask(
        np.any(mask, axis=3), graph, (1, 1, 1), flow_xyzt3=flow,
        segmask_binary_4d=mask, segmentation_labels=labels[..., 0])
    planes = [PlaneData(center=np.array([5., 10., 10.]), normal=np.array([0., 0., 1.]), label=1, segmentation_label=1) for _ in range(3)]
    planes[1].roi_edit_operations = {"1": [{"mode": "replace", "polygon": [[-1, -1], [1, -1], [1, 1], [-1, 1]]}]}
    base = compute_plane_metrics(flow, mask, (1, 1, 1), (0, 0, 0), planes,
        branch_labels_3d=branches, segmentation_labels_3d=labels[..., 0],
        segmentation_labels_4d=labels, paths=paths, path_info=path_info, forks=forks)
    field = np.where(labels == 1, 1., 100.).astype(np.float32)
    surface = create_uniform_field_grid(mask[..., 0].astype(np.uint8), (1, 1, 1), name="mask").threshold(0.5, scalars="mask").extract_surface(algorithm="dataset_surface")
    surface.point_data["wss"] = np.where(surface.points[:, 0] < 9, 1., 100.)
    calls = []
    prepare = plane_derived._prepare_tke_array
    def counted(*args, **kwargs):
        calls.append(1)
        return prepare(*args, **kwargs)
    monkeypatch.setattr(plane_derived, "_prepare_tke_array", counted)
    derived, pixels = plane_derived.augment_plane_metrics_with_derived(base, planes, mask, (1, 1, 1), (0, 0, 0),
        branch_labels_3d=branches, segmentation_labels_3d=labels[..., 0], segmentation_labels_4d=labels,
        tke_array=field, pressure_gradient_array=np.repeat(field[..., None], 3, axis=-1),
        relative_pressure_array=field * 3, wss_surfaces=[surface, surface])
    assert len(calls) == 1
    assert base[0]["area_mm2"][1] < base[0]["area_mm2"][0]
    for basic, metric, payload in zip(base, derived, pixels):
        areas = [float(np.sum(frame["cell_area_mm2"])) for frame in payload["timepoints"]]
        np.testing.assert_allclose(areas, basic["area_mm2"])
        np.testing.assert_allclose(metric["tke_mean_J_m3_t"], 1.)
        np.testing.assert_allclose(metric["relative_pressure_mean_Pa_t"], 3.)
        np.testing.assert_allclose(metric["wss_wall_mean_Pa_t"], 1.)


@pytest.mark.parametrize("width_voxels,normal,min_points", [
    (1, (0.0, 0.0, 1.0), 8),
    (2, (0.0, 0.0, 1.0), 8),
    (3, (0.0, 0.0, 1.0), 8),
    (1, (0.2, 0.3, 1.0), 8),
    (1, (0.0, 0.0, 1.0), 100),
])
def test_small_vessel_plane_contour_phantom_survives_display_filters(
    monkeypatch, width_voxels, normal, min_points,
):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets
    from autoflow.ui.ortho_viewer import OrthoViewer

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    ws = Workspace()
    ws.resolution = np.full(3, 0.5)
    ws.origin = np.array([10.0, -20.0, 30.0])
    ws.mag_raw = np.ones((24, 24, 24, 2), dtype=np.float32)
    ws.segmask_binary = np.zeros(ws.mag_raw.shape, dtype=bool)
    ws.segmask_binary[10:10 + width_voxels, 10:10 + width_voxels, :, 0] = True
    # Another component must not become the selected vessel's cyan outline.
    ws.segmask_binary[20:22, 20:22, :, 0] = True
    ws.segmask_raw = ws.segmask_binary.astype(np.int16)
    ws.segmask_labels = ws.segmask_raw.copy()
    plane = PlaneData(
        center=np.array([(10 + width_voxels / 2) * 0.5] * 2 + [6.0]),
        normal=np.asarray(normal), label=1, segmentation_label=1,
    )
    ws.planes = [plane]
    ws.derived.plane_metrics = [{"area_mm2": [width_voxels ** 2 * 0.25, 0.0]}]
    original_mask = ws.segmask_binary.copy()
    original_labels = ws.segmask_labels.copy()
    original_metrics = json.dumps(ws.derived.plane_metrics)
    viewer = OrthoViewer(ws)
    try:
        viewer._plane_display_smoothing_sigma = 0.8
        viewer._plane_display_smoothing_points = min_points
        viewer.set_selected_plane(0)
        region = viewer._effective_plane_region(plane, 0)
        sampled = viewer._sample_smooth_region_mask(region, plane, 8.0, 0.375)
        assert sampled is not None and np.max(sampled[0]) > 0.5
        view = viewer.slice_views["axial"]
        assert len(view.contour_items) == 1
        for item in view.contour_items:
            x, y = item.getData()
            points = np.column_stack((x, y))
            assert len(points) >= 4
            np.testing.assert_allclose(points[0], points[-1])
            assert np.max(np.abs(points)) < width_voxels * 0.5
            assert abs(polygon_area(points)) > 0.0

        # Even a subpixel ROI in a wide FOV must retain its vector boundary.
        wide_contours = viewer._selected_plane_contours(plane, 0, 512.0, 0.375)
        assert wide_contours
        assert all(abs(polygon_area(contour[0])) > 0.0 for contour in wide_contours)

        ws.current_t = 1
        viewer.refresh(update_plane=False)
        assert len(view.contour_items) == 1
        carried_region = viewer._effective_plane_region(plane, 1)
        np.testing.assert_allclose(carried_region.points, region.points)
        ws.current_t = 0
        viewer.refresh(update_plane=False)
        assert len(view.contour_items) == 1
        np.testing.assert_array_equal(ws.segmask_binary, original_mask)
        np.testing.assert_array_equal(ws.segmask_labels, original_labels)
        assert json.dumps(ws.derived.plane_metrics) == original_metrics
        assert not plane.roi_edit_operations and not plane.roi_polygon_uv_mm
    finally:
        viewer.close()
        viewer.deleteLater()
        app.processEvents()


def test_missing_plane_roi_phantom_carries_geometry_and_samples_current_fields(tmp_path, monkeypatch):
    from autoflow.algorithms.metrics import compute_plane_metrics, augment_plane_metrics_with_derived, save_plane_pixelwise_h5
    from autoflow.algorithms.metrics.sampling import _build_mask_phase_lookup, _get_plane_metric_slice_spec
    labels = np.zeros((18, 18, 12, 6), dtype=np.int16)
    labels[1:3, 1:3, :, :] = 16  # A different vessel must never replace this plane's label.
    labels[8:10, 8:10, :, 1] = 14
    labels[7:10, 7:10, :, 4] = 14
    mask = labels > 0
    flow = np.zeros(labels.shape + (3,), dtype=np.float32)
    flow[..., 2] = np.arange(1, 7) * 10
    plane = PlaneData(center=np.array([8.5, 8.5, 6.3]), normal=np.array([0., 0., 1.]),
                      segmentation_label=14)
    original_labels = labels.copy()
    metric = compute_plane_metrics(flow, mask, (1, 1, 1), (10, -20, 30), [plane],
                                   segmentation_labels_4d=labels)[0]
    assert metric["roi_source_frame_t"] == [None, 1, 1, 1, 4, 4]
    assert metric["roi_carried_forward_t"] == [False, False, True, True, False, True]
    np.testing.assert_allclose(metric["area_mm2"], [0, 4, 4, 4, 9, 9])
    np.testing.assert_allclose(metric["flowrate_mL_s"], [0, .8, 1.2, 1.6, 4.5, 5.4])
    lookup, cache = _build_mask_phase_lookup(mask), {}
    for frame in (5, 2, 4, 1, 3, 0):
        spec = _get_plane_metric_slice_spec(cache, mask, lookup, None, plane, frame, (1, 1, 1), (10, -20, 30),
                                            segmentation_labels_4d=labels)
        assert (None if spec is None else spec["roi_source_frame"]) == metric["roi_source_frame_t"][frame]
    field = np.broadcast_to(np.arange(1, 7), labels.shape).copy().astype(np.float32)
    derived, pixels = augment_plane_metrics_with_derived(
        [metric], [plane], mask, (1, 1, 1), (10, -20, 30),
        tke_array=field, relative_pressure_array=field * 3, segmentation_labels_4d=labels)
    # The ROI is carried, while derived-field masks and validity stay unchanged.
    np.testing.assert_allclose(derived[0]["tke_mean_J_m3_t"], [0, 2, 0, 0, 5, 0])
    np.testing.assert_allclose(derived[0]["relative_pressure_mean_Pa_t"][1:], [6, 9, 12, 15, 18])
    exported = tmp_path / "carried-roi.h5"
    save_plane_pixelwise_h5(exported, pixels)
    with h5py.File(exported) as h5:
        assert h5["planes/000/timepoints/002/roi_source_frame"][0] == 1
        assert h5["planes/000/timepoints/002/roi_carried_forward"][0] == 1
        np.testing.assert_allclose(h5["planes/000/timepoints/002/relative_pressure_Pa"][:], 9)
    np.testing.assert_array_equal(labels, original_labels)
    assert not plane.roi_edit_operations

    # Explicit deletion stays empty and stops forward filling until a real ROI returns.
    plane.roi_edit_operations = {"2": [{"mode": "remove", "polygon": [[-30, -30], [30, -30], [30, 30], [-30, 30]]}]}
    # Frame 2 has no observed ROI; an explicit empty edit is a propagation barrier.
    deleted = compute_plane_metrics(flow, mask, (1, 1, 1), (10, -20, 30), [plane],
                                   segmentation_labels_4d=labels)[0]
    assert deleted["roi_source_frame_t"] == [None, 1, None, None, 4, 4]
    plane.roi_edit_operations = {"0": [{"mode": "replace", "polygon": [[-1, -1], [1, -1], [1, 1], [-1, 1]]}]}
    created = compute_plane_metrics(flow, mask, (1, 1, 1), (10, -20, 30), [plane],
                                   segmentation_labels_4d=labels)[0]
    assert created["area_mm2"][0] > 0 and created["roi_source_frame_t"][0] == 0

    # Playback can jump directly to a missing frame. A previous manual vector
    # contour must be displayed verbatim, matching its own source phase.
    _smoke_qt_application(monkeypatch)
    from autoflow.ui.ortho_viewer import OrthoViewer
    polygon = [[-.75, -.75], [.75, -.75], [.75, .75], [-.75, .75]]
    plane.roi_edit_operations = {"4": [{"mode": "replace", "polygon": polygon}]}
    ws = Workspace()
    ws.mag_raw = np.ones(labels.shape, dtype=np.float32)
    ws.segmask_raw = labels
    ws.segmask_labels = labels
    ws.segmask_binary = mask
    ws.flow_raw = flow
    ws.origin = np.array([10., -20., 30.])
    ws.planes = [plane]
    ws.current_t = 5
    viewer = OrthoViewer(ws)
    try:
        viewer.set_selected_plane(0)
        region, source = viewer._effective_plane_region(plane, 5, return_source=True)
        assert region is not None and source == 4
        for frame in (5, 4):
            contours = viewer._selected_plane_contours(plane, frame, 20., .75)
            cyan = [contour for contour in contours if contour[1] == "#00f5d4"]
            assert len(cyan) == 1
            np.testing.assert_allclose(cyan[0][0], np.vstack((polygon, polygon[0])))
        assert viewer._effective_plane_region(plane, 0) is None
        plane.roi_edit_operations = {"1": [{"mode": "remove", "polygon": [[-30, -30], [30, -30], [30, 30], [-30, 30]]}]}
        assert viewer._effective_plane_region(plane, 1) is None
        assert not viewer._selected_plane_contours(plane, 1, 20., .75)
        np.testing.assert_array_equal(ws.segmask_labels, original_labels)
    finally:
        viewer.close()
        viewer.deleteLater()


@pytest.mark.parametrize("sources,carried,frame_only", [
    ([0, 1, 2, 3], [False] * 4, True),
    ([0, 1, 1, 1], [False, False, True, True], False),
    ([None, None, None, 3], [False] * 4, False),
])
def test_contour_edit_phantom_updates_dependent_carried_frames(sources, carried, frame_only):
    from autoflow.ui.app import MainWindow
    ws = Workspace()
    ws.planes = [PlaneData(center=np.zeros(3), normal=np.array([0., 0., 1.]))]
    ws.current_t = 1
    ws.derived.plane_metrics = [{"roi_source_frame_t": sources, "roi_carried_forward_t": carried}]
    calls = []
    window = SimpleNamespace(workspace=ws, log=lambda *_: None,
                             _recompute_dragged_plane_metrics=lambda **kwargs: calls.append(kwargs))
    MainWindow._on_plane_roi_changed(window, 0)
    assert calls == [{"persist": True, "frame_only": frame_only}]


def test_ortho_case_history_and_time_refresh_use_correct_context(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets
    from autoflow.ui.ortho_viewer import OrthoViewer
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    ws = Workspace()
    ws.mag_raw = np.stack([np.ones((4, 4, 4)), np.full((4, 4, 4), 7.)], axis=3).astype(np.float32)
    ws.planes = [PlaneData(center=np.ones(3), normal=np.array([0., 0., 1.]), label=1)]
    viewer = OrthoViewer(ws)
    viewer.set_selected_plane(0)
    snapshot = viewer._snapshot_plane_roi(ws.planes[0])
    ws.planes[0].roi_polygon_uv_mm = [[0, 0], [1, 0], [0, 1]]
    viewer._plane_roi_undo.append((0, snapshot))
    viewer._undo_plane_roi()
    assert ws.planes[0].roi_polygon_uv_mm == []
    counts = {"set": 0, "update": 0}
    for view in viewer.slice_views.values():
        original_set, original_update = view.set_slice, view.update_image
        def counted_set(*args, _method=original_set, **kwargs):
            counts["set"] += 1; return _method(*args, **kwargs)
        def counted_update(*args, _method=original_update, **kwargs):
            counts["update"] += 1; return _method(*args, **kwargs)
        monkeypatch.setattr(view, "set_slice", counted_set)
        monkeypatch.setattr(view, "update_image", counted_update)
    viewer.refresh(update_plane=False)
    counts.update(set=0, update=0)
    ws.current_t = 1
    viewer.refresh(update_plane=False)
    assert counts == {"set": 0, "update": 3}
    assert float(viewer._current_volume.mean()) == 7.
    old = ws.planes[0]
    viewer._plane_roi_undo.append((0, snapshot))
    ws.planes = [PlaneData(center=np.ones(3), normal=np.array([0., 0., 1.]), label=2)]
    viewer._undo_plane_roi()
    assert not viewer._plane_roi_undo and not viewer._plane_roi_redo
    viewer._plane_roi_undo.append((0, snapshot))
    new_case = Workspace(); new_case.planes = [old]
    viewer.workspace = new_case
    viewer._undo_plane_roi()
    assert not viewer._plane_roi_undo and not viewer._plane_roi_redo
    viewer.close()
    app.processEvents()



@pytest.mark.parametrize("statuses,expected", [(["ok"], 0), (["ok", "error"], 1), (["partial"], 1), ([], 1)])
def test_cli_exit_status_matches_batch_result(monkeypatch, statuses, expected):
    from autoflow import cli
    monkeypatch.setattr('sys.argv', ['autoflow-run', 'phantom.h5'])
    monkeypatch.setattr(cli, 'run_batch', lambda _config: ([{"status": status} for status in statuses], ''))
    assert cli.main() == expected


@pytest.mark.parametrize("video_fails", [False, True])
def test_batch_accounts_for_video_output_and_partial_failure(tmp_path, monkeypatch, video_fails):
    import time
    from autoflow import processing
    def render(_workspace, directory, **_kwargs):
        time.sleep(0.03)
        if video_fails:
            raise RuntimeError("phantom encoder failure")
        path = Path(directory) / "plane_rotate.mp4"
        path.write_bytes(b"smoke export placeholder")
        return str(path)
    monkeypatch.setattr(processing, 'render_plane_rotation_video', render)
    config = AutoFlowConfig(inputs=[str(DATA_DIR/'phantom_S.h5')], output_dir=str(tmp_path),
        skip_derived=True, requested_videos=['plane'], use_multithread=False)
    results, _ = run_batch(config)
    assert len(results) == 1
    assert results[0]['status'] == ('partial' if video_fails else 'ok')
    summary = results[0]['summary']
    assert summary['video_times_sec']['plane'] >= 0.03
    assert summary['total_time_sec'] >= summary['compute_time_sec'] + summary['video_times_sec']['plane']
    assert bool(summary['video_errors']) is video_fails
    assert (tmp_path/'time_summary.txt').is_file()


def test_fused_correction_preserves_source_sigma_without_second_decode(tmp_path, monkeypatch):
    from autoflow.algorithms import load_input_data
    from autoflow.algorithms.data import h5_loader
    path = tmp_path / 'complex_sigma.h5'
    image = np.full((3, 3, 3, 2, 4), 90+0j, dtype=np.complex64)
    image[..., 0] = 100+0j
    with h5py.File(path, 'w') as handle:
        handle.create_dataset('img_complex', data=image)
        handle.create_dataset('Resolution', data=np.ones(3))
        handle.create_dataset('VENC', data=np.full(3, 100.))
    def correction(image, config, **kwargs):
        enabled = config.enabled
        corrected = image.copy()
        if enabled: corrected[..., 1:] *= np.complex64(0.8+0.1j)
        return corrected, None, {'applied':enabled, 'enabled':enabled}
    monkeypatch.setattr(h5_loader, 'apply_background_phase_correction_to_complex', correction)
    raw = load_input_data(str(path), correction_config={'enabled':False})
    corrected = load_input_data(str(path), correction_config={'enabled':True, 'write_cache':False})
    assert not np.array_equal(raw.sigma, corrected.sigma)
    count = []
    original_load = h5_loader.load_h5_data
    def counted(*args, **kwargs):
        count.append(1)
        return original_load(*args, **kwargs)
    monkeypatch.setattr('autoflow.algorithms.inputs.load_h5_data', counted)
    ws=Workspace();ws.paths.flow_path=ws.paths.segmask_path=str(path)
    ws.loader_params.background_phase_correction.write_cache=False
    PipelineEngine().load_data(ws, lambda _message:None, apply_background_correction=True)
    assert len(count)==1
    np.testing.assert_array_equal(ws.source_sigma, raw.sigma)
    np.testing.assert_array_equal(ws.flow_raw, corrected.flow)
    assert ws.pipeline.completed.get(StepId.BACKGROUND_CORRECTION.value)


@pytest.mark.parametrize("target_size,dpr", [((225, 173), 1), ((950, 625), 1),
                                            ((317, 211), 2), ((94, 71), 1)])
def test_remote_frame_phantom_partial_repaint_matches_full(target_size, dpr):
    """Dirty X11 updates must compose exactly like a complete Qt repaint."""
    QtGui = pytest.importorskip("PySide6.QtGui")
    from PySide6 import QtCore
    from autoflow.ui.remote_plotter import _changed_frame_region

    previous = np.zeros((173, 225, 3), dtype=np.uint8)
    previous[30:75, 70:115] = [241, 76, 39]
    current = previous.copy()
    current[30:75, 70:115] = 0  # Previously drawn geometry must be erased.
    current[48:83, 98:142] = [35, 172, 251]
    current[165:, :7] = [71, 53, 19]  # Separate orientation marker at the edge.
    region = _changed_frame_region(previous, current, target_size)
    assert region is not None and not region.isEmpty()
    assert sum(rect.width() * rect.height() for rect in region) < np.prod(target_size)
    assert _changed_frame_region(current, current, target_size).isEmpty()
    assert _changed_frame_region(None, current, target_size) is None
    assert _changed_frame_region(previous[:10], current, target_size) is None
    assert _changed_frame_region(np.zeros_like(current), np.full_like(current, 255), target_size) is None

    def source_image(pixels):
        return QtGui.QImage(pixels.data, pixels.shape[1], pixels.shape[0], pixels.strides[0],
                            QtGui.QImage.Format_RGB888)

    def canvas():
        image = QtGui.QImage(target_size[0] * dpr, target_size[1] * dpr, QtGui.QImage.Format_RGB32)
        image.setDevicePixelRatio(dpr)
        return image

    def draw(image, pixels, clip=None):
        painter = QtGui.QPainter(image)
        if clip is not None:
            painter.setClipRegion(clip)
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, False)
        painter.drawImage(QtCore.QRect(0, 0, *target_size), source_image(pixels))
        painter.end()

    partial, full = canvas(), canvas()
    draw(partial, previous)
    draw(partial, current, region)
    draw(full, current)
    np.testing.assert_array_equal(np.frombuffer(partial.constBits(), dtype=np.uint8),
                                  np.frombuffer(full.constBits(), dtype=np.uint8))


def test_portable_runtime_resolves_external_models_and_configs(monkeypatch, tmp_path):
    from autoflow.config import default_config_dir, resolve_config_dir
    from autoflow.algorithms.segmentation.models import (
        bundled_nnunet_model_folder, default_nnunet_model_folder,
        resolve_nnunet_model_folder, resolve_nnunet_4d_model_folder,
    )

    root = tmp_path / "portable folder"
    models = tmp_path / "separate model drive"
    for name in ("nnunet3d", "nnunet4d", "nnunet3d-bundled"):
        (models / name).mkdir(parents=True)
    monkeypatch.setenv("AUTOFLOW_PORTABLE_ROOT", str(root))
    monkeypatch.setenv("AUTOFLOW_MODELS_DIR", str(models))
    assert default_config_dir() == root / "configs"
    assert resolve_config_dir(str(tmp_path)) == tmp_path
    assert default_nnunet_model_folder() == models / "nnunet3d"
    assert Path(resolve_nnunet_model_folder()) == models / "nnunet3d"
    assert bundled_nnunet_model_folder() == models / "nnunet3d-bundled"
    assert Path(resolve_nnunet_4d_model_folder()) == models / "nnunet4d"
    assert Path(resolve_nnunet_4d_model_folder(str(models / "nnunet3d"))) == models / "nnunet3d"
    from autoflow.algorithms.segmentation import runtime as segmentation_runtime
    monkeypatch.setattr(segmentation_runtime.shutil, "which", lambda _name: "/unrelated/python/nnUNetv2_predict_from_modelfolder")
    command = segmentation_runtime._nnunet_predict_command()
    assert command[0] == segmentation_runtime.sys.executable
    assert command[1] == "-c"
    (models / "nnunet4d").rmdir()
    with pytest.raises(FileNotFoundError, match="portable 4D nnUNet model folder"):
        resolve_nnunet_4d_model_folder()


@pytest.mark.parametrize("global_visible", [False, True])
def test_group_segmentation_phantom_keeps_preview_after_metrics(monkeypatch, tmp_path, global_visible):
    app = _smoke_qt_application(monkeypatch)
    monkeypatch.setenv("AUTOFLOW_SSH_RENDERING", "1")
    from PySide6 import QtCore
    import autoflow.ui.app as ui_app
    from autoflow.ui.remote_plotter import RemotePlotter
    monkeypatch.setattr(ui_app, "RemotePlotter", RemotePlotter, raising=False)
    window = ui_app.MainWindow()
    try:
        ws = window.workspace
        ws.paths.output_dir = str(tmp_path)
        ws.data_loaded = True
        ws.flow_raw = np.zeros((12, 10, 10, 2, 3), dtype=np.float32)
        ws.flow_raw[..., 2] = 20.0
        ws.mag_raw = np.ones((12, 10, 10, 2), dtype=np.float32)
        labels = np.zeros((12, 10, 10, 2), dtype=np.int16)
        labels[1:5, 2:7, 2:7, :] = 1
        labels[7:11, 2:7, 2:7, :] = 2
        ws.set_segmentation_source("original", labels)
        assert ws.activate_segmentation_source("original")
        ws.skeleton_params = SkeletonParams(
            label_groups={"left": {"labels": [1]}, "right": {"labels": [2]}},
            remove_small_cc=False, do_closing=False, do_opening=False,
            gaussian_enabled=False, dilation_iters=0, erosion_iters=0)
        window.pipeline.preprocess(ws)
        ws.derived_params.use_multithread = False
        ws.planes = [PlaneData(center=np.array([2.5, 4., 4.]),
                               normal=np.array([0., 0., 1.]), group_name="left")]
        window._sync_segmentation_scene_object()
        window.scene.sync_from_workspace()
        window._refresh_browser()
        window._refresh_segmentation_ui()
        window.segmentation_panel.check_visible.setChecked(global_visible)
        window._refresh_browser()
        groups = [next(obj for obj in ws.scene_objects.values()
                       if obj.data_key == f"segmask_group_{group}") for group in ("left", "right")]
        for obj, visible in zip(groups, (True, False)):
            window._find_browser_item_by_uid(obj.uid).setCheckState(
                0, QtCore.Qt.Checked if visible else QtCore.Qt.Unchecked)
        assert ws.segmentation.visible is global_visible
        assert [obj.visible for obj in groups] == [True, False]
        worker = ui_app._PipelineTaskWorker(window.pipeline, ws, [StepId.COMPUTE_PLANE_METRICS])
        failures, finished = [], []
        worker.failed.connect(failures.append)
        worker.finished.connect(finished.append)
        worker.run()
        assert not failures, failures
        assert len(finished) == 1 and len(ws.derived.plane_metrics) == 1
        assert (tmp_path / "plane_metrics.json").is_file()
        window._finish_pipeline_scene_refresh([StepId.COMPUTE_PLANE_METRICS])
        app.processEvents()
        for original, visible in zip(groups, (True, False)):
            obj = ws.scene_objects[original.uid]
            assert obj.visible is visible
            assert bool(obj.actor.GetVisibility()) is visible
            assert window._find_browser_item_by_uid(obj.uid).checkState(0) == (
                QtCore.Qt.Checked if visible else QtCore.Qt.Unchecked)
        window._on_segmentation_opacity_changed(42)
        assert [ws.scene_objects[obj.uid].visible for obj in groups] == [True, False]
        window.segmentation_panel.check_visible.setChecked(False)
        window.segmentation_panel.check_visible.setChecked(True)
        assert all(ws.scene_objects[obj.uid].visible for obj in groups)
        window.segmentation_panel.check_visible.setChecked(False)
        assert not any(ws.scene_objects[obj.uid].visible for obj in groups)
    finally:
        window.close()


@pytest.mark.parametrize("mode", ["portable", "checkout", "online"])
def test_docs_action_smoke_uses_local_pages_or_online_fallback(monkeypatch, tmp_path, mode):
    app = _smoke_qt_application(monkeypatch)
    monkeypatch.setenv("AUTOFLOW_SSH_RENDERING", "1")
    from PySide6 import QtCore, QtGui, QtTest
    import autoflow.runtime_paths as runtime_paths
    import autoflow.ui.app as ui_app
    from autoflow.ui.remote_plotter import RemotePlotter
    monkeypatch.setattr(ui_app, "RemotePlotter", RemotePlotter, raising=False)
    repository = tmp_path / "source checkout"
    monkeypatch.setattr(runtime_paths, "__file__", str(repository / "autoflow/runtime_paths.py"))
    monkeypatch.delenv("AUTOFLOW_PORTABLE_ROOT", raising=False)
    expected = None
    if mode in {"checkout", "portable"}:
        expected = repository / "build/docs-offline/index.html"
        expected.parent.mkdir(parents=True)
        expected.write_text("<html><body>Source documentation</body></html>", encoding="utf-8")
        (expected.parent / "offline-manifest.json").write_text('{"offline": true}', encoding="utf-8")
    if mode == "portable":
        portable = tmp_path / "portable app with spaces"
        monkeypatch.setenv("AUTOFLOW_PORTABLE_ROOT", str(portable))
        expected = portable / "docs/site/index.html"
        expected.parent.mkdir(parents=True)
        expected.write_text("<html><body>Packaged documentation</body></html>", encoding="utf-8")
        (expected.parent / "offline-manifest.json").write_text('{"offline": true}', encoding="utf-8")
    if mode == "online":
        ordinary_site = repository / "site/index.html"
        ordinary_site.parent.mkdir(parents=True)
        ordinary_site.write_text("<html>Online-only build</html>", encoding="utf-8")
    opened = []
    monkeypatch.setattr(QtGui.QDesktopServices, "openUrl", lambda url: opened.append(QtCore.QUrl(url)) or True)
    window = ui_app.MainWindow()
    try:
        window.show()
        app.processEvents()
        actions = window.menuBar().actions()
        assert [action.text() for action in actions[:3]] == ["File", "Docs", "Export"]
        assert actions[1] is window.action_docs
        assert window.action_docs.shortcut().toString() == "F1"
        QtTest.QTest.mouseClick(window.menuBar(), QtCore.Qt.LeftButton,
                               pos=window.menuBar().actionGeometry(window.action_docs).center())
        app.processEvents()
        assert len(opened) == 1
        if expected is None:
            assert opened[0].toString() == runtime_paths.DOCUMENTATION_URL
        else:
            assert opened[0].isLocalFile()
            assert Path(opened[0].toLocalFile()) == expected
    finally:
        window.close()



def test_streamline_phantom_background_keeps_geometry_and_rejects_obsolete_input(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtCore, QtWidgets
    from autoflow.algorithms.streamlines import generate_seed_points
    from autoflow.ui.streamline_jobs import streamline_requests

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    ws = Workspace()
    ws.flow_raw = np.zeros((12, 12, 12, 2, 3), dtype=np.float32)
    ws.flow_raw[..., 0] = 30.0
    ws.segmask_binary = np.ones((12, 12, 12, 2), dtype=bool)
    ws.segmask_3d = ws.segmask_binary[..., 0]
    p = ws.streamline_params
    p.max_steps, p.min_seeds, p.seed_ratio = 100, 4, 0.005
    ws.streamline_seeds = generate_seed_points(
        ws.segmask_3d, ws.resolution, ws.origin, ratio=p.seed_ratio,
        rng_seed=p.rng_seed, min_seeds=p.min_seeds)
    ws.streamline_active = True
    manager = streamline_requests(ws)
    assert manager.request(0) is None
    # Replace the input while the first result is still waiting for Qt delivery.
    ws.flow_raw = ws.flow_raw.copy() * 2
    expected = generate_streamlines_at_t(
        ws.flow_raw, 1, ws.streamline_seeds, ws.resolution, ws.origin,
        mask_3d=ws.segmask_binary[..., 1], max_steps=p.max_steps,
        terminal_speed=p.terminal_speed, seed_ratio=p.seed_ratio,
        min_seeds=p.min_seeds, rng_seed=p.rng_seed)
    assert manager.request(1) is None
    loop = QtCore.QEventLoop()
    delivered = []
    def ready(phase):
        delivered.append((phase, QtCore.QThread.currentThread() is app.thread()))
        loop.quit()
    manager.ready.connect(ready)
    timeout = QtCore.QTimer()
    timeout.setSingleShot(True)
    timeout.timeout.connect(loop.quit)
    timeout.start(5000)
    loop.exec()
    timeout.stop()
    manager.ready.disconnect(ready)
    assert delivered == [(1, True)]
    assert 0 not in ws.streamline_cache
    actual = manager.request(1)
    np.testing.assert_array_equal(actual.points, expected.points)
    np.testing.assert_array_equal(actual.lines, expected.lines)
    assert set(actual.point_data) == set(expected.point_data)
    for name in expected.point_data:
        np.testing.assert_array_equal(actual.point_data[name], expected.point_data[name])


def _junction_cluster_phantom():
    from itertools import combinations
    points = np.array([[4, 4, 4], [5, 4, 4], [4, 5, 4], [4, 4, 5],
                       [1, 4, 4], [4, 4, 8]], dtype=float)
    edges = np.array(list(combinations(range(4), 2)) + [(0, 4), (3, 5)], dtype=int)
    return GraphData(points=points, edges=edges), np.ones((12, 12, 12), dtype=np.int16)


def _junction_edge_geometry(graph):
    return {tuple(sorted((tuple(graph.points[a]), tuple(graph.points[b])))) for a, b in graph.edges}


@pytest.mark.parametrize('permuted', [False, True])
def test_junction_cluster_phantom_routes_interfaces_and_preserves_existing_geometry(permuted):
    import networkx as nx
    from autoflow.algorithms.graph import graph_to_networkx
    from autoflow.algorithms.graph_junctions import simplify_junction_clusters
    graph, labels = _junction_cluster_phantom()
    spacing, origin = np.array([1.2, 2.1, .8]), np.array([10., -5., 3.])
    graph.points = graph.points * spacing + origin
    before_points, before_edges = graph.points.copy(), graph.edges.copy()
    expected_endpoints = {tuple(graph.points[4]), tuple(graph.points[5])}
    if permuted:
        order = np.array([5, 1, 3, 4, 0, 2])
        graph = GraphData(points=graph.points[order], edges=np.argsort(order)[graph.edges[::-1, ::-1]])
        before_points, before_edges = graph.points.copy(), graph.edges.copy()
    cleaned, qc = simplify_junction_clusters(graph, labels, spacing, origin=origin)
    assert nx.is_tree(graph_to_networkx(cleaned)) and len(cleaned.points) == 4
    assert _junction_edge_geometry(cleaned).issubset(_junction_edge_geometry(graph))
    assert {tuple(p) for p in cleaned.points}.issubset({tuple(p) for p in graph.points})
    assert expected_endpoints == {tuple(cleaned.points[n]) for n, d in graph_to_networkx(cleaned).degree() if d == 1}
    assert qc['candidate_count'] == qc['simplified_cluster_count'] == 1
    assert qc['cycle_rank_before'] == 3 and qc['cycle_rank_after'] == 0
    assert qc['removed_edge_count'] == 5 and qc['removed_node_count'] == 2
    assert qc['stage'] == 'after_triangle_contraction_before_flow_filter'
    assert all(qc[k] for k in ['original_endpoints_preserved', 'interface_ports_preserved',
        'component_connectivity_preserved', 'label_component_connectivity_preserved'])
    np.testing.assert_array_equal(graph.points, before_points)
    np.testing.assert_array_equal(graph.edges, before_edges)


def test_junction_cluster_phantom_preserves_mixed_label_interfaces_and_cross_label_cycles():
    from itertools import combinations
    import networkx as nx
    from autoflow.algorithms.graph import graph_to_networkx
    from autoflow.algorithms.graph_junctions import simplify_junction_clusters
    points = np.array([[4, 4, 4], [5, 4, 4], [4, 5, 4], [4, 4, 5], [7, 7, 7]], dtype=float)
    graph = GraphData(points=points, edges=np.array(list(combinations(range(5), 2))))
    labels = np.ones((12, 12, 12), dtype=np.int16);labels[7, 7, 7] = 2
    original_interfaces = {tuple(sorted((tuple(points[i]), tuple(points[4])))) for i in range(4)}
    cleaned, qc = simplify_junction_clusters(graph, labels)
    assert original_interfaces.issubset(_junction_edge_geometry(cleaned))
    assert len(cleaned.points) == 5 and len(cleaned.edges) == 7
    assert nx.is_connected(graph_to_networkx(cleaned))
    assert qc['cycle_rank_before'] == 6 and qc['cycle_rank_after'] == 3
    assert qc['protected_edge_count'] == 4 and qc['interface_ports_preserved']
    assert qc['clusters'][0]['source_labels'] == {'1': 4, '2': 1}
    assert qc['clusters'][0]['label_patches'][0]['cycle_rank_after'] == 0


@pytest.mark.parametrize('kind', ['disabled', 'oversized', 'minimum_cycles', 'closed_patch', 'unknown', 'simple_ring'])
def test_junction_cluster_phantom_keeps_ineligible_regions(kind):
    from autoflow.algorithms.graph_junctions import simplify_junction_clusters
    graph, labels = _junction_cluster_phantom();kwargs = {}
    if kind == 'disabled':kwargs['enabled'] = False
    elif kind == 'oversized':
        graph.points[1, 0] = 13;labels = np.ones((20, 20, 20), dtype=np.int16)
    elif kind == 'minimum_cycles':kwargs['min_cycle_rank'] = 4
    elif kind == 'closed_patch':graph = GraphData(points=graph.points[:4], edges=graph.edges[:6])
    elif kind == 'unknown':labels[:] = 0
    elif kind == 'simple_ring':
        graph = GraphData(points=np.array([[4, 4, 4], [5, 4, 4], [5, 5, 4], [4, 5, 4], [2, 4, 4], [7, 5, 4]], dtype=float),
            edges=np.array([[0, 1], [1, 2], [2, 3], [3, 0], [0, 4], [2, 5]]))
    cleaned, qc = simplify_junction_clusters(graph, labels, **kwargs)
    assert cleaned is graph and qc['removed_edge_count'] == 0
    if kind == 'oversized':assert qc['clusters'][0]['reason'] == 'extent_exceeds_limit'
    if kind == 'closed_patch':assert qc['clusters'][0]['label_patches'][0]['reason'] == 'fewer_than_two_preserved_ports'
    if kind == 'simple_ring':assert qc['detected_cluster_count'] == 0


def test_junction_cluster_phantom_pipeline_order_and_audit_persistence(tmp_path, monkeypatch):
    import networkx as nx
    import autoflow.core.pipeline as module
    from autoflow.algorithms.graph import graph_to_networkx
    graph, labels = _junction_cluster_phantom();calls = []
    ws = Workspace();ws.paths.output_dir = str(tmp_path)
    ws.skeleton_params.label_map = {'vessel': 1};ws.skeleton_params.label_groups = {'phantom': {'labels': [1]}}
    ws.skeleton_params.min_edge_points = 0
    ws.segmask_raw = labels.copy();ws.segmask_labels_3d = labels.copy();ws.segmask_3d = labels > 0
    ws.segmask_binary = ws.segmask_3d[..., None]
    ws.flow_raw = np.zeros(labels.shape + (1, 3));ws.flow_raw[..., 2] = 20
    ws.group_order = ['phantom'];ws.skeleton_points = graph.points.copy()
    ws.multilabel_groups = {'phantom': {'labels': [1], 'skeleton_points': graph.points.copy(),
        'segmask_3d': ws.segmask_3d.copy(), 'segmask_binary': ws.segmask_binary.copy()}}
    junction, flow, terminal = module.simplify_junction_clusters, module.remove_intralabel_cycles, module.remove_short_terminal_branches
    def build(*args, **kwargs):calls.append('triangle');return graph
    def route(*args, **kwargs):calls.append('junction');return junction(*args, **kwargs)
    def filter_flow(cleaned, *args, **kwargs):
        calls.append('flow');assert nx.is_tree(graph_to_networkx(cleaned));return flow(cleaned, *args, **kwargs)
    def prune(*args, **kwargs):calls.append('terminal');return terminal(*args, **kwargs)
    monkeypatch.setattr(module, 'build_graph_from_points', build)
    monkeypatch.setattr(module, 'simplify_junction_clusters', route)
    monkeypatch.setattr(module, 'remove_intralabel_cycles', filter_flow)
    monkeypatch.setattr(module, 'remove_short_terminal_branches', prune)
    assert PipelineEngine()._step_generate_graph(ws).success
    assert calls == ['triangle', 'junction', 'flow', 'terminal']
    audit = ws.multilabel_groups['phantom']['junction_cluster_cleanup']
    assert audit['simplified_cluster_count'] == 1
    assert ws.graph_hierarchy['groups'][0]['junction_cluster_cleanup'] == audit
    restored = Workspace();restored.restore_dict(ws.snapshot_dict())
    assert restored.multilabel_groups['phantom']['junction_cluster_cleanup'] == audit
    assert restored.graph_hierarchy['groups'][0]['junction_cluster_cleanup'] == audit
    exported = json.loads((tmp_path / 'graph_hierarchy.json').read_text())
    assert exported['groups'][0]['junction_cluster_cleanup'] == audit


def test_junction_cluster_phantom_config_group_gui_cli_roundtrip():
    from PySide6 import QtWidgets
    from autoflow.cli import build_parser
    from autoflow.ui.action_settings import ActionSettingsDialog, action_sections
    cfg = AutoFlowConfig.from_config_dir()
    assert cfg.junction_cluster_enabled and cfg.junction_cluster_max_extent_voxels == 8 and cfg.junction_cluster_min_cycle_rank == 2
    ws = build_workspace(AutoFlowConfig(junction_cluster_enabled=False, junction_cluster_max_extent_voxels=6, junction_cluster_min_cycle_rank=3))
    ws.skeleton_params.label_groups['custom'] = {'labels': [1], 'preprocess': {'junction_cluster_enabled': True, 'junction_cluster_max_extent_voxels': 4}}
    group = ws.skeleton_params.params_for_group('custom')
    assert group.junction_cluster_enabled and group.junction_cluster_max_extent_voxels == 4 and group.junction_cluster_min_cycle_rank == 3
    restored = Workspace();restored.restore_dict(ws.snapshot_dict())
    assert restored.skeleton_params.to_dict() == ws.skeleton_params.to_dict()
    args = build_parser().parse_args(['case.h5', '--no-junction-cluster-cleanup', '--junction-cluster-max-extent-voxels', '5', '--junction-cluster-min-cycle-rank', '4'])
    assert not args.junction_cluster_enabled and args.junction_cluster_max_extent_voxels == 5 and args.junction_cluster_min_cycle_rank == 4
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    dialog = ActionSettingsDialog('Graph', action_sections(StepId.GENERATE_GRAPH, ws), ws, ws)
    try:
        dialog.controls['skeleton_params.junction_cluster_enabled'].setChecked(True)
        dialog.controls['skeleton_params.junction_cluster_max_extent_voxels'].setValue(7)
        dialog.controls['skeleton_params.junction_cluster_min_cycle_rank'].setValue(2)
        values = dialog.values()
        assert values['skeleton_params.junction_cluster_enabled'] is True
        assert values['skeleton_params.junction_cluster_max_extent_voxels'] == 7
        assert values['skeleton_params.junction_cluster_min_cycle_rank'] == 2
    finally:
        dialog.close();dialog.deleteLater();app.processEvents()


@pytest.mark.parametrize(('name', 'value'), [('junction_cluster_enabled', 1),
    ('junction_cluster_max_extent_voxels', .5), ('junction_cluster_max_extent_voxels', np.inf),
    ('junction_cluster_max_extent_voxels', True), ('junction_cluster_min_cycle_rank', 0),
    ('junction_cluster_min_cycle_rank', 1.5), ('junction_cluster_min_cycle_rank', True)])
def test_junction_cluster_phantom_rejects_invalid_parameters(name, value):
    with pytest.raises(ValueError, match=name):SkeletonParams.from_dict({name: value})
    with pytest.raises(ValueError, match=name):build_workspace(AutoFlowConfig(**{name: value}))
