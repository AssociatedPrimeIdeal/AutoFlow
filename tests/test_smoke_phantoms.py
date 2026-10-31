import json
import sys
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
from autoflow.algorithms.regression_phantoms import REGRESSION_CASES


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
                 "_browser_subgroup_nodes", "_browser_apply_path_hierarchy",
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


def test_graph_hierarchy_phantom_native_groups_and_legacy_snapshot(tmp_path):
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
    assert "joint_systems" not in hierarchy
    assert not hasattr(ws, "willis_ring_graph")
    for report in hierarchy["groups"]:
        by_index = {record["path_index"]: record for record in report["paths"]}
        for subgroup in report["subgroups"]:
            keys = [(by_index[index]["subgroup_level"] is None, by_index[index]["subgroup_level"] or 0)
                    for index in subgroup["path_indices"]]
            assert keys == sorted(keys)
    assert json.loads((tmp_path / "graph_hierarchy.json").read_text())["schema"] == hierarchy["schema"]
    assert all("hierarchy" in info for info in ws.path_info)
    assert not any(obj.data_key.startswith("hierarchy_path_") or obj.data_key == "willis_ring_graph" for obj in ws.scene_objects.values())
    restored = Workspace()
    saved = ws.snapshot_dict()
    assert "willis_ring_graph" not in saved and "willis_ring_status" not in saved
    saved["graph_hierarchy"]["joint_systems"] = [{"group_name": "cerebral_arterial_system"}]
    saved["willis_ring_status"] = {"status": "detected"}
    saved["willis_ring_graph"] = {"points": [[0, 0, 0]], "edges": []}
    for key in ("willis_ring_graph", "hierarchy_path_cerebral_arterial_system_0"):
        saved["scene_objects"].append(dict(saved["scene_objects"][0], uid=key, data_key=key))
    restored.restore_dict(saved)
    assert not hasattr(restored, "willis_ring_graph")
    assert not any(o.data_key == "willis_ring_graph" or o.data_key.startswith("hierarchy_path_") for o in restored.scene_objects.values())
    assert restored.graph_hierarchy == hierarchy
    assert build_quality_report(ws)["graph_hierarchy"] == hierarchy
    import pyvista as pv
    from autoflow.ui.viewer import SceneController
    plotter = pv.Plotter(off_screen=True)
    try:
        scene = SceneController(plotter, ws, lambda *_: None)
        native = scene._build_dataset("smooth_path_intracranial_anterior_arteries_0")
        assert native.n_points > 0
        assert scene._build_dataset("hierarchy_path_cerebral_arterial_system_0") is None
        assert scene._build_dataset("willis_ring_graph") is None
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
    assert qc['protected_edge_count'] == 0 and qc['stage'] == 'before_terminal_pruning'
    assert qc['cross_label_candidate_edge_count'] == 4 and qc['policy'] == 'all_positive_labels'
    assert qc['component_connectivity_preserved'] and qc['label_presence_preserved']
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
        assert qc['rejected_paths'][0]['reason'] == 'preserve_component_or_endpoint'


def test_intralabel_cycle_phantom_does_not_cut_edge_crossing_another_label():
    from autoflow.algorithms.graph_cycles import remove_intralabel_cycles
    graph = GraphData(points=np.array([[2, 2, 2], [8, 2, 2], [5, 8, 2]], dtype=float),
        edges=np.array([[0, 1], [1, 2], [2, 0]]))
    labels = np.ones((11, 11, 5), dtype=np.int16)
    labels[5, 2, 2] = 2  # Endpoints alone would miss this interface.
    cleaned, qc = remove_intralabel_cycles(graph, labels, flow=_uniform_cycle_flow(labels))
    assert cleaned is graph and qc['protected_edge_count'] == 0
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
    _seed_manual_1mm_phantom_state(ws)
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
    assert SkeletonParams.from_dict({}).min_edge_points == 9
    assert SkeletonParams.from_dict({"min_edge_points": 0}).min_edge_points == 0
    assert SkeletonParams.from_dict({"min_edge_count": 4}).min_edge_points == 4

    # The shared default removes an eight-edge terminal arm but keeps the
    # nine-edge arm. Explicit smaller values retain their existing semantics.
    points = [[0., 0., 0.]]
    edges = []
    for direction, length in [((1, 0, 0), 9), ((0, 1, 0), 8), ((0, 0, 1), 3)]:
        previous = 0
        for step in range(1, length + 1):
            index = len(points)
            points.append((np.asarray(direction) * step).tolist())
            edges.append([previous, index])
            previous = index
    boundary_graph = GraphData(points=np.asarray(points), edges=np.asarray(edges))
    default_filtered = remove_short_terminal_branches(boundary_graph)
    assert len(default_filtered.edges) == 9
    np.testing.assert_array_equal(default_filtered.points, boundary_graph.points[:10])
    assert len(remove_short_terminal_branches(boundary_graph, min_edge_points=8).edges) == 17
    np.testing.assert_array_equal(remove_short_terminal_branches(boundary_graph, min_edge_points=0).edges, boundary_graph.edges)
    np.testing.assert_array_equal(boundary_graph.points, points)
    assert AutoFlowConfig().min_edge_points == 9
    assert AutoFlowConfig.from_config_dir().min_edge_points == 9
    assert SkeletonParams().params_for_group("aorta_systemic_branches").min_edge_points == 9
    from autoflow.config import DEFAULT_CONFIG_BUNDLE
    assert bundle_to_autoflow_kwargs(DEFAULT_CONFIG_BUNDLE)["min_edge_points"] == 9
    from autoflow.ui.parameter_help import parameter_help
    assert "Default 9" in parameter_help("skeleton_params.min_edge_points", "Minimum terminal edge count")



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


def test_label_plane_phantom_unified_defaults_and_qc_omissions():
    from autoflow.algorithms.label_planes import generate_label_planes
    from autoflow.plane_rules import default_label_plane_rules

    rules = default_label_plane_rules()
    counts = dict.fromkeys(rules, 0)
    counts.update(AAO=3, DAO=3, MPA=1, BA=1, LIJV=2, RIJV=2, LICA=2, RICA=2, LVA=2, RVA=2, PV=3, SMV=1, SV=1)
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
    assert len(planes) == sum(counts.values()) == 25
    for name, count in counts.items():
        selected = [p for p in planes if p.segmentation_label == label_map[name]]
        expected = ([] if not count else [.2, .5, .8] if name in {"AAO", "PV"}
                    else [.1, .3, .5] if name == "DAO"
                    else [.1] if name in {"LTS", "RTS"} else [.9] if name in {"SSS", "StrS"}
                    else [.7] if name == "MPA" else [.5] if name in {"LPA", "RPA"}
                    else [1 / 3, 2 / 3] if name in {"LIJV", "RIJV"}
                    else [.3, .5] if name in {"LVA", "RVA"} else [.33, .66] if name in {"LICA", "RICA"} else [.5])
        assert len(selected) == len(expected) == count
        assert [p.distance / 30 for p in selected] == pytest.approx(expected)
        assert all(np.allclose(p.normal, [1, 0, 0]) for p in selected)
    assert rules["LPCA"]["plane_count"] == rules["RPCA"]["plane_count"] == 1
    assert rules["LICA"]["plane_count"] == rules["RICA"]["plane_count"] == 2
    assert rules["ACA"]["plane_count"] == 1
    assert all(rules[name]["plane_count"] == 0 for name in ("ARCH", "HA", "SMA"))
    assert len(smooth) == len(paths)
    assert all(q["status"] == ("disabled" if rules[q["label_name"]]["plane_count"] == 0 else "selected" if counts[q["label_name"]] else "error") for q in qc)
    assert all("main_path" not in rule and rule["plane_mode"] == "landmark" for rule in rules.values())
    config = AutoFlowConfig.from_config_dir()
    assert config.plane_mode == "by_label" and config.cc_filter_mode == "hybrid"
    assert config.cc_rel_min_ratio == .10 and config.min_cc_volume == 50.
    assert build_workspace(config).plane_gen_params.label_rules == default_label_plane_rules()
    from autoflow.plane_rules import load_plane_rules_file
    plan_dir = DATA_DIR.parent / "configs" / "plane_plans"
    assert {p.name for p in plan_dir.glob('*.json')} == {'default.json'}
    assert load_plane_rules_file(plan_dir/'default.json') == rules
    assert sum(rule['plane_count'] for rule in rules.values()) == 45
    for name in ('LICA','RICA','LVA','RVA'):
        spec=rules[name]['placements'][0]
        assert spec['anchor']==('flow_inlet_branch' if name in ('LICA','RICA') else 'label_branch') and 'level' not in spec
        assert spec['scope']=='label'
        assert spec['path_selection']==('unique' if name in ('LICA','RICA') else 'longest')
        assert spec['fractions']==([.33,.66] if name in ('LICA','RICA') else [.3,.5])
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
    for info in infos:info["hierarchy"]={"flow_sign":1}
    rules = {name: default_label_plane_rules()[name] for name in mapping}
    planes, _, _, qc = generate_label_planes(paths, volume, mapping, rules,
        path_info=infos, smoothing_window=1, inter_time=1)
    renal = [plane for plane in planes if plane.segmentation_label == 12]
    aorta = [plane for plane in planes if plane.segmentation_label == 4]
    assert {plane.path_index for plane in renal} == {1, 2} and len(renal) == 6
    for index in (1,2):
        assert sorted(p.placement["fraction"] for p in renal if p.path_index==index)==pytest.approx([.1,.3,.5])
    assert len(aorta) == 3 and {plane.path_index for plane in aorta} == {0}
    assert sorted(plane.placement["fraction"] for plane in aorta) == [.1, .3, .5]
    assert all(plane.placement["selection_source"] == "single_label_branch" for plane in renal)
    assert len({plane.placement["connection_id"] for plane in renal}) == 1
    assert all(p.placement["anchor"]=="label_branch" for p in aorta)
    assert {row["selection_source"] for row in qc} == {"single_label_branches"}


@pytest.mark.parametrize("name", ["PV", "SMV", "SV", "RBCT", "CCA", "LBCT"])
@pytest.mark.parametrize("length", [6, 12, 40])
def test_connection_default_phantom_positions_stay_inside_short_label(name, length):
    from autoflow.algorithms.label_planes import generate_label_planes
    from autoflow.plane_rules import default_label_plane_rules
    path = np.column_stack((np.arange(5., 56.), np.full(51, 20.), np.full(51, 10.)))
    # A supported two-label origin is sufficient; an unrelated neighbour is rejected.
    other = "SMV" if name=="PV" else "PV" if name in {"SMV","SV"} else "DAO"
    values = np.where(np.arange(51) <= length, 101, 102)
    volume, infos = _landmark_phantom([path], [values], [(0, 1)], {name: 101, other: 102})
    rule = default_label_plane_rules()[name]
    if name == 'PV': rule['placements'][0].pop('connection_fallback', None)
    planes, _, _, qc = generate_label_planes([path], volume, {name: 101, other: 102},
        {name: rule}, path_info=infos, smoothing_window=1, inter_time=1)
    if name == "PV":
        # A PV branch now requires the actual SMV/SV/PV confluence.
        assert not planes and qc[0]["selection_warnings"] == ["portal_confluence_not_found"]
        return
    distance = 10. if name in {"RBCT", "CCA", "LBCT"} else 15.
    if length < distance:
        assert not planes and qc[0]["omitted_positions"][0]["reason"] == "offset_exceeds_interval"
        return
    assert len(planes) == 1
    placement = planes[0].placement
    assert placement["anchor"] == "label_reference_branch" and placement["reference_labels"]
    assert placement["nominal_distance_in_interval_mm"] == pytest.approx(distance)
    assert placement["requested_offset_mm"] == pytest.approx(distance)
    assert not placement["short_interval_midpoint"]
    assert planes[0].center[0] == pytest.approx(5 + length - distance)


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
    rule = {"plane_mode":"landmark","plane_count":1,"placements":[{"anchor":"label_level","level":1,
        "path_selection":"longest","position":"near","offset_mm":15.}]}
    planes, _, _, qc = generate_label_planes(paths, volume, {name: 23}, {name: rule},
        path_info=infos, smoothing_window=1, inter_time=1)
    if length < 15:
        assert not planes and qc[0]["omitted_positions"][0]["reason"] == "offset_exceeds_interval"
        return
    assert len(planes) == 1 and planes[0].path_index == 0
    placement = planes[0].placement
    assert placement["anchor"] == "label_level"
    assert placement["label_level"] == 1 and placement["path_selection"] == "longest"
    distance = 15.
    assert placement["nominal_distance_in_interval_mm"] == pytest.approx(distance)
    assert not placement["short_interval_midpoint"]
    assert planes[0].center[0] == pytest.approx(5. + (distance if flow_sign == 1 else length - distance))
    assert planes[0].normal == pytest.approx([flow_sign, 0., 0.])
    assert not qc[0]["selection_warnings"]
    # Without L1, report the unavailable level rather than choosing L2.
    infos[0]["hierarchy"]["label_local_levels"]["23"] = 1
    omitted, _, _, qc = generate_label_planes(paths, volume, {name: 23}, {name: rule},
        path_info=infos, smoothing_window=1, inter_time=1)
    assert not omitted and qc[0]["selection_warnings"] == ["label_level_unavailable"]


@pytest.mark.parametrize("name", ["StrS","LTS","RTS","SSS","MPA","LPA","RPA"])
@pytest.mark.parametrize("length,sign", [(12,1),(24,1),(60,1),(60,-1)])
def test_label_offset_phantom_uses_requested_level_and_flow_reference(name,length,sign):
    from autoflow.algorithms.label_planes import generate_label_planes
    from autoflow.plane_rules import default_label_plane_rules
    paths=[np.column_stack((np.arange(2.,63.),np.full(61,2.),np.full(61,10.))),
           np.column_stack((np.full(length+1,62.),np.arange(2.,length+3.),np.full(length+1,10.))),
           np.column_stack((np.arange(62.,55.,-1),np.full(7,2.),np.full(7,10.)))]
    # Avoid overlapping the root with the shorter daughter in the sampled volume.
    paths[2][:,1]=np.linspace(2,6,7)
    hierarchy=[{"label_local_levels":{"23":level},"flow_sign":sign} for level in (0,2,2)]
    volume,infos=_landmark_phantom(paths,[np.full(len(p),23) for p in paths],[(0,1),(1,2),(1,3)],{name:23},hierarchy)
    # A connection is absent; PV/SMV/SV must use only their declared fallback.
    planes,_,_,qc=generate_label_planes(paths,volume,{name:23},{name:_legacy_flow_reference_rule(name)},
        path_info=infos,smoothing_window=1,inter_time=1)
    chosen=0 if name in {"LTS","RTS","LPA","RPA"} else 1
    if name == "MPA" and length<15:
        assert not planes and qc[0]["omitted_positions"][0]["reason"]=="offset_exceeds_interval"
        return
    assert len(planes)==(3 if name=="PV" else 1) and all(p.path_index==chosen for p in planes)
    if name=="PV":
        assert [p.placement["fraction"] for p in planes]==pytest.approx([.2,.5,.8])
        assert [p.center[1] for p in planes]==pytest.approx([2+length*(f if sign==1 else 1-f) for f in (.2,.5,.8)])
        assert all(p.placement["connection_fallback"] and p.placement["label_level"]==3 for p in planes)
        return
    placement=planes[0].placement
    selected_length=60 if chosen==0 else length
    from_start=name in {"LTS","RTS","LPA","RPA"}
    fractional=name in {"SSS","StrS","LTS","RTS"}
    distance=selected_length*(.1 if from_start else .9) if fractional else 15 if from_start else selected_length-15
    if fractional:
        assert placement["position"]=="fractions" and placement["fraction"]==pytest.approx(.1)
        assert placement["fraction_reference"]==("flow_start" if from_start else "flow_end")
        assert "requested_offset_mm" not in placement
    else:
        assert placement["position"]==("after_start" if from_start else "before_end")
        assert placement["offset_reference"]==("flow_start" if from_start else "flow_end")
        assert placement["requested_offset_mm"]==15
    assert placement["label_level"]==(1 if from_start else 3)
    assert placement["nominal_distance_in_interval_mm"]==pytest.approx(distance)
    axis=0 if chosen==0 else 1
    assert planes[0].center[axis]==pytest.approx(2+(distance if sign==1 else selected_length-distance))
    infos[chosen]["hierarchy"]["flow_sign"]=None
    omitted,_,_,qc=generate_label_planes(paths,volume,{name:23},{name:_legacy_flow_reference_rule(name)},
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
    for info in infos:info["hierarchy"]={"flow_sign":1}
    planes,_,_,_=generate_label_planes(paths,volume,{"DAO":4},{"DAO":default_label_plane_rules()["DAO"]},
        path_info=infos,smoothing_window=1,inter_time=1)
    assert len(planes)==3 and {p.path_index for p in planes}=={1}
    assert [p.center[0] for p in planes]==pytest.approx([10,20,30])
    assert all(p.placement["selection_scope"]=="label" for p in planes)


@pytest.mark.parametrize("name",["LIJV","RIJV"])
def test_jugular_phantom_highest_level_wins_over_longer_low_level(name):
    from autoflow.algorithms.label_planes import generate_label_planes
    from autoflow.plane_rules import default_label_plane_rules
    paths=[np.column_stack((np.arange(5.,56.),np.full(51,10.),np.full(51,10.))),
           np.column_stack((np.arange(5.,21.),np.full(16,30.),np.full(16,10.)))]
    hierarchy=[{"label_local_levels":{"32":level},"flow_sign":1} for level in (0,3)]
    volume,infos=_landmark_phantom(paths,[np.full(len(p),32) for p in paths],[(0,1),(2,3)],{name:32},hierarchy)
    planes,_,_,_=generate_label_planes(paths,volume,{name:32},{name:{"plane_mode":"landmark","plane_count":2,"placements":[{"anchor":"label_terminal","scope":"label","path_selection":"longest","position":"fractions","fractions":[1/3,2/3]}]}},
        path_info=infos,smoothing_window=1,inter_time=1)
    assert len(planes)==2 and {p.path_index for p in planes}=={1}
    assert all(p.placement["label_level"]==4 for p in planes)
    assert [p.center[0] for p in planes]==pytest.approx([10,15])


def test_connection_phantom_background_gap_errors_without_an_alternate():
    from autoflow.algorithms.label_planes import generate_label_planes
    from autoflow.plane_rules import default_label_plane_rules, normalize_label_plane_rules
    path = np.column_stack((np.arange(5., 56.), np.full(51, 20.), np.full(51, 10.)))
    values = np.where(np.arange(51) < 20, 12, np.where(np.arange(51) == 20, 0, 4))
    volume, infos = _landmark_phantom([path], [values], [(0, 1)], {"LRA": 12, "DAO": 4})
    rule = default_label_plane_rules()["LRA"]
    planes, _, _, qc = generate_label_planes([path], volume, {"LRA":12,"DAO":4}, {"LRA":rule},
        path_info=infos, smoothing_window=1, inter_time=1)
    assert not planes and qc[0]["status"] == "error"
    assert qc[0]["selection_warnings"] == ["label_reference_not_found"]
    rule["placements"][0]["fallback"] = {"anchor":"label_longest","position":"fractions","fractions":[.5]}
    with pytest.raises(ValueError, match="no longer supported"):
        normalize_label_plane_rules({"LRA":rule})


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
    assert len(planes) == 1 and planes[0].path_index == 1
    assert planes[0].placement["label_level"] is None
    assert planes[0].center[0] == pytest.approx(35.0)
    assert planes[0].placement["route_path_indices"] == [1]
    assert planes[0].placement["fraction"] == .5


def test_connection_default_gui_phantom_preserves_strict_levels(tmp_path):
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
    original["placements"]=[dict(_legacy_flow_reference_rule("MPA")["placements"][0],scope="label")]
    dialog = LandmarkPlanDialog("MPA", original)
    try:
        table = dialog.table
        assert table.rowCount() == 1
        anchor = table.cellWidget(0, 0)
        assert anchor.currentData() == "label_terminal"
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
                                                "position": "fractions", "fractions": [.2, .5, .8], "scope": "label"}]
        assert original["placements"][0]["anchor"] == "label_terminal"
        assert not dialog.findChildren(QtWidgets.QPlainTextEdit)
    finally:
        dialog.close()
        dialog.deleteLater()
        app.processEvents()


def test_landmark_unified_defaults_gui_phantom_roundtrips(tmp_path):
    from PySide6 import QtWidgets
    from autoflow.ui.action_settings import ActionSettingsDialog, action_sections
    from autoflow.plane_rules import default_label_plane_rules, save_plane_rules_file, load_plane_rules_file
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    ws = Workspace()
    ws.plane_gen_params.label_rules = {'MPA':default_label_plane_rules()['MPA']}
    dialog = ActionSettingsDialog("Planes", action_sections(StepId.GENERATE_PLANES, ws), ws, ws)
    try:
        assert dialog.findChild(QtWidgets.QComboBox, "plane_plan_preset") is None
        defaults=dialog.findChild(QtWidgets.QPushButton,"plane_plan_defaults")
        assert defaults is not None
        defaults.click()
        table = dialog.controls["plane_gen_params.label_rules"]
        assert table.isColumnHidden(2) and table.isColumnHidden(3)
        rules = dialog.values()["plane_gen_params.label_rules"]
        assert rules==default_label_plane_rules()
        for name in ('LICA','RICA','LVA','RVA'):
            row=next(i for i in range(table.rowCount()) if table.item(i,0).text()==name)
            assert ('Flow-entry branch' if name in ('LICA','RICA') else 'Label branch') in table.item(row,6).text() and 'L1' not in table.item(row,6).text()
            assert rules[name]['plane_count']==2
        filename = tmp_path / "planes.json"
        save_plane_rules_file(filename, rules)
        assert load_plane_rules_file(filename) == rules
    finally:
        dialog.close();dialog.deleteLater();app.processEvents()


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
    assert imported[0].placement == {**ws.planes[0].placement, "geometry_source": "imported", "manual_geometry": True}


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


@pytest.mark.parametrize("failure", ["fov_boundary_contact", "no_branch_support"])
def test_landmark_validation_phantom_omits_instead_of_moving(monkeypatch, failure):
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
        return ([plane] if valid else []), [{"valid": valid, "reason": failure}]
    monkeypatch.setattr(sampling, "filter_planes_by_branch_support", support)
    rule = default_label_plane_rules()["AAO"]
    rule["plane_count"] = 1
    rule["placements"] = [{"anchor": "label_level", "level": 1, "position": "fractions", "fractions": [.2]}]
    planes, _, _, qc = generate_label_planes([path], volume, mapping, {"AAO": rule}, path_info=infos,
        smoothing_window=1, inter_time=10, segmask_binary_4d=(volume > 0)[..., None],
        filter_defaults={"fov_boundary_max_fraction": .2})
    assert not planes
    assert len(inspected) == 1 and inspected[0][0] == pytest.approx(11.)
    assert qc[0]["requested_plane_count"] == 1
    assert qc[0]["omitted_positions"][0]["reason"] == failure


@pytest.mark.parametrize("placement", [
    {"anchor": "missing", "position": "near"},
    {"anchor": "label_connections", "position": "fractions", "fractions": [.1], "fraction_reference": "flow_end"},
    {"anchor": "label_level", "position": "near", "fraction_reference": "flow_end"},
    {"anchor": "label_level", "position": "fractions", "fractions": [.1], "fraction_reference": "unknown"},
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
def test_label_plane_branch_phantom_unreliable_flow_reports_failure(flow_available):
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
    assert not planes and qc[0]["status"] == "selection_failed"
    assert qc[0]["reason"] == "legacy_fallback_disabled"
    assert qc[0]["selection_warnings"] == ["flow_direction_unreliable" if flow_available else "flow_unavailable"]
    assert qc[0]["omitted_positions"]


def test_label_plane_cycle_phantom_reports_root_failure():
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
    assert not planes and qc[0]["status"] == "selection_failed"
    assert qc[0]["selection_warnings"] == ["directed_cycle_no_root"]


def test_label_plane_curved_phantom_smoothing_cannot_create_extra_roots():
    from scipy.ndimage import binary_dilation
    from autoflow.algorithms.label_planes import generate_label_planes, _matched_label_plane_candidates
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
    placement, _, topology, _ = _matched_label_plane_candidates(
        [path], labels, {"RRA": 1}, smoothing_window=151, inter_time=5)
    np.testing.assert_array_equal(topology[0], dense)
    assert len(placement[0]) == len(topology[0])
    assert not np.shares_memory(placement[0], topology[0])
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
    _seed_manual_1mm_phantom_state(ws)
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
    _seed_manual_1mm_phantom_state(ws)
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
    _seed_manual_1mm_phantom_state(ws)
    engine = PipelineEngine()
    result = engine._step_generate_planes(ws)
    assert result.success and len(ws.planes) == 1, result.message
    assert ws.pipeline.completed[StepId.GENERATE_PLANES.value]
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
    placement=next(item for item in report['checks'] if item['id']=='planes.placement_rules')
    assert placement['status']=='warn' and placement['threshold_status']=='fail' and 'AAO' in placement['summary']
    detail=next(item for item in placement['details'] if item['status']=='warn')
    assert detail['value']['label_name']=='AAO'
    assert detail['value']['reasons']=={'fov_boundary_contact':1}
    assert ws.derived.plane_qc['plane_layout']['failure_policy']=='qc_error'
    # Repeated measurement refreshes retain the original per-label QC findings.
    engine._compute_plane_metrics_internal(ws, save=False, include_derived=False)
    assert next(c for c in build_quality_report(ws)['checks']
                if c['id']=='planes.placement_rules')['status']=='warn'

    check = next(item for item in report["checks"] if item["id"] == "planes.exclusions")
    assert check["status"] == "warn"
    assert "FOV=8.000/32.000 mm (25.0%)" in check["details"][0]["details"][0]["summary"]
    ws.plane_gen_params.label_rules = {"AAO": {"plane_count": 1, "fov_boundary_filter": False}, "BA": 1}
    assert engine._step_generate_planes(ws).success and len(ws.planes) == 2
    records = ws.derived.plane_qc["planes"]
    assert [record["plane_index_source"] for record in records] == [0, 1]
    assert records[0]["main_path_fraction"] == .5
    assert records[0]["fov_boundary"]["status"] == "disabled"

    # A missing map is a QC Warning with zero planes, not a batch-stopping error.
    ws.skeleton_params.label_map = {}
    result = engine._step_generate_planes(ws)
    assert result.success and not ws.planes
    assert ws.pipeline.completed[StepId.GENERATE_PLANES.value]
    check=next(c for c in build_quality_report(ws)['checks'] if c['id']=='planes.placement_rules')
    assert check['status']=='warn'
    assert check['details'][0]['value']['reasons']=={'label_map_unavailable':1}


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
            method = child.controls["plane_gen_params.smoothing_method"]
            method.setCurrentIndex(method.findData("savgol"))
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
    import copy
    from autoflow.config import DEFAULT_RENDER_STYLE_CFG
    ws.render_settings["render_style_cfg"] = copy.deepcopy(DEFAULT_RENDER_STYLE_CFG)
    ws.render_settings["render_style_cfg"]["metrics"]["tke"]["mode"] = "volume"
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


@pytest.mark.skipif(sys.platform == "win32", reason="Linux SSH renderer; native Windows GUI is verified by portable_check.py --gui")
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


@pytest.mark.skipif(sys.platform == "win32", reason="Linux SSH renderer; native Windows GUI is verified by portable_check.py --gui")
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


@pytest.mark.skipif(sys.platform == "win32", reason="Linux SSH renderer; native Windows GUI is verified by portable_check.py --gui")
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


@pytest.mark.skipif(sys.platform == "win32", reason="Linux SSH renderer; native Windows GUI is verified by portable_check.py --gui")
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


@pytest.mark.skipif(sys.platform == "win32", reason="Linux SSH renderer; native Windows GUI is verified by portable_check.py --gui")
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
    assert summary["stage_times_sec"].keys() == {"load", "analysis_report"}
    assert summary["quality_report"]["overall_status"] == "incomplete"
    assert summary["quality_report_file"] == str(out_dir / "quality_report.json")
    assert (out_dir / "summary.json").is_file()
    assert (out_dir / "quality_report.json").is_file()
    assert (out_dir / "analysis_report.html").is_file()
    assert summary["analysis_report_file"] == str(out_dir / "analysis_report.html")
    assert summary["analysis_report_status"] == "complete"
    assert summary["analysis_report_snapshot_id"]
    assert summary["analysis_report_error"] == ""
    assert summary["analysis_report_time_sec"] >= 0.0
    assert summary["outputs"]["analysis_report"] == "complete"
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
    helper_script = tmp_path / "implicit_helper.sh"
    helper_script.write_text('DATASET="Dataset7020_Aorta_4DTemporalFT"\n')
    monkeypatch.setattr("autoflow.algorithms.segmentation.nnunet_temporal._NNUNET_4D_PIPELINE_DEFAULT", helper_script)
    def reject_implicit_python_switch(_script):
        pytest.fail("An implicit helper must not select a training-only interpreter")
    monkeypatch.setattr("autoflow.algorithms.segmentation.nnunet_temporal._python_from_4d_pipeline_script", reject_implicit_python_switch)
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
        import sys
        assert command[0] == sys.executable
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
    ws = Workspace()
    ws.segmask_raw = np.zeros((16, 16, 16), dtype=np.int16)
    ws.resolution = np.ones(3, dtype=float)
    ws.skeleton_params = SkeletonParams(remove_small_cc=True, min_cc_volume_mm3=10.0,
        cc_filter_mode="hybrid", cc_rel_min_ratio=0.01,
        label_groups={"vessel": {"labels": [1]}}, single_label_group_name="vessel")
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


def test_quality_report_collects_actionable_pipeline_checks(tmp_path, monkeypatch):
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

    # Numerical QC thresholds need review; actual failed steps alone are errors.
    from autoflow.core.pipeline import StepResult
    from autoflow.report_html import _status
    workspace.derived.plane_qc["path_ic"]["0"] = .1
    warning = build_quality_report(workspace)
    flow_check = next(c for c in warning["checks"] if c["id"] == "hemodynamics.flow_consistency")
    assert flow_check["status"] == "warn" and flow_check["threshold_status"] == "fail"
    assert warning["overall_status"] == "needs_review" and warning["status_counts"]["fail"] == 0
    assert _status({"status": "fail"}) == "warning"
    engine = PipelineEngine()
    monkeypatch.setattr(engine, "_step_remove_noise", lambda ws: StepResult(StepId.REMOVE_NOISE, False, True, "Solver failed"))
    assert not engine.run_step(workspace, StepId.REMOVE_NOISE, lambda _: None).success
    failed = build_quality_report(workspace)
    execution = next(c for c in failed["checks"] if c["id"] == "execution."+StepId.REMOVE_NOISE.value)
    assert execution["status"] == "fail" and execution["failure_kind"] == "execution"
    assert execution["value"]["output_status"] == "skipped" and _status(execution) == "error"
    assert failed["overall_status"] == "not_ready" and failed["status_counts"]["fail"] == 1
    monkeypatch.setattr(engine, "_step_remove_noise", lambda ws: StepResult(StepId.REMOVE_NOISE, True, True, "Optional step skipped"))
    assert engine.run_step(workspace, StepId.REMOVE_NOISE, lambda _: None).success
    assert not workspace.pipeline.errors
    assert build_quality_report(workspace)["overall_status"] == "needs_review"

    def execution_exception(ws):
        raise RuntimeError("processing exception")
    monkeypatch.setattr(engine, "_step_remove_noise", execution_exception)
    with pytest.raises(RuntimeError, match="processing exception"):
        engine.run_step(workspace, StepId.REMOVE_NOISE, lambda _: None)
    assert workspace.pipeline.errors[StepId.REMOVE_NOISE.value]["error_type"] == "RuntimeError"
    from autoflow.task_control import TaskCancelled
    workspace.pipeline.errors.clear()
    def cancelled_step(ws):
        raise TaskCancelled("cancelled")
    monkeypatch.setattr(engine, "_step_remove_noise", cancelled_step)
    with pytest.raises(TaskCancelled):
        engine.run_step(workspace, StepId.REMOVE_NOISE, lambda _: None)
    assert not workspace.pipeline.errors


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

    workspace.planes = [PlaneData(center=np.array([m["distance"], 0., 0.]), normal=np.array([1., 0., 0.]),
                                  path_index=m["path_index"], segmentation_label=m["segmentation_label"],
                                  distance=m["distance"], metrics=dict(m)) for m in metrics]
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
    assert segmentation_module._NNUNET_4D_MODEL_DEFAULT == (
        Path(segmentation_module.__file__).resolve().parents[2]
        / "segmodel" / "nnunet4d_20261029_f117a02617a2"
    )
    from autoflow.algorithms.segmentation.models import (
        _model_folder_from_4d_pipeline_script, _nnunet_4d_pipeline_assignments,
    )
    script = tmp_path / "run_7020_noCC.sh"
    script.write_text(
        'SSD_ROOT="${SSD_ROOT:-/var/tmp/temporal}"\n'
        'NNUNET_PREPROCESSED="${NNUNET_PREPROCESSED:-${SSD_ROOT}/nnpreprocessed_noCC}"\n'
        'NNUNET_RESULTS="${NNUNET_RESULTS:-/nas-data2/models/nnres_noCC}"\n'
        'DATASET="${DATASET:-Dataset7020_Aorta_4DTemporalFT}"\n'
    )
    assignments = _nnunet_4d_pipeline_assignments(script)
    assert assignments["NNUNET_PREPROCESSED"] == "/var/tmp/temporal/nnpreprocessed_noCC"
    assert _model_folder_from_4d_pipeline_script(script) == (
        Path("/nas-data2/models/nnres_noCC/Dataset7020_Aorta_4DTemporalFT")
        / "nnUNetTrainerPartBalancedTversky__nnUNetPlansIso1mm__3d_fullres"
    ).resolve()


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


@pytest.mark.parametrize("display,preload,override,expected_remote", [
    (":1.0", "libdlfaker.so:libvglfaker.so", "", False),
    ("unix:1", "/usr/lib/libvglfaker.so", "vtkXOpenGLRenderWindow", False),
    (":1", "", "", True),
    ("localhost:10.0", "libvglfaker.so", "", True),
    (":1", "libvglfaker.so", "vtkOSOpenGLRenderWindow", True),
    (":1", "libvglfaker.so", "vtkEGLRenderWindow", True),
])
def test_gui_render_startup_smoke_local_virtualgl_keeps_native_widget(
        monkeypatch, display, preload, override, expected_remote):
    from autoflow.ui import launcher
    monkeypatch.setattr(launcher.sys, "platform", "linux")
    monkeypatch.setenv("QT_QPA_PLATFORM", "xcb")
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.delenv("AUTOFLOW_SSH_RENDERING", raising=False)
    monkeypatch.setenv("DISPLAY", display)
    monkeypatch.setenv("SSH_CONNECTION", "inherited from desktop startup")
    monkeypatch.setenv("LD_PRELOAD", preload)
    monkeypatch.setenv("VTK_DEFAULT_OPENGL_WINDOW", override)
    configured = []
    monkeypatch.setattr(launcher, "_display_is_reachable", lambda _: True)
    def configure():
        configured.append(True)
        monkeypatch.setenv("AUTOFLOW_SSH_RENDERING", "1")
    monkeypatch.setattr(launcher, "_configure_ssh_rendering", configure)
    launcher.validate_display()
    assert bool(configured) is expected_remote
    assert (launcher.os.environ.get("AUTOFLOW_SSH_RENDERING") == "1") is expected_remote


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


def test_magnitude_volume_phantom_reuses_ranges_and_invalidates_changed_scalars(monkeypatch):
    from contextlib import closing
    import pyvista as pv
    from autoflow.ui.viewer import SceneController
    ws = Workspace()
    source = np.arange(1, 211, dtype=np.float32).reshape(5, 6, 7)
    ws.mag_raw = np.stack((source, source * 2, source * 3), axis=3)
    original = ws.mag_raw.copy()
    percentile = np.percentile
    calls = []
    def counted(values, q, *args, **kwargs):
        if np.ndim(q) == 0 and float(q) == 99.5:
            calls.append(True)
        return percentile(values, q, *args, **kwargs)
    monkeypatch.setattr(np, "percentile", counted)
    with closing(pv.Plotter(off_screen=True)) as plotter:
        scene = SceneController(plotter, ws, lambda _: None)
        obj = scene.add_content_render(3)
        actor = obj.actor
        def verify(phase):
            scene.update_time(phase)
            assert obj.actor is actor
            dataset = actor.GetMapper().dataset
            values = np.asarray(dataset["Magnitude"], float)
            expected = (0., float(percentile(values[np.isfinite(values)], 99.5)))
            np.testing.assert_allclose(scene._volume_scalar_range(obj, dataset), expected)
            np.testing.assert_allclose(actor.GetProperty().GetRGBTransferFunction(0).GetRange(), expected)
            np.testing.assert_allclose(actor.GetMapper().scalar_range, expected)
        for phase in (0, 1, 2):
            verify(phase)
        warm_count = len(calls)
        for phase in (2, 0, 1, 2, 1):
            verify(phase)
        assert len(calls) == warm_count
        dataset = actor.GetMapper().dataset
        dataset.point_data["Magnitude"][:] *= 2
        scene._set_volume_opacity(obj, dataset)
        assert len(calls) > warm_count
        scene.invalidate_cache("magnitude_volume")
        verify(1)
        np.testing.assert_array_equal(ws.mag_raw, original)


@pytest.mark.parametrize("mirrored", [False, True])
def test_streamline_tube_phantom_compact_geometry_and_dense_drag_phase_restore(monkeypatch, mirrored):
    from contextlib import closing
    import pyvista as pv
    from autoflow.core.models import ObjectKind
    from autoflow.ui.viewer import SceneController
    ws = Workspace()
    ws.flow_raw = np.zeros((5, 5, 5, 2, 3), np.float32)
    ws.flow_raw[..., 0] = 100.
    ws.streamline_active = True
    ws.streamline_params.display_style = "tubes"
    ws.render_settings["streamline_clim"] = (0., 1.)
    count = 50_100
    x = np.linspace(0., 30., count, dtype=np.float32)
    for phase in (0, 1):
        mesh = pv.PolyData(np.column_stack((x, np.sin(x), np.full(count, phase, np.float32))),
                           lines=np.r_[count, np.arange(count, dtype=np.int64)])
        mesh["Velocity"] = np.linspace(.1, .8, count, dtype=np.float32)
        mesh["vector"] = np.column_stack((mesh["Velocity"], np.zeros((count, 2), np.float32)))
        mesh["IntegrationTime"] = x.astype(float)
        mesh.cell_data["SeedIds"] = [phase]
        mesh.set_active_scalars("Velocity")
        ws.streamline_cache[phase] = mesh
    raw = ws.streamline_cache[0]
    source_points = raw.points.copy()
    source_arrays = {key: value.copy() for key, value in raw.point_data.items()}
    with closing(pv.Plotter(off_screen=True)) as plotter:
        scene = SceneController(plotter, ws, lambda _: None)
        if mirrored:
            scene.set_display_axis_directions(["RL", "AP", "FH"])
        uid = ws.add_object(name="flow", kind=ObjectKind.FLOW, data_key="streamlines_live",
                            scalars="Velocity", dynamic=True, tube_radius=.05)
        obj = ws.scene_objects[uid]
        expected = raw.tube(radius=.05)
        compact = scene._display_dataset(obj, raw)
        for left, right in ((compact.points, expected.points), (compact.strips, expected.strips),
                            (compact["Velocity"], expected["Velocity"]),
                            (compact["TubeNormals"], expected["TubeNormals"])):
            np.testing.assert_array_equal(left, right)
        assert compact.GetActualMemorySize() < expected.GetActualMemorySize()
        scene.render_all()
        actor = obj.actor
        tube = pv.PolyData.tube
        builds = []
        def counted(mesh, *args, **kwargs):
            builds.append(True)
            return tube(mesh, *args, **kwargs)
        monkeypatch.setattr(pv.PolyData, "tube", counted)
        scene.set_interaction_preview(True)
        assert actor.GetMapper().dataset.n_points == count
        scene.update_time(1)
        assert obj.actor is actor and actor.GetMapper().dataset.n_points == count
        assert not builds
        scene.set_interaction_preview(False, render=False)
        assert len(builds) == 1 and obj.actor is actor
        assert actor.GetMapper().dataset.n_points > count
        # Release restores the current phase even when playback advanced.
        assert np.min(actor.GetMapper().dataset.points[:, 2]) > .8
        np.testing.assert_array_equal(raw.points, source_points)
        assert set(raw.point_data) == set(source_arrays)
        for key, value in source_arrays.items():
            np.testing.assert_array_equal(raw[key], value)


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


def _send_input_drop(widget, urls):
    from PySide6 import QtCore, QtGui, QtWidgets
    mime = QtCore.QMimeData()
    mime.setUrls(urls)
    actions = QtCore.Qt.CopyAction | QtCore.Qt.MoveAction
    enter = QtGui.QDragEnterEvent(QtCore.QPoint(20, 20), actions, mime, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier)
    QtWidgets.QApplication.sendEvent(widget, enter)
    if not enter.isAccepted():
        return False
    move = QtGui.QDragMoveEvent(QtCore.QPoint(20, 20), actions, mime, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier)
    QtWidgets.QApplication.sendEvent(widget, move)
    drop = QtGui.QDropEvent(QtCore.QPointF(20, 20), actions, mime, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier)
    QtWidgets.QApplication.sendEvent(widget, drop)
    if drop.isAccepted():
        assert drop.dropAction() == QtCore.Qt.CopyAction
    return drop.isAccepted()


@pytest.mark.skipif(sys.platform == "win32", reason="Linux SSH renderer; native Windows GUI is verified by portable_check.py --gui")
def test_gui_input_drop_phantom_loads_h5_and_rejects_unsupported_or_busy(monkeypatch, tmp_path):
    app = _smoke_qt_application(monkeypatch)
    monkeypatch.setenv("AUTOFLOW_SSH_RENDERING", "1")
    from PySide6 import QtCore, QtWidgets
    import autoflow.ui.app as ui_app
    from autoflow.ui.remote_plotter import RemotePlotter
    monkeypatch.setattr(ui_app, "RemotePlotter", RemotePlotter, raising=False)
    path = tmp_path / "flow phantom \u6d4b\u8bd5.HDF5"
    shape = (3, 4, 5, 2)
    with h5py.File(path, "w") as handle:
        handle["mag"] = np.ones(shape, dtype=np.float32)
        handle["flow"] = np.full(shape + (3,), 10., dtype=np.float32)
        handle["Resolution"] = [1., 1.5, 2.]
        handle["RR"] = 1000.
        handle["VENC"] = [100., 100., 100.]
        handle["SpatialOrder"] = np.array(["LR", "AP", "FH"], dtype="S4")
        handle["VENCOrder"] = np.array(["LR", "AP", "FH"], dtype="S4")
    original = path.read_bytes()
    url = QtCore.QUrl.fromLocalFile(str(path))
    window = ui_app.MainWindow()
    assert window.spin_fps.value() == 10.
    assert window.spin_fps.minimum() == .5 and window.spin_fps.maximum() == 100.
    assert window.spin_fps.suffix() == " fps"
    window.show()
    app.processEvents()
    try:
        initial = window.workspace
        assert _send_input_drop(window.plotter, [url])
        # Load dialogs must wait until the desktop drag has ended.
        assert window.workspace is initial
        app.processEvents()
        assert window.workspace.data_loaded, window.console.toPlainText()
        assert window.workspace.mag_raw.shape == shape
        np.testing.assert_allclose(window.workspace.flow_raw, 10.)
        loaded = window.workspace
        invalid = tmp_path / "single.dcm"
        invalid.write_bytes(b"not a directory")
        for urls in ([url, url], [QtCore.QUrl("https://example.com/flow.h5")],
                     [QtCore.QUrl.fromLocalFile(str(invalid))],
                     [QtCore.QUrl.fromLocalFile(str(tmp_path / "missing.h5"))]):
            assert not _send_input_drop(window, urls)
        with monkeypatch.context() as patch:
            patch.setattr(window, "_pipeline_thread", object())
            assert not _send_input_drop(window, [url])
            assert "Cannot switch cases" in window.statusBar().currentMessage()
        broken = tmp_path / "broken.h5"
        broken.write_bytes(b"invalid H5")
        assert _send_input_drop(window, [QtCore.QUrl.fromLocalFile(str(broken))])
        app.processEvents()
        assert "H5 could not be opened" in window.statusBar().currentMessage()
        assert window.workspace is loaded and path.read_bytes() == original
        # A dropped directory still asks where to save the converted H5.
        monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", lambda *a, **k: ("", ""))
        assert _send_input_drop(window, [QtCore.QUrl.fromLocalFile(str(tmp_path))])
        app.processEvents()
        assert "no H5 destination" in window.statusBar().currentMessage()
        assert window.workspace is loaded
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()


@pytest.mark.skipif(sys.platform == "win32", reason="Linux SSH renderer; native Windows GUI is verified by portable_check.py --gui")
def test_dicom2h5_gui_phantom_destination_feedback_and_worker_progress(monkeypatch, tmp_path):
    import time
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
        assert _send_input_drop(window, [QtCore.QUrl.fromLocalFile(str(root))])
        app.processEvents()
        for _ in range(500):
            app.processEvents()
            if applied:
                break
            # Process Qt events above, then release the GIL for the Python worker.
            time.sleep(.01)
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
            # Process Qt events above, then release the GIL for the Python worker.
            time.sleep(.01)
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


@pytest.mark.skipif(sys.platform == "win32", reason="Linux SSH renderer; native Windows GUI is verified by portable_check.py --gui")
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
        for dialog in (skeleton, planes, pathlines, pwv):
            assert [dialog.tabs.tabText(i) for i in range(dialog.tabs.count())] == ["Basic", "Advanced"]
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
        assert MainWindow._selected_render_object(owner) is None
        owner._browser_selected_objects.return_value = [segmentation, noise]
        assert MainWindow._selected_render_object(owner) is None
        owner.scene._visible_volume_object.assert_not_called()
        owner._browser_selected_objects.return_value = [volume]
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


@pytest.mark.skipif(sys.platform == "win32", reason="Linux SSH renderer; native Windows GUI is verified by portable_check.py --gui")
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
        marker = window.scene._orientation_marker
        window.plotter.camera.parallel_projection = True
        window.plotter.camera.parallel_scale = 32
        for _, direction, _ in marker.VIEWS:
            viewport = marker.widget.GetViewport()
            pos = QtCore.QPoint(round((viewport[0]+viewport[2])/2*window.plotter.width()),
                round((1-(viewport[1]+viewport[3])/2)*window.plotter.height()))
            QtTest.QTest.mouseClick(window.plotter, QtCore.Qt.LeftButton, pos=pos)
            np.testing.assert_allclose(marker._camera_direction()[:3],
                np.asarray(direction)*window.scene._display_axis_signs, atol=1e-7)
            assert window.plotter.camera.parallel_projection and window.plotter.camera.parallel_scale == 32
        window.plotter.camera.Azimuth(20)
        QtTest.QTest.mouseClick(window.plotter, QtCore.Qt.LeftButton, pos=pos)
        np.testing.assert_allclose(marker._camera_direction()[:3], [0, -1, 0], atol=1e-7)
        # Plane picking is a read-only display query, including non-unit input normals.
        from autoflow.core.plane_state import stamp_metric, metric_validity
        original_normal = ws.planes[0].normal.copy()
        ws.planes[0].normal = np.array([1., 2., 3.])
        ws.planes[0].metrics = stamp_metric(ws.planes[0], {"flowrate_mL_s": [1., 2.]}, 0)
        window.scene._plane_uid_at_display_position(-1000, -1000)
        np.testing.assert_array_equal(ws.planes[0].normal, [1., 2., 3.])
        assert metric_validity(ws.planes[0]) == "current"
        ws.planes[0].normal = original_normal
        ws.planes[0].metrics = {}
        window._add_manual_plane()
        assert ws.scene_objects[first_uid].actor is first_actor
        assert ws.pathline_cache[0] == "keep existing pathline"
        assert window._plane_edit_enabled and window.scene._plane_selection_locked
        handles = window._plane_handles
        assert [widget.GetEnabled() for widget, _ in handles.handles] == [1]
        assert window.combo_plane_edit_mode.count() == 2
        assert not window.ortho_viewer.slice_views["axial"].view_box.plane_rotation_enabled
        assert all(window.ortho_viewer.slice_views[name].view_box.plane_rotation_enabled
                   for name in ("coronal", "sagittal"))
        assert "2D views" in handles.help.GetInput()
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
        def rotate_2d(name):
            view = window.ortho_viewer.slice_views[name]
            viewport = view.plot.viewport()
            radius = (view.view_box.viewRange()[0][1]-view.view_box.viewRange()[0][0])*.22
            def screen(x, y):
                return view.plot.mapFromScene(view.view_box.mapViewToScene(QtCore.QPointF(x, y)))
            start = screen(radius, 0)
            finish = screen(radius*.866, radius*.5)
            QtTest.QTest.mouseMove(viewport, start)
            QtTest.QTest.mousePress(viewport, QtCore.Qt.LeftButton, pos=start)
            QtTest.QTest.qWait(25)  # Respect pyqtgraph's scene mouse rate limit.
            from PySide6 import QtGui, QtWidgets
            move = QtGui.QMouseEvent(QtCore.QEvent.Type.MouseMove, QtCore.QPointF(finish),
                QtCore.QPointF(viewport.mapToGlobal(finish)), QtCore.Qt.NoButton,
                QtCore.Qt.LeftButton, QtCore.Qt.NoModifier)
            QtWidgets.QApplication.sendEvent(viewport, move)
            QtTest.QTest.mouseRelease(viewport, QtCore.Qt.LeftButton, pos=finish)
            QtTest.QTest.qWait(40)
            assert window.ortho_viewer._plane_rotation_reference is None
            assert not writes and not recomputes
            np.testing.assert_allclose(tuple(window.plotter.camera_position), camera)

        for name in ("coronal", "sagittal"):
            plane = ws.planes[1]
            before_normal = plane.normal.copy()
            rotate_2d(name)
            assert np.linalg.norm(plane.normal - before_normal) > 0.01
            np.testing.assert_allclose(np.dot(window._plane_axis_u, window._plane_axis_v), 0., atol=1e-10)
            np.testing.assert_allclose(np.cross(window._plane_axis_u, window._plane_axis_v), plane.normal, atol=1e-10)
        assert "preview" in window.ortho_viewer.label_plane_metric.text().lower()
        window._toggle_selected_plane_edit()
        assert len(writes) == 1 and not recomputes  # A new plane has no existing metrics to refresh.
        assert not window.scene._plane_selection_locked and window._plane_handles is None
        assert not window.scene._hud_renderer.HasViewProp(handles.help)

        # Drafts retain the previous result for Cancel; Apply queues one update.
        from autoflow.core.plane_state import metric_validity
        monkeypatch.setattr(window, "_schedule_plane_calculation", lambda index: recomputes.append(index))
        ws.planes[1].metrics = {"area_mm2": [1., 1.]}
        window._activate_plane_drag_widgets(1)
        window._update_plane_from_drag(1, center=ws.planes[1].center + [0.5, 0., 0.])
        assert ws.planes[1].metrics and metric_validity(ws.planes[1]) == "stale"
        window._toggle_selected_plane_edit()
        assert recomputes == [1]
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
    # One application serves many GUI phantoms; closing one test window must
    # not queue application shutdown while a later test runs a QThread.
    _qt_smoke_application.setQuitOnLastWindowClosed(False)
    return _qt_smoke_application


@pytest.mark.skipif(sys.platform == "win32", reason="Linux SSH renderer; native Windows GUI is verified by portable_check.py --gui")
def test_browser_plane_phantom_display_controls_and_visibility_clicks(monkeypatch, tmp_path):
    app = _smoke_qt_application(monkeypatch)
    monkeypatch.setenv("AUTOFLOW_SSH_RENDERING", "1")
    from PySide6 import QtCore, QtTest, QtWidgets
    import pyvista as pv
    import autoflow.ui.app as ui_app
    from autoflow.ui.remote_plotter import RemotePlotter
    from autoflow.core.models import ObjectKind
    monkeypatch.setattr(ui_app, "RemotePlotter", RemotePlotter, raising=False)
    window = ui_app.MainWindow()
    try:
        window.show()
        app.processEvents()
        assert not window.btn_browser_add_plane.isEnabled()
        assert not window.btn_delete_obj.isEnabled()
        assert not window.chk_all_planes_visible.isEnabled()
        assert not window.chk_all_pathlines_visible.isEnabled()
        assert not hasattr(window, "document_tabs")
        assert window.workspace_pages.currentWidget() is window.workspace_page
        assert window.pwv_dock.windowTitle() == "Measurements"
        assert window.pwv_dock.isHidden() and window.log_dock.isHidden()
        assert window.appearance_scroll.isHidden() and window.selection_tabs.isHidden()
        window._on_open_case_qc()
        assert window.workspace_pages.currentWidget() is window.qc_page
        assert window.qc_page.currentWidget() is window.qc_empty
        window._on_open_export_page()
        assert window.workspace_pages.currentWidget() is window._export_page
        assert not any(button.isEnabled() for button in window._export_case_buttons)
        window._select_workflow_page('input')
        assert window.workspace_pages.currentWidget() is window.workspace_page
        assert window.slider_plane_size.parentWidget() is window.plane_display_controls
        assert window.spin_plane_size.parentWidget() is window.plane_display_controls
        assert window.chk_plane_selected_only.parentWidget() is window.plane_display_controls
        assert window.appearance_panel.isAncestorOf(window.slider_plane_size)
        assert window.appearance_panel.isAncestorOf(window.slider_render_window)
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
        assert isinstance(window.selection_tabs, QtWidgets.QStackedWidget)
        assert not isinstance(window.selection_tabs, QtWidgets.QTabWidget)
        assert window.tree_objects.header().isHidden()
        assert window.tree_objects.inline_row.parent() is item
        assert window.appearance_scroll.isHidden() and not window.appearance_toggle.isChecked()
        window.appearance_toggle.click()
        app.processEvents()
        assert window.appearance_scroll.isVisible() and window.plane_display_controls.isVisible()
        assert window.selection_tabs.isVisible() and window.pwv_dock.isVisible()
        inspector = window._inline_inspector
        for _ in range(3):
            window._refresh_browser()
            app.processEvents()
            item = window._find_browser_item_by_uid(planes[0].uid)
            assert item.isSelected() and window.tree_objects.currentItem() is item
            assert window.tree_objects.inline_row.parent() is item
            assert window._inline_inspector is inspector
            assert window.appearance_toggle.isChecked()
        # The eye changes visibility without selecting a different row or
        # treating the inspector's display row as a scene object.
        window.tree_objects.scrollToItem(item)
        app.processEvents()
        eye = window.tree_objects.visualRect(window.tree_objects.indexFromItem(item, 1))
        QtTest.QTest.mouseClick(window.tree_objects.viewport(), QtCore.Qt.LeftButton, pos=eye.center())
        assert not planes[0].visible and item.isSelected()
        QtTest.QTest.mouseClick(window.tree_objects.viewport(), QtCore.Qt.LeftButton, pos=eye.center())
        assert planes[0].visible and window.tree_objects.currentItem() is item
        assert set(window._collect_browser_uids(item)) == {planes[0].uid, pathlines[0].uid}
        assert window.ortho_viewer.roi_edit_tools.isHidden()
        highlight = window.scene._highlight_plane_actor
        assert highlight.GetVisibility()

        def click(checkbox):
            QtTest.QTest.mouseClick(checkbox, QtCore.Qt.LeftButton,
                                   pos=QtCore.QPoint(8, checkbox.height() // 2))
            app.processEvents()

        # Exercise the actual checkbox widgets hosted in the overflow menu.
        window.browser_options_menu.popup(window.browser_options_button.mapToGlobal(QtCore.QPoint(0, window.browser_options_button.height())))
        app.processEvents()
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
        window.browser_options_menu.hide()
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
        deadline = QtCore.QElapsedTimer(); deadline.start()
        while dialog.activity.phase <= phase and deadline.elapsed() < 2000:
            QtTest.QTest.qWait(50)
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


@pytest.mark.skipif(sys.platform == "win32", reason="Linux SSH renderer; native Windows GUI is verified by portable_check.py --gui")
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


def test_missing_plane_roi_phantom_interpolates_geometry_and_samples_current_fields(tmp_path, monkeypatch):
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
    assert metric["roi_source_frame_t"] == [4, 1, 1, 1, 4, 4]
    assert not any(metric["roi_carried_forward_t"])
    assert metric['roi_interpolated_t'] == [True, False, True, True, False, True]
    assert metric['roi_source_frames_t'][0] == [4, 1]
    assert metric['roi_source_frames_t'][2] == [1, 4]
    np.testing.assert_allclose(metric['roi_interpolation_weight_t'][2], 1/3)
    assert np.all((np.asarray(metric["area_mm2"]) >= 4) & (np.asarray(metric["area_mm2"]) <= 9))
    np.testing.assert_allclose(metric["flowrate_mL_s"], np.asarray(metric['area_mm2']) * np.arange(1, 7) / 10, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(metric['area_mm2'])[[1, 4]], [4, 9])
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
        assert h5["planes/000/timepoints/002/roi_carried_forward"][0] == 0
        assert h5["planes/000/timepoints/002/roi_interpolated"][0] == 1
        np.testing.assert_array_equal(h5["planes/000/timepoints/002/roi_source_frames"][:], [1, 4])
        np.testing.assert_allclose(h5["planes/000/timepoints/002/relative_pressure_Pa"][:], 9)
    np.testing.assert_array_equal(labels, original_labels)
    assert not plane.roi_edit_operations

    # Explicit deletion stays empty and stops forward filling until a real ROI returns.
    plane.roi_edit_operations = {"2": [{"mode": "remove", "polygon": [[-30, -30], [30, -30], [30, 30], [-30, 30]]}]}
    # Frame 2 has no observed ROI; an explicit empty edit is a propagation barrier.
    deleted = compute_plane_metrics(flow, mask, (1, 1, 1), (10, -20, 30), [plane],
                                   segmentation_labels_4d=labels)[0]
    assert deleted["roi_source_frame_t"] == [4, 1, None, 4, 4, 4]
    assert deleted['roi_resolution_t'][2] == 'explicit_empty'
    assert deleted['sample_valid_t'][2] is False
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
        assert region is not None and source is None
        assert viewer._selected_plane_contours(plane, 5, 20., .75)
        for frame in (4,):
            contours = viewer._selected_plane_contours(plane, frame, 20., .75)
            cyan = [contour for contour in contours if contour[1] == "#00f5d4"]
            assert len(cyan) == 1
            np.testing.assert_allclose(cyan[0][0], np.vstack((polygon, polygon[0])))
        assert viewer._effective_plane_region(plane, 0) is not None
        plane.roi_edit_operations = {"1": [{"mode": "remove", "polygon": [[-30, -30], [30, -30], [30, 30], [-30, 30]]}]}
        assert viewer._effective_plane_region(plane, 1) is None
        assert not viewer._selected_plane_contours(plane, 1, 20., .75)
        np.testing.assert_array_equal(ws.segmask_labels, original_labels)
    finally:
        viewer.close()
        viewer.deleteLater()


@pytest.mark.parametrize("sources,carried,frame_only", [
    ([0, 1, 2, 3], [False] * 4, False),
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
def test_batch_accounts_for_video_output_and_partial_failure(tmp_path, monkeypatch, video_fails, regression_matrix):
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
    config = AutoFlowConfig(inputs=[str(regression_matrix/'circle_1x1x1mm.h5')], output_dir=str(tmp_path),
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


@pytest.mark.skipif(sys.platform == "win32", reason="Linux SSH renderer; native Windows GUI is verified by portable_check.py --gui")
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
        assert finished[0]["analysis_report"]["status"] == "complete"
        assert (tmp_path / "analysis_report.html").is_file()
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


@pytest.mark.skipif(sys.platform == "win32", reason="Linux SSH renderer; native Windows GUI is verified by portable_check.py --gui")
def test_gui_menu_smoke_has_no_docs_action_or_f1_shortcut(monkeypatch):
    app = _smoke_qt_application(monkeypatch)
    monkeypatch.setenv("AUTOFLOW_SSH_RENDERING", "1")
    from PySide6 import QtGui
    import autoflow.ui.app as ui_app
    from autoflow.ui.remote_plotter import RemotePlotter
    monkeypatch.setattr(ui_app, "RemotePlotter", RemotePlotter, raising=False)
    window = ui_app.MainWindow()
    try:
        window.show()
        app.processEvents()
        assert [action.text() for action in window.menuBar().actions()] == ["File", "Edit", "Export", "Settings", "View"]
        assert [action.text() for action in window.edit_menu.actions()] == ["Undo", "Redo"]
        assert window.file_menu.actions()[0].text() == "Load H5"
        assert not hasattr(window, "action_docs")
        assert all(shortcut.toString() != "F1"
                   for action in window.findChildren(QtGui.QAction)
                   for shortcut in action.shortcuts())
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


def test_junction_cluster_phantom_reduces_repeated_contacts_without_erasing_labels():
    from itertools import combinations
    import networkx as nx
    from autoflow.algorithms.graph import graph_to_networkx
    from autoflow.algorithms.graph_junctions import simplify_junction_clusters
    points = np.array([[4, 4, 4], [5, 4, 4], [4, 5, 4], [4, 4, 5], [7, 7, 7]], dtype=float)
    graph = GraphData(points=points, edges=np.array(list(combinations(range(5), 2))))
    labels = np.ones((12, 12, 12), dtype=np.int16);labels[7, 7, 7] = 2
    original_interfaces = {tuple(sorted((tuple(points[i]), tuple(points[4])))) for i in range(4)}
    cleaned, qc = simplify_junction_clusters(graph, labels)
    retained_interfaces = original_interfaces & _junction_edge_geometry(cleaned)
    assert retained_interfaces and len(retained_interfaces) < len(original_interfaces)
    assert nx.is_connected(graph_to_networkx(cleaned))
    assert qc['cycle_rank_before'] == 6 and qc['cycle_rank_after'] < 3
    assert all(qc['local_cleanup']['checks'].values())
    assert qc['protected_edge_count'] == 4 and qc['interface_ports_preserved']
    assert qc['clusters'][0]['source_labels'] == {'1': 4, '2': 1}
    assert qc['clusters'][0]['label_patches'][0]['cycle_rank_after'] == 0


@pytest.mark.parametrize('kind', ['disabled', 'oversized', 'minimum_cycles', 'closed_patch', 'unknown', 'simple_ring'])
def test_junction_cluster_phantom_keeps_ineligible_regions(kind):
    from autoflow.algorithms.graph_junctions import simplify_junction_clusters
    graph, labels = _junction_cluster_phantom();kwargs = {}
    if kind == 'disabled':kwargs['enabled'] = False
    elif kind == 'oversized':
        graph.points[1, 0] = 13;labels = np.ones((20, 20, 20), dtype=np.int16);kwargs['max_extent_voxels'] = 8
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


@pytest.mark.parametrize("scale", [1., 12.])
def test_junction_cluster_phantom_unbounded_default_preserves_ports_across_extent(scale):
    import networkx as nx
    from autoflow.algorithms.graph import graph_to_networkx
    from autoflow.algorithms.graph_junctions import simplify_junction_clusters
    graph, _ = _junction_cluster_phantom()
    graph.points *= scale
    labels = np.ones((int(9*scale)+2,)*3, dtype=np.int16)
    expected_endpoints = {tuple(graph.points[4]), tuple(graph.points[5])}
    cleaned, qc = simplify_junction_clusters(graph, labels)
    assert nx.is_tree(graph_to_networkx(cleaned))
    assert qc['simplified_cluster_count'] == 1 and qc['cycle_rank_after'] == 0
    assert qc['max_extent_voxels'] == 0 and qc['extent_limit_applied'] is False
    assert expected_endpoints == {tuple(cleaned.points[n]) for n,d in graph_to_networkx(cleaned).degree() if d==1}
    assert all(qc[name] for name in ('outside_edges_preserved','boundary_ports_preserved','interface_ports_preserved'))


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
    _seed_manual_1mm_phantom_state(ws)
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
    assert 'radius_tree_cleanup' not in exported['groups'][0]


def test_junction_cluster_phantom_config_group_gui_cli_roundtrip():
    from PySide6 import QtWidgets
    from autoflow.cli import build_parser
    from autoflow.ui.action_settings import ActionSettingsDialog, action_sections
    cfg = AutoFlowConfig.from_config_dir()
    assert cfg.junction_cluster_enabled and cfg.junction_cluster_max_extent_voxels == 0 and cfg.junction_cluster_min_cycle_rank == 2
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
    ('junction_cluster_max_extent_voxels', -.5), ('junction_cluster_max_extent_voxels', .5), ('junction_cluster_max_extent_voxels', np.inf),
    ('junction_cluster_max_extent_voxels', True), ('junction_cluster_min_cycle_rank', 0),
    ('junction_cluster_min_cycle_rank', 1.5), ('junction_cluster_min_cycle_rank', True)])
def test_junction_cluster_phantom_rejects_invalid_parameters(name, value):
    with pytest.raises(ValueError, match=name):SkeletonParams.from_dict({name: value})
    with pytest.raises(ValueError, match=name):build_workspace(AutoFlowConfig(**{name: value}))


@pytest.mark.parametrize("scalar_venc", [False, True])
def test_pointwise_phantom_preserves_numpy_order_and_broadcast(monkeypatch, scalar_venc):
    from autoflow.algorithms.pointwise import (wrapped_phase_from_flow,
        synthetic_complex_from_mag_flow, apply_complex_correction, velocity_from_complex)
    monkeypatch.setenv("AUTOFLOW_POINTWISE_WORKERS", "4")
    monkeypatch.setenv("AUTOFLOW_POINTWISE_BLOCK_ROWS", "4096")
    rng = np.random.default_rng(12)
    # Deliberately non-contiguous, including +/-pi boundary values and zero mag.
    flow = rng.uniform(-240, 240, (25, 24, 8, 5, 3)).astype(np.float32)[::-1]
    flow[0, 0, 0, 0] = [-120, 0, 120]
    mag = rng.uniform(0, 1, flow.shape[:-1]).astype(np.float32)
    mag[0, 0, 0, 0] = 0
    venc = np.float32(120) if scalar_venc else np.array([120, 130, 140], np.float32)
    vv = np.repeat(venc, 3) if np.ndim(venc) == 0 else venc
    expected_phase = np.angle(np.exp(1j * flow * np.pi / np.maximum(vv, 1e-6))).astype(np.float32)
    np.testing.assert_array_equal(wrapped_phase_from_flow(flow, venc), expected_phase)
    expected = np.empty(flow.shape[:-1] + (4,), np.complex64)
    expected[..., 0] = mag.astype(np.complex64)
    expected[..., 1:4] = mag[..., None] * np.exp(1j * (np.pi * flow / vv))
    np.testing.assert_array_equal(synthetic_complex_from_mag_flow(mag, flow, venc), expected)
    for corr in (rng.normal(size=flow.shape).astype(np.float32), np.zeros(flow.shape[:3] + (1, 3), np.float32)):
        corrected = expected.copy()
        corrected[..., 1:4] *= np.exp(-1j * corr)
        actual = apply_complex_correction(expected, corr)
        np.testing.assert_array_equal(actual, corrected)
        velocity = (np.angle(corrected[..., 1:4] * np.conj(corrected[..., 0:1])) / np.pi) * vv
        np.testing.assert_array_equal(velocity_from_complex(actual, venc), velocity.astype(np.float32))


def test_pointwise_phantom_cancelled_before_partial_output():
    from autoflow.algorithms.pointwise import wrapped_phase_from_flow
    from autoflow.task_control import CancellationToken, TaskCancelled, task_scope
    token = CancellationToken()
    token.cancel()
    with task_scope():
        with pytest.raises(TaskCancelled):
            with task_scope(token):
                wrapped_phase_from_flow(np.zeros((8, 8, 8, 2, 3), np.float32), 100)


def test_unwrap_tfc_dynamic_support_does_not_create_phase_jump():
    from autoflow.algorithms.phase_unwrapping._common import total_field_correction
    values = np.full((10, 10, 10, 4), 3.0, np.float32)
    mask = np.ones_like(values, bool)
    mask[..., 0] = False
    mask[:1, :1, :, 0] = True
    np.testing.assert_array_equal(total_field_correction(values, mask), values)
    mask[..., 2] = False
    np.testing.assert_array_equal(total_field_correction(values, mask), values)
    np.testing.assert_array_equal(total_field_correction(values, np.zeros_like(mask)), values)


def test_unwrap_final_wrap_count_includes_temporal_alignment():
    from autoflow.algorithms.phase_unwrapping.engine import unwrap_phase
    phase = np.broadcast_to(np.array([2.9, -2.9, -2.7, -2.5], np.float32)[None, None, None, :, None],
                            (4, 4, 4, 4, 3)).copy()
    result = unwrap_phase(phase, None, 100, "nprs", device="cpu",
                          params={"tfc": True, "nprs_upsampling_factor": 1})
    expected = np.rint((result["phase_unwrapped"] - phase) / (2 * np.pi)).astype(np.int16)
    np.testing.assert_array_equal(result["wrap_count"], expected)
    np.testing.assert_array_equal(result["wrap_mask"], expected != 0)
    assert np.any(expected[..., -1, :])


@pytest.mark.parametrize("shape", [(16, 16, 16, 5), (15, 13, 11, 5)])
def test_unwrap_laplacian_odd_dimensions_include_last_frame(shape):
    from autoflow.algorithms.phase_unwrapping.laplacian import unwrap_4D, _lap4d_gpu
    axes = np.meshgrid(*(np.linspace(-1, 1, n) for n in shape[:3]), indexing="ij")
    blob = 9.0 * np.exp(-sum(x*x for x in axes) / .2)
    wrapped = np.angle(np.exp(1j * blob)).astype(np.float32)
    phi = np.repeat(wrapped[..., None], shape[3], axis=3)
    counts = unwrap_4D(phi)
    assert counts.shape == shape and np.any(counts[..., -1] != 0)
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        return
    recovered, gpu_counts = _lap4d_gpu(phi, np.ones(shape, bool), ts=2, device="cuda", tfc=False)
    assert gpu_counts.shape == shape and np.any(gpu_counts[..., -1] != 0)
    np.testing.assert_array_equal(gpu_counts, counts)
    np.testing.assert_allclose(recovered, phi + 2*np.pi*gpu_counts, atol=2e-6, rtol=0)


def test_plane_identity_phantom_delete_and_workspace_roundtrip():
    from autoflow.core.models import PlaneData, Workspace
    from autoflow.core.plane_state import remove_planes, stamp_metric, selected_metric
    ws = Workspace()
    ws.planes = [PlaneData(np.array([i, 1, 2.]), np.array([1., 0, 0]), path_index=i) for i in range(3)]
    for i, plane in enumerate(ws.planes):
        plane.roi_edit_operations = {"0": [{"mode": "replace", "polygon": [[0, 0], [1, 0], [0, 1]]}]}
        plane.metrics = stamp_metric(plane, {"plane_index": i, "flowrate_mL_s": [i+1.]}, i)
    ws.derived.plane_metrics = [dict(p.metrics) for p in ws.planes]
    keep = [p.plane_id for p in ws.planes[1:]]
    removed = ws.planes[0].plane_id
    assert remove_planes(ws, [removed]) == {1: 0, 2: 1}
    assert [p.plane_id for p in ws.planes] == keep
    assert [m["plane_id"] for m in ws.derived.plane_metrics] == keep
    assert [m["plane_index"] for m in ws.derived.plane_metrics] == [0, 1]
    assert selected_metric(ws, 0)["flowrate_mL_s"] == [2.]
    restored = Workspace()
    restored.restore_dict(ws.snapshot_dict())
    assert [p.plane_id for p in restored.planes] == keep
    assert restored.planes[0].roi_edit_operations == ws.planes[0].roi_edit_operations


def test_held_arrow_phantom_paints_before_release_and_wraps(monkeypatch):
    pytest.importorskip("PySide6")
    pytest.importorskip("pyqtgraph")
    pytest.importorskip("pyvistaqt")
    from PySide6 import QtCore, QtGui, QtWidgets, QtTest
    from types import MethodType, SimpleNamespace
    from autoflow.ui.app import MainWindow
    from autoflow.ui.time_navigation import FrameNavigationFilter
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = QtWidgets.QMainWindow()
    target = QtWidgets.QWidget(window)
    target.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
    window.setCentralWidget(target)
    window.workspace = SimpleNamespace(current_t=4, time_count=lambda: 5)
    window._pipeline_thread = None
    window._keyboard_navigation_active = False
    window._pending_timeline_value = None
    window._play_timer = QtCore.QTimer(window)
    window.slider_t = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal, target)
    window.lab_t = QtWidgets.QLabel(target)
    frames = []
    window._refresh_timeline = lambda: None
    window._refresh_time_views = lambda: frames.append(window.workspace.current_t)
    window._refresh_segmentation_ui = lambda: None
    window._refresh_selection_info = lambda: None
    for name in ("_queue_keyboard_time_step", "_finish_keyboard_time_step", "_finish_timeline_scrub",
                 "_apply_pending_timeline_value"):
        setattr(window, name, MethodType(getattr(MainWindow, name), window))
    window._timeline_scrub_timer = QtCore.QTimer(window)
    window._timeline_scrub_timer.setSingleShot(True)
    window._timeline_scrub_timer.setInterval(30)
    window._timeline_scrub_timer.timeout.connect(window._apply_pending_timeline_value)
    nav = FrameNavigationFilter(window, window._queue_keyboard_time_step, window._finish_keyboard_time_step)
    try:
        window.show()
        target.setFocus()
        app.processEvents()
        def press(key, repeat=False):
            event = QtGui.QKeyEvent(QtCore.QEvent.Type.KeyPress, key,
                                   QtCore.Qt.KeyboardModifier.NoModifier, "", repeat, 1)
            QtCore.QCoreApplication.sendEvent(target, event)
        press(QtCore.Qt.Key.Key_Right)
        assert nav.held, (app.focusWidget(), target.window(), window)
        assert window._timeline_scrub_timer.isActive()
        for _ in range(50):
            QtTest.QTest.qWait(10)
            if frames:
                break
        assert frames and frames[-1] == 0  # still held: rendered before release
        for _ in range(8):
            press(QtCore.Qt.Key.Key_Right, True)
            previous_count = len(frames)
            for _ in range(50):
                QtTest.QTest.qWait(10)
                if len(frames) > previous_count:
                    break
        assert len(frames) >= 6 and {0, 4}.issubset(frames)
        QtTest.QTest.keyRelease(target, QtCore.Qt.Key.Key_Right)
        assert window.workspace.current_t == 3
        window.workspace.current_t = 0
        press(QtCore.Qt.Key.Key_Left)
        for _ in range(50):
            QtTest.QTest.qWait(10)
            if frames[-1] == 4:
                break
        assert frames[-1] == 4
        QtTest.QTest.keyRelease(target, QtCore.Qt.Key.Key_Left)
        editor = QtWidgets.QLineEdit("text", target)
        editor.show()
        editor.setFocus()
        app.processEvents()
        before = list(frames)
        QtTest.QTest.keyClick(editor, QtCore.Qt.Key.Key_Left)
        QtTest.QTest.qWait(35)
        assert frames == before
    finally:
        app.removeEventFilter(nav)
        window.close()
        window.deleteLater()
        app.processEvents()


@pytest.mark.skipif(sys.platform == "win32", reason="Linux SSH renderer; native Windows GUI is verified by portable_check.py --gui")
def test_plane_transaction_phantom_cancel_and_undo_preserve_results(monkeypatch, tmp_path):
    app = _smoke_qt_application(monkeypatch)
    monkeypatch.setenv("AUTOFLOW_SSH_RENDERING", "1")
    import autoflow.ui.app as ui_app
    from autoflow.ui.remote_plotter import RemotePlotter
    from autoflow.core.plane_state import metric_validity
    monkeypatch.setattr(ui_app, "RemotePlotter", RemotePlotter, raising=False)
    window = ui_app.MainWindow()
    try:
        ws = window.workspace
        ws.data_loaded = True
        ws.paths.output_dir = str(tmp_path)
        ws.mag_raw = np.ones((12, 12, 12, 2), np.float32)
        ws.flow_raw = np.zeros((12, 12, 12, 2, 3), np.float32)
        ws.segmask_binary = np.ones((12, 12, 12, 2), bool)
        ws.ortho_cursor = np.array([6, 6, 6])
        plane = PlaneData(np.array([6., 6., 6.]), np.array([0., 0., 1.]), path_index=-1)
        plane.metrics = {"area_mm2": [10., 10.], "flowrate_mL_s": [0., 0.]}
        ws.planes = [plane]
        ws.derived.plane_metrics = [dict(plane.metrics)]
        window._rebuild_plane_objects()
        uid = window._find_uid_by_indexed_data_key("plane", 0)
        actor = ws.scene_objects[uid].actor
        writes, calculations = [], []
        monkeypatch.setattr(window, "_persist_plane_outputs", lambda **kw: writes.append(kw))
        monkeypatch.setattr(window, "_schedule_plane_calculation", calculations.append)
        window.show()
        app.processEvents()
        window._add_manual_plane()
        window._update_plane_from_drag(1, center=np.array([7., 6., 6.]))
        window._cancel_plane_edit()
        assert len(ws.planes) == 1 and ws.planes[0].plane_id == plane.plane_id
        assert not writes and not calculations
        assert ws.scene_objects[uid].actor is actor
        window._selected_plane_index = 0
        original = ws.planes[0].center.copy()
        window._activate_plane_drag_widgets(0)
        window._update_plane_from_drag(0, center=original + [1, 0, 0])
        assert metric_validity(ws.planes[0]) == "stale"
        window._force_exit_edit()  # Esc follows the same true rollback path.
        np.testing.assert_array_equal(ws.planes[0].center, original)
        assert metric_validity(ws.planes[0]) == "current" and not writes
        window._activate_plane_drag_widgets(0)
        window._update_plane_from_drag(0, center=original + [1, 0, 0])
        window._toggle_selected_plane_edit()
        assert calculations == [0] and len(writes) == 1
        window._undo_plane_change()
        np.testing.assert_array_equal(ws.planes[0].center, original)
        assert metric_validity(ws.planes[0]) == "current"
        window._redo_plane_change()
        np.testing.assert_array_equal(ws.planes[0].center, original + [1, 0, 0])
        assert metric_validity(ws.planes[0]) == "stale"
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()


def test_plane_sampling_context_phantom_reuses_geometry_without_numeric_change(monkeypatch):
    import autoflow.algorithms.metrics.sampling as sampling
    from autoflow.algorithms.metrics.planes import compute_plane_metrics
    from autoflow.algorithms.metrics.plane_derived import augment_plane_metrics_with_derived
    mask = np.ones((8, 8, 8, 3), bool)
    flow = np.zeros(mask.shape + (3,), np.float32)
    flow[..., 2] = 10
    tke = np.full(mask.shape, 2.0, np.float32)
    planes = [PlaneData(np.array([4., 4., 4.]), np.array([0., 0., 1.]), path_index=-1)]
    args = (flow, mask, np.ones(3), np.zeros(3), planes)
    baseline = compute_plane_metrics(*args)
    baseline, _ = augment_plane_metrics_with_derived(baseline, planes, mask, np.ones(3), np.zeros(3), tke_array=tke)
    calls = []
    original = sampling._build_plane_support_mesh
    def counted(*a, **kw):
        calls.append(1)
        return original(*a, **kw)
    monkeypatch.setattr(sampling, "_build_plane_support_mesh", counted)
    context = sampling.PlaneSamplingContext(mask, np.ones(3), np.zeros(3), planes)
    actual = compute_plane_metrics(*args, sampling_context=context)
    actual, _ = augment_plane_metrics_with_derived(actual, planes, mask, np.ones(3), np.zeros(3),
                                                  tke_array=tke, sampling_context=context)
    assert actual == baseline
    assert len(calls) == 1 and context.hits >= 1
    context.close()
    assert not context.slice_caches and not context.support_mesh_cache


def test_qc_report_phantom_stale_identity_locations_and_missing_values(tmp_path):
    from autoflow.core.plane_state import stamp_metric, invalidate_planes, remove_planes
    from autoflow.qc_locations import issue_anchors, iter_issues
    from autoflow.report_snapshot import build_report_snapshot, save_report_html
    ws = Workspace()
    ws.data_loaded = True
    ws.paths.flow_path = "case <phantom>.h5"
    ws.flow_raw = np.zeros((8, 8, 8, 2, 3), np.float32)
    ws.mag_raw = np.ones((8, 8, 8, 2), np.float32)
    ws.origin = np.array([30., -10., 5.])
    ws.planes = [PlaneData(np.array([4., 4., 4.]), np.array([0., 0., 1.]))]
    plane = ws.planes[0]
    plane.metrics = stamp_metric(plane, {"area_mm2": [0., 0.], "flowrate_mL_s": [0., 0.], "sample_valid_t": [False, False], "netflow_mL_beat": 0.}, 0)
    ws.derived.plane_metrics = [dict(plane.metrics)]
    report = build_quality_report(ws)
    anchors = issue_anchors(ws, report)
    assert anchors and all(a["point_mm"] == plane.center.tolist() for a in anchors)
    issue = next(c for c in iter_issues(report) if c.get("plane_ids") == [plane.plane_id])
    assert issue["phase_scope"] == "whole_cycle" and issue["scope_kind"] == "plane"
    snapshot = build_report_snapshot(ws)
    assert snapshot["planes"][0]["metrics"]["netflow_mL_beat"] is None
    assert snapshot["planes"][0]["metrics"]["flowrate_mL_s"] == [None, None]
    saved = save_report_html(snapshot, tmp_path/"report.html", sections=["metrics"])
    text = __import__("pathlib").Path(saved).read_text(encoding="utf-8")
    assert "case &lt;phantom&gt;.h5" in text and "Required findings" in text and issue["issue_id"] in text
    assert text.count("yaw:0,pitch:-Math.PI/2") == 2
    invalidate_planes(ws, [plane.plane_id], "ROI edited", roi=True)
    stale = build_quality_report(ws)
    assert next(c for c in stale["checks"] if c["id"] == "hemodynamics.plane_metrics")["status"] == "not_run"
    assert any(a["status"] == "stale" for a in issue_anchors(ws, stale))
    assert build_report_snapshot(ws)["planes"][0]["metrics"] is None
    remove_planes(ws, [plane.plane_id])
    assert not build_report_snapshot(ws)["planes"] and not issue_anchors(ws, build_quality_report(ws))


def test_case_report_phantom_failure_preserves_analysis_and_propagates_cancel(monkeypatch, tmp_path):
    import time
    from autoflow import processing, report_snapshot
    from autoflow.task_control import TaskCancelled

    ws = Workspace()
    ws.data_loaded = True
    ws.mag_raw = np.ones((4, 4, 4, 2), np.float32)
    ws.flow_raw = np.zeros((4, 4, 4, 2, 3), np.float32)
    previous = tmp_path / "analysis_report.html"
    previous.write_text("previous frozen report")
    summary = {"quality_report": build_quality_report(ws), "quality_report_file": "quality_report.json",
               "segmentation_only": True, "segmentation_output": "segmentation.h5"}
    before = ws.flow_raw.copy()

    def fail_save(*args, **kwargs):
        raise PermissionError("output unavailable")

    monkeypatch.setattr(report_snapshot, "save_report_html", fail_save)
    processing._export_case_html_report(ws, summary, str(tmp_path))
    assert summary["analysis_report_status"] == "failed" and not summary["analysis_report_file"]
    assert "PermissionError" in summary["analysis_report_error"]
    assert previous.read_text() == "previous frozen report"
    np.testing.assert_array_equal(ws.flow_raw, before)
    processing._write_case_summary(summary, str(tmp_path / "summary.json"), time.perf_counter(), 0.0)
    assert summary["outputs"]["analysis_report"] == "failed"
    assert summary["output_status"] == "incomplete"

    def cancel_save(*args, **kwargs):
        raise TaskCancelled("cancel report")

    monkeypatch.setattr(report_snapshot, "save_report_html", cancel_save)
    with pytest.raises(TaskCancelled):
        processing._export_case_html_report(ws, summary, str(tmp_path))


@pytest.mark.skipif(sys.platform == "win32", reason="Linux SSH renderer; native Windows GUI is verified by portable_check.py --gui")
def test_roi_transaction_phantom_apply_cancel_and_undo(monkeypatch, tmp_path):
    app = _smoke_qt_application(monkeypatch)
    monkeypatch.setenv("AUTOFLOW_SSH_RENDERING", "1")
    import autoflow.ui.app as ui_app
    from autoflow.ui.remote_plotter import RemotePlotter
    monkeypatch.setattr(ui_app, "RemotePlotter", RemotePlotter, raising=False)
    from autoflow.core.plane_state import stamp_metric, metric_validity
    window = ui_app.MainWindow()
    try:
        ws = window.workspace
        ws.data_loaded = True
        ws.paths.output_dir = str(tmp_path)
        ws.mag_raw = np.ones((8, 8, 8, 2), np.float32)
        ws.flow_raw = np.zeros((8, 8, 8, 2, 3), np.float32)
        ws.segmask_binary = np.ones((8, 8, 8, 2), bool)
        plane = PlaneData(np.array([4., 4., 4.]), np.array([0., 0., 1.]))
        plane.metrics = stamp_metric(plane, {"area_mm2": [10., 10.], "flowrate_mL_s": [0., 0.]}, 0)
        ws.planes = [plane]
        ws.derived.plane_metrics = [dict(plane.metrics)]
        viewer = window.ortho_viewer
        viewer.set_selected_plane(0)
        writes, calculations = [], []
        monkeypatch.setattr(window, "_persist_plane_outputs", lambda **kw: writes.append(kw))
        monkeypatch.setattr(window, "_schedule_plane_calculation", calculations.append)
        monkeypatch.setattr(viewer, "refresh_plane_contours", lambda: None)
        polygon = np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])
        viewer.btn_edit_contour.setChecked(True)
        viewer._commit_contour(polygon, "phantom")
        assert viewer._roi_transaction is not None and metric_validity(plane) == "stale"
        assert not writes and not calculations
        window._force_exit_edit()
        assert not plane.roi_edit_operations and metric_validity(plane) == "current"
        viewer.btn_edit_contour.setChecked(True)
        viewer._commit_contour(polygon, "phantom")
        viewer._apply_plane_roi()
        assert plane.roi_edit_operations and calculations == [0] and plane.roi_revision == 1
        window._undo_plane_change()
        assert not ws.planes[0].roi_edit_operations and metric_validity(ws.planes[0]) == "current"
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()


@pytest.mark.skipif(sys.platform == "win32", reason="Linux SSH renderer; native Windows GUI is verified by portable_check.py --gui")
def test_roi_scope_phantom_current_all_toggle_cancel_and_undo(monkeypatch, tmp_path):
    import copy
    from autoflow.algorithms.metrics import compute_plane_metrics
    from autoflow.core.plane_state import stamp_metric
    import autoflow.ui.app as ui_app
    from autoflow.ui.remote_plotter import RemotePlotter
    monkeypatch.setattr(ui_app, "RemotePlotter", RemotePlotter, raising=False)

    app = _smoke_qt_application(monkeypatch)
    monkeypatch.setenv("AUTOFLOW_SSH_RENDERING", "1")
    window = ui_app.MainWindow()
    try:
        ws = window.workspace
        ws.data_loaded = True
        ws.paths.output_dir = str(tmp_path)
        ws.mag_raw = np.ones((14, 14, 12, 4), np.float32)
        ws.flow_raw = np.zeros((14, 14, 12, 4, 3), np.float32)
        ws.flow_raw[..., 2] = np.arange(1, 5) * 10
        ws.segmask_binary = np.zeros((14, 14, 12, 4), bool)
        ws.segmask_binary[5:8, 5:8, :, 0] = True
        ws.segmask_binary[6:8, 6:8, :, 3] = True
        plane = PlaneData(np.array([6.5, 6.5, 6.3]), np.array([0., 0., 1.]))
        other = PlaneData(np.array([2., 2., 6.3]), np.array([0., 0., 1.]))
        ws.planes = [plane, other]
        def measure():
            return compute_plane_metrics(ws.flow_raw, ws.segmask_binary, ws.resolution, ws.origin, [plane])[0]
        baseline = measure()
        plane.metrics = stamp_metric(plane, baseline, 0)
        ws.derived.plane_metrics = [dict(plane.metrics), {}]
        viewer = window.ortho_viewer
        ws.current_t = 1
        viewer.set_selected_plane(0)
        monkeypatch.setattr(viewer, "refresh_plane_contours", lambda: None)
        writes, calculations = [], []
        monkeypatch.setattr(window, "_persist_plane_outputs", lambda **kw: writes.append(kw))
        monkeypatch.setattr(window, "_schedule_plane_calculation", calculations.append)
        polygon = np.array([[-2.2, -2.2], [2.2, -2.2], [2.2, 2.2], [-2.2, 2.2]])
        assert not viewer.check_roi_all_frames.isChecked()
        viewer.btn_edit_contour.setChecked(True)
        viewer._commit_contour(polygon, "phantom")
        frame_only = copy.deepcopy(plane.roi_edit_operations)
        assert set(frame_only) == {"1"} and frame_only["1"][0]["scope"] == "frame"
        measured = measure()
        assert measured["area_mm2"][1] > baseline["area_mm2"][1]
        for frame in (0, 3):
            assert measured["area_mm2"][frame] == baseline["area_mm2"][frame]
            assert measured["flowrate_mL_s"][frame] == baseline["flowrate_mL_s"][frame]
        assert measured['roi_interpolated_t'][2]
        assert measured['roi_source_frames_t'][2] == [1, 3]
        assert measured["area_mm2"][3] <= measured["area_mm2"][2] <= measured["area_mm2"][1]
        assert measured["roi_source_frame_t"] == [0, 1, 1, 3]
        assert measured["flowrate_mL_s"][2] == pytest.approx(measured["area_mm2"][2] * .3)
        # Playback uses the same scope contract, even when jumping backwards.
        for frame in (3, 1, 2, 0):
            region, source = viewer._effective_plane_region(plane, frame, return_source=True)
            assert source == {0: 0, 1: 1, 2: None, 3: 3}[frame]
            assert region is not None
        viewer.check_roi_all_frames.setChecked(True)
        assert set(plane.roi_edit_operations) == {"0", "1", "2", "3"}
        all_metric = measure()
        np.testing.assert_allclose(all_metric["area_mm2"], all_metric["area_mm2"][0])
        assert all_metric["roi_source_frame_t"] == [0, 1, 2, 3]
        assert all(op[0]["scope"] == "all" for op in plane.roi_edit_operations.values())
        viewer.check_roi_all_frames.setChecked(False)
        assert plane.roi_edit_operations == frame_only
        viewer._undo_plane_roi()
        assert len(plane.roi_edit_operations) == 4
        viewer._redo_plane_roi()
        assert plane.roi_edit_operations == frame_only
        viewer._cancel_plane_roi()
        assert not plane.roi_edit_operations and not calculations and not writes
        viewer.check_roi_all_frames.setChecked(True)
        viewer.btn_edit_contour.setChecked(True)
        viewer._commit_contour(polygon, "phantom")
        viewer._apply_plane_roi()
        assert calculations == [0] and plane.roi_revision == 1
        assert not other.roi_edit_operations
        saved = Workspace()
        saved.restore_dict(ws.snapshot_dict())
        assert saved.planes[0].roi_edit_operations == plane.roi_edit_operations
        window._undo_plane_change()
        assert not ws.planes[0].roi_edit_operations
        window._redo_plane_change()
        assert len(ws.planes[0].roi_edit_operations) == 4
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()


def test_shared_worker_phantom_protocol_restart_and_cancel():
    from autoflow.algorithms.segmentation.shared_worker import SharedInferenceWorker
    worker = SharedInferenceWorker()
    try:
        first = worker.request("ping")
        process = worker.process
        assert worker.request("ping")["pid"] == first["pid"]
        worker.cancel()
        assert process.poll() is not None
        second = worker.request("ping")
        assert second["pid"] != first["pid"]
        with pytest.raises(RuntimeError, match="Unknown inference command"):
            worker.request("unknown")
        assert worker.process is None
    finally:
        worker.close()


def test_shared_normalization_phantom_matches_standard_temporal_channels():
    pytest.importorskip("nnunetv2")
    from types import SimpleNamespace
    from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
    from autoflow.algorithms.segmentation import shared_helpers as shared
    config = SimpleNamespace(normalization_schemes=["ZScoreNormalization"]*37, use_mask_for_norm=[False]*37)
    rng = np.random.default_rng(17)
    sources = {("global", i, -1): rng.normal(size=(5, 6, 7)).astype(np.float32) for i in range(12)}
    for phase in range(3):
        for feature in range(5):
            sources[("temporal", feature, phase)] = rng.normal(loc=phase, size=(5, 6, 7)).astype(np.float32)
    slices, _, nonzero = shared.common_4d_crop_metadata(sources)
    seg = np.where(nonzero, 0, -1).astype(np.int8)[None]
    pre = DefaultPreprocessor(verbose=False)
    properties = {str(i): {} for i in range(37)}
    normalized, _ = shared.normalize_case_source_maps(sources, slices, seg, pre, config, properties)
    by_key = dict(zip(sources, normalized))
    for phase in range(3):
        keys = shared.temporal_source_keys(phase, 3, 2)
        standard = pre._normalize(np.stack([sources[k][slices] for k in keys]), seg, config, properties)
        np.testing.assert_array_equal(standard, np.stack([by_key[k] for k in keys]))


def test_pwv_quality_phantom_retains_fit_and_original_time_resolution():
    from autoflow.algorithms.pwv import _fit_pwv
    fit = _fit_pwv([0, 10, 20, 30], [1, 3, 5, 7])
    assert fit["pwv_m_s"] == pytest.approx(5.)
    assert fit["effective_span_mm"] == 30 and fit["fitted_delay_ms"] == pytest.approx(6.)
    assert fit["fit_rmse_ms"] < 1e-12 and fit["pwv_95ci_m_s"] == pytest.approx([5., 5.])
    assert _fit_pwv([0, 10], [1, 3])["pwv_95ci_m_s"] is None


def test_plane_basis_phantom_continuous_across_reference_axis_boundary():
    pytest.importorskip("PySide6")
    from autoflow.ui.plane_editing import _basis, transport_plane_basis
    normal = np.array([.4, .400001, .8])
    normal /= np.linalg.norm(normal)
    u, v = _basis(normal)
    target = np.array([.400001, .4, .8])
    target /= np.linalg.norm(target)
    new_u, new_v = transport_plane_basis(u, v, target)
    assert np.dot(u, new_u) > .999999
    np.testing.assert_allclose(np.cross(new_u, new_v), target, atol=1e-12)
    canonical_u, canonical_v = _basis(target)
    polygon = np.array([[2., 1.], [-2., 1.], [-2., -1.], [2., -1.]])
    physical = polygon[:, :1]*new_u + polygon[:, 1:]*new_v
    coordinates = np.column_stack((physical@canonical_u, physical@canonical_v))
    np.testing.assert_allclose(coordinates[:, :1]*canonical_u + coordinates[:, 1:]*canonical_v, physical, atol=1e-12)


def test_plane_job_phantom_rejects_old_geometry_and_cross_case_undo():
    pytest.importorskip("PySide6")
    from types import SimpleNamespace
    from autoflow.core.plane_state import mutation_snapshot, plane_signature, workspace_signature
    from autoflow.ui.plane_editing import PlaneEditingMixin
    ws = Workspace()
    ws.planes = [PlaneData(np.array([2., 2., 2.]), np.array([0., 0., 1.]))]
    request = {"plane": __import__("copy").deepcopy(ws.planes[0]), "signature": plane_signature(ws.planes[0]),
               "input_signature": workspace_signature(ws, include_analysis=False)}
    before = mutation_snapshot(ws)
    ws.planes[0].center += [1, 0, 0]
    owner = SimpleNamespace(workspace=ws)
    PlaneEditingMixin._plane_result_ready(owner, {"request": request, "metric": {"netflow_mL_beat": 99.}})
    assert not ws.planes[0].metrics
    other = Workspace()
    messages = []
    owner.workspace, owner.log = other, messages.append
    PlaneEditingMixin._restore_plane_snapshot(owner, before)
    assert not other.planes and messages
    PlaneEditingMixin._plane_result_ready(owner, {"request": request, "metric": {"netflow_mL_beat": 99.}})
    assert not other.derived.plane_metrics


def test_plane_metric_family_phantom_wss_parameter_keeps_current_flow():
    from autoflow.core.plane_state import stamp_metric, invalidate_metric_families, selected_metric, metric_validity
    from autoflow.report_snapshot import build_report_snapshot, render_report_html
    ws = Workspace()
    ws.planes = [PlaneData(np.array([2., 2., 2.]), np.array([0., 0., 1.]))]
    plane = ws.planes[0]
    plane.metrics = stamp_metric(plane, {"flowrate_mL_s": [0., 3.], "netflow_mL_beat": 1.5,
                                       "wss_wall_mean_Pa": 7.}, 0)
    ws.derived.plane_metrics = [dict(plane.metrics)]
    invalidate_metric_families(ws, ["wss"])
    metric = selected_metric(ws, 0)
    assert metric_validity(plane) == "current" and metric["flowrate_mL_s"] == [0., 3.]
    assert metric["wss_wall_mean_Pa"] is None and plane.metrics["wss_wall_mean_Pa"] == 7.
    snapshot = build_report_snapshot(ws)
    assert snapshot["planes"][0]["metrics"]["wss_wall_mean_Pa"] is None
    assert any(c["id"] == "hemodynamics.derived_validity" and c["status"] == "not_run" for c in snapshot["quality"]["checks"])
    required_only = render_report_html(snapshot, sections=[])
    assert "Required findings" in required_only and "Advanced measurements</summary>" not in required_only


def test_plane_pixelwise_phantom_delete_and_undo_revalidate_ids(tmp_path):
    import copy
    import h5py
    from autoflow.core.plane_state import workspace_signature, remove_planes
    from autoflow.algorithms.metrics.export import save_plane_pixelwise_h5, refresh_pixelwise_validity
    ws = Workspace()
    ws.planes = [PlaneData(np.array([2., 2., 2.]), np.array([0., 0., 1.]))]
    original = copy.deepcopy(ws.planes)
    path = tmp_path/"planes.h5"
    payload = [{"plane_index": 0, **ws.planes[0].identity_dict(), "timepoints": []}]
    save_plane_pixelwise_h5(path, payload, input_signature=workspace_signature(ws, include_analysis=False))
    assert refresh_pixelwise_validity(path, ws)
    remove_planes(ws, [ws.planes[0].plane_id])
    assert not refresh_pixelwise_validity(path, ws)
    with h5py.File(path) as handle:
        assert handle["meta"].attrs["validity"] == "stale"
    ws.planes = original
    assert refresh_pixelwise_validity(path, ws)


def test_input_calibration_phantom_named_channels_preview_and_draft(monkeypatch):
    application = _smoke_qt_application(monkeypatch)
    from autoflow.ui.action_settings import ActionSettingsDialog, action_sections
    from autoflow.algorithms.data.input_parameters import normalize_input_parameters
    from autoflow.ui.input_calibration import source_channel_help
    ws = Workspace()
    ws.data_loaded = True
    ws.resolution = np.array([1.234567890123, 1.5, 2.75])
    ws.venc = np.array([100., 150., 220.])
    ws.spatial_order = ["FH", "RL", "AP"]
    ws.venc_order = ["HF", "RL", "AP"]
    ws.input_state.metadata["h5_layout"] = "complex_img"
    panel = ActionSettingsDialog("Reload input", action_sections("input", ws), ws, ws,
                                 run_label="Apply and reload")
    try:
        assert all(not entry.isEditable() for key in ("input.spatial_order", "input.venc_order")
                   for entry in panel.controls[key].entries)
        values = lambda: normalize_input_parameters({key.split(".")[1]: value for key, value in panel.values().items()})
        np.testing.assert_array_equal(values()["resolution"], ws.resolution)
        preview = panel.input_mapping_preview.text()
        assert "Output axis 2 · FH" in preview
        assert "Flow FH · sign −" in preview and "VENC FH = 220 cm/s" in preview
        assert "image slots 1/2/3" in source_channel_help(ws)
        # A stored-channel edit maps to physical calibration; it does not
        # reorder the LR/AP/FH value fields or mutate the loaded workspace.
        panel.controls["input.venc"].entries[2].setText("240.5")
        assert "VENC FH = 240.5 cm/s" in panel.input_mapping_preview.text()
        channel = panel.controls["input.venc_order"].entries[0]
        channel.setCurrentIndex(channel.findData("AP"))
        assert "Mapping unavailable" in panel.input_mapping_preview.text()
        with pytest.raises(ValueError):
            panel.values()
        panel.controls["input.venc_order"].setText("AP, RL, FH")
        assert values()["venc_order"] == ["AP", "RL", "FH"]
        assert "Mapping unavailable" not in panel.input_mapping_preview.text()
        panel.reject()
        np.testing.assert_array_equal(ws.venc, [100.,150.,220.])
        assert ws.venc_order == ["HF", "RL", "AP"]
    finally:
        panel.close()
        panel.deleteLater()
        application.processEvents()


def test_viewer_direction_phantom_plane_first_and_compass_visibility(monkeypatch):
    application = _smoke_qt_application(monkeypatch)
    from autoflow.ui.ortho_viewer import OrthoViewer
    from autoflow.ui.slice_view import anatomical_direction_label, plane_orientation_labels, anatomical_plane_title
    assert anatomical_direction_label([1, 0, 0]) == "R"
    assert anatomical_direction_label([0, -1, 0]) == "A"
    assert anatomical_direction_label([0, 0, -1]) == "F"
    assert anatomical_direction_label([1, -.8, 0]) == "RA"
    assert anatomical_direction_label([0, 0, 0]) == "?"
    assert anatomical_direction_label([np.nan, 0, 0]) == "?"
    assert plane_orientation_labels([0,-1,0],[1,0,0],"U","V") == (
        ("P", "A"), ("L", "R"))
    assert anatomical_plane_title([0,-1,0],[1,0,0]) == "AP × LR"
    assert anatomical_plane_title([1,0,0],[0,0,-1]) == "LR × HF"
    ws = Workspace()
    ws.mag_raw = np.ones((14,14,14,2), np.float32)
    ws.flow_raw = np.zeros(ws.mag_raw.shape+(3,), np.float32)
    ws.planes = [PlaneData(np.array([7.,7.,7.]),np.array([0.,0.,1.]))]
    saved_order = "coronal,sagittal,axial"
    ws.render_settings["viewer_display"] = {"view":{"order":saved_order}, "guides":{"orientation":True}}
    viewer = OrthoViewer(ws)
    try:
        viewer.resize(780,950)
        viewer.show()
        viewer.update_slider_ranges()
        application.processEvents()
        viewer.set_selected_plane(0)
        assert viewer.view_grid.order()[0] == "axial"
        viewer.view_grid.set_order(["sagittal","coronal","axial"])
        assert viewer.view_grid.order()[0] == "axial"
        assert ws.render_settings["viewer_display"]["view"]["order"] == saved_order
        view = viewer.slice_views["axial"]
        assert view.orientation_items["h1"].textItem.toPlainText() == "A"
        assert view.title_label.text() == "AP × LR"
        for ranges in [((3,5),(3,5)), ((50,90),(-30,-20))]:
            view.view_box.setRange(xRange=ranges[0],yRange=ranges[1],padding=0)
            application.processEvents()
            viewport = view.view_box.sceneBoundingRect()
            for item in view.orientation_items.values():
                assert viewport.contains(item.sceneBoundingRect()), (viewport,item.sceneBoundingRect())
        view.view_box.invertX(True)
        view.view_box.invertY(True)
        application.processEvents()
        assert view.orientation_items["h0"].sceneBoundingRect().center().x() > view.view_box.sceneBoundingRect().center().x()
        assert view.orientation_items["v0"].sceneBoundingRect().center().y() < view.view_box.sceneBoundingRect().center().y()
        ws.planes[0].normal = np.array([0.,0.,-1.])
        viewer.refresh()
        assert view.orientation_items["h1"].textItem.toPlainText() == "P"
        viewer.set_selected_plane(None)
        assert ",".join(viewer.view_grid.order()) == saved_order
        assert view.orientation_items["h0"].textItem.toPlainText() == "L"
        viewer.set_selected_plane(0)
        viewer.reset_state()
        assert ",".join(viewer.view_grid.order()) == saved_order
        viewer.set_selected_plane(0)
        ws.planes = []
        viewer.refresh()
        assert ",".join(viewer.view_grid.order()) == saved_order
    finally:
        viewer.close()
        viewer.deleteLater()
        application.processEvents()
def test_flow_controls_phantom_domains_and_physical_grid():
    from autoflow.algorithms.flow_seeding import volume_seed_domain, volume_seed_points, streamline_inputs_available
    flow = np.zeros((4, 6, 3, 2, 3), np.float32)
    flow[0, 0, 0, 0, 0] = np.nan
    segmentation = np.zeros(flow.shape[:4], bool); segmentation[:2] = True
    noise = np.zeros(flow.shape[:3], bool); noise[2:] = True
    domains = {region: volume_seed_domain(flow, region, segmentation=segmentation, noise_keep=noise)
               for region in ('fov', 'segmentation', 'noise_free')}
    assert domains['fov'].sum() == 71
    assert not np.any(domains['segmentation'] & domains['noise_free'])
    assert domains['noise_free'].sum() == 36  # independent of segmentation
    workspace = Workspace(); workspace.flow_raw = flow
    workspace.streamline_params.seed_region = 'fov'
    assert streamline_inputs_available(workspace)
    workspace.streamline_params.seed_region = 'noise_free'
    assert not streamline_inputs_available(workspace)
    workspace.pcmra_render_mask = noise
    assert streamline_inputs_available(workspace)
    with pytest.raises(ValueError, match='unavailable'):
        volume_seed_domain(flow, 'noise_free')
    points = volume_seed_points(np.ones((4, 6, 3), bool), (2., 1., 3.), (11., -7., 9.),
        distribution='uniform', grid_spacing_mm=2., count=1, ratio=.001)
    assert len(points) == 4 * 3 * 4  # grid ignores fixed count / ratio
    np.testing.assert_allclose((points - [11., -7., 9.]) % 2., 1.)
    random = volume_seed_points(domains['fov'], (2., 1., 3.), (11., -7., 9.), mode='fixed', count=7)
    assert len(random) == 7
    np.testing.assert_array_equal(random, volume_seed_points(domains['fov'], (2., 1., 3.),
        (11., -7., 9.), mode='fixed', count=7))


def test_flow_controls_phantom_current_contour_carry_and_segmentation_scope():
    from autoflow.algorithms.flow_seeding import plane_seed_points, plane_basis
    flow = np.zeros((12, 12, 12, 4, 3), np.float32); flow[..., 2] = 10
    plane = PlaneData(np.array([6., 6., 6.]), np.array([0., 0., 1.]))
    plane.roi_edit_operations = {'1': [{'mode': 'replace', 'polygon': [[-2,-2],[2,-2],[2,2],[-2,2]]},
        {'mode': 'add', 'polygon': [[2,-2],[4,-2],[4,2],[2,2]]}]}
    points, source = plane_seed_points(flow, plane, (1,1,1), (13, -9, 7), phase=2,
        region='contour', distribution='uniform', count=1, segmentation=None)
    assert source == 1 and len(points) > 1
    u, v = plane_basis(plane)
    uv = (points - (plane.center + [13, -9, 7])) @ np.column_stack((u,v))
    assert np.any(uv[:, 0] > 2.)  # added area survives exact-polygon clipping
    segmentation = np.zeros(flow.shape[:4], bool); segmentation[4:6,4:6] = True
    seg_points, _ = plane_seed_points(flow, plane, (1,1,1), (13,-9,7), phase=2,
        region='segmentation', distribution='uniform', segmentation=segmentation)
    assert len(seg_points) < len(points)
    assert plane.roi_edit_operations['1'][0]['mode'] == 'replace'


def test_flow_controls_phantom_display_gaps_heads_and_raw_cache_unchanged():
    import pyvista as pv
    from autoflow.algorithms.flow_display import clip_flow_lines, pathline_display
    raw = pv.PolyData(np.column_stack((np.arange(5.), np.zeros((5,2)))), lines=np.r_[5, np.arange(5)])
    raw['Velocity'] = np.array([.2,.2,0.,.2,.2]);raw['PathlinePhase'] = np.linspace(0,1,5)
    original = raw.points.copy()
    visible = clip_flow_lines(raw, 10.)
    assert visible.n_lines == 2
    assert not np.any(np.isclose(visible.points[:,0], 2.))
    assert np.min(visible['Velocity']) >= .1 - 1e-7
    for style, expected_lines in [('particles', 0), ('trails', 1), ('history', 1)]:
        shown = pathline_display(raw, .9, style=style, trail_duration_ms=100, rr_ms=1000)
        assert shown.n_lines == expected_lines
        if style != 'history':
            assert shown.n_verts == 1 and np.min(shown['PathlinePhase']) >= .8 - 1e-7
            assert shown['ParticleAge_ms'][-1] == pytest.approx(900.)
    np.testing.assert_array_equal(raw.points, original)
    assert raw.n_verts == 0 and 'ParticleAge_ms' not in raw.point_data


def test_flow_controls_phantom_spatial_pathline_step_and_cycle_limit():
    from autoflow.algorithms.streamlines import generate_pathlines_from_plane_at_t, create_pathline_temporal_source
    flow = np.zeros((12,12,80,5,3),np.float32);flow[...,2] = [1,2,3,2,1]
    mask = np.ones(flow.shape[:4],bool)
    meshes = [generate_pathlines_from_plane_at_t(flow, 0, SimpleNamespace(), (1,1,1), (0,0,0),
        mask_4d=mask, seeds=np.array([[6.,6.,20.3]]), rr=1000, terminal_speed=.001,
        integration_step_mm=step) for step in (None,.5)]
    for mesh in meshes:
        assert mesh.points[-1,2] == pytest.approx(38.3, abs=.001)
        assert mesh.field_data['CompletedCycle'][0] == 1
    assert meshes[1].n_points > meshes[0].n_points
    assert np.max(np.linalg.norm(np.diff(meshes[1].points,axis=0),axis=1)) <= .501
    short = generate_pathlines_from_plane_at_t(flow,0,SimpleNamespace(),(1,1,1),(0,0,0),
        mask_4d=mask,seeds=np.array([[6.,6.,20.3]]),rr=1000,terminal_speed=.001,
        integration_step_mm=.5,max_steps=10)
    assert short.field_data['CompletedCycle'][0] == 0
    source = create_pathline_temporal_source(flow,mask,(1,1,1),(0,0,0),0,1000.,temporal_cache_mb=0.)
    repeat = generate_pathlines_from_plane_at_t(flow,0,SimpleNamespace(),(1,1,1),(0,0,0),
        mask_4d=mask,seeds=np.array([[6.,6.,20.3]]),rr=1000,terminal_speed=.001,
        integration_step_mm=.5,temporal_source=source)
    assert len(source._datasets) <= 2
    assert repeat.points[-1,2] == pytest.approx(38.3, abs=.001)


def test_flow_controls_phantom_popup_scope_visibility_and_serialization():
    from PySide6 import QtWidgets
    from autoflow.ui.action_settings import ActionSettingsDialog, action_sections
    from autoflow.core.models import StreamlineParams
    from autoflow.config import load_config_bundle, apply_config_bundle_to_workspace
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    ws = Workspace();ws.flow_raw = np.zeros((4,4,4,2,3));ws.mag_raw = np.ones((4,4,4,2))
    assert ws.streamline_params.line_width == pytest.approx(1.0)
    assert ws.streamline_params.tube_radius == pytest.approx(.05)
    ws.streamline_params.seed_region = 'fov'
    dialog = ActionSettingsDialog('Streamlines',action_sections(StepId.GENERATE_STREAMLINES,ws),ws,Workspace())
    try:
        assert [dialog.tabs.tabText(i) for i in range(dialog.tabs.count())] == ['Basic', 'Advanced']
        combo = dialog.controls['streamline_params.seed_region']
        assert combo.count() == 3 and not combo.isEditable()
        assert not combo.model().item(combo.findData('noise_free')).isEnabled()
        distribution = dialog.controls['streamline_params.seed_distribution']
        distribution.setCurrentIndex(distribution.findData('uniform'));app.processEvents()
        assert not dialog.rows['streamline_params.grid_spacing_mm'].isHidden()
        assert dialog.rows['streamline_params.seed_ratio'].isHidden()
        assert 'streamline_seed_info' not in dialog.values()
        width = dialog.controls['streamline_params.line_width']
        radius = dialog.controls['streamline_params.tube_radius']
        style = dialog.controls['streamline_params.display_style']
        assert not width.isHidden() and radius.isHidden()
        label = dialog.forms['streamline_params.line_width'].labelForField(dialog.rows['streamline_params.line_width'])
        assert label.text() == 'Thickness (px)' and 'pixels' in label.toolTip()
        width.setValue(1.25)
        style.setCurrentIndex(style.findData('tubes'));app.processEvents()
        assert width.isHidden() and not radius.isHidden()
        radius.setValue(.7)
        style.setCurrentIndex(style.findData('lines'));app.processEvents()
        assert radius.isHidden() and width.value() == pytest.approx(1.25)
        values = dialog.values()
        assert values['streamline_params.line_width'] == pytest.approx(1.25)
        assert values['streamline_params.tube_radius'] == pytest.approx(.7)
        style.setCurrentIndex(style.findData('tubes'));app.processEvents()
        assert dialog.values()['streamline_params.line_width'] == pytest.approx(1.25)
        dialog.reject()
        assert ws.streamline_params.line_width == pytest.approx(1.0)
        assert ws.streamline_params.tube_radius == pytest.approx(.05)
    finally:
        dialog.close()
    params = StreamlineParams.from_dict(dict(ws.streamline_params.to_dict(), integration_step_mm=.5,
        line_width=2.75,tube_radius=.4,
        pathline_seed_region='contour',pathline_display_style='trails',pathline_trail_duration_ms=120.))
    assert StreamlineParams.from_dict(params.to_dict()).to_dict() == params.to_dict()
    assert params.line_width == pytest.approx(2.75) and params.tube_radius == pytest.approx(.4)
    defaults = Workspace();apply_config_bundle_to_workspace(defaults,load_config_bundle())
    assert defaults.streamline_params.seed_distribution == 'uniform'
    assert defaults.streamline_params.pathline_display_style == 'trails'
    assert defaults.streamline_params.line_width == pytest.approx(1.0)
    assert defaults.streamline_params.tube_radius == pytest.approx(.05)


def test_majority_vote_phantom_streaming_preserves_ties_and_input():
    from autoflow.algorithms.preprocess import majority_vote_labels_3d
    rng = np.random.default_rng(418)
    labels = rng.choice(np.array([-3, 0, 1, 17, 205], np.int16), size=(5, 6, 7, 8))
    labels[0, 0, 0] = np.array([17, 0, 17, 0, 17, 0, 17, 0])
    labels = labels[:, ::-1, ::2, :]
    original = labels.copy()
    values = np.unique(labels)
    expected = values[np.stack([(labels == value).sum(axis=3) for value in values], axis=-1).argmax(axis=-1)]
    np.testing.assert_array_equal(majority_vote_labels_3d(labels), expected)
    np.testing.assert_array_equal(labels, original)
    flat = labels[..., 0]
    copy_result = majority_vote_labels_3d(flat)
    np.testing.assert_array_equal(copy_result, flat)
    assert not np.shares_memory(copy_result, flat)


def test_preprocess_qc_phantom_sparse_support_translation_and_retained_extension():
    from autoflow.algorithms.preprocess_qc import build_segmentation_preprocess_qc
    raw = np.zeros((12, 11, 10, 2), dtype=np.uint8)
    raw[2:4, 3:5, 4:6, :] = 1
    source = raw.copy()
    voted = raw[..., 0].copy()
    retained = voted.copy()
    retained[0, 0, 0] = 1
    processed = retained > 0
    inputs = [raw, source, voted, retained, processed]
    result = build_segmentation_preprocess_qc(*inputs, resolution=np.array([.5, 1., 2.]))
    record = result['labels'][0]
    assert record['input_union_voxels'] == record['voted_voxels'] == 8
    assert record['retained_group_voxels'] == record['skeleton_input_voxels'] == 9
    assert all(phase['vote_overlap_voxels'] == 8 for phase in record['phases'])
    padded = [np.pad(a, [(5, 2), (3, 7), (4, 1)] + ([(0, 0)] if a.ndim == 4 else [])) for a in inputs]
    translated = build_segmentation_preprocess_qc(*padded, resolution=np.array([.5, 1., 2.]))
    assert result['labels'] == translated['labels']


def test_batch_report_phantom_offline_failed_cases_escaping_and_portable_links(tmp_path, monkeypatch):
    import copy
    from html.parser import HTMLParser
    from urllib.parse import unquote
    from autoflow.batch_report_html import save_batch_report_html
    folder = tmp_path / 'batch'
    case = folder / 'case #1 & two'
    case.mkdir(parents=True)
    (case / 'analysis_report.html').write_text('<html>Evidence</html>')
    (case / 'summary.json').write_text('{}')
    canonical = folder / 'batch_report.json'
    canonical.write_text('canonical data must remain unchanged')
    malicious = '<script>alert("case")</script> & detail'
    results = [
        {'case': malicious, 'status': 'ok', 'summary': {
            'total_time_sec': 1.25, 'output_status': 'complete',
            'output_dir': str(case), 'analysis_report_file': str(case / 'analysis_report.html'),
            'quality_report_file': str(case / 'missing.json'),
            'quality_report': {'overall_status': 'needs_review', 'status_counts': {'warn': 2, 'not_run': 1}}}},
        {'case': 'Failed case', 'status': 'error', 'error': '<missing> & failed'},
        {'case': 'Timed out case', 'status': 'timeout', 'error': 'Timed out after 12 s'},
    ]
    before = copy.deepcopy(results)
    output = Path(save_batch_report_html(results, folder / 'batch_report.html'))
    text = output.read_text(encoding="utf-8")
    assert results == before
    assert canonical.read_text() == 'canonical data must remain unchanged'
    assert '<script>alert(' not in text and '&lt;script&gt;alert' in text
    assert '&lt;missing&gt; &amp; failed' in text
    assert 'needs_review' in text and '1 not evaluated' in text
    assert 'Failed case' in text and 'Timed out case' in text
    assert '<noscript>' in text and 'All case rows remain visible' in text
    assert 'https://' not in text and 'http://' not in text
    assert 'missing.json' not in text
    class Links(HTMLParser):
        def __init__(self):
            super().__init__(); self.hrefs = []; self.statuses = []
        def handle_starttag(self, tag, attributes):
            attrs = dict(attributes)
            if tag == 'a': self.hrefs.append(attrs['href'])
            if tag == 'tr' and 'data-status' in attrs: self.statuses.append(attrs['data-status'])
    parsed = Links(); parsed.feed(text)
    assert parsed.statuses == ['ok', 'error', 'timeout']
    assert any('case%20%231%20%26%20two/analysis_report.html' == url for url in parsed.hrefs)
    # Default relative output directories must not get prepended twice.
    relative_results = copy.deepcopy(results)
    relative_results[0]['summary']['output_dir'] = 'batch/case #1 & two'
    relative_results[0]['summary']['analysis_report_file'] = 'batch/case #1 & two/analysis_report.html'
    relative_results[0]['summary']['total_time_sec'] = 10 ** 10000
    with monkeypatch.context() as local:
        local.chdir(tmp_path)
        relative_text = Path(save_batch_report_html(relative_results, 'batch/relative_report.html')).read_text(encoding="utf-8")
    relative_parsed = Links(); relative_parsed.feed(relative_text)
    assert relative_parsed.hrefs == parsed.hrefs
    assert 'Not recorded' in relative_text
    moved = tmp_path / 'moved_batch'
    folder.rename(moved)
    assert all((moved / unquote(url)).is_file() for url in parsed.hrefs)


@pytest.mark.parametrize('overrides, expected', [
    ({}, 'ok'),
    ({'skip_plane_metrics': True, 'n_planes': 0, 'plane_metrics': [],
      'requested_metrics': {'pwv': True}, 'metric_output_status': {'pwv': 'not_assessable'}}, 'ok'),
    ({'requested_metrics': {'wss': True}, 'metric_output_status': {'wss': 'missing'}}, 'partial'),
    ({'requested_videos': {'plane': True}, 'videos': {'plane': ''}}, 'partial'),
    ({'requested_metrics': {'tke': True}, 'metric_output_status': {'tke': 'unsupported'},
      'requested_videos': {'tke': True}, 'videos': {'tke': ''}}, 'ok'),
    ({'quality_report': {'overall_status': 'needs_review'}}, 'ok'),
    ({'analysis_report_status': 'failed'}, 'partial'),
])
def test_batch_outputs_phantom_skip_missing_and_review_are_distinct(tmp_path, overrides, expected):
    import time
    from autoflow.processing import _write_case_summary
    from autoflow.batch_status import case_status
    summary = {'quality_report_file': 'quality_report.json', 'analysis_report_status': 'complete',
               'quality_report': {'overall_status': 'ready'}, 'n_planes': 1, 'plane_metrics': [{}]}
    summary.update(overrides)
    _write_case_summary(summary, tmp_path / 'summary.json', time.perf_counter(), 0)
    assert case_status(summary) == expected
    if summary.get('skip_plane_metrics'):
        assert 'plane_metrics' not in summary['required_outputs']
    if summary.get('metric_output_status', {}).get('tke') == 'unsupported':
        assert summary['outputs']['tke'] == 'unsupported'
        assert 'tke' not in summary['required_outputs']


def test_batch_checkpoint_phantom_report_failure_keeps_all_results(tmp_path, monkeypatch):
    from autoflow.batch_status import checkpoint_batch, record_batch_result
    import autoflow.batch_report_html as report_html
    rows = []
    checkpoint_batch(rows, tmp_path, 2)
    record_batch_result(rows, {'status': 'ok', 'case': 'first'}, tmp_path, 2)
    progress = json.loads((tmp_path / 'batch_progress.json').read_text())
    assert progress['completed_cases'] == 1 and progress['remaining_cases'] == 1
    assert progress['all_requests_complete'] is False
    def fail_report(*args, **kwargs):
        raise OSError('simulated report write failure')
    monkeypatch.setattr(report_html, 'save_batch_report_html', fail_report)
    record_batch_result(rows, {'status': 'partial', 'case': 'second'}, tmp_path, 2)
    progress = json.loads((tmp_path / 'batch_progress.json').read_text())
    assert progress['batch_html_status'] == 'failed'
    assert progress['status_counts'] == {'ok': 1, 'partial': 1}
    assert json.loads((tmp_path / 'batch_report.json').read_text()) == rows


@pytest.mark.parametrize('entrypoint', ['api', 'legacy'])
def test_batch_cancellation_phantom_preserves_finished_cases(tmp_path, monkeypatch, entrypoint):
    from autoflow import api, processing
    from autoflow.task_control import TaskCancelled
    module = api if entrypoint == 'api' else processing
    cases = [SimpleNamespace(input_path=name+'.h5', display_name=name, output_name=name)
             for name in ('first', 'second')]
    monkeypatch.setattr(module, 'collect_input_items', lambda *args, **kwargs: cases)
    monkeypatch.setattr(module, 'build_workspace' if entrypoint == 'api' else 'build_base_workspace',
                        lambda *args, **kwargs: None)
    completed = []
    def run(*args, **kwargs):
        if completed:
            raise TaskCancelled('cancel after first case')
        completed.append('first')
        return {'output_status': 'complete'}
    monkeypatch.setattr(module, 'run_case' if entrypoint == 'api' else 'process_single', run)
    if entrypoint == 'legacy':
        monkeypatch.setattr(processing, 'INPUT', ['dummy'], raising=False)
        monkeypatch.setattr(processing, 'OUTPUT_DIR', str(tmp_path), raising=False)
    with pytest.raises(TaskCancelled):
        if entrypoint == 'api':
            api.run_batch(api.AutoFlowConfig(inputs=['dummy'], output_dir=str(tmp_path)))
        else:
            processing.run_batch()
    rows = json.loads((tmp_path / 'batch_report.json').read_text())
    assert len(rows) == 1 and rows[0]['status'] == 'ok'
    assert (tmp_path / 'batch_report.html').is_file()

def test_temporal_roi_single_source_and_missing_phantom():
    from autoflow.algorithms.metrics import compute_plane_metrics
    labels = np.zeros((12, 12, 8, 4), np.int16)
    labels[4:7, 4:7, :, 2] = 1
    flow = np.zeros(labels.shape + (3,), np.float32)
    flow[..., 2] = [10, 20, 30, 40]
    plane = PlaneData(np.array([5., 5., 4.3]), np.array([0., 0., 1.]), segmentation_label=1)
    metric = compute_plane_metrics(flow, labels > 0, (1, 1, 1), (7, -13, 4), [plane], segmentation_labels_4d=labels)[0]
    assert metric['roi_resolution_t'] == ['single_source_hold', 'single_source_hold', 'observed', 'single_source_hold']
    assert not any(metric['roi_interpolated_t'])
    np.testing.assert_allclose(metric['area_mm2'], [9] * 4)
    np.testing.assert_allclose(metric['flowrate_mL_s'], [.9, 1.8, 2.7, 3.6])
    empty = compute_plane_metrics(flow, np.zeros_like(labels, bool), (1, 1, 1), (0, 0, 0), [plane])[0]
    assert not any(empty['sample_valid_t'])
    assert empty['roi_resolution_t'] == ['missing'] * 4


def test_streamline_auto_grid_phantom_uses_three_spacings_and_keeps_override():
    from autoflow.algorithms.flow_seeding import volume_seed_points
    from autoflow.core.models import StreamlineParams
    from autoflow.config import load_config_bundle
    mask = np.ones((12, 12, 9), bool)
    spacing, origin = np.array([2., 1., 3.]), np.array([11., -7., 9.])
    points = volume_seed_points(mask, spacing, origin, distribution='uniform')
    assert len(points) == 48
    for axis in range(3):
        np.testing.assert_allclose(np.diff(np.unique(points[:, axis])), spacing[axis] * 3)
    np.testing.assert_allclose(points.min(axis=0), origin + 1.5 * spacing)
    explicit = volume_seed_points(mask, spacing, origin, distribution='uniform', grid_spacing_mm=1.)
    assert len(explicit) > len(points)
    assert StreamlineParams().grid_spacing_mm == 0
    assert StreamlineParams.from_dict({'grid_spacing_mm': 2.5}).grid_spacing_mm == 2.5
    assert StreamlineParams().pathline_grid_spacing_mm == 1


def test_case_video_phantom_browser_companion_portability_and_stale_detection(tmp_path):
    from autoflow.rendering.videos import _write_video
    from autoflow.report_media import register_report_videos, report_video_records, ensure_browser_video
    from autoflow.report_snapshot import build_report_snapshot, save_report_html
    import imageio.v2 as imageio
    import base64, re
    ws = Workspace()
    ws.paths.flow_path = 'video-phantom.h5'
    ws.mag_raw = np.ones((8, 8, 8, 2), np.float32)
    ws.flow_raw = np.zeros((8, 8, 8, 2, 3), np.float32)
    ws.segmask_raw = ws.segmask_labels = np.ones((8, 8, 8, 2), np.int16)
    ws.segmask_binary = ws.segmask_raw > 0
    source = tmp_path/'original'/'velocity.mp4'
    source.parent.mkdir()
    frames = [np.full((16, 16, 3), value, np.uint8) for value in (20, 100, 200)]
    _write_video(frames, source, fps=6)
    register_report_videos(ws, {'velocity': source}, {'fps': 6}, ensure_browser=True)
    browser = source.with_suffix('.webm')
    assert browser.is_file()
    before = browser.stat().st_mtime_ns
    assert ensure_browser_video(source) == str(browser.resolve())
    assert browser.stat().st_mtime_ns == before
    reader = imageio.get_reader(browser)
    assert sum(1 for _ in reader) == 3
    reader.close()
    snapshot = build_report_snapshot(ws)
    target = tmp_path/'exported'/'case.html'
    save_report_html(snapshot, target)
    text = target.read_text(encoding='utf-8')
    assert 'video/webm' in text and 'controls loop' in text
    assets = list((target.parent/'case_media').glob('*.webm'))
    assert assets
    import os, stat
    if os.name != 'nt':
        assert all(stat.S_IMODE(asset.stat().st_mode) & 0o444 == 0o444 for asset in assets)
    decoded = json.loads(base64.b64decode(re.search(r'<pre id="snapshot">([A-Za-z0-9+/=]+)</pre>', text).group(1)))
    assert decoded == snapshot  # Asset URLs do not mutate the signed snapshot.
    ws.analysis_revision += 1
    assert next(r for r in report_video_records(ws) if r['family'] == 'velocity')['validity'] == 'stale'


def test_qc_video_phantom_lists_all_computed_families():
    import pyvista as pv
    from autoflow.core.pipeline import PipelineEngine
    from autoflow.report_media import available_video_jobs
    ws = Workspace()
    shape = (6, 6, 6, 2)
    ws.mag_raw = np.ones(shape, np.float32)
    ws.flow_raw = np.zeros(shape + (3,), np.float32)
    ws.segmask_raw = ws.segmask_labels = np.ones(shape, np.int16)
    ws.segmask_binary = ws.segmask_raw > 0
    ws.planes = [PlaneData(np.array([3., 3., 3.]), np.array([0., 0., 1.]))]
    for key in ['wss_volume', 'tke_array', 'pressure_gradient_magnitude', 'pressure_gradient_peak',
                'relative_pressure_array', 'vorticity_magnitude', 'vorticity_magnitude_peak',
                'q_criterion_array', 'q_criterion_peak', 'swirling_strength_array', 'swirling_strength_peak']:
        setattr(ws.derived, key, np.ones(shape, np.float32))
    ws.derived.wss_surfaces = [pv.Sphere()]
    ws.derived.pressure_gradient_array = ws.derived.vorticity_array = np.ones(shape + (3,), np.float32)
    ws.derived.vortex_support_mask = ws.derived.pressure_gradient_support_mask = np.ones(shape, bool)
    ws.derived.artifact_signatures = PipelineEngine()._derived_artifact_signatures(ws)
    ws.streamline_active = True
    ws.pathline_cache = {0: {0: pv.Line()}}
    ws.pcmra_array = np.ones(shape, np.float32)
    assert set(available_video_jobs(ws)) == {'plane', 'wss', 'tke', 'pressure_gradient', 'relative_pressure',
        'vorticity', 'q_criterion', 'swirling_strength', 'velocity', 'streamlines', 'pathlines', 'pcmra'}


def test_current_view_cycle_phantom_restores_camera_phase_and_cancelled_output(tmp_path, monkeypatch):
    from autoflow.ui.view_video import export_current_view
    from autoflow.task_control import TaskCancelled
    import imageio.v2 as imageio
    _smoke_qt_application(monkeypatch)
    ws = Workspace()
    ws.mag_raw = np.ones((4, 4, 4, 3), np.float32)
    ws.flow_raw = np.zeros((4, 4, 4, 3, 3), np.float32)
    ws.current_t = 1
    camera = [(1, 2, 3), (0, 0, 0), (0, 1, 0)]
    shown, resumed = [], []
    plotter = SimpleNamespace(camera_position=camera.copy(), screenshot=lambda **_: np.full((16, 18, 3), 40+60*ws.current_t, np.uint8))
    scene = SimpleNamespace(_interaction_preview_active=False, set_interaction_preview=lambda *a, **k: None,
        update_time=lambda t: shown.append(t), flush_render=lambda: None)
    window = SimpleNamespace(workspace=ws, plotter=plotter, scene=scene,
        _play_timer=SimpleNamespace(isActive=lambda: True), _on_pause=lambda: None,
        _on_play=lambda: resumed.append(True), _refresh_timeline=lambda: None,
        ortho_viewer=SimpleNamespace(refresh=lambda: None))
    path = tmp_path/'view.mp4'
    result = export_current_view(window, path, fps=6, cycles=2)
    assert result['phases'] == [1, 2, 0, 1, 2, 0]
    assert ws.current_t == 1 and plotter.camera_position == camera and resumed
    reader = imageio.get_reader(path);assert sum(1 for _ in reader) == 6;reader.close()
    import os, stat
    if os.name != "nt":
        assert stat.S_IMODE(path.stat().st_mode) & 0o444 == 0o444
    from autoflow.report_media import ensure_browser_video
    browser = Path(ensure_browser_video(path))
    if os.name != "nt":
        assert stat.S_IMODE(browser.stat().st_mode) & 0o444 == 0o444
    reader = imageio.get_reader(browser);assert sum(1 for _ in reader) == 6;reader.close()
    previous = path.read_bytes()
    with pytest.raises(TaskCancelled):
        export_current_view(window, path, fps=6, cycles=2, cancelled=lambda: True)
    assert path.read_bytes() == previous and ws.current_t == 1 and plotter.camera_position == camera


def test_geometry_grid_phantom_keeps_native_velocity_and_roundtrips_workspace():
    from autoflow.algorithms.geometry_grid import grid_shape
    from autoflow.core.models import Workspace
    from autoflow.core.pipeline import PipelineEngine
    ws = Workspace()
    shape = (17, 17, 25)
    x, y, z = np.indices(shape)
    labels = (((x-8)**2+(y-8)**2 < 16) & (z > 2) & (z < 22)).astype(np.int16)
    ws.resolution = np.array([2., 2., 3.])
    ws.flow_raw = np.zeros(shape+(2, 3), np.float32)
    ws.flow_raw[..., 2] = 30.
    flow = ws.flow_raw.copy()
    ws.set_segmentation_source('original', labels)
    ws.activate_segmentation_source('original')
    engine = PipelineEngine()
    assert engine._step_generate_skeleton(ws).success
    assert engine._step_generate_graph(ws).success
    assert np.array_equal(ws.skeleton_spacing, np.ones(3))
    assert ws.skeleton_labels.shape == grid_shape(shape, ws.resolution)
    assert ws.branch_labels.shape == shape
    assert np.array_equal(ws.flow_raw, flow) and ws.segmask_3d.shape == shape
    assert len(ws.centerline_paths) and all(np.isfinite(p).all() for p in ws.centerline_paths)
    restored = Workspace()
    restored.restore_dict(ws.snapshot_dict())
    assert np.array_equal(restored.skeleton_spacing, ws.skeleton_spacing)
    assert np.array_equal(restored.skeleton_labels, ws.skeleton_labels)
    ws.skeleton_params.geometry_spacing_mm = 0.
    assert engine._step_generate_graph(ws).success
    assert np.array_equal(ws.skeleton_spacing, np.ones(3))
    assert ws.skeleton_params.geometry_spacing_mm == 1.
    assert np.array_equal(restored.segmentation_display_4d(), ws.segmentation_display_4d())
    assert np.array_equal(ws.flow_raw, flow)


def test_geometry_cache_phantom_h5_reload_and_edit_invalidation(tmp_path):
    from autoflow.algorithms.geometry_grid import save_geometry_cache, prepare_skeleton_geometry, resample_labels
    from autoflow.algorithms.segmentation import save_segmentation_to_source_h5
    from autoflow.algorithms.data import load_h5_data
    from autoflow.core.models import Workspace
    from autoflow.core.pipeline import PipelineEngine
    shape = (9, 11, 13)
    native = np.zeros(shape+(2,), np.int16)
    native[2:7, 2:7, 2:11] = 1
    spacing = np.array([2.4, 1.2, 3.6])
    labels = resample_labels(native[..., 0], spacing)
    path = tmp_path/'geometry.h5'
    with h5py.File(path, 'w') as f:
        f['mag'] = np.ones(shape+(2,), np.float32)
        f['flow'] = np.zeros(shape+(2, 3), np.float32)
        f['Resolution'] = spacing.astype(np.float32)
        f['Origin'] = np.zeros(3)
        f['VENC'] = np.full(3, 100.)
        f['SpatialOrder'] = np.array(['LR', 'AP', 'FH'], dtype='S4')
        f['VENCOrder'] = np.array(['LR', 'AP', 'FH'], dtype='S4')
    sidecar = tmp_path/'geometry.npz'
    fine4d = np.repeat(labels[..., None], 2, axis=3)
    fine4d[0, 0, 0, 1] = 2
    save_geometry_cache(sidecar, labels, native, spacing, np.zeros(3), 'model_probe', temporal_labels=fine4d)
    save_segmentation_to_source_h5(str(path), native, resolution=spacing, origin=np.zeros(3),
                                  provenance={'geometry_cache_path': str(sidecar)})
    loaded = load_h5_data(str(path), correction_config={'enabled': False})
    assert loaded.metadata['geometry_cache_dataset'] == '/segmask_geometry1mm'
    sidecar.unlink()  # Embedded geometry remains portable without its sidecar.
    ws = Workspace()
    ws.paths.flow_path = str(path)
    ws.resolution = loaded.resolution
    ws.set_segmentation_source('original', loaded.segmentation, {'metadata': loaded.metadata})
    ws.activate_segmentation_source('original')
    engine = PipelineEngine()
    engine.preprocess(ws)
    actual, _, receipt = prepare_skeleton_geometry(ws)
    assert receipt['cache_hit'] and np.array_equal(actual, labels)
    np.testing.assert_array_equal(ws.segmentation_display_4d(), fine4d)
    ws.segmask_raw[4, 4, 4, 0] = 0
    ws.invalidate_segmentation_display()
    engine.preprocess(ws)
    _, _, edited = prepare_skeleton_geometry(ws)
    assert not edited['cache_hit'] and edited['rejected_caches']


@pytest.mark.skipif(sys.platform == "win32", reason="Linux EGL renderer; Windows GUI is checked manually")
def test_generated_plane_phantom_locked_centers_translate_and_preserve_identity(monkeypatch, tmp_path):
    app = _smoke_qt_application(monkeypatch)
    monkeypatch.setenv("AUTOFLOW_SSH_RENDERING", "1")
    from PySide6 import QtCore, QtGui, QtTest, QtWidgets
    import autoflow.ui.app as ui_app
    from autoflow.ui.remote_plotter import RemotePlotter
    monkeypatch.setattr(ui_app, "RemotePlotter", RemotePlotter, raising=False)
    window = ui_app.MainWindow()
    try:
        ws = window.workspace
        ws.data_loaded = True
        ws.paths.output_dir = str(tmp_path)
        ws.mag_raw = np.ones((20, 20, 20, 2), np.float32)
        ws.flow_raw = np.zeros((20, 20, 20, 2, 3), np.float32)
        ws.segmask_binary = np.ones((20, 20, 20, 2), bool)
        ws.resolution = np.array([1., 1.5, 2.])
        ws.origin = np.array([11., -7., 23.])
        ws.ortho_cursor = np.array([8, 8, 8])
        ws.centerline_paths = [np.array([[1., 12., 16.], [18., 12., 16.]])]
        ws.planes = [PlaneData(center=np.array([8., 12., 16.]), normal=np.array([1., 0., 0.]),
            path_index=0, label=1, segmentation_label=7, group_name="artery", distance=7.)]
        original = ws.planes[0]
        original.metrics = {"area_mm2": [10., 10.]}
        window._rebuild_plane_objects()
        window._refresh_browser()
        window._selected_plane_index = 0
        window.ortho_viewer.set_selected_plane(0)
        window._refresh_selection_info()
        window.show()
        QtTest.QTest.qWait(60)
        writes, calculations = [], []
        monkeypatch.setattr(window, "_persist_plane_outputs", lambda **kw: writes.append(kw))
        monkeypatch.setattr(window, "_schedule_plane_calculation", calculations.append)
        assert not window.check_plane_unlocked.isChecked()
        assert window._plane_handles is None

        def center_drag(name, dx=2., dy=1.5):
            view = window.ortho_viewer.slice_views[name]
            viewport = view.plot.viewport()
            def screen(x, y):
                return view.plot.mapFromScene(view.view_box.mapViewToScene(QtCore.QPointF(x, y)))
            start, end = screen(0, 0), screen(dx, dy)
            first = view.view_box.mapSceneToView(view.plot.mapToScene(start))
            last = view.view_box.mapSceneToView(view.plot.mapToScene(end))
            before = ws.planes[0].center.copy()
            u, v = window.ortho_viewer._plane_view_bases(ws.planes[0].normal)[name]
            expected = before+(last.x()-first.x())*u+(last.y()-first.y())*v
            QtTest.QTest.mouseMove(viewport, start)
            QtTest.QTest.mousePress(viewport, QtCore.Qt.LeftButton, pos=start)
            QtTest.QTest.qWait(25)
            move = QtGui.QMouseEvent(QtCore.QEvent.Type.MouseMove, QtCore.QPointF(end),
                QtCore.QPointF(viewport.mapToGlobal(end)), QtCore.Qt.NoButton,
                QtCore.Qt.LeftButton, QtCore.Qt.NoModifier)
            QtWidgets.QApplication.sendEvent(viewport, move)
            QtTest.QTest.mouseRelease(viewport, QtCore.Qt.LeftButton, pos=end)
            QtTest.QTest.qWait(40)
            return expected

        center_drag("axial")
        np.testing.assert_allclose(ws.planes[0].center, original.center)
        window.check_plane_unlocked.setChecked(True)
        assert window._plane_edit_session and window._plane_edit_enabled
        assert not window.check_plane_path_locked.isChecked()
        for name in ("axial", "coronal", "sagittal"):
            expected = center_drag(name)
            np.testing.assert_allclose(ws.planes[0].center, expected, atol=1e-8)
            assert ws.planes[0].path_index == 0 and ws.planes[0].label == 1
            assert ws.planes[0].segmentation_label == 7 and ws.planes[0].group_name == "artery"
            assert ws.planes[0].plane_id == original.plane_id
            assert not writes and not calculations
        moved = ws.planes[0].center.copy()
        window._undo_plane_change()
        assert not np.allclose(ws.planes[0].center, moved)
        window._redo_plane_change()
        np.testing.assert_allclose(ws.planes[0].center, moved)
        window._cancel_plane_edit()
        np.testing.assert_allclose(ws.planes[0].center, original.center)
        assert not window.check_plane_unlocked.isChecked()
        window.check_plane_unlocked.setChecked(True)
        center_drag("axial")
        moved = ws.planes[0].center.copy()
        window.check_plane_unlocked.setChecked(False)  # Save and relock.
        np.testing.assert_allclose(ws.planes[0].center, moved)
        assert window._plane_edit_session is None and window._plane_handles is None
        assert calculations == [0] and len(writes) == 1
        assert all(not view.view_box.plane_translation_enabled for view in window.ortho_viewer.slice_views.values())
        window.check_plane_unlocked.setChecked(True)
        window.check_plane_path_locked.setChecked(True)
        window._update_plane_from_drag(0, center=[10., 14., 18.])
        np.testing.assert_allclose(ws.planes[0].center, [10., 12., 16.])
        window.check_plane_path_locked.setChecked(False)
        window._update_plane_from_drag(0, center=[10., 14., 18.])
        np.testing.assert_allclose(ws.planes[0].center, [10., 14., 18.])
        assert ws.planes[0].segmentation_label == 7 and ws.planes[0].path_index == 0
        window._cancel_plane_edit()
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()


def test_current_view_resolution_phantom_renders_and_restores_on_cancel(monkeypatch, tmp_path):
    _smoke_qt_application(monkeypatch)
    import pyvista as pv
    import imageio.v2 as imageio
    from autoflow.ui.video_options import CurrentViewVideoDialog
    from autoflow.ui.view_video import export_current_view
    from autoflow.task_control import TaskCancelled
    dialog = CurrentViewVideoDialog(None, 8.3)
    assert tuple(dialog.output_resolution()) == (1920, 1080)
    dialog.resolution.setCurrentIndex(0)
    assert dialog.output_resolution() is None
    dialog.resolution.setCurrentIndex(4)
    dialog.width.setValue(321);dialog.height.setValue(181)
    assert dialog.output_resolution() == (321, 181) and dialog.dimensions.isEnabled()
    dialog.deleteLater()
    plotter = pv.Plotter(off_screen=True, window_size=(96, 128))
    try:
        plotter.add_mesh(pv.Sphere(), color="red")
        text = plotter.add_text("Resolution phantom", font_size=8)
        plotter.camera.parallel_projection = True
        plotter.camera.parallel_scale = 1.5
        plotter.show(auto_close=False, interactive=False)
        camera = plotter.camera.copy()
        fonts = [(prop, prop.GetFontSize()) for prop in
                 (text.GetTextProperty(),) if hasattr(text, "GetTextProperty")]
        ws = Workspace()
        ws.mag_raw = np.ones((4, 4, 4, 2), np.float32)
        ws.flow_raw = np.zeros((4, 4, 4, 2, 3), np.float32)
        window = SimpleNamespace(workspace=ws, plotter=plotter,
            scene=SimpleNamespace(_interaction_preview_active=False,
                set_interaction_preview=lambda *a, **k: None, update_time=lambda _: None,
                flush_render=plotter.render), _play_timer=SimpleNamespace(isActive=lambda: False),
            _on_pause=lambda: None, _refresh_timeline=lambda: None,
            ortho_viewer=SimpleNamespace(refresh=lambda: None))
        target = tmp_path/'resolution.mp4'
        result = export_current_view(window, target, fps=6, resolution=(321, 181))
        assert result['resolution'] == [322, 182] and tuple(result['source_resolution']) == (96, 128)
        reader = imageio.get_reader(target)
        frames = [frame for frame in reader];reader.close()
        assert len(frames) == 2 and all(frame.shape[:2] == (182, 322) for frame in frames)
        assert tuple(plotter.window_size) == (96, 128)
        np.testing.assert_allclose(plotter.camera.position, camera.position)
        assert plotter.camera.parallel_scale == camera.parallel_scale
        assert all(prop.GetFontSize() == size for prop, size in fonts)
        previous = target.read_bytes()
        cancel = []
        with pytest.raises(TaskCancelled):
            export_current_view(window, target, fps=6, resolution=(320, 180),
                progress=lambda *args: cancel.append(True), cancelled=lambda: bool(cancel))
        assert target.read_bytes() == previous
        assert tuple(plotter.window_size) == (96, 128) and ws.current_t == 0
        assert plotter.camera.parallel_scale == camera.parallel_scale
        assert all(prop.GetFontSize() == size for prop, size in fonts)
        with pytest.raises(ValueError):
            export_current_view(window, target, fps=6, resolution=(0, 100))
    finally:
        plotter.close()


def _seed_manual_1mm_phantom_state(ws):
    """Seed canonical fields for fixtures that bypass the preprocessing stage."""
    assert np.array_equal(ws.resolution, np.ones(3))
    ws.segmentation_analysis_labels = ws.segmask_labels if ws.segmask_labels is not None else ws.segmask_raw
    ws.segmentation_analysis_labels_3d = ws.segmask_labels_3d
    ws.segmentation_analysis_binary = ws.segmask_binary if ws.segmask_binary is not None else ws.segmask_raw[..., None] > 0
    ws.segmentation_analysis_mask = ws.segmask_3d if ws.segmask_3d is not None else ws.segmask_labels_3d > 0
    ws.geometry_branch_labels = ws.branch_labels
    for state in ws.multilabel_groups.values():
        state['geometry_mask_3d'] = state.get('segmask_3d', ws.segmentation_analysis_mask)
        state['geometry_binary'] = state.get('segmask_binary', ws.segmentation_analysis_binary)


def test_shared_1mm_segmentation_preserves_model_temporal_labels_and_portable_snapshot(tmp_path):
    from autoflow.algorithms.geometry_grid import grid_shape, save_geometry_cache
    ws = Workspace();ws.resolution = np.array([2., 1.5, 2.5]);ws.origin = np.array([11., -7., 23.])
    native = np.zeros((7, 8, 6, 2), np.int16);native[1:6, 1:7, 1:5] = 1
    ws.flow_raw = np.zeros(native.shape+(3,), np.float32)
    fine = np.zeros(grid_shape(native.shape, ws.resolution)+(2,), np.int16)
    fine[2:10, 2:9, 2:10, 0] = 1;fine[3:11, 2:9, 2:10, 1] = 2
    from autoflow.algorithms.preprocess import majority_vote_labels_3d
    cache = tmp_path/'model-grid.npz'
    save_geometry_cache(cache, majority_vote_labels_3d(fine), native, ws.resolution, ws.origin, 'test_model', temporal_labels=fine)
    ws.set_segmentation_source('auto', native, {'geometry_cache_path':str(cache)});ws.activate_segmentation_source('auto')
    np.testing.assert_array_equal(ws.segmentation_display_4d(), fine)
    assert ws.segmentation_geometry['cache_hit']
    snapshot = ws.snapshot_dict();cache.unlink()
    restored = Workspace();restored.restore_dict(snapshot)
    np.testing.assert_array_equal(restored.segmentation_display_4d(), fine)
    np.testing.assert_array_equal(restored.get_active_segmentation(), native)
    np.testing.assert_array_equal(restored.flow_raw, ws.flow_raw)


def test_shared_1mm_plane_sampling_uses_native_physical_velocity_coordinates():
    from autoflow.algorithms.metrics import compute_plane_metrics
    spacing = np.array([2., 1.5, 2.5]);origin = np.array([11., -7., 23.])
    xyz = np.indices((9, 11, 9))*spacing[:,None,None,None]
    flow = np.zeros((9, 11, 9, 2, 3), np.float32)
    flow[..., 2] = (10.+2.*xyz[0]+3.*xyz[1]+4.*xyz[2])[...,None]
    fine = np.zeros((17, 16, 21, 2),bool);fine[4:10, 3:9, 3:18] = True
    plane = PlaneData(center=np.array([6., 6., 9.]), normal=np.array([0.,0.,1.]))
    metric = compute_plane_metrics(flow, fine, np.ones(3), origin, [plane], flow_spacing=spacing)[0]
    # Six-by-six 1-mm cells, centroid (6.5,5.5,9) mm; exact affine velocity.
    expected_cm_s = 10.+2.*6.5+3.*5.5+4.*9.
    np.testing.assert_allclose(metric['area_mm2'], 36., atol=1e-6)
    np.testing.assert_allclose(metric['flowrate_signed_mL_s'], expected_cm_s*36./100., rtol=1e-6)


def test_analytic_phantom_h5_roundtrip_preserves_axes_velocity_and_truth(tmp_path):
    from autoflow.algorithms.analytic_phantoms import SHAPES, create_analytic_phantom, save_analytic_phantom
    from autoflow.algorithms.data import load_h5_data
    data = create_analytic_phantom(SHAPES[3], (.8, 1.2, 2.5))
    path = save_analytic_phantom(tmp_path/'oblique.h5', data)
    loaded = load_h5_data(str(path), correction_config={'enabled':False})
    np.testing.assert_array_equal(loaded.flow, data['flow'])
    np.testing.assert_array_equal(loaded.segmentation, data['segmentation'])
    np.testing.assert_allclose(loaded.resolution, data['resolution'])
    np.testing.assert_allclose(loaded.origin, data['origin'])
    assert loaded.tke_array is None
    with h5py.File(path) as handle:
        np.testing.assert_array_equal(handle['truth/pressure_gradient_pa_m'], data['truth']['pressure_gradient_pa_m'])


def test_label_linear_phantom_competes_background_and_sparse_ids_phasewise():
    from autoflow.algorithms.label_resampling import resample_label_linear
    labels = np.zeros((3, 3, 3, 2), np.int16)
    labels[1, :, :, 0] = 300;labels[2, :, :, 0] = 7
    labels[0, :, :, 1] = 9;labels[1, :, :, 1] = 3
    actual, info = resample_label_linear(labels, (2., 1., 1.), backend='cpu', return_info=True)
    assert actual.shape == (5, 3, 3, 2)
    assert set(np.unique(actual)) <= {0, 3, 7, 9, 300}
    # x=1 ties background/300; x=3 ties 300/7. Lowest class wins.
    assert np.all(actual[1, :, :, 0] == 0)
    assert np.all(actual[3, :, :, 0] == 7)
    assert np.all(actual[1, :, :, 1] == 3)


@pytest.fixture(scope="session")
def regression_matrix(tmp_path_factory):
    from autoflow.algorithms.regression_phantoms import generate_regression_matrix
    directory = tmp_path_factory.getbasetemp()/"regression-inputs"
    rows = generate_regression_matrix(directory, reuse=True)
    assert len(rows) == 24
    return directory


def _nrms(measured, reference):
    measured, reference = np.asarray(measured), np.asarray(reference)
    return float(np.sqrt(np.mean((measured-reference)**2))/max(1e-12, np.sqrt(np.mean(reference**2))))


def _phantom_reference_errors(ws, truth, params, include_hemodynamics=True):
    from scipy.ndimage import binary_erosion
    from autoflow.algorithms.analytic_phantoms import FlowPhantom, analytical_solution, basis
    from autoflow.algorithms.phantom_references import wss_operator_reference
    reference_plane = PlaneData(center=np.asarray(params['center_world_mm'])-ws.origin,
                                normal=np.asarray(params['direction_unit']),label=0)
    from autoflow.algorithms.geometry_grid import measurement_segmentation
    geometry = measurement_segmentation(ws)
    measured = compute_plane_metrics(ws.flow_raw,geometry.binary,geometry.spacing,
                                    ws.origin,[reference_plane],RR=ws.rr,flow_spacing=ws.resolution)[0]
    native = compute_plane_metrics(ws.flow_raw,ws.segmask_raw>0,ws.resolution,ws.origin,
                                   [reference_plane],RR=ws.rr,flow_spacing=ws.resolution)[0]
    qtrue = truth['discrete_native_plane_flowrate_ml_s']
    area_true = float(truth['discrete_native_plane_area_mm2'])
    q,qn = np.asarray(measured['flowrate_signed_mL_s']),np.asarray(native['flowrate_signed_mL_s'])
    flow_errors = dict(flow_nrms=_nrms(q,qtrue),
        area_relative_error=float(abs(np.mean(measured['area_mm2'])-area_true)/area_true),
        native_flow_nrms=_nrms(qn,qtrue),
        native_area_relative_error=float(abs(np.mean(native['area_mm2'])-area_true)/area_true),
        native_reference_area_mm2=area_true,native_reference_flowrate_ml_s=qtrue.tolist(),
        pipeline_area_mm2=float(np.mean(measured['area_mm2'])),pipeline_flowrate_ml_s=q.tolist(),
        physical_flow_nrms=_nrms(q,truth['flowrate_ml_s']),
        physical_area_relative_error=float(abs(np.mean(measured['area_mm2'])-truth['cross_section_area_mm2'])/truth['cross_section_area_mm2']),
        input_physical_sampling_flow_nrms=_nrms(qtrue,truth['flowrate_ml_s']),
        input_physical_sampling_area_relative_error=float(abs(area_true-truth['cross_section_area_mm2'])/truth['cross_section_area_mm2']))
    if not include_hemodynamics:
        return dict(**flow_errors,hemodynamics_requested=False,pg_vector_nrms=None,pressure_nrms=None,
                    physical_pg_vector_nrms=None,physical_pressure_nrms=None,
                    pg_valid_samples=None,wss_vector_nrms=None,wss_magnitude_nrms=None,
                    physical_wss_vector_nrms=None,physical_wss_magnitude_nrms=None,
                    wss_valid_fraction=None,wall_samples=None)
    interior = binary_erosion(ws.segmask_raw[...,0]>0,iterations=2)&truth['pressure_valid_mask']
    support = np.asarray(ws.derived.pressure_gradient_support_mask,bool)&interior[...,None]
    expected = truth['discrete_pressure_gradient_pa_m']
    valid = support&np.isfinite(expected).all(axis=-1)
    pg = _nrms(ws.derived.pressure_gradient_array[valid],expected[valid]) if np.any(valid) else None
    physical_expected = np.broadcast_to(truth['pressure_gradient_pa_m'],expected.shape)
    physical_pg = _nrms(ws.derived.pressure_gradient_array[valid],physical_expected[valid]) if np.any(valid) else None
    pressure_errors,pressure_reference = [],[]
    for phase in range(ws.time_count()):
        region = valid[...,phase]
        if np.count_nonzero(region)>1:
            a = ws.derived.relative_pressure_array[...,phase][region]
            b = truth['pressure_pa'][...,phase][region]
            pressure_errors.extend((a-a.mean()-(b-b.mean())).tolist())
            pressure_reference.extend((b-b.mean()).tolist())
    physical_pressure = None
    if pressure_errors:
        physical_pressure = float(np.sqrt(np.mean(np.square(pressure_errors)))/max(1e-12,np.sqrt(np.mean(np.square(pressure_reference)))))
    spec = FlowPhantom(**{k:v for k,v in params.items() if k in FlowPhantom.__dataclass_fields__})
    direction,axis_u,axis_v = basis(spec)
    numerical_errors,numerical_values,physical_errors,physical_values = [],[],[],[]
    numerical_mag_errors,numerical_magnitudes,physical_mag_errors,physical_magnitudes = [],[],[],[]
    wall_count = wall_valid_count = 0
    for phase,surface in enumerate(ws.derived.wss_surfaces):
        if surface is None or not surface.n_points:
            continue
        points,vectors = np.asarray(surface.points),np.asarray(surface['wss_vectors'])
        relative = points-np.asarray(params['center_world_mm'])
        side = abs(relative@direction)<spec.length_mm/2-8.
        x,y = relative@axis_u,relative@axis_v
        if spec.kind=='ellipse':
            theta = np.arctan2(y/spec.b_mm,x/spec.a_mm)
            tau = 2*spec.mu_pa_s*spec.umax_m_s*np.sqrt((np.cos(theta)/(spec.a_mm*1e-3))**2+(np.sin(theta)/(spec.b_mm*1e-3))**2)
        else:
            shear = analytical_solution(spec,np.array([0.]),np.array([0.]),truth['times_s'])[3]
            inner = np.hypot(x,y)<(spec.a_mm+spec.inner_mm)/2 if spec.kind=='annulus' else np.zeros(len(x),bool)
            tau = np.where(inner,shear[1,phase],shear[0,phase])
        physical = tau[:,None]*direction
        reference = wss_operator_reference(ws.flow_raw[...,phase:phase+1,:],ws.resolution,ws.origin,
            points,np.asarray(surface.point_data['Normals']),float(np.min(ws.resolution)),spec.mu_pa_s*1000.)[:,0,:]
        good = side&np.isfinite(vectors).all(axis=1)&np.isfinite(reference).all(axis=1)
        wall_count += int(side.sum());wall_valid_count += int(good.sum())
        numerical_errors.extend((vectors[good]-reference[good]).ravel().tolist());numerical_values.extend(reference[good].ravel().tolist())
        numerical_mag_errors.extend((np.linalg.norm(vectors[good],axis=1)-np.linalg.norm(reference[good],axis=1)).tolist())
        numerical_magnitudes.extend(np.linalg.norm(reference[good],axis=1).tolist())
        physical_errors.extend((vectors[good]-physical[good]).ravel().tolist());physical_values.extend(physical[good].ravel().tolist())
        physical_mag_errors.extend((np.linalg.norm(vectors[good],axis=1)-np.linalg.norm(physical[good],axis=1)).tolist())
        physical_magnitudes.extend(np.linalg.norm(physical[good],axis=1).tolist())
    def normalized(errors,values):
        return float(np.sqrt(np.mean(np.square(errors)))/max(1e-12,np.sqrt(np.mean(np.square(values))))) if errors else None
    return dict(**flow_errors,hemodynamics_requested=True,pg_vector_nrms=pg,
                pg_valid_samples=int(valid.sum()),pressure_nrms=None,
                physical_pg_vector_nrms=physical_pg,physical_pressure_nrms=physical_pressure,
                wss_vector_nrms=normalized(numerical_errors,numerical_values),
                wss_magnitude_nrms=normalized(numerical_mag_errors,numerical_magnitudes),
                physical_wss_vector_nrms=normalized(physical_errors,physical_values),
                physical_wss_magnitude_nrms=normalized(physical_mag_errors,physical_magnitudes),
                wss_valid_fraction=wall_valid_count/max(1,wall_count),wall_samples=wall_count)


def _phantom_geometry_errors(ws, truth, params, automatic_planes):
    from scipy.spatial import cKDTree
    if not params['truth_type'].startswith('analytic') or not params.get('centerline_applicable', True):
        return dict(centerline_truth_applicable=False)
    center = np.asarray(params['center_world_mm']); direction = np.asarray(params['direction_unit'])
    def radial_error(points):
        points = np.asarray(points, dtype=float).reshape(-1,3)+ws.origin
        relative = points-center; axial = relative@direction
        keep = np.abs(axial) <= params['length_mm']/2-8.
        radius = np.linalg.norm(relative-axial[:,None]*direction, axis=1)
        return float(np.percentile(radius[keep],95)) if np.any(keep) else None
    paths = ws.centerline_paths_smooth or ws.centerline_paths
    path_points = np.vstack(paths) if paths else np.empty((0,3))
    truth_points = truth['centerline_mm']
    distances = cKDTree(path_points+ws.origin).query(truth_points)[0] if len(path_points) else np.full(len(truth_points),np.inf)
    normals = np.array([plane.normal/np.linalg.norm(plane.normal) for plane in automatic_planes])
    angles = np.degrees(np.arccos(np.clip(abs(normals@direction),0.,1.))) if len(normals) else []
    return dict(centerline_truth_applicable=True, skeleton_p95_axis_distance_mm=radial_error(ws.skeleton_points),
                centerline_p95_axis_distance_mm=radial_error(path_points),
                centerline_coverage_within_2mm=float(np.mean(distances<=2.)),
                automatic_plane_normal_p95_deg=float(np.percentile(angles,95)) if len(angles) else None)


@pytest.mark.parametrize("case_name", REGRESSION_CASES)
def test_regression_matrix_runs_pipeline_with_resolution_matched_reference(case_name, regression_matrix, tmp_path):
    from autoflow.config import apply_config_bundle_to_workspace, load_config_bundle
    path = regression_matrix/(case_name+'.h5')
    with h5py.File(path) as source:
        native_shape = source['flow'].shape
        truth = {key:value[()] for key,value in source['truth'].items()}
        params = json.loads(source.attrs['analytic_parameters_json'])
    ws = Workspace()
    apply_config_bundle_to_workspace(ws, load_config_bundle(str(Path(__file__).resolve().parents[1]/'configs')))
    ws.paths.flow_path = str(path); ws.paths.output_dir = str(tmp_path)
    ws.loader_params.background_phase_correction.enabled = False
    ws.skeleton_params.label_map = {'vessel': 1}; ws.skeleton_params.label_groups = {'phantom': {'labels': [1]}}
    ws.derived_params.wss_smoothing_iteration = 80
    ws.derived_params.use_multithread = False
    ws.plane_gen_params.plane_mode = 'distance'; ws.plane_gen_params.cross_section_dist = 20.
    engine = PipelineEngine(); engine.load_data(ws, lambda message: None)
    result = engine._step_generate_skeleton(ws)
    assert result.success, result.message
    if np.allclose(ws.resolution, 1.):
        assert len(ws.skeleton_points), result.message
    # An annular flow domain has a medial sheet rather than an ordinary vessel
    # centerline; use its physically prescribed cross-section for plane metrics.
    if params.get('centerline_applicable', True) and len(ws.skeleton_points):
        for step in (engine._step_generate_graph, engine._step_generate_planes):
            result = step(ws)
            assert result.success, result.message
        assert ws.graph is not None and len(ws.graph.points)
    automatic_planes = list(ws.planes)
    if not ws.planes:
        if 'plane_centers_mm' in truth:
            center, normal = truth['plane_centers_mm'][0], truth['plane_normals'][0]
        else:
            center, normal = params['center_world_mm'], params['direction_unit']
        ws.planes = [PlaneData(center=np.asarray(center)-ws.origin, normal=np.asarray(normal), label=0)]
    include_hemodynamics = np.allclose(ws.resolution, 1.)
    if include_hemodynamics:
        engine._ensure_derived_metrics(ws, save_pixelwise=True, compute_wss=True, compute_tke=True,
                                       compute_pressure_gradient=True, compute_vortex=True)
    metrics, qc, message = engine._compute_plane_metrics_internal(ws, save=True, include_derived=include_hemodynamics, ensure_derived=False)
    assert metrics, message
    engine._save_planes_json(ws)
    from autoflow.processing import _save_pixelwise_npz
    if include_hemodynamics:
        _save_pixelwise_npz(str(tmp_path/'derived_metrics_pixelwise.npz'), ws.derived.pixelwise_export)
    with h5py.File(path) as source:
        for phase in range(ws.time_count()):
            np.testing.assert_array_equal(ws.flow_raw[..., phase, :], source['flow'][..., phase, :])
    np.testing.assert_array_equal(ws.segmentation_spacing, [1., 1., 1.])
    assert ws.segmentation_analysis_binary.shape[3] == native_shape[3]
    if include_hemodynamics:
        assert ws.derived.wss_volume.shape == native_shape[:4]
        assert ws.derived.pressure_gradient_array.shape == native_shape
        assert ws.derived.vorticity_array.shape == native_shape
    else:
        assert ws.derived.wss_volume is None and ws.derived.pressure_gradient_array is None
    assert ws.derived.tke_array is None
    for name in ['plane_metrics.json', 'plane_metrics_pixelwise.h5', 'planes.json', 'planes.h5']:
        assert (tmp_path/name).is_file(), name
    errors = _phantom_reference_errors(ws, truth, params, include_hemodynamics)
    # Measure real automatic sections independently of the known reference
    # section used above to isolate ROI/field sampling. Signed Q is aligned
    # with the prescribed axis; a complete oblique section has A/abs(n.d).
    automatic_flow_errors, automatic_area_errors = [], []
    direction = np.asarray(params['direction_unit'])
    if automatic_planes:
        for plane, measurement in zip(ws.planes, metrics):
            normal = np.asarray(plane.normal)/np.linalg.norm(plane.normal)
            alignment = float(normal@direction)
            if abs(alignment) < 1e-8:
                continue
            from autoflow.algorithms.phantom_references import native_plane_reference
            discrete = native_plane_reference(ws.segmask_raw[...,0]>0,ws.flow_raw,ws.resolution,ws.origin,
                np.asarray(plane.center)+ws.origin,normal)
            raw_q = np.asarray(measurement['flowrate_signed_mL_s'])*measurement.get('forward_sign',1)
            automatic_flow_errors.append(_nrms(raw_q,discrete['flowrate_ml_s']))
            expected_area = discrete['area_mm2']
            automatic_area_errors.append(float(abs(np.mean(measurement['area_mm2'])-expected_area)/expected_area))
    record = dict(reference_contract='resolution_matched_discrete_v1',
                  geometry_truth_scope='analytic_axis_shape_diagnostic_not_unique_discrete_skeleton',
                  automatic_plane_flow_nrms=automatic_flow_errors,
                  automatic_plane_area_relative_error=automatic_area_errors,
                  path_smoothing=ws.plane_gen_params.smoothing_method,
                  case=case_name, truth_type=params['truth_type'], spacing_mm=ws.resolution.tolist(),
                  native_shape=list(native_shape), geometry_shape=list(ws.segmentation_analysis_binary.shape),
                  skeleton_points=len(ws.skeleton_points), automatic_planes=len(automatic_planes),
                  reference_plane_fallback=not bool(automatic_planes),
                  **_phantom_geometry_errors(ws, truth, params, automatic_planes), **errors)
    (tmp_path/'phantom-accuracy.json').write_text(json.dumps(record, indent=2, allow_nan=False))
    # Numerical correctness uses this input's discrete representation.
    # Physical analytic deviations are auxiliary diagnostics, never a demand
    # that a coarser input recover a separately generated finer phantom.
    np.testing.assert_array_equal(ws.segmentation_analysis_mask,truth['discrete_canonical_mask'])
    assert errors['native_flow_nrms'] < 5e-5
    assert errors['native_area_relative_error'] < 5e-5
    # A topology-only upsample must not change the quantitative input ROI.
    assert errors['flow_nrms'] < 5e-5
    assert errors['area_relative_error'] < 5e-5
    if include_hemodynamics:
        assert errors['pg_vector_nrms'] is not None and errors['pg_vector_nrms'] < 5e-4
        assert errors['wss_vector_nrms'] is not None and errors['wss_vector_nrms'] < 5e-5
        assert errors['wss_valid_fraction'] > .9
    assert np.isfinite(errors['flow_nrms'])
    if include_hemodynamics:
        assert 0. <= errors['wss_valid_fraction'] <= 1.


def test_regression_input_matrix_has_24_analytic_cases(regression_matrix):
    manifest = json.loads((regression_matrix/'manifest.json').read_text())
    assert [Path(row['file']).stem for row in manifest['cases']] == list(REGRESSION_CASES)
    assert sum(row['truth_type'].startswith('analytic') for row in manifest['cases']) == 24
    assert len(manifest['cases']) == 24
    for row in manifest['cases']:
        with h5py.File(regression_matrix/row['file']) as source:
            assert source['flow'].shape[-1] == 3
            assert source['segmask'].shape == source['flow'].shape[:-1]
            assert 'pressure_pa' in source['truth'] and 'wall_wss_vectors_pa' in source['truth']
            assert 'tke' not in source


def test_radius_smoothing_phantom_preserves_lumen_displacement_and_junctions():
    from scipy.ndimage import gaussian_filter1d
    from scipy.spatial import cKDTree
    from autoflow.algorithms.path_smoothing import PathSmoothingContext, prepare_centerline
    x = np.arange(4., 36.)
    clean = np.column_stack((x, 13.+2*np.sin(x/5), np.full(len(x),10.)))
    path = clean.copy(); path[:,1] += .35*(-1.)**np.arange(len(x))
    coordinates = np.moveaxis(np.indices((40,28,21)),0,-1)
    mask = cKDTree(clean).query(coordinates)[0] < 2.3
    source_copy = path.copy(); mask_copy = mask.copy()
    context = PathSmoothingContext(mask)
    source, fitted, receipt = prepare_centerline(path,context=context)
    arc = np.r_[0.,np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=1))]
    spacing = arc[-1]/(len(source)-1)
    radius = gaussian_filter1d(np.maximum(context.radii(source),.5),3/spacing,mode='nearest')
    assert receipt['smoothing_method'] == 'radius_adaptive' and receipt['backend'] == 'cpu'
    assert receipt['lumen_guard'] is True
    assert np.all(context.contains(fitted))
    assert np.all(np.linalg.norm(fitted-source,axis=1) <= .15*radius+1e-9)
    np.testing.assert_array_equal(fitted[[0,-1]],path[[0,-1]])
    np.testing.assert_array_equal(path,source_copy); np.testing.assert_array_equal(mask,mask_copy)
    assert receipt['max_displacement_mm'] > .01


def test_radius_smoothing_phantom_move_cannot_cross_background_into_another_vessel():
    from autoflow.algorithms.path_smoothing import PathSmoothingContext
    mask = np.ones((20,12,12),bool); mask[9:11] = False
    context = PathSmoothingContext(mask,spacing=(2.,1.,1.),origin=(11.,-7.,23.))
    source = np.array([[25.,-2.,28.]])
    proposed = np.array([[12.,0.,0.]])
    assert context.contains(source+proposed)[0]
    constrained,fraction = context.constrain(source,proposed)
    assert 0 < fraction[0] < 1
    assert context.contains(constrained)[0] and constrained[0,0] < 29.


def test_radius_smoothing_phantom_cli_api_and_workspace_roundtrip():
    from autoflow.cli import build_parser
    from autoflow.core.models import PlaneGenerationParams
    from autoflow.config import load_config_bundle,bundle_to_autoflow_kwargs
    args = build_parser().parse_args(['input.h5','--path-smoothing','radius_adaptive',
        '--path-smoothing-radius-factor','2','--path-smoothing-max-displacement-ratio','.15',
        '--path-smoothing-step-mm','.5','--path-smoothing-backend','cpu'])
    assert args.path_smoothing_method == 'radius_adaptive'
    kwargs = bundle_to_autoflow_kwargs(load_config_bundle())
    assert kwargs['path_smoothing_radius_factor'] == 2. and kwargs['path_smoothing_max_displacement_ratio'] == .15
    ws = build_workspace(AutoFlowConfig(path_smoothing_radius_factor=3.,path_smoothing_backend='cpu'))
    restored = PlaneGenerationParams.from_dict(ws.plane_gen_params.to_dict())
    assert restored.smoothing_method == 'radius_adaptive' and restored.smoothing_radius_factor == 3.
    assert restored.smoothing_max_displacement_ratio == .15 and restored.smoothing_backend == 'cpu'
    with pytest.raises(ValueError,match='smoothing_max_displacement_ratio'):
        PlaneGenerationParams.from_dict({'smoothing_max_displacement_ratio':1.5})
    with pytest.raises(ValueError,match='smoothing_sample_step_mm'):
        build_workspace(AutoFlowConfig(path_smoothing_sample_step_mm=0.))


def test_radius_smoothing_phantom_plane_builder_default_and_explicit_legacy():
    path = np.column_stack((np.arange(3.,28.),np.full(25,5.),np.full(25,5.)))
    labels = np.zeros((32,12,12),np.int16);labels[1:31,2:9,2:9] = 1
    planes,smooth,qc = generate_planes_from_paths([path],segmentation_labels=labels,
        plane_mode='count',plane_count=3,return_qc=True)
    assert len(planes) == 3 and len(smooth) == 1
    assert qc[0]['smoothing']['smoothing_method'] == 'radius_adaptive'
    assert qc[0]['smoothing']['lumen_guard'] is True
    _,_,legacy = generate_planes_from_paths([path],segmentation_labels=labels,
        smoothing_window=3,return_qc=True)
    assert legacy[0]['smoothing']['smoothing_method'] == 'savgol'


def test_resolution_specific_phantom_reference_uses_native_voxel_area_and_flux():
    from autoflow.algorithms.phantom_references import native_plane_reference,voxel_plane_cut
    from autoflow.algorithms.analytic_phantoms import create_analytic_phantom,SHAPES
    data = create_analytic_phantom(SHAPES[1],(2.,2.,2.))
    # Four native 2x2-mm cells define 16 mm² here; neither pi*3² nor the
    # independently sampled 1-mm tube is the discrete correctness reference.
    assert data['truth']['discrete_native_plane_area_mm2'] == pytest.approx(16.)
    mask = np.ones((1,1,1),bool); flow = np.array([[[[[0.,0.,100.],[0.,0.,-50.]]]]])
    reference = native_plane_reference(mask,flow,(2.,3.,4.),(11.,-7.,23.),(11.,-7.,23.),(0.,0.,1.))
    assert reference['area_mm2'] == pytest.approx(6.)
    np.testing.assert_allclose(reference['flowrate_ml_s'],[6.,-3.])
    cut = voxel_plane_cut(mask,(2.,3.,4.),(0,0,0),(0,0,0),(1.,1.,0.))
    assert sum(cut['areas']) == pytest.approx(8*np.sqrt(2))
    cut = voxel_plane_cut(mask,(2.,2.,2.),(0,0,0),(0,0,0),(1.,1.,1.))
    assert sum(cut['areas']) == pytest.approx(3*np.sqrt(3))


def test_resolution_specific_wss_reference_preserves_units_and_signed_phases():
    from autoflow.algorithms.phantom_references import wss_operator_reference
    x = np.arange(6.)
    profile = (.5*x+.1*x*x)*100.
    flow = np.zeros((6,3,3,2,3));flow[...,0,2] = profile[:,None,None]
    flow[...,1,2] = -.5*profile[:,None,None]
    measured = wss_operator_reference(flow,(1,1,1),(0,0,0),np.array([[0.,1.,1.]]),np.array([[1.,0.,0.]]),1.)
    np.testing.assert_allclose(measured,[[[0.,0.,2.],[0.,0.,-1.]]],atol=1e-12)



def test_external_segmentation_phantom_keeps_native_measurement_cells(tmp_path):
    from autoflow.algorithms.analytic_phantoms import create_analytic_phantom, SHAPES
    from autoflow.algorithms.geometry_grid import measurement_segmentation
    data = create_analytic_phantom(SHAPES[1], (2., 2., 2.))
    ws = Workspace(); ws.resolution = data['resolution']; ws.origin = data['origin']
    ws.flow_raw = data['flow']; ws.mag_raw = data['mag']; ws.rr = data['rr']
    ws.paths.output_dir = str(tmp_path)
    ws.skeleton_params.remove_small_cc = False
    ws.set_segmentation_source('imported', data['segmentation']); ws.activate_segmentation_source('imported')
    engine = PipelineEngine(); engine.preprocess(ws)
    geometry = measurement_segmentation(ws)
    assert geometry.source == 'input_native_labels'
    np.testing.assert_array_equal(geometry.spacing, [2., 2., 2.])
    np.testing.assert_array_equal(geometry.binary, data['segmentation'] > 0)
    center = np.asarray(data['parameters']['center_world_mm']) - ws.origin
    ws.planes = [PlaneData(center=center, normal=np.array([0., 0., 1.]))]
    metric = engine._compute_plane_metrics_internal(ws, save=False, include_derived=False)[0][0]
    np.testing.assert_allclose(metric['area_mm2'], 16., atol=1e-6)
    np.testing.assert_allclose(metric['flowrate_signed_mL_s'], data['truth']['discrete_native_plane_flowrate_ml_s'], rtol=1e-6)
    assert np.sum(ws.segmentation_analysis_mask[..., int(center[2])]) == 9
    # A missing/stale model cache must not silently select interpolated labels.
    ws.segmentation_geometry['cache_hit'] = True
    assert measurement_segmentation(ws).source == 'input_native_labels'



@pytest.mark.parametrize("source", ["auto", "original", "imported"])
@pytest.mark.parametrize("voxel_size", [.5, 2.])
def test_source_1mm_segmentation_phantom_routes_pipeline_and_gui_to_native_velocity(tmp_path, source, voxel_size):
    from autoflow.algorithms.geometry_grid import grid_shape, save_geometry_cache, measurement_segmentation
    from autoflow.algorithms.preprocess import majority_vote_labels_3d
    from autoflow.algorithms.phantom_references import native_plane_reference
    ws = Workspace(); ws.resolution = np.full(3,voxel_size); ws.origin = np.array([11.,-7.,23.])
    ws.paths.output_dir = str(tmp_path); ws.skeleton_params.remove_small_cc = False
    size = int(16/voxel_size)+1
    xyz = np.indices((size,size,size))*voxel_size
    ws.flow_raw = np.zeros((size,size,size,2,3),np.float32)
    ws.flow_raw[...,2] = (10.+2.*xyz[0]+3.*xyz[1]+4.*xyz[2])[...,None]
    native = np.zeros((size,size,size,2),np.int16)
    native[(xyz[0]>=4)&(xyz[0]<=10)&(xyz[1]>=4)&(xyz[1]<=10)&(xyz[2]>=2)&(xyz[2]<=14)] = 1
    # Deliberately different cached model geometry must remain topology-only:
    # no fine-to-native round trip is allowed to replace this native prediction.
    fine = np.zeros(grid_shape(native.shape,ws.resolution)+(2,),np.int16)
    fine[3:12,3:12,2:15] = 1
    cache = tmp_path/'source-model.npz'
    save_geometry_cache(cache,majority_vote_labels_3d(fine),native,ws.resolution,ws.origin,
                        'model_logits_linear_then_classify',temporal_labels=fine)
    ws.set_segmentation_source(source,native,{'geometry_cache_path':str(cache)})
    ws.activate_segmentation_source(source)
    flow_copy = ws.flow_raw.copy()
    engine = PipelineEngine(); engine.preprocess(ws)
    geometry = measurement_segmentation(ws)
    assert geometry.source == ('auto_native_labels' if source=='auto' else 'input_native_labels')
    np.testing.assert_array_equal(geometry.binary,native>0)
    np.testing.assert_array_equal(geometry.spacing,ws.resolution)
    np.testing.assert_array_equal(ws.segmentation_analysis_binary,fine>0)
    np.testing.assert_array_equal(ws.segmentation_spacing,[1.,1.,1.])
    ws.planes = [PlaneData(center=np.array([7.,7.,8.]),normal=np.array([0.,0.,1.]))]
    reference = native_plane_reference(native[...,0]>0,ws.flow_raw,ws.resolution,ws.origin,
                                       np.array([7.,7.,8.])+ws.origin,(0,0,1))
    metric = engine._compute_plane_metrics_internal(ws,save=False,include_derived=False)[0][0]
    np.testing.assert_allclose(metric['area_mm2'],reference['area_mm2'],rtol=1e-6)
    np.testing.assert_allclose(metric['flowrate_signed_mL_s'],reference['flowrate_ml_s'],rtol=1e-6)
    assert metric['roi_geometry_source'] == geometry.source
    np.testing.assert_array_equal(metric['roi_spacing_mm'],ws.resolution)
    np.testing.assert_array_equal(ws.flow_raw,flow_copy)
    from autoflow.ui.ortho_viewer import OrthoViewer
    from types import SimpleNamespace
    inputs = OrthoViewer._analysis_segmentation_inputs(SimpleNamespace(workspace=ws))
    assert inputs[0] is geometry.binary
    np.testing.assert_array_equal(inputs[3],geometry.spacing)


def test_native_measurement_phantom_wss_uses_native_model_output_and_migrates_old_results(tmp_path, monkeypatch):
    from autoflow.algorithms.geometry_grid import grid_shape,save_geometry_cache,QUANTITATIVE_POLICY
    from autoflow.algorithms.preprocess import majority_vote_labels_3d
    import autoflow.core.pipeline as module
    ws = Workspace(); ws.resolution = np.full(3,2.); ws.origin = np.array([11.,-7.,23.])
    ws.paths.output_dir = str(tmp_path); ws.skeleton_params.remove_small_cc = False
    native = np.zeros((9,9,9,2),np.int16);native[2:6,2:6,1:8] = 1
    ws.flow_raw = np.zeros(native.shape+(3,),np.float32)
    fine = np.zeros(grid_shape(native.shape,ws.resolution)+(2,),np.int16);fine[2:13,2:13,1:15] = 1
    cache = tmp_path/'native-model.npz'
    save_geometry_cache(cache,majority_vote_labels_3d(fine),native,ws.resolution,ws.origin,
                        'model_logits_linear_then_classify',temporal_labels=fine)
    ws.set_segmentation_source('auto',native,{'geometry_cache_path':str(cache)})
    ws.activate_segmentation_source('auto')
    engine = PipelineEngine();engine.preprocess(ws)
    calls = []
    def compute(**kwargs):
        calls.append(kwargs)
        return dict(wss_surfaces=[None,None],wss_volume=np.zeros(native.shape,np.float32),
                    pixelwise_export={})
    monkeypatch.setattr(module,'compute_derived_metrics',compute)
    engine._ensure_derived_metrics(ws,compute_wss=True,compute_tke=False,compute_pressure_gradient=False)
    assert len(calls)==1
    np.testing.assert_array_equal(calls[0]['wss_mask4d'],native>0)
    np.testing.assert_array_equal(calls[0]['wss_spacing'],ws.resolution)
    assert calls[0]['flow'] is ws.flow_raw
    ws.planes = [PlaneData(center=np.array([6.,6.,8.]),normal=np.array([0.,0.,1.]))]
    ws.planes[0].metrics = {'area_mm2':[121.,121.],'roi_geometry_source':'source_1mm_geometry',
                           'roi_geometry_policy':'source_segmentation_geometry_v2'}
    plane_id = ws.planes[0].plane_id
    ws.pipeline.mark_done(StepId.COMPUTE_PLANE_METRICS)
    snapshot = ws.snapshot_dict()
    snapshot['segmentation_geometry']['quantitative_policy'] = 'source_segmentation_geometry_v2'
    restored = Workspace();restored.restore_dict(snapshot)
    assert restored.planes[0].plane_id == plane_id
    assert restored.planes[0].metrics['validity'] == 'stale'
    assert StepId.COMPUTE_PLANE_METRICS.value not in restored.pipeline.completed
    assert restored.derived.wss_surfaces == []
    assert restored.segmentation_analysis_binary is None
    restored.paths.output_dir = str(tmp_path/'restored')
    engine.preprocess(restored)
    np.testing.assert_array_equal(restored.segmask_binary,native>0)
    np.testing.assert_array_equal(restored.segmentation_display_4d(),fine)
    assert restored.segmentation_geometry['quantitative_policy'] == QUANTITATIVE_POLICY
    # A manually edited 1-mm working copy is bound once to a native label
    # source; later quantitative preparation must use that native result.
    from autoflow.algorithms.geometry_grid import bind_segmentation_geometry,labels_on_flow_grid,measurement_segmentation
    edited = fine.copy(); edited[2:4] = 0
    expected_native = labels_on_flow_grid(edited,restored)
    bind_segmentation_geometry(restored,'imported',edited,{'source':'manual_edit'},tmp_path)
    restored.activate_segmentation_source('imported')
    engine.preprocess(restored)
    np.testing.assert_array_equal(measurement_segmentation(restored).binary,expected_native>0)
    np.testing.assert_array_equal(restored.segmentation_display_4d(),edited)


def test_quality_cycle_phantom_warns_on_current_undirected_ring_and_locates_it():
    from autoflow.quality import build_quality_report
    from autoflow.qc_locations import issue_anchors, iter_issues
    from autoflow.report_spatial import build_spatial_context
    ws = Workspace()
    ws.resolution = np.ones(3)
    ws.origin = np.array([20., -3., 7.])
    ws.segmask_labels_3d = np.full((8, 8, 8), 13, np.int16)
    ws.graph.points = np.array([[2., 2., 2.], [4., 2., 2.], [4., 4., 2.], [2., 4., 2.]])
    ws.graph.edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
    ws.centerline_node_paths = [[0, 1, 2], [0, 3, 2]]
    ws.centerline_paths = [ws.graph.points[nodes].copy() for nodes in ws.centerline_node_paths]
    ws.path_info = [{"group_name": "renal"}, {"group_name": "renal"}]
    # No directed loop metadata: a forward split/rejoin still needs a warning.
    ws.graph_hierarchy = {"paths": [], "groups": []}
    points_before, edges_before = ws.graph.points.copy(), ws.graph.edges.copy()
    report = build_quality_report(ws)
    check = next(row for row in report["checks"] if row["id"] == "centerline.cycles")
    assert check["status"] == "warn" and check["value"]["cycle_rank"] == 1
    cycle = check["details"][0]
    assert cycle["label_ids"] == [13] and cycle["path_ids"] == [0, 1]
    assert cycle["location_method"] == "graph_cycle_center"
    assert cycle["value"]["length_mm"] == pytest.approx(8.)
    anchor = next(row for row in issue_anchors(ws, report) if row["issue_id"] == cycle["issue_id"])
    np.testing.assert_allclose(anchor["point_mm"], [3., 3., 2.])
    spatial = build_spatial_context(ws, report)
    marker = next(row for row in spatial["markers"] if cycle["issue_id"] in row["issue_ids"])
    assert marker["location_method"] == "graph_cycle_center"
    np.testing.assert_allclose(marker["point_mm"], [3., 3., 2.])
    np.testing.assert_array_equal(ws.graph.points, points_before)
    np.testing.assert_array_equal(ws.graph.edges, edges_before)
    # Unknown/mixed labels still warn, without declaring the ring anatomical error.
    ws.segmask_labels_3d[4:, :, :] = 4
    mixed = next(row for row in build_quality_report(ws)["checks"] if row["id"] == "centerline.cycles")
    assert mixed["status"] == "warn" and mixed["value"]["cross_label_count"] == 1
    ws.graph.edges = ws.graph.edges[:-1]
    tree = next(row for row in build_quality_report(ws)["checks"] if row["id"] == "centerline.cycles")
    assert tree["status"] == "pass" and tree["value"]["cycle_rank"] == 0
    assert not tree["details"]
    ws.graph.points = np.empty((0, 3)); ws.graph.edges = np.empty((0, 2), int)
    missing = next(row for row in build_quality_report(ws)["checks"] if row["id"] == "centerline.cycles")
    assert missing["status"] == "not_run"


@pytest.mark.parametrize('confidence,available,reason', [
    (.3, True, 'flow_C_at_or_above_threshold'),
    (.2, True, 'preserve_component_or_endpoint'),
    (None, False, 'flow_signal_unavailable'),
    (0., False, 'flow_signal_unavailable'),
    (0., True, 'preserve_component_or_endpoint'),
])
def test_cycle_retention_phantom_explains_actual_cleanup_C_and_guard(confidence, available, reason, monkeypatch):
    from autoflow.algorithms.graph_cycles import remove_intralabel_cycles
    from autoflow.quality_cycles import build_cycle_check
    from autoflow.quality import _check
    import autoflow.algorithms.label_planes as label_planes
    from autoflow.core.models import GraphData, Workspace
    graph = GraphData(points=np.array([[2.,2.,2.],[4.,2.,2.],[4.,4.,2.],[2.,4.,2.]]),
                      edges=np.array([[0,1],[1,2],[2,3],[3,0]]))
    labels = np.ones((8,8,8), dtype=np.int16)
    monkeypatch.setattr(label_planes, '_flow_direction', lambda *a, **k: {
        'flow_sign':1, 'flow_confidence':confidence, 'flow_alignment':confidence,
        'flow_direction_status':'reliable' if confidence is not None else 'flow_unavailable',
        'flow_signal_available':available})
    cleaned, audit = remove_intralabel_cycles(graph, labels, flow_threshold=.3)
    assert len(cleaned.edges) == 4
    assert len(audit['path_evaluations']) == 1
    assert audit['path_evaluations'][0]['reason'] == reason
    assert audit['path_evaluations'][0]['direction_confidence'] == confidence
    ws=Workspace();ws.graph=cleaned;ws.segmentation_analysis_labels_3d=labels
    ws.centerline_node_paths=[[0,1,2,3,0]];ws.path_info=[{'group_name':'phantom'}]
    ws.graph_hierarchy={'groups':[{'group_name':'phantom','cycle_cleanup':audit}]}
    check=build_cycle_check(ws,_check);cycle=check['value']['cycles'][0]
    assert check['status']=='warn'
    assert cycle['retention_reasons']==[reason]
    assert cycle['cleanup_units'][0]['C']==(confidence if available else None)
    assert cycle['cleanup_units'][0]['raw_C']==(confidence if not available else None)
    assert cycle['cleanup_units'][0]['threshold']==.3
    assert cycle['cleanup_units'][0]['shared_cycle_edge_count']==4
    assert cycle['cleanup_units'][0]['evidence_source']=='saved_cleanup_decision'


def test_cycle_retention_phantom_includes_cross_label_edges_in_C_filter():
    from autoflow.algorithms.graph_cycles import remove_intralabel_cycles
    from autoflow.quality_cycles import build_cycle_check
    from autoflow.quality import _check
    from autoflow.core.models import GraphData, Workspace
    graph=GraphData(points=np.array([[2.,2.,2.],[4.,2.,2.],[4.,4.,2.],[2.,4.,2.]]),
                    edges=np.array([[0,1],[1,2],[2,3],[3,0]]))
    labels=np.ones((8,8,8),dtype=np.int16);labels[3:]=2
    cleaned,audit=remove_intralabel_cycles(graph,labels)
    assert audit['cross_label_candidate_edge_count']==2
    assert not audit['protected_edges']
    assert audit['path_evaluations'][0]['cross_label']
    ws=Workspace();ws.graph=cleaned;ws.segmentation_analysis_labels_3d=labels
    ws.centerline_node_paths=[[0,1,2,3,0]];ws.path_info=[{'group_name':'phantom'}]
    ws.graph_hierarchy={'groups':[{'group_name':'phantom','cycle_cleanup':audit}]}
    cycle=build_cycle_check(ws,_check)['value']['cycles'][0]
    assert cycle['retention_reasons']==['flow_signal_unavailable']
    assert 'cleanup_audit_unavailable' not in cycle['retention_reasons']
    assert all(row['C'] is None for row in cycle['cleanup_units'] if row['reason']=='cross_label_edge')


def test_plane_rule_strict_phantom_rejects_fallback_and_reports_short_interval():
    from autoflow.algorithms.label_planes import generate_label_planes
    from autoflow.plane_rules import normalize_label_plane_rules
    path=np.column_stack((np.arange(5.,18.),np.full(13,12.),np.full(13,10.)))
    volume,infos=_landmark_phantom([path],[np.full(len(path),2)],[(0,1)],{'MPA':2},
        [{'label_local_levels':{'2':0},'flow_sign':1,'direction_status':'reliable'}])
    spec={'anchor':'label_terminal','path_selection':'longest','position':'before_end','offset_mm':15.}
    rule={'plane_mode':'landmark','plane_count':1,'placements':[spec]}
    planes,_,_,qc=generate_label_planes([path],volume,{'MPA':2},{'MPA':rule},path_info=infos,
        smoothing_window=1,inter_time=1)
    assert not planes and qc[0]['status']=='error'
    omitted=qc[0]['omitted_positions'][0]
    assert omitted['reason']=='offset_exceeds_interval'
    assert omitted['requested_offset_mm']==15 and omitted['interval_length_mm']==12
    spec['fallback']={'anchor':'label_longest','position':'fractions','fractions':[.5]}
    with pytest.raises(ValueError,match='no longer supported'):
        normalize_label_plane_rules({'MPA':rule})


def test_plane_rule_strict_phantom_rejects_conditional_level_substitution():
    from autoflow.plane_rules import normalize_label_plane_rules
    rule={'plane_mode':'landmark','plane_count':1,'placements':[{'anchor':'label_level','level':2,
        'when_connected_to':['LVA','RVA'],'otherwise_level':1,'path_selection':'longest',
        'position':'fractions','fractions':[.5]}]}
    with pytest.raises(ValueError,match='no longer supported'):
        normalize_label_plane_rules({'BA':rule})


def test_pulmonary_trunk_coverage_phantom_warns_for_two_MPA_daughter_planes():
    from autoflow.core.models import Workspace,PlaneData
    from autoflow.quality_placement import build_plane_placement_checks
    from autoflow.quality import _check
    ws=Workspace();ws.planes=[PlaneData(center=np.zeros(3),normal=np.array([1.,0,0]),path_index=i,segmentation_label=2) for i in (1,2)]
    ws.derived.plane_qc={'labels':[{'label_name':'MPA','segmentation_label':2,'status':'ok'}]}
    ws.graph_hierarchy={'paths':[{'path_index':0,'group_name':'pulmonary_arteries','label_names':['MPA'],
        'is_inlet':True,'length_mm':61.}, {'path_index':1,'group_name':'pulmonary_arteries',
        'label_names':['MPA','LPA'],'length_mm':45.}]}
    check=next(row for row in build_plane_placement_checks(ws,_check) if row['id']=='planes.pulmonary_trunk')
    assert check['status']=='warn' and check['value']['candidate_path_indices']==[0]
    assert check['value']['actual_mpa_plane_path_indices']==[1,2]
    ws.planes.append(PlaneData(center=np.zeros(3),normal=np.array([1.,0,0]),path_index=0,segmentation_label=2))
    assert next(row for row in build_plane_placement_checks(ws,_check) if row['id']=='planes.pulmonary_trunk')['status']=='pass'


@pytest.mark.parametrize('fallback', [True,False])
def test_plane_rule_mode_fallback_phantom_warns_without_changing_compatibility(fallback):
    from autoflow.core.models import Workspace
    from autoflow.quality_placement import build_plane_placement_checks
    from autoflow.quality import _check
    ws=Workspace();ws.derived.plane_qc={'requested_plane_mode':'by_label' if fallback else 'fixed_step',
        'plane_mode':'fixed_step','paths':[], 'layout_warnings':['plane_mode_fallback'] if fallback else []}
    check=build_plane_placement_checks(ws,_check)[0]
    assert check['status']==('warn' if fallback else 'pass')
    assert len(check['value']['events'])==int(fallback)
    if fallback:
        assert check['value']['events'][0]['requested_anchor']=='by_label'
        assert check['value']['events'][0]['effective_anchor']=='fixed_step'


@pytest.mark.parametrize("field", ["uniform", "vortex", "stagnant"])
@pytest.mark.parametrize("step", [None, .5])
def test_cuda_streamline_phantom_matches_vtk_voxels_and_closed_mask_faces(field, step):
    try:
        import cupy as cp
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CUDA device unavailable")
    except (ImportError, RuntimeError):
        pytest.skip("CuPy/CUDA unavailable")
    shape = (13, 12, 10)
    spacing, origin = np.array([1.1, 2.3, .9]), np.array([7.25, -12.8, 30.1])
    flow = np.zeros(shape+(1, 3), np.float32)
    if field == "uniform":
        flow[..., 0] = 60.
    elif field == "vortex":
        x, y, z = np.indices(shape)
        flow[..., 0, 0] = -(y-5)*8
        flow[..., 0, 1] = (x-6)*8
        flow[..., 0, 2] = 2.
    mask = np.zeros(shape, bool)
    mask[2:11, 2:10, 1:9] = True
    mask[6, 5:7, :] = False
    seeds = origin + np.array([[3.5, 4.5, 3.5], [8.5, 7.5, 4.5],
                               [2., 3., 2.], [11., 3., 2.]])*spacing
    original = flow.copy()
    meshes = [generate_streamlines_at_t(flow, 0, seeds, spacing, origin, mask,
                integration_backend=backend, integration_step_mm=step, max_steps=60)
              for backend in ("vtk", "cuda")]
    cpu, gpu = meshes
    np.testing.assert_array_equal(cpu.lines, gpu.lines)
    np.testing.assert_allclose(cpu.points, gpu.points, rtol=0, atol=3e-4)
    for key in ("vector", "Velocity", "IntegrationTime"):
        np.testing.assert_allclose(cpu[key], gpu[key], rtol=1e-5, atol=2e-5)
    for key in ("SeedIds", "ReasonForTermination"):
        if cpu.n_cells:
            np.testing.assert_array_equal(cpu[key], gpu[key])
        else:
            assert key not in cpu.cell_data and key not in gpu.cell_data
    assert str(gpu.field_data["IntegrationBackend"][0]) == "cuda"
    np.testing.assert_array_equal(flow, original)


def test_streamline_phantom_cuda_failure_fallback_and_cancellation(monkeypatch):
    from autoflow.algorithms import streamlines_cuda
    from autoflow.task_control import TaskCancelled
    flow = np.zeros((8, 6, 5, 1, 3), np.float32)
    flow[..., 0] = 50.
    args = (flow, 0, np.array([[3.5, 2.5, 2.5]]), np.ones(3), np.zeros(3), np.ones(flow.shape[:3], bool))
    reference = generate_streamlines_at_t(*args, integration_backend="vtk")
    def fail(*args, **kwargs):
        raise MemoryError("phantom VRAM exhaustion")
    monkeypatch.setattr(streamlines_cuda, "trace_sampled_volume", fail)
    result = generate_streamlines_at_t(*args, integration_backend="auto")
    np.testing.assert_array_equal(reference.points, result.points)
    assert result.field_data["IntegrationBackend"][0] == "vtk"
    assert "phantom VRAM exhaustion" in result.field_data["IntegrationFallbackReason"][0]
    with pytest.raises(RuntimeError, match="CUDA streamline integration failed"):
        generate_streamlines_at_t(*args, integration_backend="cuda")
    with pytest.raises(ValueError, match="Integration backend"):
        generate_streamlines_at_t(*args, integration_backend="invalid")
    def cancel(*args, **kwargs):
        raise TaskCancelled("Cancelled by user")
    monkeypatch.setattr(streamlines_cuda, "trace_sampled_volume", cancel)
    with pytest.raises(TaskCancelled):
        generate_streamlines_at_t(*args, integration_backend="auto")


def test_retired_radius_routing_phantom_old_settings_are_ignored():
    from dataclasses import fields
    from autoflow.core.models import SkeletonParams
    from autoflow.cli import build_parser
    from autoflow.ui.action_settings import action_sections
    legacy = {'radius_tree_policy': 'forest', 'label_groups': {'vessel': {
        'labels': [1], 'radius_tree_policy': 'forest', 'preprocess': {'radius_tree_policy': 'auto'}}}}
    params = SkeletonParams.from_dict(legacy)
    assert 'radius_tree_policy' not in params.to_dict()
    assert 'radius_tree_policy' not in params.label_groups['vessel']
    assert 'radius_tree_policy' not in params.label_groups['vessel']['preprocess']
    assert not hasattr(params.params_for_group('vessel'), 'radius_tree_policy')
    assert 'radius_tree_policy' not in {f.name for f in fields(AutoFlowConfig)}
    parser = build_parser()
    assert '--radius-tree-policy' not in parser.format_help()
    ws = Workspace()
    assert all('radius_tree_policy' not in field.key
               for _, section in action_sections(StepId.GENERATE_GRAPH, ws) for field in section)


@pytest.mark.parametrize("fps,update_seconds,expected_delay", [(10., .02, 80), (25., .01, 30), (100., .02, 0), (.5, .003, 1997)])
def test_playback_fps_phantom_schedules_target_without_overdue_frames(monkeypatch, fps, update_seconds, expected_delay):
    from collections import deque
    from types import SimpleNamespace
    import autoflow.ui.app as ui_app
    clock = iter((5., 5.+update_seconds))
    monkeypatch.setattr(ui_app.time, "perf_counter", lambda: next(clock))
    delays, phases, labels = [], [], []
    state = SimpleNamespace(
        workspace=SimpleNamespace(current_t=1, time_count=lambda: 3),
        spin_fps=SimpleNamespace(value=lambda: fps),
        scene=SimpleNamespace(update_time=phases.append),
        ortho_viewer=SimpleNamespace(refresh=lambda **kw: None),
        _refresh_timeline=lambda: None, _refresh_render_range_control=lambda: None,
        _playback_last_tick_started_at=None, _playback_frame_times_ms=deque(maxlen=24),
        _playback_render_times_ms=deque(maxlen=24),
        lab_playback_performance=SimpleNamespace(setText=labels.append),
        _play_timer=SimpleNamespace(isSingleShot=lambda: True, start=delays.append))
    ui_app.MainWindow._on_play_tick(state)
    assert phases == [2] and state.workspace.current_t == 2
    assert delays == [expected_delay]
    assert len(state._playback_render_times_ms) == 1 and labels



def test_ortho_playback_phantom_retains_cycle_windows_and_refreshes_display_settings(monkeypatch):
    from types import MethodType, SimpleNamespace
    from autoflow.ui.ortho_viewer import OrthoViewer
    import autoflow.ui.viewer_display as display
    ws = Workspace()
    ws.mag_raw = np.arange(3*4*5*20, dtype=np.float32).reshape(3,4,5,20)+1
    ws.flow_raw = np.zeros((3,4,5,20,3),np.float32)
    viewer = SimpleNamespace(workspace=ws, _cache={})
    viewer._cached = MethodType(OrthoViewer._cached, viewer)
    viewer._array_cache_token = OrthoViewer._array_cache_token
    calls = []
    def levels(image, fallback):
        calls.append(True)
        return OrthoViewer._auto_image_levels(image, fallback)
    viewer._auto_image_levels = levels
    first = []
    for phase in range(20):
        for fallback in ((0.,2000.),(10.,1500.)):
            first.append(OrthoViewer._cached_image_levels(viewer, ws.mag_raw[...,phase], fallback))
    assert len(calls) == 40
    second = []
    for phase in range(20):
        for fallback in ((0.,2000.),(10.,1500.)):
            second.append(OrthoViewer._cached_image_levels(viewer, ws.mag_raw[...,phase], fallback))
    assert len(calls) == 40 and first == second
    # Simulate allocator address reuse for a different image owner.
    viewer._array_cache_token = lambda value: "reused-address"
    low = np.arange(60,dtype=np.float32).reshape(3,4,5)+1
    high = low+1000
    low_window = OrthoViewer._cached_image_levels(viewer, low, (0.,2000.))
    high_window = OrthoViewer._cached_image_levels(viewer, high, (0.,2000.))
    assert high_window[0] > low_window[1]
    viewer.combo_content = SimpleNamespace(currentData=lambda: 3)
    viewer.display_state = MethodType(display.OrthoDisplayMixin.display_state, viewer)
    original = display.view_display
    merges = []
    def counted(*args, **kwargs):
        merges.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(display, "view_display", counted)
    def paint(update_plane):
        states = [viewer.display_state(name) for name in (None, "axial", None, "axial")]
        assert states[0] == states[2] and states[1] == states[3]
        return states[0]["image"]["cmap"]
    viewer._refresh_views = paint
    ws.render_settings["viewer_display"] = {"image": {"cmap": "gray"}}
    assert OrthoViewer.refresh(viewer) == "gray" and len(merges) == 2
    assert viewer._refresh_display_states is None
    ws.render_settings["viewer_display"]["image"]["cmap"] = "viridis"
    assert OrthoViewer.refresh(viewer) == "viridis" and len(merges) == 4
    def fail(update_plane):
        viewer.display_state()
        raise RuntimeError("phantom repaint failure")
    viewer._refresh_views = fail
    with pytest.raises(RuntimeError, match="phantom repaint failure"):
        OrthoViewer.refresh(viewer)
    assert viewer._refresh_display_states is None


def test_ortho_playback_phantom_sample_cache_preserves_pixels_and_edit_invalidation(monkeypatch):
    from types import SimpleNamespace
    from autoflow.core.cache import ByteLRUCache
    from autoflow.ui.ortho_viewer import OrthoViewer
    import autoflow.ui.ortho_viewer as ortho
    ws = Workspace()
    ws.current_t = 2
    res = np.array([1.1, 2.3, .9])
    viewer = SimpleNamespace(workspace=ws, _playback_active=True,
        _get_resolution=lambda: res, _array_cache_token=OrthoViewer._array_cache_token,
        _axis_sample_cache=ByteLRUCache(64*1024**2))
    source = np.arange(8*9*10, dtype=np.float32).reshape(8,9,10)
    original = source.copy()
    sampler = ortho.map_coordinates
    calls = []
    def counted(*args, **kwargs):
        calls.append(True)
        return sampler(*args, **kwargs)
    monkeypatch.setattr(ortho, "map_coordinates", counted)
    for name in ("axial", "coronal", "sagittal"):
        args = (source,name,2,(5,6))
        result = OrthoViewer._axis_geometry_sample(viewer,*args,source_spacing=res,order=1)
        again = OrthoViewer._axis_geometry_sample(viewer,*args,source_spacing=res,order=1)
        assert result is again
        viewer._playback_active = False
        reference = OrthoViewer._axis_geometry_sample(viewer,*args,source_spacing=res,order=1)
        np.testing.assert_array_equal(result,reference)
        viewer._playback_active = True
    assert len(calls) == 6
    source *= 2
    ws.segmentation.revision += 1
    changed = OrthoViewer._axis_geometry_sample(viewer,source,"axial",2,(5,6),source_spacing=res,order=1)
    viewer._playback_active = False
    reference = OrthoViewer._axis_geometry_sample(viewer,source,"axial",2,(5,6),source_spacing=res,order=1)
    np.testing.assert_array_equal(changed,reference)
    np.testing.assert_array_equal(source,original*2)
    assert len(calls) == 8


@pytest.mark.parametrize("selected_plane", [False, True])
def test_gui_slice_phantom_reuses_presentation_and_updates_changed_data(monkeypatch, selected_plane):
    app = _smoke_qt_application(monkeypatch)
    import copy
    from autoflow.ui.ortho_viewer import OrthoViewer
    import autoflow.ui.ortho_viewer as ortho
    ws = Workspace()
    ws.mag_raw = np.arange(10*11*12*3,dtype=np.float32).reshape(10,11,12,3)+1
    ws.flow_raw = np.zeros(ws.mag_raw.shape+(3,),np.float32)
    ws.flow_raw[...,0] = 40.
    ws.resolution = np.array([1.1,1.7,.9])
    ws.origin = np.array([5.,-8.,12.])
    ws.segmask_raw = np.ones(ws.mag_raw.shape,np.int16)
    ws.segmask_raw[5:,...] = 2
    ws.planes = [PlaneData(np.array([5.,7.,5.]),np.array([1.,1.,1.]),path_index=-1)]
    viewer = OrthoViewer(ws)
    try:
        viewer.update_slider_ranges()
        if selected_plane:viewer.set_selected_plane(0)
        viewer.set_playback_active(True)
        viewer.refresh(update_plane=False)
        viewer.open_display_settings()
        panel = viewer._display_panel
        lut_calls,overlay_calls,panel_calls = [],[],[]
        overlay = ortho.make_label_overlay
        def counted_overlay(*args,**kwargs):
            overlay_calls.append(True)
            return overlay(*args,**kwargs)
        monkeypatch.setattr(ortho,"make_label_overlay",counted_overlay)
        panel_set = panel.set_state
        def counted_panel(*args,**kwargs):
            panel_calls.append(True)
            return panel_set(*args,**kwargs)
        monkeypatch.setattr(panel,"set_state",counted_panel)
        for view in viewer.slice_views.values():
            setter = view.image_item.setLookupTable
            def counted_lut(*args,_setter=setter,**kwargs):
                lut_calls.append(True)
                return _setter(*args,**kwargs)
            monkeypatch.setattr(view.image_item,"setLookupTable",counted_lut)
        original = {name:np.asarray(view.image_item.image).copy() for name,view in viewer.slice_views.items()}
        for _ in range(3):viewer.refresh(update_plane=False)
        assert not lut_calls and not overlay_calls and not panel_calls
        # Per-view style changes must apply even when pixels remain cached.
        state = copy.deepcopy(ws.render_settings['viewer_display'])
        state['views']['coronal'] = {'image':{'cmap':'viridis'}}
        state['segmentation']['colors']['1'] = '#00ff00'
        viewer.apply_display_settings(state)
        assert lut_calls and overlay_calls and panel_calls
        for name,view in viewer.slice_views.items():
            np.testing.assert_array_equal(view.image_item.image,original[name])
        lookup = np.asarray(viewer.slice_views['coronal'].image_item.lut).copy()
        # Full refreshes rebuild image geometry, retaining the per-view LUT.
        viewer.refresh(update_plane=True)
        np.testing.assert_array_equal(viewer.slice_views['coronal'].image_item.lut,lookup)
        viewer.set_playback_active(False)
        ws.mag_raw *= 2
        viewer.refresh(update_plane=True)
        for name,view in viewer.slice_views.items():
            np.testing.assert_allclose(view.image_item.image,original[name]*2,atol=1e-4)
        assert viewer._overlay_rgba_cache.bytes_used <= viewer._overlay_rgba_cache.max_bytes
    finally:
        if viewer._display_dock:viewer._display_dock.close()
        viewer.close();viewer.deleteLater();app.processEvents()


def test_measurements_phantom_curve_parameters_preserve_metrics_and_relocate_consistency(monkeypatch):
    """Display interpolation cannot invalidate metrics; IC stays in selection/QC."""
    _smoke_qt_application(monkeypatch)
    import copy
    from PySide6 import QtCore
    from autoflow.ui.app import _ANALYSIS_MODE_ITEMS
    from autoflow.ui.action_settings import ActionSettingsMixin, action_sections
    from autoflow.ui.measurement_summary import MeasurementSummary
    from autoflow.ui.selection_info import consistency_rows
    from autoflow.ui.pinned_metrics import DEFAULT_PINNED_METRICS, pinned_metric_keys, pinned_metric_rows

    ws = Workspace()
    ws.derived.plane_metrics = [{"path_index": 0, "flowrate_mL_s": [1., 3., 2.],
        "flowrate_signed_mL_s": [-1., -4., 2.], "meanv_cm_s": 2., "peakv_cm_s": 5.,
        "netflow_mL_beat": 8., "area_mm2": [10., 12., 14.], "sample_valid_t": [True, True, True]}]
    ws.derived.plane_qc = {"path_ic": {"0": .8}, "forks": [
        {"fork_id": 4, "left": [0], "right": [1], "ic": None}]}
    metrics = copy.deepcopy(ws.derived.plane_metrics)
    qc = copy.deepcopy(ws.derived.plane_qc)
    revision = ws.analysis_revision
    refreshes = []
    window = SimpleNamespace(workspace=ws, _refresh_analysis_panel=lambda: refreshes.append(True))
    key = "render_settings.measurements.flowrate_spline_order"
    fields = [field for _, section in action_sections("measurements", ws) for field in section]
    assert [(field.key, field.default, field.minimum, field.maximum) for field in fields if field.kind == "int"] == [(key, 3, 1, 3)]
    pinned_key = "render_settings.measurements.pinned_metrics"
    pinned_field = next(field for field in fields if field.key == pinned_key)
    assert pinned_field.kind == "checklist" and pinned_field.default == list(DEFAULT_PINNED_METRICS)
    assert pinned_metric_keys(ws) == list(DEFAULT_PINNED_METRICS)
    expected = [("Mean velocity", "2", "cm/s"), ("Peak velocity", "5", "cm/s"),
                ("Peak flow", "4", "mL/s"), ("Net flow", "8", "mL/beat")]
    assert pinned_metric_rows(metrics[0], DEFAULT_PINNED_METRICS) == expected
    assert all(value == "—" for _, value, _ in pinned_metric_rows(
        dict(metrics[0], sample_valid_t=[True, False, True]), DEFAULT_PINNED_METRICS))
    assert pinned_metric_rows({}, ["mean_wss"]) == [("Mean WSS", "—", "Pa")]
    assert pinned_metric_rows({"meanv_cm_s": 0}, ["mean_velocity"]) == [("Mean velocity", "0", "cm/s")]
    from autoflow.ui.action_settings import ActionSettingsDialog
    dialog = ActionSettingsDialog("Measurements", action_sections("measurements", ws), ws, ws)
    checklist = dialog.controls[pinned_key]
    chosen = {checklist.item(index).data(QtCore.Qt.UserRole) for index in range(checklist.count())
              if checklist.item(index).checkState() == QtCore.Qt.Checked}
    assert chosen == set(DEFAULT_PINNED_METRICS)
    values = dialog.values()
    assert values[pinned_key] == list(DEFAULT_PINNED_METRICS)
    dialog.deleteLater()
    assert key in {field.key for _, section in action_sections(StepId.COMPUTE_PLANE_METRICS, ws) for field in section}
    ActionSettingsMixin._publish_action_settings(window, {key: 1, pinned_key: ["mean_area", "mean_wss"]})
    assert pinned_metric_keys(ws) == ["mean_area", "mean_wss"]
    assert ws.render_settings["measurements"]["flowrate_spline_order"] == 1
    assert refreshes and ws.analysis_revision == revision
    assert ws.derived.plane_metrics == metrics and ws.derived.plane_qc == qc
    assert not ws.input_state.metadata.get("gui_pending_settings")
    assert "internal_consistency" not in {value for _, value in _ANALYSIS_MODE_ITEMS}
    assert consistency_rows(ws, 0) == [("Path IC", "0.800", ""), ("Fork 4 IC", "undefined", "")]
    assert consistency_rows(ws, 99) == [("Path IC", "undefined", "")]
    panel = MeasurementSummary()
    panel.set_measurement("Plane 0", "Phase 1", "3", "mL/s", [("Min", "1"), ("Mean", "2"), ("Max", "3")])
    assert panel.value.text() == "3" and panel.unit.text() == "mL/s"
    assert not panel.reading.isHidden()
    panel.set_pinned_metrics("Cycle summary", expected)
    labels = list(panel.pinned_labels)
    panel.set_measurement("Plane 0", "Phase 1", "10", "Pa", [("Min", "1")])
    panel.set_pinned_metrics("Cycle summary", expected)
    assert panel.pinned_labels == labels and [label.text() for label in labels] == ["2", "5", "4", "8"]
    assert "Peak flow: 4 mL/s" in panel.text()
    panel.setText("No plane selected.")
    assert panel.reading.isHidden() and panel.text() == "No plane selected."
    panel.deleteLater()


@pytest.mark.skipif(sys.platform != "linux", reason="Remote Linux HTML/VTK integration smoke")
def test_html_document_phantom_native_process_survives_docs_qc_and_progress(tmp_path):
    import importlib.util
    import os
    import subprocess
    for module in ("PySide6.QtWebEngineWidgets", "pyvistaqt", "pyqtgraph"):
        try:
            present = importlib.util.find_spec(module) is not None
        except ModuleNotFoundError:
            present = False
        if not present:
            pytest.skip(module + " is unavailable")
    page = tmp_path / "smoke.html"
    page.write_text("<html><body><h1>HTML smoke</h1><script>document.body.dataset.ready='yes'</script></body></html>")
    script = r'''
import sys, time
from PySide6 import QtCore, QtWidgets, QtTest
from autoflow.ui.html_view import configure_html_runtime, open_html_document
configure_html_runtime()
from autoflow.ui.app import MainWindow
from autoflow.ui.progress import TaskProgressDialog
app = QtWidgets.QApplication([])
app.setQuitOnLastWindowClosed(False)
w = MainWindow()
w.show()
def ready(pane):
    result = []
    pane.view.page().runJavaScript("document.body.dataset.ready", lambda value: result.append(value))
    deadline = time.monotonic() + 10
    while not result and time.monotonic() < deadline:
        app.processEvents()
        QtTest.QTest.qWait(20)
    return result == ['yes']
try:
    assert not hasattr(w, 'document_tabs')
    for _ in range(2):
        pane = open_html_document(w, sys.argv[1], 'Docs', 'docs')
        deadline = time.monotonic() + 10
        while not ready(pane) and time.monotonic() < deadline:
            QtTest.QTest.qWait(30)
        assert ready(pane)
        assert w._html_document_dialogs['docs'].isVisible()
        assert w.workspace_pages.currentWidget() is w.workspace_page
        w._html_document_dialogs['docs'].close()
        app.processEvents()
    pane = open_html_document(w, sys.argv[1], 'QC', 'qc')
    deadline = time.monotonic() + 10
    while not ready(pane) and time.monotonic() < deadline:
        QtTest.QTest.qWait(30)
    assert ready(pane)
    assert w.workspace_pages.currentWidget() is w.qc_page
    progress = TaskProgressDialog('Check', 'HTML and calculation coexistence', w)
    progress.show()
    QtTest.QTest.qWait(100)
    progress.finish()
    app.processEvents()
    assert ready(pane)
    w._on_open_export_page()
    assert w.workspace_pages.currentWidget() is w._export_page
    print('HTML_SMOKE_OK')
finally:
    w.close()
    w.deleteLater()
    app.processEvents()
'''
    environment = dict(os.environ, QT_QPA_PLATFORM="offscreen", AUTOFLOW_SSH_RENDERING="1",
                       VTK_DEFAULT_OPENGL_WINDOW="vtkEGLRenderWindow", PYVISTA_OFF_SCREEN="true",
                       OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    environment.pop("DISPLAY", None)
    result = subprocess.run([sys.executable, "-c", script, str(page)],
                            cwd=Path(__file__).resolve().parents[1], env=environment,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, (result.stdout + result.stderr)[-6000:]
    assert 'HTML_SMOKE_OK' in result.stdout


def _legacy_flow_reference_rule(name):
    first=name in {'LTS','RTS','LPA','RPA'}
    fractional=name in {'LTS','RTS','SSS','StrS'}
    spec={'anchor':'label_level' if first else 'label_terminal','path_selection':'longest','scope':'label',
          'position':'fractions' if fractional else 'after_start' if first else 'before_end'}
    if first:spec['level']=1
    if fractional:spec.update(fractions=[.1],fraction_reference='flow_start' if first else 'flow_end')
    else:spec['offset_mm']=15.
    return {'plane_mode':'landmark','plane_count':1,'placements':[spec]}


@pytest.mark.parametrize('name',['AAO','DAO','LICA','RICA','LVA','RVA','BA','LIJV','RIJV'])
def test_continuous_trunk_phantom_small_spur_does_not_move_fraction_planes(name):
    from autoflow.algorithms.label_planes import generate_label_planes
    from autoflow.plane_rules import default_label_plane_rules
    straight=np.column_stack((np.arange(5.,65.),np.full(60,20.),np.full(60,10.)))
    mapping={name:17}
    def run(paths,nodes,levels):
        hierarchy=[{'label_local_levels':{'17':level},'flow_sign':1} for level in levels]
        volume,infos=_landmark_phantom(paths,[np.full(len(p),17) for p in paths],nodes,mapping,hierarchy)
        original=[p.copy() for p in paths]
        planes,_,_,qc=generate_label_planes(paths,volume,mapping,{name:_legacy_label_route_rule(name)},
            path_info=infos,smoothing_window=1,inter_time=1)
        for before,after in zip(original,paths):np.testing.assert_array_equal(before,after)
        assert qc[0]['status']=='selected'
        assert all(p.placement['label_level'] is None for p in planes)
        return np.array([p.center for p in planes]),planes
    baseline,_=run([straight],[(0,2)],[0])
    spur=np.column_stack((np.full(5,35.),np.arange(20.,25.),np.full(5,10.)))
    branched,planes=run([straight[:31],straight[30:],spur],[(0,1),(1,2),(1,3)],[57,2,99])
    np.testing.assert_allclose(branched,baseline,rtol=0,atol=1e-8)
    assert all(p.placement['route_path_indices']==[0,1] for p in planes)


def test_continuous_reference_phantom_extends_across_spurious_path_breaks():
    from autoflow.algorithms.label_planes import generate_label_planes
    from autoflow.plane_rules import default_label_plane_rules
    paths=[np.column_stack((np.arange(5.,21.),np.full(16,20.),np.full(16,10.))),
           np.column_stack((np.arange(20.,36.),np.full(16,20.),np.full(16,10.))),
           np.column_stack((np.arange(35.,65.),np.full(30,20.),np.full(30,10.))),
           np.column_stack((np.full(5,35.),np.arange(20.,25.),np.full(5,10.)))]
    values=[np.full(16,4),np.where(np.arange(16)<3,4,12),np.full(30,12),np.full(5,12)]
    volume,infos=_landmark_phantom(paths,values,[(0,1),(1,2),(2,3),(2,4)],{'DAO':4,'LRA':12})
    planes,_,_,qc=generate_label_planes(paths,volume,{'DAO':4,'LRA':12},
        {'LRA':_legacy_label_route_rule('LRA')},path_info=infos,smoothing_window=1,inter_time=1)
    assert len(planes)==3 and qc[0]['status']=='selected'
    np.testing.assert_allclose([p.center[0] for p in planes],[27.1,35.3,43.5],atol=1e-8)
    assert all(p.placement['route_path_indices']==[1,2] for p in planes)


def test_continuous_reference_phantom_ambiguous_far_fork_does_not_block_known_near_plane():
    from autoflow.algorithms.label_planes import generate_label_planes
    from autoflow.plane_rules import default_label_plane_rules
    paths=[np.column_stack((np.arange(5.,31.),np.full(26,20.),np.full(26,10.))),
           np.column_stack((np.arange(30.,56.),np.full(26,20.),np.full(26,10.))),
           np.column_stack((np.full(26,30.),np.arange(20.,46.),np.full(26,10.)))]
    values=[np.where(np.arange(26)<5,17,28),np.full(26,28),np.full(26,28)]
    volume,infos=_landmark_phantom(paths,values,[(0,1),(1,2),(1,3)],{'LICA':17,'LMCA':28})
    planes,_,_,qc=generate_label_planes(paths,volume,{'LICA':17,'LMCA':28},
        {'LMCA':_legacy_label_route_rule('LMCA')},path_info=infos,smoothing_window=1,inter_time=1)
    assert len(planes)==1 and planes[0].center[0]==pytest.approx(15.)
    assert qc[0]['status']=='selected' and planes[0].placement['trunk_competing_routes']>=1
    rule={'plane_mode':'landmark','plane_count':1,'placements':[
        {'anchor':'label_reference','path_selection':'longest','position':'fractions','fractions':[.8]}]}
    planes,_,_,qc=generate_label_planes(paths,volume,{'LICA':17,'LMCA':28},{'LMCA':rule},
        path_info=infos,smoothing_window=1,inter_time=1)
    assert not planes and qc[0]['omitted_positions'][0]['reason']=='ambiguous_label_trunk'
    assert qc[0]['placements'][0]['trunk_selection'][0]['reason']=='ambiguous_label_trunk'


def test_continuous_reference_phantom_rejects_an_unrelated_parent_without_fallback():
    from autoflow.algorithms.label_planes import generate_label_planes
    from autoflow.plane_rules import default_label_plane_rules
    path=np.column_stack((np.arange(5.,56.),np.full(51,20.),np.full(51,10.)))
    volume,infos=_landmark_phantom([path],[np.where(np.arange(51)<10,4,9)],[(0,1)],{'DAO':4,'LPA':9})
    planes,_,_,qc=generate_label_planes([path],volume,{'DAO':4,'LPA':9},
        {'LPA':_legacy_label_route_rule('LPA')},path_info=infos,smoothing_window=1,inter_time=1)
    assert not planes and qc[0]['omitted_positions'][0]['reason']=='label_reference_not_found'


def test_clinical_default_plan_phantom_has_no_path_level_or_cross_vessel_plane_reference():
    from autoflow.plane_rules import default_label_plane_rules,normalize_label_plane_rules
    rules=default_label_plane_rules()
    assert len(rules)==32 and sum(r['plane_count'] for r in rules.values())==45
    assert {s['anchor'] for r in rules.values() for s in r['placements']}=={'label_branch','label_reference_branch','flow_inlet_branch','portal_branch','label_longest'}
    assert all('level' not in s and 'fallback' not in s for r in rules.values() for s in r['placements'])
    assert all(rules[n]['placements'][0]['path_selection']=='all' for n in ('LRA','RRA','ACA'))
    assert normalize_label_plane_rules(rules)==rules


@pytest.mark.parametrize('name',['LICA','MPA','LRA','SSS','PV','SMV','SV'])
def test_continuous_trunk_gui_phantom_roundtrips_clinical_anchors(name):
    from PySide6 import QtWidgets
    from autoflow.ui.landmark_editor import LandmarkPlanDialog
    from autoflow.plane_rules import default_label_plane_rules
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    rule=default_label_plane_rules()[name];dialog=LandmarkPlanDialog(name,rule)
    try:
        dialog.accept()
        assert dialog.result()==QtWidgets.QDialog.Accepted
        assert dialog.rule['placements']==rule['placements']
    finally:
        dialog.close();dialog.deleteLater();app.processEvents()


def test_continuous_trunk_phantom_long_thin_spur_does_not_displace_large_main_trunk():
    from autoflow.algorithms.label_planes import generate_label_planes
    from autoflow.plane_rules import default_label_plane_rules
    paths=[np.column_stack((np.arange(5.,46.),np.full(41,20.),np.full(41,10.))),
           np.column_stack((np.arange(45.,56.),np.full(11,20.),np.full(11,10.))),
           np.column_stack((np.full(44,45.),np.arange(20.,64.),np.full(44,10.)))]
    volume,infos=_landmark_phantom(paths,[np.full(len(p),17) for p in paths],[(0,1),(1,2),(1,3)],{'LICA':17},
        [{'flow_sign':1} for _ in paths])
    volume[4:57,17:24,7:14]=17
    planes,_,_,qc=generate_label_planes(paths,volume,{'LICA':17},{'LICA':_legacy_label_route_rule('LICA')},
        path_info=infos,smoothing_window=1,inter_time=1)
    assert len(planes)==2 and qc[0]['status']=='selected'
    np.testing.assert_allclose([p.center for p in planes],[[21.5,20.,10.],[38.,20.,10.]],atol=1e-8)
    assert all(p.placement['route_path_indices']==[0,1] for p in planes)


def test_continuous_reference_phantom_rejects_tiny_false_parent_label_patch():
    from autoflow.algorithms.label_planes import generate_label_planes
    from autoflow.plane_rules import default_label_plane_rules
    path=np.column_stack((np.arange(5.,56.),np.full(51,20.),np.full(51,10.)))
    volume,infos=_landmark_phantom([path],[np.where(np.arange(51)<2,2,9)],[(0,1)],{'MPA':2,'LPA':9})
    planes,_,_,qc=generate_label_planes([path],volume,{'MPA':2,'LPA':9},{'LPA':_legacy_label_route_rule('LPA')},
        path_info=infos,smoothing_window=1,inter_time=1)
    assert not planes and qc[0]['omitted_positions'][0]['reason']=='label_reference_too_short'
    assert qc[0]['placements'][0]['trunk_selection'][0]['reference_component_extent_mm']==pytest.approx(1.)


def test_continuous_trunk_phantom_loop_is_error_without_switching_to_another_component():
    from autoflow.algorithms.label_planes import generate_label_planes
    from autoflow.plane_rules import default_label_plane_rules
    paths=[np.array([[10.,10.,10.],[20.,10.,10.],[20.,20.,10.]]),
           np.array([[20.,20.,10.],[10.,20.,10.],[10.,10.,10.]]),
           np.column_stack((np.arange(5.,36.),np.full(31,40.),np.full(31,10.)))]
    # Dense samples ensure continuous physical source-label runs around the loop.
    paths=[np.vstack([np.linspace(a,b,11)[:-1] for a,b in zip(p,p[1:])]+[p[-1:]]) if len(p)==3 else p for p in paths]
    volume,infos=_landmark_phantom(paths,[np.full(len(p),21) for p in paths],[(0,1),(1,0),(2,3)],{'BA':21},
        [{'flow_sign':1} for _ in paths])
    planes,_,_,qc=generate_label_planes(paths,volume,{'BA':21},{'BA':_legacy_label_route_rule('BA')},
        path_info=infos,smoothing_window=1,inter_time=1)
    assert not planes and qc[0]['omitted_positions'][0]['reason']=='ambiguous_label_trunk_cycle'



@pytest.mark.parametrize('gap,patch,expected',[(0,False,True),(1,False,False),(0,True,False)])
def test_continuous_reference_phantom_cross_group_origin_requires_real_face_contact(gap,patch,expected):
    from autoflow.algorithms.label_planes import generate_label_planes
    from autoflow.plane_rules import default_label_plane_rules
    paths=[np.column_stack((np.arange(20.,51.),np.full(31,20.),np.full(31,10.))),
           np.column_stack((np.full(21,19.-gap),np.arange(10.,31.),np.full(21,10.)))]
    volume,infos=_landmark_phantom(paths,[np.full(31,32),np.full(21,17)],[(0,1),(2,3)],
                                  {'LPCA':32,'LICA':17})
    infos[0]['group_name']='Posterior';infos[1]['group_name']='Anterior'
    if patch:
        volume[volume==17]=0
        volume[19,20,10]=17
    planes,_,_,qc=generate_label_planes(paths,volume,{'LPCA':32,'LICA':17},
        {'LPCA':_legacy_label_route_rule('LPCA')},path_info=infos,smoothing_window=1,inter_time=1)
    if expected:
        assert len(planes)==1 and qc[0]['status']=='selected'
        np.testing.assert_allclose(planes[0].center,[25.,20.,10.],atol=1e-8)
        assert planes[0].placement['connection_kind']=='label_volume_interface'
        assert planes[0].placement['route_path_indices']==[0]
    else:
        assert not planes
        reason='label_reference_too_short' if patch else 'label_reference_not_found'
        assert qc[0]['omitted_positions'][0]['reason']==reason



def test_current_view_orbit_phantom_screen_axis_marker_and_cancellation(monkeypatch, tmp_path):
    _smoke_qt_application(monkeypatch)
    import pyvista as pv
    import imageio.v2 as imageio
    from autoflow.ui.view_video import export_current_view
    from autoflow.ui.video_options import CurrentViewVideoDialog, MovieCapture
    from autoflow.task_control import TaskCancelled
    dialog = CurrentViewVideoDialog(None, 4, rotate=True)
    assert dialog.motion_options() == dict(rotate=True, rotation_axis='screen_vertical',
        rotation_angle=360., rotation_speed=30., hide_orientation_marker=False)
    assert not dialog.cycles.isEnabled()
    dialog.camera_mode.setCurrentIndex(0)
    assert dialog.cycles.isEnabled() and not dialog.motion_options()['rotate']
    dialog.deleteLater()
    plotter = pv.Plotter(off_screen=True, window_size=(160, 120))
    try:
        plotter.add_mesh(pv.Cone(direction=(1, 0, 0)), color='orange')
        plotter.add_mesh(pv.Sphere(radius=.2, center=(.6, .3, .2)), color='cyan')
        marker = plotter.add_orientation_widget(pv.Arrow(), interactive=False)
        plotter.camera_position = [(3., -5., 4.), (.2, .3, .4), (.1, .2, 1.)]
        plotter.camera.OrthogonalizeViewUp()
        plotter.camera.parallel_projection = True
        plotter.camera.parallel_scale = 1.6
        plotter.show(auto_close=False, interactive=False)
        original = plotter.camera.copy()
        ws = Workspace();ws.mag_raw = np.ones((3, 3, 3, 3), np.float32)
        ws.flow_raw = np.zeros((3, 3, 3, 3, 3), np.float32);ws.current_t = 2
        snapshots = []
        capture = MovieCapture.image
        def image(self):
            pixels = capture(self)
            snapshots.append((np.array(plotter.camera.position), np.array(plotter.camera.up),
                              marker.GetEnabled(), pixels.copy()))
            return pixels
        monkeypatch.setattr(MovieCapture, 'image', image)
        window = SimpleNamespace(workspace=ws, plotter=plotter,
            scene=SimpleNamespace(_orientation_marker=SimpleNamespace(widget=marker),
                _interaction_preview_active=False, set_interaction_preview=lambda *a, **k: None,
                update_time=lambda _: None, flush_render=plotter.render),
            _play_timer=SimpleNamespace(isActive=lambda: False), _on_pause=lambda: None,
            _refresh_timeline=lambda: None, ortho_viewer=SimpleNamespace(refresh=lambda: None))
        target = tmp_path/'orbit.mp4'
        result = export_current_view(window, target, fps=4, rotate=True, rotation_speed=180,
                                     hide_orientation_marker=True)
        assert result['frames'] == 9 and result['phases'] == [2, 0, 1]*3
        assert result['rotate'] and result['hide_orientation_marker']
        assert all(not row[2] for row in snapshots) and marker.GetEnabled()
        focal = np.array(original.focal_point);offset = np.array(original.position)-focal
        np.testing.assert_allclose(snapshots[0][0], original.position)
        np.testing.assert_allclose(snapshots[-1][0], original.position, atol=1e-10)
        assert abs(np.dot(snapshots[2][0]-focal, offset)) < 1e-9
        np.testing.assert_allclose(snapshots[4][0]-focal, -offset, atol=1e-10)
        assert np.mean(np.abs(snapshots[0][3].astype(float)-snapshots[2][3])) > 1
        for position, up, _, _ in snapshots:
            np.testing.assert_allclose(np.linalg.norm(position-focal), np.linalg.norm(offset))
            np.testing.assert_allclose(up, original.up, atol=1e-10)
        reader = imageio.get_reader(target)
        assert sum(1 for _ in reader) == 9;reader.close()
        previous = target.read_bytes()
        snapshots.clear()
        result = export_current_view(window, tmp_path/'marker.mp4', fps=4)
        assert not result['hide_orientation_marker'] and all(row[2] for row in snapshots)
        events = []
        with pytest.raises(TaskCancelled):
            export_current_view(window, target, fps=4, rotate=True, rotation_speed=180,
                hide_orientation_marker=True, cancelled=lambda: len(events) >= 3,
                progress=lambda current, total: events.append(current))
        assert len(events) == 3 and target.read_bytes() == previous
        assert marker.GetEnabled() and ws.current_t == 2
        np.testing.assert_allclose(plotter.camera.position, original.position)
        np.testing.assert_allclose(plotter.camera.up, original.up)
        assert plotter.camera.parallel_scale == original.parallel_scale
        assert plotter.camera.clipping_range == original.clipping_range
    finally:
        plotter.close()


def test_colorbar_layout_phantom_preview_clamps_and_rolls_back(monkeypatch):
    _smoke_qt_application(monkeypatch)
    import copy
    import pyvista as pv
    from PySide6 import QtWidgets
    from autoflow.ui.colorbar_layout import ColorbarLayoutDialog
    from autoflow.ui.viewer import SceneController
    from autoflow.core.models import ObjectKind
    ws = Workspace();before = copy.deepcopy(ws.render_settings)
    plotter = pv.Plotter(off_screen=True, window_size=(300, 240))
    parent = QtWidgets.QWidget();parent.workspace = ws
    errors = []
    try:
        scene = parent.scene = SceneController(plotter, ws, errors.append)
        scene.initialize()
        mesh = pv.Sphere();mesh['value'] = mesh.points[:, 0]
        actor = plotter.add_mesh(mesh, scalars='value', show_scalar_bar=False)
        uid = ws.add_object('Phantom', ObjectKind.METRIC, 'phantom', visible=True,
                           scalars='value', show_scalar_bar=True, scalar_bar_title='Velocity (m/s)')
        ws.scene_objects[uid].actor = actor
        scene._refresh_shared_scalar_bar()
        dialog = ColorbarLayoutDialog(parent)
        dialog.presets.setCurrentText('Right')
        bar = next(iter(plotter.scalar_bars.values()))
        assert bar.GetOrientation() == 1
        np.testing.assert_allclose(bar.GetPosition(), (.86, .18))
        dialog.length.setValue(70);dialog.x.setValue(98)
        bar = next(iter(plotter.scalar_bars.values()))
        assert bar.GetPosition()[0]+bar.GetWidth() <= .98001
        assert abs(bar.GetHeight()-.7) < 1e-6
        dialog.reject();dialog.deleteLater()
        assert ws.render_settings == before and not errors
        dialog = ColorbarLayoutDialog(parent)
        dialog.presets.setCurrentText('Bottom')
        dialog.label_font.setValue(17)
        dialog.accept();dialog.deleteLater()
        saved = ws.snapshot_dict()
        restored = Workspace();restored.restore_dict(saved)
        assert restored.render_settings['shared_colorbar_bar_cfg']['label_font_size'] == 17
        assert not restored.render_settings['shared_colorbar_bar_cfg']['vertical']
    finally:
        parent.close();parent.deleteLater();plotter.close()


@pytest.mark.parametrize('kind',['forward','reverse','pulsatile','spatial_reversal','transverse'])
def test_branch_alignment_phantom_only_two_scores_separate_spatial_and_temporal_reversal(kind):
    from autoflow.algorithms.flow_alignment import branch_flow_evidence
    x=np.arange(3.5,36.,1.)
    points=np.column_stack((x,np.full(len(x),10.),np.full(len(x),10.)))
    field=np.zeros((41,21,21,2,3),np.float32)
    field[...,0]=2.
    if kind=='reverse':field[...,0]=-2.
    if kind=='pulsatile':field[...,1,0]=-2.
    if kind=='spatial_reversal':field[20:,...,0]=-2.
    if kind=='transverse':field[...,0]=0.;field[...,1]=2.
    result=branch_flow_evidence(points,field)
    assert set(result)=={'status','reason','axis_alignment','phase_alignment'}
    assert result['status']=='available'
    if kind=='transverse':
        assert result['axis_alignment']==pytest.approx(0.)
        assert result['phase_alignment']==pytest.approx(0.)
    elif kind=='spatial_reversal':
        assert result['axis_alignment']==pytest.approx(1.)
        assert result['phase_alignment']==pytest.approx(0.,abs=1e-12)
    else:
        assert result['axis_alignment']==pytest.approx(1.)
        assert result['phase_alignment']==pytest.approx(1.)


def _axis_phase_bridge_phantom():
    from autoflow.core.models import GraphData
    graph=GraphData(points=np.array([[10.,20.,10.],[30.,20.,10.],[10.,10.,10.],
        [10.,30.,10.],[30.,10.,10.],[30.,30.,10.]]),
        edges=np.array([[0,1],[0,2],[0,3],[1,4],[1,5]]))
    field=np.zeros((41,41,21,1,3),np.float32);field[...,1]=2.
    return graph,field


def test_branch_filter_phantom_node_pass_protects_a_short_internal_connection():
    import networkx as nx
    from autoflow.algorithms.branch_filter import filter_graph_branches
    from autoflow.algorithms.graph import graph_to_networkx
    graph,field=_axis_phase_bridge_phantom()
    field[...,0]=2.
    original_points=graph.points.copy();original_edges=graph.edges.copy()
    cleaned,report=filter_graph_branches(graph,flow=field,apply=True)
    assert len(cleaned.edges)==1
    assert nx.number_connected_components(graph_to_networkx(cleaned))==1
    assert report['stages']['flow']['candidate_count']==0
    assert report['stages']['nodes']['candidate_count']==4
    for stage in [report['stages']['nodes']]:
        connection=next(b for b in stage['branches'] if not b['terminal'])
        assert connection['decision']=='retained' and connection['reason']=='connection_preserved'
    assert report['stages']['flow']['branches'][0]['flow']['status']=='available'
    assert len(report['stages']['flow']['junctions'])==2 and len(report['stages']['flow']['branches'])==5
    assert report['computed_scores']==['axis_alignment','phase_alignment']
    np.testing.assert_array_equal(graph.points,original_points)
    np.testing.assert_array_equal(graph.edges,original_edges)


@pytest.mark.parametrize('axis,phase,removed',[(.3,.3,False),(.8,.299,True),(.299,.299,True),(.5,.5,False)])
def test_branch_filter_phantom_or_threshold_is_strict_and_phase_alone_can_remove(monkeypatch,axis,phase,removed):
    import autoflow.algorithms.branch_filter as module
    from autoflow.core.models import GraphData
    graph=GraphData(points=np.array([[2.,2.,2.],[8.,2.,2.]]),edges=np.array([[0,1]]))
    monkeypatch.setattr(module,'branch_flow_evidence',lambda *a,**k:
        dict(status='available',reason='',axis_alignment=axis,phase_alignment=phase))
    cleaned,report=module.filter_graph_branches(graph,apply=True,min_node_count=0)
    assert bool(report['removed_edge_count'])==removed
    assert cleaned.points.shape==(0,3) if removed else cleaned is graph


def test_branch_filter_phantom_audit_and_node_renumbering_preserve_input_and_decisions():
    from autoflow.algorithms.branch_filter import filter_graph_branches
    from autoflow.core.models import GraphData
    graph,field=_axis_phase_bridge_phantom()
    field[...,1]=0.;field[...,0]=2.
    same,audit=filter_graph_branches(graph,flow=field,min_node_count=0)
    assert same is graph and audit['candidate_count']==4 and not audit['removed_edge_count']
    cleaned,report=filter_graph_branches(graph,flow=field,apply=True,min_node_count=0)
    permutation=np.arange(len(graph.points))[::-1];inverse=np.argsort(permutation)
    shuffled=GraphData(points=graph.points[permutation],edges=inverse[graph.edges[::-1]])
    other,result=filter_graph_branches(shuffled,flow=field,apply=True,min_node_count=0)
    assert sorted(map(tuple,other.points))==sorted(map(tuple,cleaned.points))
    assert {b['branch_id'] for b in result['stages']['flow']['branches'] if b['decision']=='removed'}=={
        b['branch_id'] for b in report['stages']['flow']['branches'] if b['decision']=='removed'}


def test_branch_filter_phantom_missing_velocity_is_unavailable_rather_than_a_fabricated_zero():
    from autoflow.algorithms.branch_filter import filter_graph_branches
    graph,_=_axis_phase_bridge_phantom()
    same,report=filter_graph_branches(graph,apply=True,min_node_count=0)
    assert same is graph and not report['candidate_count']
    assert all(b['flow']['axis_alignment'] is None and b['flow']['phase_alignment'] is None for b in report['stages']['flow']['branches'])


@pytest.mark.parametrize('options',[{'threshold':1.1},{'apply':'false'},{'min_node_count':-1},{'min_node_count':8.5},{'min_node_count':True}])
def test_branch_filter_phantom_invalid_simple_controls_are_rejected(options):
    from autoflow.algorithms.branch_filter import filter_graph_branches
    graph,field=_axis_phase_bridge_phantom()
    with pytest.raises(ValueError):filter_graph_branches(graph,flow=field,**options)


@pytest.mark.parametrize('count,removed',[(4,True),(5,False)])
def test_branch_filter_phantom_default_node_limit_counts_both_endpoints_and_is_strict(count,removed):
    from autoflow.algorithms.branch_filter import filter_graph_branches
    from autoflow.core.models import GraphData
    points=np.column_stack((np.arange(3.,3.+count),np.full(count,5.),np.full(count,5.)))
    graph=GraphData(points=points,edges=np.column_stack((np.arange(count-1),np.arange(1,count))))
    field=np.zeros((21,11,11,1,3),np.float32);field[...,0]=2.
    cleaned,report=filter_graph_branches(graph,flow=field,apply=True)
    branch=report['stages']['nodes']['branches'][0]
    assert report['min_node_count']==5 and branch['node_count']==count
    flow_evidence=report['stages']['flow']['branches'][0]['flow']
    assert flow_evidence['axis_alignment']==pytest.approx(1.)
    assert flow_evidence['phase_alignment']==pytest.approx(1.)
    assert (branch['decision']=='removed')==removed
    assert len(cleaned.edges)==(0 if removed else count-1)


@pytest.mark.parametrize('count,removed',[(4,False),(5,False)])
def test_branch_filter_phantom_closed_arm_does_not_count_its_repeated_start_node_twice(count,removed):
    from autoflow.algorithms.branch_filter import filter_graph_branches
    from autoflow.core.models import GraphData
    angle=np.arange(count)*2*np.pi/count
    points=np.column_stack((10+3*np.cos(angle),10+3*np.sin(angle),np.full(count,10.)))
    graph=GraphData(points=points,edges=np.column_stack((np.arange(count),np.roll(np.arange(count),-1))))
    cleaned,report=filter_graph_branches(graph,apply=True)
    assert report['stages']['nodes']['branches'][0]['node_count']==count
    assert not report['stages']['nodes']['branches'][0]['terminal']
    assert report['stages']['flow']['branches'][0]['flow']['status']=='unavailable'
    assert (report['stages']['nodes']['branches'][0]['decision']=='removed')==removed
    assert len(cleaned.edges)==(0 if removed else count)


def test_branch_filter_phantom_node_limit_can_be_configured_without_velocity():
    from autoflow.algorithms.branch_filter import filter_graph_branches
    from autoflow.core.models import GraphData
    points=np.column_stack((np.arange(9.),np.zeros(9),np.zeros(9)))
    graph=GraphData(points=points,edges=np.column_stack((np.arange(8),np.arange(1,9))))
    retained,audit=filter_graph_branches(graph,min_node_count=0,apply=True)
    removed,report=filter_graph_branches(graph,min_node_count=10,apply=True)
    assert retained is graph and len(removed.edges)==0
    assert report['stages']['nodes']['branches'][0]['reason']=='short_arm'


def test_branch_filter_phantom_resegments_after_flow_before_counting_nodes(monkeypatch):
    import autoflow.algorithms.branch_filter as module
    from autoflow.core.models import GraphData
    main=np.column_stack((np.arange(5.,21.),np.full(16,10.),np.full(16,5.)))
    side_a=np.column_stack((np.full(3,9.),np.arange(11.,14.),np.full(3,5.)))
    side_b=side_a.copy();side_b[:,0]=13.
    graph=GraphData(points=np.vstack((main,side_a,side_b)),edges=np.array(
        [(i,i+1) for i in range(15)]+[(4,16),(16,17),(17,18),(8,19),(19,20),(20,21)]))
    velocity=np.zeros((31,21,11,1,3),np.float32);velocity[...,0]=2.
    original=module.branch_flow_evidence;calls=[]
    def measured(*args,**kwargs):
        calls.append(1)
        return original(*args,**kwargs)
    monkeypatch.setattr(module,'branch_flow_evidence',measured)
    cleaned,report=module.filter_graph_branches(graph,flow=velocity,apply=True)
    first=report['stages']['flow'];second=report['stages']['nodes']
    assert len(first['branches'])==5 and first['candidate_count']==2
    assert len(calls)==5  # C evaluates every initial arm; the node pass never samples flow.
    assert len(second['branches'])==1
    assert second['branches'][0]['node_count']==16 and second['branches'][0]['decision']=='retained'
    assert all('flow' not in branch for branch in second['branches'])
    np.testing.assert_array_equal(cleaned.points,main)
    assert len(cleaned.edges)==15 and len(graph.edges)==21
    same,audit=module.filter_graph_branches(graph,flow=velocity)
    assert same is graph and not audit['removed_edge_count']
    assert audit['stages']['flow']['candidate_count']==2
    assert audit['stages']['nodes']['branches'][0]['node_count']==16

def test_metric_surface_default_keeps_zero_valid_and_missing_phase_transparent():
    from autoflow.rendering.datasets import metric_display_mesh
    from autoflow.rendering.style import metric_style, configured_metric_style, rounded_display_clim
    ws = Workspace()
    mask = np.zeros((9, 9, 9, 2), bool)
    mask[2:7, 2:7, 2:7] = True
    ws.segmask_binary = mask
    energy = np.full(mask.shape, np.nan, np.float32)
    energy[..., 0][mask[..., 0]] = 0.
    ws.derived.tke_array = energy
    before = energy.copy()
    zero = metric_display_mesh(ws, "tke_volume", 0)
    missing = metric_display_mesh(ws, "tke_volume", 1)
    assert np.all(zero["display_valid"] == 1)
    np.testing.assert_allclose(zero["TKE"], 0., atol=1e-6)
    assert not np.any(missing["display_valid"])
    assert np.isnan(missing["TKE"]).all()
    assert configured_metric_style(ws, "tke_volume")["mode"] == "surface"
    style = metric_style(ws, "tke_volume")
    assert style["opacity"] == 1. and style["nan_opacity"] == 0.
    assert style["lighting"] is False
    assert rounded_display_clim((0., 2.12)) == (0., 3.)
    assert rounded_display_clim((-7511., 7511.)) == (-8000., 8000.)
    np.testing.assert_array_equal(energy, before)


def test_branch_filter_phantom_terminal_status_is_recomputed_after_flow():
    from autoflow.algorithms.branch_filter import filter_graph_branches
    from autoflow.core.models import GraphData
    left=np.array([[10.,20.,10.],[30.,20.,10.],[10.,10.,10.],[10.,30.,10.]])
    up=np.column_stack((np.full(4,30.),np.arange(21.,25.),np.full(4,10.)))
    down=up.copy();down[:,1]=np.arange(19.,15.,-1.)
    graph=GraphData(points=np.vstack((left,up,down)),edges=np.array(
        [[0,1],[0,2],[0,3],[1,4],[4,5],[5,6],[6,7],[1,8],[8,9],[9,10],[10,11]]))
    velocity=np.zeros((41,41,21,1,3),np.float32)
    velocity[:20,...,0]=2.;velocity[20:,...,1]=2.
    cleaned,report=filter_graph_branches(graph,flow=velocity,min_node_count=3,apply=True)
    first=report['stages']['flow'];second=report['stages']['nodes']
    old_bridge=next(b for b in first['branches'] if not b['terminal'])
    assert old_bridge['decision']=='retained' and first['candidate_count']==2
    new_bridge=next(b for b in second['branches'] if b['node_count']==2)
    assert new_bridge['terminal'] and new_bridge['decision']=='removed'
    assert second['candidate_count']==1 and len(cleaned.edges)==8


def test_branch_filter_phantom_c_pass_can_remove_a_nonterminal_low_alignment_connection():
    import networkx as nx
    from autoflow.algorithms.branch_filter import filter_graph_branches
    from autoflow.algorithms.graph import graph_to_networkx
    graph,velocity=_axis_phase_bridge_phantom()
    cleaned,report=filter_graph_branches(graph,flow=velocity,min_node_count=0,apply=True)
    connection=next(b for b in report['stages']['flow']['branches'] if not b['terminal'])
    assert connection['flow']['axis_alignment']==pytest.approx(0.)
    assert connection['decision']=='removed'
    assert len(cleaned.edges)==4 and nx.number_connected_components(graph_to_networkx(cleaned))==2
    assert report['terminal_only_stages']==['nodes']


def _legacy_label_route_placements(name):
    """Clinical fractions on continuous trunks or offsets from local vessel references."""
    def trunk(fractions):
        return [{"anchor":"label_trunk", "scope":"label", "path_selection":"longest",
                 "position":"fractions", "fractions":fractions}]
    def reference(*, fractions=None, offset=15., all_origins=False):
        spec={"anchor":"label_reference", "path_selection":"all" if all_origins else "longest",
              "position":"fractions" if fractions is not None else "near"}
        spec.update(fractions=fractions) if fractions is not None else spec.update(offset_mm=offset)
        return [spec]
    if name=='AAO':return trunk([.2,.5,.8])
    if name=='DAO':return trunk([.1,.3,.5])
    if name in {'LICA','RICA','LVA','RVA'}:return trunk([.33,.66])
    if name=='BA':return trunk([.5])
    if name in {'LIJV','RIJV'}:return trunk([1/3,2/3])
    if name in {'LRA','RRA'}:return reference(fractions=[.1,.3,.5],all_origins=True)
    if name=='PV':return reference(fractions=[.2,.5,.8])
    if name in {'LTS','RTS','SSS','StrS'}:return reference(fractions=[.1])
    if name in {'MPA','LPA','RPA','RBCT','CCA','LBCT','SMV','SV'}:return reference()
    if name in {'ACA','LMCA','RMCA','LPCA','RPCA'}:return reference(offset=5.,all_origins=name=='ACA')
    return reference(offset=5.)


def _legacy_label_route_rule(name):
    from autoflow.plane_rules import default_label_plane_rules
    rule = default_label_plane_rules()[name]
    rule['placements'] = _legacy_label_route_placements(name)
    return rule



@pytest.mark.parametrize('cycle',[False,True])
def test_single_branch_portal_phantom_stops_at_the_first_fork_even_with_a_downstream_ring(cycle):
    from autoflow.algorithms.label_planes import generate_label_planes
    from autoflow.plane_rules import default_label_plane_rules
    paths=[np.column_stack((np.full(16,20.),np.arange(5.,21.),np.full(16,10.))),
           np.column_stack((np.arange(5.,21.),np.full(16,20.),np.full(16,10.))),
           np.column_stack((np.arange(20.,41.),np.full(21,20.),np.full(21,10.)))]
    nodes=[(0,1),(2,1),(1,3)]
    values=[np.r_[np.full(13,15),np.full(3,14)],np.r_[np.full(13,16),np.full(3,14)],np.full(21,14)]
    if cycle:
        loop=np.vstack((np.linspace([40.,20.,10.],[60.,20.,10.],21)[:-1],
                        np.linspace([60.,20.,10.],[60.,40.,10.],21)[:-1],
                        np.linspace([60.,40.,10.],[40.,20.,10.],21)))
        paths.append(loop);nodes.append((3,3));values.append(np.full(len(loop),14))
    else:
        paths.extend([np.column_stack((np.arange(40.,61.),np.full(21,20.),np.full(21,10.))),
                      np.column_stack((np.full(21,40.),np.arange(20.,41.),np.full(21,10.)))])
        nodes.extend([(3,4),(3,5)]);values.extend([np.full(21,14),np.full(21,14)])
    mapping={'PV':14,'SMV':15,'SV':16}
    volume,infos=_landmark_phantom(paths,values,nodes,mapping)
    planes,_,_,qc=generate_label_planes(paths,volume,mapping,{'PV':default_label_plane_rules()['PV']},
        path_info=infos,smoothing_window=1,inter_time=1)
    assert len(planes)==3 and qc[0]['status']=='selected'
    np.testing.assert_allclose([p.center for p in planes],[[24.,20.,10.],[30.,20.,10.],[36.,20.,10.]])
    assert all(p.path_index==2 and p.placement['route_path_indices']==[2] for p in planes)
    assert all(p.placement['branch_placement_length_mm']==pytest.approx(20.) for p in planes)
    assert all(set(p.placement['matched_arms'])=={'PV','SMV','SV'} for p in planes)


@pytest.mark.parametrize('name,parent,value,parent_value',[
    ('LPA','MPA',9,2),('RPA','MPA',8,2),
    ('LMCA','LICA',28,17),('RMCA','RICA',29,20),
    ('LPCA','BA',26,21),('RPCA','BA',27,21),('ACA','LICA',30,17)])
def test_single_branch_reference_phantom_uses_the_configured_origin_fraction(name,parent,value,parent_value):
    from autoflow.algorithms.label_planes import generate_label_planes
    from autoflow.plane_rules import default_label_plane_rules
    paths=[np.column_stack((np.arange(5.,36.),np.full(31,20.),np.full(31,10.))),
           np.column_stack((np.arange(35.,61.),np.full(26,20.),np.full(26,10.))),
           np.column_stack((np.full(26,35.),np.arange(20.,46.),np.full(26,10.)))]
    values=[np.where(np.arange(31)<10,parent_value,value),np.full(26,value),np.full(26,value)]
    mapping={name:value,parent:parent_value}
    volume,infos=_landmark_phantom(paths,values,[(0,1),(1,2),(1,3)],mapping)
    planes,_,_,qc=generate_label_planes(paths,volume,mapping,{name:default_label_plane_rules()[name]},
        path_info=infos,smoothing_window=1,inter_time=1)
    assert len(planes)==1 and qc[0]['status']=='selected'
    fraction=.3 if name in {'MPA','LPA','RPA'} else .1
    np.testing.assert_allclose(planes[0].center,[15.+20.*fraction,20.,10.])
    assert planes[0].placement['fraction']==pytest.approx(fraction)
    assert planes[0].placement['branch_placement_length_mm']==pytest.approx(20.)
    assert planes[0].placement['route_path_indices']==[0]
    assert 'requested_offset_mm' not in planes[0].placement


@pytest.mark.parametrize('name',['LVA','RVA'])
@pytest.mark.parametrize('sign',[1,-1,None])
def test_single_branch_vertebral_phantom_uses_thirty_and_fifty_percent_in_flow_direction(name,sign):
    from autoflow.algorithms.label_planes import generate_label_planes
    from autoflow.plane_rules import default_label_plane_rules
    path=np.column_stack((np.arange(5.,56.),np.full(51,20.),np.full(51,10.)))
    volume,infos=_landmark_phantom([path],[np.full(51,18)],[(0,1)],{name:18},[{'flow_sign':sign}])
    planes,_,_,qc=generate_label_planes([path],volume,{name:18},{name:default_label_plane_rules()[name]},
        path_info=infos,smoothing_window=1,inter_time=1)
    if sign is None:
        assert not planes and qc[0]['selection_warnings']==['flow_direction_unavailable']
    else:
        assert len(planes)==2
        np.testing.assert_allclose([p.center[0] for p in planes],[20.,30.] if sign==1 else [40.,30.])


@pytest.mark.parametrize('name',['LICA','RICA'])
@pytest.mark.parametrize('reverse',[False,True])
def test_single_branch_carotid_phantom_selects_the_flow_entry_before_longer_daughters(name,reverse):
    from autoflow.algorithms.label_planes import generate_label_planes
    from autoflow.plane_rules import default_label_plane_rules
    paths=[np.column_stack((np.arange(5.,21.),np.full(16,20.),np.full(16,10.))),
           np.column_stack((np.arange(20.,61.),np.full(41,20.),np.full(41,10.))),
           np.column_stack((np.full(31,20.),np.arange(20.,51.),np.full(31,10.)))]
    nodes=[(0,1),(1,2),(1,3)];signs=[1,1,1]
    if reverse:
        paths[0]=paths[0][::-1];nodes[0]=(1,0);signs[0]=-1
    volume,infos=_landmark_phantom(paths,[np.full(len(p),17) for p in paths],nodes,{name:17},
                                  [{'flow_sign':s} for s in signs])
    planes,_,_,qc=generate_label_planes(paths,volume,{name:17},{name:default_label_plane_rules()[name]},
        path_info=infos,smoothing_window=1,inter_time=1)
    assert len(planes)==2 and all(p.path_index==0 for p in planes)
    np.testing.assert_allclose([p.center[0] for p in planes],[9.95,14.9])
    assert all(p.placement['anchor']=='flow_inlet_branch' and p.placement['route_path_indices']==[0] for p in planes)


def test_single_branch_carotid_phantom_multiple_flow_entries_are_not_selected_by_path_number():
    from autoflow.algorithms.label_planes import generate_label_planes
    from autoflow.plane_rules import default_label_plane_rules
    paths=[np.column_stack((np.arange(5.,26.),np.full(21,y),np.full(21,10.))) for y in (10.,30.)]
    volume,infos=_landmark_phantom(paths,[np.full(21,17)]*2,[(0,1),(2,3)],{'LICA':17},[{'flow_sign':1}]*2)
    planes,_,_,qc=generate_label_planes(paths,volume,{'LICA':17},{'LICA':default_label_plane_rules()['LICA']},
        path_info=infos,smoothing_window=1,inter_time=1)
    assert not planes and qc[0]['selection_warnings']==['ambiguous_label_branch']


@pytest.mark.parametrize('filled',[False,True])
@pytest.mark.parametrize('permuted',[False,True])
def test_lumen_cycle_opening_phantom_certifies_only_filled_lumen(filled,permuted):
    import networkx as nx
    from autoflow.core.models import GraphData
    from autoflow.algorithms.graph import graph_to_networkx
    from autoflow.algorithms.cycle_opening import open_redundant_cycles
    mask=np.zeros((35,35,21),bool)
    mask[4:31,4:31,3:18]=True
    if not filled:
        mask[13:22,13:22,:]=False
    points=np.array([[9.,9.,10.],[26.,9.,10.],[26.,26.,10.],[9.,26.,10.],[5.,9.,10.]])
    edges=np.array([[0,1],[1,2],[2,3],[3,0],[0,4]])
    original=GraphData(points=points.copy(),edges=edges.copy())
    expected,baseline=open_redundant_cycles(original,mask)
    if permuted:
        order=np.array([3,1,4,0,2]);inverse=np.argsort(order)
        points=points[order];edges=inverse[edges[::-1,::-1]]
    graph=GraphData(points=points.copy(),edges=edges.copy())
    output,report=open_redundant_cycles(graph,mask)
    np.testing.assert_array_equal(output.points,points)
    np.testing.assert_array_equal(graph.edges,edges)
    assert report['removed_edge_count']==int(filled)
    assert report['cycle_rank_after']==(0 if filled else 1)
    assert nx.is_connected(graph_to_networkx(output))
    assert report['component_connectivity_preserved']
    geometry=lambda g:{tuple(sorted((tuple(g.points[a]),tuple(g.points[b])))) for a,b in g.edges}
    assert geometry(output)==geometry(expected)
    assert any(np.array_equal(output.points[a],[5.,9.,10.]) or np.array_equal(output.points[b],[5.,9.,10.]) for a,b in output.edges)


def test_lumen_cycle_opening_phantom_missing_mask_retains_cycle():
    from autoflow.core.models import GraphData
    from autoflow.algorithms.cycle_opening import open_redundant_cycles
    graph=GraphData(points=np.array([[1.,1.,1.],[3.,1.,1.],[2.,3.,1.]]),edges=np.array([[0,1],[1,2],[2,0]]))
    output,report=open_redundant_cycles(graph,None)
    assert output is graph and report['reason']=='support_mask_unavailable'


def test_single_branch_loop_phantom_ordinary_midpoint_is_not_rejected_for_a_cycle():
    from autoflow.algorithms.label_planes import generate_label_planes
    from autoflow.plane_rules import default_label_plane_rules
    corners=np.array([[10.,10.,10.],[30.,10.,10.],[30.,30.,10.],[10.,30.,10.],[10.,10.,10.]])
    path=np.vstack([np.linspace(a,b,21)[:-1] for a,b in zip(corners,corners[1:])]+[corners[-1:]])
    volume,infos=_landmark_phantom([path],[np.full(len(path),21)],[(0,0)],{'BA':21},[{'flow_sign':1}])
    planes,_,_,qc=generate_label_planes([path],volume,{'BA':21},{'BA':default_label_plane_rules()['BA']},
        path_info=infos,smoothing_window=1,inter_time=1)
    assert len(planes)==1 and not qc[0]['selection_warnings']
    assert planes[0].placement['branch_closed']
    np.testing.assert_allclose(planes[0].center,[30.,30.,10.])



@pytest.mark.parametrize("sign", [1, -1, None])
@pytest.mark.parametrize("connected", [True, False])
def test_single_branch_pulmonary_phantom_uses_longest_mpa_from_downstream(sign, connected):
    from autoflow.algorithms.label_planes import generate_label_planes
    from autoflow.plane_rules import default_label_plane_rules
    paths=[np.column_stack((np.arange(5.,46.),np.full(41,20.),np.full(41,10.))),
           np.column_stack((np.arange(45.,61.),np.full(16,20.),np.full(16,10.))),
           np.column_stack((np.full(16,45.),np.arange(20.,36.),np.full(16,10.)))]
    values=[np.full(41,2),np.where(np.arange(16)<6,2,8),np.where(np.arange(16)<6,2,9)]
    mapping={'MPA':2,'RPA':8,'LPA':9}
    if not connected:
        values = [np.full(len(path), 2) for path in paths]
        mapping = {'MPA': 2}
    volume,infos=_landmark_phantom(paths,values,[(0,1),(1,2),(1,3)],mapping,[{'flow_sign':sign}]*3)
    planes,_,_,qc=generate_label_planes(paths,volume,mapping,{'MPA':default_label_plane_rules()['MPA']},
        path_info=infos,smoothing_window=1,inter_time=1)
    if sign is None:
        assert not planes and qc[0]["omitted_positions"][0]["reason"] == "flow_direction_unavailable"
        return
    assert len(planes)==1 and planes[0].path_index==0
    np.testing.assert_allclose(planes[0].center,[33. if sign == 1 else 17.,20.,10.])
    assert planes[0].placement["anchor"] == "label_longest"
    assert planes[0].placement["fraction_reference"] == "flow_end"
    assert planes[0].placement["fraction"] == pytest.approx(.3)
    assert planes[0].placement['route_path_indices']==[0]


@pytest.mark.parametrize('reverse',[False,True])
@pytest.mark.parametrize('permuted',[False,True])
def test_lowest_inlet_phantom_uses_physical_z_not_path_number(reverse,permuted):
    from autoflow.algorithms.label_planes import generate_label_planes
    from autoflow.plane_rules import default_label_plane_rules
    paths=[np.column_stack((np.arange(5.,46.),np.full(41,20.),np.full(41,z))) for z in (18.,10.)]
    nodes=[(0,1),(2,3)];signs=[1,1]
    if reverse: paths=[p[::-1] for p in paths];nodes=[n[::-1] for n in nodes];signs=[-1,-1]
    if permuted: paths=paths[::-1];nodes=nodes[::-1]
    volume,infos=_landmark_phantom(paths,[np.full(41,17)]*2,nodes,{'LICA':17},[{'flow_sign':v} for v in signs])
    planes,_,_,qc=generate_label_planes(paths,volume,{'LICA':17},{'LICA':default_label_plane_rules()['LICA']},path_info=infos,smoothing_window=1,inter_time=1)
    assert len(planes)==2
    np.testing.assert_allclose([p.center for p in planes],[[18.2,20.,10.],[31.4,20.,10.]])
    assert all(p.placement['branch_inlet_selection']=='lowest_z' for p in planes)


def test_lowest_inlet_phantom_does_not_promote_an_internal_label_fragment():
    from autoflow.algorithms.label_planes import generate_label_planes
    from autoflow.plane_rules import default_label_plane_rules
    path=np.column_stack((np.arange(5.,56.),np.full(51,20.),np.linspace(18.,3.,51)))
    values=np.full(51,17);values[24:28]=0
    volume,infos=_landmark_phantom([path],[values],[(0,1)],{'LICA':17},[{'flow_sign':1}])
    from scipy.ndimage import maximum_filter
    volume=maximum_filter(volume,size=3)
    planes,_,_,qc=generate_label_planes([path],volume,{'LICA':17},{'LICA':default_label_plane_rules()['LICA']},path_info=infos,smoothing_window=1,inter_time=1)
    assert len(planes)==2 and all(p.placement['branch_segment_index']==0 for p in planes)
    assert all(p.center[2]>10 for p in planes)


@pytest.mark.parametrize('name,value',[('PV',14),('SMV',15),('SV',16)])
@pytest.mark.parametrize('sign',[1,-1,None])
def test_portal_flow_endpoint_fallback_phantom_preserves_direction_and_missing_flow(name,value,sign):
    from autoflow.algorithms.label_planes import generate_label_planes
    from autoflow.plane_rules import default_label_plane_rules
    path=np.column_stack((np.arange(5.,56.),np.full(51,20.),np.full(51,10.)))
    volume,infos=_landmark_phantom([path],[np.full(51,value)],[(0,1)],{name:value},[{'flow_sign':sign}])
    rule=default_label_plane_rules()[name]
    planes,_,_,qc=generate_label_planes([path],volume,{name:value},{name:rule},path_info=infos,smoothing_window=1,inter_time=1)
    if sign is None:
        assert not planes and 'flow_direction_unavailable' in qc[0]['selection_warnings']
        return
    expected=([15.,30.,45.] if sign==1 else [45.,30.,15.]) if name=='PV' else [40. if sign==1 else 20.]
    np.testing.assert_allclose([p.center[0] for p in planes],expected)
    assert all(p.placement['connection_fallback'] for p in planes)
    assert all(p.placement['fallback_endpoint']==('flow_start' if name=='PV' else 'flow_end') for p in planes)
    assert all(p.placement['requested_anchor']==rule['placements'][0]['anchor'] for p in planes)
    rule['placements'][0]['connection_fallback']='none'
    strict,_,_,_=generate_label_planes([path],volume,{name:value},{name:rule},path_info=infos,smoothing_window=1,inter_time=1)
    assert not strict


def test_junction_terminal_blob_phantom_preserves_port_and_is_numbering_invariant():
    from itertools import combinations
    from autoflow.algorithms.graph_junctions import simplify_junction_clusters
    points = np.array([[1,5,5],[5,5,5],[6,5,5],[5,6,5],[6,6,5]], float)
    graph = GraphData(points=points, edges=np.array([(0,1)]+list(combinations(range(1,5),2))))
    labels = np.ones((14,14,14), np.int16)
    result, report = simplify_junction_clusters(graph, labels)
    assert report['cycle_rank_after'] == 0
    assert report['local_cleanup']['clusters'][0]['single_ports'][0]['label'] == 1
    assert tuple(points[0]) in set(map(tuple, result.points))
    assert tuple(points[4]) in set(map(tuple, result.points))
    order = np.array([4,0,3,1,2])
    changed, _ = simplify_junction_clusters(GraphData(points=points[order], edges=np.argsort(order)[graph.edges[::-1]]), labels)
    assert _junction_edge_geometry(changed) == _junction_edge_geometry(result)
    untouched, report = simplify_junction_clusters(graph, labels, enabled=False)
    assert untouched is graph and report['removed_edge_count'] == 0


def test_geometry_draft_phantom_gesture_branch_delete_and_connect_undo():
    from autoflow.ui.geometry_editor import GeometryDraft
    points = np.array([[0,0,0],[1,0,0],[2,0,0],[3,1,0],[3,-1,0],[3,0,1]], float)
    edges = np.array([[0,1],[1,2],[2,3],[2,4],[2,5]])
    draft = GeometryDraft(points, edges)
    draft.select_branch(0)
    assert draft.edge_ids == {0,1}
    draft.delete_selection()
    assert len(draft.edges) == 3 and tuple(points[2]) in set(map(tuple,draft.points))
    assert draft.undo() and np.array_equal(draft.edges, edges)
    draft.select('node', 0)
    draft.begin_gesture()
    for distance in range(1,10):draft.move_selection([0,distance,0])
    draft.finish_gesture()
    assert len(draft.undo_stack) == 1
    draft.undo()
    np.testing.assert_array_equal(draft.points, points)
    draft.redo()
    assert draft.points[0,1] == 9
    draft.begin_gesture();draft.move_selection([0,2,0]);draft.cancel_gesture()
    assert draft.points[0,1] == 9
    assert draft.connect(3,4)
    assert not draft.connect(4,3) and not draft.connect(3,3)
    assert len(draft.edges) == 6
    draft.undo();assert len(draft.edges) == 5
    np.testing.assert_array_equal(points[:,1], [0,0,0,1,-1,0])


def _geometry_edit_workspace(tmp_path):
    from autoflow.core.pipeline import PipelineEngine
    ws = Workspace();ws.paths.output_dir = str(tmp_path)
    labels = np.ones((12,12,12),np.int16)
    ws.skeleton_params.label_map = {'vessel':1}
    ws.skeleton_params.label_groups = {'phantom':{'labels':[1]}}
    ws.skeleton_params.min_edge_points = 0
    ws.segmask_raw = labels.copy();ws.segmask_labels_3d=labels.copy();ws.segmask_3d=labels>0
    ws.segmask_binary=ws.segmask_3d[...,None]
    ws.flow_raw=np.zeros(labels.shape+(1,3));ws.flow_raw[...,2]=20
    ws.skeleton_points=np.array([[2+i,5,5] for i in range(7)],float)
    ws.skeleton_spacing=np.ones(3);ws.skeleton_labels=labels.copy()
    ws.group_order=['phantom']
    ws.multilabel_groups={'phantom':{'labels':[1],'skeleton_points':ws.skeleton_points.copy(),
        'segmask_3d':ws.segmask_3d.copy(),'segmask_binary':ws.segmask_binary.copy()}}
    _seed_manual_1mm_phantom_state(ws)
    engine=PipelineEngine();assert engine._step_generate_graph(ws,write_output=False).success
    return ws,engine


def test_geometry_edit_phantom_keeps_manual_edges_and_restores_transaction(tmp_path,monkeypatch):
    from autoflow.core.geometry_edit import prepare_geometry_edit,publish_geometry_edit,geometry_snapshot,restore_geometry_snapshot
    import autoflow.core.pipeline as module
    ws,engine=_geometry_edit_workspace(tmp_path)
    original=geometry_snapshot(ws)
    points=np.array([[2+i,5,5] for i in range(7)],float)
    edges=np.array([[i,i+1] for i in range(6)])
    def must_not_rebuild(*args,**kwargs):raise AssertionError('Manual graph was rebuilt')
    monkeypatch.setattr(module,'build_graph_from_points',must_not_rebuild)
    values=prepare_geometry_edit(ws,engine,'graph','phantom',points,edges)
    np.testing.assert_array_equal(ws.graph.edges,original['values']['graph'].edges)
    publish_geometry_edit(ws,values)
    np.testing.assert_array_equal(ws.graph.points,points)
    np.testing.assert_array_equal(ws.graph.edges,edges)
    assert ws.centerline_paths and sum(len(p)-1 for p in ws.centerline_node_paths)==len(edges)
    restore_geometry_snapshot(ws,original)
    np.testing.assert_array_equal(ws.graph.edges,original['values']['graph'].edges)
    assert not list(tmp_path.iterdir())


def test_geometry_edit_phantom_empty_skeleton_is_not_silently_regenerated(tmp_path):
    from autoflow.core.geometry_edit import prepare_geometry_edit
    ws,engine=_geometry_edit_workspace(tmp_path)
    values=prepare_geometry_edit(ws,engine,'skeleton','phantom',np.empty((0,3)))
    assert len(values['graph'].points)==0 and not values['centerline_paths']
    assert len(ws.skeleton_points)>0


@pytest.mark.skipif(sys.platform == "win32", reason="Linux SSH renderer smoke")
def test_geometry_editor_gui_smoke_selection_history_and_escape(tmp_path,monkeypatch):
    from PySide6 import QtWidgets,QtCore,QtTest
    application=_smoke_qt_application(monkeypatch)
    monkeypatch.setenv("AUTOFLOW_SSH_RENDERING", "1")
    import autoflow.ui.app as ui_app
    from autoflow.ui.remote_plotter import RemotePlotter
    monkeypatch.setattr(ui_app, "RemotePlotter", RemotePlotter, raising=False)
    window=ui_app.MainWindow()
    try:
        ws,engine=_geometry_edit_workspace(tmp_path)
        render_settings=window.workspace.render_settings
        window.workspace.__dict__.update(ws.__dict__)
        window.workspace.render_settings=render_settings
        graph=window.workspace.graph
        window.show();application.processEvents()
        window._enter_interactive_edit('graph',graph.points,graph.edges,group_name='phantom')
        application.processEvents()
        assert window._edit_btn_save.text()=='Done' and window._edit_btn_edge.text()=='Connect'
        assert window._geometry_draft.nodes==set() and window.tree_objects.isEnabled()
        assert not window.btn_add_plane.isEnabled()
        window._add_manual_plane()
        assert window._plane_edit_session is None
        draft=window._geometry_draft
        draft.select('node',0);draft.begin_gesture();draft.move_selection([0,1,0]);draft.finish_gesture()
        window._render_geometry_draft()
        QtTest.QTest.keyClick(window.plotter,QtCore.Qt.Key_Z,QtCore.Qt.ControlModifier)
        np.testing.assert_array_equal(draft.points,draft.original_points)
        QtTest.QTest.keyClick(window.plotter,QtCore.Qt.Key_Z,QtCore.Qt.ControlModifier|QtCore.Qt.ShiftModifier)
        assert draft.changed
        window._force_exit_edit()
        assert window._edit_mode=='graph' and draft.changed and not draft.nodes
        window._exit_interactive_edit(False)
        np.testing.assert_array_equal(window.workspace.graph.points,graph.points)
        assert window._edit_mode is None
        window._enter_interactive_edit('graph',graph.points,graph.edges,group_name='phantom')
        monkeypatch.setattr(window,'_run_modal_task',lambda *a: (_ for _ in ()).throw(AssertionError('No-op edit recomputed')))
        window._exit_interactive_edit(True)
        assert window._edit_mode is None
    finally:
        if window._edit_mode:window._exit_interactive_edit(False)
        window.plotter.close();window.close();application.processEvents()


def test_geometry_edit_phantom_preserves_manual_plane_and_rolls_back_failed_rebuild(tmp_path, monkeypatch):
    from autoflow.core.geometry_edit import prepare_geometry_edit,publish_geometry_edit
    from autoflow.core.plane_state import stamp_metric,metric_validity
    ws,engine=_geometry_edit_workspace(tmp_path)
    plane=PlaneData(center=np.array([5.,5.,5.]),normal=np.array([0.,0.,1.]),path_index=-1,group_name='phantom',placement={'manual_geometry':True})
    plane.roi_polygon_uv_mm=[[-1,-1],[1,-1],[1,1],[-1,1]]
    plane.metrics=stamp_metric(plane,{'flowrate_mL_s':[1.]},0)
    ws.planes=[plane];ws.derived.plane_metrics=[plane.metrics]
    points=ws.graph.points.copy();points[0,1]+=.2
    before=ws.graph.points.copy()
    values=prepare_geometry_edit(ws,engine,'graph','phantom',points,ws.graph.edges)
    np.testing.assert_array_equal(ws.graph.points,before)
    assert ws.planes[0] is plane and metric_validity(plane)=='current'
    kept=next(p for p in values['planes'] if p.plane_id==plane.plane_id)
    assert len(values['planes'])==1 and kept.roi_polygon_uv_mm==plane.roi_polygon_uv_mm
    np.testing.assert_array_equal(kept.center,plane.center)
    np.testing.assert_array_equal(kept.normal,plane.normal)
    assert metric_validity(kept)=='stale'
    def fail(*args,**kwargs):raise RuntimeError('Synthetic plane rebuild failure')
    monkeypatch.setattr(engine,'_step_generate_planes',fail)
    with pytest.raises(RuntimeError,match='Synthetic'):
        prepare_geometry_edit(ws,engine,'graph','phantom',points,ws.graph.edges)
    np.testing.assert_array_equal(ws.graph.points,before)
    assert ws.planes[0] is plane and metric_validity(plane)=='current'


@pytest.mark.skipif(sys.platform == "win32", reason="Linux SSH renderer smoke")
@pytest.mark.parametrize('mirrored',[False,True])
def test_geometry_editor_gui_smoke_pointer_picking_connection_and_box(tmp_path,monkeypatch,mirrored):
    from PySide6 import QtCore,QtTest
    application=_smoke_qt_application(monkeypatch)
    monkeypatch.setenv('AUTOFLOW_SSH_RENDERING','1')
    import autoflow.ui.app as ui_app
    from autoflow.ui.remote_plotter import RemotePlotter
    monkeypatch.setattr(ui_app,'RemotePlotter',RemotePlotter,raising=False)
    window=ui_app.MainWindow()
    try:
        ws,engine=_geometry_edit_workspace(tmp_path)
        render_settings=window.workspace.render_settings
        window.workspace.__dict__.update(ws.__dict__)
        window.workspace.render_settings=render_settings
        window.workspace.origin=np.array([11.,-7.,23.])
        points=np.array([[2,5,5],[5,5,5],[8,5,5],[5,8,5]],float)
        edges=np.array([[0,1],[1,2],[1,3]])
        if mirrored:
            window.scene._display_axis_signs=np.array([-1.,1.,1.])
            window.scene.reset_display_reference()
        window.show();application.processEvents()
        window._enter_interactive_edit('graph',points,edges,group_name='phantom')
        window.plotter.view_xy();window.plotter.reset_camera()
        window.plotter.camera.parallel_projection=True
        window.plotter.render();application.processEvents()
        def at(index):
            p=window._geometry_screen_points()[index]
            return QtCore.QPoint(round(p[0]),round(p[1]))
        draft=window._geometry_draft
        first=at(0)
        QtTest.QTest.mouseMove(window.plotter,first);application.processEvents()
        QtTest.QTest.mouseClick(window.plotter,QtCore.Qt.LeftButton,pos=first)
        assert draft.nodes=={0}
        target=first+QtCore.QPoint(24,-17)
        QtTest.QTest.mousePress(window.plotter,QtCore.Qt.LeftButton,pos=first)
        QtTest.QTest.mouseMove(window.plotter,target)
        QtTest.QTest.mouseRelease(window.plotter,QtCore.Qt.LeftButton,pos=target)
        assert len(draft.undo_stack)==1
        projected=window._geometry_screen_points()[0]
        np.testing.assert_allclose(projected[:2],[target.x(),target.y()],atol=1.1)
        window._geometry_history(False)
        np.testing.assert_array_equal(draft.points,points)
        window._edit_btn_edge.click()
        for _ in range(2):
            QtTest.QTest.mouseMove(window.plotter,at(0));application.processEvents()
            QtTest.QTest.mouseClick(window.plotter,QtCore.Qt.LeftButton,pos=at(0))
            assert window._geometry_source == 0
            QtTest.QTest.mouseMove(window.plotter,at(2));application.processEvents()
            assert 'preview' in window._geometry_actors
            QtTest.QTest.mouseClick(window.plotter,QtCore.Qt.LeftButton,pos=at(2))
            assert len(draft.edges)==4  # Connecting an existing pair never deletes it.
        window._force_exit_edit()
        assert window._edit_mode=='graph' and draft.changed
        window.plotter.render();application.processEvents()
        screen=window._geometry_screen_points()
        low=np.floor(screen[:,:2].min(0)-20).astype(int);high=np.ceil(screen[:,:2].max(0)+20).astype(int)
        start,end=QtCore.QPoint(*map(int,low)),QtCore.QPoint(*map(int,high))
        QtTest.QTest.mousePress(window.plotter,QtCore.Qt.LeftButton,QtCore.Qt.ShiftModifier,pos=start)
        QtTest.QTest.mouseMove(window.plotter,end)
        QtTest.QTest.mouseRelease(window.plotter,QtCore.Qt.LeftButton,QtCore.Qt.ShiftModifier,pos=end)
        assert draft.nodes=={0,1,2,3}
        window._exit_interactive_edit(False)
    finally:
        if window._edit_mode:window._exit_interactive_edit(False)
        window.plotter.close();window.close();application.processEvents()


@pytest.mark.skipif(sys.platform == "win32", reason="Linux SSH renderer smoke")
def test_geometry_editor_gui_smoke_done_undo_restores_paths_and_files(tmp_path,monkeypatch):
    from PySide6 import QtCore,QtTest
    application=_smoke_qt_application(monkeypatch)
    monkeypatch.setenv('AUTOFLOW_SSH_RENDERING','1')
    import autoflow.ui.app as ui_app
    from autoflow.ui.remote_plotter import RemotePlotter
    monkeypatch.setattr(ui_app,'RemotePlotter',RemotePlotter,raising=False)
    window=ui_app.MainWindow()
    try:
        ws,engine=_geometry_edit_workspace(tmp_path)
        settings=window.workspace.render_settings
        window.workspace.__dict__.update(ws.__dict__);window.workspace.render_settings=settings
        before=window.workspace.graph.points.copy()
        original_edges=window.workspace.graph.edges.copy()
        window.show();application.processEvents()
        window._enter_interactive_edit('graph',before,original_edges,group_name='phantom')
        draft=window._geometry_draft
        draft.select('node',0);draft.begin_gesture();draft.move_selection([0,.25,0]);draft.finish_gesture()
        expected=draft.points.copy()
        window._exit_interactive_edit(True)
        assert window._edit_mode is None, window.console.toPlainText()[-4000:]
        np.testing.assert_array_equal(window.workspace.graph.points,expected)
        np.testing.assert_array_equal(window.workspace.graph.edges,original_edges)
        assert window.workspace.centerline_paths and (tmp_path/'graph_hierarchy.json').exists()
        assert window._plane_undo_stack.undoText()=='Edit graph'
        window.plotter.setFocus();QtTest.QTest.keyClick(window.plotter,QtCore.Qt.Key_Z,QtCore.Qt.ControlModifier)
        np.testing.assert_array_equal(window.workspace.graph.points,before)
        QtTest.QTest.keyClick(window.plotter,QtCore.Qt.Key_Z,QtCore.Qt.ControlModifier|QtCore.Qt.ShiftModifier)
        np.testing.assert_array_equal(window.workspace.graph.points,expected)
    finally:
        if window._edit_mode:window._exit_interactive_edit(False)
        window.plotter.close();window.close();application.processEvents()


@pytest.mark.parametrize('mode',['skeleton','graph'])
def test_geometry_edit_phantom_delete_all_with_existing_planes_stays_empty(tmp_path,mode):
    from autoflow.core.geometry_edit import prepare_geometry_edit
    ws,engine=_geometry_edit_workspace(tmp_path)
    plane=PlaneData(center=np.array([5.,5.,5.]),normal=np.array([0.,0.,1.]),path_index=-1,group_name='phantom',placement={'manual_geometry':True})
    ws.planes=[plane]
    values=prepare_geometry_edit(ws,engine,mode,'phantom',np.empty((0,3)),np.empty((0,2),int))
    assert len(values['graph'].points)==0 and not values['centerline_paths']
    assert len(values['planes'])==1 and values['planes'][0].plane_id==plane.plane_id
    assert values['planes'][0].path_index==-1
    assert len(ws.graph.points)>0 and ws.planes[0] is plane


def test_plane_coordinates_phantom_excludes_contours_but_analysis_exports_keep_them(tmp_path):
    from autoflow.plane_io import save_planes_json,save_planes_h5
    ws=Workspace();ws.origin=np.array([10.,20.,30.])
    plane=PlaneData(center=np.array([2.,3.,4.]),normal=np.array([0.,0.,2.]),path_index=-1,
        roi_polygon_uv_mm=[[0,0],[2,0],[0,2]],roi_edit_operations={'0':[{'mode':'replace','polygon':[[0,0],[3,0],[0,3]]}]})
    plane.metrics={'flowrate_mL_s':[123.]};ws.planes=[plane]
    save_plane_positions(ws,str(tmp_path/'coordinates.json'))
    payload=load_plane_position_payload(tmp_path/'coordinates.json')
    assert payload['content']=='plane_geometry'
    record=payload['planes'][0]
    assert 'roi_polygon_uv_mm' not in record and 'roi_edit_operations' not in record and 'flowrate_mL_s' not in record
    np.testing.assert_allclose(record['center_world_mm'],[12,23,34]);np.testing.assert_allclose(record['normal'],[0,0,1])
    save_planes_json(ws,str(tmp_path/'analysis.json'));save_planes_h5(ws,str(tmp_path/'analysis.h5'))
    assert json.loads((tmp_path/'analysis.json').read_text())[0]['roi_edit_operations']==plane.roi_edit_operations
    with h5py.File(tmp_path/'analysis.h5') as h5:
        full=json.loads(h5['plane_0000/payload_json'][()].decode())
        assert full['roi_polygon_uv_mm']==plane.roi_polygon_uv_mm
    assert plane.roi_polygon_uv_mm and plane.roi_edit_operations


@pytest.mark.parametrize('mode',['world','local','path_relative'])
def test_plane_coordinates_phantom_legacy_contours_are_ignored(mode):
    source=Workspace();source.origin=np.array([10.,20.,30.])
    plane=PlaneData(center=np.array([5.,0.,0.]),normal=np.array([1.,0.,0.]),path_index=0,distance=5.,group_name='vessel')
    plane.roi_polygon_uv_mm=[[0,0],[2,0],[0,2]]
    plane.roi_edit_operations={'0':[{'mode':'replace','polygon':[[0,0],[3,0],[0,3]]}]}
    plane.roi_revision=7;plane.metrics={'flowrate_mL_s':[123.]};source.planes=[plane]
    payload={'origin':[10,20,30],'planes':build_plane_records(source)}
    payload['planes'][0].update(group_path_index=0,path_fraction=.5)
    target=Workspace();target.origin=np.array([100.,200.,300.])
    target.centerline_paths_smooth=[np.array([[0,0,0],[20,0,0]],float)]
    target.path_info=[{'group_name':'vessel'}]
    target.multilabel_groups={'vessel':{'path_index_offset':0,'centerline_paths_smooth':target.centerline_paths_smooth}}
    imported,report=project_planes_to_workspace(payload,target,mapping_mode=mode,return_report=True)
    assert len(imported)==1 and report['ignored_contour_count']==1
    assert imported[0].roi_polygon_uv_mm==[] and imported[0].roi_edit_operations=={} and imported[0].metrics=={}
    assert imported[0].roi_revision==0
    np.testing.assert_allclose(imported[0].center,{'world':[-85,-180,-270],'local':[5,0,0],'path_relative':[10,0,0]}[mode])
    assert source.planes[0].roi_revision==7 and source.planes[0].roi_edit_operations


def test_plane_coordinates_phantom_rejects_nonfinite_geometry(tmp_path):
    ws=Workspace();ws.planes=[PlaneData(center=np.array([np.nan,0,0]),normal=np.array([1,0,0]))]
    with pytest.raises(ValueError,match='finite'):save_plane_positions(ws,str(tmp_path/'bad.json'))
    assert not (tmp_path/'bad.json').exists()
    payload=[{'center':[1,2,3],'normal':[0,0,0]},{'center':[np.inf,0,0],'normal':[1,0,0]},{'center':[1,2,3],'normal':[0,1,0]}]
    planes,report=project_planes_to_workspace(payload,Workspace(),return_report=True)
    assert len(planes)==1 and report['skipped_count']==2


def test_plane_coordinate_settings_smoke_select_clear_and_cancel(tmp_path,monkeypatch):
    from autoflow.ui.action_settings import ActionSettingsDialog,action_sections
    from PySide6 import QtWidgets
    app=_smoke_qt_application(monkeypatch)
    ws=Workspace();source=tmp_path/'reference.json';source.write_text('{"planes":[]}')
    dialog=ActionSettingsDialog('Generate Planes',action_sections(StepId.GENERATE_PLANES,ws),ws,Workspace())
    try:
        dialog.show();app.processEvents()
        file=dialog.controls['plane_gen_params.positions_file']
        assert file.placeholderText()=='Automatic'
        assert not dialog.rows['plane_gen_params.plane_mode'].isHidden()
        assert dialog.rows['plane_gen_params.import_mode'].isHidden()
        monkeypatch.setattr(QtWidgets.QFileDialog,'getOpenFileName',lambda *args,**kw:(str(source),''))
        dialog.rows['plane_gen_params.positions_file'].findChild(QtWidgets.QPushButton).click()
        app.processEvents()
        assert file.text()==str(source)
        assert dialog.rows['plane_gen_params.plane_mode'].isHidden()
        assert dialog.rows['plane_gen_params.label_rules'].isHidden()
        assert not dialog.rows['plane_gen_params.import_mode'].isHidden()
        assert not dialog.tabs.isTabVisible(1)
        assert dialog.values()['plane_gen_params.positions_file']==str(source)
        file.clear();app.processEvents()
        assert not dialog.rows['plane_gen_params.plane_mode'].isHidden() and dialog.tabs.isTabVisible(1)
        file.setText(str(tmp_path/'missing.json'))
        with pytest.raises(ValueError,match='does not exist'):dialog.values()
        dialog.reject()
        assert ws.plane_gen_params.positions_file==''
    finally:dialog.close();dialog.deleteLater();app.processEvents()


@pytest.mark.parametrize('mode',['world','local','path_relative'])
def test_plane_coordinates_generation_phantom_replaces_without_contours_and_keeps_paths(tmp_path,mode):
    from autoflow.core.pipeline import PipelineEngine
    from autoflow.core.models import ObjectKind
    source=tmp_path/'reference.json'
    source.write_text(json.dumps({'origin':[10,20,30], 'planes':[
        {'center':[5,0,0],'normal':[1,0,0],'path_index':0,'group_name':'vessel','group_path_index':0,
         'path_fraction':.5,'roi_polygon_uv_mm':[[0,0],[2,0],[0,2]],'flowrate_mL_s':[123.]}]}))
    ws=Workspace();ws.data_loaded=True;ws.origin=np.array([100,200,300.])
    path=np.array([[0,0,0],[20,0,0]],float)
    ws.centerline_paths=[path.copy()];ws.centerline_paths_smooth=[path.copy()]
    ws.path_info=[{'group_name':'vessel'}]
    ws.multilabel_groups={'vessel':{'path_index_offset':0,'centerline_paths_smooth':[path.copy()]}}
    ws.planes=[PlaneData(center=np.array([999,0,0]),normal=np.array([1,0,0]),metrics={'flowrate_mL_s':[456.]})]
    ws.derived.plane_metrics=[ws.planes[0].metrics]
    ws.pipeline.mark_done(StepId.COMPUTE_PLANE_METRICS)
    ws.add_object(name='old',kind=ObjectKind.PLANE,data_key='plane_vessel_0')
    ws.plane_gen_params.positions_file=str(source);ws.plane_gen_params.import_mode=mode
    engine=PipelineEngine();result=engine._step_generate_planes(ws,write_output=False)
    assert result.success and not result.skipped and len(ws.planes)==1
    expected={'world':[-85,-180,-270],'local':[5,0,0],'path_relative':[10,0,0]}[mode]
    np.testing.assert_allclose(ws.planes[0].center,expected)
    np.testing.assert_allclose(ws.centerline_paths_smooth[0],path)
    assert not ws.planes[0].roi_polygon_uv_mm and not ws.planes[0].metrics
    assert ws.derived.plane_qc['ignored_contour_count']==1 and not ws.derived.plane_metrics
    assert ws.pipeline.is_done(StepId.GENERATE_PLANES) and not ws.pipeline.is_done(StepId.COMPUTE_PLANE_METRICS)
    assert len([o for o in ws.scene_objects.values() if o.kind==ObjectKind.PLANE])==1
    before=ws.planes[0]
    source.write_text('{"planes":[]}')
    with pytest.raises(ValueError,match='importable'):engine._step_generate_planes(ws,write_output=False)
    assert ws.planes[0] is before
    ws.plane_gen_params.positions_file=''
    assert engine._step_generate_planes(ws,write_output=False).skipped  # automatic placement requires segmentation


def test_plane_coordinates_config_phantom_shared_source_and_workspace_roundtrip(tmp_path):
    from autoflow.config import load_config_bundle,bundle_to_autoflow_kwargs
    from autoflow.core.models import PlaneGenerationParams
    from autoflow import AutoFlowConfig,build_workspace
    (tmp_path/'planes.json').write_text(json.dumps({'positions_file':'reference.json','import_mode':'local'}))
    bundle=load_config_bundle(str(tmp_path));kwargs=bundle_to_autoflow_kwargs(bundle)
    assert kwargs['reuse_planes']==str(tmp_path/'reference.json') and kwargs['plane_import_mode']=='local'
    ws=build_workspace(AutoFlowConfig.from_config_dir(str(tmp_path)))
    assert ws.plane_gen_params.positions_file==kwargs['reuse_planes']
    restored=PlaneGenerationParams.from_dict(ws.plane_gen_params.to_dict())
    assert restored.positions_file==kwargs['reuse_planes'] and restored.import_mode=='local'
    bundle['batch']['reuse_planes']='legacy.json'
    assert bundle_to_autoflow_kwargs(bundle)['reuse_planes']=='legacy.json'



def test_plane_coordinates_batch_phantom_uses_file_in_placement_stage(tmp_path):
    source=tmp_path/'target.h5'
    with h5py.File(source,'w') as h:
        h['mag']=np.ones((4,4,4,2),np.float32)
        h['flow']=np.zeros((4,4,4,2,3),np.float32)
        h['Resolution']=np.ones(3)
        h['VENC']=np.full(3,100.)
        h['SpatialOrder']=np.asarray(['LR','AP','FH'],dtype='S4')
        h['VENCOrder']=np.asarray(['LR','AP','FH'],dtype='S4')
    positions=tmp_path/'reference';positions.mkdir()
    (positions/'plane_positions.json').write_text(json.dumps({'planes':[{
        'center_world_mm':[1.,2.,3.],'normal':[0.,0.,1.],
        'roi_polygon_uv_mm':[[0,0],[1,0],[0,1]],'flowrate_mL_s':[123.]}]}))
    output=tmp_path/'out'
    summary=run_case(str(source),output_dir=str(output),config=AutoFlowConfig(
        reuse_planes=str(positions),skip_derived=True,skip_plane_metrics=True,
        background_phase_correction=False,background_phase_write_cache=False))
    assert summary['reused_planes_file']==str(positions/'plane_positions.json')
    assert summary['plane_import_report']['imported_count']==1
    assert summary['plane_import_report']['ignored_contour_count']==1
    assert 'planes' in summary['stage_times_sec'] and 'reuse_planes' not in summary['stage_times_sec']
    payload=load_plane_position_payload(output/'plane_positions.json')
    assert len(payload['planes'])==1 and 'roi_polygon_uv_mm' not in payload['planes'][0]
    np.testing.assert_allclose(payload['planes'][0]['center_world_mm'],[1,2,3])
    qc=json.loads((output/'plane_qc.json').read_text())
    assert qc['plane_mode']=='imported' and qc['actual_plane_count']==1
