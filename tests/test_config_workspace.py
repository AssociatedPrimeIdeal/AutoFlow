import autoflow.api as api
from autoflow import AutoFlowConfig, build_workspace


def test_autoflow_config_defaults_cover_workspace_and_rendering_contract():
    cfg = AutoFlowConfig()

    assert cfg.inputs == []
    assert cfg.output_dir == "./results"

    assert cfg.skip_derived is False
    assert cfg.skip_plane_metrics is False
    assert cfg.use_multithread is True
    assert cfg.reuse_planes == ""
    assert cfg.dicom_read_workers == 1

    assert cfg.use_center_plane is True
    assert cfg.cross_section_dist == 5.0
    assert cfg.start_dist == 5.0
    assert cfg.end_dist == 0.0

    assert cfg.remove_small_cc is True
    assert cfg.min_cc_volume == 50.0

    assert cfg.seed_ratio == 0.02
    assert cfg.tube_radius == 0.05

    assert cfg.fps == 12
    assert cfg.plane_rotation_frames == 180
    assert cfg.make_plane_video is False
    assert cfg.make_wss_video is False
    assert cfg.make_streamlines_video is False
    assert cfg.make_tke_video is False

    assert cfg.camera_view == "right"
    assert cfg.camera_distance_scale == 1.5
    assert cfg.rotate_dynamic_video is True
    assert cfg.dynamic_rotation_frames == 180
    assert cfg.dynamic_rotation_elevation_deg == 10.0
    assert cfg.dynamic_time_repeat == 3

    assert cfg.add_plane_idx is False
    assert cfg.add_path_idx is False

    assert cfg.wss_clim == (0.0, 10.0)
    assert cfg.wss_bar_cfg == api.DEFAULT_WSS_BAR_CFG
    assert cfg.wss_bar_cfg is not api.DEFAULT_WSS_BAR_CFG

    assert cfg.tke_clim == (0.0, 100.0)
    assert cfg.tke_bar_cfg == api.DEFAULT_TKE_BAR_CFG
    assert cfg.tke_bar_cfg is not api.DEFAULT_TKE_BAR_CFG

    assert cfg.streamline_clim == (0.0, 1)
    assert cfg.streamline_bar_cfg == api.DEFAULT_STREAMLINE_BAR_CFG
    assert cfg.streamline_bar_cfg is not api.DEFAULT_STREAMLINE_BAR_CFG


def test_build_workspace_maps_default_config_values():
    ws = build_workspace()

    assert ws.plane_gen_params.to_dict() == {
        "use_center_plane": True,
        "cross_section_distance": 5.0,
        "start_distance": 5.0,
        "end_distance": 0.0,
        "smoothing_window": 15,
        "smoothing_polyorder": 2,
        "inter_time": 10,
    }
    assert ws.skeleton_params.to_dict() == {
        "remove_small_cc": True,
        "min_cc_volume_mm3": 50.0,
        "do_closing": True,
        "do_opening": False,
        "gaussian_sigma": 0.5,
        "gaussian_enabled": True,
    }
    assert ws.streamline_params.to_dict() == {
        "seed_ratio": 0.02,
        "max_steps": 2000,
        "min_seeds": 50,
        "terminal_speed": 0.01,
        "rng_seed": 0,
    }
    assert getattr(ws.streamline_params, "tube_radius") == 0.05
    assert ws.loader_params.dicom_read_workers == 1


def test_build_workspace_maps_custom_plane_generation_parameters():
    config = AutoFlowConfig(
        use_center_plane=False,
        cross_section_dist=12.5,
        start_dist=1.5,
        end_dist=2.5,
    )

    ws = build_workspace(config)

    assert ws.plane_gen_params.to_dict() == {
        "use_center_plane": False,
        "cross_section_distance": 12.5,
        "start_distance": 1.5,
        "end_distance": 2.5,
        "smoothing_window": 15,
        "smoothing_polyorder": 2,
        "inter_time": 10,
    }


def test_build_workspace_maps_skeleton_and_streamline_parameters():
    config = AutoFlowConfig(
        remove_small_cc=False,
        min_cc_volume=42.0,
        seed_ratio=0.03,
        tube_radius=0.12,
        dicom_read_workers=5,
    )

    ws = build_workspace(config)

    assert ws.skeleton_params.to_dict() == {
        "remove_small_cc": False,
        "min_cc_volume_mm3": 42.0,
        "do_closing": True,
        "do_opening": False,
        "gaussian_sigma": 0.5,
        "gaussian_enabled": True,
    }
    assert ws.streamline_params.to_dict() == {
        "seed_ratio": 0.03,
        "max_steps": 2000,
        "min_seeds": 50,
        "terminal_speed": 0.01,
        "rng_seed": 0,
    }
    assert getattr(ws.streamline_params, "tube_radius") == 0.12
    assert ws.loader_params.dicom_read_workers == 5


def test_run_case_forwards_rendering_defaults_to_process_single(monkeypatch, tmp_path):
    captured = {}

    def fake_process_single(input_path, case_dir, **kwargs):
        captured["input_path"] = input_path
        captured["case_dir"] = case_dir
        captured.update(kwargs)
        return {"status": "ok", "case_dir": case_dir}

    monkeypatch.setattr(api, "process_single", fake_process_single)

    cfg = AutoFlowConfig()
    out_dir = tmp_path / "case"
    summary = api.run_case("demo_input.h5", output_dir=str(out_dir), config=cfg)

    assert summary == {"status": "ok", "case_dir": str(out_dir)}
    assert captured["input_path"] == "demo_input.h5"
    assert captured["case_dir"] == str(out_dir)

    assert captured["workspace"].plane_gen_params.cross_section_distance == 5.0
    assert captured["fps"] == 12
    assert captured["plane_rotation_frames"] == 180
    assert captured["make_plane_video"] is False
    assert captured["make_wss_video"] is False
    assert captured["make_streamlines_video"] is False
    assert captured["make_tke_video"] is False
    assert captured["camera_view"] == "right"
    assert captured["camera_distance_scale"] == 1.5
    assert captured["rotate_dynamic_video"] is True
    assert captured["dynamic_rotation_frames"] == 180
    assert captured["dynamic_rotation_elevation_deg"] == 10.0
    assert captured["dynamic_time_repeat"] == 3
    assert captured["add_plane_idx"] is False
    assert captured["add_path_idx"] is False
    assert captured["wss_clim"] == (0.0, 10.0)
    assert captured["tke_clim"] == (0.0, 100.0)
    assert captured["streamline_clim"] == (0.0, 1)
    assert captured["wss_bar_cfg"] == api.DEFAULT_WSS_BAR_CFG
    assert captured["tke_bar_cfg"] == api.DEFAULT_TKE_BAR_CFG
    assert captured["streamline_bar_cfg"] == api.DEFAULT_STREAMLINE_BAR_CFG
    assert captured["wss_bar_cfg"] is not cfg.wss_bar_cfg
    assert captured["tke_bar_cfg"] is not cfg.tke_bar_cfg
    assert captured["streamline_bar_cfg"] is not cfg.streamline_bar_cfg
