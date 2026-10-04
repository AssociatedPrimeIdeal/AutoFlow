# Configuration parameters

Complete supported defaults from `load_config_bundle("configs")`, including shipped JSON overrides. Direct `AutoFlowConfig()` class defaults can differ: compare [API fields](api-parameters.md) and [configuration precedence](../developer/config-system.md).

Lengths are mm, input velocity is cm/s, RR is ms, density is kg/m3 and viscosity is mPa s unless a row states otherwise. Rendering changes display, not numerical metrics. Expand dictionary controls in [structured parameters](parameter-schemas.md).

## batch

Code owner: `autoflow/processing.py`.

| Parameter | Type | Effective default | Where configured | Effect and units | Code owner |
| --- | --- | --- | --- | --- | --- |
| `output_dir` | string | `"./results"` | `configs/batch.json` | Root directory for per-case exports. Relative paths resolve from the process working directory. | `autoflow/processing.py` |
| `skip_derived` | bool | `false` | `configs/batch.json` | Suppress requested WSS, TKE and pressure analysis and their exports; optional metrics otherwise remain opt-in. | `autoflow/processing.py` |
| `skip_wss` | bool | `false` | `configs/batch.json` | Suppress WSS only, including volume export and derived plane summaries. | `autoflow/processing.py` |
| `skip_tke` | bool | `false` | `configs/batch.json` | Suppress TKE only; does not manufacture TKE for unsupported inputs. | `autoflow/processing.py` |
| `skip_pressure_gradient` | bool | `false` | `configs/batch.json` | Suppress PG, relative-pressure reconstruction and centreline pressure-drop outputs. | `autoflow/processing.py` |
| `skip_plane_metrics` | bool | `false` | `configs/batch.json` | Suppress plane metric calculation and its JSON/H5 exports. PWV also needs plane metrics. | `autoflow/processing.py` |
| `use_multithread` | bool | `true` | `configs/batch.json` | Enable process plane metrics with memory-mapped inputs: below 128 planes, use four workers only at 1920 or more plane-phase evaluations; smaller jobs stay serial. At least 128 planes use up to eight workers. CPU/plane counts cap workers. | `autoflow/processing.py` |
| `reuse_planes` | string | `""` | `configs/batch.json` | Path to a plane-position JSON or per-case directory to reuse. Empty string generates planes normally. | `autoflow/processing.py` |

## ui

Code owner: `autoflow/ui/app.py`.

| Parameter | Type | Effective default | Where configured | Effect and units | Code owner |
| --- | --- | --- | --- | --- | --- |
| `background_color` | string | `"#000000"` | `configs/ui.json` | Background colour of the 3D scene, as a VTK colour name or hexadecimal RGB. Changes display only. | `autoflow/ui/app.py` |

## loader

Code owner: `autoflow/algorithms/data/h5_loader.py; autoflow/algorithms/inputs.py; autoflow/algorithms/dicom_conversion.py; autoflow/algorithms/phase_correction.py`.

| Parameter | Type | Effective default | Where configured | Effect and units | Code owner |
| --- | --- | --- | --- | --- | --- |
| `background_phase_correction.enabled` | bool | `false` | `configs/loader.json` | Enable an explicit background-correction stage after loading, before segmentation. Compatible H5 correction caches may be reused unless force_recompute is true. | `autoflow/algorithms/data/h5_loader.py; autoflow/algorithms/inputs.py; autoflow/algorithms/dicom_conversion.py; autoflow/algorithms/phase_correction.py` |
| `background_phase_correction.method` | string | `"wrls_arto"` | `configs/loader.json` | Correction backend: wrls_arto uses robust weighted regression with automatic rejection; msac uses polynomial sample consensus. | `autoflow/algorithms/data/h5_loader.py; autoflow/algorithms/inputs.py; autoflow/algorithms/dicom_conversion.py; autoflow/algorithms/phase_correction.py` |
| `background_phase_correction.corr_fit_order` | int | `3` | `configs/loader.json` | Polynomial order for MSAC correction. Higher orders add spatial flexibility and fitting cost; not the WRLS fixed basis order. | `autoflow/algorithms/data/h5_loader.py; autoflow/algorithms/inputs.py; autoflow/algorithms/dicom_conversion.py; autoflow/algorithms/phase_correction.py` |
| `background_phase_correction.threshold` | float | `0.2` | `configs/loader.json` | MSAC residual threshold as a fraction of VENC; controls static-tissue inlier acceptance. | `autoflow/algorithms/data/h5_loader.py; autoflow/algorithms/inputs.py; autoflow/algorithms/dicom_conversion.py; autoflow/algorithms/phase_correction.py` |
| `background_phase_correction.wrls_lambda` | float | `5.0` | `configs/loader.json` | L1 penalty on WRLS polynomial coefficients; larger values impose stronger regularization. | `autoflow/algorithms/data/h5_loader.py; autoflow/algorithms/inputs.py; autoflow/algorithms/dicom_conversion.py; autoflow/algorithms/phase_correction.py` |
| `background_phase_correction.wrls_magnitude_threshold` | float | `0.04` | `configs/loader.json` | Minimum reference-magnitude fraction for the static-tissue fitting mask; low-signal voxels are discarded. | `autoflow/algorithms/data/h5_loader.py; autoflow/algorithms/inputs.py; autoflow/algorithms/dicom_conversion.py; autoflow/algorithms/phase_correction.py` |
| `background_phase_correction.wrls_mid_fov_fraction` | float | `0.5` | `configs/loader.json` | Fraction of the two in-plane axes used for the central fitting region; restricts initial static-tissue support. | `autoflow/algorithms/data/h5_loader.py; autoflow/algorithms/inputs.py; autoflow/algorithms/dicom_conversion.py; autoflow/algorithms/phase_correction.py` |
| `background_phase_correction.wrls_mid_slice_fraction` | float | `0.65` | `configs/loader.json` | Fraction of the slice axis retained for the central fitting region. | `autoflow/algorithms/data/h5_loader.py; autoflow/algorithms/inputs.py; autoflow/algorithms/dicom_conversion.py; autoflow/algorithms/phase_correction.py` |
| `background_phase_correction.wrls_arto_iterations` | int | `4` | `configs/loader.json` | Number of automatic rejection/refitting rounds after the initial WRLS estimate. | `autoflow/algorithms/data/h5_loader.py; autoflow/algorithms/inputs.py; autoflow/algorithms/dicom_conversion.py; autoflow/algorithms/phase_correction.py` |
| `background_phase_correction.wrls_tau` | float | `3.0` | `configs/loader.json` | Residual acceptance width in multiples of the fitted central Gaussian standard deviation. | `autoflow/algorithms/data/h5_loader.py; autoflow/algorithms/inputs.py; autoflow/algorithms/dicom_conversion.py; autoflow/algorithms/phase_correction.py` |
| `background_phase_correction.wrls_delta` | float | `2.0` | `configs/loader.json` | Initial separation of the outer ARTO Gaussian means from zero, in residual standard deviations. | `autoflow/algorithms/data/h5_loader.py; autoflow/algorithms/inputs.py; autoflow/algorithms/dicom_conversion.py; autoflow/algorithms/phase_correction.py` |
| `background_phase_correction.wrls_central_probability` | float | `0.5` | `configs/loader.json` | Prior/minimum mixture probability for the static-tissue central Gaussian; lies between zero and one. | `autoflow/algorithms/data/h5_loader.py; autoflow/algorithms/inputs.py; autoflow/algorithms/dicom_conversion.py; autoflow/algorithms/phase_correction.py` |
| `background_phase_correction.wrls_fista_iterations` | int | `5000` | `configs/loader.json` | Maximum iterations of the sparse WRLS coefficient solver; increases fitting work without changing polynomial degree. | `autoflow/algorithms/data/h5_loader.py; autoflow/algorithms/inputs.py; autoflow/algorithms/dicom_conversion.py; autoflow/algorithms/phase_correction.py` |
| `background_phase_correction.wrls_gmm_iterations` | int | `1000` | `configs/loader.json` | Maximum EM iterations per ARTO Gaussian-mixture fit; early convergence may stop sooner. | `autoflow/algorithms/data/h5_loader.py; autoflow/algorithms/inputs.py; autoflow/algorithms/dicom_conversion.py; autoflow/algorithms/phase_correction.py` |
| `background_phase_correction.dual_venc_ratio1` | float | `0.0` | `configs/loader.json` | Dual-VENC decision threshold used in the 2*LV alias-shift stage. Zero derives the threshold from high/low VENC. | `autoflow/algorithms/data/h5_loader.py; autoflow/algorithms/inputs.py; autoflow/algorithms/dicom_conversion.py; autoflow/algorithms/phase_correction.py` |
| `background_phase_correction.dual_venc_ratio2` | float | `0.0` | `configs/loader.json` | Dual-VENC decision threshold used in the 4*LV alias-shift stage. Zero derives the threshold from high/low VENC. | `autoflow/algorithms/data/h5_loader.py; autoflow/algorithms/inputs.py; autoflow/algorithms/dicom_conversion.py; autoflow/algorithms/phase_correction.py` |
| `background_phase_correction.force_recompute` | bool | `false` | `configs/loader.json` | Ignore reusable embedded correction and estimate a fresh field; needed for a cold algorithm benchmark. | `autoflow/algorithms/data/h5_loader.py; autoflow/algorithms/inputs.py; autoflow/algorithms/dicom_conversion.py; autoflow/algorithms/phase_correction.py` |
| `background_phase_correction.write_cache` | bool | `true` | `configs/loader.json` | Allow newly computed correction to be stored in the source H5. Set false for read-only measurements. | `autoflow/algorithms/data/h5_loader.py; autoflow/algorithms/inputs.py; autoflow/algorithms/dicom_conversion.py; autoflow/algorithms/phase_correction.py` |
| `dicom_backend` | string | `"dicom2h5"` | `configs/loader.json` | Compatibility setting: DICOM directories always convert through Dicom2H5. Legacy API/config native values migrate to dicom2h5. | `autoflow/algorithms/data/h5_loader.py; autoflow/algorithms/inputs.py; autoflow/algorithms/dicom_conversion.py; autoflow/algorithms/phase_correction.py` |
| `dicom_h5_dir` | string | `""` | `configs/loader.json` | Destination directory for preserved Dicom2H5 outputs. Empty uses output_dir/_dicom_h5. Existing files cause an error; reopen the saved H5 to reuse it. | `autoflow/algorithms/data/h5_loader.py; autoflow/algorithms/inputs.py; autoflow/algorithms/dicom_conversion.py; autoflow/algorithms/phase_correction.py` |
| `ignore_embedded_segmentation` | bool | `false` | `configs/loader.json` | Ignore source H5 segmentation while retaining image/velocity input; use to test a fresh segmentation path. | `autoflow/algorithms/data/h5_loader.py; autoflow/algorithms/inputs.py; autoflow/algorithms/dicom_conversion.py; autoflow/algorithms/phase_correction.py` |

## noise_removal

Code owner: `autoflow/algorithms/noise_removal.py`, `autoflow/core/pipeline.py`, `autoflow/ui/viewer.py`.

| Parameter | Type | Effective default | Where configured | Effect and units | Code owner |
| --- | --- | --- | --- | --- | --- |
| `enabled` | bool | `false` | `configs/noise_removal.json` | Run PC-MRA display-region masking; does not change magnitude, velocity or segmentation inputs. | `autoflow/algorithms/noise_removal.py`, `autoflow/processing.py` |
| `method` | string | `"magnitude_temporal"` | `configs/noise_removal.json` | Choose magnitude_temporal screening or magnitude-only screening. | `autoflow/algorithms/noise_removal.py`, `autoflow/processing.py` |
| `magnitude_fraction` | float | `0.05` | `configs/noise_removal.json` | Fraction of maximum temporal-mean magnitude; AutoFlow default 0.05. Lower values retain more voxels; zero keeps all positive finite magnitude. | `autoflow/algorithms/noise_removal.py`, `autoflow/processing.py` |
| `velocity_std_max` | float | `0.8` | `configs/noise_removal.json` | Upper temporal speed SD as a fraction of maximum SD over finite velocity voxels; AutoFlow default 0.80. Higher values retain more voxels; zero disables temporal screening. Not VENC-normalized. | `autoflow/algorithms/noise_removal.py`, `autoflow/processing.py` |

## phase_unwrapping

Code owner: `autoflow/algorithms/phase_unwrapping/engine.py`.

| Parameter | Type | Effective default | Where configured | Effect and units | Code owner |
| --- | --- | --- | --- | --- | --- |
| `mask_source` | string | `"auto"` | `configs/phase_unwrapping.json` | Mask source: auto selects none for gc3D/lap4D/NPRS and pcmra_std for PUDIP/GUST. Traditional methods allow none/segmask; learned methods also allow pcmra_std/pcmra_mean. segmask requires an active segmentation. | `autoflow/algorithms/phase_unwrapping/backends.py` |
| `device` | string | `"auto"` | `configs/phase_unwrapping.json` | Compute device selector: auto, cpu, or cuda. Backend-specific CPU work can remain even with cuda selected. | `autoflow/algorithms/phase_unwrapping/backends.py` |
| `tfc` | bool | `true` | `configs/phase_unwrapping.json` | Temporal flow consistency option of supported traditional methods; couples recovery across cardiac phases. | `autoflow/algorithms/phase_unwrapping/engine.py` |
| `lap4d_ts` | float | `2.0` | `configs/phase_unwrapping.json` | Temporal-axis scaling in the lap4D Laplacian operator; controls the balance of temporal and spatial constraints. | `autoflow/algorithms/phase_unwrapping/laplacian.py` |
| `nprs_upsampling_factor` | int | `2` | `configs/phase_unwrapping.json` | Spatial Fourier upsampling factor used only by NPRS; larger factors increase memory and resampling cost. | `autoflow/algorithms/phase_unwrapping/nprs.py` |
| `nprs_pi_unwrap` | bool | `true` | `configs/phase_unwrapping.json` | Enable the additional pi-unwrapping pass of NPRS. | `autoflow/algorithms/phase_unwrapping/nprs.py` |
| `nprs_auto_crop` | bool | `true` | `configs/phase_unwrapping.json` | Return NPRS output cropped to the original array extent after FFT padding. | `autoflow/algorithms/phase_unwrapping/nprs.py` |
| `backend_params` | object | `{}` | `configs/phase_unwrapping.json` | Backend-specific nested objects, e.g. pudip and gust; supported adapter keys are listed below rather than passed blindly. | `autoflow/algorithms/phase_unwrapping/_common.py` |
| `write_output` | bool | `true` | `configs/phase_unwrapping.json` | Write phase_unwrap.npz when phase recovery runs; has no effect when the method is disabled or DV input skips recovery. | `autoflow/processing.py` |

## skeleton

Code owner: `autoflow/algorithms/preprocess.py; autoflow/algorithms/skeleton.py`.

| Parameter | Type | Effective default | Where configured | Effect and units | Code owner |
| --- | --- | --- | --- | --- | --- |
| `remove_small_cc` | bool | `true` | `configs/skeleton.json` | Filter small connected components before skeletonization. | `autoflow/algorithms/preprocess.py; autoflow/algorithms/skeleton.py` |
| `special_handling` | string | `"three_pass_merge"` | `configs/skeleton.json` | Special systemic-branch strategy: three_pass_merge separates and merges branch skeleton passes; other supported modes are described in the skeleton guide. | `autoflow/algorithms/preprocess.py; autoflow/algorithms/skeleton.py` |
| `special_merge_radius_mm` | float | `1.5` | `configs/skeleton.json` | Maximum physical distance in mm for joining the special-handling skeleton passes. | `autoflow/algorithms/preprocess.py; autoflow/algorithms/skeleton.py` |
| `separate_special_label_contacts` | bool | `true` | `configs/skeleton.json` | Separate configured special labels where they touch before skeleton generation. | `autoflow/algorithms/preprocess.py; autoflow/algorithms/skeleton.py` |
| `special_contact_labels` | list | `["RBCT", "CCA", "LBCT"]` | `configs/skeleton.json` | Anatomical label names subject to the contact-separation rule; resolve IDs through labels.label_map. | `autoflow/algorithms/preprocess.py; autoflow/algorithms/skeleton.py` |
| `min_cc_volume_mm3` | float | `50.0` | `configs/skeleton.json` | Absolute connected-component volume threshold in cubic millimetres. | `autoflow/algorithms/preprocess.py; autoflow/algorithms/skeleton.py` |
| `cc_filter_mode` | string | `"hybrid"` | `configs/skeleton.json` | Component retention rule: absolute, relative or hybrid combines physical volume and a fraction of the largest component. | `autoflow/algorithms/preprocess.py; autoflow/algorithms/skeleton.py` |
| `cc_rel_min_ratio` | float | `0.01` | `configs/skeleton.json` | Smallest retained component as a fraction of the largest component in relative/hybrid filtering. | `autoflow/algorithms/preprocess.py; autoflow/algorithms/skeleton.py` |
| `min_edge_points` | int | `3` | `configs/skeleton.json` | Minimum graph-edge point count retained during graph cleanup; controls short terminal-edge removal. | `autoflow/algorithms/preprocess.py; autoflow/algorithms/skeleton.py` |
| `do_closing` | bool | `true` | `configs/skeleton.json` | Legacy Boolean binary-closing switch used in default preprocessing. | `autoflow/algorithms/preprocess.py; autoflow/algorithms/skeleton.py` |
| `do_opening` | bool | `false` | `configs/skeleton.json` | Legacy Boolean binary-opening switch used in default preprocessing. | `autoflow/algorithms/preprocess.py; autoflow/algorithms/skeleton.py` |
| `gaussian_sigma` | float | `0.5` | `configs/skeleton.json` | Gaussian mask smoothing width in voxels before binary thresholding. | `autoflow/algorithms/preprocess.py; autoflow/algorithms/skeleton.py` |
| `gaussian_enabled` | bool | `true` | `configs/skeleton.json` | Enable Gaussian mask smoothing; per-group overrides in labels.label_groups can replace this value. | `autoflow/algorithms/preprocess.py; autoflow/algorithms/skeleton.py` |
| `dilation_iters` | int | `0` | `configs/skeleton.json` | Number of binary dilation iterations; expands support and can connect nearby vessels. Zero disables this operation. | `autoflow/algorithms/preprocess.py; autoflow/algorithms/skeleton.py` |
| `erosion_iters` | int | `0` | `configs/skeleton.json` | Number of binary erosion iterations; shrinks support and can remove narrow structures. | `autoflow/algorithms/preprocess.py; autoflow/algorithms/skeleton.py` |
| `opening_iters` | int | `0` | `configs/skeleton.json` | Number of explicit binary opening iterations; removes small protrusions/components. | `autoflow/algorithms/preprocess.py; autoflow/algorithms/skeleton.py` |
| `closing_iters` | int | `0` | `configs/skeleton.json` | Number of explicit binary closing iterations; fills small gaps and holes. | `autoflow/algorithms/preprocess.py; autoflow/algorithms/skeleton.py` |

## labels

Code owner: `autoflow/core/models.py; autoflow/core/pipeline.py`.

| Parameter | Type | Effective default | Where configured | Effect and units | Code owner |
| --- | --- | --- | --- | --- | --- |
| `single_label_group_name` | string | `"single_label"` | `configs/labels.json` | Group name for a binary/single-label input in the GUI browser and grouped analysis. | `autoflow/core/models.py; autoflow/core/pipeline.py` |
| `single_label_browser_color` | string | `"#d9480f"` | `configs/labels.json` | Display colour for the single-label group. | `autoflow/core/models.py; autoflow/core/pipeline.py` |
| `default_group_browser_color` | string | `"#1c7ed6"` | `configs/labels.json` | Fallback colour for a group without an explicit colour. | `autoflow/core/models.py; autoflow/core/pipeline.py` |
| `label_map` | object | `"See structured parameters for shipped entries"` | `configs/labels.json` | Dictionary from anatomical label names to integer IDs; background is zero. The configured mapping must match the segmentation/model. | `autoflow/core/models.py; autoflow/core/pipeline.py` |
| `label_groups` | object | `"See structured parameters for shipped entries"` | `configs/labels.json` | Named group definitions containing member labels, colours and optional preprocessing overrides; each group is processed independently. | `autoflow/core/models.py; autoflow/core/pipeline.py` |

## planes

Code owner: `autoflow/algorithms/planes.py`.

| Parameter | Type | Effective default | Where configured | Effect and units | Code owner |
| --- | --- | --- | --- | --- | --- |
| `plane_mode` | string | `"fixed_step"` | `configs/planes.json` | Placement rule: count, distance, anchored_offset, uniform or fixed_step. Controls which spacing/anchor parameters are active. | `autoflow/algorithms/planes.py` |
| `plane_count` | int | `3` | `configs/planes.json` | Number of planes in count/fixed-step placement; -1 requests all available valid locations where supported. | `autoflow/algorithms/planes.py` |
| `cross_section_distance` | float | `5.0` | `configs/planes.json` | Physical separation in mm used by distance/fixed-step layouts. | `autoflow/algorithms/planes.py` |
| `start_distance` | float | `0.0` | `configs/planes.json` | Millimetres trimmed from the start of each usable path before plane placement. | `autoflow/algorithms/planes.py` |
| `end_distance` | float | `0.0` | `configs/planes.json` | Millimetres trimmed from the end of each usable path before plane placement. | `autoflow/algorithms/planes.py` |
| `anchor` | string | `"center"` | `configs/planes.json` | Reference location: start, center, end or junction. Defines the origin for anchored placement. | `autoflow/algorithms/planes.py` |
| `anchor_offset_mm` | float | `5.0` | `configs/planes.json` | Physical offset in mm from the selected anchor toward the chosen direction. | `autoflow/algorithms/planes.py` |
| `direction` | string | `"both"` | `configs/planes.json` | Placement direction from the anchor: toward_start, toward_end or both. | `autoflow/algorithms/planes.py` |
| `spacing_mode` | string | `"fraction"` | `configs/planes.json` | Anchor-layout spacing interpretation: distance in mm or fraction of usable path length. | `autoflow/algorithms/planes.py` |
| `spacing_ratio` | float | `0.25` | `configs/planes.json` | Dimensionless usable-path-length fraction between anchored planes when spacing_mode=fraction. | `autoflow/algorithms/planes.py` |
| `segmentation_filter` | bool | `true` | `configs/planes.json` | Discard candidate planes whose centreline location lacks compatible segmentation support. | `autoflow/algorithms/planes.py` |
| `smoothing_window` | int | `15` | `configs/planes.json` | Savitzky-Golay window length for centreline smoothing; must admit the selected polynomial order. | `autoflow/algorithms/planes.py` |
| `smoothing_polyorder` | int | `2` | `configs/planes.json` | Polynomial degree of centreline Savitzky-Golay smoothing. | `autoflow/algorithms/planes.py` |
| `inter_time` | int | `10` | `configs/planes.json` | Centreline interpolation density multiplier; increases resampled path points before plane generation. | `autoflow/algorithms/planes.py` |
| `render.default.plane_color` | string | `"yellow"` | `configs/planes.json` | Plane display colour for the default or named anatomical group. | `autoflow/ui/viewer.py; autoflow/rendering/videos.py` |
| `render.default.plane_opacity` | float | `0.75` | `configs/planes.json` | Plane display opacity between zero and one for the default or named anatomical group. | `autoflow/ui/viewer.py; autoflow/rendering/videos.py` |
| `render.groups.aorta_systemic_branches.plane_color` | string | `"#ffd43b"` | `configs/planes.json` | Plane display colour for the default or named anatomical group. | `autoflow/ui/viewer.py; autoflow/rendering/videos.py` |
| `render.groups.aorta_systemic_branches.plane_opacity` | float | `0.75` | `configs/planes.json` | Plane display opacity between zero and one for the default or named anatomical group. | `autoflow/ui/viewer.py; autoflow/rendering/videos.py` |
| `render.groups.pulmonary_arteries.plane_color` | string | `"#a5d8ff"` | `configs/planes.json` | Plane display colour for the default or named anatomical group. | `autoflow/ui/viewer.py; autoflow/rendering/videos.py` |
| `render.groups.pulmonary_arteries.plane_opacity` | float | `0.75` | `configs/planes.json` | Plane display opacity between zero and one for the default or named anatomical group. | `autoflow/ui/viewer.py; autoflow/rendering/videos.py` |
| `render.groups.portal_splenic_venous.plane_color` | string | `"#b2f2bb"` | `configs/planes.json` | Plane display colour for the default or named anatomical group. | `autoflow/ui/viewer.py; autoflow/rendering/videos.py` |
| `render.groups.portal_splenic_venous.plane_opacity` | float | `0.75` | `configs/planes.json` | Plane display opacity between zero and one for the default or named anatomical group. | `autoflow/ui/viewer.py; autoflow/rendering/videos.py` |
| `render.groups.intracranial_anterior_arteries.plane_color` | string | `"#e599f7"` | `configs/planes.json` | Plane display colour for the default or named anatomical group. | `autoflow/ui/viewer.py; autoflow/rendering/videos.py` |
| `render.groups.intracranial_anterior_arteries.plane_opacity` | float | `0.75` | `configs/planes.json` | Plane display opacity between zero and one for the default or named anatomical group. | `autoflow/ui/viewer.py; autoflow/rendering/videos.py` |
| `render.groups.vertebrobasilar_arteries.plane_color` | string | `"#d0bfff"` | `configs/planes.json` | Plane display colour for the default or named anatomical group. | `autoflow/ui/viewer.py; autoflow/rendering/videos.py` |
| `render.groups.vertebrobasilar_arteries.plane_opacity` | float | `0.75` | `configs/planes.json` | Plane display opacity between zero and one for the default or named anatomical group. | `autoflow/ui/viewer.py; autoflow/rendering/videos.py` |
| `render.groups.intracranial_veins.plane_color` | string | `"#99e9f2"` | `configs/planes.json` | Plane display colour for the default or named anatomical group. | `autoflow/ui/viewer.py; autoflow/rendering/videos.py` |
| `render.groups.intracranial_veins.plane_opacity` | float | `0.75` | `configs/planes.json` | Plane display opacity between zero and one for the default or named anatomical group. | `autoflow/ui/viewer.py; autoflow/rendering/videos.py` |
| `render.groups.jugular_veins.plane_color` | string | `"#dee2e6"` | `configs/planes.json` | Plane display colour for the default or named anatomical group. | `autoflow/ui/viewer.py; autoflow/rendering/videos.py` |
| `render.groups.jugular_veins.plane_opacity` | float | `0.75` | `configs/planes.json` | Plane display opacity between zero and one for the default or named anatomical group. | `autoflow/ui/viewer.py; autoflow/rendering/videos.py` |

## streamlines

Code owner: `autoflow/algorithms/streamlines.py`.

| Parameter | Type | Effective default | Where configured | Effect and units | Code owner |
| --- | --- | --- | --- | --- | --- |
| `seed_ratio` | float | `0.1` | `configs/streamlines.json` | Fraction of slice support locations sampled as streamline seeds; min_seeds sets a lower target when possible. | `autoflow/algorithms/streamlines.py` |
| `max_steps` | int | `200` | `configs/streamlines.json` | Maximum integration steps for each instantaneous streamline. | `autoflow/algorithms/streamlines.py` |
| `min_seeds` | int | `50` | `configs/streamlines.json` | Minimum desired seed count per valid plane, limited by available support. | `autoflow/algorithms/streamlines.py` |
| `terminal_speed` | float | `0.01` | `configs/streamlines.json` | Velocity threshold in m/s for terminating streamline integration (input cm/s is converted to m/s). | `autoflow/algorithms/streamlines.py` |
| `rng_seed` | int | `0` | `configs/streamlines.json` | Random seed for reproducible seed subsampling. | `autoflow/algorithms/streamlines.py` |
| `tube_radius` | float | `0.05` | `configs/streamlines.json` | Rendered tube radius in physical mm; affects display geometry, not integration. | `autoflow/algorithms/streamlines.py` |
| `render.clim` | optional / null | `null` | `configs/streamlines.json` | Displayed scalar limits [lower,upper] in the metric's units; null selects the automatic range. Changes colours only, never numerical results. | `autoflow/ui/viewer.py; autoflow/rendering/videos.py` |
| `render.show_scalar_bar` | bool | `true` | `configs/streamlines.json` | Show the metric scalar bar in the GUI/offline renderer. | `autoflow/ui/viewer.py; autoflow/rendering/videos.py` |
| `render.bar_cfg.position_x` | float | `0.75` | `configs/streamlines.json` | Horizontal scalar-bar origin as a viewport fraction between zero and one. | `autoflow/ui/viewer.py; autoflow/rendering/videos.py` |
| `render.bar_cfg.position_y` | float | `0.2` | `configs/streamlines.json` | Vertical scalar-bar origin as a viewport fraction between zero and one. | `autoflow/ui/viewer.py; autoflow/rendering/videos.py` |
| `render.bar_cfg.height` | float | `0.22` | `configs/streamlines.json` | Scalar-bar height as a viewport fraction. | `autoflow/ui/viewer.py; autoflow/rendering/videos.py` |
| `render.bar_cfg.width` | float | `0.05` | `configs/streamlines.json` | Scalar-bar width as a viewport fraction. | `autoflow/ui/viewer.py; autoflow/rendering/videos.py` |
| `render.bar_cfg.title_font_size` | int | `40` | `configs/streamlines.json` | Scalar-bar title font size in points. | `autoflow/ui/viewer.py; autoflow/rendering/videos.py` |
| `render.bar_cfg.label_font_size` | int | `32` | `configs/streamlines.json` | Scalar-bar tick-label font size in points. | `autoflow/ui/viewer.py; autoflow/rendering/videos.py` |

## pathlines

Code owner: `autoflow/algorithms/streamlines.py; autoflow/ui/app.py`.

| Parameter | Type | Effective default | Where configured | Effect and units | Code owner |
| --- | --- | --- | --- | --- | --- |
| `seed_ratio` | float | `0.2` | `configs/pathlines.json` | Support fraction used only by ratio seeding. | `autoflow/algorithms/streamlines.py; autoflow/ui/app.py` |
| `max_steps` | int | `200` | `configs/pathlines.json` | Maximum temporal integration steps per pathline. | `autoflow/algorithms/streamlines.py; autoflow/ui/app.py` |
| `min_seeds` | int | `50` | `configs/pathlines.json` | Minimum desired seed count in ratio mode. | `autoflow/algorithms/streamlines.py; autoflow/ui/app.py` |
| `seed_mode` | string | `"fixed"` | `configs/pathlines.json` | fixed uses seed_count; ratio uses seed_ratio and min_seeds. | `autoflow/algorithms/streamlines.py; autoflow/ui/app.py` |
| `seed_count` | int | `250` | `configs/pathlines.json` | Maximum desired seeds per plane in fixed mode; support limits the actual count. | `autoflow/algorithms/streamlines.py; autoflow/ui/app.py` |
| `terminal_speed` | float | `0.01` | `configs/pathlines.json` | Velocity threshold in m/s for terminating temporal integration; the VTK temporal source uses mm/s and converts this threshold. | `autoflow/algorithms/streamlines.py; autoflow/ui/app.py` |
| `rng_seed` | int | `0` | `configs/pathlines.json` | Random seed for reproducible pathline seed placement. | `autoflow/algorithms/streamlines.py; autoflow/ui/app.py` |
| `tube_radius` | float | `0.25` | `configs/pathlines.json` | Rendered pathline tube radius in mm. | `autoflow/algorithms/streamlines.py; autoflow/ui/app.py` |
| `color` | string | `"deepskyblue"` | `configs/pathlines.json` | Uniform colour used when color_mode=uniform. | `autoflow/algorithms/streamlines.py; autoflow/ui/app.py` |
| `color_mode` | string | `"per_plane"` | `configs/pathlines.json` | uniform, per_plane or per_group colour assignment; affects display only. | `autoflow/algorithms/streamlines.py; autoflow/ui/app.py` |
| `temporal_cache_mb` | float | `512.0` | `configs/pathlines.json` | Memory limit in MiB for prepared temporal velocity grids; larger limits can accelerate repeated GUI phase changes. | `autoflow/algorithms/streamlines.py; autoflow/ui/app.py` |

## derived

Code owner: `autoflow/config.py`.

No active fields are shipped. This object is retained for legacy compatibility; configure the named derived modules instead.

## fluid

Code owner: `autoflow/algorithms/metrics/wss.py; autoflow/algorithms/metrics/pressure.py; autoflow/algorithms/metrics/tke.py`.

| Parameter | Type | Effective default | Where configured | Effect and units | Code owner |
| --- | --- | --- | --- | --- | --- |
| `rho` | float | `1060.0` | `configs/fluid.json` | Blood density in kg/m^3, shared by pressure and TKE unless their module overrides it. | `autoflow/algorithms/metrics/wss.py; autoflow/algorithms/metrics/pressure.py; autoflow/algorithms/metrics/tke.py` |
| `viscosity` | float | `4.0` | `configs/fluid.json` | Dynamic viscosity in mPa*s, shared by WSS and pressure unless their module overrides it. | `autoflow/algorithms/metrics/wss.py; autoflow/algorithms/metrics/pressure.py; autoflow/algorithms/metrics/tke.py` |

## wss

Code owner: `autoflow/algorithms/metrics/wss.py`.

| Parameter | Type | Effective default | Where configured | Effect and units | Code owner |
| --- | --- | --- | --- | --- | --- |
| `smoothing_iteration` | int | `200` | `configs/wss.json` | Taubin surface-smoothing iterations before normal sampling; zero retains the extracted voxel surface. | `autoflow/algorithms/metrics/wss.py` |
| `inward_distance` | string | `"auto"` | `configs/wss.json` | Wall-normal sampling step h in mm; auto uses the minimum voxel spacing. Parabolic fitting also probes 2h. | `autoflow/algorithms/metrics/wss.py` |
| `parabolic_fitting` | bool | `true` | `configs/wss.json` | True computes the exact wall derivative of a three-point quadratic vector fit; false uses the first-segment linear slope. | `autoflow/algorithms/metrics/wss.py` |
| `no_slip_condition` | bool | `true` | `configs/wss.json` | Set the wall velocity vector to zero by default. False samples the measured wall vector and increases sensitivity to partial volume and surface placement. | `autoflow/algorithms/metrics/wss.py` |
| `render.clim` | list | `[0.0, 5.0]` | `configs/wss.json` | Displayed scalar limits [lower,upper] in the metric's units; null selects the automatic range. Changes colours only, never numerical results. | `autoflow/ui/viewer.py; autoflow/rendering/videos.py` |
| `render.show_scalar_bar` | bool | `true` | `configs/wss.json` | Show the metric scalar bar in the GUI/offline renderer. | `autoflow/ui/viewer.py; autoflow/rendering/videos.py` |
| `render.bar_cfg.position_x` | float | `0.75` | `configs/wss.json` | Horizontal scalar-bar origin as a viewport fraction between zero and one. | `autoflow/ui/viewer.py; autoflow/rendering/videos.py` |
| `render.bar_cfg.position_y` | float | `0.2` | `configs/wss.json` | Vertical scalar-bar origin as a viewport fraction between zero and one. | `autoflow/ui/viewer.py; autoflow/rendering/videos.py` |
| `render.bar_cfg.height` | float | `0.22` | `configs/wss.json` | Scalar-bar height as a viewport fraction. | `autoflow/ui/viewer.py; autoflow/rendering/videos.py` |
| `render.bar_cfg.width` | float | `0.05` | `configs/wss.json` | Scalar-bar width as a viewport fraction. | `autoflow/ui/viewer.py; autoflow/rendering/videos.py` |
| `render.bar_cfg.title_font_size` | int | `40` | `configs/wss.json` | Scalar-bar title font size in points. | `autoflow/ui/viewer.py; autoflow/rendering/videos.py` |
| `render.bar_cfg.label_font_size` | int | `32` | `configs/wss.json` | Scalar-bar tick-label font size in points. | `autoflow/ui/viewer.py; autoflow/rendering/videos.py` |

## tke

Code owner: `autoflow/algorithms/data/h5_loader.py; autoflow/algorithms/metrics/tke.py`.

| Parameter | Type | Effective default | Where configured | Effect and units | Code owner |
| --- | --- | --- | --- | --- | --- |
| `render.clim` | list | `[0.0, 100.0]` | `configs/tke.json` | Displayed scalar limits [lower,upper] in the metric's units; null selects the automatic range. Changes colours only, never numerical results. | `autoflow/ui/viewer.py; autoflow/rendering/videos.py` |
| `render.show_scalar_bar` | bool | `true` | `configs/tke.json` | Show the metric scalar bar in the GUI/offline renderer. | `autoflow/ui/viewer.py; autoflow/rendering/videos.py` |
| `render.bar_cfg.position_x` | float | `0.75` | `configs/tke.json` | Horizontal scalar-bar origin as a viewport fraction between zero and one. | `autoflow/ui/viewer.py; autoflow/rendering/videos.py` |
| `render.bar_cfg.position_y` | float | `0.2` | `configs/tke.json` | Vertical scalar-bar origin as a viewport fraction between zero and one. | `autoflow/ui/viewer.py; autoflow/rendering/videos.py` |
| `render.bar_cfg.height` | float | `0.22` | `configs/tke.json` | Scalar-bar height as a viewport fraction. | `autoflow/ui/viewer.py; autoflow/rendering/videos.py` |
| `render.bar_cfg.width` | float | `0.05` | `configs/tke.json` | Scalar-bar width as a viewport fraction. | `autoflow/ui/viewer.py; autoflow/rendering/videos.py` |
| `render.bar_cfg.title_font_size` | int | `40` | `configs/tke.json` | Scalar-bar title font size in points. | `autoflow/ui/viewer.py; autoflow/rendering/videos.py` |
| `render.bar_cfg.label_font_size` | int | `32` | `configs/tke.json` | Scalar-bar tick-label font size in points. | `autoflow/ui/viewer.py; autoflow/rendering/videos.py` |

## pressure_gradient

Code owner: `autoflow/algorithms/metrics/pressure.py`.

| Parameter | Type | Effective default | Where configured | Effect and units | Code owner |
| --- | --- | --- | --- | --- | --- |
| `method` | string | `"ppe"` | `configs/pressure_gradient.json` | `ppe` is the merged Cartesian pressure solver; `ste` is the staggered-grid Stokes estimator. Legacy `least_squares` is accepted as an alias for `ppe`. | `autoflow/algorithms/metrics/pressure.py` |
| `smoothing_sigma` | float | `0.0` | `configs/pressure_gradient.json` | Spatial Gaussian smoothing width in voxels. Zero disables smoothing; valid-mask normalization prevents boundary attenuation. | `autoflow/algorithms/metrics/pressure.py` |
| `support_erosion_iters` | int | `1` | `configs/pressure_gradient.json` | Additional conservative 26-neighbour erosion margin. Complete measured spatial and temporal derivative stencils are mandatory even at zero. | `autoflow/algorithms/metrics/pressure.py` |
| `layer_opacity` | float | `0.6` | `configs/pressure_gradient.json` | PG layer opacity between zero (transparent) and one (opaque), affecting display only. | `autoflow/algorithms/metrics/pressure.py` |
| `relative_pressure_opacity` | float | `0.6` | `configs/pressure_gradient.json` | Relative-pressure layer opacity between zero and one. | `autoflow/algorithms/metrics/pressure.py` |
| `use_convective_acceleration` | bool | `true` | `configs/pressure_gradient.json` | Include (v dot grad)v in Navier-Stokes PG. Disabling omits real convective effects and is an explicit approximation. | `autoflow/algorithms/metrics/pressure.py` |
| `render.clim` | optional / null | `null` | `configs/pressure_gradient.json` | Displayed scalar limits [lower,upper] in the metric's units; null selects the automatic range. Changes colours only, never numerical results. | `autoflow/ui/viewer.py; autoflow/rendering/videos.py` |
| `render.show_scalar_bar` | bool | `true` | `configs/pressure_gradient.json` | Show the metric scalar bar in the GUI/offline renderer. | `autoflow/ui/viewer.py; autoflow/rendering/videos.py` |
| `render.bar_cfg.position_x` | float | `0.75` | `configs/pressure_gradient.json` | Horizontal scalar-bar origin as a viewport fraction between zero and one. | `autoflow/ui/viewer.py; autoflow/rendering/videos.py` |
| `render.bar_cfg.position_y` | float | `0.2` | `configs/pressure_gradient.json` | Vertical scalar-bar origin as a viewport fraction between zero and one. | `autoflow/ui/viewer.py; autoflow/rendering/videos.py` |
| `render.bar_cfg.height` | float | `0.22` | `configs/pressure_gradient.json` | Scalar-bar height as a viewport fraction. | `autoflow/ui/viewer.py; autoflow/rendering/videos.py` |
| `render.bar_cfg.width` | float | `0.05` | `configs/pressure_gradient.json` | Scalar-bar width as a viewport fraction. | `autoflow/ui/viewer.py; autoflow/rendering/videos.py` |
| `render.bar_cfg.title_font_size` | int | `40` | `configs/pressure_gradient.json` | Scalar-bar title font size in points. | `autoflow/ui/viewer.py; autoflow/rendering/videos.py` |
| `render.bar_cfg.label_font_size` | int | `32` | `configs/pressure_gradient.json` | Scalar-bar tick-label font size in points. | `autoflow/ui/viewer.py; autoflow/rendering/videos.py` |
| `render.relative_pressure_clim` | optional / null | `null` | `configs/pressure_gradient.json` | Displayed relative-pressure range [lower,upper] in Pa; null uses an automatic symmetric range. | `autoflow/ui/viewer.py; autoflow/rendering/videos.py` |
| `render.relative_pressure_show_scalar_bar` | bool | `true` | `configs/pressure_gradient.json` | Show the relative-pressure scalar bar. | `autoflow/ui/viewer.py; autoflow/rendering/videos.py` |
| `render.relative_pressure_bar_cfg.position_x` | float | `0.75` | `configs/pressure_gradient.json` | Horizontal scalar-bar origin as a viewport fraction between zero and one. | `autoflow/ui/viewer.py; autoflow/rendering/videos.py` |
| `render.relative_pressure_bar_cfg.position_y` | float | `0.2` | `configs/pressure_gradient.json` | Vertical scalar-bar origin as a viewport fraction between zero and one. | `autoflow/ui/viewer.py; autoflow/rendering/videos.py` |
| `render.relative_pressure_bar_cfg.height` | float | `0.22` | `configs/pressure_gradient.json` | Scalar-bar height as a viewport fraction. | `autoflow/ui/viewer.py; autoflow/rendering/videos.py` |
| `render.relative_pressure_bar_cfg.width` | float | `0.05` | `configs/pressure_gradient.json` | Scalar-bar width as a viewport fraction. | `autoflow/ui/viewer.py; autoflow/rendering/videos.py` |
| `render.relative_pressure_bar_cfg.title_font_size` | int | `40` | `configs/pressure_gradient.json` | Scalar-bar title font size in points. | `autoflow/ui/viewer.py; autoflow/rendering/videos.py` |
| `render.relative_pressure_bar_cfg.label_font_size` | int | `32` | `configs/pressure_gradient.json` | Scalar-bar tick-label font size in points. | `autoflow/ui/viewer.py; autoflow/rendering/videos.py` |

## vortex

Code owner: `autoflow/algorithms/metrics/vortex.py`.

| Parameter | Type | Effective default | Where configured | Effect and units | Code owner |
| --- | --- | --- | --- | --- | --- |
| `smoothing_sigma` | float | `0.0` | `configs/vortex.json` | Spatial Gaussian velocity smoothing width in voxels before vortex derivatives; zero disables smoothing. | `autoflow/algorithms/metrics/vortex.py` |
| `support_erosion_iters` | int | `1` | `configs/vortex.json` | Lumen-mask erosion iterations defining valid vortex derivative support. Zero disables the optional erosion. | `autoflow/algorithms/metrics/vortex.py` |

## pwv

Code owner: `autoflow/algorithms/pwv.py`.

| Parameter | Type | Effective default | Where configured | Effect and units | Code owner |
| --- | --- | --- | --- | --- | --- |
| `enabled` | bool | `true` | `configs/pwv.json` | Enable the configured PWV workflow; CLI/API must still request pwv as an optional metric. | `autoflow/algorithms/pwv.py` |
| `groups` | list | `[{"name": "Portal Vein", "labels": ["PV"]}]` | `configs/pwv.json` | List of PWV group objects: name identifies the result; labels selects the anatomical members. | `autoflow/algorithms/pwv.py` |
| `plane_interval_mm` | float | `5.0` | `configs/pwv.json` | Physical separation in mm between PWV sampling planes. | `autoflow/algorithms/pwv.py` |
| `start_distance` | float | `0.0` | `configs/pwv.json` | Millimetres trimmed from the path start for PWV plane placement. | `autoflow/algorithms/pwv.py` |
| `end_distance` | float | `0.0` | `configs/pwv.json` | Millimetres trimmed from the path end for PWV plane placement. | `autoflow/algorithms/pwv.py` |
| `smoothing_window` | int | `15` | `configs/pwv.json` | Savitzky-Golay centreline-smoothing window length for PWV plane generation. | `autoflow/algorithms/pwv.py` |
| `smoothing_polyorder` | int | `2` | `configs/pwv.json` | Polynomial degree for PWV centreline smoothing. | `autoflow/algorithms/pwv.py` |
| `inter_time` | int | `10` | `configs/pwv.json` | Centreline interpolation density for PWV distance sampling. | `autoflow/algorithms/pwv.py` |
| `waveform_key` | string | `"flowrate_mL_s"` | `configs/pwv.json` | Time-resolved plane metric used as the propagation waveform, e.g. flowrate_mL_s. | `autoflow/algorithms/pwv.py` |
| `transit_time_method` | string | `"foot_to_foot"` | `configs/pwv.json` | Delay estimator: foot_to_foot uses waveform feet; supported cross-correlation mode aligns complete waveforms. | `autoflow/algorithms/pwv.py` |
| `foot_method` | string | `"tangent"` | `configs/pwv.json` | Foot detector, such as tangent or threshold; applies to foot_to_foot delay estimation. | `autoflow/algorithms/pwv.py` |
| `foot_savgol_window` | int | `5` | `configs/pwv.json` | Odd smoothing window in cardiac samples used before waveform-foot detection. | `autoflow/algorithms/pwv.py` |
| `foot_savgol_polyorder` | int | `2` | `configs/pwv.json` | Polynomial degree of foot-detection waveform smoothing. | `autoflow/algorithms/pwv.py` |
| `foot_threshold_percent` | float | `10.0` | `configs/pwv.json` | Percentage of the waveform upstroke range used by threshold foot detection. | `autoflow/algorithms/pwv.py` |
| `xcorr_window` | string | `"full"` | `configs/pwv.json` | Waveform window used for cross-correlation; full uses the complete sampled cycle. | `autoflow/algorithms/pwv.py` |
| `xcorr_interp_factor` | int | `10` | `configs/pwv.json` | Sub-frame temporal interpolation multiplier for cross-correlation lag resolution. | `autoflow/algorithms/pwv.py` |
| `allow_cycle_wrap` | bool | `true` | `configs/pwv.json` | Allow propagation delays across the end/start boundary of the cardiac cycle. | `autoflow/algorithms/pwv.py` |
| `minimum_valid_planes` | int | `2` | `configs/pwv.json` | Minimum successfully sampled planes needed to fit distance against arrival time. | `autoflow/algorithms/pwv.py` |
| `scene_visible` | bool | `true` | `configs/pwv.json` | Initial GUI visibility of the generated PWV planes. | `autoflow/algorithms/pwv.py` |
| `scene_color` | string | `"#ffd43b"` | `configs/pwv.json` | Display colour of PWV planes. | `autoflow/algorithms/pwv.py` |
| `plot_color` | string | `"#2b8a3e"` | `configs/pwv.json` | Colour of measured arrival-time/distance points. | `autoflow/algorithms/pwv.py` |
| `fit_color` | string | `"#f08c00"` | `configs/pwv.json` | Colour of the fitted PWV regression line. | `autoflow/algorithms/pwv.py` |
| `plot_dpi` | int | `160` | `configs/pwv.json` | Resolution in dots per inch of exported PWV plots. | `autoflow/algorithms/pwv.py` |

## segmentation

Code owner: `autoflow/algorithms/segmentation/nnunet_static.py; autoflow/ui/segmentation.py`.

| Parameter | Type | Effective default | Where configured | Effect and units | Code owner |
| --- | --- | --- | --- | --- | --- |
| `visible` | bool | `true` | `configs/segmentation.json` | Initial visibility of segmentation surfaces in the 3D scene. | `autoflow/algorithms/segmentation/nnunet_static.py; autoflow/ui/segmentation.py` |
| `opacity` | float | `0.35` | `configs/segmentation.json` | Segmentation surface opacity between zero and one. | `autoflow/algorithms/segmentation/nnunet_static.py; autoflow/ui/segmentation.py` |
| `active_label` | int | `1` | `configs/segmentation.json` | Integer label painted or modified by the editor. | `autoflow/algorithms/segmentation/nnunet_static.py; autoflow/ui/segmentation.py` |
| `editing_enabled` | bool | `false` | `configs/segmentation.json` | Initial enable state of segmentation editing; enables correction controls. | `autoflow/algorithms/segmentation/nnunet_static.py; autoflow/ui/segmentation.py` |
| `tool` | string | `"brush"` | `configs/segmentation.json` | Editor tool selector, e.g. brush; controls how mouse edits modify the active mask. | `autoflow/algorithms/segmentation/nnunet_static.py; autoflow/ui/segmentation.py` |
| `brush_radius` | int | `3` | `configs/segmentation.json` | Brush radius in voxel/pixel units of the editing slice. | `autoflow/algorithms/segmentation/nnunet_static.py; autoflow/ui/segmentation.py` |
| `edit_all_timepoints` | bool | `true` | `configs/segmentation.json` | Apply a mask edit across all cardiac phases instead of the current phase only. | `autoflow/algorithms/segmentation/nnunet_static.py; autoflow/ui/segmentation.py` |
| `cleanup_4d_components` | bool | `false` | `configs/segmentation.json` | Enable connected-component cleanup of the 4D segmentation. | `autoflow/algorithms/segmentation/nnunet_static.py; autoflow/ui/segmentation.py` |
| `cleanup_4d_mode` | string | `"absolute"` | `configs/segmentation.json` | Component cleanup policy; absolute uses a physical-volume threshold. | `autoflow/algorithms/segmentation/nnunet_static.py; autoflow/ui/segmentation.py` |
| `cleanup_4d_min_volume_mm3` | float | `50.0` | `configs/segmentation.json` | Minimum physical component volume in mm^3 for configured 4D cleanup. | `autoflow/algorithms/segmentation/nnunet_static.py; autoflow/ui/segmentation.py` |
| `mode` | string | `"input"` | `configs/segmentation.json` | Segmentation workflow mode: use input, import an external mask, threshold scalar data, or run the automatic backend. | `autoflow/algorithms/segmentation/nnunet_static.py; autoflow/ui/segmentation.py` |
| `input_source` | string | `"original"` | `configs/segmentation.json` | Choose which already loaded segmentation source becomes active; original refers to embedded input segmentation. | `autoflow/algorithms/segmentation/nnunet_static.py; autoflow/ui/segmentation.py` |
| `import_path` | string | `""` | `configs/segmentation.json` | External H5/NPY/NPZ/NIfTI segmentation path; empty leaves import unconfigured. | `autoflow/algorithms/segmentation/nnunet_static.py; autoflow/ui/segmentation.py` |
| `threshold_scalar` | string | `"pcmra"` | `configs/segmentation.json` | Scalar field thresholded to create a mask, such as pcmra, mag or velocity magnitude. | `autoflow/algorithms/segmentation/nnunet_static.py; autoflow/ui/segmentation.py` |
| `threshold_value.mode` | string | `"manual"` | `configs/segmentation.json` | manual uses explicit scalar-range percentages; supported automatic modes choose thresholds from the image. | `autoflow/algorithms/segmentation/nnunet_static.py; autoflow/ui/segmentation.py` |
| `threshold_value.min_percent` | float | `10.0` | `configs/segmentation.json` | Lower cutoff as a percentage of the selected scalar range. | `autoflow/algorithms/segmentation/nnunet_static.py; autoflow/ui/segmentation.py` |
| `threshold_value.max_percent` | float | `100.0` | `configs/segmentation.json` | Upper cutoff as a percentage of the selected scalar range. | `autoflow/algorithms/segmentation/nnunet_static.py; autoflow/ui/segmentation.py` |
| `threshold_keep_largest_cc` | bool | `true` | `configs/segmentation.json` | Keep only the largest threshold-derived connected component when enabled. | `autoflow/algorithms/segmentation/nnunet_static.py; autoflow/ui/segmentation.py` |
| `threshold_min_component_volume_mm3` | float | `0.0` | `configs/segmentation.json` | Remove threshold-derived components below this physical volume in mm^3. | `autoflow/algorithms/segmentation/nnunet_static.py; autoflow/ui/segmentation.py` |
| `threshold_closing` | bool | `true` | `configs/segmentation.json` | Apply binary closing to the threshold-derived mask. | `autoflow/algorithms/segmentation/nnunet_static.py; autoflow/ui/segmentation.py` |
| `threshold_opening` | bool | `false` | `configs/segmentation.json` | Apply binary opening to the threshold-derived mask. | `autoflow/algorithms/segmentation/nnunet_static.py; autoflow/ui/segmentation.py` |
| `auto_backend` | string | `"nnUNet4D"` | `configs/segmentation.json` | Automatic backend: nnUNet4D predicts phase-resolved temporal masks; nnUNet predicts a static mask copied over phases. | `autoflow/algorithms/segmentation/nnunet_static.py; autoflow/ui/segmentation.py` |
| `auto_model` | string | `"auto"` | `configs/segmentation.json` | Model directory or auto profile. Auto resolves the backend-specific Dataset7020 temporal or Dataset7010 static model. | `autoflow/algorithms/segmentation/nnunet_static.py; autoflow/ui/segmentation.py` |
| `auto_checkpoint` | string | `"auto"` | `configs/segmentation.json` | Checkpoint basename or auto. Auto chooses the profile's checkpoint; does not download a model. | `autoflow/algorithms/segmentation/nnunet_static.py; autoflow/ui/segmentation.py` |
| `auto_folds` | string | `"single"` | `configs/segmentation.json` | single selects fold_all or the first available fold; all selects all numeric folds when available; a comma list selects explicit folds. | `autoflow/algorithms/segmentation/nnunet_static.py; autoflow/ui/segmentation.py` |
| `auto_device` | string | `"auto"` | `configs/segmentation.json` | Inference device: auto selects CUDA when available, otherwise CPU. | `autoflow/algorithms/segmentation/nnunet_static.py; autoflow/ui/segmentation.py` |
| `auto_label_map` | string | `""` | `configs/segmentation.json` | Optional file/string label mapping used to interpret automatic model output. | `autoflow/algorithms/segmentation/nnunet_static.py; autoflow/ui/segmentation.py` |
| `force_recompute_auto_cache` | bool | `false` | `configs/segmentation.json` | Ignore previous automatic-segmentation artifacts and produce a new prediction. | `autoflow/algorithms/segmentation/nnunet_static.py; autoflow/ui/segmentation.py` |
| `write_auto_cache` | bool | `true` | `configs/segmentation.json` | Permit segmentation-cache writes into supported source H5 files. Set false for read-only benchmarking. | `autoflow/algorithms/segmentation/nnunet_static.py; autoflow/ui/segmentation.py` |

## colorbar

Code owner: `autoflow/ui/viewer.py; autoflow/rendering/videos.py`.

| Parameter | Type | Effective default | Where configured | Effect and units | Code owner |
| --- | --- | --- | --- | --- | --- |
| `show` | bool | `true` | `configs/colorbar.json` | Default shared GUI scalar-bar visibility; metric visibility still follows its active layer. | `autoflow/ui/viewer.py; autoflow/rendering/videos.py` |
| `bar_cfg.position_x` | float | `0.87` | `configs/colorbar.json` | Horizontal scalar-bar origin as a viewport fraction between zero and one. | `autoflow/ui/viewer.py; autoflow/rendering/videos.py` |
| `bar_cfg.position_y` | float | `0.15` | `configs/colorbar.json` | Vertical scalar-bar origin as a viewport fraction between zero and one. | `autoflow/ui/viewer.py; autoflow/rendering/videos.py` |
| `bar_cfg.height` | float | `0.65` | `configs/colorbar.json` | Scalar-bar height as a viewport fraction. | `autoflow/ui/viewer.py; autoflow/rendering/videos.py` |
| `bar_cfg.width` | float | `0.08` | `configs/colorbar.json` | Scalar-bar width as a viewport fraction. | `autoflow/ui/viewer.py; autoflow/rendering/videos.py` |
| `bar_cfg.title_font_size` | int | `14` | `configs/colorbar.json` | Scalar-bar title font size in points. | `autoflow/ui/viewer.py; autoflow/rendering/videos.py` |
| `bar_cfg.label_font_size` | int | `11` | `configs/colorbar.json` | Scalar-bar tick-label font size in points. | `autoflow/ui/viewer.py; autoflow/rendering/videos.py` |

## video_exporting

Code owner: `autoflow/rendering/videos.py`.

| Parameter | Type | Effective default | Where configured | Effect and units | Code owner |
| --- | --- | --- | --- | --- | --- |
| `fps` | int | `12` | `configs/video_exporting.json` | Output video frame rate in frames per second; sets playback speed independently of acquired RR. | `autoflow/rendering/videos.py` |
| `plane_rotation_frames` | int | `180` | `configs/video_exporting.json` | Number of camera-rotation frames in the plane video. | `autoflow/rendering/videos.py` |
| `make_plane_video` | bool | `false` | `configs/video_exporting.json` | Request a plane video without requiring a separate requested_videos entry. | `autoflow/rendering/videos.py` |
| `make_wss_video` | bool | `false` | `configs/video_exporting.json` | Request WSS computation/rendering and a WSS video. | `autoflow/rendering/videos.py` |
| `make_pressure_gradient_video` | bool | `false` | `configs/video_exporting.json` | Request pressure computation and PG/relative-pressure videos. | `autoflow/rendering/videos.py` |
| `make_streamlines_video` | bool | `false` | `configs/video_exporting.json` | Request the streamline video. | `autoflow/rendering/videos.py` |
| `make_tke_video` | bool | `false` | `configs/video_exporting.json` | Request the TKE video when supported input TKE exists. | `autoflow/rendering/videos.py` |
| `camera_view` | string | `"right"` | `configs/video_exporting.json` | Initial camera direction, such as right; controls exported viewing orientation. | `autoflow/rendering/videos.py` |
| `camera_distance_scale` | float | `1.5` | `configs/video_exporting.json` | Multiplier of scene-based camera distance; larger values make the vessel appear smaller. | `autoflow/rendering/videos.py` |
| `rotate_dynamic_video` | bool | `true` | `configs/video_exporting.json` | Rotate the camera during dynamic metric videos; false retains a fixed view. | `autoflow/rendering/videos.py` |
| `dynamic_rotation_frames` | int | `180` | `configs/video_exporting.json` | Frames for one dynamic camera rotation cycle. | `autoflow/rendering/videos.py` |
| `dynamic_rotation_elevation_deg` | float | `10.0` | `configs/video_exporting.json` | Elevation in degrees for the dynamic camera orbit; null uses the renderer's standard setting. | `autoflow/rendering/videos.py` |
| `dynamic_time_repeat` | int | `3` | `configs/video_exporting.json` | Number of cardiac playback repeats within the dynamic export. | `autoflow/rendering/videos.py` |
| `add_plane_idx` | bool | `true` | `configs/video_exporting.json` | Annotate the plane index in plane-video labels. | `autoflow/rendering/videos.py` |
| `add_path_idx` | bool | `false` | `configs/video_exporting.json` | Annotate the owning path index in plane-video labels. | `autoflow/rendering/videos.py` |
| `window_size` | list | `[1600, 1200]` | `configs/video_exporting.json` | Render canvas [width,height] in pixels; larger dimensions increase rendering and encoding cost. | `autoflow/rendering/videos.py` |
| `plane_video.show_skeleton` | bool | `true` | `configs/video_exporting.json` | Show the skeleton in plane videos. | `autoflow/rendering/videos.py` |
| `plane_video.skeleton_point_size` | float | `5.0` | `configs/video_exporting.json` | Rendered skeleton-point size in pixels. | `autoflow/rendering/videos.py` |
| `plane_video.label.prefix` | string | `""` | `configs/video_exporting.json` | Text prepended to each plane-video label. | `autoflow/rendering/videos.py` |
| `plane_video.label.font_size` | int | `20` | `configs/video_exporting.json` | Plane-label font size in points. | `autoflow/rendering/videos.py` |
| `plane_video.label.text_color` | string | `"black"` | `configs/video_exporting.json` | Plane-label text colour. | `autoflow/rendering/videos.py` |
| `plane_video.label.shape_color` | string | `"yellow"` | `configs/video_exporting.json` | Background/outline colour of the plane-label shape. | `autoflow/rendering/videos.py` |
| `plane_video.label.shape_opacity` | float | `0.3` | `configs/video_exporting.json` | Opacity of the plane-label shape between zero and one. | `autoflow/rendering/videos.py` |
| `plane_video.default.skeleton_color` | string | `""` | `configs/video_exporting.json` | Skeleton-colour override for plane videos; empty uses the group/default scene colour. | `autoflow/rendering/videos.py` |
| `plane_video.default.plane_size` | int | `10` | `configs/video_exporting.json` | Physical side length in mm of the displayed plane square. | `autoflow/rendering/videos.py` |
| `plane_video.default.plane_color` | string | `"yellow"` | `configs/video_exporting.json` | Plane-video default colour; group settings can override it. | `autoflow/rendering/videos.py` |
| `plane_video.default.plane_opacity` | float | `0.75` | `configs/video_exporting.json` | Default plane-video opacity between zero and one. | `autoflow/rendering/videos.py` |
| `plane_video.groups` | object | `{}` | `configs/video_exporting.json` | Per-group plane-video overrides; each group may supply skeleton_color, plane_size, plane_color and plane_opacity. | `autoflow/rendering/videos.py` |
