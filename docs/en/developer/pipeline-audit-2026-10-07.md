# End-to-end pipeline and interaction audit — 2026-10-07

## Scope and evidence

This audit reviews the current H5/DICOM entry points, correction, unwrapping, PC-MRA, segmentation and review, skeletons, graphs, planes, every optional metric family, 2D/3D interaction, videos, persistence, cancellation and batch failure handling. It adds manual audit tooling and documentation, implements offline video caption/phase-scheduling improvements, and applies the non-model follow-up fixes described below.

Five real H5 samples are selected from the user-specified `h5s_autoflow` directory: one per Aorta, Carotid, Cerebrovascular, PortalVein and RenalArtery. Each category has 15 candidates. Selection is deterministic: median file size, filename as tie-breaker. Source names and full local paths remain in the local `results/audit_20261007_full/manifest.json`, not this document.

Cold means a fresh Python worker per case, recomputed WRLS+ARTO background correction, lap4D unwrapping, freshly generated nnUNet4D segmentation, and no correction/segmentation cache writes. Embedded segmentation is ignored. Operating-system/NAS caches are not flushed. Requested metrics are PWV, WSS, TKE, pressure/relative pressure and vortex, in addition to plane metrics. All six video outputs use the configured 1600 × 1200 resolution and 180 frames. The machine is shared and the configured checkpoint changed externally during the run (details below); these cold-baseline observations are not isolated-hardware or frozen-weight benchmarks. Separate paired video comparisons below measure the implemented caption optimization.

Evidence labels below distinguish **real-case measurement**, **synthetic reproduction**, **source-confirmed behavior**, and **unmeasured opportunity**. Real-case timings do not validate segmentation accuracy or every reported physiological value.

## Follow-up implementation (non-model scope)

The requested non-model fixes are implemented. All model-related changes were
withdrawn at the user's request: the existing 3D/static and 4D/temporal model
choices, checkpoint loading, labels and grouped execution behavior remain
unchanged. Model-performance findings later in this document are historical
audit observations and are deferred.

| Issue | Current behavior | Verification |
| --- | --- | --- |
| ROI history crosses cases or stale plane indices | History is bound to workspace and plane object sequence; case replacement, regeneration/import, deletion or reordering clears it | Qt smoke checks preserve same-case Undo and reject old-case/replaced-plane history |
| Basic/derived plane support differs | Basic flow/area and TKE/pressure share segmentation label, branch, connected-section and phase-ROI selection; WSS uses wall samples in the selected section | U-shaped production-branch phantom with dynamic labels, constant binary mask and a frame ROI matches areas and selected field values |
| Time-frame refresh misses image update | Geometry keys exclude cardiac time/cursor position; unchanged geometry updates image and overlays | Five real cases, 332 time transitions: **996 `update_image`, 0 `set_slice`**; selected-plane Qt smoke checks changed pixels |
| Correction decodes H5 twice | CLI/API requested correction runs during the first decode; default GUI loading and explicit source-based correction retain their workflows | Two real cases compare all 10 retained arrays/geometry fields exactly; source files unchanged |
| TKE is prepared per plane | Prepare once per multi-plane call and share with workers | Phantom counts one preparation for three planes with identical expected samples |
| CLI failures return success | Exit 0 only for a nonempty all-`ok` batch; errors/partial exports/empty input return 1 | Real invalid two-component input returns 1 with an `error` batch report; smoke covers partial video failure |
| Total time ends before outputs | Total includes positions, videos, QC, phase NPZ and first summary write; preserve `compute_time_sec` and report `output_time_sec` | Smoke verifies delayed report/video publication is included; successful and partial output cases persist |

The combined decoding/correction stage is named `load_and_background_correction`.
Fused single-VENC complex loading preserves sigma from the original encodes,
matching the former first decode. An initial replay exposed magnitude roundoff
from corrected complex encodes; that candidate was fixed before acceptance so
TKE inputs remain unchanged. Missing TKE stays optional. Failed available video
exports populate `video_errors` and produce a `partial` batch status.

### Loading comparison

| Case | Original two-pass load/correction (s) | Fused load/correction (s) | Numerical comparison |
| --- | ---: | ---: | --- |
| Aorta | 57.44 | 50.70 | Exact equality for flow, input flow, phase, magnitude, sigma, correction, segmentation, resolution, VENC and origin |
| Carotid | 52.05 | 33.21 | Exact equality for flow, input flow, phase, magnitude, sigma, correction, segmentation, resolution, VENC and origin |

These are one old-first/new-second pair per case on a shared host, using the
same fresh WRLS+ARTO settings and no source cache writes. They establish
numerical preservation and fewer decodes; they do not establish a universal
speedup or a new five-case end-to-end baseline. The two source size/mtime checks
are unchanged. Evidence: `loader_followup_comparison.json`; the rejected initial
sigma comparison remains separately recorded in `loader_followup_initial_comparison.json`.

### Current-checkout interaction verification

| Case | Earlier warm Speed p95 (ms) | Current warm Speed p95 (ms) | Current update calls |
| --- | ---: | ---: | --- |
| Aorta | 25.79 | 21.25 | 54 image updates / 0 geometry resets |
| Carotid | 99.00 | 123.19 | 39 image updates / 0 geometry resets |
| Cerebrovascular | 91.18 | 143.09 | 36 image updates / 0 geometry resets |
| PortalVein | 37.76 | 46.36 | 45 image updates / 0 geometry resets |
| RenalArtery | 44.04 | 43.67 | 75 image updates / 0 geometry resets |

The fixed invalidation path is verified, but the mixed latency changes are not
evidence of a general displayed-FPS improvement: these are CPU/offscreen
measurements under different shared-host load, with fresh correction work
running during the follow-up. Full scalar-volume work and image drawing still
contribute to latency. An initial follow-up probe imported installed code from
its results-directory script location; it was corrected to explicitly import
the current checkout. Only `ortho_followup_results.json`, which records the
actual module path, is accepted for the current-code results above.

The required existing suites pass: **195 passed** (`python -m pytest
 tests/test_smoke_phantoms.py tests/test_pressure_gradient_phantom.py -q`, Conda
`autoflow311`). No new automated test file was added. Evidence is in
`regression_non_model_final.log`, `ortho_followup_results.json`,
`cli_failure_followup/probe_result.json` and `loader_followup_comparison.json`.
Affected English feature/user pages describe the new contracts and timing
fields. Recompute earlier derived plane summaries when multiple vessels or
frame ROI edits can alter their selected support.

## Measurements

All times are seconds. These are the **original cold baseline**, captured before the video changes below. Python/config sources were frozen locally before implementation; the final cold worker explicitly imported that snapshot.

| Stage | Aorta | Carotid | Cerebrovascular | PortalVein | RenalArtery |
| --- | ---: | ---: | ---: | ---: | ---: |
| `load` | 3.39 | 11.73 | 13.82 | 5.06 | 4.39 |
| `background_correction` | 33.52 | 27.11 | 41.79 | 30.60 | 36.94 |
| `noise_removal` | 0.41 | 1.48 | 1.99 | 0.57 | 0.52 |
| `phase_unwrap` | 2.60 | 6.07 | 7.10 | 2.46 | 2.24 |
| `pcmra` | 0.41 | 1.31 | 1.43 | 0.55 | 0.57 |
| `autoseg` | 76.99 | 79.10 | 90.48 | 100.26 | 71.46 |
| `skeleton` | 0.45 | 1.12 | 0.96 | 0.97 | 0.35 |
| `graph` | 0.07 | 0.03 | 0.02 | 0.10 | 0.04 |
| `planes` | 0.55 | 0.33 | 0.26 | 1.21 | 0.48 |
| `plane_metrics` | 32.83 | 17.76 | 16.46 | 60.68 | 26.92 |
| `pwv` | 0.60 | 1.55 | 1.18 | 0.94 | 0.56 |
| `derived_export` | 4.68 | 16.31 | 19.17 | 7.40 | 6.04 |
| Video: plane | 12.91 | 11.31 | 9.97 | 20.96 | 12.28 |
| Video: streamlines | 62.02 | 62.13 | 60.07 | 60.86 | 58.70 |
| Video: wss | 60.42 | 59.80 | 58.94 | 61.33 | 57.27 |
| Video: tke | 61.12 | 63.99 | 60.47 | 62.16 | 61.92 |
| Video: pressure_gradient | 59.19 | 64.59 | 58.91 | 61.46 | 59.95 |
| Video: relative_pressure | 57.95 | 64.68 | 61.61 | 61.55 | 58.15 |
| **All six videos** | 313.63 | 326.49 | 309.98 | 328.33 | 308.27 |
| **Legacy compute total** | 157.80 | 168.83 | 201.14 | 212.86 | 152.50 |
| **End-to-end wall** | 479.83 | 511.15 | 529.90 | 551.47 | 470.84 |
| **Peak tree RSS (GiB)** | 11.07 | 13.02 | 14.11 | 20.49 | 13.66 |
| **Planes / paths** | 45 / 15 | 18 / 6 | 9 / 3 | 108 / 37 | 42 / 15 |

All five workers reported `ok`, and all source size/mtime checks were unchanged. Completion does **not** imply quality acceptance: predicted-label mismatch and fixed PWV selection affect interpretation. Video export consumed 58.5-65.5% of observed wall time before optimization. Background correction and fresh segmentation are the main remaining compute costs; skeleton/graph generation is comparatively small.

Nested timers overlap their parent stages and must not be added again. `plane_metrics` includes derived-volume preparation. Process-tree RSS sums shared pages more than once and is an upper-style observation, not unique memory consumption. External wall time includes interpreter startup, outputs, videos and process shutdown; the existing `total_time_sec` does not.

The final repository regression passed: **186 passed in 38.26 s**, using `python -m pytest tests/test_smoke_phantoms.py tests/test_pressure_gradient_phantom.py -q` in `autoflow311` with offscreen settings. Module invocation ensures the working checkout is tested instead of the separately installed package. No new automated test file was added. An earlier environment regression ran during the beginning of the Aorta cold baseline, adding temporary competition to that measurement.

## Video optimization implemented

**Status: Implemented.** All six offline exports use `autoflow/rendering/video_layout.py::VideoOverlay`. Two persistent `vtkTextActor` instances use fixed font sizes and pixel positions, updating only changed text. The header identifies the metric and, for dynamic videos, the one-based cardiac phase and acquisition time computed from a valid RR interval. The footer shows export-frame progress and rotation when enabled. The existing GUI background, foreground palette, metric styles and quantitative colorbars remain the source of visual settings.

**Measured mechanism.** In the installed PyVista 0.48.4/NVIDIA EGL environment, the old dynamic `vtkCornerAnnotation.SetText` call itself took very little time, but its update made the next render expensive. A fixed 20-frame surface probe spent 5.464 s rendering with updated text and 0.488 s when the old text was held constant. Removing explicit mapper updates, switching prebuilt phase actors, or bypassing PyVista's mapper input setter did not produce a comparable reduction. The implemented change therefore targets the caption actors; those mapper alternatives were not adopted. The installed screenshot path issues no second render, and the hardware renderer is NVIDIA EGL rather than software rendering.

**Paired synthetic verification.** Both variants use the same four cached cardiac phases, 20 frames, 1600 x 1200 pixels, default SSAA, camera trajectory, scalar fields/ranges, colorbars, and libx264 output at 12 fps. The surface and volume cases exercise the old caption and the new shared overlay sequentially:

| Probe | Old render time | New render time | Old frame loop + encoder close | New frame loop + encoder close | Observed export-loop speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| Surface | 6.872 s | 0.157 s | 7.486 s | 0.703 s | 10.65x |
| Volume | 5.922 s | 0.321 s | 6.519 s | 0.855 s | 7.63x |

The export-loop totals include per-frame data/text/camera updates, rendering, screenshot readback, encoding append and encoder close. They exclude geometry construction and plotter initialization, so these ratios are not whole-pipeline or whole-case speedups. The volume comparison retains the existing opacity-transfer refresh. Shared GPU activity varied during the runs; the values above are observations rather than isolated-device guarantees.

The raw first/middle/last frames (indices 0, 10, 19) were compared before video compression. After excluding only the upper-left caption rectangle (`x < 800, y < 220`) and lower-left footer rectangle (`x < 1000, y >= 1100`), **every remaining pixel is identical** in all six comparisons, including the scalar bars. The two overlay screenshots were also visually inspected for placement and legibility. Evidence, full per-block timings, screenshots, movies and source hashes are in `results/audit_20261007_full/video_export_trace/overlay_trace_summary.json` and its adjacent trace directories. This verifies the tested surface and volume scenes; the separate real Aorta replay evaluates the six complete export paths.

The shared `_time_index` schedule also fixes nonrotating `time_repeat > 1`: each phase is repeated equally instead of filling the final frames with the last phase. Its expected sequence and rotation cycling are covered in `tests/test_smoke_phantoms.py`; no new test file is introduced. User-facing behavior is documented in the [video feature guide](../features/videos.md).

A real Aorta replay used the same in-memory workspace with original embedded correction and segmentation for both renderers. It is separate from the cold nnUNet baseline above. Every export kept 1600 x 1200, 180 frames, 12 fps, shared GUI style, scalar bars, camera path and anti-aliasing. Renderer-local geometry caches were cleared before each variant. The updated default canvas has the same layout before and after the later small-canvas scaling fix.

| Video | Frozen renderer (s) | Updated renderer (s) | Observed speedup |
| --- | ---: | ---: | ---: |
| plane | 13.65 | 5.46 | 2.50x |
| wss | 105.98 | 4.21 | 25.20x |
| tke | 90.33 | 7.48 | 12.08x |
| pressure_gradient | 84.30 | 5.19 | 16.23x |
| relative_pressure | 88.19 | 5.41 | 16.29x |
| streamlines | 87.52 | 6.54 | 13.38x |
| **Total** | **469.98** | **34.29** | **13.71x** |

All 12 paired MP4s decoded to 180 frames. The first/middle/last uncompressed frames matched pixel-for-pixel across the full scene region (all columns, rows 160 through height minus 70); only top/bottom caption strips were excluded. This also covers the scalar bars. A visual contact sheet confirms unchanged geometry, colours and framing. Encoder compression can vary because caption pixels changed; the comparison uses frames before encoding.

These timings include geometry, plotter setup, rendering and MP4 encoding/publication, but exclude later verification/PNG writes. They are one baseline-first pair per feature on shared hardware; the other anatomy downstream controls were running concurrently. The synthetic attribution isolates the caption mechanism, while this replay demonstrates practical end-to-end export gains. Do not generalize the exact multiplier to every case or infer a new full-pipeline timing by subtracting it from the cold baseline. Evidence: `video_replay/comparison.json`, `video_replay/before_after.png`, and complete MP4s under `video_replay/{baseline,updated}/`.

## Correctness and failure handling to fix first

| Priority | Finding | Evidence and consequence | Change owner |
| --- | --- | --- | --- |
| P1 | Case switch retains the previous case's plane-ROI undo stack | Synthetic Qt reproduction executes the actual load handler with disk/3D surroundings stubbed, then actual ROI undo. New case plane 0 receives old case coordinates. Bind history to case/plane identities or clear on replacement and plane-list rebuild. Static review also confirms import-replace, deletion/index shifts, regeneration and centerline edits do not clear it (`app.py:3073,5895,6805,7453,8457`). | `autoflow/ui/app.py:6968`; `autoflow/ui/ortho_viewer.py:987`, `:2621` |
| P1 | Basic and derived plane measurements can use different support | Two-label synthetic counterexample: basic label-filtered area is 16 mm², derived support is 32 mm², and TKE mean is 50.5 instead of the selected region's 1. This demonstrates a reachable contract mismatch, not contamination established in all real samples. Share a validated label/connected-component/ROI slice specification across metric families. | `algorithms/metrics/planes.py:144`; `algorithms/metrics/plane_derived.py:77`, `:149`; `core/pipeline.py` |
| P1 | A failed CLI case exits successfully | Invalid-flow H5 reproduction produces `batch_report.status=error` and `Done: 0/1 succeeded`, but process return code is 0. Return nonzero for failed/partial batches and write actionable error text to stderr. | `autoflow/cli.py:369`; `autoflow/api.py:575` |
| P1 | Reported total time excludes substantial requested work | `total_time_sec` stops before videos, quality/summary files and phase-unwrapping NPZ. Retain a compute timer and add a genuine end-to-end timer plus per-stage outcomes. | `autoflow/processing.py:725`, `:740`, `:949`, `:1019`; `autoflow/api.py:589` |
| P2 - Implemented | Nonrotating videos now repeat every phase equally | Replaced the old final-phase clamp with shared `_time_index`: four phases with repeat 2 now schedule 0,0,1,1,2,2,3,3. Rotating and nonrotating dynamic exports share this schedule. Added coverage in the existing smoke suite; the default rotating cold runs did not exercise the old failing branch. | `autoflow/rendering/videos.py`; `tests/test_smoke_phantoms.py::test_video_phase_schedule_repeats_every_phase_and_cycles_for_rotation` |
| P2 | Moving a plane can retain an obsolete contour cache entry | Source-confirmed missing center/normal in the outer contour cache key; the inner geometry-aware key is bypassed by an outer hit. A real-geometry reproduction is still required. | `autoflow/ui/ortho_viewer.py:532`, `:575`, `:1178` |
| P2 | Changed velocity can leave old downstream results visibly available | Correction/unwrap/revert deliberately preserve results. Keep manual geometry, but show numerical/PC-MRA stale state and offer dependency-aware recomputation. | `autoflow/core/pipeline.py:925`, `:1091`, `:1109` |
| P2 | Input validation fails too late | Invalid two-component flow reaches orientation code and throws an IndexError. Validate shape, components, time dimensions and metadata before allocating/reorienting. | `autoflow/algorithms/data/h5_loader.py:246`; `orientation.py:75` |

The Aorta run also exposes a reporting distinction: the configured PWV group is globally `Portal Vein` (`PV=14`), independent of filename anatomy. Its PWV calculation returns `ok`, while PWV QC is `warn` (fit R² 0.2342) and overall QC is `not_ready`. Generated segmentation is active, but `capabilities.has_segmentation` remains the initial-loader snapshot (`false`). Report source capability, active state, completion and quality separately. The cross-anatomy validation below confirms the source/model ontology and explains why an output named `Portal Vein` is produced for every category; completion does not establish the intended vessel or physiological validity.

## Opportunities by processing stage

| Stage | Remaining work and proposed direction | Evidence / constraint | Priority |
| --- | --- | --- | --- |
| Discovery and preflight | Preview the selected case list, resolve model/checkpoint/device and output writability before correction starts. Clarify top-level H5 scanning versus recursive/DICOM discovery. | H5 previews already read only metadata. DICOM conversion was reviewed statically, not exercised by these H5 inputs. | P1/P2 |
| H5 decode and correction | Eliminate the second complete H5 decode during BGC; retain one source-decoded representation with a correction interface. In a batch, a fused load/correct path is also possible. | `core/pipeline.py:467,482,895`; `processing.py:440,455`. Preserve separate GUI actions, dual-VENC semantics, orientation, sigma and cache policy. | P1 |
| Load memory | Lazily create wrapped phase and use explicit copy-on-change snapshots for flow undo. | `core/pipeline.py:571–583,913–918`: flow/input/phase can occupy several full arrays; normalized inputs construct complex exp/angle even before unwrapping is requested. | P1/P2 |
| Normalized-input correction | Avoid round-tripping `mag+flow` through a full complex64 four-channel volume; apply a valid correction cache directly in phase space. | `phase_correction.py:363–380`. Needs wrap-boundary and numerical equality checks; not measured in the selected complex H5 set. | P2 |
| WRLS/ARTO | Profile GMM GPU synchronization and convergence; record GPU fallback causes and add cancellation checkpoints. | GPU fit, chunked CPU fit and the small CPU FISTA loop already exist. Do not claim gains by reducing polynomial order, ARTO rounds, precision or fit iterations. | P2 |
| Noise mask / PC-MRA | Reuse temporal magnitude/speed reductions and mark PC-MRA dirty when working velocity changes. Separate compute and compressed-write timings. | Rendering noise masks already preserve original magnitude/velocity; do not replace them with destructive data masking. | P2 |
| Phase unwrapping | Keep wrapped phase lazy; account for export of phase/flow/wrap arrays, offer an explicit diagnostic-output level, and expose backend-specific progress/cancel behavior. | Default CUDA lap4D is measured; gc3D, NPRS, PUDIP and GUST are not benchmarked here. A different method is not an equivalent performance optimization. | P2 |
| 4D segmentation | Replace gzip-NIfTI round trips with array/memmap transport and reuse a model worker for ordinary batches. Move grouped inference into an isolated GUI worker if enabling the grouped path there. | `segmentation/nnunet_grouped.py:182–228,263–309`. CLI grouped and GUI isolated paths differ. Unique-map reuse, shared preprocessing, single-fold and disabled TTA already exist. | P1/P2 |
| Segmentation feedback | Report progress while each future actually completes, not after the whole file-write map has finished. Run model/dependency preflight before expensive work. | `nnunet_temporal.py:251–265`; model resolution is currently inside the segmentation stage. | P2 |
| Segmentation review | Put import/save/4D cleanup on the existing worker/candidate framework. Cache label inventories by segmentation version and compute all per-frame counts in one pass. | Active GUI paths: `app.py:2166,3442,3744,3264–3345`. External Labeler is current; old internal brush history is not an active-workflow bottleneck. | P2 |
| Mask preprocessing | Materialize group masks lazily or within spatial ROIs, preserving 4D labels and topology. | `core/pipeline.py:719,741,758`: full XYZT group masks and global masks coexist. Existing preprocessing signatures already avoid some repeats. | P2 |
| Skeleton / graph | Retain current algorithms unless the real timings justify further work; avoid rebuilding graphs already suitable for PWV and plane import. | Geometry is measured separately. Do not sacrifice branch topology to shorten a small stage. | P3 |
| Plane generation / import | Skip automatic plane creation when an import will immediately replace it, while retaining graph support required by path-relative import and QC. | `processing.py:595–612`. | P2 |
| Basic and derived plane sampling | Share exact support/slice geometry, sampled fields and label policy; retain existing process isolation and cancellation. | Two slice paths currently duplicate work and have the support mismatch above. Four/eight-process policies and repeated-mask caches already exist. | P1 |
| TKE per-plane preparation | Prepare and mask a provided TKE volume once per augmentation pass instead of once per plane. | Synthetic 1/4/8-plane calls produce 1/4/8 full new arrays despite already prepared float32 input. Shared prepared-array replacement preserves every synthetic metric and pixelwise value. Real speedup not measured. | P1/P2 |
| WSS | Cache mask-specific wall probe coordinates, normals and probe-validity geometry; consider bounding-box EDT/preparation with adequate padding. | `metrics/wss.py:79–98,124–164`. Preserve wall derivative, units, invalid-probe handling and changing 4D walls. | P2 |
| Pressure / relative pressure | Bound sparse-system/preconditioner caching when every phase's mask differs; consider separating gradient-only requests from pressure reconstruction. | `metrics/pressure.py:455–473,758`. ROI cropping and reusable AMG systems already exist. Separating requested outputs is a public API/config change. | P2 |
| Vortex | Reuse shared velocity derivatives only if pressure/vortex smoothing and support semantics match; retain finite-stencil support. | Different parameter/support choices can invalidate reuse. Measure first; no speedup is claimed. | P3 |
| PWV | Reuse compatible centerline graphs, reduce full group flow/mask copies and separate PWV from unrelated plane-export skipping. Validate anatomy-specific group selection. | `algorithms/pwv.py:677–704`; `processing.py:660–668`. | P2 |
| Derived-cache checks | Use reliable data version IDs plus parameter signatures for in-session checks; reserve full-array content hashes for boundaries that need them. | `core/pipeline.py:43–52,1327–1349,1452` scans full arrays and can copy strided inputs even on hot cache checks. All edit/load/correction entry points must bump versions. | P2 |
| 2D time/slice interaction | Fix fast-path invalidation to depend on geometry/display properties, not equal time indices; separate image and overlay updates. | Synthetic Qt reproduction: nine full `set_slice` calls and zero `update_image` calls during three transitions across three views. | P1/P2 |
| 3D streams / planes | Precompute current/adjacent streamlines in a worker; let the main thread only swap rendered data. After moving a plane, show geometry immediately and update metrics/pathlines asynchronously. | `viewer.py:2000–2019,2031–2072`; `app.py:4793–4814,5152–5178,7569`. Initial pathline calculation elsewhere already has a worker. | P2 |
| Cache budgets / object tree | Replace whole-cache clearing at 24 entries with bounded LRU/byte budgets; bound 3D mesh/tube caches; retain object-tree expansion/selection by UID. | `ortho_viewer.py:153–160`; `viewer.py:1466,1956`; `app.py:5957–6077`. | P2/P3 |
| Videos - Implemented caption optimization | Replaced repeatedly resized corner annotations with persistent, fixed-size text actors across all six exports. Caption styling uses the shared GUI palette, with a title, cardiac phase/time where available, and rotation/export progress. Remaining work: geometry/preparation progress, phase-cache/worker lifetime, and separately measured GUI handoff. | Paired synthetic traces identify text-triggered rendering as the dominant cost; changing mapper updates did not materially help. Geometry, scalar mapping, colorbars, resolution, SSAA, camera and encoding quality are preserved. NVIDIA EGL is active and screenshots do not render twice. See the implemented section above. | Implemented; remaining P2 |
| GUI video job handoff | Send only metric-relevant arrays/geometry, and prefer local temporary storage/memmap before final NAS output. | `rendering/jobs.py:22–38` serializes nearly the entire workspace under the output directory; job timing starts after handoff. CLI cold timings do not include this GUI-specific cost. | P2 |
| Output I/O | Measure final NPZ/JSON/H5 writes; expose required-versus-diagnostic outputs and compression choices. Keep streamed/atomic MP4 output. | Large derived NPZ is already uncompressed; no blanket recommendation to disable compression. Final phase NPZ and video times are omitted from old total. | P2 |
| Cache persistence | Validate attrs/shape before reading correction arrays; write new datasets before replacing old caches; preserve reusable sidecars when source H5 is read-only. | `data/correction_cache.py:70–162,275`; `segmentation/io.py:153–181`. Fault injection was not performed on user files. | P2 |
| Batch recovery | Atomically persist each case/stage outcome, add resume with input/config/output identity checks, distinguish partial success and verify requested artifacts. | `api.py:417,572–583`; video exceptions currently warn and continue. | P1/P2 |
| API contracts | Separate configuration-template input from an already loaded workspace to avoid deep-copying a full workspace that is immediately reloaded. | `processing.py:397,440`. Ordinary run_batch passes an empty base workspace, so the large-copy issue is not attributed to every batch. | P2 |
| DICOM workflow | Reuse validated conversions, report actual converter stages and support cancellation through an isolated worker. | `dicom_conversion.py:31–88`. Static review only; H5 samples do not test raw DICOM import. | P2 |

Further persistence details: `core/pipeline.py:1665–1674` publishes plane JSON before atomic H5 publication, so cancellation can leave files from different generations. Publish a result set with a manifest/commit marker after all files succeed. Internal/Python workspace serialization (`core/models.py:1534–1536,1593–1622`, `io_utils.py:5–7`) expands arrays to Python lists and indented JSON; a future session format should store arrays in binary payloads with a small schema/version manifest. These helpers are not connected to current GUI save/load, so this is not claimed as measured GUI save latency. The GUI currently lacks complete session save/restore, although some display preferences already persist through QSettings. These are source-confirmed risks; no destructive fault injection was performed.

The pressure solver checks cancellation at phase boundaries but not inside iterative CG/direct factorization (`metrics/pressure.py:88–105,457`), so a difficult phase can delay cancellation. Streamline video preparation also finishes all phases before the first frame (`rendering/videos.py:856–882`); add preparation progress and a bounded prefetch queue. Graph triangle pruning copies the full graph for candidate edge removals (`algorithms/graph.py:54–67`); optimize only if a complex graph is actually costly.

## Cross-anatomy segmentation validation

**Source-confirmed label space.** The five selected H5 files already contain global labels. Each `segmask` declares `source_label_space=native`, `label_space=global`, the correct `source_part`, and a native-to-global `label_mapping` that exactly matches the source exporter. Reading the full 4D masks gives the following nonzero label inventories:

| Source category | Actual embedded global IDs across all phases | Contains global `PV=14`? |
| --- | --- | --- |
| Aorta | 1–10, 14, 16 | Yes |
| Carotid | 17, 18, 19, 20, 31, 32 | No |
| Cerebrovascular | 17, 19–30 | No |
| PortalVein | 1–16 | Yes |
| RenalArtery | 4, 10–16 | Yes |

The authority is the external Segdata `scripts/methods/datasplit/common.py`: part definitions at lines 39–135, global name-based merge at 138–160, and `label_remap_for_part` at 328–338. `Case2Case/export_testset_autoflow_h5.py` imports that module, aligns `seg_true.seq.nrrd` to the image, and applies the mapping when `--use-seg-true` is selected; `--label-space` defaults to `global` (154–230, 345–354). For example, native Carotid `1=LICA` becomes global 17, native Cerebrovascular `1=RICA` becomes 20, and native PortalVein/RenalArtery `1=PV` becomes 14. Aorta's mapping is the identity. **Do not apply the native-to-global mapping again to these exported H5 masks.** Their already-global IDs are the appropriate comparison space.

The export CSV records conversion status and shape, not label space. A `skipped` base conversion can still be followed by `_write_seg_true` (399–406), so it does not imply missing remapping. Conversely, the older/general `convert_aorta_h5_to_autoflow.py` and `Case2Dicom/export_manifest_to_autoflow_h5.py` paths only transpose/cast their source mask and do not themselves remap labels. These findings apply to the five inspected files, not every possible H5 export. Detailed attributes, mappings and voxel counts are retained in `results/audit_20261007_full/source_label_mapping_audit.json`.

**Current model ontology.** A direct read of the resolved Dataset7020 model's `dataset.json` confirms that all 33 label names and IDs, including background, exactly equal the source global ontology; no name aliasing is required for that comparison. AutoFlow's `configs/labels.json` uses alternate display/configuration names at eight of the same IDs: `A1/RBCT=5`, `A2/CCA=6`, `A3/LBCT=7`, `SS/StrS=25`, `LPCAP12/LPCA=26`, `RPCAP12/RPCA=27`, `LJGV/LIJV=31`, and `RJGV/RIJV=32`. This is a correspondence between the declared maps, not an automatic alias translator: configured label strings must exist in the active AutoFlow label map. `PV=14` has the same spelling and ID throughout. The metadata includes all five body-part categories; the folder name `Dataset7020_Aorta_4DTemporalFT` alone does not establish an aorta-only model. Metadata also does not independently prove the checkpoint's training distribution or accuracy. The comparison and metadata hash are saved in `results/audit_20261007_label_ontology.json`.

**Observed prediction anomaly; cause unresolved.** The saved first-phase cold-run inventory for Carotid contains predicted IDs 4, 11, 14, 15 and 16 instead of the embedded source set 17, 18, 19, 20, 31 and 32. Cerebrovascular contains predicted IDs 10, 14 and 16 instead of its cerebral source set. This is an anatomy-consistency alarm. The source masks are not assumed ground truth, and inventory agreement or disagreement is not a voxel accuracy score. Matching metadata rules out a declared source/model numeric-ontology mismatch in these checks; it does not establish the cause of the prediction anomaly or validate preprocessing, transport, orientation, checkpoint behavior or the optional correction/unwrapping path. Model routing uses a fixed auto model and currently does not reject an anatomically inconsistent label inventory before activation. Controlled comparisons and visual review are required before attributing the anomaly or accepting downstream metrics.

### PWV group interpretation across the five categories

The checked-in `configs/pwv.json` enables a single group named `Portal Vein` with `labels=["PV"]`. `PwvParams.from_dict` resolves that exact name through the active label map to `[14]` (`core/models.py:773–803`; `config.py:975`). This checked-in bundle differs from the built-in fallback defaults, which disable PWV and have no groups (`config.py:458–461`). `compute_pwv_groups` keeps the configured group name, selects label 14 from the temporal-majority mask, and skips an empty group; it does not infer a different vessel from the input category (`algorithms/pwv.py:635–664`).

| Cold-run category | Group in the actual PWV output | Computational outcome | Interpretation |
| --- | --- | --- | --- |
| Aorta | `Portal Vein`, `[14]` | `ok`; 3 valid planes, R² 0.2342 | Both embedded source and prediction include PV. This result describes the configured PV mask, not an aortic PWV preset. |
| Carotid | `Portal Vein`, `[14]` | `skipped`; only 1 valid plane | Source has no PV; the active prediction supplies label 14. The name reflects configuration, not a verified portal vein. |
| Cerebrovascular | `Portal Vein`, `[14]` | `ok`; 2 valid planes, R² 1.0 | Source has no PV; prediction supplies label 14. A two-point fit and `ok` do not establish correct anatomy or a valid cerebral PWV measurement. |
| PortalVein | `Portal Vein`, `[14]` | `ok`; 3 valid planes, R² 0.8312 | Category and configured target are consistent; segmentation, path, waveform and fit still require review. |
| RenalArtery | `Portal Vein`, `[14]` | `skipped`; PWV fit failed | Source includes PV as well as renal arteries. The configured target remains PV, not LRA/RRA. |

Choose and record the intended vessel group before running PWV. For an aortic analysis, a reviewed `AAO`/`ARCH`/`DAO` group is a configuration candidate; for carotid, cerebral or renal analyses, select the intended side and vessel labels explicitly and review the resulting connected path. Do not merge unrelated vessels merely to produce a fit. Keep `PV` only when portal-vein analysis is intended, or disable PWV when no suitable target is defined. Add anatomy/coverage preflight and preserve separate computational, segmentation-review and QC statuses. These are configuration and validation recommendations; no PWV configuration or production code was changed by this audit.

### Checkpoint drift and exploratory preprocessing check

The default checkpoint was being updated externally during this audit. Its initial recorded SHA-256 was `67e820...918d8`; before the no-lap4D check it was `48e3a2...cf97b18`; a later stable read was `bef3a8...f53f35` (mtime 2026-10-06 22:45:31 UTC). Full hashes, timestamps and unchanged-stat checks are in `checkpoint_drift_audit.json`. No checkpoint was written by this audit. Consequently, five cold timings describe the observed live setup, but they are not a frozen-weight comparison; the exact weights used by every inference cannot be reconstructed from these three samples. Freeze the model/checkpoint before further accuracy or preprocessing attribution tests.

A brain run requested the same model path, fresh WRLS/ARTO, noise masking, PC-MRA and nnUNet4D while omitting lap4D. It completed in 149.87 s, produced no phase-unwrapping artifact, and left the source unchanged. Its first-phase prediction contained labels `3,10,12,13,14,16`, while the embedded reference contains `17,19-30`. Because checkpoint identity changed, this is an **exploratory observation, not a valid one-variable control**; it does not establish whether lap4D caused the original anomaly. Neither model, orientation, normalization nor transport has been isolated as the cause. The paired video replay below uses one already computed workspace and is unaffected by checkpoint drift.

### Reference-label downstream coverage

A separate non-cold run reused original embedded correction and global segmentation, then exercised skeleton, graph, planes, WSS, optional TKE, pressure/relative pressure, vortex, plane metrics, PWV and volume exports. No inference or source-cache writes were performed. Source/active label counts matched for all five, and all input size/mtime checks were unchanged. Embedded segmentation remains a reference, not asserted ground truth.

| Anatomy | Planes / paths | WSS (s) | TKE (s) | Pressure (s) | Vortex (s) | Plane metrics and files (s) | PWV result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Aorta | 41 / 14 | 14.87 | 0.52 | 3.44 | 0.67 | 16.25 | ok |
| Carotid | 95 / 32 | 11.70 | 1.83 | 11.09 | 1.76 | 18.50 | skipped |
| Cerebrovascular | 204 / 71 | 23.28 | 6.86 | 27.23 | 3.06 | 37.18 | skipped |
| PortalVein | 127 / 44 | 20.62 | 0.80 | 6.32 | 1.02 | 44.19 | ok |
| RenalArtery | 39 / 14 | 7.07 | 0.63 | 2.47 | 0.51 | 11.12 | ok |

Carotid and brain PWV correctly skip the configured portal-vein group because label 14 is absent in these embedded references; no anatomy-specific PWV result was established for them. All five QC reports remain `not_ready`, so successful execution is not acceptance of physiological values. The brain reference generated 204 planes/71 paths versus 9/3 from the cold prediction, demonstrating why the earlier downstream workload was not representative of its reference anatomy. These shared-machine timings ran concurrently with Aorta video replay and must not be compared as isolated performance gains. Full stage times and QC are in `downstream_reference/<anatomy>/reference_summary.json` and `quality_report.json`.

## Workflow and startup experience

Existing functionality includes stage-specific Run All actions, automatic Skeleton → Graph → Planes prerequisites for metrics, task cancellation with completed-step retention, a static downstream-recompute reminder after correction, automatic QC refresh on Review entry, and some saved display preferences. Improvements should build on these:

| Area | Concrete change | Current evidence |
| --- | --- | --- |
| Result freshness | Show “computed from previous velocity/segmentation” and an update action while retaining manual geometry. | Ready status checks presence rather than result version (`app.py:1870–1892`), despite the existing correction reminder (`:1079`). |
| Run All semantics | Show the stage's actual execution list, what will be rebuilt and what will be reused; make “continue incomplete work” easy without repeated confirmation dialogs. | Centerline Run All always starts at Skeleton and destroys downstream graph/planes; Hemodynamics also enables streams/all-plane pathlines but omits PWV (`app.py:214–223,7523–7536,7593–7618`). Stage names/task titles already exist; the scope is not completely unlabeled. |
| Missing prerequisites | Show a direct “segmentation required” action and distinguish completed/skipped/failed counts. | Missing segmentation produces success+skipped StepResults; the worker counts success and ultimately reports completed (`pipeline.py:1115,1689,1964`; `app.py:472–485,7487`). |
| Labeler review loop | Add “apply saved Labeler version” without requiring editor closure; bind pending versions to their original case. | Current workflow is save, close Labeler, then confirm application; a case switch leaves the old-case result only in a log (`app.py:3914–4001`). Existing export reuse should be retained. |
| Output locations | Display the active case result directory, artifact list and open-folder action. Keep video export location separate or clearly show that it changes subsequent output routing. | Video completion changes `ws.paths.output_dir` (`app.py:7328–7329`), affecting later automatic metric writes. |
| Config validation | Reject a file passed as `--config-dir`; expose resolved configuration sources and startup errors before application construction fails. | A real `pwv.json` path is accepted as a directory and silently falls back to 21 module defaults, including empty PWV groups (`config.py:549,672–685`). Invalid JSON can fail before `MainWindow.show()` (`app.py:617,8649`). |
| Core-only CLI installation | Declare nibabel as required where needed or lazily import NIfTI functionality; allow help to run before importing the rendering stack. | nibabel is declared only in gui/all extras, but is imported unconditionally from segmentation IO. A missing-module simulation makes CLI help fail before parsing; no new clean environment was installed. New-process CLI import took 1.579 s versus 0.048 s for the already-light launcher. |
| Metadata discovery | Put H5 case discovery into a cancellable worker with the same error path as loading. | File selection immediately calls discovery in the GUI thread without local exception handling (`app.py:7052–7055`). |
| Session continuity | Preserve pending edits/results in a recoverable session and handle leaving an uncommitted edit explicitly. | Clear/exit discard active editor candidates; File has no full workspace save/load (`app.py:2697–2708,7349–7355,8632–8638`). Do not add confirmation to unrelated ordinary actions. |
| Reproducible reruns | Save the full resolved configuration, software identity, model/fold/checkpoint identity and segmentation provenance with every run. | Normal summary lacks a complete configuration/provenance snapshot, particularly with source-cache writes disabled (`processing.py:959–1017`). |

## Interactive validation

The five selected real inputs completed a separate interactive-review scenario using their embedded segmentation and saved correction, without rerunning correction, segmentation or preprocessing. Each case used a fresh process, a real three-view `OrthoViewer` at 840 x 960, and two full phase cycles in both Flow LR and Speed. The label-panel probe called the real `MainWindow._refresh_segmentation_ui` with a real `SegmentationDock` and phase-mask controls for phases 0 -> 1 -> 2 -> 1 -> 0; only the optional advanced model-configuration controls were absent. Source H5 size and mtime remained unchanged for all five cases.

These are **shared-machine interaction references**: other audit pipeline ablations could run concurrently on the same host. The table reports refresh plus Qt offscreen event-processing wall time. The load column starts at `PipelineEngine.load_data` and includes viewer construction and its first refresh, excluding process/import startup. Warm values are from the second full cycle, with no OS cache flush. Label-panel percentiles use only five updates, so they describe this small sample rather than a stable tail-latency estimate. No 3D scene, native display or remote-desktop transport was timed.

| Region | Phases | Load through first 2D (s) | Warm Flow LR p50 / p95 (ms) | First Speed refresh (ms) | Warm Speed p50 / p95 (ms) | Label panel p50 / p95 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Aorta | 18 | 6.42 | 16.4 / 18.2 | 40.9 | 22.2 / 25.8 | 49.4 / 60.7 |
| Carotid | 13 | 24.40 | 65.9 / 71.7 | 173.5 | 90.2 / 99.0 | 235.7 / 241.9 |
| Cerebrovascular | 12 | 16.58 | 63.0 / 77.4 | 180.9 | 83.4 / 91.2 | 208.2 / 211.0 |
| PortalVein | 15 | 6.80 | 23.7 / 25.2 | 72.2 | 33.5 / 37.8 | 83.3 / 87.9 |
| RenalArtery | 25 | 5.15 | 20.6 / 22.5 | 34.3 | 31.8 / 44.0 | 35.5 / 39.0 |

Across all 332 measured phase transitions, the three slice views made 996 `set_slice` calls and 0 `update_image` calls. This confirms the synthetic fast-path finding on real data: changing time repeatedly takes the full slice-installation path. The larger Carotid and Cerebrovascular inputs also spend roughly 0.2-0.24 s in a label-panel refresh alone. Cache label metadata/counts against segmentation changes and separate image-only phase changes from slice geometry before evaluating whole-window responsiveness; the table is not a sum of all work performed by the GUI's time callback.

Separate synthetic Qt probes reproduced three phase transitions with nine full slice installations, and reproduced old-case ROI undo history being applied to a new case. The case-load probe exercised the real load-handler and undo methods while substituting disk loading, configuration application, 3D scene work and unrelated controls; it establishes the state-transition defect, not a clinical case-loading validation. Full per-frame timings and method counts are in `ortho_interaction_results.json` and `ortho_<region>.json`; the reproducible harness is `ortho_interaction_benchmark.py` under the local audit output directory.

The small synthetic probes are intentionally separate from the two supported automated test files. They reproduce contracts and UI state transitions; they are not clinical validations. The completed real-data Ortho probe reused embedded correction/segmentation and remains a separate review scenario, not a substitute for cold measurements. Offscreen 2D updates do not measure native-screen FPS or 3D responsiveness.

## Artifact validation

The five completed cold-run output directories were inspected with `results/audit_20261007_full/validate_artifacts.py`. These are the original baseline videos, generated before the caption/phase-scheduling changes; the checks below do not substitute for validating newly rendered videos.

| Artifact check | Observed result | Scope |
| --- | --- | --- |
| Baseline MP4 decoding | 30/30 videos decode to 180 frames each: 5,400 frames total; all H.264, 1600 × 1200, 12 fps | Six requested video families for each of the five categories |
| Selected RGB frames | First, middle and last frame reads succeeded for all 30 videos | 90 sampled pixel arrays; middle-frame contact sheet also inspected |
| H5 metadata | 15/15 files open; 57,294 dataset entries expose readable shapes and dtypes | Three H5 outputs per case; dataset payloads were not fully read |
| NPZ/NPY headers | 20/20 archives pass; all 155 array headers are readable NPY 1.0 and contain no object dtype | Four NPZ outputs per case; array payloads and whole-archive CRCs were not fully checked |
| Result JSON | All five summary and quality-report pairs parse | Parsing establishes container readability, not successful QC |

The NPZ checker initially called a private NumPy helper unavailable in the installed version. It was corrected to the public version-specific header readers and the NPZ checks were repeated successfully; that initial checker error was not evidence of damaged output. Results and frame statistics are retained in `artifact_validation.json`, with the 30-panel middle-frame overview in `video_contact_sheet.png` under the same local audit directory.

Video decodability does not establish visible vessel coverage or quantitative validity. The Cerebrovascular baseline overview contains sparse structures, and its pressure views have little or no visible vessel structure at the sampled middle frame. This remains a segmentation/support/display review target alongside the label anomaly above. Container/header checks do not validate segmentation anatomy, finite numerical payloads, metric accuracy or scientific readiness.

## Reproduction and local artifacts

```bash
conda activate autoflow311
python tools/benchmark_pipeline_audit.py \
  --input-dir /nas-data2/ryy/CMR4DFlow2026/Segdata/data4seg_test/h5s_autoflow \
  --output-dir results/audit_new_run
```

| Harness parameter | Type | Default | Where configured | Effect | Code owner |
| --- | --- | --- | --- | --- | --- |
| `--input-dir` | Path | Required | Command line | Select the directory containing the five category prefixes | `tools/benchmark_pipeline_audit.py` |
| `--output-dir` | Path | Required | Command line | Create a new isolated audit output directory | `tools/benchmark_pipeline_audit.py` |
| `--regions` | List of category names | All five | Command line | Limit the deterministic median-size selection to named categories | `tools/benchmark_pipeline_audit.py` |
| `--no-videos` | Boolean flag | False | Command line | Exclude optional videos while retaining all metric families | `tools/benchmark_pipeline_audit.py` |

The output directory must not already exist. Model changes are outside the requested follow-up scope; benchmark comparability depends on external model/hardware conditions. `--regions` selects a subset and `--no-videos` explicitly excludes videos. The harness leaves the selected H5 inputs untouched, reads actual batch status in addition to exit codes, launches cases sequentially and records source size/mtime, wall time and sampled process-tree RSS. Low-overhead wrapper timers are inclusive and may overlap. Optional `PYVISTA_OFF_SCREEN=true`/`QT_QPA_PLATFORM=offscreen` are already set for workers.

Local evidence under `results/audit_20261007_full/` includes `manifest.json`, `measurements.json`, per-case logs and detail timers, the normal per-case outputs, `regression_final.log`, `checkpoint_sha256.txt`, `gui_manual_probe_results.json`, `check_tke_plane_repeated_prepare.json`, `check_plane_support_consistency.json`, and `cli_failure_probe/probe_result.json`. The renderer probe retains installed-library source and VTK capabilities. Identifiable source filenames stay in local evidence; validation data is not copied into `tests/` or demo data.

## Acceptance and remaining audit opportunities

The requested ROI-history, plane-support, 2D invalidation, duplicate-load,
repeated-TKE preparation, CLI status and complete case-timing fixes are accepted
by the follow-up checks above. Offline video caption/scheduling improvements
retain the earlier paired rendering/decoding verification.

Remaining unimplemented audit opportunities include asynchronous interactive
stream/plane operations, smaller GUI video-worker payloads, bounded long-session
caches and incremental persistence. These are separate follow-up work; model
changes and label investigations are deferred by the user. Cold pipeline
baseline tables above remain historical and have not been replaced with an
unmeasured estimate for the combined changes.

Not exercised here: raw DICOM conversion, every alternative unwrapping backend, 3D nnUNet, live human Labeler editing, all native 3D gestures, disk/network fault injection and unattended long-session cache growth. These remain explicit validation targets rather than being described as passed.