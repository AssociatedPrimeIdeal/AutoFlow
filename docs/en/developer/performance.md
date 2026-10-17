# Current pipeline performance

The [2026-10-07 cross-anatomy pipeline and interaction audit](pipeline-audit-2026-10-07.md) covers five input categories, optional metrics and videos, GUI state, failure semantics and remaining optimization opportunities.

## Validated changes on 2026-10-18

The grouped CLI/Python pipeline now includes bounded gzip/shuffle H5 decoding,
shared float32 CUDA feature maps, GPU-first padding, uncompressed temporary
source NIfTI files, and threaded CPU prediction export. WSS uses isolated phase
groups and a conservative spatial crop; repeated masks keep wall reuse. Pressure
stencils and centreline interpolation avoid repeated work. The plane worker
threshold is 512 plane-phases, with up to eight isolated processes.

The baseline is the current worktree captured immediately before integration,
including its existing graph and plane-plan changes: **44 paths and 29 planes**.
This differs from the older external experiment's geometry. Both versions use
the same frozen Dataset7020 checkpoint, NumPy/Torch seeds, deterministic CUDA
settings, input, correction, fresh 4D segmentation and WSS/pressure requests.
Embedded correction/segmentation are bypassed and source writes are disabled.
OS caches were not flushed. Videos, optional phase unwrapping, PWV, TKE and
vortex analysis are excluded from the pipeline timer.

| Stage | Current baseline | Integrated implementation |
| --- | ---: | ---: |
| Complete API call | 151.42 s | 77.72 s |
| Load and correction | 22.78 s | 19.32 s |
| Fresh grouped segmentation | 75.81 s | 33.92 s |
| Metrics and their exports | 44.12 s | 13.18 s |

The observed complete-API speedup is **1.95x**;
the 30-second target is **not reached**. NAS/cache conditions affect load time;
the table is a controlled-case observation, not a universal runtime guarantee.
The CUDA feature pool contains 112 maps (6,822,816,000 bytes) and the exporter
uses four threads. The process Torch thread count is restored after inference.

All 382 compared arrays and
297,035 scalar fields are exactly equal, including
NaN placement. Export checks also match all 10,340 H5
datasets, 9 NPZ arrays and one segmentation NIfTI plus
its affine. The existing smoke/phantom suite passes 369 checks.

GUI checks compare exact selected-plane pixels and boundary screenshots,
including in-place label invalidation. A three-frame plane video exported
through the actual child process has identical decoded pixels. A real 3D
streamline cache miss returns the same mesh in a background worker while Qt
timer events continue. Pending requests coalesce and obsolete results are
discarded. Worker objects are kept outside the serializable workspace.

The GUI's standard isolated segmentation path is retained; the grouped inference
numbers above apply to CLI/Python calls. Labeler checks use its real offscreen
window, five 4D images and one mask: initial generation, zero-rewrite reuse,
save/edit/import, missing-image regeneration and preservation of saved edits
all pass. Raw magnitude/velocity/labels and persistent exchange files remain
available. Plane-only video export trims only its private subprocess snapshot.

The [machine-readable validation record](performance-validation-2026-10-18.json)
contains the controls and numerical, GUI, video and Labeler results.

## Validated changes on 2026-10-04

The current comparison uses the registered DV H5, 20 phases, **39 paths and 113 planes**, with a frozen `fold_all/checkpoint_best.pth`. Its SHA-256 is `a3b253f440bb9b1935c6457faf9eb6790a82af098e9ccc8fbce6645ad459987c`. The baseline is a Python-source snapshot taken before these changes. Raw timings, hashes, numerical comparisons and GUI checks are in the [validation record](performance-validation-2026-10-04.json).

### Controlled plane replay

Both versions read identical frozen velocity, segmentation, branch labels, geometry, WSS and pressure arrays. Automatic base-plane processing uses four processes on both sides. The updated version also reuses label-specific/static-phase slices and parallelizes derived sampling across four isolated processes.

| Repeat | Baseline base | Updated base | Baseline derived sampling | Updated derived sampling | Total before → after |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 11.29 s | 8.82 s | 18.59 s | 5.62 s | 29.87 → 14.43 s |
| 2 | 10.75 s | 7.50 s | 18.20 s | 5.92 s | 28.95 → 13.42 s |

This reduces measured base-plus-derived plane sampling time by **51.7–53.6%**. All metric fields, QC metadata, pixelwise samples and plane ordering pass comparison at `rtol=1e-6`, `atol=1e-7`, including NaNs. These timers exclude loading, correction, segmentation, volume WSS/pressure calculation and output writing.

### Controlled WRLS fit replay

One order-2 WRLS fit on the real case's frozen velocity-derived mean/variation field retains float64 arithmetic and all 5000 FISTA iterations. GPU normal-equation construction and final volume evaluation are retained; the small coefficient loop moves to NumPy on CPU.

| Warm repeat | Original fit | Updated fit |
| --- | ---: | ---: |
| 1 | 0.673 s | 0.048 s |
| 2 | 0.575 s | 0.054 s |

The maximum correction-field difference is **6.94e-17**; comparison passes `rtol=1e-10`, `atol=1e-11`. The warm fit is about 91.8% shorter on average. This is a fit microbenchmark; ARTO/GMM, H5 I/O and the complete dual-VENC correction are excluded.

### Complete cold runs and equality

Each run recomputes correction and nnUNet4D segmentation, ignores embedded numerical caches, disables source writes, requests WSS/pressure, and excludes videos. The source size/mtime remain unchanged.

| Run pair | Original API wall | Updated API wall | GPU conditions |
| --- | ---: | ---: | --- |
| First controlled-checkpoint pair | 234.48 s | 228.36 s | GPU utilization before run: 92% vs 100% |
| Fixed-seed repeat | 281.18 s | 188.53 s | GPU utilization before run: 100% vs 37% |

GPU contention changed substantially. These full-run differences are observed timings, **not a stable end-to-end speedup percentage**. The frozen plane replay and isolated WRLS fit establish the measured algorithm gains.

With NumPy/Torch seeds fixed equally, flow, magnitude, 4D segmentation, processed 4D/3D masks, branch labels, pressure gradients and relative-pressure arrays are **bitwise identical**. The unseeded pair's relative pressure differs by at most 0.00177 Pa. An unchanged-code pressure replay confirms that changing the NumPy seed changes the existing AMG-preconditioned iterative result, while repeating the same seed gives exact equality; no pressure-solver formula or tolerance was changed.

### GUI and export validation

The real H5 loaded in 22.65 s with continuing GUI timer events; cancellation retained the previous case. A smaller-case reload verified rebinding both scene and orthogonal viewer before refresh. A real 113-plane GUI worker completed with continuing activity/progress events, actual stage/plane counts and clean QThread dismissal.

A cube phantom exported six frames through the actual isolated VTK/FFmpeg child. GUI timers continued throughout export. Cancelling a task-owned inference-style child stopped it in about 0.40 s while an unrelated process remained alive. Smoke coverage verifies modal input/shortcut locking, Escape behavior, close-to-cancel, completed-step retention, and preservation of previous MP4/H5 exports on cancellation.

Only active segmentation import/preprocessing paths were optimized. The old internal brush-history compatibility path is not used by the external Labeler workflow and is not counted as an optimization or memory gain.

## Historical measurements on 2026-10-02

## Measurement scope

Measured on 2026-10-02 with `autoflow311` (Python 3.11.15), Linux, an NVIDIA RTX PRO 6000 Blackwell and the repository's local validation H5. Canonical volume: `158 x 44 x 144`, 20 phases, 2.5-mm voxels. The run generated 34 paths and 96 planes.

This is a cold numerical pipeline: fresh WRLS+ARTO correction of both VENC sources, fresh nnUNet4D segmentation, geometry, plane metrics, WSS and PG/relative pressure. Embedded correction and segmentation are ignored; source cache writes are disabled. The source size/mtime were unchanged. No phase-unwrapping step, PWV, vortex analysis, TKE or videos were requested. GUI rendering and human review are excluded.

The GPU was already at 100% utilization before the measured run and had other users' work. These times represent shared-machine conditions, not isolated hardware throughput. Two unchanged-code cold runs took about 312 and 301 seconds inside the pipeline; their difference is not an optimization speedup.

## Baseline before the worker-policy change

The completed timing report measured 303.77 s around the API call, including final reporting. The pipeline recorded 300.93 s; stage timers sum to 300.93 s.

| Stage | Seconds | Share of stage sum | Interpretation |
| --- | ---: | ---: | --- |
| Load + fresh correction | 74.11 | 24.6% | H5 read/normalization, dual-VENC correction and alias reconstruction |
| Fresh nnUNet4D segmentation | 161.83 | 53.8% | 112 unique maps, 20 samples, 37 channels; single fold_all checkpoint |
| Skeleton | 1.30 | 0.4% | Group preprocessing and extraction |
| Graph and paths | 0.11 | <0.1% | Branch/fork/path topology |
| Planes | 1.12 | 0.4% | Placement, validation and geometry files |
| Derived preparation + plane metrics/export | 61.91 | 20.6% | 11.63 s preparation, 49.49 s plane calculation/derived sampling/export, 0.80 s final plane JSON |
| Derived NPZ export | 0.54 | 0.2% | Reuses prepared volume families |

Nested times below overlap the stages above and must not be added again.

| Derived calculation | Seconds | Calls |
| --- | ---: | ---: |
| WSS overall | 5.96 | 1 |
| WSS wall probe/fitting | 2.97 | 20 |
| PG overall, including relative pressure | 5.37 | 1 |
| Relative pressure reconstruction | 1.03 | 1 |
| Sparse matrix assembly | 0.34 | 20 |
| Sparse pressure solves | 0.43 | 20 |

WSS plus PG consumes about 3.8% of pipeline time. Even eliminating both completely would save only about 11 seconds for this case. The larger remaining costs are segmentation, correction and plane sampling/export.

## Optimized cold rerun

The updated automatic process policy was then exercised by a complete fresh correction/segmentation run. API wall time was **290.07 s**, and pipeline stage time was **289.71 s**.

| Stage | Seconds |
| --- | ---: |
| Load + fresh correction | 88.00 |
| Fresh nnUNet4D | 157.92 |
| Skeleton / graph / planes | 2.62 |
| Derived preparation + plane metrics/export | 40.80 |
| Derived NPZ export | 0.37 |

Nested WSS took **5.73 s** and PG/relative pressure **4.28 s**. Within the combined plane stage, preparation took 10.32 s, plane sampling/export 29.70 s, and plane JSON 0.78 s.

Fresh preprocessing/segmentation produced **35 paths and 105 planes**, versus 34/96 in the baseline. GPU utilization stayed at 100%, and load/correction took longer. Therefore 303.77 versus 290.07 s is an observed end-to-end timing, not a controlled percentage speedup. The frozen 96-plane comparison below is the evidence for the process-policy improvement; it preserves identical masks/flow/geometry and checks all metric/QC values.

## Reproduce a cold measurement through the Python API

Activate `autoflow311`, run from the repository root, select a fresh output directory and retain the same checkpoint for both versions:

```python
import time
from autoflow import AutoFlowConfig, run_case

config = AutoFlowConfig.from_config_dir(
    "./configs",
    output_dir="/tmp/autoflow-cold-new",
    background_phase_correction=True,
    force_recompute_corr=True,
    background_phase_write_cache=False,
    ignore_embedded_segmentation=True,
    autoseg=True,
    autoseg_backend="nnUNet4D",
    autoseg_model="/absolute/path/to/frozen/model",
    autoseg_checkpoint="checkpoint_best.pth",
    autoseg_folds="single",
    force_recompute_seg=True,
    write_segmentation_cache=False,
    requested_metrics=["wss", "pg"],
    requested_videos=[],
)
started = time.perf_counter()
summary = run_case("/absolute/path/to/case.h5", config=config)
print("API wall seconds:", time.perf_counter() - started)
print(summary["stage_times_sec"])
```

The registered validation input is `/nas-data2/ryy/CMR4DFlow2026/Segdata/qingtian_h5/DV/DV_heart_2026.03.01_B-01220711V013.h5`. It is local validation data, not shipped demo/test data. Record model hashes, source size/mtime, GPU load and stage timings. Freeze flow, masks, labels and plane geometry before comparing individual metric implementations. Source-cache reuse belongs in a separate warm-run measurement.

## Where further optimization can help

| Priority | Area | Next useful comparison | Numerical consequences |
| --- | --- | --- | --- |
| 1 | Segmentation | Profile resampling, network windows, input/export I/O under an idle GPU; retain the checkpoint and 4D context | Changing model, static/4D backend, folds or temporal context changes segmentation and needs validation |
| 2 | Plane metrics | Separate base VTK slicing from derived sampling and H5 writing; benchmark local prepared slice reuse and batched probes | Retain identical contours, labels, velocities and sampled values |
| 3 | Background correction | Profile H5/NAS I/O separately from robust fitting and GPU contention | Reducing fitting iterations/ARTO rounds can change correction |
| 4 | WSS/pressure | Batch probes or reduce avoidable conversions only after measuring cost | Retain valid support, analytic wall derivative, units and pressure gauge |

Production already reuses identical-phase geometry, crops pressure work to the vessel bounding box, caches derived families between plane/volume export, and caches pressure preconditioners when systems match. Correction/segmentation caches can accelerate repeated review runs; such warm runs must be reported separately from cold algorithm timings. [Segmentation performance history](segmentation-performance.md) documents earlier controlled changes and should not be added to today's wall-time savings.

## Plane process comparison

The measured baseline kept fewer than 128 planes serial. The updated automatic policy uses up to four processes below 128 planes when plane count times phase count reaches 1920; smaller temporal workloads remain serial. Sets with at least 128 planes still use up to eight processes. CPU and plane counts cap the worker count. The 96-plane/20-phase case now selects four workers. Run the paired replay below to test a process count on identical arrays; it reuses a generated segmentation specifically for this focused comparison and is not a cold full-pipeline baseline.

For a paired replay, load the same prepared arrays and planes and call `compute_plane_metrics(...)` followed by `compute_plane_metrics_multithread(..., max_workers=4)`. Exclude input loading, correction and segmentation from those timers. Compare every metric and QC field with `rtol=1e-6`, `atol=1e-7`, including NaNs. Also measure derived augmentation and H5 export separately; base-plane timings alone do not cover those operations.

### Controlled results

| Run | Serial | Four processes | Result check |
| --- | ---: | ---: | --- |
| First pair | 30.26 s | 13.30 s | Equivalent metrics and QC |
| Second pair | 30.14 s | 10.48 s | Equivalent metrics and QC |

This is about 56–65% less time for **base plane metrics**, not the entire pipeline. Derived sampling and file exports are additional work; the serial augmented replay took 59.83 s including WSS/PG preparation. The worker policy changed without changing any numerical formula.

## Code and validation

Measure through the public API and the metric functions above. Algorithm owners: `autoflow/algorithms/segmentation/`, `autoflow/algorithms/data/`, `phase_correction.py`, `metrics/`; orchestration: `autoflow/core/pipeline.py`, `autoflow/processing.py`. Numerical optimizations must preserve smoke/phantom truth and use controlled real-case comparisons. See [Testing](testing.md).

## Interactive Content cuts and volume dragging (2026-10-16)

Focused display replay used the default local `DV_heart_2026.03.01_B-01220711V013.h5`
case, normalized shape `(158, 44, 144, 20)`, phase 3, and a 900 × 600 offscreen
view in `autoflow311`. Loading and PC-MRA generation were outside the timers;
the source H5 size/mtime were unchanged. Each row is the median of eight
movements on shared server hardware, not a cold whole-pipeline comparison.

| Operation | Previous | Updated | Conditions |
| --- | ---: | ---: | --- |
| Orthogonal cut geometry | 68.69 ms | 2.11 ms | EGL, NVIDIA RTX PRO 6000 Blackwell |
| Cut update including render | 72.34 ms | 6.04 ms | Same GPU view, normal stationary quality |
| Orthogonal cut geometry | 73.01 ms | 3.01 ms | OSMesa/llvmpipe software fallback |
| Cut update including render | 95.79 ms | 21.38 ms | Same software view, normal stationary quality |
| Camera render during drag | 16.97 ms | 4.64 ms | Software baseline versus temporary interactive quality |

Orthogonal geometry now samples only its two adjacent image arrays and reuses
plane templates, rather than scanning the full voxel grid with VTK cutters.
Dragging temporarily disables antialiasing and uses coarser volume ray sampling;
release restores the exact previous mapper settings and render pass. SSH views
also cap the preview's longest side at 960 pixels, returning to native size on
release. Numerical source arrays and metrics remain unchanged.

The numbers above exclude X11/network transfer and do not establish the user's
desktop frame rate. Geometry preservation is checked against independent
trilinear sampling, and Qt phantoms verify direct-axis activation, mouse-driven
preview restoration and correct pointer mapping in large SSH widgets.

SSH mouse movement also coalesces pointer packets before dispatch to VTK, using
an approximately 30 ms preview interval. Frame capture throttling alone does
not prevent every incoming mouse event from triggering a render. Pending events
retain only the newest coordinate; release flushes it before ending the gesture
and restoring native resolution. Discarded packets never update VTK's last-event
position, so camera motion retains the full pointer displacement. This reduces
redundant rendering under queued input; X11/network latency still needs
measurement in the user's session.


## DICOM converter update and profiling (2026-10-16)

The Dicom2H5 submodule and both installation extras now pin
`d13d0a49e1debd08bb659d447de1a51c32cf6920`, the upstream `main` tip fetched
on 2026-10-16. The previous pin was `c6693c07f51255cd1004beb5bce9704a8f219e5c`.
The intervening commit changes Siemens single-frame velocity direction
resolution to use private encoding metadata when present; it does not change
conversion scheduling or remove discovery-time pixel decoding.

### Measurement scope

Manual comparisons used the real `dicom/SYF_880` acquisition on HP-Z8-G5 in
`autoflow311`, eight converter processes, NAS input/output, and a fresh H5
output for every run. This acquisition contains 6,623 candidate files, of
which 5,280 belong to the selected flow group (224 x 224 x 88, 15 phases).
OS/NAS caches were not flushed and became warm. These are conversion-only
experiments, not a repeated cold end-to-end pipeline benchmark. The earlier
75.72-second conversion cannot be compared directly to these warm runs as a
version-upgrade speedup.

| Variant, in execution order | Conversion wall | Filter/group | SOP-UID deduplication | Pixel decode and assembly | H5 dataset creation |
| --- | ---: | ---: | ---: | ---: | ---: |
| Latest upstream, first control | 18.97 s | 1.37 s | 2.61 s | 3.99 s | 1.66 s |
| Header-only discovery prototype | 17.91 s | 0.82 s | 2.54 s | 3.75 s | 1.55 s |
| Header-only discovery, filter chunksize 16 | 18.67 s | 0.90 s | 2.90 s | 4.07 s | 1.54 s |
| Latest upstream, last control | 19.35 s | 1.58 s | 3.03 s | 3.88 s | 1.57 s |

The prototype reads discovery metadata with `stop_before_pixels=True` and
uses Rows, Columns, NumberOfFrames and SamplesPerPixel to reproduce the array
layout used by the existing vendor classifiers. It retains all upstream
classification and velocity-calibration branches. This is an isolated
experiment: active AutoFlow continues to use the unmodified latest upstream
converter. Broader vendor and malformed/unsupported-pixel validation is
required before promoting header-only discovery to production; this single
UIH acquisition does not establish equivalence for every supported scanner.

All variants selected the same file/group inventory and produced exactly
equal values, shapes, dtypes and attributes for all 32 H5 datasets, including
magnitude, velocity, calibration, spatial ordering and metadata. The latest
upstream UIH output also exactly matches the earlier pinned-version output.
No source DICOM was modified. The retained regression suites passed:
**230 passed in 47.31 s**. An initial timing-harness type error was corrected
before the accepted runs; its failed-run records are separate from the table.

### Remaining optimization candidates

A separate cProfile run took 25.20 s, including profiling overhead. H5 file
close/flush accounted for 9.03 s, explaining much of the time outside the
function timers above. Deduplication performed another 5,280 header reads;
its normal unprofiled cost is shown in the table. Do not add cumulative
profile timers to the stage timings or use the profiled run as a speedup
control.

1. Reuse SOPInstanceUID collected during discovery for deduplication, retaining
   the same order and duplicate-selection rules. This avoids a full second
   header scan and is a larger warm-path opportunity than metadata-only
   classification by itself.
2. Use local NVMe for the preserved conversion and downstream reads, then
   overlap required NAS publication with later computation where feasible.
   If final delivery must be on NAS, include or overlap that copy; staging
   alone does not eliminate its cost. Preserve atomic no-replace publication,
   cancellation cleanup and the final converted-source provenance.
3. Retain header-only discovery as a candidate for validation on other vendors
   and uncached input. The observed warm-run saving was about one second;
   chunksize 16 did not show a convincing benefit in this comparison.
4. These costs are CPU/file-I/O work. Multiple GPUs primarily help the later
   segmentation stage rather than this converter path.

Local evidence is in `results/dicom2h5_update_20261016/comparison.json`,
`profile.txt`, `regression.log` and the isolated `metadata-filter*.patch`
files. These real validation inputs and H5 outputs are not test/demo data.


## SSH interaction and X11 presentation audit on 2026-10-17

The interactive GUI was no longer running when inspected. A fresh forwarded
X11 session to `HP-Z8-G5` (the `10.19.125.3` SSH alias) was measured using its
NVIDIA RTX PRO 6000 Blackwell, NVIDIA driver 595.99.02, `autoflow311`, and the
Moba/X client X server. VTK reported `vtkEGLRenderWindow` and the NVIDIA GPU
as its actual OpenGL renderer. This establishes hardware rendering for this
benchmark, not the renderer of the earlier closed GUI session.

The existing private default validation H5 was opened read-only. Only its first
magnitude/reference frame, embedded first-frame segmentation, and resolution
were read. The scene variants were a contour of the segmentation, magnitude
volume rendering, and orthogonal magnitude slices. This is a display benchmark,
not a cold-start analysis/segmentation pipeline benchmark.

At a measured 960 x 630 logical-pixel widget, four control groups ran in the
order full repaint, partial repaint, partial repaint, full repaint. Each group
started with the same camera and used four warm-up rotations plus 32 measured
rotations. SSH compression was disabled. Medians below pool the two groups of
each policy (64 frames); median stage values need not sum to the total median.

| Scene | Full repaint total | Partial repaint total | Full Qt presentation/sync | Partial Qt presentation/sync | Mean painted area after change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Segmentation surface | 30.86 ms | 16.56 ms | 26.22 ms | 9.91 ms | 34.2% |
| Magnitude volume | 30.40 ms | 18.74 ms | 24.19 ms | 11.07 ms | 40.5% |
| Orthogonal magnitude slices | 31.80 ms | 17.66 ms | 26.75 ms | 10.47 ms | 37.5% |

GPU render-call medians remained about 2.6-3.8 ms in the full-repaint controls.
Partial repaint adds about 2 ms of CPU frame comparison while substantially
reducing Qt/X11 presentation time in these scenes. The total measured frame
cost fell by 38-46% (1.62-1.86x throughput potential). These are render, capture,
and synchronized presentation costs, **not** measured monitor FPS or physical
mouse-to-photon latency. `QApplication.sync()` drains Qt/window-system updates;
production still coalesces pointer movement at roughly 30 ms and therefore
does not gain an uncapped frame rate from this change.

`remote_plotter.py` now compares exact RGB pixels in 64-pixel source tiles
during interaction. Touching changed tiles become a few bounding rectangles,
mapped conservatively to logical widget coordinates. Qt's backing store keeps
unchanged areas. Many tiny rectangles performed inconsistently, so components
are grouped before painting. Changed capture/widget dimensions and stationary
frames force a full update; unchanged frames need no new paint. A mostly
changing image falls back to a full repaint. Only the interactive previous
frame is retained, and release restores the existing native-size/full-quality
path. No analysis arrays, picking coordinates, or video exports are modified.

Exploratory SSH-compression runs improved the sparse surface but slowed the
volume; selecting AES-GCM or converting Qt images to RGB32 did not establish a
consistent improvement. Neither is forced by AutoFlow. TurboVNC/VirtualGL,
NoMachine, and other remote-desktop alternatives are deployment candidates,
not measured results from this audit.

Reproduce from the repository root in a forwarded SSH session:

```bash
conda activate autoflow311
QT_QPA_PLATFORM=xcb python tools/benchmark_remote_rendering.py \
  --input /nas-data2/ryy/CMR4DFlow2026/Segdata/qingtian_h5/DV/DV_heart_2026.03.01_B-01220711V013.h5 \
  --output results/x11_rendering_20261017/compare_x11.json \
  --sizes 960x630 --frames 32 --compare
```

Use `QT_QPA_PLATFORM=offscreen` for a server-only control. This helper currently
expects the validation file's legacy `img_complex`, `segmask`, and `Resolution`
datasets. It records actual widget and render sizes; a window manager can
constrain a requested size. Source owners are `ui/remote_plotter.py`,
`ui/interaction_rendering.py`, and `tools/benchmark_remote_rendering.py`.
The `--full-repaint` control bypasses the RGB comparison, and `--compare`
alternates controls within one SSH connection to reduce connection variability.

The retained smoke/phantom suite passed: **263 passed in 52.48 s**. The added
phantom checks compose partial updates with Qt and compare every output pixel
against a full repaint at native size, non-integer scaling, downscaling, and
device-pixel ratio 2, including erasure of old geometry and separate markers.
Existing drag coalescing and quality/coordinate restoration checks also pass.
Evidence is in local `results/x11_rendering_20261017/compare_x11.json`,
`summary.json`, `before_offscreen.json`, and `regression.log`; the exploratory
run files are separate from the accepted comparison above.

Documentation builds normally. Strict MkDocs validation still reports the
pre-existing unlisted page `en/developer/pipeline-audit-2026-10-07.md` in the
repository navigation; this audit did not change that navigation. Logs are
`results/x11_rendering_20261017/docs_build.log` and `docs_build_nonstrict.log`.
