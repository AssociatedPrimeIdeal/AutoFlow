# Current pipeline performance

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
