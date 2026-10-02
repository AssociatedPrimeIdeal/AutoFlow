# Current pipeline performance

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

## Reproduce the cold measurement

```bash
conda activate autoflow311
PYVISTA_OFF_SCREEN=true python tools/benchmark_pipeline.py \
  --report /tmp/autoflow-benchmark.json
```

The default input is `/nas-data2/ryy/CMR4DFlow2026/Segdata/qingtian_h5/DV/DV_heart_2026.03.01_B-01220711V013.h5`. It is validation data, not shipped demo/test data. Pass a different input as the positional argument.

| Argument | Type | Default | Where configured | Effect | Code owner |
| --- | --- | --- | --- | --- | --- |
| `input` | path | validation H5 above | benchmark CLI | Input to measure | tools/benchmark_pipeline.py |
| `--config-dir` | path | repository configs | benchmark CLI | Module bundle; cold/read-only overrides are then applied | same |
| `--output-dir` | path | fresh temporary directory | benchmark CLI | Numerical outputs; existing summary is rejected | same |
| `--report` | path | OUTPUT/benchmark.json | benchmark CLI | Timing/provenance report | same |
| `--with` | comma-separated strings | wss,pg | benchmark CLI | Optional numerical families to include | same |
| `--autoseg-folds` | string | single | benchmark CLI | Single or installed ensemble fold selection | same |
| `--single-thread` | switch | false | benchmark CLI | Disable process plane metrics; smaller automatic jobs are already serial | same |

Inspect `wall_time_sec`, `stage_times_sec`, `nested_metric_times`, GPU snapshots and cache/source flags. Compare repeated runs under comparable CPU/GPU/NAS load and unchanged checkpoints. CUDA work is asynchronous; do not interpret a Python profiler's forward-call time as pure GPU execution time.

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

```bash
PYVISTA_OFF_SCREEN=true python tools/benchmark_plane_metrics.py /path/to/case.h5 \
  --segmentation /path/to/case_auto_segmentation.nii.gz \
  --workers 4 --report /tmp/plane-comparison.json
```

The segmentation argument must be AutoFlow's generated HF/AP/RL inference NIfTI; the benchmark restores internal LR/AP/FH orientation. Serial and process results are compared with rtol=1e-6 and atol=1e-7, including NaNs and QC metadata. It alternates two serial and two process runs; its final augmented serial measurement additionally includes WSS/PG preparation and sampling.

| Argument | Type | Default | Where configured | Effect | Code owner |
| --- | --- | --- | --- | --- | --- |
| input | path | required | plane benchmark CLI | Load velocity; loading/correction are excluded from paired timers | tools/benchmark_plane_metrics.py |
| --segmentation | path | required | same | Freeze a generated mask for all runs | same |
| --workers | positive int | 4 | same | Explicit process count, bypassing automatic threshold | same |
| --report | path | required | same | Write paired timings and equivalence checks | same |

### Controlled results

| Run | Serial | Four processes | Result check |
| --- | ---: | ---: | --- |
| First pair | 30.26 s | 13.30 s | Equivalent metrics and QC |
| Second pair | 30.14 s | 10.48 s | Equivalent metrics and QC |

This is about 56–65% less time for **base plane metrics**, not the entire pipeline. Derived sampling and file exports are additional work; the serial augmented replay took 59.83 s including WSS/PG preparation. The worker policy changed without changing any numerical formula.

## Code and validation

Timer tooling: `tools/benchmark_pipeline.py`, `tools/benchmark_plane_metrics.py`. Algorithm owners: `autoflow/algorithms/segmentation.py`, `phase_correction.py`, `metrics.py`; orchestration: `autoflow/core/pipeline.py`, `autoflow/processing.py`. Numerical optimizations must preserve smoke/phantom truth and use controlled real-case comparisons. See [Testing](testing.md).
