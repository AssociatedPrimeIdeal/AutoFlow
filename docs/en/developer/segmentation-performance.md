# Segmentation and Labeler performance validation

Measured on 2026-09-30 using `autoflow311`, an NVIDIA RTX PRO 6000 Blackwell,
the GUI's standard 4D nnUNet subprocess path, and NAS-backed Labeler exchange
files. This measures automatic segmentation followed by manual label revision,
not nnUNet training. The CLI's grouped preprocessing is a different path and
was not replaced or benchmarked as the GUI baseline.

## Input and controls

The local validation input was
`/nas-data2/ryy/CMR4DFlow2026/Segdata/qingtian_h5/DV/DV_heart_2026.03.01_B-01220711V013.h5`.
Loading recomputed background correction, ignored embedded segmentation, and
disabled source cache writes. The input size and modification time were
unchanged. The normalized magnitude/velocity arrays were frozen once and then
used by both versions: spatial shape `158 x 44 x 144`, 20 phases, 2.5-mm voxels.
This removes background-correction differences from the segmentation comparison.

| Setting | Value in both versions |
| --- | --- |
| Model | Dataset7020_Aorta_4DTemporalFT, nnUNetTrainerPartBalancedTversky, nnUNetPlansIso1mm, 3d_fullres |
| Checkpoint/folds | checkpoint_best.pth, fold_all |
| Channels | 12 cycle statistics plus 25 temporal channels |
| Preprocessing/export workers | 1 / 1 |
| Sliding-window step / TTA | 0.5 / disabled |
| GUI isolation | inference in a child process; grouped_preprocessing=False |
| Labeler features | mag, flow_x, flow_y, flow_z, and time-mean PC-MRA repeated over phases |
| Exchange storage | five float32 images and one int16 mask, uncompressed NIfTI |

## Retained changes

4D input preparation now compresses 112 distinct feature maps instead of 740
repeated channels, retaining all 740 original sample filenames through hard
links. Filesystems without hard links use byte copies. The channel arrays,
affines, order, per-sample normalization/cropping, and model inference remain
unchanged. No checkpoint, precision, TTA, or temporal-context change was made.

The standard Python 4D child process also moves the unpadded input tensor to
CUDA before calling nnUNet's original padding and sliding-window code. This
keeps the same values, padding, window order, and model arithmetic while
avoiding expensive CPU padding and transfer of the padded tensor. CPU runs,
CPU-accumulation runs, and devices without enough free memory retain the
original path. The free-memory guard reserves the input, padded tensor, and
16 GiB for inference; a CUDA allocation failure retries the original path.
The helper loads from the calling AutoFlow package even when the prediction
interpreter and working directory differ. Installed nnUNet source is not edited
for these transfer optimizations; the frozen standalone predictor keeps its
existing path.

The inspected nnUNet preprocessing iterator also calls `pin_memory()` on each
tensor but discards the returned copies. The child helper disables only this
recognized implementation, removing a copy that never reaches inference.
If inspection fails or the implementation changes, it retains that iterator's
normal behavior.

Labeler export runs outside the GUI thread, with two NIfTI writers. A content
digest distinguishes unchanged images from same-shaped but changed inputs.
Image reuse and mask reuse are separate: a different active seed refreshes only
the mask; accepting the exact saved edit retains its file and label definitions.
An unchanged seed retains saved Labeler work even before applying it in AutoFlow.
Replaced masks are backed up as `segmentation.previous.<timestamp_ns>.nii`.
Legacy manifests refresh once and preserve differing saved masks before migration.

## Timings

### Input-file and Labeler changes

| Stage | Before | After | Interpretation |
| --- | ---: | ---: | --- |
| 4D input preparation through inference launch | 21.62 s | 4.53 s | Same logical channel files; fewer encodings |
| Complete automatic segmentation, controlled CUDA comparison | 156.69 s | 139.05 s | 17.64 s saved, 11.3% |
| First Labeler feature export, including PC-MRA | about 5.3 s | about 4.8 s | NAS measurements; roughly 0.5 s saved |
| Labeler image/mask load | 2.4-5.8 s observed | same implementation | NAS/cache variability; no attributed speedup |
| Labeler save | about 0.45 s | same operation | Identical edit and output |
| AutoFlow apply of generated labels | 0.68 s | same operation | Hidden GUI with EGL rendering |
| AutoFlow import of edited labels, including file read | 0.57 s | same operation | Exact imported labels |
| Optional source-H5 label cache write | 0.14 s | same operation | Scratch H5 on NAS; original input untouched |
| Reopen exchange preparation after accepting an edit | not compared | 0.33-0.49 s | Zero image/mask files rewritten |
| Cold input load plus correction | 39.49 s | same operation | Outside the frozen-input segmentation comparison |

Adding the measured stages, with about 7 seconds of unchanged GUI/load/save
work, gives approximately **169 -> 151 seconds** from loaded input through
Labeler save and AutoFlow re-import: about **18 seconds, or 11%, less machine
waiting**. Including the same cold load gives approximately **209 -> 191
seconds**, about **9%** saved. These are stage-sum estimates, not a stopwatch
measurement of an entire interactive session. Human revision, confirmation,
and visible native-window rendering time are excluded. NAS timing fluctuates;
do not attribute unrelated loading fluctuations to the optimization.

### CPU-copy bottleneck and fixed-checkpoint comparison

A subsequent child-process profile measured **52.86 s in CPU tensor padding**
and **11.45 s in discarded pinned-memory copies**. These are components of a
profiled run, not independent end-to-end savings. The 20-phase model prepares
37 channels per phase, resamples the channels to the model's 1-mm grid, and
executes 18 sliding windows per phase: 360 network calls. CUDA calls are
asynchronous, so CPU profiler time inside network `forward` is not GPU wall time.

The shared checkpoint was updated externally during investigation, and one
load encountered an incomplete archive. New comparisons therefore used a
validated read-only snapshot of `dataset.json`, `plans.json`, and
`fold_all/checkpoint_best.pth`, with checkpoint SHA-256
`3e12551d2fafff2372c49fc8a58549b360a27a8e6cb445911f02756264df7b56`.
AutoFlow did not change the shared model. The earlier timings above use the
earlier checkpoint and must not be combined with this second pair as one
continuous speedup measurement.

| Standard 4D segmentation, frozen inputs/checkpoint | Time | Result |
| --- | ---: | --- |
| Before transfer optimization, distinct-map preparation retained | 134.25 s | Fixed-checkpoint baseline |
| GPU-first padding prototype | 91.44 s | All 20,021,760 labels equal to baseline |
| Integrated production subprocess | 87.61 s | All 20,021,760 labels equal to baseline |

The integrated production path saves **46.64 s (34.7%)** in this controlled
pair. It changes machine waiting before opening Labeler; human editing time
is unaffected. The prototype and integrated timings differ by normal run
variation and are not evidence of another implementation improvement.

Increasing preprocessing workers to two did not establish a useful gain in
the exploratory timing. That run crossed the checkpoint update, so it is not
valid equality evidence and no worker-count change was retained.

## Result equality and reproducibility

All 740 channel files were decoded and their voxel/affine hashes compared:
**all identical**, with 740 versus 112 actual encodings.

The existing nnUNet runtime showed small differences across independent default
CUDA launches on identical frozen arrays: two original runs differed at 17
voxels; an original/candidate pair differed at 16. This prevents using an
uncontrolled pair as strict equality evidence. The runtime enables cuDNN
benchmarking; its sole responsibility for these differences was not established.

For the strict comparison, the benchmark set the following equally in both
child processes, after the predictor constructor:

```python
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)
```

It also set `CUBLAS_WORKSPACE_CONFIG=:4096:8`. **All 20,021,760 output labels
were exactly equal, with zero differing voxels.** These controls were applied
only in the verification harness. Production inference defaults were not
silently changed; independent CUDA reruns retain the existing reproducibility
limitation.

The fixed-checkpoint transfer comparison used the same controls on both sides.
CPU and GPU padding were also compared by float32 bit patterns and inverse
slices on four shapes, including odd padding and special values; all matched.
Manual checks covered both the low-memory guard and allocation-failure fallback.

Labeler loaded all five complete feature arrays without value changes. The
final exchange check used the identical labels from the controlled inference
comparison. A
scripted voxel revision was saved through Labeler's actual `MainWindow.save_mask`
and read back in AutoFlow; the whole 4D edited label array matched exactly.
AutoFlow's actual source-apply and import handlers also preserved the expected
labels. The launch bridge was checked for five images, magnitude last, exact
active mask, and a Qt timer firing during export. The editor was not visibly
opened during these SSH checks. Hidden-window teardown emitted Qt worker
warnings after the data checks; native visible rendering was not validated.

An eager NIfTI-read candidate was rejected: six alternating checks on the same
files gave median 0.62 s for the existing mmap conversion and 0.80 s for eager
conversion, with equal arrays. Labeler's existing loader was restored.

## Code, regression checks, and evidence

- `autoflow/algorithms/segmentation/nnunet_static.py`: distinct-map encoding and standard sample links.
- `autoflow/nnunet_runtime.py`: child-only transfer optimization and original-path fallbacks.
- `autoflow/ui/labeler_exchange.py`: content digests, bounded export, independent image/mask reuse, backups.
- `autoflow/ui/app.py`: responsive export progress and launch/import integration.
- `tests/test_smoke_phantoms.py`: circular-channel values, hard-link fallback, edited-mask reuse, changed-input invalidation, legacy-workspace migration, and CPU/low-memory/allocation-failure inference fallbacks.

```bash
conda activate autoflow311
PYVISTA_OFF_SCREEN=true python -m pytest tests/test_smoke_phantoms.py tests/test_pressure_gradient_phantom.py -q
```

Run from the repository root so that validation selects this checkout rather
than a separate installed AutoFlow version. The retained suite passed **94
tests** after the transfer integration. Real-case benchmarks and verification
arrays are outside the repository in `/tmp/autoflow-seg-labeler-20260930`;
NAS exchange artifacts are in
`/nas-data/ryy_rawdata/.autoflow-benchmark-seg-labeler-20260930`.
Neither validation images nor model weights were added to the tests or repository.

## Remaining opportunities

1. Plane metrics: the prior full-pipeline profile spent substantial time in
   repeated VTK slice construction. On fixed labels, the retained support-mesh
   crop already reduced basic metrics for 108 planes and 20 phases from 24.84 s
   to 17.91 s with equal values. Further reuse must distinguish connected-region
   selection, target labels, branch support, phase masks, and ROI operations;
   sharing slices across different settings could change results.
2. Labeler return: `MainWindow._commit_segmentation_source_change` currently
   clears segmentation-dependent topology and metrics. If a revision leaves
   the majority-vote topology unchanged, a future optimization could retain
   topology/planes and recompute only metrics for affected phases, including
   any temporal aggregates. This is not implemented; correctness requires
   explicit dependency tracking and comparison against full recomputation.
3. Cold loading/correction: the measured load is 39.49 s. An exact normalized
   data cache could help repeated opens when the source and correction settings
   match; it does not reduce the first cold-start baseline. First-load changes
   need separate profiling of reads, complex conversions, and correction.

Labeler first export (about 4.8 s), save (about 0.45 s), and re-import (about
0.57 s) have much less remaining absolute cost than segmentation and plane
metrics. No additional speedup is claimed for these unimplemented opportunities.
