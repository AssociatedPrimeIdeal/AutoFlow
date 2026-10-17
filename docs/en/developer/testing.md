# Testing

## Recommended Environment
Use:

```bash
conda activate autoflow311
python -m pytest tests/test_smoke_phantoms.py tests/test_pressure_gradient_phantom.py -q
```

## Supported Automated Suite

| Area | Main tests |
| --- | --- |
| smoke regression, including GUI input reload calibration, cancellation and downstream reset | `tests/test_smoke_phantoms.py` |
| pressure truth, reconstruction units/gauges, derivative support, and phantom regression | `tests/test_pressure_gradient_phantom.py` |
| analytic WSS vector truth, interpolation, invalid sampling, and geometry reuse | `tests/test_smoke_phantoms.py` |

## Expectations
- keep the automated suite limited to smoke and phantom regression coverage
- do not require a new targeted pytest file for every code change
- use manual verification and documentation updates when behavior changes outside the retained regression suite

## Useful Commands

```bash
conda activate autoflow311
python -m pytest tests/test_smoke_phantoms.py -q
python -m pytest tests/test_pressure_gradient_phantom.py -q
```

## Documentation checks

```bash
mkdocs build --strict
```

Review parameter tables against the merged configuration bundle, public dataclass fields and argparse declarations whenever a control changes. Every feature page must retain the eleven-section template and explain GUI, CLI and Python use or explicit unavailable status. Update the code-owner map for new modules. `mkdocs build --strict` checks site structure and links.

## Verify code provenance

Use `python -m pytest` from the repository root. A bare `pytest` command can resolve a stale installed AutoFlow package instead of this checkout. Check `autoflow.algorithms.metrics.__file__` when results contradict the source or phantom truth. Development installs should use `python -m pip install -e .` in the activated environment.

## External validation case replacements

The private validation dataset is managed under
`/nas-data2/ryy/CMR4DFlow2026/Segdata`, separately from this repository's
smoke and phantom suite. Its test selection has two synchronized manifests:
`data4seg_test/test_case_manifest.csv` and
`scripts/datastat/test_case_manifest.csv`. The AutoFlow H5 exporter reads the
second one. Each manifest row includes the source H5, source Mean label and
test Mean label paths.

`scripts/datastat/create_data4seg_test_set.py` applies the reviewed overrides
from `apply_test_set_replacements.py` after the fixed-seed draw. Rebuilding
therefore retains previous substitutions, including Center005 brain case
T23-0008 to T23-0003, without changing center/vendor quotas. The replacement
helper accepts `--dry-run`, recognizes already-applied substitutions, backs
up changed manifests and archives displaced test directories. Copying
reviewed Mean labels back into source `data4seg` now requires the explicit
`--sync-reviewed-means` option.

| Flag | Type | Default | Where configured | Effect | Code owner |
| --- | --- | --- | --- | --- | --- |
| `--dry-run` | bool | false | replacement helper CLI | Validate counts and sources without publishing the replacement. | external `scripts/datastat/apply_test_set_replacements.py` |
| `--sync-reviewed-means` | bool | false | replacement helper CLI | Copy reviewed Mean labels from the test copy back to source data. | external `scripts/datastat/apply_test_set_replacements.py` |

For an actual substitution, stage the complete new test directory, export
its strict H5 with the matching 4-D `seg_true`, and regenerate its CVI DICOM
before publishing either manifest. Validate frame count, VENC, RR,
segmentation/image alignment and DICOM series/UID consistency. Update the
affected rows in export reports, archive the old active exports/results,
and keep the original acquisition and historical reports. The 2026-10-16
replacement is recorded in
`data4seg_test/_case_replacement_20261016_T23-0003`.

The external `scripts/Case2Dicom/regenerate_segtrue_testset.py` regeneration
path reads positive TE/TR values from the strict H5's source CSV and uses
them for DICOM EchoTime/RepetitionTime instead of the v9 compatibility
template's constants. Pixel conversion, velocity signs and encoding are
unchanged. Validate the source timing together with RR, VENC and frame count.

A replacement of the software-comparison cohort does not change existing
trained weights. Independent model evaluation requires patient-level
exclusion from training and validation, including all augmented samples
and time frames, before training the model used for that evaluation.

## Dicom2H5 integration checks

The retained smoke suite checks backend/config selection, duplicate-root collection, multi-group loading, pipeline compatibility, source/destination preservation, invalid calibration and temporary cleanup without optional converter dependencies in base CI.

Manual verification with the pinned submodule converted 24 synthetic GE single-frame DICOM files into one native H5. AutoFlow loaded flow shape (4,5,3,2,3), 2 mm spacing, RR 1000 ms, and canonical component means [-10,20,-30] cm/s from the known GE polarities. TKE remained unavailable. The public CLI completed conversion and loading; absent segmentation was reported and vessel steps skipped.

On 2026-10-16 the offscreen GUI conversion worker and subsequent normal input
loader also completed a local Siemens Vida Enhanced MR head acquisition:
320 source files, magnitude `(352, 396, 80, 15)`, matching three-component flow,
0.511364 × 0.511364 × 1 mm spacing, RR 853 ms and VENC 120 cm/s. Conversion
followed by GUI loading took 53.62 s in that run, produced a 2.68 GB H5, cleared
the conversion thread and reported no load errors. Source sizes/mtimes were
unchanged. This checks a local acquisition, with chooser return values supplied
by the verification harness; it does not time X11 transfer or exercise the
user's desktop file-dialog clicks. Keep this private input outside tests/demo
data. Conversion cleanup joins the finished thread before deleting its Qt
wrapper; the retained GUI phantom covers this lifecycle.

An offscreen Qt check exercised the new GUI menu and real background conversion worker with that acquisition. It verified output/case handoff and continuing GUI timer events during conversion. The [worked example](../user/demo-case.md) documents a separate real GUI walkthrough on the supplied DV H5.

## Correction regression checks

Smoke coverage checks GUI loading/applying saved corr for normalized, complex, real-phase and dual-VENC H5; loading without valid caches never fits; repeated GUI correction runs use original source phases, replace H5 caches and never accumulate subtraction. Loading progress reports cache reads and application, with low/high VENC labels. It also checks Correction Run All order, method-dependent mask defaults, unavailable-segmask rejection, display-only noise masking and preservation of downstream artifacts after background correction, unwrap and revert. Run GUI checks in `autoflow311`: verify Input & QC → Correction → Segmentation, default LAP4D/none, method-specific mask choices and segmask availability after import/generation. The new controls were inspected with the existing software-rendering adapter on a virtual display; this does not establish clinical masking accuracy or GPU rendering performance.

Smoke/phantom checks also cover absent PC-MRA before explicit generation, PC-MRA generation last in the group, stored-image persistence and phase actor reuse, excluded-region surface bounds, and available-only Content entries with stable selection when results appear/disappear.

## Task and geometry smoke coverage

The retained smoke file covers frame-specific plane ROI sampling, static slice reuse, serial/process derived equivalence, segmentation topology retention/invalidation, modal input locking and cancellation without partial workspace publication. Qt checks use `QT_QPA_PLATFORM=offscreen` and skip only when the optional PySide6 runtime is absent. Real-data speed comparisons and rendering-process checks are recorded in [Performance](performance.md); no private validation data is copied into tests.

Slice-streamline phantoms in the same smoke file check anisotropic straight
flow, disconnected masks, vortex direction/radius, empty and invalid fields,
and exact slab crossings even when both segment endpoints are outside. Qt
checks cover simultaneous cyan planar/orange projected modes in axis and
oblique layouts, nonzero source origin, phase-dependent masks, cache/item
reuse, opacity, unavailable 3D trajectories and reset. For manual review,
enable both overlays on Magnitude, select/clear a plane, move slices, change
slab thickness, and play/pause the timeline. Verify direction arrows, preserved
scalar window/level and no joining of separate projected pieces.

Unified display smoke checks also cover gradient versus uniform segment colors,
separate legends, per-view guides, label filtering/boundaries, fixed limits over
time, dock updates and `render_settings.viewer_display` workspace round trips.
Content phantoms cover signed/unsigned fields in volume, maximum-intensity and
orthogonal-plane modes, anisotropic/nonzero-origin geometry, VTK axis-drag
events with display mirroring, phase actor reuse, missing/static data, Browser
reuse/hide/delete cleanup, workspace persistence and automatic PC-MRA handoff.
The Content GUI regression also opens two different H5 files using
`_load_selected_input_case` and clicks the actual render button after each
workspace replacement. Filling arrays into the initial workspace alone does
not cover input-load lifecycle or stale source references.
Cut phantoms compare direct orthogonal sampling with independent trilinear
interpolation at nonzero origins and anisotropic spacings. GUI checks cover
single-axis activation directly from Volume mode, the three-cut shortcut,
and actual mouse drags with preview/final-quality restoration for smart and
fixed-point volume mappers. Large SSH-widget checks verify preview pixel caps
and pointer-coordinate scaling without changing source arrays.
An occlusion phantom renders an opaque cut in front of a flow line, then checks
that selected-layer priority reveals the line and disabling priority restores
the occlusion. It also checks actor reuse, multi-selection camera bounds,
foreground legends and deletion/reset cleanup. A Qt mouse-event burst verifies
coalescing and the final release coordinate. DICOM GUI checks cover cancelled
destinations, existing-file feedback, worker-thread execution and visible stage
progress before the converted case is applied.

## Algorithm module refactors

The retained smoke suite checks default Dicom2H5 directory dispatch, legacy native-setting migration, preserved ambiguous multi-group conversions and absence of TKE for magnitude/velocity input. H5 and segmentation package re-exports preserve imports; helper instrumentation patches the implementation call site. Existing geometry/progress phantoms require one parent-thread callback per completed plane even when worker markers arrive together.

### Phase-unwrapping refactors

Keep the existing package imports and compare phase, velocity, wrap counts, masks and statistics against a small synthetic baseline. CPU traditional solvers and Torch operations on CPU can validate structural moves without a CUDA run; lightweight PUDIP/GUST stand-ins validate the adapter contracts without full training. These checks do not establish real learned-backend quality or CUDA performance. Full backend/device runs are manual checks, and no separate targeted pytest file is required.

The traditional CPU implementation is merged into the same phase-unwrapping owners as its Torch paths. Import low-level methods from `phase_unwrapping.cpu` and helpers from their algorithm modules; no standalone traditional package or compatibility forwards remain. Compare `lap3D`, `lap4D`, NPRS, graph-cut and local-gradient modes through `unwrap_data`, and verify Fourier pad/crop/shift helpers after removing duplicate definitions. CPU algorithm functions should retain their original bodies and low-level signatures.
