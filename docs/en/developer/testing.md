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
| smoke regression | `tests/test_smoke_phantoms.py` |
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

## Dicom2H5 integration checks

The retained smoke suite checks backend/config selection, duplicate-root collection, multi-group loading, pipeline compatibility, source/destination preservation, invalid calibration and temporary cleanup without optional converter dependencies in base CI.

Manual verification with the pinned submodule converted 24 synthetic GE single-frame DICOM files into one native H5. AutoFlow loaded flow shape (4,5,3,2,3), 2 mm spacing, RR 1000 ms, and canonical component means [-10,20,-30] cm/s from the known GE polarities. TKE remained unavailable. The public CLI completed conversion and loading; absent segmentation was reported and vessel steps skipped.

An offscreen Qt check exercised the new GUI menu and real background conversion worker with that acquisition. It verified output/case handoff and continuing GUI timer events during conversion. The [worked example](../user/demo-case.md) documents a separate real GUI walkthrough on the supplied DV H5.

## Correction regression checks

Smoke coverage checks GUI loading/applying saved corr for normalized, complex, real-phase and dual-VENC H5; loading without valid caches never fits; repeated GUI correction runs use original source phases, replace H5 caches and never accumulate subtraction. Loading progress reports cache reads and application, with low/high VENC labels. It also checks Correction Run All order, method-dependent mask defaults, unavailable-segmask rejection, display-only noise masking and preservation of downstream artifacts after background correction, unwrap and revert. Run GUI checks in `autoflow311`: verify Input & QC → Correction → Segmentation, default LAP4D/none, method-specific mask choices and segmask availability after import/generation. The new controls were inspected with the existing software-rendering adapter on a virtual display; this does not establish clinical masking accuracy or GPU rendering performance.

Smoke/phantom checks also cover absent PC-MRA before explicit generation, PC-MRA generation last in the group, stored-image persistence and phase actor reuse, excluded-region surface bounds, and available-only Content entries with stable selection when results appear/disappear.

## Task and geometry smoke coverage

The retained smoke file covers frame-specific plane ROI sampling, static slice reuse, serial/process derived equivalence, segmentation topology retention/invalidation, modal input locking and cancellation without partial workspace publication. Qt checks use `QT_QPA_PLATFORM=offscreen` and skip only when the optional PySide6 runtime is absent. Real-data speed comparisons and rendering-process checks are recorded in [Performance](performance.md); no private validation data is copied into tests.

## Algorithm module refactors

The retained smoke suite checks default Dicom2H5 directory dispatch, legacy native-setting migration, preserved ambiguous multi-group conversions and absence of TKE for magnitude/velocity input. H5 and segmentation package re-exports preserve imports; helper instrumentation patches the implementation call site. Existing geometry/progress phantoms require one parent-thread callback per completed plane even when worker markers arrive together.

### Phase-unwrapping refactors

Keep the existing package imports and compare phase, velocity, wrap counts, masks and statistics against a small synthetic baseline. CPU traditional solvers and Torch operations on CPU can validate structural moves without a CUDA run; lightweight PUDIP/GUST stand-ins validate the adapter contracts without full training. These checks do not establish real learned-backend quality or CUDA performance. Full backend/device runs are manual checks, and no separate targeted pytest file is required.

The traditional CPU implementation is merged into the same phase-unwrapping owners as its Torch paths. Import low-level methods from `phase_unwrapping.cpu` and helpers from their algorithm modules; no standalone traditional package or compatibility forwards remain. Compare `lap3D`, `lap4D`, NPRS, graph-cut and local-gradient modes through `unwrap_data`, and verify Fourier pad/crop/shift helpers after removing duplicate definitions. CPU algorithm functions should retain their original bodies and low-level signatures.
