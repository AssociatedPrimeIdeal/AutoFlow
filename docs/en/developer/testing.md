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
python tools/build_parameter_reference.py --check
mkdocs build --strict
```

The generator reads API/CLI declaration AST nodes without importing optional GUI/inference runtimes, and documents learned-method choices with their installation requirements. It checks 19 configuration modules plus all public API fields/CLI actions against descriptions and generated tables. Each functional feature page uses the eleven-section template (including all GUI/CLI/Python entry points or explicit unavailable status). The --check mode also verifies feature section order and entry-point explanations, English-only prose, and coverage of every Python module in the code-owner map. Keep configuration descriptions in `docs/en/developer/parameter-descriptions.json`, then regenerate rather than editing the generated tables.

## Verify code provenance

Use `python -m pytest` from the repository root. A bare `pytest` command can resolve a stale installed AutoFlow package instead of this checkout. Check `autoflow.algorithms.metrics.__file__` when results contradict the source or phantom truth. Development installs should use `python -m pip install -e .` in the activated environment.

## Dicom2H5 integration checks

The retained smoke suite checks backend/config selection, duplicate-root collection, multi-group loading, pipeline compatibility, source/destination preservation, invalid calibration and temporary cleanup without optional converter dependencies in base CI.

Manual verification with the pinned submodule converted 24 synthetic GE single-frame DICOM files into one native H5. AutoFlow loaded flow shape (4,5,3,2,3), 2 mm spacing, RR 1000 ms, and canonical component means [-10,20,-30] cm/s from the known GE polarities. TKE remained unavailable. The public CLI completed conversion and loading; absent segmentation was reported and vessel steps skipped.

An offscreen Qt check exercised the new GUI menu and real background conversion worker with that acquisition. It verified output/case handoff and continuing GUI timer events during conversion. The [worked example](../user/demo-case.md) documents a separate real GUI walkthrough on the supplied DV H5.
