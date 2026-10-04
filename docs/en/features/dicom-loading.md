# DICOM loading through Dicom2H5

## Status

| Entry point | Status | Notes |
| --- | --- | --- |
| GUI | Supported | `File > Import DICOM Directory`; requires the converter dependency |
| CLI | Supported | DICOM directories convert before analysis; all H5 groups are processed |
| Python API | Supported | `run_case()`, `run_batch()`, `load_input_data()` or explicit conversion |

The [4DFlow_Dicom2H5 project](https://github.com/AssociatedPrimeIdeal/4DFlow_Dicom2H5) is an MIT-licensed submodule at `third_party/4DFlow_Dicom2H5`, pinned to commit `c6693c07f51255cd1004beb5bce9704a8f219e5c`. The `dicom` dependency extra uses the same revision. AutoFlow no longer implements a separate native DICOM decoder.

## What it does

DICOM input follows one path: **DICOM directory → Dicom2H5 → preserved normalized H5 → AutoFlow H5 loader**. The converter recursively reads the directory and assembles supported acquisitions. AutoFlow validates the H5 group contract and finite positive RR, spacing and VENC before publishing the new file. Original DICOM files are read only.

Case discovery, normalized-array loading and downstream analysis use the same H5 path as an ordinary H5 input. Correction reruns reread the preserved H5 rather than converting the source directory again.

## When to use it

Use it when your input is a DICOM directory and you need a reusable `mag + flow` H5. The upstream recognizers cover Siemens, Philips, GE and UIH conventions; verify support and calibration for your acquisition. Use the saved H5 for later runs.

## Quick use

### Install

```bash
git submodule update --init third_party/4DFlow_Dicom2H5
pip install -e ".[dicom]"
```

Alternatively install the submodule directly with `pip install -e third_party/4DFlow_Dicom2H5`. The `all` extra includes the converter. H5-only analysis does not require it.

### GUI

1. Choose `File > Import DICOM Directory`.
2. Select the acquisition directory and a new H5 destination.
3. Wait for conversion and validation in the background worker.
4. Select an H5 group if several acquisitions were converted.
5. Load and review the case, then run Correction, segmentation and analysis as needed.

Cancelling group selection retains the converted H5. Reopen it through `File > Open H5`. The native scan/metadata-override dialog has been removed.

### CLI

```bash
autoflow-run /path/to/dicom-root --output-dir ./results/dicom-batch   --dicom-h5-dir ./results/converted-h5 --bgc --autoseg --no-cache-write
```

All converted groups become separate batch cases. Directories containing top-level H5 files are treated as H5 batches; directories without them are DICOM conversion roots. A single DICOM file is not an input: select its containing acquisition directory. Existing conversion destinations are never replaced; pass the saved H5 to reuse it.

### Python API

```python
from autoflow import AutoFlowConfig, run_batch

summaries, last_output = run_batch(AutoFlowConfig(
    inputs=["/path/to/dicom-root"],
    output_dir="./results/dicom-batch",
    dicom_h5_dir="./results/converted-h5",
))
```

For explicit conversion and group selection:

```python
from autoflow.algorithms.dicom_conversion import convert_dicom_input
from autoflow.algorithms.inputs import load_input_data

cases = convert_dicom_input("/path/to/dicom-root", "./converted-flow.h5")
loaded = load_input_data(cases[0])
```

`load_input_data(directory, dicom_h5_dir=...)` and `run_case(directory, config=...)` require exactly one H5 group. On ambiguity the converted H5 remains available; select a group or use `run_batch()`.

## Inputs

| Input | Required | Contract |
| --- | --- | --- |
| DICOM directory | yes for conversion | Acquisition tree containing magnitude and velocity components |
| Converter dependency | yes for conversion | Pinned submodule or installed `dicom` extra |
| New H5 destination | yes | Parent can be created; destination must not already exist |
| Saved H5 file | for reuse | Select an `InputCase.source_group` when necessary |

## Parameters

| Parameter or flag | Type | Default | Where configured | Effect | Code owner |
| --- | --- | --- | --- | --- | --- |
| `dicom_h5_dir` / `--dicom-h5-dir` | path string | empty | Loader config, API, CLI | API/batch resolves empty to `output_dir/_dicom_h5`; standalone loader uses `./results/_dicom_h5` | `autoflow/api.py`, `autoflow/algorithms/inputs.py` |
| `dicom_backend` / `--dicom-backend` | string | `dicom2h5` | Loader config, API, CLI | Compatibility setting; Dicom2H5 is the sole route. Legacy API/config `native` values migrate to `dicom2h5`; CLI accepts only `dicom2h5` | `autoflow/case_types.py`, `autoflow/cli.py` |
| `DICOM2H5_MAX_WORKERS` | positive integer environment variable | 8 | Set before converter import / GUI launch | Bounds upstream conversion process pools | `third_party/4DFlow_Dicom2H5/src/dicom2h5/converter.py` |
| `dicom_directory` | path | required | `convert_dicom_input()` | Source directory for upstream decoding | `autoflow/algorithms/dicom_conversion.py` |
| `output_h5` | path | required | `convert_dicom_input()` | Publish a new validated H5 | same |
| `progress_callback` | callable or None | None | `convert_dicom_input()` | Receives conversion start/completion dictionaries | same |

`dicom_read_workers`, `--dicom-read-workers` and `dicom_parameter_overrides` belonged to the removed native loader. They are no longer supported controls; old saved loader dictionaries omit them when restored. Review or correct acquisition calibration before conversion rather than applying native-reader overrides.

## Outputs

| Output | Contents |
| --- | --- |
| Converted H5 | One normalized flow group per assembled acquisition |
| Group core keys | `mag: XYZT`, `flow: XYZT3` in cm/s, `RR` ms, `Resolution` mm, `VENC`, `VENCOrder`, `SpatialOrder`, `Origin` |
| Geometry metadata | `ImageOrientationPatient`, `SliceDirectionLPS`, `RotationMatrix` |
| Metadata subgroups | `patient`, `scanner`, `acquisition` |
| Returned cases | H5 `InputCase` objects with group selection and conversion provenance |
| Analysis results | Normal geometry, metric and QC outputs after H5 loading |

Automatic filenames are `<directory-name>-<absolute-path-hash>.h5`. Temporary files are removed on failure; validated H5 files are published atomically without replacing existing destinations.

## Limitations

- Conversion can skip malformed or unrelated acquisitions while producing other valid groups. Compare the expected acquisition count against the saved H5 groups.
- Shape and calibration checks do not independently verify scanner polarity or temporal sampling.
- The downstream geometry contract is axis aligned; storing an oblique rotation matrix does not add oblique resampling.
- Converted magnitude and velocity lack the complex information needed to derive sigma. TKE stays optional and is skipped when unavailable.
- Atomic publication requires hard-link support on the output filesystem.

## Where to change code

- `autoflow/algorithms/inputs.py`: H5 discovery, directory dispatch and normalized loading.
- `autoflow/algorithms/dicom_conversion.py`: converter resolution, validation, publication and provenance.
- `autoflow/core/pipeline.py`: preserved-source loading and correction reruns.
- `autoflow/ui/app.py`, `autoflow/ui/input_dialogs.py`: conversion workflow and H5 group selection.
- `autoflow/api.py`, `autoflow/processing.py`, `autoflow/cli.py`, `autoflow/config.py`, `autoflow/core/models.py`: configured and saved input settings.
- `.gitmodules`, `pyproject.toml`: converter revision and optional installation.
- Vendor decoding belongs to the converter repository; change its pinned revision deliberately.

## Tests

`tests/test_smoke_phantoms.py` covers converter dispatch, H5 groups, optional TKE, preserved sources, temporary cleanup, existing-destination safety and configuration/CLI compatibility. Use the retained smoke/phantom suites; do not add a separate DICOM test file.

Manual vendor validation should compare known component signs and calibration with the saved H5 before analysing a real case.

## Common problems

| Problem | Next action |
| --- | --- |
| Converter unavailable | Initialize the submodule and install it, or install `.[dicom]` in the running environment |
| Destination already exists | Reopen the saved H5 or choose a new destination |
| Multiple H5 groups | Select a group in the GUI or an `InputCase`, or use `run_batch()` |
| Invalid calibration | Review DICOM metadata and upstream decoding before converting again |
| No TKE | Continue supported velocity analysis; provide measured sigma/TKE if available |
