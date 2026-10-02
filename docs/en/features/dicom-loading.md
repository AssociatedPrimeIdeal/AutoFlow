# DICOM loading and preserved conversion

## Status

| Entry point | Direct native loading | Dicom2H5 conversion |
| --- | --- | --- |
| GUI | Supported: `File > Import DICOM Directory` | Supported with optional converter: `File > Import DICOM via Dicom2H5...` |
| CLI | Supported; default `--dicom-backend native` | Supported with `--dicom-backend dicom2h5` |
| Python API | Supported | Supported through `AutoFlowConfig`, `load_input_data()`, or explicit conversion |

The optional backend is the [4DFlow_Dicom2H5 project](https://github.com/AssociatedPrimeIdeal/4DFlow_Dicom2H5), an MIT-licensed submodule at `third_party/4DFlow_Dicom2H5`, pinned to commit `c6693c07f51255cd1004beb5bce9704a8f219e5c`. Installing the dependency extra uses the same revision.

## What it does

The native loader scans and previews recognized flow acquisitions, then reads DICOM directly into `LoadedCase`. Its GUI confirmation dialog permits geometry, VENC, direction, and RR overrides.

The optional converter recursively reads a DICOM directory, assembles supported acquisitions into a new normalized H5, validates its group contract and finite positive RR/spacing/VENC, and loads its groups through the existing H5 normalization path. It preserves a reusable H5 for inspection and later loading. Original DICOM files are read only.

## When to use it

Use direct loading for an existing supported acquisition, especially when reviewing or replacing scanner metadata through the native preview.

Use Dicom2H5 when you want a persistent `mag + flow` H5, or need its vendor-specific decoding route. Its upstream recognizers cover Siemens, Philips, GE, and UIH conventions; this is not a guarantee that every export from those vendors is supported. Verify your sequence's calibration and component polarity.

## Quick use

### Install the optional backend

From a source checkout:

```bash
git submodule update --init third_party/4DFlow_Dicom2H5
pip install -e third_party/4DFlow_Dicom2H5
```

Alternatively install AutoFlow's pinned dependency extra:

```bash
pip install -e ".[dicom]"
```

The `all` extra includes it. The native route does not require this extra.

### GUI

1. Choose `File > Import DICOM via Dicom2H5...`.
2. Select a DICOM directory containing one vendor's relevant acquisition files.
3. Choose a **new** H5 filename in `Save Converted H5`.
4. Wait for background conversion and validation. The progress dialog is indeterminate; closing it hides progress rather than cancelling conversion.
5. If several H5 groups were created, choose the acquisition in the H5 case-selection dialog.
6. Choose background correction when prompted and inspect the normalized calibration in `Input & QC`.
7. Import or generate segmentation in the `Segmentation` stage, then follow the [worked example](../user/demo-case.md) from mask review onward.

The new H5 remains available if you cancel case selection or decline subsequent loading. Later use `File > Open H5` to reopen it. `File > Import DICOM Directory` retains the direct native scanner and editable preview.

### CLI

```bash
autoflow-run /path/to/dicom --output-dir ./results/dicom \
  --dicom-backend native --bgc --autoseg --no-cache-write

autoflow-run /path/to/dicom --output-dir ./results/converted-analysis \
  --dicom-backend dicom2h5 --dicom-h5-dir ./results/converted-h5 \
  --bgc --autoseg --with wss,pg --no-cache-write
```

The conversion route processes all recognized H5 groups as separate batch cases. Directory arguments with `dicom2h5` mean DICOM conversion roots; explicit H5 file arguments always use normal H5 loading. To reuse an earlier conversion, pass its H5 file rather than rerunning conversion into the same destination.

`--no-cache-write` disables correction/segmentation writes to the loaded H5. It does not suppress the new converted H5 or analysis outputs.

### Python API

```python
from autoflow import AutoFlowConfig, run_batch

config = AutoFlowConfig.from_config_dir(
    "./configs",
    inputs=["/path/to/dicom"],
    output_dir="./results/dicom",
    dicom_backend="dicom2h5",
    dicom_h5_dir="./results/converted-h5",
    background_phase_correction=True,
    background_phase_write_cache=False,
    autoseg=True,
    requested_metrics=["wss", "pg"],
)
results, last_output = run_batch(config)
```

For conversion and inspection without running the pipeline:

```python
from autoflow.algorithms.dicom_conversion import convert_dicom_input
from autoflow.algorithms.dicom import load_input_data

cases = convert_dicom_input("/path/to/dicom", "./results/new-flow.h5")
for case in cases:
    print(case.source_group, case.display_name)
loaded = load_input_data(cases[0])
print(loaded.mag.shape, loaded.flow.shape, loaded.capabilities.to_dict())
```

A direct `load_input_data(directory, dicom_backend="dicom2h5", dicom_h5_dir=...)` or `run_case(directory, config=...)` requires exactly one converted group. If several groups exist, the H5 is preserved and the error directs you to select a group or use `run_batch()`.

## Inputs

| Input | Requirement | Meaning |
| --- | --- | --- |
| DICOM directory | Required for conversion | Recursive acquisition tree; magnitude plus three velocity components |
| Vendor tags and geometry | Required for recognized decoding | Direction/polarity, VENC, pixel spacing, orientation and timing |
| New H5 destination | Required for explicit conversion / GUI | Parent directories can be created; an existing file is never replaced |
| Segmentation | Optional for loading, required for vessel analysis | Import an external mask or run nnUNet after conversion |
| TKE / complex sigma | Optional | Ordinary converted velocity data do not supply these quantities |

## Parameters

| Parameter / flag | Type | Default | Where configured | Effect | Code owner |
| --- | --- | --- | --- | --- | --- |
| `dicom_backend` / `--dicom-backend` | `native` or `dicom2h5` | `native` | `configs/loader.json`, `AutoFlowConfig`, CLI | Selects directory route; GUI exposes separate explicit menu actions | `autoflow/algorithms/dicom.py` |
| `dicom_h5_dir` / `--dicom-h5-dir` | path string | empty | Loader config, API, CLI | Empty API/batch setting resolves to `output_dir/_dicom_h5`; standalone loader defaults to `./results/_dicom_h5` | `autoflow/api.py`, `autoflow/algorithms/dicom_conversion.py` |
| `dicom_read_workers` / `--dicom-read-workers` | integer | 1 | Loader config, API, CLI | Direct native readers only; 0 selects automatic threads | `autoflow/algorithms/dicom.py` |
| `DICOM2H5_MAX_WORKERS` | positive integer environment variable | 8 | Set before converter import / GUI launch | Bounds upstream conversion process pools; independent of native reader count | `third_party/4DFlow_Dicom2H5/src/dicom2h5/converter.py` |
| `dicom_directory` | path | required | `convert_dicom_input()` | Source directory to scan recursively | `autoflow/algorithms/dicom_conversion.py` |
| `output_h5` | path | required | `convert_dicom_input()`, GUI save dialog | Explicit new H5 destination | Same adapter |
| `progress_callback` | callable or null | null | `convert_dicom_input()` | Receives coarse conversion/complete messages; no per-file progress guarantee | Same adapter |

[All loader fields](../user/parameters.md#loader), [CLI parameters](../user/cli-parameters.md), and [API fields](../user/api-parameters.md) cover correction, segmentation and metadata overrides. Native `dicom_parameter_overrides` are not applied to converter output; fix conversion calibration at the source or choose the native preview route.

## Outputs

| Output | Contents |
| --- | --- |
| New converted H5 | One flow group per assembled acquisition |
| Each group's core keys | `mag: XYZT`, `flow: XYZT3` in cm/s, `RR` ms, `Resolution` mm, `VENC`, `VENCOrder`, `SpatialOrder`, `Origin` |
| Each group's geometry metadata | `ImageOrientationPatient`, `SliceDirectionLPS`, `RotationMatrix` |
| Each group's metadata subgroups | `patient`, `scanner`, `acquisition` |
| Returned cases | H5 `InputCase` objects with source-group selection and DICOM backend/source-directory/converted-path metadata |
| Analysis outputs | Normal per-case geometry, metrics and QC outputs after loading; see [Outputs](../user/outputs.md) |

Automatic destinations use `<directory-name>-<absolute-path-hash>.h5` to avoid collisions between identically named source directories. Analysis case directories include the converted H5 group name. Temporary conversions are removed on failure. Valid H5s are published atomically without replacing existing outputs.

## Limitations

The converter may skip malformed or unrelated acquisitions while producing other valid groups. Inspect conversion logs and compare expected acquisitions against the H5 groups; conversion is not an acquisition-completeness check.

Upstream group validation checks key/shape structure. AutoFlow additionally rejects missing/non-finite/non-positive RR, resolution, or VENC and non-finite origins. This still does not independently verify scanner calibration, phase polarity, or true temporal sampling.

AutoFlow's downstream geometry contract is axis aligned. Retaining an oblique `RotationMatrix` in the H5 does not add arbitrary oblique resampling to the existing loader. Review orientation and registration before transferring planes between cases.

Converted `mag + flow` lacks the complex reference/encodes used to derive sigma. AutoFlow skips unavailable TKE cleanly; velocity magnitude alone is not a turbulence measurement.

Existing destinations cause an error. Atomic publication uses a hard link within the output filesystem; choose a filesystem that supports it if publication reports a link error.

## Where to change code

- `autoflow/algorithms/dicom_conversion.py`: dependency resolution, validation, safe publication and case provenance.
- `autoflow/algorithms/dicom.py`: native scanning and backend dispatch.
- `autoflow/api.py`, `autoflow/processing.py`, `autoflow/cli.py`: batch collection, per-case routing and public controls.
- `autoflow/core/models.py`, `autoflow/core/pipeline.py`, `autoflow/config.py`: saved/configured loader settings.
- `autoflow/ui/app.py`: conversion worker, menu, destination and case selection.
- `.gitmodules`, `pyproject.toml`: pinned converter dependency and installation.
- Upstream vendor decoding belongs to the converter repository. Update the submodule revision deliberately when accepting upstream changes.

## Tests

`tests/test_smoke_phantoms.py` checks multi-group dispatch through the H5 loader, pipeline compatibility, absent TKE, config/CLI selection, source preservation, temporary cleanup, and failed/existing-destination safety. Mock conversion avoids requiring optional vendor dependencies in base CI.

Manual validation uses the pinned converter on synthetic GE single-frame DICOM, including numeric component-sign checks after AutoFlow normalization. Real vendor acquisitions still require input/QC review. The supplied worked-example case validates the H5 analysis route.

## Common problems

| Problem | Next action |
| --- | --- |
| Converter unavailable | Initialize the submodule and install it, or install the `dicom` extra in the environment running AutoFlow |
| Destination exists | Reopen that H5, or choose a new destination; conversion never overwrites it |
| No valid groups | Confirm magnitude and all three velocity encodes exist and match a supported vendor convention; inspect skipped-group logs |
| Invalid calibration | Verify DICOM metadata; use native import with explicit overrides when appropriate |
| Several groups with `run_case()` | Select an `InputCase` from the saved H5 or use `run_batch()` |
| No vessel geometry | Import or generate segmentation; conversion does not create a lumen mask |
| TKE unavailable | Expected for velocity-only converted data; request supported analyses instead |
