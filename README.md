# AutoFlow
The GIF below showcases a demo case processed by **AutoFlow** on the demo data, highlighting a series of automatically generated results, including:

- automated plane placement
- streamline visualization
- wall shear stress (WSS)
- turbulent kinetic energy (TKE)

![Demo](https://github.com/user-attachments/assets/e2c17a9e-6a47-4f85-ba0d-c35b622802b1)

## Installation

Clone the repo and install inside your environment:

```bash
git clone https://github.com/AssociatedPrimeIdeal/AutoFlow.git
cd AutoFlow
pip install .
```

The base install is intended for library usage and non-GUI batch processing.

If you also want the desktop GUI:

```bash
git clone https://github.com/AssociatedPrimeIdeal/AutoFlow.git
cd AutoFlow
pip install ".[gui]"
```

## Running The CLI

You can also run the non-GUI pipeline from the command line:

`autoflow-run` accepts:

- legacy / normalized H5 inputs
- a DICOM directory

Video outputs are disabled by default. Pass the specific `--*-video` flags you want to generate.

Background phase correction is opt-in for both H5 and DICOM inputs:

- CLI / library batch mode keeps BGC disabled by default.
- Pass `--bgc` on the CLI to enable it for a run.
- Optional CLI tuning flags are `--bgc-fit-order` and `--bgc-threshold`.
- In the GUI, AutoFlow asks whether to enable BGC each time you load an H5 file or DICOM case.

General usage:

```bash
autoflow-run <input-path>  \
  --output-dir <output-dir> \
  [options]
```

When `<input-path>` is a DICOM directory, AutoFlow scans it recursively. If multiple 4D-flow cases are found, batch mode processes each case separately and writes each case to its own output subdirectory.

Legacy H5 example:

```bash
autoflow-run ./data/demo_data.h5 \
  --output-dir ./results/demo \
```

DICOM directory example:

```bash
autoflow-run /path/to/dicom_root \
  --output-dir ./results/dicom_batch \
```

with background phase correction enabled:

```bash
autoflow-run /path/to/dicom_root \
  --output-dir ./results/dicom_batch \
  --bgc \
  --bgc-fit-order 3 \
  --bgc-threshold 0.1 \
```

Example with evenly spaced planes and rotating dynamic videos:

```bash
autoflow-run ./data/phantom_S.h5 ./data/phantom_U.h5 ./data/phantom_Y.h5 \
  --output-dir ./results/demo \
  --plane-by-distance \
  --cross-section-dist 15 \
```

## Running The Library

See `./library_demo.ipynb` for a packaged-library workflow.

Example with rotating dynamic videos:

```python
from autoflow import AutoFlowConfig, run_batch

config = AutoFlowConfig(
    inputs=["./data/demo_data.h5"],
    output_dir="./results/demo",
    use_center_plane=True,
    cross_section_dist=15.0,
    start_dist=5.0,
    end_dist=0.0,
    skip_derived=False,
    use_multithread=True,
    make_plane_video=False,
    make_wss_video=True,
    make_streamlines_video=True,
    make_tke_video=True,
    rotate_dynamic_video=True,
    dynamic_rotation_frames=180,
    dynamic_rotation_elevation_deg=10.0,
    dynamic_time_repeat=3,
    camera_view="right",
    camera_distance_scale=1.5,
    add_path_idx=False,
)

results, case_out = run_batch(config)
```

## Running The GUI

After `pip install ".[gui]"`:

```bash
autoflow-gui
```

![app](https://github.com/user-attachments/assets/2a668f0c-f98d-4168-9e84-79fa3949adb2)
1. Use **File → Open H5** for legacy / normalized HDF5 input.
2. Use **File → Import DICOM Directory** for direct DICOM input. AutoFlow scans the selected directory recursively, lists the detected 4D-flow cases, and then imports the one you choose.
3. After you choose an H5 file or DICOM case, AutoFlow asks whether to enable background phase correction for that load.
4. The **Input / Background Correction** panel shows the current resolution, venc, and detected direction labels; you can edit these values before running downstream steps if the DICOM readout needs manual adjustment.
5. Legacy HDF5 input can still contain `img_complex`, `segmask`, `Resolution`, `VENC`, `RR`, `SpatialOrder`, and `VENCOrder`.
6. Use the **Steps** panel to run the pipeline sequentially or click **Run All**.
7. Use **Edit Skeleton** / **Edit Graph** for interactive editing:
   - Click a node to select it; drag the sphere widget to move it.
   - Press `Delete` or `Backspace` to remove the selected node or edge.
   - In graph edit mode, press `E` to toggle edge mode. With edge mode on, click two nodes sequentially to add/remove an edge between them. Click an edge directly to select it for deletion.
   - Press `Escape` to cancel edits; click the step button again to apply.
8. Adjust parameters in the bottom panels before running each step.
9. Use the **Timeline** slider to scrub through time frames.
10. The **Ortho Viewer** on the right shows reformatted cross-sectional images for the selected plane.

## Input Formats

### Direct DICOM

AutoFlow can now read 4D-flow DICOM directly without converting to H5 first.

- Directory scan reads DICOM headers first and only loads pixel data for the selected / active case.
- Spatial orientation is inferred from `ImageOrientationPatient` plus slice ordering from `ImagePositionPatient` when available.
- Velocity component direction is inferred in this order:
  - standard `MRVelocityEncodingSequence / VelocityEncodingDirection`
  - explicit anatomical labels in DICOM text fields such as `RL`, `LR`, `AP`, `PA`, `FH`, `HF`
  - `RO / PE / SS` style labels combined with `InPlanePhaseEncodingDirection`
- The currently supported direct-DICOM layouts are still limited. Different vendors, scanner models, and software versions often write substantially different 4D-flow DICOM variants, so AutoFlow cannot currently adapt to every machine's DICOM output automatically.
- Manual verification and occasional parameter adjustment are still expected for direct DICOM input.


### Legacy HDF5

Input HDF5 file must contain:

| Dataset      | Shape                           | Description                                      |
|--------------|----------------------------------|--------------------------------------------------|
| `img_complex`| `(X, Y, Z, T, 4)`              | Complex PC-MRI images: ref + 3 velocity encodes  |
| `segmask`    | `(X, Y, Z)` or `(X, Y, Z, T)`  | Segmentation mask                                |
| `Resolution` | `(3,)`                         | Voxel spacing (mm)                               |
| `VENC`       | `(3,)` or `(1,)`               | Velocity encoding (cm/s)                         |
| `RR`         | scalar                         | R-R interval (ms)                                |
| `SpatialOrder` | `(3,)`                       | Spatial axis labels                              |
| `VENCOrder`  | `(3,)`                         | Encoding direction labels                        |

## Output

Results are saved under the chosen output directory.

- CLI batch mode creates one subdirectory per case, using the H5 stem or detected DICOM case name.
- GUI DICOM import defaults to `<selected_dicom_root>/autoflow_<case_name>`.

- `planes.json` — plane geometry and per-plane metrics
- `plane_metrics.json` — detailed time-resolved metrics (see below)
- `plane_qc.json` — internal consistency quality control

## Plane Metrics

Each measurement plane exports a set of time-resolved or cycle aggregated quantities to `plane_metrics.json`.
### Output fields
 
Time-series fields have length `T` (cardiac phases); cycle aggregates are scalars.
 
**Geometry / identity**
 
| Field | Description |
|---|---|
| `center`, `normal` | plane center (mm) and unit normal |
| `label` | branch label (0 = whole vessel) |
| `path_index` | which branch this plane belongs to |
| `distance` | arc-length position along the branch (mm) |
 
**Per-timestep signals**
 
| Field | Unit | Description |
|---|---|---|
| `area_mm2` | mm² | lumen area at this plane |
| `flowrate_mL_s` | mL/s | signed flow along the raw plane normal (backward-compatible) |
| `meanv_cm_s_t` | cm/s | area-weighted mean velocity along the raw plane normal |
| `flowrate_signed_mL_s` | mL/s | `flowrate_mL_s × forward_sign`; sign is now "+ = along the branch's forward direction", stable across planes on curved branches |
| `meanv_signed_cm_s_t` | cm/s | `meanv_cm_s_t × forward_sign`; same sign convention as above |
| `flowrate_forward_mL_s` | mL/s | forward-only flowrate, ≥ 0 |
| `flowrate_reverse_mL_s` | mL/s | reverse-only flowrate, reported as a positive magnitude |
| `meanv_forward_cm_s_t` | cm/s | forward-only mean velocity |
| `meanv_reverse_cm_s_t` | cm/s | reverse-only mean velocity magnitude |
 
**Cycle aggregates**
 
| Field | Unit | Description |
|---|---|---|
| `peakv_cm_s` | cm/s | peak \|velocity\| over the cycle (backward-compatible) |
| `netflow_mL_beat` | mL/beat | \|mean flowrate\| × RR / 1000 (magnitude, backward-compatible) |
| `meanv_cm_s` | cm/s | cycle-averaged mean velocity along the raw normal |
| `meanv_signed_cm_s` | cm/s | `meanv_cm_s × forward_sign`; recommended for display — stable sign on curved branches |
| `netflow_forward_mL_beat` | mL/beat | forward volume per cardiac cycle |
| `netflow_reverse_mL_beat` | mL/beat | reverse volume per cardiac cycle |
| `net_netflow_signed_mL_beat` | mL/beat | forward − reverse. Positive: net flow agrees with the branch direction. Negative: the plane's net flow runs counter to the branch direction |
| `reflux_fraction` | ratio | `reverse / forward`; flags regurgitant or retrograde flow |
| `peakv_forward_cm_s` | cm/s | forward peak velocity |
| `peakv_reverse_cm_s` | cm/s | reverse peak velocity magnitude |
| `meanv_forward_cm_s` | cm/s | cycle-averaged forward mean velocity |
| `meanv_reverse_cm_s` | cm/s | cycle-averaged reverse mean velocity |
| `path_ic` | 0..1 | internal consistency of net-flow magnitudes across planes on the same branch |
 
**Diagnostics for the forward direction**
 
| Field | Description |
|---|---|
| `forward_sign` | ±1; sign applied to `normal` to point along the branch's forward direction at this plane |
| `forward_sign_source` | `"flow_tangent"` when driven by the local tangent, `"geometry_fallback"` when the tangent was too weak and the branch chord was used, `"none"` if no directional information was available (defaults to +1) |
| `local_path_tangent` | 3-vector, unit tangent of the reoriented branch at the plane center |
| `local_path_direction` | short text label (`HF+`, `LR-`, …) summarizing the local tangent direction |
| `normal_tangent_cos` | `dot(normal, local_path_tangent)`; magnitudes near 1 indicate the plane is well aligned with the centerline |
 
## References

- WSS calculation references:
  - Petersson et al., *Assessment of the Accuracy of MRI Wall Shear Stress Estimation Using Numerical Simulations*.
  - https://github.com/EdwardFerdian/wss_mri_calculator

- TKE calculation reference:
  - Dyverfeldt et al., *Quantification of intravoxel velocity standard deviation and turbulence intensity by generalizing phase-contrast MRI. Magn Reson Med 2006; 56: 850*.
