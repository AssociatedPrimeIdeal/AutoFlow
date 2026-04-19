# AutoFlow
The GIF below showcases a demo case processed by **AutoFlow** on the demo data, highlighting a series of automatically generated results, including:

- automated plane placement
- streamline visualization
- wall shear stress (WSS)
- turbulent kinetic energy (TKE)

![Demo](https://github.com/user-attachments/assets/e2c17a9e-6a47-4f85-ba0d-c35b622802b1)

## Requirements

- Python 3.9+
- System packages for Qt5 (e.g. `libxcb-xinerama0` on Ubuntu)

## Installation

```bash
pip install numpy scipy scikit-image networkx pyvista pyvistaqt PyQt5 matplotlib h5py imageio Pillow
```

## Running Script

See `./demo.ipynb` for example workflows, including:

- a demo showing how to run the pipeline on an aortic dataset
- demo examples for three phantom datasets

## Running Software

```bash
python app.py
```
![app](https://github.com/user-attachments/assets/2a668f0c-f98d-4168-9e84-79fa3949adb2)
1. **File → Open Data** to load an HDF5 file containing `img_complex`, `segmask`, `Resolution`, `VENC`, `RR`, `SpatialOrder`, and `VENCOrder`.
2. Use the **Steps** panel to run the pipeline sequentially or click **Run All**.
3. Use **Edit Skeleton** / **Edit Graph** for interactive editing:
   - Click a node to select it; drag the sphere widget to move it.
   - Press `Delete` or `Backspace` to remove the selected node or edge.
   - In graph edit mode, press `E` to toggle edge mode. With edge mode on, click two nodes sequentially to add/remove an edge between them. Click an edge directly to select it for deletion.
   - Press `Escape` to cancel edits; click the step button again to apply.
4. Adjust parameters in the bottom panels before running each step.
5. Use the **Timeline** slider to scrub through time frames.
6. The **Ortho Viewer** on the right shows reformatted cross-sectional images for the selected plane.

## Data Format

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

During loading, the software reconstructs `mag`, `flow`, `IVSD`, and `TKE` from `img_complex`.

## Output

Results are saved next to the input file:

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