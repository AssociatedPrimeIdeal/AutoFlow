# AutoFlow
- extracted planes
- streamline visualization
- wall shear stress (WSS)
- turbulent kinetic energy (TKE)

<video src="./demo.mp4" controls width="100%"></video>

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
- `plane_metrics.json` — detailed time-resolved metrics
- `plane_qc.json` — internal consistency quality control

## References

- WSS calculation references:
  - Petersson et al., *Assessment of the Accuracy of MRI Wall Shear Stress Estimation Using Numerical Simulations*.
  - https://github.com/EdwardFerdian/wss_mri_calculator

- TKE calculation reference:
  - Dyverfeldt et al., *Quantification of intravoxel velocity standard deviation and turbulence intensity by generalizing phase-contrast MRI. Magn Reson Med 2006; 56: 850*.