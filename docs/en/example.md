# Example: Aorta H5

This example uses the Aorta case supplied for the AutoFlow documentation:

```text
/nas-data2/ryy/CMR4DFlow2026/Segdata/data4seg_test/h5s_autoflow/Aorta_Center003_GE_15T_Voyager_Exam32324-12224896-C17-0021.h5
```

The case is a legacy complex H5 with a spatial volume, cardiac phases, one magnitude reference and three complex velocity encodes. The processing command only needs the minimum input contract described in [Input & Output](input-output.md).

## Run the case from the CLI

```bash
CASE=/path/to/Aorta_Center003_GE_15T_Voyager_Exam32324-12224896-C17-0021.h5

autoflow-run "$CASE" \
  --output-dir ./results/aorta_center003 \
  --bgc \
  --autoseg \
  --with pwv,wss,pg,vortex \
  --video plane,wss,pg,streamlines
```

The command loads and normalizes the case, prepares the active segmentation, generates skeleton/graph/paths/planes, calculates plane metrics, and then runs only the requested optional metrics and videos.

## Follow the case in the GUI

```bash
autoflow-gui --config-dir ./configs
```

1. Open the H5 from `File > Open H5`.
2. Confirm metadata in `Input & QC`.
3. In `Segmentation`, choose a source and review it in orthogonal views.
4. Run `Skeleton`, `Graph/Paths`, and `Planes`; correct geometry before hemodynamic analysis.
5. Select PWV, WSS, pressure or vortex analysis in `Hemodynamics` as needed.
6. Use `Export > Export Videos...` to export plane, WSS, pressure or streamline views.

## Inspect the output tree

```text
<output-root>/<case-name>/
├── summary.json
├── quality_report.json
├── planes.json
├── planes.h5
├── plane_positions.json
├── plane_metrics.json
├── plane_qc.json
├── pwv.json                  # if PWV was requested
├── *_wss*.npz / *_wss*.h5   # if WSS was requested
├── *_pressure*.npz / *.h5   # if pressure was requested
└── *.mp4                    # if a video was requested
```

Read `summary.json` first to see what was requested and what ran. Read `quality_report.json` and `plane_qc.json` before interpreting metric values. The video on the home page is a documentation preview generated from this type of case, not a clinical report.
