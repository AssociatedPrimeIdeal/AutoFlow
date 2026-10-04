# Scientific References

This page collects the literature behind AutoFlow's calculations and identifies where the implementation differs from a published method. A citation is not a claim that AutoFlow reproduces the complete paper, uses its parameters, or has equivalent clinical validation. Use the feature guides for current defaults, units, support masks and limitations.

## Feature-to-reference map

| Feature | References | Relationship to AutoFlow | Main code |
| --- | --- | --- | --- |
| PG and relative pressure | [PG-1 to PG-5](#pressure-gradient-and-relative-pressure) | Navier–Stokes gradients, pressure Poisson and Stokes estimators; Cartesian adaptations | `autoflow/algorithms/metrics/pressure.py` |
| WSS | [WSS-1 and WSS-2](#wall-shear-stress) | Tangential velocity derivatives and surface smoothing; not the original B-spline estimator | `autoflow/algorithms/metrics/wss.py` |
| Centerline and paths | [CL-1 and CL-2](#centerline-skeleton-and-paths) | Lee 3-D thinning and Savitzky–Golay smoothing; graph cleanup/grouping are AutoFlow logic | `autoflow/algorithms/skeleton.py`, `autoflow/algorithms/graph.py`, `autoflow/algorithms/paths.py` |
| Noise mask and PC-MRA | [PC-1 and PC-2](#noise-masking-and-pc-mra) | Magnitude/temporal-SD screening and magnitude-weighted speed; thresholds are project settings | `autoflow/algorithms/noise_removal.py`, `autoflow/core/pipeline.py` |
| Background phase correction | [BGC-1](#background-phase-correction) | Weighted regularized least squares with automatic rejection of temporally invariant outliers | `autoflow/algorithms/phase_correction.py` |
| LAP4D phase unwrapping | [PU-1](#phase-unwrapping) | Four-dimensional single-step Laplacian method | `autoflow/algorithms/phase_unwrapping/laplacian.py`, `autoflow/algorithms/phase_unwrapping/engine.py` |
| Model segmentation | [SEG-1](#segmentation) | nnU-Net framework; local models/4-D integration need separate validation | `autoflow/algorithms/segmentation/nnunet_static.py` |
| TKE | [TKE-1](#turbulent-kinetic-energy) | Intravoxel dispersion from magnitude attenuation, not cardiac-cycle speed SD | `autoflow/algorithms/data/h5_loader.py`, `autoflow/algorithms/metrics/tke.py` |
| PWV | [PWV-1](#pulse-wave-velocity) | MRI transit-time methodology review, not an exact implementation specification | `autoflow/algorithms/pwv.py` |
| Vortex kinematics | [V-1 and V-2](#vortex-kinematics) | Velocity-gradient vortex identifiers and swirling strength | `autoflow/algorithms/metrics/vortex.py` |
| Acquisition, flow metrics and QC | [CMR-1 and CMR-2](#general-4d-flow-guidance) | Consensus guidance for acquisition, retrospective analysis and validation | `autoflow/algorithms/data/h5_loader.py`, `autoflow/algorithms/inputs.py`, `autoflow/quality.py` |
| Numerical and imaging libraries | [SW-1 and SW-2](#software-foundations) | Software acknowledgements, separate from physiological validation | SciPy and scikit-image |

## Pressure gradient and relative pressure

- **PG-1.** Ebbers T, Farnebäck G. *Improving computation of cardiovascular relative pressure fields from velocity MRI.* Journal of Magnetic Resonance Imaging. 2009;30(1). [DOI: 10.1002/jmri.21775](https://doi.org/10.1002/jmri.21775). Basis for structure-defined pressure Poisson reconstruction and multigrid solution on the vessel domain.
- **PG-2.** Švihlová H, Hron J, Málek J, Rajagopal KR, Rajagopal K. *Determination of pressure data from velocity data with a view toward its application in cardiovascular mechanics. Part 1. Theoretical considerations.* International Journal of Engineering Science. 2016;105:108–127. [DOI: 10.1016/j.ijengsci.2015.11.002](https://doi.org/10.1016/j.ijengsci.2015.11.002). Original Stokes-based pressure-estimation formulation using an auxiliary incompressible field.
- **PG-3.** Bertoglio C, Nuñez R, Galarce F, Nordsletten D, Osses A. *Relative pressure estimation from velocity measurements in blood flows: State-of-the-art and new approaches.* International Journal for Numerical Methods in Biomedical Engineering. 2018;34(2):e2925. [DOI: 10.1002/cnm.2925](https://doi.org/10.1002/cnm.2925). Reviews estimators and develops formulations including Stokes-based reconstruction.
- **PG-4.** Nolte D, Urbina J, Sotelo J, et al. *Validation of 4D Flow based relative pressure maps in aortic flows.* Medical Image Analysis. 2021;74:102195. [DOI: 10.1016/j.media.2021.102195](https://doi.org/10.1016/j.media.2021.102195). Compares PPE/STE with catheter measurements and investigates resolution, segmentation and noise sensitivity.
- **PG-5.** Pacheco DRQ. *On the numerical treatment of viscous and convective effects in relative pressure reconstruction methods.* International Journal for Numerical Methods in Biomedical Engineering. 2022;38(3):e3562. [DOI: 10.1002/cnm.3562](https://doi.org/10.1002/cnm.3562); [open full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC9286393/). Discusses why equivalent viscous/convective formulations differ for noisy sampled velocity.

AutoFlow computes `grad(p) = -rho*(dv/dt + (v dot grad)v) + mu*laplacian(v)` in SI units. `ppe` uses face-gradient normal equations on the valid Cartesian voxel domain; the legacy `least_squares` name is an alias. `ste` uses an auxiliary incompressible velocity and pressure system on a staggered MAC grid, not a verbatim finite-element implementation from the papers. Temporal differentiation, support erosion, component-wise gauges and solvers are documented in [Pressure](../features/pressure.md).

Centerline pressure drops sample the reconstructed pressure field. They are not a separate Bernoulli estimator; disconnected components cannot share an absolute pressure reference. Published estimator validation does not validate AutoFlow's implementation or a particular acquisition.

## Wall shear stress

- **WSS-1.** Stalder AF, Russe MF, Frydrychowicz A, Bock J, Hennig J, Markl M. *Quantitative 2D and 3D phase contrast MRI: optimized analysis of blood flow and vessel wall parameters.* Magnetic Resonance in Medicine. 2008;60(5). [DOI: 10.1002/mrm.21778](https://doi.org/10.1002/mrm.21778). Foundational MRI vector-WSS work; also discusses flow quantification and resolution-driven WSS underestimation.
- **WSS-2.** Taubin G. *A signal processing approach to fair surface design.* Proceedings of SIGGRAPH '95. 1995:351–358. [DOI: 10.1145/218380.218473](https://doi.org/10.1145/218380.218473). Basis for the non-shrinking surface-smoothing family used by WSS geometry preparation.

AutoFlow estimates `tau_wall = mu * d(v_tangential)/dn` with inward-normal samples at `0`, `h` and `2h`, a linear/quadratic derivative, and optional no-slip. It uses VTK-interpolated probes, not Stalder's B-spline/Green's-theorem method. Taubin iterations, probe distance, segmentation-derived normals and invalid-probe handling are implementation settings, not recommendations from WSS-1. See [WSS](../features/wss.md).

## Centerline, skeleton and paths

- **CL-1.** Lee TC, Kashyap RL, Chu CN. *Building Skeleton Models via 3-D Medial Surface Axis Thinning Algorithms.* CVGIP: Graphical Models and Image Processing. 1994;56(6):462–478. [DOI: 10.1006/cgip.1994.1042](https://doi.org/10.1006/cgip.1994.1042). Underlying 3-D thinning algorithm used by scikit-image's `skeletonize` for volumetric input.
- **CL-2.** Savitzky A, Golay MJE. *Smoothing and Differentiation of Data by Simplified Least Squares Procedures.* Analytical Chemistry. 1964;36(8):1627–1639. [DOI: 10.1021/ac60214a047](https://doi.org/10.1021/ac60214a047). Basis for coordinate-wise Savitzky–Golay path smoothing through SciPy.

The production route is segmentation → component-wise 3-D thinning → voxel-neighbour graph → branch/path extraction → smoothing and segmentation filtering. It is not a VMTK Voronoi/fast-marching algorithm. Majority voting over 4-D labels, group merging, special-label multi-pass handling, triangle-cycle cleanup, terminal pruning, degree-based fork detection and flow-informed orientation are AutoFlow logic; CL-1/CL-2 do not prescribe those steps. Path smoothing preserves endpoints.

See [Skeleton](../features/skeleton.md) and [Graph and paths](../features/graph-paths.md). Centerline extraction sources are separate from the pressure references used when sampling pressure along a path.

## Noise masking and PC-MRA

- **PC-1.** Bock J, Kreher BW, Hennig J, Markl M. *Optimized pre-processing of time-resolved 2D and 3D Phase Contrast MRI data.* Proceedings of ISMRM. 2007;15:3138. [Original abstract PDF](https://cds.ismrm.org/protected/07MProceedings/PDFfiles/03138.pdf). Describes magnitude/velocity-time-course SD masks with interactive thresholds, correction, unwrapping and magnitude-weighted PC-MRA.
- **PC-2.** Bock J, Wieben O, Johnson KM, Hennig J, Markl M. *Optimal processing to derive static PC-MRA from time-resolved 3D PC-MRI data.* Proceedings of ISMRM. 2008;16:3053. [Original abstract PDF](https://cds.ismrm.org/protected/08MProceedings/PDFfiles/03053.pdf). Compares static PC-MRA formulas after noise masking and static-tissue removal.

PC-1 Table 1 reports empirical magnitude-noise thresholds of `(9 ± 3)%` of maximum magnitude and temporal-SD noise thresholds of `(20 ± 4)%` of maximum SD. Its `(10 ± 2)%` SD setting concerns static-region selection for eddy-current correction, not a required noise-mask lower cutoff. PC-2 gives no numerical noise-threshold defaults. These are thoracic-aorta studies, not validated portal-vein defaults.

**Current AutoFlow defaults are 5% magnitude and 80% temporal SD.** They are deliberately less restrictive project settings, not the papers' recommendations. AutoFlow thresholds temporal-mean magnitude and temporal SD of speed, creates one display mask, and leaves quantitative velocity/segmentation unchanged. It does not remove static tissue. Generated PC-MRA is per-frame `magnitude × speed`, not all eight static formulas in PC-2. See [Noise removal](../features/noise-removal.md) and [PC-MRA](../features/pcmra-volume-rendering.md).

## Background phase correction

- **BGC-1.** Pruitt AA, Jin N, Liu Y, Simonetti OP, Ahmad R. *A method to correct background phase offset for phase-contrast MRI in the presence of steady flow and spatial wrap-around artifact.* Magnetic Resonance in Medicine. 2019;81(4). [DOI: 10.1002/mrm.27572](https://doi.org/10.1002/mrm.27572); [open full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC6372316/).

WRLS means **weighted regularized least squares**; ARTO means **automatic rejection of temporally invariant outliers**. BGC-1 excludes steady-flow/wrap-around outliers using regression residuals and a Gaussian mixture model. AutoFlow adapts this to its 3-D polynomial fields, initialization, candidate masks, iteration controls and acceleration. The paper's validation and second-order settings do not validate all AutoFlow defaults. This citation concerns WRLS+ARTO, not the separate MSAC backend. See [Background phase correction](../features/background-phase-correction.md).

## Phase unwrapping

- **PU-1.** Loecher M, Schrauben E, Johnson KM, Wieben O. *Phase unwrapping in 4D MR flow with a 4D single-step laplacian algorithm.* Journal of Magnetic Resonance Imaging. 2016;43(4). [DOI: 10.1002/jmri.25045](https://doi.org/10.1002/jmri.25045); [upstream MATLAB implementation](https://github.com/mloecher/4dflow-lapunwrap).

This reference is explicitly named by the bundled LAP4D implementation. AutoFlow adds layout conversion, temporal scaling, device dispatch and workspace mask/rerun handling. It does not establish the same provenance for `gc3D`, `nprs`, PUDIP-Flow or GUST-Flow. Their backend/upstream links are in [Phase unwrapping](../features/phase-unwrapping.md); do not cite PU-1 for every backend.

## Segmentation

- **SEG-1.** Isensee F, Jaeger PF, Kohl SAA, Petersen J, Maier-Hein KH. *nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation.* Nature Methods. 2021;18(2). [DOI: 10.1038/s41592-020-01008-z](https://doi.org/10.1038/s41592-020-01008-z).

Reference for the automatic segmentation framework. Deployed checkpoints, training datasets, vessel labels and AutoFlow's 4-D integration require separate provenance/evaluation. Imported or embedded masks are not automatically nnU-Net results. See [Segmentation](../features/segmentation.md).

## Turbulent kinetic energy

- **TKE-1.** Dyverfeldt P, Sigfridsson A, Kvitting JP, Ebbers T. *Quantification of intravoxel velocity standard deviation and turbulence intensity by generalizing phase-contrast MRI.* Magnetic Resonance in Medicine. 2006;56(4). [DOI: 10.1002/mrm.21022](https://doi.org/10.1002/mrm.21022).

Supports magnitude-attenuation-derived intravoxel velocity dispersion. With suitable sigma, AutoFlow uses `TKE = 0.5*rho*(sigma_x² + sigma_y² + sigma_z²)` in J/m³. This is not the cardiac-cycle speed SD used by the noise overlay. Ordinary magnitude/mean velocity cannot establish turbulence; missing sigma/source TKE stays unavailable. See [TKE](../features/tke.md).

## Pulse-wave velocity

- **PWV-1.** Wentland AL, Grist TM, Wieben O. *Review of MRI-based measurements of pulse wave velocity: a biomarker of arterial stiffness.* Cardiovascular Diagnosis and Therapy. 2014;4(2). [DOI: 10.3978/j.issn.2223-3652.2014.03.04](https://doi.org/10.3978/j.issn.2223-3652.2014.03.04).

Background for path-distance/transit-time MRI estimation and arterial-stiffness interpretation. AutoFlow's plane selection, foot detection, cross-correlation, cycle handling and fit rejection are project implementations, not an exact recipe from this review. Adequate temporal resolution and a suitable arterial propagation path are required. See [PWV](../features/pwv.md).

## Vortex kinematics

- **V-1.** Jeong J, Hussain F. *On the identification of a vortex.* Journal of Fluid Mechanics. 1995;285:69–94. [DOI: 10.1017/S0022112095000462](https://doi.org/10.1017/S0022112095000462). Foundational comparison of velocity-gradient vortex identifiers.
- **V-2.** Zhou J, Adrian RJ, Balachandar S, Kendall TM. *Mechanisms for generating coherent packets of hairpin vortices in channel flow.* Journal of Fluid Mechanics. 1999;387:353–396. [DOI: 10.1017/S002211209900467X](https://doi.org/10.1017/S002211209900467X). Basis for identifying swirling motion through complex eigenvalues of the velocity-gradient tensor.

AutoFlow reports vorticity, `Q = 0.5*(||Omega||² - ||S||²)` and swirling strength `lambda_ci`. It does not implement the `lambda_2` criterion developed in V-1. These fluid-mechanics papers do not supply a clinical MRI threshold. See [Vortex kinematics](../features/vortex-kinematics.md).

## General 4D-flow guidance

- **CMR-1.** Dyverfeldt P, Bissell M, Barker AJ, et al. *4D flow cardiovascular magnetic resonance consensus statement.* Journal of Cardiovascular Magnetic Resonance. 2015;17. [DOI: 10.1186/s12968-015-0174-5](https://doi.org/10.1186/s12968-015-0174-5).
- **CMR-2.** Bissell MM, Raimondi F, Ait Ali L, et al. *4D Flow cardiovascular magnetic resonance consensus statement: 2023 update.* Journal of Cardiovascular Magnetic Resonance. 2023;25(1). [DOI: 10.1186/s12968-023-00942-z](https://doi.org/10.1186/s12968-023-00942-z).

Acquisition, retrospective plane-flow analysis, correction, visualization, quality assurance and reporting context, not AutoFlow certification. Review calibration, segmentation, temporal/spatial resolution and retained phantom results before interpreting [plane metrics](../features/metrics.md) or other biomarkers.

## Software foundations

- **SW-1.** van der Walt S, Schönberger JL, Nunez-Iglesias J, et al. *scikit-image: image processing in Python.* PeerJ. 2014;2:e453. [DOI: 10.7717/peerj.453](https://doi.org/10.7717/peerj.453).
- **SW-2.** Virtanen P, Gommers R, Oliphant TE, et al. *SciPy 1.0: fundamental algorithms for scientific computing in Python.* Nature Methods. 2020;17(3). [DOI: 10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2).

Software acknowledgements for thinning, morphology, filtering, interpolation and linear algebra, distinct from clinical validation. Optional third-party backends retain their own citations and licenses.

## Maintaining the reference list

When a calculation/backend changes, update its mapping and implementation caveat here. Prefer original articles/abstracts and verified DOI metadata. Distinguish method sources, contextual reviews and software references; do not cite an unimplemented method or label project parameters as paper recommendations. Link feature guides to the relevant section and record checkpoint-specific provenance separately for segmentation or learned reconstruction models.
