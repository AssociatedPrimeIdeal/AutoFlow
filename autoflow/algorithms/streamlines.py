import numpy as np
import pyvista as pv

from vtkmodules.util.vtkAlgorithm import VTKPythonAlgorithmBase
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from vtkmodules.vtkCommonCore import vtkFloatArray, vtkPoints
from vtkmodules.vtkCommonDataModel import (
    vtkCellArray,
    vtkDataObject,
    vtkImageData,
    vtkMultiBlockDataSet,
    vtkPolyData,
)
from vtkmodules.vtkCommonExecutionModel import vtkStreamingDemandDrivenPipeline
from vtkmodules.vtkFiltersFlowPaths import vtkParticleTracer, vtkParticleTracerBase

from .surfaces import (
    _build_branch_grid,
    create_uniform_grid,
    create_uniform_vector,
    extract_plane_cross_section,
)


_AUTOMATIC_CLIM_PERCENTILE = 99.0


def automatic_streamline_clim(flow_xyzt3, mask=None):
    """Return a robust, all-phase velocity range in m/s for streamline colors."""
    if flow_xyzt3 is None:
        return (0.0, 1.0)
    flow = np.asarray(flow_xyzt3, dtype=np.float32)
    if flow.ndim != 5 or flow.shape[-1] != 3:
        return (0.0, 1.0)

    vectors = None
    if mask is not None:
        mask_arr = np.asarray(mask, dtype=bool)
        if mask_arr.shape == flow.shape[:4]:
            vectors = flow[mask_arr]
        elif mask_arr.shape == flow.shape[:3]:
            vectors = flow[mask_arr].reshape(-1, 3)
        elif mask_arr.ndim == 4 and mask_arr.shape[:3] == flow.shape[:3] and mask_arr.shape[3] == 1:
            vectors = flow[np.broadcast_to(mask_arr, flow.shape[:4])]
    if vectors is None:
        vectors = flow.reshape(-1, 3)
    vectors = np.asarray(vectors, dtype=np.float32).reshape(-1, 3)
    if vectors.size == 0:
        return (0.0, 1.0)
    vectors = vectors[np.all(np.isfinite(vectors), axis=1)]
    if vectors.size == 0:
        return (0.0, 1.0)

    speed_sq = np.einsum("ij,ij->i", vectors, vectors, optimize=True)
    speed_cm_s = np.sqrt(speed_sq)
    speed_limit = float(
        np.percentile(speed_cm_s, _AUTOMATIC_CLIM_PERCENTILE)
    ) / 100.0
    if speed_limit <= 0.0 and np.any(speed_cm_s > 0.0):
        speed_limit = float(np.max(speed_cm_s)) / 100.0
    if not np.isfinite(speed_limit) or speed_limit <= 0.0:
        speed_limit = 1.0
    return (0.0, max(speed_limit, 1e-6))


def _set_streamline_velocity_scalar(streamlines):
    """Color streamlines by the magnitude of the vector used by the tracer."""
    if streamlines is None or "vector" not in streamlines.point_data:
        return streamlines
    vectors = np.asarray(streamlines.point_data["vector"], dtype=np.float32)
    streamlines.point_data["Velocity"] = np.linalg.norm(
        vectors, axis=1
    ).astype(np.float32, copy=False)
    streamlines.set_active_scalars("Velocity")
    return streamlines


def generate_seed_points(mask_3d, spacing, origin, ratio=0.02, rng_seed=0,
                         min_seeds=50):
    idx = np.argwhere(np.asarray(mask_3d, dtype=bool))
    K = len(idx)
    if K == 0:
        return np.empty((0, 3), dtype=float)
    n = int(min(K, max(int(min_seeds), int(round(K * float(ratio))))))
    rng = np.random.default_rng(int(rng_seed))
    pick = idx[rng.choice(K, size=n, replace=False)]
    sp = np.asarray(spacing, dtype=float).reshape(1, 3)
    org = np.asarray(origin, dtype=float).reshape(1, 3)
    return (org + pick.astype(float) * sp).astype(float)


def _crop_vector_domain_to_mask(flow_t, mask_3d, spacing, origin, padding=1):
    mask = np.asarray(mask_3d, dtype=bool)
    occupied = np.where(mask)
    if not occupied[0].size:
        return np.asarray(flow_t), mask, np.asarray(origin, dtype=float).reshape(3)
    pad = max(int(padding), 0)
    lo = np.array([max(int(np.min(axis_values)) - pad, 0) for axis_values in occupied], dtype=int)
    hi = np.array([
        min(int(np.max(axis_values)) + pad + 1, int(mask.shape[axis]))
        for axis, axis_values in enumerate(occupied)
    ], dtype=int)
    spatial_slices = tuple(slice(int(lo[axis]), int(hi[axis])) for axis in range(3))
    cropped_origin = (
        np.asarray(origin, dtype=float).reshape(3)
        + lo.astype(float) * np.asarray(spacing, dtype=float).reshape(3)
    )
    return np.asarray(flow_t)[spatial_slices], mask[spatial_slices], cropped_origin


def generate_streamlines_at_t(flow_xyzt3, t, seeds, spacing, origin, mask_3d=None,
                              max_steps=2000, terminal_speed=0.01,
                              seed_ratio=0.02, min_seeds=50, rng_seed=0):
    if mask_3d is None:
        return None
    flow_t, mask_work, work_origin = _crop_vector_domain_to_mask(
        flow_xyzt3[..., int(t), :],
        mask_3d,
        spacing,
        origin,
    )
    flow_t = np.asarray(flow_t, dtype=np.float32)
    velocity = create_uniform_vector(
        flow_t[..., 0] / 100.0,
        flow_t[..., 1] / 100.0,
        flow_t[..., 2] / 100.0,
        spacing, origin=work_origin)
    mesh = create_uniform_grid(mask_work, spacing, origin=work_origin)
    mesh = mesh.threshold(0.1)
    if mesh.n_points == 0:
        return None
    volume = mesh.sample(velocity)
    volume.set_active_scalars('Velocity')
    if volume.points.shape[0] == 0:
        return None
    if seeds is None or len(seeds) == 0:
        seeds = generate_seed_points(
            mask_3d,
            spacing,
            origin,
            ratio=seed_ratio,
            rng_seed=int(rng_seed) + int(t),
            min_seeds=min_seeds,
        )
    source = np.asarray(seeds, dtype=float)
    if source.size == 0:
        return None
    sl = volume.streamlines_from_source(
        vectors='vector',
        source=source,
        integrator_type=4,
        max_steps=int(max_steps),
        terminal_speed=float(terminal_speed),
        compute_vorticity=False,
    )
    if sl is None or sl.n_points == 0:
        return None
    return _set_streamline_velocity_scalar(sl)


def _plane_seeds(mask_3d, plane, spacing, origin, seed_ratio, min_seeds,
                 rng_seed, branch_labels_3d=None, t=0, max_seeds=None,
                 seed_mode="ratio"):
    if mask_3d is None:
        return None
    mask = np.asarray(mask_3d, dtype=bool)
    if mask.ndim != 3 or any(int(size) <= 0 for size in mask.shape):
        raise ValueError(f"plane seed mask must be a non-empty 3D array, got {mask.shape}")
    if not np.any(mask):
        return None
    spacing_arr = np.asarray(spacing, dtype=float).reshape(-1)
    origin_arr = np.asarray(origin, dtype=float).reshape(-1)
    if spacing_arr.size != 3 or not np.all(np.isfinite(spacing_arr)) or np.any(spacing_arr <= 0.0):
        raise ValueError(f"spacing must contain three positive finite values, got {spacing_arr}")
    if origin_arr.size != 3 or not np.all(np.isfinite(origin_arr)):
        raise ValueError(f"origin must contain three finite values, got {origin_arr}")
    try:
        center = np.asarray(plane.center, dtype=float).reshape(3)
        normal = np.asarray(plane.normal, dtype=float).reshape(3)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("plane must provide three-element center and normal values") from exc
    if (
        not np.all(np.isfinite(center))
        or not np.all(np.isfinite(normal))
        or np.linalg.norm(normal) <= 1e-12
    ):
        raise ValueError("plane center and normal must be finite, with a non-zero normal")
    if branch_labels_3d is not None and np.asarray(branch_labels_3d).shape != mask.shape:
        raise ValueError(
            f"branch labels must match pathline mask shape {mask.shape}, "
            f"got {np.asarray(branch_labels_3d).shape}"
        )

    mesh = create_uniform_grid(mask, spacing_arr, origin=origin_arr)
    mesh = mesh.threshold(0.1)
    if mesh.n_points == 0:
        return None
    branch_grid = _build_branch_grid(branch_labels_3d, spacing_arr, origin_arr)
    target_label = None
    if branch_labels_3d is not None:
        target_label = int(getattr(plane, 'label', 0) or 0)
        if target_label <= 0:
            # PlaneData.center is local physical space; do not subtract the
            # world origin when converting it to an array index.
            ijk = np.rint(center / (spacing_arr + 1e-12)).astype(int)
            ijk = np.clip(ijk, 0, np.array(np.asarray(branch_labels_3d).shape) - 1)
            target_label = int(np.asarray(branch_labels_3d)[ijk[0], ijk[1], ijk[2]])
    pg = extract_plane_cross_section(mask, plane, spacing_arr, origin_arr, branch_grid=branch_grid, target_label=target_label)
    if pg is None or pg.n_cells == 0:
        return None
    seeds = pg.cell_centers().points
    if len(seeds) == 0:
        return None
    mode = str(seed_mode or "ratio").strip().lower()
    if mode == "fixed":
        n_seeds = max(1, int(max_seeds if max_seeds is not None else min_seeds))
    else:
        n_seeds = int(max(int(min_seeds), int(np.ceil(len(seeds) * float(seed_ratio)))))
        if max_seeds is not None:
            n_seeds = min(n_seeds, max(1, int(max_seeds)))
    n_seeds = max(1, min(int(n_seeds), len(seeds)))
    if len(seeds) > n_seeds:
        rng = np.random.default_rng(int(rng_seed) + 104729 * int(getattr(plane, 'path_index', 0)) + int(t))
        idx = rng.choice(len(seeds), size=n_seeds, replace=False)
        seeds = seeds[np.sort(idx)]
    return np.asarray(seeds, dtype=float)


def _validate_pathline_inputs(flow, mask_4d, mask_3d, spacing, origin, time_index):
    """Validate values before handing arrays to VTK's native tracer."""
    if flow.ndim != 5 or flow.shape[-1] != 3:
        raise ValueError(f"flow must be XYZT3, got {flow.shape}")
    if any(int(size) <= 0 for size in flow.shape[:4]):
        raise ValueError(f"flow has an empty spatial or temporal dimension: {flow.shape}")
    spacing_arr = np.asarray(spacing, dtype=float).reshape(-1)
    origin_arr = np.asarray(origin, dtype=float).reshape(-1)
    if spacing_arr.size != 3 or not np.all(np.isfinite(spacing_arr)) or np.any(spacing_arr <= 0.0):
        raise ValueError(f"spacing must contain three positive finite values, got {spacing_arr}")
    if origin_arr.size != 3 or not np.all(np.isfinite(origin_arr)):
        raise ValueError(f"origin must contain three finite values, got {origin_arr}")

    n_time = int(flow.shape[3])
    if not 0 <= int(time_index) < n_time:
        raise ValueError(f"time index {time_index} is outside 0..{n_time - 1}")
    spatial_shape = tuple(int(size) for size in flow.shape[:3])
    if mask_3d is None or np.asarray(mask_3d).shape != spatial_shape:
        raise ValueError(
            f"mask_3d must match flow spatial shape {spatial_shape}, "
            f"got {None if mask_3d is None else np.asarray(mask_3d).shape}"
        )
    if mask_4d is not None:
        mask_arr = np.asarray(mask_4d)
        if mask_arr.ndim == 4 and mask_arr.shape != flow.shape[:4]:
            raise ValueError(
                f"mask_4d must match flow shape {flow.shape[:4]}, got {mask_arr.shape}"
            )
        if mask_arr.ndim not in (3, 4):
            raise ValueError(f"mask must be 3D or 4D, got {mask_arr.shape}")

    # VTK stores point and cell counts in vtkIdType. Reject impossible products
    # before vtkImageData or vtkPoints attempts a native allocation.
    point_count = 1
    for size in spatial_shape:
        point_count *= int(size) + 1
    if point_count <= 0 or point_count > np.iinfo(np.int64).max:
        raise ValueError(f"flow grid is too large for VTK point indexing: {spatial_shape}")
    return spacing_arr, origin_arr


def _coerce_pathline_seeds(seeds, max_seeds):
    if seeds is None:
        return None
    seed_arr = np.asarray(seeds, dtype=float)
    if seed_arr.size == 0:
        return None
    if seed_arr.ndim != 2 or seed_arr.shape[1] != 3:
        raise ValueError(f"pathline seeds must have shape (N, 3), got {seed_arr.shape}")
    seed_arr = seed_arr[np.all(np.isfinite(seed_arr), axis=1)]
    if seed_arr.size == 0:
        return None
    if max_seeds is not None:
        limit = max(1, int(max_seeds))
        seed_arr = seed_arr[:limit]
    return np.ascontiguousarray(seed_arr, dtype=float)


def generate_streamlines_from_plane_at_t(flow_xyzt3, t, plane, spacing, origin,
                                         mask_3d=None, max_steps=2000,
                                         terminal_speed=0.01,
                                         seed_ratio=0.02, min_seeds=50,
                                         rng_seed=0,
                                         branch_labels_3d=None):
    flow_t = np.asarray(flow_xyzt3[..., int(t), :], dtype=np.float32)
    velocity = create_uniform_vector(
        flow_t[..., 0] / 100.0,
        flow_t[..., 1] / 100.0,
        flow_t[..., 2] / 100.0,
        spacing, origin=origin)
    if mask_3d is None:
        return None
    mesh = create_uniform_grid(np.asarray(mask_3d) > 0, spacing, origin=origin)
    mesh = mesh.threshold(0.1)
    if mesh.n_points == 0:
        return None
    volume = mesh.sample(velocity)
    volume.set_active_scalars('Velocity')
    if volume.points.shape[0] == 0:
        return None
    seeds = _plane_seeds(
        mask_3d,
        plane,
        spacing,
        origin,
        seed_ratio=seed_ratio,
        min_seeds=min_seeds,
        rng_seed=rng_seed,
        branch_labels_3d=branch_labels_3d,
        t=t,
    )
    if seeds is None or len(seeds) == 0:
        return None
    seeds = np.asarray(seeds, dtype=float).reshape(-1, 3)
    sl = volume.streamlines_from_source(
        vectors='vector',
        source=seeds,
        integrator_type=4,
        max_steps=int(max_steps),
        terminal_speed=float(terminal_speed),
        compute_vorticity=False,
    )
    if sl is None or sl.n_points == 0:
        return None
    return _set_streamline_velocity_scalar(sl)


class _TemporalVelocitySource(VTKPythonAlgorithmBase):
    """Expose an XYZT flow array as the temporal input for vtkParticleTracer."""

    def __init__(self, flow, mask, spacing, origin, start_phase, rr,
                 temporal_cache_mb=0.0):
        super().__init__(nInputPorts=0, nOutputPorts=1, outputType="vtkMultiBlockDataSet")
        self._flow = np.asarray(flow, dtype=np.float32)
        self._mask = None if mask is None else np.asarray(mask, dtype=bool)
        self._spacing = np.asarray(spacing, dtype=float).reshape(3)
        self._origin = np.asarray(origin, dtype=float).reshape(3)
        self._n_time = int(self._flow.shape[3])
        self._start_phase = int(start_phase) % max(self._n_time, 1)
        self._rr_seconds = max(float(rr) / 1000.0, 1e-6)
        self._times = np.linspace(0.0, self._rr_seconds, self._n_time + 1).tolist()
        # vtkParticleTracer only needs consecutive time steps. Keep the
        # generated image data lazy and bounded so a large 4D case is not
        # duplicated in memory for every cardiac phase.
        frame_shape = np.asarray(self._flow.shape[:3], dtype=np.int64) + 1
        frame_bytes = int(np.prod(frame_shape, dtype=np.int64)) * 3 * np.dtype(np.float32).itemsize
        cache_limit = max(0, int(float(temporal_cache_mb) * 1024.0 * 1024.0))
        self._cache_all_phases = bool(
            cache_limit > 0 and frame_bytes > 0
            and frame_bytes * self._n_time <= cache_limit
        )
        self._datasets = {}

    @property
    def cache_all_phases(self):
        return self._cache_all_phases

    def _build_dataset(self, time_index):
        phase = (self._start_phase + int(time_index)) % self._n_time
        vectors = np.nan_to_num(self._flow[..., phase, :], nan=0.0, posinf=0.0, neginf=0.0)
        if self._mask is not None:
            if self._mask.ndim == 4:
                mask_phase = self._mask[..., phase % int(self._mask.shape[3])]
            else:
                mask_phase = self._mask
            vectors = np.where(mask_phase[..., None], vectors, 0.0)
        # Loaded flow is cm/s; the workspace geometry is mm and the source
        # time axis is seconds, so VTK receives mm/s vectors.
        vectors_mm_s = np.asarray(vectors * 10.0, dtype=np.float32)
        # The rest of AutoFlow represents voxel fields as cell-centered grids
        # with an outer boundary at ``shape * spacing``.  Extend the VTK point
        # field by one sample on each axis so plane cell centers remain inside
        # the tracer domain.
        point_vectors = np.pad(
            vectors_mm_s,
            ((0, 1), (0, 1), (0, 1), (0, 0)),
            mode="edge",
        )
        image = vtkImageData()
        image.SetDimensions(*[int(size) for size in point_vectors.shape[:3]])
        image.SetSpacing(*[float(value) for value in self._spacing])
        image.SetOrigin(*[float(value) for value in self._origin])
        array = numpy_to_vtk(
            np.ascontiguousarray(point_vectors.reshape(-1, 3, order="F")),
            deep=True,
        )
        array.SetName("velocity")
        image.GetPointData().SetVectors(array)
        return image

    def RequestInformation(self, request, in_info, out_info):
        info = out_info.GetInformationObject(0)
        info.Set(vtkStreamingDemandDrivenPipeline.TIME_STEPS(), self._times, len(self._times))
        info.Set(
            vtkStreamingDemandDrivenPipeline.TIME_RANGE(),
            [self._times[0], self._times[-1]],
            2,
        )
        shape = np.asarray(self._flow.shape[:3], dtype=int) + 1
        info.Set(
            vtkStreamingDemandDrivenPipeline.WHOLE_EXTENT(),
            [0, int(shape[0]) - 1, 0, int(shape[1]) - 1, 0, int(shape[2]) - 1],
            6,
        )
        # vtkParticleTracer is a temporal integrator. This tells its temporal
        # executive that updates arrive in chronological order and should be
        # accumulated instead of replayed from the beginning for every call.
        info.Set(
            vtkStreamingDemandDrivenPipeline.NO_PRIOR_TEMPORAL_ACCESS(),
            vtkStreamingDemandDrivenPipeline.NO_PRIOR_TEMPORAL_ACCESS_RESET,
        )
        return 1

    def RequestData(self, request, in_info, out_info):
        info = out_info.GetInformationObject(0)
        requested = self._times[0]
        if info.Has(vtkStreamingDemandDrivenPipeline.UPDATE_TIME_STEP()):
            requested = float(info.Get(vtkStreamingDemandDrivenPipeline.UPDATE_TIME_STEP()))
        index = int(np.argmin(np.abs(np.asarray(self._times, dtype=float) - requested)))
        phase = (self._start_phase + index) % self._n_time
        if phase not in self._datasets:
            self._datasets[phase] = self._build_dataset(index)
            if not self._cache_all_phases:
                previous_phase = (phase - 1) % self._n_time
                self._datasets = {
                    key: value for key, value in self._datasets.items()
                    if key in {phase, previous_phase}
                }
        # The temporal tracer retains the two prior input datasets. A
        # multiblock wrapper gives each time request a distinct image child;
        # mutating the reusable pipeline output cannot then overwrite the
        # previous child retained by vtkParticleTracer.
        output = vtkMultiBlockDataSet.GetData(out_info)
        output.SetNumberOfBlocks(1)
        output.SetBlock(0, self._datasets[phase])
        output.GetInformation().Set(vtkDataObject.DATA_TIME_STEP(), self._times[index])
        self._datasets[phase].GetInformation().Set(
            vtkDataObject.DATA_TIME_STEP(), self._times[index]
        )
        return 1


def create_pathline_temporal_source(flow_xyzt3, mask, spacing, origin,
                                    start_phase, rr, temporal_cache_mb=0.0):
    """Create a reusable VTK temporal velocity source for one pathline batch."""
    flow = np.asarray(flow_xyzt3, dtype=np.float32)
    phase = int(start_phase)
    mask_arr = None if mask is None else np.asarray(mask)
    mask_3d = None
    if mask_arr is not None:
        mask_3d = (
            mask_arr[..., min(max(0, phase), mask_arr.shape[3] - 1)]
            if mask_arr.ndim == 4 else mask_arr
        )
    _validate_pathline_inputs(flow, mask_arr, mask_3d, spacing, origin, phase)
    rr_value = float(rr)
    if not np.isfinite(rr_value) or rr_value <= 0.0:
        raise ValueError(f"RR interval must be positive and finite, got {rr}")
    return _TemporalVelocitySource(
        flow, mask_arr, spacing, origin, start_phase=phase, rr=rr_value,
        temporal_cache_mb=temporal_cache_mb,
    )


def _particle_tracks_to_polydata(tracks):
    """Convert vtkParticleTracer samples into the existing PolyData contract."""
    valid = [track for track in tracks.values() if len(track) >= 2]
    if not valid:
        return None
    points = vtkPoints()
    lines = vtkCellArray()
    speeds = vtkFloatArray()
    speeds.SetName("Velocity")
    speeds.SetNumberOfComponents(1)
    phases = vtkFloatArray()
    phases.SetName("PathlinePhase")
    phases.SetNumberOfComponents(1)
    for track in valid:
        lines.InsertNextCell(len(track))
        for point, vector, phase in track:
            lines.InsertCellPoint(points.InsertNextPoint(*point))
            speeds.InsertNextValue(float(np.linalg.norm(vector)) / 1000.0)
            phases.InsertNextValue(float(phase))
    poly = vtkPolyData()
    poly.SetPoints(points)
    poly.SetLines(lines)
    poly.GetPointData().AddArray(speeds)
    poly.GetPointData().AddArray(phases)
    poly.GetPointData().SetActiveScalars("Velocity")
    return pv.wrap(poly).copy(deep=True)


def pathline_prefix_at_phase(pathline, phase):
    """Return the cached t=0 trajectory through a normalized phase.

    The expensive particle integration is performed once. Playback only
    rebuilds the visible prefix of each polyline from the phase samples stored
    on its points.
    """
    if pathline is None or "PathlinePhase" not in getattr(pathline, "point_data", {}):
        return pathline
    phase_values = np.asarray(pathline.point_data["PathlinePhase"], dtype=float).reshape(-1)
    if phase_values.size == 0 or not hasattr(pathline, "lines"):
        return pathline
    target = float(np.clip(phase, 0.0, 1.0))
    points = np.asarray(pathline.points, dtype=float)
    line_values = np.asarray(pathline.lines, dtype=np.int64).reshape(-1)
    selected_points = []
    selected_lines = []
    selected_ids = []
    initial_ids = []
    cursor = 0
    point_offset = 0
    while cursor < line_values.size:
        count = int(line_values[cursor])
        cursor += 1
        ids = line_values[cursor:cursor + count]
        cursor += count
        if count < 2:
            continue
        initial_ids.append(int(ids[0]))
        keep = ids[phase_values[ids] <= target + 1e-9]
        if keep.size < 2:
            continue
        selected_points.append(points[keep])
        selected_lines.append(np.concatenate(([keep.size], np.arange(point_offset, point_offset + keep.size, dtype=np.int64))))
        selected_ids.append(keep)
        point_offset += int(keep.size)
    if not selected_points:
        if not initial_ids:
            return None
        point_ids = np.asarray(initial_ids, dtype=np.int64)
        result = pv.PolyData(points[point_ids])
        for name, values in pathline.point_data.items():
            result.point_data[name] = np.asarray(values)[point_ids]
        result.set_active_scalars("Velocity")
        return result
    result = pv.PolyData(np.vstack(selected_points))
    result.lines = np.concatenate(selected_lines).astype(np.int64)
    for name, values in pathline.point_data.items():
        values_arr = np.asarray(values)
        result.point_data[name] = np.concatenate([values_arr[ids] for ids in selected_ids])
    result.set_active_scalars("Velocity")
    return result


def generate_pathlines_from_plane_at_t(flow_xyzt3, t, plane, spacing, origin,
                                       mask_4d=None, mask_3d=None, max_steps=2000,
                                       terminal_speed=0.01,
                                       seed_ratio=0.02, min_seeds=50,
                                       rng_seed=0, rr=1000.0,
                                       branch_labels_3d=None, max_seeds=250,
                                       seeds=None, progress_callback=None,
                                       seed_mode="ratio", temporal_source=None,
                                       temporal_cache_mb=0.0):
    flow = np.asarray(flow_xyzt3, dtype=np.float32)
    time_index = int(t)
    if mask_3d is None and mask_4d is not None:
        mask_arr = np.asarray(mask_4d)
        if mask_arr.ndim == 4:
            mask_3d = mask_arr[..., min(max(0, time_index), mask_arr.shape[3] - 1)]
        else:
            mask_3d = mask_arr
    spacing_arr, origin_arr = _validate_pathline_inputs(
        flow,
        mask_4d,
        mask_3d,
        spacing,
        origin,
        time_index,
    )
    rr_ms = float(rr)
    if not np.isfinite(rr_ms) or rr_ms <= 0.0:
        raise ValueError(f"RR interval must be positive and finite, got {rr}")
    if seeds is None:
        seeds = _plane_seeds(
            mask_3d,
            plane,
            spacing_arr,
            origin_arr,
            seed_ratio=seed_ratio,
            min_seeds=min_seeds,
            rng_seed=rng_seed,
            branch_labels_3d=branch_labels_3d,
            t=time_index,
            max_seeds=max_seeds,
            seed_mode=seed_mode,
        )
    seeds = _coerce_pathline_seeds(seeds, max_seeds)
    if seeds is None:
        return None
    source = temporal_source
    if source is None:
        source = create_pathline_temporal_source(
            flow,
            mask_4d if mask_4d is not None else mask_3d,
            spacing_arr,
            origin_arr,
            start_phase=time_index,
            rr=rr_ms,
            temporal_cache_mb=temporal_cache_mb,
        )
    seed_poly = vtkPolyData()
    seed_points = vtkPoints()
    for seed in seeds:
        seed_points.InsertNextPoint(*[float(value) for value in seed])
    seed_poly.SetPoints(seed_points)

    tracer = vtkParticleTracer()
    tracer.SetInputConnection(source.GetOutputPort())
    tracer.SetInputData(1, seed_poly)
    tracer.SetInputArrayToProcess(
        0,
        0,
        0,
        vtkDataObject.FIELD_ASSOCIATION_POINTS,
        "velocity",
    )
    # Treat each frame as an independent temporal dataset. This avoids VTK
    # reusing locator state across masked frames while retaining temporal
    # interpolation between them.
    tracer.SetMeshOverTimeToDifferent()
    tracer.SetComputeVorticity(False)
    tracer.SetForceSerialExecution(True)
    tracer.SetIntegratorType(vtkParticleTracerBase.RUNGE_KUTTA4)
    # terminal_speed is configured in m/s while VTK integrates mm/s vectors.
    tracer.SetTerminalSpeed(max(0.0, float(terminal_speed)) * 1000.0)

    tracks = {}
    # vtkParticleTracer advances through temporal input frames rather than
    # exposing the fixed integration-step limit used by the retired tracer.
    # Keep max_steps meaningful as a cap on cardiac-frame updates.
    frame_limit = max(1, int(max_steps))
    time_values = source._times[:min(len(source._times), frame_limit)]
    for time_index, time_value in enumerate(time_values):
        tracer.UpdateTimeStep(float(time_value))
        output = tracer.GetOutputDataObject(0)
        if output is not None and output.GetNumberOfPoints() > 0:
            points = np.asarray(vtk_to_numpy(output.GetPoints().GetData()), dtype=float)
            particle_ids = output.GetPointData().GetArray("ParticleId")
            vectors = output.GetPointData().GetArray("velocity")
            ids = (
                np.arange(points.shape[0], dtype=np.int64)
                if particle_ids is None
                else np.asarray(vtk_to_numpy(particle_ids), dtype=np.int64)
            )
            vector_values = (
                np.zeros((points.shape[0], 3), dtype=np.float32)
                if vectors is None
                else np.asarray(vtk_to_numpy(vectors), dtype=np.float32).reshape(-1, 3)
            )
            phase = float(time_value / max(source._times[-1], 1e-12))
            for particle_id, point, vector in zip(ids, points, vector_values):
                tracks.setdefault(int(particle_id), []).append((point, vector, phase))
        if progress_callback is not None:
            progress_callback(time_index + 1, len(time_values))
    return _particle_tracks_to_polydata(tracks)
