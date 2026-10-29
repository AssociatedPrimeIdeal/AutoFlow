"""Sample a documented finite-element CFD reference onto MRI grids.

The reference mesh, pressure and wall shear come from VMR, independently of
AutoFlow's velocity derivatives and wall fitting. Pressure gradients are the
piecewise constant derivatives of the source linear tetrahedral pressure.
"""
import json
import hashlib
from pathlib import Path

import h5py
import numpy as np


def load_cfd_reference(path):
    path = Path(path)
    with np.load(path, allow_pickle=False) as source:
        result = {name: source[name] for name in source.files}
    if 'velocity_file_name' in result:
        companion = path.parent/str(result['velocity_file_name'])
        if companion.parent.resolve() != path.parent.resolve():
            raise ValueError('CFD velocity fixture must be alongside its reference mesh')
        with np.load(companion, allow_pickle=False) as source:
            result['velocity_cm_s'] = source['velocity_cm_s']
    required = {'points_mm', 'tetrahedra', 'velocity_cm_s', 'pressure_pa',
                'wall_points_mm', 'wall_triangles', 'wall_wss_vectors_pa',
                'times_s', 'period_s', 'metadata_json'}
    missing = required.difference(result)
    if missing:
        raise ValueError(f'Incomplete CFD reference: {sorted(missing)}')
    points = result['points_mm']; cells = result['tetrahedra']
    count = len(result['times_s'])
    if points.ndim != 2 or points.shape[1] != 3 or cells.ndim != 2 or cells.shape[1] != 4:
        raise ValueError('CFD reference requires XYZ points and original tetrahedra')
    if np.any(cells < 0) or np.any(cells >= len(points)):
        raise ValueError('CFD tetrahedral connectivity is outside the point array')
    if result['velocity_cm_s'].shape != (len(points), count, 3) or result['pressure_pa'].shape != (len(points), count):
        raise ValueError('CFD fields do not match the mesh and cardiac phases')
    period = float(result['period_s'])
    if period <= 0 or not np.isfinite(period) or not np.allclose(result['times_s'], np.arange(count)*period/count):
        raise ValueError('CFD phases must be uniform over one cycle without a duplicate endpoint')
    result['metadata'] = json.loads(str(result['metadata_json']))
    if 'velocity_file_name' in result:
        digest = hashlib.sha256(companion.read_bytes()).hexdigest()
        if digest != result['metadata'].get('velocity_file_sha256'):
            raise ValueError('CFD velocity fixture checksum does not match its reference mesh')
    return result


class TetrahedralSampler:
    """Reuse one original-mesh locator across bounded output slabs."""
    def __init__(self, reference):
        import pyvista as pv
        from vtkmodules.vtkCommonDataModel import vtkStaticCellLocator
        self.vertices = np.asarray(reference['points_mm'], dtype=float)
        self.cells = np.asarray(reference['tetrahedra'], dtype=np.int64)
        connectivity = np.column_stack((np.full(len(self.cells), 4), self.cells)).ravel()
        self.grid = pv.UnstructuredGrid(connectivity, np.full(len(self.cells), 10, dtype=np.uint8), self.vertices)
        self.grid.cell_data['reference_cell'] = np.arange(len(self.cells), dtype=np.int64)
        self.locator = vtkStaticCellLocator()
        self.locator.SetDataSet(self.grid); self.locator.BuildLocator()

    def sample(self, points_mm):
        import pyvista as pv
        points = np.asarray(points_mm, dtype=float).reshape(-1, 3)
        sampled = pv.PolyData(points).sample(self.grid, locator=self.locator, tolerance=1e-9,
                                           pass_point_data=False, pass_cell_data=False)
        valid = np.asarray(sampled['vtkValidPointMask'], dtype=bool)
        ids = np.asarray(sampled['reference_cell'], dtype=np.int64)[valid]
        tetra = self.cells[ids]; xyz = self.vertices[tetra]
        matrix = np.moveaxis(xyz[:, 1:] - xyz[:, :1], 1, 2)
        coordinates = np.linalg.solve(matrix, (points[valid]-xyz[:, 0])[..., None])[..., 0]
        weights = np.column_stack((1-coordinates.sum(axis=1), coordinates))
        # Cell evaluation may accept slight boundary extrapolation; reject it.
        strict = np.all((weights >= -1e-7) & (weights <= 1.0000001), axis=1)
        valid[np.flatnonzero(valid)[~strict]] = False
        return valid, tetra[strict], weights[strict], matrix[strict]


def tetrahedral_samples(reference, points_mm):
    """Locate points in the original tetrahedra, retaining the lumen boundary."""
    return TetrahedralSampler(reference).sample(points_mm)


def native_cfd_geometry(reference):
    """Enclose every source vessel on the original image's sampling lattice."""
    bounds = np.asarray(reference['domain_bounds_mm'], dtype=float).reshape(3, 2)
    spacing = np.asarray(reference['native_spacing_mm'], dtype=float)
    image_origin = np.asarray(reference['native_image_origin_mm'], dtype=float)
    origin = image_origin+np.floor((bounds[:, 0]-3.-image_origin)/spacing)*spacing
    shape = tuple((np.ceil((bounds[:, 1]+3.-origin)/spacing).astype(int)+1).tolist())
    return shape, spacing, origin


def save_native_cfd_phantom(path, reference):
    """Stream the whole CFD model onto its native image lattice, without regions.

    Only empty background is omitted from storage. Every source tetrahedron and
    all 20 selected full-cycle snapshots participate in the same input file.
    """
    from scipy.spatial import cKDTree
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    shape, spacing, origin = native_cfd_geometry(reference)
    phases = len(reference['times_s']); sampler = TetrahedralSampler(reference)
    caps = np.asarray(reference['cap_boundary_points_mm'])
    cap_tree = cKDTree(caps)
    metadata = dict(reference['metadata'], truth_type='independent_cfd_numerical_reference',
                    spacing_mm=spacing.tolist(), origin_mm=origin.tolist(),
                    spatial_samples='origin_plus_index_times_spacing',
                    sampling='linear FE point samples on original image lattice; entire model, no regional crop',
                    fov='whole vascular bounding box plus background padding; original lattice extended where the source CFD has outlet extensions',
                    cap_exclusion_mm=6.)
    block = tuple(min(n, s) for n, s in zip(shape, (32, 32, 16)))
    with h5py.File(path, 'x') as target:
        common = dict(compression='gzip', compression_opts=4, shuffle=True)
        datasets = {}
        for name, dtype, fill in [('mag', 'f4', 100.), ('segmask', 'i2', 0), ('flow', 'f4', 0.)]:
            vector = name == 'flow'
            datasets[name] = target.create_dataset(name, shape=shape+(phases,)+( (3,) if vector else ()),
                                                   dtype=dtype, fillvalue=fill, chunks=block+(1,)+((3,) if vector else ()), **common)
        for name, value in [('resolution', spacing), ('origin', origin), ('rr', 1000*float(reference['period_s'])),
                            ('venc', max(150., 1.1*float(np.max(np.abs(reference['velocity_cm_s'])))))]:
            target[name] = value
        target['SpatialOrder'] = np.array(['LR', 'AP', 'FH'], dtype='S4')
        target['VENCOrder'] = np.array(['LR', 'AP', 'FH'], dtype='S4')
        target.attrs['autoflow_internal_spatial_order'] = 'XYZ'
        target.attrs['cfd_parameters_json'] = json.dumps(metadata)
        target.attrs['reference_kind'] = 'independent_cfd_numerical_reference'
        truth = target.create_group('truth')
        pressure = truth.create_dataset('pressure_pa', shape=shape+(phases,), dtype='f4', fillvalue=np.nan, chunks=block+(1,), **common)
        gradient = truth.create_dataset('pressure_gradient_pa_m', shape=shape+(phases, 3), dtype='f4', fillvalue=np.nan, chunks=block+(1, 3), **common)
        support = truth.create_dataset('pressure_valid_mask', shape=shape, dtype='?', chunks=block, **common)
        for key in ['times_s', 'wall_points_mm', 'wall_triangles', 'wall_wss_vectors_pa',
                    'plane_centers_mm', 'plane_normals', 'plane_area_mm2', 'plane_flowrate_ml_s']:
            truth.create_dataset(key, data=reference[key], **common)
        truth.create_dataset('wall_valid_mask', data=cap_tree.query(reference['wall_points_mm'])[0] >= 6., **common)
        truth.create_dataset('cap_boundary_points_mm', data=caps, **common)
        for z0 in range(0, shape[2], block[2]):
            z1 = min(shape[2], z0+block[2]); slab_shape = (shape[0], shape[1], z1-z0)
            coordinates = np.moveaxis(np.indices(slab_shape, dtype=float), 0, -1)
            coordinates[..., 2] += z0; coordinates = coordinates*spacing+origin
            valid, tetra, weights, matrices = sampler.sample(coordinates.reshape(-1, 3))
            if not np.any(valid):
                continue
            velocity_values = np.einsum('ni,nitc->ntc', weights, reference['velocity_cm_s'][tetra]).astype(np.float32)
            pressure_values = np.einsum('ni,nit->nt', weights, reference['pressure_pa'][tetra]).astype(np.float32)
            delta = reference['pressure_pa'][tetra[:, 1:]]-reference['pressure_pa'][tetra[:, :1]]
            gradient_values = np.swapaxes(np.linalg.solve(np.swapaxes(matrices, 1, 2), delta)*1000., 1, 2).astype(np.float32)
            index = np.full(len(valid), -1, dtype=np.int32); index[valid] = np.arange(np.count_nonzero(valid))
            index = index.reshape(slab_shape)
            for x0 in range(0, shape[0], block[0]):
                for y0 in range(0, shape[1], block[1]):
                    x1, y1 = min(shape[0], x0+block[0]), min(shape[1], y0+block[1])
                    lookup = index[x0:x1, y0:y1]; mask = lookup >= 0
                    if not np.any(mask):
                        continue
                    region = (slice(x0, x1), slice(y0, y1), slice(z0, z1))
                    chunk_shape = mask.shape
                    array = np.zeros(chunk_shape+(phases, 3), dtype=np.float32)
                    array[mask] = velocity_values[lookup[mask]]; datasets['flow'][region] = array
                    array[:] = np.nan; array[mask] = gradient_values[lookup[mask]]; gradient[region] = array
                    scalar = np.full(chunk_shape+(phases,), np.nan, dtype=np.float32)
                    scalar[mask] = pressure_values[lookup[mask]]; pressure[region] = scalar
                    datasets['segmask'][region] = np.broadcast_to(mask[..., None], chunk_shape+(phases,)).astype(np.int16)
                    datasets['mag'][region] = np.broadcast_to((100.+900.*mask)[..., None], chunk_shape+(phases,)).astype(np.float32)
                    interior = np.zeros(chunk_shape, dtype=bool)
                    xyz = coordinates[x0:x1, y0:y1]
                    interior[mask] = cap_tree.query(xyz[mask])[0] >= 6.
                    support[region] = interior
            print(f'VMR 0028 native lattice: slab {z1}/{shape[2]}', flush=True)
    return path


def create_cfd_phantom(reference, spacing=(1., 1., 1.)):
    """Return a normalized MRI input and independent numerical reference fields."""
    spacing = np.asarray(spacing, dtype=float)
    if spacing.shape != (3,) or np.any(~np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError('spacing must contain three positive finite mm values')
    bounds = np.asarray(reference.get('domain_bounds_mm', reference.get('crop_bounds_mm')), dtype=float).reshape(3, 2)
    origin = bounds[:, 0]-3.
    shape = tuple((np.ceil((bounds[:, 1]+3-origin)/spacing).astype(int)+1).tolist())
    points = np.moveaxis(np.indices(shape, dtype=float), 0, -1)*spacing+origin
    flat = points.reshape(-1, 3)
    inside, tetra, weights, matrices = tetrahedral_samples(reference, flat)
    in_crop = np.all((flat >= bounds[:, 0]) & (flat <= bounds[:, 1]), axis=1)
    selected = in_crop[inside]
    inside[inside.copy()] = selected
    tetra = tetra[selected]; weights = weights[selected]; matrices = matrices[selected]
    phases = len(reference['times_s'])
    velocity = np.einsum('ni,nitc->ntc', weights, reference['velocity_cm_s'][tetra])
    pressure = np.einsum('ni,nit->nt', weights, reference['pressure_pa'][tetra])
    pressure_delta = reference['pressure_pa'][tetra[:, 1:]]-reference['pressure_pa'][tetra[:, :1]]
    gradient = np.linalg.solve(np.swapaxes(matrices, 1, 2), pressure_delta)*1000.
    gradient = np.swapaxes(gradient, 1, 2)
    flow = np.zeros((len(flat), phases, 3), dtype=np.float32); flow[inside] = velocity
    p = np.full((len(flat), phases), np.nan, dtype=np.float32); p[inside] = pressure
    g = np.full((len(flat), phases, 3), np.nan, dtype=np.float32); g[inside] = gradient
    mask = inside.reshape(shape)
    # Imaging truncation faces are not physical no-slip walls.
    crop_distance = np.minimum(flat-bounds[:, 0], bounds[:, 1]-flat).min(axis=1)
    pressure_valid = (inside & (crop_distance >= 6.)).reshape(shape)
    wall = np.asarray(reference['wall_points_mm'])
    wall_distance = np.minimum(wall-bounds[:, 0], bounds[:, 1]-wall).min(axis=1)
    truth = dict(times_s=reference['times_s'], pressure_pa=p.reshape(shape+(phases,)),
                 pressure_gradient_pa_m=g.reshape(shape+(phases, 3)),
                 pressure_valid_mask=pressure_valid,
                 wall_points_mm=wall, wall_triangles=reference['wall_triangles'],
                 wall_wss_vectors_pa=reference['wall_wss_vectors_pa'],
                 wall_valid_mask=wall_distance >= 6.)
    for key in ('plane_centers_mm', 'plane_normals', 'plane_area_mm2', 'plane_flowrate_ml_s'):
        if key in reference:
            truth[key] = reference[key]
    metadata = dict(reference['metadata'], spacing_mm=spacing.tolist(), origin_mm=origin.tolist(),
                    truth_type='independent_cfd_numerical_reference',
                    spatial_samples='origin_plus_index_times_spacing',
                    sampling='point samples of the original linear tetrahedral velocity; no voxel averaging or noise',
                    cap_exclusion_mm=6.)
    return dict(mag=np.broadcast_to((100+900*mask.astype(np.float32))[..., None], shape+(phases,)),
                flow=flow.reshape(shape+(phases, 3)),
                segmentation=np.broadcast_to(mask[..., None].astype(np.int16), shape+(phases,)),
                resolution=spacing, origin=origin, rr=1000*float(reference['period_s']),
                venc=max(150., 1.1*float(np.max(np.abs(flow)))), truth=truth, parameters=metadata)


def save_cfd_phantom(path, data):
    from .analytic_phantoms import save_analytic_phantom
    # The normalized field and truth dataset schema is shared; provenance is
    # deliberately tagged as CFD rather than an analytical solution.
    path = save_analytic_phantom(path, data)
    with h5py.File(path, 'r+') as target:
        target.attrs['cfd_parameters_json'] = target.attrs.pop('analytic_parameters_json')
        target.attrs['reference_kind'] = 'independent_cfd_numerical_reference'
    return Path(path)
