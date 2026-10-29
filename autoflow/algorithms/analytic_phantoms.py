"""Independent fully developed Newtonian flow solutions for regression data.

Lengths passed to the geometry API are mm, velocity output is cm/s. Analytical
pressure, pressure gradient and shear are SI. Truncation caps are imaging crop
planes, not physical no-slip walls. See docs/en/developer/analytic-phantoms.md.
"""
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.special import jv


@dataclass(frozen=True)
class FlowPhantom:
    name: str = 'circle'
    kind: str = 'ellipse'
    a_mm: float = 6.
    b_mm: float = 6.
    inner_mm: float = 0.
    length_mm: float = 64.
    direction: tuple = (0., 0., 1.)
    roll_rad: float = 0.
    umax_m_s: float = .6
    drive_pa_m: float = 200.
    harmonic_pa_m: float = 1400.
    period_s: float = .8
    mu_pa_s: float = .004
    rho_kg_m3: float = 1060.
    phases: int = 4


SHAPES = (
    FlowPhantom(),
    FlowPhantom(name='circle_small', a_mm=3., b_mm=3., umax_m_s=.3),
    FlowPhantom(name='ellipse', a_mm=7., b_mm=3.5, umax_m_s=.5),
    FlowPhantom(name='ellipse_oblique', a_mm=6., b_mm=4., umax_m_s=.45,
                direction=(1., 2., 3.), roll_rad=.37),
    FlowPhantom(name='annulus', kind='annulus', a_mm=7., b_mm=7., inner_mm=3.),
    FlowPhantom(name='womersley', kind='womersley', a_mm=5., b_mm=5.,
                drive_pa_m=100., phases=32),
)
RESOLUTIONS = ((1., 1., 1.), (2., 2., 2.), (3., 3., 3.), (1., 2., 3.))


def basis(spec):
    d = np.asarray(spec.direction, float)
    d /= np.linalg.norm(d)
    helper = np.eye(3)[np.argmin(np.abs(d))]
    u = np.cross(helper, d); u /= np.linalg.norm(u)
    v = np.cross(d, u)
    c, s = np.cos(spec.roll_rad), np.sin(spec.roll_rad)
    return d, c*u+s*v, -s*u+c*v


def geometry(spec, spacing, shape=None):
    """Return sample-centre coordinates; a fixed physical tube for every grid."""
    spacing = np.asarray(spacing, float)
    if spacing.shape != (3,) or np.any(~np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError('spacing must contain three positive finite mm values')
    d, u, v = basis(spec)
    half = abs(d)*spec.length_mm/2 + abs(u)*spec.a_mm + abs(v)*spec.b_mm + 4.
    shape = tuple((np.ceil(2*half/spacing).astype(int)+1).tolist()) if shape is None else tuple(shape)
    xyz = np.moveaxis(np.indices(shape, dtype=float), 0, -1)*spacing-half
    axial, x, y = xyz@d, xyz@u, xyz@v
    radial = (x/spec.a_mm)**2+(y/spec.b_mm)**2 <= 1.
    if spec.kind == 'annulus':
        radial &= x*x+y*y >= spec.inner_mm**2
    mask = radial & (np.abs(axial) <= spec.length_mm/2)
    return mask, x, y, axial, half


def analytical_solution(spec, x_mm, y_mm, times_s):
    """Return axial velocity, driving gradient, Q and signed wall shear.

    The pressure gradient is -G(t)*direction. Shear is the fluid traction along
    the flow direction, i.e. viscosity times the inward wall derivative.
    """
    x, y = np.asarray(x_mm)*1e-3, np.asarray(y_mm)*1e-3
    times = np.asarray(times_s, float)
    a, b, mu = spec.a_mm*1e-3, spec.b_mm*1e-3, spec.mu_pa_s
    r = np.hypot(x, y)
    if spec.kind == 'womersley':
        w = 2*np.pi/spec.period_s
        k = np.sqrt(-1j*w*spec.rho_kg_m3/mu)
        phase = np.exp(1j*w*times)
        coefficient = spec.harmonic_pa_m/(1j*w*spec.rho_kg_m3)
        oscillation = coefficient*(1-jv(0, k*r)/jv(0, k*a))
        speed = (spec.drive_pa_m/(4*mu)*(a*a-r*r))[..., None] + np.real(oscillation[..., None]*phase)
        drive = spec.drive_pa_m + np.real(spec.harmonic_pa_m*phase)
        q = np.pi*a**4*spec.drive_pa_m/(8*mu) + np.real(
            np.pi*a*a*coefficient*(1-2*jv(1, k*a)/(k*a*jv(0, k*a)))*phase)
        tau = spec.drive_pa_m*a/2 + np.real(-mu*coefficient*k*jv(1, k*a)/jv(0, k*a)*phase)
        shear = np.broadcast_to(tau, (2, len(times)))
    elif spec.kind == 'annulus':
        ri = spec.inner_mm*1e-3
        factor = (a*a-ri*ri)/np.log(a/ri)
        safe_r = np.maximum(r, ri)
        profile = spec.drive_pa_m/(4*mu)*(a*a-r*r-factor*np.log(a/safe_r))
        speed = np.repeat(profile[..., None], len(times), axis=-1)
        drive = np.full(len(times), spec.drive_pa_m)
        q = np.full(len(times), np.pi*spec.drive_pa_m/(8*mu)*(
            a**4-ri**4-(a*a-ri*ri)**2/np.log(a/ri)))
        shear = np.repeat((spec.drive_pa_m/4*np.array([2*a-factor/a, -2*ri+factor/ri]))[:, None], len(times), axis=1)
    elif spec.kind == 'ellipse':
        profile = spec.umax_m_s*(1-x*x/(a*a)-y*y/(b*b))
        speed = np.repeat(profile[..., None], len(times), axis=-1)
        drive = np.full(len(times), 2*mu*spec.umax_m_s*(1/(a*a)+1/(b*b)))
        q = np.full(len(times), np.pi*a*b*spec.umax_m_s/2)
        # Returned values are the extrema; complete angular shear is in truth.
        shear = np.repeat((2*mu*spec.umax_m_s/np.array([a, b]))[:, None], len(times), axis=1)
    else:
        raise ValueError(f'Unknown analytic flow kind: {spec.kind}')
    return speed, drive, q*1e6, shear


def create_analytic_phantom(spec=SHAPES[0], spacing=(1., 1., 1.)):
    mask, x, y, axial, center = geometry(spec, spacing)
    times = np.arange(spec.phases)*spec.period_s/spec.phases
    speed, drive, q, shear = analytical_solution(spec, x, y, times)
    d, u, v = basis(spec)
    flow = (np.where(mask[..., None], speed, 0.)[..., None]*d*100).astype(np.float32)
    origin = np.array([11., -7., 23.])
    # Lateral-wall quadrature, excluding the crop ends by 8 mm.
    theta = np.arange(128)*2*np.pi/128
    z = np.linspace(-spec.length_mm/2+8, spec.length_mm/2-8, 9)
    radius_pairs = [(spec.a_mm, spec.b_mm, 1)]
    if spec.kind == 'annulus':
        radius_pairs.append((spec.inner_mm, spec.inner_mm, -1))
    points, normals, stresses, wall_ids, areas = [], [], [], [], []
    for boundary, (a, b, sign) in enumerate(radius_pairs):
        edge = a*np.cos(theta)[:, None]*u+b*np.sin(theta)[:, None]*v
        normal = sign*(np.cos(theta)[:, None]*u/a+np.sin(theta)[:, None]*v/b)
        normal /= np.linalg.norm(normal, axis=1, keepdims=True)
        local = edge[None]+z[:, None, None]*d
        points.append((local+center+origin).reshape(-1, 3))
        normals.append(np.broadcast_to(normal, local.shape).reshape(-1, 3))
        if spec.kind == 'ellipse':
            tau = 2*spec.mu_pa_s*spec.umax_m_s*np.sqrt((np.cos(theta)/(a*1e-3))**2+(np.sin(theta)/(b*1e-3))**2)
            tau = np.repeat(tau[:, None], spec.phases, axis=1)
        else:
            tau = np.broadcast_to(shear[boundary], (len(theta), spec.phases))
        stresses.append(np.broadcast_to(tau[None, ..., None]*d, (len(z), len(theta), spec.phases, 3)).reshape(-1, spec.phases, 3))
        wall_ids.extend([boundary]*len(z)*len(theta))
        ds = np.sqrt((a*np.sin(theta))**2+(b*np.cos(theta))**2)*2*np.pi/len(theta)
        areas.append(np.tile(ds, len(z)))
    area = np.pi*(spec.a_mm*spec.b_mm-spec.inner_mm**2)
    truth = dict(times_s=times, pressure_gradient_pa_m=-drive[:, None]*d,
                 flowrate_ml_s=q, cross_section_area_mm2=np.asarray(area),
                 pressure_pa=(-axial[..., None]*1e-3*drive).astype(np.float32),
                 pressure_valid_mask=mask & (np.abs(axial) < spec.length_mm/2-8),
                 wall_points_mm=np.concatenate(points), wall_outward_normals=np.concatenate(normals),
                 wall_wss_vectors_pa=np.concatenate(stresses), wall_boundary_id=np.array(wall_ids),
                 wall_arc_weights_mm=np.concatenate(areas),
                 centerline_mm=center[None]+origin+np.linspace(-spec.length_mm/2+8, spec.length_mm/2-8, 65)[:, None]*d)
    params = dict(asdict(spec), spacing_mm=list(spacing), origin_mm=origin.tolist(),
                  center_world_mm=(center+origin).tolist(), direction_unit=d.tolist(),
                  truth_type='analytic_unforced_incompressible_newtonian',
                  spatial_samples='origin_plus_index_times_spacing',
                  velocity_units='cm/s', pressure_units='Pa', pressure_gradient_units='Pa/m',
                  wss_units='Pa', flowrate_units='mL/s', cap_exclusion_mm=8.,
                  centerline_applicable=spec.kind != 'annulus')
    data = dict(mag=np.broadcast_to((mask.astype(np.float32)*900+100)[..., None], mask.shape+(spec.phases,)),
                flow=flow, segmentation=np.broadcast_to(mask[..., None].astype(np.int16), mask.shape+(spec.phases,)),
                resolution=np.asarray(spacing), origin=origin, rr=1000*spec.period_s,
                venc=max(150., 1.1*float(np.max(np.abs(flow)))), truth=truth, parameters=params)
    from .phantom_references import add_discrete_references
    return add_discrete_references(data)


def save_analytic_phantom(path, data):
    """Save a normalized AutoFlow H5 plus independent analytic truth arrays."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects existing user data and earlier results.
    with h5py.File(path, 'x') as handle:
        for key, source in (('mag', 'mag'), ('flow', 'flow'), ('segmask', 'segmentation')):
            handle.create_dataset(key, data=data[source], compression='gzip', compression_opts=4)
        for key in ('resolution', 'origin', 'rr', 'venc'):
            handle[key] = data[key]
        handle['SpatialOrder'] = np.array(['LR', 'AP', 'FH'], dtype='S4')
        handle['VENCOrder'] = np.array(['LR', 'AP', 'FH'], dtype='S4')
        handle.attrs['autoflow_internal_spatial_order'] = 'XYZ'
        handle.attrs['analytic_parameters_json'] = json.dumps(data['parameters'])
        group = handle.create_group('truth')
        for key, value in data['truth'].items():
            value = np.asarray(value)
            group.create_dataset(key, data=value, **({'compression': 'gzip'} if value.ndim else {}))
    return path


def generate_matrix(directory, specs=SHAPES, resolutions=RESOLUTIONS):
    records = []
    for spec in specs:
        for spacing in resolutions:
            name = spec.name+'_'+('x'.join(f'{v:g}' for v in spacing))+'mm.h5'
            data = create_analytic_phantom(spec, spacing)
            path = save_analytic_phantom(Path(directory)/name, data)
            record = dict(file=path.name, shape=list(data['flow'].shape), **data['parameters'])
            records.append(record)
            print(path, flush=True)
    manifest = Path(directory)/'manifest.json'
    manifest.write_text(json.dumps(dict(schema='autoflow.analytic_phantoms.v1', cases=records), indent=2))
    return records
