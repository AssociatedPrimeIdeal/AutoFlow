import json
from pathlib import Path
import tempfile

import h5py
import numpy as np
import pytest

from autoflow import AutoFlowConfig, build_workspace, run_batch
from autoflow.algorithms.metrics import (
    _periodic_central_difference,
    compute_centerline_pressure_profiles,
    compute_pressure_gradient_metrics,
    reconstruct_relative_pressure_map,
)




@pytest.mark.parametrize(
    ("requested", "canonical"),
    [("least_squares", "ppe"), ("ppe", "ppe"), ("ste", "ste"), ("stokes", "ste")],
)
def test_pressure_method_aliases_are_canonical(requested, canonical):
    workspace = build_workspace(AutoFlowConfig(pressure_method=requested))
    assert workspace.derived_params.pressure_method == canonical


@pytest.mark.parametrize("method", ["least_squares", "ppe", "ste"])
@pytest.mark.parametrize("spacing", [(0.5, 0.5, 0.5), (1.0, 1.0, 1.0), (0.5, 1.5, 2.0)])
def test_pressure_reconstruction_matches_constant_gradient_in_pa(method, spacing):
    shape = (5, 6, 7, 3)
    gradient = np.zeros(shape + (3,), dtype=np.float32)
    coordinates_m = np.indices(shape[:3]).transpose(1, 2, 3, 0) * np.asarray(spacing) / 1000.0
    support = np.ones(shape, dtype=bool)
    expected = np.zeros(shape, dtype=float)
    for tidx in range(shape[3]):
        g = np.array([100.0, -30.0, 250.0]) * (tidx + 1)
        gradient[..., tidx, :] = g
        expected[..., tidx] = coordinates_m @ g
    actual = reconstruct_relative_pressure_map(gradient, support, spacing, method=method)
    np.testing.assert_allclose(actual["relative_pressure_array"], expected, rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize("method", ["least_squares", "ppe", "ste"])
def test_pressure_reconstruction_integrates_quadratic_pressure_on_irregular_support(method):
    spacing = np.array([0.8, 1.2, 1.7])
    xyz = np.indices((6, 6, 6)).transpose(1, 2, 3, 0) * spacing / 1000.0
    x, y, z = np.moveaxis(xyz, -1, 0)
    truth = 20000 * x**2 - 10000 * y**2 + 5000 * z**2 + 3000 * x * y
    gradient = np.stack((40000 * x + 3000 * y, -20000 * y + 3000 * x, 10000 * z), axis=-1)
    support = np.ones((6, 6, 6, 1), dtype=bool)
    support[2:4, 2:4, 2:4] = False
    actual = reconstruct_relative_pressure_map(gradient[..., None, :], support, spacing, method=method)
    np.testing.assert_allclose(actual["relative_pressure_array"][..., 0][support[..., 0]],
                               truth[support[..., 0]], rtol=3e-4, atol=3e-5)


@pytest.mark.parametrize("method", ["least_squares", "ppe", "ste"])
def test_pressure_reconstruction_anchors_each_disconnected_component(method):
    support = np.zeros((11, 4, 4, 1), dtype=bool)
    support[0:3, :, :, :] = True
    support[5:8, :, :, :] = True
    # A separate isolated voxel must have a well-defined zero reference.
    support[10, 3, 3, :] = True
    gradient = np.zeros(support.shape + (3,), dtype=np.float32)
    gradient[..., 0] = 100.0
    actual = reconstruct_relative_pressure_map(gradient, support, (1, 1, 1), method=method)
    p = actual["relative_pressure_array"][..., 0]
    assert p[2, 1, 1] - p[0, 1, 1] == pytest.approx(0.2, abs=2e-5)
    assert p[7, 1, 1] - p[5, 1, 1] == pytest.approx(0.2, abs=2e-5)
    assert p[0, 0, 0] == p[5, 0, 0] == p[10, 3, 3] == 0.0


@pytest.mark.parametrize("erosion", [0, 1])
@pytest.mark.parametrize("smoothing", [0.0, 1.0])
def test_uniform_flow_has_no_pg_from_zero_background_or_mask_smoothing(erosion, smoothing):
    mask = np.zeros((9, 9, 9, 8), dtype=bool)
    mask[2:7, 2:7, 2:7] = True
    flow = np.zeros(mask.shape + (3,), dtype=np.float32)
    flow[mask, 0] = 100.0
    result = compute_pressure_gradient_metrics(mask, flow, (1, 1, 1), rr=800,
                                               support_erosion_iters=erosion, smoothing_sigma=smoothing)
    assert np.any(result["pressure_gradient_support_mask"])
    assert not np.any(result["pressure_gradient_support_mask"][2, :, :])
    assert float(np.max(result["pressure_gradient_magnitude"])) < 0.2


def test_pressure_support_rejects_missing_temporal_and_nonfinite_spatial_samples():
    mask = np.zeros((9, 9, 9, 8), dtype=bool)
    mask[2:7, 2:7, 2:7] = True
    mask[4, 4, 4, 2] = False
    flow = np.zeros(mask.shape + (3,), dtype=np.float32)
    flow[..., 0] = 100.0
    flow[3, 3, 3, 5, 0] = np.nan
    result = compute_pressure_gradient_metrics(mask, flow, (1, 1, 1), rr=800, support_erosion_iters=0)
    support = result["pressure_gradient_support_mask"]
    assert support[4, 4, 4, 1] == support[4, 4, 4, 3] == 0
    assert support[4, 3, 3, 5] == 0
    assert np.isfinite(result["pressure_gradient_array"]).all()
    assert np.isfinite(result["relative_pressure_array"]).all()


def test_pressure_gradient_matches_analytic_periodic_acceleration():
    nt = 32
    mask = np.zeros((9, 9, 9, nt), dtype=bool)
    mask[2:7, 2:7, 2:7] = True
    phase = np.arange(nt) * 2.0 * np.pi / nt
    flow = np.zeros(mask.shape + (3,), dtype=np.float32)
    flow[..., 2] = 100.0 + 10.0 * np.sin(phase)
    result = compute_pressure_gradient_metrics(mask, flow, (1, 1, 1), rr=800)
    truth = -1060.0 * 0.1 * (2.0 * np.pi / 0.8) * np.cos(phase)
    measured = result["pressure_gradient_array"][4, 4, 4, :, 2]
    assert np.sqrt(np.mean((measured - truth)**2) / np.mean(truth**2)) < 0.01


def test_pressure_temporal_derivative_wraps_first_and_last_phases():
    phase = np.arange(8, dtype=np.float32)
    waveform = np.sin(2.0 * np.pi * phase / 8.0)
    values = waveform.reshape(1, 1, 1, 8, 1)
    derivative = _periodic_central_difference(values, 0.1)
    expected = (np.roll(waveform, -1) - np.roll(waveform, 1)) / 0.2

    assert derivative.reshape(-1) == pytest.approx(expected)
    assert abs(float(derivative[0, 0, 0, 0, 0])) > 0.0
    assert abs(float(derivative[0, 0, 0, -1, 0])) > 0.0


def test_centerline_pressure_sampling_uses_local_coordinates_with_nonzero_origin():
    pressure = np.zeros((5, 3, 3, 2), dtype=np.float32)
    pressure[:] = np.arange(5, dtype=np.float32).reshape(5, 1, 1, 1)
    path_local = np.array([[0.0, 1.0, 1.0], [4.0, 1.0, 1.0]], dtype=float)

    at_zero = compute_centerline_pressure_profiles(
        pressure, [path_local], (1.0, 1.0, 1.0), (0.0, 0.0, 0.0)
    )
    shifted = compute_centerline_pressure_profiles(
        pressure, [path_local], (1.0, 1.0, 1.0), (100.0, 200.0, 300.0)
    )

    assert shifted == at_zero
    assert shifted[0]["relative_pressure_Pa_t"][0] == pytest.approx([0.0, 4.0])


def test_centerline_pressure_rejects_unsupported_endpoints_and_disconnected_gauges():
    pressure = np.zeros((9, 3, 3, 2), dtype=np.float32)
    pressure[:] = np.arange(9, dtype=float).reshape(9, 1, 1, 1)
    support = np.zeros(pressure.shape, dtype=bool)
    support[1:4] = True
    support[5:8] = True
    paths = [np.array([[0., 1., 1.], [3., 1., 1.]]),
             np.array([[1., 1., 1.], [3., 1., 1.]]),
             np.array([[1., 1., 1.], [6., 1., 1.]])]
    profiles = compute_centerline_pressure_profiles(
        pressure, paths, (1, 1, 1), (50, 60, 70), support_mask=support)
    assert profiles[0]['relative_pressure_Pa_t'][0] == [None, 3.0]
    assert profiles[0]['pressure_drop_Pa_t'] == [None, None]
    assert profiles[0]['pressure_drop_mean_Pa'] is None
    assert profiles[1]['pressure_drop_Pa_t'] == pytest.approx([-2., -2.])
    assert all(profiles[1]['pressure_drop_valid_t'])
    assert profiles[2]['pressure_drop_Pa_t'] == [None, None]
    assert profiles[2]['pressure_drop_peak_Pa'] is None
    # The new unavailable samples remain valid JSON, with null rather than NaN.
    json.dumps(profiles, allow_nan=False)


@pytest.mark.parametrize("workers", [1, 2])
def test_plane_pressure_support_excludes_padding_and_preserves_true_zero(workers):
    from types import SimpleNamespace
    from autoflow.algorithms.metrics.plane_derived import augment_plane_metrics_with_derived
    mask = np.zeros((9, 9, 9, 3), dtype=bool)
    mask[1:8, 1:8, 1:8] = True
    support = np.zeros_like(mask)
    support[1:5, 1:8, 1:8, 0] = True
    support[..., 2] = mask[..., 2]
    gradient = np.zeros(mask.shape + (3,), dtype=np.float32)
    gradient[support[..., 0], 0, 2] = 3.0
    pressure = np.zeros(mask.shape, dtype=np.float32)
    pressure[..., 0][support[..., 0]] = 5.0
    plane = SimpleNamespace(center=np.array([4.5, 4.5, 4.5]), normal=np.array([0., 0., 1.]),
                            label=0, path_index=0, roi_edit_operations={})
    metrics, samples = augment_plane_metrics_with_derived(
        [{}, {}], [plane, plane], mask, (1, 1, 1), (0, 0, 0),
        pressure_gradient_array=gradient, relative_pressure_array=pressure,
        pressure_gradient_support_mask=support, max_workers=workers, use_multithread=workers > 1)
    for metric, payload in zip(metrics, samples):
        assert metric['pressure_gradient_mag_mean_Pa_m_t'] == [3.0, None, 0.0]
        assert metric['relative_pressure_mean_Pa_t'] == [5.0, None, 0.0]
        assert metric['pressure_gradient_mag_mean_Pa_m'] == pytest.approx(1.5)
        assert metric['relative_pressure_mean_Pa'] == pytest.approx(2.5)
        assert metric['pressure_gradient_valid_cell_count_t'][1] == 0
        assert not np.any(payload['timepoints'][1]['pressure_gradient_valid'])
        assert np.all(np.isnan(payload['timepoints'][1]['pressure_gradient_vec_Pa_m']))
        assert np.all(np.isfinite(payload['timepoints'][2]['pressure_gradient_vec_Pa_m']))
        json.dumps(metric, allow_nan=False)


@pytest.mark.parametrize("shape_index", range(6))
def test_analytic_flow_phantom_closed_form_matches_independent_quadrature_and_wall_derivative(shape_index):
    from autoflow.algorithms.analytic_phantoms import SHAPES, analytical_solution
    spec = SHAPES[shape_index]
    nodes, weights = np.polynomial.legendre.leggauss(72)
    theta = np.arange(144)*2*np.pi/144
    lo = spec.inner_mm/spec.a_mm if spec.kind == "annulus" else 0.
    r = lo+(nodes+1)*(1-lo)/2
    radial_weights = weights*(1-lo)/2
    x = spec.a_mm*r[:, None]*np.cos(theta)
    y = spec.b_mm*r[:, None]*np.sin(theta)
    times = np.arange(spec.phases)*spec.period_s/spec.phases
    speed, drive, q, shear = analytical_solution(spec, x, y, times)
    area_weights = np.broadcast_to(spec.a_mm*spec.b_mm*(radial_weights*r)[:, None]*2*np.pi/len(theta), x.shape)
    integrated = np.einsum("ij,ijt->t", area_weights, speed)
    np.testing.assert_allclose(integrated, q, rtol=2e-9, atol=2e-9)
    boundary = analytical_solution(spec, np.array([spec.a_mm]), np.array([0.]), times)[0]
    np.testing.assert_allclose(boundary, 0., atol=1e-12)
    epsilon_mm = 1e-4
    inward = analytical_solution(spec, np.array([spec.a_mm-epsilon_mm]), np.array([0.]), times)[0][0]
    derivative = spec.mu_pa_s*inward/(epsilon_mm*1e-3)
    np.testing.assert_allclose(derivative, shear[0], rtol=5e-4, atol=1e-4)
    if spec.kind == "womersley":
        assert np.min(shear[0]) < 0. < np.max(shear[0])


@pytest.mark.parametrize("shape_index", [0, 2, 3])
@pytest.mark.parametrize("spacing", [(1., 1., 1.), (2., 2., 2.), (1., 2., 3.)])
def test_analytic_poiseuille_pg_matches_prescribed_driving_gradient(shape_index, spacing):
    from autoflow.algorithms.analytic_phantoms import SHAPES, create_analytic_phantom
    data = create_analytic_phantom(SHAPES[shape_index], spacing)
    center = np.asarray(data['parameters']['center_world_mm'])-data['origin']
    index = np.rint(center/spacing).astype(int)
    roi = tuple(slice(max(0, int(i)-4), min(int(i)+5, n)) for i,n in zip(index,data['flow'].shape))
    flow = data['flow'][roi]
    mask = data['segmentation'][roi] > 0
    result = compute_pressure_gradient_metrics(mask, flow, spacing, data['rr'], viscosity=4., support_erosion_iters=1)
    support = np.asarray(result['pressure_gradient_support_mask'], dtype=bool)
    if not np.any(support):
        assert not np.allclose(spacing, 1.), "Resolved phantom must provide derivative support"
        return
    assert int(np.count_nonzero(support)) >= 8
    expected = np.broadcast_to(data['truth']['pressure_gradient_pa_m'], result['pressure_gradient_array'].shape)
    np.testing.assert_allclose(result['pressure_gradient_array'][support], expected[support], rtol=3e-3, atol=.15)
