"""Optional metric-family selection and combined derived result payloads."""

import numpy as np

from ._common import _ensure_mask4d
from .pressure import compute_pressure_gradient_metrics
from .tke import compute_tke_metrics
from .vortex import compute_vortex_metrics
from .wss import compute_wss_metrics


def compute_derived_metrics(mask4d, flow, spacing, origin=(0, 0, 0),
                            smoothing_iteration=200, viscosity=4.0,
                            inward_distance=None, parabolic_fitting=True,
                            no_slip_condition=True, step_size=5,
                            tube_radius=0.1, rho=1060.0,
                            save_pixelwise=False, tke_array=None, sigma=None,
                            rr=1000.0, pressure_gradient_smoothing_sigma=0.0,
                            pressure_gradient_support_erosion_iters=1,
                            pressure_gradient_use_convective_acceleration=True,
                            compute_wss=True, compute_tke=True,
                            compute_pressure_gradient=True,
                            compute_vortex=False,
                            wss_smoothing_iteration=None,
                            wss_viscosity=None,
                            wss_inward_distance=None,
                            wss_parabolic_fitting=None,
                            wss_no_slip_condition=None,
                            tke_rho=None,
                            pressure_gradient_rho=None,
                            pressure_gradient_viscosity=None,
                            vortex_smoothing_sigma=0.0,
                            vortex_support_erosion_iters=1,
                            pressure_method="ppe",
                            centerline_paths=None):
    mask4d = _ensure_mask4d(mask4d)
    wss_smoothing_iteration = smoothing_iteration if wss_smoothing_iteration is None else wss_smoothing_iteration
    wss_viscosity = viscosity if wss_viscosity is None else wss_viscosity
    wss_inward_distance = inward_distance if wss_inward_distance is None else wss_inward_distance
    wss_parabolic_fitting = parabolic_fitting if wss_parabolic_fitting is None else wss_parabolic_fitting
    wss_no_slip_condition = no_slip_condition if wss_no_slip_condition is None else wss_no_slip_condition
    tke_rho = rho if tke_rho is None else tke_rho
    pressure_gradient_rho = rho if pressure_gradient_rho is None else pressure_gradient_rho
    pressure_gradient_viscosity = viscosity if pressure_gradient_viscosity is None else pressure_gradient_viscosity
    wss = None
    if compute_wss:
        wss = compute_wss_metrics(
            mask4d, flow, spacing, origin=origin,
            smoothing_iteration=wss_smoothing_iteration,
            viscosity=wss_viscosity,
            inward_distance=wss_inward_distance,
            parabolic_fitting=wss_parabolic_fitting,
            no_slip_condition=wss_no_slip_condition,
        )
    tke = None
    if compute_tke and (tke_array is not None or sigma is not None):
        tke = compute_tke_metrics(
            mask4d, spacing, origin=origin, tke_array=tke_array, sigma=sigma, rho=tke_rho,
        )
    pressure_gradient = None
    if compute_pressure_gradient:
        pressure_gradient = compute_pressure_gradient_metrics(
            mask4d,
            flow,
            spacing,
            rr=rr,
            rho=pressure_gradient_rho,
            viscosity=pressure_gradient_viscosity,
            smoothing_sigma=pressure_gradient_smoothing_sigma,
            support_erosion_iters=pressure_gradient_support_erosion_iters,
            use_convective_acceleration=pressure_gradient_use_convective_acceleration,
            pressure_method=pressure_method,
            centerline_paths=centerline_paths,
            origin=origin,
        )
    vortex = None
    if compute_vortex:
        vortex = compute_vortex_metrics(
            mask4d,
            flow,
            spacing,
            smoothing_sigma=vortex_smoothing_sigma,
            support_erosion_iters=vortex_support_erosion_iters,
        )

    spacing = np.asarray(spacing, dtype=float).reshape(3)
    origin = np.asarray(origin, dtype=float).reshape(3)
    result = {
        "wss_surfaces": [] if wss is None else wss["wss_surfaces"],
        "wss_volume": None if wss is None else wss["wss_volume"],
        "tke_volume": None if tke is None else tke["tke_volume"],
        "tke_array": None if tke is None else tke["tke_array"],
        "tke_peak": None if tke is None else tke["tke_peak"],
        "pressure_gradient_array": None if pressure_gradient is None else pressure_gradient["pressure_gradient_array"],
        "pressure_gradient_magnitude": None if pressure_gradient is None else pressure_gradient["pressure_gradient_magnitude"],
        "pressure_gradient_peak": None if pressure_gradient is None else pressure_gradient["pressure_gradient_peak"],
        "pressure_gradient_dt_s": None if pressure_gradient is None else pressure_gradient["pressure_gradient_dt_s"],
        "pressure_gradient_temporal_scheme": (
            None if pressure_gradient is None else pressure_gradient["pressure_gradient_temporal_scheme"]
        ),
        "pressure_gradient_support_mask": None if pressure_gradient is None else pressure_gradient["pressure_gradient_support_mask"],
        "pressure_gradient_display_clim": None if pressure_gradient is None else pressure_gradient["pressure_gradient_display_clim"],
        "relative_pressure_array": None if pressure_gradient is None else pressure_gradient["relative_pressure_array"],
        "relative_pressure_peak": None if pressure_gradient is None else pressure_gradient["relative_pressure_peak"],
        "relative_pressure_display_clim": None if pressure_gradient is None else pressure_gradient["relative_pressure_display_clim"],
        "centerline_pressure_profiles": [] if pressure_gradient is None else pressure_gradient["centerline_pressure_profiles"],
        "pressure_method": None if pressure_gradient is None else pressure_gradient["pressure_method"],
        "vorticity_array": None if vortex is None else vortex["vorticity_array"],
        "vorticity_magnitude": None if vortex is None else vortex["vorticity_magnitude"],
        "vorticity_magnitude_peak": None if vortex is None else vortex["vorticity_magnitude_peak"],
        "q_criterion_array": None if vortex is None else vortex["q_criterion_array"],
        "q_criterion_peak": None if vortex is None else vortex["q_criterion_peak"],
        "swirling_strength_array": None if vortex is None else vortex["swirling_strength_array"],
        "swirling_strength_peak": None if vortex is None else vortex["swirling_strength_peak"],
        "vortex_support_mask": None if vortex is None else vortex["vortex_support_mask"],
        "streamlines": [],
        "tube_radius": float(tube_radius),
    }
    if save_pixelwise:
        pixelwise_export = {
            "spacing": np.asarray(spacing, dtype=np.float32),
            "origin": np.asarray(origin, dtype=np.float32),
        }
        if wss is not None:
            pixelwise_export["wss"] = np.asarray(wss["wss_volume"], dtype=np.float32)
        if pressure_gradient is not None:
            pixelwise_export["pressure_gradient"] = np.asarray(pressure_gradient["pressure_gradient_array"], dtype=np.float32)
            pixelwise_export["pressure_gradient_mag"] = np.asarray(pressure_gradient["pressure_gradient_magnitude"], dtype=np.float32)
            pixelwise_export["pressure_gradient_peak"] = np.asarray(pressure_gradient["pressure_gradient_peak"], dtype=np.float32)
            pixelwise_export["pressure_gradient_support_mask"] = np.asarray(pressure_gradient["pressure_gradient_support_mask"], dtype=np.uint8)
            pixelwise_export["relative_pressure"] = np.asarray(pressure_gradient["relative_pressure_array"], dtype=np.float32)
            pixelwise_export["relative_pressure_peak"] = np.asarray(pressure_gradient["relative_pressure_peak"], dtype=np.float32)
        if tke is not None:
            pixelwise_export["tke"] = np.asarray(tke["tke_peak"], dtype=np.float32)
            pixelwise_export["tke_time"] = np.asarray(tke["tke_array"], dtype=np.float32)
        if vortex is not None:
            pixelwise_export["vorticity"] = np.asarray(vortex["vorticity_array"], dtype=np.float32)
            pixelwise_export["vorticity_magnitude"] = np.asarray(vortex["vorticity_magnitude"], dtype=np.float32)
            pixelwise_export["vorticity_magnitude_peak"] = np.asarray(vortex["vorticity_magnitude_peak"], dtype=np.float32)
            pixelwise_export["q_criterion"] = np.asarray(vortex["q_criterion_array"], dtype=np.float32)
            pixelwise_export["q_criterion_peak"] = np.asarray(vortex["q_criterion_peak"], dtype=np.float32)
            pixelwise_export["swirling_strength"] = np.asarray(vortex["swirling_strength_array"], dtype=np.float32)
            pixelwise_export["swirling_strength_peak"] = np.asarray(vortex["swirling_strength_peak"], dtype=np.float32)
            pixelwise_export["vortex_support_mask"] = np.asarray(vortex["vortex_support_mask"], dtype=np.uint8)
        result["pixelwise_export"] = pixelwise_export
    else:
        result["pixelwise_export"] = {}
    return result
