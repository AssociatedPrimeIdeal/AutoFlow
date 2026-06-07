"""Generate the pressure-gradient phantom as ``data/phantom_P.h5``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
from scipy.special import jv


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
PHANTOM_H5_PATH = DATA_DIR / "phantom_P.h5"
TRUTH_GROUP_NAME = "truth"


@dataclass(frozen=True)
class PhantomConfig:
    radius_mm: float = 12.0
    length_mm: float = 48.0
    period_s: float = 0.8
    rho_kg_m3: float = 1060.0
    mu_pa_s: float = 0.004
    nx: int = 41
    ny: int = 41
    nz: int = 81
    nt: int = 49
    mag_inside: float = 1.0
    mag_outside: float = 0.05
    steady_dp_dz_pa_m: float = -70.0
    harmonic_dp_dz_pa_m: tuple[tuple[int, complex], ...] = (
        (1, complex(-900.0, 250.0)),
        (2, complex(350.0, -160.0)),
    )
    secondary_axial_amplitude_m_s: float = 0.14
    secondary_axial_second_harmonic_weight: float = 0.35
    swirl_modes: tuple[tuple[float, float], ...] = (
        (3.8317059702075125, 0.30),
        (7.015586669815619, -0.11),
    )

    @property
    def radius_m(self) -> float:
        return self.radius_mm / 1000.0

    @property
    def length_m(self) -> float:
        return self.length_mm / 1000.0

    @property
    def nu_m2_s(self) -> float:
        return self.mu_pa_s / self.rho_kg_m3

    @property
    def omega_rad_s(self) -> float:
        return 2.0 * np.pi / self.period_s

    @property
    def rr_ms(self) -> float:
        return self.period_s * 1000.0


def axial_pressure_gradient(times_s: np.ndarray, cfg: PhantomConfig) -> np.ndarray:
    times_s = np.asarray(times_s, dtype=float)
    out = np.full_like(times_s, float(cfg.steady_dp_dz_pa_m), dtype=float)
    for harmonic, coeff in cfg.harmonic_dp_dz_pa_m:
        phase = np.exp(1j * harmonic * cfg.omega_rad_s * times_s)
        out += np.real(coeff * phase)
    return out


def axial_secondary_profile(r_m: np.ndarray, cfg: PhantomConfig) -> np.ndarray:
    r_m = np.asarray(r_m, dtype=float)
    s = np.square(r_m / cfg.radius_m)
    profile = 1.0 - 8.0 * s + 18.0 * s**2 - 16.0 * s**3 + 5.0 * s**4
    return np.where(s <= 1.0, profile, 0.0)


def axial_secondary_radial_laplacian(r_m: np.ndarray, cfg: PhantomConfig) -> np.ndarray:
    r_m = np.asarray(r_m, dtype=float)
    s = np.square(r_m / cfg.radius_m)
    dq_ds = -8.0 + 36.0 * s - 48.0 * s**2 + 20.0 * s**3
    d2q_ds2 = 36.0 - 96.0 * s + 60.0 * s**2
    laplacian = 4.0 * (s * d2q_ds2 + dq_ds) / (cfg.radius_m**2)
    return np.where(s <= 1.0, laplacian, 0.0)


def axial_secondary_waveform(
    z_m: np.ndarray,
    t_s: float,
    cfg: PhantomConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    z_m = np.asarray(z_m, dtype=float)
    kz = 2.0 * np.pi / cfg.length_m
    omega = cfg.omega_rad_s
    mix = float(cfg.secondary_axial_second_harmonic_weight)

    phase1 = kz * z_m
    phase2 = 2.0 * phase1
    time1 = omega * t_s + 0.35
    time2 = 2.0 * omega * t_s - 0.65

    waveform = np.sin(phase1) * np.cos(time1) + mix * np.cos(phase2) * np.sin(time2)
    d_wave_dt = -omega * np.sin(phase1) * np.sin(time1) + 2.0 * omega * mix * np.cos(phase2) * np.cos(time2)
    d_wave_dz = kz * np.cos(phase1) * np.cos(time1) - 2.0 * kz * mix * np.sin(phase2) * np.sin(time2)
    d2_wave_dz2 = -(kz**2) * np.sin(phase1) * np.cos(time1) - 4.0 * (kz**2) * mix * np.cos(phase2) * np.sin(time2)
    return waveform, d_wave_dt, d_wave_dz, d2_wave_dz2


def base_axial_velocity_m_s(r_m: np.ndarray, t_s: float, cfg: PhantomConfig) -> np.ndarray:
    r_m = np.asarray(r_m, dtype=float)
    radius = cfg.radius_m
    rho = cfg.rho_kg_m3
    mu = cfg.mu_pa_s

    uz = -cfg.steady_dp_dz_pa_m * (radius**2 - r_m**2) / (4.0 * mu)
    for harmonic, coeff in cfg.harmonic_dp_dz_pa_m:
        omega = harmonic * cfg.omega_rad_s
        beta = np.sqrt(-1j * omega * rho / mu)
        c = -coeff / (1j * omega * rho)
        shape = 1.0 - jv(0, beta * r_m) / jv(0, beta * radius)
        uz += np.real(c * shape * np.exp(1j * omega * t_s))
    return uz


def axial_velocity_m_s(r_m: np.ndarray, z_m: np.ndarray | float, t_s: float, cfg: PhantomConfig) -> np.ndarray:
    base = base_axial_velocity_m_s(r_m, t_s, cfg)
    profile = axial_secondary_profile(r_m, cfg)
    waveform, _, _, _ = axial_secondary_waveform(z_m, t_s, cfg)
    amplitude = float(cfg.secondary_axial_amplitude_m_s)

    if np.ndim(waveform) == 0:
        return base + amplitude * profile * float(waveform)
    return base[..., None] + amplitude * profile[..., None] * waveform


def swirl_velocity_m_s(r_m: np.ndarray, t_s: float, cfg: PhantomConfig) -> np.ndarray:
    r_m = np.asarray(r_m, dtype=float)
    out = np.zeros_like(r_m, dtype=float)
    for root, coeff_m_s in cfg.swirl_modes:
        decay = np.exp(-cfg.nu_m2_s * (root / cfg.radius_m) ** 2 * t_s)
        out += coeff_m_s * jv(1, root * r_m / cfg.radius_m) * decay
    return out


def true_pressure_gradient_pa_per_m(
    x_m: np.ndarray,
    y_m: np.ndarray,
    z_m: np.ndarray,
    times_s: np.ndarray,
    cfg: PhantomConfig,
) -> np.ndarray:
    x_m = np.asarray(x_m, dtype=float)
    y_m = np.asarray(y_m, dtype=float)
    z_m = np.asarray(z_m, dtype=float)
    times_s = np.asarray(times_s, dtype=float)
    r_m = np.hypot(x_m, y_m)
    safe_r = np.where(r_m > 1e-12, r_m, 1.0)
    secondary_profile = axial_secondary_profile(r_m, cfg)
    secondary_radial_laplacian = axial_secondary_radial_laplacian(r_m, cfg)
    secondary_amplitude = float(cfg.secondary_axial_amplitude_m_s)

    grad = np.zeros(x_m.shape + (len(z_m), len(times_s), 3), dtype=float)
    for tidx, t_s in enumerate(times_s):
        u_theta = swirl_velocity_m_s(r_m, float(t_s), cfg)
        dp_dr = np.where(r_m > 1e-12, cfg.rho_kg_m3 * u_theta**2 / safe_r, 0.0)
        grad[:, :, :, tidx, 0] = (dp_dr * x_m / safe_r)[:, :, None]
        grad[:, :, :, tidx, 1] = (dp_dr * y_m / safe_r)[:, :, None]

        base_uz = base_axial_velocity_m_s(r_m, float(t_s), cfg)
        base_grad = float(axial_pressure_gradient(np.array([t_s], dtype=float), cfg)[0])
        waveform, d_wave_dt, d_wave_dz, d2_wave_dz2 = axial_secondary_waveform(z_m, float(t_s), cfg)
        secondary_uz = secondary_amplitude * secondary_profile[..., None] * waveform[None, None, :]
        secondary_dt = secondary_amplitude * secondary_profile[..., None] * d_wave_dt[None, None, :]
        secondary_dz = secondary_amplitude * secondary_profile[..., None] * d_wave_dz[None, None, :]
        secondary_laplacian = secondary_amplitude * (
            secondary_radial_laplacian[..., None] * waveform[None, None, :]
            + secondary_profile[..., None] * d2_wave_dz2[None, None, :]
        )
        grad[:, :, :, tidx, 2] = base_grad + (
            -cfg.rho_kg_m3 * (secondary_dt + (base_uz[..., None] + secondary_uz) * secondary_dz)
            + cfg.mu_pa_s * secondary_laplacian
        )
    return grad


def true_wss_components_pa(times_s: np.ndarray, cfg: PhantomConfig) -> np.ndarray:
    radius = cfg.radius_m
    rho = cfg.rho_kg_m3
    mu = cfg.mu_pa_s
    times_s = np.asarray(times_s, dtype=float)

    tau_z = np.full_like(times_s, -cfg.steady_dp_dz_pa_m * radius / 2.0, dtype=float)
    for harmonic, coeff in cfg.harmonic_dp_dz_pa_m:
        omega = harmonic * cfg.omega_rad_s
        beta = np.sqrt(-1j * omega * rho / mu)
        c = -coeff / (1j * omega * rho)
        du_dr_wall = c * beta * jv(1, beta * radius) / jv(0, beta * radius)
        phase = np.exp(1j * omega * times_s)
        tau_z += np.real(mu * (-du_dr_wall) * phase)

    tau_theta = np.zeros_like(times_s, dtype=float)
    for root, coeff_m_s in cfg.swirl_modes:
        decay = np.exp(-cfg.nu_m2_s * (root / radius) ** 2 * times_s)
        du_dr_wall = coeff_m_s * (root / radius) * jv(0, root) * decay
        tau_theta += mu * (-du_dr_wall)

    return np.stack([tau_theta, tau_z], axis=-1)


def centerline_flowrate_ml_s(times_s: np.ndarray, cfg: PhantomConfig) -> np.ndarray:
    times_s = np.asarray(times_s, dtype=float)
    radius = cfg.radius_m
    mu = cfg.mu_pa_s
    area_scale = np.pi * radius**4 / (8.0 * mu)

    q = -cfg.steady_dp_dz_pa_m * area_scale * np.ones_like(times_s)
    for harmonic, coeff in cfg.harmonic_dp_dz_pa_m:
        omega = harmonic * cfg.omega_rad_s
        beta = np.sqrt(-1j * omega * cfg.rho_kg_m3 / mu)
        f_beta = 1.0 - (2.0 * jv(1, beta * radius)) / (beta * radius * jv(0, beta * radius))
        q_complex = -(np.pi * radius**2) * coeff / (1j * omega * cfg.rho_kg_m3) * f_beta
        q += np.real(q_complex * np.exp(1j * omega * times_s))
    return q * 1.0e6


def relative_rms_error(estimated: np.ndarray, truth: np.ndarray) -> float:
    estimated = np.asarray(estimated, dtype=float)
    truth = np.asarray(truth, dtype=float)
    denom = np.sqrt(np.mean(truth**2)) + 1e-12
    return float(np.sqrt(np.mean((estimated - truth) ** 2)) / denom)


def estimate_pressure_gradient_from_velocity_m_s(
    velocity_xyz_t3_m_s: np.ndarray,
    spacing_m: tuple[float, float, float],
    dt_s: float,
    rho_kg_m3: float,
    mu_pa_s: float,
) -> np.ndarray:
    v = np.asarray(velocity_xyz_t3_m_s, dtype=float)
    dx, dy, dz = [float(s) for s in spacing_m]
    vc = v[1:-1, 1:-1, 1:-1, 1:-1, :]
    du_dt = (v[1:-1, 1:-1, 1:-1, 2:, :] - v[1:-1, 1:-1, 1:-1, :-2, :]) / (2.0 * dt_s)

    conv = np.zeros_like(vc)
    lap = np.zeros_like(vc)
    for comp in range(3):
        du_dx = (v[2:, 1:-1, 1:-1, 1:-1, comp] - v[:-2, 1:-1, 1:-1, 1:-1, comp]) / (2.0 * dx)
        du_dy = (v[1:-1, 2:, 1:-1, 1:-1, comp] - v[1:-1, :-2, 1:-1, 1:-1, comp]) / (2.0 * dy)
        du_dz = (v[1:-1, 1:-1, 2:, 1:-1, comp] - v[1:-1, 1:-1, :-2, 1:-1, comp]) / (2.0 * dz)
        d2u_dx2 = (
            v[2:, 1:-1, 1:-1, 1:-1, comp]
            - 2.0 * v[1:-1, 1:-1, 1:-1, 1:-1, comp]
            + v[:-2, 1:-1, 1:-1, 1:-1, comp]
        ) / (dx * dx)
        d2u_dy2 = (
            v[1:-1, 2:, 1:-1, 1:-1, comp]
            - 2.0 * v[1:-1, 1:-1, 1:-1, 1:-1, comp]
            + v[1:-1, :-2, 1:-1, 1:-1, comp]
        ) / (dy * dy)
        d2u_dz2 = (
            v[1:-1, 1:-1, 2:, 1:-1, comp]
            - 2.0 * v[1:-1, 1:-1, 1:-1, 1:-1, comp]
            + v[1:-1, 1:-1, :-2, 1:-1, comp]
        ) / (dz * dz)
        conv[..., comp] = vc[..., 0] * du_dx + vc[..., 1] * du_dy + vc[..., 2] * du_dz
        lap[..., comp] = d2u_dx2 + d2u_dy2 + d2u_dz2

    return -rho_kg_m3 * (du_dt + conv) + mu_pa_s * lap


def build_phantom(cfg: PhantomConfig):
    x_mm = np.linspace(-cfg.radius_mm, cfg.radius_mm, cfg.nx)
    y_mm = np.linspace(-cfg.radius_mm, cfg.radius_mm, cfg.ny)
    z_mm = np.linspace(-0.5 * cfg.length_mm, 0.5 * cfg.length_mm, cfg.nz)
    times_s = np.linspace(0.0, cfg.period_s, cfg.nt)

    x_m = x_mm / 1000.0
    y_m = y_mm / 1000.0
    z_m = z_mm / 1000.0

    xx_m, yy_m = np.meshgrid(x_m, y_m, indexing="ij")
    rr_m = np.hypot(xx_m, yy_m)
    safe_r = np.where(rr_m > 1e-12, rr_m, 1.0)
    mask_xy = rr_m <= cfg.radius_m

    velocity_xyz_t3_m_s = np.zeros((cfg.nx, cfg.ny, cfg.nz, cfg.nt, 3), dtype=np.float32)
    for tidx, t_s in enumerate(times_s):
        u_theta = swirl_velocity_m_s(rr_m, float(t_s), cfg)
        velocity_xyz_t3_m_s[:, :, :, tidx, 0] = (-u_theta * yy_m / safe_r)[:, :, None].astype(np.float32)
        velocity_xyz_t3_m_s[:, :, :, tidx, 1] = (u_theta * xx_m / safe_r)[:, :, None].astype(np.float32)
        velocity_xyz_t3_m_s[:, :, :, tidx, 2] = axial_velocity_m_s(rr_m, z_m, float(t_s), cfg).astype(np.float32)

    velocity_xyz_t3_m_s *= mask_xy[:, :, None, None, None].astype(np.float32)
    flow_xyz_t3_cm_s = velocity_xyz_t3_m_s * 100.0

    mag_xy = np.where(mask_xy, cfg.mag_inside, cfg.mag_outside).astype(np.float32)
    mag_xyzt = np.repeat(mag_xy[:, :, None, None], cfg.nz, axis=2)
    mag_xyzt = np.repeat(mag_xyzt, cfg.nt, axis=3)

    seg_xyz = np.repeat(mask_xy[:, :, None], cfg.nz, axis=2).astype(np.int16)
    seg_xyzt = np.repeat(seg_xyz[..., None], cfg.nt, axis=3)

    grad_xyz_t3 = true_pressure_gradient_pa_per_m(xx_m, yy_m, z_m, times_s, cfg).astype(np.float32)
    grad_xyz_t3 *= mask_xy[:, :, None, None, None].astype(np.float32)

    tau_components_pa = true_wss_components_pa(times_s, cfg).astype(np.float32)
    flowrate_ml_s = centerline_flowrate_ml_s(times_s, cfg).astype(np.float32)

    centerline_path = np.column_stack(
        [
            np.zeros(cfg.nz, dtype=np.float32),
            np.zeros(cfg.nz, dtype=np.float32),
            z_mm.astype(np.float32),
        ]
    )
    plane_area_mm2 = np.full(cfg.nt, np.pi * (cfg.radius_mm ** 2), dtype=np.float32)

    return {
        "config": cfg,
        "x_mm": x_mm.astype(np.float32),
        "y_mm": y_mm.astype(np.float32),
        "z_mm": z_mm.astype(np.float32),
        "x_m": x_m.astype(np.float32),
        "y_m": y_m.astype(np.float32),
        "z_m": z_m.astype(np.float32),
        "times_s": times_s.astype(np.float32),
        "mask_xy": mask_xy.astype(bool),
        "segmentation_xyzt": seg_xyzt,
        "mag_xyzt": mag_xyzt.astype(np.float32),
        "flow_xyzt3_cm_s": flow_xyz_t3_cm_s.astype(np.float32),
        "flow_xyzt3_m_s": velocity_xyz_t3_m_s.astype(np.float32),
        "pressure_gradient_xyzt3_pa_m": grad_xyz_t3.astype(np.float32),
        "wss_components_pa": tau_components_pa,
        "flowrate_center_plane_ml_s": flowrate_ml_s,
        "centerline_path_mm": centerline_path,
        "plane_center_mm": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "plane_normal": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        "plane_area_mm2": plane_area_mm2,
        "resolution_mm": np.array(
            [
                float(x_mm[1] - x_mm[0]),
                float(y_mm[1] - y_mm[0]),
                float(z_mm[1] - z_mm[0]),
            ],
            dtype=np.float32,
        ),
        "origin_mm": np.array([x_mm[0], y_mm[0], z_mm[0]], dtype=np.float32),
        "venc_cm_s": np.array([150.0, 150.0, 150.0], dtype=np.float32),
    }


def truth_to_serializable(truth: dict) -> dict:
    cfg = truth["config"]
    meta = np.array(
        [
            cfg.radius_mm,
            cfg.length_mm,
            cfg.period_s,
            cfg.rho_kg_m3,
            cfg.mu_pa_s,
            cfg.rr_ms,
        ],
        dtype=np.float32,
    )
    harmonic = np.array(
        [[float(h), float(c.real), float(c.imag)] for h, c in cfg.harmonic_dp_dz_pa_m],
        dtype=np.float32,
    )
    swirl = np.array([[float(root), float(coeff)] for root, coeff in cfg.swirl_modes], dtype=np.float32)
    secondary = np.array(
        [
            cfg.secondary_axial_amplitude_m_s,
            cfg.secondary_axial_second_harmonic_weight,
        ],
        dtype=np.float32,
    )

    payload = {key: value for key, value in truth.items() if key != "config"}
    payload["config_summary"] = meta
    payload["harmonic_dp_dz_terms"] = harmonic
    payload["swirl_modes"] = swirl
    payload["secondary_axial_mode"] = secondary
    return payload


def load_truth_h5(path: Path = PHANTOM_H5_PATH, group_name: str = TRUTH_GROUP_NAME) -> dict:
    with h5py.File(path, "r") as handle:
        if group_name not in handle:
            raise KeyError(f"missing truth group '{group_name}' in {path}")
        group = handle[group_name]
        return {key: group[key][()] for key in group.keys()}


def save_truth_h5(group: h5py.Group, truth: dict) -> None:
    serializable = truth_to_serializable(truth)
    if TRUTH_GROUP_NAME in group:
        del group[TRUTH_GROUP_NAME]
    truth_group = group.create_group(TRUTH_GROUP_NAME)
    for key, value in serializable.items():
        truth_group.create_dataset(key, data=np.asarray(value))


def save_normalized_h5(path: Path, truth: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle["flow"] = np.asarray(truth["flow_xyzt3_cm_s"], dtype=np.float32)
        handle["mag"] = np.asarray(truth["mag_xyzt"], dtype=np.float32)
        handle["segmentation"] = np.asarray(truth["segmentation_xyzt"], dtype=np.int16)
        handle["Resolution"] = np.asarray(truth["resolution_mm"], dtype=np.float32)
        handle["Origin"] = np.asarray(truth["origin_mm"], dtype=np.float32)
        handle["VENC"] = np.asarray(truth["venc_cm_s"], dtype=np.float32)
        handle["RR"] = np.float32(truth["config"].rr_ms)
        handle["SpatialOrder"] = np.asarray(["LR", "AP", "FH"], dtype="S2")
        handle["VENCOrder"] = np.asarray(["LR", "AP", "FH"], dtype="S2")
        save_truth_h5(handle, truth)


def generate_and_save(
    phantom_h5_path: Path = PHANTOM_H5_PATH,
    cfg: PhantomConfig | None = None,
) -> dict:
    truth = build_phantom(cfg or PhantomConfig())
    save_normalized_h5(Path(phantom_h5_path), truth)
    return truth


def summary_lines(truth: dict, phantom_h5_path: Path) -> Iterable[str]:
    cfg = truth["config"]
    flow_cm_s = np.asarray(truth["flow_xyzt3_cm_s"], dtype=float)
    grad = np.asarray(truth["pressure_gradient_xyzt3_pa_m"], dtype=float)
    mask = np.asarray(truth["segmentation_xyzt"], dtype=bool)
    flow_mag = np.linalg.norm(flow_cm_s, axis=-1)
    grad_mag = np.linalg.norm(grad, axis=-1)
    flow_vals = flow_mag[mask]
    grad_vals = grad_mag[mask]
    axial_grad_vals = np.abs(grad[..., 2])[mask]
    yield "Pressure-gradient phantom generated"
    yield f"  h5: {phantom_h5_path}"
    yield f"  truth: embedded at /{TRUTH_GROUP_NAME}"
    yield f"  shape XYZT: {flow_cm_s.shape[:4]}"
    yield f"  spacing mm: {truth['resolution_mm'].tolist()}"
    yield f"  RR ms: {cfg.rr_ms:.3f}"
    yield f"  flow |v| range inside mask cm/s: {flow_vals.min():.3f} .. {flow_vals.max():.3f}"
    yield f"  |grad p| range inside mask Pa/m: {grad_vals.min():.3f} .. {grad_vals.max():.3f}"
    yield f"  |dp/dz| range inside mask Pa/m: {axial_grad_vals.min():.3f} .. {axial_grad_vals.max():.3f}"
    yield (
        "  true |WSS| range Pa: "
        f"{np.linalg.norm(truth['wss_components_pa'], axis=1).min():.5f} .. "
        f"{np.linalg.norm(truth['wss_components_pa'], axis=1).max():.5f}"
    )


def main() -> None:
    truth = generate_and_save()
    for line in summary_lines(truth, PHANTOM_H5_PATH):
        print(line)


if __name__ == "__main__":
    main()
