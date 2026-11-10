"""PC-MRA display masking and continuous spatiotemporal coherence weights.

Magnitude and temporal-SD masking follow Bock et al., ISMRM 2007, abstract 3138,
and ISMRM 2008, abstract 3053. AutoFlow defaults are 5% of maximum magnitude
and 80% of maximum temporal SD, not the papers' empirical 9% and 20% settings.
Temporal-mean magnitude and omission of static-tissue removal remain AutoFlow
choices, not a verbatim reproduction.
Acquired magnitude/velocity and nnUNet feature inputs are never modified here.
Coherence weighting is a separate display heuristic, not a noise probability
or a dual-VENC correction. It is never a quantitative velocity/ROI weight.
"""
import numpy as np


COHERENCE_RAMPS = {"medium": (0.55, 0.92), "strong": (0.70, 0.98)}


def pcmra_coherence_weight(mag, flow, *, strength="strong", backend="auto"):
    """Return a static XYZ display weight, leaving all input buffers intact."""
    from scipy.ndimage import gaussian_filter
    from ..task_control import check_cancelled, TaskCancelled
    if backend not in {"auto", "cpu", "cuda"}:
        raise ValueError("Coherence backend must be auto, cpu or cuda")
    strength = str(strength).strip().lower()
    if strength not in COHERENCE_RAMPS:
        raise ValueError("coherence strength must be medium or strong")
    velocity = np.asarray(flow)
    magnitude = np.asarray(mag)
    if velocity.ndim != 5 or velocity.shape[-1] != 3 or velocity.shape[3] < 1:
        raise ValueError("flow must have shape XYZT3 with at least one phase")
    if magnitude.ndim == 3:
        magnitude = magnitude[..., None]
    if magnitude.ndim != 4 or magnitude.shape[:3] != velocity.shape[:3] or magnitude.shape[3] not in (1, velocity.shape[3]):
        raise ValueError("mag must match flow spatial dimensions and have 1 or T frames")
    check_cancelled()
    score = None
    used_backend = "cpu"
    if backend == "cuda" or (backend == "auto" and velocity.size >= 262_144):
        try:
            from .flow_cuda import coherence_score_cuda
            score = coherence_score_cuda(magnitude, velocity)
            used_backend = "cuda"
        except TaskCancelled:
            raise
        except Exception:
            if backend == "cuda":
                raise
    if score is None:
        score = _coherence_score_cpu(magnitude, velocity)
    check_cancelled()
    score = gaussian_filter(score, sigma=0.6, mode="reflect")
    low, high = COHERENCE_RAMPS[strength]
    ramp = np.clip((score - low) / (high - low), 0, 1)
    weight = np.asarray(ramp * ramp * (3 - 2 * ramp), dtype=np.float32)
    return weight, {"coherence_enabled": True, "coherence_strength": strength,
                    "coherence_ramp": [low, high], "coherence_spatial_window_voxels": 3,
                    "coherence_temporal_window_phases": 3, "coherence_temporal_percentile": 90,
                    "coherence_smoothing_sigma_voxels": 0.6, "coherence_time_boundary": "cyclic",
                    "coherence_mean_display_weight": float(np.mean(weight)),
                    "coherence_backend": used_backend,
                    "coherence_scope": "pcmra_rendering_only"}


def _coherence_score_cpu(magnitude, velocity):
    from scipy.ndimage import uniform_filter
    from ..task_control import check_cancelled, report_progress
    # Sanitizing uses private arrays: readonly acquired arrays are valid inputs.
    v = np.nan_to_num(velocity, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    m = np.maximum(np.nan_to_num(magnitude, nan=0.0, posinf=0.0, neginf=0.0), 0).astype(np.float32, copy=False)
    modes = ("reflect", "reflect", "reflect", "wrap")
    denominator = uniform_filter(m * np.linalg.norm(v, axis=-1), size=(3, 3, 3, 3), mode=modes)
    numerator = np.zeros_like(denominator)
    for component in range(3):
        check_cancelled()
        moment = uniform_filter(m * v[..., component], size=(3, 3, 3, 3), mode=modes)
        numerator += moment * moment
        report_progress({"stage": "noise_coherence", "current": component + 1, "total": 3,
                         "message": "Calculating PC-MRA display coherence"})
    np.sqrt(numerator, out=numerator)
    score = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-12)
    np.clip(score, 0, 1, out=score)
    check_cancelled()
    return np.percentile(score, 90, axis=3, overwrite_input=True).astype(np.float32)


def pcmra_display_frame(pcmra, phase=0, *, mask=None, weight=None):
    """Apply display filters to a private XYZ frame, never to stored PC-MRA."""
    source = np.asarray(pcmra)
    if source.ndim not in (3, 4):
        raise ValueError("PC-MRA must have shape XYZ or XYZT")
    tidx = min(max(0, int(phase)), source.shape[3] - 1) if source.ndim == 4 else 0
    values = np.array(source[..., tidx] if source.ndim == 4 else source, dtype=np.float32, copy=True)
    for field, binary in ((mask, True), (weight, False)):
        if field is None:
            continue
        region = np.asarray(field)
        if region.ndim == 4:
            region = region[..., min(tidx, region.shape[3] - 1)]
        if region.shape != values.shape:
            raise ValueError("PC-MRA display mask/weight must match the acquired spatial grid")
        if binary:
            values = np.where(region, values, 0.0)
        else:
            values *= np.clip(np.nan_to_num(region, nan=0.0, posinf=0.0, neginf=0.0), 0, 1)
    return values


def pcmra_render_mask(mag, flow, venc, *, method="magnitude_temporal",
                      magnitude_fraction=0.05, velocity_std_max=0.80):
    from ..case_types import NoiseRemovalConfig
    cfg = NoiseRemovalConfig(method=method, magnitude_fraction=magnitude_fraction,
                             velocity_std_max=velocity_std_max, coherence_enabled=False)
    velocity = np.asarray(flow)
    magnitude = np.asarray(mag)
    if velocity.ndim != 5 or velocity.shape[-1] != 3:
        raise ValueError("flow must have shape XYZT3")
    if magnitude.ndim == 3:
        magnitude = magnitude[..., None]
    if magnitude.ndim != 4 or magnitude.shape[:3] != velocity.shape[:3] or magnitude.shape[3] not in (1, velocity.shape[3]):
        raise ValueError("mag must match flow spatial dimensions and have 1 or T frames")
    finite_m = np.all(np.isfinite(magnitude), axis=3)
    mean_m = np.mean(np.nan_to_num(magnitude, nan=0.0, posinf=0.0, neginf=0.0), axis=3, dtype=np.float64)
    positive = mean_m[finite_m & (mean_m > 0)]
    magnitude_max = float(np.max(positive)) if positive.size else 0.0
    threshold = cfg.magnitude_fraction * magnitude_max
    mode = "fraction_of_max" if positive.size else "no_positive_signal"
    # A static spatial mask avoids flicker across cardiac phases and survives a
    # later unwrap. It is a display region, never a vessel segmentation.
    finite_v = np.all(np.isfinite(velocity), axis=(3, 4))
    mask = finite_m & finite_v & (mean_m > threshold)
    temporal_applied = cfg.method == "magnitude_temporal" and velocity.shape[3] >= 2 and cfg.velocity_std_max > 0
    temporal_max = None
    temporal_threshold = None
    if temporal_applied:
        speed = np.hypot(velocity[..., 0], velocity[..., 1], dtype=np.float64)
        np.hypot(speed, velocity[..., 2], out=speed)
        np.nan_to_num(speed, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        speed -= speed[..., :1].copy()
        temporal_sd = np.std(speed, axis=3)
        temporal_max = float(np.max(temporal_sd[finite_v])) if np.any(finite_v) else 0.0
        temporal_threshold = cfg.velocity_std_max * temporal_max
        mask &= temporal_sd <= temporal_threshold
    return mask, {"parameters": cfg.to_dict(), "magnitude_threshold": threshold,
                  "magnitude_threshold_mode": mode, "magnitude_reference_max": magnitude_max,
                  "magnitude_statistic": "temporal_mean", "temporal_screening_applied": temporal_applied,
                  "temporal_std_threshold": temporal_threshold, "temporal_std_reference_max": temporal_max,
                  "temporal_std_statistic": "speed",
                  "retained_fraction": float(np.mean(mask)) if mask.size else 0.0,
                  "scope": "pcmra_rendering_only",
                  "reference": "Bock et al., ISMRM 2007, 3138; ISMRM 2008, 3053"}
