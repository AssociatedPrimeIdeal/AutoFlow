"""PC-MRA masking with magnitude and temporal-speed SD thresholds.

Magnitude and temporal-SD masking follow Bock et al., ISMRM 2007, abstract 3138,
and ISMRM 2008, abstract 3053. AutoFlow defaults are 5% of maximum magnitude
and 80% of maximum temporal SD, not the papers' empirical 9% and 20% settings.
Temporal-mean magnitude and omission of static-tissue removal remain AutoFlow
choices, not a verbatim reproduction.
Acquired magnitude/velocity and nnUNet feature inputs are never modified here.
"""
import numpy as np


def pcmra_render_mask(mag, flow, venc, *, method="magnitude_temporal",
                      magnitude_fraction=0.05, velocity_std_max=0.80):
    from ..case_types import NoiseRemovalConfig
    cfg = NoiseRemovalConfig(method=method, magnitude_fraction=magnitude_fraction,
                             velocity_std_max=velocity_std_max)
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
