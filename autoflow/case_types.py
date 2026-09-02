import copy
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class LoaderCapabilities:
    has_segmentation: bool = False
    has_tke: bool = False
    has_complex_source: bool = False
    supports_wss: bool = False
    supports_plane_metrics: bool = False
    has_wrapped_phase: bool = False
    supports_phase_unwrap: bool = False

    def to_dict(self):
        return {
            "has_segmentation": bool(self.has_segmentation),
            "has_tke": bool(self.has_tke),
            "has_complex_source": bool(self.has_complex_source),
            "supports_wss": bool(self.supports_wss),
            "supports_plane_metrics": bool(self.supports_plane_metrics),
            "has_wrapped_phase": bool(self.has_wrapped_phase),
            "supports_phase_unwrap": bool(self.supports_phase_unwrap),
        }

    @staticmethod
    def from_dict(d):
        return LoaderCapabilities(
            has_segmentation=bool(d.get("has_segmentation", False)),
            has_tke=bool(d.get("has_tke", False)),
            has_complex_source=bool(d.get("has_complex_source", False)),
            supports_wss=bool(d.get("supports_wss", False)),
            supports_plane_metrics=bool(d.get("supports_plane_metrics", False)),
            has_wrapped_phase=bool(d.get("has_wrapped_phase", False)),
            supports_phase_unwrap=bool(d.get("supports_phase_unwrap", False)),
        )


@dataclass
class BackgroundPhaseCorrectionConfig:
    enabled: bool = True
    method: str = "wrls_arto"
    corr_fit_order: int = 3
    threshold: float = 0.1
    wrls_lambda: float = 5.0
    wrls_magnitude_threshold: float = 0.04
    wrls_mid_fov_fraction: float = 0.5
    wrls_mid_slice_fraction: float = 0.65
    wrls_arto_iterations: int = 2
    wrls_tau: float = 3.0
    wrls_delta: float = 2.0
    wrls_central_probability: float = 0.5
    wrls_fista_iterations: int = 5000
    wrls_gmm_iterations: int = 1000
    dual_venc_ratio1: float = 0.0
    dual_venc_ratio2: float = 0.0
    force_recompute: bool = False
    # Keep normal cache reuse fast, but allow read-only cold-start profiling.
    write_cache: bool = True

    def to_dict(self):
        return {
            "enabled": bool(self.enabled),
            "method": str(self.method),
            "corr_fit_order": int(self.corr_fit_order),
            "threshold": float(self.threshold),
            "wrls_lambda": float(self.wrls_lambda),
            "wrls_magnitude_threshold": float(self.wrls_magnitude_threshold),
            "wrls_mid_fov_fraction": float(self.wrls_mid_fov_fraction),
            "wrls_mid_slice_fraction": float(self.wrls_mid_slice_fraction),
            "wrls_arto_iterations": int(self.wrls_arto_iterations),
            "wrls_tau": float(self.wrls_tau),
            "wrls_delta": float(self.wrls_delta),
            "wrls_central_probability": float(self.wrls_central_probability),
            "wrls_fista_iterations": int(self.wrls_fista_iterations),
            "wrls_gmm_iterations": int(self.wrls_gmm_iterations),
            "dual_venc_ratio1": float(self.dual_venc_ratio1),
            "dual_venc_ratio2": float(self.dual_venc_ratio2),
            "force_recompute": bool(self.force_recompute),
            "write_cache": bool(self.write_cache),
        }

    @staticmethod
    def from_dict(d):
        payload = d or {}
        return BackgroundPhaseCorrectionConfig(
            enabled=bool(payload.get("enabled", False)),
            # A bare legacy dict historically meant MSAC; fully specified config
            # bundles use WRLS+ARTO as the new default.
            method=str(payload.get("method", "msac" if "method" not in payload else "wrls_arto") or "wrls_arto").strip().lower(),
            corr_fit_order=int(payload.get("corr_fit_order", 3)),
            threshold=float(payload.get("threshold", 0.1)),
            wrls_lambda=float(payload.get("wrls_lambda", 5.0)),
            wrls_magnitude_threshold=float(payload.get("wrls_magnitude_threshold", 0.04)),
            wrls_mid_fov_fraction=float(payload.get("wrls_mid_fov_fraction", 0.5)),
            wrls_mid_slice_fraction=float(payload.get("wrls_mid_slice_fraction", 0.65)),
            wrls_arto_iterations=int(payload.get("wrls_arto_iterations", 2)),
            wrls_tau=float(payload.get("wrls_tau", 3.0)),
            wrls_delta=float(payload.get("wrls_delta", 2.0)),
            wrls_central_probability=float(payload.get("wrls_central_probability", 0.5)),
            wrls_fista_iterations=int(payload.get("wrls_fista_iterations", 5000)),
            wrls_gmm_iterations=int(payload.get("wrls_gmm_iterations", 1000)),
            dual_venc_ratio1=float(payload.get("dual_venc_ratio1", 0.0)),
            dual_venc_ratio2=float(payload.get("dual_venc_ratio2", 0.0)),
            force_recompute=bool(payload.get("force_recompute", False)),
            write_cache=bool(payload.get("write_cache", True)),
        )


@dataclass
class PhaseUnwrappingConfig:
    """Optional traditional phase-unwrapping workflow settings."""
    # Retained only so workspaces written by older AutoFlow versions can be
    # loaded.  Selecting ``method`` is now the sole user-facing opt-in.
    enabled: bool = False
    method: str = "none"
    mask_source: str = "segmentation"
    device: str = "auto"
    tfc: bool = True
    lap4d_ts: float = 2.0
    nprs_upsampling_factor: int = 2
    nprs_pi_unwrap: bool = True
    nprs_auto_crop: bool = True
    write_output: bool = True

    def to_dict(self):
        return {
            "method": str(self.method),
            "mask_source": str(self.mask_source),
            "device": str(self.device),
            "tfc": bool(self.tfc),
            "lap4d_ts": float(self.lap4d_ts),
            "nprs_upsampling_factor": int(self.nprs_upsampling_factor),
            "nprs_pi_unwrap": bool(self.nprs_pi_unwrap),
            "nprs_auto_crop": bool(self.nprs_auto_crop),
            "write_output": bool(self.write_output),
        }

    @staticmethod
    def from_dict(d):
        payload = dict(d or {})
        method = str(payload.get("method", "none") or "none").strip()
        aliases = {"gc3d": "gc3D", "lap4d": "lap4D"}
        method = aliases.get(method.lower(), method)
        if method not in {"none", "gc3D", "lap4D", "nprs"}:
            method = "none"
        mask_source = str(payload.get("mask_source", "segmentation") or "segmentation").strip().lower()
        if mask_source in {"active_segmentation", "seg", "mask"}:
            mask_source = "segmentation"
        if mask_source not in {"segmentation", "all"}:
            mask_source = "segmentation"
        return PhaseUnwrappingConfig(
            enabled=bool(payload.get("enabled", False)),
            method=method,
            mask_source=mask_source,
            device=str(payload.get("device", "auto") or "auto"),
            tfc=bool(payload.get("tfc", True)),
            lap4d_ts=float(payload.get("lap4d_ts", 2.0)),
            nprs_upsampling_factor=max(1, int(payload.get("nprs_upsampling_factor", 2) or 2)),
            nprs_pi_unwrap=bool(payload.get("nprs_pi_unwrap", True)),
            nprs_auto_crop=bool(payload.get("nprs_auto_crop", True)),
            write_output=bool(payload.get("write_output", True)),
        )


@dataclass
class LoadedCase:
    mag: np.ndarray
    flow: np.ndarray
    resolution: np.ndarray
    origin: np.ndarray
    venc: np.ndarray
    rr: float
    segmentation: Optional[np.ndarray] = None
    tke_array: Optional[np.ndarray] = None
    sigma: Optional[np.ndarray] = None
    correction: Optional[np.ndarray] = None
    correction_high: Optional[np.ndarray] = None
    phase_wrapped: Optional[np.ndarray] = None
    phase_wrapped_high: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_format: str = ""
    source_group: Optional[str] = None
    capabilities: LoaderCapabilities = field(default_factory=LoaderCapabilities)

    @property
    def segmask(self):
        return self.segmentation


@dataclass
class InputCase:
    input_path: str
    input_kind: str
    display_name: str = ""
    output_name: str = ""
    source_group: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InputState:
    source_format: str = ""
    source_group: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    capabilities: LoaderCapabilities = field(default_factory=LoaderCapabilities)

    def to_dict(self):
        return {
            "source_format": self.source_format,
            "source_group": self.source_group,
            "metadata": copy.deepcopy(self.metadata),
            "capabilities": self.capabilities.to_dict(),
        }

    @staticmethod
    def from_dict(d):
        return InputState(
            source_format=str(d.get("source_format", "")),
            source_group=d.get("source_group"),
            metadata=copy.deepcopy(d.get("metadata", {})),
            capabilities=LoaderCapabilities.from_dict(d.get("capabilities", {})),
        )
