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

    def to_dict(self):
        return {
            "has_segmentation": bool(self.has_segmentation),
            "has_tke": bool(self.has_tke),
            "has_complex_source": bool(self.has_complex_source),
            "supports_wss": bool(self.supports_wss),
            "supports_plane_metrics": bool(self.supports_plane_metrics),
        }

    @staticmethod
    def from_dict(d):
        return LoaderCapabilities(
            has_segmentation=bool(d.get("has_segmentation", False)),
            has_tke=bool(d.get("has_tke", False)),
            has_complex_source=bool(d.get("has_complex_source", False)),
            supports_wss=bool(d.get("supports_wss", False)),
            supports_plane_metrics=bool(d.get("supports_plane_metrics", False)),
        )


@dataclass
class BackgroundPhaseCorrectionConfig:
    enabled: bool = True
    method: str = "msac"
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
        }

    @staticmethod
    def from_dict(d):
        payload = d or {}
        return BackgroundPhaseCorrectionConfig(
            enabled=bool(payload.get("enabled", False)),
            method=str(payload.get("method", "msac") or "msac").strip().lower(),
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
