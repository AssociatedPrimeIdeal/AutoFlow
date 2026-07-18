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
    corr_fit_order: int = 3
    threshold: float = 0.1
    dual_venc_ratio1: float = 0.0
    dual_venc_ratio2: float = 0.0
    force_recompute: bool = False

    def to_dict(self):
        return {
            "enabled": bool(self.enabled),
            "corr_fit_order": int(self.corr_fit_order),
            "threshold": float(self.threshold),
            "dual_venc_ratio1": float(self.dual_venc_ratio1),
            "dual_venc_ratio2": float(self.dual_venc_ratio2),
            "force_recompute": bool(self.force_recompute),
        }

    @staticmethod
    def from_dict(d):
        payload = d or {}
        return BackgroundPhaseCorrectionConfig(
            enabled=bool(payload.get("enabled", False)),
            corr_fit_order=int(payload.get("corr_fit_order", 3)),
            threshold=float(payload.get("threshold", 0.1)),
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
