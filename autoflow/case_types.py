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
