"""Array-shape normalization shared by metric families."""

import numpy as np


def _ensure_mask4d(mask4d):
    mask4d = np.asarray(mask4d, dtype=bool)
    if mask4d.ndim == 3:
        mask4d = mask4d[..., np.newaxis]
    if mask4d.ndim != 4:
        raise ValueError(f"mask4d must be XYZ or XYZT, got {mask4d.shape}")
    return mask4d


def _ensure_flow5d(flow):
    flow = np.asarray(flow, dtype=np.float32)
    if flow.ndim != 5 or flow.shape[-1] != 3:
        raise ValueError(f"flow must be XYZTV, got {flow.shape}")
    return flow
