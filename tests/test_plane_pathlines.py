import numpy as np

from autoflow.core.models import PlaneData, Workspace
from autoflow.core.pipeline import PipelineEngine


def test_pathline_step_uses_per_plane_colors():
    ws = Workspace()
    ws.segmask_raw = np.ones((16, 16, 16, 2), dtype=np.int16)
    ws.flow_raw = np.zeros((16, 16, 16, 2, 3), dtype=np.float32)
    ws.flow_raw[..., 0] = 10.0
    ws.planes = [
        PlaneData(center=np.array([4.0, 8.0, 8.0], dtype=float), normal=np.array([1.0, 0.0, 0.0], dtype=float)),
        PlaneData(center=np.array([10.0, 8.0, 8.0], dtype=float), normal=np.array([1.0, 0.0, 0.0], dtype=float)),
    ]
    ws.streamline_params.pathline_color = 'deepskyblue'
    ws.pathline_colors = {0: 'lime', 1: '#ff8800'}

    engine = PipelineEngine()
    engine.preprocess(ws)
    result = engine._step_plane_streamlines(ws)

    assert result.success and not result.skipped
    colors = {
        obj.data_key: obj.color
        for obj in ws.scene_objects.values()
        if obj.data_key.startswith('pathline_')
    }
    assert colors == {
        'pathline_0': 'lime',
        'pathline_1': '#ff8800',
    }
