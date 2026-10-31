"""Render a matched before/after comparison without recomputing measurements."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import shutil

os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
os.environ.setdefault("VTK_DEFAULT_OPENGL_WINDOW", "vtkEGLRenderWindow")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
import pyvista as pv
from PIL import Image, ImageDraw, ImageFont
from matplotlib.colors import to_rgb
from segmentation import labeled_surface, change_statistics
from runtime import font_path

PALETTE = ['#ee6c71','#cdb34b','#489bd2','#48b1a2','#df9a51','#aa7acc',
           '#75bccc','#e06e9d','#766ac9','#8ebc62','#ed9271','#567dc2','#c3a0d5','#4cae83']


def render_review(source, output, phase=2, movie=True):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    data = np.load(source)
    before, after = data['before'], data['after']
    spacing, origin = data['spacing'], data['origin']
    stats = change_statistics(before, after)
    if before.ndim == 3:
        before, after = before[..., None], after[..., None]
    phases = before.shape[3]; phase = min(phase, phases - 1)
    pts = np.argwhere(np.any(before > 0, axis=3)) * spacing + origin
    center = (pts.min(0) + pts.max(0)) / 2
    extent = np.ptp(pts, axis=0)
    scale = float(max(extent[2] / 2, max(extent[:2]) / (2 * 1280/800)) * 1.17)
    lookup = np.zeros((int(before.max()) + 1, 3))
    for value in range(1, len(lookup)):
        lookup[value] = to_rgb(PALETTE[(value-17) % len(PALETTE)])
    meshes = {'before': [], 'after': [], 'removed': []}
    for t in range(phases):
        for name, array in [('before', before[..., t]), ('after', after[..., t]),
                            ('removed', np.where(after[..., t] == 0, before[..., t], 0))]:
            mesh = labeled_surface(array, spacing, origin)
            if mesh is not None:
                mesh['RGB'] = (lookup[np.asarray(mesh['Label'], int)] * 255).astype(np.uint8)
            meshes[name].append(mesh)
        print('review meshes', t+1, phases, flush=True)
    plots = [pv.Plotter(off_screen=True, window_size=(1280, 800)) for _ in range(3)]
    for p in plots:
        p.set_background('white'); p.enable_lightkit(); p.enable_anti_aliasing('msaa')
    def view(kind, t, angle):
        p = plots[['before', 'after', 'removed'].index(kind)]
        p.clear()
        p.enable_lightkit()
        if kind == 'removed' and meshes['after'][t] is not None:
            p.add_mesh(meshes['after'][t], color='#cbd4db', opacity=.18,
                       smooth_shading=True, show_scalar_bar=False)
        mesh = meshes[kind][t]
        if mesh is not None:
            kwargs = dict(color='#ed7d22') if kind == 'removed' else dict(scalars='RGB', rgb=True)
            p.add_mesh(mesh, **kwargs, smooth_shading=True, show_scalar_bar=False,
                       ambient=.35, diffuse=.72, specular=.18, specular_power=25)
        a = math.radians(angle)
        p.camera.position = tuple(center + 500 * np.array([math.sin(a), -math.cos(a), 0]))
        p.camera.focal_point = tuple(center); p.camera.up = (0, 0, 1)
        p.camera.parallel_projection = True; p.camera.parallel_scale = scale
        p.renderer.ResetCameraClippingRange(); p.render()
        return Image.fromarray(p.screenshot(return_img=True)[..., :3])
    font = ImageFont.truetype(font_path(True), 46)
    sheet = Image.new('RGB', (3840, 1700), 'white'); draw = ImageDraw.Draw(sheet)
    for i, name in enumerate(['Before', 'After', 'Removed']):
        draw.text((i*1280+640, 18), name, font=font, fill='#243442', anchor='mt')
    for row, angle in enumerate((0, 120)):
        for col, kind in enumerate(('before', 'after', 'removed')):
            sheet.paste(view(kind, phase, angle), (col*1280, 100+row*800))
    image_path = output/'Segmentation_before_after.png'; sheet.save(image_path)
    if movie:
        movie_path = output/'Segmentation_before_after.mp4'
        import imageio_ffmpeg
        ffmpeg = shutil.which('ffmpeg') or imageio_ffmpeg.get_ffmpeg_exe()
        p = subprocess.Popen([ffmpeg,'-v','warning','-y','-f','rawvideo','-pix_fmt','rgb24',
            '-s','2560x900','-r','30','-i','-','-an','-c:v','libx264','-preset','fast',
            '-crf','17','-threads','4','-pix_fmt','yuv420p','-movflags','+faststart',str(movie_path)], stdin=subprocess.PIPE)
        for i in range(180):
            t = (phase + int(i * phases / 60)) % phases
            canvas = Image.new('RGB', (2560, 900), 'white'); d = ImageDraw.Draw(canvas)
            for col, kind in enumerate(('before', 'after')):
                d.text((col*1280+640, 18), kind.title(), font=font, fill='#243442', anchor='mt')
                canvas.paste(view(kind, t, i*2), (col*1280, 100))
            p.stdin.write(canvas.tobytes())
        p.stdin.close(); assert p.wait() == 0
        subprocess.run([ffmpeg,'-v','error','-i',str(movie_path),'-f','null','-'], check=True)
    for p in plots: p.close()
    (output/'review.json').write_text(json.dumps({'source':str(source),'frames':stats,
        'static_phase':phase,'angles_degrees':[0,120],'movie_duration_s':6 if movie else None,
        'surface_smoothing':'Display only: 60 surface iterations + 80 Taubin iterations, pass band 0.08',
        'metrics_recalculated':False,'production_videos_updated':False}, indent=2))
    return image_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path); parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--phase', type=int, default=2); parser.add_argument('--stills', action='store_true')
    args = parser.parse_args()
    render_review(args.source, args.output, args.phase, not args.stills)
