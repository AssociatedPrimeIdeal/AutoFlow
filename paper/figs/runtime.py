"""Configuration, paths, encoding and provenance shared by all three figures."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO))


def configure_runtime():
    for key, value in {
        'QT_QPA_PLATFORM':'offscreen', 'PYVISTA_OFF_SCREEN':'true',
        'VTK_DEFAULT_OPENGL_WINDOW':'vtkEGLRenderWindow', 'MPLBACKEND':'Agg',
        'OMP_NUM_THREADS':'4', 'OPENBLAS_NUM_THREADS':'4', 'MKL_NUM_THREADS':'4',
    }.items():
        os.environ.setdefault(key, value)


def load_config(path, *, input_path=None, output=None, work=None):
    path = Path(path).resolve()
    cfg = json.loads(path.read_text(encoding='utf-8'))
    cfg['_config_file'] = str(path)
    cfg.setdefault('case_name', 'demo')
    if input_path:
        cfg['source'] = {'input':str(Path(input_path).resolve()),'input_layout':'native'}
        cfg['case_name'] = Path(input_path).stem
        cfg.pop('work_dir',None);cfg.pop('output_dir',None)
        cfg.pop('existing_assets', None)
        cfg.setdefault('correction',{}).update(mode='measured',slice=None,phase=None,amplified_background_phase=False)
        for key in ('camera','template_case','template_videos'):
            cfg.setdefault('workflow',{}).pop(key,None)
        cfg['workflow']['selected_plane']='auto'
        cfg.setdefault('results',{}).pop('rotation_seconds',None)
    default_output = ('D:/CodexData/outputs/AutoFlow-paper' if os.name == 'nt'
                      else '/tmp/autoflow-paper-outputs')
    default_work = ('D:/CodexData/tmp/AutoFlow-paper' if os.name == 'nt'
                    else '/tmp/autoflow-paper')
    cfg['output_dir'] = str(Path(output or cfg.get('output_dir', str(Path(default_output)/cfg['case_name']))).expanduser())
    cfg['work_dir'] = str(Path(work or cfg.get('work_dir', str(Path(default_work)/cfg['case_name']))).expanduser())
    cfg['config_dir'] = str(resolve_path(cfg.get('config_dir', str(REPO/'configs')), cfg))
    return cfg


def resolve_path(value, cfg):
    value = str(value).replace('{repo}', str(REPO)).replace('{work}', cfg['work_dir'])
    value = value.replace('{output}', cfg['output_dir'])
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path(cfg['_config_file']).parent/path


def paths(cfg):
    work, output = Path(cfg['work_dir']), Path(cfg['output_dir'])
    for p in (work, output, work/'assets', work/'analysis'):
        p.mkdir(parents=True, exist_ok=True)
    return work, output


def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    def small(x):
        if isinstance(x, Path): return str(x)
        if hasattr(x, 'item') and getattr(x, 'ndim', 1) == 0: return x.item()
        if hasattr(x, 'size') and x.size <= 1000: return x.tolist()
        raise TypeError(f'Only small metadata belongs in JSON: {type(x).__name__}')
    path.write_text(json.dumps(value, indent=2, default=small), encoding='utf-8')


def file_identity(path):
    path = Path(path).resolve(); st = path.stat()
    return dict(path=str(path), bytes=st.st_size, mtime_ns=st.st_mtime_ns)


def config_identity(cfg):
    # Render timing, cameras and output locations do not invalidate measurements.
    value = {key:copy.deepcopy(cfg.get(key)) for key in ('source','config_dir','cleanup','analysis')}
    value['plane_spacing_mm'] = cfg.get('workflow',{}).get('plane_spacing_mm',10)
    value['source_files'] = {}
    source = cfg.get('source',{})
    for key,item in [('input',source.get('input')),('workspace',source.get('workspace')),
                     ('segmentation',source.get('segmentation',{}).get('path'))]:
        if item:
            p=resolve_path(item,cfg)
            if p.is_file():value['source_files'][key]=file_identity(p)
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def logical_identity(cfg):
    value={key:cfg.get(key) for key in ('source','config_dir','cleanup','analysis')}
    value['config_dir']=str(value['config_dir']).replace(str(REPO),'{repo}')
    value['plane_spacing_mm']=cfg.get('workflow',{}).get('plane_spacing_mm',10)
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''): digest.update(chunk)
    return digest.hexdigest()


def ffmpeg():
    found = shutil.which('ffmpeg')
    if found: return found
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def font_path(bold=False, configured=None):
    candidates = [configured] if configured else []
    candidates += [str(ROOT/'assets'/('DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf')),
                   '/usr/share/fonts/truetype/dejavu/'+('DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'),
                   'C:/Windows/Fonts/'+('arialbd.ttf' if bold else 'arial.ttf')]
    for path in candidates:
        if path and Path(path).is_file(): return str(path)
    raise FileNotFoundError('Configure render.font/render.bold_font or install DejaVu Sans')


class VideoWriter:
    def __init__(self, path, size, fps=30, encoder='libx264'):
        self.path, self.size, self.fps, self.frames = Path(path), tuple(size), fps, 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        codec = (['-c:v','h264_nvenc','-preset','p6','-rc','vbr','-cq','16','-b:v','0']
                 if encoder == 'h264_nvenc' else
                 ['-c:v','libx264','-preset','fast','-crf','17','-threads','4'])
        self.process = subprocess.Popen([ffmpeg(),'-v','error','-y','-f','rawvideo',
            '-pix_fmt','rgb24','-s',f'{size[0]}x{size[1]}','-r',str(fps),'-i','-',
            '-an',*codec,'-pix_fmt','yuv420p','-movflags','+faststart',str(self.path)], stdin=subprocess.PIPE)

    def append(self, image):
        import numpy as np
        values = np.asarray(image, dtype=np.uint8)
        if values.shape != (self.size[1], self.size[0], 3): raise ValueError(values.shape)
        self.process.stdin.write(values.tobytes()); self.frames += 1

    def close(self):
        self.process.stdin.close()
        if self.process.wait(): raise RuntimeError(f'Video encoding failed: {self.path}')
        subprocess.run([ffmpeg(),'-v','error','-i',str(self.path),'-f','null','-'], check=True)
        return dict(file=str(self.path), size=self.size, fps=self.fps, frames=self.frames,
                    duration_s=self.frames/self.fps, sha256=sha256_file(self.path), decode='passed')
