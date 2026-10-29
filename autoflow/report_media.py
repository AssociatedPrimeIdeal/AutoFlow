"""Case-bound report media and the available, already-computed video families."""

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import subprocess
import time
from urllib.parse import quote
import numpy as np

from .core.plane_state import plane_signature
from .task_control import TaskCancelled, check_cancelled, stop_process


VIDEO_TITLES = {
    'plane': 'Measurement planes', 'wss': 'Wall shear stress', 'tke': 'Turbulent kinetic energy',
    'pressure_gradient': 'Pressure-gradient magnitude', 'relative_pressure': 'Relative pressure',
    'streamlines': 'Streamlines', 'velocity': 'Velocity magnitude', 'vorticity': 'Vorticity magnitude',
    'q_criterion': 'Q-criterion', 'swirling_strength': 'Swirling strength', 'pcmra': 'PC-MRA',
    'current_view': 'Current 3D view', 'pathlines': 'Pathlines',
}


def media_signature(ws):
    def token(value):
        return None if value is None else (id(value), np.shape(value), str(getattr(value, 'dtype', 'mesh')))
    payload = [str(ws.paths.flow_path), ws.analysis_revision, ws.segmentation.revision,
               token(ws.flow_raw), token(ws.segmask_binary), token(ws.segmask_labels),
               list(np.asarray(ws.resolution)), list(np.asarray(ws.origin)), float(ws.rr),
               [plane_signature(p) for p in ws.planes], ws.derived.artifact_signatures,
               ws.derived_params.to_dict(), ws.streamline_params.to_dict(), ws.plane_gen_params.to_dict()]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def video_config_signature(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()


def metric_family_status(ws):
    from .core.pipeline import PipelineEngine
    signature = media_signature(ws)
    cached = getattr(ws, '_report_family_status_cache', None)
    if cached is not None and cached[0] == signature:
        return dict(cached[1])
    engine = PipelineEngine()
    valid = engine._derived_artifact_validity(ws, engine._derived_artifact_signatures(ws))
    results = {}
    for family, attrs in [('wss', ['wss_surfaces']), ('tke', ['tke_array', 'tke_volume']),
                          ('pressure', ['pressure_gradient_array', 'relative_pressure_array']),
                          ('vortex', ['vorticity_magnitude', 'q_criterion_array', 'swirling_strength_array'])]:
        present = any(getattr(ws.derived, name, None) is not None and
                      (len(getattr(ws.derived, name)) > 0 if isinstance(getattr(ws.derived, name), (list, tuple))
                       else np.size(getattr(ws.derived, name)) > 0) for name in attrs)
        results['pg' if family == 'pressure' else family] = 'complete' if valid.get(family) else 'stale' if present else 'not_calculated'
    if results['tke'] == 'not_calculated' and ws.source_tke_array is None and ws.source_sigma is None:
        results['tke'] = 'unsupported'
    results['pwv'] = ('complete' if all(p.get('status') == 'ok' and p.get('validity', 'current') == 'current' for p in ws.derived.pwv_results)
                      else 'not_assessable') if ws.derived.pwv_results else 'not_calculated'
    ws._report_family_status_cache = (signature, dict(results))
    return results


def available_video_jobs(ws):
    status = metric_family_status(ws)
    jobs = []
    if ws.planes:
        jobs.append('plane')
    for key in ['wss', 'tke']:
        if status[key] == 'complete':
            jobs.append(key)
    if status['pg'] == 'complete':
        jobs.append('pressure_gradient')
        if ws.derived.relative_pressure_array is not None:
            jobs.append('relative_pressure')
    if status['vortex'] == 'complete':
        jobs.extend(['vorticity', 'q_criterion', 'swirling_strength'])
    if ws.flow_raw is not None:
        jobs.append('velocity')
        if ws.streamline_active or ws.streamline_cache or ws.derived.streamlines:
            jobs.append('streamlines')
    if ws.pcmra_array is not None:
        jobs.append('pcmra')
    if any(mesh is not None and getattr(mesh, 'n_points', 0) for frames in ws.pathline_cache.values() for mesh in frames.values()):
        jobs.append('pathlines')
    return jobs


def ensure_browser_video(filename):
    """A VP9 companion plays in Qt builds without proprietary MP4 codecs."""
    import imageio_ffmpeg
    from .core.output import publish_video
    source = Path(filename).resolve()
    if source.suffix.lower() == '.webm':
        return str(source)
    target = source.with_suffix('.webm')
    receipt = target.with_suffix('.webm.source.json')
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    try:
        previous = json.loads(receipt.read_text())
        if (target.is_file() and previous['source_sha256'] == source_hash and
                previous['webm_bytes'] == target.stat().st_size and previous['webm_mtime_ns'] == target.stat().st_mtime_ns):
            return str(target)
    except (OSError, ValueError, KeyError):
        pass
    descriptor, temporary = tempfile.mkstemp(prefix='.browser-video-', suffix='.webm', dir=source.parent)
    os.close(descriptor)
    process = None
    try:
        with tempfile.TemporaryFile() as log:
            process = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), '-nostdin', '-y', '-i', str(source),
                '-an', '-c:v', 'libvpx-vp9', '-b:v', '0', '-crf', '32', '-deadline', 'realtime',
                '-cpu-used', '5', '-threads', '2', '-pix_fmt', 'yuv420p', temporary],
                stdout=log, stderr=log, start_new_session=os.name != 'nt',
                creationflags=subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0)
            while process.poll() is None:
                check_cancelled()
                time.sleep(.05)
            if process.returncode:
                log.seek(0)
                raise RuntimeError('Browser video encoding failed: ' + log.read().decode(errors='replace')[-1500:])
        check_cancelled()
        publish_video(temporary, target)
        stat = target.stat()
        receipt.write_text(json.dumps(dict(source_sha256=source_hash, webm_bytes=stat.st_size, webm_mtime_ns=stat.st_mtime_ns)))
        return str(target)
    finally:
        if process is not None and process.poll() is None:
            stop_process(process)
        if os.path.exists(temporary):
            os.unlink(temporary)


def register_report_videos(ws, outputs, config, *, camera_behavior='rotation', phase_behavior='cardiac_cycle', ensure_browser=False):
    records = ws.input_state.metadata.setdefault('report_videos', {})
    signature = media_signature(ws)
    config_signature = video_config_signature(config)
    for key, filename in outputs.items():
        if not filename:
            continue
        p = Path(filename).resolve()
        if not p.is_file():
            continue
        stat = p.stat()
        records[key] = dict(family=key, title=VIDEO_TITLES.get(key, key), path=str(p),
            data_signature=signature, config_signature=config_signature,
            sha256=hashlib.sha256(p.read_bytes()).hexdigest(), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
            camera_behavior=camera_behavior, phase_behavior=phase_behavior,
            fps=config.get('fps'), analysis_revision=ws.analysis_revision)
        try:
            companion = ensure_browser_video(p) if ensure_browser else str(p.with_suffix('.webm'))
        except TaskCancelled:
            raise
        except Exception as exc:
            records[key]['browser_error'] = f'{type(exc).__name__}: {exc}'
            continue
        browser = Path(companion)
        matches = ensure_browser or browser == p
        if not matches:
            try:
                receipt = json.loads(browser.with_suffix('.webm.source.json').read_text())
                matches = (receipt['source_sha256'] == records[key]['sha256'] and
                           receipt['webm_bytes'] == browser.stat().st_size and receipt['webm_mtime_ns'] == browser.stat().st_mtime_ns)
            except (OSError, ValueError, KeyError):
                pass
        if browser.is_file() and matches:
            stat = browser.stat()
            records[key].update(browser_path=str(browser), browser_sha256=hashlib.sha256(browser.read_bytes()).hexdigest(),
                                browser_bytes=stat.st_size, browser_mtime_ns=stat.st_mtime_ns)


def report_video_records(ws):
    signature = media_signature(ws)
    source = copy.deepcopy(ws.input_state.metadata.get('report_videos', {}))
    jobs = available_video_jobs(ws)
    records = []
    for key in dict.fromkeys([*jobs, *source]):
        record = source.get(key, dict(family=key, title=VIDEO_TITLES.get(key, key)))
        path = Path(record.get('path', ''))
        state = 'missing'
        if record.get('path') and path.is_file():
            stat = path.stat()
            state = 'current' if (record.get('data_signature') == signature and
                record.get('bytes') == stat.st_size and record.get('mtime_ns') == stat.st_mtime_ns) else 'stale'
        record['validity'] = state
        records.append(record)
    return records


def prepare_report_media(snapshot, report_path):
    """Keep videos portable beside an exported HTML without base64 inflation."""
    records = copy.deepcopy(snapshot.get('videos', []))
    destination = Path(report_path).resolve().parent
    assets = destination / (Path(report_path).stem + '_media')
    def publish(source, digest, family):
        if source.parent == destination:
            return quote(source.relative_to(destination).as_posix())
        assets.mkdir(parents=True, exist_ok=True)
        target = assets / (family + '-' + digest[:12] + source.suffix.lower())
        if not target.is_file() or target.stat().st_size != source.stat().st_size:
            descriptor, temporary = tempfile.mkstemp(prefix='.video-', suffix=source.suffix, dir=assets)
            os.close(descriptor)
            try:
                from .core.output import publish_video
                shutil.copyfile(source, temporary)
                publish_video(temporary, target)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        return quote(target.relative_to(destination).as_posix())
    for record in records:
        record.pop('url', None)
        if record.get('validity') != 'current':
            continue
        source = Path(record.get('path', ''))
        if not source.is_file():
            record['validity'] = 'missing'
            continue
        family = record.get('family', 'video')
        record['url'] = publish(source, record['sha256'], family)
        record['mime'] = 'video/webm' if source.suffix.lower() == '.webm' else 'video/mp4'
        browser = Path(record.get('browser_path', ''))
        if browser.is_file() and record.get('browser_bytes') == browser.stat().st_size and record.get('browser_mtime_ns') == browser.stat().st_mtime_ns:
            record['download_url'] = record['url']
            record['url'] = publish(browser, record['browser_sha256'], family)
            record['mime'] = 'video/webm'
    return records
