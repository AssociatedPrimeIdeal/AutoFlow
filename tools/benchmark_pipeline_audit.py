"""Manual, read-only-input full-pipeline audit (not part of the test suite)."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REGIONS = ('Aorta', 'Carotid', 'Cerebrovascular', 'PortalVein', 'RenalArtery')

def worker(argv):
    # An audit may preserve its baseline while video-only improvements are developed.
    frozen_root = Path(os.environ['AUTOFLOW_AUDIT_DETAIL']).resolve().parent / 'baseline_source'
    if (frozen_root / 'autoflow').is_dir():
        sys.path.insert(0, str(frozen_root))
        argv = list(argv)
        if '--config-dir' in argv:
            argv[argv.index('--config-dir') + 1] = str(frozen_root / 'configs')
    import functools
    from collections import defaultdict
    timings = defaultdict(lambda: {'calls': 0, 'seconds': 0.0})
    import autoflow.core.pipeline as pipeline
    import autoflow.processing as processing
    def instrument(owner, name):
        original = getattr(owner, name)
        @functools.wraps(original)
        def measured(*args, **kwargs):
            begin = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                row = timings[name]
                row['calls'] += 1
                row['seconds'] += time.perf_counter() - begin
        setattr(owner, name, measured)
    for name in ('load_data', 'preprocess', 'refresh_segmentation_dependents', '_ensure_derived_metrics', '_derived_artifact_signatures', '_compute_plane_metrics_internal', '_save_planes_json'):
        instrument(pipeline.PipelineEngine, name)
    for name in ('_save_pixelwise_npz', 'save_plane_positions', 'build_quality_report', 'save_quality_report', 'load_metrics_from_output'):
        if hasattr(processing, name):
            instrument(processing, name)
    import autoflow.algorithms.metrics.derived as derived
    for name in ('compute_wss_metrics', 'compute_tke_metrics', 'compute_pressure_gradient_metrics', 'compute_vortex_metrics'):
        instrument(derived, name)
    for name in ('load_input_data', 'compute_plane_metrics_multithread', 'augment_plane_metrics_with_derived', 'save_plane_pixelwise_h5'):
        instrument(pipeline, name)
    import autoflow.cli as cli
    sys.argv = ['autoflow-run', *argv]
    try:
        raise SystemExit(cli.main())
    finally:
        Path(os.environ['AUTOFLOW_AUDIT_DETAIL']).write_text(json.dumps(dict(timings), indent=2))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--regions', nargs='+', choices=REGIONS, default=list(REGIONS))
    parser.add_argument('--no-videos', action='store_true')
    args = parser.parse_args()
    args.input_dir = args.input_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    import h5py
    import psutil
    root = Path(__file__).resolve().parents[1]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    selected = []
    for region in args.regions:
        candidates = sorted(args.input_dir.glob(region + '*.h5'), key=lambda p: (p.stat().st_size, p.name))
        if not candidates:
            raise ValueError(f'No sample for {region}')
        path = candidates[len(candidates) // 2]
        with h5py.File(path, 'r') as handle:
            datasets = {}
            handle.visititems(lambda name, item: datasets.update({name: {'shape': list(item.shape), 'dtype': str(item.dtype)}}) if isinstance(item, h5py.Dataset) else None)
        selected.append({'region': region, 'path': str(path), 'bytes': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns, 'datasets': datasets, 'candidate_count': len(candidates)})
    def command_output(argv):
        result = subprocess.run(argv, capture_output=True, text=True)
        return result.stdout.strip() or result.stderr.strip()
    environment = {'python': sys.executable, 'host': os.uname().nodename, 'cpu_count': os.cpu_count(), 'memory_available_bytes': psutil.virtual_memory().available, 'git_head': command_output(['git', '-C', str(root), 'rev-parse', 'HEAD']), 'gpu_before': command_output(['nvidia-smi', '--query-gpu=name,memory.used,memory.total,utilization.gpu', '--format=csv,noheader']), 'policy': 'Application cold start per case; force correction, ignore embedded segmentation, disable input cache writes; OS filesystem cache is not flushed; shared machine, sequential cases; default quality and video resolution/frame counts.'}
    (args.output_dir / 'manifest.json').write_text(json.dumps({'environment': environment, 'samples': selected}, indent=2))
    rows = []
    for sample in selected:
        region = sample['region']
        output = args.output_dir / region
        command = [sys.executable, str(Path(__file__).resolve()), '--worker', sample['path'], '--output-dir', str(output), '--config-dir', str(root / 'configs'), '--correction', '--force-recompute-corr', '--ignore-embedded-segmentation', '--force-recompute-seg', '--no-cache-write', '--autoseg', '--autoseg-backend', 'nnUNet4D', '--with', 'pwv,wss,tke,pg,vortex']
        if not args.no_videos:
            command += ['--video', 'plane,wss,tke,pg,streamlines']
        env = dict(os.environ, PYVISTA_OFF_SCREEN='true', QT_QPA_PLATFORM='offscreen', PYTHONUNBUFFERED='1', AUTOFLOW_AUDIT_DETAIL=str(args.output_dir / (region + '_detail.json')))
        env['PYTHONPATH'] = str(root) + os.pathsep + env.get('PYTHONPATH', '')
        print(f'BEGIN {region}', flush=True)
        begin = time.perf_counter()
        peak_rss = 0
        with (args.output_dir / (region + '.log')).open('w') as log:
            proc = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
            monitored = psutil.Process(proc.pid)
            while proc.poll() is None:
                try:
                    rss = monitored.memory_info().rss
                    for child in monitored.children(recursive=True):
                        try:
                            rss += child.memory_info().rss
                        except psutil.Error:
                            pass
                    peak_rss = max(peak_rss, rss)
                except psutil.Error:
                    pass
                time.sleep(1)
        row = {'region': region, 'wall_seconds': time.perf_counter() - begin, 'returncode': proc.returncode, 'peak_process_tree_rss_bytes': peak_rss, 'input_unchanged_size_mtime': Path(sample['path']).stat().st_size == sample['bytes'] and Path(sample['path']).stat().st_mtime_ns == sample['mtime_ns'], 'command': command}
        report = output / 'batch_report.json'
        if report.exists():
            reports = json.loads(report.read_text())
            row['status'] = reports[0].get('status') if reports else 'empty'
            if reports and reports[0].get('status') == 'ok':
                summary = reports[0]['summary']
                row['summary'] = {key: summary.get(key) for key in ('total_time_sec', 'stage_times_sec', 'video_times_sec', 'n_planes', 'n_paths', 'n_graph_nodes', 'capabilities', 'phase_unwrap', 'videos', 'requested_metrics')}
            else:
                row['error'] = reports[0].get('error') if reports else 'No report entries'
        else:
            row['status'] = 'process_failed'
        rows.append(row)
        (args.output_dir / 'measurements.json').write_text(json.dumps(rows, indent=2))
        print(f'END {region}: {row["status"]}, {row["wall_seconds"]:.1f}s, peak RSS {peak_rss / 2**30:.2f} GiB', flush=True)
    (args.output_dir / 'gpu_after.txt').write_text(environment['gpu_before'] + '\nAFTER\n' + command_output(['nvidia-smi', '--query-gpu=name,memory.used,memory.total,utilization.gpu', '--format=csv,noheader']))
    return 0 if all(row['status'] == 'ok' for row in rows) else 1

if __name__ == '__main__':
    if '--worker' in sys.argv:
        worker(sys.argv[sys.argv.index('--worker') + 1:])
    else:
        raise SystemExit(main())