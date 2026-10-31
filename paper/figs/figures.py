"""Build the sequential workflow, simultaneous workflow and result overview."""
from __future__ import annotations
import argparse
from pathlib import Path
from runtime import configure_runtime, load_config, paths, dump


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['check','verify','review','prepare','assets','panel-assets','capture','capture-intro','workflow','results','all','import-existing'])
    parser.add_argument('--config', type=Path, default=Path(__file__).parent/'presets/vol023-seg-true.json')
    parser.add_argument('--input', type=Path, help='New case; disables workspace and old segmentation/cache reuse')
    parser.add_argument('--segmentation', type=Path, help='External labels for --input; already aligned on the input grid')
    parser.add_argument('--input-layout', choices=['native','matlab_vtzyx'])
    parser.add_argument('--region', choices=['cardiopulmonary','cerebrovascular','portal-venous','carotid','renal-artery'])
    parser.add_argument('--encoder', choices=['libx264','h264_nvenc'])
    parser.add_argument('--output', type=Path); parser.add_argument('--work', type=Path)
    parser.add_argument('--mode', choices=['sequence','simultaneous','both'], default='both')
    parser.add_argument('--stills', action='store_true', help='Prepare review images without encoding full movies')
    parser.add_argument('--resolution', choices=['4k','1080p'], default='4k')
    args = parser.parse_args(); configure_runtime()
    cfg = load_config(args.config, input_path=args.input, output=args.output, work=args.work)
    if args.input_layout:cfg['source']['input_layout']=args.input_layout
    if args.region:cfg['region']=args.region
    if args.encoder:cfg.setdefault('render',{})['encoder']=args.encoder
    if args.segmentation:
        cfg['source']['segmentation'] = {'path':str(args.segmentation.resolve()), 'kind':'imported', 'layout':'normalized'}
    if args.command == 'check':
        from prepare_case import check_config
        print(check_config(cfg)); return
    work, out = paths(cfg); dump(work/'effective-config.json', cfg)
    if args.command=='verify':
        from verify_sources import verify_sources
        verify_sources(cfg);return
    if args.command=='panel-assets':
        from prepare_workflow_panels import prepare_workflow_panels
        prepare_workflow_panels(cfg);return
    if args.command=='capture-intro':
        from capture_gui import capture_gui
        capture_gui(cfg,intro_only=True);return
    if args.command == 'import-existing':
        from import_existing import import_existing
        import_existing(cfg); return
    if args.command == 'review':
        from prepare_case import segmentation_review
        from review_segmentation import render_review
        candidate = segmentation_review(cfg)
        print(render_review(candidate, out/'segmentation-review', movie=not args.stills)); return
    if args.command in ('prepare','all'):
        from prepare_case import prepare_case
        prepare_case(cfg)
    if args.command in ('assets','all'):
        from prepare_assets import prepare_assets
        prepare_assets(cfg)
    if args.command in ('capture','all'):
        from capture_gui import capture_gui
        from capture_qc import capture_qc
        capture_gui(cfg); capture_qc(cfg)
    if args.command in ('workflow','all'):
        from workflow import render_workflows
        render_workflows(cfg, mode=args.mode, stills=args.stills, resolution=args.resolution)
    if args.command in ('results','all'):
        from results import render_results
        render_results(cfg, stills=args.stills)


if __name__ == '__main__': main()
