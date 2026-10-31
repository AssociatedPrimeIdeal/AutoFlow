"""Explicitly stage an existing figure asset bundle for layout reproduction."""
import json
from pathlib import Path
import shutil
from runtime import dump, paths, resolve_path


def import_existing(cfg):
    work,_=paths(cfg);sources=cfg.get('existing_assets',{})
    if not sources:raise ValueError('No existing_assets entry in this preset; run prepare/assets instead')
    expected=cfg['source'].get('segmentation',{}).get('kind','embedded')
    audit={}
    for kind in ('results','workflow'):
        if kind not in sources:continue
        declared=sources.get(kind+'_segmentation_kind')
        if declared!=expected:raise ValueError(f'{kind} asset segmentation {declared!r} differs from {expected!r}')
        source=resolve_path(sources[kind],cfg);target=work/'assets'/kind
        metadata=json.loads((source/'assets.json').read_text())
        actual=str(metadata.get('source',''))
        if expected=='seg_true' and 'seg_true' not in actual:
            raise ValueError(f'{kind} asset manifest does not identify seg_true')
        shutil.copytree(source,target,dirs_exist_ok=True)
        audit[kind]={'source':str(source),'target':str(target),'segmentation_kind':expected,
                     'mode':'Reproduction of existing assets; no new metric calculation is claimed'}
    dump(work/'imported-assets.json',audit)
    print(json.dumps(audit,indent=2))
