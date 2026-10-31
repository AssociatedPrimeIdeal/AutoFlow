"""Display-only centerline fitting; measurements and plane geometry stay fixed."""
from pathlib import Path
import json
from autoflow.algorithms.path_smoothing import PathSmoothingContext, prepare_centerline


def display_paths(ws, cfg):
    parameters=json.loads((Path(cfg['config_dir'])/'planes.json').read_text())
    options={k:v for k,v in parameters.items() if k.startswith('smoothing_')}
    contexts={};paths=[];reports=[]
    for index,path in enumerate(ws.centerline_paths):
        name=ws.path_info[index].get('group_name','') if index<len(ws.path_info) else ''
        group=ws.multilabel_groups.get(name,{})
        if name not in contexts:
            contexts[name]=PathSmoothingContext(group.get('geometry_mask_3d',ws.segmentation_analysis_mask),ws.segmentation_spacing)
        _,smooth,report=prepare_centerline(path,context=contexts[name],**options)
        paths.append(smooth);reports.append(dict(path_index=index,group=name,**report))
    return paths,reports
