"""Manual source/mask audit, separate from the repository's pytest suite."""
import hashlib
import json
import numpy as np
from runtime import dump, paths, logical_identity
from prepare_case import load_workspace


def verify_sources(cfg):
    work,_=paths(cfg);ws=load_workspace(cfg,prepared=True)
    original=np.asarray(ws.get_active_segmentation());native=np.asarray(ws.segmask_binary)
    topology=np.asarray(ws.segmentation_analysis_binary)
    exact_native=np.array_equal(original>0,native)
    exact_labels=np.array_equal(original,ws.segmask_labels)
    if not cfg.get('cleanup',{}).get('remove_small_cc',True):
        if not exact_native or not exact_labels:
            raise ValueError('Original segmentation is not retained exactly on the native measurement grid')
    report={'case':cfg['case_name'],'segmentation':ws.get_active_segmentation_provenance(),
        'cleanup_enabled':cfg.get('cleanup',{}).get('remove_small_cc',True),
        'native_mask_matches_original':exact_native,'native_labels_match_original':exact_labels,
        'native_shape':list(native.shape),'native_spacing_mm':ws.resolution.tolist(),
        'topology_shape':list(topology.shape),'topology_spacing_mm':np.asarray(ws.segmentation_spacing).tolist(),
        'metric_mask_sha256':hashlib.sha256(native.tobytes()).hexdigest(),
        'native_labels_sha256':hashlib.sha256(np.asarray(ws.segmask_labels).tobytes()).hexdigest(),
        'topology_mask_sha256':hashlib.sha256(topology.tobytes()).hexdigest(),
        'tke_source':'sigma from complex source' if ws.source_sigma is not None else 'supplied TKE' if ws.source_tke_array is not None else 'unavailable',
        'plane_count':len(ws.planes),'phase_count':ws.time_count()}
    report['stale_unused_asset_bundles']=[]
    for kind in ('results','workflow'):
        path=work/'assets'/kind/'assets.json'
        if not path.exists():continue
        data=json.loads(path.read_text())
        if data['mask_sha256']!=report['topology_mask_sha256']:
            report['stale_unused_asset_bundles'].append(kind);continue
        data['native_metric_mask_sha256']=report['metric_mask_sha256'];dump(path,data)
    replacement=work/'workflow-panels/assets.json'
    if replacement.exists():
        data=json.loads(replacement.read_text())
        assert data['native_metric_mask_sha256']==report['metric_mask_sha256']
        assert data['topology_mask_sha256']==report['topology_mask_sha256']
        assert data['cleanup']==cfg.get('cleanup')
        data['analysis_identity']=logical_identity(cfg);dump(replacement,data)
    stamp=work/'workspace.json'
    if stamp.exists():
        data=json.loads(stamp.read_text());data['metric_segmentation_sha256']=report['metric_mask_sha256']
        data['topology_segmentation_sha256']=report['topology_mask_sha256'];dump(stamp,data)
    dump(work/'source-verification.json',report);print(json.dumps(report,indent=2));return report
