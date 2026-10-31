"""Load one declared segmentation source and prepare shared scientific results."""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
from pathlib import Path
import pickle

import numpy as np
from runtime import REPO, dump, file_identity, config_identity, paths, resolve_path
from segmentation import clean_grouped_labels, change_statistics


def check_config(cfg):
    source = cfg['source']
    expected = source.get('segmentation', {}).get('kind', 'embedded')
    values = {'case':cfg['case_name'], 'segmentation':expected,
              'output':cfg['output_dir'], 'work':cfg['work_dir'], 'sources':{}}
    for key in ('input','workspace'):
        if source.get(key):
            p = resolve_path(source[key], cfg)
            values['sources'][key] = {'path':str(p), 'exists':p.exists()}
    if source.get('segmentation', {}).get('path'):
        p = resolve_path(source['segmentation']['path'], cfg)
        values['sources']['segmentation'] = {'path':str(p), 'exists':p.exists()}
    return json.dumps(values, indent=2)


def load_workspace(cfg, *, prepared=False):
    work, _ = paths(cfg)
    local = work/'workspace.pkl'
    stamp = work/'workspace.json'
    if local.exists():
        if not stamp.exists() or json.loads(stamp.read_text())['config_identity'] != config_identity(cfg):
            raise ValueError('Workspace configuration changed; run prepare again or use a new --work directory')
        source = local
    elif cfg['source'].get('workspace'):
        source = resolve_path(cfg['source']['workspace'], cfg)
    else:
        raise FileNotFoundError('Run prepare before assets/capture, or explicitly configure source.workspace')
    with source.open('rb') as handle: ws = pickle.load(handle)
    provenance = ws.get_active_segmentation_provenance()
    expected = cfg['source'].get('segmentation', {}).get('kind', '')
    actual = str(provenance.get('path', provenance.get('source', '')))
    if expected == 'seg_true' and 'seg_true' not in actual:
        raise ValueError(f'The selected workspace does not contain seg_true: {actual}')
    if expected == 'nnunet' and not any(x in actual.lower() for x in ('nn_', 'nnunet', 'prediction')):
        raise ValueError(f'The selected workspace does not identify a model prediction: {actual}')
    return ws


def cleanup_groups(ws, cfg, labels):
    overrides = cfg.get('cleanup', {})
    present = set(int(x) for x in np.unique(labels) if x)
    groups, assigned = {}, set()
    for name, group in ws.skeleton_params.label_groups.items():
        ids = [int(ws.skeleton_params.label_map[x]) for x in group.get('labels', [])
               if x in ws.skeleton_params.label_map and int(ws.skeleton_params.label_map[x]) in present]
        if not ids: continue
        params = ws.skeleton_params.params_for_group(name).to_dict()
        params.update({k:overrides[k] for k in ('remove_small_cc','cc_filter_mode','min_cc_volume_mm3','cc_rel_min_ratio') if k in overrides})
        groups[name] = {'labels':ids, 'params':params}; assigned.update(ids)
    for label in sorted(present-assigned):
        groups[f'other_{label}'] = {'labels':[label], 'params':{
            'remove_small_cc':overrides.get('remove_small_cc',True), 'cc_filter_mode':'absolute',
            'min_cc_volume_mm3':float(overrides.get('min_cc_volume_mm3',50)), 'cc_rel_min_ratio':0}}
    return groups


def segmentation_review(cfg):
    work, _ = paths(cfg); ws = load_workspace(cfg)
    labels = ws.segmentation_analysis_labels
    if labels is None:
        from autoflow.core.pipeline import PipelineEngine
        PipelineEngine().preprocess(ws); labels = ws.segmentation_analysis_labels
    labels = np.asarray(labels)
    groups = cleanup_groups(ws, cfg, labels)
    cleaned = clean_grouped_labels(labels, ws.segmentation_spacing, groups)
    candidate = work/'segmentation-review.npz'
    np.savez_compressed(candidate, before=labels, after=cleaned,
                        spacing=ws.segmentation_spacing, origin=ws.origin)
    dump(work/'segmentation-review.json', {'case':cfg['case_name'],
        'source':ws.get_active_segmentation_provenance(), 'groups':groups,
        'frames':change_statistics(labels,cleaned),
        'before_sha256':hashlib.sha256(labels.tobytes()).hexdigest(),
        'after_sha256':hashlib.sha256(cleaned.tobytes()).hexdigest(),
        'metrics_recalculated':False})
    return candidate


def adapt_input(cfg, work):
    """The optional legacy adapter is explicit; normalized H5/DICOM use AutoFlow."""
    source = resolve_path(cfg['source']['input'], cfg)
    if cfg['source'].get('input_layout', 'native') != 'matlab_vtzyx': return source
    import h5py
    target = work/'input-normalized.h5'
    stamp = work/'input-normalized.json'
    identity = file_identity(source)
    csv_path = source.with_suffix('.csv')
    identity['csv'] = file_identity(csv_path)
    if target.exists() and stamp.exists() and json.loads(stamp.read_text()) == identity: return target
    with csv_path.open(newline='') as handle: row = next(csv.DictReader(handle))
    with h5py.File(source,'r') as src, h5py.File(target,'w') as dst:
        dst.attrs['autoflow_layout']='img_complex_xyztv'; dst.attrs['corr_layout']='corr_xyztv'
        for key,name in [('Resolution','resolution'),('VENC','VENC')]:
            dst.create_dataset(key,data=np.array([float(x) for x in row[name].split(';')],np.float32))
        dst.create_dataset('RR',data=np.float32(row['RR'])); dst.create_dataset('Origin',data=np.zeros(3,np.float32))
        for key,name in [('SpatialOrder','spatial_order'),('VENCOrder','venc_order')]:
            dst.create_dataset(key,data=np.array(row[name].split(';'),dtype=h5py.string_dtype()))
        image = np.asarray(src['img'],np.complex64).transpose(4,3,2,1,0)
        dst.create_dataset('img_complex',data=image)
        if 'corr' in src:
            corr=dst.create_dataset('corr',data=np.asarray(src['corr'],np.float32).transpose(4,3,2,1,0))
            for k,v in src['corr'].attrs.items():corr.attrs[k]=v
        if 'segmask' in src:dst.create_dataset('segmask',data=np.asarray(src['segmask']).transpose(2,1,0))
    dump(stamp,identity); return target


def attach_segmentation(ws, cfg, input_path):
    spec = cfg['source'].get('segmentation')
    if not spec or not spec.get('path'): return
    source = resolve_path(spec['path'],cfg)
    if spec.get('layout') != 'nrrd_raw':
        from autoflow.algorithms.segmentation import load_segmentation_file
        labels, provenance = load_segmentation_file(source, spatial_shape=ws.flow_raw.shape[:3], time_count=ws.time_count())
    else:
        import h5py, nrrd
        from autoflow.algorithms.data.orientation import _reorient_spatial_only
        labels,_ = nrrd.read(str(source))
        mag_path = resolve_path(spec['magnitude'],cfg)
        magnitude,_ = nrrd.read(str(mag_path))
        magnitude = magnitude[...,0] if magnitude.ndim == 4 else magnitude
        with h5py.File(input_path,'r') as handle:
            reference=np.abs(np.asarray(handle['img_complex'][:,:,:,0,0]))
            raw_order=[x.decode() if isinstance(x,bytes) else str(x) for x in handle['SpatialOrder'][...]]
        best=(-np.inf,None,None)
        for perm in itertools.permutations(range(3)):
            candidate=magnitude.transpose(perm)
            if candidate.shape != reference.shape: continue
            for flips in itertools.product((False,True),repeat=3):
                value=candidate[tuple(slice(None,None,-1 if x else 1) for x in flips)][::3,::3,::2].ravel()
                score=float(np.corrcoef(value,reference[::3,::3,::2].ravel())[0,1])
                if score>best[0]:best=(score,perm,flips)
        score,perm,flips=best
        if score < float(spec.get('minimum_alignment_correlation',.99)):
            raise ValueError(f'Segmentation magnitude alignment failed: {score}')
        labels=labels.transpose(tuple(perm)+(3,))
        for axis,flip in enumerate(flips):
            if flip: labels=np.flip(labels,axis)
        mapping={int(k):int(v) for k,v in spec.get('label_mapping',{}).items()}
        if mapping:
            unknown=set(int(x) for x in np.unique(labels))-set(mapping)
            if unknown:raise ValueError(f'Unmapped segmentation labels: {sorted(unknown)}')
            lut=np.zeros(max(mapping)+1,np.int16)
            for key,value in mapping.items():lut[key]=value
            labels=lut[labels]
        labels=_reorient_spatial_only(labels,raw_order,ws.spatial_order)
        provenance=dict(path=str(source),source=spec['kind'],axis_permutation=perm,
                        axis_flips=flips,alignment_correlation=score,label_mapping=mapping)
    if tuple(labels.shape)!=tuple(ws.flow_raw.shape[:4]):raise ValueError((labels.shape,ws.flow_raw.shape))
    provenance['inference_performed_for_this_run']=False
    ws.set_segmentation_source('imported',labels,provenance=provenance)
    ws.activate_segmentation_source('imported')


def make_planes(ws, spacing_mm):
    from autoflow.algorithms.planes import generate_planes_from_paths
    result=[]; audit=[]
    for index,path in enumerate(ws.centerline_paths_smooth or ws.centerline_paths):
        points=np.asarray(path,float)
        if len(points)<2:continue
        length=float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum())
        if length<=1e-7:continue
        count=1 if length<spacing_mm else max(1,int(np.floor((length-spacing_mm/2)/spacing_mm))+1)
        found,_=generate_planes_from_paths([points],cross_section_distance=spacing_mm,plane_mode='fixed_step',
            plane_count=count,anchor='center',direction='both',spacing_mode='distance',
            segmentation_filter=False,smoothing_method='none',inter_time=1)
        for plane in found:
            plane.path_index=index;plane.label=index+1
            plane.group_name=ws.path_info[index].get('group_name','') if index<len(ws.path_info) else ''
            ijk=np.rint(np.asarray(plane.center)/ws.segmentation_spacing).astype(int)
            ijk=np.clip(ijk,0,np.array(ws.segmentation_analysis_labels_3d.shape)-1)
            plane.segmentation_label=int(ws.segmentation_analysis_labels_3d[tuple(ijk)])
        distances=[float(p.distance) for p in found]
        if len(distances)>1:assert np.allclose(np.diff(distances),spacing_mm,atol=1e-4)
        if length<spacing_mm:assert len(found)==1 and np.isclose(distances[0],length/2,atol=1e-4)
        result.extend(found);audit.append(dict(path_index=index,length_mm=length,positions_mm=distances))
    return result,audit


def prepare_case(cfg):
    from autoflow.api import AutoFlowConfig, build_workspace
    from autoflow.core.pipeline import PipelineEngine
    from autoflow.core.models import StepId
    from autoflow.quality import build_quality_report, save_quality_report
    from autoflow.report_snapshot import build_report_snapshot, save_report_html
    work,_=paths(cfg); engine=PipelineEngine()
    if cfg['source'].get('workspace'):
        # Load the declared source directly; never silently reuse old metrics.
        source=resolve_path(cfg['source']['workspace'],cfg)
        with source.open('rb') as handle:ws=pickle.load(handle)
        expected=cfg['source'].get('segmentation',{}).get('kind')
        if expected=='seg_true' and 'seg_true' not in str(ws.get_active_segmentation_provenance()):
            raise ValueError('source.workspace is not the declared seg_true source')
        ws.reset_segmentation_results()
    else:
        input_path=adapt_input(cfg,work)
        options=AutoFlowConfig.from_config_dir(cfg['config_dir'],output_dir=str(work/'analysis'),
            background_phase_write_cache=False,write_segmentation_cache=False,autoseg=False,
            requested_metrics=['wss','tke','pg','vortex'],requested_videos=[])
        ws=build_workspace(options)
        engine.load_data(ws,print,input_source=str(input_path),reuse_existing_corr=True,apply_background_correction=True)
        attach_segmentation(ws,cfg,input_path)
    ws.paths.output_dir=str(work/'analysis'); ws.segmentation.write_auto_cache=False
    for key,value in cfg.get('cleanup',{}).items():
        if hasattr(ws.skeleton_params,key):setattr(ws.skeleton_params,key,value)
    ws.segmentation.cleanup_4d_components=False
    for group in ws.skeleton_params.label_groups.values():
        group.setdefault('preprocess',{})['remove_small_cc']=bool(cfg.get('cleanup',{}).get('remove_small_cc',True))
    if ws.segmask_raw is None:raise ValueError('Provide embedded or external segmentation before making figures')
    def step(which):
        outcome=engine.run_step(ws,which,print)
        if not outcome.success:raise RuntimeError(outcome.message)
    if not ws.phase_unwrap_result and cfg.get('analysis',{}).get('unwrap',True):
        ws.phase_unwrap_params.enabled=True;ws.phase_unwrap_params.method='lap4D';step(StepId.UNWRAP_PHASE)
    for which in (StepId.REMOVE_NOISE,StepId.GENERATE_PCMRA,StepId.GENERATE_SKELETON,StepId.GENERATE_GRAPH):step(which)
    ws.planes,audit=make_planes(ws,float(cfg['workflow'].get('plane_spacing_mm',10)))
    ws.pipeline.mark_done(StepId.GENERATE_PLANES)
    engine._ensure_derived_metrics(ws,save_pixelwise=False,refresh_scene_objects=True,
        compute_wss=True,compute_tke=True,compute_pressure_gradient=True,compute_vortex=True)
    step(StepId.COMPUTE_PLANE_METRICS)
    report=build_quality_report(ws,source_path=ws.paths.flow_path)
    save_quality_report(ws,str(work/'analysis/quality_report.json'),report=report)
    save_report_html(build_report_snapshot(ws,quality_report=report),str(work/'quality-report.html'))
    for obj in ws.scene_objects.values():obj.actor=obj.label_actor=None
    with (work/'workspace.pkl').open('wb') as handle:pickle.dump(ws,handle,protocol=5)
    dump(work/'workspace.json',{'config_identity':config_identity(cfg),'case':cfg['case_name'],
        'segmentation':ws.get_active_segmentation_provenance(),'planes':len(ws.planes),'plane_placement':audit,
        'metric_segmentation_sha256':hashlib.sha256(np.asarray(ws.segmask_binary).tobytes()).hexdigest(),
        'topology_segmentation_sha256':hashlib.sha256(np.asarray(ws.segmentation_analysis_binary).tobytes()).hexdigest()})
    return work/'workspace.pkl'
