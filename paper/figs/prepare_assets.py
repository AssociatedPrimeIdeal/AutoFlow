"""Export current measurements once for the workflow and result renderers."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pyvista as pv
from matplotlib import colormaps
from matplotlib.colors import to_rgb, ListedColormap
from PIL import Image
from runtime import dump, paths, logical_identity
from prepare_case import load_workspace, make_planes
from segmentation import labeled_surface

PALETTE=['#ee6c71','#cdb34b','#489bd2','#48b1a2','#df9a51','#aa7acc','#75bccc',
         '#e06e9d','#766ac9','#8ebc62','#ed9271','#567dc2','#c3a0d5','#4cae83']
METRICS=[('wss','wss_surface_live','wss','WSS (Pa)',False),
         ('tke','tke_volume','TKE','TKE (J/m³)',False),
         ('relative-pressure','relative_pressure_volume','RelativePressure','Pressure (Pa)',True),
         ('pressure-gradient','pressure_gradient_volume','PressureGradient','PG (Pa/m)',False),
         ('vorticity','vorticity_magnitude_volume','Vorticity Magnitude','Vorticity (s⁻¹)',False),
         ('q-criterion','q_criterion_volume','Q-Criterion','Q (s⁻²)',True),
         ('swirling-strength','swirling_strength_volume','Swirling Strength','Swirling strength (s⁻¹)',False)]


def camera(plotter, center, scale, angle):
    a=math.radians(angle)
    plotter.camera.position=tuple(center+500*np.array([math.sin(a),-math.cos(a),0.]))
    plotter.camera.focal_point=tuple(center);plotter.camera.up=(0,0,1)
    plotter.camera.parallel_projection=True;plotter.camera.parallel_scale=scale
    plotter.renderer.ResetCameraClippingRange()


def prepare_assets(cfg):
    from scipy.spatial import cKDTree
    from autoflow.algorithms.surfaces import build_cell_mask_surface
    from autoflow.algorithms.streamlines import generate_streamlines_at_t
    from autoflow.rendering.datasets import metric_display_mesh
    from autoflow.rendering.style import configured_metric_style, rounded_display_clim
    from autoflow.rendering.videos import _build_group_surfaces, _path_group_name
    from autoflow.ui.viewer import SceneController
    from PySide6 import QtWidgets

    work,_=paths(cfg); root=work/'assets'; result=root/'results'; workflow=root/'workflow'
    for p in (result,workflow):p.mkdir(parents=True,exist_ok=True)
    ws=load_workspace(cfg,prepared=True)
    from display_geometry import display_paths
    ws.centerline_paths_smooth,display_fit=display_paths(ws,cfg)
    labels=np.asarray(ws.segmentation_analysis_labels)
    mask=np.asarray(ws.segmentation_analysis_binary,dtype=bool)
    if not cfg.get('cleanup',{}).get('remove_small_cc',True):
        if not np.array_equal(labels>0,mask):
            raise ValueError('The prepared quantitative mask excludes segmentation voxels despite cleanup=false')
        if not np.array_equal(np.asarray(ws.get_active_segmentation())>0,ws.segmask_binary):
            raise ValueError('The native measurement mask differs from the declared original segmentation')
    display_labels=np.where(mask,labels,0)
    spacing=np.asarray(ws.segmentation_spacing);origin=np.asarray(ws.origin);n=ws.time_count()
    points=np.argwhere(np.any(mask,axis=3))*spacing+origin
    center=(points.min(0)+points.max(0))/2;extent=np.ptp(points,axis=0)
    radial=float(np.linalg.norm((points-center)[:,:2],axis=1).max())
    data=dict(case=cfg['case_name'],source=ws.get_active_segmentation_provenance(),analysis_identity=logical_identity(cfg),
        mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),
        native_metric_mask_sha256=hashlib.sha256(np.asarray(ws.segmask_binary).tobytes()).hexdigest(),
        phase_count=n,base_phase=int(ws.current_t),
        rr_ms=float(ws.rr),center=center.tolist(),parallel_scale=max(radial*.825*1.1,float(extent[2])*.58),
        groups=[],paths=[],planes=[],metrics=[],unavailable_metrics=[])
    groups=_build_group_surfaces(ws,smoothing_iteration=60)
    for i,(name,surface,color) in enumerate(groups):
        filename=f'group-{i}.vtp';surface.save(result/filename)
        data['groups'].append(dict(name=name,color=color,mesh=filename,labels=ws.multilabel_groups[name]['labels']))
    for i,path in enumerate(ws.centerline_paths_smooth or ws.centerline_paths):
        group=_path_group_name(ws,i)
        data['paths'].append(dict(index=i,group=group,points=(np.asarray(path)+origin).tolist(),
            color=ws.skeleton_params.scene_color_for_group(group,'path'),
            plane_color=ws.skeleton_params.scene_color_for_group(group,'plane')))
    figure_planes,_=make_planes(ws,float(cfg.get('results',{}).get('plane_spacing_mm',20)))
    for p in figure_planes:
        group=_path_group_name(ws,p.path_index)
        data['planes'].append(dict(path_index=p.path_index,group=group,center=(p.center+origin).tolist(),
            normal=np.asarray(p.normal).tolist(),color=ws.skeleton_params.scene_color_for_group(group,'plane')))
    names={int(v):k for k,v in ws.skeleton_params.label_map.items()};seg_meshes=[];ids=set()
    for t in range(n):
        mesh=labeled_surface(display_labels[...,t],spacing,origin)
        if mesh is None:raise ValueError(f'Empty segmentation at phase {t}')
        filename=f'segmentation-{t:02d}.vtp';mesh.save(result/filename);seg_meshes.append(mesh)
        ids.update(int(x) for x in np.unique(mesh['Label']))
    data['segmentation']=dict(meshes=[f'segmentation-{t:02d}.vtp' for t in range(n)],
        labels=sorted(ids),names={str(i):names.get(i,str(i)) for i in ids},
        source_mask='segmentation_analysis_labels restricted to segmentation_analysis_binary',
        display_smoothing={'surface_iterations':60,'taubin_iterations':80,'pass_band':.08})
    print('segmentation assets',n,'phases',flush=True)
    metric_meshes={}
    for name,key,scalar,units,signed in METRICS:
        meshes=[metric_display_mesh(ws,key,t) for t in range(n)]
        if any(m is None for m in meshes):
            data['unavailable_metrics'].append(name);continue
        if scalar not in meshes[0].array_names:
            matches=[x for x in meshes[0].array_names if x.lower() in (scalar.lower(),'tke_mj')]
            if not matches:raise ValueError((name,meshes[0].array_names))
            scalar=matches[0]
        shape=meshes[0].copy(deep=True);shape.clear_data();geometry=name+'-surface.vtp';shape.save(result/geometry)
        if any(m.n_points!=shape.n_points or not np.allclose(m.points,shape.points) for m in meshes):
            raise ValueError(f'{name}: temporal display geometry changed; export per-phase meshes before rendering')
        values=np.stack([np.asarray(m[scalar],np.float32) for m in meshes]);finite=values[np.isfinite(values)]
        if not len(finite):data['unavailable_metrics'].append(name);continue
        upper=max(float(np.percentile(np.abs(finite) if signed else finite,95)),1e-6)
        limits=list(rounded_display_clim((-upper,upper) if signed else (0,upper)))
        filename=name+'-cardiac.npz';np.savez_compressed(result/filename,values=values)
        style=configured_metric_style(ws,key)
        data['metrics'].append(dict(name=name,key=key,scalar=scalar,units=units,signed=signed,
            clim=limits,cmap=style['cmap'],color_gamma=style['color_gamma'],data=filename,geometry=geometry))
        metric_meshes[name]=meshes
        print('metric assets',name,n,'phases',flush=True)
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    helper=pv.Plotter(off_screen=True,window_size=(16,16));scene=SceneController(helper,ws,lambda _:None)
    mag=cfg.get('results',{}).get('magnitude',{})
    ws.render_settings.setdefault('content_renders',{})['magnitude_volume']=dict(content=3,mode='slices',
        planes=[True,True,True],positions=mag.get('positions',[.125,.77,.05]),handles=False,
        threshold=mag.get('threshold',.06),shade=False,plane_index=None)
    mag_values=[]
    for t in range(n):
        ws.current_t=t;cut=scene._build_dataset('magnitude_volume')
        if t==0:shape=cut.copy(deep=True);shape.clear_data();shape.save(result/'magnitude-planes.vtp')
        mag_values.append(np.asarray(cut['Magnitude'],np.float32))
    helper.close();np.savez_compressed(result/'magnitude-cardiac.npz',values=np.stack(mag_values))
    window=float(mag.get('window',.5155));level=float(mag.get('level',.2578))
    data['magnitude']=dict(mag,geometry='magnitude-planes.vtp',data='magnitude-cardiac.npz',clim=[level-window/2,level+window/2])
    data['magnitude'].setdefault('threshold',.06);data['magnitude'].setdefault('opacity',.85)
    stream_meshes=[];speed=[];prm=ws.streamline_params
    for t in range(n):
        mesh=generate_streamlines_at_t(ws.flow_raw,t,None,ws.resolution,ws.origin,
            mask_3d=ws.segmask_binary[...,t],max_steps=prm.max_steps,terminal_speed=prm.terminal_speed,
            seed_ratio=prm.seed_ratio,min_seeds=prm.min_seeds,rng_seed=prm.rng_seed,
            seed_distribution=prm.seed_distribution,seed_mode=prm.seed_mode,seed_count=prm.seed_count,
            grid_spacing_mm=prm.grid_spacing_mm,integration_step_mm=prm.integration_step_mm,
            integration_backend=prm.integration_backend)
        mesh['Speed']=np.asarray(mesh['Velocity'])*100.;mesh.save(result/f'streamlines-{t:02d}.vtp')
        speed.append(np.asarray(mesh['Speed']));stream_meshes.append(mesh)
    all_speed=np.concatenate([x for x in speed if len(x)]) if any(len(x) for x in speed) else np.array([1.])
    data['streamlines']=dict(meshes=[f'streamlines-{t:02d}.vtp' for t in range(n)],cmap='turbo',units='Velocity (cm/s)',
        clim=list(rounded_display_clim((0,float(np.percentile(all_speed,99))))))
    dump(result/'assets.json',data)
    candidates=[i for i,p in enumerate(ws.planes) if p.metrics.get('flowrate_mL_s') is not None]
    wanted=cfg.get('workflow',{}).get('selected_plane','auto')
    index=int(wanted) if wanted!='auto' else max(candidates,key=lambda i:float(np.nanmax(np.abs(ws.planes[i].metrics['flowrate_mL_s']))))
    p=ws.planes[index];m=p.metrics
    plane_metrics=dict(plane_index=index,flowrate=np.asarray(m['flowrate_mL_s']).tolist(),
        area=np.asarray(m['area_mm2']).tolist(),peak_velocity=np.asarray(m['peakv_cm_s_t']).tolist(),rr_ms=float(ws.rr))
    dump(workflow/'plane-metrics.json',plane_metrics)
    # Preserve the approved voxel/metric styles; all three products use one case.
    import shutil
    from prepare_workflow_panels import prepare_workflow_panels
    (workflow/'topology').mkdir(exist_ok=True)
    total=round(float(cfg['workflow'].get('rotation_seconds',6))*30)
    view=cfg['workflow'].get('camera',{})
    workflow_center=np.asarray(view.get('center',center.tolist()))
    workflow_scale=float(view.get('segmentation_scale',max(extent[2]/2,max(extent[:2])/(2*1120/680))*1.22))
    plot=pv.Plotter(off_screen=True,window_size=(1120,680));plot.set_background('white');plot.enable_lightkit()
    for name,_,color in groups:
        support=ws.multilabel_groups[name]['geometry_mask_3d']
        surface=build_cell_mask_surface(support,spacing,origin-.5*spacing,smooth_iter=1000)
        plot.add_mesh(surface,color=color,opacity=.24,smooth_shading=True,ambient=.6,diffuse=.4,specular=.02,show_scalar_bar=False)
    for row in data['paths']:
        points=np.asarray(row['points'])
        if len(points)>1:plot.add_mesh(pv.lines_from_points(points).tube(radius=.4,n_sides=12),color=row['color'],lighting=False,show_scalar_bar=False)
    for frame in range(total):
        camera(plot,workflow_center,workflow_scale,frame*360/total);plot.render();plot.screenshot(str(workflow/'topology'/f'{frame:03d}.png'))
    plot.close()
    prepare_workflow_panels(cfg,workspace=ws)
    replacement=work/'workflow-panels'
    for name in ('segmentation','hemodynamics'):shutil.copytree(replacement/name,workflow/name,dirs_exist_ok=True)
    metadata=json.loads((replacement/'assets.json').read_text())
    metadata.update(case=cfg['case_name'],mask_sha256=data['mask_sha256'],selected_plane_index=index,
                    center=workflow_center.tolist(),parallel_scale=workflow_scale)
    dump(workflow/'assets.json',metadata)
    from correction import prepare_correction
    prepare_correction(ws,cfg,workflow)
    print('assets complete',flush=True)
