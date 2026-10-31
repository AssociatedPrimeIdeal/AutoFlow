"""Render segmentation and metrics with the approved workflow presentation style."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import pyvista as pv
from matplotlib import colormaps
from matplotlib.colors import to_rgb, ListedColormap
from runtime import dump, paths, logical_identity
from prepare_case import load_workspace
from prepare_assets import PALETTE


def prepare_workflow_panels(cfg, workspace=None):
    from autoflow.algorithms.surfaces import build_cell_mask_surface
    from autoflow.algorithms.streamlines import generate_streamlines_at_t
    from autoflow.rendering.datasets import metric_display_mesh
    work,_=paths(cfg);out=work/'workflow-panels';out.mkdir(exist_ok=True)
    ws=load_workspace(cfg,prepared=True) if workspace is None else workspace;n=ws.time_count()
    labels=np.where(ws.segmentation_analysis_binary,ws.segmentation_analysis_labels,0).astype(np.int16)
    spacing=np.asarray(ws.segmentation_spacing);origin=np.asarray(ws.origin)
    points=np.argwhere(ws.segmask_3d>0)*ws.resolution+origin;extent=np.ptp(points,axis=0)
    view=cfg.get('workflow',{}).get('camera',{})
    center=np.asarray(view.get('center',((points.min(0)+points.max(0))/2).tolist()))
    seg_scale=float(view.get('segmentation_scale',max(extent[2]/2,max(extent[:2])/(2*1120/680))*1.22))
    hemo_scale=float(view.get('hemodynamics_scale',max(extent[2]/2,max(extent[:2])/(2*1280/800))*1.17))
    def camera(plot,angle,scale):
        a=math.radians(angle);plot.camera.position=tuple(center+500*np.array([math.sin(a),-math.cos(a),0.]))
        plot.camera.focal_point=tuple(center);plot.camera.up=(0,0,1);plot.camera.parallel_projection=True
        plot.camera.parallel_scale=scale;plot.renderer.ResetCameraClippingRange()
    for name in ('segmentation','hemodynamics'):(out/name).mkdir(exist_ok=True)
    total=round(float(cfg['workflow'].get('rotation_seconds',6))*30)
    cycles=float(cfg['workflow'].get('cardiac_cycles_per_rotation',3))
    lut=np.zeros((int(labels.max())+1,3))
    for value in (int(x) for x in np.unique(labels) if x):lut[value]=to_rgb(PALETTE[(value-17 if value>=17 else value-1)%len(PALETTE)])
    rgba=[]
    for t in range(n):
        values=labels[...,t];array=np.zeros(values.shape+(4,),np.uint8)
        array[...,:3]=(lut[values]*255).astype(np.uint8);array[...,3]=np.where(values>0,255,0)
        rgba.append(array.reshape(-1,4,order='F'))
    plot=pv.Plotter(off_screen=True,window_size=(1120,680));plot.set_background('white');plot.enable_lightkit()
    grid=pv.ImageData(dimensions=labels.shape[:3],spacing=spacing,origin=origin);grid['RGBA']=rgba[0]
    actor=plot.add_volume(grid,scalars='RGBA',show_scalar_bar=False,shade=True,ambient=.5,diffuse=.7,
        specular=.04,opacity_unit_distance=float(min(spacing))*.5,reset_camera=False)
    actor.prop.interpolation_type='nearest'
    for frame in range(total):
        phase=int(frame/total*cycles*n)%n;grid['RGBA']=rgba[phase]
        camera(plot,360*frame/total,seg_scale);plot.render();plot.screenshot(str(out/'segmentation'/f'{frame:03d}.png'))
    plot.close();print('original-style voxel segmentation ready',flush=True)
    groups=[]
    for name,group in ws.skeleton_params.label_groups.items():
        ids=[int(ws.skeleton_params.label_map[x]) for x in group.get('labels',[]) if x in ws.skeleton_params.label_map]
        mask=np.isin(ws.segmentation_analysis_labels_3d,ids)
        if not np.any(mask):continue
        surface=build_cell_mask_surface(mask,spacing,origin-.5*spacing,smooth_iter=1000)
        if surface is not None:groups.append(surface)
    meshsets={};metric_ranges={}
    for mode,key,scalar,cmap,signed in [('wss','wss_surface_live','wss','turbo',False),
        ('pressure','relative_pressure_volume','RelativePressure','RdBu_r',True),('tke','tke_volume','TKE','magma',False)]:
        meshes=[metric_display_mesh(ws,key,t) for t in range(n)]
        if any(mesh is None for mesh in meshes):continue
        if scalar not in meshes[0].array_names:
            scalar=next((x for x in meshes[0].array_names if x.lower() in ('tke','tke_mj')),scalar)
        observed=np.concatenate([np.asarray(mesh[scalar])[np.isfinite(mesh[scalar])] for mesh in meshes])
        if not len(observed):continue
        high=max(float(np.ceil(np.percentile(np.abs(observed),98))),1.)
        limits=[-high if signed else 0,high];x=np.linspace(0,1,256);gamma=.65 if signed else .8
        x=.5+.5*np.sign(2*x-1)*np.abs(2*x-1)**gamma if signed else x**gamma
        colors=colormaps[cmap](x)
        meshsets[mode]=(meshes,scalar,ListedColormap(colors),limits)
        metric_ranges[mode]={'label':{'wss':'WSS (Pa)','pressure':'Relative pressure (Pa)','tke':'TKE (J/m³)'}[mode],
                             'clim':limits,'colors':colors[:,:3].tolist()}
        print('measured frames',mode,n,flush=True)
    streams=[];prm=ws.streamline_params
    for t in range(n):
        mesh=generate_streamlines_at_t(ws.flow_raw,t,None,ws.resolution,ws.origin,mask_3d=ws.segmask_binary[...,t],
            max_steps=prm.max_steps,terminal_speed=prm.terminal_speed,seed_ratio=prm.seed_ratio,min_seeds=prm.min_seeds,
            rng_seed=prm.rng_seed,seed_distribution=prm.seed_distribution,seed_mode=prm.seed_mode,seed_count=prm.seed_count,
            grid_spacing_mm=prm.grid_spacing_mm,integration_step_mm=prm.integration_step_mm,integration_backend=prm.integration_backend)
        mesh['Speed']=np.asarray(mesh['Velocity'])*100.;streams.append(mesh.tube(radius=.17,n_sides=6))
    values=np.concatenate([mesh['Speed'] for mesh in streams]);high=max(1.,float(np.percentile(values,99)))
    meshsets['streamlines']=(streams,'Speed','turbo',[0,high]);metric_ranges['streamlines']={
        'label':'Velocity (cm/s)','clim':[0,high],'colors':colormaps['turbo'](np.linspace(0,1,256))[:,:3].tolist()}
    plot=pv.Plotter(off_screen=True,window_size=(1280,800));plot.set_background('white')
    for surface in groups:plot.add_mesh(surface,color='#a8becb',opacity=.045,lighting=False,show_scalar_bar=False)
    order=[x for x in ('streamlines','wss','pressure','tke') if x in meshsets];actor=None;timeline=[]
    for frame in range(total):
        phase=int(frame/total*cycles*n)%n;mode=order[min(len(order)-1,int(frame/total*len(order)))]
        if actor is not None:plot.remove_actor(actor,render=False)
        meshes,scalar,cmap,limits=meshsets[mode]
        actor=plot.add_mesh(meshes[phase],scalars=scalar,cmap=cmap,clim=limits,nan_opacity=0,
            show_scalar_bar=False,lighting=False,reset_camera=False)
        camera(plot,360*frame/total,hemo_scale);plot.render();plot.screenshot(str(out/'hemodynamics'/f'{frame:03d}.png'))
        timeline.append(dict(frame=frame,mode=mode,phase=phase,azimuth_deg=360*frame/total))
    plot.close()
    dump(out/'assets.json',{'source':ws.get_active_segmentation_provenance(),'phase_count':n,'analysis_identity':logical_identity(cfg),
        'frames_per_rotation':total,'continuous_hemodynamics':timeline,'metric_ranges':metric_ranges,
        'native_metric_mask_sha256':hashlib.sha256(np.asarray(ws.segmask_binary).tobytes()).hexdigest(),
        'topology_mask_sha256':hashlib.sha256(np.asarray(ws.segmentation_analysis_binary).tobytes()).hexdigest(),
        'cleanup':cfg.get('cleanup'), 'render_style':'Original v22 categorical voxel volume and original hemodynamic colormaps/opacity',
        'camera':{'center':center.tolist(),'segmentation_scale':seg_scale,'hemodynamics_scale':hemo_scale}})
    print('replacement assets complete',flush=True)
