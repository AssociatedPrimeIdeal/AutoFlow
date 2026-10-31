"""The ten-panel result figure, consolidated from the approved reference style."""


def render_results(cfg, *, stills=False):
    from pathlib import Path
    import os,sys,json,time,math,traceback
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    os.environ.setdefault('PYVISTA_OFF_SCREEN','true')
    os.environ.setdefault('VTK_DEFAULT_OPENGL_WINDOW','vtkEGLRenderWindow')
    import numpy as np
    import pyvista as pv
    import imageio.v2 as imageio
    from PIL import Image,ImageDraw,ImageFont
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import colormaps
    from PySide6 import QtWidgets
    from vtkmodules.vtkRenderingCore import vtkRenderer
    from scipy.ndimage import gaussian_filter
    from scipy.sparse import csr_matrix
    from runtime import REPO, font_path
    OUT=Path(cfg['output_dir'])/'results'; OUT.mkdir(parents=True,exist_ok=True)
    ASSET=Path(cfg['work_dir'])/'assets/results'
    sys.path.insert(0,str(REPO))
    from autoflow.ui.orientation_marker import anatomical_marker
    data=json.loads((ASSET/'assets.json').read_text())
    baseline_limits={m['name']:list(m['clim']) for m in data['metrics']}
    for m in data['metrics']:
     if m['name']=='wss':
      m['clim']=list(cfg.get('results',{}).get('wss_clim',[0.,5.]))
     elif m['name'] in cfg.get('results',{}).get('double_clim',[]):
      m['clim']=[2.*value for value in m['clim']]
    data['frame_timing']='Four cardiac cycles per level 360-degree turn; half previous cardiac speed'
    data['cardiac_cycles_per_turn']=4
    data['cardiac_playback_speed_ratio']=.5
    data['baseline_limits']=baseline_limits
    data['font_style']={'weight':'bold','title':32,'panel_letter':36,'units_and_ticks':26,'segmentation_labels':20,'group_labels':22,'cardiac_phase':28}
    (OUT/'figure-config.json').write_text(json.dumps(data,indent=2))
    center=np.asarray(data['center'])
    S=800;ART_H=660;TILE_H=760
    FPS=int(cfg.get('results',{}).get('fps',24))
    FRAMES=int(round(FPS*float(cfg.get('results',{}).get('rotation_seconds',float(data['rr_ms'])/1000.*8))))
    CYCLES=int(cfg.get('results',{}).get('cardiac_cycles_per_rotation',4))
    BASE_PHASE=int(data['base_phase']);N_PHASES=int(data['phase_count'])
    DRAFT=stills
    FONT=font_path(configured=cfg.get('render',{}).get('font'))
    BOLD=font_path(True,cfg.get('render',{}).get('bold_font'))
    title_font=ImageFont.truetype(BOLD,32)
    letter_font=ImageFont.truetype(BOLD,36)
    small_font=ImageFont.truetype(BOLD,26)
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    (OUT/'panels').mkdir(exist_ok=True);(OUT/'movies').mkdir(exist_ok=True)
    panels=[]
    def log(stage,**kw):
     (OUT/'render-status.json').write_text(json.dumps(dict(stage=stage,updated=time.time(),**kw),indent=2))
     print(stage,kw,flush=True)
    def camera(p,angle):
     r=math.radians(angle);direction=np.array([math.sin(r),-math.cos(r),0.])
     p.camera.position=tuple(center+direction*(350. if p is panels[0]['p'] else 500.))
     p.camera.focal_point=tuple(center)
     p.camera.up=(0.,0.,1.)
     p.camera.parallel_projection=p is not panels[0]['p']
     if p is panels[0]['p']:p.camera.view_angle=26.
     p.camera.parallel_scale=76. if p is panels[0]['p'] else float(data['parallel_scale'])
     p.renderer.ResetCameraClippingRange()
    def panel(name,title,letter):
     p=pv.Plotter(off_screen=True,window_size=(S,ART_H),border=False)
     p.set_background('white');p.enable_lightkit()
     try:p.enable_anti_aliasing('msaa')
     except Exception:pass
     entry=dict(name=name,title=title,letter=letter,p=p,metric=None)
     panels.append(entry)
     if name in ('magnitude-streamlines','topology-planes'):
      try:p.enable_depth_peeling(number_of_peels=8,occlusion_ratio=.05)
      except Exception:pass
     return entry
    def color_variant(color,light=0):
     a=np.asarray(matplotlib.colors.to_rgb(color))
     return tuple(a+(1-a)*light) if light>=0 else tuple(a*(1+light))
    def decorate(entry,raw):
     tile=Image.new('RGB',(S,TILE_H),'white')
     tile.paste(Image.fromarray(raw[:,:,:3]),(0,56))
     dr=ImageDraw.Draw(tile)
     dr.text((22,13),entry['letter'],font=letter_font,fill='#1c2932')
     dr.text((62,15),entry['title'],font=title_font,fill='#1c2932')
     metric=entry['metric']
     if metric:
      lo,hi=metric['clim'];x0,x1=190,748;y0=706;y1=723
      lut=(lookup(metric,x1-x0)[:,:3]*255).astype(np.uint8)
      bar=np.tile(lut[None,...],(y1-y0,1,1))
      tile.paste(Image.fromarray(bar),(x0,y0))
      dr.text((190,668),metric['units'],font=small_font,fill='#46515b')
      ticks=(list(range(int(lo),int(hi)+1)) if hi-lo<=5 else [int(lo),int(round((lo+hi)/2)),int(hi)])
      for v in ticks:
       x=x0+(x1-x0)*(v-lo)/(hi-lo)
       dr.text((x,730),str(v),font=small_font,fill='#46515b',anchor='mt')
     if entry['name']=='multilabel-segmentation':
      ids=data['segmentation']['labels'];bw=(S-52)/len(ids)
      font=ImageFont.truetype(BOLD,20)
      while max(dr.textbbox((0,0),data['segmentation']['names'][str(label)],font=font)[2] for label in ids)>bw-2:
       font=ImageFont.truetype(BOLD,font.size-1)
      for i,label in enumerate(ids):
       x=int(26+i*bw);x1=int(26+(i+1)*bw)-1
       dr.rectangle((x,706,x1,723),fill=palette[i%len(palette)])
       dr.text(((x+x1)/2,730),data['segmentation']['names'][str(label)],font=font,fill='#46515b',anchor='mt')
     if entry['name']=='topology-planes':
      group_names={'intracranial_anterior_arteries':'Anterior arteries','vertebrobasilar_arteries':'Vertebrobasilar','intracranial_veins':'Venous sinuses'}
      groups=data['groups'];bw=(S-52)/len(groups);font=ImageFont.truetype(BOLD,22)
      for i,g in enumerate(groups):
       x=int(26+i*bw);x1=int(26+(i+1)*bw)-1
       dr.rectangle((x,706,x1,723),fill=g['color'])
       dr.text(((x+x1)/2,730),group_names.get(g['name'],g['name'].replace('_',' ')),font=font,fill='#46515b',anchor='mt')
     return np.asarray(tile)
    def lookup(metric,n=256):
     x=np.linspace(0,1,n)
     if metric.get('name')=='streamlines':return colormaps[metric['cmap']](x)
     if metric.get('signed'):
      x=.5+.5*np.sign(2*x-1)*np.abs(2*x-1)**.65
     else:x=x**(.55 if metric['name'] in ('tke','pressure-gradient') else .8)
     return colormaps[metric['cmap']](x)
    def add_scalar(entry,mesh,metric):
     entry['metric']=metric;entry['values']=np.load(ASSET/metric['data'])['values']
     mesh['Value']=entry['values'][BASE_PHASE]
     entry['mesh']=mesh
     entry['actor']=entry['p'].add_mesh(mesh,scalars='Value',cmap=matplotlib.colors.ListedColormap(lookup(metric)),clim=metric['clim'],
      opacity=1.,show_scalar_bar=False,nan_color='#dde4e9',nan_opacity=0.,smooth_shading=True,lighting=False)
    try:
     log('build revised scenes')
     a=panel('magnitude-streamlines','Magnitude + streamlines','A')
     mag_mesh=pv.read(ASSET/data['magnitude']['geometry'])
     mag_values=np.load(ASSET/data['magnitude']['data'])['values']
     mag_mesh['Magnitude']=mag_values[BASE_PHASE]
     lo,hi=data['magnitude']['clim']
     levels=np.linspace(lo,hi,256)
     alpha=np.where(levels>lo+(hi-lo)*data['magnitude']['threshold'],255,0).astype(np.uint8)
     mag_actor=a['p'].add_mesh(mag_mesh,scalars='Magnitude',cmap='gray',clim=[lo,hi],opacity=alpha,
      show_scalar_bar=False,lighting=False)
     mag_actor.GetProperty().SetOpacity(data['magnitude']['opacity'])
     stream_meshes=[pv.read(ASSET/f) for f in data['streamlines']['meshes']]
     stream_actor=a['p'].add_mesh(stream_meshes[BASE_PHASE],scalars='Speed',cmap='turbo',
      clim=data['streamlines']['clim'],show_scalar_bar=False,line_width=1.6,render_lines_as_tubes=True,lighting=False)
     a['p'].renderer.RemoveActor(stream_actor)
     fg=vtkRenderer();fg.SetLayer(2);fg.SetBackgroundAlpha(0);fg.SetPreserveColorBuffer(True);fg.SetPreserveDepthBuffer(False)
     fg.SetActiveCamera(a['p'].camera);fg.AddActor(stream_actor);fg.InteractiveOff()
     a['p'].render_window.SetNumberOfLayers(3);a['p'].render_window.AddRenderer(fg);a['foreground']=fg
     a['metric']=dict(name='streamlines',units='Velocity (cm/s)',clim=data['streamlines']['clim'],cmap='turbo')
     a['p'].add_orientation_widget(anatomical_marker(np.ones(3)),interactive=False,viewport=(.005,.005,.205,.245))
     b=panel('multilabel-segmentation','Multilabel segmentation','B')
     ids=data['segmentation']['labels']
     palette=['#ee6c71','#cdb34b','#489bd2','#48b1a2','#df9a51','#aa7acc','#75bccc',
              '#e06e9d','#766ac9','#8ebc62','#ed9271','#567dc2','#c3a0d5','#4cae83']
     lut=np.zeros((max(ids)+1,3))
     for n,label in enumerate(ids):lut[label]=matplotlib.colors.to_rgb(palette[n%len(palette)])
     seg_meshes=[pv.read(ASSET/f) for f in data['segmentation']['meshes']]
     for multi in seg_meshes:multi['RGB']=(lut[np.asarray(multi['Label'],int)]*255).astype(np.uint8)
     seg_actor=b['p'].add_mesh(seg_meshes[BASE_PHASE],scalars='RGB',rgb=True,smooth_shading=True,show_scalar_bar=False,
      ambient=.35,diffuse=.72,specular=.18,specular_power=25)
     c=panel('topology-planes','Centerlines + planes','C')
     for g in data['groups']:
      c['p'].add_mesh(pv.read(ASSET/g['mesh']),color=color_variant(g['color'],.12),opacity=.10,
       smooth_shading=True,show_scalar_bar=False,ambient=.45,diffuse=.65)
     for path in data['paths']:
      points=np.asarray(path['points']);line=pv.lines_from_points(points)
      c['p'].add_mesh(line.tube(radius=.62,n_sides=8),color=color_variant(path['color'],-.38),
       smooth_shading=True,show_scalar_bar=False,ambient=.7,diffuse=.5)
     for plane in data['planes']:
      plane_mesh=pv.Plane(center=plane['center'],direction=plane['normal'],i_size=8.5,j_size=8.5,i_resolution=1,j_resolution=1)
      col=color_variant(plane['color'],-.26)
      c['p'].add_mesh(plane_mesh,color=col,opacity=.32,show_scalar_bar=False,lighting=False)
      edge=plane_mesh.extract_feature_edges(boundary_edges=True,feature_edges=False,non_manifold_edges=False,manifold_edges=False)
      c['p'].add_mesh(edge,color=col,line_width=3.2,show_scalar_bar=False,lighting=False)
     specs=[('wss','Wall shear stress','D'),('tke','Turbulent kinetic energy','E'),
            ('relative-pressure','Relative pressure','F'),('pressure-gradient','Pressure gradient','G'),
            ('vorticity','Vorticity','H'),('q-criterion','Q-criterion','I'),('swirling-strength','Swirling strength','J')]
     for name,title,letter in specs:
      entry=panel(name,title,letter);metric=next((x for x in data['metrics'] if x['name']==name),None)
      if metric is None:
       entry['p'].add_text('Not available',position='lower_left',font_size=16,color='#6e6e73')
       continue
      add_scalar(entry,pv.read(ASSET/metric.get('geometry','metric-surface.vtp')),metric)
     last_phase=[None]
     def render_angle(angle,phase=BASE_PHASE):
      if last_phase[0]!=phase:
       mag_mesh['Magnitude']=mag_values[phase];mag_actor.GetMapper().dataset=mag_mesh
       stream_actor.GetMapper().dataset=stream_meshes[phase]
       seg_actor.GetMapper().dataset=seg_meshes[phase]
       for entry in panels:
        if 'values' in entry:
         entry['mesh']['Value']=entry['values'][phase]
         entry['actor'].GetMapper().dataset=entry['mesh']
       last_phase[0]=phase
      tiles=[]
      for entry in panels:
       camera(entry['p'],angle);entry['p'].render()
       tiles.append(decorate(entry,entry['p'].screenshot(return_img=True)))
      return tiles
     def compose(tiles,phase=BASE_PHASE):
      body=np.concatenate([np.concatenate(tiles[:5],axis=1),np.concatenate(tiles[5:],axis=1)],axis=0)
      im=Image.new('RGB',(S*5,TILE_H*2+48),'white');im.paste(Image.fromarray(body),(0,0))
      dr=ImageDraw.Draw(im);font=ImageFont.truetype(BOLD,28)
      dr.text((22,TILE_H*2+12),f"Cardiac phase {phase+1} / {N_PHASES}   |   {phase*data['rr_ms']/N_PHASES:.0f} ms",font=font,fill='#46515b')
      x0,x1=900,S*5-40;y=TILE_H*2+25
      dr.line((x0,y,x1,y),fill='#e0e7ed',width=4)
      x=x0+(x1-x0)*phase/(N_PHASES-1)
      dr.ellipse((x-7,y-7,x+7,y+7),fill='#248c9c')
      return np.asarray(im)
     tiles=render_angle(0,BASE_PHASE)
     for entry,tile in zip(panels,tiles):Image.fromarray(tile).save(OUT/'panels'/(entry['letter']+'-'+entry['name']+'.png'),dpi=(300,300))
     Image.fromarray(compose(tiles)).save(OUT/'results-overview.png',dpi=(300,300))
     sample=[]
     for i,angle in enumerate((0,45,90,135)):
      phase=(BASE_PHASE+i*4)%N_PHASES
      sample.append(Image.fromarray(compose(render_angle(angle,phase),phase)).resize((1800,706),Image.Resampling.LANCZOS))
     check=Image.new('RGB',(1800,706*len(sample)),'white')
     for i,im in enumerate(sample):check.paste(im,(0,706*i))
     check.save(OUT/'orbit-check.png')
     settings=dict(data)
     settings['display_surface_interpolation']='Shared cleaned segmentation surface and five local validity-weighted interpolation passes; display only'
     settings['anti_aliasing']='msaa'
     settings['transparent_shell_opacity']=0.
     settings['valid_opacity']=1.
     settings['cardiac_cycles_per_turn']=CYCLES
     settings['cardiac_playback_speed_ratio']=.5
     settings['rotation_speed_ratio']=1.
     settings['phase_index_by_frame']=[int(BASE_PHASE+frame/FRAMES*CYCLES*N_PHASES)%N_PHASES for frame in range(FRAMES)]
     settings.update(orbit_axis=[0,0,1],camera_start=[0,-1,0],camera_elevation_degrees=0,
      full_rotation_degrees=360,frames=FRAMES,fps=FPS,layout='2 rows x 5 columns',panel_order=[p['letter']+': '+p['title'] for p in panels])
     settings.pop('paths',None)
     (OUT/'render-settings.json').write_text(json.dumps(settings,indent=2))
     if DRAFT:
      log('draft complete',planes=len(data['planes']))
     else:
      log('render level synchronized orbit',frame=0,total=FRAMES)
      writers={p['name']:imageio.get_writer(str(OUT/'movies'/(p['letter']+'-'+p['name']+'.mp4')),
       fps=FPS,codec='libx264',quality=8,macro_block_size=2,ffmpeg_params=['-pix_fmt','yuv420p']) for p in panels}
      together=imageio.get_writer(str(OUT/'results-rotation.mp4'),fps=FPS,codec='libx264',quality=8,
       macro_block_size=2,ffmpeg_params=['-pix_fmt','yuv420p'])
      try:
       for frame in range(FRAMES):
        phase=settings['phase_index_by_frame'][frame]
        tiles=render_angle(frame*360./FRAMES,phase)
        for p,tile in zip(panels,tiles):writers[p['name']].append_data(tile)
        together.append_data(compose(tiles,phase))
        if frame%12==0:log('render level synchronized orbit',frame=frame,total=FRAMES)
      finally:
       for w in writers.values():w.close()
       together.close()
      log('render complete',frames=FRAMES,total_seconds=FRAMES/FPS)
    finally:
     for entry in panels:entry['p'].close()
