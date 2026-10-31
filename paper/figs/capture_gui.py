"""Capture the real Qt GUI, including synchronized plane selection and readouts."""
from __future__ import annotations
import os
from pathlib import Path
import subprocess
import sys
from runtime import dump, paths


def capture_gui(cfg, *, intro_only=False):
    work,_=paths(cfg); config=work/'capture-config.json';dump(config,cfg)
    command=[sys.executable,__file__,str(config)]
    if intro_only:command.append('--intro-only')
    subprocess.run(command,check=True)


def _capture(cfg, *, intro_only=False):
    os.environ.update(QT_QPA_PLATFORM='offscreen',QT_SCALE_FACTOR='1.5',AUTOFLOW_SSH_RENDERING='1')
    import copy,json,math,time
    import numpy as np
    from PIL import Image
    from scipy.ndimage import label as connected
    from PySide6 import QtWidgets,QtCore
    from autoflow.ui.app import MainWindow,configure_high_dpi,apply_application_theme
    from autoflow.core.models import ObjectKind
    from autoflow.api import AutoFlowConfig,build_workspace
    from prepare_case import adapt_input,load_workspace

    work,_=paths(cfg);root=work/'assets/workflow';gui=root/'gui';gui.mkdir(parents=True,exist_ok=True)
    configure_high_dpi();app=QtWidgets.QApplication([]);apply_application_theme(app,'light')
    QtCore.QSettings.setDefaultFormat(QtCore.QSettings.IniFormat)
    QtCore.QSettings.setPath(QtCore.QSettings.IniFormat,QtCore.QSettings.UserScope,str(work/'gui-preferences'))
    window=MainWindow(cfg['config_dir']);apply_application_theme(app,'light')
    app.setStyleSheet(app.styleSheet()+'\nQWidget { font-size:16px; } QTreeView#browserObjects { font-size:16px; }')
    window.resize(2212,1352);window.show();window.scene.set_background('white')
    def events(seconds=.1):
        until=time.monotonic()+seconds
        while time.monotonic()<until:app.processEvents();time.sleep(.008)
    def rect(widget):
        p=widget.mapTo(window,QtCore.QPoint(0,0));return [p.x(),p.y(),widget.width(),widget.height()]
    def capture(name,widget=None):
        window.plotter.render()
        if hasattr(window.plotter,'_capture_frame'):window.plotter._capture_frame()
        events();(widget or window).grab().save(str(gui/(name+'.png')))
    window._select_workflow_page('correction');events(.3);capture('empty-correction')
    # A completed workspace is not a valid post-import state, even if its
    # actors are hidden: the Browser and processing status still expose results.
    input_path=adapt_input(cfg,work)
    options=AutoFlowConfig.from_config_dir(cfg['config_dir'],output_dir=str(work/'intro'),
        ignore_embedded_segmentation=True,background_phase_write_cache=False,
        write_segmentation_cache=False,autoseg=False,requested_metrics=[],requested_videos=[])
    loaded=build_workspace(options)
    window.pipeline.load_data(loaded,print,input_source=str(input_path),
        reuse_existing_corr=False,apply_background_correction=False)
    window.workspace=loaded;window.scene.workspace=loaded;window.ortho_viewer.workspace=loaded
    window.scene.reset_display_reference();window.scene.reset_scene();window.scene.set_background('white')
    window.ortho_viewer._initialize_display_settings();window.ortho_viewer.reset_state();window.ortho_viewer.update_slider_ranges()
    window._refresh_all();window._refresh_time_views();events(.3)
    if loaded.scene_objects or loaded.planes or loaded.segmask_raw is not None or loaded.pipeline.completed:
        raise ValueError('The import capture contains analysis results before Run All')
    if window.tree_objects.topLevelItemCount():
        raise ValueError('The post-import Browser must not contain analysis result rows')
    capture('run-ready');run_rect=rect(window.btn_run_all)
    window.btn_run_all.setDown(True);capture('run-click');window.btn_run_all.setDown(False)
    intro_meta=dict(run_rect=run_rect,gui_scale=1.5,intro_size_pixels=[3318,2028],
        intro=dict(state='input_loaded_before_analysis',loader_input=str(input_path),
            source_format=loaded.input_state.source_format,scene_object_count=0,browser_row_count=0,
            segmentation_available=False,completed_steps=[],correction_applied=False,
            note='Native Qt input-only state; the DICOM import gesture is presentation choreography.'))
    dump(root/'intro-manifest.json',intro_meta)
    if intro_only:
        window.close();events(.05);print('Input-only GUI capture complete',flush=True);return
    data=json.loads((root/'assets.json').read_text());result=json.loads((work/'assets/results/assets.json').read_text())
    ws=load_workspace(cfg,prepared=True)
    from display_geometry import display_paths
    ws.centerline_paths_smooth,display_fit=display_paths(ws,cfg)
    for obj in ws.scene_objects.values():obj.actor=obj.label_actor=None;obj.visible=False
    ws.segmentation.visible=False
    window.workspace=ws;window.scene.workspace=ws;window.ortho_viewer.workspace=ws
    window.scene.reset_display_reference();window.scene.reset_scene();window.scene.set_background('white')
    window.ortho_viewer._initialize_display_settings();window.ortho_viewer.reset_state();window.ortho_viewer.update_slider_ranges()
    window._select_workflow_page('correction');window._refresh_all();window._refresh_time_views();events(.3)
    window.resize(2100,1100);events(.2)
    for i,path in enumerate(ws.centerline_paths_smooth or ws.centerline_paths):
        group=ws.path_info[i].get('group_name','') if i<len(ws.path_info) else ''
        ws.add_object(name=f'Centerline {i}',kind=ObjectKind.BRANCH,data_key=f'smooth_path_{i}',
            group_name=group,visible=True,color=ws.skeleton_params.scene_color_for_group(group,'path'),line_width=4.5)
    for i,plane in enumerate(ws.planes):
        key=f'plane_{i}'
        if not any(o.data_key==key for o in ws.scene_objects.values()):
            ws.add_object(name=f'plane {i}',kind=ObjectKind.PLANE,data_key=key,group_name=plane.group_name,
                visible=True,color='#dfa240',opacity=.32)
    for obj in ws.scene_objects.values():
        obj.visible=obj.kind==ObjectKind.PLANE or obj.data_key.startswith(('smooth_path_','segmask_group_'))
        if obj.data_key.startswith('segmask_group_'):obj.opacity=.18;obj.show_scalar_bar=False
        if obj.kind==ObjectKind.PLANE:obj.opacity=.32;obj.color='#dfa240'
    window._select_workflow_page('centerline');window.scene.sync_from_workspace();window._refresh_all();window._refresh_time_views()
    from autoflow.algorithms.surfaces import build_cell_mask_surface
    display_surfaces=[]
    for obj in ws.scene_objects.values():
        if obj.data_key.startswith('segmask_group_') and obj.actor is not None:
            group=ws.multilabel_groups[obj.data_key[len('segmask_group_'):]]
            surface=build_cell_mask_surface(group['geometry_mask_3d'],ws.segmentation_spacing,
                np.asarray(ws.origin)-.5*np.asarray(ws.segmentation_spacing),smooth_iter=1000)
            display_surfaces.append(surface);obj.actor.GetMapper().SetInputData(surface)
    window.workspace_splitter.setSizes([360,1210,530]);window.pwv_dock.show();window._show_measurements('plane_curve')
    pts=np.argwhere(ws.segmask_3d>0)*ws.resolution+ws.origin
    view=cfg.get('workflow',{}).get('camera',{})
    center=np.asarray(view.get('center',((pts.min(0)+pts.max(0))/2).tolist()))
    scale=float(view.get('gui_parallel_scale',float(np.ptp(pts,axis=0).max())*.53))
    def camera(angle):
        a=math.radians(angle);p=window.plotter
        p.camera.position=tuple(center+500*np.array([math.sin(a),-math.cos(a),0.]))
        p.camera.focal_point=tuple(center);p.camera.up=(0,0,1);p.camera.parallel_projection=True
        p.camera.parallel_scale=scale;p.renderer.ResetCameraClippingRange()
    index=int(data['selected_plane_index']);obj=next(o for o in ws.scene_objects.values() if o.data_key==f'plane_{index}')
    uid=obj.uid
    phase0=int(result['base_phase']);ws.current_t=phase0
    with QtCore.QSignalBlocker(window.slider_t):window.slider_t.setValue(phase0)
    camera(0);window._on_3d_plane_picked(uid,index);window._show_measurements('plane_curve')
    window._refresh_selection_info();window._refresh_time_views();window._refresh_pwv_plot();events(.4)
    window.tree_objects.verticalScrollBar().setValue(0);events(.1)
    selected_rect=rect(window.plotter)
    window._on_3d_plane_picked(None,None);window.pwv_dock.show();window._show_measurements('plane_curve')
    window._refresh_selection_info();window._refresh_pwv_plot();events(.2)
    assert np.max(np.abs(np.array(rect(window.plotter))-selected_rect))<2
    capture('planes-unselected')
    marker=window.scene._orientation_marker.widget;marker.SetEnabled(False);capture('unselected-card-raw',window.plotter);marker.SetEnabled(True)
    window._on_3d_plane_picked(uid,index);window._show_measurements('plane_curve')
    window.tree_objects.verticalScrollBar().setValue(0);events(.1)
    folder=gui/'planes-orbit';raw=gui/'planes-card-raw';cards=gui/'planes-card'
    for directory in (folder,raw,cards):directory.mkdir(exist_ok=True)
    total=int(data['frames_per_rotation']);cycles=float(cfg['workflow'].get('cardiac_cycles_per_rotation',3));n=int(data['phase_count'])
    records=[];boxes=[]
    for frame in range(total):
        phase=(phase0+int(frame/total*cycles*n))%n;ws.current_t=phase
        with QtCore.QSignalBlocker(window.slider_t):window.slider_t.setValue(phase)
        camera(frame*360/total);window._refresh_time_views();window._refresh_selection_info();window._refresh_pwv_plot();events(.03)
        window.tree_objects.verticalScrollBar().setValue(0);window.plotter.render()
        if hasattr(window.plotter,'_capture_frame'):window.plotter._capture_frame()
        events(.02);window.grab().save(str(folder/f'{frame:03d}.jpg'),'JPG',96)
        marker.SetEnabled(False);window.plotter.render()
        if hasattr(window.plotter,'_capture_frame'):window.plotter._capture_frame()
        events(.02);window.plotter.grab().save(str(raw/f'{frame:03d}.png'));marker.SetEnabled(True)
        actual=window.label_pwv_status.value.text();expected=float(ws.planes[index].metrics['flowrate_mL_s'][phase])
        if abs(float(actual)-expected)>=.011:raise ValueError((phase,actual,expected))
        records.append(dict(frame=frame,phase=phase,angle_deg=frame*360/total,plane_index=index,
                            flowrate_expected=expected,GUI_flowrate_text=actual))
        pixels=np.asarray(Image.open(raw/f'{frame:03d}.png').convert('RGB')).astype(int)
        foreground=(pixels.max(2)-pixels.min(2)>18)&(pixels.min(2)<230);where=np.argwhere(foreground)
        boxes.append([where[:,1].min(),where[:,0].min(),where[:,1].max()+1,where[:,0].max()+1])
        if frame%30==0:print('GUI orbit',frame,total,flush=True)
    boxes=np.array(boxes);lo=boxes[:,:2].min(0);hi=boxes[:,2:].max(0);pad=np.maximum(12,((hi-lo)*.08).astype(int))
    dimensions=np.array(Image.open(raw/'000.png').size);bounds=tuple(np.r_[np.maximum(0,lo-pad),np.minimum(dimensions,hi+pad)].astype(int))
    for file in raw.glob('*.png'):Image.open(file).crop(bounds).save(cards/(file.stem+'.jpg'),'JPEG',quality=96)
    Image.open(gui/'unselected-card-raw.png').crop(bounds).save(gui/'unselected-card.png')
    shot=np.asarray(Image.open(folder/'000.jpg').convert('RGB'));viewport=np.rint(np.asarray(selected_rect)*1.5).astype(int)
    crop=shot[viewport[1]:viewport[1]+viewport[3],viewport[0]:viewport[0]+viewport[2]]
    pink=(crop[:,:,0]>235)&(crop[:,:,2]>225)&(crop[:,:,1]<80)
    components,_=connected(pink);sizes=np.bincount(components.ravel());sizes[0]=0
    if not np.any(sizes):raise ValueError('Selected-plane outline is not visible in the native GUI capture')
    where=np.argwhere(components==sizes.argmax());point=(where.min(0)+where.max(0))/2
    meta=dict(**intro_meta,plane_cine=records,
        plane=dict(index=index,plotter_rect=selected_rect,click_widget=[float((viewport[0]+point[1])/1.5),float((viewport[1]+point[0])/1.5)],
                   card_crop_bbox=list(bounds),card_orientation_marker=False),
        source=ws.get_active_segmentation_provenance(),display_centerline_fit=display_fit,
        interaction='Authored import/Run All gestures over real Qt states; selected plane and every displayed flow value are native GUI data.')
    dump(root/'gui-manifest.json',meta);window.close();events(.05)
    print('GUI complete',index,len(records),flush=True)


if __name__=='__main__':
    from runtime import configure_runtime, load_config
    configure_runtime()
    try:_capture(load_config(sys.argv[1]),intro_only='--intro-only' in sys.argv[2:])
    except BaseException:
        import traceback;traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
    sys.stdout.flush();os._exit(0)
