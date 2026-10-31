"""Correction-panel images with an explicit measured/simulated provenance."""
from __future__ import annotations
import numpy as np
from PIL import Image
from scipy.ndimage import label, binary_closing, binary_fill_holes
from runtime import dump


def prepare_correction(ws, cfg, output):
    spec=cfg.get('correction',{});component=int(spec.get('component',0))
    source=getattr(ws,'phase_wrapped',None)
    if source is None:
        raise ValueError('The correction panel needs stored FC-referenced phase; provide it or use pre-rendered correction assets')
    phase=spec.get('phase')
    if phase is None:phase=int(ws.current_t)
    phase=min(int(phase),source.shape[3]-1)
    z=spec.get('slice')
    if z is None:
        support=ws.segmask_binary[...,phase]
        z=int(np.argmax(support.sum(axis=(0,1))))
    z=int(z)
    if not 0<=z<source.shape[2]:raise ValueError(f'Correction slice {z} is outside {source.shape[2]} slices')
    clean=np.asarray(source[:,:,z,phase,component],float)
    mag=ws.mag_raw[:,:,z,phase];support=(mag>.08)|ws.segmask_binary[:,:,z,phase]
    regions,n=label(support);sizes=np.bincount(regions.ravel());sizes[0]=0
    body=binary_fill_holes(binary_closing(regions==sizes.argmax(),iterations=2)) if n else support
    mode=spec.get('mode','measured');gain=0.;reference_venc=float(np.asarray(ws.venc).reshape(-1)[component])
    if mode=='fc_venc_simulation':
        target=float(spec.get('simulation_venc_cm_s',30))
        if target<=0:raise ValueError('simulation_venc_cm_s must be positive')
        corr=np.zeros_like(clean) if ws.correction_raw is None else np.asarray(ws.correction_raw[:,:,z,phase,component])
        if spec.get('amplified_background_phase',False):
            gain=.65*np.pi/max(float(np.percentile(np.abs(corr[body]),98)),1e-6)
        before=np.angle(np.exp(1j*(reference_venc/target*clean+gain*corr)))
        explanation='FC is fixed. Rescale only angle(encoded * conj(FC)); add the configured recorded-correction field and wrap to [-pi, pi]. After shows the original corrected phase at the original VENC.'
    elif mode=='measured':
        target=reference_venc
        corr=np.zeros_like(clean) if ws.correction_raw is None else np.asarray(ws.correction_raw[:,:,z,phase,component])
        before=np.angle(np.exp(1j*(clean+corr)))
        explanation='Recorded uncorrected relative phase reconstructed by adding the recorded correction field to the corrected FC-referenced phase.'
    else:raise ValueError(f'Unknown correction mode: {mode}')
    for name,values in [('before',before),('after',clean)]:
        gray=np.clip((values+np.pi)/(2*np.pi),0,1)
        if name=='after':gray=np.where(support,gray,.5);gray[~body]=0
        im=Image.fromarray(np.uint8(np.rint(gray.T*255))).convert('RGB')
        factor=1080/max(im.size);im=im.resize((round(im.width*factor),round(im.height*factor)),Image.Resampling.LANCZOS)
        canvas=Image.new('RGB',(1080,1080));canvas.paste(im,((1080-im.width)//2,(1080-im.height)//2))
        canvas.save(output/f'phase-{name}.png')
    dump(output/'phase-provenance.json',dict(mode=mode,description=explanation,slice=z,phase=phase,component=component,
        reference_venc_cm_s=reference_venc,simulated_venc_cm_s=target,clim_rad=[-float(np.pi),float(np.pi)],
        background_gain=gain,full_native_slice=True,phase_smoothing=False))
