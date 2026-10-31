"""One compositor for the sequential and simultaneous workflow presentations."""
from __future__ import annotations

from functools import lru_cache
import json
import math
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from runtime import ROOT, VideoWriter, dump, ffmpeg, font_path, paths, sha256_file, resolve_path, logical_identity

W,H=3840,2160
BG='#f5f5f7';INK='#1d1d1f';BLUE='#007aff';GRAY='#6e6e73'
TITLES=['Correction','Segmentation','Topology','Plane Metrics','Hemodynamics','QC Report']
REGIONS=['cardiopulmonary','cerebrovascular','portal-venous','carotid','renal-artery']
REGION_LABELS=['Cardiopulmonary','Cerebrovascular','Portal Venous','Carotid','Renal Artery']
MODE_LABELS={'streamlines':'Streamlines','wss':'Wall shear stress','relative-pressure':'Relative pressure',
             'pressure':'Relative pressure','tke':'Turbulent kinetic energy'}


def ease(x):
    x=max(0,min(1,float(x)));return x*x*(3-2*x)


def paste(dst,im,xy,alpha=1):
    if alpha<=0:return
    if im.mode!='RGBA':im=im.convert('RGBA')
    if alpha<.999:
        im=im.copy();im.putalpha(im.getchannel('A').point(lambda p:round(p*alpha)))
    dst.alpha_composite(im,(round(xy[0]),round(xy[1])))


def rounded(im,radius=32):
    im=im.copy().convert('RGBA');mask=Image.new('L',im.size)
    ImageDraw.Draw(mask).rounded_rectangle((0,0,im.width-1,im.height-1),radius,fill=255)
    im.putalpha(mask);return im


def panel(size):return rounded(Image.new('RGBA',size,'white'))


def cursor(dst,point,click=0):
    x,y=point;draw=ImageDraw.Draw(dst)
    if 0<click<1:
        r=24+64*click;draw.ellipse((x-r,y-r,x+r,y+r),outline=(0,122,255,round(220*(1-click))),width=7)
    points=[(0,0),(0,58),(16,43),(34,72),(45,66),(27,38),(54,38)]
    draw.polygon([(x+a+4,y+b+4) for a,b in points],fill=(0,0,0,60))
    draw.polygon([(x+a,y+b) for a,b in points],fill='white',outline='#26394a',width=3)


def magnifier(dst,point,click=0):
    x,y=map(float,point);layer=Image.new('RGBA',dst.size);draw=ImageDraw.Draw(layer)
    if 0<click<1:
        r=40+60*click;draw.ellipse((x-r,y-r,x+r,y+r),outline=(0,122,255,round(230*(1-click))),width=7)
    draw.line((x+23,y+23,x+62,y+62),fill=(255,255,255,240),width=20)
    draw.line((x+23,y+23,x+62,y+62),fill='#1d3c56',width=12)
    draw.ellipse((x-33,y-33,x+33,y+33),fill=(238,248,255,245),outline='white',width=10)
    draw.ellipse((x-29,y-29,x+29,y+29),outline=BLUE,width=6)
    draw.line((x-13,y,x+13,y),fill=BLUE,width=5);draw.line((x,y-13,x,y+13),fill=BLUE,width=5)
    paste(dst,layer,(0,0))


class Workflow:
    def __init__(self,cfg):
        self.cfg=cfg;self.root=Path(cfg['work_dir'])/'assets/workflow';self.gui=self.root/'gui'
        self.data=json.loads((self.root/'assets.json').read_text())
        if self.data.get('analysis_identity') and self.data['analysis_identity']!=logical_identity(cfg):
            raise ValueError('Workflow assets belong to a different source/cleanup configuration; prepare assets again')
        if self.data.get('cleanup') and self.data.get('cleanup')!=cfg.get('cleanup'):
            raise ValueError('Replacement segmentation cleanup differs from the configured policy')
        self.meta=json.loads((self.root/'gui-manifest.json').read_text())
        intro_path=self.root/'intro-manifest.json'
        if not intro_path.exists():
            raise ValueError('Run capture-intro to capture a verified input-only opening')
        self.meta.update(json.loads(intro_path.read_text()))
        intro_state=self.meta['intro']
        if (intro_state['scene_object_count'] or intro_state['browser_row_count'] or
                intro_state['segmentation_available'] or intro_state['completed_steps']):
            raise ValueError('The opening contains premature analysis results')
        qc=self.root/'qc-manifest.json'
        if qc.exists():self.meta.update(json.loads(qc.read_text()))
        self.metrics=json.loads((self.root/'plane-metrics.json').read_text())
        self.records=self.meta.get('plane_cine',[]);self.frames=int(self.data['frames_per_rotation'])
        if len(self.records)!=self.frames:raise ValueError('Native plane frames are incomplete')
        for row in self.records:
            expected=self.metrics['flowrate'][row['phase']]
            if row['plane_index']!=self.metrics['plane_index'] or abs(float(row['GUI_flowrate_text'])-expected)>.011:
                raise ValueError('Plane identity/phase/readout mismatch')
        regular=cfg.get('workflow',{}).get('font',cfg.get('render',{}).get('font'))
        bold=cfg.get('workflow',{}).get('bold_font',cfg.get('render',{}).get('bold_font'))
        for font in (regular,bold):
            if font and not resolve_path(font,cfg).is_file():
                raise FileNotFoundError(f'Configured presentation font is missing: {resolve_path(font,cfg)}')
        self.font=font_path(configured=str(resolve_path(regular,cfg)) if regular else None)
        self.bold=font_path(True,str(resolve_path(bold,cfg)) if bold else None)
        self.rects=[np.array([60+(i%3)*1260,65+(i//3)*1040,1200,990],float) for i in range(6)]
        options=cfg['workflow'];self.intro=float(options.get('import_seconds',3.2));interval=float(options.get('step_interval_seconds',1.2))
        self.starts=[self.intro+i*interval for i in range(4)]
        self.plane_start=self.starts[3]+1.4;self.plane_full=self.plane_start+.7
        self.mouse_arrives=self.plane_full+.6;self.select=self.mouse_arrives+.1;self.orbit=self.select+.4
        self.plane_end=self.plane_start+float(options.get('plane_expand_seconds',4.1))
        self.starts.extend([self.plane_end,self.plane_end+interval])
        self.qc_start=self.starts[5]+1.8;self.hold=float(options.get('qc_detail_hold_seconds',1))
        self.qc_end=self.qc_start+3.4+self.hold+.7
        self.final_start={'sequence':self.qc_end,'simultaneous':self.intro+3.9+self.hold}
        self.rotation=float(options.get('rotation_seconds',6));self.closing=float(options.get('closing_rotations',2))*self.rotation
        self.gui_box=np.array([460,65,3320,2030],float)
        self.icons={}
        for name in REGIONS:
            icon=Image.open(ROOT/'assets/regions'/(name+'.png')).convert('RGBA')
            icon=icon.crop(icon.getchannel('A').getbbox());scale=165/max(icon.size)
            self.icons[name]=icon.resize((round(icon.width*scale),round(icon.height*scale)),Image.Resampling.LANCZOS)
        self.phase_before=rounded(self.load(str(self.root/'phase-before.png')),25)
        self.phase_after=rounded(self.load(str(self.root/'phase-after.png')),25)

    @lru_cache(maxsize=80)
    def ft(self,size,bold=False):return ImageFont.truetype(self.bold if bold else self.font,int(size))

    @lru_cache(maxsize=30)
    def load(self,path):return Image.open(path).convert('RGBA')

    @lru_cache(maxsize=80)
    def fit(self,path,width,height):
        im=self.load(path);scale=min(width/im.width,height/im.height)
        return im.resize((max(1,round(im.width*scale)),max(1,round(im.height*scale))),Image.Resampling.LANCZOS)

    def inside(self,dst,src,box):
        x,y,width,height=box
        if isinstance(src,(str,Path)):im=self.fit(str(src),round(width),round(height))
        else:
            scale=min(width/src.width,height/src.height)
            im=src.resize((max(1,round(src.width*scale)),max(1,round(src.height*scale))),Image.Resampling.LANCZOS)
        paste(dst,im,(x+(width-im.width)/2,y+(height-im.height)/2))

    def gui_point(self,raw,box,source):
        size=np.array(self.load(str(source)).size);scale=min(box[2:]/size)
        return box[:2]+(box[2:]-size*scale)/2+np.array(raw)*float(self.meta.get('gui_scale',1.5))*scale

    def animation_frame(self,t,index,mode):
        if mode=='closing':return round(t/self.rotation*self.frames)%self.frames
        if mode=='simultaneous':return round((t-self.final_start[mode])/self.rotation*self.frames)%self.frames
        if t>=self.qc_end-.7:return round(max(0,t-self.qc_end)/self.rotation*self.frames)%self.frames
        if index==3:return round(max(0,t-self.orbit)/self.rotation*self.frames)%self.frames
        return round(max(0,t-self.starts[index])/self.rotation*self.frames)%self.frames

    def flow_chart(self,im,t,mode):
        draw=ImageDraw.Draw(im);phase=self.records[self.animation_frame(t,3,mode)]['phase']
        flow=np.array(self.metrics['flowrate']);area=np.array(self.metrics['area']);peak=max(self.metrics['peak_velocity'])
        draw.text((60,626),'Flow rate',font=self.ft(38,True),fill=INK)
        draw.text((1140,622),f'{flow[phase]:.2f} mL/s',font=self.ft(48,True),fill=BLUE,anchor='ra')
        draw.text((60,684),f"Plane {self.metrics['plane_index']}   ·   Area {area[phase]:.1f} mm²   ·   Peak velocity {peak:.1f} cm/s",font=self.ft(31),fill=GRAY)
        x0,y0,width,height=100,770,1000,140;lo=float(min(0,flow.min()));hi=float(max(lo+1e-6,flow.max()*1.12))
        xs=np.linspace(x0,x0+width,len(flow));ys=y0+height-(flow-lo)/(hi-lo)*height
        for value in (lo,(lo+hi)/2,hi):
            y=y0+height-(value-lo)/(hi-lo)*height;draw.line((x0,y,x0+width,y),fill='#e8e9ed',width=2)
            draw.text((x0-15,y),f'{value:.0f}',font=self.ft(25),fill=GRAY,anchor='rm')
        draw.line(list(zip(xs,ys)),fill=BLUE,width=5);x,y=xs[phase],ys[phase]
        draw.line((x,y0,x,y0+height),fill='#97caff',width=2);draw.ellipse((x-9,y-9,x+9,y+9),fill=BLUE,outline='white',width=2)
        draw.text((x0,y0+height+30),'0',font=self.ft(25),fill=GRAY,anchor='ma')
        draw.text((x0+width,y0+height+30),str(len(flow)-1),font=self.ft(25),fill=GRAY,anchor='ma')
        draw.text((600,970),'Cardiac phase',font=self.ft(27),fill=GRAY,anchor='md')
        draw.text((1140,735),f'Phase {phase}',font=self.ft(28),fill=GRAY,anchor='rd')

    def qc_thumb(self,t,mode):
        if mode=='closing':return self.load(str(self.gui/'selected-check-short.png'))
        start=self.intro if mode=='simultaneous' else self.starts[5];age=max(0,t-start)
        if age<1:return self.load(str(self.gui/'qc-overview.png'))
        if age<3.2:return self.load(str(self.gui/'qc-rotate'/f'{min(99,int((age-1)/2.2*100)):03d}.png'))
        if mode=='sequence' or age>=3.9:return self.load(str(self.gui/'selected-check-short.png'))
        source=self.load(str(self.gui/'qc-rotate/099.png'));q=ease((age-3.5)/.4)
        x,y,width,height=self.meta['qc']['inspector_rect_pixels']
        box=np.array([0,0,source.width,source.height])*(1-q)+np.array([x,y,x+width,min(source.height,y+height)])*q
        return source.crop(tuple(np.rint(box).astype(int)))

    def card(self,index,t,mode):
        im=panel((1200,990));draw=ImageDraw.Draw(im)
        draw.text((38,26),str(index+1),font=self.ft(52,True),fill=BLUE)
        draw.text((105,22),TITLES[index],font=self.ft(58,True),fill=INK)
        if index==0:
            progress=1 if mode=='closing' else ease((t-self.intro)/(3.9 if mode=='simultaneous' else 6))
            phase=self.phase_before.copy();edge=round(progress*phase.width)
            if edge>0:paste(phase,self.phase_after.crop((0,0,edge,phase.height)),(0,0))
            if 0<edge<phase.width:ImageDraw.Draw(phase).line((edge,15,edge,phase.height-15),fill=(255,255,255,220),width=2)
            self.inside(im,phase,(130,125,940,820))
        elif index in (1,2):
            folder='segmentation' if index==1 else 'topology';frame=self.animation_frame(t,index,mode)
            self.inside(im,self.root/folder/f'{frame:03d}.png',(25,125,1150,820) if index==1 else (20,128,1160,812))
        elif index==3:
            selected=mode!='sequence' or t>=self.select;frame=self.animation_frame(t,3,mode)
            source=self.gui/'planes-card'/f'{frame:03d}.jpg' if selected else self.gui/'unselected-card.png'
            self.inside(im,source,(20,127,1160,480) if selected else (20,128,1160,812))
            if selected:self.flow_chart(im,t,mode)
        elif index==4:
            frame=self.animation_frame(t,4,mode);record=self.data['continuous_hemodynamics'][frame]
            mode_name=record['mode'];metric=self.data['metric_ranges'][mode_name]
            draw.text((600,142),MODE_LABELS.get(mode_name,mode_name),font=self.ft(42),fill=INK,anchor='mm')
            self.inside(im,self.root/'hemodynamics'/f'{frame:03d}.png',(25,185,1150,615))
            colors=np.array(metric['colors']);ramp=np.stack([np.interp(np.linspace(0,len(colors)-1,850),np.arange(len(colors)),colors[:,k]) for k in range(3)],axis=1)
            paste(im,Image.fromarray(np.tile((ramp*255).astype('uint8')[None],(20,1,1))),(175,865))
            draw.text((175,807),metric['label'],font=self.ft(40),fill=INK)
            draw.text((175,901),f"{metric['clim'][0]:.0f}",font=self.ft(34),fill=GRAY)
            draw.text((1025,901),f"{metric['clim'][1]:.0f}",font=self.ft(34),fill=GRAY,anchor='ra')
            draw.text((1170,975),f"Cardiac phase {record['phase']+1}/{self.data['phase_count']}",font=self.ft(31),fill=GRAY,anchor='rd')
        else:self.inside(im,self.qc_thumb(t,mode),(22,128,1156,842))
        return im

    @lru_cache(maxsize=8)
    def plane_plate(self,frame):
        source=self.gui/'planes-unselected.png' if frame<0 else self.gui/'planes-orbit'/f'{frame:03d}.jpg'
        im=panel((3680,1940));draw=ImageDraw.Draw(im)
        draw.text((42,22),'4',font=self.ft(60,True),fill=BLUE);draw.text((127,22),'Plane Metrics',font=self.ft(60,True),fill=INK)
        self.inside(im,source,(25,135,3630,1780));return im

    def plane_content(self,t):
        selected=t>=self.select;frame=self.animation_frame(t,3,'sequence') if selected else -1
        im=self.plane_plate(frame)
        if self.plane_full<=t<self.orbit:
            im=im.copy();source=self.gui/'planes-unselected.png' if not selected else self.gui/'planes-orbit'/f'{frame:03d}.jpg'
            target=self.gui_point(self.meta['plane']['click_widget'],np.array([25,135,3630,1780]),source)
            point=target+np.array([300.,180.])*(1-ease((t-self.plane_full)/.6))
            cursor(im,point,max(0,min(.999,(t-self.select)/.4)) if selected else 0)
        return im

    @lru_cache(maxsize=1)
    def qc_detail(self):
        im=panel((3680,1940));draw=ImageDraw.Draw(im)
        draw.text((42,22),'6',font=self.ft(60,True),fill=BLUE);draw.text((127,22),'QC Report',font=self.ft(60,True),fill=INK)
        self.inside(im,self.gui/'qc-scene.png',(65,185,1685,1670))
        self.inside(im,self.gui/'selected-check.png',(1840,135,1720,1760));return im

    def qc_content(self,t):
        age=t-self.qc_start
        if age>=3.4:return self.qc_detail()
        index=None
        if age<1:source=self.gui/'qc-overview.png'
        elif age<2.4:index=min(99,int((age-1)/1.4*100));source=self.gui/'qc-rotate'/f'{index:03d}.png'
        else:source=self.gui/'qc-rotate/099.png'
        im=panel((3680,1940));draw=ImageDraw.Draw(im)
        draw.text((42,22),'6',font=self.ft(60,True),fill=BLUE);draw.text((127,22),'QC Report',font=self.ft(60,True),fill=INK)
        body=np.array([24,120,3632,1790]);self.inside(im,source,body)
        if index is not None:cursor(im,self.gui_point(self.meta['qc']['cursor_positions_logical'][index],body,source))
        if age>=2.95:im=Image.blend(im,self.qc_detail(),ease((age-2.95)/.45))
        if 2.4<age<3.15:
            target=np.array([2700.,1140.]);point=target+np.array([160,110])*(1-ease((age-2.4)/.3))
            magnifier(im,point,max(0,(age-2.75)/.4))
        return im

    def expand(self,dst,index,t,start,end,content):
        z=ease((t-start)/.7)*ease((end-t)/.7)
        if z<=0:return
        box=self.rects[index]*(1-z)+np.array([80,100,3680,1940])*z
        paste(dst,Image.new('RGBA',(W,H),(245,245,247,round(255*z))),(0,0));self.inside(dst,content,box)

    def approach(self,dst,t,index,click):
        if not click-.65<t<click+.35:return
        target=self.rects[index][:2]+[920,560]
        magnifier(dst,target+np.array([160,110])*(1-ease((t-click+.65)/.6)),max(0,(t-click)/.35))

    def opening(self,t):
        im=Image.new('RGBA',(W,H),BG);draw=ImageDraw.Draw(im)
        for i,name in enumerate(REGIONS):
            y=350+i*350;icon=self.icons[name];paste(im,icon,(220-icon.width/2,y-icon.height/2))
            draw.text((220,y+112),REGION_LABELS[i],font=self.ft(46),fill=BLUE if name==self.cfg.get('region','cerebrovascular') else GRAY,anchor='mm')
        click_at=self.intro-.5
        source=self.gui/'empty-correction.png' if t<1.7 else self.gui/('run-ready.png' if t<click_at else 'run-click.png')
        self.inside(im,source,self.gui_box)
        if .2<t<.8:
            row=REGIONS.index(self.cfg.get('region','cerebrovascular'))
            u=ease((t-.2)/.6);x,y=np.array([240.,340.+row*350])*(1-u)+np.array([1750.,850.])*u
            draw.rounded_rectangle((x,y,x+110,y+43),radius=12,fill='#90c9ff')
            draw.rounded_rectangle((x,y+22,x+210,y+162),radius=16,fill='#6ab6f9')
            draw.rounded_rectangle((x+16,y+52,x+194,y+146),radius=11,fill='#98d0ff')
            draw.text((x+105,y+92),'DICOM',font=self.ft(34,True),fill='#164c78',anchor='mm');cursor(im,[x+190,y+142])
        if .8<=t<1.7:
            x,y=1610,870;draw.rounded_rectangle((x,y,x+850,y+245),radius=30,fill='white',outline='#dce0e5',width=2)
            draw.text((x+55,y+48),'Loading DICOM',font=self.ft(44,True),fill=INK)
            draw.rounded_rectangle((x+55,y+145,x+795,y+161),radius=8,fill='#e8ebee')
            draw.rounded_rectangle((x+55,y+145,x+55+max(16,740*ease((t-.8)/.9)),y+161),radius=8,fill=BLUE)
        if click_at-.6<t<click_at+.25:
            rr=np.array(self.meta['run_rect']);target=self.gui_point(rr[:2]+rr[2:]/2,self.gui_box,self.gui/'run-ready.png')
            u=ease((t-click_at+.6)/.55);cursor(im,np.array([1960,1190])*(1-u)+target*u,max(0,(t-click_at)/.25))
        fade=ease((self.intro-t)/.3)
        if fade<1:im=Image.blend(Image.new('RGBA',(W,H),BG),im,fade)
        return im.convert('RGB')

    @lru_cache(maxsize=180)
    def closing_frame(self,frame):
        im=Image.new('RGBA',(W,H),BG);t=frame/30
        for i,box in enumerate(self.rects):paste(im,self.card(i,t,'closing'),box[:2])
        return im.convert('RGB')

    def render(self,t,mode):
        if t<self.intro:return self.opening(t)
        if t>=self.final_start[mode]:return self.closing_frame(round((t-self.final_start[mode])*30)%self.frames)
        if mode=='sequence' and self.plane_full<=t<self.plane_end-.7:
            im=Image.new('RGBA',(W,H),BG);self.inside(im,self.plane_content(t),(80,100,3680,1940));return im.convert('RGB')
        if mode=='sequence' and self.qc_start+3.4<=t<self.qc_end-.7:
            im=Image.new('RGBA',(W,H),BG);self.inside(im,self.qc_detail(),(80,100,3680,1940));return im.convert('RGB')
        im=Image.new('RGBA',(W,H),BG)
        for i,box in enumerate(self.rects):
            start=self.intro if mode=='simultaneous' else self.starts[i]
            if t<start:continue
            alpha=ease((t-start)/.45);paste(im,self.card(i,t,mode),box[:2]+[0,24*(1-alpha)],alpha)
        if mode=='sequence':
            if self.plane_start<t<self.plane_end:self.expand(im,3,t,self.plane_start,self.plane_end,self.plane_content(t))
            if self.qc_start<t<self.qc_end:self.expand(im,5,t,self.qc_start,self.qc_end,self.qc_content(t))
            self.approach(im,t,3,self.plane_start-.25);self.approach(im,t,5,self.qc_start-.25)
        else:
            age=t-self.intro
            if 2.85<age<3.85:
                target=np.array([3500.,1640.]);point=target+np.array([140,100])*(1-ease((age-2.85)/.6))
                magnifier(im,point,max(0,(age-3.5)/.35))
        return im.convert('RGB')


def render_workflows(cfg,*,mode='both',stills=False,resolution='4k'):
    _,out=paths(cfg);out=out/'workflow';out.mkdir(parents=True,exist_ok=True)
    compositor=Workflow(cfg);modes=['sequence','simultaneous'] if mode=='both' else [mode]
    from narration import write_narration
    write_narration(compositor,out)
    for variant in modes:
        tag='Workflow' if variant=='sequence' else 'Six_Panels'
        times=[.5,2.2,compositor.final_start[variant]]
        if variant=='sequence':times += [compositor.plane_full,compositor.select+.1,compositor.plane_end+.5,compositor.qc_start+3.6]
        else:times += [compositor.intro+.6,compositor.intro+3.6]
        for t in times:compositor.render(t,variant).save(out/f'{tag}-{t:05.2f}.jpg',quality=95)
        duration=compositor.final_start[variant]+compositor.closing
        timeline={'mode':variant,'step_starts_s':compositor.starts if variant=='sequence' else [compositor.intro]*6,
            'plane_expansion_s':[compositor.plane_start,compositor.plane_end] if variant=='sequence' else None,
            'qc_detail_hold_s':compositor.hold,'closing_start_s':compositor.final_start[variant],
            'closing_rotations':cfg['workflow'].get('closing_rotations',2),'duration_s':duration,
            'source':compositor.data.get('source'),'metric_mask_sha256':compositor.data.get('native_metric_mask_sha256'),
            'topology_mask_sha256':compositor.data.get('mask_sha256'),
            'plane_index':compositor.metrics['plane_index'],'native_readouts_checked':len(compositor.records),
            'intro':compositor.meta['intro'],'run_click_s':compositor.intro-.5}
        if stills:dump(out/(tag+'-timeline.json'),timeline);continue
        size=(3840,2160) if resolution=='4k' else (1920,1080)
        file=out/f'AutoFlow_{tag}_{"4K" if resolution=="4k" else "1080p"}.mp4'
        writer=VideoWriter(file,size,30,cfg.get('render',{}).get('encoder','libx264'))
        total=round(duration*30)
        for frame in range(total):
            image=compositor.render(frame/30,variant)
            if size!=(3840,2160):image=image.resize(size,Image.Resampling.LANCZOS)
            writer.append(image)
            if frame%90==0:print(tag,frame,total,flush=True)
        records=[writer.close()]
        if resolution=='4k':
            small=out/f'AutoFlow_{tag}_1080p.mp4'
            subprocess.run([ffmpeg(),'-v','error','-y','-i',str(file),'-vf','scale=1920:1080:flags=lanczos',
                '-c:v','libx264','-preset','fast','-crf','18','-threads','4','-pix_fmt','yuv420p','-movflags','+faststart',str(small)],check=True)
            subprocess.run([ffmpeg(),'-v','error','-i',str(small),'-f','null','-'],check=True)
            records.append(dict(file=str(small),size=[1920,1080],fps=30,frames=total,duration_s=duration,decode='passed',sha256=sha256_file(small)))
        dump(out/(tag+'-verification.json'),{'videos':records,'timeline':timeline})
