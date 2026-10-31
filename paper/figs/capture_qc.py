"""Capture the generated QC report and its actual evidence text."""
from __future__ import annotations
import os
from pathlib import Path
import subprocess
import sys
from runtime import dump, paths


def capture_qc(cfg):
    work,_=paths(cfg);config=work/'capture-config.json';dump(config,cfg)
    subprocess.run([sys.executable,__file__,str(config)],check=True)


def _capture(cfg):
    os.environ.update(QT_QPA_PLATFORM='offscreen',QT_SCALE_FACTOR='1.5',QTWEBENGINE_DISABLE_SANDBOX='1',
        QTWEBENGINE_CHROMIUM_FLAGS='--no-sandbox --disable-gpu --disable-dev-shm-usage',QT_QUICK_BACKEND='software')
    import json,time
    from PIL import Image
    from PySide6 import QtWidgets,QtCore,QtTest
    from PySide6.QtWebEngineWidgets import QWebEngineView
    work,_=paths(cfg);root=work/'assets/workflow';gui=root/'gui';gui.mkdir(parents=True,exist_ok=True)
    app=QtWidgets.QApplication([]);view=QWebEngineView();view.resize(1680,1220);view.show();view.setZoomFactor(1.28)
    def events(seconds=.1):
        until=time.monotonic()+seconds
        while time.monotonic()<until:app.processEvents();time.sleep(.008)
    done=[];view.loadFinished.connect(lambda ok:done.append(ok));view.load(QtCore.QUrl.fromLocalFile(str(work/'quality-report.html')))
    until=time.monotonic()+45
    while not done and time.monotonic()<until:events(.1)
    if not done or not done[-1]:raise RuntimeError('QC report did not load')
    def js(script):
        result=[];view.page().runJavaScript(script,lambda x:result.append(x));until=time.monotonic()+5
        while not result and time.monotonic()<until:events(.01)
        if not result:raise TimeoutError('QC JavaScript callback')
        return result[0]
    def capture(name):
        events(.12);view.grab().save(str(gui/(name+'.png')))
    def bounds(selector):
        return json.loads(js("JSON.stringify((()=>{const e=document.querySelector("+json.dumps(selector)+");if(!e)return null;const r=e.getBoundingClientRect();return [r.x,r.y,r.width,r.height]})())"))
    def heading(text):
        js("(()=>{const e=[...document.querySelectorAll('h2,h3')].find(e=>e.textContent.includes("+json.dumps(text)+"));if(e)window.scrollTo(0,e.getBoundingClientRect().top+scrollY-20)})()")
    events(.5);capture('qc-overview');headline=js("document.querySelector('.headline').innerText")
    # Select a real review item; never impose a warning when this case passes.
    js("(()=>{const issues=[...document.querySelectorAll('.finding.error [data-select-issue],.finding.warning [data-select-issue]')];const b=issues.find(b=>/normal angle|plane geometry/i.test(b.closest('.finding').innerText))||issues[0]||document.querySelector('[data-select-issue]');if(b)b.click()})()")
    events(.25);heading('Segmentation, skeleton');events(.2)
    highlight="""(()=>{const dl=document.querySelector('#marker-inspector .evidence-grid');if(!dl)return;
      for(const dt of dl.querySelectorAll('dt')){const dd=dt.nextElementSibling;if(!dd)continue;
        const key=dt.textContent.trim();if(!['Message','Threshold','Next action'].includes(key))continue;
        if(key==='Next action'){dd.style.background='#ffe36a';dd.style.fontWeight='700';continue;}
        const text=dd.textContent;const re=key==='Message'?/Normal angle[^;]+/i:/normal angle[^;]+/i;
        const match=text.match(re);if(match){dd.replaceChildren(document.createTextNode(text.slice(0,match.index)));
          const span=document.createElement('span');span.textContent=match[0];span.style.background='#ffe36a';
          dd.append(span,document.createTextNode(text.slice(match.index+match[0].length)));}
        else if(key==='Message'){dd.style.background='#ffe36a';}
      }})()"""
    js(highlight);js("document.getElementById('zoom-in')?.click()");events(.2)
    canvas=bounds('#spatial-canvas');zoom=view.zoomFactor()
    start=QtCore.QPoint(round((canvas[0]+canvas[2]*.42)*zoom),round((canvas[1]+canvas[3]*.45)*zoom))
    target=view.focusProxy() or view;QtTest.QTest.mousePress(target,QtCore.Qt.LeftButton,QtCore.Qt.NoModifier,start)
    folder=gui/'qc-rotate';folder.mkdir(exist_ok=True);positions=[]
    for i in range(100):
        delta=i if i<50 else 99-i;point=start+QtCore.QPoint(delta*3,delta);positions.append([point.x(),point.y()])
        QtTest.QTest.mouseMove(target,point,delay=0);events(.03);view.grab().save(str(folder/f'{i:03d}.png'))
    QtTest.QTest.mouseRelease(target,QtCore.Qt.LeftButton,QtCore.Qt.NoModifier,start)
    inspector=bounds('#inspector');ratio=1.5*zoom
    meta={}
    meta['qc']={'rotation_frames':100,'cursor_positions_logical':positions,'headline':headline,
        'inspector_rect_pixels':[v*ratio for v in inspector],
        'selected_text':js("document.querySelector('#inspector').innerText"),
        'selection_policy':'Prefer a real plane-geometry review item, then another error/warning, otherwise the first supplied issue; actual report statuses unchanged.'}
    # Independent native crops keep the evidence readable at all case-specific text lengths.
    view.resize(1680,1800);events(.2);heading('Segmentation, skeleton');events(.2);js(highlight)
    scene=bounds('.hero-grid > .panel');shot=view.grab();shot.save(str(gui/'qc-scene-source.png'))
    toolbar=bounds('.hero-grid > .panel .scene-toolbar')
    if toolbar:
        bottom=scene[1]+scene[3];scene[1]=toolbar[1];scene[3]=bottom-scene[1]
    scene_pixels=[v*ratio for v in scene]
    Image.open(gui/'qc-scene-source.png').crop((int(scene_pixels[0]),int(scene_pixels[1]),
        int(scene_pixels[0]+scene_pixels[2]),int(scene_pixels[1]+scene_pixels[3]))).save(gui/'qc-scene.png')
    js("(()=>{const e=document.getElementById('inspector');window.scrollTo(0,e.getBoundingClientRect().top+scrollY-20)})()")
    events(.2);box=bounds('#inspector');view.grab().save(str(gui/'qc-detail-source.png'));px=[v*ratio for v in box]
    detail=Image.open(gui/'qc-detail-source.png').crop((int(px[0]),int(px[1]),int(px[0]+px[2]),int(px[1]+px[3])))
    review_end=js("(()=>{const e=document.querySelector('#inspector .review-record');if(!e)return null;return e.getBoundingClientRect().bottom-document.getElementById('inspector').getBoundingClientRect().top+12})()")
    if review_end:detail=detail.crop((0,0,detail.width,min(detail.height,round(review_end*ratio))))
    detail.save(gui/'selected-check.png')
    cutoff=js("(()=>{const dt=[...document.querySelectorAll('#marker-inspector dt')].find(e=>e.textContent.trim()==='Next action');if(!dt)return null;const r=dt.nextElementSibling.getBoundingClientRect(),b=document.getElementById('inspector').getBoundingClientRect();return r.bottom-b.top+20})()")
    if cutoff:detail.crop((0,0,detail.width,min(detail.height,round(cutoff*ratio)))).save(gui/'selected-check-short.png')
    else:detail.save(gui/'selected-check-short.png')
    dump(root/'qc-manifest.json',meta);view.close();events(.1)
    print('QC capture complete',headline,flush=True)


if __name__=='__main__':
    from runtime import configure_runtime,load_config
    configure_runtime()
    try:_capture(load_config(sys.argv[1]))
    except BaseException:
        import traceback;traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
    sys.stdout.flush();os._exit(0)
