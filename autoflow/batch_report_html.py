'''Portable, offline batch overview built from the canonical case results.'''
from collections import Counter
from datetime import datetime, timezone
from html import escape
import math
import os
from pathlib import Path
import tempfile
from urllib.parse import quote

_STYLE = '''*{box-sizing:border-box}body{font:15px/1.5 system-ui,sans-serif;color:#243447;background:#f3f6f8;margin:0}main{max-width:1440px;margin:auto;padding:28px}h1{margin:0 0 8px;font-size:28px}p{margin:8px 0 18px}.muted{color:#5b6d7d}.cards{display:flex;gap:12px;flex-wrap:wrap;margin:22px 0}.card{background:white;padding:16px 24px;border-radius:8px;border:1px solid #d8e2e9}.card b{display:block;font-size:26px}.controls{display:flex;gap:18px;align-items:end;flex-wrap:wrap;margin:20px 0}label{display:flex;flex-direction:column;gap:5px}input,select{font:inherit;padding:8px;border:1px solid #98aabb;border-radius:5px}input{min-width:280px}button{font:inherit;border:0;background:none;color:inherit;cursor:pointer;text-align:left;padding:4px}button:focus-visible,a:focus-visible{outline:3px solid #147d96;outline-offset:3px}.table{overflow:auto;border-radius:8px;border:1px solid #d8e2e9}table{border-collapse:collapse;width:100%;background:white}th,td{padding:12px;border-bottom:1px solid #e2e9ee;text-align:left;vertical-align:top}th{background:#e9f0f5;white-space:nowrap}td.case{min-width:190px;overflow-wrap:anywhere}td.error{min-width:160px;max-width:420px;overflow-wrap:anywhere}.badge{display:inline-block;padding:3px 8px;border-radius:4px;background:#e9eef3;white-space:nowrap}.ok{color:#125d40;background:#def4e8}.partial{color:#855000;background:#fff0ce}.error,.failed,.timeout{color:#9b2432}.badge.error,.badge.failed,.badge.timeout{background:#ffe5e8}a{color:#076d87;text-underline-offset:3px}.links a{display:block;margin-bottom:5px}.small{font-size:13px}tr[hidden]{display:none}.notice{padding:12px 16px;border-left:4px solid #ad7800;background:#fff2d5}footer{margin:24px 0;color:#5b6d7d;font-size:13px}@media(max-width:700px){main{padding:16px}.controls,input{width:100%}}'''
_SCRIPT = '''const body=document.querySelector('tbody');const rows=[...body.rows];const search=document.getElementById('search');const status=document.getElementById('status');function filter(){const query=search.value.trim().toLocaleLowerCase();let count=0;for(const row of rows){const match=(!status.value||row.dataset.status===status.value)&&row.textContent.toLocaleLowerCase().includes(query);row.hidden=!match;if(match)count++;}document.getElementById('visible').textContent=count+' / '+rows.length+' cases shown';}search.addEventListener('input',filter);status.addEventListener('change',filter);for(const button of document.querySelectorAll('th button')){button.addEventListener('click',()=>{const key=button.dataset.key;const ascending=button.dataset.direction!=='ascending';for(const peer of document.querySelectorAll('th button')){delete peer.dataset.direction;peer.parentElement.removeAttribute('aria-sort');}button.dataset.direction=ascending?'ascending':'descending';button.parentElement.setAttribute('aria-sort',button.dataset.direction);rows.sort((a,b)=>{let comparison;if(key==='seconds'){const x=Number(a.dataset.seconds),y=Number(b.dataset.seconds);if(!Number.isFinite(x))return Number.isFinite(y)?1:0;if(!Number.isFinite(y))return -1;comparison=x-y;}else comparison=(a.dataset[key]||'').localeCompare(b.dataset[key]||'',undefined,{numeric:true,sensitivity:'base'});return ascending?comparison:-comparison;});for(const row of rows)body.appendChild(row);});}filter();'''


def _mapping(value):
    return value if isinstance(value, dict) else {}


def _local_link(value, root, label):
    if not value:
        return ''
    path = Path(str(value))
    # Canonical case summaries can contain either report-root-relative paths
    # or the caller's relative output directory (for example results/case).
    # Resolve both interpretations, but only link existing files inside root.
    candidates = [path] if path.is_absolute() else [root / path, Path.cwd() / path]
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
            relative = resolved.relative_to(root)
            if not resolved.is_file():
                continue
        except (ValueError, OSError, RuntimeError):
            # A malformed path or symlink loop must not suppress other cases.
            continue
        url = quote(relative.as_posix(), safe='/')
        return '<a href="' + escape(url, quote=True) + '">' + escape(label) + '</a>'
    return ''


def render_batch_report_html(results, output_dir):
    '''Render every supplied case, keeping execution and QC status separate.

    Only existing files below the report directory are linked. Relative links,
    inline CSS/JavaScript, and complete static rows work after moving the folder;
    JavaScript is optional for reading, and used only for sorting/filtering.
    '''
    root = Path(output_dir).resolve()
    records = list(results)
    counts = Counter(str(_mapping(row).get('status', 'unknown')) for row in records)
    rows = []
    for row in records:
        row = _mapping(row)
        summary = _mapping(row.get('summary'))
        qc = _mapping(summary.get('quality_report'))
        status = str(row.get('status', 'unknown'))
        name = str(row.get('case') or summary.get('input_display_name') or Path(str(row.get('file', 'Unknown case'))).name)
        output_status = str(summary.get('output_status') or 'not evaluated')
        errors = [str(row['error'])] if row.get('error') else []
        if summary.get('analysis_report_error'):
            errors.append(str(summary['analysis_report_error']))
        for key, value in _mapping(summary.get('video_errors')).items():
            errors.append(str(key) + ': ' + str(value))
        qc_status = str(qc.get('overall_status') or 'not evaluated')
        qc_counts = _mapping(qc.get('status_counts'))
        qc_text = ', '.join(str(qc_counts[key]) + ' ' + label for key, label in
                            [('fail', 'error'), ('warn', 'warning'), ('not_run', 'not evaluated'), ('pass', 'pass')]
                            if key in qc_counts)
        try:
            seconds = float(summary.get('total_time_sec'))
            if not math.isfinite(seconds) or seconds < 0:
                raise ValueError('Invalid duration')
        except (TypeError, ValueError, OverflowError):
            seconds = None
        links = _local_link(summary.get('analysis_report_file'), root, 'Case report')
        report_url = links.split('"')[1] if links else ''
        selection = ('<input class="case-pick" type="checkbox" aria-label="Select case report" data-report="'+report_url+'"> ') if report_url else '' 
        report_url = links.split('"')[1] if links else ''
        selection = ('<input class="case-pick" type="checkbox" aria-label="Select case report" data-report="'+report_url+'"> ') if report_url else '' 
        links += _local_link(summary.get('quality_report_file'), root, 'QC data')
        case_dir = summary.get('output_dir')
        if case_dir:
            links += _local_link(Path(str(case_dir)) / 'summary.json', root, 'Summary data')
        token = status if status in {'ok', 'partial', 'error', 'failed', 'timeout'} else 'other'
        rows.append('<tr data-case="'+escape(name, quote=True)+'" data-status="'+escape(status, quote=True)+'" data-seconds="'+('NaN' if seconds is None else str(seconds))+'">'
                    '<td class="case">'+selection+escape(name)+'</td><td><span class="badge '+token+'">'+escape(status)+'</span></td>'
                    '<td>'+escape(output_status)+'</td><td>'+escape(qc_status)+'<div class="muted small">'+escape(qc_text)+'</div></td>'
                    '<td>'+('Not recorded' if seconds is None else f'{seconds:.2f} s')+'</td><td class="links">'+(links or 'No report available')+'</td>'
                    '<td class="error">'+('<br>'.join(escape(error) for error in errors) or '—')+'</td></tr>')
    cards = '<div class="card"><b>'+str(len(records))+'</b>Total cases</div>'
    for key in sorted(counts):
        cards += '<div class="card"><b>'+str(counts[key])+'</b>'+escape(key)+'</div>'
    options = '<option value="">All execution statuses</option>' + ''.join('<option value="'+escape(key, quote=True)+'">'+escape(key)+'</option>' for key in sorted(counts))
    generated = datetime.now(timezone.utc).isoformat(timespec='seconds')
    merge_script = """
const picks=()=>[...document.querySelectorAll('.case-pick')];
document.getElementById('select-visible').onclick=()=>{for(const box of picks())if(!box.closest('tr').hidden)box.checked=true;};
document.getElementById('clear-selection').onclick=()=>picks().forEach(box=>box.checked=false);
document.getElementById('merge-reports').onclick=()=>{const container=document.getElementById('merged-reports');container.replaceChildren();
for(const box of picks().filter(x=>x.checked)){const details=document.createElement('details'),heading=document.createElement('summary');heading.textContent=box.closest('tr').dataset.case+' · case QC report';details.append(heading);
details.addEventListener('toggle',()=>{if(details.open&&!details.querySelector('iframe')){const frame=document.createElement('iframe');frame.title=heading.textContent;frame.loading='lazy';frame.src=box.dataset.report;frame.style.cssText='width:100%;height:850px;border:1px solid #d8e2e9';details.append(frame);}});container.append(details);}
if(!container.children.length)container.textContent='No case reports selected.';};
""" 
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>AutoFlow batch review</title><style>'+_STYLE+'.case-pick{min-width:0;width:auto;display:inline}.merge-controls button{padding:8px}#merged-reports details{margin:14px 0}#merged-reports summary{cursor:pointer;padding:8px;background:white}</style></head><body><main><h1>AutoFlow batch review</h1>'
            '<p class="muted">All recorded cases, including failures and incomplete outputs. Open a case report for measurements and review evidence.</p>'
            '<p class="notice">Execution success and QC findings are separate. An ok run may still require scientific review; missing QC is not a pass.</p>'
            '<div class="cards">'+cards+'</div><div class="controls"><label>Search cases and errors<input id="search" type="search" placeholder="Case name or error text"></label>'
            '<label>Execution status<select id="status">'+options+'</select></label><span id="visible" role="status" aria-live="polite"></span></div>'
            '<div class="table"><table><thead><tr><th><button data-key="case">Case ↕</button></th><th><button data-key="status">Execution ↕</button></th>'
            '<th>Outputs</th><th>QC findings</th><th><button data-key="seconds">Elapsed ↕</button></th><th>Results</th><th>Failure details</th></tr></thead><tbody>'
            +''.join(rows)+'</tbody></table></div><p class="merge-controls"><button type="button" id="select-visible">Select visible reports</button> · <button type="button" id="clear-selection">Clear selection</button> · <button type="button" id="merge-reports">Merge selected reports</button></p><section id="merged-reports" aria-live="polite"></section><noscript><p>Filtering and sorting require JavaScript. All case rows remain visible above.</p></noscript>'
            '<footer>Generated '+escape(generated)+' · '+_local_link(root/'batch_report.json',root,'Canonical batch JSON')+'</footer></main><script>'+_SCRIPT+merge_script+'</script></body></html>')


def save_batch_report_html(results, html_path):
    '''Atomically replace the overview; the caller owns canonical JSON writes.'''
    target = Path(html_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = render_batch_report_html(results, target.parent)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=target.parent,
                                         prefix='.'+target.name+'.', suffix='.tmp', delete=False) as handle:
            temporary = handle.name
            handle.write(text)
        os.replace(temporary, target)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)
    return str(target)
