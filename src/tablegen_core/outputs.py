"""Portable CSV/NPY/JSON output and a small self-contained HTML preview."""
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import tempfile
import numpy as np


def _preview(data, columns, metadata):
    head = ''.join('<th>' + html.escape(str(name)) + '</th>' for name in columns)
    rows = ''.join('<tr>' + ''.join(f'<td>{float(value):.6g}</td>' for value in row) + '</tr>' for row in data[:20])
    # The canvas uses deterministic row sampling; rendering never consumes RNG.
    sampled = data[np.linspace(0, len(data)-1, min(len(data), 1000), dtype=int)]
    payload = json.dumps({'columns': columns, 'rows': sampled.tolist()}, allow_nan=False).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    detail = html.escape(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False))
    return '''<!doctype html><html lang="ko"><meta charset="utf-8">
<title>합성 표 미리보기</title><style>
body{font:15px system-ui;max-width:1100px;margin:30px auto;padding:0 18px;color:#172c39;background:#f5f7fa}
canvas{background:white;width:100%;height:420px;border:1px solid #ccd5df;touch-action:none}
table{border-collapse:collapse;background:white}td,th{padding:7px 12px;border:1px solid #ccd5df;text-align:right}
pre{white-space:pre-wrap;overflow-wrap:anywhere}select{margin:5px}section{overflow:auto}select:hover{background:#e6efec}select:active{background:#d6e5e0}select:focus-visible{outline:2px solid #215a4b}@media(prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
</style><label>Language / 언어 <select id="language"><option value="ko">한국어</option><option value="en">English</option></select></label><h1 id="page-title">합성 표 미리보기</h1><p id="instructions">마우스로 드래그하면 3D 그림이 회전합니다. X·Y·Z 열을 선택하세요. 그림은 최대 1,000행을 일정 간격으로 표시하며 축별로 표준화합니다. 저장된 표의 값은 바뀌지 않습니다. 아래 표는 앞 20행입니다.</p>
<label>X <select id="x"></select></label><label>Y <select id="y"></select></label><label>Z <select id="z"></select></label>
<canvas id="plot" width="1000" height="420"></canvas><h2 id="preview-title">표 미리보기</h2><section><table><thead><tr>''' + head + '</tr></thead><tbody>' + rows + '''</tbody></table></section>
<details><summary id="settings-title">생성 설정·기록</summary><pre>''' + detail + '''</pre></details>
<script type="application/json" id="data">''' + payload + '''</script><script>
'use strict';
const translations={ko:{title:'합성 표 미리보기',instructions:'마우스로 드래그하면 3D 그림이 회전합니다. X·Y·Z 열을 선택하세요. 그림은 최대 1,000행을 일정 간격으로 표시하며 축별로 표준화합니다. 저장된 표의 값은 바뀌지 않습니다. 아래 표는 앞 20행입니다.',preview:'표 미리보기',settings:'생성 설정·기록'},en:{title:'Synthetic table preview',instructions:'Drag to rotate the 3D plot. Choose the X, Y and Z columns. The plot shows at most 1,000 evenly sampled rows, standardized per axis for display only. Saved data is unchanged. The table below shows the first 20 rows.',preview:'Table preview',settings:'Generation settings and record'}};
document.getElementById('language').onchange=e=>{const lang=e.target.value,t=translations[lang];document.documentElement.lang=lang;document.title=t.title;document.getElementById('page-title').textContent=t.title;document.getElementById('instructions').textContent=t.instructions;document.getElementById('preview-title').textContent=t.preview;document.getElementById('settings-title').textContent=t.settings;};
const d=JSON.parse(document.getElementById('data').textContent),c=document.getElementById('plot'),ctx=c.getContext('2d');
let yaw=.5,pitch=.3,drag=null;const sels=['x','y','z'].map((id,j)=>{const s=document.getElementById(id);d.columns.forEach((n,i)=>{const o=document.createElement('option');o.textContent=n;o.value=i;s.append(o)});s.value=Math.min(j,d.columns.length-1);s.onchange=draw;return s});
function draw(){ctx.clearRect(0,0,c.width,c.height);const ids=sels.map(s=>+s.value),mu=ids.map(i=>d.rows.reduce((a,r)=>a+r[i],0)/d.rows.length),sd=ids.map((i,k)=>Math.sqrt(d.rows.reduce((a,r)=>a+(r[i]-mu[k])**2,0)/d.rows.length)||1);
const pts=d.rows.map(r=>{const v=ids.map((i,k)=>(r[i]-mu[k])/sd[k]),x=v[0]*Math.cos(yaw)+v[2]*Math.sin(yaw),z=-v[0]*Math.sin(yaw)+v[2]*Math.cos(yaw);return [x,v[1]*Math.cos(pitch)-z*Math.sin(pitch),v[1]*Math.sin(pitch)+z*Math.cos(pitch)]}).sort((a,b)=>a[2]-b[2]);
ctx.strokeStyle='#ccd5df';ctx.beginPath();ctx.moveTo(0,210);ctx.lineTo(1000,210);ctx.moveTo(500,0);ctx.lineTo(500,420);ctx.stroke();pts.forEach(p=>{ctx.fillStyle='rgba(15,100,160,.65)';ctx.beginPath();ctx.arc(500+p[0]*52,210-p[1]*52,2.5,0,Math.PI*2);ctx.fill()})}
c.onpointerdown=e=>{drag=[e.clientX,e.clientY];c.setPointerCapture(e.pointerId)};c.onpointermove=e=>{if(!drag)return;yaw+=(e.clientX-drag[0])*.01;pitch+=(e.clientY-drag[1])*.01;drag=[e.clientX,e.clientY];draw()};c.onpointerup=c.onpointercancel=()=>drag=null;draw();
</script></html>'''


def write_result(result, directory):
    data = np.asarray(result.data)
    if data.ndim != 2 or not np.issubdtype(data.dtype, np.number) or np.iscomplexobj(data) or not np.isfinite(data).all():
        raise ValueError('data must be a finite real numeric matrix')
    if len(result.columns) != data.shape[1] or len(set(result.columns)) != len(result.columns):
        raise ValueError('columns must be unique and match the matrix width')
    target = Path(directory).expanduser().absolute()
    if target.exists():
        raise FileExistsError(f'output directory already exists: {target}')
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f'.tablegen-core-{os.getpid()}-', dir=target.parent))
    try:
        with (staging/'table.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow(result.columns)
            writer.writerows(data.tolist())
        np.save(staging/'table.npy', data, allow_pickle=False)
        (staging/'report.html').write_text(_preview(data, result.columns, result.metadata), encoding='utf-8')
        metadata = dict(result.metadata)
        metadata['output_files'] = {name: {'sha256': hashlib.sha256((staging/name).read_bytes()).hexdigest(), 'bytes': (staging/name).stat().st_size} for name in ('table.csv', 'table.npy', 'report.html')}
        (staging/'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        publish_staged(staging, target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return {name: str(target/name) for name in ('table.csv', 'table.npy', 'metadata.json', 'report.html')}


def publish_staged(staging, target):
    """Reserve destination exclusively; metadata last, rollback on copy error."""
    staging, target = Path(staging), Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir()  # Must fail even if another process just created an empty dir.
    published = []
    try:
        for name in ('table.csv', 'table.npy', 'report.html', 'metadata.json'):
            with (staging/name).open('rb') as source, (target/name).open('xb') as destination:
                published.append(target/name)
                shutil.copyfileobj(source, destination)
    except BaseException:
        for path in published:
            path.unlink(missing_ok=True)
        try:
            target.rmdir()
        except OSError:
            pass
        raise
