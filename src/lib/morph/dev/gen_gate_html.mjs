// Reads gate_viz.json (dumped by expE.ts GATE_VIZ) and writes a self-contained
// HTML that animates the grown XOR machine computing all 4 cases in lockstep.
import { readFileSync, writeFileSync } from 'node:fs';

const inPath = process.argv[2];
const outPath = process.argv[3];
const d = JSON.parse(readFileSync(inPath, 'utf8'));
const DATA = JSON.stringify(d);

const html = `<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Developmental computation — a grown XOR gate</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 32px 24px 56px;
    background: radial-gradient(1100px 520px at 50% -8%, rgba(45,212,191,.09), transparent 60%), #090c11;
    color: #e6edf3;
    font: 15px/1.55 ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  }
  .wrap { max-width: 980px; margin: 0 auto; }
  h1 { font-size: 27px; margin: 0 0 4px; letter-spacing: -.02em;
       background: linear-gradient(120deg,#e6edf3,#7dd3c8); -webkit-background-clip: text; background-clip:text; color: transparent; }
  .sub { color: #9aa7b4; margin: 0 0 22px; max-width: 68ch; }
  .controls { display:flex; align-items:center; gap:14px; margin: 8px 0 22px; flex-wrap: wrap; }
  button { background:#14324a; color:#cdeee7; border:1px solid #23566f; border-radius:9px;
           padding:8px 16px; font-size:14px; font-weight:600; cursor:pointer; }
  button:hover { background:#1a415f; }
  input[type=range] { width: 260px; accent-color:#2dd4bf; }
  .tlabel { color:#9aa7b4; font-variant-numeric: tabular-nums; min-width: 118px; }
  .cases { display:grid; grid-template-columns: repeat(4, 1fr); gap:16px; }
  @media (max-width: 720px){ .cases{ grid-template-columns: repeat(2,1fr);} }
  .case { background: rgba(255,255,255,.025); border:1px solid rgba(255,255,255,.07);
          border-radius:14px; padding:14px; display:flex; flex-direction:column; align-items:center; gap:8px; }
  .case h3 { margin:0; font-size:14px; font-weight:600; letter-spacing:.01em; }
  .case .want { color:#9aa7b4; font-weight:400; }
  canvas { border-radius:8px; image-rendering: pixelated; background:#05070a; }
  .readout { font-variant-numeric: tabular-nums; font-size:13px; }
  .ok { color:#34d399; } .bad { color:#f87171; }
  .legend { display:flex; gap:16px; align-items:center; color:#9aa7b4; font-size:12.5px; margin-top:18px; flex-wrap:wrap; }
  .swatch { display:inline-block; width:44px; height:11px; border-radius:3px; vertical-align:middle;
            background: linear-gradient(90deg,#2b6cff,#0a0f16 50%,#ff8a3d); }
  .marks { display:flex; gap:14px; font-size:12.5px; color:#9aa7b4; }
  .hid { margin-top:34px; }
  .hid h2 { font-size:16px; margin:0 0 4px; } .hid p { color:#9aa7b4; margin:0 0 14px; max-width:70ch; }
  .hgrid { display:grid; grid-template-columns: repeat(3, max-content); gap:22px; }
  .hcol { display:flex; flex-direction:column; align-items:center; gap:6px; }
  .hcol span { font-size:12px; color:#9aa7b4; }
  .foot { margin-top:34px; color:#6b7785; font-size:12.5px; border-top:1px solid rgba(255,255,255,.06); padding-top:16px; }
  code { color:#7dd3c8; }
</style></head>
<body><div class="wrap">
  <h1>A logic gate that was <em>grown</em>, not wired</h1>
  <p class="sub">One cellular-automaton rule — trained by gradient descent through development —
  reads two input cells and, after ${d.T} steps of growth, reports their <strong>XOR</strong> at a distant output cell.
  The same rule handles all four input cases: it can't memorize an answer, it has to build a computation.
  Watch the signal (channel 0) flow from the inputs (○) and combine at the output (□).</p>

  <div class="controls">
    <button id="play">▶ Play</button>
    <input type="range" id="scrub" min="0" max="${d.T}" value="0"/>
    <span class="tlabel">step <span id="tnum">0</span> / ${d.T}</span>
    <span class="marks">○ input&nbsp;&nbsp;□ output readout</span>
  </div>

  <div class="cases" id="cases"></div>

  <div class="legend"><span>signal value</span>
    <span>−1&nbsp;<span class="swatch"></span>&nbsp;+1</span>
    <span>(tanh-bounded field carried in cell memory)</span></div>

  <div class="hid">
    <h2>The wiring it invented</h2>
    <p>Hidden morphogen channels at the final step for case <strong>[0,1]</strong> — the internal
    structure the rule builds to route and combine the signals. Nobody designed these; gradient descent did.</p>
    <div class="hgrid" id="hgrid"></div>
  </div>

  <div class="foot">Algocell · developmental computation (E1) · a step toward self-repairing machines grown by gradient.</div>
</div>
<script>
const D = ${DATA};
const CELL = 22, W = D.SW, H = D.SH;
const inSet = new Set(D.inputCells), outCell = D.outputCell;

function color(v){ // diverging: blue(−) → dark(0) → orange(+)
  const t = Math.max(-1, Math.min(1, v));
  const bg = [10,15,22];
  const pos = [255,138,61], neg = [43,108,255];
  const c = t>=0 ? pos : neg, a = Math.abs(t);
  return 'rgb('+Math.round(bg[0]+(c[0]-bg[0])*a)+','+Math.round(bg[1]+(c[1]-bg[1])*a)+','+Math.round(bg[2]+(c[2]-bg[2])*a)+')';
}
function drawGrid(ctx, vals, markOut){
  for(let y=0;y<H;y++) for(let x=0;x<W;x++){
    const i=y*W+x;
    ctx.fillStyle = color(vals[i]);
    ctx.fillRect(x*CELL, y*CELL, CELL-1, CELL-1);
  }
  // input rings
  for(const i of inSet){ const x=i%W, y=(i/W|0);
    ctx.strokeStyle='#e6edf3'; ctx.lineWidth=2; ctx.beginPath();
    ctx.arc(x*CELL+CELL/2, y*CELL+CELL/2, CELL/2-4, 0, 7); ctx.stroke(); }
  if(markOut){ const x=outCell%W, y=(outCell/W|0);
    ctx.strokeStyle='#2dd4bf'; ctx.lineWidth=2.5;
    ctx.strokeRect(x*CELL+2, y*CELL+2, CELL-5, CELL-5); }
}

// build case panels
const casesEl = document.getElementById('cases');
const canvases = [];
D.cases.forEach((c, k)=>{
  const el = document.createElement('div'); el.className='case';
  const want = c.tgt, got = c.out, ok = Math.abs(got-want)<0.3;
  el.innerHTML = '<h3>in ['+c.in.join(',')+'] <span class="want">→ want '+want+'</span></h3>';
  const cv = document.createElement('canvas'); cv.width=W*CELL; cv.height=H*CELL;
  el.appendChild(cv);
  const ro = document.createElement('div'); ro.className='readout';
  ro.innerHTML = 'output: <span class="'+(ok?'ok':'bad')+'">'+got.toFixed(3)+'</span> '+(ok?'✓':'✗');
  el.appendChild(ro);
  casesEl.appendChild(el);
  canvases.push(cv.getContext('2d'));
});

// hidden channels for case [0,1] (index 1)
const hg = document.getElementById('hgrid');
const hcase = D.cases[1];
D.hiddenChannels.forEach((ch, j)=>{
  const col = document.createElement('div'); col.className='hcol';
  const cv = document.createElement('canvas'); cv.width=W*CELL; cv.height=H*CELL;
  col.appendChild(cv);
  const lab = document.createElement('span'); lab.textContent='hidden ch '+ch; col.appendChild(lab);
  hg.appendChild(col);
  // normalize hidden channel for contrast
  const vals = hcase.hidden[j]; let mx=1e-6; for(const v of vals) mx=Math.max(mx,Math.abs(v));
  drawGrid(cv.getContext('2d'), vals.map(v=>v/mx), false);
});

let t=0, playing=false, timer=null;
const scrub=document.getElementById('scrub'), tnum=document.getElementById('tnum'), playBtn=document.getElementById('play');
function render(){
  tnum.textContent=t; scrub.value=t;
  D.cases.forEach((c,k)=> drawGrid(canvases[k], c.frames[t], true));
}
function step(){ t=(t+1)%(D.T+1); render(); }
playBtn.onclick=()=>{ playing=!playing; playBtn.textContent=playing?'❚❚ Pause':'▶ Play';
  if(playing){ timer=setInterval(step, 180); } else clearInterval(timer); };
scrub.oninput=()=>{ t=+scrub.value; render(); };
render();
</script>
</body></html>`;

writeFileSync(outPath, html);
console.log('wrote', outPath);
