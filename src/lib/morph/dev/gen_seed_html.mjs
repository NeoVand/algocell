// Reads gseed_viz.json (dumped by expG.ts GVIZ) and writes a self-contained HTML
// animating the full lifecycle: a single seed cell GROWS the computational
// structure, computes XOR, is DAMAGED mid-run, and HEALS — one rule, four cases.
import { readFileSync, writeFileSync } from 'node:fs';

const inPath = process.argv[2], outPath = process.argv[3];
const d = JSON.parse(readFileSync(inPath, 'utf8'));
const DATA = JSON.stringify(d);

const html = `<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Grow-from-seed — a machine grown, computing, and self-healing</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin:0; padding:32px 24px 56px;
    background: radial-gradient(1100px 520px at 50% -8%, rgba(45,212,191,.09), transparent 60%), #090c11;
    color:#e6edf3; font:15px/1.55 ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
  .wrap { max-width:1000px; margin:0 auto; }
  h1 { font-size:27px; margin:0 0 4px; letter-spacing:-.02em;
       background:linear-gradient(120deg,#e6edf3,#7dd3c8); -webkit-background-clip:text; background-clip:text; color:transparent; }
  .sub { color:#9aa7b4; margin:0 0 20px; max-width:72ch; }
  .controls { display:flex; align-items:center; gap:14px; margin:8px 0 6px; flex-wrap:wrap; }
  button { background:#14324a; color:#cdeee7; border:1px solid #23566f; border-radius:9px; padding:8px 16px; font-size:14px; font-weight:600; cursor:pointer; }
  button:hover { background:#1a415f; }
  input[type=range] { width:340px; accent-color:#2dd4bf; }
  .tlabel { color:#9aa7b4; font-variant-numeric:tabular-nums; min-width:108px; }
  .phase { font-weight:700; min-width:210px; }
  .phase.seed{color:#34d399;} .phase.grow{color:#7dd3c8;} .phase.compute{color:#a5b4fc;} .phase.damage{color:#f87171;} .phase.heal{color:#fbbf24;} .phase.done{color:#34d399;}
  .track { position:relative; height:8px; border-radius:6px; margin:4px 0 22px; max-width:340px;
           background:linear-gradient(90deg,#1f6f63 0%, #1f6f63 var(--grow), #2a5674 var(--grow), #2a5674 var(--dmg), #f8717133 var(--dmg), #b4832e55 100%); }
  .track .m { position:absolute; top:-4px; width:2px; height:16px; }
  .track .mg { background:#7dd3c8; left:var(--grow); } .track .md { background:#f87171; left:var(--dmg); }
  .cases { display:grid; grid-template-columns:repeat(4,1fr); gap:16px; }
  @media (max-width:720px){ .cases{ grid-template-columns:repeat(2,1fr);} }
  .case { background:rgba(255,255,255,.025); border:1px solid rgba(255,255,255,.07); border-radius:14px; padding:14px; display:flex; flex-direction:column; align-items:center; gap:8px; }
  .case h3 { margin:0; font-size:14px; font-weight:600; } .case .want{ color:#9aa7b4; font-weight:400; }
  canvas { border-radius:8px; image-rendering:pixelated; background:#05070a; }
  .readout { font-variant-numeric:tabular-nums; font-size:13px; }
  .ok{color:#34d399;} .bad{color:#f87171;}
  .legend { display:flex; gap:16px; align-items:center; color:#9aa7b4; font-size:12.5px; margin-top:16px; flex-wrap:wrap; }
  .swatch { display:inline-block; width:44px; height:11px; border-radius:3px; vertical-align:middle; background:linear-gradient(90deg,#2b6cff,#0a0f16 50%,#ff8a3d); }
  .foot { margin-top:30px; color:#6b7785; font-size:12.5px; border-top:1px solid rgba(255,255,255,.06); padding-top:16px; }
  em { color:#7dd3c8; font-style:italic; }
</style></head>
<body><div class="wrap">
  <h1>Grown from a <em>single seed</em> — and it computes</h1>
  <p class="sub">The medium starts empty except one <span style="color:#34d399">seed cell</span> (🌱, centre).
  The same rule grows the whole computational structure outward, wires the two inputs (○) to the output (□),
  computes their <strong>XOR</strong> — then, when a <span style="color:#f87171">patch is destroyed</span>, regrows and
  keeps computing. Grown, functional, self-healing: one rule, four input cases, learned by gradient through development.</p>

  <div class="controls">
    <button id="play">▶ Play</button>
    <input type="range" id="scrub" min="0" max="${d.T_TOTAL}" value="0"/>
    <span class="tlabel">step <span id="tnum">0</span> / ${d.T_TOTAL}</span>
    <span class="phase seed" id="phase">🌱 seed</span>
  </div>
  <div class="track" style="--grow:${(100 * d.T_GROW / d.T_TOTAL).toFixed(1)}%; --dmg:${(100 * d.DMG_AT / d.T_TOTAL).toFixed(1)}%"><div class="m mg"></div><div class="m md"></div></div>

  <div class="cases" id="cases"></div>

  <div class="legend"><span>signal</span><span>−1&nbsp;<span class="swatch"></span>&nbsp;+1</span>
    <span style="color:#34d399">🌱 seed</span><span style="color:#f87171">▢ damage zone</span><span>○ input&nbsp;&nbsp;□ output</span></div>

  <div class="foot">Algocell · developmental computation (E3) · grow-from-seed → compute → self-repair, one rule.</div>
</div>
<script>
const D = ${DATA};
const CELL=24, W=D.SW, H=D.SH;
const inSet=new Set(D.inputCells), outCell=D.outputCell, dmgSet=new Set(D.damagedCells), seedCell=D.seedCell;
const DMG_AT=D.DMG_AT, T_GROW=D.T_GROW, T=D.T_TOTAL;
function color(v){ const t=Math.max(-1,Math.min(1,v)); const bg=[10,15,22],pos=[255,138,61],neg=[43,108,255];
  const c=t>=0?pos:neg,a=Math.abs(t);
  return 'rgb('+Math.round(bg[0]+(c[0]-bg[0])*a)+','+Math.round(bg[1]+(c[1]-bg[1])*a)+','+Math.round(bg[2]+(c[2]-bg[2])*a)+')'; }
function drawGrid(ctx, vals, t){
  for(let y=0;y<H;y++) for(let x=0;x<W;x++){ const i=y*W+x; ctx.fillStyle=color(vals[i]); ctx.fillRect(x*CELL,y*CELL,CELL-1,CELL-1); }
  // damage zone
  ctx.strokeStyle = t>=DMG_AT ? '#ff5b5b' : 'rgba(248,113,113,.5)'; ctx.lineWidth = t>=DMG_AT?2.5:1.5; ctx.setLineDash([4,3]);
  for(const i of dmgSet){ const x=i%W,y=(i/W|0); ctx.strokeRect(x*CELL+1.5,y*CELL+1.5,CELL-3,CELL-3); } ctx.setLineDash([]);
  // seed marker (green dot) — where growth began
  { const x=seedCell%W,y=(seedCell/W|0); ctx.fillStyle='#34d399'; ctx.beginPath(); ctx.arc(x*CELL+CELL/2,y*CELL+CELL/2,3,0,7); ctx.fill(); }
  // inputs
  for(const i of inSet){ const x=i%W,y=(i/W|0); ctx.strokeStyle='#e6edf3'; ctx.lineWidth=2; ctx.beginPath(); ctx.arc(x*CELL+CELL/2,y*CELL+CELL/2,CELL/2-4,0,7); ctx.stroke(); }
  // output
  { const x=outCell%W,y=(outCell/W|0); ctx.strokeStyle='#2dd4bf'; ctx.lineWidth=2.5; ctx.strokeRect(x*CELL+2,y*CELL+2,CELL-5,CELL-5); }
}
const casesEl=document.getElementById('cases'); const ctxs=[], readouts=[];
D.cases.forEach((c)=>{ const el=document.createElement('div'); el.className='case';
  el.innerHTML='<h3>in ['+c.in.join(',')+'] <span class="want">→ want '+c.tgt+'</span></h3>';
  const cv=document.createElement('canvas'); cv.width=W*CELL; cv.height=H*CELL; el.appendChild(cv);
  const ro=document.createElement('div'); ro.className='readout'; el.appendChild(ro);
  casesEl.appendChild(el); ctxs.push(cv.getContext('2d')); readouts.push({el:ro,c}); });
let t=0, playing=false, timer=null;
const scrub=document.getElementById('scrub'), tnum=document.getElementById('tnum'), playBtn=document.getElementById('play'), phaseEl=document.getElementById('phase');
function phase(t){
  if(t===0) return ['🌱 seed','seed'];
  if(t<T_GROW) return ['growing…','grow'];
  if(t<DMG_AT) return ['computing / holding','compute'];
  if(t===DMG_AT) return ['💥 damage!','damage'];
  if(t<T) return ['healing…','heal'];
  return ['grown · computed · healed ✓','done'];
}
function render(){ tnum.textContent=t; scrub.value=t; const [pl,cls]=phase(t); phaseEl.textContent=pl; phaseEl.className='phase '+cls;
  D.cases.forEach((c,k)=>{ drawGrid(ctxs[k], c.frames[t], t);
    const val=c.frames[t][outCell], ok=Math.abs(val-c.tgt)<0.3;
    readouts[k].el.innerHTML='output: <span class="'+(ok?'ok':'bad')+'">'+val.toFixed(3)+'</span> '+(ok?'✓':''); }); }
function step(){ t=(t+1)%(T+1); render(); }
playBtn.onclick=()=>{ playing=!playing; playBtn.textContent=playing?'❚❚ Pause':'▶ Play'; if(playing) timer=setInterval(step,170); else clearInterval(timer); };
scrub.oninput=()=>{ t=+scrub.value; render(); };
render();
</script>
</body></html>`;

writeFileSync(outPath, html);
console.log('wrote', outPath);
