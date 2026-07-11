// WGSL kernels for GPU/browser TRAINING of the field-CA rule (reverse-mode BPTT).
// Companion to the forward `shader.ts`; adds the backward pass + Adam so training runs on
// the GPU (fast, and in-browser). Math mirrors the finite-diff-checked CPU reference
// (expI.ts / rule.ts) and is validated by /devcomp/traingpu against it.
//
// Batch packing: B independent samples (each = one port placement × one input case) along a
// batch axis; buffers are [B][N][C]. The rule is sample-independent → one invocation per
// (b, cell). Param grads are SHARED across samples/steps → accumulated by small matmul
// reductions over cells (no f32 atomics).
//
// Buffers are packed to fit WebGPU's baseline 8-storage-buffers/stage limit:
//   optim = [grad(P) | m(P) | v(P)] ; portsU = [isInput(BN) | isOutput(BN)] ;
//   portsF = [inputVal(BN) | targetVal(BN)] ; scratch cell = [gp|gpre1|hAct|perc|gperc].
// isInput encodes 0=none else (injection channel + 1), same as the forward shader.

import type { RuleConfig } from './rule';

export interface TrainDims { B: number; T: number; aliveFrom: number; whold?: number; }

export function trainShaderWGSL(cfg: RuleConfig, dims: TrainDims): string {
	const { SW, SH, N, C, HD, PERC, W1O, B1O, W2O, B2O, P } = cfg;
	const { B, T, aliveFrom } = dims;
	const whold = dims.whold ?? 1; // persistence window: score the output over the last `whold` states
	const BN = N * B;
	const OFF_GP = 0, OFF_GPRE1 = C, OFF_HACT = C + HD, OFF_PERC = C + 2 * HD, OFF_GPERC = C + 2 * HD + PERC;
	const SCR = C + 2 * HD + 2 * PERC;
	const IN_MARK = 1, OUT_MARK = 2;
	// Marker rules (movable ports) stamp ch1=IN_MARK, ch2=OUT_MARK each step and carry no
	// gradient through them. Non-marker rules (fixed-layout gate/adder) use ch1.. as ordinary
	// hidden channels, so those lines must be OMITTED — else they corrupt hidden state/grad.
	const mk = cfg.markers;
	const stampFwd = mk
		? `if (c == ${IN_MARK}) { v = select(0.0, 1.0, inp); }\n    if (c == ${OUT_MARK}) { v = select(0.0, 1.0, outp); }`
		: '';
	const zeroMarkGrad = mk ? `if (c == ${IN_MARK} || c == ${OUT_MARK}) { gsv = 0.0; }` : '';
	return /* wgsl */ `
const SW : i32 = ${SW};
const SH : i32 = ${SH};
const N  : i32 = ${N};
const C  : i32 = ${C};
const HD : i32 = ${HD};
const PERC : i32 = ${PERC};
const BATCH : i32 = ${B};
const NB : i32 = ${BN};
const NBC : u32 = ${BN * C}u;
const W1O : u32 = ${W1O}u;
const B1O : u32 = ${B1O}u;
const W2O : u32 = ${W2O}u;
const B2O : u32 = ${B2O}u;
const PP : u32 = ${P}u;
const MOFF : u32 = ${P}u;         // m offset inside optim
const VOFF : u32 = ${2 * P}u;     // v offset inside optim
const NORMOFF : u32 = ${3 * P}u;  // ‖grad‖² scalar slot inside optim (optim sized 3P+4)
const ALIVE0 : i32 = ${aliveFrom};
const SCR : i32 = ${SCR};
const OFF_GP : i32 = ${OFF_GP};
const OFF_GPRE1 : i32 = ${OFF_GPRE1};
const OFF_HACT : i32 = ${OFF_HACT};
const OFF_PERC : i32 = ${OFF_PERC};
const OFF_GPERC : i32 = ${OFF_GPERC};
const OUToff : u32 = ${BN}u;      // isOutput offset inside portsU
const INV1 : u32 = ${BN}u;        // portsF: inputVal1 (post-switch input)
const TGT0 : u32 = ${2 * BN}u;    // portsF: target0 (pre-switch answer)
const TGT1 : u32 = ${3 * BN}u;    // portsF: target1 (post-switch answer)
const TFINAL : u32 = ${T}u;
const NORMF : f32 = ${(B * whold).toFixed(1)}; // batch × hold-window (mean over both)
const FIRE_THRESH : u32 = ${Math.min(Math.floor((cfg.fireRate ?? 1) * 4294967296), 4294967295) >>> 0}u;

// ctrl: tsw = input-switch state (0 = non-reactive); seed = per-iter stochastic-update seed
struct Ctrl { t : u32, dir : u32, tsw : u32, seed : u32, lr : f32, beta1c : f32, beta2c : f32, pad3 : f32 };

// NCA stochastic update: cell fires (updates) this step? Bit-identical to rule.ts cellFires().
fn fires(cell : i32, step : u32, seed : u32) -> bool {
  ${(cfg.fireRate ?? 1) >= 1 ? 'return true;' : `
  var x = (u32(cell) * 0x9e3779b1u) ^ ((step + 1u) * 0x85ebca77u) ^ ((seed + 1u) * 0xc2b2ae3du);
  x = (x ^ (x >> 16u)) * 0x7feb352du;
  x = (x ^ (x >> 15u)) * 0x846ca68bu;
  x = x ^ (x >> 16u);
  return x < FIRE_THRESH;`}
}

@group(0) @binding(0) var<storage, read_write>  params  : array<f32>;   // Adam writes this
@group(0) @binding(1) var<storage, read_write>  optim   : array<f32>;   // [grad(P)|m(P)|v(P)]
@group(0) @binding(2) var<storage, read_write>  traj    : array<f32>;   // [(T+1)*B*N*C]
@group(0) @binding(3) var<storage, read>        portsU  : array<u32>;   // [isInput(BN)|isOutput(BN)]
@group(0) @binding(4) var<storage, read>        portsF  : array<f32>;   // [inVal0|inVal1|tgt0|tgt1] each BN
@group(0) @binding(5) var<storage, read_write>  gsA     : array<f32>;   // [B*N*C]
@group(0) @binding(6) var<storage, read_write>  gsB     : array<f32>;   // [B*N*C]
@group(0) @binding(7) var<storage, read_write>  scratch : array<f32>;   // [B*N*SCR]
@group(0) @binding(8) var<uniform>              ctrl    : Ctrl;

fn perc4(trajBase : u32, b : i32, i : i32, ch : i32) -> vec4<f32> {
  let sc = traj[trajBase + u32((b * N + i)      * C + ch)];
  let sr = traj[trajBase + u32((b * N + (i + 1))  * C + ch)];
  let sl = traj[trajBase + u32((b * N + (i - 1))  * C + ch)];
  let su = traj[trajBase + u32((b * N + (i - SW)) * C + ch)];
  let sd = traj[trajBase + u32((b * N + (i + SW)) * C + ch)];
  return vec4<f32>(sc, (sr - sl) * 0.5, (sd - su) * 0.5, sr + sl + su + sd - 4.0 * sc);
}

// ---- FORWARD: traj[t] -> traj[t+1] (rule + stamp) ----
@compute @workgroup_size(64)
fn fwd(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = i32(gid.x);
  if (idx >= NB) { return; }
  let b = idx / N;
  let i = idx - b * N;
  let x = i - (i / SW) * SW;
  let y = i / SW;
  let t = ctrl.t;
  let inBase  = t * NBC;
  let outBase = (t + 1u) * NBC;
  let cellBase = u32((b * N + i) * C);
  if (x < 1 || x >= SW - 1 || y < 1 || y >= SH - 1) {
    for (var c = 0; c < C; c = c + 1) { traj[outBase + cellBase + u32(c)] = 0.0; }
    return;
  }
  let fire = fires(i, t, ctrl.seed);           // NCA stochastic update
  var h : array<f32, ${HD}>;
  if (fire) {
    var perc : array<f32, ${PERC}>;
    for (var ch = 0; ch < C; ch = ch + 1) {
      let p = perc4(inBase, b, i, ch);
      let bb = ch * 4;
      perc[bb] = p.x; perc[bb + 1] = p.y; perc[bb + 2] = p.z; perc[bb + 3] = p.w;
    }
    for (var u = 0; u < HD; u = u + 1) {
      var a = params[B1O + u32(u)];
      let base = W1O + u32(u * PERC);
      for (var k = 0; k < PERC; k = k + 1) { a = a + params[base + u32(k)] * perc[k]; }
      h[u] = max(a, 0.0);
    }
  }
  let sIdx = u32(b * N + i);
  let inCode = portsU[sIdx];
  let inp = inCode != 0u;
  let outp = portsU[OUToff + sIdx] != 0u;
  for (var c = 0; c < C; c = c + 1) {
    var v = traj[inBase + cellBase + u32(c)];   // non-firing cell keeps its state
    if (fire) {
      var dl = params[B2O + u32(c)];
      let base = W2O + u32(c * HD);
      for (var u = 0; u < HD; u = u + 1) { dl = dl + params[base + u32(u)] * h[u]; }
      v = tanh(v + dl);
    }
    ${stampFwd}
    if (inp && c == i32(inCode) - 1) {
      let useV1 = (t + 1u) >= ctrl.tsw;          // state t+1 carries post-switch input (tsw=0 → always v1==v0)
      v = select(portsF[sIdx], portsF[INV1 + sIdx], useV1);
    }
    traj[outBase + cellBase + u32(c)] = v;
  }
}

// ---- SEED GRAD: gsA = dL/d state[T]; output cells' ch0 = 2*(o - tgt)/B ----
@compute @workgroup_size(64)
fn seedGrad(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = i32(gid.x);
  if (idx >= NB) { return; }
  let sIdx = u32(idx);
  let base = sIdx * u32(C);
  for (var c = 0; c < C; c = c + 1) { gsA[base + u32(c)] = 0.0; }
  if (portsU[OUToff + sIdx] != 0u) {
    let o = traj[TFINAL * NBC + base];
    gsA[base] = 2.0 * (o - portsF[TGT1 + sIdx]) / NORMF; // final state = last window-B step (post-switch)
  }
}

// ---- PERSISTENCE: at a windowed step t<T, add the direct output-loss gradient to gsOut ----
// (dispatched only for t in [T-whold+1, T-1]; ctrl.t = t, ctrl.dir picks which gs holds dL/dstate[t])
@compute @workgroup_size(64)
fn injectOut(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = i32(gid.x);
  if (idx >= NB) { return; }
  let sIdx = u32(idx);
  if (portsU[OUToff + sIdx] == 0u) { return; } // output cells only
  let base = sIdx * u32(C);
  let o = traj[ctrl.t * NBC + base]; // output readout at state t
  let tgt = select(portsF[TGT1 + sIdx], portsF[TGT0 + sIdx], ctrl.t < ctrl.tsw); // window A→tgt0, B→tgt1
  let g = 2.0 * (o - tgt) / NORMF;
  if (ctrl.dir == 0u) { gsB[base] = gsB[base] + g; } else { gsA[base] = gsA[base] + g; }
}

// ---- BWD1: residual gsOut + store scratch(gp,gpre1,hAct,perc,gperc). dir picks gs read/write ----
@compute @workgroup_size(64)
fn bwd1(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = i32(gid.x);
  if (idx >= NB) { return; }
  let b = idx / N;
  let i = idx - b * N;
  let x = i - (i / SW) * SW;
  let y = i / SW;
  let gBase = u32((b * N + i) * C);
  let scrBase = u32((b * N + i) * SCR);
  let dir = ctrl.dir;
  if (x < 1 || x >= SW - 1 || y < 1 || y >= SH - 1) {
    for (var c = 0; c < C; c = c + 1) {
      if (dir == 0u) { gsB[gBase + u32(c)] = 0.0; } else { gsA[gBase + u32(c)] = 0.0; }
    }
    return;
  }
  let t = ctrl.t;
  let inBase  = t * NBC;
  let outBase = (t + 1u) * NBC;
  let sIdx = u32(b * N + i);
  let inCode = portsU[sIdx];
  let inp = inCode != 0u;
  if (!fires(i, t, ctrl.seed)) {
    // non-firing cell was identity (ns=s) → clamped grad passes straight through; no MLP → clear scratch
    for (var c = 0; c < C; c = c + 1) {
      var gsv = select(gsB[gBase + u32(c)], gsA[gBase + u32(c)], dir == 0u);
      ${zeroMarkGrad}
      if (inp && c == i32(inCode) - 1) { gsv = 0.0; }
      if (dir == 0u) { gsB[gBase + u32(c)] = gsv; } else { gsA[gBase + u32(c)] = gsv; }
      scratch[scrBase + u32(OFF_GP + c)] = 0.0;
    }
    for (var u = 0; u < HD; u = u + 1) { scratch[scrBase + u32(OFF_GPRE1 + u)] = 0.0; scratch[scrBase + u32(OFF_HACT + u)] = 0.0; }
    for (var k = 0; k < PERC; k = k + 1) { scratch[scrBase + u32(OFF_PERC + k)] = 0.0; scratch[scrBase + u32(OFF_GPERC + k)] = 0.0; }
    return;
  }
  var perc : array<f32, ${PERC}>;
  for (var ch = 0; ch < C; ch = ch + 1) {
    let p = perc4(inBase, b, i, ch);
    let bb = ch * 4;
    perc[bb] = p.x; perc[bb + 1] = p.y; perc[bb + 2] = p.z; perc[bb + 3] = p.w;
  }
  var pre1 : array<f32, ${HD}>;
  var hh   : array<f32, ${HD}>;
  for (var u = 0; u < HD; u = u + 1) {
    var a = params[B1O + u32(u)];
    let base = W1O + u32(u * PERC);
    for (var k = 0; k < PERC; k = k + 1) { a = a + params[base + u32(k)] * perc[k]; }
    pre1[u] = a; hh[u] = max(a, 0.0);
  }
  var gh : array<f32, ${HD}>;
  for (var u = 0; u < HD; u = u + 1) { gh[u] = 0.0; }
  for (var c = 0; c < C; c = c + 1) {
    var gsv = select(gsB[gBase + u32(c)], gsA[gBase + u32(c)], dir == 0u);
    ${zeroMarkGrad}
    if (inp && c == i32(inCode) - 1) { gsv = 0.0; }
    let sp = traj[outBase + gBase + u32(c)];
    let gp = gsv * (1.0 - sp * sp);
    scratch[scrBase + u32(OFF_GP + c)] = gp;
    if (dir == 0u) { gsB[gBase + u32(c)] = gp; } else { gsA[gBase + u32(c)] = gp; }
    let wbase = W2O + u32(c * HD);
    for (var u = 0; u < HD; u = u + 1) { gh[u] = gh[u] + params[wbase + u32(u)] * gp; }
  }
  var gperc : array<f32, ${PERC}>;
  for (var k = 0; k < PERC; k = k + 1) { gperc[k] = 0.0; }
  for (var u = 0; u < HD; u = u + 1) {
    var g = gh[u];
    if (pre1[u] <= 0.0) { g = 0.0; }
    scratch[scrBase + u32(OFF_GPRE1 + u)] = g;
    scratch[scrBase + u32(OFF_HACT + u)] = hh[u];
    let base = W1O + u32(u * PERC);
    for (var k = 0; k < PERC; k = k + 1) { gperc[k] = gperc[k] + params[base + u32(k)] * g; }
  }
  for (var k = 0; k < PERC; k = k + 1) {
    scratch[scrBase + u32(OFF_PERC + k)] = perc[k];
    scratch[scrBase + u32(OFF_GPERC + k)] = gperc[k];
  }
}

// ---- BWD GATHER: gsOut[j] += perc-transpose gather from gperc of j and its 4 neighbours ----
@compute @workgroup_size(64)
fn bwdGather(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = i32(gid.x);
  if (idx >= NB) { return; }
  let b = idx / N;
  let j = idx - b * N;
  let x = j - (j / SW) * SW;
  let y = j / SW;
  if (x < 1 || x >= SW - 1 || y < 1 || y >= SH - 1) { return; }
  let dir = ctrl.dir;
  let gBase = u32((b * N + j) * C);
  let sJ = u32((b * N + j)      * SCR + OFF_GPERC);
  let sR = u32((b * N + (j + 1))  * SCR + OFF_GPERC);
  let sL = u32((b * N + (j - 1))  * SCR + OFF_GPERC);
  let sD = u32((b * N + (j + SW)) * SCR + OFF_GPERC);
  let sU = u32((b * N + (j - SW)) * SCR + OFF_GPERC);
  for (var ch = 0; ch < C; ch = ch + 1) {
    let o = u32(ch * 4);
    var g = scratch[sJ + o] - 4.0 * scratch[sJ + o + 3u];
    g = g + 0.5 * scratch[sL + o + 1u] - 0.5 * scratch[sR + o + 1u];
    g = g + 0.5 * scratch[sU + o + 2u] - 0.5 * scratch[sD + o + 2u];
    g = g + scratch[sR + o + 3u] + scratch[sL + o + 3u] + scratch[sD + o + 3u] + scratch[sU + o + 3u];
    if (dir == 0u) { gsB[gBase + u32(ch)] = gsB[gBase + u32(ch)] + g; }
    else { gsA[gBase + u32(ch)] = gsA[gBase + u32(ch)] + g; }
  }
}

// ---- PARAM-GRAD reductions over all cells (accumulate across steps into optim[0..P)) ----
@compute @workgroup_size(64)
fn gradW1(@builtin(global_invocation_id) gid : vec3<u32>) {
  let o = i32(gid.x);
  if (o >= HD * PERC) { return; }
  let hh = o / PERC;
  let k = o - hh * PERC;
  var acc = 0.0;
  for (var cell = 0; cell < NB; cell = cell + 1) {
    let sb = u32(cell * SCR);
    acc = acc + scratch[sb + u32(OFF_GPRE1 + hh)] * scratch[sb + u32(OFF_PERC + k)];
  }
  optim[W1O + u32(o)] = optim[W1O + u32(o)] + acc;
}

@compute @workgroup_size(64)
fn gradW2(@builtin(global_invocation_id) gid : vec3<u32>) {
  let o = i32(gid.x);
  if (o >= C * HD) { return; }
  let c = o / HD;
  let hh = o - c * HD;
  var acc = 0.0;
  for (var cell = 0; cell < NB; cell = cell + 1) {
    let sb = u32(cell * SCR);
    acc = acc + scratch[sb + u32(OFF_GP + c)] * scratch[sb + u32(OFF_HACT + hh)];
  }
  optim[W2O + u32(o)] = optim[W2O + u32(o)] + acc;
}

@compute @workgroup_size(64)
fn gradBias(@builtin(global_invocation_id) gid : vec3<u32>) {
  let o = i32(gid.x);
  if (o < HD) {
    var acc = 0.0;
    for (var cell = 0; cell < NB; cell = cell + 1) { acc = acc + scratch[u32(cell * SCR + OFF_GPRE1 + o)]; }
    optim[B1O + u32(o)] = optim[B1O + u32(o)] + acc;
  } else if (o < HD + C) {
    let c = o - HD;
    var acc = 0.0;
    for (var cell = 0; cell < NB; cell = cell + 1) { acc = acc + scratch[u32(cell * SCR + OFF_GP + c)]; }
    optim[B2O + u32(c)] = optim[B2O + u32(c)] + acc;
  }
}

@compute @workgroup_size(64)
fn zeroGrad(@builtin(global_invocation_id) gid : vec3<u32>) {
  let o = gid.x;
  if (o >= PP) { return; }
  optim[o] = 0.0;
}

// ---- GRAD NORM: ‖grad‖² into optim[NORMOFF] (single thread; P is small, runs in µs) ----
@compute @workgroup_size(1)
fn gradNormSq(@builtin(global_invocation_id) gid : vec3<u32>) {
  if (gid.x != 0u) { return; }
  var s = 0.0;
  for (var o = 0u; o < PP; o = o + 1u) { s = s + optim[o] * optim[o]; }
  optim[NORMOFF] = s;
}

// ---- ADAM with gradient clipping (‖grad‖→1) + weight decay. ctrl.lr, beta1c, beta2c ----
@compute @workgroup_size(64)
fn adam(@builtin(global_invocation_id) gid : vec3<u32>) {
  let o = gid.x;
  if (o >= PP) { return; }
  let norm = sqrt(optim[NORMOFF]);
  let clip = select(1.0, 1.0 / norm, norm > 1.0);
  let g = optim[o] * clip + 2.0e-5 * params[o];
  let m = 0.9 * optim[MOFF + o] + 0.1 * g;
  let v = 0.999 * optim[VOFF + o] + 0.001 * g * g;
  optim[MOFF + o] = m; optim[VOFF + o] = v;
  let mh = m / ctrl.beta1c;
  let vh = v / ctrl.beta2c;
  params[o] = params[o] - ctrl.lr * mh / (sqrt(vh) + 1e-8);
}
`;
}
