// WGSL compute kernel for the developmental-computation field CA.
// One invocation per interior cell: perceive → MLP(ReLU) → tanh(state+dl).
// Dimensions are baked from a RuleConfig so the GPU rule is identical to the
// reference `forward()` for any grid/channel size. Damage and input-clamp are
// folded into the single pass, in the reference order (damage zeros the cell,
// then input clamp wins).

import type { RuleConfig } from './rule';

export function fieldShaderWGSL(cfg: RuleConfig): string {
	const { SW, SH, C, HD, PERC, W1O, B1O, W2O, B2O, markers } = cfg;
	const markerBinding = markers ? '@group(0) @binding(7) var<storage, read> isOutput : array<u32>;' : '';
	const markerDecl = markers ? 'let outp = isOutput[u32(i)] == 1u;' : '';
	// markers: ch1 = IN_MARK (1 at inputs), ch2 = OUT_MARK (1 at outputs), everywhere else 0.
	const markerClamp = markers
		? 'if (c == 1) { v = select(0.0, 1.0, inp); }\n    if (c == 2) { v = select(0.0, 1.0, outp); }'
		: '';
	const async = (cfg.fireRate ?? 1) < 1; // NCA stochastic updates (ctrl.y = per-step seed)
	const firesFn = `fn fires(cell : i32, seed : u32) -> bool {\n  ${async ? `var x = (u32(cell) * 0x9e3779b1u) ^ ((seed + 1u) * 0x85ebca77u) ^ 0xc2b2ae3du;\n  x = (x ^ (x >> 16u)) * 0x7feb352du;\n  x = (x ^ (x >> 15u)) * 0x846ca68bu;\n  x = x ^ (x >> 16u);\n  return x < ${Math.min(Math.floor((cfg.fireRate ?? 1) * 4294967296), 4294967295) >>> 0}u;` : 'return true;'}\n}`;
	return /* wgsl */ `
const SW : i32 = ${SW};
const SH : i32 = ${SH};
const C  : i32 = ${C};
const HD : i32 = ${HD};
const PERC : i32 = ${PERC};
const W1O : u32 = ${W1O}u;
const B1O : u32 = ${B1O}u;
const W2O : u32 = ${W2O}u;
const B2O : u32 = ${B2O}u;

@group(0) @binding(0) var<storage, read>        stateIn   : array<f32>;
@group(0) @binding(1) var<storage, read_write>  stateOut  : array<f32>;
@group(0) @binding(2) var<storage, read>        params    : array<f32>;
@group(0) @binding(3) var<storage, read>        isInput   : array<u32>;   // per-cell: 0 if not input, else (injection channel + 1)
@group(0) @binding(4) var<storage, read>        inputVal  : array<f32>;   // per-cell: clamped input bit
@group(0) @binding(5) var<storage, read>        damageKeep: array<u32>;   // per-cell: 1 keep, 0 destroy
@group(0) @binding(6) var<uniform>              ctrl      : vec4<u32>;     // x = applyDamage flag, y = step seed
${markerBinding}
${firesFn}

@compute @workgroup_size(8, 8)
fn step(@builtin(global_invocation_id) gid : vec3<u32>) {
  let x = i32(gid.x);
  let y = i32(gid.y);
  if (x < 1 || x >= SW - 1 || y < 1 || y >= SH - 1) { return; } // interior only; border stays 0
  let i  = y * SW + x;
  let r  = i + 1;
  let l  = i - 1;
  let up = i - SW;
  let dn = i + SW;

  let fire = fires(i, ctrl.y);          // NCA stochastic update (synchronous when fireRate=1)
  var h : array<f32, ${HD}>;
  if (fire) {
    var perc : array<f32, ${PERC}>;
    for (var ch = 0; ch < C; ch = ch + 1) {
      let sc   = stateIn[u32(i  * C + ch)];
      let sr   = stateIn[u32(r  * C + ch)];
      let sl   = stateIn[u32(l  * C + ch)];
      let su   = stateIn[u32(up * C + ch)];
      let sd   = stateIn[u32(dn * C + ch)];
      let b = ch * 4;
      perc[b]     = sc;
      perc[b + 1] = (sr - sl) * 0.5;
      perc[b + 2] = (sd - su) * 0.5;
      perc[b + 3] = sr + sl + su + sd - 4.0 * sc;
    }
    for (var u = 0; u < HD; u = u + 1) {
      var a = params[B1O + u32(u)];
      let base = W1O + u32(u * PERC);
      for (var k = 0; k < PERC; k = k + 1) { a = a + params[base + u32(k)] * perc[k]; }
      h[u] = max(a, 0.0);
    }
  }

  let dmg = (ctrl.x == 1u) && (damageKeep[u32(i)] == 0u);
  let inCode = isInput[u32(i)];        // 0 = not input; else (injection channel + 1)
  let inp = inCode != 0u;
  ${markerDecl}
  for (var c = 0; c < C; c = c + 1) {
    var v = stateIn[u32(i * C + c)];    // non-firing cell keeps its state
    if (fire) {
      var dl = params[B2O + u32(c)];
      let base = W2O + u32(c * HD);
      for (var u = 0; u < HD; u = u + 1) { dl = dl + params[base + u32(u)] * h[u]; }
      v = tanh(v + dl);
    }
    if (dmg) { v = 0.0; }              // damage zeros the cell...
    ${markerClamp}
    if (inp && c == i32(inCode) - 1) { v = inputVal[u32(i)]; }  // ...then input clamp wins, into that input's channel (matches reference)
    stateOut[u32(i * C + c)] = v;
  }
}
`;
}
