// GRADIENT SURVIVAL under quantization (Phase 1.5) — still no assembly.
//
// Value-survival (quantize.ts) ≠ gradient-survival. Here we carry a forward-mode
// DUAL tangent d(output)/d(one weight) through a fixed-point cell update and
// compare it to the f64 analytic gradient (finite differences of the real rule).
// This is the exact dual arithmetic Exp A proved on the Z80 — now for the rule's
// datapath. It tells us the Q-format whose GRADIENT is clean, before we commit
// the format to assembly (Phase 3 carries this same tangent on the real Z80).
//
//   npx tsx src/lib/devcomp/z80/gradfixed.ts

import { readFileSync } from 'node:fs';
import { EDIM, experimentById, seedGrid, step, readOutputs, type RuleConfig, type Experiment } from '../rule';
import {
	fmt, fmtName, toQ, fromQ, K, dotDual, reluDual, tanhDualIdeal, type Dual, type Fmt
} from './fixed';

const cfg = EDIM;
const exp = experimentById('e1_gate')!;
const par = new Float64Array(JSON.parse(readFileSync(new URL('../params/e1_gate.json', import.meta.url), 'utf8')));

// ---- one fixed-point DUAL cell update at cell i (mirrors rule.ts step) -----
// parD: params as duals (θ's tangent = 1, all others 0). sD: seed field as duals
// (all tangents 0 — the initial condition is constant). Returns cell i's C duals.
function cellDual(cfg: RuleConfig, parD: Dual[], sD: Dual[], i: number, f: Fmt): Dual[] {
	const { SW, C, HD, PERC, FEAT, W1O, B1O, W2O, B2O } = cfg;
	const r = i + 1, l = i - 1, u = i - SW, d = i + SW;
	const perc: Dual[] = new Array(PERC);
	for (let ch = 0; ch < C; ch++) {
		const b = ch * FEAT;
		const self = sD[i * C + ch], sr = sD[r * C + ch], sl = sD[l * C + ch], su = sD[u * C + ch], sd = sD[d * C + ch];
		perc[b] = self;
		perc[b + 1] = { v: (sr.v - sl.v) >> 1, d: (sr.d - sl.d) >> 1 };
		perc[b + 2] = { v: (sd.v - su.v) >> 1, d: (sd.d - su.d) >> 1 };
		perc[b + 3] = { v: sr.v + sl.v + su.v + sd.v - 4 * self.v, d: sr.d + sl.d + su.d + sd.d - 4 * self.d };
	}
	const h: Dual[] = new Array(HD);
	for (let hh = 0; hh < HD; hh++) {
		const w: Dual[] = new Array(PERC);
		for (let k = 0; k < PERC; k++) w[k] = parD[W1O + hh * PERC + k];
		h[hh] = reluDual(dotDual(w, perc, parD[B1O + hh], f));
	}
	const out: Dual[] = new Array(C);
	for (let c = 0; c < C; c++) {
		const w: Dual[] = new Array(HD);
		for (let hh = 0; hh < HD; hh++) w[hh] = parD[W2O + c * HD + hh];
		const dl = dotDual(w, h, parD[B2O + c], f);
		const pre: Dual = { v: sD[i * C + c].v + dl.v, d: sD[i * C + c].d + dl.d };
		out[c] = tanhDualIdeal(pre, f);
	}
	return out;
}

/** d(output-cell ch0 after 1 step)/dθ, carried through fixed-point dual arithmetic. */
function dualGrad(theta: number, inputs: number[], f: Fmt): number {
	const s0 = seedGrid(cfg, exp, inputs);
	const sD: Dual[] = Array.from(s0, (v) => ({ v: toQ(v, f), d: 0 }));
	const parD: Dual[] = Array.from(par, (v) => K(v, f));
	parD[theta] = { v: toQ(par[theta], f), d: toQ(1, f) }; // seed θ's tangent = 1.0
	const out = cellDual(cfg, parD, sD, exp.outputCells[0], f);
	return fromQ(out[0].d, f);
}

/** f64 ground-truth gradient d(out ch0 after 1 step)/dθ via central differences. */
function f64Grad(theta: number, inputs: number[], h = 1e-4): number {
	const s0 = seedGrid(cfg, exp, inputs);
	const outAt = (delta: number): number => {
		const p = Float64Array.from(par);
		p[theta] += delta;
		const ns = step(cfg, p, s0, exp, inputs);
		return readOutputs(cfg, ns, exp)[0];
	};
	return (outAt(h) - outAt(-h)) / (2 * h);
}

// ---- run ------------------------------------------------------------------
console.log('=== Phase 1.5: does the GRADIENT survive fixed-point? (dual vs f64 finite-diff) ===\n');
console.log('d(output ch0, 1 step)/dθ for representative weights, case [0,1]:\n');

const inputs = [0, 1];
// representative params: bias→ch0, a W2 weight into ch0, a W1 weight, a hidden bias
const THETAS: { name: string; idx: number }[] = [
	{ name: 'b2[0]        ', idx: cfg.B2O + 0 },
	{ name: 'W2[ch0,h5]   ', idx: cfg.W2O + 0 * cfg.HD + 5 },
	{ name: 'W2[ch0,h20]  ', idx: cfg.W2O + 0 * cfg.HD + 20 },
	{ name: 'b1[10]       ', idx: cfg.B1O + 10 },
	{ name: 'W1[h10,perc7]', idx: cfg.W1O + 10 * cfg.PERC + 7 },
	{ name: 'W1[h3,perc44]', idx: cfg.W1O + 3 * cfg.PERC + 44 }
];

const FORMATS: Fmt[] = [fmt(16, 8), fmt(24, 16), fmt(32, 16), fmt(32, 24)];

for (const f of FORMATS) {
	console.log(`  ${fmtName(f)}:`);
	console.log('    weight          dual grad      f64 grad       |err|');
	let worst = 0;
	for (const th of THETAS) {
		const g = dualGrad(th.idx, inputs, f);
		const gRef = f64Grad(th.idx, inputs);
		const err = Math.abs(g - gRef);
		worst = Math.max(worst, err);
		console.log(`    ${th.name}   ${g.toFixed(6).padStart(10)}   ${gRef.toFixed(6).padStart(10)}   ${err.toExponential(2)}`);
	}
	console.log(`    worst |err| = ${worst.toExponential(2)}  (Q resolution = ${(1 / 2 ** f.F).toExponential(2)})\n`);
}

console.log('If the dual tangent tracks the f64 gradient to ~Q resolution, the substrate');
console.log('carries the rule\'s training gradient — forward-mode, exactly as Exp A proved.');
