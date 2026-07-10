// QUANTIZATION STUDY (Phase 1 of the Z80-substrate proof) — no assembly yet.
//
// Question: does the trained developmental-computation rule still COMPUTE when
// every value/weight is a signed fixed-point integer (as it must be on a Z80)?
// And at what precision does the XOR attractor survive the full rollout?
//
// We re-run the E1 gate two ways from the same seed — the f64 reference
// (`rule.ts`) and a BIT-FAITHFUL fixed-point emulation (`fixed.ts`, which does
// exactly what the Z80 datapath will do) — sweeping Q-formats, and check the
// truth table + output trajectory. The winning format is what we then build in
// Z80 assembly (Phase 2). If NOTHING survives, S9 is in trouble and we know now.
//
//   npx tsx src/lib/devcomp/z80/quantize.ts

import { readFileSync } from 'node:fs';
import { EDIM, experimentById, seedGrid, perceive, readOutputs, forward, type RuleConfig, type Experiment } from '../rule';
import { fmt, fmtName, toQ, fromQ, dotQ, reluQ, tanhIdeal, buildTanhTable, tanhTable, newOverflow, type Fmt, type OverflowStats } from './fixed';

const cfg = EDIM;
const exp = experimentById('e1_gate')!;
const par = new Float64Array(JSON.parse(readFileSync(new URL('../params/e1_gate.json', import.meta.url), 'utf8')));
const STEPS = exp.tGrow; // 24

// ---- fixed-point rollout: mirrors rule.ts step(), integers only -----------

interface FixedOpts {
	f: Fmt;
	tanh: (qx: number) => number; // Q→Q tanh (ideal or table)
	ov: OverflowStats;
}

/** Quantize the whole param block once. */
function quantParams(f: Fmt): Int32Array {
	const q = new Int32Array(cfg.P);
	for (let i = 0; i < cfg.P; i++) q[i] = toQ(par[i], f);
	return q;
}

/** One fixed-point step — identical structure to rule.ts step(), all Q(F) ints. */
function stepFixed(cfg: RuleConfig, parQ: Int32Array, s: Int32Array, exp: Experiment, inputs: number[], o: FixedOpts): Int32Array {
	const { SW, SH, C, HD, PERC, FEAT, W1O, B1O, W2O, B2O, N } = cfg;
	const f = o.f;
	const ns = new Int32Array(N * C);
	const perc: number[] = new Array(PERC);
	const h: number[] = new Array(HD);
	const w1row: number[] = new Array(PERC);
	const w2row: number[] = new Array(HD);
	for (let y = 1; y < SH - 1; y++)
		for (let x = 1; x < SW - 1; x++) {
			const i = y * SW + x;
			const r = i + 1, l = i - 1, u = i - SW, d = i + SW;
			// perceive (fixed): identity, gx=(sr-sl)>>1, gy=(sd-su)>>1, lap=sr+sl+su+sd-4·self
			for (let ch = 0; ch < C; ch++) {
				const b = ch * FEAT;
				const self = s[i * C + ch];
				const sr = s[r * C + ch], sl = s[l * C + ch], su = s[u * C + ch], sd = s[d * C + ch];
				perc[b] = self;
				perc[b + 1] = (sr - sl) >> 1; // ·0.5 (arithmetic shift, floor — a Z80 SRA)
				perc[b + 2] = (sd - su) >> 1;
				perc[b + 3] = sr + sl + su + sd - 4 * self; // exact int
			}
			for (let hh = 0; hh < HD; hh++) {
				const base = W1O + hh * PERC;
				for (let k = 0; k < PERC; k++) w1row[k] = parQ[base + k];
				h[hh] = reluQ(dotQ(w1row, perc, parQ[B1O + hh], f, o.ov));
			}
			for (let c = 0; c < C; c++) {
				const base = W2O + c * HD;
				for (let hh = 0; hh < HD; hh++) w2row[hh] = parQ[base + hh];
				const dl = dotQ(w2row, h, parQ[B2O + c], f, o.ov);
				ns[i * C + c] = o.tanh(s[i * C + c] + dl); // tanh(state + dl), residual inside
			}
		}
	// input clamp (ch0) every step
	for (let k = 0; k < exp.inputCells.length; k++) ns[exp.inputCells[k] * C + 0] = toQ(inputs[k], f);
	return ns;
}

function seedFixed(f: Fmt): (inputs: number[]) => Int32Array {
	return (inputs: number[]) => {
		const s0 = seedGrid(cfg, exp, inputs); // f64 seed (0s and 1s + clamped inputs)
		const q = new Int32Array(cfg.N * cfg.C);
		for (let i = 0; i < q.length; i++) q[i] = toQ(s0[i], f);
		return q;
	};
}

function rolloutFixed(f: Fmt, tanh: (qx: number) => number, inputs: number[]): { out: number[]; ov: OverflowStats } {
	const ov = newOverflow();
	const parQ = quantParams(f);
	const seed = seedFixed(f);
	let s = seed(inputs);
	const outTraj: number[] = [];
	for (let t = 0; t < STEPS; t++) {
		s = stepFixed(cfg, parQ, s, exp, inputs, { f, tanh, ov });
		outTraj.push(fromQ(s[exp.outputCells[0] * cfg.C + 0], f));
	}
	return { out: outTraj, ov };
}

// ---- f64 reference --------------------------------------------------------

function rolloutF64(inputs: number[]): number[] {
	const states = forward(cfg, par, exp, inputs, { steps: STEPS });
	return states.slice(1).map((s) => readOutputs(cfg, s, exp)[0]);
}

// measure pre-activation and dl magnitude ranges in f64 → sets integer-bit budget
function magnitudeStats(): { maxPre: number; maxDl: number; maxState: number } {
	const { SW, SH, C, HD, PERC, W1O, B1O, W2O, B2O } = cfg;
	let maxPre = 0, maxDl = 0, maxState = 0;
	for (const cs of exp.cases) {
		const states = forward(cfg, par, exp, cs.in, { steps: STEPS });
		const percBuf = new Float64Array(PERC);
		for (const s of states) {
			for (let j = 0; j < s.length; j++) maxState = Math.max(maxState, Math.abs(s[j]));
			for (let y = 1; y < SH - 1; y++)
				for (let x = 1; x < SW - 1; x++) {
					const i = y * SW + x;
					perceive(cfg, s, i, percBuf);
					const hh2: number[] = [];
					for (let hh = 0; hh < HD; hh++) {
						let a = par[B1O + hh]; const base = W1O + hh * PERC;
						for (let k = 0; k < PERC; k++) a += par[base + k] * percBuf[k];
						maxPre = Math.max(maxPre, Math.abs(a));
						hh2.push(a > 0 ? a : 0);
					}
					for (let c = 0; c < C; c++) {
						let dl = par[B2O + c]; const base = W2O + c * HD;
						for (let hh = 0; hh < HD; hh++) dl += par[base + hh] * hh2[hh];
						maxDl = Math.max(maxDl, Math.abs(dl));
					}
				}
		}
	}
	return { maxPre, maxDl, maxState };
}

// ---- run ------------------------------------------------------------------

function classify(v: number): number { return v > 0.5 ? 1 : 0; }
function truthOk(outs: number[][]): boolean {
	return exp.cases.every((cs, i) => classify(outs[i][outs[i].length - 1]) === cs.out[0]);
}

console.log('=== Phase 1: does the E1 gate survive fixed-point? (bit-faithful Z80 emulation) ===\n');

// f64 reference truth table
const f64outs = exp.cases.map((cs) => rolloutF64(cs.in));
console.log('f64 reference (the trained rule):');
console.log('  case      out(f64)   class  expected');
exp.cases.forEach((cs, i) => {
	const o = f64outs[i][f64outs[i].length - 1];
	console.log(`  [${cs.in.join(',')}] → ${o.toFixed(4).padStart(8)}    ${classify(o)}      ${cs.out[0]}`);
});
console.log(`  f64 truth table: ${truthOk(f64outs) ? 'PASS' : 'FAIL'}\n`);

const mag = magnitudeStats();
const intBitsPre = Math.ceil(Math.log2(mag.maxPre)) + 1; // +1 sign
console.log('magnitude budget (over all cases, all cells, all steps):');
console.log(`  max|state| = ${mag.maxState.toFixed(3)}   max|pre-act| = ${mag.maxPre.toFixed(3)}   max|dl| = ${mag.maxDl.toFixed(3)}`);
console.log(`  → need ≥ ${intBitsPre} integer bits (incl. sign) to hold pre-activations without saturating\n`);

const FORMATS: Fmt[] = [
	fmt(16, 8),  // Q8.8   — expA's format (1 reg pair)
	fmt(16, 12), // Q4.12
	fmt(24, 16), // Q8.16
	fmt(32, 16), // Q16.16 — expB's format (2 reg pairs)
	fmt(32, 24)  // Q8.24
];

console.log('fixed-point sweep (ideal tanh — isolates the MAC quantization):');
console.log('  format    truth   max|Δout vs f64|   saturations   final outputs');
let winner: Fmt | null = null;
for (const f of FORMATS) {
	const outs = exp.cases.map((cs) => rolloutFixed(f, (qx) => tanhIdeal(qx, f), cs.in));
	const trajs = outs.map((o) => o.out);
	let maxErr = 0;
	for (let i = 0; i < trajs.length; i++)
		for (let t = 0; t < STEPS; t++) maxErr = Math.max(maxErr, Math.abs(trajs[i][t] - f64outs[i][t]));
	const sat = outs.reduce((a, o) => a + o.ov.saturations, 0);
	const ok = truthOk(trajs);
	const finals = trajs.map((tr) => tr[tr.length - 1].toFixed(3)).join(' ');
	if (ok && !winner) winner = f;
	console.log(`  ${fmtName(f).padEnd(8)}  ${ok ? 'PASS' : 'FAIL'}    ${maxErr.toExponential(2).padStart(10)}         ${String(sat).padStart(6)}      [${finals}]`);
}

// tanh TABLE (what the Z80 actually uses) for the winning format
if (winner) {
	console.log(`\nwinner: ${fmtName(winner)}. Re-run with a real tanh LOOKUP TABLE (Z80 uses a table, not Math.tanh):`);
	const XR = 4; // tanh(±4) ≈ ±0.9993
	for (const N of [64, 128, 256, 512]) {
		const table = buildTanhTable(winner, N, XR);
		const outs = exp.cases.map((cs) => rolloutFixed(winner!, (qx) => tanhTable(qx, winner!, table, N, XR), cs.in));
		const trajs = outs.map((o) => o.out);
		let maxErr = 0;
		for (let i = 0; i < trajs.length; i++)
			for (let t = 0; t < STEPS; t++) maxErr = Math.max(maxErr, Math.abs(trajs[i][t] - f64outs[i][t]));
		const ok = truthOk(trajs);
		const finals = trajs.map((tr) => tr[tr.length - 1].toFixed(3)).join(' ');
		console.log(`  ${String(N).padStart(3)}-entry table:  ${ok ? 'PASS' : 'FAIL'}   max|Δout|=${maxErr.toExponential(2)}   [${finals}]`);
	}
	console.log(`\n→ PHASE 1 RESULT: the trained XOR gate computes correctly in ${fmtName(winner)} fixed-point.`);
	console.log('  The paradigm quantizes. Build the Z80 datapath in this format (Phase 2).');
} else {
	console.log('\n→ PHASE 1 RESULT: NO tested format preserved the truth table. Investigate before assembly.');
}
