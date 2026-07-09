// EXPERIMENT H — DEVELOPMENTAL ARITHMETIC (the adder). Scaling developmental
// computation past a single gate: a 1-bit FULL ADDER (3 inputs a,b,cin → 2
// outputs sum, carry) grown by one CA rule, then (later stages) made
// long-horizon-stable, input-reactive, self-repairing, and seed-growable.
//
// This file starts with the hard part: COMPUTE the adder. sum = a⊕b⊕cin
// (3-input parity), carry = majority(a,b,cin). Two spatially-separated outputs,
// 8 input cases — substantially harder than the XOR gate (E1).
//
//   npx tsx src/lib/morph/dev/expH.ts
//   ITERS=… PARAMS_OUT=… HVIZ=… npx tsx …/expH.ts

import { writeFileSync } from 'node:fs';

const W = 11, H = 11, N = W * H;
const C = 16, FEAT = 4, PERC = FEAT * C, HD = 64;
const W1O = 0, B1O = HD * PERC, W2O = B1O + HD, B2O = W2O + C * HD, P = B2O + C; // 5200

const iy = H >> 1, IN_COL = 2, OUT_COL = W - 3;
const inputCells = [(iy - 1) * W + IN_COL, iy * W + IN_COL, (iy + 1) * W + IN_COL]; // a, b, cin (adjacent rows)
// output cells are retargeted during the distance curriculum; fixed after.
let outCells = [(iy - 1) * W + OUT_COL, (iy + 1) * W + OUT_COL]; // sum, carry

interface Case { in: number[]; out: number[]; }
const CASES: Case[] = [];
for (let a = 0; a < 2; a++) for (let b = 0; b < 2; b++) for (let cin = 0; cin < 2; cin++) {
	const s = a ^ b ^ cin, cout = (a + b + cin) >= 2 ? 1 : 0;
	CASES.push({ in: [a, b, cin], out: [s, cout] });
}

function mulberry32(seed: number): () => number {
	let a = seed >>> 0;
	return () => {
		a |= 0; a = (a + 0x6d2b79f5) | 0;
		let t = Math.imul(a ^ (a >>> 15), 1 | a);
		t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
}

function seedGrid(inputs: number[]): Float64Array {
	const s = new Float64Array(N * C);
	for (let y = 1; y < H - 1; y++) for (let x = 1; x < W - 1; x++)
		for (let c = 1; c < C; c++) s[(y * W + x) * C + c] = 1;
	clampInputs(s, inputs);
	return s;
}
function clampInputs(f: Float64Array, inputs: number[]): void {
	for (let k = 0; k < inputCells.length; k++) f[inputCells[k] * C + 0] = inputs[k];
}

function perceive(s: Float64Array, i: number, out: Float64Array): void {
	const r = i + 1, l = i - 1, u = i - W, d = i + W;
	for (let ch = 0; ch < C; ch++) {
		const b = ch * FEAT, self = s[i * C + ch];
		const sr = s[r * C + ch], sl = s[l * C + ch], su = s[u * C + ch], sd = s[d * C + ch];
		out[b] = self;
		out[b + 1] = (sr - sl) * 0.5;
		out[b + 2] = (sd - su) * 0.5;
		out[b + 3] = sr + sl + su + sd - 4 * self;
	}
}

const T = Number(process.env.T ?? 30); // developmental steps

function forward(par: Float64Array, inputs: number[]): Float64Array[] {
	const s0 = seedGrid(inputs);
	const states: Float64Array[] = [s0];
	let s = s0;
	const perc = new Float64Array(PERC), h = new Float64Array(HD);
	for (let t = 0; t < T; t++) {
		const ns = new Float64Array(N * C);
		for (let y = 1; y < H - 1; y++)
			for (let x = 1; x < W - 1; x++) {
				const i = y * W + x;
				perceive(s, i, perc);
				for (let hh = 0; hh < HD; hh++) {
					let a = par[B1O + hh]; const base = W1O + hh * PERC;
					for (let k = 0; k < PERC; k++) a += par[base + k] * perc[k];
					h[hh] = a > 0 ? a : 0;
				}
				for (let c = 0; c < C; c++) {
					let dl = par[B2O + c]; const base = W2O + c * HD;
					for (let hh = 0; hh < HD; hh++) dl += par[base + hh] * h[hh];
					ns[i * C + c] = Math.tanh(s[i * C + c] + dl);
				}
			}
		clampInputs(ns, inputs);
		states.push(ns);
		s = ns;
	}
	return states;
}

/** Loss + gradient: readout both output cells at step T, over all 8 cases. */
function lossAndGrad(par: Float64Array): { L: number; grad: Float64Array; outs: number[][] } {
	const grad = new Float64Array(P);
	let L = 0; const outs: number[][] = [];
	const norm = CASES.length * outCells.length;
	const perc = new Float64Array(PERC), pre1 = new Float64Array(HD), hbuf = new Float64Array(HD), gh = new Float64Array(HD), gperc = new Float64Array(PERC);
	for (const cse of CASES) {
		const states = forward(par, cse.in);
		const sT = states[T];
		outs.push(outCells.map((cell) => sT[cell * C + 0]));
		const gsT = new Float64Array(N * C);
		for (let k = 0; k < outCells.length; k++) {
			const diff = sT[outCells[k] * C + 0] - cse.out[k];
			L += (diff * diff) / norm;
			gsT[outCells[k] * C + 0] = (2 * diff) / norm;
		}
		let gs = gsT;
		for (let t = T - 1; t >= 0; t--) {
			for (const ic of inputCells) gs[ic * C + 0] = 0;
			const s = states[t], sp = states[t + 1];
			const gsPrev = new Float64Array(N * C);
			for (let y = 1; y < H - 1; y++)
				for (let x = 1; x < W - 1; x++) {
					const i = y * W + x;
					perceive(s, i, perc);
					for (let hh = 0; hh < HD; hh++) {
						let a = par[B1O + hh]; const base = W1O + hh * PERC;
						for (let k = 0; k < PERC; k++) a += par[base + k] * perc[k];
						pre1[hh] = a; hbuf[hh] = a > 0 ? a : 0;
					}
					gh.fill(0);
					for (let c = 0; c < C; c++) {
						const spv = sp[i * C + c];
						const gp = gs[i * C + c] * (1 - spv * spv);
						gsPrev[i * C + c] += gp;
						grad[B2O + c] += gp;
						const base = W2O + c * HD;
						for (let hh = 0; hh < HD; hh++) { grad[base + hh] += gp * hbuf[hh]; gh[hh] += par[base + hh] * gp; }
					}
					gperc.fill(0);
					for (let hh = 0; hh < HD; hh++) {
						let g = gh[hh]; if (pre1[hh] <= 0) g = 0;
						grad[B1O + hh] += g;
						const base = W1O + hh * PERC;
						for (let k = 0; k < PERC; k++) { grad[base + k] += g * perc[k]; gperc[k] += par[base + k] * g; }
					}
					const r = i + 1, l = i - 1, u = i - W, d = i + W;
					for (let ch = 0; ch < C; ch++) {
						const bb = ch * FEAT, gId = gperc[bb], gGx = gperc[bb + 1], gGy = gperc[bb + 2], gLap = gperc[bb + 3];
						gsPrev[i * C + ch] += gId - 4 * gLap;
						gsPrev[r * C + ch] += 0.5 * gGx + gLap;
						gsPrev[l * C + ch] += -0.5 * gGx + gLap;
						gsPrev[d * C + ch] += 0.5 * gGy + gLap;
						gsPrev[u * C + ch] += -0.5 * gGy + gLap;
					}
				}
			gs = gsPrev;
		}
	}
	return { L, grad, outs };
}

function gradientCheck(): boolean {
	const rng = mulberry32(3);
	const par = new Float64Array(P).map(() => (rng() - 0.5) * 0.1);
	for (let j = W2O; j < P; j++) par[j] = 0; // near-identity rule => unsaturated field => clean finite-diff
	const { grad } = lossAndGrad(par);
	const eps = 1e-4;
	let maxRel = 0;
	for (const j of [10, 900, B1O + 3, W2O + 7, B2O + 1]) {
		const pp = par.slice(); pp[j] += eps;
		const pm = par.slice(); pm[j] -= eps;
		const fd = (lossAndGrad(pp).L - lossAndGrad(pm).L) / (2 * eps);
		maxRel = Math.max(maxRel, Math.abs(grad[j] - fd) / (Math.abs(fd) + 1e-8));
	}
	console.log(`  gradient check: max rel err ${maxRel.toExponential(2)} -> ${maxRel < 0.02 ? 'PASS' : 'FAIL'}`);
	return maxRel < 0.02;
}

function train(iters: number, init: Float64Array | undefined, seed = 7): { par: Float64Array; L: number } {
	let par: Float64Array;
	if (init) par = init.slice();
	else {
		const rng = mulberry32(seed);
		par = new Float64Array(P);
		for (let j = 0; j < P; j++) par[j] = (rng() - 0.5) * 0.12;
		for (let j = W2O; j < P; j++) par[j] *= 0.5;
	}
	const warm = init !== undefined;
	const lrHi = warm ? 0.003 : 0.008, lrLo = warm ? 0.0006 : 0.003;
	const m = new Float64Array(P), v = new Float64Array(P), b1 = 0.9, b2 = 0.999;
	let bestLoss = Infinity, bestPar = par.slice();
	for (let it = 1; it <= iters; it++) {
		const cos = 0.5 * (1 + Math.cos(Math.PI * (it / iters)));
		const lr = Math.min(1, it / 20) * (lrLo + (lrHi - lrLo) * cos);
		const { L, grad } = lossAndGrad(par);
		let gn = 0; for (let j = 0; j < P; j++) gn += grad[j] * grad[j]; gn = Math.sqrt(gn);
		const clip = gn > 1 ? 1 / gn : 1;
		if (L < bestLoss) { bestLoss = L; bestPar = par.slice(); }
		const c1 = 1 - Math.pow(b1, it), c2 = 1 - Math.pow(b2, it);
		for (let j = 0; j < P; j++) {
			const g = grad[j] * clip + 2e-5 * par[j];
			m[j] = b1 * m[j] + (1 - b1) * g;
			v[j] = b2 * v[j] + (1 - b2) * g * g;
			par[j] -= (lr * (m[j] / c1)) / (Math.sqrt(v[j] / c2) + 1e-8);
		}
		if (it % 200 === 0 || it === 1) console.log(`      iter ${String(it).padStart(4)}  loss ${L.toFixed(5)}  (best ${bestLoss.toFixed(5)})`);
	}
	return { par: bestPar, L: bestLoss };
}

/** Distance curriculum: outputs start near the inputs, extend one column at a time. */
function computeCurriculum(iters: number): Float64Array {
	console.log('  COMPUTE — 1-bit full adder (distance curriculum)');
	let par: Float64Array | undefined;
	for (let dist = 1; dist <= OUT_COL - IN_COL; dist++) {
		outCells = [(iy - 1) * W + (IN_COL + dist), (iy + 1) * W + (IN_COL + dist)];
		const restarts = dist === 1 ? 6 : 1;
		const si = dist === 1 ? Math.min(iters, 400) : iters;
		let best: { par: Float64Array; L: number } | null = null;
		for (let s = 0; s < restarts; s++) { const r = train(si, par, 7 + s * 101); if (!best || r.L < best.L) best = r; }
		par = best!.par;
		const { outs } = lossAndGrad(par);
		const acc = CASES.filter((c, i) => c.out.every((t, k) => Math.abs(outs[i][k] - t) < 0.3)).length;
		console.log(`    d=${dist}: loss ${best!.L.toFixed(4)}  acc ${acc}/${CASES.length}`);
	}
	outCells = [(iy - 1) * W + OUT_COL, (iy + 1) * W + OUT_COL];
	return par!;
}

function report(par: Float64Array): void {
	const { L, outs } = lossAndGrad(par);
	let acc = 0;
	console.log(`\n  FINAL full adder (loss ${L.toFixed(4)}):  a b cin -> sum carry`);
	CASES.forEach((c, i) => {
		const ok = c.out.every((t, k) => Math.abs(outs[i][k] - t) < 0.3);
		if (ok) acc++;
		console.log(`    ${c.in.join(' ')}  ->  ${outs[i].map((o) => o.toFixed(2)).join(' ')}   (want ${c.out.join(' ')})  ${ok ? '✓' : '✗'}`);
	});
	console.log(`  accuracy ${acc}/${CASES.length} cases`);
}

function main() {
	console.log(`developmental arithmetic (1-bit full adder): ${W}x${H}, C=${C}, HD=${HD}, ${P} params, T=${T}`);
	if (!gradientCheck()) { console.error('FAIL: gradient wrong'); process.exit(1); }
	const iters = Number(process.env.ITERS ?? 900);
	const par = computeCurriculum(iters);
	report(par);
	if (process.env.PARAMS_OUT) { writeFileSync(process.env.PARAMS_OUT, JSON.stringify(Array.from(par))); console.log(`  params saved to ${process.env.PARAMS_OUT}`); }
}

main();
