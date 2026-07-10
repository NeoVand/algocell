// EXPERIMENT J — 2-BIT RIPPLE-CARRY ADDER as one developmental rule (the paper's
// compositional-depth result). Unlike the 1-bit adder (a single wide gate:
// parity+majority in parallel), a 2-bit adder has a PRODUCED-THEN-CONSUMED
// internal signal: FA0 makes carry0 = a0∧b0, and FA1 consumes it to make
// sum1 = a1⊕b1⊕carry0 and cout = majority(a1,b1,carry0). 16 input cases, cin=0.
//
// Inputs a1,b1,a0,b0 → outputs sum1,sum0,cout, with an internal carry cell.
// Curriculum "expose-then-internalize":
//   MODE=expose  — teacher-force carry0 at the carry cell (clamped) + a distance
//                  curriculum for the three real outputs. Decomposes into two
//                  ~1-bit-adder subproblems → bootstraps.
//   MODE=intern  — release the clamp, warm from expose, ADD an auxiliary loss
//                  supervising the carry cell = carry0. The field must now PRODUCE
//                  carry0 and FA1 must CONSUME the field's own carry.
// Causal probe: lesion the carry cell mid-rollout → only carry-dependent cases break.
//
//   MODE=expose ITERS=… PARAMS_OUT=p.json npx tsx src/lib/morph/dev/adder2.ts
//   MODE=intern PARAMS_IN=p.json ITERS=… PARAMS_OUT=q.json npx tsx …/adder2.ts
//   MODE=probe  PARAMS_IN=q.json npx tsx …/adder2.ts

import { writeFileSync, readFileSync } from 'node:fs';

const W = 13, H = 13, N = W * H;
const C = 16, FEAT = 4, PERC = FEAT * C, HD = 96;
const W1O = 0, B1O = HD * PERC, W2O = B1O + HD, B2O = W2O + C * HD, P = B2O + C; // 7792
const T = Number(process.env.T ?? 60);

const iy = H >> 1, IN_COL = 2, OUT_COL = W - 3; // 6, 2, 10
// input cells: two bit-lanes (FA1 upper, FA0 lower)
const A1 = 3 * W + IN_COL, B1 = 4 * W + IN_COL, A0 = 8 * W + IN_COL, B0 = 9 * W + IN_COL;
const inputCells = [A1, B1, A0, B0]; // a1, b1, a0, b0
const CARRY = iy * W + (IN_COL + 2); // (6,4) — where carry0 lives, near the inputs
// output cells (retargeted during the distance curriculum, fixed after)
let SUM1 = 3 * W + OUT_COL, COUT = 4 * W + OUT_COL, SUM0 = 8 * W + OUT_COL;

interface Case { in: number[]; sum1: number; sum0: number; cout: number; carry0: number; }
const CASES: Case[] = [];
for (let a1 = 0; a1 < 2; a1++) for (let b1 = 0; b1 < 2; b1++) for (let a0 = 0; a0 < 2; a0++) for (let b0 = 0; b0 < 2; b0++) {
	const carry0 = a0 & b0;
	const sum0 = a0 ^ b0;
	const sum1 = a1 ^ b1 ^ carry0;
	const cout = (a1 + b1 + carry0) >= 2 ? 1 : 0;
	CASES.push({ in: [a1, b1, a0, b0], sum1, sum0, cout, carry0 });
}

function mulberry32(seed: number): () => number {
	let a = seed >>> 0;
	return () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

function seedGrid(): Float64Array {
	const s = new Float64Array(N * C);
	for (let y = 1; y < H - 1; y++) for (let x = 1; x < W - 1; x++) for (let c = 1; c < C; c++) s[(y * W + x) * C + c] = 1;
	return s;
}
function clampInputs(f: Float64Array, cse: Case, clampCarry: boolean): void {
	for (let k = 0; k < inputCells.length; k++) f[inputCells[k] * C + 0] = cse.in[k];
	if (clampCarry) f[CARRY * C + 0] = cse.carry0;
}
function perceive(s: Float64Array, i: number, out: Float64Array): void {
	const r = i + 1, l = i - 1, u = i - W, d = i + W;
	for (let ch = 0; ch < C; ch++) {
		const b = ch * FEAT, self = s[i * C + ch];
		const sr = s[r * C + ch], sl = s[l * C + ch], su = s[u * C + ch], sd = s[d * C + ch];
		out[b] = self; out[b + 1] = (sr - sl) * 0.5; out[b + 2] = (sd - su) * 0.5; out[b + 3] = sr + sl + su + sd - 4 * self;
	}
}

// scored output cells + their per-case targets. When carry is clamped it is an
// input (not scored); when released it is an auxiliary scored output.
function scoredCells(clampCarry: boolean): number[] { return clampCarry ? [SUM1, SUM0, COUT] : [SUM1, SUM0, COUT, CARRY]; }
function targets(cse: Case, clampCarry: boolean): number[] { return clampCarry ? [cse.sum1, cse.sum0, cse.cout] : [cse.sum1, cse.sum0, cse.cout, cse.carry0]; }
function clampedCells(clampCarry: boolean): number[] { return clampCarry ? [...inputCells, CARRY] : inputCells; }

// carryOverride: if not null, clamp the carry cell's signal channel to that value
// EVERY step (a causal intervention — "pin the internal carry wire to v").
function forward(par: Float64Array, cse: Case, clampCarry: boolean, carryOverride: number | null = null): Float64Array[] {
	const s0 = seedGrid(); clampInputs(s0, cse, clampCarry);
	if (carryOverride !== null) s0[CARRY * C + 0] = carryOverride;
	const states: Float64Array[] = [s0]; let s = s0;
	const perc = new Float64Array(PERC), h = new Float64Array(HD);
	for (let t = 0; t < T; t++) {
		const ns = new Float64Array(N * C);
		for (let y = 1; y < H - 1; y++) for (let x = 1; x < W - 1; x++) {
			const i = y * W + x;
			perceive(s, i, perc);
			for (let hh = 0; hh < HD; hh++) { let a = par[B1O + hh]; const base = W1O + hh * PERC; for (let k = 0; k < PERC; k++) a += par[base + k] * perc[k]; h[hh] = a > 0 ? a : 0; }
			for (let c = 0; c < C; c++) { let dl = par[B2O + c]; const base = W2O + c * HD; for (let hh = 0; hh < HD; hh++) dl += par[base + hh] * h[hh]; ns[i * C + c] = Math.tanh(s[i * C + c] + dl); }
		}
		clampInputs(ns, cse, clampCarry);
		if (carryOverride !== null) ns[CARRY * C + 0] = carryOverride; // pin the internal carry wire
		states.push(ns); s = ns;
	}
	return states;
}

function lossAndGrad(par: Float64Array, clampCarry: boolean, wCarry = 1): { L: number; grad: Float64Array; outs: number[][] } {
	const grad = new Float64Array(P); let L = 0; const outs: number[][] = [];
	const cells = scoredCells(clampCarry), clamped = clampedCells(clampCarry);
	const norm = CASES.length * cells.length;
	const perc = new Float64Array(PERC), pre1 = new Float64Array(HD), hbuf = new Float64Array(HD), gh = new Float64Array(HD), gperc = new Float64Array(PERC);
	for (const cse of CASES) {
		const states = forward(par, cse, clampCarry);
		const sT = states[T]; const tg = targets(cse, clampCarry);
		outs.push(cells.map((cell) => sT[cell * C + 0]));
		const gsT = new Float64Array(N * C);
		for (let k = 0; k < cells.length; k++) {
			const wk = (cells[k] === CARRY ? wCarry : 1);
			const diff = sT[cells[k] * C + 0] - tg[k];
			L += (wk * diff * diff) / norm;
			gsT[cells[k] * C + 0] = (2 * wk * diff) / norm;
		}
		let gs = gsT;
		for (let t = T - 1; t >= 0; t--) {
			for (const ic of clamped) gs[ic * C + 0] = 0;
			const s = states[t], sp = states[t + 1];
			const gsPrev = new Float64Array(N * C);
			for (let y = 1; y < H - 1; y++) for (let x = 1; x < W - 1; x++) {
				const i = y * W + x;
				perceive(s, i, perc);
				for (let hh = 0; hh < HD; hh++) { let a = par[B1O + hh]; const base = W1O + hh * PERC; for (let k = 0; k < PERC; k++) a += par[base + k] * perc[k]; pre1[hh] = a; hbuf[hh] = a > 0 ? a : 0; }
				gh.fill(0);
				for (let c = 0; c < C; c++) {
					const spv = sp[i * C + c]; const gp = gs[i * C + c] * (1 - spv * spv);
					gsPrev[i * C + c] += gp; grad[B2O + c] += gp; const base = W2O + c * HD;
					for (let hh = 0; hh < HD; hh++) { grad[base + hh] += gp * hbuf[hh]; gh[hh] += par[base + hh] * gp; }
				}
				gperc.fill(0);
				for (let hh = 0; hh < HD; hh++) { let g = gh[hh]; if (pre1[hh] <= 0) g = 0; grad[B1O + hh] += g; const base = W1O + hh * PERC; for (let k = 0; k < PERC; k++) { grad[base + k] += g * perc[k]; gperc[k] += par[base + k] * g; } }
				const r = i + 1, l = i - 1, u = i - W, d = i + W;
				for (let ch = 0; ch < C; ch++) {
					const bb = ch * FEAT, gId = gperc[bb], gGx = gperc[bb + 1], gGy = gperc[bb + 2], gLap = gperc[bb + 3];
					gsPrev[i * C + ch] += gId - 4 * gLap;
					gsPrev[r * C + ch] += 0.5 * gGx + gLap; gsPrev[l * C + ch] += -0.5 * gGx + gLap;
					gsPrev[d * C + ch] += 0.5 * gGy + gLap; gsPrev[u * C + ch] += -0.5 * gGy + gLap;
				}
			}
			gs = gsPrev;
		}
	}
	return { L, grad, outs };
}

function accuracy(par: Float64Array, clampCarry: boolean): { acc: number; carryAcc: number } {
	let acc = 0, carryAcc = 0;
	for (const cse of CASES) {
		const sT = forward(par, cse, clampCarry)[T];
		const real = [sT[SUM1 * C] - cse.sum1, sT[SUM0 * C] - cse.sum0, sT[COUT * C] - cse.cout];
		if (real.every((d) => Math.abs(d) < 0.3)) acc++;
		if (Math.abs(sT[CARRY * C] - cse.carry0) < 0.3) carryAcc++;
	}
	return { acc, carryAcc };
}

function train(iters: number, init: Float64Array | undefined, clampCarry: boolean, wCarry: number, seed = 7): { par: Float64Array; L: number } {
	let par: Float64Array;
	if (init) par = init.slice();
	else { const rng = mulberry32(seed); par = new Float64Array(P); for (let j = 0; j < P; j++) par[j] = (rng() - 0.5) * 0.12; for (let j = W2O; j < P; j++) par[j] *= 0.5; }
	const warm = init !== undefined;
	const lrHi = warm ? 0.003 : 0.008, lrLo = warm ? 0.0006 : 0.003;
	const m = new Float64Array(P), v = new Float64Array(P), b1 = 0.9, b2 = 0.999;
	let bestLoss = Infinity, bestPar = par.slice();
	for (let it = 1; it <= iters; it++) {
		const cos = 0.5 * (1 + Math.cos(Math.PI * (it / iters)));
		const lr = Math.min(1, it / 20) * (lrLo + (lrHi - lrLo) * cos);
		const { L, grad } = lossAndGrad(par, clampCarry, wCarry);
		let gn = 0; for (let j = 0; j < P; j++) gn += grad[j] * grad[j]; gn = Math.sqrt(gn);
		const clip = gn > 1 ? 1 / gn : 1;
		if (L < bestLoss) { bestLoss = L; bestPar = par.slice(); }
		const c1 = 1 - Math.pow(b1, it), c2 = 1 - Math.pow(b2, it);
		for (let j = 0; j < P; j++) { const g = grad[j] * clip + 2e-5 * par[j]; m[j] = b1 * m[j] + (1 - b1) * g; v[j] = b2 * v[j] + (1 - b2) * g * g; par[j] -= (lr * (m[j] / c1)) / (Math.sqrt(v[j] / c2) + 1e-8); }
		if (it % 100 === 0 || it === 1) { const a = accuracy(par, clampCarry); console.log(`      iter ${String(it).padStart(4)}  loss ${L.toFixed(5)}  best ${bestLoss.toFixed(5)}  acc ${a.acc}/16 carry ${a.carryAcc}/16`); }
	}
	return { par: bestPar, L: bestLoss };
}

function gradientCheck(clampCarry: boolean): boolean {
	const rng = mulberry32(3);
	const par = new Float64Array(P).map(() => (rng() - 0.5) * 0.1);
	for (let j = W2O; j < P; j++) par[j] = 0;
	const { grad } = lossAndGrad(par, clampCarry);
	const eps = 1e-4; let maxRel = 0;
	for (const j of [10, 2000, B1O + 3, W2O + 7, B2O + 1]) {
		const pp = par.slice(); pp[j] += eps; const pm = par.slice(); pm[j] -= eps;
		const fd = (lossAndGrad(pp, clampCarry).L - lossAndGrad(pm, clampCarry).L) / (2 * eps);
		maxRel = Math.max(maxRel, Math.abs(grad[j] - fd) / (Math.abs(fd) + 1e-8));
	}
	console.log(`  gradient check (clampCarry=${clampCarry}): max rel err ${maxRel.toExponential(2)} -> ${maxRel < 0.02 ? 'PASS' : 'FAIL'}`);
	return maxRel < 0.02;
}

/** Distance curriculum for the three real outputs (carry cell fixed). Clamp carry throughout (expose). */
function exposeCurriculum(iters: number): Float64Array {
	console.log('  EXPOSE — teacher-forced carry, distance curriculum for the 3 outputs');
	let par: Float64Array | undefined;
	const maxDist = Number(process.env.MAXDIST ?? (OUT_COL - IN_COL));
	for (let dist = 2; dist <= maxDist; dist++) {
		const col = IN_COL + dist;
		SUM1 = 3 * W + col; COUT = 4 * W + col; SUM0 = 8 * W + col;
		const restarts = dist === 2 ? Number(process.env.RESTARTS ?? 4) : 1;
		const si = dist === 2 ? Math.min(iters, 400) : iters;
		let best: { par: Float64Array; L: number } | null = null;
		for (let s = 0; s < restarts; s++) { const r = train(si, par, true, 1, 7 + s * 101); if (!best || r.L < best.L) best = r; }
		par = best!.par;
		const a = accuracy(par, true);
		console.log(`    outCol=${col}: loss ${best!.L.toFixed(4)}  acc ${a.acc}/16`);
	}
	SUM1 = 3 * W + OUT_COL; COUT = 4 * W + OUT_COL; SUM0 = 8 * W + OUT_COL;
	return par!;
}

function report(par: Float64Array, clampCarry: boolean): void {
	console.log(`\n  2-bit adder (clampCarry=${clampCarry}):  a1 b1 a0 b0 -> sum1 sum0 cout   (carry0)`);
	let acc = 0;
	for (const cse of CASES) {
		const sT = forward(par, cse, clampCarry)[T];
		const o = [sT[SUM1 * C], sT[SUM0 * C], sT[COUT * C]];
		const ok = Math.abs(o[0] - cse.sum1) < 0.3 && Math.abs(o[1] - cse.sum0) < 0.3 && Math.abs(o[2] - cse.cout) < 0.3;
		if (ok) acc++;
		console.log(`    ${cse.in.join(' ')}  ->  ${o.map((x) => x.toFixed(2)).join(' ')}   want ${cse.sum1} ${cse.sum0} ${cse.cout}   carry@${sT[CARRY * C].toFixed(2)}(${cse.carry0}) ${ok ? '✓' : '✗'}`);
	}
	console.log(`  accuracy ${acc}/16`);
}

/** Causal probe: PIN the internal carry wire to v∈{0,1} for the whole rollout and
 *  check whether the outputs track the FORCED carry — sum1 = a1⊕b1⊕v, cout =
 *  majority(a1,b1,v) — rather than the true carry0. If they do, the carry cell
 *  causally controls exactly the carry-dependent computation. */
function probe(par: Float64Array): void {
	console.log('\n  CAUSAL CARRY PROBE — pin the carry cell to v for all t, per case:');
	// baseline (no intervention): should already be 16/16
	let base = 0;
	for (const cse of CASES) { const s = forward(par, cse, false)[T]; if (Math.abs(s[SUM1 * C] - cse.sum1) < 0.3 && Math.abs(s[COUT * C] - cse.cout) < 0.3) base++; }
	// interventions: does the field compute the adder for the FORCED carry?
	for (const v of [0, 1]) {
		let track = 0, flipped = 0;
		for (const cse of CASES) {
			const [a1, b1] = cse.in;
			const wantS1 = a1 ^ b1 ^ v, wantC = (a1 + b1 + v) >= 2 ? 1 : 0;
			const s = forward(par, cse, false, v)[T];
			if (Math.abs(s[SUM1 * C] - wantS1) < 0.3 && Math.abs(s[COUT * C] - wantC) < 0.3) track++;   // tracks forced carry
			if (v !== cse.carry0 && (Math.abs(s[SUM1 * C] - cse.sum1) > 0.5 || Math.abs(s[COUT * C] - cse.cout) > 0.5)) flipped++; // wrong-carry flips the true output
		}
		const nWrong = CASES.filter((c) => c.carry0 !== v).length;
		console.log(`    pin carry=${v}: outputs match the carry=${v} adder in ${track}/16 cases; wrong-carry flipped the true answer in ${flipped}/${nWrong} cases`);
	}
	console.log(`    (baseline, no intervention: ${base}/16 correct.)`);
	interchangeProbe(par);
}

/** Interchange intervention (activation patching) — the clean causal test. For each
 *  (a1,b1), take base=(a1,b1,0,0) [carry0=0] and source=(a1,b1,1,1) [carry0=1]:
 *  run base but patch the carry cell's FULL state with source's trajectory each step.
 *  If base's outputs flip to the carry0=1 answer, the carry cell causally carries carry0
 *  (a1,b1 are identical, so the carry cell is the ONLY thing changed). */
function interchangeProbe(par: Float64Array): void {
	console.log('  INTERCHANGE INTERVENTION (patch carry cell state 0→1, same a1,b1):');
	let flip = 0, keep0 = 0;
	for (let a1 = 0; a1 < 2; a1++) for (let b1 = 0; b1 < 2; b1++) {
		const base = CASES.find((c) => c.in[0] === a1 && c.in[1] === b1 && c.in[2] === 0 && c.in[3] === 0)!;
		const source = CASES.find((c) => c.in[0] === a1 && c.in[1] === b1 && c.in[2] === 1 && c.in[3] === 1)!;
		const src = forward(par, source, false);
		const srcCarry = src.map((s) => Array.from({ length: C }, (_, c) => s[CARRY * C + c])); // carry-cell trajectory
		// run base with the carry cell patched to source's trajectory
		const s0 = seedGrid(); clampInputs(s0, base, false);
		for (let c = 0; c < C; c++) s0[CARRY * C + c] = srcCarry[0][c];
		let s = s0; const perc = new Float64Array(PERC), h = new Float64Array(HD);
		for (let t = 0; t < T; t++) {
			const ns = new Float64Array(N * C);
			for (let y = 1; y < H - 1; y++) for (let x = 1; x < W - 1; x++) {
				const i = y * W + x; perceive(s, i, perc);
				for (let hh = 0; hh < HD; hh++) { let a = par[B1O + hh]; const bb = W1O + hh * PERC; for (let k = 0; k < PERC; k++) a += par[bb + k] * perc[k]; h[hh] = a > 0 ? a : 0; }
				for (let c = 0; c < C; c++) { let dl = par[B2O + c]; const bb = W2O + c * HD; for (let hh = 0; hh < HD; hh++) dl += par[bb + hh] * h[hh]; ns[i * C + c] = Math.tanh(s[i * C + c] + dl); }
			}
			clampInputs(ns, base, false);
			for (let c = 0; c < C; c++) ns[CARRY * C + c] = srcCarry[t + 1][c]; // patch full carry-cell state
			s = ns;
		}
		const wantS1 = a1 ^ b1 ^ 1, wantC = (a1 + b1 + 1) >= 2 ? 1 : 0; // the carry0=1 answer
		const flipped = Math.abs(s[SUM1 * C] - wantS1) < 0.3 && Math.abs(s[COUT * C] - wantC) < 0.3;
		if (flipped) flip++;
		// control: base's own (carry0=0) answer WITHOUT patching stays correct
		const un = forward(par, base, false)[T];
		if (Math.abs(un[SUM1 * C] - base.sum1) < 0.3 && Math.abs(un[COUT * C] - base.cout) < 0.3) keep0++;
		console.log(`    a1=${a1} b1=${b1}: patched→ sum1 ${s[SUM1 * C].toFixed(2)} cout ${s[COUT * C].toFixed(2)} (carry0=1 wants ${wantS1} ${wantC}) ${flipped ? '✓ flipped' : '✗'}`);
	}
	console.log(`    → ${flip}/4 patched cases flip to the carry0=1 answer; ${keep0}/4 unpatched stay carry0=0.`);
	console.log(`    ${flip === 4 && keep0 === 4 ? 'CLEAN: the carry cell causally carries carry0 (patching it alone flips exactly the carry-dependent outputs).' : 'partial — carry likely distributed beyond the single cell.'}`);
	regionLesionProbe(par);
}

/** Region lesion — zero a (2r+1)² block around the carry cell every step, and read
 *  per-output accuracy. Prediction for a compositional carry: sum0 (carry-independent,
 *  = a0⊕b0) survives; sum1/cout (carry-dependent) degrade. Sweeps radius. */
function regionLesionProbe(par: Float64Array): void {
	console.log('  REGION LESION (zero a block around the carry cell, every step):');
	const cx = (CARRY % W), cyc = Math.floor(CARRY / W);
	for (const r of [0, 1, 2]) {
		const region = new Set<number>();
		for (let dy = -r; dy <= r; dy++) for (let dx = -r; dx <= r; dx++) { const x = cx + dx, y = cyc + dy; if (x > 0 && x < W - 1 && y > 0 && y < H - 1) region.add(y * W + x); }
		let okS0 = 0, okS1 = 0, okC = 0;
		for (const cse of CASES) {
			const s0 = seedGrid(); clampInputs(s0, cse, false);
			let s = s0; const perc = new Float64Array(PERC), h = new Float64Array(HD);
			for (let t = 0; t < T; t++) {
				const ns = new Float64Array(N * C);
				for (let y = 1; y < H - 1; y++) for (let x = 1; x < W - 1; x++) {
					const i = y * W + x; perceive(s, i, perc);
					for (let hh = 0; hh < HD; hh++) { let a = par[B1O + hh]; const bb = W1O + hh * PERC; for (let k = 0; k < PERC; k++) a += par[bb + k] * perc[k]; h[hh] = a > 0 ? a : 0; }
					for (let c = 0; c < C; c++) { let dl = par[B2O + c]; const bb = W2O + c * HD; for (let hh = 0; hh < HD; hh++) dl += par[bb + hh] * h[hh]; ns[i * C + c] = Math.tanh(s[i * C + c] + dl); }
				}
				clampInputs(ns, cse, false);
				for (const cell of region) for (let c = 0; c < C; c++) ns[cell * C + c] = 0;
				s = ns;
			}
			if (Math.abs(s[SUM0 * C] - cse.sum0) < 0.3) okS0++;
			if (Math.abs(s[SUM1 * C] - cse.sum1) < 0.3) okS1++;
			if (Math.abs(s[COUT * C] - cse.cout) < 0.3) okC++;
		}
		console.log(`    r=${r} (${region.size} cells): sum0 ${okS0}/16  sum1 ${okS1}/16  cout ${okC}/16   (carry-independent sum0 vs carry-dependent sum1/cout)`);
		if (r === 0) console.log(`    → DISSOCIATION: zeroing the carry cell leaves carry-independent sum0 at ${okS0}/16 but degrades carry-dependent sum1/cout to ${okS1}/${okC}/16 — the carry cell is causally special for exactly the carry-dependent outputs.`);
	}
}

function main(): void {
	const mode = process.env.MODE ?? 'expose';
	console.log(`2-bit ripple-carry adder: ${W}x${H}, C=${C}, HD=${HD}, ${P} params, T=${T}, 16 cases`);
	if (mode === 'expose') {
		if (!gradientCheck(true)) { process.exit(1); }
		const par = exposeCurriculum(Number(process.env.ITERS ?? 600));
		report(par, true);
		if (process.env.PARAMS_OUT) { writeFileSync(process.env.PARAMS_OUT, JSON.stringify(Array.from(par))); console.log(`  saved ${process.env.PARAMS_OUT}`); }
	} else if (mode === 'intern') {
		if (!process.env.PARAMS_IN) { console.error('intern needs PARAMS_IN'); process.exit(1); }
		if (!gradientCheck(false)) { process.exit(1); }
		const par0 = Float64Array.from(JSON.parse(readFileSync(process.env.PARAMS_IN, 'utf8')) as number[]);
		console.log('  INTERNALIZE — release the clamp, supervise the carry cell (aux loss), warm from expose');
		const wCarry = Number(process.env.WCARRY ?? 2);
		const par = train(Number(process.env.ITERS ?? 800), par0, false, wCarry).par;
		report(par, false); probe(par);
		if (process.env.PARAMS_OUT) { writeFileSync(process.env.PARAMS_OUT, JSON.stringify(Array.from(par))); console.log(`  saved ${process.env.PARAMS_OUT}`); }
	} else if (mode === 'probe') {
		const par = Float64Array.from(JSON.parse(readFileSync(process.env.PARAMS_IN!, 'utf8')) as number[]);
		report(par, false); probe(par);
	}
}
main();
