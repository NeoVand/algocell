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

// ===========================================================================
// STABILIZE + SELF-REPAIR + REACTIVE — make the 2-bit adder an ongoing attractor
// that holds its answer indefinitely, heals damage, and re-settles on live input
// changes. Warm-started from the internalized compute rule. The internal carry is
// kept alive by a light auxiliary score on the carry cell so it persists too.
// ===========================================================================

// scored cells over hold windows: the 3 real outputs + the internal carry (aux).
function scoredCellsHold(): { cell: number; tgt: (c: Case) => number; w: number }[] {
	return [
		{ cell: SUM1, tgt: (c) => c.sum1, w: 1 }, { cell: SUM0, tgt: (c) => c.sum0, w: 1 },
		{ cell: COUT, tgt: (c) => c.cout, w: 1 }, { cell: CARRY, tgt: (c) => c.carry0, w: Number(process.env.WCARRY ?? 0.5) }
	];
}
function damageMask(cx: number, cy: number, size: number): Uint8Array {
	const mask = new Uint8Array(N).fill(1); const h = size >> 1;
	for (let y = cy - h; y <= cy - h + size - 1; y++) for (let x = cx - h; x <= cx - h + size - 1; x++)
		if (x >= 0 && x < W && y >= 0 && y < H) mask[y * W + x] = 0;
	return mask;
}
function candidatePatches(): { cx: number; cy: number }[] {
	const list: { cx: number; cy: number }[] = [];
	for (let cx = IN_COL + 1; cx <= OUT_COL; cx++) for (let cy = 2; cy <= H - 3; cy++) list.push({ cx, cy });
	return list;
}
// timeline: grow → hold → (damage) → re-hold. Weights sum to 1 across scored steps.
const S_GROW = Number(process.env.SGROW ?? 40), S_HOLD = Number(process.env.SHOLD ?? 26), S_REPAIR = Number(process.env.SREPAIR ?? 22);
const S_DMG_AT = S_GROW + S_HOLD, S_TOTAL = S_DMG_AT + S_REPAIR, S_TAIL = 8;
interface Sched { w: Float64Array; nSteps: number; dmgAt: number; }
function schedule(mode: 'persist' | 'repair'): Sched {
	const w = new Float64Array(S_TOTAL + 1);
	const hn = S_DMG_AT - 1 - S_GROW + 1; for (let s = S_GROW; s <= S_DMG_AT - 1; s++) w[s] = 0.5 / hn;
	if (mode === 'persist') { const tn = S_TOTAL - S_DMG_AT + 1; for (let s = S_DMG_AT; s <= S_TOTAL; s++) w[s] = 0.5 / tn; return { w, nSteps: S_TOTAL, dmgAt: -1 }; }
	const rn = S_TAIL; for (let s = S_TOTAL - S_TAIL + 1; s <= S_TOTAL; s++) w[s] += 0.5 / rn;
	return { w, nSteps: S_TOTAL, dmgAt: S_DMG_AT };
}

function forwardHold(par: Float64Array, cse: Case, sched: Sched, mask: Uint8Array | null): Float64Array[] {
	const s0 = seedGrid(); clampInputs(s0, cse, false);
	const states: Float64Array[] = [s0]; let s = s0;
	const perc = new Float64Array(PERC), h = new Float64Array(HD);
	for (let t = 0; t < sched.nSteps; t++) {
		const ns = new Float64Array(N * C);
		for (let y = 1; y < H - 1; y++) for (let x = 1; x < W - 1; x++) {
			const i = y * W + x; perceive(s, i, perc);
			for (let hh = 0; hh < HD; hh++) { let a = par[B1O + hh]; const base = W1O + hh * PERC; for (let k = 0; k < PERC; k++) a += par[base + k] * perc[k]; h[hh] = a > 0 ? a : 0; }
			for (let c = 0; c < C; c++) { let dl = par[B2O + c]; const base = W2O + c * HD; for (let hh = 0; hh < HD; hh++) dl += par[base + hh] * h[hh]; ns[i * C + c] = Math.tanh(s[i * C + c] + dl); }
		}
		if (mask && t + 1 === sched.dmgAt) for (let i = 0; i < N; i++) if (mask[i] === 0) for (let c = 0; c < C; c++) ns[i * C + c] = 0;
		clampInputs(ns, cse, false);
		states.push(ns); s = ns;
	}
	return states;
}

// backward shared with the reactive path: given per-step output-gradient seeds, backprop to grad.
function backpropHold(par: Float64Array, states: Float64Array[], nSteps: number, dmgAt: number, mask: Uint8Array | null,
	seedAt: (t: number, gs: Float64Array) => void, grad: Float64Array): void {
	const perc = new Float64Array(PERC), pre1 = new Float64Array(HD), hbuf = new Float64Array(HD), gh = new Float64Array(HD), gperc = new Float64Array(PERC);
	let gs = new Float64Array(N * C);
	for (let t = nSteps - 1; t >= 0; t--) {
		seedAt(t + 1, gs);                                   // add output-loss gradient at state t+1
		for (const p of inputCells) gs[p * C + 0] = 0;       // clamped inputs: no grad through override
		if (mask && t + 1 === dmgAt) for (let i = 0; i < N; i++) if (mask[i] === 0) for (let c = 0; c < C; c++) gs[i * C + c] = 0;
		const st = states[t], sp = states[t + 1];
		const gsPrev = new Float64Array(N * C);
		for (let y = 1; y < H - 1; y++) for (let x = 1; x < W - 1; x++) {
			const i = y * W + x; perceive(st, i, perc);
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

function lossHold(par: Float64Array, sched: Sched, mask: Uint8Array | null): { L: number; grad: Float64Array } {
	const grad = new Float64Array(P); let L = 0;
	const scored = scoredCellsHold();
	const norm = CASES.length * scored.reduce((a, s) => a + s.w, 0);
	for (const cse of CASES) {
		const states = forwardHold(par, cse, sched, mask);
		backpropHold(par, states, sched.nSteps, sched.dmgAt, mask, (step, gs) => {
			if (sched.w[step] <= 0) return;
			for (const sc of scored) { const oc = sc.cell * C + 0; const diff = states[step][oc] - sc.tgt(cse); L += (sched.w[step] * sc.w * diff * diff) / norm; gs[oc] += (2 * sched.w[step] * sc.w * diff) / norm; }
		}, grad);
	}
	return { L, grad };
}

// evalBest (optional): keep the params that MAXIMISE this score (evaluated every 30 iters),
// instead of minimising loss. Used for reactivity, where low loss ≠ a rule that both holds
// and migrates — and it guarantees we never save worse than the warm-start (init is scored too).
function adamTrain(iters: number, init: Float64Array, lossFn: (par: Float64Array, it: number) => { L: number; grad: Float64Array }, lrHi = 0.003, lrLo = 0.0006, evalBest?: (par: Float64Array) => number): { par: Float64Array; L: number } {
	let par = init.slice();
	const m = new Float64Array(P), v = new Float64Array(P), b1 = 0.9, b2 = 0.999;
	let bestLoss = Infinity, bestPar = par.slice();
	let bestScore = evalBest ? evalBest(init) : -Infinity;
	if (evalBest) console.log(`      init score ${bestScore.toFixed(3)}`);
	for (let it = 1; it <= iters; it++) {
		const cos = 0.5 * (1 + Math.cos(Math.PI * (it / iters)));
		const lr = Math.min(1, it / 20) * (lrLo + (lrHi - lrLo) * cos);
		const { L, grad } = lossFn(par, it);
		let gn = 0; for (let j = 0; j < P; j++) gn += grad[j] * grad[j]; gn = Math.sqrt(gn);
		const clip = gn > 1 ? 1 / gn : 1;
		if (!evalBest && L < bestLoss) { bestLoss = L; bestPar = par.slice(); }
		const c1 = 1 - Math.pow(b1, it), c2 = 1 - Math.pow(b2, it);
		for (let j = 0; j < P; j++) { const g = grad[j] * clip + 2e-5 * par[j]; m[j] = b1 * m[j] + (1 - b1) * g; v[j] = b2 * v[j] + (1 - b2) * g * g; par[j] -= (lr * (m[j] / c1)) / (Math.sqrt(v[j] / c2) + 1e-8); }
		if (evalBest && (it % 30 === 0)) { const sc = evalBest(par); if (sc > bestScore) { bestScore = sc; bestPar = par.slice(); } console.log(`      iter ${String(it).padStart(4)}  loss ${L.toFixed(4)}  score ${sc.toFixed(3)}  best ${bestScore.toFixed(3)}`); }
		else if (it % 50 === 0 || it === 1) console.log(`      iter ${String(it).padStart(4)}  loss ${L.toFixed(5)}  best ${(evalBest ? bestScore : bestLoss).toFixed(5)}`);
	}
	return { par: bestPar, L: bestLoss };
}

function accHold(par: Float64Array, mask: Uint8Array | null): number {
	const sched = mask ? schedule('repair') : schedule('persist');
	let acc = 0;
	for (const cse of CASES) {
		const sT = forwardHold(par, cse, sched, mask)[sched.nSteps];
		if (Math.abs(sT[SUM1 * C] - cse.sum1) < 0.3 && Math.abs(sT[SUM0 * C] - cse.sum0) < 0.3 && Math.abs(sT[COUT * C] - cse.cout) < 0.3) acc++;
	}
	return acc;
}
/** long-horizon hold from a plain rollout (no schedule). */
function driftCheck(par: Float64Array): void {
	for (const steps of [50, 150, 400]) {
		let acc = 0;
		for (const cse of CASES) {
			let s = seedGrid(); clampInputs(s, cse, false);
			const perc = new Float64Array(PERC), h = new Float64Array(HD);
			for (let t = 0; t < steps; t++) {
				const ns = new Float64Array(N * C);
				for (let y = 1; y < H - 1; y++) for (let x = 1; x < W - 1; x++) { const i = y * W + x; perceive(s, i, perc); for (let hh = 0; hh < HD; hh++) { let a = par[B1O + hh]; const base = W1O + hh * PERC; for (let k = 0; k < PERC; k++) a += par[base + k] * perc[k]; h[hh] = a > 0 ? a : 0; } for (let c = 0; c < C; c++) { let dl = par[B2O + c]; const base = W2O + c * HD; for (let hh = 0; hh < HD; hh++) dl += par[base + hh] * h[hh]; ns[i * C + c] = Math.tanh(s[i * C + c] + dl); } }
				clampInputs(ns, cse, false); s = ns;
			}
			if (Math.abs(s[SUM1 * C] - cse.sum1) < 0.3 && Math.abs(s[SUM0 * C] - cse.sum0) < 0.3 && Math.abs(s[COUT * C] - cse.cout) < 0.3) acc++;
		}
		console.log(`    long-horizon @${steps}: ${acc}/16`);
	}
}

function gradientCheckHold(): boolean {
	const rng = mulberry32(5); const par = new Float64Array(P).map(() => (rng() - 0.5) * 0.1);
	for (let j = W2O; j < P; j++) par[j] = 0; // near-identity → unsaturated → clean finite diff
	const sched = schedule('repair'); const mask = damageMask(6, 6, 3);
	const { grad } = lossHold(par, sched, mask);
	const eps = 1e-4; let maxRel = 0;
	for (const j of [12, 2500, B1O + 4, W2O + 9, B2O + 2]) {
		const pp = par.slice(); pp[j] += eps; const pm = par.slice(); pm[j] -= eps;
		const fd = (lossHold(pp, sched, mask).L - lossHold(pm, sched, mask).L) / (2 * eps);
		maxRel = Math.max(maxRel, Math.abs(grad[j] - fd) / (Math.abs(fd) + 1e-8));
	}
	console.log(`  hold gradient check: max rel err ${maxRel.toExponential(2)} -> ${maxRel < 0.02 ? 'PASS' : 'FAIL'}`);
	return maxRel < 0.02;
}

function stabilizeMain(): void {
	if (!process.env.PARAMS_IN) { console.error('stabilize needs PARAMS_IN (compute params)'); process.exit(1); }
	if (!gradientCheckHold()) { console.error('FAIL: hold gradient wrong'); process.exit(1); }
	const par0 = Float64Array.from(JSON.parse(readFileSync(process.env.PARAMS_IN, 'utf8')) as number[]);
	console.log(`STABILIZE 2-bit adder: grow ${S_GROW}+hold ${S_HOLD}+repair ${S_REPAIR}=${S_TOTAL}`);
	const iters = Number(process.env.ITERS ?? 500);
	const patches = candidatePatches(), schedP = schedule('persist'), schedR = schedule('repair');
	console.log('  [A] persist (hold window, warm from compute)');
	const parA = adamTrain(iters, par0, (par) => lossHold(par, schedP, null)).par;
	console.log(`    held ${accHold(parA, null)}/16`);
	console.log('  [B] + damage (self-repair)');
	const parB = adamTrain(Math.round(iters * 1.4), parA, (par, it) => {
		const pc = patches[Math.floor(mulberry32((it * 2654435761) >>> 0)() * patches.length)];
		return lossHold(par, schedR, damageMask(pc.cx, pc.cy, 3));
	}).par;
	const midMask = damageMask(Math.round((IN_COL + OUT_COL) / 2), iy, 3);
	console.log(`    held ${accHold(parB, null)}/16, +damage ${accHold(parB, midMask)}/16`);
	driftCheck(parB);
	if (process.env.PARAMS_OUT) { writeFileSync(process.env.PARAMS_OUT, JSON.stringify(Array.from(parB))); console.log(`  saved ${process.env.PARAMS_OUT}`); }
}

// ---- REACTIVE: hold prior answer, flip an input mid-rollout, re-settle to the new answer ----
// Long post-switch window (R_T2): the internal carry must re-propagate FA0→FA1 after a flip,
// which takes many steps — a short window is why the first attempt got 0/256.
const R_T1 = Number(process.env.RT1 ?? 40), R_T2 = Number(process.env.RT2 ?? 85), R_TAIL = Number(process.env.RTAIL ?? 16), R_TOTAL = R_T1 + R_T2, R_DMG_AT = R_T1 + 8;
function caseOf(inp: number[]): Case { return CASES.find((c) => c.in.every((v, i) => v === inp[i]))!; }
function correct(st: Float64Array, c: Case): boolean { return Math.abs(st[SUM1 * C] - c.sum1) < 0.3 && Math.abs(st[SUM0 * C] - c.sum0) < 0.3 && Math.abs(st[COUT * C] - c.cout) < 0.3; }
// keep-best evals (cheap subsets): reactivity over sampled transitions + hold accuracy.
const EVAL_PAIRS: [number, number][] = []; { const rng = mulberry32(4242); for (let i = 0; i < 24; i++) EVAL_PAIRS.push([Math.floor(rng() * 16), Math.floor(rng() * 16)]); }
function reactAccSample(par: Float64Array): number { let ok = 0; for (const [pi, ti] of EVAL_PAIRS) if (correct(forwardReact(par, CASES[pi], CASES[ti], null)[R_TOTAL], CASES[ti])) ok++; return ok / EVAL_PAIRS.length; }
function holdAccQuick(par: Float64Array): number { const sched = schedule('persist'); let ok = 0; for (const c of CASES) if (correct(forwardHold(par, c, sched, null)[sched.nSteps], c)) ok++; return ok / 16; }
function forwardReact(par: Float64Array, prior: Case, target: Case, mask: Uint8Array | null): Float64Array[] {
	const s0 = seedGrid(); clampInputs(s0, prior, false);
	const states: Float64Array[] = [s0]; let s = s0;
	const perc = new Float64Array(PERC), h = new Float64Array(HD);
	for (let t = 0; t < R_TOTAL; t++) {
		const cse = t < R_T1 ? prior : target;
		const ns = new Float64Array(N * C);
		for (let y = 1; y < H - 1; y++) for (let x = 1; x < W - 1; x++) { const i = y * W + x; perceive(s, i, perc); for (let hh = 0; hh < HD; hh++) { let a = par[B1O + hh]; const base = W1O + hh * PERC; for (let k = 0; k < PERC; k++) a += par[base + k] * perc[k]; h[hh] = a > 0 ? a : 0; } for (let c = 0; c < C; c++) { let dl = par[B2O + c]; const base = W2O + c * HD; for (let hh = 0; hh < HD; hh++) dl += par[base + hh] * h[hh]; ns[i * C + c] = Math.tanh(s[i * C + c] + dl); } }
		if (mask && t + 1 === R_DMG_AT) for (let i = 0; i < N; i++) if (mask[i] === 0) for (let c = 0; c < C; c++) ns[i * C + c] = 0;
		clampInputs(ns, cse, false); states.push(ns); s = ns;
	}
	return states;
}
function lossReact(par: Float64Array, priors: Case[], masks: (Uint8Array | null)[]): { L: number; grad: Float64Array; acc: number } {
	const grad = new Float64Array(P); let L = 0, acc = 0;
	const scored = scoredCellsHold(); const norm = CASES.length * scored.reduce((a, s) => a + s.w, 0) * R_TAIL;
	const p1s = R_T1 - R_TAIL + 1, p2s = R_TOTAL - R_TAIL + 1;
	CASES.forEach((target, ci) => {
		const prior = priors[ci], mask = masks[ci];
		const states = forwardReact(par, prior, target, mask);
		const fin = states[R_TOTAL];
		if (Math.abs(fin[SUM1 * C] - target.sum1) < 0.3 && Math.abs(fin[SUM0 * C] - target.sum0) < 0.3 && Math.abs(fin[COUT * C] - target.cout) < 0.3) acc++;
		backpropReact(par, states, mask, (step, gs) => {
			const cse = step <= R_T1 ? prior : target;
			const inWinA = step >= p1s && step <= R_T1, inWinB = step >= p2s;
			if (!inWinA && !inWinB) return;
			for (const sc of scored) { const oc = sc.cell * C + 0; const diff = states[step][oc] - sc.tgt(cse); L += (sc.w * diff * diff) / norm; gs[oc] += (2 * sc.w * diff) / norm; }
		}, grad);
	});
	return { L, grad, acc };
}
function backpropReact(par: Float64Array, states: Float64Array[], mask: Uint8Array | null, seedAt: (t: number, gs: Float64Array) => void, grad: Float64Array): void {
	backpropHold(par, states, R_TOTAL, R_DMG_AT, mask, seedAt, grad);
}
function reactReport(par: Float64Array): void {
	let ok = 0, tot = 0;
	for (const prior of CASES) for (const target of CASES) { const st = forwardReact(par, prior, target, null)[R_TOTAL]; tot++; if (Math.abs(st[SUM1 * C] - target.sum1) < 0.3 && Math.abs(st[SUM0 * C] - target.sum0) < 0.3 && Math.abs(st[COUT * C] - target.cout) < 0.3) ok++; }
	console.log(`  reactivity: ${ok}/${tot} prior→target transitions land on the new answer`);
}
function reactiveMain(): void {
	if (!process.env.PARAMS_IN) { console.error('reactive needs PARAMS_IN (stable params)'); process.exit(1); }
	const par0 = Float64Array.from(JSON.parse(readFileSync(process.env.PARAMS_IN, 'utf8')) as number[]);
	console.log(`REACTIVE 2-bit adder: hold ${R_T1} + switch + hold ${R_T2}, tail ${R_TAIL}, damage @${R_DMG_AT}`);
	const iters = Number(process.env.ITERS ?? 500), patches = candidatePatches();
	// keep-best: maximise migration but only among rules that still HOLD (≥70%); else keep the
	// best-holding rule → the retry can never end up worse than the stable warm-start.
	const evalBest = (par: Float64Array) => { const h = holdAccQuick(par); return h >= 0.7 ? reactAccSample(par) + h : h - 2; };
	console.log('  [A] reactive (learn input transitions, warm from stable, gentle lr)');
	const parA = adamTrain(iters, par0, (par, it) => { const rng = mulberry32((it * 40503) >>> 0); const priors = CASES.map(() => CASES[Math.floor(rng() * 16)]); return lossReact(par, priors, CASES.map(() => null)); }, 0.0012, 0.0003, evalBest).par;
	reactReport(parA); console.log(`    hold ${(holdAccQuick(parA) * 16).toFixed(0)}/16`);
	console.log('  [B] reactive + damage (retain self-repair)');
	const parB = adamTrain(Math.round(iters * 1.2), parA, (par, it) => { const rng = mulberry32((it * 22699) >>> 0); const priors = CASES.map(() => CASES[Math.floor(rng() * 16)]); const masks = CASES.map(() => { const pc = patches[Math.floor(rng() * patches.length)]; return damageMask(pc.cx, pc.cy, 3); }); return lossReact(par, priors, masks); }, 0.0012, 0.0003, evalBest).par;
	reactReport(parB); console.log(`    hold ${(holdAccQuick(parB) * 16).toFixed(0)}/16`); driftCheck(parB);
	if (process.env.PARAMS_OUT) { writeFileSync(process.env.PARAMS_OUT, JSON.stringify(Array.from(parB))); console.log(`  saved ${process.env.PARAMS_OUT}`); }
}

// JOINT: hold + self-repair + reactivity trained TOGETHER from the start (no staging),
// warm from the COMPUTE rule (has the internal carry, but is not yet a rigid attractor).
// Hypothesis: the sequential stabilize→reactive staging built a rigid attractor that
// fights migration; training all objectives jointly should find a soft attractor instead.
function jointMain(): void {
	if (!process.env.PARAMS_IN) { console.error('joint needs PARAMS_IN (compute params)'); process.exit(1); }
	const par0 = Float64Array.from(JSON.parse(readFileSync(process.env.PARAMS_IN, 'utf8')) as number[]);
	console.log(`JOINT 2-bit adder: hold+repair+reactive from iter 1 (no staging). hold ${R_T1}+switch+hold ${R_T2}, tail ${R_TAIL}, damage@${R_DMG_AT}`);
	const iters = Number(process.env.ITERS ?? 700), patches = candidatePatches();
	const midMask = damageMask(Math.round((IN_COL + OUT_COL) / 2), iy, 3);
	// keep-best values ALL three: reactivity + hold + repair, once the rule holds at all (≥0.5).
	const evalBest = (par: Float64Array) => { const h = holdAccQuick(par); if (h < 0.5) return h - 3; return reactAccSample(par) + h + accHold(par, midMask) / 16; };
	const par = adamTrain(iters, par0, (par, it) => {
		const rng = mulberry32((it * 2246822519) >>> 0);
		const priors = CASES.map(() => CASES[Math.floor(rng() * 16)]);                        // random prior → any transition
		const masks = CASES.map((_, ci) => (ci % 2 === 0 ? (() => { const pc = patches[Math.floor(rng() * patches.length)]; return damageMask(pc.cx, pc.cy, 3); })() : null)); // half damaged (repair), half clean (reactive)
		return lossReact(par, priors, masks);
	}, 0.0018, 0.0004, evalBest).par;
	reactReport(par);
	console.log(`  hold ${(holdAccQuick(par) * 16).toFixed(0)}/16, +damage ${accHold(par, midMask)}/16`);
	driftCheck(par);
	if (process.env.PARAMS_OUT) { writeFileSync(process.env.PARAMS_OUT, JSON.stringify(Array.from(par))); console.log(`  saved ${process.env.PARAMS_OUT}`); }
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
	} else if (mode === 'stabilize') { stabilizeMain(); }
	else if (mode === 'reactive') { reactiveMain(); }
	else if (mode === 'joint') { jointMain(); }
}
main();
