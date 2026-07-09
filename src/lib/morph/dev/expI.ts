// EXPERIMENT I — POSITIONAL INVARIANCE (the self-wiring circuit).
//
// One CA rule that computes regardless of WHERE the input/output ports are — move
// a port anywhere and the plane rewires to make it work. The terminals announce
// themselves with MARKER channels (IN_MARK / OUT_MARK), so the (translation-
// invariant, grid-size-agnostic) rule reads markers instead of absolute positions.
// Trained on randomized port placements → it can't memorize positions, so it must
// learn to route: emit beacons from the markers and follow them ("waves finding
// each other"). Params are grid-independent, so it transfers to bigger grids.
//
//   TASK=wire  — movable WIRE (1 input → 1 output, output = input).  [de-risk]
//   TASK=xor   — movable XOR  (2 inputs → 1 output, output = a⊕b).   [default]
//
//   TASK=xor ITERS=… PARAMS_OUT=… npx tsx src/lib/morph/dev/expI.ts

import { writeFileSync, readFileSync } from 'node:fs';

// Rule is grid-INDEPENDENT: channels + MLP only. ch0 = signal, ch1 = IN_MARK,
// ch2 = OUT_MARK, ch3.. = hidden (beacons/structure the rule invents).
const C = 16, FEAT = 4, PERC = FEAT * C, HD = 96;
const W1O = 0, B1O = HD * PERC, W2O = B1O + HD, B2O = W2O + C * HD, P = B2O + C;
const IN_MARK = 1, OUT_MARK = 2;

const TASK = process.env.TASK ?? 'xor';
const N_IN = TASK === 'xor' ? 2 : 1;
const CASES: number[][] = TASK === 'xor'
	? [[0, 0], [0, 1], [1, 0], [1, 1]]
	: [[0], [1]];
const target = (bits: number[]): number => (TASK === 'xor' ? bits[0] ^ bits[1] : bits[0]);

const GW = 11, GH = 11; // training grid (rule works on any size)
const T = Number(process.env.T ?? (TASK === 'xor' ? 36 : 32)); // developmental steps

function mulberry32(seed: number): () => number {
	let a = seed >>> 0;
	return () => {
		a |= 0; a = (a + 0x6d2b79f5) | 0;
		let t = Math.imul(a ^ (a >>> 15), 1 | a);
		t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
}

interface Place { gw: number; gh: number; ins: number[]; out: number; }

/** Re-assert markers + inputs every step (fixed boundary conditions the rule reads). */
function stamp(f: Float64Array, N: number, ins: number[], bits: number[], out: number): void {
	for (let i = 0; i < N; i++) { f[i * C + IN_MARK] = 0; f[i * C + OUT_MARK] = 0; }
	for (let k = 0; k < ins.length; k++) { f[ins[k] * C + IN_MARK] = 1; f[ins[k] * C + 0] = bits[k]; }
	f[out * C + OUT_MARK] = 1;
}

function seed(gw: number, gh: number, ins: number[], bits: number[], out: number): Float64Array {
	const N = gw * gh, s = new Float64Array(N * C);
	for (let y = 1; y < gh - 1; y++) for (let x = 1; x < gw - 1; x++)
		for (let c = 3; c < C; c++) s[(y * gw + x) * C + c] = 1; // uniform alive interior (position-invariant)
	stamp(s, N, ins, bits, out);
	return s;
}

function perceive(gw: number, s: Float64Array, i: number, out: Float64Array): void {
	const r = i + 1, l = i - 1, u = i - gw, d = i + gw;
	for (let ch = 0; ch < C; ch++) {
		const b = ch * FEAT, self = s[i * C + ch];
		const sr = s[r * C + ch], sl = s[l * C + ch], su = s[u * C + ch], sd = s[d * C + ch];
		out[b] = self;
		out[b + 1] = (sr - sl) * 0.5;
		out[b + 2] = (sd - su) * 0.5;
		out[b + 3] = sr + sl + su + sd - 4 * self;
	}
}

function forward(par: Float64Array, gw: number, gh: number, ins: number[], bits: number[], out: number, nSteps: number): Float64Array[] {
	const N = gw * gh;
	const s0 = seed(gw, gh, ins, bits, out);
	const states: Float64Array[] = [s0];
	let s = s0;
	const perc = new Float64Array(PERC), h = new Float64Array(HD);
	for (let t = 0; t < nSteps; t++) {
		const ns = new Float64Array(N * C);
		for (let y = 1; y < gh - 1; y++)
			for (let x = 1; x < gw - 1; x++) {
				const i = y * gw + x;
				perceive(gw, s, i, perc);
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
		stamp(ns, N, ins, bits, out);
		states.push(ns);
		s = ns;
	}
	return states;
}

/** Loss + gradient over a batch of placements, each scored on all input cases. */
function lossAndGrad(par: Float64Array, places: Place[]): { L: number; grad: Float64Array; acc: number } {
	const grad = new Float64Array(P);
	let L = 0, correct = 0, total = 0;
	const norm = places.length * CASES.length;
	const perc = new Float64Array(PERC), pre1 = new Float64Array(HD), hbuf = new Float64Array(HD), gh = new Float64Array(HD), gperc = new Float64Array(PERC);
	for (const pl of places) {
		const { gw, gh: ghh, ins, out } = pl;
		const N = gw * ghh;
		for (const bits of CASES) {
			const tgt = target(bits);
			const states = forward(par, gw, ghh, ins, bits, out, T);
			const o = states[T][out * C + 0];
			const diff = o - tgt;
			L += (diff * diff) / norm;
			total++; if (Math.abs(diff) < 0.3) correct++;
			const gsT = new Float64Array(N * C);
			gsT[out * C + 0] = (2 * diff) / norm;
			let gs = gsT;
			for (let t = T - 1; t >= 0; t--) {
				for (let i = 0; i < N; i++) { gs[i * C + IN_MARK] = 0; gs[i * C + OUT_MARK] = 0; } // markers clamped everywhere
				for (const inc of ins) gs[inc * C + 0] = 0; // input ch0 clamped
				const s = states[t], sp = states[t + 1];
				const gsPrev = new Float64Array(N * C);
				for (let y = 1; y < ghh - 1; y++)
					for (let x = 1; x < gw - 1; x++) {
						const i = y * gw + x;
						perceive(gw, s, i, perc);
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
						const r = i + 1, l = i - 1, u = i - gw, d = i + gw;
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
	}
	return { L, grad, acc: correct / total };
}

/** Random placement. `maxDist` caps the Manhattan spread of the ports around the
 *  first input — a curriculum knob: small = ports clustered (easy routing, learn
 *  the combine); large = ports anywhere. */
function randPlace(rng: () => number, gw = GW, gh = GH, maxDist = 9999): Place {
	const rx = () => 1 + Math.floor(rng() * (gw - 2));
	const ry = () => 1 + Math.floor(rng() * (gh - 2));
	const x0 = rx(), y0 = ry();
	const used = new Set<number>([y0 * gw + x0]);
	const near = () => {
		for (let t = 0; t < 50; t++) {
			const x = rx(), y = ry();
			if (Math.abs(x - x0) + Math.abs(y - y0) <= maxDist && !used.has(y * gw + x)) { used.add(y * gw + x); return y * gw + x; }
		}
		let c = ry() * gw + rx(); while (used.has(c)) c = ry() * gw + rx();
		used.add(c); return c;
	};
	const ins = [y0 * gw + x0];
	for (let k = 1; k < N_IN; k++) ins.push(near());
	const out = near();
	return { gw, gh, ins, out };
}

function gradientCheck(): boolean {
	const rng = mulberry32(3);
	const par = new Float64Array(P).map(() => (rng() - 0.5) * 0.1);
	for (let j = W2O; j < P; j++) par[j] = 0;
	const places = [randPlace(rng), randPlace(rng)];
	const { grad } = lossAndGrad(par, places);
	const eps = 1e-4; let maxRel = 0;
	for (const j of [11, 1200, B1O + 6, W2O + 10, B2O + 2]) {
		const pp = par.slice(); pp[j] += eps;
		const pm = par.slice(); pm[j] -= eps;
		const fd = (lossAndGrad(pp, places).L - lossAndGrad(pm, places).L) / (2 * eps);
		maxRel = Math.max(maxRel, Math.abs(grad[j] - fd) / (Math.abs(fd) + 1e-8));
	}
	console.log(`  gradient check: max rel err ${maxRel.toExponential(2)} -> ${maxRel < 0.02 ? 'PASS' : 'FAIL'}`);
	return maxRel < 0.02;
}

const BATCH = Number(process.env.BATCH ?? 6);

function train(iters: number, init?: Float64Array): Float64Array {
	let par: Float64Array;
	if (init) { par = init.slice(); }
	else {
		const rng0 = mulberry32(7);
		par = new Float64Array(P);
		for (let j = 0; j < P; j++) par[j] = (rng0() - 0.5) * 0.1;
		// ZERO=1 → near-identity rule (last layer 0): builds routing+combine gradually,
		// avoids the constant-0.5 collapse (as the compute experiments do).
		if (process.env.ZERO === '1') for (let j = W2O; j < P; j++) par[j] = 0;
		else for (let j = W2O; j < P; j++) par[j] *= 0.4;
	}
	const FULL = GW + GH; // full-grid Manhattan span
	const warm = init !== undefined;
	const curr = Number(process.env.CURR ?? 0.5); // curriculum: fraction of training over which ports spread to full
	const phase1 = Number(process.env.PHASE1 ?? 0); // fraction of training at a single FIXED placement first (master XOR, then generalize)
	const lrPeak = Number(process.env.LR ?? (warm ? 0.002 : 0.006));
	const iyc = GH >> 1, ic = GW >> 1;
	// phase-1 canonical placement: ports ADJACENT (output next to the inputs, like
	// E1's d=1) so XOR is learnable from scratch before we spread the ports out.
	const canonical = (): Place => ({ gw: GW, gh: GH, ins: N_IN === 2 ? [(iyc - 1) * GW + (ic - 1), (iyc + 1) * GW + (ic - 1)] : [iyc * GW + (ic - 1)], out: iyc * GW + ic });
	const m = new Float64Array(P), v = new Float64Array(P), b1 = 0.9, b2 = 0.999;
	let bestLoss = Infinity, bestPar = par.slice();
	for (let it = 1; it <= iters; it++) {
		const frac = it / iters;
		// cosine-decay lr to a low floor: find the solution, then settle into its basin
		// instead of bouncing out (the stability fix used across all the other rules).
		const cos = 0.5 * (1 + Math.cos(Math.PI * frac));
		const lr = Math.min(1, it / 40) * lrPeak * (0.1 + 0.9 * cos);
		let places: Place[];
		if (frac < phase1) places = Array.from({ length: BATCH }, canonical); // fixed placement — learn XOR first
		else {
			// then spread to random placements (distance curriculum over the remaining fraction)
			const p2 = Math.max(1e-6, (frac - phase1) / Math.max(1e-6, curr - phase1));
			const maxDist = Math.min(FULL, Math.round(2 + p2 * FULL));
			const rng = mulberry32(1000 + it);
			places = Array.from({ length: BATCH }, () => randPlace(rng, GW, GH, maxDist));
		}
		const { L, grad, acc } = lossAndGrad(par, places);
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
		if (it % 100 === 0 || it === 1) console.log(`    iter ${String(it).padStart(4)}  loss ${L.toFixed(5)}  batch-acc ${(acc * 100).toFixed(0)}%  (best ${bestLoss.toFixed(5)})`);
	}
	return bestPar;
}

/** Test on many random placements at a given grid size; report accuracy. */
function evalAcc(par: Float64Array, gw: number, gh: number, nSteps: number, trials = 60, seedN = 555): number {
	const rng = mulberry32(seedN);
	let ok = 0, total = 0;
	for (let n = 0; n < trials; n++) {
		const pl = randPlace(rng, gw, gh);
		for (const bits of CASES) {
			const tgt = target(bits);
			const st = forward(par, gw, gh, pl.ins, bits, pl.out, nSteps);
			const o = st[nSteps][pl.out * C + 0];
			total++; if (Math.abs(o - tgt) < 0.3) ok++;
		}
	}
	return ok / total;
}

function main() {
	console.log(`positional invariance — movable ${TASK.toUpperCase()}: grid-agnostic rule, C=${C}, HD=${HD}, ${P} params, train ${GW}x${GH}, T=${T}, batch ${BATCH}, ${N_IN} inputs`);
	if (!gradientCheck()) { console.error('FAIL: gradient wrong'); process.exit(1); }
	const iters = Number(process.env.ITERS ?? 800);
	const init = process.env.PARAMS_IN ? Float64Array.from(JSON.parse(readFileSync(process.env.PARAMS_IN, 'utf8')) as number[]) : undefined;
	if (init) console.log(`  warm-starting from ${process.env.PARAMS_IN}`);
	const par = train(iters, init);
	console.log(`\n  ${TASK} accuracy (random placements, all cases):`);
	console.log(`    ${GW}x${GH} @T=${T}:   ${(evalAcc(par, GW, GH, T) * 100).toFixed(0)}%`);
	console.log(`    13x13 @T=${T}:   ${(evalAcc(par, 13, 13, T) * 100).toFixed(0)}%   (bigger grid, same rule)`);
	console.log(`    17x17 @T=${T + 16}: ${(evalAcc(par, 17, 17, T + 16) * 100).toFixed(0)}%   (much bigger, more steps)`);
	if (process.env.PARAMS_OUT) { writeFileSync(process.env.PARAMS_OUT, JSON.stringify(Array.from(par))); console.log(`  params saved to ${process.env.PARAMS_OUT}`); }
}

main();
