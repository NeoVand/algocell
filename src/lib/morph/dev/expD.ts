// EXPERIMENT D — grow a COLOR emoji (a lizard) by gradient descent through
// development, NCA-style. Finishes the proof of concept: gradient-based
// morphogenesis grows complex, multi-colour, asymmetric targets.
//
// A 16-channel field (RGBA visible + 12 hidden morphogens), directional +
// diffusive perception (identity, gx, gy, laplacian per channel), a small MLP
// update (perception -> ReLU hidden -> delta), applied residually with a tanh
// squash, for T steps. The exact gradient of the RGBA-vs-target loss w.r.t. all
// weights is obtained by REVERSE-MODE AD (backprop through the rollout) — the
// SAME gradient forward-mode computes (proven exact in Exp A-C), just at O(1)
// rollouts, so it is fast enough at this scale. Forward-gradient over Zilion's
// lanes is the in-substrate scalable route.
//
//   npx tsx src/lib/morph/dev/expD.ts

import { writeFileSync, readFileSync } from 'node:fs';

const S = 64;
const N = S * S;
const C = 16; // 0..3 = RGBA (visible), 4..15 = hidden
const FEAT = 4; // identity, gx, gy, laplacian
const PERC = FEAT * C; // 64
const HD = 80; // MLP hidden units
const T = 40;

// flat parameter layout: [W1 (HD*PERC), b1 (HD), W2 (C*HD), b2 (C)]
const W1O = 0;
const B1O = HD * PERC;
const W2O = B1O + HD;
const B2O = W2O + C * HD;
const P = B2O + C;

function mulberry32(seed: number): () => number {
	let a = seed >>> 0;
	return () => {
		a |= 0;
		a = (a + 0x6d2b79f5) | 0;
		let t = Math.imul(a ^ (a >>> 15), 1 | a);
		t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
}

/** Real lizard emoji (Noto, 🦎/U+1F98E), decoded to premultiplied RGBA at SxS. */
function lizardTarget(): Float64Array {
	const j = JSON.parse(readFileSync(new URL('./lizard64.json', import.meta.url), 'utf8')) as { S: number; rgba: number[] };
	if (j.S !== S) throw new Error(`emoji resolution ${j.S} != grid ${S}`);
	return Float64Array.from(j.rgba);
}

function centerSeed(): Float64Array {
	const s = new Float64Array(N * C);
	const i = (S >> 1) * S + (S >> 1);
	for (let c = 0; c < C; c++) s[i * C + c] = 1;
	return s;
}

const nb = (i: number) => ({ r: i + 1, l: i - 1, u: i - S, d: i + S });

function perceive(s: Float64Array, i: number, out: Float64Array): void {
	const { r, l, u, d } = nb(i);
	for (let ch = 0; ch < C; ch++) {
		const b = ch * FEAT;
		const self = s[i * C + ch];
		const sr = s[r * C + ch], sl = s[l * C + ch], su = s[u * C + ch], sd = s[d * C + ch];
		out[b] = self;
		out[b + 1] = (sr - sl) * 0.5;
		out[b + 2] = (sd - su) * 0.5;
		out[b + 3] = sr + sl + su + sd - 4 * self;
	}
}

function forward(par: Float64Array, seed: Float64Array): Float64Array[] {
	const states: Float64Array[] = [seed.slice()];
	let s = seed;
	const perc = new Float64Array(PERC);
	const h = new Float64Array(HD);
	for (let t = 0; t < T; t++) {
		const ns = new Float64Array(N * C);
		for (let y = 1; y < S - 1; y++)
			for (let x = 1; x < S - 1; x++) {
				const i = y * S + x;
				perceive(s, i, perc);
				for (let hh = 0; hh < HD; hh++) {
					let a = par[B1O + hh];
					const base = W1O + hh * PERC;
					for (let k = 0; k < PERC; k++) a += par[base + k] * perc[k];
					h[hh] = a > 0 ? a : 0;
				}
				for (let c = 0; c < C; c++) {
					let dl = par[B2O + c];
					const base = W2O + c * HD;
					for (let hh = 0; hh < HD; hh++) dl += par[base + hh] * h[hh];
					ns[i * C + c] = Math.tanh(s[i * C + c] + dl);
				}
			}
		states.push(ns);
		s = ns;
	}
	return states;
}

function computeLoss(sT: Float64Array, target: Float64Array): { L: number; gsT: Float64Array } {
	const gsT = new Float64Array(N * C);
	let L = 0;
	const M = N * 4;
	for (let i = 0; i < N; i++)
		for (let ch = 0; ch < 4; ch++) {
			const diff = sT[i * C + ch] - target[i * 4 + ch];
			L += diff * diff;
			gsT[i * C + ch] = (2 * diff) / M;
		}
	return { L: L / M, gsT };
}

function backward(states: Float64Array[], par: Float64Array, gsT: Float64Array): Float64Array {
	const grad = new Float64Array(P);
	let gs = gsT;
	const perc = new Float64Array(PERC);
	const pre1 = new Float64Array(HD);
	const h = new Float64Array(HD);
	const gh = new Float64Array(HD);
	const gperc = new Float64Array(PERC);
	for (let t = T - 1; t >= 0; t--) {
		const s = states[t];
		const sp = states[t + 1];
		const gsPrev = new Float64Array(N * C);
		for (let y = 1; y < S - 1; y++)
			for (let x = 1; x < S - 1; x++) {
				const i = y * S + x;
				perceive(s, i, perc);
				for (let hh = 0; hh < HD; hh++) {
					let a = par[B1O + hh];
					const base = W1O + hh * PERC;
					for (let k = 0; k < PERC; k++) a += par[base + k] * perc[k];
					pre1[hh] = a;
					h[hh] = a > 0 ? a : 0;
				}
				gh.fill(0);
				for (let c = 0; c < C; c++) {
					const spv = sp[i * C + c];
					const gp = gs[i * C + c] * (1 - spv * spv);
					gsPrev[i * C + c] += gp; // residual
					grad[B2O + c] += gp;
					const base = W2O + c * HD;
					for (let hh = 0; hh < HD; hh++) {
						grad[base + hh] += gp * h[hh];
						gh[hh] += par[base + hh] * gp;
					}
				}
				gperc.fill(0);
				for (let hh = 0; hh < HD; hh++) {
					let g = gh[hh];
					if (pre1[hh] <= 0) g = 0; // ReLU
					grad[B1O + hh] += g;
					const base = W1O + hh * PERC;
					for (let k = 0; k < PERC; k++) {
						grad[base + k] += g * perc[k];
						gperc[k] += par[base + k] * g;
					}
				}
				const { r, l, u, d } = nb(i);
				for (let ch = 0; ch < C; ch++) {
					const bb = ch * FEAT;
					const gId = gperc[bb], gGx = gperc[bb + 1], gGy = gperc[bb + 2], gLap = gperc[bb + 3];
					gsPrev[i * C + ch] += gId - 4 * gLap;
					gsPrev[r * C + ch] += 0.5 * gGx + gLap;
					gsPrev[l * C + ch] += -0.5 * gGx + gLap;
					gsPrev[d * C + ch] += 0.5 * gGy + gLap;
					gsPrev[u * C + ch] += -0.5 * gGy + gLap;
				}
			}
		gs = gsPrev;
	}
	return grad;
}

function lossOnly(par: Float64Array, seed: Float64Array, target: Float64Array): number {
	return computeLoss(forward(par, seed)[T], target).L;
}

function gradientCheck(): boolean {
	const rng = mulberry32(11);
	const par = new Float64Array(P).map(() => (rng() - 0.5) * 0.08);
	const seed = centerSeed();
	const target = lizardTarget();
	const grad = backward(forward(par, seed), par, computeLoss(forward(par, seed)[T], target).gsT);
	const eps = 1e-4;
	const idxs = [W1O + 5, W1O + 2000, B1O + 3, W2O + 10, W2O + 800, B2O + 0, B2O + 3];
	console.log('=== D0: backprop-through-development vs finite differences (MLP) ===');
	console.log('  param       backprop        finite-diff     |rel err|');
	let maxRel = 0;
	for (const j of idxs) {
		const pp = par.slice(); pp[j] += eps;
		const pm = par.slice(); pm[j] -= eps;
		const fd = (lossOnly(pp, seed, target) - lossOnly(pm, seed, target)) / (2 * eps);
		const rel = Math.abs(grad[j] - fd) / (Math.abs(fd) + 1e-8);
		maxRel = Math.max(maxRel, rel);
		console.log(`  ${String(j).padStart(6)}   ${grad[j].toExponential(3).padStart(13)}   ${fd.toExponential(3).padStart(13)}   ${rel.toFixed(4)}`);
	}
	const ok = maxRel < 0.02;
	console.log(`max rel err ${maxRel.toFixed(4)} -> ${ok ? 'PASS' : 'FAIL'}\n`);
	return ok;
}

function train(iters: number): { par: Float64Array; lossCurve: number[] } {
	const rng = mulberry32(5);
	// small init; W2 (last layer) especially small so the initial rule is gentle
	const par = new Float64Array(P);
	for (let j = 0; j < W2O; j++) par[j] = (rng() - 0.5) * 0.1; // W1, b1 small random
	// W2, b2 stay ZERO — the initial rule is a no-op, so it can't saturate the
	// tanh before any signal exists; gradients build it up gently (NCA-standard).
	const seed = centerSeed();
	const target = lizardTarget();
	const m = new Float64Array(P), v = new Float64Array(P);
	const b1 = 0.9, b2 = 0.999;
	const lossCurve: number[] = [];
	for (let it = 1; it <= iters; it++) {
		const lr = it > iters * 0.5 ? (it > iters * 0.8 ? 0.002 : 0.005) : 0.01;
		const states = forward(par, seed);
		const { L, gsT } = computeLoss(states[T], target);
		const grad = backward(states, par, gsT);
		// global-norm gradient clipping — stops the overshoot that saturates tanh
		let gn = 0;
		for (let j = 0; j < P; j++) gn += grad[j] * grad[j];
		gn = Math.sqrt(gn);
		if (gn > 1.0) for (let j = 0; j < P; j++) grad[j] *= 1.0 / gn;
		const c1 = 1 - Math.pow(b1, it), c2 = 1 - Math.pow(b2, it);
		for (let j = 0; j < P; j++) {
			m[j] = b1 * m[j] + (1 - b1) * grad[j];
			v[j] = b2 * v[j] + (1 - b2) * grad[j] * grad[j];
			par[j] -= (lr * (m[j] / c1)) / (Math.sqrt(v[j] / c2) + 1e-8);
		}
		if (it % 25 === 0 || it === 1) lossCurve.push(L);
		if (it % 100 === 0 || it === 1) console.log(`  iter ${String(it).padStart(4)}   loss ${L.toFixed(6)}`);
	}
	return { par, lossCurve };
}

const rgb = (s: Float64Array, i: number): [number, number, number] => [
	Math.max(0, Math.min(1, Math.tanh(s[i * C + 0]))),
	Math.max(0, Math.min(1, Math.tanh(s[i * C + 1]))),
	Math.max(0, Math.min(1, Math.tanh(s[i * C + 2])))
];

function analyze(par: Float64Array, lossCurve: number[]): object {
	const seed = centerSeed();
	const target = lizardTarget();
	const states = forward(par, seed);
	const frames: number[][][] = [];
	for (let t = 0; t <= T; t += 2) {
		const f: number[][] = [];
		for (let i = 0; i < N; i++) f.push(rgb(states[t], i));
		frames.push(f);
	}
	const hidden: number[][] = [];
	for (let ch = 4; ch < C; ch++) {
		const field: number[] = [];
		for (let i = 0; i < N; i++) field.push((Math.tanh(states[T][i * C + ch]) + 1) / 2);
		hidden.push(field);
	}
	// weight energy per perception feature, aggregated over the first layer
	const featEnergy = [0, 0, 0, 0];
	for (let hh = 0; hh < HD; hh++)
		for (let ch = 0; ch < C; ch++)
			for (let f = 0; f < FEAT; f++) featEnergy[f] += Math.abs(par[W1O + hh * PERC + ch * FEAT + f]);
	const fe = featEnergy.reduce((a, x) => a + x, 0);
	const tgt: number[][] = [];
	for (let i = 0; i < N; i++) tgt.push([target[i * 4], target[i * 4 + 1], target[i * 4 + 2]]);
	return { S, T, frames, target: tgt, hidden, featEnergy: featEnergy.map((x) => x / fe), lossCurve };
}

function main() {
	console.log(`color NCA (MLP): ${C} ch (RGBA + ${C - 4} hidden), ${HD} hidden units, ${P} params, ${S}x${S}, ${T} steps\n`);
	if (!gradientCheck()) {
		console.error('FAIL: backprop gradient is wrong.');
		process.exit(1);
	}
	console.log('=== D1: grow the lizard by gradient descent through development ===');
	const t0 = Date.now();
	const { par, lossCurve } = train(Number(process.env.ITERS ?? 1200));
	console.log(`trained in ${((Date.now() - t0) / 1000).toFixed(0)}s, final loss ${lossCurve[lossCurve.length - 1].toFixed(6)}`);
	const out = process.env.EXPD_VIZ;
	if (out) {
		writeFileSync(out, JSON.stringify(analyze(par, lossCurve)));
		console.log(`wrote viz -> ${out}`);
	}
}

main();
