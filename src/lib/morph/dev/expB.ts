// EXPERIMENT B — gradient-based morphogenesis via in-substrate forward-mode AD.
//
// A smooth, DIRECTIONAL, single-channel field-CA (a tiny NCA): each cell's
// continuous value is updated by a learnable weighted combination of itself and
// its four neighbors, plus a bias, residually, then clamped to [0,1]. We run T
// developmental steps from a seed, then a loss vs a target pattern. Carrying
// DUAL NUMBERS through the entire rollout yields d(loss)/d(each weight) by the
// chain rule — the same forward-mode AD that Exp A proved runs exactly on the
// Z80 (this rollout is just that dual-multiply primitive composed, so it is
// Zilion-runnable). We verify the gradient vs finite differences, then gradient
// -descend the weights to GROW the target. This replaces blind CA-rule evolution
// with gradient descent through development — NCA-style, but forward-mode AD, no
// backprop framework.
//
//   npx tsx src/lib/morph/dev/expB.ts

import { writeFileSync } from 'node:fs';

// ---- fixed-point dual numbers (Q-format, signed; mirrors the Z80 arithmetic) --
// Q16.16 fixed-point: enough fractional bits that the gradient survives the
// developmental rollout cleanly. On the Z80 this is 32-bit fixed-point — a
// direct extension of Exp A's 16-bit multiply (same shift-add, wider).
const SCALE = 1 << 16;
const qmul = (a: number, b: number): number => Math.trunc((a * b) / SCALE);

interface D {
	v: number; // value   (fixed-point)
	d: number; // tangent (fixed-point) = d(value)/d(the seeded parameter)
}
const K = (real: number): D => ({ v: Math.round(real * SCALE), d: 0 }); // constant
const add = (a: D, b: D): D => ({ v: a.v + b.v, d: a.d + b.d });
const sub = (a: D, b: D): D => ({ v: a.v - b.v, d: a.d - b.d });
const mul = (a: D, b: D): D => ({ v: qmul(a.v, b.v), d: qmul(a.v, b.d) + qmul(a.d, b.v) });
// Smooth saturating nonlinearity (tanh): bounds the field with NO kinks, so the
// gradient is alive everywhere (a hard clamp zeroed it on saturation, collapsing
// the disc to empty; a leaky clamp still has corners that break the finite-diff
// check). The dual carries tanh's exact derivative 1−tanh². (On the Z80 this is
// a small fixed-point table; here Math.tanh stands in for it.)
const squash = (a: D): D => {
	const tx = Math.tanh(a.v / SCALE);
	return { v: Math.round(tx * SCALE), d: Math.trunc(a.d * (1 - tx * tx)) };
};

// ---- the model ------------------------------------------------------------
const W = 12;
const H = 12;
const T = 16;
const NPARAM = 6; // [w_self, w_up, w_down, w_left, w_right, bias]

/** One developmental step: residual, directional, clamped. p are dual params. */
function step(field: D[], p: D[]): D[] {
	const nf: D[] = new Array(W * H);
	const ZERO: D = { v: 0, d: 0 };
	for (let i = 0; i < W * H; i++) nf[i] = ZERO; // borders stay 0
	for (let y = 1; y < H - 1; y++)
		for (let x = 1; x < W - 1; x++) {
			const i = y * W + x;
			const self = field[i];
			const up = field[i - W];
			const dn = field[i + W];
			const lf = field[i - 1];
			const rt = field[i + 1];
			let delta = mul(p[0], self);
			delta = add(delta, mul(p[1], up));
			delta = add(delta, mul(p[2], dn));
			delta = add(delta, mul(p[3], lf));
			delta = add(delta, mul(p[4], rt));
			delta = add(delta, p[5]); // bias
			nf[i] = squash(add(self, delta));
		}
	return nf;
}

/** Mean-squared error of the final field vs the target, as a dual number. */
function loss(field: D[], target: number[]): D {
	let L: D = { v: 0, d: 0 };
	for (let i = 0; i < W * H; i++) {
		const diff = sub(field[i], K(target[i]));
		L = add(L, mul(diff, diff));
	}
	return mul(L, K(1 / (W * H)));
}

/** Roll out T steps from the seed with params `theta`, seeding the tangent on param `wrt` (or none). */
function rollout(theta: number[], seed: number[], wrt: number): D[] {
	const p: D[] = theta.map((val, j) => ({ v: Math.round(val * SCALE), d: j === wrt ? SCALE : 0 }));
	let f: D[] = seed.map((v) => ({ v: Math.round(v * SCALE), d: 0 }));
	for (let t = 0; t < T; t++) f = step(f, p);
	return f;
}

/** Loss + its gradient w.r.t. all params via forward-mode AD (one rollout per param). */
function lossAndGrad(theta: number[], seed: number[], target: number[]): { L: number; grad: number[] } {
	const grad: number[] = [];
	let L = 0;
	for (let i = 0; i < NPARAM; i++) {
		const Ld = loss(rollout(theta, seed, i), target);
		grad.push(Ld.d / SCALE);
		L = Ld.v / SCALE;
	}
	return { L, grad };
}

function lossOnly(theta: number[], seed: number[], target: number[]): number {
	return loss(rollout(theta, seed, -1), target).v / SCALE;
}

// ---- seeds, targets, rendering -------------------------------------------
function centerSeed(): number[] {
	const s = new Array(W * H).fill(0);
	s[(H >> 1) * W + (W >> 1)] = 1;
	return s;
}
function asymSeed(): number[] {
	const s = centerSeed();
	s[(H >> 1) * W + (W >> 1) - 1] = 0.6; // break symmetry
	return s;
}
function discTarget(r: number, level = 0.9): number[] {
	const t = new Array(W * H).fill(0);
	const cx = W / 2;
	const cy = H / 2;
	for (let y = 1; y < H - 1; y++)
		for (let x = 1; x < W - 1; x++) {
			const dx = x - cx + 0.5;
			const dy = y - cy + 0.5;
			if (dx * dx + dy * dy <= r * r) t[y * W + x] = level;
		}
	return t;
}
function rampTarget(): number[] {
	// horizontal intensity ramp — ASYMMETRIC, impossible for an isotropic rule.
	const t = new Array(W * H).fill(0);
	for (let y = 1; y < H - 1; y++)
		for (let x = 1; x < W - 1; x++) t[y * W + x] = (x - 1) / (W - 3);
	return t;
}
function render(field: number[]): string {
	const ramp = ' .:-=+*#%@';
	const rows: string[] = [];
	for (let y = 0; y < H; y++) {
		let r = '';
		for (let x = 0; x < W; x++) {
			const v = Math.max(0, Math.min(1, field[y * W + x]));
			r += ramp[Math.min(ramp.length - 1, Math.floor(v * ramp.length))];
		}
		rows.push(r);
	}
	return rows.join('\n');
}
const fieldValues = (f: D[]): number[] => f.map((c) => c.v / SCALE);

// ---- (B0) gradient check --------------------------------------------------
function gradientCheck(): boolean {
	const seed = asymSeed();
	const target = rampTarget();
	// A gentle, UNSATURATED operating point so the field sits in (0,1) and the
	// gradient is genuinely nonzero — otherwise the check is vacuous.
	const theta = [0.0, 0.04, 0.03, 0.05, 0.02, 0.01];
	const { L, grad } = lossAndGrad(theta, seed, target);
	const eps = 0.001; // small enough for the high-curvature bias, large enough vs quantization
	console.log('=== B0: AD-through-time gradient vs finite differences ===');
	console.log(`L = ${L.toFixed(6)}`);
	console.log('  param     dual dL/dθ      finite-diff      |rel err|');
	let maxRel = 0;
	let maxMag = 0;
	for (let i = 0; i < NPARAM; i++) {
		const tp = theta.slice();
		tp[i] += eps;
		const tm = theta.slice();
		tm[i] -= eps;
		const fd = (lossOnly(tp, seed, target) - lossOnly(tm, seed, target)) / (2 * eps);
		const rel = Math.abs(grad[i] - fd) / (Math.abs(fd) + 1e-5);
		maxRel = Math.max(maxRel, rel);
		maxMag = Math.max(maxMag, Math.abs(grad[i]));
		console.log(`  θ${i}      ${grad[i].toExponential(3).padStart(12)}   ${fd.toExponential(3).padStart(12)}   ${rel.toFixed(4)}`);
	}
	const nonzero = maxMag > 1e-4;
	const ok = nonzero && maxRel < 0.15;
	console.log(`max relative error: ${maxRel.toFixed(4)} | max |grad|: ${maxMag.toExponential(2)} (nonzero=${nonzero}) -> ${ok ? 'PASS' : 'FAIL'}\n`);
	return ok;
}

/** Roll out and capture the field at every `everyN` steps (for visualization). */
function rolloutFrames(theta: number[], seed: number[], everyN: number): number[][] {
	const p: D[] = theta.map((val) => ({ v: Math.round(val * SCALE), d: 0 }));
	let f: D[] = seed.map((v) => ({ v: Math.round(v * SCALE), d: 0 }));
	const frames: number[][] = [fieldValues(f)];
	for (let t = 0; t < T; t++) {
		f = step(f, p);
		if ((t + 1) % everyN === 0) frames.push(fieldValues(f));
	}
	return frames;
}

// ---- (B1) grow a target by gradient descent -------------------------------
function grow(name: string, seed: number[], target: number[], iters: number): { L: number; theta: number[] } {
	let theta = [0.02, 0.05, 0.05, 0.05, 0.05, 0.01];
	let L = lossOnly(theta, seed, target);
	const L0 = L;
	for (let it = 0; it < iters; it++) {
		const g = lossAndGrad(theta, seed, target);
		L = g.L;
		// backtracking line search along -grad (proves the gradient points downhill)
		let stepSize = 2.0;
		const gnorm2 = g.grad.reduce((s, x) => s + x * x, 0);
		if (gnorm2 < 1e-12) break;
		while (stepSize > 1e-4) {
			const cand = theta.map((v, i) => v - stepSize * g.grad[i]);
			if (lossOnly(cand, seed, target) < L) {
				theta = cand;
				break;
			}
			stepSize *= 0.5;
		}
		if (stepSize <= 1e-4) break;
	}
	const finalField = fieldValues(rollout(theta, seed, -1));
	console.log(`--- ${name}: loss ${L0.toFixed(5)} -> ${L.toFixed(5)} ---`);
	console.log('target:\n' + render(target));
	console.log('grown (gradient descent through development):\n' + render(finalField));
	console.log('weights:', theta.map((v) => v.toFixed(3)).join(', '), '\n');
	return { L, theta };
}

function main() {
	const ok = gradientCheck();
	if (!ok) {
		console.error('FAIL: AD-through-time gradient is wrong; stopping before descent.');
		process.exit(1);
	}
	console.log('=== B1: grow targets by gradient descent through development ===\n');
	const disc = grow('disc', asymSeed(), discTarget(3.5), 120);
	const ramp = grow('horizontal ramp (ASYMMETRIC — impossible for an isotropic rule)', asymSeed(), rampTarget(), 200);

	console.log('SUMMARY: gradient-based morphogenesis reduced loss on both targets.');
	console.log(`disc ${disc.L.toFixed(5)} | ramp ${ramp.L.toFixed(5)}`);
	console.log('The gradient came from differentiating the DEVELOPMENTAL ROLLOUT — not from evolution.');

	// Dump visualization data (development frames + target/grown) if asked.
	const out = process.env.EXPB_VIZ;
	if (out) {
		const data = {
			W, H, T,
			disc: {
				target: discTarget(3.5),
				frames: rolloutFrames(disc.theta, asymSeed(), 2),
				weights: disc.theta
			},
			ramp: {
				target: rampTarget(),
				frames: rolloutFrames(ramp.theta, asymSeed(), 2),
				weights: ramp.theta
			}
		};
		writeFileSync(out, JSON.stringify(data));
		console.log(`\nwrote viz data -> ${out}`);
	}
}

main();
