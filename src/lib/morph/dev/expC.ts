// EXPERIMENT C — grow the letter F by GRADIENT DESCENT through development.
//
// A multi-channel neural cellular automaton (NCA): each cell holds C channels
// (channel 0 visible, 1..C-1 hidden "morphogens"). Perception is directional —
// identity + signed gradients gx=right-left, gy=down-up per channel — so the
// rule is NOT reflection-equivariant and can break symmetry from a symmetric
// seed (an isotropic rule provably cannot; that is why evolution stalled on the
// F). The update is a learned linear map over perception + bias, applied
// residually with a tanh squash, for T steps. We differentiate the whole
// rollout by FORWARD-MODE AD (dual numbers, here in float with typed arrays;
// Exp B proved the same survives Q16.16 fixed-point on the Z80) to get the exact
// gradient of the final-vs-target loss w.r.t. every weight, and descend it.
//
//   npx tsx src/lib/morph/dev/expC.ts

import { writeFileSync } from 'node:fs';

const Wd = 12;
const Hd = 12;
const N = Wd * Hd;
const C = 6; // channels: 1 visible + 5 hidden
const PERC = 3 * C; // [self, gx, gy] per channel
const T = 20; // developmental steps
const P = C * PERC + C; // weights (C x PERC) + biases (C)
const BIAS0 = C * PERC; // offset of the biases in the param vector

// ---- deterministic PRNG ---------------------------------------------------
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

// ---- forward-mode dual rollout (value + one tangent direction) -------------
// vF/tF: field value + tangent, length N*C. vP/tP: param value + tangent, length P.
function step(vF: Float64Array, tF: Float64Array, vP: Float64Array, tP: Float64Array): [Float64Array, Float64Array] {
	const nvF = new Float64Array(N * C);
	const ntF = new Float64Array(N * C);
	const pv = new Float64Array(PERC);
	const pt = new Float64Array(PERC);
	for (let y = 1; y < Hd - 1; y++)
		for (let x = 1; x < Wd - 1; x++) {
			const i = y * Wd + x;
			const r = i + 1;
			const l = i - 1;
			const u = i - Wd;
			const dn = i + Wd;
			for (let ch = 0; ch < C; ch++) {
				const b = ch * 3;
				pv[b] = vF[i * C + ch];
				pt[b] = tF[i * C + ch];
				pv[b + 1] = vF[r * C + ch] - vF[l * C + ch]; // gx
				pt[b + 1] = tF[r * C + ch] - tF[l * C + ch];
				pv[b + 2] = vF[dn * C + ch] - vF[u * C + ch]; // gy
				pt[b + 2] = tF[dn * C + ch] - tF[u * C + ch];
			}
			for (let c = 0; c < C; c++) {
				let dv = vP[BIAS0 + c];
				let dt = tP[BIAS0 + c];
				const row = c * PERC;
				for (let k = 0; k < PERC; k++) {
					const wv = vP[row + k];
					dv += wv * pv[k];
					dt += wv * pt[k] + tP[row + k] * pv[k];
				}
				const preV = vF[i * C + c] + dv;
				const preT = tF[i * C + c] + dt;
				const th = Math.tanh(preV);
				nvF[i * C + c] = th;
				ntF[i * C + c] = (1 - th * th) * preT;
			}
		}
	return [nvF, ntF];
}

function rollout(vP: Float64Array, tP: Float64Array, seedV: Float64Array): [Float64Array, Float64Array] {
	let vF: Float64Array = seedV.slice();
	let tF: Float64Array = new Float64Array(N * C); // seed has no parameter dependence
	for (let t = 0; t < T; t++) [vF, tF] = step(vF, tF, vP, tP);
	return [vF, tF];
}

/** Loss = MSE(visible channel 0, target); returns value and its directional derivative. */
function loss(vF: Float64Array, tF: Float64Array, target: Float64Array): [number, number] {
	let Lv = 0;
	let Lt = 0;
	for (let i = 0; i < N; i++) {
		const diff = vF[i * C] - target[i];
		Lv += diff * diff;
		Lt += 2 * diff * tF[i * C];
	}
	return [Lv / N, Lt / N];
}

/** Full gradient by forward-mode AD: one rollout per parameter. */
function fullGradient(vP: Float64Array, seedV: Float64Array, target: Float64Array): { L: number; grad: Float64Array } {
	const grad = new Float64Array(P);
	const tP = new Float64Array(P);
	let L = 0;
	for (let j = 0; j < P; j++) {
		tP.fill(0);
		tP[j] = 1;
		const [vF, tF] = rollout(vP, tP, seedV);
		const [Lv, Lt] = loss(vF, tF, target);
		grad[j] = Lt;
		L = Lv;
	}
	return { L, grad };
}

function lossOnly(vP: Float64Array, seedV: Float64Array, target: Float64Array): number {
	const zero = new Float64Array(P);
	const [vF, tF] = rollout(vP, zero, seedV);
	return loss(vF, tF, target)[0];
}

// ---- seed, target, visible field ------------------------------------------
function centerSeed(): Float64Array {
	const s = new Float64Array(N * C);
	const i = (Hd >> 1) * Wd + (Wd >> 1);
	for (let c = 0; c < C; c++) s[i * C + c] = 1; // all channels lit at the seed
	return s;
}
function letterFTarget(): Float64Array {
	const rows = ['#####', '#....', '####.', '#....', '#....', '#....'];
	const t = new Float64Array(N);
	const ox = (Wd - 5) >> 1;
	const oy = (Hd - rows.length) >> 1;
	for (let r = 0; r < rows.length; r++)
		for (let c = 0; c < rows[r].length; c++)
			if (rows[r][c] === '#') t[(oy + r) * Wd + (ox + c)] = 1;
	return t;
}
const visible = (vF: Float64Array): number[] => {
	const out: number[] = [];
	for (let i = 0; i < N; i++) out.push(vF[i * C]);
	return out;
};
function render(field: number[]): string {
	const ramp = ' .:-=+*#%@';
	const rows: string[] = [];
	for (let y = 0; y < Hd; y++) {
		let r = '';
		for (let x = 0; x < Wd; x++) {
			const v = Math.max(0, Math.min(1, field[y * Wd + x]));
			r += ramp[Math.min(ramp.length - 1, Math.floor(v * ramp.length))];
		}
		rows.push(r);
	}
	return rows.join('\n');
}

// ---- (C0) gradient check on the multi-channel model -----------------------
function gradientCheck(): boolean {
	const rng = mulberry32(7);
	const vP = new Float64Array(P).map(() => (rng() - 0.5) * 0.3);
	const seed = centerSeed();
	const target = letterFTarget();
	const { grad } = fullGradient(vP, seed, target);
	const eps = 1e-3;
	// spot-check a spread of parameters against finite differences
	const idxs = [0, 3, 7, 20, 40, BIAS0, BIAS0 + 2, P - 1];
	console.log('=== C0: multi-channel AD-through-time vs finite differences ===');
	console.log('  param     dual dL/dθ      finite-diff      |rel err|');
	let maxRel = 0;
	let maxMag = 0;
	for (const j of idxs) {
		const vp = vP.slice();
		vp[j] += eps;
		const vm = vP.slice();
		vm[j] -= eps;
		const fd = (lossOnly(vp, seed, target) - lossOnly(vm, seed, target)) / (2 * eps);
		const rel = Math.abs(grad[j] - fd) / (Math.abs(fd) + 1e-6);
		maxRel = Math.max(maxRel, rel);
		maxMag = Math.max(maxMag, Math.abs(grad[j]));
		console.log(`  θ${String(j).padStart(3)}   ${grad[j].toExponential(3).padStart(12)}   ${fd.toExponential(3).padStart(12)}   ${rel.toFixed(4)}`);
	}
	const ok = maxMag > 1e-4 && maxRel < 0.05;
	console.log(`max rel err ${maxRel.toFixed(4)} | max |grad| ${maxMag.toExponential(2)} -> ${ok ? 'PASS' : 'FAIL'}\n`);
	return ok;
}

// ---- (C1) grow the F by Adam on the forward-mode gradient ------------------
function growF(iters: number): { grownVisible: number[]; frames: number[][]; loss: number; target: number[] } {
	const rng = mulberry32(3);
	const vP = new Float64Array(P).map(() => (rng() - 0.5) * 0.2);
	const seed = centerSeed();
	const target = letterFTarget();
	// Adam
	const m = new Float64Array(P);
	const v = new Float64Array(P);
	const lr = 0.02;
	const b1 = 0.9;
	const b2 = 0.999;
	let L = lossOnly(vP, seed, target);
	const L0 = L;
	for (let it = 1; it <= iters; it++) {
		const g = fullGradient(vP, seed, target);
		L = g.L;
		for (let j = 0; j < P; j++) {
			m[j] = b1 * m[j] + (1 - b1) * g.grad[j];
			v[j] = b2 * v[j] + (1 - b2) * g.grad[j] * g.grad[j];
			const mh = m[j] / (1 - Math.pow(b1, it));
			const vh = v[j] / (1 - Math.pow(b2, it));
			vP[j] -= (lr * mh) / (Math.sqrt(vh) + 1e-8);
		}
		if (it % 50 === 0 || it === 1) console.log(`  iter ${String(it).padStart(4)}   loss ${L.toFixed(5)}`);
	}
	// capture development frames of the visible channel
	const zero = new Float64Array(P);
	let vF: Float64Array = seed.slice();
	let tF: Float64Array = new Float64Array(N * C);
	const frames: number[][] = [visible(vF)];
	for (let t = 0; t < T; t++) {
		[vF, tF] = step(vF, tF, vP, zero);
		if ((t + 1) % 2 === 0) frames.push(visible(vF));
	}
	console.log(`\ngrown F: loss ${L0.toFixed(5)} -> ${L.toFixed(5)}`);
	return { grownVisible: visible(vF), frames, loss: L, target: Array.from(target) };
}

function main() {
	console.log(`model: ${C} channels, ${P} params, ${Wd}x${Hd} grid, ${T} steps\n`);
	if (!gradientCheck()) {
		console.error('FAIL: multi-channel AD-through-time gradient is wrong.');
		process.exit(1);
	}
	console.log('=== C1: grow the letter F by gradient descent through development ===');
	const res = growF(400);
	console.log('\ntarget:\n' + render(res.target));
	console.log('\ngrown (channel 0):\n' + render(res.grownVisible));

	const out = process.env.EXPC_VIZ;
	if (out) {
		writeFileSync(out, JSON.stringify({ W: Wd, H: Hd, T, frames: res.frames, grown: res.grownVisible, target: res.target, loss: res.loss }));
		console.log(`\nwrote viz -> ${out}`);
	}
}

main();
