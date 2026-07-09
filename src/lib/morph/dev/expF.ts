// EXPERIMENT F — SELF-REPAIR (the paper's headline).
//
// A grown XOR gate that, when a patch of its own structure is destroyed
// mid-computation, REGROWS and still computes XOR. This is the demo that
// separates developmental computation from both NCA (images only, no
// computation) and classic A-life (no gradients): a machine, grown by gradient
// descent through development, that heals.
//
// Built on E1 (expE.ts). Three warm-started stages, gentle refinement lr:
//   A  COMPUTE  — the clean XOR gate (distance curriculum), read once at T_GROW.
//   B  PERSIST  — score the output over a WINDOW so it's a stable attractor,
//                 not a one-shot spike (output must HOLD the answer).
//   C  REPAIR   — zero a random patch mid-rollout, then require the output to
//                 return to correct. Backprop flows through the damage mask.
//
//   npx tsx src/lib/morph/dev/expF.ts
//   ITERS=… FVIZ=path.json npx tsx src/lib/morph/dev/expF.ts

import { writeFileSync } from 'node:fs';

const SW = 9, SH = 9, N = SW * SH;
const C = 12, FEAT = 4, PERC = FEAT * C, HD = 48;
const W1O = 0, B1O = HD * PERC, W2O = B1O + HD, B2O = W2O + C * HD, P = B2O + C;

// Timeline (steps): grow & compute, then hold, then damage, then heal.
const T_GROW = 24;
const T_HOLD = 8; // output must stay correct across this window (persistence)
const T_REPAIR = 18; // steps allowed to heal after damage
const DMG_AT = T_GROW + T_HOLD; // damage strikes after it has computed and held
const T_TOTAL = DMG_AT + T_REPAIR;
const REPAIR_TAIL = 5; // the last few steps after damage must be correct again
const DMG_SIZE = Number(process.env.DMG_SIZE ?? 3); // side of the square damage patch

const iy = SH >> 1, GATE_IC = 2, OUT_COL = SW - 2; // inputs at col 2 (rows iy±1), output at col 7

function mulberry32(seed: number): () => number {
	let a = seed >>> 0;
	return () => {
		a |= 0; a = (a + 0x6d2b79f5) | 0;
		let t = Math.imul(a ^ (a >>> 15), 1 | a);
		t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
}

const inputCells = [(iy - 1) * SW + GATE_IC, (iy + 1) * SW + GATE_IC];
let OUTPUT = iy * SW + OUT_COL; // mutable: retargeted during the distance curriculum, then fixed
const CASES = [
	{ in: [0, 0], tgt: 0 },
	{ in: [0, 1], tgt: 1 },
	{ in: [1, 0], tgt: 1 },
	{ in: [1, 1], tgt: 0 }
];

/** Live substrate: interior cells alive (hidden channels = 1), signal = 0. */
function seedGrid(inputs: number[]): Float64Array {
	const s = new Float64Array(N * C);
	for (let y = 1; y < SH - 1; y++)
		for (let x = 1; x < SW - 1; x++) {
			const i = y * SW + x;
			for (let c = 1; c < C; c++) s[i * C + c] = 1;
		}
	clampInputs(s, inputs);
	return s;
}
function clampInputs(f: Float64Array, inputs: number[]): void {
	for (let k = 0; k < inputCells.length; k++) f[inputCells[k] * C + 0] = inputs[k];
}

const nb = (i: number) => ({ r: i + 1, l: i - 1, u: i - SW, d: i + SW });

function perceive(s: Float64Array, i: number, out: Float64Array): void {
	const { r, l, u, d } = nb(i);
	for (let ch = 0; ch < C; ch++) {
		const b = ch * FEAT, self = s[i * C + ch];
		const sr = s[r * C + ch], sl = s[l * C + ch], su = s[u * C + ch], sd = s[d * C + ch];
		out[b] = self;
		out[b + 1] = (sr - sl) * 0.5;
		out[b + 2] = (sd - su) * 0.5;
		out[b + 3] = sr + sl + su + sd - 4 * self;
	}
}

/** A square damage patch as a 0/1 keep-mask over cells (1 = keep, 0 = destroyed). */
function damageMask(cx: number, cy: number, size: number): Uint8Array {
	const mask = new Uint8Array(N).fill(1);
	const h = size >> 1;
	for (let y = cy - h; y <= cy - h + size - 1; y++)
		for (let x = cx - h; x <= cx - h + size - 1; x++)
			if (x >= 0 && x < SW && y >= 0 && y < SH) mask[y * SW + x] = 0;
	return mask;
}

interface Schedule { w: Float64Array; nSteps: number; damageAt: number; }

/** Readout weights over steps (each scored step wants output == case target). */
function schedule(mode: 'compute' | 'persist' | 'repair'): Schedule {
	const w = new Float64Array(T_TOTAL + 1);
	if (mode === 'compute') { w[T_GROW] = 1; return { w, nSteps: T_GROW, damageAt: -1 }; }
	// hold window [T_GROW .. DMG_AT-1] carries half the weight in every mode
	const hs = T_GROW, he = DMG_AT - 1, hn = he - hs + 1;
	for (let s = hs; s <= he; s++) w[s] = 0.5 / hn;
	if (mode === 'persist') { // no damage: hold answer over the whole tail too
		const ts = DMG_AT, te = T_TOTAL, tn = te - ts + 1;
		for (let s = ts; s <= te; s++) w[s] = 0.5 / tn;
		return { w, nSteps: T_TOTAL, damageAt: -1 };
	}
	// repair: last REPAIR_TAIL steps (post-damage) must be correct again
	const rs = T_TOTAL - REPAIR_TAIL + 1, re = T_TOTAL, rn = re - rs + 1;
	for (let s = rs; s <= re; s++) w[s] += 0.5 / rn;
	return { w, nSteps: T_TOTAL, damageAt: DMG_AT };
}

/** Forward rollout. Signal clamped at inputs every step; damage mask (if any)
 *  applied at damageAt. Returns states[0..nSteps]. */
function forward(par: Float64Array, inputs: number[], sched: Schedule, mask: Uint8Array | null): Float64Array[] {
	const s0 = seedGrid(inputs);
	const states: Float64Array[] = [s0];
	let s = s0;
	const perc = new Float64Array(PERC), h = new Float64Array(HD);
	for (let t = 0; t < sched.nSteps; t++) {
		const ns = new Float64Array(N * C);
		for (let y = 1; y < SH - 1; y++)
			for (let x = 1; x < SW - 1; x++) {
				const i = y * SW + x;
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
		if (mask && t + 1 === sched.damageAt) // destroy a patch: all channels -> 0
			for (let i = 0; i < N; i++) if (mask[i] === 0) for (let c = 0; c < C; c++) ns[i * C + c] = 0;
		clampInputs(ns, inputs); // inputs are external — they survive damage
		states.push(ns);
		s = ns;
	}
	return states;
}

/** Loss + gradient over all 4 cases for one schedule / damage mask. */
function lossAndGrad(par: Float64Array, sched: Schedule, mask: Uint8Array | null): { L: number; grad: Float64Array; outs: number[][] } {
	const grad = new Float64Array(P);
	let L = 0; const outs: number[][] = [];
	const perc = new Float64Array(PERC), pre1 = new Float64Array(HD), hbuf = new Float64Array(HD), gh = new Float64Array(HD), gperc = new Float64Array(PERC);
	for (const cse of CASES) {
		const states = forward(par, cse.in, sched, mask);
		// record the output trajectory at scored steps (for reporting)
		const traj: number[] = [];
		for (let step = 0; step <= sched.nSteps; step++) if (sched.w[step] > 0) traj.push(states[step][OUTPUT * C + 0]);
		outs.push(traj);
		// per-case loss over scored steps
		for (let step = 0; step <= sched.nSteps; step++) {
			if (sched.w[step] <= 0) continue;
			const diff = states[step][OUTPUT * C + 0] - cse.tgt;
			L += (sched.w[step] * diff * diff) / CASES.length;
		}
		// backprop
		let gs = new Float64Array(N * C); // grad wrt states[nSteps], grows as we descend
		for (let t = sched.nSteps - 1; t >= 0; t--) {
			const step = t + 1;
			if (sched.w[step] > 0) // readout at this step
				gs[OUTPUT * C + 0] += (2 * sched.w[step] * (states[step][OUTPUT * C + 0] - cse.tgt)) / CASES.length;
			for (const ic of inputCells) gs[ic * C + 0] = 0; // clamp: no grad through the override
			if (mask && step === sched.damageAt) // damage: no grad through destroyed cells
				for (let i = 0; i < N; i++) if (mask[i] === 0) for (let c = 0; c < C; c++) gs[i * C + c] = 0;
			const s = states[t], sp = states[step];
			const gsPrev = new Float64Array(N * C);
			for (let y = 1; y < SH - 1; y++)
				for (let x = 1; x < SW - 1; x++) {
					const i = y * SW + x;
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
					const { r, l, u, d } = nb(i);
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
	const sched = schedule('repair');
	const mask = damageMask(4, iy, DMG_SIZE); // fixed patch so the check is deterministic
	const { grad } = lossAndGrad(par, sched, mask);
	const eps = 1e-4;
	let maxRel = 0;
	for (const j of [10, 500, B1O + 2, W2O + 5, B2O + 0]) {
		const pp = par.slice(); pp[j] += eps;
		const pm = par.slice(); pm[j] -= eps;
		const fd = (lossAndGrad(pp, sched, mask).L - lossAndGrad(pm, sched, mask).L) / (2 * eps);
		maxRel = Math.max(maxRel, Math.abs(grad[j] - fd) / (Math.abs(fd) + 1e-8));
	}
	console.log(`  gradient check (repair schedule + damage): max rel err ${maxRel.toExponential(2)} -> ${maxRel < 0.02 ? 'PASS' : 'FAIL'}`);
	return maxRel < 0.02;
}

/** Candidate damage patches, spread across the computational structure. */
function candidatePatches(): { cx: number; cy: number }[] {
	const list: { cx: number; cy: number }[] = [];
	for (let cx = GATE_IC + 1; cx <= OUT_COL; cx++) for (const cy of [iy - 1, iy, iy + 1]) list.push({ cx, cy });
	return list;
}

/** One Adam training run. `dmg`: if true, each iteration damages a random patch. */
function train(sched: Schedule, iters: number, init: Float64Array | undefined, dmg: boolean, seed = 7): { par: Float64Array; L: number } {
	let par: Float64Array;
	if (init) par = init.slice();
	else {
		const rng = mulberry32(seed);
		par = new Float64Array(P);
		for (let j = 0; j < P; j++) par[j] = (rng() - 0.5) * 0.12;
		for (let j = W2O; j < P; j++) par[j] *= 0.5;
	}
	const warm = init !== undefined;
	const lrHi = warm ? 0.003 : 0.01, lrLo = warm ? 0.0006 : 0.004;
	const patches = candidatePatches();
	const m = new Float64Array(P), v = new Float64Array(P), b1 = 0.9, b2 = 0.999;
	let bestLoss = Infinity, bestPar = par.slice();
	for (let it = 1; it <= iters; it++) {
		const cos = 0.5 * (1 + Math.cos(Math.PI * (it / iters)));
		const lr = Math.min(1, it / 20) * (lrLo + (lrHi - lrLo) * cos);
		// random damage patch this iteration (deterministic per (seed,it))
		let mask: Uint8Array | null = null;
		if (dmg) { const rp = mulberry32(seed * 1000 + it)(); const pc = patches[Math.floor(rp * patches.length)]; mask = damageMask(pc.cx, pc.cy, DMG_SIZE); }
		const { L, grad } = lossAndGrad(par, sched, mask);
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

/** Stage A: the clean XOR gate via a distance curriculum (as in E1). */
function computeCurriculum(iters: number): Float64Array {
	console.log('  [A] COMPUTE — clean XOR gate (distance curriculum)');
	let par: Float64Array | undefined;
	const sched = schedule('compute');
	for (let dist = 1; dist <= OUT_COL - GATE_IC; dist++) {
		OUTPUT = iy * SW + (GATE_IC + dist); // retarget output for this curriculum stage
		const restarts = dist === 1 ? 6 : 1;
		const si = dist === 1 ? Math.min(iters, 300) : iters;
		let best: { par: Float64Array; L: number } | null = null;
		for (let s = 0; s < restarts; s++) { const r = train(sched, si, par, false, 7 + s * 101); if (!best || r.L < best.L) best = r; }
		par = best!.par;
		console.log(`    d=${dist}: loss ${best!.L.toFixed(4)}`);
	}
	OUTPUT = iy * SW + OUT_COL; // fix output at the far cell for persist/repair
	return par!;
}

function main() {
	console.log(`self-repair: ${SW}x${SH}, C=${C}, ${P} params, timeline grow ${T_GROW} + hold ${T_HOLD} + repair ${T_REPAIR} = ${T_TOTAL}, damage ${DMG_SIZE}x${DMG_SIZE} @ step ${DMG_AT}`);
	if (!gradientCheck()) { console.error('FAIL: gradient wrong'); process.exit(1); }
	const iters = Number(process.env.ITERS ?? 700);

	const parA = computeCurriculum(iters);
	report('after COMPUTE', parA);

	console.log('  [B] PERSIST — output must hold the answer (stable attractor)');
	const parB = train(schedule('persist'), iters, parA, false).par;
	report('after PERSIST', parB);

	console.log('  [C] REPAIR — damage a random patch mid-run, then heal');
	const parC = train(schedule('repair'), Math.round(iters * 1.5), parB, true).par;
	report('after REPAIR (undamaged)', parC);
	reportDamaged(parC);

	if (process.env.SWEEP !== '0') robustnessSweep(parC);
	if (process.env.PARAMS_OUT) { writeFileSync(process.env.PARAMS_OUT, JSON.stringify(Array.from(parC))); console.log(`  params saved to ${process.env.PARAMS_OUT}`); }
	if (process.env.FVIZ) dumpViz(parC, process.env.FVIZ);
}

/** Forward-only healed outputs for a given damage mask (fast eval, no backprop). */
function healedOutputs(par: Float64Array, mask: Uint8Array | null): number[] {
	const sched = schedule('repair');
	return CASES.map((cse) => { const st = forward(par, cse.in, sched, mask); return st[st.length - 1][OUTPUT * C + 0]; });
}

/** Robustness characterization: does it heal damage at ANY location, up to what
 *  SIZE? Sweep every interior patch center × several sizes; count how many leave
 *  all 4 XOR outputs correct after healing (|out-target| < 0.3). */
function robustnessSweep(par: Float64Array): void {
	console.log('  [D] ROBUSTNESS — heal rate over all damage positions × sizes');
	for (const size of [2, 3, 4, 5]) {
		let heal = 0, total = 0;
		const h = size >> 1;
		for (let cy = 1 + h; cy <= SH - 2 - (size - 1 - h); cy++)
			for (let cx = 1 + h; cx <= SW - 2 - (size - 1 - h); cx++) {
				const outs = healedOutputs(par, damageMask(cx, cy, size));
				const ok = CASES.every((c, k) => Math.abs(outs[k] - c.tgt) < 0.3);
				total++; if (ok) heal++;
			}
		const pct = total ? Math.round((100 * heal) / total) : 0;
		console.log(`    ${size}x${size} patch: healed ${heal}/${total} positions  (${pct}%)`);
	}
}

/** Report output on the undamaged persist schedule (final held value per case). */
function report(label: string, par: Float64Array): void {
	const sched = schedule('persist');
	const r = lossAndGrad(par, sched, null);
	const finals = r.outs.map((tr) => tr[tr.length - 1]);
	console.log(`    ${label}: held outputs [${finals.map((o) => o.toFixed(3)).join(' ')}]  (want 0 1 1 0)`);
}

/** Report output AFTER a fixed central damage — the self-repair test. */
function reportDamaged(par: Float64Array): void {
	const sched = schedule('repair');
	const mask = damageMask(Math.round((GATE_IC + OUT_COL) / 2), iy, DMG_SIZE);
	const r = lossAndGrad(par, sched, mask);
	const healed = r.outs.map((tr) => tr[tr.length - 1]);
	const ok = CASES.every((c, k) => Math.abs(healed[k] - c.tgt) < 0.2);
	console.log(`    after central ${DMG_SIZE}x${DMG_SIZE} DAMAGE, healed outputs [${healed.map((o) => o.toFixed(3)).join(' ')}]  (want 0 1 1 0)  ${ok ? '✓ SELF-REPAIRED' : '(soft)'}`);
	console.log(`    repair loss ${r.L.toFixed(4)} (baseline 0.250)`);
}

/** Dump full field trajectories (signal + hidden) for the damage-and-regrow viz. */
function dumpViz(par: Float64Array, path: string): void {
	const sched = schedule('repair');
	const dmgCx = Math.round((GATE_IC + OUT_COL) / 2);
	const mask = damageMask(dmgCx, iy, DMG_SIZE);
	const hiddenChannels = [1, 2, 3];
	const cases = CASES.map((cse) => {
		const states = forward(par, cse.in, sched, mask);
		const frames = states.map((s) => Array.from({ length: N }, (_, i) => s[i * C + 0]));
		const last = states[states.length - 1];
		const hidden = hiddenChannels.map((ch) => Array.from({ length: N }, (_, i) => last[i * C + ch]));
		return { in: cse.in, tgt: cse.tgt, out: last[OUTPUT * C + 0], frames, hidden };
	});
	const damagedCells: number[] = [];
	for (let i = 0; i < N; i++) if (mask[i] === 0) damagedCells.push(i);
	writeFileSync(path, JSON.stringify({ SW, SH, C, T_TOTAL, DMG_AT, damagedCells, inputCells, outputCell: OUTPUT, hiddenChannels, cases }));
	console.log(`  viz dumped to ${path}`);
}

main();
