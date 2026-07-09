// EXPERIMENT E — DEVELOPMENTAL COMPUTATION (the paper's headline demo).
//
// Not "grow a picture" but "grow a MACHINE": a cellular-automaton rule, trained
// by gradient descent through development, that routes/computes a signal from
// input cell(s) to an output cell — the same rule must give the RIGHT output for
// every input case, so it cannot memorize a pattern; it has to build a
// computational structure. This is what neither NCA (images only) nor classic
// A-life (no gradients) does.
//
// Milestone ladder (this file starts at E0 and E1):
//   E0  WIRE  — 1 input, output = input (faithful transport over distance).
//   E1  GATE  — 2 inputs, output = XOR/AND (actual computation, 4 cases).
//   (next) grow-from-seed; then damage-and-still-compute (self-repair).
//
//   npx tsx src/lib/morph/dev/expE.ts            # runs WIRE then GATE
//   TASK=wire|gate  ITERS=… npx tsx …/expE.ts
//   TASK=gate GATE_VIZ=path.json …               # also dump development frames

import { writeFileSync } from 'node:fs';

const SW = 9, SH = 9, N = SW * SH;
const C = 12; // channel 0 = signal (clamped at input, read at output); 1..11 = hidden structure
const FEAT = 4; // identity, gx, gy, laplacian
const PERC = FEAT * C;
const HD = 48;
const T = 24;
const W1O = 0, B1O = HD * PERC, W2O = B1O + HD, B2O = W2O + C * HD, P = B2O + C;

function mulberry32(seed: number): () => number {
	let a = seed >>> 0;
	return () => {
		a |= 0; a = (a + 0x6d2b79f5) | 0;
		let t = Math.imul(a ^ (a >>> 15), 1 | a);
		t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
}

interface Task {
	name: string;
	inputCells: number[];
	outputCell: number;
	cases: { in: number[]; tgt: number }[];
}
const iy = SH >> 1;
const wireTask: Task = {
	name: 'wire',
	inputCells: [iy * SW + 1],
	outputCell: iy * SW + (SW - 2),
	cases: [{ in: [0], tgt: 0 }, { in: [1], tgt: 1 }]
};
const gateTask: Task = {
	name: 'gate (XOR)',
	inputCells: [(iy - 1) * SW + 2, (iy + 1) * SW + 2],
	outputCell: iy * SW + (SW - 2),
	cases: [
		{ in: [0, 0], tgt: 0 },
		{ in: [0, 1], tgt: 1 },
		{ in: [1, 0], tgt: 1 },
		{ in: [1, 1], tgt: 0 }
	]
};

/** Live substrate: every interior cell alive (hidden channels = 1); signal channel 0 = 0. */
function seedGrid(): Float64Array {
	const s = new Float64Array(N * C);
	for (let y = 1; y < SH - 1; y++)
		for (let x = 1; x < SW - 1; x++) {
			const i = y * SW + x;
			for (let c = 1; c < C; c++) s[i * C + c] = 1;
		}
	return s;
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

/** Forward rollout for one input case. Signal channel 0 is clamped at input cells every step. */
function forward(par: Float64Array, task: Task, inputs: number[]): Float64Array[] {
	const clampInputs = (f: Float64Array) => {
		for (let k = 0; k < task.inputCells.length; k++) f[task.inputCells[k] * C + 0] = inputs[k];
	};
	const s0 = seedGrid();
	clampInputs(s0);
	const states: Float64Array[] = [s0];
	let s = s0;
	const perc = new Float64Array(PERC), h = new Float64Array(HD);
	for (let t = 0; t < T; t++) {
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
		clampInputs(ns); // input is an external boundary condition
		states.push(ns);
		s = ns;
	}
	return states;
}

function caseLoss(sT: Float64Array, task: Task, tgt: number): { L: number; gsT: Float64Array } {
	const gsT = new Float64Array(N * C);
	const o = sT[task.outputCell * C + 0];
	const diff = o - tgt;
	gsT[task.outputCell * C + 0] = 2 * diff;
	return { L: diff * diff, gsT };
}

/** Backprop through one case. gs is zeroed at clamped input cells each step (their value is external). */
function backward(states: Float64Array[], par: Float64Array, task: Task, gsT: Float64Array, grad: Float64Array): void {
	let gs = gsT;
	const perc = new Float64Array(PERC), pre1 = new Float64Array(HD), h = new Float64Array(HD), gh = new Float64Array(HD), gperc = new Float64Array(PERC);
	for (let t = T - 1; t >= 0; t--) {
		for (const ic of task.inputCells) gs[ic * C + 0] = 0; // clamped: no gradient through the override
		const s = states[t], sp = states[t + 1];
		const gsPrev = new Float64Array(N * C);
		for (let y = 1; y < SH - 1; y++)
			for (let x = 1; x < SW - 1; x++) {
				const i = y * SW + x;
				perceive(s, i, perc);
				for (let hh = 0; hh < HD; hh++) {
					let a = par[B1O + hh]; const base = W1O + hh * PERC;
					for (let k = 0; k < PERC; k++) a += par[base + k] * perc[k];
					pre1[hh] = a; h[hh] = a > 0 ? a : 0;
				}
				gh.fill(0);
				for (let c = 0; c < C; c++) {
					const spv = sp[i * C + c];
					const gp = gs[i * C + c] * (1 - spv * spv);
					gsPrev[i * C + c] += gp;
					grad[B2O + c] += gp;
					const base = W2O + c * HD;
					for (let hh = 0; hh < HD; hh++) { grad[base + hh] += gp * h[hh]; gh[hh] += par[base + hh] * gp; }
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

/** Total loss + gradient over all input cases. */
function lossAndGrad(par: Float64Array, task: Task): { L: number; grad: Float64Array; outs: number[] } {
	const grad = new Float64Array(P);
	let L = 0; const outs: number[] = [];
	for (const cse of task.cases) {
		const states = forward(par, task, cse.in);
		outs.push(states[T][task.outputCell * C + 0]);
		const { L: cl, gsT } = caseLoss(states[T], task, cse.tgt);
		L += cl / task.cases.length;
		backward(states, par, task, gsT, grad);
	}
	for (let j = 0; j < P; j++) grad[j] /= task.cases.length;
	return { L, grad, outs };
}

function gradientCheck(task: Task): boolean {
	const rng = mulberry32(3);
	const par = new Float64Array(P).map(() => (rng() - 0.5) * 0.1);
	for (let j = W2O; j < P; j++) par[j] = 0;
	const { grad } = lossAndGrad(par, task);
	const eps = 1e-4;
	let maxRel = 0;
	for (const j of [10, 500, B1O + 2, W2O + 5, B2O + 0]) {
		const pp = par.slice(); pp[j] += eps;
		const pm = par.slice(); pm[j] -= eps;
		const fd = (lossAndGrad(pp, task).L - lossAndGrad(pm, task).L) / (2 * eps);
		maxRel = Math.max(maxRel, Math.abs(grad[j] - fd) / (Math.abs(fd) + 1e-8));
	}
	console.log(`  gradient check: max rel err ${maxRel.toExponential(2)} -> ${maxRel < 0.02 ? 'PASS' : 'FAIL'}`);
	return maxRel < 0.02;
}

function train(task: Task, iters: number, init?: Float64Array, seed = 7): { par: Float64Array; L: number; outs: number[] } {
	let par: Float64Array;
	if (init) {
		par = init.slice(); // warm-start (curriculum)
	} else {
		const rng = mulberry32(seed);
		par = new Float64Array(P);
		// full nonzero init (incl. last layer) so the CA has initial dynamics.
		for (let j = 0; j < P; j++) par[j] = (rng() - 0.5) * 0.12;
		for (let j = W2O; j < P; j++) par[j] *= 0.5;
	}
	// A warm-started stage is REFINEMENT, not exploration: a high lr kicks the
	// good incoming solution out of its basin (loss bounces back to baseline). So
	// warm stages get a gentler peak and a cosine decay to a low floor that lets
	// the transport re-saturate and settle. From-scratch stages keep the hot lr.
	const warm = init !== undefined;
	const lrHi = warm ? 0.003 : 0.01;
	const lrLo = warm ? 0.0006 : 0.004;
	const m = new Float64Array(P), v = new Float64Array(P), b1 = 0.9, b2 = 0.999;
	let bestLoss = Infinity, bestPar = par.slice();
	let last = { L: 0, outs: [] as number[] };
	for (let it = 1; it <= iters; it++) {
		const cos = 0.5 * (1 + Math.cos(Math.PI * (it / iters))); // 1 → 0 over the run
		const lr = Math.min(1, it / 20) * (lrLo + (lrHi - lrLo) * cos);
		const { L, grad, outs } = lossAndGrad(par, task);
		last = { L, outs };
		let gn = 0; for (let j = 0; j < P; j++) gn += grad[j] * grad[j]; gn = Math.sqrt(gn);
		const clip = gn > 1 ? 1 / gn : 1;
		if (L < bestLoss) { bestLoss = L; bestPar = par.slice(); } // snapshot BEFORE the step
		const c1 = 1 - Math.pow(b1, it), c2 = 1 - Math.pow(b2, it);
		for (let j = 0; j < P; j++) {
			const g = grad[j] * clip + 2e-5 * par[j];
			m[j] = b1 * m[j] + (1 - b1) * g;
			v[j] = b2 * v[j] + (1 - b2) * g * g;
			par[j] -= (lr * (m[j] / c1)) / (Math.sqrt(v[j] / c2) + 1e-8);
		}
		if (it % 200 === 0 || it === 1) console.log(`    iter ${String(it).padStart(4)}  loss ${L.toFixed(5)}  (best ${bestLoss.toFixed(5)})`);
	}
	const final = lossAndGrad(bestPar, task);
	return { par: bestPar, L: final.L, outs: final.outs };
}

/** Train several random restarts, keep the best. XOR-from-scratch has a strong
 *  constant-0.5 local minimum; restarts + curriculum are how we escape it. */
function trainBest(
	task: Task,
	iters: number,
	restarts: number,
	init?: Float64Array
): { par: Float64Array; L: number; outs: number[] } {
	let best: { par: Float64Array; L: number; outs: number[] } | null = null;
	for (let s = 0; s < restarts; s++) {
		const r = train(task, iters, init, 7 + s * 101);
		if (!best || r.L < best.L) best = r;
		if (restarts > 1) console.log(`      restart ${s}: loss ${r.L.toFixed(5)}`);
	}
	return best!;
}

function runTask(task: Task, iters: number): void {
	console.log(`\n=== ${task.name} ===`);
	if (!gradientCheck(task)) { console.error('FAIL: gradient wrong'); process.exit(1); }
	const { L, outs } = train(task, iters);
	console.log(`  final loss ${L.toFixed(5)}`);
	console.log('  input -> output (want target):');
	task.cases.forEach((cse, k) => console.log(`    [${cse.in.join(',')}] -> ${outs[k].toFixed(3)}   (want ${cse.tgt})`));
	// constant-output baseline loss = variance of targets around their mean (what "ignore the input" scores)
	const mt = task.cases.reduce((a, c) => a + c.tgt, 0) / task.cases.length;
	const baseline = task.cases.reduce((a, c) => a + (c.tgt - mt) ** 2, 0) / task.cases.length;
	console.log(`  (constant-output baseline loss would be ${baseline.toFixed(3)}; beating it means it uses the input)`);
}

/** A wire whose output is `dist` cells to the right of the input. */
function wireAt(dist: number): Task {
	return {
		name: `wire d=${dist}`,
		inputCells: [iy * SW + 1],
		outputCell: iy * SW + (1 + dist),
		cases: [{ in: [0], tgt: 0 }, { in: [1], tgt: 1 }]
	};
}

/** Distance curriculum: learn to transport 1 cell, warm-start, extend one cell at a time. */
function wireCurriculum(iters: number): void {
	console.log(`\n=== WIRE (distance curriculum) ===`);
	console.log(`  (long-range transport is a vanishing-gradient problem from scratch; the curriculum bootstraps it)`);
	let par: Float64Array | undefined;
	const maxDist = SW - 3; // output stays interior
	for (let dist = 1; dist <= maxDist; dist++) {
		const task = wireAt(dist);
		const r = train(task, iters, par);
		par = r.par;
		const ok = Math.abs(r.outs[0]) < 0.15 && Math.abs(r.outs[1] - 1) < 0.15;
		console.log(`  d=${dist}: loss ${r.L.toFixed(4)}  [0]->${r.outs[0].toFixed(2)} [1]->${r.outs[1].toFixed(2)}  ${ok ? 'OK' : '(soft)'}`);
	}
}

const GATE_IC = 2; // input column; output moves right from here across the curriculum

/** XOR gate whose output cell is `dist` columns to the right of the input column.
 *  Inputs stay fixed (rows iy±1, col GATE_IC) so a warm-started rule transfers. */
function gateAt(dist: number): Task {
	return {
		name: `gate d=${dist}`,
		inputCells: [(iy - 1) * SW + GATE_IC, (iy + 1) * SW + GATE_IC],
		outputCell: iy * SW + (GATE_IC + dist),
		cases: [
			{ in: [0, 0], tgt: 0 },
			{ in: [0, 1], tgt: 1 },
			{ in: [1, 0], tgt: 1 },
			{ in: [1, 1], tgt: 0 }
		]
	};
}

/** Distance curriculum for XOR. Stage 1 (output adjacent to the inputs) is the
 *  hard part — the rule must discover the nonlinear combination locally — so it
 *  gets several restarts; later stages just extend the transport by one cell. */
function gateCurriculum(iters: number): void {
	console.log(`\n=== GATE (XOR, distance curriculum) ===`);
	console.log(`  (XOR = long-range transport + a nonlinear combine; both are hard from scratch)`);
	let par: Float64Array | undefined;
	const maxDist = SW - 2 - GATE_IC; // output column stays interior (<= SW-2)
	for (let dist = 1; dist <= maxDist; dist++) {
		const task = gateAt(dist);
		// Stage 1 discovers XOR and converges fast (~200 iters), so cap it and spend
		// the budget on the transport stages, which need to re-saturate after each hop.
		const restarts = dist === 1 ? 6 : 1; // only the first stage explores; rest warm-start
		const stageIters = dist === 1 ? Math.min(iters, 300) : iters;
		const r = trainBest(task, stageIters, restarts, par);
		par = r.par;
		const correct = task.cases.every((c, k) => Math.abs(r.outs[k] - c.tgt) < 0.2);
		const outs = r.outs.map((o) => o.toFixed(2)).join(' ');
		console.log(
			`  d=${dist}: loss ${r.L.toFixed(4)}  outs[${outs}]  ${correct ? '✓ XOR solved' : '(not yet)'}`
		);
	}
	if (par) {
		const finalTask = gateAt(maxDist);
		const f = lossAndGrad(par, finalTask);
		console.log(`\n  FINAL XOR (output at col ${GATE_IC + maxDist}):`);
		finalTask.cases.forEach((c, k) =>
			console.log(`    [${c.in.join(',')}] -> ${f.outs[k].toFixed(3)}   (want ${c.tgt})`)
		);
		console.log(`  (constant-output baseline loss = 0.250; XOR must beat it by using both inputs)`);
		if (process.env.GATE_VIZ) dumpGateViz(par, finalTask, process.env.GATE_VIZ);
	}
}

/** Dump development frames for each XOR case: the signal channel over all T+1
 *  steps (watch the two inputs flow in and combine) plus a few hidden channels
 *  at the final step (the internal wiring the rule invented). */
function dumpGateViz(par: Float64Array, task: Task, path: string): void {
	const hiddenChannels = [1, 2, 3];
	const cases = task.cases.map((cse) => {
		const states = forward(par, task, cse.in);
		const frames = states.map((s) => Array.from({ length: N }, (_, i) => s[i * C + 0]));
		const hidden = hiddenChannels.map((ch) =>
			Array.from({ length: N }, (_, i) => states[T][i * C + ch])
		);
		return { in: cse.in, tgt: cse.tgt, out: states[T][task.outputCell * C + 0], frames, hidden };
	});
	writeFileSync(
		path,
		JSON.stringify({ SW, SH, C, T, inputCells: task.inputCells, outputCell: task.outputCell, hiddenChannels, cases })
	);
	console.log(`  viz dumped to ${path}`);
}

function main() {
	console.log(`developmental computation: ${SW}x${SH}, C=${C}, ${P} params, ${T} steps`);
	const which = process.env.TASK;
	const iters = Number(process.env.ITERS ?? 500);
	if (!which || which === 'wire') wireCurriculum(iters);
	if (which === 'gate') gateCurriculum(Number(process.env.ITERS ?? 800));
}

main();
