// S8 — ablations + multi-seed statistics for the developmental XOR gate.
//
// Trains the E1 gate (9×9, XOR at a displaced output via a distance curriculum)
// under ablation conditions, over many random seeds, and reports success RATE +
// loss stats. This is the rigor the paper needs: the mechanism claims (directional
// perception is necessary; a hidden nonlinearity is necessary; capacity vs
// success) become measured curves with error bars, not single anecdotes.
//
// One process runs ONE condition over a seed list (fan out conditions in parallel).
//   COND=baseline|iso|id|norelu|hd8|hd16|hd32|hd96  SEEDS=0-15  ITERS=500 npx tsx s8.ts
// Emits a human log + a final line  RESULT <json>  for machine aggregation.

const SW = 9, SH = 9, N = SW * SH, C = 12, T = 24;
const GATE_IC = 2;
const iy = SH >> 1;

type PercMode = 'full' | 'iso' | 'id';
interface Cond { name: string; feat: number; perc: PercMode; hd: number; relu: boolean; }
const CONDS: Record<string, Cond> = {
	baseline: { name: 'baseline (full percep, HD48, ReLU)', feat: 4, perc: 'full', hd: 48, relu: true },
	iso: { name: 'isotropic perception (id+lap only)', feat: 2, perc: 'iso', hd: 48, relu: true },
	id: { name: 'identity-only perception (no diffusion sense)', feat: 1, perc: 'id', hd: 48, relu: true },
	norelu: { name: 'no hidden nonlinearity (linear MLP)', feat: 4, perc: 'full', hd: 48, relu: false },
	hd4: { name: 'capacity HD=4', feat: 4, perc: 'full', hd: 4, relu: true },
	hd8: { name: 'capacity HD=8', feat: 4, perc: 'full', hd: 8, relu: true },
	hd16: { name: 'capacity HD=16', feat: 4, perc: 'full', hd: 16, relu: true },
	hd32: { name: 'capacity HD=32', feat: 4, perc: 'full', hd: 32, relu: true },
	hd96: { name: 'capacity HD=96', feat: 4, perc: 'full', hd: 96, relu: true }
};

function mulberry32(seed: number): () => number {
	let a = seed >>> 0;
	return () => {
		a |= 0; a = (a + 0x6d2b79f5) | 0;
		let t = Math.imul(a ^ (a >>> 15), 1 | a);
		t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
}

interface Task { inputCells: number[]; outputCell: number; cases: { in: number[]; tgt: number }[]; }
function gateAt(dist: number): Task {
	return {
		inputCells: [(iy - 1) * SW + GATE_IC, (iy + 1) * SW + GATE_IC],
		outputCell: iy * SW + (GATE_IC + dist),
		cases: [{ in: [0, 0], tgt: 0 }, { in: [0, 1], tgt: 1 }, { in: [1, 0], tgt: 1 }, { in: [1, 1], tgt: 0 }]
	};
}

// ---- model (parameterized by condition) -----------------------------------
class Model {
	FEAT: number; PERC: number; HD: number; relu: boolean; perc: PercMode;
	W1O = 0; B1O: number; W2O: number; B2O: number; P: number;
	constructor(c: Cond) {
		this.FEAT = c.feat; this.PERC = c.feat * C; this.HD = c.hd; this.relu = c.relu; this.perc = c.perc;
		this.B1O = this.HD * this.PERC; this.W2O = this.B1O + this.HD; this.B2O = this.W2O + C * this.HD; this.P = this.B2O + C;
	}
	perceive(s: Float64Array, i: number, out: Float64Array): void {
		const r = i + 1, l = i - 1, u = i - SW, d = i + SW;
		for (let ch = 0; ch < C; ch++) {
			const b = ch * this.FEAT, self = s[i * C + ch];
			const sr = s[r * C + ch], sl = s[l * C + ch], su = s[u * C + ch], sd = s[d * C + ch];
			out[b] = self;
			if (this.perc === 'full') { out[b + 1] = (sr - sl) * 0.5; out[b + 2] = (sd - su) * 0.5; out[b + 3] = sr + sl + su + sd - 4 * self; }
			else if (this.perc === 'iso') { out[b + 1] = sr + sl + su + sd - 4 * self; }
		}
	}
	// scatter d(perc)/d(state) back to neighbours (transpose of perceive)
	scatter(gperc: Float64Array, gsPrev: Float64Array, i: number): void {
		const r = i + 1, l = i - 1, u = i - SW, d = i + SW;
		for (let ch = 0; ch < C; ch++) {
			const b = ch * this.FEAT, gId = gperc[b];
			if (this.perc === 'full') {
				const gGx = gperc[b + 1], gGy = gperc[b + 2], gLap = gperc[b + 3];
				gsPrev[i * C + ch] += gId - 4 * gLap;
				gsPrev[r * C + ch] += 0.5 * gGx + gLap; gsPrev[l * C + ch] += -0.5 * gGx + gLap;
				gsPrev[d * C + ch] += 0.5 * gGy + gLap; gsPrev[u * C + ch] += -0.5 * gGy + gLap;
			} else if (this.perc === 'iso') {
				const gLap = gperc[b + 1];
				gsPrev[i * C + ch] += gId - 4 * gLap;
				gsPrev[r * C + ch] += gLap; gsPrev[l * C + ch] += gLap; gsPrev[d * C + ch] += gLap; gsPrev[u * C + ch] += gLap;
			} else { gsPrev[i * C + ch] += gId; }
		}
	}
}

function seedGrid(): Float64Array {
	const s = new Float64Array(N * C);
	for (let y = 1; y < SH - 1; y++) for (let x = 1; x < SW - 1; x++) { const i = y * SW + x; for (let c = 1; c < C; c++) s[i * C + c] = 1; }
	return s;
}

function forward(M: Model, par: Float64Array, task: Task, inputs: number[]): Float64Array[] {
	const clamp = (f: Float64Array) => { for (let k = 0; k < task.inputCells.length; k++) f[task.inputCells[k] * C + 0] = inputs[k]; };
	const s0 = seedGrid(); clamp(s0);
	const states = [s0]; let s = s0;
	const perc = new Float64Array(M.PERC), h = new Float64Array(M.HD);
	for (let t = 0; t < T; t++) {
		const ns = new Float64Array(N * C);
		for (let y = 1; y < SH - 1; y++) for (let x = 1; x < SW - 1; x++) {
			const i = y * SW + x;
			M.perceive(s, i, perc);
			for (let hh = 0; hh < M.HD; hh++) { let a = par[M.B1O + hh]; const base = M.W1O + hh * M.PERC; for (let k = 0; k < M.PERC; k++) a += par[base + k] * perc[k]; h[hh] = M.relu ? (a > 0 ? a : 0) : a; }
			for (let c = 0; c < C; c++) { let dl = par[M.B2O + c]; const base = M.W2O + c * M.HD; for (let hh = 0; hh < M.HD; hh++) dl += par[base + hh] * h[hh]; ns[i * C + c] = Math.tanh(s[i * C + c] + dl); }
		}
		clamp(ns); states.push(ns); s = ns;
	}
	return states;
}

function lossAndGrad(M: Model, par: Float64Array, task: Task): { L: number; grad: Float64Array; outs: number[] } {
	const grad = new Float64Array(M.P); let L = 0; const outs: number[] = [];
	const perc = new Float64Array(M.PERC), pre1 = new Float64Array(M.HD), h = new Float64Array(M.HD), gh = new Float64Array(M.HD), gperc = new Float64Array(M.PERC);
	for (const cse of task.cases) {
		const states = forward(M, par, task, cse.in);
		const o = states[T][task.outputCell * C + 0]; outs.push(o);
		const diff = o - cse.tgt; L += (diff * diff) / task.cases.length;
		let gs = new Float64Array(N * C); gs[task.outputCell * C + 0] = 2 * diff;
		for (let t = T - 1; t >= 0; t--) {
			for (const ic of task.inputCells) gs[ic * C + 0] = 0;
			const s = states[t], sp = states[t + 1]; const gsPrev = new Float64Array(N * C);
			for (let y = 1; y < SH - 1; y++) for (let x = 1; x < SW - 1; x++) {
				const i = y * SW + x;
				M.perceive(s, i, perc);
				for (let hh = 0; hh < M.HD; hh++) { let a = par[M.B1O + hh]; const base = M.W1O + hh * M.PERC; for (let k = 0; k < M.PERC; k++) a += par[base + k] * perc[k]; pre1[hh] = a; h[hh] = M.relu ? (a > 0 ? a : 0) : a; }
				gh.fill(0);
				for (let c = 0; c < C; c++) {
					const spv = sp[i * C + c]; const gp = gs[i * C + c] * (1 - spv * spv);
					gsPrev[i * C + c] += gp; grad[M.B2O + c] += gp; const base = M.W2O + c * M.HD;
					for (let hh = 0; hh < M.HD; hh++) { grad[base + hh] += gp * h[hh]; gh[hh] += par[base + hh] * gp; }
				}
				gperc.fill(0);
				for (let hh = 0; hh < M.HD; hh++) { let g = gh[hh]; if (M.relu && pre1[hh] <= 0) g = 0; grad[M.B1O + hh] += g; const base = M.W1O + hh * M.PERC; for (let k = 0; k < M.PERC; k++) { grad[base + k] += g * perc[k]; gperc[k] += par[base + k] * g; } }
				M.scatter(gperc, gsPrev, i);
			}
			gs = gsPrev;
		}
	}
	for (let j = 0; j < M.P; j++) grad[j] /= task.cases.length;
	return { L, grad, outs };
}

function train(M: Model, task: Task, iters: number, seed: number, init?: Float64Array): { par: Float64Array; L: number; outs: number[] } {
	let par: Float64Array;
	if (init) par = init.slice();
	else { const rng = mulberry32(seed); par = new Float64Array(M.P); for (let j = 0; j < M.P; j++) par[j] = (rng() - 0.5) * 0.12; for (let j = M.W2O; j < M.P; j++) par[j] *= 0.5; }
	const warm = init !== undefined;
	const lrHi = warm ? 0.003 : 0.01, lrLo = warm ? 0.0006 : 0.004;
	const m = new Float64Array(M.P), v = new Float64Array(M.P), b1 = 0.9, b2 = 0.999;
	let bestLoss = Infinity, bestPar = par.slice();
	for (let it = 1; it <= iters; it++) {
		const cos = 0.5 * (1 + Math.cos(Math.PI * (it / iters)));
		const lr = Math.min(1, it / 20) * (lrLo + (lrHi - lrLo) * cos);
		const { L, grad } = lossAndGrad(M, par, task);
		let gn = 0; for (let j = 0; j < M.P; j++) gn += grad[j] * grad[j]; gn = Math.sqrt(gn);
		const clip = gn > 1 ? 1 / gn : 1;
		if (L < bestLoss) { bestLoss = L; bestPar = par.slice(); }
		const c1 = 1 - Math.pow(b1, it), c2 = 1 - Math.pow(b2, it);
		for (let j = 0; j < M.P; j++) { const g = grad[j] * clip + 2e-5 * par[j]; m[j] = b1 * m[j] + (1 - b1) * g; v[j] = b2 * v[j] + (1 - b2) * g * g; par[j] -= (lr * (m[j] / c1)) / (Math.sqrt(v[j] / c2) + 1e-8); }
	}
	const final = lossAndGrad(M, bestPar, task);
	return { par: bestPar, L: final.L, outs: final.outs };
}

/** One seed: distance curriculum (adjacent → displaced). Returns solved-distance + final loss. */
function runSeed(M: Model, iters: number, seed: number): { solved: boolean; maxSolvedDist: number; finalLoss: number; outs: number[] } {
	let par: Float64Array | undefined;
	const maxDist = SW - 2 - GATE_IC; // 5
	let maxSolvedDist = 0, finalLoss = 1, finalOuts: number[] = [];
	const R1 = Number(process.env.RESTARTS ?? 4);
	for (let dist = 1; dist <= maxDist; dist++) {
		const task = gateAt(dist);
		const restarts = dist === 1 ? R1 : 1;
		const stageIters = dist === 1 ? Math.min(iters, 300) : iters;
		let best: { par: Float64Array; L: number; outs: number[] } | null = null;
		for (let s = 0; s < restarts; s++) { const r = train(M, task, stageIters, seed * 131 + s * 977, par); if (!best || r.L < best.L) best = r; }
		par = best!.par;
		const correct = task.cases.every((c, k) => Math.abs(best!.outs[k] - c.tgt) < 0.2);
		if (correct) maxSolvedDist = dist;
		finalLoss = best!.L; finalOuts = best!.outs;
		if (!correct && dist > 1) break; // curriculum stalled — stop early
	}
	return { solved: maxSolvedDist === maxDist, maxSolvedDist, finalLoss, outs: finalOuts };
}

function parseSeeds(spec: string): number[] {
	if (spec.includes('-')) { const [a, b] = spec.split('-').map(Number); return Array.from({ length: b - a + 1 }, (_, i) => a + i); }
	return spec.split(',').map(Number);
}

const condKey = process.env.COND ?? 'baseline';
const cond = CONDS[condKey];
if (!cond) { console.error(`unknown COND ${condKey}; have ${Object.keys(CONDS).join(', ')}`); process.exit(1); }
const seeds = parseSeeds(process.env.SEEDS ?? '0-7');
const iters = Number(process.env.ITERS ?? 500);
const M = new Model(cond);
console.log(`S8 condition: ${condKey} — ${cond.name}  (P=${M.P}, seeds ${seeds[0]}..${seeds[seeds.length - 1]}, iters ${iters})`);
const per: { seed: number; solved: boolean; maxSolvedDist: number; finalLoss: number }[] = [];
const t0 = Date.now();
for (const sd of seeds) {
	const r = runSeed(M, iters, sd);
	per.push({ seed: sd, solved: r.solved, maxSolvedDist: r.maxSolvedDist, finalLoss: r.finalLoss });
	console.log(`  seed ${String(sd).padStart(2)}: ${r.solved ? 'SOLVED' : 'failed'}  maxDist ${r.maxSolvedDist}/5  loss ${r.finalLoss.toFixed(4)}  outs[${r.outs.map((o) => o.toFixed(2)).join(' ')}]`);
}
const nSolved = per.filter((p) => p.solved).length;
const losses = per.map((p) => p.finalLoss).sort((a, b) => a - b);
const mean = losses.reduce((a, b) => a + b, 0) / losses.length;
const std = Math.sqrt(losses.reduce((a, b) => a + (b - mean) ** 2, 0) / losses.length);
const secs = (Date.now() - t0) / 1000;
const result = { cond: condKey, name: cond.name, P: M.P, nSeeds: seeds.length, nSolved, successRate: nSolved / seeds.length, meanLoss: mean, stdLoss: std, medianLoss: losses[losses.length >> 1], meanSolvedDist: per.reduce((a, p) => a + p.maxSolvedDist, 0) / per.length, secs, per };
console.log(`\nsuccess ${nSolved}/${seeds.length} (${(100 * nSolved / seeds.length).toFixed(0)}%)  meanLoss ${mean.toFixed(4)}±${std.toFixed(4)}  meanSolvedDist ${result.meanSolvedDist.toFixed(2)}/5  ${secs.toFixed(1)}s`);
console.log('RESULT ' + JSON.stringify(result));
