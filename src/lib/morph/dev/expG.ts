// EXPERIMENT G — GROW-FROM-SEED (the complete "grown machine").
//
// E1 computes, E2 self-repairs — but both start from a substrate that is already
// fully alive. Here the medium starts EMPTY except a single seed cell: the rule
// must grow the whole computational structure from that seed AND then compute
// XOR (and, since self-repair ⊇ growth, still heal damage). One rule; grown,
// functional, self-healing.
//
// Key idea: growth is extreme repair. So we warm-start from the E2 rule and
// fine-tune with the seed initial condition in the training distribution.
//
//   PARAMS_IN=frepair_params.json npx tsx src/lib/morph/dev/expG.ts        # test E2 from seed
//   MODE=train ITERS=… PARAMS_IN=… PARAMS_OUT=… GVIZ=… npx tsx …/expG.ts   # fine-tune grow-from-seed

import { writeFileSync, readFileSync } from 'node:fs';

const SW = 9, SH = 9, N = SW * SH;
const C = 12, FEAT = 4, PERC = FEAT * C, HD = 48;
const W1O = 0, B1O = HD * PERC, W2O = B1O + HD, B2O = W2O + C * HD, P = B2O + C;

const T_GROW = 24, T_HOLD = 8, T_REPAIR = 18;
const DMG_AT = T_GROW + T_HOLD, T_TOTAL = DMG_AT + T_REPAIR;
const REPAIR_TAIL = 5;
const DMG_SIZE = Number(process.env.DMG_SIZE ?? 3);

const iy = SH >> 1, GATE_IC = 2, OUT_COL = SW - 2;
const SEED_CELL = iy * SW + (SW >> 1); // single seed cell at the grid centre
const inputCells = [(iy - 1) * SW + GATE_IC, (iy + 1) * SW + GATE_IC];
const OUTPUT = iy * SW + OUT_COL;
const CASES = [
	{ in: [0, 0], tgt: 0 }, { in: [0, 1], tgt: 1 }, { in: [1, 0], tgt: 1 }, { in: [1, 1], tgt: 0 }
];

function mulberry32(seed: number): () => number {
	let a = seed >>> 0;
	return () => {
		a |= 0; a = (a + 0x6d2b79f5) | 0;
		let t = Math.imul(a ^ (a >>> 15), 1 | a);
		t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
}

type IC = 'full' | 'seed';
/** Initial substrate. 'full' = all interior cells alive (E1/E2). 'seed' = only
 *  the centre cell alive; the rule must grow the rest. Inputs always clamped. */
function seedGrid(inputs: number[], ic: IC): Float64Array {
	const s = new Float64Array(N * C);
	if (ic === 'full') {
		for (let y = 1; y < SH - 1; y++) for (let x = 1; x < SW - 1; x++)
			for (let c = 1; c < C; c++) s[(y * SW + x) * C + c] = 1;
	} else {
		for (let c = 1; c < C; c++) s[SEED_CELL * C + c] = 1;
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

function damageMask(cx: number, cy: number, size: number): Uint8Array {
	const mask = new Uint8Array(N).fill(1);
	const h = size >> 1;
	for (let y = cy - h; y <= cy - h + size - 1; y++)
		for (let x = cx - h; x <= cx - h + size - 1; x++)
			if (x >= 0 && x < SW && y >= 0 && y < SH) mask[y * SW + x] = 0;
	return mask;
}

interface Schedule { w: Float64Array; nSteps: number; damageAt: number; }
function schedule(mode: 'persist' | 'repair'): Schedule {
	const w = new Float64Array(T_TOTAL + 1);
	const hs = T_GROW, he = DMG_AT - 1, hn = he - hs + 1;
	for (let s = hs; s <= he; s++) w[s] = 0.5 / hn;
	if (mode === 'persist') {
		const ts = DMG_AT, te = T_TOTAL, tn = te - ts + 1;
		for (let s = ts; s <= te; s++) w[s] = 0.5 / tn;
		return { w, nSteps: T_TOTAL, damageAt: -1 };
	}
	const rs = T_TOTAL - REPAIR_TAIL + 1, re = T_TOTAL, rn = re - rs + 1;
	for (let s = rs; s <= re; s++) w[s] += 0.5 / rn;
	return { w, nSteps: T_TOTAL, damageAt: DMG_AT };
}

function forward(par: Float64Array, inputs: number[], sched: Schedule, mask: Uint8Array | null, ic: IC): Float64Array[] {
	const s0 = seedGrid(inputs, ic);
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
		if (mask && t + 1 === sched.damageAt)
			for (let i = 0; i < N; i++) if (mask[i] === 0) for (let c = 0; c < C; c++) ns[i * C + c] = 0;
		clampInputs(ns, inputs);
		states.push(ns);
		s = ns;
	}
	return states;
}

function lossAndGrad(par: Float64Array, sched: Schedule, mask: Uint8Array | null, ic: IC): { L: number; grad: Float64Array; outs: number[][] } {
	const grad = new Float64Array(P);
	let L = 0; const outs: number[][] = [];
	const perc = new Float64Array(PERC), pre1 = new Float64Array(HD), hbuf = new Float64Array(HD), gh = new Float64Array(HD), gperc = new Float64Array(PERC);
	for (const cse of CASES) {
		const states = forward(par, cse.in, sched, mask, ic);
		const traj: number[] = [];
		for (let step = 0; step <= sched.nSteps; step++) if (sched.w[step] > 0) traj.push(states[step][OUTPUT * C + 0]);
		outs.push(traj);
		for (let step = 0; step <= sched.nSteps; step++) {
			if (sched.w[step] <= 0) continue;
			const diff = states[step][OUTPUT * C + 0] - cse.tgt;
			L += (sched.w[step] * diff * diff) / CASES.length;
		}
		let gs = new Float64Array(N * C);
		for (let t = sched.nSteps - 1; t >= 0; t--) {
			const step = t + 1;
			if (sched.w[step] > 0)
				gs[OUTPUT * C + 0] += (2 * sched.w[step] * (states[step][OUTPUT * C + 0] - cse.tgt)) / CASES.length;
			for (const ic2 of inputCells) gs[ic2 * C + 0] = 0;
			if (mask && step === sched.damageAt)
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

/** Train grow-from-seed: mix seed-IC and full-IC cases so the ONE rule handles
 *  both, with optional damage. Warm-started (gentle cosine lr). */
function train(iters: number, init: Float64Array, dmg: boolean, seed = 7): { par: Float64Array; L: number } {
	let par = init.slice();
	const lrHi = 0.003, lrLo = 0.0006;
	const patches: { cx: number; cy: number }[] = [];
	for (let cx = GATE_IC + 1; cx <= OUT_COL; cx++) for (const cy of [iy - 1, iy, iy + 1]) patches.push({ cx, cy });
	const m = new Float64Array(P), v = new Float64Array(P), b1 = 0.9, b2 = 0.999;
	let bestLoss = Infinity, bestPar = par.slice();
	const schedP = schedule('persist'), schedR = schedule('repair');
	for (let it = 1; it <= iters; it++) {
		const cos = 0.5 * (1 + Math.cos(Math.PI * (it / iters)));
		const lr = Math.min(1, it / 20) * (lrLo + (lrHi - lrLo) * cos);
		// accumulate gradient over both initial conditions (seed is the focus, full keeps it from forgetting)
		const grad = new Float64Array(P); let L = 0;
		let mask: Uint8Array | null = null;
		if (dmg) { const rp = mulberry32(seed * 1000 + it)(); const pc = patches[Math.floor(rp * patches.length)]; mask = damageMask(pc.cx, pc.cy, DMG_SIZE); }
		const sched = dmg ? schedR : schedP;
		const rSeed = lossAndGrad(par, sched, mask, 'seed');
		const rFull = lossAndGrad(par, sched, mask, 'full');
		for (let j = 0; j < P; j++) grad[j] = 0.75 * rSeed.grad[j] + 0.25 * rFull.grad[j];
		L = 0.75 * rSeed.L + 0.25 * rFull.L;
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
		if (it % 200 === 0 || it === 1) console.log(`      iter ${String(it).padStart(4)}  loss ${L.toFixed(5)} (seed ${rSeed.L.toFixed(4)} full ${rFull.L.toFixed(4)})  best ${bestLoss.toFixed(5)}`);
	}
	return { par: bestPar, L: bestLoss };
}

function loadParams(path: string): Float64Array {
	return Float64Array.from(JSON.parse(readFileSync(path, 'utf8')) as number[]);
}

/** Report held (persist) and healed (repair, central damage) outputs for an IC. */
function report(label: string, par: Float64Array, ic: IC): void {
	const held = lossAndGrad(par, schedule('persist'), null, ic).outs.map((tr) => tr[tr.length - 1]);
	const mask = damageMask(Math.round((GATE_IC + OUT_COL) / 2), iy, DMG_SIZE);
	const healed = lossAndGrad(par, schedule('repair'), mask, ic).outs.map((tr) => tr[tr.length - 1]);
	const okH = CASES.every((c, k) => Math.abs(held[k] - c.tgt) < 0.2);
	const okR = CASES.every((c, k) => Math.abs(healed[k] - c.tgt) < 0.2);
	console.log(`  ${label} (${ic} IC):`);
	console.log(`    grown/held   [${held.map((o) => o.toFixed(3)).join(' ')}]  ${okH ? '✓' : '(soft)'}`);
	console.log(`    +damage heal [${healed.map((o) => o.toFixed(3)).join(' ')}]  ${okR ? '✓' : '(soft)'}   (want 0 1 1 0)`);
}

function dumpViz(par: Float64Array, path: string): void {
	const sched = schedule('repair'); // grow -> hold -> damage -> heal, from a SEED
	const dmgCx = Math.round((GATE_IC + OUT_COL) / 2);
	const mask = damageMask(dmgCx, iy, DMG_SIZE);
	const hiddenChannels = [1, 2, 3];
	const cases = CASES.map((cse) => {
		const states = forward(par, cse.in, sched, mask, 'seed');
		const frames = states.map((s) => Array.from({ length: N }, (_, i) => s[i * C + 0]));
		const last = states[states.length - 1];
		const hidden = hiddenChannels.map((ch) => Array.from({ length: N }, (_, i) => last[i * C + ch]));
		return { in: cse.in, tgt: cse.tgt, out: last[OUTPUT * C + 0], frames, hidden };
	});
	const damagedCells: number[] = [];
	for (let i = 0; i < N; i++) if (mask[i] === 0) damagedCells.push(i);
	writeFileSync(path, JSON.stringify({ SW, SH, C, T_TOTAL, DMG_AT, T_GROW, seedCell: SEED_CELL, damagedCells, inputCells, outputCell: OUTPUT, hiddenChannels, cases }));
	console.log(`  viz dumped to ${path}`);
}

function main() {
	console.log(`grow-from-seed: ${SW}x${SH}, C=${C}, seed@${SEED_CELL} (row ${iy}, col ${SW >> 1}), timeline grow ${T_GROW}+hold ${T_HOLD}+repair ${T_REPAIR}=${T_TOTAL}`);
	const inPath = process.env.PARAMS_IN;
	if (!inPath) { console.error('set PARAMS_IN to the saved E2 params json'); process.exit(1); }
	const parE2 = loadParams(inPath);
	console.log('\n[TEST] does the E2 self-repair rule already grow from a seed?');
	report('E2 rule, full substrate', parE2, 'full');
	report('E2 rule, from a SEED', parE2, 'seed');

	if (process.env.MODE === 'train') {
		const iters = Number(process.env.ITERS ?? 700);
		console.log('\n[TRAIN A] grow-from-seed + persist (warm-start from E2)');
		const parA = train(iters, parE2, false).par;
		report('after grow+persist', parA, 'seed');
		console.log('\n[TRAIN B] grow-from-seed + persist + damage (heal too)');
		const parB = train(Math.round(iters * 1.5), parA, true).par;
		report('after grow+persist+repair', parB, 'seed');
		report('  cross-check full substrate', parB, 'full');
		if (process.env.PARAMS_OUT) { writeFileSync(process.env.PARAMS_OUT, JSON.stringify(Array.from(parB))); console.log(`  params saved to ${process.env.PARAMS_OUT}`); }
		if (process.env.GVIZ) dumpViz(parB, process.env.GVIZ);
	} else if (process.env.GVIZ) {
		dumpViz(parE2, process.env.GVIZ); // viz the E2 rule from seed (test mode)
	}
}

main();
