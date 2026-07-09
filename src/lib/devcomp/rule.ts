// Shared spec for developmental computation (E1/E2/E3 and beyond).
//
// Single source of truth for the CA rule: constants, parameter layout, the
// reference `forward()` stepper (identical to the trainer in
// `src/lib/morph/dev/expF.ts`), and per-experiment I/O layouts. The trainer
// (Node), the WGSL kernel (GPU), and the demo UI all agree with THIS file.
//
// The rule (per interior cell, per step):
//   perceive = [identity, gx=(right−left)/2, gy=(down−up)/2, laplacian] per channel
//   dl       = W2 · relu(W1 · perceive + b1) + b2
//   state'   = tanh(state + dl)      ← residual is INSIDE the tanh
// Inputs are clamped (channel 0) every step; border cells stay 0 forever.

export const SW = 9, SH = 9, N = SW * SH;
export const C = 12, FEAT = 4, PERC = FEAT * C, HD = 48;
// Parameter block layout: [ W1(HD×PERC), b1(HD), W2(C×HD), b2(C) ]
export const W1O = 0;
export const B1O = W1O + HD * PERC;
export const W2O = B1O + HD;
export const B2O = W2O + C * HD;
export const P = B2O + C; // 2940

export type IC = 'full' | 'seed';

export interface Experiment {
	id: string;
	name: string;
	blurb: string;
	inputCells: number[]; // cells whose channel-0 is clamped to the input bit
	outputCells: number[]; // cells whose channel-0 is read as an output bit
	cases: { in: number[]; out: number[] }[]; // truth table
	ic: IC; // initial condition: full substrate, or a single seed cell
	seedCell?: number; // for ic === 'seed'
	paramsUrl: string; // JSON number[] of length P
	tGrow: number; // steps until the answer has settled (for a readout)
}

const iy = SH >> 1, IN_COL = 2, OUT_COL = SW - 2;
const IN_TOP = (iy - 1) * SW + IN_COL, IN_BOT = (iy + 1) * SW + IN_COL;
const OUT = iy * SW + OUT_COL;
const XOR_CASES = [
	{ in: [0, 0], out: [0] },
	{ in: [0, 1], out: [1] },
	{ in: [1, 0], out: [1] },
	{ in: [1, 1], out: [0] }
];

/** The trained experiments. Param URLs resolve against the app's base path. */
export const EXPERIMENTS: Experiment[] = [
	{
		id: 'e1_gate', name: 'XOR gate', blurb: 'Two inputs, one output: computes their XOR at a cell 5 away.',
		inputCells: [IN_TOP, IN_BOT], outputCells: [OUT], cases: XOR_CASES,
		ic: 'full', paramsUrl: 'e1_gate.json', tGrow: 24
	},
	{
		id: 'e2_repair', name: 'Self-repair', blurb: 'Destroy a patch mid-computation — it regrows and still computes XOR.',
		inputCells: [IN_TOP, IN_BOT], outputCells: [OUT], cases: XOR_CASES,
		ic: 'full', paramsUrl: 'e2_repair.json', tGrow: 24
	},
	{
		id: 'e3_seed', name: 'Grow from seed', blurb: 'Grow the whole computer from a single seed cell, then compute + heal.',
		inputCells: [IN_TOP, IN_BOT], outputCells: [OUT], cases: XOR_CASES,
		ic: 'seed', seedCell: iy * SW + (SW >> 1), paramsUrl: 'e3_seed.json', tGrow: 24
	}
];

export function experimentById(id: string): Experiment | undefined {
	return EXPERIMENTS.find((e) => e.id === id);
}

/** Parse a params JSON payload (number[]) into a typed array; validate length. */
export function loadParams(arr: number[]): Float64Array {
	if (arr.length !== P) throw new Error(`params length ${arr.length} !== ${P}`);
	return Float64Array.from(arr);
}

/** Initial substrate for an experiment + input assignment. */
export function seedGrid(exp: Experiment, inputs: number[]): Float64Array {
	const s = new Float64Array(N * C);
	if (exp.ic === 'full') {
		for (let y = 1; y < SH - 1; y++) for (let x = 1; x < SW - 1; x++)
			for (let c = 1; c < C; c++) s[(y * SW + x) * C + c] = 1;
	} else {
		const sc = exp.seedCell ?? iy * SW + (SW >> 1);
		for (let c = 1; c < C; c++) s[sc * C + c] = 1;
	}
	clampInputs(s, exp, inputs);
	return s;
}

export function clampInputs(f: Float64Array, exp: Experiment, inputs: number[]): void {
	for (let k = 0; k < exp.inputCells.length; k++) f[exp.inputCells[k] * C + 0] = inputs[k];
}

/** Perception features for cell i into `out` (length PERC). */
export function perceive(s: Float64Array, i: number, out: Float64Array): void {
	const r = i + 1, l = i - 1, u = i - SW, d = i + SW;
	for (let ch = 0; ch < C; ch++) {
		const b = ch * FEAT, self = s[i * C + ch];
		const sr = s[r * C + ch], sl = s[l * C + ch], su = s[u * C + ch], sd = s[d * C + ch];
		out[b] = self;
		out[b + 1] = (sr - sl) * 0.5;
		out[b + 2] = (sd - su) * 0.5;
		out[b + 3] = sr + sl + su + sd - 4 * self;
	}
}

/** One CA step (pure): returns the next state. Optionally destroys `damage` cells. */
export function step(par: Float64Array, s: Float64Array, exp: Experiment, inputs: number[], damage?: Uint8Array): Float64Array {
	const ns = new Float64Array(N * C);
	const perc = new Float64Array(PERC), h = new Float64Array(HD);
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
	if (damage) for (let i = 0; i < N; i++) if (damage[i] === 0) for (let c = 0; c < C; c++) ns[i * C + c] = 0;
	clampInputs(ns, exp, inputs);
	return ns;
}

export interface RolloutOpts { steps: number; damage?: { at: number; mask: Uint8Array }; }

/** Full developmental rollout. Returns states[0..steps]. */
export function forward(par: Float64Array, exp: Experiment, inputs: number[], opts: RolloutOpts): Float64Array[] {
	const states: Float64Array[] = [seedGrid(exp, inputs)];
	for (let t = 0; t < opts.steps; t++) {
		const dmg = opts.damage && opts.damage.at === t + 1 ? opts.damage.mask : undefined;
		states.push(step(par, states[t], exp, inputs, dmg));
	}
	return states;
}

/** Read the output bits (channel 0 of each output cell) from a state. */
export function readOutputs(s: Float64Array, exp: Experiment): number[] {
	return exp.outputCells.map((cell) => s[cell * C + 0]);
}

/** A square keep-mask (1 = keep, 0 = destroy) centred at (cx,cy). */
export function damageMask(cx: number, cy: number, size: number): Uint8Array {
	const mask = new Uint8Array(N).fill(1);
	const h = size >> 1;
	for (let y = cy - h; y <= cy - h + size - 1; y++)
		for (let x = cx - h; x <= cx - h + size - 1; x++)
			if (x >= 0 && x < SW && y >= 0 && y < SH) mask[y * SW + x] = 0;
	return mask;
}
