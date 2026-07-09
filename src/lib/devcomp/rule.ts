// Shared spec for developmental computation (E1/E2/E3, the adder, and beyond).
//
// Single source of truth for the CA rule: dimensions/parameter layout (a
// RuleConfig, so different experiments can use different grid/channel sizes),
// the reference `forward()` stepper (identical to the trainers in
// `src/lib/morph/dev/exp{F,G,H}.ts`), and per-experiment I/O layouts. The trainer
// (Node), the WGSL kernel (GPU), and the demo UI all agree with THIS file.
//
// The rule (per interior cell, per step):
//   perceive = [identity, gx=(right−left)/2, gy=(down−up)/2, laplacian] per channel
//   dl       = W2 · relu(W1 · perceive + b1) + b2
//   state'   = tanh(state + dl)      ← residual is INSIDE the tanh
// Inputs are clamped (channel 0) every step; border cells stay 0 forever.

export interface RuleConfig {
	SW: number; SH: number; N: number; C: number; FEAT: number; PERC: number; HD: number;
	W1O: number; B1O: number; W2O: number; B2O: number; P: number;
	markers: boolean; // if true: ch1 = IN_MARK, ch2 = OUT_MARK — a position-invariant rule with movable ports
}

/** Build a config from grid + channel + hidden sizes. Param block layout is
 *  [ W1(HD×PERC), b1(HD), W2(C×HD), b2(C) ]. `markers` enables the movable-port rule. */
export function makeConfig(SW: number, SH: number, C: number, HD: number, markers = false): RuleConfig {
	const FEAT = 4, PERC = FEAT * C, N = SW * SH;
	const W1O = 0, B1O = W1O + HD * PERC, W2O = B1O + HD, B2O = W2O + C * HD, P = B2O + C;
	return { SW, SH, N, C, FEAT, PERC, HD, W1O, B1O, W2O, B2O, P, markers };
}

// Marker channel indices (for markers configs).
export const IN_MARK = 1, OUT_MARK = 2;

export const EDIM = makeConfig(9, 9, 12, 48); // E-series: gate / self-repair / grow (P=2940)
export const ADIM = makeConfig(11, 11, 16, 64); // 1-bit full adder (P=5200)
export const IDIM = makeConfig(17, 17, 16, 96, true); // movable XOR — position-invariant, draggable ports (P=7792)

// Back-compat exports (E-series dims). New code should read dims from an experiment's cfg.
export const { SW, SH, N, C, FEAT, PERC, HD, W1O, B1O, W2O, B2O, P } = EDIM;

export type IC = 'full' | 'seed';

export interface Experiment {
	id: string;
	name: string;
	blurb: string;
	cfg: RuleConfig;
	inputCells: number[]; // cells whose channel-0 is clamped to the input bit
	outputCells: number[]; // cells whose channel-0 is read as an output bit
	outputLabels?: string[]; // optional per-output labels (e.g. ['sum','carry'])
	cases: { in: number[]; out: number[] }[]; // truth table
	ic: IC; // initial condition: full substrate, or a single seed cell
	seedCell?: number; // for ic === 'seed'
	paramsUrl: string; // JSON number[] of length cfg.P
	tGrow: number; // steps until the answer has settled (for a readout)
	stable: boolean; // true if the rule is a long-horizon attractor (run indefinitely); else cap at tGrow
	reactive?: boolean; // true if it re-settles on live input changes (toggle re-clamps, no re-seed)
	movable?: boolean; // true if ports are draggable (a markers/position-invariant rule)
}

// --- E-series I/O (9×9) ---
const e_iy = EDIM.SH >> 1, e_inCol = 2, e_outCol = EDIM.SW - 2;
const IN_TOP = (e_iy - 1) * EDIM.SW + e_inCol, IN_BOT = (e_iy + 1) * EDIM.SW + e_inCol;
const E_OUT = e_iy * EDIM.SW + e_outCol;
const XOR_CASES = [
	{ in: [0, 0], out: [0] }, { in: [0, 1], out: [1] }, { in: [1, 0], out: [1] }, { in: [1, 1], out: [0] }
];

// --- Adder I/O (11×11): 3 inputs (adjacent rows), 2 outputs (sum, carry) ---
const a_iy = ADIM.SH >> 1, a_inCol = 2, a_outCol = ADIM.SW - 3;
const ADD_IN = [(a_iy - 1) * ADIM.SW + a_inCol, a_iy * ADIM.SW + a_inCol, (a_iy + 1) * ADIM.SW + a_inCol];
const ADD_OUT = [(a_iy - 1) * ADIM.SW + a_outCol, (a_iy + 1) * ADIM.SW + a_outCol];
const ADD_CASES: { in: number[]; out: number[] }[] = [];
for (let a = 0; a < 2; a++) for (let b = 0; b < 2; b++) for (let cin = 0; cin < 2; cin++)
	ADD_CASES.push({ in: [a, b, cin], out: [a ^ b ^ cin, a + b + cin >= 2 ? 1 : 0] });

export const EXPERIMENTS: Experiment[] = [
	{
		id: 'e1_gate', name: 'XOR gate', blurb: 'Two inputs, one output: computes their XOR at a cell 5 away.',
		cfg: EDIM, inputCells: [IN_TOP, IN_BOT], outputCells: [E_OUT], cases: XOR_CASES,
		ic: 'full', paramsUrl: 'e1_gate.json', tGrow: 24, stable: false
	},
	{
		id: 'e2_repair', name: 'Self-repair', blurb: 'Destroy a patch mid-computation — it regrows and still computes XOR.',
		cfg: EDIM, inputCells: [IN_TOP, IN_BOT], outputCells: [E_OUT], cases: XOR_CASES,
		ic: 'full', paramsUrl: 'e3_seed.json', tGrow: 24, stable: true // uses the stable E3 rule
	},
	{
		id: 'e3_seed', name: 'Grow from seed', blurb: 'Grow the whole computer from a single seed cell, then compute + heal.',
		cfg: EDIM, inputCells: [IN_TOP, IN_BOT], outputCells: [E_OUT], cases: XOR_CASES,
		ic: 'seed', seedCell: e_iy * EDIM.SW + (EDIM.SW >> 1), paramsUrl: 'e3_seed.json', tGrow: 24, stable: true
	},
	{
		id: 'adder', name: '1-bit adder', blurb: 'Three inputs → two outputs: a full adder (sum = a⊕b⊕cin, carry = majority) that holds its answer, tracks live input changes, and self-repairs. Arithmetic, grown by gradient.',
		cfg: ADIM, inputCells: ADD_IN, outputCells: ADD_OUT, outputLabels: ['sum', 'carry'], cases: ADD_CASES,
		ic: 'full', paramsUrl: 'adder_reactive.json', tGrow: 30, stable: true, reactive: true // stable + self-repairing + input-reactive
	},
	{
		id: 'movable_wire', name: 'Movable wire', blurb: 'One rule, no fixed layout: drag the input (○) or output (□) port anywhere and the plane rewires to route the bit. Works on any grid size.',
		cfg: IDIM, inputCells: [movCell(IDIM, IDIM.SH >> 1, 4)], outputCells: [movCell(IDIM, IDIM.SH >> 1, IDIM.SW - 5)],
		cases: [{ in: [0], out: [0] }, { in: [1], out: [1] }],
		ic: 'full', paramsUrl: 'wire_invariant.json', tGrow: 40, stable: true, movable: true
	},
	{
		id: 'movable_xor', name: 'Movable XOR', blurb: 'Position-invariant computation: drag the two inputs (○) or the output (□) anywhere and the plane rewires to compute their XOR. Watch the waves from the ports find each other.',
		cfg: IDIM, inputCells: [movCell(IDIM, (IDIM.SH >> 1) - 2, 4), movCell(IDIM, (IDIM.SH >> 1) + 2, 4)], outputCells: [movCell(IDIM, IDIM.SH >> 1, IDIM.SW - 5)],
		cases: XOR_CASES,
		ic: 'full', paramsUrl: 'xor_invariant.json', tGrow: 44, stable: true, movable: true
	}
];

function movCell(cfg: RuleConfig, row: number, col: number): number { return row * cfg.SW + col; }

export function experimentById(id: string): Experiment | undefined {
	return EXPERIMENTS.find((e) => e.id === id);
}

export function loadParams(cfg: RuleConfig, arr: number[]): Float64Array {
	if (arr.length !== cfg.P) throw new Error(`params length ${arr.length} !== ${cfg.P}`);
	return Float64Array.from(arr);
}

export function clampInputs(cfg: RuleConfig, f: Float64Array, exp: Experiment, inputs: number[]): void {
	for (let k = 0; k < exp.inputCells.length; k++) f[exp.inputCells[k] * cfg.C + 0] = inputs[k];
}

/** Stamp markers + input bits at the given port positions (for movable rules). */
export function stampMarkers(cfg: RuleConfig, f: Float64Array, inPorts: number[], outPorts: number[], bits: number[]): void {
	for (let i = 0; i < cfg.N; i++) { f[i * cfg.C + IN_MARK] = 0; f[i * cfg.C + OUT_MARK] = 0; }
	inPorts.forEach((p, k) => { f[p * cfg.C + IN_MARK] = 1; f[p * cfg.C + 0] = bits[k]; });
	for (const p of outPorts) f[p * cfg.C + OUT_MARK] = 1;
}

/** Uniform-alive interior + stamped markers — the initial state for a movable rule. */
export function seedMarkers(cfg: RuleConfig, inPorts: number[], outPorts: number[], bits: number[]): Float64Array {
	const s = new Float64Array(cfg.N * cfg.C);
	for (let y = 1; y < cfg.SH - 1; y++) for (let x = 1; x < cfg.SW - 1; x++)
		for (let c = 3; c < cfg.C; c++) s[(y * cfg.SW + x) * cfg.C + c] = 1;
	stampMarkers(cfg, s, inPorts, outPorts, bits);
	return s;
}

export function seedGrid(cfg: RuleConfig, exp: Experiment, inputs: number[]): Float64Array {
	const s = new Float64Array(cfg.N * cfg.C);
	if (exp.ic === 'full') {
		for (let y = 1; y < cfg.SH - 1; y++) for (let x = 1; x < cfg.SW - 1; x++)
			for (let c = 1; c < cfg.C; c++) s[(y * cfg.SW + x) * cfg.C + c] = 1;
	} else {
		const sc = exp.seedCell ?? (cfg.SH >> 1) * cfg.SW + (cfg.SW >> 1);
		for (let c = 1; c < cfg.C; c++) s[sc * cfg.C + c] = 1;
	}
	clampInputs(cfg, s, exp, inputs);
	return s;
}

export function perceive(cfg: RuleConfig, s: Float64Array, i: number, out: Float64Array): void {
	const { SW, C, FEAT } = cfg;
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

export function step(cfg: RuleConfig, par: Float64Array, s: Float64Array, exp: Experiment, inputs: number[], damage?: Uint8Array): Float64Array {
	const { SW, SH, C, HD, PERC, W1O, B1O, W2O, B2O, N } = cfg;
	const ns = new Float64Array(N * C);
	const perc = new Float64Array(PERC), h = new Float64Array(HD);
	for (let y = 1; y < SH - 1; y++)
		for (let x = 1; x < SW - 1; x++) {
			const i = y * SW + x;
			perceive(cfg, s, i, perc);
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
	clampInputs(cfg, ns, exp, inputs);
	return ns;
}

export interface RolloutOpts { steps: number; damage?: { at: number; mask: Uint8Array }; }

export function forward(cfg: RuleConfig, par: Float64Array, exp: Experiment, inputs: number[], opts: RolloutOpts): Float64Array[] {
	const states: Float64Array[] = [seedGrid(cfg, exp, inputs)];
	for (let t = 0; t < opts.steps; t++) {
		const dmg = opts.damage && opts.damage.at === t + 1 ? opts.damage.mask : undefined;
		states.push(step(cfg, par, states[t], exp, inputs, dmg));
	}
	return states;
}

export function readOutputs(cfg: RuleConfig, s: Float64Array, exp: Experiment): number[] {
	return exp.outputCells.map((cell) => s[cell * cfg.C + 0]);
}

export function damageMask(cfg: RuleConfig, cx: number, cy: number, size: number): Uint8Array {
	const mask = new Uint8Array(cfg.N).fill(1);
	const h = size >> 1;
	for (let y = cy - h; y <= cy - h + size - 1; y++)
		for (let x = cx - h; x <= cx - h + size - 1; x++)
			if (x >= 0 && x < cfg.SW && y >= 0 && y < cfg.SH) mask[y * cfg.SW + x] = 0;
	return mask;
}
