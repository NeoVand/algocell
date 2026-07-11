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
	fireRate: number; // NCA-style STOCHASTIC updates: prob a cell updates each step (1 = synchronous).
	// <1 desynchronizes the CA → damps the period-3 limit cycle (the pole near ±120°) so attractors
	// are genuine fixed points, not blinkers. Non-firing cells keep their state; markers/inputs still clamp.
}

/** Build a config from grid + channel + hidden sizes. Param block layout is
 *  [ W1(HD×PERC), b1(HD), W2(C×HD), b2(C) ]. `markers` enables the movable-port rule. */
export function makeConfig(SW: number, SH: number, C: number, HD: number, markers = false, fireRate = 1): RuleConfig {
	const FEAT = 4, PERC = FEAT * C, N = SW * SH;
	const W1O = 0, B1O = W1O + HD * PERC, W2O = B1O + HD, B2O = W2O + C * HD, P = B2O + C;
	return { SW, SH, N, C, FEAT, PERC, HD, W1O, B1O, W2O, B2O, P, markers, fireRate };
}

/** Per-cell stochastic-update mask. MUST be bit-identical to the WGSL `fires()` so the GPU
 *  trainer validates against this CPU reference. Returns true if cell fires (updates) this step. */
export function cellFires(cell: number, step: number, seed: number, fireRate: number): boolean {
	if (fireRate >= 1) return true;
	let x = (Math.imul(cell, 0x9e3779b1) ^ Math.imul(step + 1, 0x85ebca77) ^ Math.imul(seed + 1, 0xc2b2ae3d)) >>> 0;
	x = Math.imul(x ^ (x >>> 16), 0x7feb352d) >>> 0;
	x = Math.imul(x ^ (x >>> 15), 0x846ca68b) >>> 0;
	x = (x ^ (x >>> 16)) >>> 0;
	return x < fireRate * 4294967296;
}

// Marker channel indices (for markers configs).
export const IN_MARK = 1, OUT_MARK = 2;

export const EDIM = makeConfig(9, 9, 12, 48); // E-series: gate / self-repair / grow (P=2940)
export const ADIM = makeConfig(11, 11, 16, 64); // 1-bit full adder (P=5200)
export const IDIM = makeConfig(17, 17, 16, 96, true); // movable wire — position-invariant, draggable ports (P=7792)
export const XDIM = makeConfig(17, 17, 20, 128, true); // movable XOR — more capacity for routing 2 signals + combine (P≈12948)
export const RDIM = makeConfig(17, 17, 16, 96, true, 0.5); // reactive movable XOR — STOCHASTIC updates (fireRate 0.5) damp the ring (P=7792)

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
	sigChannels?: number[]; // per-input injection channel (default: all ch0). XOR uses distinct
	// channels [3,4] so the two input bits stay separable and can be routed + XOR'd at the output;
	// ch0 stays the readout. A single shared channel blends the two waves → XOR unlearnable.
}

/** The channel each input's bit is clamped into. Defaults to ch0 for every input (wire/adder/
 *  E-series). Movable XOR overrides with distinct channels so the two bits stay decodable. */
export function inputChannels(exp: Experiment): number[] {
	return exp.sigChannels ?? exp.inputCells.map(() => 0);
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
		ic: 'full', paramsUrl: 'xor_invariant.json', tGrow: 50, stable: true, movable: true, sigChannels: [3, 4]
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

/** Stamp markers + input bits at the given port positions (for movable rules). `inCh[k]`
 *  is input k's injection channel (default ch0 — the wire's shared signal/readout channel). */
export function stampMarkers(cfg: RuleConfig, f: Float64Array, inPorts: number[], outPorts: number[], bits: number[], inCh?: number[]): void {
	const chans = inCh ?? inPorts.map(() => 0);
	for (let i = 0; i < cfg.N; i++) { f[i * cfg.C + IN_MARK] = 0; f[i * cfg.C + OUT_MARK] = 0; }
	inPorts.forEach((p, k) => { f[p * cfg.C + IN_MARK] = 1; f[p * cfg.C + chans[k]] = bits[k]; });
	for (const p of outPorts) f[p * cfg.C + OUT_MARK] = 1;
}

/** Uniform-alive interior + stamped markers — the initial state for a movable rule. Alive
 *  (hidden) channels start ABOVE the signal channels so the injected bits aren't pre-filled. */
export function seedMarkers(cfg: RuleConfig, inPorts: number[], outPorts: number[], bits: number[], inCh?: number[]): Float64Array {
	const chans = inCh ?? inPorts.map(() => 0);
	const aliveFrom = chans.some((c) => c > 0) ? Math.max(...chans) + 1 : 3;
	const s = new Float64Array(cfg.N * cfg.C);
	for (let y = 1; y < cfg.SH - 1; y++) for (let x = 1; x < cfg.SW - 1; x++)
		for (let c = aliveFrom; c < cfg.C; c++) s[(y * cfg.SW + x) * cfg.C + c] = 1;
	stampMarkers(cfg, s, inPorts, outPorts, bits, chans);
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

export interface RolloutOpts { steps: number; damage?: { at: number; mask: Uint8Array }; switchAt?: number; bits2?: number[]; seed?: number; }

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

/** One step of a MOVABLE (markers) rule: apply the rule everywhere, optionally damage,
 *  then RE-STAMP markers + inputs at the (possibly relocated) ports — exactly what the
 *  trainer and the WGSL kernel do. Ports are given explicitly (not read from the
 *  experiment) so it works at any placement, on any grid size. */
export function stepMarkers(cfg: RuleConfig, par: Float64Array, s: Float64Array, inPorts: number[], outPorts: number[], bits: number[], inCh: number[], damage?: Uint8Array, step = 0, seed = 0): Float64Array {
	const { SW, SH, C, HD, PERC, W1O, B1O, W2O, B2O, N, fireRate } = cfg;
	const ns = new Float64Array(N * C);
	const perc = new Float64Array(PERC), h = new Float64Array(HD);
	for (let y = 1; y < SH - 1; y++)
		for (let x = 1; x < SW - 1; x++) {
			const i = y * SW + x;
			if (!cellFires(i, step, seed, fireRate)) { for (let c = 0; c < C; c++) ns[i * C + c] = s[i * C + c]; continue; } // async: keep state
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
	stampMarkers(cfg, ns, inPorts, outPorts, bits, inCh); // damage first, then clamp wins (matches the kernel)
	return ns;
}

/** Reference rollout for a movable rule (CPU ground truth for the WGSL kernel). If
 *  `opts.switchAt`/`opts.bits2` are set, the injected input switches to `bits2` from that
 *  state onward (mid-rollout input change → trains/tests a REACTIVE rule). */
export function forwardMarkers(cfg: RuleConfig, par: Float64Array, inPorts: number[], outPorts: number[], bits: number[], inCh: number[], opts: RolloutOpts): Float64Array[] {
	const sw = opts.switchAt ?? Infinity, b2 = opts.bits2 ?? bits, seed = opts.seed ?? 0;
	const states: Float64Array[] = [seedMarkers(cfg, inPorts, outPorts, bits, inCh)];
	for (let t = 0; t < opts.steps; t++) {
		const stepBits = t + 1 >= sw ? b2 : bits; // state t+1 carries the post-switch input
		const dmg = opts.damage && opts.damage.at === t + 1 ? opts.damage.mask : undefined;
		states.push(stepMarkers(cfg, par, states[t], inPorts, outPorts, stepBits, inCh, dmg, t, seed));
	}
	return states;
}

export interface MovSample {
	inPorts: number[]; outPort: number; bits: number[]; inCh: number[]; target: number;
	// optional REACTIVE transition: input switches to bits2 at state `tSwitch`, output must
	// re-settle to target2. Absent → non-reactive (single input, single answer).
	tSwitch?: number; bits2?: number[]; target2?: number;
}

/** Loss + gradient for a batch of movable samples — the CPU ground truth for the GPU trainer
 *  (identical BPTT to expI.ts, marker/input-channel clamped). norm = batch size (mean over
 *  samples). `whold` averages the output MSE over the LAST `whold` states (a persistence /
 *  long-horizon-stability objective: hold the answer, don't drift) — whold=1 is single-readout.
 *  Reuses forwardMarkers for the rollout. f64 throughout. */
export function lossAndGradMarkers(cfg: RuleConfig, par: Float64Array, samples: MovSample[], steps: number, whold = 1, seed = 0): { L: number; grad: Float64Array } {
	const { SW, SH, C, HD, PERC, FEAT, W1O, B1O, W2O, B2O, P, N, fireRate } = cfg;
	const grad = new Float64Array(P);
	let L = 0;
	const norm = samples.length * whold;
	const perc = new Float64Array(PERC), pre1 = new Float64Array(HD), hbuf = new Float64Array(HD), gh = new Float64Array(HD), gperc = new Float64Array(PERC);
	for (const s of samples) {
		const sw = s.tSwitch ?? 0; // 0 = non-reactive (single input); >0 = flip to bits2 at state sw
		const b2 = s.bits2 ?? s.bits, tgt2 = s.target2 ?? s.target;
		// window A holds the pre-switch answer (target); window B holds the post-switch answer (tgt2).
		const inWindowA = (t: number) => sw > 0 && t >= sw - whold && t < sw;
		const inWindowB = (t: number) => t >= steps - whold + 1 && t < steps;
		const states = forwardMarkers(cfg, par, s.inPorts, [s.outPort], s.bits, s.inCh, { steps, switchAt: sw > 0 ? sw : undefined, bits2: b2, seed });
		const oc = s.outPort * C + 0;
		const dT = states[steps][oc] - tgt2; // final state = post-switch answer
		L += (dT * dT) / norm;
		let gs = new Float64Array(N * C);
		gs[oc] = (2 * dT) / norm; // seed at final state (last window-B step)
		for (let t = steps - 1; t >= 0; t--) {
			for (let i = 0; i < N; i++) { gs[i * C + IN_MARK] = 0; gs[i * C + OUT_MARK] = 0; }
			for (let k = 0; k < s.inPorts.length; k++) gs[s.inPorts[k] * C + s.inCh[k]] = 0;
			const st = states[t], sp = states[t + 1];
			const gsPrev = new Float64Array(N * C);
			for (let y = 1; y < SH - 1; y++)
				for (let x = 1; x < SW - 1; x++) {
					const i = y * SW + x;
					// async: a non-firing cell was identity (ns=s) → gradient passes straight through, no param grad
					if (!cellFires(i, t, seed, fireRate)) { for (let c = 0; c < C; c++) gsPrev[i * C + c] += gs[i * C + c]; continue; }
					perceive(cfg, st, i, perc);
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
					const r = i + 1, l = i - 1, u = i - SW, d = i + SW;
					for (let ch = 0; ch < C; ch++) {
						const bb = ch * FEAT, gId = gperc[bb], gGx = gperc[bb + 1], gGy = gperc[bb + 2], gLap = gperc[bb + 3];
						gsPrev[i * C + ch] += gId - 4 * gLap;
						gsPrev[r * C + ch] += 0.5 * gGx + gLap;
						gsPrev[l * C + ch] += -0.5 * gGx + gLap;
						gsPrev[d * C + ch] += 0.5 * gGy + gLap;
						gsPrev[u * C + ch] += -0.5 * gGy + gLap;
					}
				}
			// persistence + reactivity: state[t]'s output is scored when t is in a hold window →
			// add the direct loss-gradient at the output cell (window A → target, window B → tgt2).
			if (inWindowA(t) || inWindowB(t)) {
				const dt = states[t][oc] - (t < sw ? s.target : tgt2);
				L += (dt * dt) / norm;
				gsPrev[oc] += (2 * dt) / norm;
			}
			gs = gsPrev;
		}
	}
	return { L, grad };
}

// --- Fixed-layout, non-marker, multi-input/multi-output rule (gate / adder) -------------
// CPU ground truth for the GPU trainer's NON-marker path. Mirrors it exactly: seed the
// interior (channels aliveFrom..C-1 = 1), clamp N inputs into their channels each step,
// score M outputs' channel 0 over the last `whold` states. No markers, synchronous (fireRate=1),
// non-reactive. Loss normalization = samples.length * whold (matches WGSL NORMF), summed over
// output cells (no division by #outputs) — so it validates the GPU gradient to ~f32 precision.
export interface FixedSample { inPorts: number[]; inCh: number[]; bits: number[]; outPorts: number[]; targets: number[]; }

export function seedFixed(cfg: RuleConfig, aliveFrom: number, s: FixedSample): Float64Array {
	const { N, C, SW, SH } = cfg;
	const st = new Float64Array(N * C);
	for (let y = 1; y < SH - 1; y++) for (let x = 1; x < SW - 1; x++) for (let c = aliveFrom; c < C; c++) st[(y * SW + x) * C + c] = 1;
	s.inPorts.forEach((p, k) => { st[p * C + s.inCh[k]] = s.bits[k]; });
	return st;
}

function stepFixed(cfg: RuleConfig, par: Float64Array, s: Float64Array, sample: FixedSample): Float64Array {
	const { SW, SH, C, HD, PERC, W1O, B1O, W2O, B2O, N } = cfg;
	const ns = new Float64Array(N * C);
	const perc = new Float64Array(PERC), h = new Float64Array(HD);
	for (let y = 1; y < SH - 1; y++) for (let x = 1; x < SW - 1; x++) {
		const i = y * SW + x;
		perceive(cfg, s, i, perc);
		for (let hh = 0; hh < HD; hh++) { let a = par[B1O + hh]; const base = W1O + hh * PERC; for (let k = 0; k < PERC; k++) a += par[base + k] * perc[k]; h[hh] = a > 0 ? a : 0; }
		for (let c = 0; c < C; c++) { let dl = par[B2O + c]; const base = W2O + c * HD; for (let hh = 0; hh < HD; hh++) dl += par[base + hh] * h[hh]; ns[i * C + c] = Math.tanh(s[i * C + c] + dl); }
	}
	sample.inPorts.forEach((p, k) => { ns[p * C + sample.inCh[k]] = sample.bits[k]; });
	return ns;
}

export function lossAndGradFixed(cfg: RuleConfig, par: Float64Array, samples: FixedSample[], steps: number, aliveFrom: number, whold = 1): { L: number; grad: Float64Array } {
	const { SW, SH, C, HD, PERC, FEAT, W1O, B1O, W2O, B2O, P, N } = cfg;
	const grad = new Float64Array(P);
	let L = 0;
	const norm = samples.length * whold;
	const perc = new Float64Array(PERC), pre1 = new Float64Array(HD), hbuf = new Float64Array(HD), gh = new Float64Array(HD), gperc = new Float64Array(PERC);
	for (const s of samples) {
		const states: Float64Array[] = [seedFixed(cfg, aliveFrom, s)];
		for (let t = 0; t < steps; t++) states.push(stepFixed(cfg, par, states[t], s));
		let gs = new Float64Array(N * C);
		for (let k = 0; k < s.outPorts.length; k++) {
			const oc = s.outPorts[k] * C + 0;
			const d = states[steps][oc] - s.targets[k];
			L += (d * d) / norm;
			gs[oc] += (2 * d) / norm;
		}
		for (let t = steps - 1; t >= 0; t--) {
			for (let k = 0; k < s.inPorts.length; k++) gs[s.inPorts[k] * C + s.inCh[k]] = 0;
			const st = states[t], sp = states[t + 1];
			const gsPrev = new Float64Array(N * C);
			for (let y = 1; y < SH - 1; y++) for (let x = 1; x < SW - 1; x++) {
				const i = y * SW + x;
				perceive(cfg, st, i, perc);
				for (let hh = 0; hh < HD; hh++) { let a = par[B1O + hh]; const base = W1O + hh * PERC; for (let k = 0; k < PERC; k++) a += par[base + k] * perc[k]; pre1[hh] = a; hbuf[hh] = a > 0 ? a : 0; }
				gh.fill(0);
				for (let c = 0; c < C; c++) {
					const spv = sp[i * C + c];
					const gp = gs[i * C + c] * (1 - spv * spv);
					gsPrev[i * C + c] += gp; grad[B2O + c] += gp;
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
				const r = i + 1, l = i - 1, u = i - SW, d = i + SW;
				for (let ch = 0; ch < C; ch++) {
					const bb = ch * FEAT, gId = gperc[bb], gGx = gperc[bb + 1], gGy = gperc[bb + 2], gLap = gperc[bb + 3];
					gsPrev[i * C + ch] += gId - 4 * gLap;
					gsPrev[r * C + ch] += 0.5 * gGx + gLap; gsPrev[l * C + ch] += -0.5 * gGx + gLap;
					gsPrev[d * C + ch] += 0.5 * gGy + gLap; gsPrev[u * C + ch] += -0.5 * gGy + gLap;
				}
			}
			// persistence window (whold>1): also score states[t]'s outputs
			if (whold > 1 && t >= steps - whold + 1 && t < steps) {
				for (let k = 0; k < s.outPorts.length; k++) {
					const oc = s.outPorts[k] * C + 0;
					const d = states[t][oc] - s.targets[k];
					L += (d * d) / norm;
					gsPrev[oc] += (2 * d) / norm;
				}
			}
			gs = gsPrev;
		}
	}
	return { L, grad };
}

export function damageMask(cfg: RuleConfig, cx: number, cy: number, size: number): Uint8Array {
	const mask = new Uint8Array(cfg.N).fill(1);
	const h = size >> 1;
	for (let y = cy - h; y <= cy - h + size - 1; y++)
		for (let x = cx - h; x <= cx - h + size - 1; x++)
			if (x >= 0 && x < cfg.SW && y >= 0 && y < cfg.SH) mask[y * cfg.SW + x] = 0;
	return mask;
}
