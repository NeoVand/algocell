// Signed fixed-point emulation of the developmental-computation rule — a
// BIT-FAITHFUL model of what a Z80 program would compute, so the quantization
// study (`quantize.ts`) predicts the real substrate exactly.
//
// Format Q(W.F): a value x is stored as the signed integer round(x·2^F),
// saturated to W bits (range [-2^(W-1), 2^(W-1)-1]). Every operation here is
// exactly realizable on the Z80 (add, arithmetic shift, signed multiply, a
// tanh table) — no float ever touches the datapath. The one modelling choice
// that matters for accuracy is the multiply-accumulate: products W1·perceive
// are summed in a WIDE accumulator (Q(2F), exact — BigInt here, a 32/48-bit
// register pair on the Z80) and the >>F reduction happens ONCE at the end. That
// is standard DSP practice and the most accurate faithful choice.

export interface Fmt {
	W: number; // total bits (incl. sign)
	F: number; // fractional bits  → integer bits = W - F (incl. sign)
}
export const fmt = (W: number, F: number): Fmt => ({ W, F });
export const fmtName = (f: Fmt): string => `Q${f.W - f.F}.${f.F}`;

const SCALE = (F: number): number => 2 ** F;

/** Saturate an integer to W signed bits (models Z80 saturation on store). */
export function clampW(x: number, W: number): number {
	const hi = 2 ** (W - 1) - 1;
	const lo = -(2 ** (W - 1));
	return x > hi ? hi : x < lo ? lo : x;
}
function clampWBig(x: bigint, W: number): number {
	const hi = (1n << BigInt(W - 1)) - 1n;
	const lo = -(1n << BigInt(W - 1));
	const c = x > hi ? hi : x < lo ? lo : x;
	return Number(c);
}

export const toQ = (x: number, f: Fmt): number => clampW(Math.round(x * SCALE(f.F)), f.W);
export const fromQ = (q: number, f: Fmt): number => q / SCALE(f.F);

/** Q(F)·Q(F) → Q(F): exact wide product, single arithmetic >>F (floor), W-bit store. */
export function mulQ(a: number, b: number, f: Fmt): number {
	const p = BigInt(a) * BigInt(b);
	return clampWBig(p >> BigInt(f.F), f.W);
}

/** Track how often saturation fires — a saturating value means too few integer bits. */
export interface OverflowStats {
	stores: number;
	saturations: number;
	maxAbsPre: number; // largest |pre-activation| seen (real units) — sets the integer-bit budget
	maxAbsDl: number; // largest |dl| seen
	accBitsMax: number; // widest accumulator magnitude seen (bits) — sets the accumulator width
}
export const newOverflow = (): OverflowStats => ({ stores: 0, saturations: 0, maxAbsPre: 0, maxAbsDl: 0, accBitsMax: 0 });

/**
 * Fixed-point dot product  Σ w[k]·x[k] + bias  (all Q(F)), reduced to a Q(F)
 * integer. Products accumulate exactly in Q(2F) (BigInt), bias is shifted in,
 * then a single arithmetic >>F (floor — what the Z80 signed shift does) and a
 * W-bit saturating store. Mirrors a Z80 MAC loop with a wide accumulator.
 */
export function dotQ(w: number[], x: number[], bias: number, f: Fmt, ov?: OverflowStats): number {
	const F = BigInt(f.F);
	let acc = BigInt(bias) << F; // bias Q(F) → Q(2F)
	for (let k = 0; k < w.length; k++) acc += BigInt(w[k]) * BigInt(x[k]); // Q(F)·Q(F) = Q(2F)
	if (ov) {
		const mag = acc < 0n ? -acc : acc;
		ov.accBitsMax = Math.max(ov.accBitsMax, mag === 0n ? 0 : mag.toString(2).length);
	}
	const q = acc >> F; // arithmetic shift right = floor, matches signed Z80 shift
	const hi = (1n << BigInt(f.W - 1)) - 1n;
	const lo = -(1n << BigInt(f.W - 1));
	if (ov) {
		ov.stores++;
		if (q > hi || q < lo) ov.saturations++;
	}
	return clampWBig(q, f.W);
}

/** ReLU in fixed point — a signed compare against 0, trivial on the Z80. */
export const reluQ = (q: number): number => (q > 0 ? q : 0);

// ---- tanh in fixed point -------------------------------------------------
// Two models: `ideal` (round-trip through Math.tanh at Q(F) resolution — the
// best a table could do) and a real N-entry lookup table with linear interp,
// to measure the table-size the Z80 actually needs.

export function tanhIdeal(qx: number, f: Fmt): number {
	return toQ(Math.tanh(fromQ(qx, f)), f);
}

/** Build an N-entry tanh table over [-XR, XR] in Q(F). Beyond ±XR, tanh≈±1. */
export function buildTanhTable(f: Fmt, N: number, XR: number): Int32Array {
	const t = new Int32Array(N + 1);
	for (let i = 0; i <= N; i++) {
		const x = -XR + (2 * XR * i) / N;
		t[i] = toQ(Math.tanh(x), f);
	}
	return t;
}
export function tanhTable(qx: number, f: Fmt, table: Int32Array, N: number, XR: number): number {
	const x = fromQ(qx, f);
	if (x <= -XR) return toQ(-1, f);
	if (x >= XR) return toQ(1, f);
	const pos = ((x + XR) / (2 * XR)) * N;
	const i = Math.floor(pos);
	const frac = pos - i;
	const a = table[i];
	const b = table[i + 1];
	return clampW(Math.round(a + (b - a) * frac), f.W);
}

// ---- forward-mode dual numbers in fixed point ----------------------------
// A value carries its tangent v̇ = d(value)/d(the one seeded weight). Every op
// below is a fixed-point op with a fixed-point tangent — the exact dual
// arithmetic Exp A proved runs on the Z80, now for the rule's actual datapath.
// This is how the substrate hands back its own training gradient (forward mode:
// O(params) passes, O(1) memory, embarrassingly parallel over weights).

export interface Dual {
	v: number; // value   Q(F)
	d: number; // tangent Q(F)
}
export const K = (x: number, f: Fmt): Dual => ({ v: toQ(x, f), d: 0 }); // a constant (no tangent)

/** Dual dot product Σ w·x + bias, wide-accumulated in both value and tangent. */
export function dotDual(w: Dual[], x: Dual[], bias: Dual, f: Fmt): Dual {
	const F = BigInt(f.F);
	let av = BigInt(bias.v) << F;
	let ad = BigInt(bias.d) << F;
	for (let k = 0; k < w.length; k++) {
		const wv = BigInt(w[k].v), wd = BigInt(w[k].d), xv = BigInt(x[k].v), xd = BigInt(x[k].d);
		av += wv * xv;
		ad += wv * xd + wd * xv; // product rule
	}
	return { v: clampWBig(av >> F, f.W), d: clampWBig(ad >> F, f.W) };
}

/** ReLU on a dual — gate value and tangent together (subgradient 0 at/below 0). */
export const reluDual = (a: Dual): Dual => (a.v > 0 ? a : { v: 0, d: 0 });

/** tanh on a dual using a fixed-point derivative table (1−tanh²), Q-multiplied
 *  into the incoming tangent — exactly what the Z80 does (two table lookups). */
export function tanhDualTab(a: Dual, f: Fmt, tanhTab: Int32Array, derTab: Int32Array, N: number, XR: number): Dual {
	const v = tanhTable(a.v, f, tanhTab, N, XR);
	const der = tanhTable_(a.v, f, derTab, N, XR); // (1−tanh²) sampled at a.v
	return { v, d: mulQ(der, a.d, f) };
}
/** ideal-tanh dual (isolates MAC/gradient quantization from table error). */
export function tanhDualIdeal(a: Dual, f: Fmt): Dual {
	const t = Math.tanh(fromQ(a.v, f));
	return { v: toQ(t, f), d: mulQ(toQ(1 - t * t, f), a.d, f) };
}
// table sampler that returns the raw Q value at x (no ±1 clamp — used for the
// derivative table, whose ends are ≈0 not ±1).
function tanhTable_(qx: number, f: Fmt, table: Int32Array, N: number, XR: number): number {
	const x = fromQ(qx, f);
	if (x <= -XR || x >= XR) return table[x <= -XR ? 0 : N];
	const pos = ((x + XR) / (2 * XR)) * N;
	const i = Math.floor(pos);
	const frac = pos - i;
	return clampW(Math.round(table[i] + (table[i + 1] - table[i]) * frac), f.W);
}
export function buildDerivTable(f: Fmt, N: number, XR: number): Int32Array {
	const t = new Int32Array(N + 1);
	for (let i = 0; i <= N; i++) {
		const x = -XR + (2 * XR * i) / N;
		const th = Math.tanh(x);
		t[i] = toQ(1 - th * th, f);
	}
	return t;
}
