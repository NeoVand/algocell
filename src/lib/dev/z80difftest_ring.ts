// Ring-size differential test of the GPU Z80 (DEV ONLY — never imported by the app runtime).
//
// z80difftest.ts compares the shipping core (zilion + Algocell's host bits) with the z80-emulator
// reference on 32- and 38-byte pair memories. The soup also runs pair rings of other sizes, and on a
// ring whose size P does not divide 65,536 (38, 40, 100, ...) the 16-bit wrap 0xFFFF → 0x0000 is a
// seam: address 0xFFFF sits at position 65535 mod P (15 at P = 40), address 0x0000 at position 0. A
// jump, stack move or block copy that crosses address 0 therefore lands at (65,536 + t) mod P, not
// at t mod P. This module extends the test to any even P (default sizes 32, 38, 40, 64, 100, 128):
//
//   • random cases — uniformly random pair memories under the soup's conventions: every register 0,
//     SP = spInit(P) (the largest 16-bit value aliasing the last byte of the partner), PC = 0,
//     128 instructions, no suppression;
//   • targeted cases — programs that cross address 0 by JR, DJNZ, sequential PC run-off, PUSH/POP
//     (explicit SP and the soup's initial SP), CALL/RET, EX (SP),HL, LDIR/LDDR, (IX+d)/(IY+d) and
//     16-bit memory operands. Each terminated probe carries an ANALYTIC expectation derived from first
//     principles (form the 16-bit address, then take it mod P), checked against the reference and the
//     GPU separately, so the reference's own memory mapping is verified, not assumed.
//
// The reference is ./z80oracle.ts, unchanged: its HAL maps every bus address as `address % P`, and
// z80-emulator masks every address it puts on the bus to 16 bits (PC/SP/HL/IX arithmetic goes through
// inc16/add16/`& 0xFFFF`), so it already implements (16-bit address) mod P for any P.
//
// Programs whose final state differs from the reference are re-compared instruction by instruction to the
// end of the run (traceAgainstReference): known z80-emulator defects are corrected where they occur and each
// correction is checked against the GPU at that instruction; known zilion deviations are named and stepped
// over; anything else is reported as unexplained.
//
// The module is GPU-backend agnostic (results come in as arrays). The headless driver is
// ./z80difftest_ring.node.ts, which dispatches createZ80TestShader(L, P) through wgpu-py
// (experiments/z80_ring_gpu.py).

import { Z80, type Hal } from 'z80-emulator';
import type { OracleResult } from './z80oracle';
import { spInit } from '$lib/sim/constants';
import { disassemble } from '$lib/z80-disasm';

export const RING_SIZES = [32, 38, 40, 64, 100, 128] as const;
export const REGS_PER_CASE = 12; // a,f,b,c,d,e,h,l,sp,pc,writes_a,writes_b (shader readback layout)
export const REG_NAMES = ['a', 'f', 'b', 'c', 'd', 'e', 'h', 'l', 'sp', 'pc'] as const;
export type RegName = (typeof REG_NAMES)[number];
export type Regs = Record<RegName, number>;

const F3F5 = 0x08 | 0x20; // undocumented flag bits

// ── Deterministic RNG (same generator as z80difftest.ts) ───────────────────────

export interface Rng {
	(): number;
	int(lo: number, hi: number): number; // inclusive
	bool(): boolean;
	pick<T>(xs: readonly T[]): T;
}

export function makeRng(seed: number): Rng {
	let a = seed >>> 0;
	const next = () => {
		a |= 0;
		a = (a + 0x6d2b79f5) | 0;
		let t = Math.imul(a ^ (a >>> 15), 1 | a);
		t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
	const r = next as Rng;
	r.int = (lo, hi) => lo + Math.floor(next() * (hi - lo + 1));
	r.bool = () => next() < 0.5;
	r.pick = (xs) => xs[Math.floor(next() * xs.length)];
	return r;
}

// ── Address model: 16-bit address, then position on the P-byte ring ─────────────

export const m16 = (x: number) => x & 0xffff;
export const ringPos = (addr: number, P: number) => m16(addr) % P;

// ── Cases ─────────────────────────────────────────────────────────────────────

export interface Expectation {
	regs?: Partial<Regs>;
	mem?: Array<[pos: number, value: number]>;
}

export type Variant = 'random' | 'halt-fill' | 'random-fill' | 'open';

export interface RingCase {
	family: string; // 'random' or a targeted family name
	variant: Variant; // halt-fill / random-fill: terminated probe with expectation; open: probe runs on into random bytes
	input: Uint8Array; // P bytes: tape A then tape B
	expect?: Expectation;
}

export function randomCases(P: number, n: number, seed: number): RingCase[] {
	const rng = makeRng(seed);
	const out: RingCase[] = [];
	for (let c = 0; c < n; c++) {
		const b = new Uint8Array(P);
		for (let i = 0; i < P; i++) b[i] = (rng() * 256) | 0;
		out.push({ family: 'random', variant: 'random', input: b });
	}
	return out;
}

/** Pair memory under construction: filler everywhere, probe bytes placed at 16-bit addresses. */
class Builder {
	readonly mem: Uint8Array;
	readonly used = new Set<number>();
	constructor(
		readonly P: number,
		rng: Rng,
		fill: 'halt' | 'random'
	) {
		this.mem = new Uint8Array(P);
		for (let i = 0; i < P; i++) this.mem[i] = fill === 'halt' ? 0x76 : (rng() * 256) | 0;
	}
	/** Place bytes at consecutive 16-bit addresses; returns their ring positions, or null on a collision. */
	put(addr: number, bytes: number[]): number[] | null {
		const ps = bytes.map((_, i) => ringPos(addr + i, this.P));
		if (new Set(ps).size !== ps.length || ps.some((p) => this.used.has(p))) return null;
		ps.forEach((p, i) => {
			this.used.add(p);
			this.mem[p] = bytes[i] & 0xff;
		});
		return ps;
	}
	at(addr: number): number {
		return this.mem[ringPos(addr, this.P)];
	}
	word(addr: number): number {
		return this.at(addr) | (this.at(m16(addr + 1)) << 8);
	}
}

const HALT = 0x76;
const lo = (v: number) => v & 0xff;
const hi = (v: number) => (v >> 8) & 0xff;

interface PairOps {
	name: 'bc' | 'de' | 'hl' | 'sp' | 'ix';
	ld: number[]; // LD rr,nn opcode bytes (operand follows)
	push?: number[];
	pop?: number[];
	st?: number[]; // LD (nn),rr opcode bytes
	ldm?: number[]; // LD rr,(nn) opcode bytes
	h?: RegName;
	l?: RegName;
}
const BC: PairOps = {
	name: 'bc',
	ld: [0x01],
	push: [0xc5],
	pop: [0xc1],
	st: [0xed, 0x43],
	ldm: [0xed, 0x4b],
	h: 'b',
	l: 'c'
};
const DE: PairOps = {
	name: 'de',
	ld: [0x11],
	push: [0xd5],
	pop: [0xd1],
	st: [0xed, 0x53],
	ldm: [0xed, 0x5b],
	h: 'd',
	l: 'e'
};
const HL: PairOps = {
	name: 'hl',
	ld: [0x21],
	push: [0xe5],
	pop: [0xe1],
	st: [0x22],
	ldm: [0x2a],
	h: 'h',
	l: 'l'
};
const HL_ED: PairOps = { ...HL, st: [0xed, 0x63], ldm: [0xed, 0x6b] };
const SP: PairOps = { name: 'sp', ld: [0x31], st: [0xed, 0x73], ldm: [0xed, 0x7b] };
const IX: PairOps = { name: 'ix', ld: [0xdd, 0x21], push: [0xdd, 0xe5] };

function pairExpect(p: PairOps, v: number): Partial<Regs> {
	if (p.name === 'sp') return { sp: v };
	return { [p.h as RegName]: hi(v), [p.l as RegName]: lo(v) } as Partial<Regs>;
}

/** A targeted family: build one probe into `b`; return its expectation (null = collision, resample). */
type Family = (b: Builder, rng: Rng, term: boolean) => Expectation | null;

/** LD A,m (then HALT when terminated) at the landing address; expectation A = m, PC parked on the HALT. */
function marker(
	b: Builder,
	rng: Rng,
	at: number,
	term: boolean,
	extra: Partial<Regs> = {}
): Expectation | null {
	const m = rng.int(0, 255);
	if (!b.put(at, term ? [0x3e, m, HALT] : [0x3e, m])) return null;
	return { regs: { a: m, pc: m16(at + 2), ...extra } };
}

/** JR / DJNZ backwards across address 0 (from address k ∈ {0} ∪ [3, 100], reached by JP k). */
const relBack =
	(op: 0x18 | 0x10): Family =>
	(b, rng, term) => {
		const P = b.P;
		const k = rng.bool() ? 0 : rng.int(3, Math.min(P - 3, 100));
		if (k > 0 && !b.put(0, [0xc3, lo(k), hi(k)])) return null;
		const e = rng.int(-128, -(k + 3)); // k + 2 + e ≤ -1: the target is below address 0
		if (!b.put(k, [op, e & 0xff])) return null;
		return marker(b, rng, m16(k + 2 + e), term, op === 0x10 ? { b: 0xff } : {});
	};

/** JR / DJNZ forwards across address 0 from an address N near the top, reached by JP N. */
const relFwd =
	(op: 0x18 | 0x10): Family =>
	(b, rng, term) => {
		const N = rng.int(0xffff - 24, 0xfffe);
		if (!b.put(0, [0xc3, lo(N), hi(N)])) return null;
		if (!b.put(N, [op, 0])) return null; // offset patched below
		const e = rng.int(Math.max(0, 0x10000 - (N + 2)), 127); // N + 2 + e ≥ 0x10000
		b.mem[ringPos(N + 1, b.P)] = e;
		return marker(b, rng, m16(N + 2 + e), term, op === 0x10 ? { b: 0xff } : {});
	};

/** Sequential run-off: JP NZ,0x10000−k; k × XOR A up to 0xFFFF; PC wraps to 0x0000; JP NZ falls through. */
const pcSeq: Family = (b, rng, term) => {
	const k = rng.int(1, 10);
	const N = 0x10000 - k;
	if (!b.put(0, [0xc2, lo(N), hi(N)])) return null; // F = 0 → NZ → taken
	if (!b.put(N, Array(k).fill(0xaf))) return null; // XOR A sets Z, so the second visit falls through
	return marker(b, rng, 3, term);
};

/** PUSH across address 0 from SP ∈ {0, 1}, optionally popped straight back into another pair. */
const pushWrap: Family = (b, rng, term) => {
	const s = rng.int(0, 1);
	const v = rng.int(0, 0xffff);
	const X = rng.pick([BC, DE, HL, IX]);
	const Y = rng.bool() ? rng.pick([BC, DE, HL].filter((p) => p !== X)) : null;
	const prog = [0x31, lo(s), hi(s), ...X.ld, lo(v), hi(v), ...X.push!];
	const after = prog.length;
	if (Y) prog.push(...Y.pop!);
	if (term) prog.push(HALT);
	const ps = b.put(0, prog);
	if (!ps) return null;
	const hiPos = ringPos(s - 1, b.P);
	const loPos = ringPos(s - 2, b.P);
	if (ps.slice(after).some((p) => p === hiPos || p === loPos)) return null; // would overwrite code not yet run
	const regs: Partial<Regs> = Y ? { ...pairExpect(Y, v), sp: s } : { sp: m16(s - 2) };
	return {
		regs: { ...regs, pc: prog.length - 1 },
		mem: [
			[hiPos, hi(v)],
			[loPos, lo(v)]
		]
	};
};

/** Two POPs starting at SP ∈ {0xFFFE, 0xFFFF}: the stack pointer runs through 0xFFFF into 0x0000. */
const popWrap: Family = (b, rng, term) => {
	const s = rng.pick([0xfffe, 0xffff]);
	const [Y1, Y2] = rng.bool() ? [BC, DE] : rng.bool() ? [DE, HL] : [HL, BC];
	const prog = [0x31, lo(s), hi(s), ...Y1.pop!, ...Y2.pop!];
	if (term) prog.push(HALT);
	if (!b.put(0, prog)) return null;
	return {
		regs: {
			...pairExpect(Y1, b.word(s)),
			...pairExpect(Y2, b.word(m16(s + 2))),
			sp: m16(s + 4),
			pc: prog.length - 1
		}
	};
};

/** The soup's own convention: POPs from SP = spInit(P) until the stack pointer crosses 0xFFFF → 0x0000. */
const popInit: Family = (b, rng, term) => {
	const sp0 = spInit(b.P); // odd for every even P, so one POP reads 0xFFFF then 0x0000
	const jStar = (0xffff - sp0) / 2;
	const n = jStar + 1 + rng.int(0, 2);
	const Y = rng.pick([BC, DE, HL]);
	const prog = [...Array(n).fill(Y.pop![0])];
	if (term) prog.push(HALT);
	if (!b.put(0, prog)) return null;
	return {
		regs: { ...pairExpect(Y, b.word(m16(sp0 + 2 * (n - 1)))), sp: m16(sp0 + 2 * n), pc: n }
	};
};

/** CALL with SP ∈ {0, 1}: the return address is pushed across address 0. */
const callWrap: Family = (b, rng, term) => {
	const s = rng.int(0, 1);
	const N = rng.int(6, 0xffff);
	if (!b.put(0, [0x31, lo(s), hi(s), 0xcd, lo(N), hi(N)])) return null;
	const hiPos = ringPos(s - 1, b.P);
	const loPos = ringPos(s - 2, b.P);
	const m = rng.int(0, 255);
	const ps = b.put(N, term ? [0x3e, m, HALT] : [0x3e, m]);
	if (!ps || ps.some((p) => p === hiPos || p === loPos)) return null;
	return {
		regs: { a: m, sp: m16(s - 2), pc: m16(N + 2) },
		mem: [
			[hiPos, 0x00],
			[loPos, 0x06]
		]
	};
};

/** RET with SP = 0xFFFF: the return address is read from 0xFFFF (low) and 0x0000 (high). */
const retWrap: Family = (b, rng, term) => {
	const k = rng.bool() ? 0 : rng.int(3, Math.min(b.P - 4, 60)); // byte at 0x0000 is LD SP's or JP's opcode
	if (k > 0 && !b.put(0, [0xc3, lo(k), hi(k)])) return null;
	if (!b.put(k, [0x31, 0xff, 0xff, 0xc9])) return null;
	const r = rng.int(0, 255);
	if (!b.put(0xffff, [r])) return null;
	const T = (b.at(0) << 8) | r;
	return marker(b, rng, T, term, { sp: 0x0001 });
};

/** EX (SP),HL with SP = 0xFFFF: a 16-bit read and write straddling address 0. */
const exSpWrap: Family = (b, rng, term) => {
	const v = rng.int(0, 0xffff);
	const prog = [0x31, 0xff, 0xff, 0x21, lo(v), hi(v), 0xe3];
	if (term) prog.push(HALT);
	if (!b.put(0, prog)) return null;
	const r = rng.int(0, 255);
	if (!b.put(0xffff, [r])) return null;
	return {
		regs: { h: b.at(0), l: r, sp: 0xffff, pc: prog.length - 1 },
		mem: [
			[ringPos(0xffff, b.P), lo(v)],
			[0, hi(v)]
		]
	};
};

/** LDIR (dir = +1) / LDDR (dir = −1) whose source, destination or both run across address 0. */
const blockWrap =
	(dir: 1 | -1): Family =>
	(b, rng, term) => {
		const P = b.P;
		const k = rng.int(3, P - 12); // the copy block sits at k, reached by JP k (address 0 is free to be overwritten)
		if (!b.put(0, [0xc3, lo(k), hi(k)])) return null;
		const n = rng.int(2, Math.min(P - 12, 120)); // 1 + 3 + n + 1 steps ≤ 128
		const crossing = () => (dir > 0 ? 0x10000 - rng.int(1, n - 1) : rng.int(0, n - 2));
		const which = rng.pick(['src', 'dst', 'both'] as const);
		const src = which === 'dst' ? rng.int(0, 0xffff) : crossing();
		const dst = which === 'src' ? rng.int(0, 0xffff) : crossing();
		const prog = [
			0x21,
			lo(src),
			hi(src),
			0x11,
			lo(dst),
			hi(dst),
			0x01,
			lo(n),
			hi(n),
			0xed,
			dir > 0 ? 0xb0 : 0xb8
		];
		if (term) prog.push(HALT);
		const block = b.put(k, prog);
		if (!block) return null;
		// Expected memory: byte-serial copy on 16-bit addresses mapped onto the ring.
		const m = new Uint8Array(b.mem);
		for (let i = 0; i < n; i++) {
			const dp = ringPos(dst + dir * i, P);
			if (block.includes(dp)) return null; // would rewrite the copy loop itself (valid Z80, but not analytic)
			m[dp] = m[ringPos(src + dir * i, P)];
		}
		const mem: Array<[number, number]> = Array.from(m, (v, p) => [p, v]);
		const regs: Partial<Regs> = {
			h: hi(m16(src + dir * n)),
			l: lo(m16(src + dir * n)),
			d: hi(m16(dst + dir * n)),
			e: lo(m16(dst + dir * n)),
			b: 0,
			c: 0,
			pc: k + prog.length - 1
		};
		return { regs, mem };
	};

/** (IX+d)/(IY+d) whose effective address crosses address 0 upward or downward; read or write. */
const idxWrap: Family = (b, rng, term) => {
	const pre = rng.bool() ? 0xdd : 0xfd;
	const up = rng.bool();
	const x = up ? rng.int(0xff81, 0xffff) : rng.int(0, 127);
	const d = up ? rng.int(0x10000 - x, 127) : rng.int(-128, -(x + 1));
	const addr = m16(x + d);
	if (rng.bool()) {
		const m = rng.int(0, 255);
		const prog = [pre, 0x21, lo(x), hi(x), 0x3e, m, pre, 0x77, d & 0xff];
		if (term) prog.push(HALT);
		const ps = b.put(0, prog);
		if (!ps) return null;
		if (term && ps[ps.length - 1] === ringPos(addr, b.P)) return null; // would overwrite the HALT before it runs
		return { regs: { a: m, pc: prog.length - 1 }, mem: [[ringPos(addr, b.P), m]] };
	}
	const prog = [pre, 0x21, lo(x), hi(x), pre, 0x7e, d & 0xff];
	if (term) prog.push(HALT);
	if (!b.put(0, prog)) return null;
	const r = rng.int(0, 255);
	if (!b.put(addr, [r])) return null;
	return { regs: { a: r, pc: prog.length - 1 } };
};

/** 16-bit memory operand at 0xFFFF: LD (0xFFFF),X writes 0xFFFF and 0x0000; LD Y,(0xFFFF) reads them back. */
const wordWrap: Family = (b, rng, term) => {
	const v = rng.int(0, 0xffff);
	const X = rng.pick([BC, DE, HL, HL_ED, SP]);
	const Y = rng.pick([BC, DE, HL, HL_ED, SP].filter((p) => p.name !== X.name));
	const prog = [...X.ld, lo(v), hi(v), ...X.st!, 0xff, 0xff, ...Y.ldm!, 0xff, 0xff];
	if (term) prog.push(HALT);
	const ps = b.put(0, prog);
	if (!ps || ps.includes(ringPos(0xffff, b.P))) return null; // the store must not land on code
	return {
		regs: { ...pairExpect(X, v), ...pairExpect(Y, v), pc: prog.length - 1 },
		mem: [
			[ringPos(0xffff, b.P), lo(v)],
			[0, hi(v)]
		]
	};
};

export const FAMILIES: Record<string, Family> = {
	'jr-back': relBack(0x18),
	'jr-fwd': relFwd(0x18),
	'djnz-back': relBack(0x10),
	'djnz-fwd': relFwd(0x10),
	'pc-seq': pcSeq,
	'push-wrap': pushWrap,
	'pop-wrap': popWrap,
	'pop-init-sp': popInit,
	'call-wrap': callWrap,
	'ret-wrap': retWrap,
	'ex-sp-wrap': exSpWrap,
	'ldir-wrap': blockWrap(1),
	'lddr-wrap': blockWrap(-1),
	'idx-wrap': idxWrap,
	'word-wrap': wordWrap
};

/** `perVariant` probes of each family in each of the three variants (halt-fill, random-fill, open). */
export function targetedCases(P: number, perVariant: number, seed: number): RingCase[] {
	const rng = makeRng((seed ^ 0x7a3c9e11 ^ Math.imul(P, 0x9e3779b1)) >>> 0);
	const out: RingCase[] = [];
	for (const [family, build] of Object.entries(FAMILIES)) {
		for (const variant of ['halt-fill', 'random-fill', 'open'] as const) {
			for (let i = 0; i < perVariant; i++) {
				for (let tries = 0; ; tries++) {
					if (tries > 1000) throw new Error(`cannot place ${family} on a ${P}-byte ring`);
					const b = new Builder(P, rng, variant === 'halt-fill' ? 'halt' : 'random');
					const e = build(b, rng, variant !== 'open');
					if (!e) continue;
					out.push({ family, variant, input: b.mem, expect: variant === 'open' ? undefined : e });
					break;
				}
			}
		}
	}
	return out;
}

// ── Comparison ────────────────────────────────────────────────────────────────

export function oracleRegs(o: OracleResult): Regs {
	return { a: o.a, f: o.f, b: o.b, c: o.c, d: o.d, e: o.e, h: o.h, l: o.l, sp: o.sp, pc: o.pc };
}

/** GPU registers as read back, NOT masked: a PC or SP left above 0xFFFF would itself be a divergence. */
export function gpuRegs(regs: Uint32Array, idx: number): Regs {
	const o = idx * REGS_PER_CASE;
	const r = {} as Regs;
	REG_NAMES.forEach((n, i) => (r[n] = regs[o + i]));
	return r;
}

export interface Diff {
	memDiffBytes: number[]; // ring positions that differ
	regDiffs: { name: RegName; gpu: number; cpu: number }[];
}

export function diffState(
	gMem: Uint8Array,
	g: Regs,
	oMem: Uint8Array,
	o: Regs,
	strictFlags = true
): Diff {
	const memDiffBytes: number[] = [];
	for (let i = 0; i < oMem.length; i++) if (gMem[i] !== oMem[i]) memDiffBytes.push(i);
	const regDiffs: Diff['regDiffs'] = [];
	for (const n of REG_NAMES) {
		const differs = n === 'f' && !strictFlags ? ((g.f ^ o.f) & ~F3F5 & 0xff) !== 0 : g[n] !== o[n];
		if (differs) regDiffs.push({ name: n, gpu: g[n], cpu: o[n] });
	}
	return { memDiffBytes, regDiffs };
}

/** Same categories as z80difftest.ts: benign = only the undocumented F3/F5 bits of F differ. */
export function isBenign(d: Diff): boolean {
	return (
		d.memDiffBytes.length === 0 &&
		d.regDiffs.length === 1 &&
		d.regDiffs[0].name === 'f' &&
		((d.regDiffs[0].gpu ^ d.regDiffs[0].cpu) & ~F3F5) === 0
	);
}

/** Which entries of an analytic expectation a final state violates (empty = meets it). */
export function checkExpectation(e: Expectation, mem: Uint8Array, regs: Regs): string[] {
	const bad: string[] = [];
	for (const [k, v] of Object.entries(e.regs ?? {})) {
		const got = regs[k as RegName];
		if (got !== v) bad.push(`${k}=${hex(got, 4)} (expected ${hex(v as number, 4)})`);
	}
	for (const [p, v] of e.mem ?? [])
		if (mem[p] !== v) bad.push(`mem[${p}]=${hex(mem[p])} (expected ${hex(v)})`);
	return bad;
}

export function hex(n: number, w = 2): string {
	return (n >>> 0).toString(16).toUpperCase().padStart(w, '0');
}

// ── First-divergence classification ────────────────────────────────────────────

export type Cause =
	| 'oracle:unhandled-dd-fd' // z80-emulator cannot decode a DD/FD-prefixed instruction and NOPs it (a real Z80 runs it)
	| 'oracle:ld-a-r' // z80-emulator's LD A,R returns R & 0xF7 (bit 3 cleared, counter bit 7 leaked)
	| 'oracle:ld-r-a' // z80-emulator's LD R,A does not set R7 (Fuse: R = R7 = A); visible only through a later LD A,R
	| 'oracle:in-f-c' // z80-emulator's IN F,(C) (ED 70) loses the carry flag (its IN A,(C) keeps it, as a Z80 does)
	| 'oracle:adc-sbc-hl-z' // z80-emulator's ADC/SBC HL,rr take Z from the unmasked 17-bit sum (Z wrong when it is ±0x10000)
	| 'zilion:bit-mem-f3f5' // BIT n,(HL)/(IX+d): F3/F5 come from MEMPTR on a real Z80; documented as not modelled by zilion
	| 'zilion:wasted-prefix-r' // zilion adds 2 to R for a wasted DD/FD prefix (a real Z80 adds 1); visible only via LD A,R
	| 'unexplained';

export interface DivergenceEvent {
	step: number; // 1-based instruction
	pc: number; // PC entering that instruction
	bytes: number[]; // 4 bytes at that PC (ring-mapped, from memory at that point)
	mnemonic: string;
	cause: Cause;
	verified: boolean; // GPU state right after this instruction equals the (corrected) reference state
	diff?: Diff; // GPU vs corrected reference after this instruction, when they disagree
}

export type TraceOutcome = 'match-modulo-oracle-quirks' | 'zilion-deviation' | 'unexplained';

export interface TraceResult {
	outcome: TraceOutcome;
	events: DivergenceEvent[]; // every reference quirk and known zilion deviation met, in order
	stop?: DivergenceEvent; // the instruction at which the comparison stopped (unexplained divergence)
}

/** z80-emulator with exactly the reference's conventions (z80oracle.ts), exposed for single-stepping. */
function mirror(input: Uint8Array, P: number): { z: Z80; mem: Uint8Array } {
	const mem = new Uint8Array(input);
	const hal: Hal = {
		tStateCount: 0,
		readMemory: (a: number) => mem[a % P],
		writeMemory: (a: number, v: number) => {
			mem[a % P] = v & 0xff;
		},
		contendMemory: () => {},
		readPort: () => 0,
		writePort: () => {},
		contendPort: () => {}
	};
	const z = new Z80(hal);
	z.reset(); // every register 0
	z.regs.sp = spInit(P);
	return { z, mem };
}

/** Final state of the uncorrected mirror (used only to check that it reproduces runOracle exactly). */
export function mirrorFinal(
	input: Uint8Array,
	P: number,
	steps: number
): { mem: Uint8Array; regs: Regs } {
	const { z, mem } = mirror(input, P);
	for (let s = 0; s < steps && !z.regs.halted; s++) z.step();
	return { mem, regs: mirrorRegs(z) };
}

function mirrorRegs(z: Z80): Regs {
	const r = z.regs;
	return {
		a: r.a,
		f: r.f,
		b: r.b,
		c: r.c,
		d: r.d,
		e: r.e,
		h: r.h,
		l: r.l,
		sp: r.sp & 0xffff,
		pc: r.pc & 0xffff
	};
}

/** Does z80-emulator fail to decode this instruction (it logs "Unhandled opcode ..." and executes nothing)? */
const undecodable = new Map<string, boolean>();
export function oracleCannotDecode(bytes: number[]): boolean {
	const key = bytes.join(',');
	const hit = undecodable.get(key);
	if (hit !== undefined) return hit;
	const { z } = mirror(Uint8Array.from([...bytes, 0, 0, 0, 0]), 8);
	const log = console.log; // z80oracle.ts has already wrapped console.log; wrap it once more, briefly
	let unhandled = false;
	console.log = (...a: unknown[]) => {
		if (typeof a[0] === 'string' && a[0].startsWith('Unhandled opcode')) unhandled = true;
		else log(...a);
	};
	try {
		z.step();
	} finally {
		console.log = log;
	}
	undecodable.set(key, unhandled);
	return unhandled;
}

/**
 * Re-runs a program instruction by instruction on z80-emulator (same conventions as z80oracle.ts) next to the
 * GPU state after every step budget 1..steps (trace run), and compares the full pair memory and the main
 * registers (all flag bits) after EVERY instruction, to the end of the run.
 *
 * Five reference defects are corrected where they occur — at the wrapper level, around z80-emulator's
 * unmodified step() — and each correction is checked against the GPU at that very instruction (`verified`):
 *   • an instruction z80-emulator cannot decode behind DD/FD (it logs "Unhandled opcode" and executes
 *     nothing): the prefix is consumed as one M1 cycle (R + 1) and the rest executes unprefixed, which is what
 *     a real Z80 does; a prefix followed by DD/FD/ED is a wasted prefix, one instruction of its own in
 *     zilion's count;
 *   • LD A,R (z80-emulator returns R & 0xF7): A := (R7 & 0x80) | (R & 0x7F), F recomputed with z80-emulator's
 *     own LD A,R flag formula on that value;
 *   • LD R,A (z80-emulator sets its 8-bit counter but not R7): R7 := A & 0x80, as in Fuse (R = R7 = A);
 *   • IN F,(C) (ED 70; z80-emulator overwrites F with the port byte before masking, so the old carry is lost):
 *     F := (old F & C) | sz53p(port byte), the formula of its own IN A,(C) (ED 78);
 *   • ADC HL,rr / SBC HL,rr (z80-emulator sets Z from the unmasked sum, so a result of exactly ±0x10000 leaves
 *     Z clear although HL = 0): Z := (HL == 0), as in Fuse, from which z80-emulator descends.
 *
 * Two zilion deviations from a real Z80 are recognised, counted, and stepped over (the reference adopts the
 * GPU's value so that the rest of the run is still compared):
 *   • BIT n,(HL)/(IX+d) with only F3/F5 differing (MEMPTR-derived on a real Z80; documented as not modelled);
 *   • LD A,R reading R + w, where w is the number of wasted DD/FD prefixes since R was last loaded (zilion
 *     adds 2 to R for a wasted prefix instead of 1).
 * The comparison stops at the first disagreement that is none of these, and reports it as unexplained.
 */
export function traceAgainstReference(
	input: Uint8Array,
	P: number,
	gpuAt: (k: number) => { mem: Uint8Array; regs: Regs },
	steps: number
): TraceResult {
	const { z, mem } = mirror(input, P);
	const events: DivergenceEvent[] = [];
	let wasted = 0; // wasted prefixes since R was last loaded (LD R,A)
	for (let k = 1; k <= steps; k++) {
		const pc = z.regs.pc & 0xffff;
		const bytes = [0, 1, 2, 3].map((i) => mem[ringPos(pc + i, P)]);
		const isIdx = bytes[0] === 0xdd || bytes[0] === 0xfd;
		const f0 = z.regs.f;
		let cause: Cause | null = null;
		if (!z.regs.halted) {
			if (isIdx && oracleCannotDecode(bytes)) {
				cause = 'oracle:unhandled-dd-fd';
				z.regs.pc = m16(z.regs.pc + 1);
				z.regs.r = (z.regs.r + 1) & 0xff;
				if (bytes[1] !== 0xdd && bytes[1] !== 0xfd && bytes[1] !== 0xed) z.step();
				else wasted++;
			} else if (bytes[0] === 0xed && bytes[1] === 0x5f) {
				cause = 'oracle:ld-a-r';
				z.step();
				const realR = (z.regs.r7 & 0x80) | (z.regs.r & 0x7f);
				z.regs.a = realR;
				z.regs.f = (f0 & 0x01) | z.sz53Table[realR] | (z.regs.iff2 ? 0x04 : 0);
			} else if (bytes[0] === 0xed && bytes[1] === 0x70) {
				cause = 'oracle:in-f-c';
				z.step();
				z.regs.f = (f0 & 0x01) | z.sz53pTable[0]; // port reads 0 (reference HAL and shader alike)
			} else if (bytes[0] === 0xed && (bytes[1] & 0xc7) === 0x42) {
				cause = 'oracle:adc-sbc-hl-z'; // ED 42/4A/52/5A/62/6A/72/7A
				z.step();
				z.regs.f = (z.regs.f & ~0x40) | (z.regs.hl === 0 ? 0x40 : 0); // Z from the 16-bit result
			} else if (bytes[0] === 0xed && bytes[1] === 0x4f) {
				cause = 'oracle:ld-r-a';
				z.step();
				z.regs.r7 = z.regs.a & 0x80; // R = R7 = A
				wasted = 0;
			} else z.step();
		}
		const g = gpuAt(k);
		let d = diffState(g.mem, g.regs, mem, mirrorRegs(z), true);
		const differs = () => d.memDiffBytes.length > 0 || d.regDiffs.length > 0;
		const ev = (c: Cause, verified: boolean): DivergenceEvent => {
			const e: DivergenceEvent = {
				step: k,
				pc,
				bytes,
				mnemonic: disassemble(new Uint8Array(bytes))[0]?.mnemonic ?? '?',
				cause: c,
				verified
			};
			if (differs()) e.diff = d;
			return e;
		};
		if (!differs()) {
			if (cause) events.push(ev(cause, true));
			continue;
		}
		// zilion deviation 1: LD A,R after wasted prefixes reads R + w.
		if (cause === 'oracle:ld-a-r' && wasted > 0) {
			const zr = (z.regs.r7 & 0x80) | ((z.regs.r + wasted) & 0x7f);
			if (g.regs.a === zr) {
				const e = ev('zilion:wasted-prefix-r', false);
				z.regs.a = zr;
				z.regs.f = (f0 & 0x01) | z.sz53Table[zr] | (z.regs.iff2 ? 0x04 : 0);
				d = diffState(g.mem, g.regs, mem, mirrorRegs(z), true);
				e.verified = !differs();
				events.push(e);
				if (e.verified) continue;
				return { outcome: 'unexplained', events, stop: e };
			}
		}
		// zilion deviation 2: BIT n,(HL)/(IX+d) with only F3/F5 different.
		const onlyF35 =
			d.memDiffBytes.length === 0 &&
			d.regDiffs.every((r) => r.name === 'f' && ((r.gpu ^ r.cpu) & ~F3F5) === 0);
		const bitMem =
			(bytes[0] === 0xcb && (bytes[1] & 0xc7) === 0x46) ||
			(isIdx && bytes[1] === 0xcb && (bytes[3] & 0xc0) === 0x40);
		if (!cause && bitMem && onlyF35) {
			events.push(ev('zilion:bit-mem-f3f5', true));
			z.regs.f = g.regs.f;
			continue;
		}
		const stop = cause ? ev(cause, false) : ev('unexplained', false);
		if (cause) events.push(stop);
		return { outcome: 'unexplained', events, stop };
	}
	const zil = events.some((e) => e.cause.startsWith('zilion:'));
	return { outcome: zil ? 'zilion-deviation' : 'match-modulo-oracle-quirks', events };
}

// ── Seam coverage of the random programs (measurement only; not part of the comparison) ──────────

export interface SeamCoverage {
	any: boolean;
	pc: boolean; // PC moved across 0xFFFF ↔ 0x0000 (run-off, JR/DJNZ, or a jump landing near the seam)
	sp: boolean; // stack pointer crossed (PUSH/POP/CALL/RET)
	ptr: boolean; // HL/DE/BC/IX/IY crossed (block ops, INC/DEC rr)
}

/**
 * Re-runs a program on z80-emulator with the reference's conventions and reports whether any 16-bit
 * register stepped across the 0xFFFF ↔ 0x0000 seam (before ≥ 0xFF00 and after ≤ 0x00FF, or the reverse).
 */
export function seamCoverage(input: Uint8Array, P: number, steps: number): SeamCoverage {
	const { z } = mirror(input, P);
	const r = z.regs;
	const cross = (x: number, y: number) =>
		(x >= 0xff00 && y <= 0x00ff) || (x <= 0x00ff && y >= 0xff00);
	const cov: SeamCoverage = { any: false, pc: false, sp: false, ptr: false };
	for (let s = 0; s < steps && !r.halted; s++) {
		const before = [r.pc, r.sp, r.hl, r.de, r.bc, r.ix, r.iy];
		z.step();
		const after = [r.pc, r.sp, r.hl, r.de, r.bc, r.ix, r.iy];
		if (cross(before[0], after[0])) cov.pc = true;
		if (cross(before[1], after[1])) cov.sp = true;
		for (let i = 2; i < 7; i++) if (cross(before[i], after[i])) cov.ptr = true;
	}
	cov.any = cov.pc || cov.sp || cov.ptr;
	return cov;
}
