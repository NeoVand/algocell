// Real-Z80 oracle (DEV ONLY) for the differential test harness.
//
// Wraps `z80-emulator` (lkesteloot/trs80, a Fuse-lineage core with full IX/IY,
// shadow registers, R refresh, and the WZ/memptr register — accurate down to
// the undocumented F3/F5 flags and passes zexall-class conformance). This is
// GROUND TRUTH: our WebGPU Z80 must match this, not the other way around.
//
// Configured to mirror the simulation's per-interaction environment: a
// `pairLength`-byte buffer (32 for square cells, 38 for hex) that the 16-bit
// address space wraps onto, all registers zeroed, SP = 0xFFFF, PC = 0
// (matching a real Z80 reset), and optionally the same instruction
// suppression the GPU core applies.

import { Z80, type Hal } from 'z80-emulator';
import type { SuppressSets } from '$lib/z80-opcodes';

// z80-emulator has one known deviation from real hardware: it NOPs DD/FD-prefixed
// opcodes that have no IX/IY form (a real Z80 — and superzazu, the paper's own
// emulator — executes them as the base opcode). It announces every such NOP with
// console.log("Unhandled opcode in DD/FD: XX"). We intercept that once so we can
// (a) silence the spam and (b) flag any oracle run that relied on the quirk, so
// the harness can separate genuine Zilion bugs from this reference-emulator wart.
let ddQuirkFired = false;
const _origConsoleLog = console.log.bind(console);
console.log = (...args: unknown[]) => {
	const m = args[0];
	if (typeof m === 'string' && m.startsWith('Unhandled opcode in ')) {
		ddQuirkFired = true;
		return;
	}
	_origConsoleLog(...args);
};

export interface OracleResult {
	mem: Uint8Array;
	/** True if this run hit z80-emulator's DD/FD-NOP quirk (see above). */
	ddQuirk: boolean;
	a: number;
	f: number;
	b: number;
	c: number;
	d: number;
	e: number;
	h: number;
	l: number;
	sp: number;
	pc: number;
}

// Suppression on the reference: the GPU core skips a suppressed instruction
// as a NOP *inside* its fetch (after prefix resolution). We mirror that here by
// peeking at the bytes at PC before each step and, if the decoded instruction
// is suppressed, advancing PC (and R) past the opcode without executing it —
// 1 byte for a base opcode, 2 for CB/ED-page and IX/IY-prefixed opcodes, 4 for
// DDCB/FDCB. Operand bytes of a base opcode are not skipped (they run next).
function skipIfSuppressed(
	mem: Uint8Array,
	pairLength: number,
	regs: { pc: number; r: number },
	s: SuppressSets
): boolean {
	const rd = (a: number) => mem[(a & 0xffff) % pairLength];
	const pc = regs.pc & 0xffff;
	const skip = (len: number, m1: number) => {
		regs.pc = (pc + len) & 0xffff;
		regs.r = (regs.r & 0x80) | ((regs.r + m1) & 0x7f);
		return true;
	};
	const b0 = rd(pc);
	if (b0 === 0xdd || b0 === 0xfd) {
		const b1 = rd(pc + 1);
		if (b1 === 0xdd || b1 === 0xfd || b1 === 0xed) return false; // wasted prefix
		if (b1 === 0xcb) return s.cb.has(rd(pc + 3)) ? skip(4, 2) : false;
		return s.base.has(b1) ? skip(2, 2) : false;
	}
	if (b0 === 0xcb) return s.cb.has(rd(pc + 1)) ? skip(2, 2) : false;
	if (b0 === 0xed) return s.ed.has(rd(pc + 1)) ? skip(2, 2) : false;
	return s.base.has(b0) ? skip(1, 1) : false;
}

export function runOracle(
	input: Uint8Array,
	steps: number,
	pairLength = 32,
	suppress?: SuppressSets
): OracleResult {
	const mem = new Uint8Array(input); // fresh copy — the program mutates itself
	const hal: Hal = {
		tStateCount: 0,
		readMemory: (address: number) => mem[address % pairLength],
		writeMemory: (address: number, value: number) => {
			mem[address % pairLength] = value & 0xff;
		},
		contendMemory: () => {},
		readPort: () => 0, // no I/O device (matches zff inPort→0)
		writePort: () => {},
		contendPort: () => {}
	};

	const z80 = new Z80(hal);
	z80.reset();
	const r = z80.regs;
	// Match the shader's per-pair init exactly: everything 0 except SP=0xFFFF.
	r.af = 0;
	r.bc = 0;
	r.de = 0;
	r.hl = 0;
	r.afPrime = 0;
	r.bcPrime = 0;
	r.dePrime = 0;
	r.hlPrime = 0;
	r.ix = 0;
	r.iy = 0;
	r.sp = 0xffff;
	r.pc = 0;
	r.memptr = 0;
	r.i = 0;
	r.r = 0;
	r.r7 = 0;
	r.iff1 = 0;
	r.iff2 = 0;
	r.im = 0;
	r.halted = 0;

	ddQuirkFired = false;
	for (let s = 0; s < steps; s++) {
		if (r.halted) break;
		if (suppress && skipIfSuppressed(mem, pairLength, r, suppress)) continue;
		z80.step();
	}

	return {
		mem,
		ddQuirk: ddQuirkFired,
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
