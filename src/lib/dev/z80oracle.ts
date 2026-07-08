// Real-Z80 oracle (DEV ONLY) for the differential test harness.
//
// Wraps `z80-emulator` (lkesteloot/trs80, a Fuse-lineage core with full IX/IY,
// shadow registers, R refresh, and the WZ/memptr register — accurate down to
// the undocumented F3/F5 flags and passes zexall-class conformance). This is
// GROUND TRUTH: our WebGPU Z80 must match this, not the other way around.
//
// Configured to mirror the simulation's per-interaction environment: a
// `pairLength`-byte buffer that the 16-bit address space wraps onto, all
// registers zeroed, SP = 0xFFFF, PC = 0 (matching a real Z80 reset).

import { Z80, type Hal } from 'z80-emulator';

export interface OracleResult {
	mem: Uint8Array;
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

export function runOracle(input: Uint8Array, steps: number, pairLength = 32): OracleResult {
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

	for (let s = 0; s < steps; s++) {
		if (r.halted) break;
		z80.step();
	}

	return {
		mem,
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
