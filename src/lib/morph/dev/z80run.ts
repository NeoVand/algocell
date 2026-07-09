// Shared dev helper: run a tape on a real Z80 (z80-emulator) until HALT.
// Used by the M0 (v1) and v2 differential tests. Wraps the address space onto
// `memBytes` and matches Zilion's per-lane init (SP=0xFFFF, everything else 0).

import { Z80, type Hal } from 'z80-emulator';

export function runOnRealZ80(
	tape: Uint8Array,
	memBytes: number,
	cap: number
): { mem: Uint8Array; steps: number; halted: boolean } {
	const mem = new Uint8Array(tape);
	const hal: Hal = {
		tStateCount: 0,
		readMemory: (a: number) => mem[a % memBytes],
		writeMemory: (a: number, v: number) => {
			mem[a % memBytes] = v & 0xff;
		},
		contendMemory: () => {},
		readPort: () => 0,
		writePort: () => {},
		contendPort: () => {}
	};
	const z80 = new Z80(hal);
	z80.reset();
	const r = z80.regs;
	r.af = 0; r.bc = 0; r.de = 0; r.hl = 0;
	r.ix = 0; r.iy = 0; r.sp = 0xffff; r.pc = 0;
	r.i = 0; r.r = 0; r.r7 = 0; r.iff1 = 0; r.iff2 = 0; r.im = 0; r.halted = 0;

	let steps = 0;
	for (; steps < cap; steps++) {
		if (r.halted) break;
		z80.step();
	}
	return { mem, steps, halted: !!r.halted };
}
