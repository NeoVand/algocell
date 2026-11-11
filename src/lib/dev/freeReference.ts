// Reference for the free-running world (DEV ONLY).
//
// The world's rule is written here a second time, in plain TypeScript, with a third-party Z80
// (`z80-emulator`, the same ground truth as z80oracle.ts) executing the instructions. The GPU world
// in gpu/free.ts must match this, tick for tick: every head reads the sheet as it stood when the
// tick began, writes go to a log, and each byte goes to the lowest-numbered head that wrote it.

import { Z80, type Hal } from 'z80-emulator';
import { HEAD_DIRECTION, HEAD_ORIGIN_X, HEAD_ORIGIN_Y, HEAD_WORDS } from '../gpu/free';

// z80-emulator announces DD/FD-prefixed opcodes it treats as NOPs; keep the console quiet.
const originalLog = console.log.bind(console);
console.log = (...args: unknown[]) => {
	if (typeof args[0] === 'string' && args[0].startsWith('Unhandled opcode')) return;
	originalLog(...args);
};

export interface ReferenceWorld {
	sheet: Uint8Array;
	tick(): void;
	registers(head: number): Record<string, number>;
	/** Bytes two or more heads tried to write in the same tick, summed over ticks. */
	contested: number;
	/** Head-ticks spent on a HALT byte. */
	halts: number;
	/**
	 * Ticks on which some head fetched a DD or FD prefix. Zilion and z80-emulator split prefix chains
	 * into steps differently (and z80-emulator has a known DD/FD deviation), so a comparison is only
	 * meaningful up to the first such tick.
	 */
	firstIndexPrefixTick: number | null;
	ticks: number;
}

export function createReference(
	sheetW: number,
	sheetH: number,
	sheet: Uint8Array,
	heads: Uint32Array,
	haltRule: 'wait' | 'pass'
): ReferenceWorld {
	const count = heads.length / HEAD_WORDS;
	const world: ReferenceWorld = {
		sheet: new Uint8Array(sheet),
		contested: 0,
		halts: 0,
		firstIndexPrefixTick: null,
		ticks: 0,
		tick,
		registers
	};
	let log = new Map<number, number>();
	const machines: { z80: Z80; place: (address: number) => number }[] = [];
	for (let h = 0; h < count; h++) {
		const ox = heads[h * HEAD_WORDS + HEAD_ORIGIN_X];
		const oy = heads[h * HEAD_WORDS + HEAD_ORIGIN_Y];
		const dir = heads[h * HEAD_WORDS + HEAD_DIRECTION];
		const place = (address: number) => {
			const a = address & 0xffff;
			let x = ox;
			let y = oy;
			if (dir === 0) x = (ox + a) % sheetW;
			else if (dir === 1) y = (oy + a) % sheetH;
			else if (dir === 2) x = (ox + sheetW - (a % sheetW)) % sheetW;
			else y = (oy + sheetH - (a % sheetH)) % sheetH;
			return y * sheetW + x;
		};
		const hal: Hal = {
			tStateCount: 0,
			readMemory: (address: number) => {
				const p = place(address);
				return log.get(p) ?? world.sheet[p];
			},
			writeMemory: (address: number, value: number) => {
				log.set(place(address), value & 0xff);
			},
			contendMemory: () => {},
			readPort: () => 0,
			writePort: () => {},
			contendPort: () => {}
		};
		const z80 = new Z80(hal);
		z80.reset();
		const r = z80.regs;
		r.af = r.bc = r.de = r.hl = 0;
		r.afPrime = r.bcPrime = r.dePrime = r.hlPrime = 0;
		r.ix = r.iy = r.sp = r.pc = r.memptr = 0;
		r.i = r.r = r.r7 = r.iff1 = r.iff2 = r.im = r.halted = 0;
		machines.push({ z80, place });
	}

	function tick(): void {
		const writes = new Map<number, number>(); // position -> value of the lowest-numbered writer
		const writers = new Map<number, number>();
		for (let h = 0; h < count; h++) {
			const { z80, place } = machines[h];
			const r = z80.regs;
			const first = world.sheet[place(r.pc)];
			if ((first === 0xdd || first === 0xfd) && world.firstIndexPrefixTick === null)
				world.firstIndexPrefixTick = world.ticks;
			log = new Map();
			z80.step();
			if (r.halted) {
				// HALT leaves the program counter on the HALT byte.
				r.halted = 0;
				world.halts++;
				if (haltRule === 'pass') r.pc = (r.pc + 1) & 0xffff;
			}
			for (const [p, value] of log) {
				writers.set(p, (writers.get(p) ?? 0) + 1);
				if (!writes.has(p)) writes.set(p, value); // heads run in ascending order
			}
		}
		for (const n of writers.values()) if (n > 1) world.contested++;
		for (const [p, value] of writes) world.sheet[p] = value;
		world.ticks++;
	}

	function registers(head: number): Record<string, number> {
		const r = machines[head].z80.regs;
		return {
			a: r.a,
			f: r.f,
			b: r.b,
			c: r.c,
			d: r.d,
			e: r.e,
			h: r.h,
			l: r.l,
			ix: r.ix & 0xffff,
			iy: r.iy & 0xffff,
			sp: r.sp & 0xffff,
			pc: r.pc & 0xffff
		};
	}

	return world;
}
