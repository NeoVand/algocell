// Reference for the free-running world (DEV ONLY).
//
// The world's rule is written here a second time, in plain TypeScript, with a third-party Z80
// (`z80-emulator`, the same ground truth as z80oracle.ts) executing the instructions. The GPU world
// in gpu/free.ts must match this, tick for tick: every head reads the sheet as it stood when the
// tick began, writes go to a log, each byte goes to the lowest-numbered head that wrote it, and a
// head that executes HALT (or draws the turnover chance) lands somewhere else with a blank mind.

import { Z80, type Hal } from 'z80-emulator';
import {
	FACING_MASKS,
	FACING_RULES,
	FACING_UNDECIDED,
	FREE_RNG,
	HEAD_FACING,
	HEAD_ORIGIN_X,
	HEAD_ORIGIN_Y,
	HEAD_RNG,
	HEAD_WORDS,
	addressOffset,
	chanceThreshold,
	faceOffset,
	facingFromNeighbours
} from '../gpu/free';

// z80-emulator announces DD/FD-prefixed opcodes it treats as NOPs; keep the console quiet.
const originalLog = console.log.bind(console);
console.log = (...args: unknown[]) => {
	if (typeof args[0] === 'string' && args[0].startsWith('Unhandled opcode')) return;
	originalLog(...args);
};

/** Zilion's zil_mix32. */
export function mix32(x: number): number {
	let h = Math.imul(x, 0x9e3779b9);
	h ^= h >>> 16;
	h = Math.imul(h, 0x85ebca6b);
	h ^= h >>> 13;
	h = Math.imul(h, 0xc2b2ae35);
	h ^= h >>> 16;
	return h >>> 0;
}

export interface ReferenceRules {
	haltRule: 'restart' | 'wait' | 'pass';
	/** Chance per instruction of lifting off. */
	turnover: number;
	/** Chance per instruction of one random byte. */
	mutation: number;
	/** Side of the square a head can address, in bytes (see REACHES); 256 if omitted. */
	reach?: number;
	/** Which ways a landing head may face (see FACING_MASKS). */
	facing: keyof typeof FACING_MASKS;
}

export interface ReferenceWorld {
	sheet: Uint8Array;
	/** The facing of the head that wrote each byte last. */
	grain: Uint8Array;
	tick(): void;
	registers(head: number): Record<string, number>;
	origin(head: number): { x: number; y: number; facing: number };
	/** Bytes two or more heads tried to write in the same tick, summed over ticks. */
	contested: number;
	/** Head-ticks on which a HALT was executed. */
	halts: number;
	/** Times a head lifted off and landed again. */
	restarts: number;
	/** Random bytes written by mutation (bids, before any contest). */
	mutations: number;
	/**
	 * Ticks on which some head fetched a DD or FD prefix. Zilion and z80-emulator split prefix chains
	 * into steps differently (and z80-emulator has a known DD/FD deviation), so a comparison is only
	 * meaningful up to the first such tick.
	 */
	firstIndexPrefixTick: number | null;
	/**
	 * Ticks on which some head fetched a repeating block input or output instruction (INIR, OTIR,
	 * INDR, OTDR: ED B2, B3, BA, BB). Zilion's 0.3 core adjusts H and P/V on a repeating step, as its
	 * test vectors require; z80-emulator does not. The same field records the first LD A,R
	 * (ED 5F), where z80-emulator returns the refresh register with bit 3 cleared. A comparison is
	 * only meaningful up to the first such tick as well.
	 */
	firstRepeatIoTick: number | null;
	ticks: number;
}

export function createReference(
	sheetW: number,
	sheetH: number,
	sheet: Uint8Array,
	heads: Uint32Array,
	rules: ReferenceRules,
	grain?: ArrayLike<number>
): ReferenceWorld {
	const count = heads.length / HEAD_WORDS;
	const turnover = chanceThreshold(rules.turnover);
	const mutation = chanceThreshold(rules.mutation);
	const world: ReferenceWorld = {
		sheet: new Uint8Array(sheet),
		grain: grain ? Uint8Array.from(grain) : new Uint8Array(sheet.length),
		contested: 0,
		halts: 0,
		restarts: 0,
		mutations: 0,
		firstIndexPrefixTick: null,
		firstRepeatIoTick: null,
		ticks: 0,
		tick,
		registers,
		origin: (head) => ({
			x: machines[head].ox,
			y: machines[head].oy,
			facing: machines[head].facing
		})
	};
	// Position -> value; a mutation carries bit 8 (it leaves the grain of the byte alone).
	let log = new Map<number, number>();
	const MUTATION = 0x100;
	// The facing of a head landing on (x, y), by the rule; `random` is a random word of its own.
	const landingFace = (x: number, y: number, random: number): number => {
		const rule = FACING_RULES[rules.facing];
		if (rule === 1) return world.grain[y * sheetW + x] & 7;
		if (rule !== 2) return random & FACING_MASKS[rules.facing];
		return facingFromNeighbours(
			world.sheet[y * sheetW + ((x + 1) % sheetW)],
			world.sheet[y * sheetW + ((x + sheetW - 1) % sheetW)],
			world.sheet[((y + 1) % sheetH) * sheetW + x],
			world.sheet[((y + sheetH - 1) % sheetH) * sheetW + x],
			random,
			mix32
		);
	};
	interface Machine {
		z80: Z80;
		ox: number;
		oy: number;
		facing: number;
		rng: number;
		place(address: number): number;
	}
	const machines: Machine[] = [];
	const bits = Math.log2(rules.reach ?? 256);
	const wrap = (v: number, n: number) => ((v % n) + n) % n;
	const blank = (z80: Z80) => {
		z80.reset();
		const r = z80.regs;
		r.af = r.bc = r.de = r.hl = 0;
		r.afPrime = r.bcPrime = r.dePrime = r.hlPrime = 0;
		r.ix = r.iy = r.sp = r.pc = r.memptr = 0;
		r.i = r.r = r.r7 = r.iff1 = r.iff2 = r.im = r.halted = 0;
	};
	for (let h = 0; h < count; h++) {
		const machine: Machine = {
			z80: null as unknown as Z80,
			ox: heads[h * HEAD_WORDS + HEAD_ORIGIN_X],
			oy: heads[h * HEAD_WORDS + HEAD_ORIGIN_Y],
			facing: heads[h * HEAD_WORDS + HEAD_FACING],
			rng: heads[h * HEAD_WORDS + HEAD_RNG],
			// An address is an offset from where the head landed, in the head's own frame: the low byte
			// counts bytes along the way it faces, the high byte counts lines across, each a signed byte.
			place(address: number) {
				const [dx, dy] = faceOffset(machine.facing, ...addressOffset(address, bits));
				return wrap(machine.oy + dy, sheetH) * sheetW + wrap(machine.ox + dx, sheetW);
			}
		};
		const place = machine.place;
		const hal: Hal = {
			tStateCount: 0,
			readMemory: (address: number) => {
				const p = place(address);
				const own = log.get(p);
				return own === undefined ? world.sheet[p] : own & 0xff;
			},
			writeMemory: (address: number, value: number) => {
				log.set(place(address), value & 0xff);
			},
			contendMemory: () => {},
			readPort: () => 0,
			writePort: () => {},
			contendPort: () => {}
		};
		machine.z80 = new Z80(hal);
		blank(machine.z80);
		machines.push(machine);
	}

	function tick(): void {
		// Position -> value and facing of the lowest-numbered writer.
		const writes = new Map<number, { value: number; facing: number }>();
		const writers = new Map<number, number>();
		for (let h = 0; h < count; h++) {
			const machine = machines[h];
			const r = machine.z80.regs;
			const pc = r.pc;
			// A head that has only been placed takes its facing from the rule before its first instruction.
			if (machine.facing === FACING_UNDECIDED)
				machine.facing = landingFace(
					machine.ox,
					machine.oy,
					mix32((machine.rng ^ FREE_RNG.face) >>> 0)
				);
			const wroteFacing = machine.facing;
			const first = world.sheet[machine.place(pc)];
			if ((first === 0xdd || first === 0xfd) && world.firstIndexPrefixTick === null)
				world.firstIndexPrefixTick = world.ticks;
			if (world.firstRepeatIoTick === null) {
				const second = world.sheet[machine.place((pc + 1) & 0xffff)];
				if (
					first === 0xed &&
					(second === 0xb2 ||
						second === 0xb3 ||
						second === 0xba ||
						second === 0xbb ||
						second === 0x5f)
				)
					world.firstRepeatIoTick = world.ticks;
			}
			log = new Map();
			machine.z80.step();

			machine.rng = (machine.rng + FREE_RNG.step) >>> 0;
			const random = mix32(machine.rng);
			let restart = turnover !== 0 && random < turnover;
			if (r.halted) {
				// HALT leaves the program counter on the HALT byte.
				r.halted = 0;
				world.halts++;
				if (rules.haltRule === 'restart') restart = true;
				else if (rules.haltRule === 'pass') r.pc = (r.pc + 1) & 0xffff;
			}
			if (restart) {
				blank(machine.z80);
				const a = mix32(random ^ FREE_RNG.landX);
				const b = mix32(a ^ FREE_RNG.landY);
				machine.ox = a % sheetW;
				machine.oy = b % sheetH;
				machine.facing = landingFace(machine.ox, machine.oy, mix32((b ^ FREE_RNG.face) >>> 0));
				world.restarts++;
			}
			if (mutation !== 0 && mix32(random ^ FREE_RNG.mutate) < mutation) {
				log.set(
					mix32(random ^ FREE_RNG.mutatePlace) % (sheetW * sheetH),
					(mix32(random ^ FREE_RNG.mutateValue) & 0xff) | MUTATION
				);
				world.mutations++;
			}
			for (const [p, value] of log) {
				writers.set(p, (writers.get(p) ?? 0) + 1);
				// Heads run in ascending order, so the first writer of a byte is the lowest-numbered.
				if (!writes.has(p)) writes.set(p, { value, facing: wroteFacing });
			}
		}
		for (const n of writers.values()) if (n > 1) world.contested++;
		for (const [p, { value, facing }] of writes) {
			world.sheet[p] = value & 0xff;
			if (!(value & MUTATION)) world.grain[p] = facing;
		}
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
