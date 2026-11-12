// Reference for the two cell-wise worlds (DEV ONLY): the rule of gpu/around.ts written a second
// time in plain TypeScript, with the third-party `z80-emulator` executing the instructions.
//
// One call computes one update of every cell from the lattice and the processors as they stood
// before it, so a test can compare the GPU's update with this one cell by cell and start the next
// comparison from the GPU's own state. A cell is `flagged` when its run fetched an instruction on
// which the two Z80s are known to differ (DD/FD prefix chains; H and P/V on a repeating block
// input or output; PUSH AF, which puts the two undocumented flag bits into memory; LD A,R, where
// z80-emulator returns the refresh register with bit 3 cleared: R = 13 gives A = 7 where the count
// is 15); such cells are not comparable.

import { Z80, type Hal } from 'z80-emulator';
import {
	FREE_RNG,
	HEAD_AGE,
	HEAD_FACING,
	HEAD_RNG,
	HEAD_WAITING,
	HEAD_WORDS,
	chanceThreshold
} from '../gpu/free';
import { AROUND_RNG, NEIGHBOUR_STEPS, STOP, neighbourMet } from '../gpu/around';
import { mix32 } from './freeReference';

export interface AroundRules {
	rule: 'neighbourhood' | 'cells';
	gather: number;
	/** Instructions after which a run is cut off. */
	budget: number;
	/** Chance of one random byte: per update (`neighbourhood`) or per instruction (`cells`). */
	mutation: number;
	/** `neighbourhood`: chance that a cell forgets its previous tape first (1 if omitted). */
	dissipation?: number;
	/** `neighbourhood`: the update undoes an earlier one (the random words are drawn in reverse). */
	backward?: boolean;
}

export interface AroundCell {
	/** The cell's tape after the update. */
	tape: Uint8Array;
	/** The cell's remembered tape after the update (`neighbourhood`: its tape before it). */
	before: Uint8Array;
	flagged: boolean;
	/** `neighbourhood`: instructions executed and why the run ended. */
	steps: number;
	reason: number;
	/** `cells`: whether the run ended (HALT or a full cell) and the processor started afresh. */
	restarted: boolean;
	/** The order of neighbours the processor has after the update. */
	order: number;
	registers: Record<string, number>;
	/** The bytes at the program counter before the last instruction looked at (for reports). */
	fetched: string;
}

export function referenceUpdate(
	width: number,
	height: number,
	tapes: Uint8Array,
	heads: Uint32Array,
	state: Record<string, number>,
	rules: AroundRules,
	before?: Uint8Array
): AroundCell[] {
	const dissipation = rules.dissipation ?? 1;
	// The engine's threshold for a chance of forgetting.
	const forgetBelow = Math.min(0xffffffff, Math.floor(dissipation * 4294967296));
	const mutation = chanceThreshold(rules.mutation);
	const half = rules.gather * 16;
	const out: AroundCell[] = [];
	for (let c = 0; c < width * height; c++) {
		const o = c * HEAD_WORDS;
		const fresh = rules.rule === 'neighbourhood';
		const counter =
			fresh && rules.backward ? heads[o + HEAD_RNG] : (heads[o + HEAD_RNG] + FREE_RNG.step) >>> 0;
		const random = mix32(counter);
		const drawn = mix32((random ^ FREE_RNG.face) >>> 0) & 7;
		let order = fresh ? drawn : heads[o + HEAD_FACING] & 7;

		const cx = c % width;
		const cy = Math.floor(c / width);
		const program = new Uint8Array(half);
		for (let j = 0; j < rules.gather; j++) {
			const [dx, dy] = NEIGHBOUR_STEPS[neighbourMet(order, j)];
			const n = ((cy + dy + height) % height) * width + ((cx + dx + width) % width);
			program.set(tapes.subarray(n * 16, n * 16 + 16), j * 16);
		}
		const tape = tapes.slice(c * 16, c * 16 + 16);
		let written = 0;
		const read = (address: number) => {
			const a = address & (2 * half - 1);
			return a < half ? program[a] : tape[a & 15];
		};
		const hal: Hal = {
			tStateCount: 0,
			readMemory: read,
			writeMemory: (address: number, value: number) => {
				const a = address & (2 * half - 1);
				if (a < half) program[a] = value & 0xff;
				else {
					tape[a & 15] = value & 0xff;
					written |= 1 << (a & 15);
				}
			},
			contendMemory: () => {},
			readPort: () => 0,
			writePort: () => {},
			contendPort: () => {}
		};
		const z80 = new Z80(hal);
		z80.reset();
		const r = z80.regs;
		const word = (name: string) => (fresh ? 0 : heads[o + state[name]]);
		r.af = (word('a') << 8) | word('f');
		r.bc = (word('b') << 8) | word('c');
		r.de = (word('d') << 8) | word('e');
		r.hl = (word('h') << 8) | word('l');
		r.afPrime = (word('a_') << 8) | word('f_');
		r.bcPrime = (word('b_') << 8) | word('c_');
		r.dePrime = (word('d_') << 8) | word('e_');
		r.hlPrime = (word('h_') << 8) | word('l_');
		r.ix = word('ix');
		r.iy = word('iy');
		r.sp = word('sp');
		r.pc = word('pc');
		r.memptr = 0;
		r.i = word('i');
		r.r = word('r') & 0x7f;
		r.r7 = word('r') & 0x80;
		r.iff1 = word('iff1');
		r.iff2 = word('iff2');
		r.im = 0;
		r.halted = 0;

		let flagged = false;
		let fetched = '';
		const look = () => {
			const first = read(r.pc);
			const second = read((r.pc + 1) & 0xffff);
			fetched = [0, 1, 2, 3]
				.map((i) =>
					read((r.pc + i) & 0xffff)
						.toString(16)
						.padStart(2, '0')
				)
				.join(' ');
			if (
				first === 0xdd ||
				first === 0xfd ||
				first === 0xf5 ||
				(first === 0xed &&
					(second === 0xb2 ||
						second === 0xb3 ||
						second === 0xba ||
						second === 0xbb ||
						second === 0x5f))
			)
				flagged = true;
		};
		let steps = 0;
		let reason = STOP.budget as number;
		let restarted = false;
		if (fresh) {
			while (steps < rules.budget) {
				look();
				z80.step();
				steps++;
				if (r.halted) {
					reason = STOP.halt;
					break;
				}
				if (written === 0xffff) {
					reason = STOP.full;
					break;
				}
			}
		} else {
			look();
			z80.step();
			// A run ends by HALT or when it has written every byte of the cell (over all its ticks).
			if (
				r.halted ||
				(heads[o + HEAD_WAITING] | written) === 0xffff ||
				heads[o + HEAD_AGE] + 1 >= rules.budget
			) {
				restarted = true;
				order = drawn;
			}
		}
		// The two-layer law: now' = out xor now xor before, before' = now; a cell that forgets first
		// (before := now) gets now' = out. Free-running cells keep the plain rule.
		const now = tapes.subarray(c * 16, c * 16 + 16);
		const forgets =
			!fresh ||
			!before ||
			forgetBelow === 0xffffffff ||
			mix32((random ^ AROUND_RNG.forget) >>> 0) < forgetBelow;
		const next = new Uint8Array(16);
		for (let i = 0; i < 16; i++)
			next[i] = tape[i] ^ now[i] ^ (forgets ? now[i] : before![c * 16 + i]);
		if (mutation !== 0 && mix32((random ^ FREE_RNG.mutate) >>> 0) < mutation)
			next[mix32((random ^ FREE_RNG.mutatePlace) >>> 0) & 15] =
				mix32((random ^ FREE_RNG.mutateValue) >>> 0) & 0xff;
		out.push({
			tape: next,
			before: fresh ? now.slice() : (before?.slice(c * 16, c * 16 + 16) ?? now.slice()),
			flagged,
			steps,
			reason,
			restarted,
			order,
			fetched,
			registers: restarted
				? { a: 0, f: 0, b: 0, c: 0, d: 0, e: 0, h: 0, l: 0, sp: 0, pc: 0 }
				: {
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
					}
		});
	}
	return out;
}
