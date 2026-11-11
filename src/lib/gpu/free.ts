// The free-running world: no cells, no encounters, no step budget.
//
// The world is a sheet of bytes (the same bytes the renderer draws, wrapped as a torus) and nothing
// else. Processors land on it:
//
//   1. A processor lands on a random byte, facing a random way, with every register at zero, and
//      starts executing there.
//   2. Its addresses are measured from where it landed, in its own frame. A Z80 address is two bytes:
//      the low byte counts bytes along the way it faces, the high byte counts lines across, each read
//      as a signed offset (-128..127). Address 0 is the landing byte, the program counter walks
//      forward, and the stack (SP = 0, so the first push writes 0xFFFF and 0xFFFE) grows backwards
//      along the neighbouring line. "Facing" is one of the eight symmetries of the square (four
//      directions, the stack on either side), so the rule itself has no preferred direction and no
//      handedness. (`facing: 'same'` gives every processor the same frame, for comparison: forward is
//      right and the stack is the row above. `written` and `bytes` let the sheet set the facing; see
//      FACING_MASKS.)
//   3. One tick is one instruction for every processor, all at once.
//   4. A processor that executes HALT lifts off and lands somewhere else (rule 1). Nothing else ends a
//      run: there is no step budget. (`turnover` adds an optional chance of lifting off per
//      instruction; it is 0 by default.)
//
// Mutation: with a small chance per instruction, one random byte of the sheet is replaced.
//
// A tick is a function of the sheet and the processors, with no race:
//   step   every processor executes one instruction against the sheet as it stood at the start of the
//          tick (plus its own writes within that instruction). Its writes go to a private log and it
//          bids for each byte it wants to write.
//   apply  each byte goes to the lowest-numbered processor that bid for it.
// All randomness comes from a counter each processor carries, so many ticks can be queued at once.
//
// Zilion supplies the instruction core and the processor-state load and store (library mode with
// per-lane state). The one host adapter is memory: the generated lane-private memory helpers are
// replaced by helpers that read the shared sheet and write to the log.

import { buildMachine } from '@neovand/zilion';
import { z80 } from '@neovand/zilion/z80';
import { SplitMix64 } from '../sim/prng';
import type { LabSettings } from '../lab/types';

/** Words per head: processor state first, then the fields below. */
export const HEAD_WORDS = 40;
export const HEAD_ORIGIN_X = 32;
export const HEAD_ORIGIN_Y = 33;
/** Instructions executed since it landed. */
export const HEAD_AGE = 34;
/** 1 while it sits on a HALT byte under the `wait` rule. */
export const HEAD_WAITING = 35;
/** Runs it has finished (times it lifted off). */
export const HEAD_RUNS = 36;
/** Length of the run it finished last, in instructions. */
export const HEAD_LAST_RUN = 37;
/** Its random counter. */
export const HEAD_RNG = 38;
/**
 * The way it faces: bit 0 swaps the two axes, bit 1 negates x, bit 2 negates y (applied in that
 * order to the offset "along, across"). The eight values are the eight symmetries of the square.
 */
export const HEAD_FACING = 39;
/** Words per head in the write log: count, then up to five (position, value) pairs. */
export const LOG_WORDS = 12;
const LOG_ENTRIES = 5;
export const NO_CLAIM = 0xffffffff;

/** The constants of the per-head random draws, shared with the reference (src/lib/dev). */
export const FREE_RNG = {
	step: 0x9e3779b9,
	mutate: 0x51ed270b,
	mutatePlace: 0x2545f491,
	mutateValue: 0x7f4a7c15,
	landX: 0x85ebca6b,
	landY: 0xc2b2ae35,
	face: 0x27d4eb2f,
	faceTie: 0x165667b1
} as const;

/** One mutation per this many instructions at mutation exponent 0, so the dial reads like the pair worlds'. */
export const FREE_MUTATION_UNIT = 128;

export const HALT_RULES = { restart: 0, wait: 1, pass: 2 } as const;

/**
 * How a landing head's facing is set.
 *
 * Drawn at random under a mask: `any` (all eight), `same` (always the first: forward is right, stack
 * on the row above), and two in-between rules that exist to find out which part of the symmetry
 * matters: `rows` (forward is right or left, stack above or below) and `two-axes` (right or down).
 *
 * Set by the sheet, by rules that themselves have no preferred direction:
 * `written`: every byte remembers the facing of the head that wrote it last (its "grain"); a head
 * takes the grain of the byte it lands on. Mutation changes a byte's value, not its grain.
 * `bytes`: forward is towards the largest of the four neighbouring bytes (ties at random); the side
 * of the stack is random.
 */
export const FACING_MASKS = {
	any: 7,
	rows: 6,
	'two-axes': 1,
	same: 0,
	written: 7,
	bytes: 7
} as const;
/** 0: random under the mask. 1: the grain of the landing byte. 2: from the neighbouring bytes. */
export const FACING_RULES = {
	any: 0,
	rows: 0,
	'two-axes': 0,
	same: 0,
	written: 1,
	bytes: 2
} as const;
/** A head whose facing has not been set yet (it is set, by the rule, before its first instruction). */
export const FACING_UNDECIDED = 0xffffffff;
/** Bit 8 of a logged value marks a mutation, which leaves the grain of the byte alone. */
export const LOG_MUTATION = 0x100;

/**
 * The `bytes` rule. Forward is towards the largest of the neighbouring bytes to the east, west,
 * south and north; `random` breaks ties without favouring a direction and picks the stack side.
 * `mix` is the 32-bit mixer the random words come from (Zilion's zil_mix32).
 */
export function facingFromNeighbours(
	east: number,
	west: number,
	south: number,
	north: number,
	random: number,
	mix: (x: number) => number
): number {
	const second = mix((random ^ FREE_RNG.faceTie) >>> 0);
	const turn = second & 3;
	const side = (second >>> 2) & 1;
	// Value first, then a random byte, then the direction's index under a random relabelling.
	const score = (value: number, index: number) =>
		value * 1024 + ((random >>> (8 * index)) & 0xff) * 4 + (index ^ turn);
	// Forward east is facing 0 or 4, west 2 or 6, south 1 or 3, north 5 or 7 (see faceOffset).
	let best = score(east, 0);
	let facing = side ? 4 : 0;
	if (score(west, 1) > best) {
		best = score(west, 1);
		facing = side ? 6 : 2;
	}
	if (score(south, 2) > best) {
		best = score(south, 2);
		facing = side ? 3 : 1;
	}
	if (score(north, 3) > best) facing = side ? 7 : 5;
	return facing;
}

/** A facing applied to an offset (along, across): the offset on the sheet as [dx, dy]. */
export function faceOffset(facing: number, along: number, across: number): [number, number] {
	let dx = facing & 1 ? across : along;
	let dy = facing & 1 ? along : across;
	if (facing & 2) dx = -dx;
	if (facing & 4) dy = -dy;
	return [dx, dy];
}

/** A chance in [0, 1] as the 32-bit threshold the shader compares a random word against. */
export function chanceThreshold(chance: number): number {
	if (!(chance > 0)) return 0;
	return Math.min(0xffffffff, Math.round(chance * 2 ** 32));
}

export function mutationChance(mutationExponent: number): number {
	return 1 / (FREE_MUTATION_UNIT * 2 ** mutationExponent);
}

export interface FreeLayout {
	/** Index of each processor-state field inside a head, e.g. `pc`, `sp`, `a`. */
	state: Record<string, number>;
	stateWords: number;
}

function build(settings: LabSettings) {
	if (settings.isa !== 'z80') throw new Error('The free-running world runs the Z80 only.');
	return buildMachine({
		core: { isa: z80, compat: settings.compat, variant: settings.variant },
		mode: 'library',
		world: {
			// Lane memory is replaced below; the declaration only has to be valid.
			memory: {
				kind: 'segments',
				segments: [
					{ bytes: 16, name: 'A' },
					{ bytes: 16, name: 'B' }
				],
				ring: 32
			},
			init: { state: 'perLane' },
			budget: { unit: 'steps', value: 1 },
			outputs: { memory: false, state: true, writeCounts: false, fetchMap: false }
		},
		bindingNames: { out: 'heads', stateIn: 'heads' }
	} as Parameters<typeof buildMachine>[0]);
}

export function freeLayout(settings: LabSettings): FreeLayout {
	const record = build(settings).manifest.record;
	const field = record.fields.find((f) => f.name === 'state');
	if (!field?.items) throw new Error('Unexpected Zilion state record.');
	if (field.items.length > HEAD_ORIGIN_X) throw new Error('Processor state does not fit a head.');
	return {
		state: Object.fromEntries(field.items.map((name, index) => [name, index])),
		stateWords: field.items.length
	};
}

export function headCount(settings: LabSettings): number {
	return Math.max(1, Math.round(settings.width * settings.height * settings.heads));
}

/**
 * The starting heads: every processor in its reset state (all zero), landed on a random byte of the
 * sheet, with its own random counter.
 */
export function createHeads(settings: LabSettings): Uint32Array<ArrayBuffer> {
	const count = headCount(settings);
	const heads = new Uint32Array(count * HEAD_WORDS);
	const rng = new SplitMix64((settings.seed ^ 0x5eed1e55) >>> 0);
	// A separate stream, so that `same` starts from the same landing sites as `any`.
	const faces = new SplitMix64((settings.seed ^ 0x0fac1e55) >>> 0);
	const mask = FACING_MASKS[settings.facing];
	const drawn = FACING_RULES[settings.facing] === 0;
	const sheetW = settings.width * 4;
	const sheetH = settings.height * 4;
	for (let h = 0; h < count; h++) {
		const o = h * HEAD_WORDS;
		heads[o + HEAD_ORIGIN_X] = rng.nextU32() % sheetW;
		heads[o + HEAD_ORIGIN_Y] = rng.nextU32() % sheetH;
		heads[o + HEAD_RNG] = rng.nextU32();
		heads[o + HEAD_FACING] = drawn ? faces.nextU32() & mask : FACING_UNDECIDED;
	}
	return heads;
}

/** The starting grain: a random facing for every byte of the sheet (one word per byte, row-major). */
export function createGrain(settings: LabSettings): Uint32Array<ArrayBuffer> {
	const grain = new Uint32Array(settings.width * settings.height * 16);
	const rng = new SplitMix64((settings.seed ^ 0x67a11e55) >>> 0);
	for (let i = 0; i < grain.length; i += 8) {
		// One 64-bit draw gives eight facings.
		let word = rng.next();
		for (let k = 0; k < 8 && i + k < grain.length; k++) {
			grain[i + k] = Number(word & 7n);
			word >>= 3n;
		}
	}
	return grain;
}

const hex = (n: number) => `0x${(n >>> 0).toString(16)}u`;

export function createFreeShader(settings: LabSettings): string {
	const { wgsl } = build(settings);
	const { stateWords } = freeLayout(settings);
	const generated =
		/fn zil_pos\(addr: u32\) -> u32 \{[^}]*\}\nfn mem_read\(addr: u32\) -> u32 \{[^}]*\}\nfn mem_write\(addr: u32, val: u32\) \{[^}]*\}\n/;
	if (!generated.test(wgsl) || !wgsl.includes('fn zil_load_state(o: u32)'))
		throw new Error('Unexpected Zilion library memory contract.');
	const machine = wgsl.replace(
		generated,
		/* wgsl */ `
// --- Host memory: the shared sheet, read in place; writes are logged and applied after the step ---
// An address is an offset from where the head landed, in the head's own frame: the low byte counts
// bytes along the way it faces, the high byte counts lines across, each a signed byte.
fn free_place(addr: u32) -> u32 {
    let lo = i32(addr & 0xffu);
    let hi = i32((addr >> 8u) & 0xffu);
    let along = lo - select(0, 256, lo >= 128);
    let across = hi - select(0, 256, hi >= 128);
    let swap = (free_face & 1u) != 0u;
    var dx = select(along, across, swap);
    var dy = select(across, along, swap);
    if ((free_face & 2u) != 0u) { dx = -dx; }
    if ((free_face & 4u) != 0u) { dy = -dy; }
    // "+ 256 * size" keeps the sum positive and is a whole number of turns round the torus.
    let x = u32(i32(free_ox) + dx + i32(256u * fp.sheet_w)) % fp.sheet_w;
    let y = u32(i32(free_oy) + dy + i32(256u * fp.sheet_h)) % fp.sheet_h;
    return y * fp.sheet_w + x;
}
// Sheet position to the lattice's storage: tile (x/4, y/4), one word per row of four bytes.
fn free_word(pos: u32) -> u32 {
    let x = pos % fp.sheet_w;
    let y = pos / fp.sheet_w;
    return ((y >> 2u) * (fp.sheet_w >> 2u) + (x >> 2u)) * 4u + (y & 3u);
}
fn free_shift(pos: u32) -> u32 { return ((pos % fp.sheet_w) & 3u) * 8u; }
fn free_peek(pos: u32) -> u32 { return (atomicLoad(&soup[free_word(pos)]) >> free_shift(pos)) & 0xffu; }
fn free_poke(pos: u32, val: u32) {
    let w = free_word(pos);
    let shift = free_shift(pos);
    loop {
        let old = atomicLoad(&soup[w]);
        let next = (old & ~(0xffu << shift)) | ((val & 0xffu) << shift);
        if (atomicCompareExchangeWeak(&soup[w], old, next).exchanged) { break; }
    }
}
fn free_log(p: u32, val: u32) {
    for (var i = 0u; i < free_n; i++) {
        if (free_pos[i] == p) { free_val[i] = val & 0x1ffu; return; }
    }
    if (free_n < ${LOG_ENTRIES}u) {
        free_pos[free_n] = p;
        free_val[free_n] = val & 0x1ffu;
        free_n++;
    } else {
        free_dropped = 1u;
    }
}
fn mem_read(addr: u32) -> u32 {
    let p = free_place(addr);
    for (var i = free_n; i > 0u; i--) {
        if (free_pos[i - 1u] == p) { return free_val[i - 1u] & 0xffu; }
    }
    return free_peek(p);
}
fn mem_write(addr: u32, val: u32) { free_log(free_place(addr), val & 0xffu); }
// The facing of a head that lands on (x, y); rnd is a random word of its own.
fn free_landing_face(x: u32, y: u32, rnd: u32) -> u32 {
    if (fp.facing_rule == 1u) { return grain[y * fp.sheet_w + x] & 7u; }
    if (fp.facing_rule != 2u) { return rnd & fp.facing_mask; }
    // Forward is towards the largest neighbouring byte. Value first, then a random byte, then the
    // direction's index under a random relabelling, so that ties favour no direction.
    let w = fp.sheet_w;
    let h = fp.sheet_h;
    let second = zil_mix32(rnd ^ ${hex(FREE_RNG.faceTie)});
    let turn = second & 3u;
    let side = ((second >> 2u) & 1u) == 1u;
    let east = (free_peek(y * w + (x + 1u) % w) << 10u) | ((rnd & 0xffu) << 2u) | turn;
    let west = (free_peek(y * w + (x + w - 1u) % w) << 10u) | (((rnd >> 8u) & 0xffu) << 2u) | (1u ^ turn);
    let south = (free_peek(((y + 1u) % h) * w + x) << 10u) | (((rnd >> 16u) & 0xffu) << 2u) | (2u ^ turn);
    let north = (free_peek(((y + h - 1u) % h) * w + x) << 10u) | (((rnd >> 24u) & 0xffu) << 2u) | (3u ^ turn);
    var best = east;
    var face = select(0u, 4u, side);
    if (west > best) { best = west; face = select(2u, 6u, side); }
    if (south > best) { best = south; face = select(1u, 3u, side); }
    if (north > best) { face = select(5u, 7u, side); }
    return face;
}
`
	);
	return /* wgsl */ `
struct FreeParams {
    sheet_w: u32,
    sheet_h: u32,
    head_count: u32,
    halt_rule: u32,      // 0 restart, 1 wait, 2 pass
    turnover: u32,       // chance per instruction of lifting off, as a 32-bit threshold
    mutation: u32,       // chance per instruction of one random byte, as a 32-bit threshold
    facing_mask: u32,    // facing rule 0: a random word under this mask (7: any of eight)
    facing_rule: u32,    // 0 random under the mask, 1 the grain of the landing byte, 2 from the bytes
}

@group(0) @binding(0) var<storage, read_write> soup: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> heads: array<u32>;
@group(0) @binding(2) var<storage, read_write> head_log: array<u32>;
@group(0) @binding(3) var<storage, read_write> claims: array<atomic<u32>>;
@group(0) @binding(4) var<uniform> fp: FreeParams;
// The facing of the head that wrote each byte last, one word per byte of the sheet, row-major.
@group(0) @binding(5) var<storage, read_write> grain: array<u32>;

var<private> free_ox: u32;
var<private> free_oy: u32;
var<private> free_face: u32;
var<private> free_n: u32;
var<private> free_dropped: u32;
var<private> free_pos: array<u32, ${LOG_ENTRIES}>;
var<private> free_val: array<u32, ${LOG_ENTRIES}>;

${machine}

// 1. Every head executes one instruction and bids for the bytes it wants to write.
@compute @workgroup_size(64)
fn free_step(@builtin(global_invocation_id) id: vec3u) {
    let h = id.x;
    if (h >= fp.head_count) { return; }
    let o = h * ${HEAD_WORDS}u;
    zil_load_state(o);
    idx_mode = 0u;
    idx_disp = 0u;
    idx_uses_mem = 0u;
    free_ox = heads[o + ${HEAD_ORIGIN_X}u];
    free_oy = heads[o + ${HEAD_ORIGIN_Y}u];
    free_face = heads[o + ${HEAD_FACING}u];
    // A head that has only been placed takes its facing from the rule before its first instruction.
    if (free_face > 7u) {
        free_face = free_landing_face(free_ox, free_oy, zil_mix32(heads[o + ${HEAD_RNG}u] ^ ${hex(FREE_RNG.face)}));
    }
    free_n = 0u;
    free_dropped = 0u;

    z80_step();

    // This head's random word for the tick.
    let counter = heads[o + ${HEAD_RNG}u] + ${hex(FREE_RNG.step)};
    heads[o + ${HEAD_RNG}u] = counter;
    let r = zil_mix32(counter);

    // HALT leaves the program counter on the HALT byte. The head lifts off and lands elsewhere
    // (restart), stays and fetches the byte again next tick (wait), or steps over it (pass).
    var waiting = 0u;
    var restart = fp.turnover != 0u && r < fp.turnover;
    if (zil_halt_reason() != ZIL_RUNNING) {
        cpu_halted = 0u;
        if (fp.halt_rule == 0u) { restart = true; }
        else if (fp.halt_rule == 2u) { cpu_pc = (cpu_pc + 1u) & 0xffffu; }
        else { waiting = 1u; }
    }
    let age = heads[o + ${HEAD_AGE}u] + 1u;
    if (restart) {
        for (var i = 0u; i < ${stateWords}u; i++) { heads[o + i] = 0u; }
        let a = zil_mix32(r ^ ${hex(FREE_RNG.landX)});
        let b = zil_mix32(a ^ ${hex(FREE_RNG.landY)});
        heads[o + ${HEAD_ORIGIN_X}u] = a % fp.sheet_w;
        heads[o + ${HEAD_ORIGIN_Y}u] = b % fp.sheet_h;
        heads[o + ${HEAD_FACING}u] =
            free_landing_face(a % fp.sheet_w, b % fp.sheet_h, zil_mix32(b ^ ${hex(FREE_RNG.face)}));
        heads[o + ${HEAD_AGE}u] = 0u;
        heads[o + ${HEAD_RUNS}u] = heads[o + ${HEAD_RUNS}u] + 1u;
        heads[o + ${HEAD_LAST_RUN}u] = age;
        waiting = 0u;
    } else {
        zil_store_state(o);
        heads[o + ${HEAD_AGE}u] = age;
        heads[o + ${HEAD_FACING}u] = free_face;
    }
    heads[o + ${HEAD_WAITING}u] = waiting;

    // Mutation: with a small chance, one random byte anywhere on the sheet. It is one more entry in
    // the log, so it goes through the same bidding as any write.
    let m = zil_mix32(r ^ ${hex(FREE_RNG.mutate)});
    if (fp.mutation != 0u && m < fp.mutation) {
        let place = zil_mix32(r ^ ${hex(FREE_RNG.mutatePlace)}) % (fp.sheet_w * fp.sheet_h);
        free_log(place, (zil_mix32(r ^ ${hex(FREE_RNG.mutateValue)}) & 0xffu) | ${LOG_MUTATION}u);
    }

    let l = h * ${LOG_WORDS}u;
    // The facing goes with the writes: the head may already have landed somewhere else.
    head_log[l] = free_n | (free_dropped << 16u) | (free_face << 20u);
    for (var i = 0u; i < free_n; i++) {
        head_log[l + 1u + i * 2u] = free_pos[i];
        head_log[l + 2u + i * 2u] = free_val[i];
        atomicMin(&claims[free_pos[i]], h);
    }
}

// 2. Each byte goes to the lowest-numbered head that bid for it; the winner clears the bid.
@compute @workgroup_size(64)
fn free_apply(@builtin(global_invocation_id) id: vec3u) {
    let h = id.x;
    if (h >= fp.head_count) { return; }
    let l = h * ${LOG_WORDS}u;
    let n = head_log[l] & 0xffffu;
    let face = (head_log[l] >> 20u) & 7u;
    for (var i = 0u; i < n; i++) {
        let pos = head_log[l + 1u + i * 2u];
        if (atomicLoad(&claims[pos]) == h) {
            let val = head_log[l + 2u + i * 2u];
            free_poke(pos, val & 0xffu);
            // A written byte takes the grain of its writer; a mutated byte keeps the grain it had.
            if ((val & ${LOG_MUTATION}u) == 0u) { grain[pos] = face; }
            atomicStore(&claims[pos], ${NO_CLAIM}u);
        }
    }
}
`;
}
