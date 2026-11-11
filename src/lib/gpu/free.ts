// The free-running world: no cells, no encounters, no step budget.
//
// The lattice's bytes are read as one continuous sheet (the same bytes the renderer draws, wrapped as
// a torus). A "head" is a processor that is never reset: its registers and program counter persist
// from one tick to the next. Each head lives on one line of the sheet (a row or a column, in one of
// the two senses) and its 16-bit address space is laid along that line from its origin, so the only
// space a program has is what it can address. One tick is one instruction for every head.
//
// A tick is a function of the sheet and the heads, with no race:
//   1. step   every head executes one instruction against the sheet as it stood at the start of the
//             tick (plus its own writes within that instruction). Its writes go to a private log and
//             it bids for each byte it wants to write.
//   2. apply  each byte goes to the lowest-numbered head that bid for it.
// HALT has no interrupt to end it. Under the `wait` rule the head stays on the HALT byte and fetches
// it again every tick, so it sleeps until something else changes that byte; under `pass` it moves on.
//
// Zilion supplies the instruction core and the processor-state load and store (library mode with
// per-lane state). The one host adapter is memory: the generated lane-private memory helpers are
// replaced by helpers that read the shared sheet and write to the log.

import { buildMachine } from '@neovand/zilion';
import { z80 } from '@neovand/zilion/z80';
import { SplitMix64 } from '../sim/prng';
import type { LabSettings } from '../lab/types';

/** Words per head: processor state first, then origin x, origin y, direction, waiting flag. */
export const HEAD_WORDS = 40;
export const HEAD_ORIGIN_X = 32;
export const HEAD_ORIGIN_Y = 33;
export const HEAD_DIRECTION = 34;
export const HEAD_WAITING = 35;
/** Words per head in the write log: count, then up to four (position, value) pairs. */
export const LOG_WORDS = 12;
const LOG_ENTRIES = 4;
export const NO_CLAIM = 0xffffffff;

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

const DIRECTIONS: Record<LabSettings['headDirections'], number[]> = {
	rows: [0],
	weave: [0, 1],
	all: [0, 1, 2, 3]
};

/**
 * The starting heads: every processor in its reset state (all zero), each with a random origin on
 * the sheet and a direction drawn from the allowed set. 0 east, 1 south, 2 west, 3 north.
 */
export function createHeads(settings: LabSettings): Uint32Array<ArrayBuffer> {
	const count = headCount(settings);
	const heads = new Uint32Array(count * HEAD_WORDS);
	const rng = new SplitMix64((settings.seed ^ 0x5eed1e55) >>> 0);
	const sheetW = settings.width * 4;
	const sheetH = settings.height * 4;
	const directions = DIRECTIONS[settings.headDirections];
	for (let h = 0; h < count; h++) {
		const o = h * HEAD_WORDS;
		heads[o + HEAD_ORIGIN_X] = rng.nextU32() % sheetW;
		heads[o + HEAD_ORIGIN_Y] = rng.nextU32() % sheetH;
		heads[o + HEAD_DIRECTION] = directions[rng.nextU32() % directions.length];
	}
	return heads;
}

export function createFreeShader(settings: LabSettings): string {
	const { wgsl } = build(settings);
	const generated =
		/fn zil_pos\(addr: u32\) -> u32 \{[^}]*\}\nfn mem_read\(addr: u32\) -> u32 \{[^}]*\}\nfn mem_write\(addr: u32, val: u32\) \{[^}]*\}\n/;
	if (!generated.test(wgsl) || !wgsl.includes('fn zil_load_state(o: u32)'))
		throw new Error('Unexpected Zilion library memory contract.');
	const machine = wgsl.replace(
		generated,
		/* wgsl */ `
// --- Host memory: the shared sheet, read in place; writes are logged and applied after the step ---
fn free_place(addr: u32) -> u32 {
    let a = addr & 0xffffu;
    var x = free_ox;
    var y = free_oy;
    switch (free_dir) {
        case 0u: { x = (free_ox + a) % fp.sheet_w; }
        case 1u: { y = (free_oy + a) % fp.sheet_h; }
        case 2u: { x = (free_ox + fp.sheet_w - (a % fp.sheet_w)) % fp.sheet_w; }
        default: { y = (free_oy + fp.sheet_h - (a % fp.sheet_h)) % fp.sheet_h; }
    }
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
fn mem_read(addr: u32) -> u32 {
    let p = free_place(addr);
    for (var i = free_n; i > 0u; i--) {
        if (free_pos[i - 1u] == p) { return free_val[i - 1u]; }
    }
    return free_peek(p);
}
fn mem_write(addr: u32, val: u32) {
    let p = free_place(addr);
    for (var i = 0u; i < free_n; i++) {
        if (free_pos[i] == p) { free_val[i] = val & 0xffu; return; }
    }
    if (free_n < ${LOG_ENTRIES}u) {
        free_pos[free_n] = p;
        free_val[free_n] = val & 0xffu;
        free_n++;
    } else {
        free_dropped = 1u;
    }
}
`
	);
	return /* wgsl */ `
struct FreeParams {
    sheet_w: u32,
    sheet_h: u32,
    head_count: u32,
    mutation_count: u32,
    seed: u32,
    halt_pass: u32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<storage, read_write> soup: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> heads: array<u32>;
@group(0) @binding(2) var<storage, read_write> head_log: array<u32>;
@group(0) @binding(3) var<storage, read_write> claims: array<atomic<u32>>;
@group(0) @binding(4) var<uniform> fp: FreeParams;

var<private> free_ox: u32;
var<private> free_oy: u32;
var<private> free_dir: u32;
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
    free_dir = heads[o + ${HEAD_DIRECTION}u];
    free_n = 0u;
    free_dropped = 0u;

    z80_step();

    // HALT leaves the program counter on the HALT byte. With no interrupt to end it, the head either
    // waits there and fetches the byte again next tick, or passes over it.
    var waiting = 0u;
    if (zil_halt_reason() != ZIL_RUNNING) {
        cpu_halted = 0u;
        if (fp.halt_pass != 0u) { cpu_pc = (cpu_pc + 1u) & 0xffffu; } else { waiting = 1u; }
    }
    zil_store_state(o);
    heads[o + ${HEAD_WAITING}u] = waiting;

    let l = h * ${LOG_WORDS}u;
    head_log[l] = free_n | (free_dropped << 16u);
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
    for (var i = 0u; i < n; i++) {
        let pos = head_log[l + 1u + i * 2u];
        if (atomicLoad(&claims[pos]) == h) {
            free_poke(pos, head_log[l + 2u + i * 2u]);
            atomicStore(&claims[pos], ${NO_CLAIM}u);
        }
    }
}

// 3. Mutation: random bytes of the sheet are replaced. The sheet is cut into as many equal stretches
// as there are mutations (from a random starting byte) and each mutation lands in its own stretch, so
// no two ever meet on a byte and the tick stays a function of its inputs.
@compute @workgroup_size(64)
fn free_mutate(@builtin(global_invocation_id) id: vec3u) {
    let m = id.x;
    if (m >= fp.mutation_count) { return; }
    let total = fp.sheet_w * fp.sheet_h;
    let stretch = total / fp.mutation_count;
    let start = zil_mix32(fp.seed ^ 0x51ed270bu) % total;
    let a = zil_mix32(fp.seed * 3266489917u + m * 668265263u + 7u);
    let b = zil_mix32(a ^ 0x9e3779b9u);
    free_poke((start + m * stretch + a % stretch) % total, b & 0xffu);
}
`;
}
