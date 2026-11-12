// Two worlds in which every cell is rewritten only by its own processor, from the tapes around it.
//
// Each cell holds a 16-byte tape, as in the cell soups. Each cell also has a processor. What the
// processor sees as memory is a ring made of two halves:
//
//   program half   the tapes of `gather` of its four neighbours (1, 2 or 4 of them), one after the
//                  other, starting from a random neighbour and going round in a random sense;
//   cell half      the cell's own tape, repeated to the same length.
//
// Execution starts at the first byte of the first neighbour, with every register at zero, so the
// stack (SP = 0) grows down from the end of the cell's own tape. The processor may read everything
// and write anywhere, but only what it writes to its own tape is kept: writes to a neighbour's tape
// change a private copy that is thrown away. So every cell can be updated at the same time from the
// same old lattice, with no two processors ever wanting the same byte.
//
// `neighbourhood` (a rule of the cellular-automaton kind): at every update each processor starts
// afresh and runs until it executes HALT, or until it has written every byte of its cell, or (for
// programs caught in a loop) until `steps` instructions. The cell's new tape is what it wrote.
//
// `cells` (free-running cells): the same run, spread over time. The processor keeps its registers
// from tick to tick and executes one instruction per tick, reading its neighbours as they are at
// that tick; writes outside its own tape are dropped. A run ends as above, by HALT or by a full
// cell, and the processor then starts afresh with a new order of neighbours. There is no step
// budget: a processor caught in a loop stays in it until the tapes it is reading change.
//
// Mutation: a chance per update (`neighbourhood`) or per instruction (`cells`) that one random
// byte of the cell is replaced.
//
// The order in which a processor meets its neighbours is drawn among the eight symmetries of the
// square each time, so the rule has no preferred direction.

import { buildMachine } from '@neovand/zilion';
import { z80 } from '@neovand/zilion/z80';
import { SplitMix64 } from '../sim/prng';
import type { LabSettings } from '../lab/types';
import {
	FREE_MUTATION_UNIT,
	FREE_RNG,
	HEAD_AGE,
	HEAD_FACING,
	HEAD_LAST_RUN,
	HEAD_RNG,
	HEAD_RUNS,
	HEAD_WAITING,
	HEAD_WORDS
} from './free';

export const AROUND_MODES = { neighbourhood: 0, cells: 1 } as const;
export type AroundRule = keyof typeof AROUND_MODES;
export const GATHERS = [1, 2, 4] as const;
/** Why the last run of a `neighbourhood` cell ended (stored in HEAD_WAITING). */
export const STOP = { halt: 1, full: 2, budget: 3 } as const;
/** East, south, west, north: the order in which neighbours are met before turning and flipping. */
export const NEIGHBOUR_STEPS = [
	[1, 0],
	[0, 1],
	[-1, 0],
	[0, -1]
] as const;

export function isAround(interaction: LabSettings['interaction']): interaction is AroundRule {
	return interaction === 'neighbourhood' || interaction === 'cells';
}

/** The j-th neighbour met under an order (0..7): which of east, south, west, north. */
export function neighbourMet(order: number, j: number): number {
	const start = order & 3;
	return (start + (order & 4 ? 4 - j : j)) & 3;
}

/** Chance that one byte of a cell is replaced: per update (`neighbourhood`), per instruction (`cells`). */
export function aroundMutationChance(rule: AroundRule, mutationExponent: number): number {
	return rule === 'cells'
		? 1 / (FREE_MUTATION_UNIT * 2 ** mutationExponent)
		: 1 / 2 ** (mutationExponent + 1);
}

/** One processor per cell (random counter, random order of neighbours), then one all-zero block. */
export function createCellHeads(settings: LabSettings): Uint32Array<ArrayBuffer> {
	const cells = settings.width * settings.height;
	const heads = new Uint32Array((cells + 1) * HEAD_WORDS);
	const rng = new SplitMix64((settings.seed ^ 0x0ce11e55) >>> 0);
	for (let c = 0; c < cells; c++) {
		heads[c * HEAD_WORDS + HEAD_RNG] = rng.nextU32();
		heads[c * HEAD_WORDS + HEAD_FACING] = rng.nextU32() & 7;
	}
	return heads;
}

function build(settings: LabSettings) {
	if (settings.isa !== 'z80') throw new Error('These worlds run the Z80 only.');
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

export function aroundLayout(settings: LabSettings): {
	state: Record<string, number>;
	stateWords: number;
} {
	const field = build(settings).manifest.record.fields.find((f) => f.name === 'state');
	if (!field?.items) throw new Error('Unexpected Zilion state record.');
	return {
		state: Object.fromEntries(field.items.map((name, index) => [name, index])),
		stateWords: field.items.length
	};
}

const hex = (n: number) => `0x${(n >>> 0).toString(16)}u`;

export function createAroundShader(settings: LabSettings): string {
	const { wgsl } = build(settings);
	const { stateWords } = aroundLayout(settings);
	const generated =
		/fn zil_pos\(addr: u32\) -> u32 \{[^}]*\}\nfn mem_read\(addr: u32\) -> u32 \{[^}]*\}\nfn mem_write\(addr: u32, val: u32\) \{[^}]*\}\n/;
	if (!generated.test(wgsl) || !wgsl.includes('fn zil_load_state(o: u32)'))
		throw new Error('Unexpected Zilion library memory contract.');
	const machine = wgsl.replace(
		generated,
		/* wgsl */ `
// --- Host memory: a private ring. Bytes 0..63 hold the gathered neighbours, 64..79 the cell. ---
fn around_index(addr: u32) -> u32 {
    let half = ap.gather * 16u;
    let a = addr & (2u * half - 1u);
    if (a < half) { return a; }
    return 64u + (a & 15u);
}
fn mem_read(addr: u32) -> u32 { return around_mem[around_index(addr)]; }
fn mem_write(addr: u32, val: u32) {
    let i = around_index(addr);
    around_mem[i] = val & 0xffu;
    if (i >= 64u) { around_written |= 1u << (i - 64u); }
}
`
	);
	return /* wgsl */ `
struct AroundParams {
    width: u32,
    height: u32,
    gather: u32,     // neighbours whose tapes make the program: 1, 2 or 4
    budget: u32,     // instructions after which a run is cut off
    mutation: u32,   // chance of one random byte, as a 32-bit threshold
    mode: u32,       // 0 neighbourhood (a whole run per update), 1 cells (one instruction per tick)
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<storage, read_write> soup: array<u32>;
@group(0) @binding(1) var<storage, read_write> heads: array<u32>;
@group(0) @binding(2) var<storage, read_write> next_soup: array<u32>;
@group(0) @binding(3) var<uniform> ap: AroundParams;

var<private> around_mem: array<u32, 80>;
var<private> around_written: u32;

${machine}

fn around_tape_byte(cell: u32, i: u32) -> u32 {
    return (soup[cell * 4u + (i >> 2u)] >> ((i & 3u) * 8u)) & 0xffu;
}

@compute @workgroup_size(64)
fn around_update(@builtin(global_invocation_id) id: vec3u) {
    let c = id.x;
    let cells = ap.width * ap.height;
    if (c >= cells) { return; }
    let o = c * ${HEAD_WORDS}u;
    let counter = heads[o + ${HEAD_RNG}u] + ${hex(FREE_RNG.step)};
    heads[o + ${HEAD_RNG}u] = counter;
    let r = zil_mix32(counter);

    // The order of neighbours: drawn afresh for a whole run, kept by a free-running processor.
    var order = heads[o + ${HEAD_FACING}u] & 7u;
    if (ap.mode == 0u) {
        order = zil_mix32(r ^ ${hex(FREE_RNG.face)}) & 7u;
        zil_load_state(cells * ${HEAD_WORDS}u); // the all-zero block: a blank processor
    } else {
        zil_load_state(o);
    }
    idx_mode = 0u;
    idx_disp = 0u;
    idx_uses_mem = 0u;

    let cx = c % ap.width;
    let cy = c / ap.width;
    let start = order & 3u;
    let flip = (order & 4u) != 0u;
    for (var j = 0u; j < ap.gather; j++) {
        let d = (start + select(j, 4u - j, flip)) & 3u;
        // east, south, west, north
        let nx = (cx + ap.width + select(0u, 1u, d == 0u) - select(0u, 1u, d == 2u)) % ap.width;
        let ny = (cy + ap.height + select(0u, 1u, d == 1u) - select(0u, 1u, d == 3u)) % ap.height;
        let n = ny * ap.width + nx;
        for (var i = 0u; i < 16u; i++) { around_mem[j * 16u + i] = around_tape_byte(n, i); }
    }
    for (var i = 0u; i < 16u; i++) { around_mem[64u + i] = around_tape_byte(c, i); }
    around_written = 0u;

    if (ap.mode == 0u) {
        var steps = 0u;
        var reason = ${STOP.budget}u;
        loop {
            if (steps >= ap.budget) { break; }
            z80_step();
            steps++;
            if (zil_halt_reason() != ZIL_RUNNING) { reason = ${STOP.halt}u; break; }
            if (around_written == 0xffffu) { reason = ${STOP.full}u; break; }
        }
        heads[o + ${HEAD_AGE}u] = steps;
        heads[o + ${HEAD_WAITING}u] = reason;
        heads[o + ${HEAD_FACING}u] = order;
    } else {
        z80_step();
        let age = heads[o + ${HEAD_AGE}u] + 1u;
        // Which bytes of the cell this run has written so far.
        let written = heads[o + ${HEAD_WAITING}u] | around_written;
        heads[o + ${HEAD_WAITING}u] = written;
        if (zil_halt_reason() != ZIL_RUNNING || written == 0xffffu || age >= ap.budget) {
            for (var i = 0u; i < ${stateWords}u; i++) { heads[o + i] = 0u; }
            heads[o + ${HEAD_FACING}u] = zil_mix32(r ^ ${hex(FREE_RNG.face)}) & 7u;
            heads[o + ${HEAD_AGE}u] = 0u;
            heads[o + ${HEAD_WAITING}u] = 0u;
            heads[o + ${HEAD_RUNS}u] = heads[o + ${HEAD_RUNS}u] + 1u;
            heads[o + ${HEAD_LAST_RUN}u] = age;
        } else {
            zil_store_state(o);
            heads[o + ${HEAD_AGE}u] = age;
        }
    }

    if (ap.mutation != 0u && zil_mix32(r ^ ${hex(FREE_RNG.mutate)}) < ap.mutation) {
        around_mem[64u + (zil_mix32(r ^ ${hex(FREE_RNG.mutatePlace)}) & 15u)] =
            zil_mix32(r ^ ${hex(FREE_RNG.mutateValue)}) & 0xffu;
    }
    for (var w = 0u; w < 4u; w++) {
        let b = 64u + w * 4u;
        next_soup[c * 4u + w] = around_mem[b] | (around_mem[b + 1u] << 8u)
            | (around_mem[b + 2u] << 16u) | (around_mem[b + 3u] << 24u);
    }
}
`;
}
