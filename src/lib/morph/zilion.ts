// MorphEngine — runs the CA on Zilion's GPU Z80 core.
//
// One Zilion "program" is one full tape (code + genome + grid), so a *batch* of
// genomes is one GPU dispatch: build a tape per genome (same fixed bootstrap and
// seed, differing genome bytes), run them all at once, read each grid back. This
// is the primitive the evolution loop will call to score a whole population per
// dispatch. Correctness of the bootstrap itself is proven headlessly against a
// real Z80 in `dev/m0.ts`; Zilion's GPU core is conformance-tested against that
// same emulator, so this path inherits that guarantee.

import { Zilion } from '@neovand/zilion';
import { assembleBootstrap } from './bootstrap';
import {
	type MorphParams,
	type MemoryMap,
	computeMemoryMap,
	buildTape,
	readFront,
	centerSeed
} from './ca';

/**
 * Instruction budget for one run. HALT stops each lane early, but the budget
 * must exceed the true count or the program won't finish. ~64 instr/cell
 * (measured), plus the per-step LDIR copy and loop overhead, times T, with a
 * safety margin. Kept reasonably tight so idle halted lanes don't spin.
 */
export function stepsFor(p: MorphParams, margin = 1.5): number {
	const interior = (p.W - 2) * (p.H - 2);
	const perStep = interior * 64 + p.W * p.H + (p.H - 2) * 4 + 16;
	return Math.ceil((p.T * perStep + 32) * margin);
}

export interface MorphEngine {
	readonly params: MorphParams;
	readonly map: MemoryMap;
	readonly code: Uint8Array;
	readonly seed: Uint8Array;
	readonly steps: number;
	readonly memBytes: number;
	/** Grow a batch of genomes to their final grids (one GPU dispatch). */
	grow(genomes: Uint8Array[]): Promise<Uint8Array[]>;
	destroy(): void;
}

export interface MorphEngineOptions {
	/** Share an existing device (e.g. `engine.gpuDevice`); omit to request one. */
	device?: GPUDevice;
	/** Override the initial grid (defaults to a single center seed of state 1). */
	seed?: Uint8Array;
	/** Override the instruction budget. */
	steps?: number;
}

export async function createMorphEngine(
	p: MorphParams,
	opts: MorphEngineOptions = {}
): Promise<MorphEngine> {
	const map = computeMemoryMap(p);
	const code = assembleBootstrap(p, map);
	const seed = opts.seed ?? centerSeed(p, 1);
	if (seed.length !== p.W * p.H) throw new Error(`seed length ${seed.length} != ${p.W * p.H}`);
	const steps = opts.steps ?? stepsFor(p);
	const zilion = await Zilion.create({ memBytes: map.memBytes, device: opts.device });

	return {
		params: p,
		map,
		code,
		seed,
		steps,
		memBytes: map.memBytes,
		async grow(genomes: Uint8Array[]): Promise<Uint8Array[]> {
			if (genomes.length === 0) return [];
			const tapes = genomes.map((g) => buildTape(code, g, seed, map, p));
			const res = await zilion.run(tapes, { steps });
			return genomes.map((_, i) => readFront(res.memoryOf(i), map, p));
		},
		destroy() {
			zilion.destroy();
		}
	};
}
