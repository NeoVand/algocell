// M2 evolution test (DEV, headless — run with `npx tsx`).
//
// Exact-STATE (multi-color) evolution. Two cases:
//   1. bullseye  — concentric diamond colour bands; radially symmetric, so an
//      isotropic outer-totalistic rule can reach it. This must converge high.
//   2. flag      — three vertical colour bands; left/right asymmetric, which an
//      isotropic rule CANNOT produce from a centered seed. Shown to plateau,
//      documenting the wall (this case is reported, not asserted).
//
//   npx tsx src/lib/morph/dev/m2.ts

import { type MorphParams, genomeLength, centerSeed, asciiGrid } from '../ca';
import { makeBullseye, makeFrenchFlag } from '../targets';
import { Evolver, makeCpuScorer } from '../evolve';

async function evolveTarget(
	label: string,
	p: MorphParams,
	target: Uint8Array,
	gens: number
): Promise<number> {
	const seed = centerSeed(p, 1);
	const evolver = new Evolver(
		{
			params: p,
			target,
			genomeLength: genomeLength(p.S),
			states: p.S,
			popSize: 600,
			eliteFrac: 0.08,
			mutationsPerChild: 2,
			fgWeight: 2,
			mode: 'exact',
			seed: 7
		},
		makeCpuScorer(p, seed)
	);

	let first = 0;
	for (let g = 0; g < gens; g++) {
		const s = await evolver.step();
		if (g === 0) first = s.bestFitness;
	}
	const best = evolver.best!;
	console.log(`\n=== ${label} (exact multi-color) ===`);
	console.log('--- target ---\n' + asciiGrid(target, p));
	console.log(`--- best grown (fitness ${best.fitness.toFixed(4)}, gen0 ${first.toFixed(4)}) ---\n` + asciiGrid(best.grid, p));
	return best.fitness;
}

async function run() {
	const p: MorphParams = { W: 16, H: 16, S: 4, T: 14 };
	const GENS = 250;

	// Achievable: concentric bands (states 3,2,1 from the center outward).
	const bull = await evolveTarget('bullseye', p, makeBullseye(p, [3, 2, 1], 2), GENS);

	// Wall: left/right asymmetric bands — reported, not asserted.
	const flag = await evolveTarget('french flag', p, makeFrenchFlag(p, [1, 2, 3]), GENS);

	console.log(`\nbullseye best: ${bull.toFixed(4)}  |  flag best: ${flag.toFixed(4)} (expected to plateau — isotropy wall)`);
	if (bull < 0.9) {
		console.error(`\nFAIL: bullseye should reach exact-color >= 0.90 (got ${bull.toFixed(4)})`);
		process.exit(1);
	}
	console.log('\nPASS: exact multi-color evolution works for radially-symmetric targets.');
	console.log('NOTE: the flag plateau is expected and documents the isotropy limit (see MORPHOGENESIS notes).');
}

run();
