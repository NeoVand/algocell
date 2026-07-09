// Substrate-v2 isolation study (DEV, headless): what actually breaks the
// isotropy wall — directional PERCEPTION (v2 rule) or the asymmetric SEED?
// MAP-Elites + Dice on the French flag, across the full rule×seed matrix.
//
//   npx tsx src/lib/morph/dev/v2mapelites.ts

import {
	type MorphParams,
	genomeLength,
	genomeLengthV2,
	asymmetricSeed,
	centerSeed,
	asciiGrid,
	liftV1toV2
} from '../ca';
import { makeFrenchFlag, arrow, letterF } from '../targets';
import { growthGenome } from '../genomes';
import { makeCpuScorer, makeCpuScorerV2 } from '../evolve';
import { MapElites, type MapElitesStats } from '../mapelites';
import { diceExact, diceBinary, softShape, distanceTransform } from '../fitness';

/** Evolve one (rule × seed) cell of the isolation matrix. */
async function evolve(
	rule: 'v1' | 'v2',
	seedKind: 'center' | 'asym',
	target: Uint8Array,
	fitnessFn: (grid: Uint8Array) => number,
	p: MorphParams,
	gens: number
): Promise<{ fitness: number; grid: Uint8Array; filled: number }> {
	const seed = seedKind === 'asym' ? asymmetricSeed(p) : centerSeed(p, 1);
	const isV2 = rule === 'v2';
	const me = new MapElites(
		{
			params: p,
			genomeLength: isV2 ? genomeLengthV2(p.S) : genomeLength(p.S),
			states: p.S,
			batchSize: 256,
			nichesCoverage: 24,
			nichesAsym: 24,
			mutationRate: 0.03,
			minMutationRate: 0.004,
			annealGens: gens,
			crossoverFrac: 0.15,
			seed: 7,
			seedGenomes: isV2 ? [liftV1toV2(growthGenome(p.S), p.S)] : [growthGenome(p.S)],
			fitnessFn
		},
		isV2 ? makeCpuScorerV2(p, seed) : makeCpuScorer(p, seed)
	);
	let last: MapElitesStats = { generation: 0, championFitness: 0, filled: 0, totalNiches: 0, championGrid: new Uint8Array(p.W * p.H) };
	for (let g = 0; g < gens; g++) last = await me.step();
	return { fitness: last.championFitness, grid: last.championGrid, filled: last.filled };
}

async function run() {
	const p: MorphParams = { W: 16, H: 16, S: 4, T: 14 };
	const GENS = 400;
	console.log('=== Isolation matrix: what breaks the isotropy wall — perception or seed? ===');
	console.log(`(MAP-Elites + Dice; ${GENS} gens; French flag, exact 3-color, mean per-class Dice)\n`);

	const flag = makeFrenchFlag(p, [1, 2, 3]);
	const flagFit = (grid: Uint8Array) => diceExact(grid, flag, p.S);
	console.log('target:\n' + asciiGrid(flag, p) + '\n');

	const cells: Record<string, { fitness: number; grid: Uint8Array; filled: number }> = {};
	for (const rule of ['v1', 'v2'] as const)
		for (const seed of ['center', 'asym'] as const) {
			const r = await evolve(rule, seed, flag, flagFit, p, GENS);
			cells[`${rule}/${seed}`] = r;
			console.log(`${rule} + ${seed} seed:  Dice ${r.fitness.toFixed(3)}  (archive ${r.filled})`);
		}

	const winner = Object.entries(cells).sort((a, b) => b[1].fitness - a[1].fitness)[0];
	console.log(`\nbest grown flag [${winner[0]}]:\n` + asciiGrid(winner[1].grid, p));

	// Symmetric seed isolates PERCEPTION; center→asym at v1 isolates the SEED.
	const dPerc = cells['v2/center'].fitness - cells['v1/center'].fitness;
	const dSeed = cells['v1/asym'].fitness - cells['v1/center'].fitness;
	console.log(`\nPerception effect (v2−v1 @ center seed): ${dPerc >= 0 ? '+' : ''}${dPerc.toFixed(3)}`);
	console.log(`Seed effect        (asym−center @ v1):    ${dSeed >= 0 ? '+' : ''}${dSeed.toFixed(3)}`);

	// The money test: the letter F silhouette (no symmetry at all).
	// Fitness = hard Dice (genuine overlap) blended with a small distance-transform
	// term for landscape smoothing — the DT is a *bonus*, not the metric, so it
	// can't be gamed by a compact blob.
	const F = letterF(p);
	const dtF = distanceTransform(F, p);
	const fFit = (g: Uint8Array) => 0.8 * diceBinary(g, F) + 0.2 * softShape(g, F, dtF);
	const FGENS = 800;
	console.log(`\n=== LETTER F silhouette (hard Dice + DT smoothing, ${FGENS} gens) — the asymmetry acid test ===`);
	console.log('target:\n' + asciiGrid(F, p));
	const Fv1 = await evolve('v1', 'asym', F, fFit, p, FGENS);
	const Fv2 = await evolve('v2', 'asym', F, fFit, p, FGENS);
	console.log(`v1 (isotropic): fit ${Fv1.fitness.toFixed(3)}  Dice ${diceBinary(Fv1.grid, F).toFixed(3)}\n` + asciiGrid(Fv1.grid, p));
	console.log(`v2 (directional): fit ${Fv2.fitness.toFixed(3)}  Dice ${diceBinary(Fv2.grid, F).toFixed(3)}\n` + asciiGrid(Fv2.grid, p));

	// Arrow sanity (hard Dice).
	const arr = arrow(p);
	const dtA = distanceTransform(arr, p);
	const av2 = await evolve('v2', 'asym', arr, (g) => 0.8 * diceBinary(g, arr) + 0.2 * softShape(g, arr, dtA), p, GENS);
	console.log(`\nArrow (v2 + asym): fit ${av2.fitness.toFixed(3)}  Dice ${diceBinary(av2.grid, arr).toFixed(3)}\n` + asciiGrid(av2.grid, p));

	console.log(
		`\nSUMMARY (hard Dice): letter-F  v1 ${diceBinary(Fv1.grid, F).toFixed(3)}  ->  v2 ${diceBinary(Fv2.grid, F).toFixed(3)}  |  arrow v2 ${diceBinary(av2.grid, arr).toFixed(3)}`
	);
}

run();
