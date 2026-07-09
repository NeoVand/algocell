// Substrate-v2 Step B (DEV, headless) — does directional perception break the
// isotropy wall? Evolve ASYMMETRIC targets with v2 (SUM×DIR16) vs v1 (isotropic)
// as a control, with the SAME asymmetric seed so the only difference is the rule.
//
//   npx tsx src/lib/morph/dev/v2evolve.ts

import {
	type MorphParams,
	genomeLength,
	genomeLengthV2,
	asymmetricSeed,
	asciiGrid,
	liftV1toV2
} from '../ca';
import { makeFrenchFlag, arrow } from '../targets';
import { growthGenome } from '../genomes';
import { Evolver, makeCpuScorer, makeCpuScorerV2, type FitnessMode } from '../evolve';

async function evolve(
	kind: 'v1' | 'v2',
	target: Uint8Array,
	mode: FitnessMode,
	p: MorphParams,
	gens: number
): Promise<{ fitness: number; first: number; grid: Uint8Array }> {
	const seed = asymmetricSeed(p);
	const isV2 = kind === 'v2';
	const ev = new Evolver(
		{
			params: p,
			target,
			genomeLength: isV2 ? genomeLengthV2(p.S) : genomeLength(p.S),
			states: p.S,
			popSize: 300,
			eliteFrac: 0.08,
			mutationsPerChild: isV2 ? 12 : 2, // ~rate-matched to the 16x bigger genome
			fgWeight: mode === 'exact' ? 2 : 3,
			mode,
			seed: 7,
			seedGenomes: isV2 ? [liftV1toV2(growthGenome(p.S), p.S)] : [growthGenome(p.S)]
		},
		isV2 ? makeCpuScorerV2(p, seed) : makeCpuScorer(p, seed)
	);
	let first = 0;
	for (let g = 0; g < gens; g++) {
		const s = await ev.step();
		if (g === 0) first = s.bestFitness;
	}
	const best = ev.best!;
	return { fitness: best.fitness, first, grid: best.grid };
}

async function run() {
	const p: MorphParams = { W: 16, H: 16, S: 4, T: 14 };
	const GENS = 150;
	console.log('=== Step B: does directional perception break the isotropy wall? ===');
	console.log(`(same asymmetric seed for both rules; only the perception differs; ${GENS} gens)\n`);

	console.log('--- ARROW (binary, left/right asymmetric) ---');
	console.log('target:\n' + asciiGrid(arrow(p), p));
	const av1 = await evolve('v1', arrow(p), 'binary', p, GENS);
	const av2 = await evolve('v2', arrow(p), 'binary', p, GENS);
	console.log(`v1 (isotropic): ${av1.fitness.toFixed(3)}   v2 (directional): ${av2.fitness.toFixed(3)}`);
	console.log('v2 arrow grown:\n' + asciiGrid(av2.grid, p));

	console.log('\n--- FRENCH FLAG (exact 3-color — the documented ~0.60 wall) ---');
	const fv1 = await evolve('v1', makeFrenchFlag(p), 'exact', p, GENS);
	const fv2 = await evolve('v2', makeFrenchFlag(p), 'exact', p, GENS);
	console.log(`v1 (isotropic): ${fv1.fitness.toFixed(3)}   v2 (directional): ${fv2.fitness.toFixed(3)}`);
	console.log('v2 flag grown:\n' + asciiGrid(fv2.grid, p));

	console.log(
		`\nSUMMARY  arrow: v1 ${av1.fitness.toFixed(3)} -> v2 ${av2.fitness.toFixed(3)}   |   flag: v1 ${fv1.fitness.toFixed(3)} -> v2 ${fv2.fitness.toFixed(3)}`
	);
	// Diagnostic only (plain truncation GA + accuracy fitness): v2 clears the ~0.60
	// wall but plain truncation under-exploits the bigger genome. See v2mapelites.ts
	// (MAP-Elites + Dice) for the proper break — v2 grows a recognizable F there.
	console.log(
		`\nNOTE: plain truncation shows v2 >= v1 on the flag but doesn't fully break through; the MAP-Elites + Dice run (v2mapelites.ts) grows a real asymmetric F.`
	);
}

run();
