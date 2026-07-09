// M1 evolution test (DEV, headless — run with `npx tsx`).
//
// Evolves a CA genome toward a target shape using the pure-TS reference as the
// scorer (deterministic, no GPU). Proves the GA + fitness converge: fitness must
// climb substantially from the random baseline. The GPU path scores identically
// (M0), so this transfers to the live demo.
//
//   npx tsx src/lib/morph/dev/m1.ts

import { type MorphParams, genomeLength, centerSeed, asciiGrid } from '../ca';
import { makeDiamond } from '../targets';
import { Evolver, makeCpuScorer, fitness } from '../evolve';

async function run() {
	const p: MorphParams = { W: 16, H: 16, S: 4, T: 14 };
	const seed = centerSeed(p, 1);
	const target = makeDiamond(p, 5, 1);
	const scorer = makeCpuScorer(p, seed);

	const evolver = new Evolver(
		{
			params: p,
			target,
			genomeLength: genomeLength(p.S),
			states: p.S,
			popSize: 400,
			eliteFrac: 0.08,
			mutationsPerChild: 2,
			fgWeight: 3,
			mode: 'binary',
			seed: 42
		},
		scorer
	);

	console.log('=== M1: evolve a CA genome toward a diamond (CPU scorer) ===');
	console.log(`grid ${p.W}x${p.H}, S=${p.S}, T=${p.T}, pop=${evolver.config.popSize}\n`);
	console.log('--- target ---\n' + asciiGrid(target, p) + '\n');

	const GENS = 150;
	const t0 = Date.now();
	let firstFitness = 0;
	for (let g = 0; g < GENS; g++) {
		const s = await evolver.step();
		if (g === 0) firstFitness = s.bestFitness;
		if (g % 25 === 0 || g === GENS - 1) {
			console.log(`gen ${String(g).padStart(3)} | best ${s.bestFitness.toFixed(4)} | mean ${s.meanFitness.toFixed(4)}`);
		}
	}
	const dt = Date.now() - t0;

	const best = evolver.best!;
	console.log(`\n--- best grown grid (fitness ${best.fitness.toFixed(4)}) ---\n` + asciiGrid(best.grid, p));
	console.log(`\ngen0 best: ${firstFitness.toFixed(4)}  final best: ${best.fitness.toFixed(4)}  (binary, fg-weighted)`);
	console.log(`plain cell accuracy: ${(fitness(best.grid, target, 'binary', 1) * 100).toFixed(1)}%`);
	console.log(`${GENS} generations in ${dt} ms (${(dt / GENS).toFixed(1)} ms/gen, CPU scorer)`);

	const improved = best.fitness >= firstFitness + 0.1;
	const good = best.fitness >= 0.85;
	if (!improved || !good) {
		console.error(`\nFAIL: expected clear learning (final>=0.85 and +0.10 over gen0). improved=${improved} good=${good}`);
		process.exit(1);
	}
	console.log('\nPASS: evolution measurably grows the genome toward the target.');
}

run();
