// Evolution of CA genomes toward a target shape.
//
// The bootstrap (Z80 code) is fixed; only the genome table evolves. A truncation
// -selection GA maps perfectly onto batched scoring: one generation = score the
// whole population in a single dispatch, keep the elites, refill by mutation.
// The scorer is pluggable — the headless test scores with the pure-TS reference
// (`makeCpuScorer`), the browser scores on the GPU (`MorphEngine.grow`) — and
// the two are byte-identical (proven in M0), so results transfer.

import { type MorphParams, runCA, runCAv2 } from './ca';
import { mulberry32 } from './genomes';

/** Grows a batch of genomes to their final grids. May be sync (CPU) or async (GPU). */
export type ScoreFn = (genomes: Uint8Array[]) => Promise<Uint8Array[]> | Uint8Array[];

export type FitnessMode = 'binary' | 'exact';

export interface EvolveConfig {
	params: MorphParams;
	/** Target grid (W*H) of states; 0 = background. */
	target: Uint8Array;
	/** Genome length (S*K). */
	genomeLength: number;
	/** Number of states (mutation draws in 0..S-1). */
	states: number;
	popSize?: number;
	/** Fraction of the population kept as elites each generation. */
	eliteFrac?: number;
	/** Table entries randomized per child. */
	mutationsPerChild?: number;
	/** Extra weight on foreground (non-zero target) cells, to beat the "all background" local optimum. */
	fgWeight?: number;
	/** 'binary' = alive/dead match (ignores color); 'exact' = state must match. */
	mode?: FitnessMode;
	seed?: number;
	/** Warm-start genomes seeded into the initial population before random fill. */
	seedGenomes?: Uint8Array[];
}

export interface GenStats {
	generation: number;
	bestFitness: number;
	meanFitness: number;
	bestGenome: Uint8Array;
	bestGrid: Uint8Array;
}

/** Fitness in [0,1]: foreground-weighted fraction of correctly-matched cells. */
export function fitness(
	grid: Uint8Array,
	target: Uint8Array,
	mode: FitnessMode,
	fgWeight: number
): number {
	let num = 0;
	let den = 0;
	for (let i = 0; i < target.length; i++) {
		const w = target[i] ? fgWeight : 1;
		const hit = mode === 'binary' ? grid[i] > 0 === target[i] > 0 : grid[i] === target[i];
		if (hit) num += w;
		den += w;
	}
	return num / den;
}

/** A scorer backed by the pure-TS reference CA (deterministic, no GPU). */
export function makeCpuScorer(p: MorphParams, seed: Uint8Array): ScoreFn {
	return (genomes) => genomes.map((g) => runCA(seed, g, p));
}

/** A scorer backed by the v2 (SUM×DIR16) reference CA. */
export function makeCpuScorerV2(p: MorphParams, seed: Uint8Array): ScoreFn {
	return (genomes) => genomes.map((g) => runCAv2(seed, g, p));
}

export class Evolver {
	readonly config: Required<EvolveConfig>;
	private score: ScoreFn;
	private rng: () => number;
	private population: Uint8Array[];
	generation = 0;
	best: { genome: Uint8Array; grid: Uint8Array; fitness: number } | null = null;

	constructor(config: EvolveConfig, score: ScoreFn) {
		this.config = {
			popSize: 256,
			eliteFrac: 0.1,
			mutationsPerChild: 2,
			fgWeight: 3,
			mode: 'binary',
			seed: 1,
			seedGenomes: [],
			...config
		};
		this.score = score;
		this.rng = mulberry32(this.config.seed);
		this.population = this.initialPopulation();
	}

	private randInt(n: number): number {
		return Math.floor(this.rng() * n);
	}

	private randomGenome(): Uint8Array {
		const g = new Uint8Array(this.config.genomeLength);
		for (let i = 0; i < g.length; i++) g[i] = this.randInt(this.config.states);
		return g;
	}

	/** Initial population: warm-start genomes first (truncated/padded to popSize), then random fill. */
	private initialPopulation(): Uint8Array[] {
		const pop: Uint8Array[] = [];
		for (const g of this.config.seedGenomes) {
			if (pop.length >= this.config.popSize) break;
			if (g.length !== this.config.genomeLength) {
				throw new Error(`seed genome length ${g.length} != ${this.config.genomeLength}`);
			}
			pop.push(g.slice());
		}
		while (pop.length < this.config.popSize) pop.push(this.randomGenome());
		return pop;
	}

	/** Run one generation; returns its statistics (and updates `this.best`). */
	async step(): Promise<GenStats> {
		const { target, mode, fgWeight, eliteFrac, popSize, mutationsPerChild, genomeLength, states } =
			this.config;

		const grids = await this.score(this.population);
		const scored = this.population.map((genome, i) => ({
			genome,
			grid: grids[i],
			f: fitness(grids[i], target, mode, fgWeight)
		}));
		scored.sort((a, b) => b.f - a.f);

		const top = scored[0];
		if (!this.best || top.f > this.best.fitness) {
			this.best = { genome: top.genome.slice(), grid: top.grid.slice(), fitness: top.f };
		}
		const meanFitness = scored.reduce((s, x) => s + x.f, 0) / scored.length;

		// Next generation: elitism + mutated offspring of elites.
		const eliteCount = Math.max(1, Math.floor(popSize * eliteFrac));
		const elites = scored.slice(0, eliteCount).map((x) => x.genome);
		const next: Uint8Array[] = elites.map((g) => g.slice());
		while (next.length < popSize) {
			const child = elites[this.randInt(eliteCount)].slice();
			for (let m = 0; m < mutationsPerChild; m++) child[this.randInt(genomeLength)] = this.randInt(states);
			next.push(child);
		}
		this.population = next;

		const stats: GenStats = {
			generation: this.generation,
			bestFitness: top.f,
			meanFitness,
			bestGenome: top.genome.slice(),
			bestGrid: top.grid.slice()
		};
		this.generation++;
		return stats;
	}
}
