// MAP-Elites for morphogenesis genomes.
//
// Truncation selection collapses into strong attractors (all-background, the
// symmetric blob) — the cause of the French-flag plateau. MAP-Elites keeps ONE
// best genome per behavior niche, so a mediocre-but-DIFFERENT individual (e.g.
// the first faintly-asymmetric organism) survives as a stepping stone. Making
// *horizontal asymmetry* an explicit archive axis forces the search to populate
// the exact dimension the isotropy wall could not cross.
//
// It reuses evolve.ts's batch `ScoreFn` (genomes -> grids) unchanged: each
// generation samples parents from the archive, varies them, scores the whole
// batch in one dispatch, and re-inserts winners — so it drops onto the GPU with
// no throughput loss.

import type { MorphParams } from './ca';
import type { ScoreFn } from './evolve';
import { mulberry32 } from './genomes';
import { behaviorDescriptor } from './fitness';

export interface MapElitesConfig {
	params: MorphParams;
	genomeLength: number;
	states: number;
	/** children scored per generation (one dispatch). */
	batchSize?: number;
	/** archive resolution on the coverage axis. */
	nichesCoverage?: number;
	/** archive resolution on the horizontal-asymmetry axis. */
	nichesAsym?: number;
	/** initial fraction of genome entries mutated per child (annealed to min). */
	mutationRate?: number;
	minMutationRate?: number;
	annealGens?: number;
	/** fraction of children made by self-row crossover of two niches. */
	crossoverFrac?: number;
	seed?: number;
	seedGenomes?: Uint8Array[];
	/** grid -> objective in [0,1]. */
	fitnessFn: (grid: Uint8Array) => number;
}

interface Cell {
	genome: Uint8Array;
	grid: Uint8Array;
	fitness: number;
}

export interface MapElitesStats {
	generation: number;
	championFitness: number;
	filled: number;
	totalNiches: number;
	championGrid: Uint8Array;
}

export class MapElites {
	readonly cfg: Required<MapElitesConfig>;
	private score: ScoreFn;
	private rng: () => number;
	private rowStride: number;
	archive = new Map<number, Cell>();
	champion: Cell | null = null;
	generation = 0;

	constructor(cfg: MapElitesConfig, score: ScoreFn) {
		this.cfg = {
			batchSize: 512,
			nichesCoverage: 24,
			nichesAsym: 24,
			mutationRate: 0.03,
			minMutationRate: 0.004,
			annealGens: 250,
			crossoverFrac: 0.15,
			seed: 1,
			seedGenomes: [],
			...cfg
		};
		this.score = score;
		this.rng = mulberry32(this.cfg.seed);
		this.rowStride = this.cfg.genomeLength / this.cfg.states; // one "self"-row
	}

	private randInt(n: number): number {
		return Math.floor(this.rng() * n);
	}

	private nicheKey(grid: Uint8Array): number {
		const { coverage, hAsym } = behaviorDescriptor(grid, this.cfg.params);
		const nc = this.cfg.nichesCoverage;
		const na = this.cfg.nichesAsym;
		const cx = Math.min(nc - 1, Math.max(0, Math.floor(coverage * nc)));
		const cy = Math.min(na - 1, Math.max(0, Math.floor(((hAsym + 1) / 2) * na)));
		return cy * nc + cx;
	}

	private randomGenome(): Uint8Array {
		const g = new Uint8Array(this.cfg.genomeLength);
		for (let i = 0; i < g.length; i++) g[i] = this.randInt(this.cfg.states);
		return g;
	}

	private mutationCount(): number {
		const t = Math.min(1, this.generation / this.cfg.annealGens);
		const rate = this.cfg.mutationRate + (this.cfg.minMutationRate - this.cfg.mutationRate) * t;
		return Math.max(1, Math.round(rate * this.cfg.genomeLength));
	}

	private mutate(g: Uint8Array): Uint8Array {
		const child = g.slice();
		const k = this.mutationCount();
		for (let m = 0; m < k; m++) child[this.randInt(this.cfg.genomeLength)] = this.randInt(this.cfg.states);
		return child;
	}

	/** Swap one whole self-row (a coherent sub-behavior) from B into A, then mutate. */
	private crossover(a: Uint8Array, b: Uint8Array): Uint8Array {
		const child = a.slice();
		const row = this.randInt(this.cfg.states);
		const start = row * this.rowStride;
		child.set(b.subarray(start, start + this.rowStride), start);
		return this.mutate(child);
	}

	private insert(genome: Uint8Array, grid: Uint8Array, fitness: number): void {
		const key = this.nicheKey(grid);
		const cur = this.archive.get(key);
		if (!cur || fitness > cur.fitness) {
			const cell: Cell = { genome, grid, fitness };
			this.archive.set(key, cell);
			if (!this.champion || fitness > this.champion.fitness) this.champion = cell;
		}
	}

	private sampleCells(n: number): Cell[] {
		const cells = [...this.archive.values()];
		const out: Cell[] = [];
		for (let i = 0; i < n; i++) out.push(cells[this.randInt(cells.length)]);
		return out;
	}

	async step(): Promise<MapElitesStats> {
		const { batchSize, seedGenomes, crossoverFrac } = this.cfg;

		let batch: Uint8Array[];
		if (this.archive.size === 0) {
			// seed generation: warm-start genomes + random fill
			batch = [];
			for (const g of seedGenomes) {
				if (batch.length >= batchSize) break;
				if (g.length !== this.cfg.genomeLength) throw new Error(`seed genome length ${g.length} != ${this.cfg.genomeLength}`);
				batch.push(g.slice());
			}
			while (batch.length < batchSize) batch.push(this.randomGenome());
		} else {
			batch = [];
			for (let i = 0; i < batchSize; i++) {
				if (this.archive.size >= 2 && this.rng() < crossoverFrac) {
					const [a, b] = this.sampleCells(2);
					batch.push(this.crossover(a.genome, b.genome));
				} else {
					const [a] = this.sampleCells(1);
					batch.push(this.mutate(a.genome));
				}
			}
		}

		const grids = await this.score(batch);
		for (let i = 0; i < batch.length; i++) {
			this.insert(batch[i], grids[i], this.cfg.fitnessFn(grids[i]));
		}

		this.generation++;
		return {
			generation: this.generation,
			championFitness: this.champion?.fitness ?? 0,
			filled: this.archive.size,
			totalNiches: this.cfg.nichesCoverage * this.cfg.nichesAsym,
			championGrid: this.champion?.grid ?? new Uint8Array(this.cfg.params.W * this.cfg.params.H)
		};
	}
}
