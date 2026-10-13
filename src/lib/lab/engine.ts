import {
	createEngine,
	type Engine,
	type Machine,
	type Batch,
	type MachineResult,
	type ShaderManifest
} from '@neovand/zilion';
import { createInitialSoup, validateSettings, validateInitial, machineSpec } from '../gpu/machine';
export { createInitialSoup, validateSettings, machineSpec } from '../gpu/machine';
import { ablationMasks } from './ablation';
import {
	DEFAULT_SETTINGS,
	type LabSettings,
	type InitialCondition,
	type LabMetrics,
	type LabSnapshot,
	type EncounterObservation,
	type CultureResult,
	type CultureGeneration,
	type MarkerResult
} from './types';

export { DEFAULT_SETTINGS } from './types';

const SQUARE_DIRECTIONS = [
	[1, 0],
	[0, 1],
	[-1, 0],
	[0, -1]
];
const HEX_DIRECTIONS = [
	[
		[1, 0],
		[-1, 0],
		[0, -1],
		[-1, -1],
		[0, 1],
		[-1, 1]
	],
	[
		[1, 0],
		[-1, 0],
		[1, -1],
		[0, -1],
		[1, 1],
		[0, 1]
	]
];

/** PCG permutation over a 32-bit LCG, matching the published host arithmetic. */
class Pcg {
	constructor(public state: number) {
		this.state >>>= 0;
	}
	next(): number {
		const s = this.state;
		this.state = (Math.imul(s, 747796405) + 2891336453) >>> 0;
		const word = Math.imul((s >>> ((s >>> 28) + 4)) ^ s, 277803737) >>> 0;
		return ((word >>> 22) ^ word) >>> 0;
	}
	bounded(n: number): number {
		return this.next() % n;
	}
}

function initialIdentity(initial: InitialCondition): string {
	if (initial.kind === 'random') return 'random';
	if (initial.kind === 'tiled') return JSON.stringify(['tiled', initial.bytes]);
	return JSON.stringify([
		'competition',
		initial.regenerator,
		initial.transmitterCore,
		initial.share,
		initial.randomPayload
	]);
}

export function cyclicSimilarity(a: ArrayLike<number>, b: ArrayLike<number>): number {
	if (!a.length || a.length !== b.length)
		throw new Error('Similarity needs two tapes of equal, positive length.');
	let best = 0;
	for (let shift = 0; shift < a.length; shift++) {
		let matches = 0;
		for (let k = 0; k < a.length; k++) if (a[k] === b[(k + shift) % a.length]) matches++;
		best = Math.max(best, matches);
	}
	return best / a.length;
}

function same(a: ArrayLike<number>, b: ArrayLike<number>): boolean {
	if (a.length !== b.length) return false;
	for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
	return true;
}

export function populationMetrics(
	soup: Uint8Array,
	settings: LabSettings,
	initial: InitialCondition = { kind: 'random' }
): LabMetrics {
	const counts = new Map<string, number>();
	let zeros = 0,
		topCount = 0,
		topKey = '',
		rCount = 0,
		tCount = 0,
		coreCount = 0;
	const L = settings.length,
		N = settings.width * settings.height;
	if (soup.length !== N * L)
		throw new Error('Population bytes do not match the declared dimensions.');
	const rOffsets =
		L === 32
			? [0x04, 0x44, 0x84, 0x08, 0x48, 0x88, 0xc8, 0x50, 0x90]
			: [0x04, 0x24, 0x44, 0x64, 0x84, 0xa4, 0x08, 0x48, 0x68, 0x88, 0xa8, 0xc8, 0xe8];
	const tOffsets = L === 32 ? [0x60, 0xa0] : [0x50, 0x90, 0xb0];
	for (let cell = 0; cell < N; cell++) {
		const tape = soup.subarray(cell * L, (cell + 1) * L);
		let key = '';
		for (const value of tape) {
			key += String.fromCharCode(value);
			if (value === 0) zeros++;
		}
		const count = (counts.get(key) ?? 0) + 1;
		counts.set(key, count);
		if (count > topCount) {
			topCount = count;
			topKey = key;
		}
		if (
			initial.kind === 'competition' &&
			settings.isa === 'z80' &&
			(L === 16 || L === 32) &&
			tape[1] === 0x5e &&
			tape[2] === 0xed &&
			tape[3] === 0xb0
		) {
			coreCount++;
			if (rOffsets.includes(tape[0])) rCount++;
			else if (tOffsets.includes(tape[0])) tCount++;
		}
	}
	return {
		update: 0,
		encounters: 0,
		executedSteps: 0,
		zeroFraction: zeros / soup.length,
		topShare: topCount / N,
		diversity: counts.size / N,
		uniqueTapes: counts.size,
		topTape: Array.from(topKey, (c) => c.charCodeAt(0)),
		topCount,
		activePairs: 0,
		mutationAttempts: 0,
		regeneratorShare: rCount + tCount ? rCount / (rCount + tCount) : null,
		transmitterShare: rCount + tCount ? tCount / (rCount + tCount) : null,
		coreShare: initial.kind === 'competition' ? coreCount / N : null
	};
}

function observation(result: MachineResult, before: Uint8Array, lane = 0): EncounterObservation {
	return {
		before: Array.from(before),
		after: Array.from(result.memoryOf(lane)),
		fetched: result.fetchedOf(lane),
		starts: result.startsOf(lane),
		state: result.stateOf(lane),
		writes: result.writesOf(lane),
		steps: result.steps[lane],
		halt: result.haltReasonOf(lane),
		trap: result.trapCauseOf(lane)
	};
}

/** Population orchestration is deliberately independent of the instruction-set implementation. */
export class LabEngine {
	readonly soup: Uint8Array;
	readonly fetchStatus: Uint8Array;
	readonly settings: LabSettings;
	readonly initialCondition: InitialCondition;
	readonly manifest: ShaderManifest;
	metrics: LabMetrics;
	update = 0;
	busy = false;
	private rng: Pcg;
	private encounters = 0;
	private executedSteps = 0;
	private dead = false;
	private deviceProblem = '';
	private pending: Promise<unknown> = Promise.resolve();
	private pendingOperations = 0;
	private readonly masks: Uint32Array;
	private readonly capacity: number;
	private readonly batch: Batch;
	private readonly claims: Uint8Array;
	private readonly pairs: Uint32Array;
	private readonly lanes: Uint8Array;

	private constructor(
		private engine: Engine,
		private machine: Machine,
		settings: LabSettings,
		initial: InitialCondition
	) {
		this.settings = structuredClone(settings);
		this.initialCondition = structuredClone(initial);
		Object.freeze(this.settings.ablation);
		Object.freeze(this.settings);
		for (const value of Object.values(this.initialCondition))
			if (Array.isArray(value)) Object.freeze(value);
		Object.freeze(this.initialCondition);
		this.soup = createInitialSoup(settings, initial);
		this.fetchStatus = new Uint8Array(settings.width * settings.height);
		this.claims = new Uint8Array(this.fetchStatus.length);
		this.capacity = Math.min(settings.pairAttempts, Math.floor(this.fetchStatus.length / 2));
		this.pairs = new Uint32Array(this.capacity * 2);
		this.lanes = new Uint8Array(this.capacity * settings.length * 2);
		this.batch = machine.batch(this.capacity);
		this.manifest = machine.manifest;
		this.masks = ablationMasks(settings.isa, settings.ablation);
		this.rng = new Pcg(settings.seed);
		this.rng.next();
		this.metrics = populationMetrics(this.soup, settings, initial);
		void engine.device.lost.then((info) => {
			if (!this.dead)
				this.deviceProblem = `WebGPU device lost: ${info.message || info.reason}. Save your current population and restart the laboratory.`;
		});
	}

	static async create(
		settings: LabSettings = DEFAULT_SETTINGS,
		initial: InitialCondition = { kind: 'random' }
	): Promise<LabEngine> {
		validateSettings(settings);
		validateInitial(initial);
		const engine = await createEngine();
		try {
			const machine = await engine.machine(machineSpec(settings));
			return new LabEngine(engine, machine, settings, initial);
		} catch (error) {
			engine.destroy();
			throw error;
		}
	}

	private enqueue<T>(operation: () => Promise<T>): Promise<T> {
		if (this.dead) return Promise.reject(new Error('This laboratory has been closed.'));
		this.pendingOperations++;
		this.busy = true;
		const task = this.pending.then(async () => {
			try {
				if (this.dead) throw new Error('This laboratory has been closed.');
				if (this.deviceProblem) throw new Error(this.deviceProblem);
				return await operation();
			} finally {
				this.pendingOperations--;
				this.busy = this.pendingOperations > 0;
			}
		});
		this.pending = task.catch(() => {});
		return task;
	}

	private neighbor(cell: number): number {
		const { width: W, height: H, topology } = this.settings;
		const x = cell % W,
			y = Math.floor(cell / W);
		const [dx, dy] = (topology === 'hex' ? HEX_DIRECTIONS[y & 1] : SQUARE_DIRECTIONS)[
			this.rng.bounded(topology === 'hex' ? 6 : 4)
		];
		const nx = x + dx < 0 || x + dx >= W ? x - dx : x + dx;
		const ny = y + dy < 0 || y + dy >= H ? y - dy : y + dy;
		return Math.max(0, Math.min(H - 1, ny)) * W + Math.max(0, Math.min(W - 1, nx));
	}

	step(): Promise<LabMetrics> {
		return this.enqueue(async () => {
			const { length: L, pairAttempts, mutationExponent } = this.settings;
			this.claims.fill(0);
			let accepted = 0;
			for (let draw = 0; draw < pairAttempts; draw++) {
				const a = this.rng.bounded(this.claims.length),
					b = this.neighbor(a);
				if (a === b || this.claims[a] || this.claims[b]) continue;
				this.claims[a] = this.claims[b] = 1;
				this.pairs[accepted * 2] = a;
				this.pairs[accepted * 2 + 1] = b;
				this.lanes.set(this.soup.subarray(a * L, (a + 1) * L), accepted * L * 2);
				this.lanes.set(this.soup.subarray(b * L, (b + 1) * L), accepted * L * 2 + L);
				accepted++;
			}
			const seed = this.rng.next();
			if (accepted) {
				const input = this.lanes.subarray(0, accepted * L * 2);
				this.batch.upload(input);
				const result = await this.batch.run({
					lanes: accepted,
					seed,
					budget: this.settings.steps,
					ablation: this.masks
				});
				for (let k = 0; k < accepted; k++) {
					const a = this.pairs[k * 2],
						b = this.pairs[k * 2 + 1],
						memory = result.memoryOf(k);
					const unchangedA = same(input.subarray(k * L * 2, k * L * 2 + L), memory.subarray(0, L));
					const unchangedB = same(
						input.subarray(k * L * 2 + L, (k + 1) * L * 2),
						memory.subarray(L)
					);
					this.soup.set(memory.subarray(0, L), a * L);
					this.soup.set(memory.subarray(L), b * L);
					this.fetchStatus[a] = unchangedA ? (result.fetchedOf(k).some((p) => p >= L) ? 2 : 1) : 0;
					if (!unchangedB) this.fetchStatus[b] = 0;
					this.executedSteps += result.steps[k];
				}
			}
			const mutations =
				mutationExponent >= 31 ? 0 : Math.floor(pairAttempts / 2 ** mutationExponent);
			for (let n = 0; n < mutations; n++) {
				const at = this.rng.bounded(this.soup.length),
					value = this.rng.next() & 255;
				if (this.soup[at] !== value) this.fetchStatus[Math.floor(at / L)] = 0;
				this.soup[at] = value;
			}
			this.update++;
			this.encounters += accepted;
			this.metrics = {
				...populationMetrics(this.soup, this.settings, this.initialCondition),
				update: this.update,
				encounters: this.encounters,
				executedSteps: this.executedSteps,
				activePairs: accepted,
				mutationAttempts: mutations
			};
			return this.metrics;
		});
	}

	encounter(
		tape: Uint8Array | number[],
		partner?: Uint8Array | number[],
		seed = 42
	): Promise<EncounterObservation> {
		return this.enqueue(async () => {
			this.checkTape(tape);
			if (partner) this.checkTape(partner);
			if (!Number.isInteger(seed) || seed < 0 || seed > 0xffffffff)
				throw new Error('Encounter seed must be an unsigned 32-bit integer.');
			const rng = new Pcg(seed),
				lane = new Uint8Array(this.settings.length * 2);
			lane.set(tape);
			for (let i = this.settings.length; i < lane.length; i++)
				lane[i] = partner ? partner[i - this.settings.length] : rng.next() & 255;
			return observation(await this.machine.run(lane, { seed, ablation: this.masks }), lane);
		});
	}

	private checkTape(tape: ArrayLike<number>): void {
		if (
			tape.length !== this.settings.length ||
			Array.from(tape).some((b) => !Number.isInteger(b) || b < 0 || b > 255)
		)
			throw new Error(`A tape must contain exactly ${this.settings.length} bytes.`);
	}

	assay(
		tape: Uint8Array | number[],
		options: { trials?: number; seed?: number; generations?: number } = {}
	): Promise<CultureResult> {
		return this.enqueue(async () => (await this.culture(tape, options)).result);
	}

	private async culture(
		tape: Uint8Array | number[],
		options: { trials?: number; seed?: number; generations?: number },
		marker?: { position: number; value: number }
	): Promise<{ result: CultureResult; retention: number[] }> {
		this.checkTape(tape);
		const trials = options.trials ?? 64,
			generations = options.generations ?? 2,
			seed = options.seed ?? 4242;
		if (
			!Number.isInteger(trials) ||
			trials < 1 ||
			trials > 256 ||
			!Number.isInteger(generations) ||
			generations < 1 ||
			generations > 8
		)
			throw new Error('Culture supports 1–256 partners and 1–8 generations.');
		if (!Number.isInteger(seed) || seed < 0 || seed > 0xffffffff)
			throw new Error('Culture seed must be an unsigned 32-bit integer.');
		const L = this.settings.length,
			rng = new Pcg(seed);
		let parents = new Uint8Array(trials * L);
		for (let k = 0; k < trials; k++) parents.set(tape, k * L);
		const summary: CultureGeneration[] = [],
			retention: number[] = [];
		let example: EncounterObservation | undefined;
		for (let generation = 1; generation <= generations; generation++) {
			const input = new Uint8Array(trials * L * 2);
			for (let k = 0; k < trials; k++) {
				input.set(parents.subarray(k * L, (k + 1) * L), k * L * 2);
				for (let b = 0; b < L; b++) input[k * L * 2 + L + b] = rng.next() & 255;
			}
			const result = await this.machine.run(input, { seed: rng.next(), ablation: this.masks });
			if (!example) example = observation(result, input.subarray(0, L * 2));
			const offspring = new Uint8Array(trials * L);
			let gain = 0,
				informative = 0,
				exact = 0,
				cyclic = 0,
				copies = 0,
				similarity = 0,
				entries = 0,
				retained = 0;
			for (let k = 0; k < trials; k++) {
				const child = result.memoryOf(k).subarray(L),
					before = cyclicSimilarity(tape, input.subarray(k * L * 2 + L, (k + 1) * L * 2)),
					after = cyclicSimilarity(tape, child);
				offspring.set(child, k * L);
				if (before < 0.75) {
					gain += (after - before) / (1 - before);
					informative++;
				}
				if (same(tape, child)) exact++;
				if (after === 1) cyclic++;
				if (after >= 0.75) copies++;
				similarity += after;
				if (result.fetchedOf(k).some((p) => p >= L)) entries++;
				if (marker && child[marker.position] === marker.value) retained++;
			}
			summary.push({
				generation,
				gain: informative ? gain / informative : null,
				informativePartners: informative,
				exactCopies: exact,
				cyclicCopies: cyclic,
				copyFraction: copies / trials,
				meanSimilarity: similarity / trials,
				partnerEntries: entries
			});
			retention.push(retained / trials);
			parents = offspring;
		}
		const gain = summary[1]?.gain ?? null;
		return {
			result: {
				parent: Array.from(tape),
				trials,
				seed,
				generations: summary,
				gain,
				positive: gain === null ? null : gain >= 0.3,
				confinement:
					summary[0].partnerEntries === 0
						? 'confined'
						: summary[0].partnerEntries >= trials / 2
							? 'open'
							: 'intermediate',
				example: example!,
				manifest: this.manifest,
				settings: structuredClone(this.settings)
			},
			retention
		};
	}

	marker(
		tape: Uint8Array | number[],
		options: {
			position: number;
			value: number;
			trials?: number;
			seed?: number;
			generations?: number;
		}
	): Promise<MarkerResult> {
		return this.enqueue(async () => {
			this.checkTape(tape);
			if (
				!Number.isInteger(options.position) ||
				options.position < 0 ||
				options.position >= tape.length ||
				!Number.isInteger(options.value) ||
				options.value < 0 ||
				options.value > 255
			)
				throw new Error('Marker position or byte value is out of range.');
			if (tape[options.position] === options.value)
				throw new Error('Choose a marker value different from the parental byte.');
			const mutant = Uint8Array.from(tape);
			mutant[options.position] = options.value;
			const configuration = {
				trials: options.trials ?? 16,
				seed: options.seed ?? 4242,
				generations: options.generations ?? 4
			};
			const wild = await this.culture(tape, configuration, options),
				marked = await this.culture(mutant, configuration, options);
			return {
				position: options.position,
				value: options.value,
				wildType: wild.result,
				mutant: marked.result,
				retention: wild.retention.map((value, index) => ({
					generation: index + 1,
					mutant: marked.retention[index],
					wildType: value
				})),
				note: 'Same-position byte-value retention in serial transfers, with matched fresh partners and a wild-type control. Values can be regenerated or reacquired; this is not causal ancestry or the registered branching assay.'
			};
		});
	}

	/** Planting is an intervention; every changed tape loses its previous fetch observation. */
	plant(cell: number, tape: Uint8Array | number[]): void {
		if (this.busy || this.dead)
			throw new Error('Pause and wait for the current operation before planting.');
		this.checkTape(tape);
		if (!Number.isInteger(cell) || cell < 0 || cell >= this.fetchStatus.length)
			throw new Error('Cell index is out of range.');
		this.soup.set(tape, cell * this.settings.length);
		this.fetchStatus[cell] = 0;
		this.metrics = {
			...populationMetrics(this.soup, this.settings, this.initialCondition),
			update: this.update,
			encounters: this.encounters,
			executedSteps: this.executedSteps
		};
	}

	snapshot(): LabSnapshot {
		if (this.busy || this.dead)
			throw new Error('Pause and wait for the current operation before saving.');
		return {
			schema: 'algocell.lab/1',
			scheduler: 'serial-reflected-pcg/1',
			settings: structuredClone(this.settings),
			initialCondition: structuredClone(this.initialCondition),
			update: this.update,
			encounters: this.encounters,
			executedSteps: this.executedSteps,
			rngState: this.rng.state,
			bytes: Array.from(this.soup),
			manifest: this.manifest,
			createdAt: new Date().toISOString()
		};
	}

	restore(snapshot: LabSnapshot): void {
		if (this.busy || this.dead) throw new Error('Wait for the current operation before restoring.');
		if (snapshot.schema !== 'algocell.lab/1' || snapshot.scheduler !== 'serial-reflected-pcg/1')
			throw new Error('Unsupported checkpoint format or population scheduler.');
		if (
			Object.keys(DEFAULT_SETTINGS).some(
				(key) =>
					JSON.stringify(snapshot.settings[key as keyof LabSettings]) !==
					JSON.stringify(this.settings[key as keyof LabSettings])
			)
		)
			throw new Error('Create a laboratory with the checkpoint settings before restoring.');
		validateInitial(snapshot.initialCondition);
		// The initial design determines exported provenance and the competition classifier.
		if (initialIdentity(snapshot.initialCondition) !== initialIdentity(this.initialCondition))
			throw new Error(
				'Create a laboratory with the checkpoint initial condition before restoring.'
			);
		if (
			snapshot.bytes.length !== this.soup.length ||
			snapshot.bytes.some((b) => !Number.isInteger(b) || b < 0 || b > 255)
		)
			throw new Error('Checkpoint population bytes are invalid.');
		for (const value of [
			snapshot.update,
			snapshot.encounters,
			snapshot.executedSteps,
			snapshot.rngState
		])
			if (!Number.isSafeInteger(value) || value < 0)
				throw new Error('Checkpoint counters are invalid.');
		if (snapshot.rngState > 0xffffffff) throw new Error('Checkpoint random state is invalid.');
		if (snapshot.manifest.sha256.wgsl !== this.manifest.sha256.wgsl)
			throw new Error('The checkpoint machine identity differs from this engine build.');
		this.soup.set(snapshot.bytes);
		this.update = snapshot.update;
		this.encounters = snapshot.encounters;
		this.executedSteps = snapshot.executedSteps;
		this.rng.state = snapshot.rngState;
		this.fetchStatus.fill(0);
		this.metrics = {
			...populationMetrics(this.soup, this.settings, this.initialCondition),
			update: this.update,
			encounters: this.encounters,
			executedSteps: this.executedSteps
		};
	}

	destroy(): void {
		if (this.dead) return;
		this.dead = true;
		void this.pending.finally(() => {
			this.batch.destroy();
			this.machine.destroy();
			this.engine.destroy();
		});
	}
}
