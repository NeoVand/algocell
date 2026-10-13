import { createEngine, type MachineResult } from '@neovand/zilion';
import { machineSpec } from './machine';
import { ablationMasks } from '../lab/ablation';
import type {
	LabSettings,
	CultureResult,
	CultureGeneration,
	EncounterObservation
} from '../lab/types';

// Culture runs only on demand, on a separate machine. Its readbacks never enter the soup loop.
class Pcg {
	constructor(public state: number) {
		this.state >>>= 0;
	}
	next(): number {
		const s = this.state;
		this.state = (Math.imul(s, 747796405) + 2891336453) >>> 0;
		const w = Math.imul((s >>> ((s >>> 28) + 4)) ^ s, 277803737) >>> 0;
		return ((w >>> 22) ^ w) >>> 0;
	}
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

export function observation(
	result: MachineResult,
	before: Uint8Array,
	lane = 0
): EncounterObservation {
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

export async function runCulture(
	device: GPUDevice,
	settings: LabSettings,
	tape: number[] | Uint8Array,
	options: {
		trials?: number;
		seed?: number;
		generations?: number;
		marker?: { position: number; value: number };
	} = {}
): Promise<CultureResult> {
	if (
		tape.length !== settings.length ||
		Array.from(tape).some((b) => !Number.isInteger(b) || b < 0 || b > 255)
	)
		throw new Error(`A tape must contain exactly ${settings.length} bytes.`);
	const engine = await createEngine({ device });
	const machine = await engine.machine(machineSpec(settings)).catch((error: unknown) => {
		engine.destroy();
		throw error;
	});
	const masks = ablationMasks(settings.isa, settings.ablation);
	try {
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
		const marker = options.marker;
		if (
			marker &&
			(!Number.isInteger(marker.position) ||
				marker.position < 0 ||
				marker.position >= tape.length ||
				!Number.isInteger(marker.value) ||
				marker.value < 0 ||
				marker.value > 255)
		)
			throw new Error('Marker position and value must identify a byte in the selected tape.');
		const L = settings.length,
			rng = new Pcg(seed);
		let parents = new Uint8Array(trials * L);
		for (let k = 0; k < trials; k++) parents.set(tape, k * L);
		const summary: CultureGeneration[] = [];
		const retention: number[] = [];
		let example: EncounterObservation | undefined;
		for (let generation = 1; generation <= generations; generation++) {
			const input = new Uint8Array(trials * L * 2);
			for (let k = 0; k < trials; k++) {
				input.set(parents.subarray(k * L, (k + 1) * L), k * L * 2);
				for (let b = 0; b < L; b++) input[k * L * 2 + L + b] = rng.next() & 255;
			}
			const result = await machine.run(input, { seed: rng.next(), ablation: masks });
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
			parents = offspring;
			if (marker) retention.push(retained / trials);
		}
		const gain = summary[1]?.gain ?? null;
		return {
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
			manifest: machine.manifest,
			settings: structuredClone(settings),
			...(marker ? { retention } : {})
		};
	} finally {
		machine.destroy();
		engine.destroy();
	}
}
