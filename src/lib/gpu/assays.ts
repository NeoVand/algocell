import { createEngine, type ShaderManifest } from '@neovand/zilion';
import { ablationMasks } from '../lab/ablation';
import type { CultureResult, EncounterObservation, LabSettings } from '../lab/types';
import { SplitMix64 } from '../sim/prng';
import { machineSpec } from './machine';
import { observation, runCulture } from './culture';

export interface AssayOptions {
	seed?: number;
	position?: number;
	value?: number;
	trials?: number;
}

export interface AssayResult {
	schema: 'algocell.encounter-bench/1';
	seed: number;
	parent: number[];
	selfPair: EncounterObservation;
	partnerSensitivity: {
		trials: number;
		changedOffspring: number;
		changedParent: number;
		meanChangedFraction: number;
		perturbation: string;
		example: { baseline: EncounterObservation; perturbed: EncounterObservation };
	};
	marker: {
		position: number;
		value: number;
		wildType: CultureResult;
		mutant: CultureResult;
		retention: { generation: number; mutant: number; wildType: number }[];
		note: string;
	};
	settings: LabSettings;
	manifest: ShaderManifest;
}

/** These bounded, isolated tests never read or step the resident population. */
export async function runAssays(
	device: GPUDevice,
	settings: LabSettings,
	tape: number[] | Uint8Array,
	options: AssayOptions = {}
): Promise<AssayResult> {
	const L = settings.length;
	if (tape.length !== L || Array.from(tape).some((b) => !Number.isInteger(b) || b < 0 || b > 255))
		throw new Error(`Select a complete ${L}-byte tape before testing.`);
	const trials = options.trials ?? 16;
	const seed = options.seed ?? 4242;
	const position = options.position ?? Math.min(8, L - 1);
	const value = options.value ?? (tape[position] === 0x7f ? 0x80 : 0x7f);
	if (!Number.isInteger(trials) || trials < 1 || trials > 64)
		throw new Error('The encounter bench supports 1–64 matched partners.');
	if (!Number.isInteger(seed) || seed < 0 || seed > 0xffffffff)
		throw new Error('The assay seed must be an unsigned 32-bit integer.');
	if (
		!Number.isInteger(position) ||
		position < 0 ||
		position >= L ||
		!Number.isInteger(value) ||
		value < 0 ||
		value > 255
	)
		throw new Error('Choose a position in the tape and a byte value from 0 to 255.');
	if (tape[position] === value)
		throw new Error('Choose a marker value different from the selected byte to compare a change.');
	const engine = await createEngine({ device });
	const machine = await engine.machine(machineSpec(settings)).catch((error: unknown) => {
		engine.destroy();
		throw error;
	});
	const masks = ablationMasks(settings.isa, settings.ablation);
	try {
		const selfInput = Uint8Array.from([...tape, ...tape]);
		const self = await machine.run(selfInput, { seed, ablation: masks });
		const rng = new SplitMix64(seed);
		const input = new Uint8Array(trials * L * 2);
		for (let lane = 0; lane < trials; lane++) {
			input.set(tape, lane * L * 2);
			for (let b = 0; b < L; b++) input[lane * L * 2 + L + b] = rng.nextU32() & 255;
		}
		const changed = input.slice();
		for (let lane = 0; lane < trials; lane++) changed[lane * L * 2 + L + (lane % L)] ^= 1;
		// Identical lane indices and seed hold reset and chemistry draws fixed.
		const baseline = await machine.run(input, { seed, ablation: masks });
		const perturbed = await machine.run(changed, { seed, ablation: masks });
		let changedOffspring = 0,
			changedParent = 0,
			changedBytes = 0;
		for (let lane = 0; lane < trials; lane++) {
			const a = baseline.memoryOf(lane),
				b = perturbed.memoryOf(lane);
			let parentDiff = false,
				childDiff = false;
			for (let pos = 0; pos < L; pos++) {
				parentDiff ||= a[pos] !== b[pos];
				if (a[L + pos] !== b[L + pos]) {
					childDiff = true;
					changedBytes++;
				}
			}
			if (parentDiff) changedParent++;
			if (childDiff) changedOffspring++;
		}
		const mutantTape = Uint8Array.from(tape);
		mutantTape[position] = value;
		const cultureOptions = { trials, seed, generations: 2, marker: { position, value } };
		const wildType = await runCulture(device, settings, tape, cultureOptions);
		const mutant = await runCulture(device, settings, mutantTape, cultureOptions);
		return {
			schema: 'algocell.encounter-bench/1',
			seed,
			parent: Array.from(tape),
			selfPair: observation(self, selfInput),
			partnerSensitivity: {
				trials,
				changedOffspring,
				changedParent,
				meanChangedFraction: changedBytes / (trials * L),
				perturbation:
					'Flip the low bit of one initial partner byte; position = trial index modulo tape length. Hold parent, lane, reset seed and chemistry seed fixed. Unchanged original partner bytes can persist without being read.',
				example: {
					baseline: observation(baseline, input.subarray(0, L * 2)),
					perturbed: observation(perturbed, changed.subarray(0, L * 2))
				}
			},
			marker: {
				position,
				value,
				wildType,
				mutant,
				retention: wildType.retention!.map((fraction, i) => ({
					generation: i + 1,
					wildType: fraction,
					mutant: mutant.retention![i]
				})),
				note: 'Same-position value matches after serial transfer against matched fresh partners. Values may be regenerated or reacquired; this measures retention, not causal ancestry. All offspring enter the next transfer, including unsuccessful copies.'
			},
			settings: structuredClone(settings),
			manifest: machine.manifest
		};
	} finally {
		machine.destroy();
		engine.destroy();
	}
}

export function stoppingReason(record: EncounterObservation): string {
	if (record.trap && record.trap !== 'none') return `Trap: ${record.trap}`;
	if (!record.halt || record.halt === 'running' || record.halt === 'none') return 'Budget reached';
	return String(record.halt);
}
