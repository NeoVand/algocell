import {
	z80,
	m6502,
	i8086,
	type WorldSpec,
	type MachineSpec,
	type RawHooks
} from '@neovand/zilion';
import { SplitMix64 } from '../sim/prng';
import { ablationMasks } from '../lab/ablation';
import type { LabSettings, InitialCondition } from '../lab/types';

export function validateSettings(settings: LabSettings): void {
	if (!['z80', 'm6502', 'i8086'].includes(settings.isa))
		throw new Error('Unsupported instruction set.');
	if (
		!['0.2', '0.3'].includes(settings.compat) ||
		(settings.isa !== 'z80' && settings.compat !== '0.3')
	)
		throw new Error('Only Z80 supports historical compatibility 0.2.');
	if (
		settings.isa === 'z80'
			? !['classic', 'zilog-nmos'].includes(settings.variant) ||
				(settings.compat === '0.2' && settings.variant !== 'classic')
			: settings.variant !== 'nmos'
	)
		throw new Error(
			'The selected core variant is incompatible with this instruction set or version.'
		);
	const integer = (key: keyof LabSettings, min: number, max: number) => {
		const value = settings[key];
		if (typeof value !== 'number' || !Number.isInteger(value) || value < min || value > max)
			throw new Error(`${key} must be an integer from ${min} to ${max}.`);
	};
	integer('length', 4, 128);
	integer('ring', 2 * settings.length, 256);
	integer('width', 2, 400);
	integer('height', 2, 400);
	if (settings.width * settings.height * settings.length > 4_194_304)
		throw new Error('Population exceeds the 4 MiB interactive memory limit.');
	integer('steps', 1, 4096);
	integer('pairAttempts', 1, 65536);
	integer('seed', 0, 0xffffffff);
	integer('mutationExponent', 0, 31);
	if (!['square', 'hex'].includes(settings.topology)) throw new Error('Unknown lattice topology.');
	if (!['top', 'first-push', 'legacy-sp'].includes(settings.stack))
		throw new Error('Unknown stack convention.');
	if (settings.stack === 'legacy-sp' && settings.isa !== 'z80')
		throw new Error('The legacy SP convention belongs to Z80.');
	if (!['counter', 'historical-pcg'].includes(settings.dialMode))
		throw new Error('Unknown random-draw convention.');
	if (
		!Number.isFinite(settings.lethalProbability) ||
		settings.lethalProbability < 0 ||
		settings.lethalProbability > 1
	)
		throw new Error('Lethality must be between zero and one.');
	if (
		typeof settings.randomRegisters !== 'boolean' ||
		!Array.isArray(settings.ablation) ||
		settings.ablation.some((s) => typeof s !== 'string')
	)
		throw new Error('Invalid reset or ablation settings.');
	ablationMasks(settings.isa, settings.ablation);
}

export function validateInitial(initial: InitialCondition): void {
	const bytes = (values: number[]) =>
		Array.isArray(values) &&
		values.length > 0 &&
		values.length <= 128 &&
		values.every((v) => Number.isInteger(v) && v >= 0 && v <= 255);
	if (initial.kind === 'random') return;
	if (initial.kind === 'tiled' && bytes(initial.bytes)) return;
	if (
		initial.kind === 'competition' &&
		bytes(initial.regenerator) &&
		bytes(initial.transmitterCore) &&
		typeof initial.randomPayload === 'boolean' &&
		Number.isFinite(initial.share) &&
		initial.share >= 0 &&
		initial.share <= 1
	)
		return;
	throw new Error('Invalid initial tape design.');
}

/** CPU-only preview uses exactly the same initialization as a live run. */
export function createInitialSoup(
	settings: LabSettings,
	initial: InitialCondition = { kind: 'random' }
): Uint8Array {
	validateSettings(settings);
	validateInitial(initial);
	const soup = new Uint8Array(settings.width * settings.height * settings.length);
	const rng = new SplitMix64(settings.seed);
	for (let i = 0; i < soup.length; i += 4) {
		const word = rng.nextU32();
		for (let b = 0; b < 4 && i + b < soup.length; b++) soup[i + b] = word >>> (8 * b);
	}
	if (initial.kind === 'random') return soup;
	for (let cell = 0; cell < settings.width * settings.height; cell++) {
		const start = cell * settings.length;
		if (initial.kind === 'tiled') {
			for (let b = 0; b < settings.length; b++)
				soup[start + b] = initial.bytes[b % initial.bytes.length];
		} else if (rng.nextU32() / 4294967296 < initial.share) {
			if (!initial.randomPayload) soup.fill(0, start, start + settings.length);
			soup.set(initial.transmitterCore.slice(0, settings.length), start);
		} else {
			for (let b = 0; b < settings.length; b++)
				soup[start + b] = initial.regenerator[b % initial.regenerator.length];
		}
	}
	return soup;
}

const pcgWGSL = (state: string, draw: string) => `var<private> ${state}: u32;
fn ${draw}() -> u32 {
  let s = ${state}; ${state} = s * 747796405u + 2891336453u;
  let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}`;

/** Raw hooks preserve registered PCG draws; the manifest truthfully marks them non-resumable. */
function historicalHooks(settings: LabSettings): RawHooks | undefined {
	if (settings.isa !== 'z80' || settings.dialMode !== 'historical-pcg') return undefined;
	const declarations: string[] = [],
		reset: string[] = [];
	let onInstructionStart: string | undefined;
	if (settings.lethalProbability > 0 && settings.lethalProbability < 1) {
		declarations.push(pcgWGSL('lethal_state', 'lethal_draw'));
		reset.push(
			'lethal_state = zil_seed * 2654435761u + zil_lane * 2246822519u + 0x165667b1u; lethal_draw();'
		);
		onInstructionStart = `if (unit == 0u && lethal_draw() < ${Math.floor(settings.lethalProbability * 4294967296)}u) { return ZIL_HALT; }`;
	}
	if (settings.randomRegisters) {
		declarations.push(pcgWGSL('conv_state', 'conv_draw'));
		reset.push(
			'conv_state = zil_seed * 2246822519u + zil_lane * 3266489917u + 0x27d4eb2fu; conv_draw();'
		);
		for (const register of [
			'a',
			'f',
			'b',
			'c',
			'd',
			'e',
			'h',
			'l',
			'a2',
			'f2',
			'b2',
			'c2',
			'd2',
			'e2',
			'h2',
			'l2'
		])
			reset.push(`cpu_${register} = conv_draw() & 0xffu;`);
		reset.push('cpu_ix = conv_draw() & 0xffffu;', 'cpu_iy = conv_draw() & 0xffffu;');
	}
	return declarations.length
		? {
				declarations: declarations.join('\n'),
				onReset: reset.join('\n'),
				...(onInstructionStart ? { onInstructionStart } : {})
			}
		: undefined;
}

export function machineSpec(settings: LabSettings): MachineSpec {
	validateSettings(settings);
	const isa = { z80, m6502, i8086 }[settings.isa];
	const historic = settings.isa === 'z80' && settings.dialMode === 'historical-pcg';
	const world: WorldSpec = {
		memory: {
			kind: 'segments',
			segments: [
				{ bytes: settings.length, name: 'A' },
				{ bytes: settings.length, name: 'B' }
			],
			ring: settings.ring
		},
		entry: { position: 0 },
		init: {
			...(settings.stack === 'legacy-sp'
				? { set: { sp: 65535 } }
				: {
						stack:
							settings.stack === 'first-push'
								? { firstPushAt: 'lastByte' as const }
								: { topAliases: 'lastByte' as const }
					}),
			...(settings.isa === 'm6502'
				? { set: { a: 0, x: 0, y: 0, p: 0x20, pc: 0 } }
				: settings.isa === 'i8086'
					? {
							set: {
								ax: 0,
								cx: 0,
								dx: 0,
								bx: 0,
								bp: 0,
								si: 0,
								di: 0,
								cs: 0,
								ds: 0,
								es: 0,
								ss: 0,
								ip: 0,
								flags: 0xf002
							}
						}
					: {}),
			...(settings.randomRegisters && !historic ? { random: { salt: 0 } } : {})
		},
		budget: { unit: 'steps', value: settings.steps },
		...(settings.lethalProbability > 0 && (!historic || settings.lethalProbability === 1)
			? { halting: { lethal: { firstUnit: 0, probability: settings.lethalProbability, salt: 0 } } }
			: {}),
		ablation: { table: 'uniform' },
		outputs: { memory: true, state: true, fetchMap: true, startMap: true, writeCounts: true },
		...(historicalHooks(settings) ? { hooks: historicalHooks(settings) } : {})
	};
	return {
		core: {
			isa,
			compat: settings.compat,
			variant: settings.variant,
			...(settings.isa === 'i8086'
				? { params: { prefetch: 'coherent', hidden_latch: 'zero_ea' } }
				: {})
		},
		world
	};
}

/** Exact original Algocell condition, including SP=FFFF in hex mode. */
export const ORIGINAL_SETTINGS: LabSettings = {
	isa: 'z80',
	compat: '0.2',
	variant: 'classic',
	length: 16,
	ring: 32,
	steps: 128,
	width: 200,
	height: 200,
	topology: 'square',
	seed: 6,
	pairAttempts: 8192,
	mutationExponent: 4,
	lethalProbability: 0,
	randomRegisters: false,
	ablation: [],
	stack: 'legacy-sp',
	dialMode: 'historical-pcg'
};
