import { validateSettings } from './engine';
import type { InitialCondition, LabSettings, LabSnapshot } from './types';

const MAX_JSON_CHARACTERS = 40_000_000;

function record(value: unknown): value is Record<string, unknown> {
	return !!value && typeof value === 'object' && !Array.isArray(value);
}

function tape(value: unknown): value is number[] {
	return (
		Array.isArray(value) &&
		value.length > 0 &&
		value.length <= 128 &&
		value.every((n) => Number.isInteger(n) && n >= 0 && n <= 255)
	);
}

/** Checkpoints are data only. No HTML, expressions or raw world hooks are imported for execution. */
export function parseCheckpoint(text: string): LabSnapshot {
	if (text.length > MAX_JSON_CHARACTERS)
		throw new Error('Checkpoint exceeds the 40 MB import limit.');
	const data: unknown = JSON.parse(text);
	if (
		!record(data) ||
		data.schema !== 'algocell.lab/1' ||
		data.scheduler !== 'serial-reflected-pcg/1'
	)
		throw new Error(
			'This is not a supported Algocell laboratory checkpoint. Older PNG files open in the legacy app.'
		);
	if (!record(data.settings)) throw new Error('Checkpoint settings are missing.');
	validateSettings(data.settings as unknown as LabSettings);
	const settings = data.settings as unknown as LabSettings;
	const initial = data.initialCondition;
	if (
		!record(initial) ||
		!(
			initial.kind === 'random' ||
			(initial.kind === 'tiled' && tape(initial.bytes)) ||
			(initial.kind === 'competition' &&
				tape(initial.regenerator) &&
				tape(initial.transmitterCore) &&
				typeof initial.randomPayload === 'boolean' &&
				typeof initial.share === 'number' &&
				initial.share >= 0 &&
				initial.share <= 1)
		)
	)
		throw new Error('Checkpoint initial condition is invalid.');
	if (
		!Array.isArray(data.bytes) ||
		data.bytes.length !== settings.width * settings.height * settings.length ||
		data.bytes.some((n) => !Number.isInteger(n) || n < 0 || n > 255)
	)
		throw new Error('Checkpoint byte count or values are invalid.');
	for (const key of ['update', 'encounters', 'executedSteps', 'rngState'] as const)
		if (!Number.isSafeInteger(data[key]) || (data[key] as number) < 0)
			throw new Error(`Checkpoint ${key} is invalid.`);
	if ((data.rngState as number) > 0xffffffff)
		throw new Error('Checkpoint random state is invalid.');
	if (
		!record(data.manifest) ||
		data.manifest.schema !== 'zilion.manifest/2' ||
		!record(data.manifest.sha256) ||
		['wgsl', 'core', 'world', 'options'].some(
			(key) =>
				typeof (data.manifest as Record<string, unknown>).sha256 !== 'object' ||
				!/^[0-9a-f]{64}$/.test((data.manifest as { sha256: Record<string, string> }).sha256[key])
		)
	)
		throw new Error('Checkpoint machine identity is missing or invalid.');
	if (typeof data.createdAt !== 'string' || !Number.isFinite(Date.parse(data.createdAt)))
		throw new Error('Checkpoint creation time is invalid.');
	return {
		...data,
		settings,
		initialCondition: initial as unknown as InitialCondition
	} as unknown as LabSnapshot;
}

export function serializeCheckpoint(snapshot: LabSnapshot): string {
	const text = JSON.stringify(snapshot);
	parseCheckpoint(text);
	return text;
}
