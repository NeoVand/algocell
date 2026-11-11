import type { ShaderManifest } from '@neovand/zilion';

export type LabIsa = 'z80' | 'm6502' | 'i8086';
export type LabCompat = '0.2' | '0.3';
export type LabVariant = 'classic' | 'zilog-nmos' | 'nmos';
export type StackConvention = 'top' | 'first-push' | 'legacy-sp';

/** Every number here is part of the run contract, not merely a display setting. */
export interface LabSettings {
	isa: LabIsa;
	compat: LabCompat;
	variant: LabVariant;
	length: number;
	ring: number;
	steps: number;
	width: number;
	height: number;
	topology: 'square' | 'hex';
	seed: number;
	pairAttempts: number;
	mutationExponent: number;
	lethalProbability: number;
	randomRegisters: boolean;
	ablation: string[];
	stack: StackConvention;
	dialMode: 'historical-pcg' | 'counter';
	/**
	 * How an encounter's result reaches the lattice. `overwrite` is the historical rule: the pair's
	 * memory after execution replaces both tapes. `reversible` keeps a second layer per cell (the tape
	 * before its last event) and applies the programs' change on top of it, so an event can be undone.
	 */
	interaction: InteractionRule;
	/**
	 * Reversible rule only: the chance, per cell and event, that the cell's stored past is discarded
	 * (before := now) ahead of the event. 1 reproduces `overwrite` exactly; 0 is a closed, fully
	 * reversible world.
	 */
	dissipation: number;
	/**
	 * What a cell at the edge of the lattice meets. `reflect` is the historical rule: a draw that
	 * points off the lattice is turned inward, so the world has walls. The other rules glue opposite
	 * edges together so that no cell is special: `cylinder` joins left to right, `torus` joins both
	 * pairs, `mobius` and `klein` join left to right with a flip, `projective` flips both pairs.
	 */
	boundary: BoundaryRule;
}

export type InteractionRule = 'overwrite' | 'reversible';
export type BoundaryRule = 'reflect' | 'cylinder' | 'torus' | 'mobius' | 'klein' | 'projective';
export const BOUNDARY_RULES: BoundaryRule[] = [
	'reflect',
	'cylinder',
	'torus',
	'mobius',
	'klein',
	'projective'
];

/**
 * Settings saved before the interaction rule existed (checkpoints, presets) carry neither field.
 * They meant the historical rule, so fill it in instead of rejecting them.
 */
export function withInteractionDefaults(
	settings: Omit<LabSettings, 'interaction' | 'dissipation' | 'boundary'> & Partial<LabSettings>
): LabSettings {
	return {
		...settings,
		interaction: settings.interaction ?? 'overwrite',
		dissipation: settings.dissipation ?? 1,
		boundary: settings.boundary ?? 'reflect'
	};
}

export type InitialCondition =
	| { kind: 'random' }
	| { kind: 'tiled'; bytes: number[] }
	| {
			kind: 'competition';
			regenerator: number[];
			transmitterCore: number[];
			share: number;
			randomPayload: boolean;
	  };

export const DEFAULT_SETTINGS: LabSettings = {
	isa: 'z80',
	compat: '0.3',
	variant: 'classic',
	length: 16,
	ring: 32,
	steps: 128,
	width: 160,
	height: 125,
	topology: 'square',
	seed: 2001,
	pairAttempts: 8192,
	mutationExponent: 4,
	lethalProbability: 0,
	randomRegisters: false,
	ablation: [],
	stack: 'top',
	dialMode: 'counter',
	interaction: 'overwrite',
	dissipation: 1,
	boundary: 'reflect'
};

/** Unknown, no partner entry, partner entry in the last initiating encounter. */
export type FetchStatus = 0 | 1 | 2;

export interface LabMetrics {
	update: number;
	encounters: number;
	executedSteps: number;
	zeroFraction: number;
	topShare: number;
	diversity: number;
	uniqueTapes: number;
	topTape: number[];
	topCount: number;
	activePairs: number;
	mutationAttempts: number;
	regeneratorShare: number | null;
	transmitterShare: number | null;
	coreShare: number | null;
}

export interface EncounterObservation {
	before: number[];
	after: number[];
	fetched: number[];
	starts: number[];
	state: Record<string, number>;
	writes: Record<string, number>;
	steps: number;
	halt: string | number;
	trap: string | number;
}

export interface CultureGeneration {
	generation: number;
	gain: number | null;
	informativePartners: number;
	exactCopies: number;
	cyclicCopies: number;
	copyFraction: number;
	meanSimilarity: number;
	partnerEntries: number;
}

export interface CultureResult {
	parent: number[];
	trials: number;
	seed: number;
	generations: CultureGeneration[];
	/** Generation two's gain; null if only one generation was requested. */
	gain: number | null;
	positive: boolean | null;
	confinement: 'confined' | 'intermediate' | 'open';
	example: EncounterObservation;
	manifest: ShaderManifest;
	settings: LabSettings;
	/** Optional same-position byte-value matches after each serial transfer. Not ancestry. */
	retention?: number[];
}

export interface MarkerResult {
	position: number;
	value: number;
	wildType: CultureResult;
	mutant: CultureResult;
	retention: { generation: number; mutant: number; wildType: number }[];
	note: string;
}

export interface LabSnapshot {
	schema: 'algocell.lab/1';
	scheduler: 'serial-reflected-pcg/1';
	settings: LabSettings;
	initialCondition: InitialCondition;
	update: number;
	encounters: number;
	executedSteps: number;
	rngState: number;
	bytes: number[];
	manifest: ShaderManifest;
	createdAt: string;
}

export interface DisassemblyLine {
	offset: number;
	bytes: number[];
	mnemonic: string;
	length: number;
	truncated?: boolean;
}
