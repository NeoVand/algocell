// Preset system for Algocell.
//
// A preset is a named bundle of setting values. The set of *what* is tunable
// lives in one declarative schema (see SETTING_KEYS below); the component
// supplies the get/set wiring for each key. Adding a new tunable setting is a
// one-line change here plus one entry in the component's schema — nothing in
// the apply/serialize path needs to change.
//
// Each setting declares an "apply mode" that says how disruptive a change is:
//   - 'live'  : takes effect immediately on the running simulation (no reset)
//   - 'reset' : requires re-seeding the soup (e.g. the RNG seed)
//   - 'grid'  : requires rebuilding GPU buffers (grid size / topology)
// When a preset is applied we take the union of the modes of the settings that
// actually changed and perform the single most disruptive action needed — so
// switching between two presets that differ only in live settings steers the
// running simulation without ever resetting it.

export type ApplyMode = 'live' | 'reset' | 'grid';

// Canonical ordered list of tunable setting keys. This is the single source of
// truth for what a preset can carry.
export const SETTING_KEYS = [
	'gridType',
	'gridWidth',
	'gridHeight',
	'seed',
	'noiseExp',
	'pairCount',
	'z80Steps',
	'suppressPatterns',
	'colormapName',
	'brightness',
	'contrast',
	'saturation',
	'showGridLines',
	'simpleView'
] as const;

export type SettingKey = (typeof SETTING_KEYS)[number];

// A snapshot of every (or some) setting values.
export type SettingValues = Partial<Record<SettingKey, unknown>>;

export interface Preset {
	id: string;
	name: string;
	values: SettingValues;
	builtin?: boolean;
}

// ── Built-in presets ──────────────────────────────────────────────────────
// Kept intentionally small and honest. "No-copy" is a research starting point,
// not a claimed magic recipe — users capture their own refined configurations.

export const BUILTIN_PRESETS: Preset[] = [
	{
		id: 'builtin:classic',
		name: 'Classic',
		builtin: true,
		values: {
			gridType: 'square',
			seed: 6,
			noiseExp: 4,
			z80Steps: 128,
			suppressPatterns: []
		}
	},
	{
		id: 'builtin:hex-organic',
		name: 'Hex organic',
		builtin: true,
		values: {
			gridType: 'hex',
			seed: 6,
			noiseExp: 4,
			z80Steps: 128,
			suppressPatterns: []
		}
	},
	{
		id: 'builtin:no-copy',
		name: 'No-copy',
		builtin: true,
		values: {
			// Suppress the Z80 load/copy families. Starting point for exploring the
			// multi-species regime that emerges when direct byte-copying is blocked.
			suppressPatterns: ['LD', 'PUSH', 'POP', 'EX']
		}
	}
];

// ── Persistence (user presets only; built-ins live in code) ────────────────

const STORAGE_KEY = 'algocell.presets.v1';

function hasStorage(): boolean {
	try {
		return typeof localStorage !== 'undefined';
	} catch {
		return false;
	}
}

export function loadUserPresets(): Preset[] {
	if (!hasStorage()) return [];
	try {
		const raw = localStorage.getItem(STORAGE_KEY);
		if (!raw) return [];
		const parsed = JSON.parse(raw);
		if (!Array.isArray(parsed)) return [];
		// Keep only well-formed entries; drop anything that isn't a preset.
		return parsed.filter(
			(p): p is Preset =>
				p && typeof p.id === 'string' && typeof p.name === 'string' && typeof p.values === 'object'
		);
	} catch {
		return [];
	}
}

export function saveUserPresets(presets: Preset[]): void {
	if (!hasStorage()) return;
	try {
		const userOnly = presets.filter((p) => !p.builtin);
		localStorage.setItem(STORAGE_KEY, JSON.stringify(userOnly));
	} catch {
		// storage full or unavailable — non-fatal
	}
}

// Deterministic-enough id without relying on Date.now/Math.random (which are
// fine at runtime, but a counter keyed off existing ids avoids collisions on
// rapid saves).
export function makePresetId(existing: Preset[]): string {
	let n = existing.length + 1;
	let id = `user:${n}`;
	const ids = new Set(existing.map((p) => p.id));
	while (ids.has(id)) {
		n++;
		id = `user:${n}`;
	}
	return id;
}

// Structural equality for setting values (handles arrays like suppressPatterns).
export function valuesEqual(a: unknown, b: unknown): boolean {
	if (a === b) return true;
	if (Array.isArray(a) && Array.isArray(b)) {
		if (a.length !== b.length) return false;
		for (let i = 0; i < a.length; i++) if (!valuesEqual(a[i], b[i])) return false;
		return true;
	}
	return false;
}
