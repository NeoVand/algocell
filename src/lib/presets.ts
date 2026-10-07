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
	'tapeLength',
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
// Kept intentionally small and honest. The suppression presets are rungs of
// the instruction-ablation ladder, not claimed recipes — users capture their
// own refined configurations. Patterns use the grammar in $lib/z80-opcodes.

export const BUILTIN_PRESETS: Preset[] = [
	{
		id: 'builtin:classic',
		name: 'Classic',
		builtin: true,
		values: {
			gridType: 'square',
			// Pin the default organism size so Classic restores 4×4 cells.
			tapeLength: 16,
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
		id: 'builtin:no-block-copy',
		name: 'No block copy',
		builtin: true,
		values: {
			// Remove only the Z80's dedicated copy loop (LDI/LDD/LDIR/LDDR). This is
			// the ablation of Cicala et al. (2026) extended to LDD; everything else
			// — stack pushes, LD (HL),r loops, 16-bit loads — stays available.
			suppressPatterns: ['family:block-copy']
		}
	},
	{
		id: 'builtin:no-copy',
		name: 'No-copy',
		builtin: true,
		values: {
			// Every load, stack and exchange family, including the ED-page block
			// copies and 16-bit loads the old substring preset could not reach.
			// Memory can then only be written by CALL/RST pushes, INC/DEC (HL),
			// RLD/RRD, SET/RES (HL) and the CB-page shifts on (HL).
			suppressPatterns: [
				'family:ld8',
				'family:ld8-imm',
				'family:ld8-mem',
				'family:ld16-imm',
				'family:ld16-mem',
				'family:ld-special',
				'family:stack',
				'family:ex',
				'family:block-copy'
			]
		}
	}
];

// ── Persistence (user presets only; built-ins live in code) ────────────────

const STORAGE_KEY = 'algocell.presets.v1';
// Ids of built-ins that have ever been seeded into this browser's storage, so a
// built-in added in a later release is seeded exactly once and a built-in the
// user deleted stays deleted. Absent key = the original three were seeded.
const SEEDED_KEY = 'algocell.presets.seeded.v1';
const LEGACY_SEEDED = ['builtin:classic', 'builtin:hex-organic', 'builtin:no-copy'];

function hasStorage(): boolean {
	try {
		return typeof localStorage !== 'undefined';
	} catch {
		return false;
	}
}

function isValidPreset(p: unknown): p is Preset {
	return (
		!!p &&
		typeof (p as Preset).id === 'string' &&
		typeof (p as Preset).name === 'string' &&
		typeof (p as Preset).values === 'object'
	);
}

// Load the full preset list. On first run (no stored key) the built-ins are
// seeded into storage so that, from then on, EVERY preset — built-in or user —
// is a regular stored entry that can be renamed via overwrite or deleted, and
// the change sticks across reloads. If the user has deleted all of them the
// stored value is an empty array (not absent), so built-ins do not reappear.
export function loadPresets(): Preset[] {
	if (!hasStorage()) return [...BUILTIN_PRESETS];
	try {
		const raw = localStorage.getItem(STORAGE_KEY);
		if (raw === null) {
			const seeded = [...BUILTIN_PRESETS];
			savePresets(seeded);
			markSeeded(BUILTIN_PRESETS.map((p) => p.id));
			return seeded;
		}
		const parsed = JSON.parse(raw);
		if (!Array.isArray(parsed)) return [...BUILTIN_PRESETS];
		return seedNewBuiltins(parsed.filter(isValidPreset));
	} catch {
		return [...BUILTIN_PRESETS];
	}
}

function readSeeded(): Set<string> {
	try {
		const raw = localStorage.getItem(SEEDED_KEY);
		if (raw === null) return new Set(LEGACY_SEEDED);
		const parsed = JSON.parse(raw);
		return new Set(
			Array.isArray(parsed) ? parsed.filter((x) => typeof x === 'string') : LEGACY_SEEDED
		);
	} catch {
		return new Set(LEGACY_SEEDED);
	}
}

function markSeeded(ids: string[]): void {
	try {
		const all = new Set([...readSeeded(), ...ids]);
		localStorage.setItem(SEEDED_KEY, JSON.stringify([...all]));
	} catch {
		// non-fatal
	}
}

// Insert built-ins this browser has never seen, right after the last stored
// built-in (or at the front), and remember that they were seeded.
//
// Also backfill settings a stored built-in lacks but its current definition
// carries (a key added in a later release, e.g. Classic pinning tapeLength: 16).
// Only missing keys are filled, so values the user edited are left alone, and a
// pre-release copy keeps the meaning it had when only that value existed.
function seedNewBuiltins(stored: Preset[]): Preset[] {
	const seeded = readSeeded();
	let changed = false;
	const result = stored.map((p) => {
		const builtin = BUILTIN_PRESETS.find((b) => b.id === p.id);
		if (!builtin) return p;
		const missing = (Object.keys(builtin.values) as SettingKey[]).filter((k) => !(k in p.values));
		if (missing.length === 0) return p;
		changed = true;
		const values: SettingValues = { ...p.values };
		for (const k of missing) values[k] = builtin.values[k];
		return { ...p, values };
	});
	const fresh = BUILTIN_PRESETS.filter(
		(b) => !seeded.has(b.id) && !result.some((p) => p.id === b.id)
	);
	if (fresh.length > 0) {
		let at = -1;
		for (let i = 0; i < result.length; i++) if (result[i].id.startsWith('builtin:')) at = i;
		result.splice(at + 1, 0, ...fresh);
		markSeeded(fresh.map((p) => p.id));
		changed = true;
	}
	if (changed) savePresets(result);
	return result;
}

export function savePresets(presets: Preset[]): void {
	if (!hasStorage()) return;
	try {
		localStorage.setItem(STORAGE_KEY, JSON.stringify(presets));
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
