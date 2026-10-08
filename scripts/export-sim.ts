// Export the simulation's WGSL and the Z80 ISA model for the headless Python
// runner (experiments/). The runner never re-implements the simulation: it
// executes exactly these shader strings, so its results are comparable with
// the deployed site. Run with `npm run export:sim` after changing shaders.ts,
// z80-opcodes.ts or the zilion dependency.
import { writeFileSync, mkdirSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { createSimShader, createZ80TestShader } from '../src/lib/gpu/shaders';
import { SQUARE_TAPE_LENGTHS } from '../src/lib/sim/constants';
import {
	PAGE_TABLES,
	FAMILIES,
	resolveSuppression,
	suppressionMasks
} from '../src/lib/z80-opcodes';
import { readFileSync } from 'node:fs';

const out = new URL('../experiments/algocell_exp/shader/', import.meta.url).pathname;
mkdirSync(out, { recursive: true });

const sha = (s: string) => createHash('sha256').update(s).digest('hex').slice(0, 16);
const zilionVersion = JSON.parse(
	readFileSync(new URL('../node_modules/@neovand/zilion/package.json', import.meta.url), 'utf8')
).version as string;

// Non-square lengths are exported for the headless runner only (the UI tiles cells as √L×√L).
const HEADLESS_TAPE_LENGTHS = [3, 5, 6, 7, 8, 10, 12, 18, 20, 24, 32, 50] as const;
const ALL_TAPE_LENGTHS = [...SQUARE_TAPE_LENGTHS, ...HEADLESS_TAPE_LENGTHS];
// Ring-length variants for the gcd experiments (Stage F): pair memory padded to P bytes.
const RING_VARIANTS: Array<[number, number]> = [[16, 33], [16, 34], [16, 35], [16, 37], [36, 73], [36, 74], [36, 75], [36, 79],
  // dead-zone probes (Stage E: the pusher's heritability dips at L = 9–12): longer rings at L = 8, 9, 10, 12, 16
  [8, 20], [8, 24], [8, 32], [9, 20], [9, 24], [9, 32], [10, 24], [10, 28], [10, 32], [10, 40], [12, 28], [12, 32], [12, 36], [12, 48], [16, 40], [16, 48],
  // local ring sweep (post Stage F): every even P from 2L to 2L+24 at L = 8, 10, 12, 16
  [8, 18], [8, 22], [8, 26], [8, 28], [8, 30], [8, 36], [8, 40], [10, 22], [10, 26], [10, 30], [10, 34], [10, 36], [10, 38], [10, 44],
  [12, 26], [12, 30], [12, 34], [12, 38], [12, 40], [12, 42], [12, 44], [12, 46], [16, 36], [16, 38], [16, 42], [16, 44], [16, 46], [16, 50], [16, 52], [16, 54], [16, 56]];
const shaders: Record<string, string> = { hex: createSimShader('hex') };
for (const L of ALL_TAPE_LENGTHS) shaders[`square_L${L}`] = createSimShader('square', L);
for (const [L, P] of RING_VARIANTS) shaders[`square_L${L}_P${P}`] = createSimShader('square', L, P);
for (const [k, v] of Object.entries(shaders)) writeFileSync(`${out}sim_${k}.wgsl`, v);
// Single-pair executor (the differential-test shader): used by the replication
// assay, one per tape length so the private memory fits the pair.
writeFileSync(`${out}z80_test.wgsl`, createZ80TestShader());
for (const L of ALL_TAPE_LENGTHS) writeFileSync(`${out}z80_test_L${L}.wgsl`, createZ80TestShader(L));
for (const [L, P] of RING_VARIANTS) writeFileSync(`${out}z80_test_L${L}_P${P}.wgsl`, createZ80TestShader(L, P));

// Golden vectors so the Python port of the pattern grammar can be checked.
const goldenPatterns = [
	['LD', 'PUSH', 'POP', 'EX'],
	['family:block-copy'],
	['family:writes-mem'],
	['family:stack', 'family:call-ret', 'family:rst', 'family:ex', 'family:block-copy'],
	['ed:B0'],
	['C5'],
	['0x3e'],
	['RST n'],
	['ADD A, B'],
	['LDIR'],
	['base:ED', 'cb:00', 'ed:00'],
	['family:writes-mem', '-family:incdec-mem', '-family:rotate-mem', '-family:bit-set-mem'],
	['LD', '-family:block-copy']
];
const golden = goldenPatterns.map((patterns) => ({
	patterns,
	masks: Array.from(suppressionMasks(resolveSuppression(patterns)))
}));

const isa = {
	zilion: zilionVersion,
	families: FAMILIES,
	pages: {
		base: PAGE_TABLES.base,
		cb: PAGE_TABLES.cb,
		ed: PAGE_TABLES.ed
	},
	golden
};
writeFileSync(`${out}isa.json`, JSON.stringify(isa));

const meta = {
	exportedAt: new Date().toISOString(),
	zilion: zilionVersion,
	shaderSha256_16: Object.fromEntries(Object.entries(shaders).map(([k, v]) => [k, sha(v)])),
	entryPoints: [
		'clear_collision',
		'prepare_batch',
		'z80_execute_batch',
		'absorb_results',
		'mutate_soup',
		'count_bytes',
		'clear_byte_counts',
		'hash_cells'
	]
};
writeFileSync(`${out}meta.json`, JSON.stringify(meta, null, 2) + '\n');
console.log('exported', Object.keys(shaders).join(', '), 'zilion', zilionVersion, meta.shaderSha256_16);
