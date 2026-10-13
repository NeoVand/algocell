import { DEFAULT_SETTINGS, type InitialCondition, type LabSettings } from './types';

export interface PresetSource {
	registration: string;
	world?: string;
	results: string[];
}

export interface PublishedCondition {
	width: number | null;
	height: number | null;
	pairAttempts: number | null;
	mutationAttempts: number | null;
	horizon: number;
	horizonUnit: 'updates' | 'transfers';
	seeds: number[];
	reset: string;
	note: string;
}

export interface PaperPreset {
	id: string;
	name: string;
	category: 'discovery' | 'intervention' | 'demonstration';
	description: string;
	whatToLookFor: string;
	finding: string;
	limits: string;
	source: PresetSource;
	published: PublishedCondition;
	config: LabSettings;
	initialCondition: InitialCondition;
}

export const DEFAULT_PRESET_ID = 'stage-g-l16';
export const PAPER_REPOSITORY = 'algocell-paper';
export const HOST_CONTRACT_NOTE =
	'This app uses a new serial host scheduler and mutation stream. These are source-bound world settings, not a replay of an archived population trajectory. A seed alone does not reproduce the paper’s GPU scheduling.';

const PLAN = 'registrations/originals/PLAN.md';
const REVISION = 'registrations/originals/REVISION_PREREG.md';
const Z80_WORLD = 'src/algocell_exp/machines/worlds.json';
const Z80_RESET =
	'All registers and flags zero except SP, whose top aliases the last byte of B; PC = 0. The first pushed byte lands one byte before that alias.';
const range = (start: number, count = 20) => Array.from({ length: count }, (_, i) => start + i);

function config(overrides: Partial<LabSettings> = {}): LabSettings {
	return {
		...DEFAULT_SETTINGS,
		compat: '0.2',
		dialMode: 'historical-pcg',
		...overrides,
		ablation: [...(overrides.ablation ?? [])]
	};
}

function published(overrides: Partial<PublishedCondition> = {}): PublishedCondition {
	return {
		width: 160,
		height: 125,
		pairAttempts: 8192,
		mutationAttempts: 512,
		horizon: 300_000,
		horizonUnit: 'updates',
		seeds: range(2001),
		reset: Z80_RESET,
		note: 'Reflecting square lattice; pair slots and byte replacements are attempts, not guaranteed completed encounters or changed bytes.',
		...overrides
	};
}

function preset(
	entry: Omit<PaperPreset, 'config' | 'published' | 'initialCondition'> & {
		config?: Partial<LabSettings>;
		published?: Partial<PublishedCondition>;
		initialCondition?: InitialCondition;
	}
): PaperPreset {
	return {
		...entry,
		config: config(entry.config),
		published: published(entry.published),
		initialCondition: entry.initialCondition ?? { kind: 'random' }
	};
}

const stageG = preset({
	id: DEFAULT_PRESET_ID,
	name: 'Historical Z80 · Stage G',
	category: 'discovery',
	description:
		'Twenty thousand random 16-byte tapes meet on a reflecting lattice. This is the historical benign Z80 world, with the frozen 0.2 processor semantics.',
	whatToLookFor:
		'Watch pale zero-rich material accumulate, then inspect abundant tapes. Run a culture test: visible order and a large exact-tape share do not establish reproduction.',
	finding:
		'All 80 historical Stage G worlds had open first detected copiers. At length 16, 19 of 20 first exemplars were load–push tilings; median detection was 450 updates.',
	limits:
		'These are cohort observations, not a guaranteed sequence for a fresh world. Detection is a sampled abundance threshold, not the first physical birth of a copier.',
	source: {
		registration: `${PLAN} · Stage G`,
		world: `${Z80_WORLD} · z80c02_pair_maps_L16`,
		results: ['results/stageG/FINDINGS.md', 'results/stageG/stageG/stage_g_runs.csv']
	}
});

const corrected = preset({
	id: 'z80-corrected',
	name: 'Corrected Z80 · 0.3',
	category: 'discovery',
	description:
		'The familiar 16-byte world on the corrected Z80 core. Compare this explicit processor contract with historical compatibility 0.2.',
	whatToLookFor:
		'Inspect the same candidate under both cores. Keep geometry, reset, budget and chemistry fixed when exploring instruction-level differences.',
	finding:
		'The paper’s corrected-core ecological reruns retained 19 of 22 claim verdicts; four of 36 component verdicts changed.',
	limits:
		'This is a current-engine reference world. Fresh stochastic trajectories do not isolate a causal effect of an emulator repair, and this preset does not reproduce the full rerun cohort.',
	source: {
		registration: `${REVISION} · I1 / Tier 2`,
		world: 'src/algocell_exp/machines/worlds_tier2.json',
		results: ['results/tier2/T2_CLAIMS.csv', 'results/tier2/T2_COMPONENTS.csv']
	},
	config: { compat: '0.3', dialMode: 'counter' },
	published: {
		note: 'A current-engine reference derived from the standard geometry; the cited Tier 2 evidence contains separate, explicitly identified cohorts.'
	}
});

const lethal = preset({
	id: 'lethal-zero',
	name: 'Lethal zero · Stage I',
	category: 'intervention',
	description:
		'Change one rule of the historical Z80 world: fetching 00 as the first byte of an instruction halts that encounter. Zero used as an operand or data is not automatically lethal.',
	whatToLookFor:
		'Compare with benign Stage G. Track zero fraction, then culture selected candidates and inspect whether instruction fetching enters the partner.',
	finding:
		'All ten first detected copiers in the historical lethal cohort were confined. By the endpoint, transmitters were a majority of culture-positive cells in nine of ten worlds.',
	limits:
		'The result is conditional on this chemistry and cohort. The mechanism behind the ecological sorting remains unresolved; confinement alone does not establish transmission.',
	source: {
		registration: `${PLAN} · Stage I`,
		world: `${Z80_WORLD} · z80c02_lethal_L16`,
		results: ['results/stageI/per_world.csv', 'results/ladder/population_worlds.csv']
	},
	config: { lethalProbability: 1, seed: 4001 },
	published: { seeds: range(4001, 10) }
});

const dialPresets = [0, 0.01, 0.03, 0.1, 0.3, 1].map((probability) =>
	preset({
		id: `dial-${probability}`,
		name: `Lethality dial · ${probability * 100}%`,
		category: 'intervention',
		description: `Each fetch of a zero opcode independently halts with probability ${probability}. This changes encounter chemistry, while keeping the Stage G geometry, reset and budget.`,
		whatToLookFor:
			'Compare several probabilities, including benign zero and fully lethal zero. The draw happens at every zero instruction start, not once per tape or update.',
		finding:
			'Historical open-first counts across probabilities 0, 0.01, 0.03, 0.1, 0.3 and 1 were 10, 10, 10, 10, 2 and 0 out of ten worlds. Several timing and endpoint clauses failed.',
		limits:
			'A quick run illustrates the rule, not an emergence-time distribution. Corrected-core counts and verdicts differ and must be reported separately. The historical PCG dial is retained for encounters.',
		source: {
			registration: `${REVISION} · DZ`,
			results: ['results/dial/DIAL.md', 'results/dial/dial_worlds.csv', 'results/tier2/T2_DZ.csv']
		},
		config: { lethalProbability: probability, seed: 9001 },
		published: { seeds: range(9001, 10) }
	})
);

const randomRegisters = preset({
	id: 'random-registers',
	name: 'Random registers',
	category: 'intervention',
	description:
		'Every encounter begins with independently randomized Z80 registers and flags, including alternates and IX/IY. PC and the historical stack position remain fixed.',
	whatToLookFor:
		'The pusher loads its own data, while many compact block copiers depend on the reset’s supplied zeros. Culture a candidate under both reset rules.',
	finding:
		'Load–push designs appeared first in all 20 random-register worlds. None of that initial cohort reached population confinement by 300,000 updates; later, longer runs did produce confined designs.',
	limits:
		'Random registers do not make confinement impossible. Randomizing SP is a separate intervention. The new host’s per-encounter seeds do not recreate archived population streams.',
	source: {
		registration: `${REVISION} · R3 C`,
		results: ['results/conv/REPORT.md', 'results/conv/conv_worlds.csv', 'results/r4/R4_REP.md']
	},
	config: { randomRegisters: true, seed: 8001 },
	published: {
		seeds: range(8001),
		reset:
			'A, F, B, C, D, E, H, L and their alternates are uniform bytes; IX/IY uniform 16-bit values. PC = 0; SP keeps the historical B-end alias.'
	}
});

const geometryPresets = [24, 28].map((ring) =>
	preset({
		id: `ring-l12-p${ring}`,
		name: `Ring geometry · 12 on ${ring}`,
		category: 'intervention',
		description:
			ring === 24
				? 'Two 12-byte tapes fill their native 24-byte ring. Compare the same tape size on a padded ring to separate inherited length from address geometry.'
				: 'Two 12-byte tapes share a 28-byte ring with four zero padding bytes. Padding is reset each encounter and discarded at writeback; SP still aliases B’s last byte, position 23.',
		whatToLookFor:
			'Compare the 01 c5 pusher’s culture and population behavior at P = 24 and P = 28. Adding space changes where stack writes meet instruction reads.',
		finding:
			'Historical L12/P24 worlds had 6 of 10 detections, with Kaplan–Meier median 245,500 updates. P28 had 10 of 10, median 350; all P28 first tapes had period two.',
		limits:
			'The native ring delayed detection; it was not a no-life boundary. Other registered arithmetic predictions failed, and visible repetition need not divide the ring.',
		source: {
			registration: `${PLAN} · Stage F / F4`,
			results: ['results/stageF/stage_f/rings.csv', 'results/stageF/stage_f/NUMBERS_F.md']
		},
		config: { length: 12, ring, seed: 101 },
		published: { seeds: range(101, 10) }
	})
);

export const NO_COPY_ABLATION = [
	'family:ld8',
	'family:ld8-imm',
	'family:ld8-mem',
	'family:ld16-imm',
	'family:ld16-mem',
	'family:ld-special',
	'family:stack',
	'family:ex',
	'family:block-copy'
];

const noBlock = preset({
	id: 'no-block-copy',
	name: 'No block copy',
	category: 'intervention',
	description:
		'Suppress LDI, LDD, LDIR and LDDR, the Z80’s dedicated memory-copy operations. Stack pushes, loads, exchanges and other possible routes remain.',
	whatToLookFor:
		'Look for load–push and exchange/stack-based copiers. Removing a convenient copy loop does not remove every way to reproduce.',
	finding:
		'The historical Stage C condition at budget 128 and k = 4 detected copiers in 10 of 10 worlds, with median detection 350 updates.',
	limits:
		'Suppression consumes opcode and prefix bytes only; old operand bytes may execute next. The result concerns this finite condition, not a minimal instruction set for life.',
	source: {
		registration: `${PLAN} · Stage C2`,
		results: ['src/make_conds.py · ABLATIONS[block-copy]', 'results/stageC/stage_c/NUMBERS_C.md']
	},
	config: { ablation: ['family:block-copy'], seed: 101 },
	published: { seeds: range(101, 10) }
});

const noCopy = preset({
	id: 'no-copy',
	name: 'No-copy ablation',
	category: 'intervention',
	description:
		'The historical strict load/stack/exchange ablation: remove all load families, stack operations, exchanges and block copies. CALL/RST and read-modify-write memory operations remain.',
	whatToLookFor:
		'Orderly products can still accumulate. Test reproduction rather than treating a repeated pattern or a change in diversity as life.',
	finding:
		'No replicator was detected in the 20 historical no-copy worlds across two budget/mutation settings by one million updates.',
	limits:
		'This is a named ablation, not “all writes disabled.” Finite non-detection does not prove that reproduction is impossible.',
	source: {
		registration: `${PLAN} · Stage A / C1`,
		results: ['src/make_conds.py · ABLATIONS[no-copy]', 'results/stageC/stage_c/NUMBERS_C.md']
	},
	config: { ablation: NO_COPY_ABLATION, seed: 101 },
	published: { seeds: range(101, 10), horizon: 1_000_000 }
});

const m6502 = preset({
	id: 'm6502-s1-a',
	name: 'NMOS 6502 · S1 arm A',
	category: 'discovery',
	description:
		'The paper’s primary 6502 world: 16-byte tapes, faithful BRK, JAM halting and a first push onto the last byte of the partner.',
	whatToLookFor:
		'Read the 6502 disassembly: 00 means BRK, not Z80 NOP. This cohort’s detections took much longer than a short browser tour, so use cultures to inspect candidates.',
	finding:
		'All 20 primary first copiers were confined. Only 14 met the additional registered loop requirement, so that prediction was unsupported. Median detection was 1,125,000 updates.',
	limits:
		'The supplied reset and architecture differ from the Z80. These timing differences do not isolate an ISA-only causal effect. The separate eligible retention study is a different assay.',
	source: {
		registration: `${REVISION} · S1, arm A`,
		world: 'src/algocell_exp/machines/worlds_6502.json · m6502_pair_maps_L16',
		results: ['results/s6502/CORRECTION.md', 'results/s6502/v2/arm_comparison.csv']
	},
	config: {
		isa: 'm6502',
		compat: '0.3',
		variant: 'nmos',
		stack: 'first-push',
		dialMode: 'counter',
		seed: 65001
	},
	published: {
		seeds: range(65001),
		horizon: 10_000_000,
		reset:
			'A = X = Y = 0; P = 0x20; PC = 0. The first push lands at B’s last byte. NMOS JAM halts; BRK executes faithfully.'
	}
});

const i8086 = preset({
	id: 'i8086-s2-a',
	name: 'Coherent 8086 · S2 arm A',
	category: 'discovery',
	description:
		'The registered 16-byte 8086 world with coherent fetch, zero-EA hidden-latch abstraction and SP = 1. REP iterations each consume one execution step.',
	whatToLookFor:
		'Inspect variable-length instructions and the stack seam: the first pushed word straddles the end and beginning of the ring. An available pusher does not guarantee that it establishes.',
	finding:
		'Copiers were detected in 14 of 20 worlds: two first copiers were open and twelve confined. The predicted open beginning failed and its registered kill rule was triggered.',
	limits:
		'Coherent fetch does not model a hardware prefetch queue. Both registered evolved-law assays admitted zero eligible candidates and remain untested. L32 was withdrawn as a population arm.',
	source: {
		registration: `${REVISION} · S2, amended arm A`,
		world: 'src/algocell_exp/machines/worlds_8086.json · i8086_pair_maps_L16',
		results: ['results/s8086/S2.md', 'results/s8086/tables/worlds.csv']
	},
	config: {
		isa: 'i8086',
		compat: '0.3',
		variant: 'nmos',
		stack: 'first-push',
		dialMode: 'counter',
		seed: 8086001
	},
	published: {
		seeds: range(8086001),
		horizon: 3_000_000,
		reset:
			'AX, CX, DX, BX, BP, SI, DI, CS, DS, ES, SS and IP = 0; FLAGS = 0xF002; SP = 1. Coherent fetch and hidden_latch = zero_ea; no extra lethal-byte rule.'
	}
});

const competitionPresets = [false, true].map((isLethal) =>
	preset({
		id: `offset-competition-${isLethal ? 'lethal' : 'benign'}`,
		name: `R/T competition · ${isLethal ? 'lethal' : 'benign'}`,
		category: 'demonstration',
		description:
			'Start with a random 50:50 mixture of a partial-copy R design and full-tape T designs at length 32. Every T cell receives its own 28-byte random payload.',
		whatToLookFor:
			'Compare chemistry using T/(R + T), absolute core prevalence and diversity. Exact top-tape share alone can hide a diverse transmitting population.',
		finding:
			'Corrected-core equal-start competitions ended with median full-tape shares of 0.05 under benign chemistry and 0.76 under lethal chemistry; all ten lethal worlds ended above one half.',
		limits:
			'This is seeded competition, not spontaneous discovery. R and T differ in payload background as well as displacement. The finite result does not prove equilibrium or selection for capacity alone.',
		source: {
			registration: `${REVISION} · E / E-L32L, corrected-core rerun`,
			world: 'src/algocell_exp/machines/worlds_tier2.json',
			results: [
				'src/offset_switch.py',
				'results/tier2/scored/E/REPORT.md',
				'results/tier2/scored/E/offset_switch.csv'
			]
		},
		config: {
			compat: '0.3',
			length: 32,
			ring: 64,
			seed: isLethal ? 7151 : 7101,
			lethalProbability: Number(isLethal)
		},
		published: {
			seeds: range(1, 10),
			note: 'Equal-start condition seeds 1–10; historical soup seed = 7100 + 50 × lethal + condition seed. Initial tails were drawn by NumPy seeded [soup_seed, 99]. Browser planting uses its separately identified host stream.'
		},
		initialCondition: {
			kind: 'competition',
			regenerator: [0x04, 0x5e, 0xed, 0xb0],
			transmitterCore: [0xa0, 0x5e, 0xed, 0xb0],
			share: 0.5,
			randomPayload: true
		}
	})
);

const displacementPresets = [4, 32].map((distance) => {
	const bytes = Array.from({ length: 32 }, (_, i) => [0x04, 0x5e, 0xed, 0xb0][i % 4]);
	bytes[0] = distance === 4 ? 0x04 : 0x60;
	bytes[8] = 0x7f;
	return preset({
		id: `displacement-${distance}`,
		name: `Displacement · ${distance === 4 ? '4-byte prefix' : 'full tape'}`,
		category: 'demonstration',
		description: `A seeded 32-byte copier with a 7F marker at position 8 and mutation disabled. Its first byte sets displacement ${distance}; ${distance === 4 ? '04 5E ED B0 regenerates the four-byte prefix' : '60 5E ED B0 sets 96 mod 64 = 32, a full-tape distance'}.`,
		whatToLookFor:
			'Culture the selected tape immediately, before stepping the population, and inspect the partner output. Compare the same position-8 marker in short-prefix and full-tape copying.',
		finding:
			'For the qualifying forward-copy family, a displacement d and held four-byte core leave d − 4 independently free positions: zero here at d = 4, and 28 at d = 32.',
		limits:
			'This is a constructed marker demonstration of an archived mechanism. The conditional law requires coverage, alignment, intact machinery and no interfering writes. Marker equality alone is not a causal ancestry trace.',
		source: {
			registration: `${REVISION} · C1 displacement interventions`,
			results: [
				'results/capacity/CAPACITY.md',
				'results/capacity/edit_eligibility.csv',
				'manuscript/MAIN_v9.md · Conditional copying law'
			]
		},
		config: { length: 32, ring: 64, mutationExponent: 31 },
		published: {
			width: null,
			height: null,
			pairAttempts: null,
			mutationAttempts: null,
			horizon: 8,
			horizonUnit: 'transfers',
			seeds: [],
			note: 'An isolated mechanism demonstration; eight serial generations in the cited study are not eight population updates. The app’s planted marker is illustrative, not an archived population. Mutation is disabled in the browser demonstration.'
		},
		initialCondition: { kind: 'tiled', bytes }
	});
});

const pusher = preset({
	id: 'pusher',
	name: 'The load–push pusher',
	category: 'demonstration',
	description:
		'A seeded culture of 01 C5 repeated across 16 bytes. The same bytes supply both instructions and literal data: load a word, then push it towards the partner’s end.',
	whatToLookFor:
		'Run the culture test and inspect fetched positions. The pusher can reproduce while executing into the partner; continued reproduction is a different measurement from carrying a variant.',
	finding:
		'In the registered branching test, 48 of 64 unmutated pusher populations survived eight generations. None of 512 tested mutation types met the majority-carriage criterion.',
	limits:
		'The planted population is an example, not discovery. Failure of the carriage criterion does not mean every marker copy vanished. The app’s short culture is not the full registered branching protocol.',
	source: {
		registration: `${REVISION} · R3 B / X1`,
		results: ['results/branching/branching_summary.csv', 'results/branching/SUPP_TABLE.md']
	},
	config: { mutationExponent: 31 },
	published: {
		width: null,
		height: null,
		pairAttempts: null,
		mutationAttempts: null,
		horizon: 8,
		horizonUnit: 'transfers',
		seeds: [],
		note: 'Frozen tape example; the cited branching assay has eight generations, two partners per surviving member and a population cap of 32. Mutation is disabled in the browser demonstration.'
	},
	initialCondition: { kind: 'tiled', bytes: [0x01, 0xc5] }
});

const returnCloser = preset({
	id: 'return-closer',
	name: 'The return closer',
	category: 'demonstration',
	description:
		'A seeded culture of AD E3 21 E3 21 C0 AD C0 repeated twice. This return-containing design is a separate evolved example from the historical 16-byte worlds.',
	whatToLookFor:
		'Compare its culture and fetch map with the pusher. Reliable copying and confined execution still do not tell you which introduced differences survive.',
	finding:
		'The return closer retained all 64 unmutated populations in the branching assay but scored no transmissible site under its declared criterion.',
	limits:
		'Do not infer a direct pusher-to-return ancestry from these two examples. No scored site is not proof that every working core variant or rare marker is impossible.',
	source: {
		registration: `${REVISION} · R3 B / X1`,
		results: ['results/stageG/stageG/stage_g_runs.csv', 'results/branching/branching_summary.csv']
	},
	config: { mutationExponent: 31 },
	published: {
		width: null,
		height: null,
		pairAttempts: null,
		mutationAttempts: null,
		horizon: 8,
		horizonUnit: 'transfers',
		seeds: [],
		note: 'Frozen tape example and branching-assay result; this app’s seeded lattice is an explanatory demonstration. Mutation is disabled in the browser demonstration.'
	},
	initialCondition: { kind: 'tiled', bytes: [0xad, 0xe3, 0x21, 0xe3, 0x21, 0xc0, 0xad, 0xc0] }
});

export const PAPER_PRESETS: PaperPreset[] = [
	stageG,
	corrected,
	lethal,
	...dialPresets,
	randomRegisters,
	...geometryPresets,
	noBlock,
	noCopy,
	m6502,
	i8086,
	...competitionPresets,
	...displacementPresets,
	pusher,
	returnCloser
];

export function getPaperPreset(id: string): PaperPreset {
	return PAPER_PRESETS.find((entry) => entry.id === id) ?? stageG;
}

export function clonePresetSettings(entry: PaperPreset): LabSettings {
	return { ...entry.config, ablation: [...entry.config.ablation] };
}
