export interface DocumentationLink {
	label: string;
	href: string;
}
export interface DocumentationSection {
	id: string;
	title: string;
	paragraphs: string[];
	steps?: string[];
	links?: DocumentationLink[];
}

export const ABOUT_SECTIONS: DocumentationSection[] = [
	{
		id: 'copying-and-inheriting',
		title: 'Copying is not inheriting',
		paragraphs: [
			'Imagine a printing press that builds another press, yet always prints the page engraved on its own rollers. It can fill a room with copies while forgetting every change made to the page. Algocell asks a related question: when a program reproduces, which differences can its offspring preserve?',
			'Each cell holds a short tape of machine code. Neighbouring tapes meet in shared circular memory. A processor executes the first tape, both tapes keep the resulting writes, and random byte replacements introduce variation. There is no assigned goal, fitness score or rule that rewards a successful tape. Programs spread by writing material that can reproduce in later encounters.'
		]
	},
	{
		id: 'four-measurements',
		title: 'Different questions need different tests',
		paragraphs: [
			'Visible order describes composition. Culture asks whether offspring continue producing similar material against fresh partners. A fetch trace asks whether instruction-stream reading leaves the focal tape. A marker comparison asks whether an introduced byte value survives serial transfer. None of these measurements can stand in for the others.',
			'A copier may preserve only a short pattern and erase changes elsewhere. Another may preserve bytes that it never executes. Confinement does not establish independence from partner data, and value matching does not establish causal ancestry. Partner sensitivity holds the parent and reset fixed while changing one initial partner byte; even an untouched partner byte can survive without being read.',
			'The instrument does not establish adaptive use of a payload, unlimited heredity or open-ended evolution. Short local observations are finite samples under declared conditions. A failed short test does not prove that a mechanism is impossible.'
		]
	},
	{
		id: 'zilion',
		title: 'The machine and its world',
		paragraphs: [
			'Algocell runs locally on WebGPU. Zilion 0.3.1 library mode supplies processor semantics inside the resident population kernel. Available cores are Z80 compatibility 0.2, corrected Z80 0.3 classic and Zilog NMOS, NMOS 6502, and coherent-fetch 8086. The reset state, ring geometry, execution budget and chemistry are part of the experiment.',
			'The coherent-fetch 8086 reads current memory after writes. It does not simulate the hardware prefetch queue or bus timing of an 8088. Different instruction sets, variants and reset conventions are different worlds, even when their tapes begin with the same bytes.',
			'Culture and encounter tests use a separate machine on demand. Their exported evidence includes settings and core/world manifests. A content hash identifies a contract; it does not certify a scientific conclusion. Parallel GPU scheduling means a shared seed does not promise an identical future trajectory.'
		],
		links: [{ label: 'Zilion source and documentation', href: 'https://github.com/NeoVand/zilion' }]
	},
	{
		id: 'credits',
		title: 'People, precedents and local work',
		paragraphs: [
			'Algocell and Zilion are by Mostafa ‘Neo’ Mohsenvand at the MIT Media Lab. This work draws on digital evolution, including von Neumann’s self-reproduction, Tierra, Avida, the interacting random programs studied by Blaise Agüera y Arcas and colleagues, and the Z80 primordial soups studied by Cicala and colleagues.',
			'The interface preserves Algocell’s full-screen byte lattice, dark instrument panels and familiar colour maps. Spatial positions remain real lattice positions. Observation tools add ways to inspect the same world, without replacing it with an abstract dashboard.',
			'Simulation, recordings and checkpoint handling remain in your browser. No account or remote simulation service is required. Files are downloaded locally; links open only when you choose them.'
		],
		links: [
			{ label: 'Algocell source', href: 'https://github.com/NeoVand/algocell' },
			{ label: 'Computational Life (2024)', href: 'https://arxiv.org/abs/2406.19108v2' },
			{
				label: 'Coevolution of self-replication and function (2026)',
				href: 'https://arxiv.org/abs/2607.09211v2'
			}
		]
	}
];

export const GUIDE_SECTIONS: DocumentationSection[] = [
	{
		id: 'first-five-minutes',
		title: 'Choose one question',
		paragraphs: [
			'Random-code experiments begin without a planted copier. Demonstrations begin with known tapes so you can inspect a mechanism immediately. Starting with a demonstration is useful for learning the tools; it is not evidence of spontaneous discovery.'
		],
		steps: [
			'Open Experiments and choose Full-tape copier. Read its condition and what to look for, then choose Start new world with these settings.',
			'Select Bytes and zoom in. Open Probe, select a cell and inspect its tape.',
			'Open the encounter bench in Measure and run a culture. Compare first and second transfers, then inspect the fetched bytes, instruction starts and writes. The live world keeps running.',
			'Compare a changed byte with the original parent in the encounter bench. Record the result, including its seed, position and value.',
			'Try Short-prefix copier with the same marker position. Ask whether copying and retention give the same answer.'
		]
	},
	{
		id: 'controls-and-time',
		title: 'Run, pause and reset',
		paragraphs: [
			'One population update proposes a batch of neighbouring pairs, executes accepted encounters, returns writes and attempts mutations. It is not a synchronized generation. Pair attempts and instruction slots include inactive pairs and early stops; throughput is not a count of executed hardware instructions.',
			'Play schedules updates; Pause stops new work; Step advances one update. Restart begins the applied condition again. Core, mutation and chemistry changes apply to the existing tapes. Seed and geometry edits are staged until you choose Start new world with these settings. Tabs, views and tools preserve playback. Hidden tabs suspend GPU work while keeping your playback choice.'
		]
	},
	{
		id: 'machine-controls',
		title: 'Set the machine and geometry',
		paragraphs: [
			'Tape length L counts inherited bytes. The encounter ring P contains A followed by B, plus zero-initialized discarded padding if P exceeds 2L. Address wrapping changes with P. Stack top and first-push placement are different reset conventions; in padded rings, the stack is placed relative to B rather than the end of padding.',
			'Random registers change the supplied state at the beginning of each encounter. Mutation independently attempts random byte replacements. With 8,192 pair attempts and mutation exponent 4, an update attempts 512 replacements, not one sixteenth of all bytes. The off setting disables population mutation.',
			'Instruction ablation suppresses decoded operations through a per-ISA table. Suppressed opcode and prefix bytes are consumed; remaining operands may become instructions. Static disassembly describes ordinary decoding, so use the measured starts and fetches to inspect execution under ablation.'
		]
	},
	{
		id: 'chemistry',
		title: 'Make zero dangerous',
		paragraphs: [
			'The zero-fetch probability applies when 00 is the first byte of an instruction. Zero leaves its ordinary behaviour intact; one stops every qualifying encounter. Intermediate probabilities make a new decision at each qualifying fetch. A zero operand or data read is not automatically lethal. Ordinary 00 is NOP on Z80 and BRK on 6502.',
			'Compare benign and lethal conditions with the same displayed seed, geometry and budget when you want a controlled local comparison. Preset cohort seeds may differ. Observe zero fraction and test candidates separately; zero-rich material alone says nothing about reproduction.'
		]
	},
	{
		id: 'field-modes',
		title: 'Read the six field views',
		paragraphs: [
			'Bytes preserves the original palette and byte mosaics. The same byte colour does not mean the same instruction across processors. Tape identity assigns a stable fingerprint colour to complete tapes; colours and hashes can collide, so use exact region counts to verify population composition.',
			'Open / confined shows the last measured initiating encounter: unknown, no observed departure, or instruction-stream fetching outside A into B or padding. Modified tapes lose their observation. It does not describe a permanent property of a lineage or measure partner-data reads.',
			'Zero density measures zero bytes divided by inherited tape length. Local diversity counts distinct tapes in the valid 3×3 window of lattice coordinates and divides by the number of cells in that window. It measures local mixing, not fitness.',
			'Marker matches highlights a chosen value at a chosen offset. The same value may already be present or arise independently. Use the paired marker assay to compare marked and original parents; the field is not a descendant tracker.'
		]
	},
	{
		id: 'probe-region',
		title: 'Probe a cell or measure a region',
		paragraphs: [
			'Wheel or pinch to zoom; drag to pan; Fit returns to the whole lattice. Probe keeps the selected cell visible and reads only its small tape. Use an address or arrow keys when pointing is inconvenient. Open the tape inspector for per-ISA disassembly and tests.',
			'Region selects a rectangle in lattice coordinates. A sample counts exact complete sequences, zero fraction and marker matches without changing playback. Endpoints are inclusive. The update stamp identifies the sampled population; the result does not silently follow later changes.'
		]
	},
	{
		id: 'encounter-bench',
		title: 'Look inside an encounter',
		paragraphs: [
			'Fetched bytes include prefixes, operands and displacements. Instruction starts are a smaller set. A byte may be instruction, operand or data on different paths; static disassembly and a measured trace answer different questions. Self-modification can make the executed stream differ from the initial tape.',
			'Write counts include repeated writes and writes of an unchanged value. A/B totals are not counts of distinct changed bytes. Budget exhaustion, HALT, lethal-byte stops and traps are reported separately. Padded addresses belong to the encounter ring and are discarded after execution.',
			'The self-pair test executes the selected parent against an identical partner. Preservation of this pair is distinct from reproduction against fresh partners. Partner sensitivity flips the low bit at one position in each fresh partner and compares matched outputs with identical reset and chemistry seeds. Output differences may reflect simple persistence of an original partner byte; no difference in a finite sample does not prove independence.'
		]
	},
	{
		id: 'culture',
		title: 'Culture and marker transfer',
		paragraphs: [
			'Culture runs the selected parent against fresh uniform random partners with no population mutation. Every resulting B tape becomes a parent in the next transfer, including unsuccessful copies. Exact-copy frequency and best cyclic similarity are different measurements.',
			'Culture gain is normalized improvement in similarity to the original parent, excluding partners already at least 75% similar. Second-transfer gain of at least 0.3 passes this operational screen. No informative partners means unknown. Confinement summarizes sampled fetches: none outside A is confined, at least half outside is open, and the rest is intermediate.',
			'Marker tests compare the original and changed parent against matched fresh partners. Retention is the fraction of outputs with the chosen value at the chosen position in each transfer. This controls for background matching but is not ordered-write ancestry or a branching inheritance assay. Changing a machinery byte can disable the copier.'
		]
	},
	{
		id: 'visual-records',
		title: 'History, comparison and recordings',
		paragraphs: [
			'History is an optional, bounded collection of view images sampled at most once per second. Scrubbing or replaying shows those observations, with their update labels. The world continues if it was playing; history does not rewind it. Return live to see the present population.',
			'Pin a reference to compare its captured appearance with the live field. Camera or display changes can make the images unlike; read the reference mode and update. This is not an independent control world.',
			'Snapshot downloads a PNG. Timelapse exports sampled frames through browser media support; it records visible observations rather than every update. Save a checkpoint to keep actual population bytes, and export assay evidence to keep settings, results and manifests.'
		]
	},
	{
		id: 'exports',
		title: 'Keep the condition with the evidence',
		paragraphs: [
			'A preset specifies how to start; a checkpoint saves a population. The .acell format is a viewable PNG with attached compressed data. It retains current tape bytes, geometry and machine settings. It does not promise identical future GPU scheduling or preserve every random lane’s continuation state.',
			'Assay JSON stores the selected tape, settings, seed, sample size, measured records and Zilion core/world manifest. Retain these with any comparison. Files remain local; no checkpoint is uploaded.'
		]
	},
	{
		id: 'support',
		title: 'Support and shortcuts',
		paragraphs: [
			'Execution needs a WebGPU-enabled browser on localhost or HTTPS. Documentation remains available if the device cannot be created. Try a browser with WebGPU support if initialization fails.',
			'Space toggles play and . steps. V selects navigation, P probes, R selects a region, M marks a byte and C compares. Keys 1–5 open World, Measure, Experiments, Guide and About; ? shows shortcuts. F fits; + and − zoom. Cmd/Ctrl-S saves a checkpoint and Cmd/Ctrl-O opens one. Single-key shortcuts yield to text inputs. Touch users can pinch, drag and use the labelled controls.'
		]
	}
];
