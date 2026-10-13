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
			'Imagine a printing press that builds another press, yet always prints the page engraved on its own rollers. It can fill a room with copies while forgetting every change made to the page. Algocell asks a related question in an inspectable digital world: when a program reproduces, which differences can its offspring preserve?',
			'Each cell holds a short tape of machine code. Neighbouring tapes meet in shared circular memory. A processor executes the first tape, both tapes keep the resulting writes, and random byte replacements introduce variation. There is no assigned goal, fitness score or rule that rewards a successful tape. Programs spread by writing material that can reproduce in later encounters.'
		]
	},
	{
		id: 'paper',
		title: 'From random code to heritable variation',
		paragraphs: [
			'The companion manuscript, “From random code to heritable variation in digital primordial soups,” by Mostafa ‘Neo’ Mohsenvand, follows reproduction and transmission separately. Orderly material can accumulate without detectable reproduction. Reproducing populations can persist while introduced variants fail a carriage test. Some copiers regenerate a short pattern; others preserve variable sequences that they never execute.',
			'Within a specified family of forward copiers, the distance between reading and writing determines which parental positions remain independently free to vary. A byte in the machinery can set that distance and itself be inherited. Environmental interventions change the prevalence of different designs. These are findings under declared digital contracts, not a universal sequence that every soup must follow.',
			'The paper does not demonstrate adaptive use of the transmitted payload, selection for increased payload capacity, unlimited heredity or open-ended evolution. The browser presets cite source conditions and keep these limits visible. Your live run is a new realization, not a replay of an archived population. A public paper accession is not yet linked here.'
		]
	},
	{
		id: 'four-measurements',
		title: 'Four questions, four measurements',
		paragraphs: [
			'A culture asks whether a tape produces offspring that reproduce against fresh partners. A fetch trace asks whether instruction-stream reading enters the partner: execution is “confined” when it stays in the focal tape and “open” when it enters its partner. Inflow fixes the parent, varies partners and measures changes in the offspring. A transfer test introduces a parental marker and follows it separately from the survival of its carriers.',
			'These questions are not interchangeable. A program can read partner data without executing it, or execute partner addresses after overwriting their original bytes. A confined copier can erase parental variation. Exact-tape abundance and diversity describe population composition; neither is a measure of heredity. Full inflow, branching transfer and byte-ancestry workflows remain part of the next instrument phase.'
		]
	},
	{
		id: 'zilion',
		title: 'The instrument: Zilion',
		paragraphs: [
			'Zilion separates processor semantics from the world that supplies memory, reset state and execution time. This app uses its 0.3 engine and world API, with frozen Z80 compatibility 0.2, corrected Z80 compatibility 0.3, the Zilog NMOS variant, NMOS 6502 and coherent-fetch 8086. A supplied zero register can matter as much as a program’s own code, so the world contract is part of every experiment.',
			'The coherent-fetch 8086 reads current memory after writes and retains decoded REP state across its iterations. It does not model an Intel hardware prefetch queue or bus timing. The paper’s historical BFF experiments use a separate research implementation; BFF is not an exported core in the npm package and is not offered as a runnable ISA here.',
			'Processor validation and ecological evidence are different checks. SingleStepTests supplies large instruction-test collections; the 8088 collection is hardware recorded, while the Z80 and 6502 collections use reference emulators. The documented contracts preserve expected differences instead of treating every comparison as the same kind of evidence. Core and world manifests carry SHA-256 identities; an identity tells you what was run, not whether a scientific claim follows from it.'
		],
		links: [
			{ label: 'Zilion source and documentation', href: 'https://github.com/NeoVand/zilion' },
			{ label: 'SingleStepTests Z80', href: 'https://github.com/SingleStepTests/z80' },
			{ label: 'SingleStepTests 6502', href: 'https://github.com/SingleStepTests/65x02' },
			{ label: 'SingleStepTests 8088', href: 'https://github.com/SingleStepTests/8088' }
		]
	},
	{
		id: 'credits',
		title: 'People, precedents and local work',
		paragraphs: [
			'Algocell and Zilion are by Mostafa ‘Neo’ Mohsenvand at the MIT Media Lab. This research builds on digital evolution, including von Neumann’s self-reproduction, Tierra and Avida; the interacting random programs of Blaise Agüera y Arcas and colleagues; and the Z80 primordial soups of Cicala and colleagues. Earlier work established load–push and block-copy architectures and faithful copying of random tails. The present research asks what introduced differences those mechanisms preserve.',
			'The interface’s visual language follows the manuscript’s figure plan: a white field, tangible byte blocks, quiet structural lines and separate encodings for byte value, execution and transmission. It preserves lattice positions so that a visible colony corresponds to actual neighbours.',
			'Simulation and file handling run locally in your browser, with no telemetry, account or remote simulation service. Outbound reference links open only when you choose them. The legacy instrument remains available for its original rendering, controls and checkpoints.'
		],
		links: [
			{ label: 'Algocell source', href: 'https://github.com/NeoVand/algocell' },
			{
				label: 'Agüera y Arcas et al. — Computational Life (2024)',
				href: 'https://arxiv.org/abs/2406.19108v2'
			},
			{
				label: 'Cicala et al. — Coevolution of self-replication and function (2026)',
				href: 'https://arxiv.org/abs/2607.09211v2'
			}
		]
	}
];

export const GUIDE_SECTIONS: DocumentationSection[] = [
	{
		id: 'first-five-minutes',
		title: 'Your first five minutes',
		paragraphs: [
			'Start with one question. Historical Stage G begins with random code; the pusher and return closer begin with known tapes so you can inspect copying immediately. Discovery presets are stochastic. Demonstrations plant their examples deliberately.'
		],
		steps: [
			'Choose a preset and read its description, what to look for and finding. Its source details separate the paper’s condition from this app’s host scheduler.',
			'Press Run to advance the world, Pause to keep its current state, or Step for one population update. Each run is bounded to a short burst; the update counter measures completed work, not seconds.',
			'Select a tape on the lattice or by cell index. Inspect the hexadecimal bytes and the disassembly for the selected processor.',
			'Pause and run the culture test. Compare first- and second-generation reproduction, then examine the example encounter’s fetched bytes and output.',
			'Switch the lattice’s visual layer or open the measurements. Ask what each display actually measures before inferring reproduction or inheritance.'
		]
	},
	{
		id: 'controls-and-time',
		title: 'Run, step, reset and time',
		paragraphs: [
			'One population update proposes a batch of neighbouring pairs, runs accepted encounters, returns writes and attempts mutations. It is not one generation: cells can participate at different times and there is no synchronized parent-to-offspring census. One encounter step is a processor execution step; Z80 block repeats and 8086 REP consume their budgets one iteration at a time.',
			'Pause stops scheduling new updates. Reset creates a new population from the current condition. Changing a machine or geometry starts a new run so measurements from unlike worlds are not silently combined. The interface pauses when hidden and bounds continuous GPU work to 60 seconds or less. You can begin another burst when you are ready.',
			'The seed identifies this app’s random initialization and host stream. Published GPU worlds contain scheduling and packed-mutation races, so equal seeds do not promise an identical historical trajectory. Record the full settings and manifest when comparing runs.'
		]
	},
	{
		id: 'machine-controls',
		title: 'Choose the processor and reset',
		paragraphs: [
			'ISA selects the instruction vocabulary. Z80 compatibility 0.2 preserves the historical core; 0.3 selects the corrected core. The Zilog NMOS variant is a separate 0.3 option. NMOS 6502 and coherent-fetch 8086 have their own registers, flags and stack behavior. Changing the ISA changes what every byte means.',
			'Stack top and first push are different conventions. The historical Z80 top aliases the final byte of the partner, then decrements before storing. The 6502 first push lands on that final byte. In the paper’s 8086 ring, first-push placement gives SP = 1 and the first pushed word crosses the ring seam. A legacy SP = 65535 setting is a distinct convention on rings that do not divide 65536.',
			'Random registers alter the state supplied at the start of every encounter; they do not mutate tape bytes. The historical Z80 intervention randomizes main and alternate registers and IX/IY while keeping PC and SP fixed. A different reset or random stream is a different experiment. The budget limits execution even if a program would otherwise keep running.'
		]
	},
	{
		id: 'world-controls',
		title: 'Set the world',
		paragraphs: [
			'Width × height is the number of tapes. Tape length L is the number of inherited bytes per cell. The pair ring P is at least 2L; extra padding starts at zero in each encounter and is discarded on writeback. In a padded paper world, the stack aliases the end of B, not the end of the padding.',
			'Pair attempts are proposed pair slots per update, not guaranteed active encounters. Collision claims prevent a cell from participating in two accepted encounters in the same update. The paper presets use a square lattice with reflecting boundaries. Hexagonal legacy worlds have a different neighbourhood and are not the registered paper condition.',
			'Mutation exponent k sets approximately pair attempts / 2ᵏ random byte-replacement attempts per update. At 8,192 attempts and k = 4, that is 512 replacements, not 1/16 of all soup bytes. A replacement can select the same value. The off setting disables mutation. Mutation is independent of whether a program reproduces.',
			'The paper presets list the full 160 × 125 population and 8,192 pair slots. If you change the size for an interactive run, it becomes a modified condition. Smaller worlds can be useful for exploration but do not have the same arrival probabilities or population statistics.'
		]
	},
	{
		id: 'chemistry-and-ablation',
		title: 'Chemistry and instruction ablation',
		paragraphs: [
			'The lethal-zero dial applies when 00 is fetched as the first byte of an instruction. Probability zero leaves the byte’s ordinary behavior intact; probability one halts every such encounter before that instruction executes. Intermediate values draw a new decision at every qualifying fetch. A zero operand or data read is not automatically lethal. On the 6502, ordinary 00 is BRK; on the Z80 it is NOP.',
			'Ablation suppresses decoded opcode operations through a per-run table. It does not delete bytes or preserve the full original instruction’s length: suppressed opcode and prefix bytes are consumed, while remaining operand bytes can become later instructions. The static disassembly shows the ordinary ISA decoding; under ablation, use the measured instruction starts and fetches to see the actual path.',
			'No block copy removes only LDI, LDD, LDIR and LDDR. The named no-copy ablation removes load, stack, exchange and block-copy families; CALL/RST and read-modify-write operations still write memory. Neither name is a theorem about every possible reproducer. Each ISA has its own opcode table; a Z80 family selection cannot be silently carried into another ISA.'
		]
	},
	{
		id: 'lattice',
		title: 'Read the lattice',
		paragraphs: [
			'Byte view uses a stable color for every byte value and makes zero nearly white. Each tape occupies a small block at its actual cell position. Unused display pixels in a non-square tape block are white; they are not additional inherited bytes. Spatial order is preserved, so a spreading texture corresponds to real lattice neighbours.',
			'Identity view colors exact complete tape sequences. An abundant sequence is not automatically a functional species or a replicator. A transmitting population can contain many distinct payloads and therefore have no very common exact tape.',
			'The fetch layer records the last measured initiating encounter. “Partner entered” means that instruction-stream fetching left A; “no observed entry” means it did not in that encounter. Unknown means no current measurement. Mutated or modified cells can lose their measurement. This layer does not measure partner-data reads, partner-information independence or heredity.'
		]
	},
	{
		id: 'inspector',
		title: 'Use the tape inspector',
		paragraphs: [
			'Select a cell to read its address, bytes and ISA-aware disassembly. Hexadecimal is base 16: 10 means sixteen, and 60 means ninety-six. Offsets count from zero. A byte can be an opcode on one execution path, an operand on another, and data read by a copier.',
			'Instruction starts and all fetched bytes answer different questions. Fetch maps include prefixes, immediates and displacements. Static disassembly is a reading of the current tape; a self-modifying program can execute a different stream. Unknown or truncated instructions must remain visibly uncertain rather than inventing bytes beyond the selected tape.',
			'The example culture encounter exposes the same engine’s before/after memory, fetched positions, starts, write counts, state and stopping reason. A budget stop differs from a HALT, lethal-byte stop or trap. A write count is not a count of distinct changed bytes: repeated and identical writes still count.'
		]
	},
	{
		id: 'culture',
		title: 'Test reproduction in culture',
		paragraphs: [
			'The selected tape runs against fresh uniformly random partners with population mutation disabled. Its resulting partner halves become the next parents, each tested against another fresh partner. Unsuccessful first-generation outputs remain in the second-generation test; selecting only successful copies would answer a different question.',
			'Exact-copy frequency counts offspring identical to the parent in the original phase. Best cyclic similarity allows rotations and reports another quantity. The paper’s culture score is normalized gain in similarity to the original selected tape, averaged over partners that were not already at least 75% similar. Generation-two gain of at least 0.3 passes that operational screen. No informative partners means unknown, not sterile.',
			'Always retain the number of partners, seed, budget, machine and chemistry with the result. A short sample can miss rare behavior. In the paper’s 16-trace convention, no partner entry in all 16 is confined, entry in at least eight is open, and one to seven is intermediate. A sampled result does not prove behavior against every possible partner.',
			'A positive culture measures continuing production of similar material. It does not establish retention of an introduced variant, adaptive function or biological life. The registered branching protocol is larger and has its own offspring and carriage rules.'
		]
	},
	{
		id: 'measurements',
		title: 'Interpret the three charts',
		paragraphs: [
			'Zero fraction is the number of zero bytes divided by all inherited tape bytes; padding is excluded. It can rise because reset-dependent instructions generate inert material. A high value is not evidence of reproduction.',
			'Top-tape share is the count of the most abundant exact sequence divided by the cell count. Diversity is the number of distinct exact sequences divided by the cell count, with the absolute unique count shown alongside it. Identical values are counted by their bytes, not by a color or a short hash.',
			'The horizontal axis is population updates. These are descriptive composition measurements, not fitness, complexity, heredity or progress. In seeded R/T competition, interpret the separately named motif-class share and absolute core prevalence; exact top-tape share cannot substitute for them. A ratio with no R or T cells is undefined, not zero.'
		]
	},
	{
		id: 'experiment-zero',
		title: 'Experiment: make zero dangerous',
		paragraphs: [
			'Changing the zero-fetch rule intervenes on the encounter chemistry. The historical benign and lethal presets differ in their source cohort seeds, so set the same displayed seed yourself if you want a controlled browser comparison. These remain new host-scheduled worlds.'
		],
		steps: [
			'Run Historical Stage G for a bounded burst, note its zero fraction and culture a common candidate.',
			'Select Lethal zero, use the same geometry, seed and budget, and run another bounded burst.',
			'Compare the measurements and fetched positions. Try an intermediate dial setting to see how risk is applied per zero fetch.',
			'Keep finite observations separate from the paper’s multi-world findings. One failed culture or a short run without a copier is not an impossibility result.'
		]
	},
	{
		id: 'experiment-marker',
		title: 'Experiment: follow a marked byte',
		paragraphs: [
			'The displacement demonstrations place 7F at parent position 8 and turn off population mutation. A short-prefix copier overwrites original material beyond its protected prefix; a full-tape copier can preserve it under the declared contract. Both mechanisms can remain execution-confined.',
			'The first byte is both instruction and data. At length 32, changing 04 to 60 gives displacement 96 mod 64 = 32 while preserving a viable instruction. Simply writing the numeric length into the first byte can change the executed instruction and break the mechanism.',
			'Inspect the marker’s value in the offspring and compare it with a wild-type control. Value equality or a short marked sequence is not a complete causal ancestry measurement: the same value can arise independently. Full ordered-write ancestry and registered branching carriage remain future workflows.'
		],
		steps: [
			'Choose Displacement · 4-byte prefix. Before stepping the population, select any cell and run its culture.',
			'Inspect the partner output at the corresponding position. The marker lies outside the protected four-byte prefix.',
			'Choose Displacement · full tape and repeat the culture with the same assay settings. The marker now lies inside the copied full-length seed.',
			'Compare marker fate separately from the culture score. The conditional count d − 4 describes free positions with held machinery, not useful genes or all working core alternatives.'
		]
	},
	{
		id: 'experiment-geometry',
		title: 'Experiment: change the ring, keep the tape',
		paragraphs: [
			'Compare the 12-on-24 and 12-on-28 presets. Both inherit 12-byte tapes, but the second encounter has four discarded padding bytes. Address wrapping and stack collisions change even though tape length does not. Keep reset and budget fixed.',
			'The historical experiment found much earlier detection on the padded ring. It did not establish that the native ring could never produce a copier. If a fresh browser run differs, record it rather than treating the preset description as a required script.'
		]
	},
	{
		id: 'exports',
		title: 'Save a condition and its population',
		paragraphs: [
			'A preset describes how to begin; a checkpoint stores the current population. Export a checkpoint before changing conditions. The new format identifies its schema, complete tape bytes, geometry, settings, scheduler/RNG continuation, counters and machine manifest. An imported checkpoint must pass format and shape checks before replacing the current run.',
			'The manifest separates core and world SHA-256 identities. Runtime budget, ablation masks, seeds and host scheduling also matter, so keep them with the manifest. A content hash is useful provenance, not a guarantee of scientific validity or identity with the paper’s frozen export.',
			'Legacy .acell files belong to the legacy instrument. Their old metadata lacks the complete machine and continuation state required by the new schema. Files remain on your computer; no checkpoint is uploaded to a remote service.'
		]
	},
	{
		id: 'legacy',
		title: 'The legacy instrument',
		paragraphs: [
			'The legacy view preserves the original optimized Z80 app while the research instrument is refactored. Its controls include square and hex layouts, 1×/2×/4×/8× speed, wheel/pinch zoom, drag pan, fit, averaged cells, Rainbow/Ocean/Thermal palettes, brightness, contrast, saturation and grid lines. Settings include user presets and Z80 suppression patterns.',
			'Legacy shortcuts are Space to toggle running, R to reset, H for help, Cmd/Ctrl-S to save and Cmd/Ctrl-O to open. Its .acell drag-and-drop and original byte, exact-tape and diversity tabs remain available. Those historical displays and its approximate CPU trace should not be mistaken for the new engine’s measured multi-ISA assays.'
		]
	},
	{
		id: 'support-and-glossary',
		title: 'Support and a short glossary',
		paragraphs: [
			'Execution needs a WebGPU-enabled browser on localhost or HTTPS. If a device cannot be created, the status message should explain the failure; presets and documentation remain readable. Retry after correcting browser support. No account or cloud worker is required.',
			'A tape is one cell’s inherited byte sequence. An encounter joins two tapes for execution. An update proposes many encounters and mutations. A genotype is an exact complete sequence. A culture is a repeated-reproduction assay. Confinement describes instruction-stream addresses. Inflow describes dependence on the original partner. Retention describes an introduced difference surviving a declared test. These definitions keep observation separate from interpretation.'
		]
	}
];
