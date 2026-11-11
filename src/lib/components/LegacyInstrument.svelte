<script lang="ts">
	import { GPUEngine } from '$lib/gpu/engine';
	import {
		DEFAULT_SEED,
		DEFAULT_NOISE_EXP,
		MAX_BATCH_PAIR_N,
		Z80_STEPS,
		type GridType,
		type GridConfig
	} from '$lib/sim/constants';
	import {
		disassemble as disassembleZ80,
		byteToMnemonic as z80ByteToMnemonic
	} from '$lib/z80-disasm';
	import { disassembleTape } from '$lib/lab/disassemble';
	import { ablationInstructions, ablationMasks } from '$lib/lab/ablation';
	import {
		PAPER_PRESETS as EXPERIMENT_PRESETS,
		type PaperPreset as ExperimentPreset
	} from '$lib/lab/presets';
	import InstrumentSelect from '$lib/components/InstrumentSelect.svelte';
	import type { FieldMode } from '$lib/observation';
	import ObservationTools from '$lib/components/ObservationTools.svelte';
	import EncounterBench from '$lib/components/EncounterBench.svelte';
	import { ABOUT_SECTIONS, GUIDE_SECTIONS, type DocumentationSection } from '$lib/lab/content';
	import type { LabSettings, LabIsa, InitialCondition, BoundaryRule } from '$lib/lab/types';
	import { ORIGINAL_SETTINGS } from '$lib/gpu/machine';
	import {
		resolveSuppression,
		countSuppressed,
		mnemonicOf,
		describeSuppression
	} from '$lib/z80-opcodes';
	import { unpackRGBA, createColormap, COLORMAP_NAMES } from '$lib/colormap';
	import type { ColormapName } from '$lib/colormap';
	import { untrack, onMount, tick } from 'svelte';
	import Mermaid from '$lib/components/Mermaid.svelte';
	import { saveCheckpoint, loadCheckpoint, type CheckpointMetadata } from '$lib/checkpoint';
	import {
		type ApplyMode,
		type SettingKey,
		type SettingValues,
		type Preset,
		loadPresets,
		savePresets,
		makePresetId,
		valuesEqual
	} from '$lib/presets';
	import katex from 'katex';
	import 'katex/dist/katex.min.css';

	let colormapName: ColormapName = $state('rainbow');
	let colormap = $state(createColormap('rainbow'));
	let simpleView = $state(false);
	let showGridLines = $state(true);
	let brightness = $state(0);
	let contrast = $state(1);
	let saturation = $state(1);

	let canvas = $state<HTMLCanvasElement>(null!);
	let canvasContainer: HTMLDivElement;
	let engine = $state.raw<GPUEngine | null>(null);
	let gpuError = $state<string | null>(null);

	// Controls
	let seed = $state(DEFAULT_SEED);
	let noiseExp = $state(DEFAULT_NOISE_EXP);
	let pairCount = $state(MAX_BATCH_PAIR_N);
	let z80Steps = $state(Z80_STEPS);
	let playing = $state(true);

	let speed = $state(8); // steps per frame; 8x is the default
	let gridType = $state<GridType>('square');
	type MachineChoice = 'z80-0.2' | 'z80-0.3' | 'z80-nmos' | 'm6502' | 'i8086';
	let machine = $state<MachineChoice>('z80-0.2');
	let tapeLength = $state(16);
	let pairLength = $state(32);
	let lethalProbability = $state(0);
	let randomRegisters = $state(false);
	let stackConvention = $state<LabSettings['stack']>('legacy-sp');
	let dialMode = $state<LabSettings['dialMode']>('historical-pcg');
	let interaction = $state<LabSettings['interaction']>('overwrite');
	let dissipation = $state(1);
	let boundary = $state<BoundaryRule>('reflect');
	let machineBusy = $state(false);
	let observationBusy = $state(false);
	let checkpointBusy = $state(false);
	let machineError = $state<string | null>(null);
	let coreNotice = $state('');
	let selectedExperimentId = $state('');
	let appliedExperimentId = $state('');
	let stagedExperimentPending = $state(false);
	let initialCondition = $state.raw<InitialCondition>({ kind: 'random' });
	const selectedExperiment = $derived(
		EXPERIMENT_PRESETS.find((preset) => preset.id === selectedExperimentId)
	);
	const isa = $derived<LabIsa>(
		machine === 'm6502' ? 'm6502' : machine === 'i8086' ? 'i8086' : 'z80'
	);
	const instructionTable = $derived(ablationInstructions(isa));
	const isaSuppressionDescription = $derived(
		isa === 'm6502'
			? 'Use a mnemonic (STA, PHA), family (family:stack, family:writes-mem) or exact opcode (opcode:8D). A suppressed 6502 operation consumes only its opcode byte; operand bytes can execute next as code. The byte values remain in the soup.'
			: 'Use a mnemonic (MOV, PUSH), family (family:block-copy, family:stack) or exact opcode (opcode:A4). Suppression consumes prefixes and the opcode only; ModRM, displacement and immediate bytes can execute next as code. Groups are suppressed at opcode level, including every suboperation sharing the selected opcode. The byte values remain in the soup.'
	);
	let inspectedTape = $state.raw<Uint8Array | null>(null);
	let inspectedCell = $state(-1);
	let inspectedSampleId = $state(0);
	let inspectedUpdate = $state(0);
	let speciesSampleUpdate = $state(0);
	let cultureBusy = $state(false);
	let observation = $state<ObservationTools>();
	let inspectReadPending = false;
	let pointerDownX = 0;
	let pointerDownY = 0;
	let pointerMoved = false;
	let inspectClickTimer: ReturnType<typeof setTimeout> | undefined;

	function settings(): LabSettings {
		return {
			isa,
			compat: machine === 'z80-0.2' ? '0.2' : '0.3',
			variant: machine === 'z80-nmos' ? 'zilog-nmos' : isa === 'z80' ? 'classic' : 'nmos',
			length: tapeLength,
			ring: pairLength,
			steps: z80Steps,
			width: gridWidth,
			height: gridHeight,
			topology: gridType,
			seed,
			pairAttempts: pairCount,
			mutationExponent: noiseExp,
			lethalProbability,
			randomRegisters,
			ablation: [...suppressPatterns],
			stack: stackConvention,
			dialMode,
			interaction,
			dissipation,
			boundary
		};
	}

	function disassemble(data: Uint8Array) {
		// Keep historical Z80 labels and use the added ISA decoders for the other cores.
		return isa === 'z80' ? disassembleZ80(data) : disassembleTape(isa, data);
	}

	function byteToMnemonic(byte: number): string {
		return isa === 'z80'
			? z80ByteToMnemonic(byte)
			: (instructionTable.find((row) => row.code === byte)?.mnemonic ?? hexByte(byte));
	}

	function getCellData(soup: Uint8Array, index: number): Uint8Array {
		const stride = Math.ceil(tapeLength / 4) * 4;
		return soup.slice(index * stride, index * stride + tapeLength);
	}

	function resolveIsaSuppression(patterns: string[]) {
		if (isa === 'z80') return resolveSuppression(patterns);
		const masks = ablationMasks(isa, patterns);
		// These sets are rebuilt only when a control changes, not on soup updates.
		// eslint-disable-next-line svelte/prefer-svelte-reactivity
		const base = new Set<number>();
		for (let byte = 0; byte < 256; byte++)
			if ((masks[byte >> 5] & (1 << (byte & 31))) !== 0) base.add(byte);
		return { base, cb: new Set<number>(), ed: new Set<number>() };
	}

	function suppressionPreview(input: string): number {
		try {
			return countSuppressed(
				resolveIsaSuppression(
					input
						.split(';')
						.map((p) => p.trim())
						.filter(Boolean)
				)
			);
		} catch {
			return 0;
		}
	}

	function assignMachineSettings(next: LabSettings) {
		machine =
			next.isa === 'z80'
				? next.compat === '0.2'
					? 'z80-0.2'
					: next.variant === 'zilog-nmos'
						? 'z80-nmos'
						: 'z80-0.3'
				: next.isa;
		tapeLength = next.length;
		pairLength = next.ring;
		lethalProbability = next.lethalProbability;
		randomRegisters = next.randomRegisters;
		stackConvention = next.stack;
		dialMode = next.dialMode;
		interaction = next.interaction;
		dissipation = next.dissipation;
		// Joined edges exist on the square lattice only.
		boundary = next.topology === 'hex' ? 'reflect' : next.boundary;
		gridWidth = next.width;
		gridHeight = next.height;
		gridType = next.topology;
		seed = next.seed;
		pairCount = next.pairAttempts;
		z80Steps = next.steps;
		noiseExp = next.mutationExponent;
		suppressPatterns = [...next.ablation];
	}

	async function updateMachine(reset = false) {
		if (!engine || machineBusy || checkpointBusy) return;
		machineBusy = true;
		machineError = null;
		try {
			if (reset) {
				await engine.configure(settings(), initialCondition);
				clearMeasurements();
			} else await engine.setMachine(settings());
			const previous = applied;
			applied = engine.settings;
			syncLiveDraft(previous, applied);
		} catch (error) {
			assignMachineSettings(engine.settings);
			machineError = error instanceof Error ? error.message : String(error);
		} finally {
			machineBusy = false;
		}
	}

	async function chooseMachine(next: string) {
		if (!engine || machineBusy || checkpointBusy || next === machine) return;
		coreNotice = '';
		const current = engine.settings;
		const nextIsa: LabIsa = next === 'm6502' ? 'm6502' : next === 'i8086' ? 'i8086' : 'z80';
		assignMachineSettings({
			...current,
			isa: nextIsa,
			compat: next === 'z80-0.2' ? '0.2' : '0.3',
			variant: next === 'z80-nmos' ? 'zilog-nmos' : nextIsa === 'z80' ? 'classic' : 'nmos',
			ablation: [],
			stack: nextIsa === 'z80' ? 'top' : 'first-push',
			dialMode: next === 'z80-0.2' ? 'historical-pcg' : 'counter'
		});
		await updateMachine();
		if (!machineError) {
			appliedExperimentId = '';
			const changedDefaults: string[] = [];
			if (current.stack !== applied.stack) {
				const placement =
					applied.stack === 'top'
						? 'top at B’s last byte'
						: applied.stack === 'first-push'
							? 'first push at B’s last byte'
							: 'legacy SP = FFFF';
				changedDefaults.push(`Stack: ${placement}`);
			}
			if (current.dialMode !== applied.dialMode)
				changedDefaults.push(
					`dial: ${applied.dialMode === 'counter' ? 'counter' : 'historical PCG'}`
				);
			if (current.ablation.length > 0) changedDefaults.push('instruction suppression cleared');
			coreNotice = `Core changed on the existing lattice.${changedDefaults.length ? ` ${changedDefaults.join('; ')}.` : ''}`;
		}
	}

	function applyExperimentPreset(preset: ExperimentPreset) {
		worldDraft = hexSafe(structuredClone(preset.config));
		stagedInitial = preset.initialCondition;
		stagedPreset = null;
		selectedExperimentId = preset.id;
		stagedExperimentPending = true;
		worldNotice = 'Condition staged. Start it when you are ready.';
	}

	async function inspectCell(index: number) {
		if (!engine || index < 0 || inspectReadPending) return;
		inspectReadPending = true;
		try {
			const sampledUpdate = engine.batchCount;
			const bytes = await engine.readCell(index);
			if (!bytes.length) return;
			activeTab = 'measure';
			sidebarCollapsed = false;
			inspectedTape = bytes;
			inspectedCell = index;
			inspectedUpdate = sampledUpdate;
			inspectedSampleId++;
			speciesModal = null;
			measureSection = 'bench';
		} finally {
			inspectReadPending = false;
		}
	}

	function scientificText(text: string): string {
		return text
			.replaceAll('new host-scheduled worlds', 'new GPU-scheduled worlds')
			.replaceAll('host scheduler', 'GPU scheduler')
			.replaceAll(
				'the update counter measures completed work, not seconds',
				'ops/s measures instruction slots of completed GPU work'
			)
			.replaceAll('host stream', 'GPU random stream')
			.replaceAll('pale zero-rich', 'zero-rich')
			.replaceAll(
				'instruction-stream reading enters the partner',
				'instruction-stream fetching leaves the focal tape (B or padding)'
			)
			.replaceAll(
				'“open” when it enters its partner',
				'“open” when it fetches outside that tape (B or padding)'
			)
			.replaceAll(
				'instruction fetching enters the partner',
				'instruction fetching leaves the focal tape (B or padding)'
			)
			.replaceAll(
				'no partner entry in all 16',
				'no fetch outside the focal tape (B or padding) in all 16'
			)
			.replaceAll('entry in at least eight is open', 'such a fetch in at least eight is open')
			.replaceAll(
				'Confinement describes instruction-stream addresses.',
				'Confinement describes whether instruction-stream fetching crosses the focal tape boundary into B or discarded padding.'
			)
			.replaceAll('new host’s per-encounter seeds', 'GPU kernel’s per-encounter seeds')
			.replaceAll(
				'0.3 engine and world API',
				'0.3.1 library core inside Algocell’s GPU soup kernel'
			)
			.replace(
				'The legacy instrument remains available for its original rendering, controls and checkpoints.',
				'The original Algocell rendering, controls and checkpoints remain the base of this instrument.'
			)
			.replace(
				'The interface’s visual language follows the manuscript’s figure plan: a white field, tangible byte blocks, quiet structural lines and separate encodings for byte value, execution and transmission.',
				'The interface keeps Algocell’s original dark instrument panels, tangible byte blocks and familiar color maps.'
			)
			.replace(
				'Press Run to advance the world, Pause to keep its current state, or Step for one population update.',
				'Press Play to advance the world or Pause to keep its current state.'
			)
			.replace(
				'Select a tape on the lattice or by cell index.',
				'Click a tape on the lattice or inspect an abundant species.'
			)
			.replace('Hexagonal legacy worlds', 'Hexagonal worlds');
	}

	const guideSections: DocumentationSection[] = GUIDE_SECTIONS.filter(
		(section) => section.id !== 'legacy'
	).map((section) => {
		if (section.id === 'lattice')
			return {
				...section,
				paragraphs: [
					'Byte view keeps Algocell’s original color maps. Each tape occupies a block at its actual lattice position; simple view averages its byte colors. A color or a visible texture is not a test of reproduction. Square and hexagonal neighborhoods preserve their original layouts.',
					'Colors continue to identify the same byte values when the processor changes. Their historical Z80 category colors do not imply equivalent 6502 or 8086 instruction meanings. Read the selected processor’s disassembly.',
					'The fetch layer records the last initiating encounter only. “Outside focal tape” means instruction-stream fetching reached any ring position at or beyond L, including B or discarded padding; “no observed exit” means it stayed in A during that encounter. Unknown means no current measurement. Mutated or modified cells can lose their measurement. This is not a lifetime confinement classification, and it does not measure partner-data reads, partner-information independence or heredity.'
				]
			};
		if (section.id === 'measurements')
			return {
				...section,
				title: 'Interpret the original charts',
				paragraphs: [
					'The opcode chart counts inherited byte values, even when a value is an operand or data rather than an executed instruction. The concentration factor compares a byte’s fraction with a uniform 1/256 baseline. Padding is excluded.',
					'The species and diversity tabs group full tapes by a GPU content hash. This fast composition view can have hash collisions; it is not an exact ancestry census. Shannon entropy, Simpson diversity and richness describe how those groups are distributed, not fitness, heredity or progress.',
					'An abundant exact-looking tape is a candidate to inspect and culture. A transmitting population can contain many different payloads, so a single top-tape share can hide it. Short visual runs do not reproduce longer cohort evidence or emergence-time distributions.'
				]
			};
		if (section.id === 'exports')
			return {
				...section,
				paragraphs: [
					'A preset describes how to begin; a checkpoint stores the current population. Algocell keeps its original .acell format: a PNG preview with local metadata and the complete word-aligned soup attached. New checkpoints include the processor, ring geometry, reset, chemistry, suppression and planted condition. Original files restore historical Z80 compatibility 0.2 with SP = FFFF.',
					'Checkpoint populations can be inspected and continued, but the original format does not preserve every GPU random lane’s continuation state. Reloading a population is not a promise of the same subsequent trajectory. Keep the complete condition with any comparison; equal seeds alone do not identify an archived run.',
					'Files remain on your computer. No checkpoint is uploaded to a remote service. Source paths in the experiment notes identify the source registration, machine world and results.'
				]
			};
		return section;
	});

	// Compute grid dimensions from viewport aspect ratio
	// ~20K cells: enough for emergence, large enough to see patterns clearly
	// For hex grids, cells are visually compressed vertically by sqrt(3)/2 ≈ 0.866
	// so we can fit more rows in the same vertical space
	const HEX_Y_RATIO = 0.866025404; // sqrt(3)/2
	function computeGridDimensions(
		vw: number,
		vh: number,
		type: GridType = 'square'
	): { w: number; h: number } {
		const aspect = vw / vh;
		const TARGET_CELLS = 20_000;
		if (type === 'hex') {
			// Hex cells are compressed vertically: effective aspect = aspect / HEX_Y_RATIO
			// This means we need more rows to fill the same visual height
			const effectiveAspect = aspect * HEX_Y_RATIO;
			const h = Math.round(Math.sqrt(TARGET_CELLS / effectiveAspect));
			const w = Math.round(h * effectiveAspect);
			return {
				w: Math.max(40, Math.min(400, w)),
				h: Math.max(40, Math.min(400, h))
			};
		}
		const h = Math.round(Math.sqrt(TARGET_CELLS / aspect));
		const w = Math.round(h * aspect);
		return {
			w: Math.max(40, Math.min(400, w)),
			h: Math.max(40, Math.min(400, h))
		};
	}

	// Initial dimensions — will be recomputed on mount from actual viewport
	let gridWidth = $state(200);
	let gridHeight = $state(200);
	let gridInitialized = false;

	// Stats
	let opsPerSec = $state(0);
	let helpTab = $state<
		'overview' | 'visuals' | 'z80' | 'params' | 'keys' | 'analysis' | 'about' | 'guide'
	>('guide');
	let lastGuideTab: typeof helpTab = 'guide';

	let keyboardModality = $state(false);
	let showInfoChart = $state(true);
	let suppressPatterns: string[] = $state([]);
	let suppressInput = $state('');

	// Resolve patterns → per-page suppression sets (base / CB / ED), see $lib/z80-opcodes.
	let suppression = $derived(resolveIsaSuppression(suppressPatterns));
	// Base-page set: what the frequency tiles (which show base-page bytes) can mark.
	let suppressedOpcodes = $derived(suppression.base);
	let suppressedTotal = $derived(countSuppressed(suppression));

	// Frequency chart: track top N bytes normalized over time
	// Total bytes counted by shader = cells * wordsPerCell * 4 (word-aligned)
	let TOTAL_BYTES = $derived(gridWidth * gridHeight * tapeLength);
	const MAX_TRACKED = 10;
	const MAX_HISTORY = 300;

	// Efficient ring-buffer chart storage: Float32Array(256) per point for O(1) byte lookup
	// Non-reactive — we only trigger Svelte updates via chartVersion counter
	const chartRing: Float32Array[] = []; // ring buffer of Float32Array(256), each indexed by byte
	let chartLen = $state(0); // current number of valid entries
	let chartVersion = $state(0); // bump to trigger reactive re-derive

	// Tracked bytes: stable set of bytes we draw lines for, with colormap-derived colors
	let trackedBytes: { byte: number; mnemonic: string }[] = $state([]);

	// Chart hover
	let chartHoveredByte = $state(-1);

	// Info widget tab: opcodes (byte frequency) or species (block identity)
	let infoTab = $state<'opcodes' | 'species' | 'diversity'>('opcodes');

	// === Species tracking ===
	interface SpeciesEntry {
		hash: number;
		count: number;
		fraction: number;
		color: string;
		textColor: string;
		exemplar: Uint8Array | null; // cell bytes for this species
		exemplarCells: GenomeCell[]; // disassembled genome grid
	}
	interface SpeciesSnapshot {
		species: SpeciesEntry[];
		richness: number; // unique hashes above threshold
		shannon: number; // Shannon entropy H
		simpson: number; // Simpson's index D = 1 - sum(p_i^2)
		totalUnique: number;
	}
	const MAX_SPECIES_DISPLAY = 10;
	const MAX_SPECIES_TRACKED = 20;
	const SPECIES_THRESHOLD = 0.001; // 0.1% of cells

	// Species ring buffer: stores fraction for each tracked species hash over time
	// speciesHashOrder[i] = hash for index i (stable across frames when possible)
	const speciesRing: Float32Array[] = [];
	let speciesHashOrder: number[] = $state([]); // maps index -> hash
	/* eslint-disable svelte/prefer-svelte-reactivity -- not reactive, only read inside derived */
	let speciesExemplars: Map<number, Uint8Array> = new Map(); // hash -> exemplar bytes
	/* eslint-enable svelte/prefer-svelte-reactivity */
	let speciesLen = $state(0);
	let speciesVersion = $state(0);
	let currentSpecies = $state<SpeciesSnapshot | null>(null);
	let speciesHoveredIdx = $state(-1);
	let speciesModal = $state<SpeciesEntry | null>(null);
	let speciesLoading = false;

	// Diversity metrics time series ring buffer
	// Each slot: [shannon, simpson, richness, totalUnique]
	const diversityRing: Float32Array[] = [];
	let diversityLen = $state(0);
	let diversityVersion = $state(0);

	function speciesAvgColor(exemplar: Uint8Array | null): { color: string; textColor: string } {
		if (!exemplar || exemplar.length === 0) {
			return { color: 'rgba(128,128,128,0.5)', textColor: 'rgba(255,255,255,0.9)' };
		}
		let rSum = 0,
			gSum = 0,
			bSum = 0;
		for (let i = 0; i < exemplar.length; i++) {
			const [r, g, b] = unpackRGBA(colormap[exemplar[i]]);
			rSum += r;
			gSum += g;
			bSum += b;
		}
		const n = exemplar.length;
		const r = Math.round(rSum / n);
		const g = Math.round(gSum / n);
		const b = Math.round(bSum / n);
		const lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
		return {
			color: `rgb(${r},${g},${b})`,
			textColor: lum > 0.4 ? 'rgba(0,0,0,0.85)' : 'rgba(255,255,255,0.9)'
		};
	}

	/** Brightened species color for chart lines */
	function speciesChartColor(exemplar: Uint8Array | null): string {
		const { color } = speciesAvgColor(exemplar);
		// Parse and brighten if too dark
		const m = color.match(/rgb\((\d+),(\d+),(\d+)\)/);
		if (!m) return color;
		let [r, g, b] = [+m[1], +m[2], +m[3]];
		const lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
		if (lum < 0.2) {
			const boost = 0.35;
			r = Math.round(r + (255 - r) * boost);
			g = Math.round(g + (255 - g) * boost);
			b = Math.round(b + (255 - b) * boost);
		}
		return `rgb(${r},${g},${b})`;
	}

	// Hover / genome tooltip
	let hoveredCell = $state(-1);
	let cellData = $state.raw<Uint8Array | null>(null);
	let disasmLines: ReturnType<typeof disassemble> = $state.raw([]);
	let canvasW = $state(800);
	let canvasH = $state(800);
	let tooltipRefreshTimer: ReturnType<typeof setInterval> | undefined;

	// Panning state
	let isPanning = $state(false);
	let panStartX = 0;
	let panStartY = 0;

	let frameCount = 0;
	let animFrameId: number | undefined;
	let statsLoading = false;

	// Build frequency lines using log-scaled concentration factor
	// concentration = frac * 256 (uniform = 1.0, a byte at 50% = 128.0)
	// Log scale ensures both dominant and minor bytes are visible
	let freqLines = $derived.by(() => {
		// Touch chartVersion to subscribe to updates
		void chartVersion;
		const len = chartLen;
		if (len < 2) return [];

		const chartBytes = displayTiles.filter((t) => t.type === 'byte');
		const allSeries = chartBytes.map((tb) => {
			const values = new Float64Array(len);
			for (let i = 0; i < len; i++) {
				const frac = chartRing[i][tb.byte];
				const conc = frac * 256; // concentration factor
				values[i] = conc > 0 ? Math.log2(conc + 1) : 0;
			}
			return {
				byte: tb.byte,
				mnemonic: tb.mnemonic,
				color: byteChartColor(tb.byte),
				values
			};
		});
		// Shared max across all lines
		let globalMax = Math.log2(2.5);
		for (const s of allSeries) {
			for (let i = 0; i < s.values.length; i++) {
				if (s.values[i] > globalMax) globalMax = s.values[i];
			}
		}
		return allSeries.map((s) => ({
			byte: s.byte,
			mnemonic: s.mnemonic,
			color: s.color,
			path: buildSparklinePathTyped(s.values, 200, 120, globalMax)
		}));
	});

	function buildSparklinePathTyped(
		values: Float64Array,
		w: number,
		h: number,
		max: number
	): string {
		if (values.length < 2) return '';
		const step = w / (values.length - 1);
		const parts: string[] = new Array(values.length);
		for (let i = 0; i < values.length; i++) {
			const x = i * step;
			const y = h - (values[i] / max) * (h - 4) - 2;
			parts[i] = `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`;
		}
		return parts.join(' ');
	}

	onMount(() => {
		if (!canvas) return;
		sidebarCollapsed = window.innerWidth < 700;
		let disposed = false;
		let mountedEngine: GPUEngine | undefined;

		async function initialize() {
			// Narrow layouts start with the sidebar collapsed. Measure only after that
			// change reaches the DOM, so the first world and camera fit its actual canvas.
			await tick();
			if (disposed) return;
			if (!gridInitialized) {
				const initialGridType = untrack(() => gridType);
				const dims = computeGridDimensions(
					canvas.clientWidth,
					canvas.clientHeight,
					initialGridType
				);
				gridWidth = dims.w;
				gridHeight = dims.h;
				gridInitialized = true;
			}

			const initialSeed = untrack(() => seed);
			const initialConfig: GridConfig = untrack(() => ({
				width: gridWidth,
				height: gridHeight,
				gridType: gridType,
				tapeLength,
				pairLength
			}));
			// Set canvas pixel dimensions before init so resetView gets correct aspect.
			const dpr = window.devicePixelRatio || 1;
			canvas.width = canvasW = Math.round(canvas.clientWidth * dpr);
			canvas.height = canvasH = Math.round(canvas.clientHeight * dpr);

			const eng = new GPUEngine(initialSeed, initialConfig);
			mountedEngine = eng;
			const ok = await eng.init(canvas);
			if (disposed) {
				eng.destroy();
				return;
			}
			if (!ok) {
				gpuError = 'WebGPU not available. Please use Chrome 113+ or Edge 113+.';
				return;
			}
			engine = eng;
			applied = eng.settings;
			worldDraft = structuredClone(eng.settings);
			startLoop();
			if (import.meta.env.DEV) {
				// Dev-only handle for headless experiments and debugging from the console.
				(window as unknown as { __algocell?: unknown }).__algocell = {
					engine: eng,
					get playing() {
						return playing;
					},
					setPlaying(value: boolean) {
						playing = value;
					},
					resolveSuppression
				};
			}
		}

		void initialize().catch((error) => {
			if (!disposed) gpuError = String(error);
		});

		return () => {
			disposed = true;
			clearInterval(tooltipRefreshTimer);
			clearTimeout(inspectClickTimer);
			if (animFrameId !== undefined) {
				cancelAnimationFrame(animFrameId);
				animFrameId = undefined;
			}
			mountedEngine?.destroy();
		};
	});

	// ResizeObserver for canvas
	onMount(() => {
		if (!canvasContainer) return;
		const ro = new ResizeObserver((entries) => {
			for (const entry of entries) {
				const { width, height } = entry.contentRect;
				canvasW = Math.round(width * devicePixelRatio);
				canvasH = Math.round(height * devicePixelRatio);
				if (canvas) {
					canvas.width = canvasW;
					canvas.height = canvasH;
				}
			}
		});
		ro.observe(canvasContainer);
		return () => ro.disconnect();
	});

	function startLoop() {
		function tick() {
			animFrameId = requestAnimationFrame(tick);
			if (!engine) return;

			if (playing && !document.hidden && !machineBusy && !checkpointBusy) {
				for (let i = 0; i < speed; i++) {
					engine.simulateStep();
				}
			}

			if (!machineBusy && !checkpointBusy) {
				engine.render(canvas);
				flushCellRefresh();
			}
			frameCount++;
			if (frameCount % 15 === 0) {
				epoch = engine.batchCount;
				opsPerSec = engine.opsPerSec;
			}

			// Update stats: immediately on first frame, then every 15 frames
			if (
				!machineBusy &&
				!observationBusy &&
				playing &&
				(frameCount === 1 || frameCount % 15 === 0) &&
				!statsLoading &&
				activeTab === 'measure' &&
				measureSection === 'charts'
			) {
				opsPerSec = engine.opsPerSec;
				updateTopBytes();
				if (
					activeTab === 'measure' &&
					measureSection === 'charts' &&
					(infoTab === 'species' || infoTab === 'diversity')
				) {
					updateSpecies();
				}
			}
		}
		tick();
	}

	async function updateTopBytes() {
		if (!engine || statsLoading) return;
		statsLoading = true;
		try {
			const counts = await engine.readStats();
			const entries: { byte: number; count: number; mnemonic: string }[] = [];
			for (let i = 0; i < 256; i++) {
				if (counts[i] > 0) {
					entries.push({ byte: i, count: counts[i], mnemonic: byteToMnemonic(i) });
				}
			}
			entries.sort((a, b) => b.count - a.count);

			// Track every non-zero byte, ranked by frequency. displayTiles filters
			// out suppressed opcodes and shows the top non-suppressed ones, so we
			// must keep the full ranked list here — otherwise broad suppression
			// (e.g. "LD" matches 100+ opcodes) can consume every tracked slot and
			// leave the top-10 empty. Keeping all non-zero bytes lets the freed
			// slots refill with the next-most-common non-suppressed opcodes.
			trackedBytes = entries.map((e) => ({ byte: e.byte, mnemonic: e.mnemonic }));

			// Only record chart history when chart is visible
			if (showInfoChart) {
				// Reuse or allocate a Float32Array(256) for O(1) byte lookup
				let slot: Float32Array;
				if (chartLen < MAX_HISTORY) {
					slot = new Float32Array(256);
					chartRing[chartLen] = slot;
					chartLen++;
				} else {
					// Shift ring: move slot 0 to end, shift everything left
					slot = chartRing[0];
					for (let i = 1; i < MAX_HISTORY; i++) chartRing[i - 1] = chartRing[i];
					chartRing[MAX_HISTORY - 1] = slot;
					slot.fill(0);
				}
				const total = TOTAL_BYTES;
				for (const e of entries) {
					slot[e.byte] = e.count / total;
				}
				chartVersion++;
			}
		} catch {
			// stats readback failed silently
		} finally {
			statsLoading = false;
		}
	}

	async function updateSpecies() {
		if (!engine || speciesLoading) return;
		speciesLoading = true;
		try {
			const hashes = await engine.readCellHashes();
			if (hashes.length === 0) return;

			// Count unique hashes and find first exemplar cell index for each
			/* eslint-disable svelte/prefer-svelte-reactivity -- local function variables, not reactive */
			const counts = new Map<number, number>();
			const exemplarIdx = new Map<number, number>(); // hash -> first cell index
			/* eslint-enable svelte/prefer-svelte-reactivity */
			for (let i = 0; i < hashes.length; i++) {
				const h = hashes[i];
				counts.set(h, (counts.get(h) || 0) + 1);
				if (!exemplarIdx.has(h)) exemplarIdx.set(h, i);
			}

			const total = hashes.length;
			const threshold = Math.max(1, Math.floor(total * SPECIES_THRESHOLD));

			// Compute diversity metrics
			let richness = 0;
			let shannonH = 0;
			let simpsonSum = 0;
			for (const [, count] of counts) {
				if (count >= threshold) richness++;
				const p = count / total;
				if (p > 0) {
					shannonH -= p * Math.log(p);
					simpsonSum += p * p;
				}
			}

			// Read soup data to extract exemplar cell bytes
			const soupData = await engine.readSoupData();

			// Sort by frequency, take top species
			const sorted = [...counts.entries()].sort((a, b) => b[1] - a[1]);
			const topSpecies: SpeciesEntry[] = sorted
				.slice(0, MAX_SPECIES_TRACKED)
				.map(([hash, count]) => {
					const cellIdx = exemplarIdx.get(hash) ?? 0;
					const exemplar = soupData.length > 0 ? getCellData(soupData, cellIdx) : null;
					const { color, textColor } = speciesAvgColor(exemplar);
					const exemplarCells = exemplar ? buildGenomeGrid(exemplar, disassemble(exemplar)) : [];
					return {
						hash,
						count,
						fraction: count / total,
						color,
						textColor,
						exemplar,
						exemplarCells
					};
				});

			currentSpecies = {
				species: topSpecies,
				richness,
				shannon: shannonH,
				simpson: 1 - simpsonSum,
				totalUnique: counts.size
			};

			// Record diversity metrics time series
			if (showInfoChart) {
				let slot: Float32Array;
				if (diversityLen < MAX_HISTORY) {
					slot = new Float32Array(4);
					diversityRing[diversityLen] = slot;
					diversityLen++;
				} else {
					slot = diversityRing[0];
					for (let i = 1; i < MAX_HISTORY; i++) diversityRing[i - 1] = diversityRing[i];
					diversityRing[MAX_HISTORY - 1] = slot;
				}
				slot[0] = shannonH;
				slot[1] = 1 - simpsonSum;
				slot[2] = richness;
				slot[3] = counts.size;
				diversityVersion++;
			}

			// Cache exemplar data for chart colors
			for (const sp of topSpecies) {
				if (sp.exemplar) speciesExemplars.set(sp.hash, sp.exemplar);
			}

			// Update ring buffer for sparklines
			if (showInfoChart) {
				// Build or update hash order (keep stable identity when possible)
				const newHashes = topSpecies.map((s) => s.hash);
				if (speciesHashOrder.length === 0) {
					speciesHashOrder = newHashes;
				} else {
					// Add any new hashes not yet tracked
					for (const h of newHashes) {
						if (!speciesHashOrder.includes(h)) {
							if (speciesHashOrder.length < MAX_SPECIES_TRACKED) {
								speciesHashOrder.push(h);
							}
						}
					}
				}

				// Record fractions for all tracked hashes
				let slot: Float32Array;
				if (speciesLen < MAX_HISTORY) {
					slot = new Float32Array(MAX_SPECIES_TRACKED);
					speciesRing[speciesLen] = slot;
					speciesLen++;
				} else {
					slot = speciesRing[0];
					for (let i = 1; i < MAX_HISTORY; i++) speciesRing[i - 1] = speciesRing[i];
					speciesRing[MAX_HISTORY - 1] = slot;
					slot.fill(0);
				}
				for (let idx = 0; idx < speciesHashOrder.length; idx++) {
					const c = counts.get(speciesHashOrder[idx]);
					slot[idx] = c ? c / total : 0;
				}
				speciesVersion++;
			}
		} catch {
			// silent fail
		} finally {
			speciesLoading = false;
		}
	}

	// Species sparklines (parallel to freqLines)
	let speciesLines = $derived.by(() => {
		void speciesVersion;
		const len = speciesLen;
		if (len < 2 || speciesHashOrder.length === 0) return [];

		const lines: { hash: number; color: string; path: string }[] = [];
		let globalMax = 0.1;

		// Pre-compute all series values to find globalMax
		const allValues: Float64Array[] = [];
		for (let idx = 0; idx < Math.min(speciesHashOrder.length, MAX_SPECIES_DISPLAY); idx++) {
			const values = new Float64Array(len);
			for (let i = 0; i < len; i++) {
				const frac = speciesRing[i][idx];
				values[i] = frac > 0 ? Math.log2(frac * 100 + 1) : 0;
			}
			allValues.push(values);
			for (let i = 0; i < values.length; i++) {
				if (values[i] > globalMax) globalMax = values[i];
			}
		}

		for (let idx = 0; idx < allValues.length; idx++) {
			const hash = speciesHashOrder[idx];
			const exemplar = speciesExemplars.get(hash) ?? null;
			lines.push({
				hash,
				color: speciesChartColor(exemplar),
				path: buildSparklinePathTyped(allValues[idx], 200, 120, globalMax)
			});
		}
		return lines;
	});

	// Diversity sparklines: each metric gets its own line with independent scaling
	const DIVERSITY_METRICS = [
		{ key: 0, label: 'H', name: 'Shannon entropy', color: 'rgb(255,180,50)' },
		{ key: 1, label: 'D', name: 'Simpson diversity', color: 'rgb(100,200,255)' },
		{ key: 2, label: 'spp', name: 'Species richness', color: 'rgb(120,255,120)' },
		{ key: 3, label: 'uniq', name: 'Unique blocks', color: 'rgb(255,130,180)' }
	] as const;

	let diversityHoveredIdx = $state(-1);

	let diversityLines = $derived.by(() => {
		void diversityVersion;
		const len = diversityLen;
		if (len < 2) return [];

		return DIVERSITY_METRICS.map((m, idx) => {
			const values = new Float64Array(len);
			let maxVal = 0;
			for (let i = 0; i < len; i++) {
				values[i] = diversityRing[i][m.key];
				if (values[i] > maxVal) maxVal = values[i];
			}
			// Each metric is normalized to its own max
			const normMax = maxVal > 0 ? maxVal : 1;
			return {
				idx,
				label: m.label,
				name: m.name,
				color: m.color,
				path: buildSparklinePathTyped(values, 200, 120, normMax),
				current: len > 0 ? diversityRing[len - 1][m.key] : 0
			};
		});
	});

	function handleCanvasMouseMove(e: MouseEvent) {
		if (touchHandled) return; // ignore synthetic mouse events from touch
		if (!engine || !canvas) return;
		const rect = canvas.getBoundingClientRect();
		const sx = e.clientX - rect.left;
		const sy = e.clientY - rect.top;

		if (isPanning) {
			if (Math.hypot(e.clientX - pointerDownX, e.clientY - pointerDownY) > 4) pointerMoved = true;
			const dx = e.clientX - panStartX;
			const dy = e.clientY - panStartY;
			engine.pan(dx, dy, rect.width, rect.height);
			panStartX = e.clientX;
			panStartY = e.clientY;
			return;
		}

		const scaleX = canvasW / rect.width;
		const scaleY = canvasH / rect.height;
		const cell = engine.screenToCell(sx * scaleX, sy * scaleY, canvasW, canvasH);

		if (cell >= 0) {
			if (cell !== hoveredCell) {
				hoveredCell = cell;
				engine.setHoverCell(cell);
				clearInterval(tooltipRefreshTimer);
				tooltipRefreshTimer = setInterval(() => refreshCellData(cell), 500);
			}
			refreshCellData(cell);
		} else if (cell < 0) {
			hoveredCell = -1;
			cellData = null;
			disasmLines = [];
			engine.setHoverCell(-1);
			clearInterval(tooltipRefreshTimer);
		}
	}

	let pendingCellRefresh = -1;
	let cellRefreshPending = false;
	let lastCellRefresh = 0;

	function hoverPreviewVisible() {
		return (
			activeTab === 'measure' &&
			measureSection === 'probe' &&
			activeTool === 'navigate' &&
			!sidebarCollapsed
		);
	}

	function refreshCellData(cell: number) {
		if (!hoverPreviewVisible()) {
			clearInterval(tooltipRefreshTimer);
			pendingCellRefresh = -1;
			return;
		}
		// Mark for readback after next render (so data matches what's on screen)
		pendingCellRefresh = cell;
	}

	function flushCellRefresh() {
		if (!hoverPreviewVisible()) {
			pendingCellRefresh = -1;
			return;
		}
		if (
			pendingCellRefresh < 0 ||
			!engine ||
			cellRefreshPending ||
			performance.now() - lastCellRefresh < 500
		)
			return;
		const cell = pendingCellRefresh;
		pendingCellRefresh = -1;
		cellRefreshPending = true;
		lastCellRefresh = performance.now();
		engine
			.readCell(cell)
			.then((data) => {
				if (data.length === 0 || hoveredCell !== cell || !hoverPreviewVisible()) return;
				cellData = data;
				disasmLines = disassemble(data);
			})
			.finally(() => {
				cellRefreshPending = false;
			});
	}

	// Track if touch just handled an interaction, to suppress synthetic mouse events
	let touchHandled = false;

	function handleCanvasMouseDown(e: MouseEvent) {
		if (touchHandled) return; // ignore synthetic mouse events from touch
		if (e.button === 0 || e.button === 1) {
			e.preventDefault();
			isPanning = true;
			pointerMoved = false;
			pointerDownX = e.clientX;
			pointerDownY = e.clientY;
			panStartX = e.clientX;
			panStartY = e.clientY;
			hoveredCell = -1;
			cellData = null;
			disasmLines = [];
			engine?.setHoverCell(-1);
		}
	}

	function handleCanvasMouseUp(e: MouseEvent) {
		if (touchHandled) return;
		if (e.button === 0 || e.button === 1) {
			isPanning = false;
			if (e.button === 0 && !pointerMoved && engine && canvas) {
				const rect = canvas.getBoundingClientRect();
				const cell = engine.screenToCell(
					((e.clientX - rect.left) * canvasW) / rect.width,
					((e.clientY - rect.top) * canvasH) / rect.height,
					canvasW,
					canvasH
				);
				clearTimeout(inspectClickTimer);
				inspectClickTimer = setTimeout(() => {
					void observation?.selectCell(cell);
				}, 220);
			}
		}
	}

	function handleCanvasWheel(e: WheelEvent) {
		if (!engine || !canvas) return;
		e.preventDefault();
		const rect = canvas.getBoundingClientRect();
		const sx = (e.clientX - rect.left) * (canvasW / rect.width);
		const sy = (e.clientY - rect.top) * (canvasH / rect.height);
		// Proportional to scroll amount, clamped for consistency between trackpad/mouse
		const normalizedDelta = Math.sign(e.deltaY) * Math.min(Math.abs(e.deltaY), 100);
		const zoomSpeed = 0.002;
		const factor = Math.exp(normalizedDelta * zoomSpeed);
		engine.zoomAt(sx, sy, canvasW, canvasH, factor);
	}

	// --- Touch events for mobile pinch-zoom, pan & tap-to-inspect ---
	let lastTouchDist = 0;
	let lastTouchX = 0;
	let lastTouchY = 0;
	let touchPanning = false;
	let touchStartX = 0;
	let touchStartY = 0;
	let touchMoved = false;
	const TAP_THRESHOLD = 8; // pixels of movement allowed for a tap

	function handleTouchStart(e: TouchEvent) {
		if (!engine || !canvas) return;
		e.preventDefault();
		// Suppress synthetic mouse events that fire after touch
		touchHandled = true;
		if (e.touches.length === 2) {
			// Pinch start
			const dx = e.touches[1].clientX - e.touches[0].clientX;
			const dy = e.touches[1].clientY - e.touches[0].clientY;
			lastTouchDist = Math.sqrt(dx * dx + dy * dy);
			lastTouchX = (e.touches[0].clientX + e.touches[1].clientX) / 2;
			lastTouchY = (e.touches[0].clientY + e.touches[1].clientY) / 2;
			touchPanning = false;
			touchMoved = true; // multi-touch is never a tap
		} else if (e.touches.length === 1) {
			// Pan start — also track for potential tap
			touchPanning = true;
			touchMoved = false;
			touchStartX = panStartX = e.touches[0].clientX;
			touchStartY = panStartY = e.touches[0].clientY;
		}
	}

	function handleTouchMove(e: TouchEvent) {
		if (!engine || !canvas) return;
		e.preventDefault();
		const rect = canvas.getBoundingClientRect();
		if (e.touches.length === 2) {
			// Pinch zoom
			const dx = e.touches[1].clientX - e.touches[0].clientX;
			const dy = e.touches[1].clientY - e.touches[0].clientY;
			const dist = Math.sqrt(dx * dx + dy * dy);
			if (lastTouchDist > 0) {
				const midX = (e.touches[0].clientX + e.touches[1].clientX) / 2;
				const midY = (e.touches[0].clientY + e.touches[1].clientY) / 2;
				const sx = (midX - rect.left) * (canvasW / rect.width);
				const sy = (midY - rect.top) * (canvasH / rect.height);
				const factor = lastTouchDist / dist; // invert: pinch out = zoom in
				engine.zoomAt(sx, sy, canvasW, canvasH, factor);
				// Also pan with pinch movement
				const panDx = midX - lastTouchX;
				const panDy = midY - lastTouchY;
				engine.pan(panDx, panDy, rect.width, rect.height);
				lastTouchX = midX;
				lastTouchY = midY;
			}
			lastTouchDist = dist;
			touchPanning = false;
		} else if (e.touches.length === 1 && touchPanning) {
			const dx = e.touches[0].clientX - panStartX;
			const dy = e.touches[0].clientY - panStartY;
			// Check if finger moved enough to count as pan (not tap)
			const totalDx = e.touches[0].clientX - touchStartX;
			const totalDy = e.touches[0].clientY - touchStartY;
			if (Math.abs(totalDx) > TAP_THRESHOLD || Math.abs(totalDy) > TAP_THRESHOLD) {
				touchMoved = true;
			}
			if (touchMoved) {
				// Dismiss tooltip when panning starts
				if (hoveredCell >= 0) {
					hoveredCell = -1;
					cellData = null;
					disasmLines = [];
					engine.setHoverCell(-1);
					clearInterval(tooltipRefreshTimer);
				}
				engine.pan(dx, dy, rect.width, rect.height);
			}
			panStartX = e.touches[0].clientX;
			panStartY = e.touches[0].clientY;
		}
	}

	function handleTouchEnd(e: TouchEvent) {
		// Reset touchHandled after synthetic events have fired (~300ms)
		setTimeout(() => {
			touchHandled = false;
		}, 400);

		if (e.touches.length === 0) {
			// All fingers lifted — check for tap
			if (!touchMoved && engine && canvas) {
				const rect = canvas.getBoundingClientRect();
				const sx = (touchStartX - rect.left) * (canvasW / rect.width);
				const sy = (touchStartY - rect.top) * (canvasH / rect.height);
				const cell = engine.screenToCell(sx, sy, canvasW, canvasH);

				if (cell >= 0 && cell !== hoveredCell) {
					// Tap on a cell — preserve the hover view and open an inspector.
					void observation?.selectCell(cell);
					hoveredCell = cell;
					engine.setHoverCell(cell);
					clearInterval(tooltipRefreshTimer);
					tooltipRefreshTimer = setInterval(() => refreshCellData(cell), 500);
					refreshCellData(cell);
				} else {
					// Tap on same cell or empty space — dismiss tooltip
					hoveredCell = -1;
					cellData = null;
					disasmLines = [];
					engine.setHoverCell(-1);
					clearInterval(tooltipRefreshTimer);
				}
			}
			lastTouchDist = 0;
			touchPanning = false;
			touchMoved = false;
		} else if (e.touches.length === 1) {
			// Went from pinch to single finger — start pan
			lastTouchDist = 0;
			touchPanning = true;
			touchMoved = true; // already moved (was pinching)
			panStartX = e.touches[0].clientX;
			panStartY = e.touches[0].clientY;
		}
	}

	function handleCanvasMouseLeave() {
		hoveredCell = -1;
		cellData = null;
		disasmLines = [];
		isPanning = false;
		engine?.setHoverCell(-1);
		clearInterval(tooltipRefreshTimer);
	}

	function handleCanvasDblClick() {
		clearTimeout(inspectClickTimer);
		inspectedTape = null;
		engine?.resetView(canvasW, canvasH);
	}

	async function handleReset() {
		if (!engine || machineBusy || checkpointBusy) return;
		machineBusy = true;
		try {
			await engine.configure(engine.settings, initialCondition);
			clearMeasurements();
		} catch (error) {
			machineError = error instanceof Error ? error.message : String(error);
		} finally {
			machineBusy = false;
		}
	}

	function clearMeasurements() {
		coreNotice = '';
		lastProbeTarget = -1;
		pendingCellRefresh = -1;
		cellData = null;
		hoveredCell = -1;
		clearInterval(tooltipRefreshTimer);
		observation?.resetSelection();
		inspectedTape = null;
		opsPerSec = 0;
		chartRing.length = 0;
		chartLen = 0;
		chartVersion++;
		trackedBytes = [];
		speciesRing.length = 0;
		speciesHashOrder = [];
		speciesLen = 0;
		speciesVersion++;
		currentSpecies = null;
		speciesExemplars = new Map();
		speciesModal = null;
		diversityRing.length = 0;
		diversityLen = 0;
		diversityVersion++;
	}

	// Checkpoint save/load
	let fileInput: HTMLInputElement;

	async function handleSave() {
		if (!engine || !canvas || machineBusy || observationBusy || checkpointBusy || cultureBusy)
			return;
		checkpointBusy = true;
		machineError = null;
		try {
			await engine.waitForIdle();
			const applied = engine.settings;
			engine.render(canvas);
			const blob = await new Promise<Blob>((resolve, reject) => {
				canvas.toBlob(
					(value) =>
						value
							? resolve(value)
							: reject(new Error('The checkpoint image could not be captured.')),
					'image/png'
				);
			});
			let soupData = await engine.readSoupData();
			const expectedBytes = applied.width * applied.height * Math.ceil(applied.length / 4) * 4;
			for (let retry = 0; soupData.length !== expectedBytes && retry < 3; retry++) {
				await new Promise((resolve) => setTimeout(resolve, 50));
				soupData = await engine.readSoupData();
			}
			if (soupData.length !== expectedBytes)
				throw new Error(
					'A population measurement is still finishing. Save the checkpoint again in a moment.'
				);
			const metadata: CheckpointMetadata = {
				version: 1,
				machineSettings: applied,
				initialCondition,
				gridWidth: applied.width,
				gridHeight: applied.height,
				gridType: applied.topology,
				seed: applied.seed,
				noiseExp: applied.mutationExponent,
				z80Steps: applied.steps,
				pairCount: applied.pairAttempts,
				suppressPatterns: [...applied.ablation],
				batchCount: engine.batchCount,
				timestamp: Date.now()
			};
			const checkpoint = await saveCheckpoint(blob, metadata, soupData);
			const url = URL.createObjectURL(checkpoint),
				a = document.createElement('a');
			a.href = url;
			const ts = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
			a.download = `algocell-${ts}.acell`;
			a.click();
			setTimeout(() => URL.revokeObjectURL(url), 1000);
		} catch (cause) {
			machineError = cause instanceof Error ? cause.message : String(cause);
		} finally {
			checkpointBusy = false;
		}
	}

	let pendingCheckpoint = $state.raw<{
		name: string;
		metadata: CheckpointMetadata;
		soupData: Uint8Array;
	} | null>(null);
	async function handleLoadFile(file: File) {
		if (!engine || machineBusy || checkpointBusy) return;
		machineError = null;
		try {
			const result = await loadCheckpoint(new Uint8Array(await file.arrayBuffer()));
			if (!result) throw new Error('This file does not contain an Algocell checkpoint.');
			pendingCheckpoint = { name: file.name, metadata: result.metadata, soupData: result.soupData };
			showMeasure('record');
		} catch (error) {
			machineError = error instanceof Error ? error.message : String(error);
		}
	}
	async function restoreCheckpoint() {
		if (!engine || !pendingCheckpoint || machineBusy || checkpointBusy) return;
		machineBusy = true;
		machineError = null;
		try {
			const { metadata, soupData } = pendingCheckpoint;
			const restored: LabSettings = metadata.machineSettings ?? {
				...ORIGINAL_SETTINGS,
				width: metadata.gridWidth,
				height: metadata.gridHeight,
				topology: metadata.gridType,
				length: metadata.gridType === 'hex' ? 19 : 16,
				ring: metadata.gridType === 'hex' ? 38 : 32,
				seed: metadata.seed,
				mutationExponent: metadata.noiseExp,
				steps: metadata.z80Steps,
				pairAttempts: metadata.pairCount,
				ablation: [...metadata.suppressPatterns]
			};
			const restoredInitial = metadata.initialCondition ?? { kind: 'random' };
			await engine.configure(restored, restoredInitial);
			engine.writeSoupData(soupData);
			assignMachineSettings(restored);
			initialCondition = restoredInitial;
			applied = engine.settings;
			worldDraft = structuredClone(applied);
			stagedInitial = initialCondition;
			selectedExperimentId = '';
			appliedExperimentId = '';
			stagedExperimentPending = false;
			stagedPreset = null;
			clearMeasurements();
			pendingCheckpoint = null;
			worldNotice = 'Checkpoint restored. Playback state preserved.';
		} catch (error) {
			machineError = error instanceof Error ? error.message : String(error);
		} finally {
			machineBusy = false;
		}
	}

	function handleFileInputChange(e: Event) {
		const input = e.target as HTMLInputElement;
		if (input.files?.[0]) {
			handleLoadFile(input.files[0]);
			input.value = '';
		}
	}

	function handleDrop(e: DragEvent) {
		e.preventDefault();
		const file = e.dataTransfer?.files[0];
		if (file) handleLoadFile(file);
	}

	function handleColormapChange(name: ColormapName) {
		colormapName = name;
		colormap = createColormap(name);
		engine?.updateColormap(colormap);
	}

	function addSuppressPattern(pattern: string) {
		if (machineBusy || checkpointBusy) return;
		// Support multiple patterns separated by semicolons
		// (commas are NOT delimiters because Z80 mnemonics contain commas, e.g. LD A,B)
		const parts = pattern
			.split(';')
			.map((s) => s.trim().toUpperCase())
			.filter(Boolean);
		let added = false;
		for (const p of parts) {
			if (!p || suppressPatterns.includes(p)) continue;
			try {
				ablationMasks(isa, [p]);
			} catch (error) {
				machineError = error instanceof Error ? error.message : String(error);
				continue;
			}
			suppressPatterns.push(p);
			added = true;
		}
		if (added) {
			// Svelte needs identity change for array reactivity
			suppressPatterns = [...suppressPatterns];
			commitAblation();
		}
	}

	function removeSuppressPattern(pattern: string) {
		if (machineBusy || checkpointBusy) return;
		suppressPatterns = suppressPatterns.filter((p) => p !== pattern);
		commitAblation();
	}

	function toggleSuppressOpcode(opcode: number) {
		if (machineBusy || checkpointBusy) return;
		tileTooltip = null;
		chartHoveredByte = -1;
		const mnemonic = isa === 'z80' ? mnemonicOf('base', opcode) : `opcode:${hexByte(opcode)}`;
		if (suppressedOpcodes.has(opcode)) {
			// Find and remove patterns that match this opcode
			// If the exact mnemonic is a pattern, remove it; otherwise remove patterns that only match this one
			const upper = mnemonic.toUpperCase();
			if (suppressPatterns.includes(upper)) {
				removeSuppressPattern(upper);
			}
		} else {
			addSuppressPattern(mnemonic);
		}
	}

	function clearSuppressedOpcodes() {
		if (machineBusy || checkpointBusy) return;
		suppressPatterns = [];
		commitAblation();
	}

	// ── Presets ────────────────────────────────────────────────────────────
	// Declarative schema: one entry per tunable setting. `mode` says how
	// disruptive a change is (see presets.ts). `get`/`set` wire the setting to
	// this component's state + engine. To add a new tunable setting later, add
	// one row here and one key to SETTING_KEYS in presets.ts — nothing else.
	interface SettingDef {
		key: SettingKey;
		mode: ApplyMode;
		get: () => unknown;
		set: (v: unknown) => void;
	}
	const SETTINGS: SettingDef[] = [
		{
			key: 'machine',
			mode: 'grid',
			get: () => machine,
			set: (v) => (machine = v as MachineChoice)
		},
		{
			key: 'tapeLength',
			mode: 'grid',
			get: () => tapeLength,
			set: (v) => (tapeLength = v as number)
		},
		{
			key: 'pairLength',
			mode: 'grid',
			get: () => pairLength,
			set: (v) => (pairLength = v as number)
		},
		{
			key: 'lethalProbability',
			mode: 'live',
			get: () => lethalProbability,
			set: (v) => (lethalProbability = v as number)
		},
		{
			key: 'randomRegisters',
			mode: 'live',
			get: () => randomRegisters,
			set: (v) => (randomRegisters = v as boolean)
		},
		{
			key: 'stackConvention',
			mode: 'grid',
			get: () => stackConvention,
			set: (v) => (stackConvention = v as LabSettings['stack'])
		},
		{
			key: 'dialMode',
			mode: 'live',
			get: () => dialMode,
			set: (v) => (dialMode = v as LabSettings['dialMode'])
		},
		{
			key: 'interaction',
			mode: 'live',
			get: () => interaction,
			set: (v) => (interaction = v as LabSettings['interaction'])
		},
		{
			key: 'dissipation',
			mode: 'live',
			get: () => dissipation,
			set: (v) => (dissipation = v as number)
		},
		{
			key: 'boundary',
			mode: 'live',
			get: () => boundary,
			set: (v) => (boundary = v as BoundaryRule)
		},
		{
			key: 'initialCondition',
			mode: 'grid',
			get: () => JSON.stringify(initialCondition),
			set: (v) => {
				initialCondition = JSON.parse(v as string) as InitialCondition;
			}
		},
		{ key: 'gridType', mode: 'grid', get: () => gridType, set: (v) => (gridType = v as GridType) },
		{ key: 'gridWidth', mode: 'grid', get: () => gridWidth, set: (v) => (gridWidth = v as number) },
		{
			key: 'gridHeight',
			mode: 'grid',
			get: () => gridHeight,
			set: (v) => (gridHeight = v as number)
		},
		{ key: 'seed', mode: 'reset', get: () => seed, set: (v) => (seed = v as number) },
		{
			key: 'noiseExp',
			mode: 'live',
			get: () => noiseExp,
			set: (v) => {
				noiseExp = v as number;
				if (engine) engine.noiseExp = noiseExp;
			}
		},
		{
			key: 'pairCount',
			mode: 'live',
			get: () => pairCount,
			set: (v) => {
				pairCount = v as number;
				if (engine) engine.pairCount = pairCount;
			}
		},
		{
			key: 'z80Steps',
			mode: 'live',
			get: () => z80Steps,
			set: (v) => {
				z80Steps = v as number;
				if (engine) engine.z80Steps = z80Steps;
			}
		},
		{
			key: 'suppressPatterns',
			mode: 'live',
			get: () => [...suppressPatterns],
			// suppression (derived) -> $effect pushes the masks to the engine
			set: (v) => (suppressPatterns = [...((v as string[]) ?? [])])
		},
		{
			key: 'colormapName',
			mode: 'live',
			get: () => colormapName,
			set: (v) => handleColormapChange(v as ColormapName)
		},
		{
			key: 'brightness',
			mode: 'live',
			get: () => brightness,
			set: (v) => (brightness = v as number)
		},
		{ key: 'contrast', mode: 'live', get: () => contrast, set: (v) => (contrast = v as number) },
		{
			key: 'saturation',
			mode: 'live',
			get: () => saturation,
			set: (v) => (saturation = v as number)
		},
		{
			key: 'showGridLines',
			mode: 'live',
			get: () => showGridLines,
			set: (v) => (showGridLines = v as boolean)
		},
		{
			key: 'simpleView',
			mode: 'live',
			get: () => simpleView,
			set: (v) => (simpleView = v as boolean)
		}
	];

	let presets = $state<Preset[]>(loadPresets());
	let newPresetName = $state('');
	let showPresetSave = $state(false);
	// When saving would collide with an existing name, hold that preset here to
	// prompt the user to confirm overwrite (which is how saved presets are edited).
	let overwriteTarget = $state<Preset | null>(null);

	function captureSettings(): SettingValues {
		const values: SettingValues = {};
		for (const def of SETTINGS) values[def.key] = def.get();
		return values;
	}

	// Which stored preset (if any) exactly matches the current settings.
	let activePresetId = $derived.by(() => {
		// Re-read current values reactively.
		const current: SettingValues = {};
		for (const def of SETTINGS) current[def.key] = def.get();
		for (const p of presets) {
			let match = true;
			for (const def of SETTINGS) {
				if (!(def.key in p.values)) continue; // partial presets only constrain the keys they carry
				if (!valuesEqual(current[def.key], p.values[def.key])) {
					match = false;
					break;
				}
			}
			if (match) return p.id;
		}
		return null;
	});

	function applyPreset(preset: Preset) {
		const values = preset.values;
		const choice = String(values.machine ?? 'z80-0.2');
		const presetIsa: LabIsa = choice === 'm6502' ? 'm6502' : choice === 'i8086' ? 'i8086' : 'z80';
		const topology = (values.gridType ?? applied.topology) as GridType;
		const legacyGeometry = preset.id === 'builtin:classic' || preset.id === 'builtin:hex-organic';
		worldDraft = {
			...structuredClone(applied),
			isa: presetIsa,
			compat: choice === 'z80-0.2' ? '0.2' : '0.3',
			variant: choice === 'z80-nmos' ? 'zilog-nmos' : presetIsa === 'z80' ? 'classic' : 'nmos',
			topology,
			width: Number(values.gridWidth ?? applied.width),
			height: Number(values.gridHeight ?? applied.height),
			length: Number(
				values.tapeLength ?? (legacyGeometry ? (topology === 'hex' ? 19 : 16) : applied.length)
			),
			ring: Number(
				values.pairLength ?? (legacyGeometry ? (topology === 'hex' ? 38 : 32) : applied.ring)
			),
			seed: Number(values.seed ?? applied.seed),
			steps: Number(values.z80Steps ?? applied.steps),
			pairAttempts: Number(values.pairCount ?? applied.pairAttempts),
			mutationExponent: Number(values.noiseExp ?? applied.mutationExponent),
			lethalProbability: Number(values.lethalProbability ?? 0),
			randomRegisters: Boolean(values.randomRegisters ?? false),
			stack: (values.stackConvention ?? 'legacy-sp') as LabSettings['stack'],
			dialMode: (values.dialMode ?? 'historical-pcg') as LabSettings['dialMode'],
			interaction: (values.interaction ?? 'overwrite') as LabSettings['interaction'],
			dissipation: Number(values.dissipation ?? 1),
			boundary: topology === 'hex' ? 'reflect' : ((values.boundary ?? 'reflect') as BoundaryRule),
			ablation: (values.suppressPatterns ?? applied.ablation) as string[]
		};
		stagedInitial = values.initialCondition
			? JSON.parse(String(values.initialCondition))
			: legacyGeometry
				? { kind: 'random' }
				: initialCondition;
		stagedPreset = preset;
		selectedExperimentId = '';
		stagedExperimentPending = false;
		worldNotice = `${preset.name} staged. Start it when you are ready.`;
	}

	// Case-insensitive name match so "Classic" and "classic" are the same preset.
	function findPresetByName(name: string): Preset | undefined {
		const key = name.trim().toLowerCase();
		return presets.find((p) => p.name.toLowerCase() === key);
	}

	function requestSavePreset(name: string) {
		const trimmed = name.trim();
		if (!trimmed) return;
		const existing = findPresetByName(trimmed);
		if (existing) {
			// Name taken — confirm overwrite (this is how a saved preset is edited).
			overwriteTarget = existing;
		} else {
			commitSavePreset(trimmed, null);
		}
	}

	function commitSavePreset(name: string, existing: Preset | null) {
		const values = captureSettings();
		if (existing) {
			// Overwrite only replaces the saved settings; the existing name is kept
			// (so re-saving "Classic" as "classic" doesn't silently rename it).
			existing.values = values;
			presets = [...presets];
		} else {
			presets = [...presets, { id: makePresetId(presets), name: name.trim(), values }];
		}
		savePresets(presets);
		newPresetName = '';
		showPresetSave = false;
		overwriteTarget = null;
	}

	function confirmOverwrite() {
		if (overwriteTarget) commitSavePreset(overwriteTarget.name, overwriteTarget);
	}

	function deletePreset(id: string) {
		presets = presets.filter((p) => p.id !== id);
		savePresets(presets);
	}

	// Delete whichever preset currently matches the settings (the highlighted one).
	function deleteActivePreset() {
		if (activePresetId) deletePreset(activePresetId);
	}

	function commitAblation() {
		if (!engine || machineBusy || checkpointBusy) return;
		engine.setAblation(suppressPatterns);
		const previous = applied;
		applied = engine.settings;
		syncLiveDraft(previous, applied);
	}

	// Sync simple view mode to GPU engine
	$effect(() => {
		if (!engine) return;
		engine.setShowAverage(simpleView);
	});

	// Sync brightness/contrast/saturation to GPU
	$effect(() => {
		if (!engine) return;
		engine.setBCS(brightness, contrast, saturation);
	});

	// Sync grid lines visibility to GPU
	$effect(() => {
		if (!engine) return;
		engine.setShowGrid(showGridLines);
	});

	// Build display tiles: collapse all suppressed into one card, fill remaining with non-suppressed
	interface DisplayTile {
		type: 'byte' | 'suppressed';
		byte: number; // for 'byte' type
		mnemonic: string;
		count: number; // for 'suppressed': how many suppressed bytes in top rankings
	}
	let displayTiles = $derived.by(() => {
		const tiles: DisplayTile[] = [];
		const hasSuppression = suppressedOpcodes.size > 0;
		// Reserve one slot for the collapsed "suppressed" card when suppression
		// is active, so the ranked non-suppressed opcodes always fill the rest.
		const byteLimit = hasSuppression ? MAX_TRACKED - 1 : MAX_TRACKED;
		let suppressedInTop = 0;

		for (const tb of trackedBytes) {
			if (hasSuppression && suppressedOpcodes.has(tb.byte)) {
				suppressedInTop++;
				continue;
			}
			if (tiles.length < byteLimit) {
				tiles.push({ type: 'byte', byte: tb.byte, mnemonic: tb.mnemonic, count: 0 });
			}
		}

		// Append the collapsed card summarizing suppressed opcodes present in the soup.
		if (hasSuppression && suppressedInTop > 0) {
			tiles.push({
				type: 'suppressed',
				byte: -1,
				mnemonic: `⊘ ${suppressedTotal}`,
				count: suppressedInTop
			});
		}

		return tiles;
	});

	function togglePlay() {
		if (!engine) return;
		playing = !playing;
	}

	function setSpeed(s: number) {
		speed = s;
	}

	function formatNumber(n: number): string {
		// Fixed 5-char output: "XXXX" + suffix or padded digits
		// Always produces exactly 5 visible characters to prevent layout shift
		if (n >= 1_000_000_000) {
			const v = (n / 1_000_000_000).toFixed(1);
			return v.padStart(4, '\u2007') + 'B'; // e.g. " 1.2B", "12.3B"
		}
		if (n >= 1_000_000) {
			const v = (n / 1_000_000).toFixed(1);
			return v.padStart(4, '\u2007') + 'M';
		}
		if (n >= 1_000) {
			const v = (n / 1_000).toFixed(1);
			return v.padStart(4, '\u2007') + 'K';
		}
		return n.toFixed(0).padStart(5, '\u2007');
	}

	function hexByte(b: number): string {
		return b.toString(16).toUpperCase().padStart(2, '0');
	}

	function tex(expr: string): string {
		return katex.renderToString(expr, { throwOnError: false, displayMode: false });
	}

	function texBlock(expr: string): string {
		return katex.renderToString(expr, { throwOnError: false, displayMode: true });
	}

	// Z80 opcode descriptions — plain-English decoder for non-experts
	// Register glossary:
	//   A = main working value, HL = address pointer (2 bytes), SP = stack pointer (read/write position)
	//   BC/DE = auxiliary 2-byte registers, AF = working value + status flags
	function opcodeDescription(op: number): string {
		const x = (op >> 6) & 3;
		const y = (op >> 3) & 7;
		const z = op & 7;
		const p = y >> 1;
		const q = y & 1;
		const R_DESC = [
			'counter B',
			'counter C',
			'register D',
			'register E',
			'high byte of address pointer',
			'low byte of address pointer',
			'the byte in memory that the address pointer points to',
			'the working value'
		];
		const RP_DESC = [
			'the BC counter pair',
			'the DE register pair',
			'the address pointer',
			'the read/write position (stack pointer)'
		];
		const RP2_DESC = [
			'the BC counter pair',
			'the DE register pair',
			'the address pointer',
			'the working value + status flags'
		];
		const CC_DESC = [
			'result was not zero',
			'result was zero',
			'no carry occurred',
			'a carry occurred',
			'parity is odd',
			'parity is even',
			'result is positive',
			'result is negative'
		];
		const ALUDESC = [
			'Add {0} to the working value',
			'Add {0} to the working value (with carry)',
			'Subtract {0} from the working value',
			'Subtract {0} from the working value (with borrow)',
			'Bitwise AND the working value with {0}',
			'Bitwise XOR the working value with {0}',
			'Bitwise OR the working value with {0}',
			'Compare the working value with {0} (updates status flags)'
		];
		if (x === 0) {
			if (z === 0) {
				if (y === 0) return 'No operation — does nothing, wastes one cycle';
				if (y === 1) return 'Swap the working value and status flags with a hidden backup copy';
				if (y === 2)
					return 'Subtract 1 from counter B; if B ≠ 0, jump by an offset (creates a tight loop)';
				if (y === 3) return 'Jump forward or backward by a short offset (unconditional)';
				return `Jump by a short offset if ${CC_DESC[y - 4]}`;
			}
			if (z === 1)
				return q === 0
					? `Load a 2-byte number from the code into ${RP_DESC[p]}`
					: `Add ${RP_DESC[p]} to the address pointer (shifts where it points)`;
			if (z === 2) {
				if (q === 0) {
					if (p === 0) return 'Write the working value into memory at the address in BC';
					if (p === 1) return 'Write the working value into memory at the address in DE';
					if (p === 2)
						return 'Write the address pointer (2 bytes) into memory at a specific address';
					return 'Write the working value into memory at a specific address';
				}
				if (p === 0) return 'Read a byte from memory (at the address in BC) into the working value';
				if (p === 1) return 'Read a byte from memory (at the address in DE) into the working value';
				if (p === 2) return 'Read 2 bytes from a specific memory address into the address pointer';
				return 'Read a byte from a specific memory address into the working value';
			}
			if (z === 3) return q === 0 ? `Add 1 to ${RP_DESC[p]}` : `Subtract 1 from ${RP_DESC[p]}`;
			if (z === 4)
				return y === 6
					? 'Add 1 to the byte in memory that the address pointer points to'
					: `Add 1 to ${R_DESC[y]}`;
			if (z === 5)
				return y === 6
					? 'Subtract 1 from the byte in memory that the address pointer points to'
					: `Subtract 1 from ${R_DESC[y]}`;
			if (z === 6)
				return y === 6
					? 'Write a byte from the code into the memory location the address pointer points to'
					: `Load a byte from the code into ${R_DESC[y]}`;
			// z === 7
			const rot = [
				'Rotate the working value left one bit (fast bit shift)',
				'Rotate the working value right one bit (fast bit shift)',
				'Rotate the working value left through the carry flag',
				'Rotate the working value right through the carry flag',
				'Adjust the working value for decimal (BCD) arithmetic',
				'Flip every bit of the working value (0↔1)',
				'Set the carry flag to 1',
				'Toggle the carry flag (0↔1)'
			];
			return rot[y];
		}
		if (x === 1) {
			if (y === 6 && z === 6) return 'Halt — stop executing until reset (cell goes dormant)';
			if (z === 6) return `Read a byte from memory (at the address pointer) into ${R_DESC[y]}`;
			if (y === 6) return `Write ${R_DESC[z]} into memory at the address pointer`;
			if (y === z) return `Copy ${R_DESC[y]} to itself (does nothing)`;
			return `Copy ${R_DESC[z]} into ${R_DESC[y]}`;
		}
		if (x === 2) {
			const target = z === 6 ? 'the byte at the address pointer' : R_DESC[z];
			return ALUDESC[y].replace('{0}', target);
		}
		// x === 3
		if (z === 0) return `Return from subroutine if ${CC_DESC[y]}`;
		if (z === 1) {
			if (q === 0) return `Read 2 bytes from memory (at the stack pointer) into ${RP2_DESC[p]}`;
			if (p === 0) return 'Return from subroutine (jump back to where we came from)';
			if (p === 1) return 'Swap BC, DE, and the address pointer with hidden backup copies';
			if (p === 2) return 'Jump to the memory address stored in the address pointer';
			return 'Set the stack pointer (read/write position) to the address pointer value';
		}
		if (z === 2) return `Jump to a specific address if ${CC_DESC[y]}`;
		if (z === 3) {
			if (y === 0) return 'Jump to a specific address (unconditional)';
			if (y === 1) return 'Prefix for extended bit/rotate instructions';
			if (y === 2)
				return 'Output to an I/O port — no device is attached, so this has no effect (a 2-byte no-op that skips the port number)';
			if (y === 3)
				return 'Input from an I/O port — no device is attached, so it always reads 0 into the working value (and skips the port-number byte)';
			if (y === 4)
				return "Swap the address pointer with the 2 bytes at the stack pointer. The stack starts at the end of the address space (inside the neighbor cell), so this is a key self-replication instruction — it moves data into the neighbor cell's memory";
			if (y === 5) return 'Swap the address pointer with the DE register pair';
			if (y === 6) return 'Disable interrupts (no interrupts exist here, so this has no effect)';
			return 'Enable interrupts (no interrupts exist here, so this has no effect)';
		}
		if (z === 4) return `Call a subroutine at a specific address if ${CC_DESC[y]}`;
		if (z === 5) {
			if (q === 0)
				return `Push ${RP2_DESC[p]} (2 bytes) onto the stack. The stack starts at the end of the address space (inside the neighbor cell), so pushing writes into the neighbor's memory — a building block of self-replication`;
			if (p === 0)
				return 'Call a subroutine at a specific address (pushes the return point onto the stack)';
			if (p === 1)
				return 'Index-register (IX) prefix — not implemented here, so it is skipped and the next instruction runs on the address pointer (HL)';
			if (p === 2) return 'Prefix for extended instructions (block copy, 16-bit memory moves)';
			return 'Index-register (IY) prefix — not implemented here, so it is skipped and the next instruction runs on the address pointer (HL)';
		}
		if (z === 6) {
			return ALUDESC[y].replace('{0}', 'a byte from the code');
		}
		// z === 7
		return `Save the return position on the stack and jump to the fixed low address ${(y * 8).toString(16).toUpperCase().padStart(2, '0')}h within this buffer (a compact call)`;
	}

	// Freq tile tooltip state
	let tileTooltip = $state<{
		byte: number;
		mnemonic: string;
		rank: number;
		x: number;
		y: number;
	} | null>(null);

	let tileTooltipSuppressUntil = 0;

	function showTileTooltip(e: MouseEvent, byte: number, mnemonic: string, rank: number) {
		if (Date.now() < tileTooltipSuppressUntil) return;
		const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
		tileTooltip = {
			byte,
			mnemonic,
			rank,
			x: rect.right + 6,
			y: rect.top - 6
		};
	}

	function hideTileTooltip() {
		tileTooltip = null;
	}

	function toggleChart() {
		showInfoChart = !showInfoChart;
		tileTooltip = null;
		tileTooltipSuppressUntil = Date.now() + 300;
	}

	function byteColor(b: number): string {
		const [r, g, bl] = unpackRGBA(colormap[b]);
		return `rgb(${r},${g},${bl})`;
	}

	function byteLuminance(b: number): number {
		const [r, g, bl] = unpackRGBA(colormap[b]);
		return (0.299 * r + 0.587 * g + 0.114 * bl) / 255;
	}

	/** Brightened color for chart lines — ensures visibility on dark backgrounds */
	function byteChartColor(b: number): string {
		const [r, g, bl] = unpackRGBA(colormap[b]);
		const lum = (0.299 * r + 0.587 * g + 0.114 * bl) / 255;
		if (lum < 0.2) {
			// Brighten dark colors: lift toward a lighter version
			const boost = 0.35;
			return `rgb(${Math.round(r + (255 - r) * boost)},${Math.round(g + (255 - g) * boost)},${Math.round(bl + (255 - bl) * boost)})`;
		}
		return `rgb(${r},${g},${bl})`;
	}

	/** Text color for cells — white on dark, black on light */
	function cellTextColor(b: number): string {
		const lum = byteLuminance(b);
		return lum > 0.4 ? 'rgba(0,0,0,0.85)' : 'rgba(255,255,255,0.9)';
	}

	interface GenomeCell {
		byteVal: number;
		label: string;
		isOpcode: boolean;
	}

	function buildGenomeGrid(data: Uint8Array, disasm: ReturnType<typeof disassemble>): GenomeCell[] {
		const cells: GenomeCell[] = [];
		// eslint-disable-next-line svelte/prefer-svelte-reactivity -- local variable, not reactive state
		const byteInfo = new Map<number, { label: string; isOpcode: boolean }>();

		for (const line of disasm) {
			byteInfo.set(line.offset, { label: line.mnemonic, isOpcode: true });
			for (let i = 1; i < line.length && line.offset + i < data.length; i++) {
				byteInfo.set(line.offset + i, {
					label: hexByte(data[line.offset + i]),
					isOpcode: false
				});
			}
		}

		for (let i = 0; i < data.length; i++) {
			const info = byteInfo.get(i);
			cells.push({
				byteVal: data[i],
				label: info ? info.label : hexByte(data[i]),
				isOpcode: info ? info.isOpcode : false
			});
		}
		return cells;
	}

	type SidebarTab = 'world' | 'measure' | 'experiments' | 'guide' | 'about';
	type DirectTool = 'navigate' | 'probe' | 'region' | 'marker' | 'history' | 'compare' | 'record';
	let activeTab = $state<SidebarTab>('world');
	let sidebarCollapsed = $state(false);
	let measureSection = $state('probe');
	let fieldMode = $state<FieldMode>('bytes');
	let activeMarkerPosition = $state(0);
	let activeMarkerValue = $state(165);
	let activeTool = $state<DirectTool>('navigate');
	let epoch = $state(0);
	let applied = $state.raw<LabSettings>({ ...ORIGINAL_SETTINGS });
	let worldDraft = $state<LabSettings>({ ...ORIGINAL_SETTINGS });
	let stagedInitial = $state.raw<InitialCondition>({ kind: 'random' });
	let stagedPreset = $state.raw<Preset | null>(null);
	let worldNotice = $state('');
	let worldSection = $state<'condition' | 'look'>('condition');
	const pendingWorld = $derived(
		JSON.stringify(worldDraft) !== JSON.stringify(applied) ||
			Boolean(stagedPreset) ||
			stagedExperimentPending
	);
	// `status` is the short name the footer appends when the edges are not walls.
	const boundaryOptions: { value: BoundaryRule; label: string; status: string }[] = [
		{ value: 'reflect', label: 'Walls (standard)', status: '' },
		{ value: 'cylinder', label: 'Cylinder', status: 'cylinder' },
		{ value: 'torus', label: 'Torus', status: 'torus' },
		{ value: 'mobius', label: 'Möbius strip', status: 'Möbius' },
		{ value: 'klein', label: 'Klein bottle', status: 'Klein' },
		{ value: 'projective', label: 'Projective plane', status: 'projective' }
	];
	const boundaryStatus = $derived(
		boundaryOptions.find((option) => option.value === applied.boundary)?.status ?? ''
	);
	const machineOptions = [
		{ value: 'z80-0.2', label: 'Z80 · compat 0.2' },
		{ value: 'z80-0.3', label: 'Z80 · 0.3 classic' },
		{ value: 'z80-nmos', label: 'Z80 · 0.3 Zilog NMOS' },
		{ value: 'm6502', label: 'NMOS 6502' },
		{ value: 'i8086', label: '8086 · coherent fetch' }
	];
	const fieldOptions = [
		{ value: 'bytes', label: 'Bytes' },
		{ value: 'identity', label: 'Tape identity' },
		{ value: 'confinement', label: 'Open / confined' },
		{ value: 'zeros', label: 'Zero density' },
		{ value: 'diversity', label: 'Local diversity' },
		{ value: 'marker', label: 'Marker matches' }
	];
	const shellTabs: { id: SidebarTab; name: string; glyph: string }[] = [
		{ id: 'world', name: 'World', glyph: '◎' },
		{ id: 'measure', name: 'Measure', glyph: '⌖' },
		{ id: 'experiments', name: 'Experiments', glyph: '⏣' },
		{ id: 'guide', name: 'Guide', glyph: '?' },
		{ id: 'about', name: 'About', glyph: 'i' }
	];
	const railTools: { id: DirectTool; label: string; key: string; path: string }[] = [
		{ id: 'navigate', label: 'Select / pan', key: 'V', path: 'M5 3v13l4-4 4 6 3-2-4-6 6-1Z' },
		{
			id: 'probe',
			label: 'Probe',
			key: 'P',
			path: 'M10 2v4m0 8v4M2 10h4m8 0h4M10 6a4 4 0 1 0 0 8 4 4 0 0 0 0-8'
		},
		{ id: 'region', label: 'Region', key: 'R', path: 'M3 7V3h4m6 0h4v4m0 6v4h-4m-6 0H3v-4' },
		{ id: 'marker', label: 'Marker', key: 'M', path: 'M6 3h8l-2 5 3 3H5l3-3-2-5Zm4 8v7' },
		{ id: 'compare', label: 'Compare', key: 'C', path: 'M10 2v16M7 4H3v12h4m6-12h4v12h-4' }
	];
	function selectTab(tab: SidebarTab) {
		activeTab = tab;
		sidebarCollapsed = false;
		if (tab === 'guide' && helpTab === 'about') helpTab = lastGuideTab;
		if (tab === 'about') {
			if (helpTab !== 'about') lastGuideTab = helpTab;
			helpTab = 'about';
		}
	}
	function activateTool(tool: DirectTool) {
		if (machineBusy || observationBusy || checkpointBusy) return;
		activeTool = tool;
		if (tool !== 'navigate') {
			selectTab('measure');
			measureSection = tool;
		}
		observation?.activateTool(tool);
	}
	function showMeasure(section: string) {
		selectTab('measure');
		measureSection = section;
		if (section !== 'charts' && section !== 'bench') activateTool(section as DirectTool);
	}
	function jumpWorld(section: 'condition' | 'look' = 'condition', target?: string) {
		selectTab('world');
		worldSection = section;
		if (target) requestAnimationFrame(() => document.getElementById(target)?.focus());
	}
	let lastProbeTarget = -1;
	function observationState(next: {
		tool: string;
		mode: FieldMode;
		update: number;
		selected: number | null;
		markerPosition?: number;
		markerValue?: number;
	}) {
		const toolChanged = activeTool !== next.tool;
		activeTool = next.tool as DirectTool;
		if (toolChanged && activeTab === 'measure' && !sidebarCollapsed && next.tool !== 'navigate')
			measureSection = next.tool;
		fieldMode = next.mode;
		if (next.markerPosition !== undefined) activeMarkerPosition = next.markerPosition;
		if (next.markerValue !== undefined) activeMarkerValue = next.markerValue;
		if (next.selected !== null && next.selected >= 0 && next.selected !== lastProbeTarget) {
			lastProbeTarget = next.selected;
			if (measureSection === 'bench' && inspectedTape && next.selected !== inspectedCell)
				void inspectCell(next.selected);
			else {
				selectTab('measure');
				measureSection = next.tool === 'marker' ? 'marker' : 'probe';
			}
		}
	}
	async function startNewWorld() {
		if (!engine || machineBusy || checkpointBusy) return;
		machineBusy = true;
		machineError = null;
		const next = $state.snapshot(worldDraft);
		const nextInitial = structuredClone(stagedInitial);
		const nextPreset = stagedPreset ? structuredClone(stagedPreset) : null;
		const nextExperimentId = selectedExperimentId;
		const stagedSignature = JSON.stringify([next, nextInitial, nextPreset, nextExperimentId]);
		try {
			await engine.configure(next, nextInitial);
			assignMachineSettings(next);
			applied = engine.settings;
			initialCondition = nextInitial;
			if (nextPreset)
				for (const def of SETTINGS)
					if (
						[
							'colormapName',
							'brightness',
							'contrast',
							'saturation',
							'showGridLines',
							'simpleView'
						].includes(def.key) &&
						def.key in nextPreset.values
					)
						def.set(nextPreset.values[def.key]);
			appliedExperimentId = nextExperimentId;
			const draftUnchanged =
				stagedSignature ===
				JSON.stringify([
					$state.snapshot(worldDraft),
					stagedInitial,
					stagedPreset,
					selectedExperimentId
				]);
			if (draftUnchanged) {
				worldDraft = structuredClone(applied);
				stagedPreset = null;
				stagedExperimentPending = false;
			}
			clearMeasurements();
			worldNotice = 'New world started. Your playback state is preserved.';
		} catch (error) {
			machineError = error instanceof Error ? error.message : String(error);
		} finally {
			machineBusy = false;
		}
	}
	function discardDraft() {
		worldDraft = structuredClone(applied);
		stagedInitial = initialCondition;
		stagedPreset = null;
		selectedExperimentId = '';
		stagedExperimentPending = false;
		worldNotice = '';
	}
	function stepOnce() {
		if (engine && !machineBusy && !checkpointBusy) {
			engine.simulateStep();
			epoch = engine.batchCount;
		}
	}
	function liveBudget(key: 'noiseExp' | 'pairCount' | 'z80Steps', value: number) {
		if (!engine || machineBusy || checkpointBusy) return;
		const previous = applied;
		if (key === 'noiseExp') noiseExp = engine.noiseExp = value;
		if (key === 'pairCount') pairCount = engine.pairCount = value;
		if (key === 'z80Steps') z80Steps = engine.z80Steps = value;
		applied = engine.settings;
		syncLiveDraft(previous, applied);
	}
	function syncLiveDraft(previous: LabSettings, next: LabSettings) {
		if (stagedPreset || stagedExperimentPending) return;
		const patch: Partial<LabSettings> = {};
		for (const key of [
			'isa',
			'compat',
			'variant',
			'mutationExponent',
			'pairAttempts',
			'steps',
			'lethalProbability',
			'ablation',
			'stack',
			'dialMode',
			'interaction',
			'dissipation',
			'boundary'
		] as const) {
			if (JSON.stringify(worldDraft[key]) === JSON.stringify(previous[key]))
				Object.assign(patch, { [key]: next[key] });
		}
		// The draft may already be staged as hex while the running square world changes its edges.
		worldDraft = hexSafe({ ...$state.snapshot(worldDraft), ...patch });
	}
	// Only walls exist on a hex lattice; any other edge rule would fail validation.
	function hexSafe(next: LabSettings): LabSettings {
		return next.topology === 'hex' && next.boundary !== 'reflect'
			? { ...next, boundary: 'reflect' }
			: next;
	}
	function setDraftTopology(topology: GridType) {
		if (worldDraft.topology === topology) return;
		worldDraft.topology = topology;
		// Back on square, the draft picks the running world's edges up again.
		worldDraft.boundary = topology === 'hex' ? 'reflect' : applied.boundary;
	}
	function keyboard(event: KeyboardEvent) {
		keyboardModality = true;
		if (event.metaKey || event.ctrlKey) {
			if (event.code === 'KeyS') {
				event.preventDefault();
				void handleSave();
			}
			if (event.code === 'KeyO') {
				event.preventDefault();
				fileInput?.click();
			}
			return;
		}
		const target = event.target;
		if (
			target instanceof HTMLElement &&
			target.closest(
				'input, textarea, select, [contenteditable=true], [role=combobox], [role=listbox]'
			)
		)
			return;
		if (event.code === 'Space') {
			event.preventDefault();
			togglePlay();
		} else if (event.key === '.') {
			event.preventDefault();
			stepOnce();
		} else if (event.key === '?') {
			selectTab('guide');
			helpTab = 'keys';
		} else if (event.key >= '1' && event.key <= '5') selectTab(shellTabs[Number(event.key) - 1].id);
		else if (event.code === 'KeyF') observation?.fitView();
		else if (event.key === '+' || event.key === '=') observation?.zoomBy(1 / 1.5);
		else if (event.key === '-') observation?.zoomBy(1.5);
		else if (event.code === 'KeyS') showMeasure('charts');
		else if (event.code === 'KeyH') selectTab('guide');
		else if (event.key === 'Escape') {
			inspectedTape = null;
			speciesModal = null;
			overwriteTarget = null;
			activateTool('navigate');
		} else {
			const tool = railTools.find((entry) => entry.key.toLowerCase() === event.key.toLowerCase());
			if (tool) {
				event.preventDefault();
				activateTool(tool.id);
			}
		}
	}
</script>

<svelte:window onpointerdown={() => (keyboardModality = false)} onkeydown={keyboard} />
<svelte:head><title>Algocell · Digital primordial soup</title></svelte:head>

<div
	class="legacy-scope unified-shell"
	class:pointer-focus={!keyboardModality}
	class:sidebar-collapsed={sidebarCollapsed}
>
	<input
		bind:this={fileInput}
		type="file"
		accept=".acell,.png"
		onchange={handleFileInputChange}
		class="sr-only"
		aria-label="Import checkpoint file"
	/>
	<header class="app-header">
		<div class="wordmark">
			<svg viewBox="0 0 24 24" width="22" height="22" aria-hidden="true"
				><path
					d="M4 4h6v6H4zm10 0h6v6h-6zM4 14h6v6H4zm10 0h6v6h-6z"
					fill="currentColor"
					opacity=".7"
				/><path d="M10 10h4v4h-4z" fill="currentColor" /></svg
			><span>Algocell</span>
		</div>
		<div class="field-choice">
			<InstrumentSelect
				options={fieldOptions}
				value={fieldMode}
				label="Field view"
				disabled={!engine || machineBusy || observationBusy || checkpointBusy}
				onchange={(value) => void observation?.changeMode(value as FieldMode)}
			/>
		</div>
		<button
			class="header-look"
			class:chosen={activeTab === 'world' && worldSection === 'look'}
			onclick={() => jumpWorld('look')}>Look</button
		>
		<div class="camera-controls">
			<button aria-label="Zoom out" title="Zoom out (−)" onclick={() => observation?.zoomBy(1.5)}
				>−</button
			><button aria-label="Zoom in" title="Zoom in (+)" onclick={() => observation?.zoomBy(1 / 1.5)}
				>+</button
			><button title="Fit lattice (F)" onclick={() => observation?.fitView()}>Fit</button>
		</div>
		<div class="header-space"></div>
	</header>
	<div
		bind:this={canvasContainer}
		class="canvas-viewport"
		ondragover={(event) => event.preventDefault()}
		ondrop={handleDrop}
		role="region"
		aria-label="Live lattice"
	>
		{#if gpuError}<div class="gpu-message">
				<h2>A local machine-code world</h2>
				<p>{gpuError}</p>
				<button onclick={() => selectTab('guide')}>Open guide</button>
			</div>{/if}
		<canvas
			bind:this={canvas}
			class="lattice-canvas"
			tabindex="0"
			aria-label="Lattice canvas"
			onpointerdown={(event) => event.currentTarget.focus({ preventScroll: true })}
			style="cursor:{isPanning
				? 'grabbing'
				: activeTool === 'navigate'
					? 'grab'
					: 'crosshair'};touch-action:none"
			onmousemove={handleCanvasMouseMove}
			onmouseleave={handleCanvasMouseLeave}
			onmousedown={handleCanvasMouseDown}
			onmouseup={handleCanvasMouseUp}
			onwheel={handleCanvasWheel}
			ondblclick={handleCanvasDblClick}
			ontouchstart={handleTouchStart}
			ontouchmove={handleTouchMove}
			ontouchend={handleTouchEnd}
			oncontextmenu={(event) => event.preventDefault()}
		></canvas>
		<div class="viewport-caption" aria-label="Field legend">
			{#if fieldMode === 'confinement'}<div class="viewport-key">
					<span><i style="background:rgb(12% 12% 15%)"></i>Unknown</span><span
						><i style="background:rgb(25% 52% 72%)"></i>No exit</span
					><span><i style="background:rgb(90% 58% 28%)"></i>Outside A</span>
				</div>{:else if fieldMode === 'zeros' || fieldMode === 'diversity'}<div
					class="viewport-key"
				>
					<span>0%</span><i class="field-gradient" class:diversity={fieldMode === 'diversity'}
					></i><span>100%</span>
				</div>{:else if fieldMode === 'marker'}<div class="viewport-key">
					<span
						><i style="background:rgb(98% 68% 35%)"></i>Byte {activeMarkerPosition} = {hexByte(
							activeMarkerValue
						)}</span
					><span><i style="background:rgb(12% 13% 15%)"></i>Other</span>
				</div>{/if}<span
				>{fieldMode === 'bytes'
					? 'Inherited byte values'
					: fieldMode === 'identity'
						? 'Complete-tape fingerprints · collisions possible'
						: fieldMode === 'confinement'
							? 'Last initiating encounter · outside A includes padding'
							: fieldMode === 'zeros'
								? 'Zero bytes per tape · 0–100%'
								: fieldMode === 'diversity'
									? 'Distinct tapes in 3 × 3 coordinates · 0–100%'
									: 'Same-position value matches · not ancestry'}</span
			>
		</div>
	</div>
	<nav class="tool-rail" aria-label="Lattice tools">
		{#each railTools as tool (tool.id)}<button
				class:active={activeTool === tool.id}
				aria-label={tool.label}
				disabled={!engine || machineBusy || observationBusy || checkpointBusy}
				aria-pressed={activeTool === tool.id}
				title={`${tool.label} (${tool.key})`}
				onclick={() => activateTool(tool.id)}
				><svg
					width="20"
					height="20"
					viewBox="0 0 20 20"
					fill="none"
					stroke="currentColor"
					stroke-width="1.2"
					stroke-linejoin="round"
					stroke-linecap="round"
					aria-hidden="true"><path d={tool.path} /></svg
				></button
			>{/each}
		<div class="rail-space"></div>
		<button
			aria-label={sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
			title={sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
			onclick={() => (sidebarCollapsed = !sidebarCollapsed)}
			><span aria-hidden="true">{sidebarCollapsed ? '‹' : '›'}</span></button
		>
	</nav>
	<aside class="app-sidebar" aria-label="Instrument sidebar">
		<div class="sidebar-tabs" role="tablist" aria-label="Workspace">
			{#each shellTabs as tab (tab.id)}<button
					role="tab"
					aria-label={tab.name}
					aria-selected={activeTab === tab.id}
					class:active={activeTab === tab.id}
					title={`${tab.name} (${shellTabs.indexOf(tab) + 1})`}
					onclick={() => selectTab(tab.id)}
					><span class="tab-glyph" aria-hidden="true">{tab.glyph}</span><span class="tab-label"
						>{tab.name}</span
					></button
				>{/each}
		</div>
		<div class="sidebar-scroll">
			<section
				class="world-panel sidebar-section"
				hidden={activeTab !== 'world' || sidebarCollapsed}
				aria-label="World settings"
			>
				<div class="section-heading">
					<h1>
						{worldSection === 'look' ? 'Look' : 'World'}
					</h1>
					<span class="section-context"
						>{worldSection === 'condition'
							? 'A continuous experiment'
							: 'The same living lattice'}</span
					>
				</div>
				<nav class="section-tabs" aria-label="World sections">
					<button
						class:active={worldSection === 'condition'}
						onclick={() => (worldSection = 'condition')}>Condition</button
					><button class:active={worldSection === 'look'} onclick={() => (worldSection = 'look')}
						>Appearance</button
					>
				</nav>
				{#if worldSection === 'condition'}
					<p class="intro-copy">
						Small programs meet and change each other. Follow the lattice, or begin with a question.
					</p>
					<button class="text-action" onclick={() => selectTab('experiments')}
						>Explore guided experiments <span>↗</span></button
					>
					<div class="section-divider"></div>
					<fieldset class="control-lock" disabled={!engine || machineBusy || checkpointBusy}>
						<div class="section-title">
							<h2>Live condition</h2>
							<span class="live-label">Applies while running</span>
						</div>
						<div class="control-field">
							<label for="machine-select">Processor</label><InstrumentSelect
								id="machine-select"
								label="Processor"
								options={machineOptions}
								value={machine}
								disabled={machineBusy || checkpointBusy || !engine}
								onchange={(value) => void chooseMachine(value)}
							/>
							<p class="control-note">Switch cores using the bytes already on the lattice.</p>
							{#if coreNotice}<p class="inline-notice core-notice" role="status">
									{coreNotice}
								</p>{/if}
						</div>
						<div class="control-field">
							<div class="label-row">
								<label for="lethal-probability">Lethal zero</label><output
									>{Math.round(lethalProbability * 100)}%</output
								>
							</div>
							<input
								id="lethal-probability"
								aria-label="Lethal zero"
								type="range"
								min="0"
								max="1"
								step="0.01"
								bind:value={lethalProbability}
								onchange={() => void updateMachine()}
								disabled={machineBusy || checkpointBusy}
							/>
							<p class="control-note">Chance that a zero instruction stops an encounter.</p>
						</div>
						<div class="control-field">
							<label for="interaction-rule">Interaction</label><InstrumentSelect
								id="interaction-rule"
								label="Interaction"
								value={interaction}
								options={[
									{ value: 'overwrite', label: 'Overwrite (standard)' },
									{ value: 'reversible', label: 'Reversible (now and before)' }
								]}
								onchange={(value) => {
									interaction = value as LabSettings['interaction'];
									void updateMachine();
								}}
								disabled={machineBusy || checkpointBusy || !engine}
							/>
							<p class="control-note">
								Overwrite replaces both tapes with the result. Reversible keeps each cell's previous
								tape and applies the change on top of it, so every event can be undone.
							</p>
						</div>
						{#if interaction === 'reversible'}
							<div class="control-field">
								<div class="label-row">
									<label for="dissipation">Dissipation</label><output
										>{Math.round(dissipation * 100)}%</output
									>
								</div>
								<input
									id="dissipation"
									aria-label="Dissipation"
									type="range"
									min="0"
									max="1"
									step="0.01"
									bind:value={dissipation}
									onchange={() => void updateMachine()}
									disabled={machineBusy || checkpointBusy || !engine}
								/>
								<p class="control-note">
									Chance that a cell forgets its previous tape before an event. 100% is the standard
									rule; 0% is a closed, fully reversible world.
								</p>
							</div>
						{/if}
						<div class="control-field">
							<label for="boundary-rule">Edges</label><InstrumentSelect
								id="boundary-rule"
								label="Edges"
								value={boundary}
								options={gridType === 'hex' ? boundaryOptions.slice(0, 1) : boundaryOptions}
								onchange={(value) => {
									boundary = value as BoundaryRule;
									void updateMachine();
								}}
								disabled={machineBusy || checkpointBusy || !engine || gridType === 'hex'}
							/>
							<p class="control-note">
								What a cell at the edge meets. Walls turn it inward; the others glue opposite edges,
								some with a flip.{gridType === 'hex'
									? ' Joined edges need the square lattice.'
									: ''}
							</p>
						</div>
						<div class="control-field">
							<div class="label-row">
								<label for="noise-input">Mutation rate</label><output
									>{noiseExp >= 31 ? 'Off' : `1 / ${2 ** noiseExp}`}</output
								>
							</div>
							<input
								id="noise-input"
								aria-label="Mutation rate"
								type="range"
								min="1"
								max="12"
								value={noiseExp >= 31 ? 1 : 13 - noiseExp}
								oninput={(event) => liveBudget('noiseExp', 13 - Number(event.currentTarget.value))}
							/><label class="inline-check"
								><input
									type="checkbox"
									checked={noiseExp >= 31}
									onchange={(event) => liveBudget('noiseExp', event.currentTarget.checked ? 31 : 4)}
								/> Mutation off</label
							>
						</div>
						<details class="shell-disclosure">
							<summary>Budgets & instruction ablation</summary>
							<div class="disclosure-body">
								<div class="control-field">
									<div class="label-row">
										<label for="pairs-input">Pairs / update</label><output
											>{pairCount.toLocaleString()}</output
										>
									</div>
									<input
										id="pairs-input"
										aria-label="Pairs per update"
										type="range"
										min="256"
										max={MAX_BATCH_PAIR_N}
										step="256"
										value={pairCount}
										oninput={(event) => liveBudget('pairCount', Number(event.currentTarget.value))}
									/>
								</div>
								<div class="control-field">
									<div class="label-row">
										<label for="steps-input">Execution steps</label><output>{z80Steps}</output>
									</div>
									<input
										id="steps-input"
										aria-label="Execution steps"
										type="range"
										min="16"
										max="1024"
										step="16"
										value={z80Steps}
										oninput={(event) => liveBudget('z80Steps', Number(event.currentTarget.value))}
									/>
								</div>
								<label class="field-label" for="ablation-input">Suppress instructions</label><input
									id="ablation-input"
									class="shell-input"
									aria-label="Suppress instructions"
									placeholder={isa === 'z80'
										? 'family:block-copy; LDIR; PUSH'
										: isa === 'm6502'
											? 'family:stack; STA; opcode:8D'
											: 'family:block-copy; MOV; opcode:A4'}
									bind:value={suppressInput}
									onkeydown={(event) => {
										if (event.key === 'Enter' && suppressInput.trim()) {
											addSuppressPattern(suppressInput);
											suppressInput = '';
										}
									}}
								/>
								{#if suppressInput.trim()}<p class="control-note">
										Enter to suppress {suppressionPreview(suppressInput)} instructions.
									</p>{/if}
								<div class="suppress-chips">
									{#each suppressPatterns as pat (pat)}<button
											class="suppress-chip"
											title={`Remove pattern: ${pat}`}
											onclick={() => removeSuppressPattern(pat)}>{pat} ×</button
										>{/each}
								</div>
								{#if suppressPatterns.length}<div class="label-row">
										<span class="control-note">{suppressedTotal} instructions suppressed</span
										><button class="quiet-button" onclick={clearSuppressedOpcodes}>Clear all</button
										>
									</div>{/if}
								<p class="control-note">
									{isa === 'z80'
										? 'Suppress a mnemonic, family or exact opcode. Bytes stay in the lattice; their execution changes.'
										: isaSuppressionDescription}
								</p>
							</div>
						</details>
						<div class="section-divider"></div>
						<div class="section-title">
							<h2>Next world</h2>
							<span class:pending={pendingWorld} class="draft-state"
								>{pendingWorld ? 'Changes staged' : 'Current settings'}</span
							>
						</div>
						<p class="control-note">
							Edit the starting condition. Your current world keeps running until you apply it.
						</p>
						<div class="control-field">
							<label for="seed-input">Seed</label><input
								id="seed-input"
								class="shell-input"
								aria-label="Seed"
								type="number"
								min="0"
								bind:value={worldDraft.seed}
							/>
						</div>
						<div class="control-field">
							<span class="field-label">Lattice</span>
							<div class="segmented">
								<button
									class:active={worldDraft.topology === 'square'}
									onclick={() => setDraftTopology('square')}>Square</button
								><button
									class:active={worldDraft.topology === 'hex'}
									onclick={() => setDraftTopology('hex')}>Hexagonal</button
								>
							</div>
						</div>
						<div class="field-pair">
							<label
								>Width<input
									class="shell-input"
									aria-label="Lattice width"
									type="number"
									min="2"
									max="400"
									bind:value={worldDraft.width}
								/></label
							><span>×</span><label
								>Height<input
									class="shell-input"
									aria-label="Lattice height"
									type="number"
									min="2"
									max="400"
									bind:value={worldDraft.height}
								/></label
							>
						</div>
						<div class="field-pair">
							<label
								>Tape length<input
									id="tape-length"
									class="shell-input"
									aria-label="Tape length"
									type="number"
									min="4"
									max="128"
									bind:value={worldDraft.length}
									oninput={(event) => {
										const length = event.currentTarget.valueAsNumber;
										if (Number.isInteger(length) && length >= 4 && length <= 128)
											worldDraft.ring = Math.max(worldDraft.ring, length * 2);
									}}
								/></label
							><span>/</span><label
								>Ring length<input
									id="pair-length"
									class="shell-input"
									aria-label="Ring length"
									type="number"
									min={worldDraft.length * 2}
									max="256"
									bind:value={worldDraft.ring}
								/></label
							>
						</div>
						<details class="shell-disclosure">
							<summary>Encounter reset</summary>
							<div class="disclosure-body">
								<label class="field-label" for="stack-convention">Stack placement</label
								><InstrumentSelect
									id="stack-convention"
									label="Stack placement"
									value={worldDraft.stack}
									options={[
										{ value: 'top', label: 'Top at B’s last byte' },
										{ value: 'first-push', label: 'First push at B’s last byte' },
										...(worldDraft.isa === 'z80'
											? [{ value: 'legacy-sp', label: 'Original SP = FFFF' }]
											: [])
									]}
									onchange={(value) => (worldDraft.stack = value as LabSettings['stack'])}
								/><label class="inline-check"
									><input type="checkbox" bind:checked={worldDraft.randomRegisters} /> Random registers</label
								><label class="field-label" for="dial-mode">Dial stream</label><InstrumentSelect
									id="dial-mode"
									label="Dial stream"
									value={worldDraft.dialMode}
									options={[
										{ value: 'historical-pcg', label: 'Historical PCG' },
										{ value: 'counter', label: 'Counter' }
									]}
									onchange={(value) => (worldDraft.dialMode = value as LabSettings['dialMode'])}
								/>
							</div>
						</details>
						<div class="apply-world">
							{@render nextWorldSummary()}
							<button
								class="primary-action"
								disabled={!engine || machineBusy || checkpointBusy}
								onclick={() => void startNewWorld()}>Start new world with these settings</button
							>{#if pendingWorld}<button class="quiet-button" onclick={discardDraft}
									>Discard staged changes</button
								>{/if}
						</div>
					</fieldset>
				{:else if worldSection === 'look'}
					<p class="intro-copy">
						Change how you see the world. Its bytes, time and camera stay yours.
					</p>
					<div class="control-field">
						<span class="field-label">Byte palette</span>
						<div class="palette-options">
							{#each COLORMAP_NAMES as name (name)}<button
									class:active={colormapName === name}
									onclick={() => handleColormapChange(name)}
									><span class={`cmap-preview cmap-preview-${name}`}></span>{name}</button
								>{/each}
						</div>
					</div>
					<div class="control-field">
						<span class="field-label">Tape detail</span>
						<div class="segmented">
							<button
								class:active={!simpleView}
								title="Detailed — show individual bytes"
								onclick={() => (simpleView = false)}>Individual bytes</button
							><button
								class:active={simpleView}
								title="Simple — one color per cell"
								onclick={() => (simpleView = true)}>Average colour</button
							>
						</div>
					</div>
					<label class="inline-check"
						><input type="checkbox" bind:checked={showGridLines} /> Grid lines</label
					>
					<div class="section-divider"></div>
					<h2>Colour adjustments</h2>
					{#each ['Brightness', 'Contrast', 'Saturation'] as label (label)}<div
							class="control-field"
						>
							<div class="label-row">
								<label for={`colour-${label}`}>{label}</label><output
									>{(label === 'Brightness'
										? brightness
										: label === 'Contrast'
											? contrast
											: saturation
									).toFixed(2)}</output
								>
							</div>
							<input
								id={`colour-${label}`}
								aria-label={label}
								type="range"
								min={label === 'Brightness' ? -0.5 : label === 'Contrast' ? 0.3 : 0}
								max={label === 'Brightness' ? 0.5 : label === 'Contrast' ? 2.5 : 3}
								step=".01"
								value={label === 'Brightness'
									? brightness
									: label === 'Contrast'
										? contrast
										: saturation}
								oninput={(event) => {
									const v = Number(event.currentTarget.value);
									if (label === 'Brightness') brightness = v;
									else if (label === 'Contrast') contrast = v;
									else saturation = v;
								}}
							/>
						</div>{/each}
					<button
						class="quiet-button"
						onclick={() => {
							brightness = 0;
							contrast = 1;
							saturation = 1;
						}}>Reset colour adjustments</button
					>
				{/if}
				{#if machineBusy}<p class="inline-notice" role="status">
						Applying the condition… playback will continue.
					</p>{/if}{#if machineError}<p class="instrument-error" role="alert">
						{machineError}
					</p>{/if}{#if worldNotice && worldSection === 'condition'}<p
						class="inline-notice"
						role="status"
					>
						{worldNotice}
					</p>{/if}
			</section>
			<div
				class="measure-heading sidebar-section"
				hidden={activeTab !== 'measure' || sidebarCollapsed}
			>
				<div class="section-heading">
					<h1>Measure</h1>
					<span class="section-context">Evidence in context</span>
				</div>
				<nav class="measure-sections" aria-label="Measure sections">
					{#each [{ id: 'probe', label: 'Probe' }, { id: 'region', label: 'Region' }, { id: 'marker', label: 'Markers' }, { id: 'history', label: 'History' }, { id: 'compare', label: 'Compare' }, { id: 'charts', label: 'Charts' }, { id: 'record', label: 'Export' }] as section (section.id)}<button
							class:active={measureSection === section.id}
							onclick={() => showMeasure(section.id)}>{section.label}</button
						>{/each}
					{#if inspectedTape || speciesModal}<button
							class:active={measureSection === 'bench'}
							onclick={() => showMeasure('bench')}>Encounter bench</button
						>{/if}
				</nav>
			</div>
			{#if activeTab === 'measure' && measureSection === 'probe' && activeTool === 'navigate' && !sidebarCollapsed && cellData && hoveredCell >= 0}<section
					class="sidebar-section hover-preview"
					aria-label="Tape preview"
				>
					<h2>Hover preview · cell {hoveredCell}</h2>
					<div class="tip-grid inline-tape-grid">
						{#each buildGenomeGrid(cellData, disasmLines) as cell, index (index)}<div
								class="tip-cell"
								class:operand={!cell.isOpcode}
								style={`background:${byteColor(cell.byteVal)};color:${cellTextColor(cell.byteVal)}`}
							>
								{cell.label}
							</div>{/each}
					</div>
				</section>{/if}
			{#if engine}<ObservationTools
					bind:this={observation}
					{engine}
					{canvas}
					{playing}
					embedded={true}
					panelVisible={activeTab === 'measure' &&
						measureSection !== 'charts' &&
						measureSection !== 'bench' &&
						!sidebarCollapsed}
					onstate={observationState}
					externalBusy={machineBusy || checkpointBusy}
					benchOpen={measureSection === 'bench'}
					onbusy={(value) => (observationBusy = value)}
					condition={EXPERIMENT_PRESETS.find((preset) => preset.id === appliedExperimentId)?.name ??
						''}
					{byteColor}
					onpause={() => {}}
					oninspect={(index) => void inspectCell(index)}
				/>{/if}
			<section
				class="sidebar-section checkpoint-section"
				hidden={activeTab !== 'measure' || measureSection !== 'record' || sidebarCollapsed}
				aria-label="Checkpoints"
			>
				<h2>Checkpoints</h2>
				<p class="intro-copy">
					Save the population and its condition to continue from these bytes.
				</p>
				<button
					class="primary-action"
					title="Save checkpoint (Cmd+S)"
					onclick={() => void handleSave()}
					disabled={!engine || machineBusy || observationBusy || checkpointBusy || cultureBusy}
					>{checkpointBusy ? 'Saving checkpoint…' : 'Save checkpoint'}</button
				><button
					class="secondary-action"
					title="Load checkpoint (Cmd+O)"
					onclick={() => fileInput?.click()}>Import checkpoint</button
				>{#if pendingCheckpoint}<div class="inline-confirmation">
						<h2>Restore checkpoint</h2>
						<p>{pendingCheckpoint.name}</p>
						<p class="control-note">
							{pendingCheckpoint.metadata.gridWidth} × {pendingCheckpoint.metadata.gridHeight} tapes ·
							seed {pendingCheckpoint.metadata.seed}. This replaces the current population.
						</p>
						<button
							class="primary-action"
							disabled={machineBusy || checkpointBusy}
							onclick={() => void restoreCheckpoint()}>Restore checkpoint</button
						><button class="quiet-button" onclick={() => (pendingCheckpoint = null)}
							>Cancel import</button
						>
					</div>{/if}
				<p class="control-note">
					Checkpoints keep bytes and settings on your computer. They do not preserve every random
					lane’s continuation.
				</p>
				<p class="control-note">Download measurement data beside each result in Measure.</p>
				{#if machineError}<p class="instrument-error" role="alert">{machineError}</p>{/if}
			</section>
			<section
				class="sidebar-section charts-panel"
				hidden={activeTab !== 'measure' || measureSection !== 'charts' || sidebarCollapsed}
				aria-label="Population charts"
			>
				<!-- Bottom-left info bar -->
				<div class="info-bar">
					<div class="info-row">
						<span
							class="info-item"
							title="Instruction-slot budget per second, including unused slots in halted encounters"
						>
							<span class="info-label">ops/s</span>
							<span class="info-value" data-numeric>{formatNumber(opsPerSec)}</span>
						</span>
						<span class="info-sep"></span>
						<span class="info-item">
							<span class="info-label">mut</span>
							<span class="info-value"
								>{#if noiseExp >= 31}off{:else}1/2<sup>{noiseExp}</sup>{/if}</span
							>
						</span>
						<button class="info-chart-toggle" onclick={toggleChart} title="Toggle chart">
							<svg
								width="10"
								height="10"
								viewBox="0 0 10 10"
								fill="currentColor"
								style="opacity:{showInfoChart ? 1 : 0.4}"
							>
								<polyline
									points="0,8 3,4 6,6 10,1"
									fill="none"
									stroke="currentColor"
									stroke-width="1.5"
								/>
							</svg>
						</button>
					</div>
					{#if showInfoChart}
						<div class="info-tabs">
							<button
								class="info-tab"
								class:active={infoTab === 'opcodes'}
								onclick={() => (infoTab = 'opcodes')}
								title="Opcode frequency — most common byte values for the selected processor over time"
								>Opcodes</button
							>
							<button
								class="info-tab"
								class:active={infoTab === 'species'}
								onclick={() => (infoTab = 'species')}
								title="Species — distinct cell genomes (by content hash) and their populations"
								>Species</button
							>
							<button
								class="info-tab"
								class:active={infoTab === 'diversity'}
								onclick={() => (infoTab = 'diversity')}
								title="Diversity — richness and Shannon/Simpson indices of the species distribution"
								>Diversity</button
							>
						</div>
					{/if}
					{#if showInfoChart && infoTab === 'opcodes' && chartVersion > 0 && chartLen > 1}
						<div class="info-chart">
							<svg
								viewBox="0 0 200 120"
								class="info-chart-svg"
								role="img"
								aria-label="Historical measurement trend"
								preserveAspectRatio="none"
								onmouseleave={() => {
									chartHoveredByte = -1;
								}}
							>
								{#each freqLines as fl (fl.byte)}
									{#if fl.path}
										<polyline
											points={fl.path.replace(/[ML]/g, (m) => (m === 'M' ? '' : ' ')).trim()}
											fill="none"
											stroke={fl.color}
											stroke-width={chartHoveredByte === fl.byte ? '4' : '2.5'}
											opacity={chartHoveredByte >= 0
												? chartHoveredByte === fl.byte
													? '1'
													: '0.15'
												: '0.8'}
										/>
									{/if}
								{/each}
							</svg>
							<div class="freq-grid">
								{#each displayTiles as tile, i (tile.type === 'suppressed' ? 'suppressed' : tile.byte)}
									{#if tile.type === 'suppressed'}
										<div class="freq-cell freq-cell-suppressed-group" role="button" tabindex="0">
											<span class="freq-label">{tile.mnemonic}</span>
										</div>
									{:else}
										<div
											class="freq-cell"
											class:dimmed={chartHoveredByte >= 0 && chartHoveredByte !== tile.byte}
											class:highlighted={chartHoveredByte === tile.byte}
											style="background:{byteColor(tile.byte)};color:{cellTextColor(tile.byte)}"
											role="button"
											tabindex="0"
											onclick={() => toggleSuppressOpcode(tile.byte)}
											onkeydown={(e) => {
												if (e.key === 'Enter' || e.key === ' ') {
													e.preventDefault();
													toggleSuppressOpcode(tile.byte);
												}
											}}
											onmouseenter={(e) => {
												chartHoveredByte = tile.byte;
												showTileTooltip(e, tile.byte, tile.mnemonic, i + 1);
											}}
											onmouseleave={() => {
												chartHoveredByte = -1;
												hideTileTooltip();
											}}
										>
											<span class="freq-rank">{i + 1}</span>
											<span class="freq-label">{tile.mnemonic || hexByte(tile.byte)}</span>
										</div>
									{/if}
								{/each}
							</div>
							{#if suppressPatterns.length > 0}
								<div class="freq-suppressed-row">
									{#each suppressPatterns as pat (pat)}
										<button
											class="freq-suppress-chip"
											onclick={() => removeSuppressPattern(pat)}
											title="Remove pattern: {pat}"
										>
											{pat}
											<svg
												width="7"
												height="7"
												viewBox="0 0 8 8"
												stroke="currentColor"
												stroke-width="1.5"
											>
												<path d="M1.5 1.5l5 5M6.5 1.5l-5 5" stroke-linecap="round" />
											</svg>
										</button>
									{/each}
									<span class="freq-suppress-count" title={describeSuppression(suppression)}
										>⊘ {suppressedTotal}</span
									>
								</div>
							{/if}
						</div>
					{:else if showInfoChart && infoTab === 'species'}
						<div class="info-chart">
							{#if speciesLen > 1 && speciesLines.length > 0}
								<svg
									viewBox="0 0 200 120"
									class="info-chart-svg"
									role="img"
									aria-label="Historical measurement trend"
									preserveAspectRatio="none"
									onmouseleave={() => {
										speciesHoveredIdx = -1;
									}}
								>
									{#each speciesLines as sl, idx (sl.hash)}
										{#if sl.path}
											<!-- Thick invisible hit area for hover -->
											<polyline
												points={sl.path.replace(/[ML]/g, (m) => (m === 'M' ? '' : ' ')).trim()}
												fill="none"
												stroke="transparent"
												role="presentation"
												stroke-width="12"
												onmouseenter={() => {
													speciesHoveredIdx = idx;
												}}
											/>
											<polyline
												points={sl.path.replace(/[ML]/g, (m) => (m === 'M' ? '' : ' ')).trim()}
												fill="none"
												stroke={sl.color}
												stroke-width={speciesHoveredIdx === idx ? '4' : '2.5'}
												opacity={speciesHoveredIdx >= 0
													? speciesHoveredIdx === idx
														? '1'
														: '0.15'
													: '0.8'}
												style="pointer-events:none"
											/>
										{/if}
									{/each}
								</svg>
							{:else}
								<div class="species-waiting">Collecting species data...</div>
							{/if}
							{#if currentSpecies}
								<div class="freq-grid">
									{#each currentSpecies.species.slice(0, MAX_SPECIES_DISPLAY) as sp, i (sp.hash)}
										{@const chartIdx = speciesHashOrder.indexOf(sp.hash)}
										<div
											class="freq-cell"
											class:dimmed={speciesHoveredIdx >= 0 && speciesHoveredIdx !== chartIdx}
											class:highlighted={speciesHoveredIdx === chartIdx}
											style="background:{sp.color};color:{sp.textColor}"
											role="button"
											tabindex="0"
											title="Click to inspect"
											onmouseenter={() => {
												speciesHoveredIdx = chartIdx;
											}}
											onmouseleave={() => {
												speciesHoveredIdx = -1;
											}}
											onclick={() => {
												speciesModal = sp;
												speciesSampleUpdate = engine?.batchCount ?? 0;
												measureSection = 'bench';
												inspectedTape = null;
											}}
											onkeydown={(e) => {
												if (e.key === 'Enter' || e.key === ' ') {
													e.preventDefault();
													speciesModal = sp;
													speciesSampleUpdate = engine?.batchCount ?? 0;
													measureSection = 'bench';
													inspectedTape = null;
												}
											}}
										>
											<span class="freq-rank">{i + 1}</span>
											<span class="freq-label">{(sp.fraction * 100).toFixed(1)}%</span>
										</div>
									{/each}
								</div>
							{/if}
							{#if suppressPatterns.length > 0}
								<div class="freq-suppressed-row">
									{#each suppressPatterns as pat (pat)}
										<button
											class="freq-suppress-chip"
											onclick={() => removeSuppressPattern(pat)}
											title="Remove pattern: {pat}"
										>
											{pat}
											<svg
												width="7"
												height="7"
												viewBox="0 0 8 8"
												stroke="currentColor"
												stroke-width="1.5"
											>
												<path d="M1.5 1.5l5 5M6.5 1.5l-5 5" stroke-linecap="round" />
											</svg>
										</button>
									{/each}
									<span class="freq-suppress-count" title={describeSuppression(suppression)}
										>⊘ {suppressedTotal}</span
									>
								</div>
							{/if}
						</div>
					{:else if showInfoChart && infoTab === 'diversity'}
						<div class="info-chart">
							{#if diversityLen > 1 && diversityLines.length > 0}
								<svg
									viewBox="0 0 200 120"
									class="info-chart-svg"
									role="img"
									aria-label="Historical measurement trend"
									preserveAspectRatio="none"
									onmouseleave={() => {
										diversityHoveredIdx = -1;
									}}
								>
									{#each diversityLines as dl (dl.idx)}
										{#if dl.path}
											<polyline
												points={dl.path.replace(/[ML]/g, (m) => (m === 'M' ? '' : ' ')).trim()}
												fill="none"
												stroke="transparent"
												role="presentation"
												stroke-width="12"
												onmouseenter={() => {
													diversityHoveredIdx = dl.idx;
												}}
											/>
											<polyline
												points={dl.path.replace(/[ML]/g, (m) => (m === 'M' ? '' : ' ')).trim()}
												fill="none"
												stroke={dl.color}
												stroke-width={diversityHoveredIdx === dl.idx ? '4' : '2'}
												opacity={diversityHoveredIdx >= 0
													? diversityHoveredIdx === dl.idx
														? '1'
														: '0.15'
													: '0.8'}
												style="pointer-events:none"
											/>
										{/if}
									{/each}
								</svg>
							{:else}
								<div class="species-waiting">Collecting diversity data...</div>
							{/if}
							<div class="diversity-grid">
								{#each diversityLines as dl (dl.idx)}
									<div
										class="diversity-card"
										role="group"
										class:dimmed={diversityHoveredIdx >= 0 && diversityHoveredIdx !== dl.idx}
										class:highlighted={diversityHoveredIdx === dl.idx}
										onmouseenter={() => {
											diversityHoveredIdx = dl.idx;
										}}
										onmouseleave={() => {
											diversityHoveredIdx = -1;
										}}
									>
										<span class="diversity-swatch" style="background:{dl.color}"></span>
										<span class="diversity-label">{dl.label}</span>
										<span class="diversity-value"
											>{dl.current < 100
												? dl.current.toFixed(dl.current < 10 ? 2 : 1)
												: formatNumber(dl.current)}</span
										>
									</div>
								{/each}
							</div>
							{#if suppressPatterns.length > 0}
								<div class="freq-suppressed-row">
									{#each suppressPatterns as pat (pat)}
										<button
											class="freq-suppress-chip"
											onclick={() => removeSuppressPattern(pat)}
											title="Remove pattern: {pat}"
										>
											{pat}
											<svg
												width="7"
												height="7"
												viewBox="0 0 8 8"
												stroke="currentColor"
												stroke-width="1.5"
											>
												<path d="M1.5 1.5l5 5M6.5 1.5l-5 5" stroke-linecap="round" />
											</svg>
										</button>
									{/each}
									<span class="freq-suppress-count" title={describeSuppression(suppression)}
										>⊘ {suppressedTotal}</span
									>
								</div>
							{/if}
						</div>
					{/if}
				</div>

				{#if tileTooltip}<div class="chart-detail">
						<strong
							>{hexByte(tileTooltip.byte)} · {tileTooltip.mnemonic} · rank {tileTooltip.rank}</strong
						>
						<p>
							{isa === 'z80'
								? opcodeDescription(tileTooltip.byte)
								: 'Inherited byte value in the selected processor.'}
						</p>
						<span>Click the tile to toggle instruction suppression.</span>
					</div>{/if}
			</section>
			{#if inspectedTape || speciesModal}
				{@const tape = inspectedTape ?? speciesModal!.exemplar}
				<section
					class="sidebar-section inspector-panel"
					hidden={activeTab !== 'measure' || measureSection !== 'bench' || sidebarCollapsed}
					aria-label="Tape inspector"
				>
					<div class="inspector-return">
						<button
							class="quiet-button"
							aria-label="Close tape inspector"
							title={speciesModal ? 'Return to population charts' : 'Return to live probe'}
							onclick={() => {
								showMeasure(speciesModal ? 'charts' : 'probe');
							}}>‹</button
						>
						<h2>Encounter bench</h2>
						<span
							>{speciesModal
								? `${(speciesModal.fraction * 100).toFixed(1)}% of population`
								: `Cell ${inspectedCell}`}</span
						>
					</div>
					{#if tape}
						<p class="control-note">
							{isa.toUpperCase()} · {tape.length} bytes. The lattice continues while you test a sample.
						</p>
						<div class="tip-grid inline-tape-grid">
							{#each buildGenomeGrid(tape, disassemble(tape)) as cell, index (index)}<div
									class="tip-cell"
									class:operand={!cell.isOpcode}
									style={`background:${byteColor(cell.byteVal)};color:${cellTextColor(cell.byteVal)}`}
									title={`Byte ${index} · ${hexByte(cell.byteVal)} · ${cell.label}`}
								>
									{cell.label}
								</div>{/each}
						</div>
						{@render disassemblyPanel(tape)}{@render culturePanel(tape)}{/if}
				</section>
			{/if}
			<section
				class="sidebar-section experiments-panel"
				hidden={activeTab !== 'experiments' || sidebarCollapsed}
				aria-label="Experiment presets"
			>
				<div class="section-heading">
					<h1>Experiments</h1>
					<span class="section-context">Begin with a question</span>
				</div>
				<p class="intro-copy">
					Choose a condition, read what to look for, then start when you are ready.
				</p>
				<InstrumentSelect
					id="experiment-preset"
					label="Experiments"
					placeholder="Choose an experiment…"
					value={selectedExperimentId}
					options={EXPERIMENT_PRESETS.map((preset) => ({
						value: preset.id,
						label: preset.name,
						group: preset.category
					}))}
					onchange={(value) => {
						const preset = EXPERIMENT_PRESETS.find((entry) => entry.id === value);
						if (preset) applyExperimentPreset(preset);
					}}
				/>
				{#if selectedExperiment}
					<div class="experiment-condition">
						<p>{scientificText(selectedExperiment.description)}</p>
						<p><strong>Look for</strong> {scientificText(selectedExperiment.whatToLookFor)}</p>
						<details>
							<summary>Finding, limits &amp; sources</summary>
							<p>{selectedExperiment.finding}</p>
							<p>{scientificText(selectedExperiment.limits)}</p>
							<p>
								<strong>Registered condition</strong>
								{selectedExperiment.published.width ?? 'isolated'} × {selectedExperiment.published
									.height ?? 'culture'}, {selectedExperiment.published.pairAttempts ?? 'assay'} pair attempts;
								horizon
								{selectedExperiment.published.horizon.toLocaleString()}
								{selectedExperiment.published.horizonUnit}. {scientificText(
									selectedExperiment.published.note
								)}
							</p>
							<p>{selectedExperiment.published.reset}</p>
							<p class="source-path">{selectedExperiment.source.registration}</p>
							{#if selectedExperiment.source.world}<p class="source-path">
									{selectedExperiment.source.world}
								</p>{/if}
							{#each selectedExperiment.source.results as source (source)}<p class="source-path">
									{source}
								</p>{/each}
							<p>
								Sources identify the registered experiment and its archived results. This GPU run is
								a fresh realization; equal seeds do not promise the archived scheduling trajectory.
							</p>
						</details>
					</div>
				{/if}

				{#if selectedExperiment || stagedPreset}<div class="apply-world">
						{@render nextWorldSummary()}
						<button
							class="primary-action"
							disabled={!engine || machineBusy || checkpointBusy}
							onclick={() => void startNewWorld()}>Start new world with these settings</button
						><button class="quiet-button" onclick={() => jumpWorld()}>Review world settings</button>
					</div>{/if}
				<div class="section-divider"></div>
				<div class="section-title">
					<h2>Your presets</h2>
					<button
						class="quiet-button"
						title="Save current settings as a preset"
						onclick={() => (showPresetSave = !showPresetSave)}>Save current</button
					>
				</div>
				{#if showPresetSave}<div class="preset-save-row">
						<input
							class="shell-input"
							aria-label="Preset name"
							placeholder="Preset name…"
							bind:value={newPresetName}
							onkeydown={(event) => {
								if (event.key === 'Enter') requestSavePreset(newPresetName);
							}}
						/><button
							class="quiet-button"
							disabled={!newPresetName.trim()}
							onclick={() => requestSavePreset(newPresetName)}>Save</button
						>
					</div>{/if}
				<div class="saved-presets">
					{#each presets as preset (preset.id)}<button
							class:active={stagedPreset?.id === preset.id}
							title={`Stage preset: ${preset.name}`}
							onclick={() => applyPreset(preset)}
							>{preset.name}<span>{preset.builtin ? 'Built in' : 'Saved'}</span></button
						>{/each}
				</div>
				{#if activePresetId}<button
						class="quiet-button"
						title="Delete selected preset"
						onclick={deleteActivePreset}>Delete current saved preset</button
					>{/if}
				{#if overwriteTarget}<div class="inline-confirmation">
						<p>Replace “{overwriteTarget.name}” with your current settings?</p>
						<button class="quiet-button" onclick={() => (overwriteTarget = null)}>Cancel</button
						><button class="quiet-button" onclick={confirmOverwrite}>Overwrite</button>
					</div>{/if}
			</section>
			<section
				class="sidebar-section guide-panel"
				hidden={(activeTab !== 'guide' && activeTab !== 'about') || sidebarCollapsed}
				aria-label={activeTab === 'about' ? 'About Algocell' : 'Guide'}
			>
				<div class="section-heading">
					<h1>{activeTab === 'about' ? 'About Algocell' : 'Guide'}</h1>
					<span class="section-context"
						>{activeTab === 'about' ? 'A world of small programs' : 'Learn by looking'}</span
					>
				</div>
				{#if activeTab === 'guide'}<InstrumentSelect
						label="Guide section"
						value={helpTab}
						options={[
							{ value: 'guide', label: 'Getting started' },
							{ value: 'overview', label: 'How the soup works' },
							{ value: 'visuals', label: 'Visual explanations' },
							{ value: 'z80', label: 'The Z80 machine' },
							{ value: 'params', label: 'Parameters' },
							{ value: 'analysis', label: 'Interpreting measurements' },
							{ value: 'keys', label: 'Keyboard shortcuts' }
						]}
						onchange={(value) => (helpTab = value as typeof helpTab)}
					/>{/if}
				<div class="modal-body documentation-content">
					{#if helpTab === 'guide'}
						{#each guideSections as section (section.id)}<h4>{section.title}</h4>
							{#each section.paragraphs as paragraph (paragraph)}<p>
									{scientificText(paragraph)}
								</p>{/each}{#if section.steps}<ol class="guide-steps">
									{#each section.steps as step (step)}<li>{scientificText(step)}</li>{/each}
								</ol>{/if}{/each}
					{:else if helpTab === 'overview'}
						<p class="cmap-note">
							Based on <a
								href="https://arxiv.org/abs/2406.19108"
								target="_blank"
								rel="noopener"
								class="help-link">Agüera y Arcas et al. (2024)</a
							>. Re-implemented in WebGPU + SvelteKit from the
							<a href="https://github.com/znah/zff" target="_blank" rel="noopener" class="help-link"
								>original JavaScript/WASM implementation</a
							>.
						</p>

						<h4>Cells &amp; Tapes</h4>
						<p>
							A grid of roughly 20,000 cells, fitted to the viewport, begins with random bytes. Each
							cell holds a short <strong>tape</strong> interpreted by the selected processor. The original
							layouts below show the default 16-byte square and 19-byte hex tapes; experiment conditions
							can use other tape and ring lengths:
						</p>

						<!-- Tape layout visuals -->
						<div class="tape-layouts">
							<div class="tape-layout-item">
								<div class="tape-layout-label">Square &mdash; 16 bytes (4&times;4)</div>
								<svg viewBox="0 0 160 160" xmlns="http://www.w3.org/2000/svg" class="tape-svg">
									{#each Array(16).keys() as i (i)}
										{@const col = i % 4}
										{@const row = Math.floor(i / 4)}
										<rect
											x={col * 40}
											y={row * 40}
											width="38"
											height="38"
											rx="3"
											fill="hsl({i * 22}, 45%, 32%)"
											stroke="rgba(255,255,255,0.12)"
											stroke-width="1"
										/>
										<text
											x={col * 40 + 19}
											y={row * 40 + 23}
											text-anchor="middle"
											fill="rgba(255,255,255,0.7)"
											font-size="11"
											font-family="monospace">{i.toString(16).toUpperCase().padStart(2, '0')}</text
										>
									{/each}
								</svg>
								<div class="tape-layout-note">4 neighbors (up, down, left, right)</div>
							</div>
							<div class="tape-layout-item">
								<div class="tape-layout-label">Hexagonal &mdash; 19 bytes (3-4-5-4-3)</div>
								<svg viewBox="0 -2 160 148" xmlns="http://www.w3.org/2000/svg" class="tape-svg">
									{#each [{ x: 49, y: 16, i: 0 }, { x: 80, y: 16, i: 1 }, { x: 111, y: 16, i: 2 }, { x: 33, y: 44, i: 3 }, { x: 64, y: 44, i: 4 }, { x: 95, y: 44, i: 5 }, { x: 126, y: 44, i: 6 }, { x: 18, y: 72, i: 7 }, { x: 49, y: 72, i: 8 }, { x: 80, y: 72, i: 9 }, { x: 111, y: 72, i: 10 }, { x: 142, y: 72, i: 11 }, { x: 33, y: 100, i: 12 }, { x: 64, y: 100, i: 13 }, { x: 95, y: 100, i: 14 }, { x: 126, y: 100, i: 15 }, { x: 49, y: 128, i: 16 }, { x: 80, y: 128, i: 17 }, { x: 111, y: 128, i: 18 }] as h (h.i)}
										<polygon
											points={Array.from({ length: 6 }, (_, k) => {
												const a = (Math.PI / 180) * (60 * k - 30);
												return `${h.x + 15 * Math.cos(a)},${h.y + 15 * Math.sin(a)}`;
											}).join(' ')}
											fill="hsl({h.i * 19}, 45%, 32%)"
											stroke="rgba(255,255,255,0.12)"
											stroke-width="1"
										/>
										<text
											x={h.x}
											y={h.y + 4}
											text-anchor="middle"
											fill="rgba(255,255,255,0.7)"
											font-size="9"
											font-family="monospace"
											>{h.i.toString(16).toUpperCase().padStart(2, '0')}</text
										>
									{/each}
								</svg>
								<div class="tape-layout-note">6 neighbors per cell</div>
							</div>
						</div>

						<h4>What Happens Each Step</h4>
						<p>
							Every simulation step, the following cycle runs thousands of times in parallel on the
							GPU:
						</p>
						<ol class="help-steps">
							<li><strong>Pick a pair.</strong> Choose a random cell and one of its neighbors.</li>
							<li>
								<strong>Combine their tapes.</strong> Both cells' bytes are copied into a single contiguous
								buffer &mdash; Cell&nbsp;A's tape first, then Cell&nbsp;B's. The buffer wraps around,
								so in the default 32-byte ring, address 31 is followed by address 0. Extra ring padding
								starts at zero and is discarded on writeback.
							</li>
						</ol>

						<!-- Full-cycle tape combination visual -->
						{@const subs = [
							'₀',
							'₁',
							'₂',
							'₃',
							'₄',
							'₅',
							'₆',
							'₇',
							'₈',
							'₉',
							'₁₀',
							'₁₁',
							'₁₂',
							'₁₃',
							'₁₄',
							'₁₅'
						]}
						{@const sz = 18}
						{@const gap = 2}
						{@const step = sz + gap}
						{@const gridW = 4 * step - gap}
						{@const axL = 160 - gridW - 20}
						{@const bxL = 160 + 20}
						{@const bufY = 140}
						{@const bufCellW = 9}
						{@const bufGap = 0.5}
						{@const bufTotalW = 32 * (bufCellW + bufGap) - bufGap}
						{@const bufXStart = (320 - bufTotalW) / 2}
						{@const z80Y = bufY + 34}
						{@const wbArrowY = z80Y + 22}
						{@const wbCellY = wbArrowY + 32}
						<svg viewBox="0 0 320 325" xmlns="http://www.w3.org/2000/svg" class="tape-combine-svg">
							<!-- ── Row 1: Source cells ── -->
							<text
								x={axL + gridW / 2}
								y="10"
								text-anchor="middle"
								fill="#c8a060"
								font-size="9.5"
								font-family="-apple-system, sans-serif"
								font-weight="600">Cell A</text
							>
							{#each Array(16).keys() as i (i)}
								{@const col = i % 4}
								{@const row = Math.floor(i / 4)}
								<rect
									x={axL + col * step}
									y={14 + row * step}
									width={sz}
									height={sz}
									rx="2"
									fill="hsl({i * 22}, 45%, 32%)"
									stroke="rgba(255,255,255,0.12)"
									stroke-width="0.5"
								/>
								<text
									x={axL + col * step + sz / 2}
									y={14 + row * step + sz / 2 + 3}
									text-anchor="middle"
									fill="rgba(255,255,255,0.85)"
									font-size="7"
									font-family="monospace">a{subs[i]}</text
								>
							{/each}

							<text
								x="160"
								y="58"
								text-anchor="middle"
								fill="rgba(255,255,255,0.35)"
								font-size="16"
								font-family="monospace"
								font-weight="300">+</text
							>

							<text
								x={bxL + gridW / 2}
								y="10"
								text-anchor="middle"
								fill="#60a0c8"
								font-size="9.5"
								font-family="-apple-system, sans-serif"
								font-weight="600">Cell B</text
							>
							{#each Array(16).keys() as i (i)}
								{@const col = i % 4}
								{@const row = Math.floor(i / 4)}
								<rect
									x={bxL + col * step}
									y={14 + row * step}
									width={sz}
									height={sz}
									rx="2"
									fill="hsl({i * 19 + 200}, 40%, 32%)"
									stroke="rgba(255,255,255,0.12)"
									stroke-width="0.5"
								/>
								<text
									x={bxL + col * step + sz / 2}
									y={14 + row * step + sz / 2 + 3}
									text-anchor="middle"
									fill="rgba(255,255,255,0.85)"
									font-size="7"
									font-family="monospace">b{subs[i]}</text
								>
							{/each}

							<!-- ── Row 2: Arrow down ── -->
							<line
								x1="160"
								y1="100"
								x2="160"
								y2="116"
								stroke="rgba(255,255,255,0.3)"
								stroke-width="1.2"
							/>
							<polygon points="160,122 155,114 165,114" fill="rgba(255,255,255,0.3)" />
							<text
								x="160"
								y="134"
								text-anchor="middle"
								fill="rgba(255,255,255,0.45)"
								font-size="8"
								font-family="-apple-system, sans-serif">Shared buffer (wrapping)</text
							>

							<!-- ── Row 3: Combined buffer ── -->
							{#each Array(32).keys() as i (i)}
								{@const isA = i < 16}
								{@const bx = bufXStart + i * (bufCellW + bufGap)}
								<rect
									x={bx}
									y={bufY}
									width={bufCellW}
									height="14"
									rx="1"
									fill={isA ? `hsl(${i * 22}, 45%, 32%)` : `hsl(${(i - 16) * 19 + 200}, 40%, 32%)`}
									stroke="rgba(255,255,255,0.08)"
									stroke-width="0.5"
								/>
								{#if i % 4 === 0}
									<text
										x={bx + bufCellW / 2}
										y={bufY + 10}
										text-anchor="middle"
										fill="rgba(255,255,255,0.7)"
										font-size="5"
										font-family="monospace">{isA ? 'a' : 'b'}{subs[isA ? i : i - 16]}</text
									>
								{/if}
							{/each}
							<!-- Labels under buffer -->
							<text
								x={bufXStart + (16 * (bufCellW + bufGap)) / 2 - bufGap / 2}
								y={bufY + 24}
								text-anchor="middle"
								fill="#c8a060"
								font-size="7"
								font-family="-apple-system, sans-serif">a₀ – a₁₅</text
							>
							<text
								x={bufXStart +
									16 * (bufCellW + bufGap) +
									(16 * (bufCellW + bufGap)) / 2 -
									bufGap / 2}
								y={bufY + 24}
								text-anchor="middle"
								fill="#60a0c8"
								font-size="7"
								font-family="-apple-system, sans-serif">b₀ – b₁₅</text
							>
							<!-- Wrap indicator -->
							<path
								d="M {bufXStart + bufTotalW + 2},{bufY + 7} C {bufXStart + bufTotalW + 8},{bufY +
									7} {bufXStart + bufTotalW + 8},{bufY - 4} {bufXStart + bufTotalW + 2},{bufY -
									4} L {bufXStart - 2},{bufY - 4} C {bufXStart - 8},{bufY - 4} {bufXStart -
									8},{bufY + 7} {bufXStart - 2},{bufY + 7}"
								fill="none"
								stroke="rgba(255,255,255,0.12)"
								stroke-width="0.7"
								stroke-dasharray="2,2"
							/>
							<text
								x={bufXStart + bufTotalW + 6}
								y={bufY + 2}
								fill="rgba(255,255,255,0.2)"
								font-size="6"
								font-family="-apple-system, sans-serif">↻</text
							>

							<!-- ── Z80 execution arrow ── -->
							<line
								x1={bufXStart}
								y1={z80Y}
								x2={bufXStart + bufTotalW - 6}
								y2={z80Y}
								stroke="var(--accent)"
								stroke-width="1"
								opacity="0.5"
							/>
							<polygon
								points="{bufXStart + bufTotalW},{z80Y} {bufXStart + bufTotalW - 6},{z80Y -
									3} {bufXStart + bufTotalW - 6},{z80Y + 3}"
								fill="var(--accent)"
								opacity="0.5"
							/>
							<text
								x="160"
								y={z80Y + 12}
								text-anchor="middle"
								fill="var(--accent)"
								font-size="7.5"
								opacity="0.7"
								font-family="-apple-system, sans-serif">⚡ Z80 executes (PC=0, regs=0)</text
							>

							<!-- ── Row 4: Write-back ── -->
							<!-- Split arrows -->
							<line
								x1="160"
								y1={wbArrowY}
								x2="160"
								y2={wbArrowY + 12}
								stroke="rgba(255,255,255,0.25)"
								stroke-width="1"
							/>
							<line
								x1="160"
								y1={wbArrowY + 12}
								x2={axL + gridW / 2}
								y2={wbArrowY + 24}
								stroke="rgba(255,255,255,0.25)"
								stroke-width="1"
							/>
							<line
								x1="160"
								y1={wbArrowY + 12}
								x2={bxL + gridW / 2}
								y2={wbArrowY + 24}
								stroke="rgba(255,255,255,0.25)"
								stroke-width="1"
							/>
							<polygon
								points="{axL + gridW / 2},{wbArrowY + 28} {axL + gridW / 2 - 4},{wbArrowY +
									21} {axL + gridW / 2 + 4},{wbArrowY + 21}"
								fill="rgba(255,255,255,0.25)"
							/>
							<polygon
								points="{bxL + gridW / 2},{wbArrowY + 28} {bxL + gridW / 2 - 4},{wbArrowY +
									21} {bxL + gridW / 2 + 4},{wbArrowY + 21}"
								fill="rgba(255,255,255,0.25)"
							/>

							<!-- Write-back label -->
							<text
								x="160"
								y={wbArrowY + 8}
								text-anchor="middle"
								fill="rgba(255,255,255,0.35)"
								font-size="7"
								font-family="-apple-system, sans-serif">write back</text
							>

							<!-- Modified Cell A -->
							<text
								x={axL + gridW / 2}
								y={wbCellY}
								text-anchor="middle"
								fill="#c8a060"
								font-size="8"
								font-family="-apple-system, sans-serif"
								opacity="0.7">Cell A′</text
							>
							{#each Array(16).keys() as i (i)}
								{@const col = i % 4}
								{@const row = Math.floor(i / 4)}
								{@const swapped = i === 2 || i === 7 || i === 11}
								<rect
									x={axL + col * step}
									y={wbCellY + 4 + row * step}
									width={sz}
									height={sz}
									rx="2"
									fill={swapped ? `hsl(${i * 19 + 200}, 40%, 32%)` : `hsl(${i * 22}, 45%, 32%)`}
									stroke={swapped ? 'rgba(96,160,200,0.5)' : 'rgba(255,255,255,0.12)'}
									stroke-width={swapped ? 1 : 0.5}
								/>
								<text
									x={axL + col * step + sz / 2}
									y={wbCellY + 4 + row * step + sz / 2 + 3}
									text-anchor="middle"
									fill="rgba(255,255,255,0.85)"
									font-size="7"
									font-family="monospace">{swapped ? `b${subs[i]}` : `a${subs[i]}`}</text
								>
							{/each}

							<!-- Modified Cell B -->
							<text
								x={bxL + gridW / 2}
								y={wbCellY}
								text-anchor="middle"
								fill="#60a0c8"
								font-size="8"
								font-family="-apple-system, sans-serif"
								opacity="0.7">Cell B′</text
							>
							{#each Array(16).keys() as i (i)}
								{@const col = i % 4}
								{@const row = Math.floor(i / 4)}
								{@const swapped = i === 1 || i === 5 || i === 13}
								<rect
									x={bxL + col * step}
									y={wbCellY + 4 + row * step}
									width={sz}
									height={sz}
									rx="2"
									fill={swapped ? `hsl(${i * 22}, 45%, 32%)` : `hsl(${i * 19 + 200}, 40%, 32%)`}
									stroke={swapped ? 'rgba(200,160,96,0.5)' : 'rgba(255,255,255,0.12)'}
									stroke-width={swapped ? 1 : 0.5}
								/>
								<text
									x={bxL + col * step + sz / 2}
									y={wbCellY + 4 + row * step + sz / 2 + 3}
									text-anchor="middle"
									fill="rgba(255,255,255,0.85)"
									font-size="7"
									font-family="monospace">{swapped ? `a${subs[i]}` : `b${subs[i]}`}</text
								>
							{/each}

							<!-- Mutate annotation -->
							<text
								x="160"
								y={wbCellY + 4 * step + 12}
								text-anchor="middle"
								fill="rgba(255,255,255,0.3)"
								font-size="7.5"
								font-family="-apple-system, sans-serif">~ random mutations applied ~</text
							>
						</svg>

						<ol class="help-steps" start="3">
							<li>
								<strong>Run the processor.</strong> The chosen machine starts at byte 0 with its declared
								reset. The original Z80 starts with zero registers except SP = FFFF. Instructions can
								read and write the shared ring, including the partner’s bytes. The diagrams illustrate
								the default Z80 world.
							</li>
							<li>
								<strong>Write back.</strong> The modified buffer is split and written back to both cells
								in the grid. Note some bytes have been cross-contaminated (highlighted).
							</li>
							<li>
								<strong>Mutate.</strong> Random byte-replacement attempts are applied across the grid.
							</li>
						</ol>
						<p>
							After all pairs are processed, the next update begins. Copiers can emerge and spread
							in this world, but a short run need not find one. Culture a candidate before
							interpreting visible order as reproduction.
						</p>

						<h4>Order, copying and evidence</h4>
						<p>
							In the default benign Z80 reset, A, H and L begin at zero. Instructions such as
							<code>LD (HL),A</code> (store&nbsp;A at address&nbsp;HL) write zeros &mdash; since A, H,
							and L are all zero. This makes NOP (0x00) accumulate rapidly. Random code acts as a &ldquo;zero
							pump,&rdquo; flooding the grid with NOPs.
						</p>
						<p>
							Some compact load–push or exchange/stack loops can copy material into a neighbor.
							Whether a candidate keeps reproducing, enters its partner during instruction fetching,
							or preserves an introduced difference are separate measurements. The schematic below
							illustrates possible composition changes; it is not a guaranteed phase sequence.
						</p>
						<svg
							viewBox="0 0 320 180"
							xmlns="http://www.w3.org/2000/svg"
							style="width:100%;margin:8px 0;display:block;border-radius:8px;background:rgba(0,0,0,0.2);border:1px solid rgba(255,255,255,0.04);padding:12px 8px;"
						>
							<!-- axes -->
							<line
								x1="40"
								y1="20"
								x2="40"
								y2="140"
								stroke="rgba(255,255,255,0.15)"
								stroke-width="1"
							/>
							<line
								x1="40"
								y1="140"
								x2="300"
								y2="140"
								stroke="rgba(255,255,255,0.15)"
								stroke-width="1"
							/>
							<!-- axis labels -->
							<text
								x="170"
								y="168"
								text-anchor="middle"
								fill="#8a7a6a"
								font-size="9"
								font-family="-apple-system, BlinkMacSystemFont, sans-serif">Time</text
							>
							<text
								x="14"
								y="80"
								text-anchor="middle"
								fill="#8a7a6a"
								font-size="9"
								font-family="-apple-system, BlinkMacSystemFont, sans-serif"
								transform="rotate(-90,14,80)">Concentration</text
							>
							<!-- NOP curve (white/gray, starts low, rises to moderate peak, then declines) -->
							<path
								d="M 40,125 C 50,95 60,68 80,62 S 120,64 150,78 Q 200,108 300,122"
								fill="none"
								stroke="rgba(220,215,210,0.7)"
								stroke-width="2"
							/>
							<text
								x="68"
								y="54"
								fill="rgba(220,215,210,0.8)"
								font-size="10"
								font-family="-apple-system, BlinkMacSystemFont, sans-serif">NOP</text
							>
							<!-- Self-replicating curve (orange, stays low then rises smoothly) -->
							<path
								d="M 40,130 C 100,130 150,120 186,88 C 220,58 260,42 300,42"
								fill="none"
								stroke="var(--accent)"
								stroke-width="2"
							/>
							<text
								x="195"
								y="36"
								fill="var(--accent)"
								font-size="10"
								font-family="-apple-system, BlinkMacSystemFont, sans-serif"
								>Self-Replicating Bytes</text
							>
							<!-- Phase transition dashed line at crossing (x≈180) -->
							<line
								x1="180"
								y1="22"
								x2="180"
								y2="140"
								stroke="rgba(120,160,220,0.5)"
								stroke-width="1"
								stroke-dasharray="4,3"
							/>
							<!-- Phase transition label -->
							<text
								x="180"
								y="155"
								text-anchor="middle"
								fill="rgba(120,160,220,0.7)"
								font-size="9"
								font-family="-apple-system, BlinkMacSystemFont, sans-serif">Phase Transition</text
							>
						</svg>
					{:else if helpTab === 'visuals'}
						<p>
							Each cell is colored by its first byte. The colormap assigns distinct colors to Z80
							opcode categories. Other bytes get a muted hue from a continuous sweep.
						</p>

						<div class="cmap-section">
							<div class="cmap-label">Special</div>
							<div class="cmap-grid">
								<div class="cmap-entry">
									<span class="help-swatch" style="background:{byteColor(0x00)}"></span><span
										class="cmap-hex">00</span
									> NOP
								</div>
								<div class="cmap-entry">
									<span class="help-swatch" style="background:{byteColor(0x76)}"></span><span
										class="cmap-hex">76</span
									> HALT
								</div>
							</div>
						</div>

						<div class="cmap-section">
							<div class="cmap-label">16-bit loads</div>
							<div class="cmap-grid">
								<div class="cmap-entry">
									<span class="help-swatch" style="background:{byteColor(0x01)}"></span><span
										class="cmap-hex">01</span
									> LD BC,nn
								</div>
								<div class="cmap-entry">
									<span class="help-swatch" style="background:{byteColor(0x11)}"></span><span
										class="cmap-hex">11</span
									> LD DE,nn
								</div>
								<div class="cmap-entry">
									<span class="help-swatch" style="background:{byteColor(0x21)}"></span><span
										class="cmap-hex">21</span
									> LD HL,nn
								</div>
								<div class="cmap-entry">
									<span class="help-swatch" style="background:{byteColor(0x31)}"></span><span
										class="cmap-hex">31</span
									> LD SP,nn
								</div>
							</div>
						</div>

						<div class="cmap-section">
							<div class="cmap-label">Memory access</div>
							<div class="cmap-grid">
								<div class="cmap-entry">
									<span class="help-swatch" style="background:{byteColor(0x2a)}"></span><span
										class="cmap-hex">2A</span
									> LD HL,(nn)
								</div>
								<div class="cmap-entry">
									<span class="help-swatch" style="background:{byteColor(0x3a)}"></span><span
										class="cmap-hex">3A</span
									> LD A,(nn)
								</div>
								<div class="cmap-entry">
									<span class="help-swatch" style="background:{byteColor(0x22)}"></span><span
										class="cmap-hex">22</span
									> LD (nn),HL
								</div>
								<div class="cmap-entry">
									<span class="help-swatch" style="background:{byteColor(0x32)}"></span><span
										class="cmap-hex">32</span
									> LD (nn),A
								</div>
							</div>
						</div>

						<div class="cmap-section">
							<div class="cmap-label">Block transfer</div>
							<div class="cmap-grid">
								<div class="cmap-entry">
									<span class="help-swatch" style="background:{byteColor(0xed)}"></span><span
										class="cmap-hex">ED</span
									> ED prefix
								</div>
								<div class="cmap-entry">
									<span class="help-swatch" style="background:{byteColor(0xb0)}"></span><span
										class="cmap-hex">B0</span
									> LDIR
								</div>
								<div class="cmap-entry">
									<span class="help-swatch" style="background:{byteColor(0xb8)}"></span><span
										class="cmap-hex">B8</span
									> LDDR
								</div>
								<div class="cmap-entry">
									<span class="help-swatch" style="background:{byteColor(0xa0)}"></span><span
										class="cmap-hex">A0</span
									> LDI
								</div>
								<div class="cmap-entry">
									<span class="help-swatch" style="background:{byteColor(0xa8)}"></span><span
										class="cmap-hex">A8</span
									> LDD
								</div>
							</div>
						</div>

						<div class="cmap-section">
							<div class="cmap-label">Stack</div>
							<div class="cmap-grid">
								<div class="cmap-entry">
									<span class="help-swatch" style="background:{byteColor(0xc5)}"></span><span
										class="cmap-hex">C5</span
									> PUSH BC
								</div>
								<div class="cmap-entry">
									<span class="help-swatch" style="background:{byteColor(0xd5)}"></span><span
										class="cmap-hex">D5</span
									> PUSH DE
								</div>
								<div class="cmap-entry">
									<span class="help-swatch" style="background:{byteColor(0xe5)}"></span><span
										class="cmap-hex">E5</span
									> PUSH HL
								</div>
								<div class="cmap-entry">
									<span class="help-swatch" style="background:{byteColor(0xf5)}"></span><span
										class="cmap-hex">F5</span
									> PUSH AF
								</div>
								<div class="cmap-entry">
									<span class="help-swatch" style="background:{byteColor(0xc1)}"></span><span
										class="cmap-hex">C1</span
									> POP BC
								</div>
								<div class="cmap-entry">
									<span class="help-swatch" style="background:{byteColor(0xd1)}"></span><span
										class="cmap-hex">D1</span
									> POP DE
								</div>
								<div class="cmap-entry">
									<span class="help-swatch" style="background:{byteColor(0xe1)}"></span><span
										class="cmap-hex">E1</span
									> POP HL
								</div>
								<div class="cmap-entry">
									<span class="help-swatch" style="background:{byteColor(0xf1)}"></span><span
										class="cmap-hex">F1</span
									> POP AF
								</div>
							</div>
						</div>

						<div class="cmap-section">
							<div class="cmap-label">Flow control</div>
							<div class="cmap-grid">
								<div class="cmap-entry">
									<span class="help-swatch" style="background:{byteColor(0xc3)}"></span><span
										class="cmap-hex">C3</span
									> JP nn
								</div>
								<div class="cmap-entry">
									<span class="help-swatch" style="background:{byteColor(0xcd)}"></span><span
										class="cmap-hex">CD</span
									> CALL nn
								</div>
								<div class="cmap-entry">
									<span class="help-swatch" style="background:{byteColor(0xc9)}"></span><span
										class="cmap-hex">C9</span
									> RET
								</div>
								<div class="cmap-entry">
									<span class="help-swatch" style="background:{byteColor(0x18)}"></span><span
										class="cmap-hex">18</span
									> JR
								</div>
								<div class="cmap-entry">
									<span class="help-swatch" style="background:{byteColor(0x10)}"></span><span
										class="cmap-hex">10</span
									> DJNZ
								</div>
							</div>
						</div>

						<div class="cmap-section">
							<div class="cmap-label">Prefixes &amp; exchange</div>
							<div class="cmap-grid">
								<div class="cmap-entry">
									<span class="help-swatch" style="background:{byteColor(0xcb)}"></span><span
										class="cmap-hex">CB</span
									> Bit ops
								</div>
								<div class="cmap-entry">
									<span class="help-swatch" style="background:{byteColor(0xdd)}"></span><span
										class="cmap-hex">DD</span
									> IX ops
								</div>
								<div class="cmap-entry">
									<span class="help-swatch" style="background:{byteColor(0xfd)}"></span><span
										class="cmap-hex">FD</span
									> IY ops
								</div>
								<div class="cmap-entry">
									<span class="help-swatch" style="background:{byteColor(0xe3)}"></span><span
										class="cmap-hex">E3</span
									> EX (SP),HL
								</div>
								<div class="cmap-entry">
									<span class="help-swatch" style="background:{byteColor(0xeb)}"></span><span
										class="cmap-hex">EB</span
									> EX DE,HL
								</div>
								<div class="cmap-entry">
									<span class="help-swatch" style="background:{byteColor(0x08)}"></span><span
										class="cmap-hex">08</span
									> EX AF,AF'
								</div>
							</div>
						</div>

						<div class="cmap-section">
							<div class="cmap-label">Ranges</div>
							<div class="cmap-grid wide">
								<div class="cmap-entry">
									<span class="help-swatch" style="background:{byteColor(0x60)}"></span><span
										class="cmap-hex">40-7F</span
									> Register LD
								</div>
								<div class="cmap-entry">
									<span class="help-swatch" style="background:{byteColor(0xa0)}"></span><span
										class="cmap-hex">80-BF</span
									> ALU ops
								</div>
							</div>
						</div>

						<p class="cmap-note">
							All other bytes get a muted tone from a continuous hue sweep. Switch colormaps in
							World → Appearance.
						</p>

						<h4>Cell Tooltips</h4>
						<p>
							Hover any cell to see its bytes disassembled as Z80 instructions. Square mode shows a
							4&times;4 grid of 16 bytes; hex mode shows a 3-4-5-4-3 hexagonal cluster of 19 bytes
							matching the cell's honeycomb shape.
						</p>
						<ul class="help-list">
							<li>
								<strong>Opcode bytes</strong> show the instruction mnemonic (e.g. <code>NOP</code>,
								<code>POP HL</code>, <code>LD B,C</code>)
							</li>
							<li>
								<strong>Operand bytes</strong> show the raw hex value (e.g. <code>00</code>,
								<code>3F</code>) and appear dimmed &mdash; these are data consumed by the preceding
								instruction
							</li>
						</ul>
						<p>
							For example, byte 0x00 appears as <code>NOP</code> when executed as an instruction,
							but as <code>00</code> when it is an operand of a multi-byte instruction like
							<code>LD BC,nn</code>.
						</p>

						<h4>Frequency Chart</h4>
						<p>
							The bottom-left widget shows the most common byte values in real time. When
							suppression is active, suppressed bytes are collapsed into a single card so you always
							see the top non-suppressed opcodes. Hover any tile for a plain-English description;
							click to suppress or un-suppress it. Toggle the chart sparkline with <kbd>S</kbd>.
						</p>

						<h4>Instruction Suppression (ablation)</h4>
						{#if isa === 'z80'}
							<p>
								Type a pattern in World → Budgets &amp; instruction ablation and press Enter to
								remove the matching instructions from the CPU. The Z80 then executes a suppressed
								opcode as a NOP. Every instruction on the base, <code>CB</code> and <code>ED</code>
								pages is addressable, so block copies like <code>LDIR</code> can be removed on their own.
							</p>
							<ul class="help-list">
								<li>
									<strong>Families</strong> &mdash; <code>family:block-copy</code>
									(LDI/LDD/LDIR/LDDR),
									<code>family:stack</code> (PUSH/POP), <code>family:ld8-mem</code>,
									<code>family:call-ret</code>, &hellip; and the pseudo-family
									<code>family:writes-mem</code> (every instruction that writes memory)
								</li>
								<li>
									<strong>Exact opcodes</strong> &mdash; <code>ed:B0</code> (LDIR),
									<code>cb:FF</code>
									(SET 7,A), <code>3E</code> or <code>base:3E</code> (LD A,n)
								</li>
								<li>
									<strong>Mnemonic substrings</strong> &mdash; <code>LDIR</code>,
									<code>LD (HL)</code>,
									<code>PUSH</code>; <code>LD</code> matches every load on every page
								</li>
								<li>
									<strong>Click-to-suppress</strong> &mdash; click any tile in the frequency widget to
									add its exact mnemonic as a pattern
								</li>
								<li>
									<strong>Remove patterns</strong> &mdash; click the &times; on any chip to remove it,
									or use the clear button to remove all
								</li>
							</ul>
							<p>
								Semantics: a suppressed base-page opcode is a 1-byte NOP, so its operand bytes run
								as the next instruction (as if the opcode had been removed from the ISA). Suppressed
								<code>CB</code>/<code>ED</code> instructions are 2-byte NOPs. IX/IY forms follow
								their base opcode. Suppression prevents <em>execution</em>, not <em>existence</em>:
								a suppressed byte value can still be written into memory or carried as data.
							</p>
						{:else}
							<p>
								Type a pattern in World → Budgets &amp; instruction ablation and press Enter. {isaSuppressionDescription}
							</p>
							<p>
								Click an opcode frequency tile to add its exact opcode as a pattern. Remove a chip
								to restore its operations, or use the clear button to restore all operations. Static
								disassembly still shows the ordinary instruction decoding; measured fetches show the
								execution path under suppression.
							</p>
						{/if}
					{:else if helpTab === 'z80'}
						<h4>Registers</h4>
						<p>The Z80 has 8-bit and 16-bit registers used as operands:</p>
						<div class="z80-reg-table">
							<div class="z80-reg-row head">
								<span>Name</span><span>Size</span><span>Role</span>
							</div>
							<div class="z80-reg-row">
								<span class="z80-reg">A</span><span>8-bit</span><span
									>Accumulator &mdash; main arithmetic register</span
								>
							</div>
							<div class="z80-reg-row">
								<span class="z80-reg">F</span><span>8-bit</span><span
									>Flags (zero, carry, sign, parity)</span
								>
							</div>
							<div class="z80-reg-row">
								<span class="z80-reg">B, C</span><span>8-bit</span><span
									>General purpose; B is loop counter for DJNZ</span
								>
							</div>
							<div class="z80-reg-row">
								<span class="z80-reg">D, E</span><span>8-bit</span><span
									>General purpose; DE = destination for block ops</span
								>
							</div>
							<div class="z80-reg-row">
								<span class="z80-reg">H, L</span><span>8-bit</span><span
									>General purpose; HL = primary memory pointer</span
								>
							</div>
							<div class="z80-reg-row">
								<span class="z80-reg">SP</span><span>16-bit</span><span>Stack pointer</span>
							</div>
							<div class="z80-reg-row">
								<span class="z80-reg">BC, DE, HL</span><span>16-bit</span><span
									>Register pairs (B+C, D+E, H+L)</span
								>
							</div>
							<div class="z80-reg-row">
								<span class="z80-reg">AF</span><span>16-bit</span><span
									>Accumulator + flags pair</span
								>
							</div>
						</div>

						<h4>Notation</h4>
						<div class="z80-notation-table">
							<div class="z80-notation-row head"><span>Syntax</span><span>Meaning</span></div>
							<div class="z80-notation-row">
								<span class="z80-reg">nn</span><span>16-bit immediate value (e.g. $1234)</span>
							</div>
							<div class="z80-notation-row">
								<span class="z80-reg">n</span><span>8-bit immediate value (e.g. $FF)</span>
							</div>
							<div class="z80-notation-row">
								<span class="z80-reg">d</span><span>Signed offset for relative jumps</span>
							</div>
							<div class="z80-notation-row">
								<span class="z80-reg">(HL)</span><span>Memory at address in HL</span>
							</div>
							<div class="z80-notation-row">
								<span class="z80-reg">(nn)</span><span>Memory at absolute address nn</span>
							</div>
							<div class="z80-notation-row">
								<span class="z80-reg">(SP)</span><span>Memory at top of stack</span>
							</div>
						</div>

						<h4>Instruction Reference</h4>

						<div class="z80-cat">
							<div class="z80-cat-label">Data Movement</div>
							<div class="z80-instr-grid">
								<div class="z80-instr"><span class="z80-op">LD x,y</span> Copy y into x</div>
								<div class="z80-instr">
									<span class="z80-op">PUSH rr</span> Push 16-bit pair onto stack
								</div>
								<div class="z80-instr">
									<span class="z80-op">POP rr</span> Pop 16 bits from stack into pair
								</div>
								<div class="z80-instr">
									<span class="z80-op">EX x,y</span> Exchange (swap) x and y
								</div>
							</div>
						</div>

						<div class="z80-cat">
							<div class="z80-cat-label">Arithmetic &amp; Logic</div>
							<div class="z80-instr-grid">
								<div class="z80-instr"><span class="z80-op">ADD A,x</span> A = A + x</div>
								<div class="z80-instr"><span class="z80-op">ADC A,x</span> A = A + x + carry</div>
								<div class="z80-instr"><span class="z80-op">SUB x</span> A = A &minus; x</div>
								<div class="z80-instr">
									<span class="z80-op">SBC A,x</span> A = A &minus; x &minus; carry
								</div>
								<div class="z80-instr"><span class="z80-op">AND x</span> A = A &amp; x</div>
								<div class="z80-instr"><span class="z80-op">OR x</span> A = A | x</div>
								<div class="z80-instr"><span class="z80-op">XOR x</span> A = A ^ x</div>
								<div class="z80-instr">
									<span class="z80-op">CP x</span> Compare A with x (sets flags)
								</div>
								<div class="z80-instr"><span class="z80-op">INC x</span> x = x + 1</div>
								<div class="z80-instr"><span class="z80-op">DEC x</span> x = x &minus; 1</div>
							</div>
						</div>

						<div class="z80-cat">
							<div class="z80-cat-label">Flow Control</div>
							<div class="z80-instr-grid">
								<div class="z80-instr"><span class="z80-op">JP nn</span> Jump to address</div>
								<div class="z80-instr">
									<span class="z80-op">JR d</span> Jump relative by offset d
								</div>
								<div class="z80-instr">
									<span class="z80-op">DJNZ d</span> Decrement B, jump if B &ne; 0
								</div>
								<div class="z80-instr">
									<span class="z80-op">CALL nn</span> Push PC, jump to address
								</div>
								<div class="z80-instr">
									<span class="z80-op">RET</span> Pop PC (return from call)
								</div>
								<div class="z80-instr">
									<span class="z80-op">NOP</span> No operation (do nothing)
								</div>
								<div class="z80-instr"><span class="z80-op">HALT</span> Stop execution</div>
							</div>
						</div>

						<div class="z80-cat">
							<div class="z80-cat-label">Block Operations</div>
							<div class="z80-instr-grid">
								<div class="z80-instr">
									<span class="z80-op">LDI</span> Copy (HL)&rarr;(DE), inc both, dec BC
								</div>
								<div class="z80-instr">
									<span class="z80-op">LDD</span> Copy (HL)&rarr;(DE), dec both, dec BC
								</div>
								<div class="z80-instr">
									<span class="z80-op">LDIR</span> LDI repeated until BC = 0
								</div>
								<div class="z80-instr">
									<span class="z80-op">LDDR</span> LDD repeated until BC = 0
								</div>
							</div>
						</div>

						<div class="z80-cat">
							<div class="z80-cat-label">Bit &amp; Rotate</div>
							<div class="z80-instr-grid">
								<div class="z80-instr"><span class="z80-op">BIT n,x</span> Test bit n of x</div>
								<div class="z80-instr"><span class="z80-op">SET n,x</span> Set bit n of x</div>
								<div class="z80-instr"><span class="z80-op">RES n,x</span> Reset bit n of x</div>
								<div class="z80-instr">
									<span class="z80-op">RL x</span> Rotate left through carry
								</div>
								<div class="z80-instr">
									<span class="z80-op">RR x</span> Rotate right through carry
								</div>
								<div class="z80-instr"><span class="z80-op">SLA x</span> Shift left arithmetic</div>
								<div class="z80-instr"><span class="z80-op">SRL x</span> Shift right logical</div>
							</div>
						</div>

						<h4>An exchange/stack copying route</h4>
						<p>
							<span class="z80-op">POP HL</span> reads two bytes and advances the stack pointer;
							<span class="z80-op">EX (SP),HL</span> exchanges that word with the next stack position.
							A working loop can move its machinery into the partner under a suitable ring and reset contract.
							This is one available route, not a claim that every occurrence copies or that it always
							dominates. Historical 16-byte first exemplars were usually load–push designs.
						</p>

						<Mermaid
							chart={`
graph TD
    A("POP HL\nread 2 bytes from memory") -->|"HL now holds data"| B("EX (SP),HL\nwrite 2 bytes forward")
    B -->|"repeat"| A
`}
						/>
					{:else if helpTab === 'params'}
						<h4>Grid Type</h4>
						<p>
							Switch between <strong>Square</strong> and <strong>Hex</strong> topologies. Square cells
							hold 16 bytes (4&times;4) with 4 neighbors. Hex cells hold 19 bytes (3-4-5-4-3 honeycomb)
							with 6 neighbors. Changing resets the simulation.
						</p>

						<h4>Grid Size (W &times; H)</h4>
						<p>
							Width and height of the grid in cells. The initial grid fits roughly 20,000 cells to
							the viewport. Larger worlds provide more encounters and use more GPU memory and
							compute. Edits stay staged until you choose Start new world with these settings.
						</p>

						<h4>Seed</h4>
						<p>
							The random seed used to initialize the grid. Same seed produces the same starting
							state. Edit it in Next world, then explicitly start the staged condition.
						</p>

						<h4>Mutation Rate</h4>
						<p>
							Mutation exponent k sets floor(pair attempts / 2ᵏ) random byte-replacement attempts
							per population update. With 8,192 pair slots and k = 4, that is 512 attempts.
							Replacement can repeat the old value; this is not a fraction of all population bytes.
							The off checkbox disables mutation.
						</p>

						<h4>Pairs / Batch</h4>
						<p>
							The number of proposed neighboring pair slots per update. Collision claims can reject
							a proposal, so the number of accepted encounters is lower. More pair attempts usually
							require more GPU time; an update is not a synchronized generation.
						</p>

						<h4>{isa === 'z80' ? 'Z80 Steps' : 'Execution steps'}</h4>
						<p>
							Maximum instruction slots per encounter, with default 128. These are execution steps
							rather than hardware clock cycles; Z80 block repeats and 8086 REP consume one slot per
							iteration. Larger budgets permit longer execution and usually reduce update
							throughput. A halted lane does not execute its unused slots.
						</p>

						<h4>Suppress Opcodes</h4>
						{#if isa === 'z80'}
							<p>
								Remove instructions from the CPU: a mnemonic substring (<code>LDIR</code>,
								<code>PUSH</code>), a family (<code>family:block-copy</code>,
								<code>family:writes-mem</code>) or an exact opcode (<code>ed:B0</code>). The Z80
								skips them as NOPs. Use this to take the usual replication machinery away and
								observe which alternative strategies evolve. Patterns appear as removable chips; the
								count shows how many instructions are affected across the base, CB and ED pages.
							</p>
						{:else}<p>
								{isaSuppressionDescription} Patterns appear as removable chips; the count shows the affected
								opcodes.
							</p>{/if}

						<h4>Colormap</h4>
						<p>
							Choose between three visual themes: Rainbow (warm opcode-aware), Ocean (cool blues),
							and Thermal (heat map).
						</p>
					{:else if helpTab === 'analysis'}
						<h4>Species Tracking</h4>
						<p>
							The <strong>Species</strong> tab identifies distinct cell types by hashing each cell's full
							byte content (FNV-1a). Identical tapes share a fingerprint; different tapes can also collide.
							These are composition groups, not measured species or lineages. The top 10 are shown as
							colored tiles whose color matches the cell's average appearance on the grid. Click a tile
							to inspect its tape and selected-processor disassembly in this sidebar. The groups use complete-tape
							fingerprints; hash collisions are possible, and these groups do not establish ancestry.
						</p>

						<h4>Diversity Metrics</h4>
						<p>
							The <strong>Diversity</strong> tab tracks four metrics over time, each independently scaled:
						</p>
						<!-- eslint-disable svelte/no-at-html-tags -- KaTeX output is safe (static strings) -->
						<div class="help-metric-list">
							<div class="help-metric-item">
								<span class="help-metric-swatch" style="background:rgb(255,180,50)"></span>
								<div>
									<strong>H</strong> (Shannon entropy) &mdash; measures evenness of species
									distribution. High H = many species at similar frequency. Low H = monoculture.
									<div class="help-metric-eq">{@html texBlock('H = -\\sum_{i} p_i \\ln p_i')}</div>
								</div>
							</div>
							<div class="help-metric-item">
								<span class="help-metric-swatch" style="background:rgb(100,200,255)"></span>
								<div>
									<strong>D</strong> (Simpson's diversity) &mdash; probability that two random cells
									are different species. {@html tex('D \\to 1')} = diverse, {@html tex('D \\to 0')} =
									monoculture.
									<div class="help-metric-eq">{@html texBlock('D = 1 - \\sum_{i} p_i^{\\,2}')}</div>
								</div>
							</div>
							<div class="help-metric-item">
								<span class="help-metric-swatch" style="background:rgb(120,255,120)"></span>
								<div>
									<strong>SPP</strong> (species richness) &mdash; count of species with {@html tex(
										'p_i > 0.1\\%'
									)}. Tracks how many distinct genomes are meaningfully present.
								</div>
							</div>
							<div class="help-metric-item">
								<span class="help-metric-swatch" style="background:rgb(255,130,180)"></span>
								<div>
									<strong>UNIQ</strong> (fingerprint groups) &mdash; distinct tape hashes in the grid.
									Hash-based composition before any abundance threshold. Region samples verify exact tapes.
								</div>
							</div>
						</div>
						<!-- eslint-enable svelte/no-at-html-tags -->

						<h4>Checkpoints</h4>
						<p>
							<kbd>{navigator.platform.includes('Mac') ? 'Cmd' : 'Ctrl'}+S</kbd> saves a checkpoint
							as an <code>.acell</code> file. This is a valid PNG image (viewable in any image app)
							with the population and its condition appended after the image data. In Measure →
							Export, choose Import checkpoint and then Restore checkpoint. You can also choose a
							file with
							<kbd>{navigator.platform.includes('Mac') ? 'Cmd' : 'Ctrl'}+O</kbd> or drag &amp; drop onto
							the canvas. Both paths stage the file for explicit restore. The file contains the complete
							grid, all parameters, and suppression patterns &mdash; everything needed to resume from
							these bytes. Legacy files do not store the complete random stream, so they do not reproduce
							the exact historical continuation.
						</p>

						<h4>Opcode Suppression</h4>
						<p>
							Suppressing opcodes (e.g. LD, EX, ADD) forces the system to find replication
							strategies using only the remaining instructions. This can push the system past
							trivial self-replication into richer emergent behavior like spatial speciation,
							error-correcting patterns, and division of labor between cell types.
						</p>
					{:else if helpTab === 'keys'}
						<div class="modal-shortcuts">
							{#each [['Space', 'Play / pause'], ['.', 'Step one update'], ['V / P / R / M / C', 'Select, probe, region, marker, compare'], ['1–5', 'World, Measure, Experiments, Guide, About'], ['F', 'Fit lattice'], ['+ / −', 'Zoom'], ['Arrow keys', 'Move the probe'], ['Cmd/Ctrl + S', 'Save checkpoint'], ['Cmd/Ctrl + O', 'Import checkpoint'], ['?', 'Show shortcuts'], ['Escape', 'Close current detail']] as shortcut (shortcut[0])}<div
									class="shortcut"
								>
									<kbd>{shortcut[0]}</kbd><span>{shortcut[1]}</span>
								</div>{/each}
						</div>
					{:else if helpTab === 'about'}
						{#each ABOUT_SECTIONS as section (section.id)}
							<h4>{section.title}</h4>
							{#each section.paragraphs as paragraph (paragraph)}<p>
									{scientificText(paragraph)}
								</p>{/each}
							{#if section.links}<p class="cmap-note">
									{#each section.links as link, index (link.href)}{#if index > 0}
											·
										{/if}
										<!-- eslint-disable svelte/no-navigation-without-resolve -- Curated scientific references are absolute external URLs. -->
										<a class="help-link" href={link.href} target="_blank" rel="noopener noreferrer"
											>{link.label}</a
										>
										<!-- eslint-enable svelte/no-navigation-without-resolve -->
									{/each}
								</p>{/if}
						{/each}
						<p class="cmap-note">
							Original WebGPU instrument inspired by Alexander Mordvintsev’s <a
								class="help-link"
								href="https://github.com/znah/zff"
								target="_blank"
								rel="noopener noreferrer">zff implementation</a
							>. Developed with Claude Code. SvelteKit · TypeScript · WebGPU.
						</p>
					{/if}
				</div>
			</section>
		</div>
		<footer class="sidebar-footer" hidden={sidebarCollapsed}>
			<button onclick={() => showMeasure('record')}>Export & checkpoints</button><button
				title="Keyboard shortcuts (?)"
				aria-label="Keyboard shortcuts"
				onclick={() => {
					selectTab('guide');
					helpTab = 'keys';
				}}>⌨</button
			>
		</footer>
	</aside>
	<footer class="status-bar" aria-label="Run and world status">
		<div class="transport">
			<button
				class="play-control"
				title={`${playing ? 'Pause' : 'Play'} (Space)`}
				aria-label={playing ? 'Pause' : 'Play'}
				disabled={!engine || machineBusy || checkpointBusy}
				onclick={togglePlay}
				><svg width="13" height="13" viewBox="0 0 14 14" fill="currentColor" aria-hidden="true"
					>{#if playing}<path d="M3 2h3v10H3zm5 0h3v10H8z" />{:else}<path
							d="M3 1.5v11l9-5.5z"
						/>{/if}</svg
				><span>{playing ? 'Pause' : 'Play'}</span></button
			><button
				title="Step one update (.)"
				aria-label="Step"
				disabled={!engine || machineBusy || checkpointBusy}
				onclick={stepOnce}>▸│</button
			>
			<div class="speed-control" title="Speed">
				<InstrumentSelect
					label="Speed"
					value={String(speed)}
					options={[1, 2, 4, 8].map((value) => ({ value: String(value), label: `${value}×` }))}
					onchange={(value) => setSpeed(Number(value))}
				/>
			</div>
			<button
				title="Restart current world"
				aria-label="Restart current world"
				disabled={!engine || machineBusy || checkpointBusy}
				onclick={() => void handleReset()}>↻</button
			>
		</div>
		<button class="status-run" onclick={togglePlay}
			><i class:running={playing}></i><span>{playing ? 'Running' : 'Paused'}</span></button
		>
		<button
			class="status-item status-rate"
			title="Completed population updates per second"
			onclick={() => showMeasure('charts')}
			><strong data-numeric>{(opsPerSec / Math.max(1, pairCount * z80Steps)).toFixed(0)}</strong
			><span>updates/s</span></button
		>
		<button
			class="status-item"
			title="Population update count"
			onclick={() => showMeasure('history')}
			><span>Epoch</span><strong data-numeric>{epoch.toLocaleString()}</strong></button
		>
		<div class="status-spacer"></div>
		<button class="status-item status-core" onclick={() => jumpWorld('condition', 'machine-select')}
			><strong
				>{applied.isa === 'z80'
					? `Z80 ${applied.compat}${applied.variant === 'zilog-nmos' ? ' NMOS' : ''}`
					: applied.isa === 'm6502'
						? 'NMOS 6502'
						: '8086'}</strong
			></button
		>
		<button class="status-item status-seed" onclick={() => jumpWorld('condition', 'seed-input')}
			><span>Seed</span><strong>{applied.seed}</strong></button
		>
		<button class="status-item status-size" onclick={() => jumpWorld('condition', 'tape-length')}
			><strong>{applied.width} × {applied.height}</strong><span
				>L{applied.length}/P{applied.ring}</span
			></button
		>
		<button
			class="status-item status-chemistry"
			onclick={() => jumpWorld('condition', 'lethal-probability')}
			><span>Zero</span><strong
				>{Math.round(applied.lethalProbability * 100)}% lethal{applied.interaction === 'reversible'
					? ` · reversible ${Math.round(applied.dissipation * 100)}%`
					: ''}{boundaryStatus ? ` · ${boundaryStatus}` : ''}</strong
			></button
		>
	</footer>
	{#snippet nextWorldSummary()}
		<div class="next-world-summary" role="group" aria-label="Next world summary">
			<strong
				>{worldDraft.isa === 'z80'
					? worldDraft.variant === 'zilog-nmos'
						? 'Z80 · Zilog NMOS'
						: 'Z80'
					: worldDraft.isa === 'm6502'
						? 'NMOS 6502'
						: '8086 · coherent fetch'} · compat {worldDraft.compat}</strong
			>
			<p>
				Mutation {worldDraft.mutationExponent >= 31
					? 'off'
					: `1 in ${2 ** worldDraft.mutationExponent}`} · Zero {Math.round(
					worldDraft.lethalProbability * 100
				)}% lethal
			</p>
			<p>
				{worldDraft.pairAttempts.toLocaleString()} pairs/update · {worldDraft.steps} steps/encounter
			</p>
			<p>
				{worldDraft.width} × {worldDraft.height} tapes · L{worldDraft.length}/P{worldDraft.ring} · seed
				{worldDraft.seed}
			</p>
			<p class="replacement-note">Replaces the current population. Playback stays as it is.</p>
		</div>
	{/snippet}

	{#snippet disassemblyPanel(tape: Uint8Array)}
		<details class="inspector-disassembly">
			<summary>Disassembly</summary>
			{#each disassemble(tape) as line (line.offset)}<div class="disassembly-line">
					<span>{line.offset.toString(16).toUpperCase().padStart(2, '0')}</span><span
						>{line.bytes.map(hexByte).join(' ')}</span
					><span>{line.mnemonic}</span>
				</div>{/each}
		</details>
	{/snippet}

	{#snippet culturePanel(tape: Uint8Array)}
		{#if engine}
			<EncounterBench
				{engine}
				{tape}
				sampleId={inspectedTape
					? `cell:${inspectedSampleId}`
					: `species:${speciesModal?.hash}:${speciesSampleUpdate}`}
				sampleCell={inspectedTape ? inspectedCell : null}
				sampleUpdate={inspectedTape ? inspectedUpdate : speciesSampleUpdate}
				sampleLabel={inspectedTape
					? `Cell ${inspectedCell}`
					: `Population fingerprint ${speciesModal?.hash}`}
				{seed}
				{byteColor}
				onpause={() => {}}
				onbusy={(value) => {
					cultureBusy = value;
				}}
			/>
		{/if}
	{/snippet}
</div>

<style>
	.control-lock {
		border: 0;
		padding: 0;
		margin: 0;
		min-width: 0;
	}
	.legacy-scope {
		color: var(--text-primary);
	}

	.legacy-scope :global(input:focus),
	.legacy-scope :global(select:focus),
	.legacy-scope :global(textarea:focus) {
		outline: none;
		box-shadow: none;
		border-color: var(--border-default);
	}
	.legacy-scope :global(input:focus-visible),
	.legacy-scope :global(select:focus-visible),
	.legacy-scope :global(textarea:focus-visible) {
		outline: 1px solid var(--accent);
		outline-offset: 2px;
	}

	/* Native selects can match :focus-visible after a pointer click in Chrome. */
	.legacy-scope.pointer-focus :global(:focus) {
		outline: none;
		box-shadow: none;
	}

	/* Prevent mobile zoom/scroll escaping on all UI overlays */
	:global(html),
	:global(body) {
		touch-action: manipulation;
		overscroll-behavior: none;
		-webkit-text-size-adjust: 100%;
	}

	/* Preserve visible keyboard focus in the historical interface. */
	button:focus-visible {
		outline: 1px solid var(--accent);
		outline-offset: 2px;
	}

	/* ── Toolbar ── */

	/* Speed menu */

	/* ── Info bar ── */
	.info-bar {
		position: fixed;
		bottom: 12px;
		left: 12px;
		z-index: 40;
		display: flex;
		flex-direction: column;
		gap: 0;
		padding: 8px;
		background: var(--bg-panel);
		border: 1px solid var(--border-subtle);
		border-radius: 10px;
		backdrop-filter: blur(16px);
		box-shadow: 0 2px 20px rgba(0, 0, 0, 0.4);
		font-size: 11px;
		font-family: monospace;
		width: 252px;
		overflow: visible;
	}
	.info-row {
		display: flex;
		align-items: center;
		gap: 8px;
	}
	.info-item {
		display: flex;
		align-items: center;
		gap: 4px;
	}
	.info-label {
		color: var(--text-subtle);
		text-transform: uppercase;
		font-size: 10px;
		letter-spacing: 0.05em;
	}
	.info-value {
		color: var(--text-secondary);
		font-variant-numeric: tabular-nums;
		width: 5ch;
		text-align: right;
		display: inline-block;
		white-space: pre;
	}
	.info-sep {
		width: 1px;
		height: 10px;
		background: var(--border-subtle);
	}
	.info-chart-toggle {
		display: flex;
		align-items: center;
		justify-content: center;
		width: 22px;
		height: 22px;
		border: none;
		background: transparent;
		cursor: pointer;
		color: var(--text-subtle);
		margin-left: auto;
		margin-right: 4px;
		border-radius: 4px;
		padding: 4px;
	}
	.info-chart-toggle:hover {
		background: var(--bg-hover);
	}
	.info-chart {
		margin-top: 6px;
		border-top: 1px solid var(--border-subtle);
		padding-top: 6px;
	}
	.info-chart-svg {
		width: 100%;
		height: 90px;
		background: var(--bg-subtle);
		border-radius: 4px;
	}
	.freq-grid {
		display: grid;
		grid-template-columns: repeat(5, 44px);
		gap: 4px;
		margin-top: 6px;
	}
	.freq-cell {
		position: relative;
		display: flex;
		align-items: center;
		justify-content: center;
		width: 44px;
		height: 44px;
		border: none;
		border-radius: 6px;
		cursor: pointer;
		font-size: 8.5px;
		font-family: monospace;
		font-weight: 600;
		text-align: center;
		line-height: 1.15;
		padding: 2px;
		overflow: hidden;
		transition: opacity 0.1s;
	}
	.freq-cell.highlighted {
		outline: 2px solid rgba(255, 255, 255, 0.7);
		z-index: 1;
	}
	.freq-cell.dimmed {
		opacity: 0.35;
	}
	.freq-cell-suppressed-group {
		background: rgba(255, 255, 255, 0.06);
		border: 1px dashed var(--border-muted);
		color: var(--text-muted);
		cursor: default;
		font-size: 9px;
	}

	.freq-rank {
		position: absolute;
		top: 2px;
		left: 3px;
		font-size: 6px;
		opacity: 0.5;
		font-weight: 400;
	}
	.freq-label {
		overflow: hidden;
		word-break: break-all;
		max-height: 100%;
		display: -webkit-box;
		-webkit-line-clamp: 2;
		line-clamp: 2;
		-webkit-box-orient: vertical;
	}
	.freq-suppressed-row {
		display: flex;
		flex-wrap: wrap;
		gap: 3px;
		margin-top: 4px;
		padding-top: 4px;
		border-top: 1px solid var(--border-subtle);
		align-items: center;
	}
	.freq-suppress-chip {
		display: inline-flex;
		align-items: center;
		gap: 3px;
		padding: 1px 6px;
		font-size: 9px;
		font-family: var(--font-mono, monospace);
		font-weight: 600;
		color: var(--text-secondary);
		background: rgba(255, 255, 255, 0.08);
		border: 1px solid var(--border-muted);
		border-radius: 3px;
		cursor: pointer;
		line-height: 1.4;
		transition: all 0.12s;
	}
	.freq-suppress-chip:hover {
		background: rgba(220, 80, 60, 0.25);
		border-color: rgba(220, 80, 60, 0.5);
	}
	.freq-suppress-chip svg {
		opacity: 0.4;
		flex-shrink: 0;
	}
	.freq-suppress-chip:hover svg {
		opacity: 1;
	}
	.freq-suppress-count {
		font-size: 8.5px;
		color: var(--text-muted);
		margin-left: 2px;
	}

	/* ── Info tabs ── */
	.info-tabs {
		display: flex;
		gap: 0;
		margin-top: 6px;
		border-bottom: 1px solid var(--border-subtle);
	}
	.info-tab {
		background: none;
		border: none;
		padding: 4px 10px;
		font-size: 9px;
		color: var(--text-muted);
		cursor: pointer;
		border-bottom: 2px solid transparent;
		margin-bottom: -1px;
		font-family: monospace;
		text-transform: uppercase;
		letter-spacing: 0.05em;
		transition:
			color 0.15s,
			border-color 0.15s;
	}
	.info-tab:hover {
		color: var(--text-secondary);
	}
	.info-tab.active {
		color: var(--accent);
		border-bottom-color: var(--accent);
	}

	.species-waiting {
		height: 90px;
		display: flex;
		align-items: center;
		justify-content: center;
		font-size: 9px;
		color: var(--text-muted);
		background: var(--bg-subtle);
		border-radius: 4px;
	}

	/* ── Diversity tab ── */
	.diversity-grid {
		display: grid;
		grid-template-columns: repeat(2, 1fr);
		gap: 4px;
		margin-top: 6px;
	}
	.diversity-card {
		display: flex;
		align-items: center;
		gap: 5px;
		padding: 4px 6px;
		border-radius: 5px;
		background: var(--bg-subtle);
		cursor: default;
		transition: opacity 0.1s;
		font-size: 10px;
		font-family: monospace;
	}
	.diversity-card.dimmed {
		opacity: 0.35;
	}
	.diversity-card.highlighted {
		outline: 1px solid rgba(255, 255, 255, 0.3);
	}
	.diversity-swatch {
		width: 8px;
		height: 8px;
		border-radius: 2px;
		flex-shrink: 0;
	}
	.diversity-label {
		color: var(--text-muted);
		text-transform: uppercase;
		font-size: 8px;
		letter-spacing: 0.05em;
		min-width: 24px;
	}
	.diversity-value {
		color: var(--text-primary);
		font-weight: 600;
		margin-left: auto;
	}

	/* ── Species inspect modal ── */

	/* ── Tile tooltip ── */

	/* ── Panels (shared) ── */

	/* ── Settings panel ── */

	/* Seed input */

	/* Grid type toggle — inline with param label */

	/* Custom slider */

	/* Suppress opcodes */

	.suppress-chips {
		display: flex;
		flex-wrap: wrap;
		gap: 4px;
		margin-top: 6px;
		align-items: center;
	}
	.suppress-chip {
		display: inline-flex;
		align-items: center;
		gap: 4px;
		padding: 2px 6px;
		font-size: 9px;
		font-family: var(--font-mono, monospace);
		font-weight: 600;
		color: var(--text-primary);
		background: rgba(255, 255, 255, 0.1);
		border: 1px solid var(--border-muted);
		border-radius: 3px;
		cursor: pointer;
		transition: all 0.12s;
		line-height: 1.4;
	}
	.suppress-chip:hover {
		background: rgba(220, 80, 60, 0.25);
		border-color: rgba(220, 80, 60, 0.5);
	}

	/* Presets */

	.preset-save-row {
		display: flex;
		gap: 4px;
		margin-top: 6px;
	}

	/* Preset overwrite confirmation dialog */

	/* Colormap selector */

	.cmap-preview-rainbow {
		background: linear-gradient(
			90deg,
			#1a0404,
			#c85a5a,
			#c8a05a,
			#5ac85a,
			#5a8ac8,
			#8a5ac8,
			#c85a8a
		);
	}
	.cmap-preview-ocean {
		background: linear-gradient(
			90deg,
			#0a1218,
			#2a4a5a,
			#3a7a8a,
			#4a9aaa,
			#5ab0c0,
			#6ac0d0,
			#8ad0e0
		);
	}
	.cmap-preview-thermal {
		background: linear-gradient(
			90deg,
			#000000,
			#4a0a2a,
			#8a1a1a,
			#c84a0a,
			#e88a2a,
			#f0c050,
			#f8f0c0
		);
	}

	/* Color adjustments */

	/* ── Genome tooltip ── */

	.tip-grid {
		display: grid;
		grid-template-columns: repeat(4, 46px);
		gap: 2px;
	}
	.tip-cell {
		width: 46px;
		height: 46px;
		display: flex;
		align-items: center;
		justify-content: center;
		overflow: hidden;
		font-size: 8.5px;
		font-family: monospace;
		text-align: center;
		line-height: 1.15;
		font-weight: 600;
		padding: 2px;
	}
	.tip-cell.operand {
		opacity: 0.75;
		font-weight: 400;
		font-size: 10px;
	}

	/* ── Hex byte tooltip: 19 hex cells in 3-4-5-4-3 hexagonal arrangement ── */

	/* Byte positions in 3-4-5-4-3 hexagonal arrangement */
	/* Byte 0: center (row 2, col 2) */

	/* Ring 1: bytes 1-6 (clockwise from N) */

	/* Ring 2: bytes 7-18 (clockwise from NE) */

	/* ── Help modal ── */

	.modal-body {
		padding: 16px;
		font-size: 13px;
		line-height: 1.6;
		color: var(--text-muted);
		max-height: 65vh;
		overflow-y: auto;
	}
	.modal-body h4 {
		color: var(--text-secondary);
		font-size: 13px;
		font-weight: 600;
		margin: 14px 0 4px;
	}
	.modal-body h4:first-of-type {
		margin-top: 2px;
	}
	.modal-body p {
		margin-bottom: 10px;
	}
	.cmap-section {
		margin-bottom: 10px;
	}
	.cmap-label {
		font-size: 10px;
		font-weight: 600;
		color: var(--text-subtle);
		text-transform: uppercase;
		letter-spacing: 0.5px;
		margin-bottom: 4px;
	}
	.cmap-grid {
		display: grid;
		grid-template-columns: 1fr 1fr 1fr;
		gap: 1px 8px;
	}
	.cmap-entry {
		display: flex;
		align-items: center;
		gap: 5px;
		font-size: 11px;
		line-height: 1.8;
	}
	.help-swatch {
		width: 12px;
		height: 12px;
		border-radius: 2px;
		flex-shrink: 0;
		border: 1px solid rgba(255, 255, 255, 0.08);
	}
	.cmap-hex {
		font-family: monospace;
		font-size: 10px;
		color: var(--text-subtle);
		min-width: 22px;
	}
	.cmap-note {
		font-size: 11px;
		color: var(--text-subtle);
		margin: 8px 0 12px;
	}
	.help-steps {
		margin: 8px 0;
		padding-left: 20px;
	}
	.help-steps li {
		margin-bottom: 6px;
		line-height: 1.45;
	}
	.tape-layouts {
		display: flex;
		gap: 16px;
		margin: 10px 0;
		flex-wrap: wrap;
		align-items: flex-start;
		justify-content: center;
	}
	.tape-layout-item {
		flex: 1;
		min-width: 140px;
		max-width: 200px;
		text-align: center;
	}
	.tape-layout-label {
		font-size: 11px;
		font-weight: 600;
		color: var(--text-secondary);
		margin-bottom: 6px;
	}
	.tape-layout-note {
		font-size: 10px;
		color: var(--text-muted);
		margin-top: 4px;
	}
	.tape-svg {
		width: 100%;
		max-width: 110px;
		display: block;
		margin: 0 auto;
	}
	.tape-combine-svg {
		width: 100%;
		margin: 8px 0;
		display: block;
		border-radius: 8px;
		background: rgba(0, 0, 0, 0.2);
		border: 1px solid rgba(255, 255, 255, 0.04);
		padding: 8px;
	}
	.help-link {
		color: var(--accent);
		text-decoration: none;
	}
	.help-link:hover {
		text-decoration: underline;
	}
	/* ── Analysis help tab ── */
	.help-metric-list {
		display: flex;
		flex-direction: column;
		gap: 6px;
		margin: 8px 0;
	}
	.help-metric-item {
		display: flex;
		align-items: baseline;
		gap: 6px;
		font-size: 12px;
		line-height: 1.4;
	}
	.help-metric-swatch {
		width: 8px;
		height: 8px;
		border-radius: 2px;
		flex-shrink: 0;
		margin-top: 5px;
	}
	.help-metric-eq {
		margin: 4px 0 2px;
	}
	.help-metric-eq :global(.katex) {
		font-size: 1em;
	}
	/* ── Z80 reference tab ── */
	.z80-reg-table {
		margin-bottom: 10px;
	}
	.z80-reg-row {
		display: grid;
		grid-template-columns: 80px 48px 1fr;
		gap: 6px;
		font-size: 11px;
		line-height: 1.9;
		color: var(--text-muted);
		border-bottom: 1px solid rgba(255, 255, 255, 0.03);
	}
	.z80-reg-row.head {
		font-weight: 600;
		color: var(--text-subtle);
		text-transform: uppercase;
		font-size: 10px;
		letter-spacing: 0.5px;
		border-bottom: 1px solid var(--border-subtle);
	}
	.z80-reg {
		font-family: monospace;
		color: var(--accent);
		font-weight: 600;
	}
	.z80-notation-table {
		margin-bottom: 10px;
	}
	.z80-notation-row {
		display: grid;
		grid-template-columns: 56px 1fr;
		gap: 8px;
		font-size: 11px;
		line-height: 1.9;
		color: var(--text-muted);
		border-bottom: 1px solid rgba(255, 255, 255, 0.03);
	}
	.z80-notation-row.head {
		font-weight: 600;
		color: var(--text-subtle);
		text-transform: uppercase;
		font-size: 10px;
		letter-spacing: 0.5px;
		border-bottom: 1px solid var(--border-subtle);
	}
	.z80-cat {
		margin-bottom: 10px;
	}
	.z80-cat-label {
		font-size: 10px;
		font-weight: 600;
		color: var(--text-subtle);
		text-transform: uppercase;
		letter-spacing: 0.5px;
		margin-bottom: 3px;
	}
	.z80-instr-grid {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 1px 12px;
	}
	.z80-instr {
		font-size: 11px;
		line-height: 1.8;
		color: var(--text-muted);
	}
	.z80-op {
		font-family: monospace;
		color: var(--accent);
		font-weight: 600;
		font-size: 11px;
	}

	.modal-shortcuts {
		margin-top: 14px;
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 4px 16px;
	}
	.shortcut {
		display: flex;
		align-items: center;
		gap: 8px;
		font-size: 11px;
		color: var(--text-muted);
	}
	.shortcut kbd {
		display: inline-block;
		padding: 2px 6px;
		background: var(--bg-muted);
		border: 1px solid var(--border-muted);
		border-radius: 4px;
		font-size: 10px;
		font-family: monospace;
		color: var(--text-secondary);
		min-width: 20px;
		text-align: center;
	}

	/* ── Mobile ── */
	@media (max-width: 768px) {
		/* Toolbar: horizontal pill at bottom center */

		/* Info bar: compact at top-left, scaled down */
		.info-bar {
			bottom: auto;
			top: 8px;
			left: 8px;
			width: auto;
			min-width: 0;
			max-width: min(210px, calc(100vw - 16px));
			padding: 6px 8px;
			font-size: 10px;
			border-radius: 8px;
		}
		.info-row {
			gap: 5px;
		}
		.info-label {
			font-size: 8px;
		}
		.info-value {
			font-size: 10px;
		}
		.info-chart-svg {
			height: 68px;
		}
		.freq-grid {
			grid-template-columns: repeat(5, 34px);
			gap: 3px;
		}
		.freq-cell {
			width: 34px;
			height: 34px;
			font-size: 8px;
			border-radius: 4px;
			padding: 1px;
		}
		.freq-rank {
			font-size: 5px;
			top: 1px;
			left: 2px;
		}

		/* Settings panel: compact, anchored bottom-right above toolbar */

		/* Bigger slider thumbs for touch (track stays thin) */

		/* Help modal: full-screen on mobile */

		.modal-body {
			max-height: none;
			flex: 1;
			overflow-y: auto;
			padding: 12px;
			font-size: 12px;
		}

		/* Hide keys tab on mobile (no keyboard) */

		/* Prevent iOS zoom on input focus — font-size must be ≥ 16px */

		/* Genome tooltip: smaller on mobile */
	}

	.experiment-condition {
		margin-top: 6px;
		font-size: 10px;
		line-height: 1.45;
		color: var(--text-muted);
	}
	.experiment-condition p {
		margin: 0 0 6px;
	}
	.experiment-condition strong {
		color: var(--text-secondary);
		font-weight: 500;
	}
	.experiment-condition summary {
		cursor: pointer;
		color: var(--accent);
		font-size: 10px;
	}
	.experiment-condition details p:first-of-type {
		margin-top: 6px;
	}
	.source-path {
		overflow-wrap: anywhere;
		font-family: monospace;
		font-size: 9px;
	}

	.instrument-error {
		color: var(--accent-rose);
		font-size: 10px;
		overflow-wrap: anywhere;
		margin-top: 5px;
	}

	.guide-steps {
		list-style: decimal;
		padding-left: 20px;
		margin: 0 0 12px;
	}
	.guide-steps li {
		margin-bottom: 6px;
	}
	.inspector-disassembly {
		max-width: 340px;
		margin-top: 7px;
	}
	.inspector-disassembly summary {
		color: var(--accent);
		font-size: 10px;
		cursor: pointer;
	}
	.disassembly-line {
		display: grid;
		grid-template-columns: 18px 66px minmax(0, 1fr);
		gap: 5px;
		padding-top: 4px;
		font: 9px/1.4 monospace;
		color: var(--text-muted);
		overflow-wrap: anywhere;
	}

	@media (max-width: 768px) {
		.info-bar {
			top: auto;
			bottom: 52px;
			left: 8px;
			max-width: calc(100vw - 16px);
			padding: 5px 8px;
		}
	}
	.unified-shell {
		--sidebar-width: 376px;
		--header-height: 48px;
		--status-height: 44px;
		--rail-width: 40px;
		position: fixed;
		inset: 0;
		overflow: hidden;
		background: #0e0e0e;
		font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
		font-size: 12px;
		font-variant-numeric: tabular-nums;
	}
	.unified-shell.sidebar-collapsed {
		--sidebar-width: 48px;
	}
	.app-header {
		position: absolute;
		left: 0;
		top: 0;
		right: var(--sidebar-width);
		height: var(--header-height);
		display: flex;
		align-items: center;
		gap: 12px;
		padding: 0 12px 0 16px;
		background: #171715;
		border-bottom: 1px solid #ffffff0b;
		z-index: 30;
	}
	.wordmark {
		display: flex;
		align-items: center;
		gap: 8px;
		font-size: 16px;
		font-weight: 600;
		letter-spacing: -0.5px;
		flex-shrink: 0;
		margin-right: 8px;
	}
	.wordmark svg {
		color: var(--accent);
	}
	.field-choice {
		width: 170px;
	}
	.header-space,
	.rail-space,
	.status-spacer {
		flex: 1;
	}
	.unified-shell :global(button) {
		font-family: inherit;
	}
	.unified-shell :global(button:disabled) {
		opacity: 0.45;
		cursor: default;
	}
	.header-look,
	.camera-controls button {
		min-height: 30px;
		padding: 4px 10px;
		color: var(--text-muted);
		border: 1px solid transparent;
		border-radius: 5px;
		background: transparent;
		font-size: 12px;
		cursor: pointer;
	}
	.header-look:hover,
	.camera-controls button:hover,
	.header-look.chosen {
		color: var(--text-primary);
		background: #ffffff08;
	}
	.camera-controls {
		display: flex;
		align-items: center;
		gap: 2px;
	}
	.camera-controls button:first-child,
	.camera-controls button:nth-child(2) {
		width: 28px;
		padding: 0;
		font-size: 17px;
	}
	.canvas-viewport {
		position: absolute;
		top: var(--header-height);
		bottom: var(--status-height);
		left: 0;
		right: calc(var(--sidebar-width) + var(--rail-width));
		overflow: hidden;
		background: #09090b;
	}
	.lattice-canvas {
		display: block;
		width: 100%;
		height: 100%;
	}
	.lattice-canvas:focus-visible {
		outline: 1px solid var(--accent);
		outline-offset: -2px;
	}
	.viewport-caption {
		position: absolute;
		left: 12px;
		bottom: 10px;
		max-width: calc(100% - 24px);
		pointer-events: none;
		color: #c8c1b8;
		font-size: 10px;
		line-height: 18px;
		text-shadow: 0 1px 4px #000;
	}
	.viewport-caption span {
		display: inline;
		background: #10100fd9;
		padding: 3px 6px;
		border-radius: 3px;
	}
	.gpu-message {
		position: absolute;
		inset: 0;
		z-index: 2;
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		padding: 32px;
		text-align: center;
		gap: 12px;
		background: #0e0e0e;
	}
	.gpu-message p {
		max-width: 360px;
		color: var(--text-muted);
	}
	.gpu-message button {
		color: var(--accent);
	}
	.tool-rail {
		position: absolute;
		right: var(--sidebar-width);
		top: var(--header-height);
		bottom: var(--status-height);
		width: var(--rail-width);
		display: flex;
		flex-direction: column;
		gap: 4px;
		padding: 8px 4px;
		background: #171715;
		border-right: 1px solid #ffffff08;
		border-left: 1px solid #ffffff0b;
		z-index: 20;
	}
	.tool-rail button {
		width: 32px;
		height: 34px;
		border-radius: 5px;
		display: flex;
		align-items: center;
		justify-content: center;
		color: #938d83;
		border: 1px solid transparent;
		background: transparent;
		cursor: pointer;
		transition:
			color 160ms ease,
			background 160ms ease;
	}
	.tool-rail button:hover {
		background: #ffffff08;
		color: #e8e4e0;
	}
	.tool-rail button.active {
		color: #dda579;
		background: #c8875a16;
		border-color: #c8875a26;
	}
	.app-sidebar {
		position: absolute;
		top: 0;
		right: 0;
		bottom: var(--status-height);
		width: var(--sidebar-width);
		background: #161615;
		border-left: 1px solid #ffffff0c;
		display: flex;
		flex-direction: column;
		z-index: 35;
		min-width: 0;
	}
	.sidebar-tabs {
		flex-shrink: 0;
		height: var(--header-height);
		display: flex;
		align-items: stretch;
		padding: 0 8px;
		gap: 2px;
		border-bottom: 1px solid #ffffff0c;
	}
	.sidebar-tabs button {
		position: relative;
		flex: 1;
		min-width: 0;
		color: #aaa297;
		font-size: 11px;
		border: 0;
		background: transparent;
		cursor: pointer;
		padding: 0 4px;
	}
	.sidebar-tabs button.active {
		color: #e5b18b;
	}
	.sidebar-tabs button.active::after {
		content: '';
		position: absolute;
		height: 2px;
		bottom: 0;
		left: 6px;
		right: 6px;
		background: var(--accent);
	}
	.tab-glyph {
		display: none;
		font-size: 18px;
	}
	.sidebar-scroll {
		flex: 1;
		min-height: 0;
		overflow-y: auto;
		overflow-x: hidden;
		scrollbar-gutter: stable;
	}
	.sidebar-section {
		padding: 20px 20px 16px;
	}
	/* Canvas annotations are clipped to the viewport and sit below shell content. */
	.unified-shell :global(.canvas-observations) {
		z-index: 0;
	}
	.sidebar-section,
	.sidebar-tabs,
	.sidebar-footer,
	.unified-shell :global(.observation-dock.embedded) {
		position: relative;
		z-index: 1;
	}
	.unified-shell :global(.observation-dock.embedded) {
		inset: auto;
	}
	.sidebar-section:not([hidden]),
	.unified-shell :global(.observation-dock.embedded:not([hidden])) {
		animation: panel-reveal 160ms cubic-bezier(0.2, 0.6, 0.3, 1);
	}
	@keyframes panel-reveal {
		from {
			opacity: 0.82;
		}
		to {
			opacity: 1;
		}
	}

	.sidebar-section[hidden],
	.measure-heading[hidden],
	.sidebar-footer[hidden] {
		display: none;
	}
	.section-heading {
		display: flex;
		justify-content: space-between;
		align-items: baseline;
		gap: 12px;
		margin-bottom: 20px;
	}
	.section-heading h1 {
		color: #e8e4e0;
		font-size: 19px;
		letter-spacing: -0.5px;
		font-weight: 550;
		margin: 0;
	}
	.section-context {
		color: #817b72;
		font-size: 10px;
	}
	.sidebar-section h2 {
		font-size: 13px;
		font-weight: 550;
		color: var(--text-primary);
		margin: 0;
	}
	.section-tabs,
	.measure-sections,
	.segmented {
		display: flex;
		padding: 3px;
		gap: 3px;
		background: #0e0e0d;
		border: 1px solid #ffffff06;
		border-radius: 6px;
	}
	.section-tabs button,
	.measure-sections button,
	.segmented button {
		border: 0;
		background: transparent;
		flex: 1;
		color: #91897f;
		padding: 6px;
		min-height: 28px;
		border-radius: 4px;
		cursor: pointer;
		font-size: 11px;
		white-space: nowrap;
	}
	.section-tabs button.active,
	.measure-sections button.active,
	.segmented button.active {
		color: #dbaa83;
		background: #2a2722;
	}
	.measure-sections {
		flex-wrap: wrap;
	}
	.measure-sections button {
		flex: 1 0 20%;
	}
	.measure-heading {
		padding-bottom: 8px;
	}
	.intro-copy {
		color: #aaa298;
		font-size: 12px;
		line-height: 1.7;
		margin: 16px 0 10px;
	}
	.text-action {
		width: 100%;
		display: flex;
		justify-content: space-between;
		color: #d9a681;
		background: transparent;
		padding: 6px 0;
		border: 0;
		cursor: pointer;
		font-size: 12px;
	}
	.text-action span {
		color: #786a5e;
	}
	.section-divider {
		height: 1px;
		background: #ffffff0c;
		margin: 24px 0;
	}
	.world-panel > .intro-copy {
		margin-top: 12px;
		margin-bottom: 8px;
	}
	.world-panel .section-divider {
		margin: 16px 0;
	}
	.checkpoint-section {
		padding-top: 0;
	}
	.checkpoint-section > h2 {
		padding-top: 16px;
		border-top: 1px solid #ffffff0c;
	}
	.section-title {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 8px;
		margin-bottom: 12px;
	}
	.live-label,
	.draft-state {
		font-size: 10px;
		color: #817b72;
	}
	.live-label {
		color: #92a390;
	}
	.draft-state.pending {
		color: #d2a279;
	}
	.control-field {
		margin: 16px 0;
	}
	.control-field > label,
	.field-label {
		display: block;
		color: #c4bcb1;
		font-size: 11px;
		margin: 0 0 8px;
	}
	.control-note {
		color: #817b72;
		font-size: 11px;
		line-height: 1.6;
		margin: 7px 0 0;
	}
	.label-row {
		display: flex;
		justify-content: space-between;
		align-items: center;
		gap: 12px;
		margin-bottom: 9px;
		color: #b9b1a6;
		font-size: 11px;
	}
	.label-row output {
		color: #d6cbbf;
		font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
	}
	.shell-input {
		height: 32px;
		width: 100%;
		padding: 5px 9px;
		border: 1px solid #ffffff13;
		background: #20201e;
		border-radius: 5px;
		color: #ddd4c9;
		font-size: 12px;
		box-shadow: none;
	}
	.shell-input[type='number'] {
		font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
	}
	.unified-shell input[type='range'] {
		display: block;
		width: 100%;
		height: 14px;
		padding: 0;
		margin: 0;
		accent-color: var(--accent);
		background: transparent;
		border: 0;
		box-shadow: none;
		cursor: pointer;
	}
	.inline-check {
		display: flex;
		align-items: center;
		gap: 8px;
		color: #b8b0a4;
		font-size: 11px;
		margin: 12px 0;
	}
	.unified-shell input[type='checkbox'] {
		appearance: auto;
		width: 13px;
		height: 13px;
		padding: 0;
		border: 1px solid #ffffff25;
		border-radius: 3px;
		accent-color: var(--accent);
		background: none;
		color: var(--accent);
		box-shadow: none;
	}
	.shell-disclosure {
		border-top: 1px solid #ffffff08;
		margin-top: 16px;
	}
	.shell-disclosure summary {
		color: #bcb3a7;
		font-size: 11px;
		cursor: pointer;
		padding: 12px 0;
	}
	.disclosure-body {
		padding-bottom: 8px;
	}
	.field-pair {
		display: flex;
		align-items: flex-end;
		gap: 8px;
		margin: 14px 0;
	}
	.field-pair label {
		flex: 1;
		min-width: 0;
		font-size: 11px;
		color: #a9a095;
	}
	.field-pair label input {
		margin-top: 7px;
	}
	.field-pair > span {
		padding-bottom: 7px;
		color: #69655e;
	}
	.apply-world {
		margin: 18px 0 4px;
		padding-top: 12px;
		border-top: 1px solid #c8875a24;
	}
	.apply-world p {
		font-size: 11px;
		color: #968e82;
		line-height: 1.6;
		margin: 0 0 12px;
	}
	.next-world-summary {
		margin-bottom: 12px;
	}
	.next-world-summary strong {
		display: block;
		color: #d1b194;
		font-size: 11px;
		font-weight: 500;
		line-height: 1.6;
	}
	.next-world-summary p {
		margin: 4px 0;
	}
	.next-world-summary .replacement-note {
		margin-top: 8px;
	}
	.primary-action,
	.secondary-action {
		display: block;
		min-height: 34px;
		width: 100%;
		border: 1px solid #c8875a40;
		border-radius: 5px;
		color: #e8c2a3;
		background: #c8875a16;
		font-size: 11px;
		padding: 8px 10px;
		cursor: pointer;
		text-align: center;
	}
	.primary-action:hover {
		background: #c8875a23;
		border-color: #c8875a70;
	}
	.secondary-action {
		margin: 12px 0;
		color: #c8bfb4;
		background: #ffffff04;
		border-color: #ffffff14;
	}
	.quiet-button {
		min-height: 28px;
		font-size: 11px;
		color: #b69a80;
		padding: 4px 0;
		border: 0;
		background: transparent;
		cursor: pointer;
	}
	.quiet-button:hover {
		color: #e0b890;
	}
	.apply-world > .quiet-button {
		display: block;
		margin: 5px auto 0;
	}
	.palette-options {
		display: flex;
		gap: 6px;
	}
	.palette-options button {
		flex: 1;
		min-width: 0;
		padding: 8px 6px;
		display: flex;
		flex-direction: column;
		gap: 7px;
		color: #999184;
		font-size: 11px;
		text-transform: capitalize;
		border: 1px solid #ffffff10;
		border-radius: 5px;
		background: transparent;
		cursor: pointer;
	}
	.palette-options button.active {
		color: #d4ac88;
		border-color: #c8875a66;
		background: #c8875a07;
	}

	.inspector-return {
		position: sticky;
		top: 0;
		z-index: 3;
		display: flex;
		align-items: center;
		gap: 8px;
		margin: -4px -20px 12px;
		padding: 4px 20px 8px;
		background: #161615;
		border-bottom: 1px solid #ffffff0c;
		color: #a39789;
		font-size: 11px;
	}
	.inspector-return .quiet-button {
		width: 28px;
		min-height: 32px;
		padding: 0;
		font-size: 20px;
	}
	.inspector-return h2 {
		flex: 1;
	}
	.inspector-return > span {
		white-space: nowrap;
		font-variant-numeric: tabular-nums;
	}
	.inline-tape-grid {
		margin: 12px 0;
		grid-template-columns: repeat(8, minmax(0, 1fr));
		gap: 4px;
	}
	.inline-tape-grid .tip-cell {
		min-width: 0;
		width: 100%;
		height: 32px;
		font-size: 9px;
		border-radius: 3px;
		overflow-wrap: anywhere;
	}
	.inline-notice {
		margin: 12px 0 0;
		color: #b8a18a;
		font-size: 11px;
		line-height: 1.5;
	}
	.saved-presets {
		display: flex;
		flex-direction: column;
		gap: 4px;
	}
	.saved-presets > button {
		min-height: 34px;
		padding: 6px 9px;
		display: flex;
		align-items: center;
		justify-content: space-between;
		border: 1px solid #ffffff0a;
		border-radius: 5px;
		color: #bdb5a9;
		background: #ffffff03;
		font-size: 12px;
		cursor: pointer;
	}
	.saved-presets > button span {
		font-size: 10px;
		color: #746d63;
	}
	.saved-presets > button.active {
		color: #d8ac87;
		border-color: #c8875a40;
	}
	.inline-confirmation {
		padding: 12px;
		border: 1px solid #c8875a44;
		border-radius: 5px;
		color: #bfb4a6;
		font-size: 12px;
	}
	.inline-confirmation button {
		margin-right: 16px;
	}
	.sidebar-footer {
		flex-shrink: 0;
		display: flex;
		height: 38px;
		align-items: center;
		justify-content: space-between;
		border-top: 1px solid #ffffff0b;
		padding: 0 16px;
	}
	.sidebar-footer button {
		color: #8c8377;
		font-size: 11px;
		border: 0;
		background: transparent;
		cursor: pointer;
	}
	.sidebar-footer button:hover {
		color: #d5c0ac;
	}
	.status-bar {
		position: absolute;
		bottom: 0;
		inset-inline: 0;
		height: var(--status-height);
		display: flex;
		align-items: center;
		gap: 4px;
		padding: 0 8px;
		background: #181816;
		border-top: 1px solid #ffffff0d;
		z-index: 40;
		white-space: nowrap;
	}
	.transport {
		display: flex;
		align-items: center;
		gap: 3px;
		padding-right: 12px;
		margin-right: 8px;
		border-right: 1px solid #ffffff0c;
	}
	.transport > button {
		height: 30px;
		min-width: 30px;
		padding: 0 7px;
		border: 1px solid transparent;
		border-radius: 4px;
		color: #aba092;
		background: transparent;
		cursor: pointer;
	}
	.transport > .play-control {
		display: flex;
		gap: 7px;
		align-items: center;
		color: #dfae88;
		padding: 0 8px;
		min-width: 70px;
	}
	.transport > button:hover {
		background: #ffffff08;
	}
	.speed-control {
		width: 58px;
	}
	.speed-control :global(button[role='combobox']) {
		min-height: 28px;
		padding: 3px 7px;
		border-color: transparent;
		background: transparent;
		font-size: 11px;
	}
	.speed-control :global(.choices) {
		top: auto;
		bottom: calc(100% + 8px);
		min-width: 58px;
	}
	.status-run,
	.status-item {
		display: flex;
		align-items: center;
		gap: 6px;
		height: 30px;
		border: 0;
		background: transparent;
		color: #c6bcb0;
		font-size: 11px;
		padding: 0 8px;
		border-radius: 4px;
		cursor: pointer;
	}
	.status-run {
		color: #a19b90;
	}
	.status-run i {
		width: 5px;
		height: 5px;
		background: #726c60;
		border-radius: 50%;
	}
	.status-run i.running {
		background: #9aa589;
		box-shadow: 0 0 0 3px #9aa5890c;
	}
	.status-item span {
		color: #777166;
	}
	.status-item strong {
		font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
		font-size: 11px;
		font-weight: 400;
	}
	.status-item [data-numeric] {
		min-width: 35px;
		text-align: right;
	}
	.status-item:hover,
	.status-run:hover {
		background: #ffffff06;
	}
	.charts-panel .info-bar {
		position: static;
		width: 100%;
		min-width: 0;
		max-width: none;
		padding: 0;
		border: 0;
		border-radius: 0;
		background: transparent;
		box-shadow: none;
		backdrop-filter: none;
	}
	.charts-panel .info-chart {
		width: 100%;
	}
	.charts-panel .info-chart-svg {
		width: 100%;
		height: 180px;
	}
	.charts-panel .info-tabs {
		margin: 12px 0;
	}
	.charts-panel .info-tab {
		min-height: 32px;
		font-size: 11px;
	}
	.charts-panel .freq-grid {
		grid-template-columns: repeat(5, minmax(0, 1fr));
	}
	.charts-panel .freq-cell {
		min-width: 0;
		min-height: 32px;
	}
	.charts-panel .info-row {
		padding: 0;
	}
	.documentation-content {
		padding: 16px 0 0;
		overflow: visible;
		max-height: none;
		font-size: 12px;
		line-height: 1.75;
		color: #aaa297;
	}
	.documentation-content h4 {
		font-family: inherit;
		color: #d7c7b7;
		margin: 24px 0 10px;
		font-size: 13px;
	}
	.documentation-content > h4:first-child {
		margin-top: 6px;
	}
	.documentation-content p {
		margin: 10px 0;
	}
	.documentation-content :global(svg) {
		max-width: 100%;
	}
	.documentation-content :global(.katex-display) {
		overflow-x: auto;
		overflow-y: hidden;
		max-width: 100%;
		font-size: 0.85em;
	}
	.documentation-content :global(table) {
		width: 100%;
		font-size: 10px;
	}
	.documentation-content .modal-shortcuts {
		grid-template-columns: 1fr;
	}
	.documentation-content .shortcut {
		display: flex;
		align-items: flex-start;
		gap: 12px;
		margin: 10px 0;
		line-height: 1.6;
	}
	.documentation-content .shortcut kbd {
		min-width: 70px;
		flex-shrink: 0;
		font-size: 10px;
	}
	.experiments-panel .experiment-condition {
		padding: 0;
		background: transparent;
		border: 0;
		margin: 16px 0;
		font-size: 12px;
	}
	.experiments-panel .experiment-condition p {
		line-height: 1.7;
		color: #a89f93;
	}
	.experiments-panel .experiment-condition details {
		margin-top: 12px;
		font-size: 11px;
	}
	.experiments-panel .source-path {
		overflow-wrap: anywhere;
		font-size: 10px;
	}
	.sidebar-collapsed .sidebar-tabs {
		height: auto;
		flex-direction: column;
		padding: 8px 4px;
		gap: 4px;
	}
	.sidebar-collapsed .sidebar-tabs button {
		flex: none;
		height: 36px;
		border-radius: 4px;
	}
	.sidebar-collapsed .sidebar-tabs button.active::after {
		width: 2px;
		height: auto;
		top: 8px;
		bottom: 8px;
		left: 0;
		right: auto;
	}
	.sidebar-collapsed .tab-glyph {
		display: inline;
	}
	.sidebar-collapsed .tab-label {
		display: none;
	}
	@media (min-width: 1600px) {
		.unified-shell {
			--sidebar-width: 392px;
		}
	}
	@media (max-width: 1300px) {
		.unified-shell {
			--sidebar-width: 352px;
		}
	}
	@media (max-width: 900px) {
		.unified-shell {
			--sidebar-width: 328px;
		}
		.status-seed,
		.status-core,
		.status-chemistry {
			display: none;
		}
		.section-context {
			display: none;
		}
		.app-header {
			gap: 6px;
			padding-left: 10px;
		}
		.wordmark {
			font-size: 14px;
			margin: 0;
		}
		.wordmark svg {
			display: none;
		}
		.field-choice {
			width: 145px;
		}
		.header-look {
			display: none;
		}
		.camera-controls button {
			padding: 4px 6px;
		}
	}
	@media (max-width: 700px) {
		.unified-shell {
			--sidebar-width: min(320px, calc(100vw - 48px));
			--rail-width: 36px;
		}
		.unified-shell.sidebar-collapsed {
			--sidebar-width: 44px;
		}
		.app-header {
			right: 0;
			padding-right: 8px;
			z-index: 45;
		}
		.app-sidebar {
			top: var(--header-height);
		}
		.sidebar-tabs {
			height: 44px;
		}
		.tool-rail button {
			width: 28px;
		}
		.status-size,
		.status-run,
		.status-rate {
			display: none;
		}
		.status-bar {
			gap: 0;
		}
		.transport {
			padding-right: 4px;
			margin-right: 4px;
		}
		.header-space {
			min-width: 0;
		}
		.wordmark {
			font-size: 14px;
		}
		.field-choice {
			width: 145px;
		}
		.sidebar-section {
			padding: 16px;
		}
		.app-sidebar {
			z-index: 30;
		}
		.viewport-caption {
			font-size: 9px;
			max-width: 180px;
		}
	}
	@media (prefers-reduced-motion: reduce) {
		.unified-shell :global(*) {
			transition: none !important;
			animation: none !important;
		}
	}

	.viewport-caption {
		left: 0;
		bottom: 0;
		right: 0;
		max-width: none;
		display: flex;
		align-items: center;
		flex-wrap: wrap;
		gap: 8px 16px;
		min-height: 26px;
		padding: 4px 12px;
		background: #141413e8;
		border-top: 1px solid #ffffff09;
		font-size: 10px;
		text-shadow: none;
	}
	.viewport-caption > span {
		padding: 0;
		background: transparent;
		color: #8e877b;
	}
	.viewport-key {
		display: flex;
		align-items: center;
		gap: 10px;
	}
	.viewport-key span {
		display: flex;
		align-items: center;
		gap: 5px;
		color: #b9afa1;
		padding: 0;
		background: transparent;
	}
	.viewport-key i {
		width: 8px;
		height: 8px;
		border-radius: 2px;
		border: 1px solid #ffffff14;
	}
	.viewport-key .field-gradient {
		width: 72px;
		height: 6px;
		border: 0;
		background: linear-gradient(90deg in srgb, rgb(9% 16% 23%), rgb(97% 87% 66%));
	}
	.viewport-key .field-gradient.diversity {
		background: linear-gradient(90deg in srgb, rgb(15% 31% 43%), rgb(91% 59% 35%));
	}

	.chart-detail {
		margin-top: 16px;
		padding-top: 12px;
		border-top: 1px solid #ffffff0c;
		font-size: 11px;
		line-height: 1.6;
		color: #a99e90;
	}
	.chart-detail strong {
		color: #d3b596;
		font-weight: 400;
	}
	.chart-detail span {
		color: #7f776b;
	}
</style>
